"""Turn the processed corpus into the Stage 1 review artefacts.

    python external_data/scripts/make_reports.py

Writes into reports/:
    QualitySummary.csv      per match: frame counts for every quality flag
    QualityIntervals.csv    the same flags grouped into contiguous time intervals
    SplitManifest.csv       proposed match-level train / validation / test split
    FeatureCompatibility.csv  which canonical fields can also be produced at deployment

Everything here is derived from the processed parquet files, so it is reproducible and it cannot
drift from what was actually parsed.
"""

import csv
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROCESSED = os.path.join(ROOT, "processed")
REPORTS = os.path.join(ROOT, "reports")

FLAGS = ["q_ball_missing", "q_ball_motion_unreliable", "q_label_missing",
         "q_ball_and_label_missing", "q_source_disagreement", "q_alignment_problem"]

REASON = {
    "q_ball_missing": "ball position not observed on this frame",
    "q_ball_motion_unreliable": "frame-to-frame ball displacement implies an implausible speed",
    "q_label_missing": "no usable possession label for this frame",
    "q_ball_and_label_missing": "ball unobserved AND no usable label",
    "q_source_disagreement": "tracking and event sources disagree",
    "q_alignment_problem": "no players tracked at all - camera away from the pitch, or an "
                           "alignment fault",
}

# How each source may be used. Set from what was actually verified during parsing, not from
# the dataset's marketing.
ROLE = {
    "idsse": {
        "split_pool": True,
        "eval_eligible": True,
        "note": "Provider reference fields BallStatus/BallPossession. 25 fps. Ball observed on "
                "100% of frames, which is CLEANER than the competition.",
    },
    "metrica": {
        "split_pool": True,
        "eval_eligible": False,
        "note": "Event-derived. Dead-ball spans law-based (strong); team spans touch-based "
                "(weak). 25 fps. Ball-missing rate close to the competition's.",
    },
    "skillcorner": {
        "split_pool": False,
        "eval_eligible": False,
        "note": "Broadcast tracking, 10 fps, algorithmic possession estimate, NO dead-ball "
                "field. Domain-shift and robustness testing only.",
    },
}


def dfl_match_meta():
    """{match_id: {competition_id, home, away, title}} read from the raw DFL match information."""
    import re
    import xml.etree.ElementTree as ET
    out = {}
    pattern = os.path.join(ROOT, "raw", "idsse", "DFL_02_01_matchinformation_*.xml")
    for path in glob.glob(pattern):
        hit = re.search(r"(DFL-MAT-[A-Z0-9]+)\.xml$", os.path.basename(path))
        if not hit:
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        general = next(root.iter("General"), None)
        teams = {}
        for team in root.iter("Team"):
            role = (team.get("Role") or "").lower()
            if role in ("home", "guest", "away"):
                teams["home" if role == "home" else "away"] = team.get("TeamName")
        out[hit.group(1)] = {
            "competition_id": general.get("CompetitionId") if general is not None else "?",
            "competition": general.get("CompetitionName") if general is not None else "",
            "home": teams.get("home", "?"), "away": teams.get("away", "?"),
            "title": general.get("MatchTitle") if general is not None else "",
        }
    return out


def load_all():
    out = []
    for path in sorted(glob.glob(os.path.join(PROCESSED, "*", "*", "frames.parquet"))):
        frame = pd.read_parquet(path)
        out.append(frame)
    return out


def intervals(frame, column):
    """[(period, start_frame, end_frame, n_frames, seconds)] for contiguous True runs."""
    rows = []
    fps = float(frame["fps"].iloc[0]) if len(frame) else 25.0
    for period, part in frame.groupby("period", sort=True):
        part = part.sort_values("frame_native")
        flag = part[column].fillna(False).to_numpy().astype(np.int8)
        if not flag.any():
            continue
        edges = np.diff(np.concatenate(([0], flag, [0])))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        native = part["frame_native"].to_numpy()
        for a, b in zip(starts, stops):
            rows.append((int(period), int(native[a]), int(native[b - 1]), int(b - a),
                         round((b - a) / fps, 2)))
    return rows


def main():
    os.makedirs(REPORTS, exist_ok=True)
    frames_list = load_all()
    if not frames_list:
        print("no processed frames found - run the parsers first")
        return 1

    summary, interval_rows = [], []
    for frame in frames_list:
        source = frame["source"].iloc[0]
        match_id = frame["match_id"].iloc[0]
        fps = float(frame["fps"].iloc[0])
        n = len(frame)
        row = {
            "source": source, "match_id": match_id, "fps": fps, "n_frames": n,
            "minutes": round(n / fps / 60.0, 1),
            "periods": frame["period"].nunique(),
            "ball_observed_pct": round(100.0 * frame["ball_observed"].mean(), 2),
            "label_valid_pct": round(100.0 * frame["label_valid"].mean(), 2),
            "label_quality_best": max(
                frame["label_quality"].dropna().unique().tolist() or ["none"],
                key=lambda q: ("none", "ambiguous", "weak", "strong", "reference").index(q)
                if q in ("none", "ambiguous", "weak", "strong", "reference") else 0),
            "eval_eligible": ROLE.get(source, {}).get("eval_eligible", False),
        }
        for label in ("H", "A", "D"):
            row["pct_" + label] = round(100.0 * (frame["label"] == label).mean(), 2)
        for flag in FLAGS:
            series = frame[flag].fillna(False)
            row["n_" + flag] = int(series.sum())
            row["pct_" + flag] = round(100.0 * series.mean(), 2)
            for period, start, stop, count, seconds in intervals(frame, flag):
                interval_rows.append({
                    "source": source, "match_id": match_id, "flag": flag,
                    "period": period, "frame_start": start, "frame_end": stop,
                    "n_frames": count, "seconds": seconds, "fps": fps,
                    "reason": REASON[flag],
                })
        summary.append(row)

    summary_df = pd.DataFrame(summary).sort_values(["source", "match_id"])
    summary_df.to_csv(os.path.join(REPORTS, "QualitySummary.csv"), index=False)

    interval_df = pd.DataFrame(interval_rows)
    interval_df.to_csv(os.path.join(REPORTS, "QualityIntervals.csv"), index=False)

    # --- proposed split ------------------------------------------------------------------
    # Match-level, and TEAM-aware. Both halves of a match are one row here so they cannot be
    # separated. SkillCorner is deliberately excluded from train/val/test: its labels are
    # another vendor's estimates, so training on them teaches us to imitate that vendor.
    #
    # The IDSSE corpus is not seven independent matches. Five of the seven are Fortuna
    # Düsseldorf home matches in the same competition and stadium, and only two come from the
    # other competition. Splitting those five across train and test would let a model learn
    # Düsseldorf-specific tracking and pitch characteristics and then be rewarded for it. So
    # the two competitions are kept whole and on opposite sides of the split.
    meta = dfl_match_meta()
    split_rows = []
    for source in sorted(summary_df["source"].unique()):
        part = summary_df[summary_df["source"] == source].sort_values("match_id")
        ids = part["match_id"].tolist()
        if not ROLE.get(source, {}).get("split_pool"):
            for match_id in ids:
                split_rows.append({
                    "source": source, "match_id": match_id, "group": "", "split": "robustness",
                    "reason": "domain-shift probe only; never trained or scored on"})
            continue

        if source == "idsse":
            groups = {}
            for match_id in ids:
                groups.setdefault(meta.get(match_id, {}).get("competition_id", "?"),
                                  []).append(match_id)
            # Largest competition block trains, its last match validates; the other block,
            # which shares no team with it, is the held-out test set.
            ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
            for rank, (competition, members) in enumerate(ordered):
                for i, match_id in enumerate(sorted(members)):
                    if rank == 0:
                        split = "validation" if i == len(members) - 1 else "train"
                        reason = ("largest competition block ({}); held-out match validates "
                                  "within-domain".format(competition))
                    else:
                        split = "test"
                        reason = ("different competition ({}) and no team shared with the "
                                  "training block - genuine held-out evaluation".format(
                                      competition))
                    split_rows.append({
                        "source": source, "match_id": match_id,
                        "group": "{} | home {}".format(
                            competition, meta.get(match_id, {}).get("home", "?")),
                        "split": split, "reason": reason})
        else:
            for match_id in ids:
                split_rows.append({
                    "source": source, "match_id": match_id, "group": "",
                    "split": "train",
                    "reason": "weak labels; usable as training signal, never for scoring"})
    pd.DataFrame(split_rows).to_csv(os.path.join(REPORTS, "SplitManifest.csv"), index=False)

    # --- feature compatibility ------------------------------------------------------------
    # The question that decides whether a feature is allowed to exist: can it also be computed
    # from the competition's own ATD at inference time? Anything that cannot is a trap.
    compat = [
        ("ball_x, ball_y", "yes", "ATD carries ball x,y normalised 0-1 when the ball is tracked"),
        ("ball_observed", "yes", "ATD writes NaN for an untracked ball"),
        ("ball_speed_mps", "yes", "derivable from consecutive frames; ATD also ships a speed field"),
        ("n_home_observed, n_away_observed", "yes",
         "ATD metadata assigns every track slot to a team"),
        ("home/away centroid, spread", "yes", "collective, computed from slot positions"),
        ("nearest_home_m, nearest_away_m, nearest_margin_m", "yes",
         "collective, no identity needed"),
        ("period", "yes", "games.half_video_blocks gives the two graded halves"),
        ("ball height (Z)", "NO",
         "IDSSE/DFL provides a Z coordinate. The competition ATD does not. MUST NOT be used as "
         "a feature, and must not be used to derive one."),
        ("player identity / shirt number", "NO",
         "the competition has 46 anonymous track slots that start, stop and swap; any feature "
         "keyed on a person cannot be computed at deployment"),
        ("per-player speed/acceleration", "partial",
         "DFL ships S and A per player and the ATD ships a speed field, but SkillCorner does "
         "not; a feature using it cannot be trained on SkillCorner"),
        ("Smart Tagging phases", "competition-only",
         "the competition has 638 tagged phases; SkillCorner has phases_of_play but Metrica and "
         "DFL have nothing equivalent, so a tagging-dependent feature cannot be trained externally"),
        ("image_corners_projection / homography", "partial",
         "the competition ships a per-frame homography; only SkillCorner has an analogue"),
    ]
    with open(os.path.join(REPORTS, "FeatureCompatibility.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "available_in_competition_data", "note"])
        writer.writerows(compat)

    # --- console summary -------------------------------------------------------------------
    pd.set_option("display.width", 200)
    print("\n=== corpus ===")
    for source in sorted(summary_df["source"].unique()):
        part = summary_df[summary_df["source"] == source]
        print("  {:<12} {:>2} matches  {:>8,} frames  {:>6.0f} min  "
              "ball seen {:>5.1f}%  labels valid {:>5.1f}%".format(
                  source, len(part), int(part["n_frames"].sum()), part["minutes"].sum(),
                  part["ball_observed_pct"].mean(), part["label_valid_pct"].mean()))
    print("\n=== label mix, mean per source ===")
    print(summary_df.groupby("source")[["pct_H", "pct_A", "pct_D"]].mean().round(1).to_string())
    print("\n=== quality flags, mean % of frames ===")
    print(summary_df.groupby("source")[["pct_" + f for f in FLAGS]].mean().round(2).to_string())
    print("\n=== missing-ball interval structure ===")
    gaps = interval_df[interval_df["flag"] == "q_ball_missing"]
    if len(gaps):
        for source, part in gaps.groupby("source"):
            print("  {:<12} {:>6,} gaps   median {:>5.1f}s  p90 {:>5.1f}s  max {:>6.1f}s".format(
                source, len(part), part["seconds"].median(),
                part["seconds"].quantile(0.9), part["seconds"].max()))
    else:
        print("  (none)")
    print("\nwrote QualitySummary.csv, QualityIntervals.csv, SplitManifest.csv, "
          "FeatureCompatibility.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
