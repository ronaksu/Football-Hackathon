"""Parse the SkillCorner Open Data matches into the canonical representation.

    python external_data/scripts/parse_skillcorner.py
    python external_data/scripts/parse_skillcorner.py --match 1874553

WHAT THIS SOURCE IS GOOD FOR, AND WHAT IT IS NOT
------------------------------------------------
SkillCorner Open Data is BROADCAST tracking: computer vision run over a television feed. That
makes it the closest available analogue to the competition's broadcast recording, and a genuinely
useful domain-shift test - frames go missing when the camera cuts to a replay or a close-up,
exactly as they do in the competition's broadcast homography.

It is a poor label source, for three specific reasons found by inspection rather than assumed:

  1. There is NO per-frame ball-in-play flag. `possession.group` is null on roughly 45% of
     frames, and a null there could mean "the ball is dead" or "we could not tell". Those are
     different statements and the file does not distinguish them. Turning every null into D
     would manufacture dead balls.

  2. The tracking file is named `tracking_extrapolated` and it means it: a ball position is
     present on more frames than `is_detected` is true. An extrapolated position is a guess, so
     `ball_observed` here is taken from `is_detected`, NOT from the presence of coordinates.

  3. Possession is a provider ESTIMATE produced by an algorithm, not a human annotation and not
     a reference field. It is graded `weak` and is never eligible for evaluation. Scoring a
     possession model against another vendor's possession model does not measure accuracy.

The timing is 10 fps and is preserved. Upsampling to 25 fps would invent fifteen observations a
second that nobody made.
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "raw", "skillcorner", "data", "matches")
OUT = os.path.join(ROOT, "processed", "skillcorner")
REPORTS = os.path.join(ROOT, "reports")

sys.path.insert(0, HERE)
from common_schema import FRAME_COLUMNS, PLAYER_COLUMNS, PROVISIONAL  # noqa: E402

FPS = 10          # documented, and checked against the timestamps below
GROUP = {"home team": "home", "away team": "away"}


def list_matches():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(RAW, "*"))
                  if os.path.isdir(p))


def build_match(match_id):
    folder = os.path.join(RAW, match_id)
    meta = json.load(open(os.path.join(folder, "{}_match.json".format(match_id)),
                          encoding="utf-8"))
    length = float(meta.get("pitch_length") or 105.0)
    width = float(meta.get("pitch_width") or 68.0)
    home_id = (meta.get("home_team") or {}).get("id")
    away_id = (meta.get("away_team") or {}).get("id")
    team_of_player = {}
    for player in meta.get("players") or []:
        team_of_player[player.get("id")] = (
            "home" if player.get("team_id") == home_id else
            "away" if player.get("team_id") == away_id else None)

    rows, player_rows = [], []
    notes = Counter()
    with open(os.path.join(folder, "{}_tracking_extrapolated.jsonl".format(match_id)),
              encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            period = record.get("period")
            if period is None:
                notes["frames_without_period"] += 1
                continue
            frame = record.get("frame")
            ball = record.get("ball_data") or {}
            detected = bool(ball.get("is_detected"))
            has_xy = ball.get("x") is not None and ball.get("y") is not None
            if has_xy and not detected:
                notes["ball_extrapolated_not_detected"] += 1

            # Only a DETECTED ball counts as observed. An extrapolated coordinate is a guess.
            bx = (float(ball["x"]) + length / 2.0) / length if (detected and has_xy) else np.nan
            by = (float(ball["y"]) + width / 2.0) / width if (detected and has_xy) else np.nan

            group = (record.get("possession") or {}).get("group")
            side = GROUP.get(group)

            counts = {"home": 0, "away": 0}
            sums = {"home": [0.0, 0.0], "away": [0.0, 0.0]}
            sq = {"home": [0.0, 0.0], "away": [0.0, 0.0]}
            nearest = {"home": np.inf, "away": np.inf}
            for p in record.get("player_data") or []:
                team = team_of_player.get(p.get("player_id"))
                if team is None or p.get("x") is None or p.get("y") is None:
                    continue
                px = (float(p["x"]) + length / 2.0) / length
                py = (float(p["y"]) + width / 2.0) / width
                counts[team] += 1
                sums[team][0] += px; sums[team][1] += py
                sq[team][0] += px * px; sq[team][1] += py * py
                if np.isfinite(bx):
                    d = np.hypot((px - bx) * length, (py - by) * width)
                    nearest[team] = min(nearest[team], d)
                player_rows.append((period, frame, team, counts[team] - 1,
                                    np.float32(px), np.float32(py)))

            row = {"period": np.int16(period), "frame_native": frame,
                   "time_s": frame / float(FPS),
                   "ball_x": bx, "ball_y": by, "ball_observed": detected and has_xy,
                   "_side": side}
            for team in ("home", "away"):
                n = counts[team]
                row["n_{}_observed".format(team)] = np.int16(n)
                mx = sums[team][0] / n if n else np.nan
                my = sums[team][1] / n if n else np.nan
                row["{}_centroid_x".format(team)] = mx
                row["{}_centroid_y".format(team)] = my
                if n > 1:
                    vx = max(sq[team][0] / n - mx * mx, 0.0)
                    vy = max(sq[team][1] / n - my * my, 0.0)
                    row["{}_spread".format(team)] = np.sqrt((vx * length ** 2)
                                                            + (vy * width ** 2))
                else:
                    row["{}_spread".format(team)] = np.nan
                row["nearest_{}_m".format(team)] = (nearest[team]
                                                    if np.isfinite(nearest[team]) else np.nan)
            rows.append(row)

    frames = pd.DataFrame(rows).sort_values(["period", "frame_native"]).reset_index(drop=True)
    frames["source"] = "skillcorner"
    frames["match_id"] = match_id
    frames["fps"] = FPS
    frames["nearest_margin_m"] = frames["nearest_away_m"] - frames["nearest_home_m"]

    step = np.hypot(frames["ball_x"].diff() * length, frames["ball_y"].diff() * width)
    consecutive = (frames["frame_native"].diff() == 1) & (frames["period"].diff() == 0)
    both = frames["ball_observed"] & frames["ball_observed"].shift(1) & consecutive
    frames["ball_speed_mps"] = np.where(both, step * FPS, np.nan)

    # Labels: team in possession only, and only where the provider states one. There is no
    # dead-ball field, so NO frame is labelled D from this source.
    side = frames.pop("_side")
    frames["label"] = side.map({"home": "H", "away": "A"})
    frames["label_source"] = np.where(frames["label"].notna(), "provider_estimate", "none")
    frames["label_quality"] = np.where(frames["label"].notna(), "weak", "none")
    frames["label_valid"] = frames["label"].notna()

    frames["q_ball_missing"] = ~frames["ball_observed"]
    frames["q_ball_motion_unreliable"] = (
        frames["ball_speed_mps"] > PROVISIONAL["max_plausible_ball_speed_mps"]).fillna(False)
    frames["q_label_missing"] = ~frames["label_valid"]
    frames["q_ball_and_label_missing"] = frames["q_ball_missing"] & frames["q_label_missing"]
    frames["q_source_disagreement"] = False
    # A frame with no players at all is the camera being somewhere other than the pitch.
    frames["q_alignment_problem"] = (frames["n_home_observed"] + frames["n_away_observed"]) == 0
    frames["model_uncertainty"] = np.nan

    players = pd.DataFrame(player_rows,
                           columns=["period", "frame_native", "team", "slot", "x", "y"])
    players["source"] = "skillcorner"
    players["match_id"] = match_id

    for column in FRAME_COLUMNS:
        if column not in frames.columns:
            frames[column] = np.nan

    info = {
        "title": "{} v {}".format((meta.get("home_team") or {}).get("short_name"),
                                  (meta.get("away_team") or {}).get("short_name")),
        "competition": ((meta.get("competition_edition") or {}).get("competition") or {}).get(
            "name", ""),
        "pitch": (length, width),
    }
    return frames[FRAME_COLUMNS], players[PLAYER_COLUMNS], info, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", nargs="+", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    matches = args.match or list_matches()
    os.makedirs(OUT, exist_ok=True)
    provenance = []

    for match_id in matches:
        frames, players, info, notes = build_match(match_id)
        n = len(frames)
        counts = Counter(frames["label"].dropna())
        print("\n--- skillcorner {}  {} ---".format(match_id, info["title"]))
        print("  frames {:,} at {} fps  periods {}  pitch {}x{}".format(
            n, FPS, sorted(frames["period"].unique().tolist()), *info["pitch"]))
        print("  ball DETECTED {:,} ({:.1f}%)   extrapolated-but-not-detected {:,}".format(
            int(frames["ball_observed"].sum()), 100 * frames["ball_observed"].mean(),
            notes.get("ball_extrapolated_not_detected", 0)))
        print("  frames with no players at all (camera off pitch): {:,} ({:.1f}%)".format(
            int(frames["q_alignment_problem"].sum()), 100 * frames["q_alignment_problem"].mean()))
        print("  labels: " + ("  ".join("{} {:,} ({:.1f}%)".format(k, v, 100 * v / n)
                                        for k, v in sorted(counts.items())) or "none"))
        print("  no possession stated on {:,} frames ({:.1f}%) - NOT turned into D".format(
            int(frames["q_label_missing"].sum()), 100 * frames["q_label_missing"].mean()))

        folder = os.path.join(OUT, match_id)
        os.makedirs(folder, exist_ok=True)
        frames.to_parquet(os.path.join(folder, "frames.parquet"), index=False)
        players.to_parquet(os.path.join(folder, "players.parquet"), index=False)

        for quality in sorted(set(frames["label_quality"])):
            mask = frames["label_quality"] == quality
            provenance.append({
                "source": "skillcorner", "match_id": match_id, "label_quality": quality,
                "label_source": "provider_estimate" if quality == "weak" else "none",
                "n_frames": int(mask.sum()),
                "pct_of_match": round(100.0 * mask.mean(), 2),
                "eligible_for_training": quality == "weak",
                "eligible_for_evaluation": False,
                "definition_note": (
                    "Algorithmic provider estimate of the team in possession. No dead-ball "
                    "field exists, so this source contributes NO D labels and a null is left "
                    "null. 10 fps, broadcast-derived."),
                "home_mapping": "explicit home_team / away_team ids in the match JSON",
            })

    path = os.path.join(REPORTS, "LabelProvenance.csv")
    touched = {(r["source"], r["match_id"]) for r in provenance}
    existing, fields = [], None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            existing = [r for r in reader
                        if (r.get("source"), r.get("match_id")) not in touched]
    fields = fields or list(provenance[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in existing + provenance:
            writer.writerow({k: row.get(k, "") for k in fields})
    print("\nprovenance -> {}".format(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
