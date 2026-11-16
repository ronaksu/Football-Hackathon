"""Turn the competition's ATD into the canonical representation the models already consume.

WHY THIS FILE IS THE WHOLE BRIDGE
---------------------------------
Every model in this project was trained on open-access tracking converted into one schema
(external_data/scripts/common_schema.py). Nothing in stage 2 or stage 3 knows what a Metrica
EPTS file is. This script is the only place that does: it reads the delivered ATD pair and
writes the same two parquet files every training match has, so the rest of the pipeline runs on
the competition match without a single change.

    processed/<match_id>/frames.parquet      one row per frame
    processed/<match_id>/players.parquet     one row per observed track slot per frame

WHAT IS AND IS NOT READ
-----------------------
  read      the raw ATD text (positions and ball), the ATD metadata XML (two teams, and which
            team each of the 46 track slots belongs to), and the published half boundaries
  NOT read  the ground truth, public or private; the Smart Tagging; the video; the homography

`label` is written as null and `label_valid` as False on every row, because for this match there
is no label and inventing one would be the one thing the rules forbid. Downstream code must
therefore never filter this match on `label_valid` -- submit/predict_match.py does not.

ONLY THE GRADED FRAMES ARE BUILT
--------------------------------
DATA.md publishes the two half boundaries in video frames. They are boundaries, not answers.
Outside them -- before kickoff, halftime, after the final whistle -- nothing is scored, so the
submission writes `X` there and there is no reason to carry 40,462 frames of an empty pitch
through a feature builder. Building only the graded blocks also means `period` is exact rather
than guessed, which is the one thing a tracking file this anonymous cannot tell us itself.

FRAME NUMBERING
---------------
`frame_native` is the VIDEO frame, 0-based, so it is already the submission's row number. The ATD
counts from 1, so `frame_native = frame_id - 1` -- the single conversion in DATA.md's alignment
table, applied once, here.

    python -I submit/atd_to_canonical.py --data-dir "D:/Football Hackathon/data"
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "external_data", "scripts"))
from common_schema import FRAME_COLUMNS, PLAYER_COLUMNS, PROVISIONAL  # noqa: E402

FPS = 25
MATCH_ID = "BAR-ATH"
SOURCE = "mtah2027"

# The pipeline measures everything on a 105.0 x 68.0 pitch -- stage2/features.py,
# stage3/features_team.py and stage3/restart_logic.py all carry those two constants. The ATD
# metadata reports 105 x 68 and games.py carries 105.3 x 68.0, measured from corrected tracking.
# Using the pipeline's own numbers keeps every feature on the scale the models were fitted on;
# 30 cm of pitch length is far smaller than that consistency is worth.
PITCH_X, PITCH_Y = 105.0, 68.0

# Published in DATA.md, inclusive, in video frames. Boundaries, not labels.
HALVES = ((1, 11185, 90651), (2, 110808, 190197))
N_VIDEO_FRAMES = 199319

# Raw ATD line: "<frame_id>:<x,y,speed>;<x,y,speed>;...:<ballx,bally,ballspeed>", NaN when absent.
FIELD_SEP, GROUP_SEP, MISSING = ":", ";", "N"


def team_of_each_slot(meta_path):
    """(is_home per track slot, (home name, away name)).

    The first <Team> in the metadata is the home side -- the FIFA/EPTS convention, which kloppy
    follows when it reports metadata.teams and which the competition's own baseline relies on.
    DATA.md states independently that home is Barcelona, so the convention and the documentation
    agree and neither has to be taken on trust alone.

    A slot is a tracklet, not a person: 46 of them for 22 players, appearing, vanishing and
    swapping. Team membership is the only thing about a slot that holds for the whole match, and
    it is the only thing taken from here.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(team_ids)))
    names = [t.findtext("Name") for t in root.iter("Team")]
    slots = [p.get("teamId") for p in root.iter("Player")]
    if not slots:
        raise ValueError("no <Player> entries in {}".format(meta_path))
    return np.array([tid == team_ids[0] for tid in slots], dtype=bool), names


def period_of_video_frame():
    """Lookup array over video frames: the half a frame belongs to, 0 where it is ungraded."""
    out = np.zeros(N_VIDEO_FRAMES, dtype=np.int16)
    for half, first, last in HALVES:
        out[first:last + 1] = half
    return out


def block_of(video_frame):
    """Which ungraded stretch a frame falls in. Derived from the published half boundaries."""
    if video_frame < HALVES[0][1]:
        return "before_kickoff"
    if video_frame > HALVES[1][2]:
        return "after_whistle"
    return "halftime"


def _pair(group):
    """(x, y) out of an "x,y,speed" group. The provided speed is deliberately dropped.

    The ATD ships a per-object speed; the training sources do not all ship one, and every speed
    feature in this project is differenced from positions. Taking the provided speed here would
    feed the models a quantity computed differently from the one they were fitted on.
    """
    comma = group.index(",")
    second = group.index(",", comma + 1)
    return float(group[:comma]), float(group[comma + 1:second])


def read_atd(raw_path, is_home_slot, period_of):
    """Stream the raw ATD once. Returns (frames, players, n_lines).

    One pass, because the file is 199,319 lines and the only thing needed from a line is its
    frame id, the ball, and the slots actually observed on it. Rows outside the graded blocks
    are discarded as they are read rather than built and dropped later.
    """
    n_slots = len(is_home_slot)
    rows, ungraded_rows = [], []
    p_frame, p_home, p_slot, p_x, p_y = [], [], [], [], []
    n_lines = 0

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            n_lines += 1
            video_frame = int(parts[0]) - 1
            if not 0 <= video_frame < N_VIDEO_FRAMES:
                continue
            period = int(period_of[video_frame])

            ball = parts[2]
            if ball[0] == MISSING:
                bx = by = np.nan
                observed = False
            else:
                bx, by = _pair(ball)
                observed = True

            if period == 0:
                # The ball track on the three stretches where no football is being played. No
                # label is involved: DATA.md publishes these boundaries, and what the ball does
                # across them is the only way to test a rule that reads the ball WITHOUT using
                # an answer. submit/predict_match.py uses it exactly for that.
                ungraded_rows.append((video_frame, block_of(video_frame), bx, by, observed))
                continue
            rows.append((video_frame, period, bx, by, observed))

            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if slot >= n_slots or group[0] == MISSING:
                    continue
                x, y = _pair(group)
                p_frame.append(video_frame)
                p_slot.append(slot)
                p_home.append(is_home_slot[slot])
                p_x.append(x)
                p_y.append(y)

    frames = pd.DataFrame(rows, columns=["frame_native", "period", "ball_x", "ball_y",
                                         "ball_observed"])
    frames = frames.sort_values("frame_native").reset_index(drop=True)
    ungraded = pd.DataFrame(ungraded_rows, columns=["frame_native", "block", "ball_x", "ball_y",
                                                    "ball_observed"])
    ungraded = ungraded.sort_values("frame_native").reset_index(drop=True)

    home = np.asarray(p_home, dtype=bool)
    period_lookup = frames.set_index("frame_native")["period"]
    players = pd.DataFrame({
        "period": period_lookup.reindex(p_frame).to_numpy().astype(np.int16),
        "frame_native": np.asarray(p_frame, dtype=np.int64),
        "team": np.where(home, "home", "away"),
        # The slot index is kept as the ATD gives it. The feature builders pivot on
        # "<team>_<slot>", so the two sides cannot collide even though they share the numbering.
        "slot": np.asarray(p_slot, dtype=np.int16),
        "x": np.asarray(p_x, dtype=np.float32),
        "y": np.asarray(p_y, dtype=np.float32),
    })
    return frames, players, ungraded, n_lines


def derive(frames, players):
    """Fill the rest of the canonical frame schema from the ball and the player positions.

    The same arithmetic as external_data/scripts/parse_dfl.py, which built every training match:
    ball speed only across two consecutive observed frames of one period, team centroids and
    spread, and the distance from the ball to the nearest player of each side. Stage 2 and
    stage 3 rebuild their own geometry from players.parquet and do not read these columns, so
    they are here for inspection and schema parity, not to feed a model.
    """
    frames = frames.copy()
    frames["source"] = SOURCE
    frames["match_id"] = MATCH_ID
    frames["fps"] = FPS
    frames["time_s"] = frames["frame_native"] / float(FPS)

    step = np.hypot(frames["ball_x"].diff() * PITCH_X, frames["ball_y"].diff() * PITCH_Y)
    consecutive = (frames["frame_native"].diff() == 1) & (frames["period"].diff() == 0)
    both = (frames["ball_observed"] & frames["ball_observed"].shift(1).fillna(False)
            & consecutive.fillna(False))
    frames["ball_speed_mps"] = np.where(both, step * FPS, np.nan)

    ball = frames.set_index("frame_native")[["ball_x", "ball_y"]]
    for side in ("home", "away"):
        g = players[players["team"] == side]
        agg = g.groupby("frame_native").agg(n=("slot", "size"), cx=("x", "mean"),
                                            cy=("y", "mean"), sx=("x", "std"), sy=("y", "std"))
        frames["n_{}_observed".format(side)] = (
            frames["frame_native"].map(agg["n"]).fillna(0).astype(np.int16))
        frames["{}_centroid_x".format(side)] = frames["frame_native"].map(agg["cx"])
        frames["{}_centroid_y".format(side)] = frames["frame_native"].map(agg["cy"])
        frames["{}_spread".format(side)] = np.sqrt(
            (frames["frame_native"].map(agg["sx"]) * PITCH_X) ** 2
            + (frames["frame_native"].map(agg["sy"]) * PITCH_Y) ** 2)

        j = g.join(ball, on="frame_native")
        d = np.hypot((j["x"] - j["ball_x"]) * PITCH_X, (j["y"] - j["ball_y"]) * PITCH_Y)
        frames["nearest_{}_m".format(side)] = frames["frame_native"].map(
            d.groupby(j["frame_native"]).min())
    frames["nearest_margin_m"] = frames["nearest_away_m"] - frames["nearest_home_m"]

    # No label exists for this match, and none may be invented. Everything downstream that needs
    # to know which frames to predict uses `period`, never `label_valid`.
    frames["label"] = None
    frames["label_valid"] = False
    frames["label_source"] = "none"
    frames["label_quality"] = "none"
    frames["model_uncertainty"] = np.nan

    frames["q_ball_missing"] = ~frames["ball_observed"]
    unreliable = frames["ball_speed_mps"] > PROVISIONAL["max_plausible_ball_speed_mps"]
    frames["q_ball_motion_unreliable"] = unreliable.fillna(False)
    frames["q_label_missing"] = True
    frames["q_ball_and_label_missing"] = frames["q_ball_missing"]
    frames["q_source_disagreement"] = False
    frames["q_alignment_problem"] = False

    players = players.copy()
    players["source"] = SOURCE
    players["match_id"] = MATCH_ID
    return frames[FRAME_COLUMNS], players[PLAYER_COLUMNS]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"),
                    help="the documented bundle layout: <data-dir>/tactical/...")
    ap.add_argument("--source", default="tactical", choices=["tactical", "broadcast"],
                    help="tactical is the recording a submission is indexed to")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "submit", "processed"))
    ap.add_argument("--report", default=os.path.join(ROOT, "submit", "reports",
                                                     "atd_adapter_report.txt"))
    args = ap.parse_args()

    folder = os.path.join(args.data_dir, args.source)
    stem = "{}_{}".format(MATCH_ID, args.source)
    meta_path = os.path.join(folder, stem + "_atd.xml")
    raw_path = os.path.join(folder, stem + "_atd.txt")
    for path in (meta_path, raw_path):
        if not os.path.exists(path):
            raise SystemExit("missing: {}\nRun local_checks/consolidate_data.py first."
                             .format(path))

    is_home_slot, names = team_of_each_slot(meta_path)
    frames, players, ungraded, n_lines = read_atd(raw_path, is_home_slot,
                                                  period_of_video_frame())
    frames, players = derive(frames, players)

    out = os.path.join(args.out_dir, MATCH_ID)
    os.makedirs(out, exist_ok=True)
    frames.to_parquet(os.path.join(out, "frames.parquet"), index=False)
    players.to_parquet(os.path.join(out, "players.parquet"), index=False)
    ungraded.to_parquet(os.path.join(out, "ball_ungraded.parquet"), index=False)

    expected_graded = sum(last - first + 1 for _, first, last in HALVES)
    gaps = set(frames["frame_native"].diff().dropna().astype(int).tolist())
    lines = [
        "ATD -> canonical, {} recording".format(args.source),
        "",
        "  teams          H = {}   A = {}".format(names[0], names[1]),
        "  track slots    {}  ({} home, {} away)".format(
            len(is_home_slot), int(is_home_slot.sum()), int((~is_home_slot).sum())),
        "  ATD records    {:,}   doc {:,}   {}".format(
            n_lines, N_VIDEO_FRAMES, "PASS" if n_lines == N_VIDEO_FRAMES else "MISMATCH"),
        "  graded frames  {:,}   doc {:,}   {}".format(
            len(frames), expected_graded,
            "PASS" if len(frames) == expected_graded else "MISMATCH"),
        "  frame steps    {}   (1 within a half, one jump across halftime)".format(sorted(gaps)),
        "  periods        {}".format(frames.groupby("period").size().to_dict()),
        "  ball observed  {:,} of {:,} graded frames ({:.1f}%)   doc 59.3%".format(
            int(frames["ball_observed"].sum()), len(frames),
            100 * frames["ball_observed"].mean()),
        "  tracked/frame  mean home {:.1f}, away {:.1f}, total {:.1f}   doc 20.5".format(
            frames["n_home_observed"].mean(), frames["n_away_observed"].mean(),
            (frames["n_home_observed"] + frames["n_away_observed"]).mean()),
        "  labels         none, on every row, by design",
        "",
        "  ungraded ball track, for testing ball rules without any label:",
    ] + [
        "    {:<16} {:>7,} frames, ball reported on {:5.1f}%".format(
            blk, len(g), 100 * g["ball_observed"].mean())
        for blk, g in ungraded.groupby("block", sort=True)
    ] + [
        "",
        "  frames  -> {}".format(os.path.join(out, "frames.parquet")),
        "  players -> {}  ({:,} observations)".format(
            os.path.join(out, "players.parquet"), len(players)),
    ]
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nreport -> {}".format(args.report))


if __name__ == "__main__":
    main()
