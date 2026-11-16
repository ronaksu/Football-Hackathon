"""Parse the three Metrica sample games into the canonical representation, with honest labels.

    python external_data/scripts/parse_metrica.py
    python external_data/scripts/parse_metrica.py --game 3

Writes, per game:
    processed/metrica/<match_id>/frames.parquet
    processed/metrica/<match_id>/players.parquet
and appends to reports/LabelProvenance.csv and reports/StructuralChecks.csv.

WHAT THE LABELS ARE, AND ARE NOT
--------------------------------
Metrica's own event definitions (documentation/events-definitions.pdf, read in full) define
possession as beginning the moment a player TOUCHES the ball. The competition defines a change
of possession as the moment an opposing player gains CONTROLLED possession, and says explicitly
that a simple touch, a deflection that does not intercept, and an aerial duel won without
subsequent control are NOT changes of possession.

Those two definitions are not the same, and the difference is not cosmetic: Metrica's convention
changes possession more often than the competition's does. So:

  - Dead-ball spans derived here are STRONG. They come from the laws of the game rather than
    from a judgement about control: BALL OUT means the ball crossed the line, FAULT RECEIVED
    means the referee stopped play, and a SET PIECE means play restarted. Both conventions
    agree about those.

  - Team-possession spans derived here are WEAK. They are built from touch-level events and
    will place hand-overs earlier, and more often, than the competition's convention would.

  - Challenges are AMBIGUOUS and are masked unless the event carries an explicit result.

The quality column records this per frame, and nothing marked `ambiguous` is given a label at
all. Masking is the honest option; filling the gap and calling it truth is not.
"""

import argparse
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "raw", "metrica", "data")
OUT = os.path.join(ROOT, "processed", "metrica")
REPORTS = os.path.join(ROOT, "reports")

sys.path.insert(0, HERE)
from common_schema import FRAME_COLUMNS, PLAYER_COLUMNS, PROVISIONAL  # noqa: E402

FPS = 25
PITCH = (105.0, 68.0)      # metres; stated in the Metrica README for games 1 and 2

# --- event semantics, taken from Metrica's own definitions document -----------------------------
# Events after which the ball is OUT OF PLAY.
STOP_TYPES = {"BALL OUT", "FAULT RECEIVED"}
# BALL LOST is normally a turnover, EXCEPT with these Intervention subtypes, which mean the
# referee or the player stopped play rather than an opponent winning the ball.
STOP_SUBTYPES_ON_BALL_LOST = {"FORCED", "VOLUNTARY", "END HALF"}
# Events that bring the ball back into play.
RESTART_TYPES = {"SET PIECE", "RECOVERY"}
# Events that change which TEAM has the ball.
TURNOVER_TYPES = {"RECOVERY", "BALL LOST"}
# Events that keep possession within the same team.
CONTINUE_TYPES = {"PASS", "SHOT"}
# Events that tell us nothing about control on their own.
NEUTRAL_TYPES = {"CARD"}
# A goal stops play.
GOAL_SUBTYPE = "GOAL"


def subtypes_of(raw):
    """Metrica packs several subtypes into one dash- or comma-separated string."""
    if not raw:
        return set()
    text = str(raw).upper()
    for sep in ("-", ",", ";", "/"):
        text = text.replace(sep, " ")
    return {t.strip() for t in text.split() if t.strip()}


# -- tracking --------------------------------------------------------------------------------

def read_metrica_csv_side(path, team):
    """(frames DataFrame, players long DataFrame) for one team's CSV.

    The file has three header rows: team names, shirt numbers, then column names. Positions are
    x,y pairs per player and the last pair is the ball.
    """
    with open(path, encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    names = rows[2]
    body = np.array([[np.nan if c in ("", "NaN") else float(c) for c in r[:len(names)]]
                     for r in rows[3:] if r and r[0] != ""], dtype=np.float64)

    period = body[:, 0].astype(np.int16)
    frame = body[:, 1].astype(np.int64)
    time_s = body[:, 2]

    # Player columns are pairs starting at index 3; the last pair is the ball.
    labels = [n.strip() for n in names]
    ball_at = labels.index("Ball") if "Ball" in labels else len(labels) - 2
    xs = list(range(3, ball_at, 2))

    players = []
    for slot, col in enumerate(xs):
        x = body[:, col]
        y = body[:, col + 1]
        seen = ~(np.isnan(x) | np.isnan(y))
        if not seen.any():
            continue
        players.append(pd.DataFrame({
            "period": period[seen], "frame_native": frame[seen],
            "team": team, "slot": slot,
            "x": x[seen].astype(np.float32), "y": y[seen].astype(np.float32),
        }))

    ball_x = body[:, ball_at]
    ball_y = body[:, ball_at + 1]
    frames = pd.DataFrame({"period": period, "frame_native": frame, "time_s": time_s,
                           "ball_x": ball_x, "ball_y": ball_y})
    long = pd.concat(players, ignore_index=True) if players else pd.DataFrame(
        columns=["period", "frame_native", "team", "slot", "x", "y"])
    return frames, long


def read_metrica_epts(path, n_players_per_team=11):
    """Game 3: '<frame>:<x,y>;<x,y>;...:<ballx,bally>' - the same shape as the competition ATD.

    The competition's ATD carries x,y,speed triples; this file carries x,y pairs only. That is a
    real schema difference between two files from the same provider, and it is why the parser
    reads the group width from the data rather than assuming it.
    """
    frames, longs = [], []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0])
            groups = parts[1].split(";")
            for slot, group in enumerate(groups):
                bits = group.split(",")
                if not bits or bits[0].startswith("N"):
                    continue
                try:
                    x, y = float(bits[0]), float(bits[1])
                except (ValueError, IndexError):
                    continue
                team = "home" if slot < n_players_per_team else "away"
                longs.append((frame, team, slot if team == "home" else slot - n_players_per_team,
                              x, y))
            bits = parts[2].split(",")
            try:
                bx, by = float(bits[0]), float(bits[1])
            except (ValueError, IndexError):
                bx = by = np.nan
            frames.append((frame, bx, by))

    frame_df = pd.DataFrame(frames, columns=["frame_native", "ball_x", "ball_y"])
    long_df = pd.DataFrame(longs, columns=["frame_native", "team", "slot", "x", "y"])
    return frame_df, long_df


# -- events ----------------------------------------------------------------------------------

def read_events_csv(path):
    out = []
    with open(path, encoding="utf-8-sig") as handle:
        for r in csv.DictReader(handle):
            try:
                out.append({
                    "team": (r["Team"] or "").strip().lower(),
                    "type": (r["Type"] or "").strip().upper(),
                    "subtypes": subtypes_of(r.get("Subtype")),
                    "period": int(r["Period"]),
                    "sf": int(r["Start Frame"]), "ef": int(r["End Frame"]),
                })
            except (KeyError, ValueError):
                continue
    return sorted(out, key=lambda e: (e["period"], e["sf"], e["ef"]))


def read_events_json(path, home_team_id):
    blob = json.load(open(path, encoding="utf-8"))
    data = blob["data"] if isinstance(blob, dict) and "data" in blob else blob
    out = []
    for e in data:
        team = (e.get("team") or {}).get("id")
        sub = e.get("subtypes")
        names = set()
        if isinstance(sub, dict):
            names = subtypes_of(sub.get("name"))
        elif isinstance(sub, list):
            for s in sub:
                names |= subtypes_of((s or {}).get("name"))
        try:
            out.append({
                "team": "home" if team == home_team_id else ("away" if team else ""),
                "type": ((e.get("type") or {}).get("name") or "").strip().upper(),
                "subtypes": names,
                "period": int(e.get("period") or 1),
                "sf": int((e.get("start") or {}).get("frame")),
                "ef": int((e.get("end") or {}).get("frame") or (e.get("start") or {}).get("frame")),
            })
        except (TypeError, ValueError):
            continue
    return sorted(out, key=lambda e: (e["period"], e["sf"], e["ef"]))


# -- labels ----------------------------------------------------------------------------------

def derive_labels(events, frame_index):
    """(label, quality) per frame, following Metrica's documented event semantics.

    State machine over each period: `alive` and `team`. Every frame inherits the state the last
    event left behind. Where an event genuinely does not settle who is in control, the span is
    marked ambiguous and gets no label at all.
    """
    label = {}
    quality = {}

    for period in sorted({e["period"] for e in events}):
        half = [e for e in events if e["period"] == period]
        if not half:
            continue
        alive, team = False, None
        cursor = half[0]["sf"]

        for event in half:
            # Frames between the previous event and this one keep the previous state.
            _paint(label, quality, cursor, event["sf"] - 1, alive, team,
                   "strong" if not alive else "weak")

            kind, subs = event["type"], event["subtypes"]
            span_quality = "weak"

            if kind in NEUTRAL_TYPES:
                pass

            elif kind in STOP_TYPES:
                _paint(label, quality, event["sf"], event["ef"], alive, team, "weak")
                alive = False
                span_quality = "strong"
                cursor = event["ef"] + 1
                _paint(label, quality, cursor, cursor - 1, alive, team, span_quality)
                continue

            elif kind == "BALL LOST" and (subs & STOP_SUBTYPES_ON_BALL_LOST):
                # The referee or the player stopped play. This is NOT a turnover.
                _paint(label, quality, event["sf"], event["ef"], alive, team, "weak")
                alive = False
                cursor = event["ef"] + 1
                continue

            elif kind == "SHOT" and GOAL_SUBTYPE in subs:
                _paint(label, quality, event["sf"], event["ef"], alive, team, "weak")
                alive = False
                cursor = event["ef"] + 1
                continue

            elif kind in RESTART_TYPES and not alive:
                alive = True
                team = event["team"] or team
                span_quality = "strong"   # a restart is law-defined

            elif kind == "RECOVERY":
                alive = True
                team = event["team"] or team

            elif kind == "BALL LOST":
                alive = True
                team = _other(event["team"]) or team

            elif kind == "CHALLENGE":
                result = subs & {"WON", "LOST"}
                if result == {"WON"}:
                    team = event["team"] or team
                elif result == {"LOST"}:
                    team = _other(event["team"]) or team
                else:
                    # Two players contesting the ball with no recorded result. The competition
                    # says an aerial duel won without subsequent control is not a change of
                    # possession, so inferring one here would be inventing an answer.
                    span_quality = "ambiguous"

            elif kind in CONTINUE_TYPES:
                alive = True
                team = event["team"] or team

            _paint(label, quality, event["sf"], event["ef"], alive, team, span_quality)
            cursor = event["ef"] + 1

        last = max(e["ef"] for e in half)
        _paint(label, quality, cursor, last, alive, team, "weak")

    labels = np.array([label.get(f, None) for f in frame_index], dtype=object)
    quals = np.array([quality.get(f, "none") for f in frame_index], dtype=object)
    return labels, quals


def _other(team):
    return {"home": "away", "away": "home"}.get(team)


def _paint(label, quality, lo, hi, alive, team, qual):
    if hi < lo:
        return
    value = None
    if qual == "ambiguous":
        value = None
    elif not alive:
        value = "D"
    elif team == "home":
        value = "H"
    elif team == "away":
        value = "A"
    for f in range(lo, hi + 1):
        label[f] = value
        quality[f] = qual


# -- assembly --------------------------------------------------------------------------------

def build_game(game, pitch=PITCH):
    folder = os.path.join(RAW, "Sample_Game_{}".format(game))
    match_id = "metrica_sample_game_{}".format(game)
    notes = []

    if game in (1, 2):
        home_f, home_p = read_metrica_csv_side(
            os.path.join(folder, "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(game)),
            "home")
        away_f, away_p = read_metrica_csv_side(
            os.path.join(folder, "Sample_Game_{}_RawTrackingData_Away_Team.csv".format(game)),
            "away")
        if len(home_f) != len(away_f):
            notes.append("home/away tracking row counts differ: {} vs {}".format(
                len(home_f), len(away_f)))
        frames = home_f[["period", "frame_native", "time_s", "ball_x", "ball_y"]].copy()
        # The ball is repeated in both files; check they agree before trusting either.
        merged = home_f[["frame_native", "ball_x"]].merge(
            away_f[["frame_native", "ball_x"]], on="frame_native", suffixes=("_h", "_a"))
        disagree = int((~np.isclose(merged["ball_x_h"], merged["ball_x_a"], equal_nan=True)).sum())
        notes.append("ball disagreement between home/away files: {} frames".format(disagree))
        players = pd.concat([home_p, away_p], ignore_index=True)
        events = read_events_csv(
            os.path.join(folder, "Sample_Game_{}_RawEventsData.csv".format(game)))
        home_label = "Home (explicit in the file name and header)"
    else:
        meta = ET.parse(os.path.join(folder, "Sample_Game_3_metadata.xml")).getroot()
        teams = [t.get("id") for t in meta.iter("Team")]
        home_id = teams[0]
        notes.append("Game 3 metadata gives no home/away designation; first team in the file "
                     "({}) is treated as home by the FIFA/EPTS convention the competition's own "
                     "baseline uses. This mapping is a CONVENTION, not a verified fact.".format(
                         home_id))
        bounds = {}
        for param in meta.iter("ProviderParameter"):
            name = (param.findtext("Name") or "").strip()
            value = (param.findtext("Value") or "").strip()
            if name.endswith("_start") or name.endswith("_end"):
                bounds[name] = value
        frames, players = read_metrica_epts(
            os.path.join(folder, "Sample_Game_3_tracking.txt"))
        h1e = int(bounds.get("first_half_end") or 0)
        h2s = int(bounds.get("second_half_start") or 0)
        frames["period"] = np.where(frames["frame_native"] <= h1e, 1, 2).astype(np.int16)
        frames["time_s"] = frames["frame_native"] / float(FPS)
        players = players.merge(frames[["frame_native", "period"]], on="frame_native", how="left")
        if h2s == h1e + 1:
            notes.append("halves are CONTIGUOUS (first_half_end={}, second_half_start={}): there "
                         "is no halftime gap, unlike the competition match which has 20,156 "
                         "ungraded frames between halves.".format(h1e, h2s))
        events = read_events_json(os.path.join(folder, "Sample_Game_3_events.json"), home_id)
        home_label = "FIFATMA by convention (NOT stated in the data)"

    frames = frames.sort_values("frame_native").reset_index(drop=True)
    frames["source"] = "metrica"
    frames["match_id"] = match_id
    frames["fps"] = FPS

    length, width = pitch
    scale = float(np.hypot(length, width))

    frames["ball_observed"] = ~(frames["ball_x"].isna() | frames["ball_y"].isna())

    # Ball speed, only where two consecutive frames are both observed.
    dx = frames["ball_x"].diff() * length
    dy = frames["ball_y"].diff() * width
    step = np.sqrt(dx ** 2 + dy ** 2)
    consecutive = frames["frame_native"].diff() == 1
    both = frames["ball_observed"] & frames["ball_observed"].shift(1) & consecutive
    frames["ball_speed_mps"] = np.where(both, step * FPS, np.nan)

    # Per-team geometry.
    for team in ("home", "away"):
        side = players[players["team"] == team]
        agg = side.groupby("frame_native").agg(
            n=("slot", "size"), cx=("x", "mean"), cy=("y", "mean"),
            sx=("x", "std"), sy=("y", "std"))
        frames = frames.merge(agg, on="frame_native", how="left", suffixes=("", "_" + team))
        frames["n_{}_observed".format(team)] = frames.pop("n").fillna(0).astype(np.int16)
        frames["{}_centroid_x".format(team)] = frames.pop("cx")
        frames["{}_centroid_y".format(team)] = frames.pop("cy")
        frames["{}_spread".format(team)] = np.sqrt(
            (frames.pop("sx") * length) ** 2 + (frames.pop("sy") * width) ** 2)

    # Distance from the ball to the closest player of each side.
    nearest = {}
    ball = frames.set_index("frame_native")[["ball_x", "ball_y"]]
    for team in ("home", "away"):
        side = players[players["team"] == team].join(ball, on="frame_native")
        d = np.sqrt(((side["x"] - side["ball_x"]) * length) ** 2
                    + ((side["y"] - side["ball_y"]) * width) ** 2)
        nearest[team] = d.groupby(side["frame_native"]).min()
    frames["nearest_home_m"] = frames["frame_native"].map(nearest["home"])
    frames["nearest_away_m"] = frames["frame_native"].map(nearest["away"])
    frames["nearest_margin_m"] = frames["nearest_away_m"] - frames["nearest_home_m"]

    # Labels.
    labels, quals = derive_labels(events, frames["frame_native"].to_numpy())
    frames["label"] = labels
    frames["label_quality"] = quals
    frames["label_source"] = np.where(frames["label"].notna(), "event_derived", "none")
    frames["label_valid"] = frames["label"].notna() & (frames["label_quality"] != "ambiguous")

    # Quality flags.
    frames["q_ball_missing"] = ~frames["ball_observed"]
    frames["q_ball_motion_unreliable"] = (
        frames["ball_speed_mps"] > PROVISIONAL["max_plausible_ball_speed_mps"]).fillna(False)
    frames["q_label_missing"] = ~frames["label_valid"]
    frames["q_ball_and_label_missing"] = frames["q_ball_missing"] & frames["q_label_missing"]
    frames["q_source_disagreement"] = False
    frames["q_alignment_problem"] = False
    frames["model_uncertainty"] = np.nan

    players["source"] = "metrica"
    players["match_id"] = match_id

    for column in FRAME_COLUMNS:
        if column not in frames.columns:
            frames[column] = np.nan
    frames = frames[FRAME_COLUMNS]
    players = players[PLAYER_COLUMNS]

    return match_id, frames, players, events, notes, home_label


def summarise(match_id, frames, players, events, notes, home_label):
    n = len(frames)
    counts = Counter(frames["label"].dropna())
    print("\n--- {} ---".format(match_id))
    print("  frames {:,}   periods {}   fps {}".format(
        n, sorted(frames["period"].unique().tolist()), int(frames["fps"].iloc[0])))
    print("  frame_native {}..{}   contiguous: {}".format(
        int(frames["frame_native"].min()), int(frames["frame_native"].max()),
        bool((frames["frame_native"].diff().dropna() == 1).all())))
    print("  home mapping: {}".format(home_label))
    print("  players: {:,} observations, {} home slots, {} away slots".format(
        len(players),
        players[players["team"] == "home"]["slot"].nunique(),
        players[players["team"] == "away"]["slot"].nunique()))
    print("  tracked per frame: mean home {:.1f}, away {:.1f}".format(
        frames["n_home_observed"].mean(), frames["n_away_observed"].mean()))
    print("  ball observed on {:,} frames ({:.1f}%)".format(
        int(frames["ball_observed"].sum()), 100 * frames["ball_observed"].mean()))
    print("  events {:,}   types: {}".format(
        len(events), dict(Counter(e["type"] for e in events).most_common(5))))
    print("  labels: " + "  ".join("{} {:,} ({:.1f}%)".format(k, v, 100 * v / n)
                                   for k, v in sorted(counts.items())))
    print("  label_valid {:,} ({:.1f}%)   ambiguous {:,}   unlabelled {:,}".format(
        int(frames["label_valid"].sum()), 100 * frames["label_valid"].mean(),
        int((frames["label_quality"] == "ambiguous").sum()),
        int(frames["label"].isna().sum())))
    print("  quality mix: {}".format(dict(Counter(frames["label_quality"]))))
    print("  ball missing {:.1f}%   motion unreliable {:,} frames".format(
        100 * frames["q_ball_missing"].mean(), int(frames["q_ball_motion_unreliable"].sum())))
    for note in notes:
        print("  NOTE: {}".format(note))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", type=int, nargs="+", default=[1, 2, 3], choices=(1, 2, 3))
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(REPORTS, exist_ok=True)
    provenance = []

    for game in args.game:
        match_id, frames, players, events, notes, home_label = build_game(game)
        summarise(match_id, frames, players, events, notes, home_label)

        folder = os.path.join(OUT, match_id)
        os.makedirs(folder, exist_ok=True)
        frames.to_parquet(os.path.join(folder, "frames.parquet"), index=False)
        players.to_parquet(os.path.join(folder, "players.parquet"), index=False)

        for quality in sorted(set(frames["label_quality"])):
            mask = frames["label_quality"] == quality
            provenance.append({
                "source": "metrica", "match_id": match_id, "label_quality": quality,
                "label_source": "event_derived" if quality != "none" else "none",
                "n_frames": int(mask.sum()),
                "pct_of_match": round(100.0 * mask.mean(), 2),
                "eligible_for_training": quality in ("weak", "strong"),
                "eligible_for_evaluation": quality == "strong",
                "definition_note": (
                    "Metrica assigns possession on TOUCH; the competition requires CONTROL. "
                    "Dead-ball spans are law-based and strong; team spans are touch-based and weak."
                ),
                "home_mapping": home_label,
            })

    # Merge on (source, match_id), NOT on source alone: filtering by source would silently
    # delete the rows of games that this run did not touch.
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
