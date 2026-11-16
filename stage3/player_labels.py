"""Derive per-frame PLAYER-level possession labels from the DFL event feed.

Why this exists. Four attempts at team attribution all landed at spellF1 0.428-0.432 with team
accuracy stuck in a 0.843-0.856 band. The one material difference left between our path model and
the published PathCRF method is that theirs trains its emissions on per-frame *player-level*
possession; ours are hand-set. The DFL event feed names the player on every on-ball action, so
those labels are derivable here.

Three things have to be got right.

1. SLOT MAPPING. processed/idsse/*/players.parquet stores an anonymous `slot`, which parse_dfl.py
   assigned as the document order of FrameSets per (half, side). That order is replayed here to
   recover slot -> PersonId, so event players can be matched to tracked slots.

2. SYNCHRONISATION. The data descriptor is explicit that event timestamps and tracking are
   produced separately: the mean offset is -0.37 s with a standard deviation of 1.82 s and a
   maximum of 27 s. Possessions average 3.9 s, so raw timestamps would inject noise at the same
   scale as the quantity being labelled. PathCRF uses ELASTIC for this; here each event is snapped
   to the frame within a search window that minimises the distance between the ball and the named
   player, which is the standard cost-function approach.

3. THE CONTROL CONVENTION. The DFL defines possession by the player in control. A TacklingGame
   carries WinnerRole / LoserRole of withBallControl or withoutBallControl and an explicit
   PossessionChange flag, so a duel only moves possession when the feed says it did.

Output: one parquet per match under stage3/player_labels/ with the holder's slot and team per
frame, plus a validation of the derived team against the DFL's own BallPossession field.
"""

import argparse
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import pandas as pd

SEGMENTS = {"firstHalf": 1, "secondHalf": 2}
FPS = 25
SYNC_WINDOW_S = 3.0          # how far either side of the stated timestamp to look
PITCH_DEFAULT = (105.0, 68.0)


def match_files(raw_dir, match_id):
    short = match_id.replace("DFL-MAT-", "")
    out = {}
    for f in os.listdir(raw_dir):
        if short not in f:
            continue
        if "matchinformation" in f:
            out["info"] = os.path.join(raw_dir, f)
        elif "events" in f:
            out["events"] = os.path.join(raw_dir, f)
        elif "positions" in f:
            out["positions"] = os.path.join(raw_dir, f)
    return out


def read_teams(path):
    root = ET.parse(path).getroot()
    teams = {}
    for team in root.iter("Team"):
        role = (team.get("Role") or "").lower()
        if role in ("home", "guest"):
            teams["home" if role == "home" else "away"] = team.get("TeamId")
    env = root.find(".//Environment")
    pitch = PITCH_DEFAULT
    if env is not None and env.get("PitchX"):
        pitch = (float(env.get("PitchX")), float(env.get("PitchY")))
    return teams, pitch


def slot_map(positions_path, home_id, away_id):
    """(period, side, slot) -> PersonId, replaying parse_dfl.py's FrameSet document order."""
    order = defaultdict(list)
    frame_times = {}
    for _, element in ET.iterparse(positions_path, events=("end",)):
        if not element.tag.endswith("FrameSet"):
            continue
        segment = element.get("GameSection")
        team_id = (element.get("TeamId") or "").upper()
        person = element.get("PersonId")
        period = SEGMENTS.get(segment)
        if period is not None:
            if team_id == "BALL":
                # Keep the ball's frame clock: it is the timeline everything is snapped to.
                rows = [(int(f.get("N")), f.get("T")) for f in element]
                frame_times[period] = rows
            elif team_id in (home_id.upper(), away_id.upper()):
                side = "home" if team_id == home_id.upper() else "away"
                order[(period, side)].append(person)
        element.clear()

    mapping = {}
    for (period, side), persons in order.items():
        for slot, person in enumerate(persons):
            mapping[(period, side, slot)] = person
    return mapping, frame_times


def parse_events(events_path, teams):
    """[(iso, period_hint, player, team_side, kind)] for every on-ball action."""
    id_to_side = {teams["home"]: "home", teams["away"]: "away"}
    out = []
    for _, element in ET.iterparse(events_path, events=("end",)):
        if not element.tag.endswith("Event"):
            continue
        iso = element.get("EventTime")
        # Set-piece restarts carry the taker on a NESTED Play (ThrowIn/Play, FreeKick/Play,
        # GoalKick/Play, CornerKick/Play), so descendants are walked rather than just children.
        # Those are the most precisely timed possession starts in the whole feed, and reading
        # only the immediate children dropped about ninety of them per match.
        children = [c for c in element.iter() if c is not element]
        if any(c.tag.split("}")[-1] == "Delete" for c in children):
            element.clear()
            continue
        for child in children:
            tag = child.tag.split("}")[-1]
            if tag == "TacklingGame":
                # Possession only moves when the feed says it did; otherwise the player who had
                # control keeps it.
                changed = (child.get("PossessionChange") or "").lower() == "true"
                if changed:
                    person, team = child.get("Winner"), child.get("WinnerTeam")
                else:
                    if (child.get("LoserRole") or "") == "withBallControl":
                        person, team = child.get("Loser"), child.get("LoserTeam")
                    else:
                        person, team = child.get("Winner"), child.get("WinnerTeam")
                if person and team in id_to_side:
                    out.append((iso, person, id_to_side[team], tag))
            elif child.get("Player"):
                team = child.get("Team")
                if team in id_to_side:
                    out.append((iso, child.get("Player"), id_to_side[team], tag))
        element.clear()
    return out


def iso_to_seconds(iso):
    # 2023-05-27T13:30:12.600+00:00 -> seconds within the day, in UTC
    m = re.match(r".*T(\d\d):(\d\d):(\d\d)(?:\.(\d+))?([+-]\d\d):(\d\d)", iso or "")
    if not m:
        return np.nan
    h, mi, s, frac, oh, om = m.groups()
    base = int(h) * 3600 + int(mi) * 60 + int(s) + (float("0." + frac) if frac else 0.0)
    offset = int(oh) * 3600 + (1 if int(oh) >= 0 else -1) * int(om) * 60
    return base - offset


def build_match(match_id, raw_dir, proc_dir, out_dir):
    paths = match_files(raw_dir, match_id)
    teams, pitch = read_teams(paths["info"])
    mapping, frame_times = slot_map(paths["positions"], teams["home"], teams["away"])
    events = parse_events(paths["events"], teams)

    players = pd.read_parquet(os.path.join(proc_dir, match_id, "players.parquet"))
    frames = pd.read_parquet(os.path.join(proc_dir, match_id, "frames.parquet"),
                             columns=["period", "frame_native", "label", "label_valid",
                                      "ball_x", "ball_y", "ball_observed"])

    # person -> (side, slot), per period
    person_slot = {}
    for (period, side, slot), person in mapping.items():
        person_slot[(period, person)] = (side, slot)

    parts = []
    for period in sorted(frames["period"].unique()):
        rows = frame_times.get(period)
        if not rows:
            continue
        rows.sort()
        ns = np.array([r[0] for r in rows])
        ts = np.array([iso_to_seconds(r[1]) for r in rows])

        fsub = frames[frames["period"] == period].sort_values("frame_native")
        psub = players[players["period"] == period]
        wide_x = psub.pivot_table(index="frame_native", columns=["team", "slot"],
                                  values="x", aggfunc="first")
        wide_y = psub.pivot_table(index="frame_native", columns=["team", "slot"],
                                  values="y", aggfunc="first")
        wide_x = wide_x.reindex(ns)
        wide_y = wide_y.reindex(ns).reindex(columns=wide_x.columns)
        bx = fsub.set_index("frame_native").reindex(ns)["ball_x"].to_numpy() * pitch[0]
        by = fsub.set_index("frame_native").reindex(ns)["ball_y"].to_numpy() * pitch[1]

        holder = np.full(len(ns), -1, dtype=np.int32)
        holder_side = np.full(len(ns), "", dtype=object)

        win = int(SYNC_WINDOW_S * FPS)
        marks = []
        for iso, person, side, kind in events:
            key = (period, person)
            if key not in person_slot:
                continue
            _, slot = person_slot[key]
            if (side, slot) not in wide_x.columns:
                continue
            t = iso_to_seconds(iso)
            if not np.isfinite(t):
                continue
            i = int(np.searchsorted(ts, t))
            if i <= 0 or i >= len(ns):
                continue
            lo, hi = max(0, i - win), min(len(ns), i + win)
            px = wide_x[(side, slot)].to_numpy()[lo:hi] * pitch[0]
            py = wide_y[(side, slot)].to_numpy()[lo:hi] * pitch[1]
            d = np.hypot(px - bx[lo:hi], py - by[lo:hi])
            if not np.isfinite(d).any():
                continue
            j = lo + int(np.nanargmin(d))
            marks.append((j, slot, side))

        marks.sort()
        for k, (j, slot, side) in enumerate(marks):
            end = marks[k + 1][0] if k + 1 < len(marks) else len(ns)
            holder[j:end] = slot
            holder_side[j:end] = side

        parts.append(pd.DataFrame({
            "period": period, "frame_native": ns,
            "holder_slot": holder, "holder_side": holder_side,
            "n_events_synced": len(marks),
        }))

    out = pd.concat(parts, ignore_index=True)
    merged = frames.merge(out, on=["period", "frame_native"], how="left")
    merged = merged[merged["label_valid"].astype(bool)]

    # Validation: does the team of the derived holder agree with the DFL's own label?
    alive = merged[merged["label"].isin(["H", "A"])]
    known = alive[alive["holder_side"].isin(["home", "away"])]
    agree = ((known["holder_side"] == "home") == (known["label"] == "H")).mean()
    cover = len(known) / max(1, len(alive))

    os.makedirs(out_dir, exist_ok=True)
    out.to_parquet(os.path.join(out_dir, match_id + ".parquet"), index=False)
    print("  {}  events synced {:>5,}   holder known on {:.1%} of live frames   "
          "team agrees with DFL label {:.1%}".format(
              match_id.replace("DFL-MAT-", ""), int(out["n_events_synced"].max()), cover, agree))
    return agree, cover


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="D:/Football Hackathon/external_data/raw/idsse")
    ap.add_argument("--proc-dir", default="D:/Football Hackathon/external_data/processed/idsse")
    ap.add_argument("--out-dir", default="D:/Football Hackathon/stage3/player_labels")
    ap.add_argument("--matches", nargs="*", default=None)
    args = ap.parse_args()

    ids = args.matches or sorted(
        d for d in os.listdir(args.proc_dir)
        if os.path.exists(os.path.join(args.proc_dir, d, "frames.parquet")))
    print("deriving player-level possession for {} matches".format(len(ids)))
    for m in ids:
        build_match(m, args.raw_dir, args.proc_dir, args.out_dir)


if __name__ == "__main__":
    main()
