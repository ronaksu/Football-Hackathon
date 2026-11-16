"""Measure what SoccerNet-GAR's tracking actually covers, from its own metadata.

Question this answers: the corpus ships tracking as short clips cut around touches. Touches stop
happening when play stops. So how much of a match -- and how much of its DEAD time -- has any
tracking at all?

Reads only the downloaded metadata.parquet. Nothing is inferred from the dataset card.
"""

import json
import sys
from collections import Counter, defaultdict

import pandas as pd

SRC = ("D:/Football Hackathon/external_data/raw/soccernet_gar_probe/"
       "sngar_tracking-full_test_metadata.parquet")

# From the dataset card and confirmed by the payload: each clip is 144 native frames at
# ~29.97 fps, centred on the event timestamp.
CLIP_S = 144 / 29.97
HALF = CLIP_S / 2.0

# A stoppage runs from the ball leaving play to the next restart.
STOP_ACTIONS = {"OUT"}
RESTART_ACTIONS = {"THROW IN", "FREE KICK", "CROSS", "GOAL"}


def load():
    df = pd.read_parquet(SRC)
    rows = []
    for payload in df["sample_payload"]:
        d = json.loads(payload)
        m = d.get("metadata", {})
        rows.append({
            "game_id": str(m.get("game_id")),
            "t": m.get("position_ms", 0) / 1000.0,
            "half": (m.get("game_time") or "? - ").split(" - ")[0].strip(),
            "action": d["labels"]["action"]["label"],
            "team": m.get("team"),
            "fps": d["inputs"][0].get("fps"),
        })
    return pd.DataFrame(rows)


def merge(intervals):
    """Union of [start, end] intervals."""
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def total(intervals):
    return sum(b - a for a, b in intervals)


def overlap(intervals, a, b):
    """How much of [a,b] is inside the merged interval list."""
    s = 0.0
    for x, y in intervals:
        lo, hi = max(x, a), min(y, b)
        if hi > lo:
            s += hi - lo
    return s


def main():
    df = load()
    print("rows                 {:,}".format(len(df)))
    print("matches              {}".format(df["game_id"].nunique()))
    print("fps values           {}".format(sorted(set(df['fps'].dropna()))))
    print("actions              {}".format(dict(Counter(df['action']).most_common())))
    print("team values          {}".format(dict(Counter(df['team'].fillna('<null>')))))
    print()

    per_match = []
    stop_cov, stop_tot, stop_n = 0.0, 0.0, 0
    gap_lengths = []

    for gid, g in df.groupby("game_id"):
        g = g.sort_values("t")
        ts = g["t"].to_numpy()
        acts = g["action"].tolist()

        span = ts[-1] - ts[0]
        if span <= 0:
            continue

        clips = merge([[t - HALF, t + HALF] for t in ts])
        covered = total(clips)

        # inter-event gaps, measured between clip edges
        for i in range(1, len(ts)):
            gap = (ts[i] - HALF) - (ts[i - 1] + HALF)
            if gap > 0:
                gap_lengths.append(gap)

        # stoppages: OUT -> next restart
        for i, a in enumerate(acts):
            if a not in STOP_ACTIONS:
                continue
            for j in range(i + 1, len(acts)):
                if acts[j] in RESTART_ACTIONS:
                    a0, b0 = ts[i], ts[j]
                    if b0 > a0:
                        stop_n += 1
                        stop_tot += b0 - a0
                        stop_cov += overlap(clips, a0, b0)
                    break

        per_match.append({
            "game_id": gid, "events": len(g), "span_s": span,
            "covered_s": covered, "pct": 100.0 * covered / span,
        })

    pm = pd.DataFrame(per_match)
    print("=== tracking coverage of elapsed match time ===")
    print("matches measured     {}".format(len(pm)))
    print("events per match     median {:.0f}   min {}   max {}".format(
        pm["events"].median(), pm["events"].min(), pm["events"].max()))
    print("span per match       median {:.1f} min".format(pm["span_s"].median() / 60))
    print("covered by clips     median {:.1f}%   min {:.1f}%   max {:.1f}%".format(
        pm["pct"].median(), pm["pct"].min(), pm["pct"].max()))
    print("UNCOVERED            median {:.1f}% of elapsed match time has NO tracking".format(
        100 - pm["pct"].median()))
    print()

    gl = pd.Series(gap_lengths)
    print("=== holes between clips (seconds with no tracking at all) ===")
    print("number of holes      {:,}".format(len(gl)))
    print("total hole time      {:,.0f} s across {} matches".format(gl.sum(), len(pm)))
    print("hole length          median {:.1f}s   p90 {:.1f}s   max {:.1f}s".format(
        gl.median(), gl.quantile(0.90), gl.max()))
    print("holes over 10s       {:,}  ({:.1f}% of holes)".format(
        (gl > 10).sum(), 100.0 * (gl > 10).mean()))
    print()

    print("=== the decisive number: dead-ball coverage ===")
    print("stoppages found      {:,}   (OUT -> next restart)".format(stop_n))
    print("total dead time      {:,.0f} s".format(stop_tot))
    print("of which tracked     {:,.0f} s  =  {:.1f}%".format(
        stop_cov, 100.0 * stop_cov / stop_tot if stop_tot else 0))
    print("UNTRACKED dead time  {:.1f}%".format(
        100 - (100.0 * stop_cov / stop_tot if stop_tot else 0)))


if __name__ == "__main__":
    sys.exit(main())
