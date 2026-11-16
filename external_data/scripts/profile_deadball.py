"""Characterise the dead ball in the IDSSE matches, and validate our labels against the DFL's own
official net playing time.

Two things this settles.

1. VALIDATION. The match information file carries the DFL's own TotalTime and PlayingTime per
   half -- the official gross and net playing time. If the dead share we derive from the
   per-frame BallStatus agrees with 1 - net/gross, then our parsing is right AND BallStatus means
   what the league means by "ball in play".

2. SHAPE. How long are stoppages, how many are there, and how much of the dead time sits in short
   stoppages versus long ones. This drives the duration distributions in the sequence model, and
   it says where spell boundaries will actually be won or lost.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

BASE = "D:/Football Hackathon/external_data"
PROC = os.path.join(BASE, "processed", "idsse")
RAW = os.path.join(BASE, "raw", "idsse")


def official_times(match_id):
    """Gross and net playing time, in seconds, from the DFL match information file."""
    hits = glob.glob(os.path.join(RAW, "*matchinformation*{}.xml".format(match_id.split("-")[-1])))
    if not hits:
        return None
    text = open(hits[0], encoding="utf-8").read()
    m = re.search(r"<OtherGameInformation([^/]*)/>", text)
    if not m:
        return None
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
    try:
        gross = (int(attrs["TotalTimeFirstHalf"]) + int(attrs["TotalTimeSecondHalf"])) / 1000.0
        net = (int(attrs["PlayingTimeFirstHalf"]) + int(attrs["PlayingTimeSecondHalf"])) / 1000.0
    except KeyError:
        return None
    return gross, net


def intervals(mask, fps):
    """Lengths in seconds of maximal True runs."""
    out, n = [], 0
    for v in mask:
        if v:
            n += 1
        elif n:
            out.append(n / fps)
            n = 0
    if n:
        out.append(n / fps)
    return out


def main():
    rows, all_dead, all_live = [], [], []

    for d in sorted(os.listdir(PROC)):
        f = os.path.join(PROC, d, "frames.parquet")
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f, columns=["match_id", "period", "fps", "label", "label_valid"])
        df = df[df["label_valid"].astype(bool)]
        if df.empty:
            continue
        fps = float(df["fps"].iloc[0])

        dead_frac = (df["label"] == "D").mean()

        dead_iv, live_iv = [], []
        for _, g in df.groupby("period", sort=True):
            lab = g["label"].to_numpy()
            dead_iv += intervals(lab == "D", fps)
            live_iv += intervals(lab != "D", fps)
        all_dead += dead_iv
        all_live += live_iv

        ot = official_times(d)
        off_dead = (1 - ot[1] / ot[0]) if ot else np.nan

        rows.append({
            "match": d.replace("DFL-MAT-", ""),
            "frames": len(df),
            "dead_labels": 100 * dead_frac,
            "dead_official": 100 * off_dead if ot == ot and ot else np.nan,
            "stoppages": len(dead_iv),
            "median_stop_s": np.median(dead_iv) if dead_iv else np.nan,
        })

    t = pd.DataFrame(rows)
    t["diff_pp"] = t["dead_labels"] - t["dead_official"]
    print("=== 1. our labels vs the DFL's own official net playing time ===")
    print(t.to_string(index=False, float_format=lambda v: "{:.1f}".format(v)))
    print()
    print("mean absolute disagreement: {:.2f} percentage points".format(
        t["diff_pp"].abs().mean()))
    print()

    d = pd.Series(all_dead)
    l = pd.Series(all_live)
    print("=== 2. how long is a stoppage? ({:,} stoppages across 7 matches) ===".format(len(d)))
    for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99):
        print("  p{:<4.0f} {:>8.1f} s".format(q * 100, d.quantile(q)))
    print("  max   {:>8.1f} s".format(d.max()))
    print()
    print("stoppages under 5 s   {:>5.1f}% of all stoppages, but only {:>4.1f}% of dead time".format(
        100 * (d < 5).mean(), 100 * d[d < 5].sum() / d.sum()))
    print("stoppages over 30 s   {:>5.1f}% of all stoppages, and      {:>4.1f}% of dead time".format(
        100 * (d > 30).mean(), 100 * d[d > 30].sum() / d.sum()))
    print()
    print("=== 3. how long is a passage of live play? ({:,} passages) ===".format(len(l)))
    for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99):
        print("  p{:<4.0f} {:>8.1f} s".format(q * 100, l.quantile(q)))
    print("  max   {:>8.1f} s".format(l.max()))
    print()
    print("A geometric (memoryless) model would make the SHORTEST stoppage the most likely one.")
    print("Observed mode is around {:.0f} s, so dwell time must be modelled explicitly.".format(
        d.quantile(0.5)))


if __name__ == "__main__":
    main()
