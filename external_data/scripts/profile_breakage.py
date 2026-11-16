"""Which stoppages break a possession if you miss them?

spellF1 matches H and A possession spells one-to-one by temporal overlap. A dead interval is not
itself scored -- but it decides where the spells either side begin and end. So the cost of getting
a stoppage wrong depends entirely on what sits on each side of it:

  H -> D -> H   missing this D MERGES two Home spells into one. Expensive: two true spells now
                face one prediction, so one of them cannot be matched at all.
  H -> D -> A   missing this D only shifts a boundary. Both spells still exist. Cheap.

This counts how many stoppages are of the expensive kind, and how long they are, so the model can
be pointed at the ones that actually cost ranking places.
"""

import os

import numpy as np
import pandas as pd

PROC = "D:/Football Hackathon/external_data/processed/idsse"


def spans(labels, fps):
    """[(label, start_s, end_s)] for maximal runs."""
    out, cur, i0 = [], labels[0], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != cur:
            out.append((cur, i0 / fps, i / fps))
            if i < len(labels):
                cur, i0 = labels[i], i
    return out


def main():
    same, diff, edge = [], [], 0

    for d in sorted(os.listdir(PROC)):
        f = os.path.join(PROC, d, "frames.parquet")
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f, columns=["period", "fps", "label", "label_valid"])
        df = df[df["label_valid"].astype(bool)]
        if df.empty:
            continue
        fps = float(df["fps"].iloc[0])

        for _, g in df.groupby("period", sort=True):
            sp = spans(g["label"].tolist(), fps)
            for k, (lab, a, b) in enumerate(sp):
                if lab != "D":
                    continue
                before = sp[k - 1][0] if k > 0 else None
                after = sp[k + 1][0] if k + 1 < len(sp) else None
                if before is None or after is None:
                    edge += 1
                elif before == after:
                    same.append(b - a)
                else:
                    diff.append(b - a)

    s, t = pd.Series(same), pd.Series(diff)
    tot = len(s) + len(t)
    print("=== stoppages classified by what sits either side ===")
    print("total scored stoppages        {:,}   (+{} at the edge of a half)".format(tot, edge))
    print()
    print("SAME team either side (H-D-H or A-D-A)  {:,}  = {:.1f}%".format(
        len(s), 100 * len(s) / tot))
    print("   -> missing one MERGES two spells. This is the expensive error.")
    print("   median {:.1f} s | p25 {:.1f} s | under 10 s: {:.1f}% of them".format(
        s.median(), s.quantile(0.25), 100 * (s < 10).mean()))
    print()
    print("DIFFERENT team either side (H-D-A)      {:,}  = {:.1f}%".format(
        len(t), 100 * len(t) / tot))
    print("   -> missing one only shifts a boundary. Cheap.")
    print("   median {:.1f} s | p25 {:.1f} s | under 10 s: {:.1f}% of them".format(
        t.median(), t.quantile(0.25), 100 * (t < 10).mean()))
    print()

    short_same = s[s < 10]
    print("=== the danger zone ===")
    print("stoppages that are BOTH same-team AND under 10 s: {:,}".format(len(short_same)))
    print("that is {:.1f}% of all stoppages, but they carry only {:.1f}% of total dead time".format(
        100 * len(short_same) / tot,
        100 * short_same.sum() / (s.sum() + t.sum())))
    print()
    print("So they are nearly invisible to a per-frame score, and every one of them that is")
    print("missed costs a whole possession spell. {:.0f} per match, on average.".format(
        len(short_same) / 7))


if __name__ == "__main__":
    main()
