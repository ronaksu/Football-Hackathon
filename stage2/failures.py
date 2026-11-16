"""Catalogue every place the Stage 2 detector disagrees with the truth, sorted by what it costs.

Four error types, in descending order of damage to spellF1:

  SPELL_MERGE    a true stoppage was missed, and the SAME team had the ball either side. Two real
                 possessions fuse into one that never happened. The most expensive error there is.
  SPELL_SPLIT    a stoppage was invented inside a real possession. One possession becomes two.
  MISSED_STOP    a true stoppage was missed, but different teams had the ball either side. The
                 possessions survive; only the boundary between them moves. Cheap.
  BOUNDARY       the stoppage was found but starts or ends more than a second out.

Writes a CSV that can be read match by match, with the clock time of every case, so the frames
can be looked at directly.
"""

import argparse
import os

import numpy as np
import pandas as pd

FPS = 25
PROC = "D:/Football Hackathon/external_data/processed/idsse"


def runs_of(mask):
    m = np.asarray(mask).astype(np.int8)
    e = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def clock(frame_idx, period, time_s):
    """mm:ss on the match clock, using the period's own start as zero."""
    t = time_s[frame_idx] - time_s[0]
    base = 0 if period == 1 else 45 * 60
    total = base + t
    return "{:d}:{:02d}".format(int(total // 60), int(total % 60))


def overlap_frac(a, b, mask):
    """Fraction of [a,b) where mask is True."""
    return float(np.mean(mask[a:b])) if b > a else 0.0


def analyse_match(pred_path, decoder):
    df = pd.read_parquet(pred_path)
    match = df["match_id"].iloc[0]

    frames = pd.read_parquet(
        os.path.join(PROC, match, "frames.parquet"),
        columns=["period", "frame_native", "time_s", "label_valid"],
    )
    df = df.merge(frames, on=["period", "frame_native"], how="left")
    df = df[df["label_valid"].astype(bool)].reset_index(drop=True)

    cases = []
    for period, g in df.groupby("period", sort=True):
        g = g.sort_values("frame_native").reset_index(drop=True)
        lab = g["label"].to_numpy()
        pred = g["pred_" + decoder].to_numpy().astype(bool)
        prob = g["prob"].to_numpy()
        fn = g["frame_native"].to_numpy()
        ts = g["time_s"].to_numpy()
        truth = lab == "D"

        # --- missed stoppages, split by what sits either side
        for a, b in runs_of(truth):
            cov = overlap_frac(a, b, pred)
            if cov >= 0.5:
                continue
            before = lab[a - 1] if a > 0 else None
            after = lab[b] if b < len(lab) else None
            same = before is not None and after is not None and before == after
            cases.append({
                "match": match.replace("DFL-MAT-", ""), "period": period,
                "type": "SPELL_MERGE" if same else "MISSED_STOP",
                "clock": clock(a, period, ts), "duration_s": round((b - a) / FPS, 1),
                "frame_start": int(fn[a]), "frame_end": int(fn[b - 1]),
                "team_before": before, "team_after": after,
                "covered": round(cov, 2), "mean_prob": round(float(prob[a:b].mean()), 3),
                "peak_prob": round(float(prob[a:b].max()), 3),
            })

        # --- invented stoppages inside a live possession
        for a, b in runs_of(pred):
            cov = overlap_frac(a, b, truth)
            if cov >= 0.5:
                continue
            before = lab[a - 1] if a > 0 else None
            after = lab[b] if b < len(lab) else None
            same = before is not None and after is not None and before == after and before != "D"
            # A predicted stoppage that touches no true stoppage at all is invented out of
            # nothing. One that overlaps a real stoppage but spills past it on either side is a
            # different fault -- the stoppage was found, then allowed to swallow the live play
            # around it. They need different fixes, so they get different names.
            if cov > 0:
                kind = "OVER_EXTENDED"
            else:
                kind = "SPELL_SPLIT" if same else "PHANTOM_STOP"
            cases.append({
                "match": match.replace("DFL-MAT-", ""), "period": period,
                "type": kind,
                "clock": clock(a, period, ts), "duration_s": round((b - a) / FPS, 1),
                "frame_start": int(fn[a]), "frame_end": int(fn[b - 1]),
                "team_before": before, "team_after": after,
                "covered": round(cov, 2), "mean_prob": round(float(prob[a:b].mean()), 3),
                "peak_prob": round(float(prob[a:b].max()), 3),
            })

        # --- boundary errors on stoppages that WERE found
        pred_spans = runs_of(pred)
        for a, b in runs_of(truth):
            if overlap_frac(a, b, pred) < 0.5:
                continue
            best, bi = 0.0, None
            for c, d in pred_spans:
                lo, hi = max(a, c), min(b, d)
                if hi > lo and (hi - lo) > best:
                    best, bi = hi - lo, (c, d)
            if bi is None:
                continue
            ds, de = (bi[0] - a) / FPS, (bi[1] - b) / FPS
            if abs(ds) < 1.0 and abs(de) < 1.0:
                continue
            cases.append({
                "match": match.replace("DFL-MAT-", ""), "period": period,
                "type": "BOUNDARY", "clock": clock(a, period, ts),
                "duration_s": round((b - a) / FPS, 1),
                "frame_start": int(fn[a]), "frame_end": int(fn[b - 1]),
                "team_before": lab[a - 1] if a > 0 else None,
                "team_after": lab[b] if b < len(lab) else None,
                "covered": round(overlap_frac(a, b, pred), 2),
                "mean_prob": round(float(prob[a:b].mean()), 3),
                "peak_prob": round(float(prob[a:b].max()), 3),
                "start_err_s": round(ds, 1), "end_err_s": round(de, 1),
            })

    return pd.DataFrame(cases)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", default="D:/Football Hackathon/stage2/predictions")
    ap.add_argument("--decoder", default="hsmm", choices=["raw", "smooth", "hsmm"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or "D:/Football Hackathon/stage2/failures_{}.csv".format(args.decoder)

    parts = []
    for f in sorted(os.listdir(args.pred_dir)):
        if f.endswith(".parquet"):
            parts.append(analyse_match(os.path.join(args.pred_dir, f), args.decoder))
    cases = pd.concat(parts, ignore_index=True)

    order = {"SPELL_MERGE": 0, "SPELL_SPLIT": 1, "OVER_EXTENDED": 2, "MISSED_STOP": 3,
             "PHANTOM_STOP": 4, "BOUNDARY": 5}
    cases["rank"] = cases["type"].map(order)
    cases = cases.sort_values(["rank", "duration_s"], ascending=[True, True]).drop(columns="rank")
    cases.to_csv(out, index=False)

    print("decoder: {}   cases: {:,}   across {} matches".format(
        args.decoder, len(cases), cases["match"].nunique()))
    print()
    print("=== by type, over 5 development matches ===")
    summary = cases.groupby("type").agg(
        n=("type", "size"),
        per_match=("type", lambda s: round(len(s) / 5.0, 1)),
        median_s=("duration_s", "median"),
        median_peak_prob=("peak_prob", "median"),
    ).reindex([k for k in order if k in set(cases["type"])])
    print(summary.to_string())
    print()
    print("=== the expensive ones, every case ===")
    bad = cases[cases["type"].isin(["SPELL_MERGE", "SPELL_SPLIT"])]
    cols = ["match", "period", "clock", "type", "duration_s", "team_before", "team_after",
            "covered", "peak_prob", "frame_start", "frame_end"]
    print(bad[cols].to_string(index=False) if len(bad) else "  none")
    print()
    print("wrote " + out)


if __name__ == "__main__":
    main()
