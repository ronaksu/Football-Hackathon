"""Before/after comparison on the metrics the brief asks for, from saved per-frame predictions.

For each saved prediction set: spellF1, macroF1, dead-ball precision and recall, the error
counts by type, and a check that genuine short possessions survive -- a detector can always buy
a tidy spell count by smoothing the short ones away, and that has to be ruled out explicitly.
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

FPS = 25
SCORER = ("D:/Football Hackathon/more-than-a-hack-2027-main/more-than-a-hack-2027-main/"
          "evaluation/possession_metrics.py")


def load_scorer():
    spec = importlib.util.spec_from_file_location("possession_metrics", SCORER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["possession_metrics"] = mod
    spec.loader.exec_module(mod)
    return mod


def runs_of(mask):
    m = np.asarray(mask).astype(np.int8)
    e = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def hybrid(pm, labels, pred_dead, periods):
    gt, pr = [], []
    for p in sorted(pd.unique(periods)):
        m = periods == p
        t = np.asarray(labels[m], dtype=object)
        d = np.asarray(pred_dead[m])
        if gt:
            gt.append(np.array(["X"]))
            pr.append(np.array(["X"]))
        gt.append(t)
        pr.append(np.where(d, "D", t))
    return pm.label_codes(np.concatenate(gt)), pm.label_codes(np.concatenate(pr))


def spell_durations(labels, pred_dead, periods):
    """Possession-spell lengths in seconds, for truth and for the hybrid prediction."""
    gt_d, pr_d = [], []
    for p in sorted(pd.unique(periods)):
        m = periods == p
        t = np.asarray(labels[m], dtype=object)
        pred = np.where(np.asarray(pred_dead[m]), "D", t)
        for seq, out in ((t, gt_d), (pred, pr_d)):
            cur, n = None, 0
            for v in seq:
                if v == cur:
                    n += 1
                else:
                    if cur in ("H", "A"):
                        out.append(n / FPS)
                    cur, n = v, 1
            if cur in ("H", "A"):
                out.append(n / FPS)
    return np.array(gt_d), np.array(pr_d)


def summarise(pm, pred_dir, decoder="hsmm"):
    tot = {"tp": 0, "fp": 0, "fn": 0}
    spells, macros = [], []
    gt_all, pr_all = [], []
    gt_dur, pr_dur = [], []

    for f in sorted(os.listdir(pred_dir)):
        if not f.endswith(".parquet"):
            continue
        d = pd.read_parquet(os.path.join(pred_dir, f))
        labels = d["label"].to_numpy()
        pred = d["pred_" + decoder].to_numpy().astype(bool)
        per = d["period"].to_numpy()

        truth = labels == "D"
        tot["tp"] += int((truth & pred).sum())
        tot["fp"] += int((~truth & pred).sum())
        tot["fn"] += int((truth & ~pred).sum())

        g, p = hybrid(pm, labels, pred, per)
        r = pm.evaluate(g, p)
        spells.append(r["spell_f1_mean"])
        macros.append(r["macro_f1"])
        gt_all.append(r["n_gt_spells"])
        pr_all.append(r["n_pred_spells"])

        a, b = spell_durations(labels, pred, per)
        gt_dur.append(a)
        pr_dur.append(b)

    prec = tot["tp"] / max(1, tot["tp"] + tot["fp"])
    rec = tot["tp"] / max(1, tot["tp"] + tot["fn"])
    gt_dur = np.concatenate(gt_dur)
    pr_dur = np.concatenate(pr_dur)

    return {
        "spellF1": np.mean(spells),
        "macroF1": np.mean(macros),
        "dead_precision": prec,
        "dead_recall": rec,
        "dead_f1": 2 * prec * rec / max(1e-9, prec + rec),
        "gt_spells": np.mean(gt_all),
        "pred_spells": np.mean(pr_all),
        "gt_short_lt3s": int((gt_dur < 3).sum()),
        "pred_short_lt3s": int((pr_dur < 3).sum()),
        "gt_median_spell_s": float(np.median(gt_dur)),
        "pred_median_spell_s": float(np.median(pr_dur)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="+", required=True,
                    help="name=dir pairs, e.g. base=stage2/predictions")
    ap.add_argument("--decoder", default="hsmm")
    args = ap.parse_args()

    pm = load_scorer()
    rows = {}
    for spec in args.sets:
        name, path = spec.split("=", 1)
        rows[name] = summarise(pm, path, args.decoder)

    t = pd.DataFrame(rows).T
    order = ["spellF1", "macroF1", "dead_precision", "dead_recall", "dead_f1",
             "pred_spells", "gt_spells", "pred_short_lt3s", "gt_short_lt3s",
             "pred_median_spell_s", "gt_median_spell_s"]
    print("decoder: {}\n".format(args.decoder))
    print(t[order].to_string(float_format=lambda v: "{:.3f}".format(v)))


if __name__ == "__main__":
    main()
