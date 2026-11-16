"""Snapshot the model's scores and failure counts, and compare two snapshots.

The point is to make "did that change help?" answerable without re-reading five log files. A
snapshot records the headline metrics and the failure population together, because a change that
moves spellF1 by a hundredth while halving the expensive failure type is worth knowing about, and
one that trades fifty cheap failures for five expensive ones is worth catching.

    python review/evaluate.py --label baseline
    python review/evaluate.py --compare baseline my_change

RULES THIS SCRIPT ENFORCES
  * External development matches only. The two sealed test matches are never read.
  * spellF1 differences below the organisers' own noise band (0.04) are reported as "noise", not
    as improvements. macroF1's band is 0.01.
  * Failure counts are split confirmed / contested. A change that only moves contested cases has
    not been shown to improve anything -- it has moved its agreement with one of two disagreeing
    reference products.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPDIR = os.path.join(HERE, "snapshots")
SPELL_BAND = 0.04      # possession_metrics.SPELL_NOISE_BAND
MACRO_BAND = 0.01
SEALED = {"J03WMX", "J03WN1"}


def collect(args):
    snap = {"taken": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": args.note or "",
            "scope": "external IDSSE development matches only; sealed test matches untouched"}

    s2 = pd.read_csv(args.stage2)
    hsmm = s2[s2["decoder"] == "hsmm"]
    snap["stage2"] = {
        "spellF1": float(hsmm["spell_f1_mean"].mean()),
        "stoppage_recall_iou50": float(hsmm["stoppage_recall_iou50"].mean()),
        "danger_recall": float(hsmm["danger_recall"].mean()),
        "pred_spells": float(hsmm["n_pred_spells"].mean()),
        "gt_spells": float(hsmm["n_gt_spells"].mean()),
        "by_decoder": {d: float(g["spell_f1_mean"].mean())
                       for d, g in s2.groupby("decoder")},
    }

    s3 = pd.read_csv(args.stage3)
    snap["stage3"] = {}
    for which, g in s3.groupby("which"):
        snap["stage3"][which] = {"spellF1": float(g["spellF1"].mean()),
                                 "macroF1": float(g["macroF1"].mean()),
                                 "pred_spells": float(g["pred_spells"].mean())}
        if "team_acc" in g and g["team_acc"].notna().any():
            snap["stage3"][which]["team_acc"] = float(g["team_acc"].mean())

    with open(args.index) as fh:
        idx = json.load(fh)
    bad = sorted({c["match"] for c in idx["cases"]} & SEALED)
    if bad:
        raise SystemExit("REFUSING: the case index contains sealed matches {}".format(bad))
    cases = pd.DataFrame(idx["cases"])
    # The index also carries the cases the model gets RIGHT. They are counted separately: mixing
    # them into the failure table would make a snapshot look better the moment the exporter finds
    # more successes, which is not the same thing as the model improving.
    if "outcome" in cases:
        snap["correct"] = {k: int(v) for k, v in
                           cases[cases["outcome"] == "correct"]["kind"].value_counts().items()}
        cases = cases[cases["outcome"] == "failure"].reset_index(drop=True)
    snap["failures"] = {
        "total": int(len(cases)),
        "by_kind_bucket": {k: {b: int(v) for b, v in g["bucket"].value_counts().items()}
                           for k, g in cases.groupby("kind")},
        "per_match": {m: int(v) for m, v in cases["match"].value_counts().items()},
    }
    return snap


def fmt_delta(a, b, band, higher_is_better=True):
    d = b - a
    if abs(d) < band:
        return "{:+.3f}  (noise, band {:.2f})".format(d, band)
    good = (d > 0) if higher_is_better else (d < 0)
    return "{:+.3f}  {}".format(d, "BETTER" if good else "WORSE")


def compare(a_name, b_name):
    def load(n):
        p = os.path.join(SNAPDIR, n + ".json")
        if not os.path.exists(p):
            raise SystemExit("no snapshot named '{}' in {}".format(n, SNAPDIR))
        with open(p) as fh:
            return json.load(fh)
    a, b = load(a_name), load(b_name)

    print("=== {}  ->  {} ===".format(a_name, b_name))
    print("{:<34}{:>12}{:>12}   {}".format("metric", a_name[:11], b_name[:11], "change"))
    print("-" * 86)
    print("{:<34}{:>12.3f}{:>12.3f}   {}".format(
        "Stage 2 spellF1 (hsmm)", a["stage2"]["spellF1"], b["stage2"]["spellF1"],
        fmt_delta(a["stage2"]["spellF1"], b["stage2"]["spellF1"], SPELL_BAND)))
    for key in ("stoppage_recall_iou50", "danger_recall"):
        print("{:<34}{:>12.3f}{:>12.3f}   {:+.3f}".format(
            "Stage 2 " + key, a["stage2"][key], b["stage2"][key],
            b["stage2"][key] - a["stage2"][key]))
    for which in sorted(set(a["stage3"]) | set(b["stage3"])):
        if which not in a["stage3"] or which not in b["stage3"]:
            continue
        for metric, band in (("spellF1", SPELL_BAND), ("macroF1", MACRO_BAND)):
            av, bv = a["stage3"][which][metric], b["stage3"][which][metric]
            print("{:<34}{:>12.3f}{:>12.3f}   {}".format(
                "{} {}".format(which, metric), av, bv, fmt_delta(av, bv, band)))

    print()
    print("=== failure counts, confirmed / contested ===")
    kinds = sorted(set(a["failures"]["by_kind_bucket"]) | set(b["failures"]["by_kind_bucket"]))
    print("{:<18}{:>18}{:>18}{:>14}".format("kind", a_name[:16], b_name[:16], "confirmed +/-"))
    print("-" * 70)
    for k in kinds:
        ka = a["failures"]["by_kind_bucket"].get(k, {})
        kb = b["failures"]["by_kind_bucket"].get(k, {})
        ac, bc = ka.get("confirmed", 0), kb.get("confirmed", 0)
        print("{:<18}{:>18}{:>18}{:>14}".format(
            k,
            "{} / {}".format(ac, ka.get("contested", 0)),
            "{} / {}".format(bc, kb.get("contested", 0)),
            "{:+d}".format(bc - ac)))
    print()
    print("A change that only moves the contested column has not been shown to improve anything.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", help="name this snapshot and save it")
    ap.add_argument("--note", default="", help="one line on what changed")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--stage2", default="D:/Football Hackathon/stage2/res_dur_logkde.csv")
    ap.add_argument("--stage3", default="D:/Football Hackathon/stage3/results_review.csv")
    ap.add_argument("--index", default="D:/Football Hackathon/review/data/index.json")
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)
    if not args.label:
        ap.error("give --label to save a snapshot, or --compare BEFORE AFTER")

    snap = collect(args)
    os.makedirs(SNAPDIR, exist_ok=True)
    path = os.path.join(SNAPDIR, args.label + ".json")
    with open(path, "w") as fh:
        json.dump(snap, fh, indent=1)

    print("Stage 2 spellF1 (hsmm)        {:.3f}".format(snap["stage2"]["spellF1"]))
    print("  stoppage recall @ IoU 0.5   {:.3f}".format(snap["stage2"]["stoppage_recall_iou50"]))
    print("  short same-team recall      {:.3f}".format(snap["stage2"]["danger_recall"]))
    for which in ("stage2_only", "stage3_only", "end_to_end"):
        if which in snap["stage3"]:
            v = snap["stage3"][which]
            print("{:<30}spellF1 {:.3f}   macroF1 {:.3f}".format(
                which, v["spellF1"], v["macroF1"]))
    print()
    if snap.get("correct"):
        print("correct cases: " + ", ".join("{} {}".format(k, v)
                                            for k, v in sorted(snap["correct"].items())))
    print("failures: {} total".format(snap["failures"]["total"]))
    for k, v in sorted(snap["failures"]["by_kind_bucket"].items()):
        print("  {:<16} confirmed {:>5}   contested {:>5}".format(
            k, v.get("confirmed", 0), v.get("contested", 0)))
    print("\nwrote " + path)


if __name__ == "__main__":
    main()
