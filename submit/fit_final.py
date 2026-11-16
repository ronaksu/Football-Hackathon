"""Fit the shipped model once, on every development match, and write it to disk.

WHY THIS FILE EXISTS
--------------------
Everything measured so far was leave-one-match-out: five folds, five models, each trained on four
matches and scored on the fifth. That is the right way to get an honest number and the wrong way
to produce a submission -- a LOMO fold is a model that deliberately ignores a fifth of the data,
and there are five of them with no reason to prefer any one.

So this script trains the same two models, with the same hyper-parameters and the same ball
masking, on all five development matches at once, and persists them along with the duration
priors and the exact feature order. `submit/predict_match.py` then loads them and never trains.

    python -I submit/fit_final.py

Writes into submit/models/:
    stage2_dead.txt        LightGBM, dead-ball probability from player geometry alone
    stage2_durations.npz   log-duration priors for Alive and Dead, and the dead prior
    stage3_team.txt        LightGBM, P(home in control) from geometry plus the ball when present
    stage3_durations.npz   log-duration priors for Home and Away spells, and the home prior
    manifest.json          feature order, row counts, hyper-parameters, source of every number

WHAT THIS CHANGES ABOUT THE REPORTED SCORE, AND WHAT IT DOES NOT
----------------------------------------------------------------
The published development figures -- spellF1 0.519, macroF1 0.884 after hardening -- stay the
leave-one-match-out numbers, because those are the only ones estimated on a match the model had
not seen. This model is trained on more data than any of those folds, so it should be no worse;
but nothing here measures that, and nothing here may, because the only unseen match left is the
competition's and it has no label we are allowed to read.

The two sealed IDSSE matches (J03WMX, J03WN1) are not loaded by this script either. They were
set aside to be opened once, after the model was frozen, and they remain unopened.
"""

import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "stage2"))
sys.path.insert(0, os.path.join(ROOT, "stage3"))
from train import fit_durations  # noqa: E402  stage2
from train_team import fit_team_durations, mask_ball  # noqa: E402  stage3

# stage3/train_team.py masks its four training matches at (0.00, 0.25, 0.45, 0.65). Cycling those
# four over five matches would hand two of them a fully visible ball, dropping the mean masking
# rate from 0.34 to 0.27 -- less ball-blindness than the folds were trained with, on a match that
# hides the ball 41% of the time. The fifth rate is therefore 0.41: the competition's own
# documented ball-missing rate on graded frames, which is a published property of the dataset
# (DATA.md) and not a label, a score or anything derived from one.
TRAIN_RATES_FINAL = (0.00, 0.25, 0.45, 0.65, 0.41)

DEV = ["DFL-MAT-J03WOH", "DFL-MAT-J03WOY", "DFL-MAT-J03WPY",
       "DFL-MAT-J03WQQ", "DFL-MAT-J03WR9"]
SEALED = ["DFL-MAT-J03WMX", "DFL-MAT-J03WN1"]          # never loaded, here to be explicit

S2_FEAT = os.path.join(ROOT, "stage2", "features_h")
S3_FEAT = os.path.join(ROOT, "stage3", "features")
S3_PC = os.path.join(ROOT, "stage3", "pitch_control")
OUT = os.path.join(HERE, "models")

# Exactly the parameters stage2/train.py and stage3/train_team.py use. Repeated rather than
# imported because a submission model must not change when a training script is next edited.
LGB_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
              "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "verbosity": -1, "seed": 0, "num_threads": 0}
S2_ROUNDS = 400          # stage2/train.py default
S3_ROUNDS = 300          # stage3/train_team.py default

S2_NON_FEATURES = {"match_id", "period", "frame_native", "label", "label_valid", "is_dead"}
S3_NON_FEATURES = {"match_id", "period", "frame_native", "label", "label_valid", "is_home",
                   "home_attacks_right"}


def read(path):
    """One feature table, labelled rows only, float32. Same reader both stages use."""
    t = pd.read_parquet(path)
    t = t[t["label_valid"].astype(bool)].reset_index(drop=True)
    for c in t.columns:
        if t[c].dtype == "float64":
            t[c] = t[c].astype("float32")
    return t


def fit_stage2():
    """Dead-ball detector: player geometry only, no ball feature anywhere in it."""
    data = {m: read(os.path.join(S2_FEAT, m + ".parquet")) for m in DEV}
    cols = [c for c in data[DEV[0]].columns if c not in S2_NON_FEATURES]
    tr = pd.concat([data[m] for m in DEV], ignore_index=True)

    model = lgb.train(LGB_PARAMS, lgb.Dataset(tr[cols], label=tr["is_dead"]),
                      num_boost_round=S2_ROUNDS)

    # The duration prior is fitted on the same five matches, per match, exactly as the folds did.
    log_a, log_d = fit_durations([data[m]["label"].to_numpy() == "D" for m in DEV],
                                 step=5, max_steps=1000, mode="logkde")
    prior_dead = float(tr["is_dead"].mean())
    return model, cols, log_a, log_d, prior_dead, len(tr)


def fit_stage3():
    """Team attribution, trained with the ball hidden at four different rates across matches."""
    data = {}
    for m in DEV:
        t = read(os.path.join(S3_FEAT, m + ".parquet"))
        pc = pd.read_parquet(os.path.join(S3_PC, m + ".parquet"))
        t = t.merge(pc, on=["period", "frame_native"], how="left")
        for c in t.columns:
            if t[c].dtype == "float64":
                t[c] = t[c].astype("float32")
        data[m] = t
    cols = [c for c in data[DEV[0]].columns if c not in S3_NON_FEATURES]

    # Live frames only -- stage 3 answers "which team", and a dead frame has no answer. Each
    # match is masked at a different rate from the same spread the folds used, so the model sees
    # a ball that is sometimes there and sometimes not rather than one guessed rate.
    parts, rates = [], {}
    for i, m in enumerate(DEV):
        rate = TRAIN_RATES_FINAL[i % len(TRAIN_RATES_FINAL)]
        rates[m] = rate
        d = data[m]
        parts.append(mask_ball(d[d["is_home"] >= 0], rate, seed=i))
    tr = pd.concat(parts, ignore_index=True)

    model = lgb.train(LGB_PARAMS, lgb.Dataset(tr[cols], label=tr["is_home"]),
                      num_boost_round=S3_ROUNDS)

    # Possession lengths, measured inside each contiguous alive stretch of each period, so a
    # possession before a stoppage is never glued to the one after it.
    label_arrays = []
    for m in DEV:
        for _, g in data[m].groupby("period", sort=True):
            label_arrays.append(g["label"].to_numpy())
    log_a, log_h = fit_team_durations(label_arrays)
    prior_home = float(tr["is_home"].mean())
    return model, cols, log_a, log_h, prior_home, len(tr), rates


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()

    print("fitting the frozen model on {} development matches".format(len(DEV)))
    print("sealed and not loaded: {}\n".format(", ".join(SEALED)))

    s2, s2_cols, s2_la, s2_ld, prior_dead, n2 = fit_stage2()
    s2.save_model(os.path.join(args.out_dir, "stage2_dead.txt"))
    np.savez(os.path.join(args.out_dir, "stage2_durations.npz"),
             log_alive=s2_la, log_dead=s2_ld, prior_dead=prior_dead)
    print("  stage 2  {:>9,} rows  {} features  dead prior {:.3f}".format(
        n2, len(s2_cols), prior_dead))

    s3, s3_cols, s3_la, s3_lh, prior_home, n3, rates = fit_stage3()
    s3.save_model(os.path.join(args.out_dir, "stage3_team.txt"))
    np.savez(os.path.join(args.out_dir, "stage3_durations.npz"),
             log_away=s3_la, log_home=s3_lh, prior_home=prior_home)
    print("  stage 3  {:>9,} live rows  {} features  home prior {:.3f}".format(
        n3, len(s3_cols), prior_home))
    print("           ball hidden per match: {}".format(
        {m.replace("DFL-MAT-", ""): r for m, r in rates.items()}))

    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "training_matches": DEV,
        "sealed_matches_not_loaded": SEALED,
        "competition_match_used_in_training": False,
        "ground_truth_used": "none, public or private",
        "stage2": {"model": "stage2_dead.txt", "rounds": S2_ROUNDS,
                   "n_rows": n2, "features": s2_cols, "prior_dead": prior_dead,
                   "duration_prior": "logkde, step 5, max 1000, fitted per match"},
        "stage3": {"model": "stage3_team.txt", "rounds": S3_ROUNDS,
                   "n_rows": n3, "features": s3_cols, "prior_home": prior_home,
                   "ball_mask_rate_per_match": {m: r for m, r in rates.items()},
                   "duration_prior": "possession lengths inside alive stretches, per period"},
        "lgb_params": LGB_PARAMS,
        "versions": {"lightgbm": lgb.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "python": sys.version.split()[0]},
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nwrote 5 files to {}  ({:.0f}s)".format(args.out_dir, time.time() - started))


if __name__ == "__main__":
    main()
