"""Stage 3: which team is in control, trained with the ball deliberately taken away.

Leave-one-match-out over the five development matches. The two test matches are never loaded.

THE BALL IS THE PROBLEM THIS SCRIPT EXISTS TO SOLVE. When it is visible, the margin between the
nearest home player and the nearest away player settles possession on its own -- 76.5% accuracy
from that one comparison. The competition loses the ball on 41% of graded frames and IDSSE never
loses it, so a model trained here learns to lean on something that will not be there.

So every training match is masked at a different rate drawn from a spread, and the held-out match
is scored at several rates including the competition's own 41%. That produces a robustness curve
rather than a single number, which is the only honest way to report this.

Finally the two stages are joined and scored end to end with the organisers' own code, decomposed
three ways so it is clear which stage any shortfall belongs to.
"""

import argparse
import importlib.util
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
from train import decode_hsmm, load_scorer, runs_of  # noqa: E402

FPS = 25
DEV = ["DFL-MAT-J03WOH", "DFL-MAT-J03WOY", "DFL-MAT-J03WPY", "DFL-MAT-J03WQQ",
       "DFL-MAT-J03WR9"]
NON_FEATURES = {"match_id", "period", "frame_native", "label", "label_valid", "is_home",
                "home_attacks_right"}
BALL_COLS = ("ball_x_n", "ball_y", "ball_third", "d_ball_home", "d_ball_away", "ball_margin",
             "ball_nearer_home", "ball_adv_x", "ball_observed",
             "ball_margin_m25", "ball_margin_m75",
             "ball_nearer_home_m25", "ball_nearer_home_m75")

# Masking rates used across the four training matches of each fold, and the rates the held-out
# match is scored at. 0.41 is the competition's documented figure.
TRAIN_RATES = (0.00, 0.25, 0.45, 0.65)
TEST_RATES = (0.00, 0.20, 0.41, 0.60)


def mask_ball(df, rate, seed, mean_run_s=2.0):
    """Hide the ball for contiguous stretches totalling about `rate` of frames.

    Contiguous, not per-frame: a camera loses the ball for a second or two at a time, and a model
    trained on independent per-frame dropout learns a problem nobody has.
    """
    if rate <= 0:
        return df
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(out)
    run = max(1, int(mean_run_s * FPS))
    hidden = np.zeros(n, dtype=bool)
    target = int(n * rate)
    while hidden.sum() < target:
        start = int(rng.integers(0, max(1, n - run)))
        length = max(1, int(rng.exponential(run)))
        hidden[start:start + length] = True
    for c in BALL_COLS:
        if c in out.columns:
            out.loc[hidden, c] = np.nan
    if "ball_observed" in out.columns:
        out.loc[hidden, "ball_observed"] = 0.0
    return out


def decode_team(prob_home, alive_mask, log_a, log_h, prior_home):
    """H/A within each contiguous alive stretch, with explicit possession-duration modelling."""
    out = np.zeros(len(prob_home), dtype=bool)  # True = home
    for a, b in runs_of(alive_mask):
        seg = prob_home[a:b]
        if len(seg) < 10:
            out[a:b] = seg.mean() >= 0.5
            continue
        out[a:b] = decode_hsmm(seg, log_a, log_h, prior_home, step=5, max_steps=400)
    return out


def fit_team_durations(label_arrays, step=5, max_steps=400):
    """Empirical log-duration distributions for away-spells and home-spells, in grid steps.

    Takes FULL label sequences including D, and measures possession lengths only inside each
    contiguous alive stretch. The first version dropped the dead frames and concatenated what was
    left, which silently glued the possession before a stoppage to the one after it -- so the
    fitted spells were far longer than real ones, and the decoder then merged real possessions to
    match. That was the cause of 243 predicted possessions against 410.
    """
    out = {}
    for state, want in (("home", "H"), ("away", "A")):
        lens = []
        for lab in label_arrays:
            lab = np.asarray(lab, dtype=object)
            for a0, b0 in runs_of(lab != "D"):
                seg = lab[a0:b0]
                for a, b in runs_of(seg == want):
                    lens.append(max(1, int(round((b - a) / step))))
        h = np.bincount(np.clip(lens, 1, max_steps), minlength=max_steps + 1).astype(float)
        h[0] = 0.0
        h = np.convolve(h, np.ones(5) / 5.0, mode="same") + 1e-3
        out[state] = np.log(h / h.sum())
    return out["away"], out["home"]


def to_labels(dead, home):
    return np.where(dead, "D", np.where(home, "H", "A"))


def score(pm, truth, pred, periods):
    gt, pr = [], []
    for p in sorted(pd.unique(periods)):
        m = periods == p
        if gt:
            gt.append(np.array(["X"]))
            pr.append(np.array(["X"]))
        gt.append(np.asarray(truth[m], dtype=object))
        pr.append(np.asarray(pred[m], dtype=object))
    r = pm.evaluate(pm.label_codes(np.concatenate(gt)), pm.label_codes(np.concatenate(pr)))
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feat-dir", default="D:/Football Hackathon/stage3/features")
    ap.add_argument("--stage2-pred", default="D:/Football Hackathon/stage2/predictions_h")
    ap.add_argument("--pitch-control", default=None,
                    help="directory of pitch-control features to merge in; omit to run without")
    ap.add_argument("--duration-source", default="possession",
                    choices=["possession", "stage2"],
                    help="possession = fit the spell-length prior on possessions themselves")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--out", default="D:/Football Hackathon/stage3/results.csv")
    ap.add_argument("--save-pred", default=None,
                    help="directory to write per-frame team probability and decoded output, for "
                         "the failure-review tool. Written at --only-rate.")
    ap.add_argument("--only-rate", type=float, default=None,
                    help="score a single ball-masking rate instead of the whole curve")
    args = ap.parse_args()

    pm = load_scorer()
    data = {}
    for m in DEV:
        t = pd.read_parquet(os.path.join(args.feat_dir, m + ".parquet"))
        if args.pitch_control:
            pc = pd.read_parquet(os.path.join(args.pitch_control, m + ".parquet"))
            t = t.merge(pc, on=["period", "frame_native"], how="left")
        t = t[t["label_valid"].astype(bool)].reset_index(drop=True)
        for c in t.columns:
            if t[c].dtype == "float64":
                t[c] = t[c].astype("float32")
        data[m] = t

    feat_cols = [c for c in data[DEV[0]].columns if c not in NON_FEATURES]
    print("Stage 3: {} features over {} matches\n".format(len(feat_cols), len(DEV)))

    rows = []
    for held in DEV:
        train_parts = []
        for i, m in enumerate([x for x in DEV if x != held]):
            d = data[m]
            d = d[d["is_home"] >= 0]
            train_parts.append(mask_ball(d, TRAIN_RATES[i % len(TRAIN_RATES)], seed=i))
        tr = pd.concat(train_parts, ignore_index=True)

        model = lgb.train(
            {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
             "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
             "bagging_freq": 1, "verbosity": -1, "seed": 0, "num_threads": 0},
            lgb.Dataset(tr[feat_cols], label=tr["is_home"]),
            num_boost_round=args.rounds)

        # Full label sequences, split by period so nothing spans half-time.
        label_arrays = []
        for m in DEV:
            if m == held:
                continue
            for _, g in data[m].groupby("period", sort=True):
                label_arrays.append(g["label"].to_numpy())
        log_a, log_h = fit_team_durations(label_arrays)
        prior_home = float(tr["is_home"].mean())

        te_full = data[held]
        s2 = pd.read_parquet(os.path.join(args.stage2_pred, held + ".parquet"))
        te = te_full.merge(s2[["period", "frame_native", "pred_hsmm"]],
                           on=["period", "frame_native"], how="left")
        te["pred_hsmm"] = te["pred_hsmm"].fillna(False).astype(bool)

        truth = te["label"].to_numpy()
        true_dead = truth == "D"
        per = te["period"].to_numpy()

        rates = (args.only_rate,) if args.only_rate is not None else TEST_RATES
        for rate in rates:
            tm = mask_ball(te, rate, seed=99)
            p_home = model.predict(tm[feat_cols])

            # Team accuracy on frames that are genuinely alive -- Stage 3 in isolation.
            alive = ~true_dead
            acc = float(((p_home[alive] >= 0.5) == te.loc[alive, "is_home"]).mean())

            # 1. Stage 3 alone: true dead-ball, our teams.
            home_true_alive = decode_team(p_home, alive, log_a, log_h, prior_home)
            a_only = to_labels(true_dead, home_true_alive)

            # 2. Stage 2 alone: our dead-ball, true teams.
            b_only = to_labels(te["pred_hsmm"].to_numpy(), truth == "H")

            # 3. End to end: both ours.
            pred_alive = ~te["pred_hsmm"].to_numpy()
            home_pred = decode_team(p_home, pred_alive, log_a, log_h, prior_home)
            full = to_labels(te["pred_hsmm"].to_numpy(), home_pred)

            if args.save_pred and (args.only_rate is None or rate == args.only_rate):
                os.makedirs(args.save_pred, exist_ok=True)
                dump = te[["period", "frame_native", "label"]].copy()
                dump["ball_rate"] = rate
                dump["p_home"] = p_home
                dump["ball_observed_masked"] = tm["ball_observed"].to_numpy().astype(bool)
                dump["pred_dead"] = te["pred_hsmm"].to_numpy()
                dump["home_stage3_only"] = home_true_alive
                dump["home_end_to_end"] = home_pred
                dump["pred_end_to_end"] = full
                dump.to_parquet(
                    os.path.join(args.save_pred, held + ".parquet"), index=False)

            for name, pred in (("stage3_only", a_only), ("stage2_only", b_only),
                               ("end_to_end", full)):
                sf, mf, nps, ngs = score(pm, truth, pred, per)
                rows.append({"held_out": held.replace("DFL-MAT-", ""), "ball_rate": rate,
                             "which": name, "team_acc": acc if name != "stage2_only" else np.nan,
                             "spellF1": sf, "macroF1": mf,
                             "pred_spells": nps, "gt_spells": ngs})
            print("  {}  ball hidden {:>3.0f}%   team acc {:.3f}   stage3 {:.3f}   "
                  "end-to-end {:.3f}".format(
                      held.replace("DFL-MAT-", ""), 100 * rate, acc,
                      rows[-3]["spellF1"], rows[-1]["spellF1"]))

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    print("\n=== means over 5 development matches ===")
    for which in ("stage3_only", "stage2_only", "end_to_end"):
        sub = out[out["which"] == which]
        print("\n{}:".format(which))
        g = sub.groupby("ball_rate").agg(
            team_acc=("team_acc", "mean"), spellF1=("spellF1", "mean"),
            macroF1=("macroF1", "mean"), pred_spells=("pred_spells", "mean"))
        print(g.to_string(float_format=lambda v: "{:.3f}".format(v)))
    print("\nwrote " + args.out)


if __name__ == "__main__":
    main()
