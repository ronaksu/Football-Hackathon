"""Stage 2: the dead-ball detector.

Leave-one-match-out over the five development matches. The two test matches (J03WMX, J03WN1) are
never loaded here.

Three decoders are compared on identical probabilities, because the argument of the strategy deck
was that duration modelling -- not the classifier -- is what makes interval output usable:

  raw    threshold the per-frame probability at 0.5. No temporal model at all.
  smooth threshold a rolling median, then delete runs shorter than a minimum length.
  hsmm   Viterbi over an explicit-duration semi-Markov model whose Alive and Dead duration
         distributions are fitted on the training matches of each fold.

Scoring uses the organisers' own `evaluation/possession_metrics.py`, not a reimplementation.

To isolate what this stage contributes, the prediction scored is a hybrid: the true H/A label
wherever the detector says alive, D wherever it says dead. A perfect detector therefore scores
spell_f1_mean = 1.0, and any shortfall is attributable to dead-ball detection alone.
"""

import argparse
import importlib.util
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

FPS = 25
DEV_MATCHES = [
    "DFL-MAT-J03WOH", "DFL-MAT-J03WOY", "DFL-MAT-J03WPY",
    "DFL-MAT-J03WQQ", "DFL-MAT-J03WR9",
]
NON_FEATURES = {"match_id", "period", "frame_native", "label", "label_valid", "is_dead"}

SCORER = ("D:/Football Hackathon/more-than-a-hack-2027-main/more-than-a-hack-2027-main/"
          "evaluation/possession_metrics.py")


def load_scorer():
    spec = importlib.util.spec_from_file_location("possession_metrics", SCORER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["possession_metrics"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- decoders

def runs_of(mask):
    """[(start, stop)] half-open spans where mask is True."""
    m = np.asarray(mask).astype(np.int8)
    edges = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def decode_raw(prob, threshold=0.5):
    return prob >= threshold


def decode_smooth(prob, threshold=0.5, med=51, min_dead=38, min_alive=38):
    """Rolling median, then remove runs shorter than a floor. min_* default to ~1.5 s."""
    s = pd.Series(prob).rolling(med, center=True, min_periods=1).median().to_numpy()
    out = s >= threshold
    for value, floor in ((True, min_dead), (False, min_alive)):
        for a, b in runs_of(out == value):
            if b - a < floor:
                out[a:b] = not value
    return out


def fit_durations(labels_by_match, step, max_steps, mode="logkde", floor=1e-5):
    """Empirical log-duration distributions for Alive and Dead, in grid steps.

    mode="hist" is the original: a linear histogram over max_steps bins, box-smoothed over five
    of them and floored at 1e-3. It was measured to be unusable as a prior. There are only about
    500 stoppages per fold spread over 1000 bins, so **76% of the bins inside the observed range
    are empty** and the largest count is 8. An empty bin lands on the floor, which charges about
    13 nats, while an occupied neighbour charges about 5. Whether a stoppage of a given length is
    allowed therefore came down to whether some training match happened to contain one within
    half a second of it -- a 1.0 s stoppage cost 13.1 nats and a 2.0 s stoppage 6.0, for no
    reason in the football.

    mode="logkde" (the default) fits the same data with a Gaussian kernel in LOG time. Durations
    run from 0.2 s to 200 s, so a fixed-width linear window is far too narrow at the long end and
    far too wide at the short end; in log time one bandwidth suits the whole range. The bandwidth
    is Silverman's rule on the log durations of that fold, not a tuned constant, so nothing here
    is fitted to a held-out match. The aggregate shape is almost unchanged -- both models put
    about 22% of the mass under ten seconds against a true 21% -- but the holes are gone.
    """
    out = {}
    grid = np.arange(1, max_steps + 1, dtype=float)
    log_grid = np.log(grid)
    for state, want in (("dead", True), ("alive", False)):
        lens = []
        for lab in labels_by_match:
            for a, b in runs_of(lab == want):
                lens.append(max(1, int(round((b - a) / step))))
        if mode == "hist":
            h = np.bincount(np.clip(lens, 1, max_steps), minlength=max_steps + 1).astype(float)
            h[0] = 0.0
            h = np.convolve(h, np.ones(5) / 5.0, mode="same") + 1e-3
            out[state] = np.log(h / h.sum())
            continue
        observed = np.clip(np.asarray(lens, dtype=float), 1.0, float(max_steps))
        log_obs = np.log(observed)
        bandwidth = max(0.15, 1.06 * float(np.std(log_obs)) * len(observed) ** -0.2)
        z = (log_grid[:, None] - log_obs[None, :]) / bandwidth
        # 1/d is the Jacobian that turns a density in log time into a density in steps.
        density = np.exp(-0.5 * z * z).sum(axis=1) / grid
        density /= density.sum()
        # A floor, so an unseen duration stays improbable rather than impossible.
        density = (1.0 - floor) * density + floor / max_steps
        out[state] = np.log(np.concatenate(([1e-300], density)))
    return out["alive"], out["dead"]


def decode_hsmm(prob, log_dur_alive, log_dur_dead, prior_dead, step=5, max_steps=1000):
    """Viterbi over a two-state explicit-duration semi-Markov model.

    Alive and Dead strictly alternate, so the predecessor state is always the other one and no
    transition matrix is needed. Emissions are the classifier's log-odds against the class prior,
    summed over a segment via a cumulative sum.
    """
    n = len(prob)
    g = prob[::step]
    t_n = len(g)

    p = np.clip(g, 1e-6, 1 - 1e-6)
    # log p(obs | state), up to a constant: the posterior divided by the prior.
    em_dead = np.log(p) - np.log(prior_dead)
    em_alive = np.log(1 - p) - np.log(1 - prior_dead)
    cum = {0: np.concatenate(([0.0], np.cumsum(em_alive))),
           1: np.concatenate(([0.0], np.cumsum(em_dead)))}
    log_dur = {0: log_dur_alive, 1: log_dur_dead}

    NEG = -1e18
    V = np.full((t_n + 1, 2), NEG)
    back = np.zeros((t_n + 1, 2), dtype=np.int32)
    V[0, :] = 0.0

    for t in range(1, t_n + 1):
        d_max = min(t, max_steps)
        d = np.arange(1, d_max + 1)
        starts = t - d
        for s in (0, 1):
            seg = cum[s][t] - cum[s][starts]
            prev = V[starts, 1 - s]
            score = prev + log_dur[s][d] + seg
            k = int(np.argmax(score))
            V[t, s] = score[k]
            back[t, s] = starts[k]

    # Backtrace.
    s = int(np.argmax(V[t_n]))
    t = t_n
    out = np.zeros(t_n, dtype=bool)
    while t > 0:
        a = back[t, s]
        out[a:t] = bool(s)
        t, s = a, 1 - s

    full = np.repeat(out, step)[:n]
    if len(full) < n:
        full = np.concatenate([full, np.full(n - len(full), full[-1])])
    return full


# ---------------------------------------------------------------- evaluation

def hybrid_codes(pm, true_labels, pred_dead, periods):
    """(gt_codes, pred_codes) with an X frame between periods so runs never merge.

    The hybrid is our dead-ball decision plus the true team wherever we say the ball is alive,
    so a perfect detector scores 1.0 and any shortfall is Stage 2's alone.

    The team must be carried in from the nearest live frame. Writing it as
    `np.where(pred_dead, "D", true_labels)` looks equivalent and is not: the truth is H, A *or*
    D, so wherever the detector MISSED a stoppage the expression fell through to the truth and
    emitted D anyway. Missed stoppages then cost nothing and only invented ones were ever
    charged -- which is precisely the failure this stage is trying to reduce. That bug is what
    put the reported figure at 0.947 instead of 0.862.
    """
    gt, pr = [], []
    for period in sorted(pd.unique(periods)):
        m = periods == period
        t = np.asarray(true_labels[m], dtype=object)
        d = np.asarray(pred_dead[m])
        team = pd.Series(np.where(t != "D", t, None)).ffill().bfill().to_numpy()
        p = np.where(d, "D", team).astype(object)
        if gt:
            gt.append(np.array(["X"]))
            pr.append(np.array(["X"]))
        gt.append(t)
        pr.append(p)
    return pm.label_codes(np.concatenate(gt)), pm.label_codes(np.concatenate(pr))


def danger_zone(true_labels, pred_dead, periods, max_len=10 * FPS):
    """Recall on short same-team stoppages: the ones whose loss merges two possession spells."""
    hit = tot = 0
    for period in sorted(pd.unique(periods)):
        m = periods == period
        lab = np.asarray(true_labels[m], dtype=object)
        pd_ = np.asarray(pred_dead[m])
        spans = runs_of(lab == "D")
        for k, (a, b) in enumerate(spans):
            if b - a >= max_len or a == 0 or b >= len(lab):
                continue
            if lab[a - 1] != lab[b]:
                continue  # different teams either side: missing it is cheap
            tot += 1
            if pd_[a:b].mean() > 0.5:
                hit += 1
    return hit, tot


def interval_recall(true_labels, pred_dead, periods, iou=0.5):
    """Fraction of true stoppages matched by a predicted stoppage at the given IoU."""
    hit = tot = 0
    for period in sorted(pd.unique(periods)):
        m = periods == period
        lab = np.asarray(true_labels[m], dtype=object)
        pred_spans = runs_of(np.asarray(pred_dead[m]))
        for a, b in runs_of(lab == "D"):
            tot += 1
            best = 0.0
            for c, d in pred_spans:
                lo, hi = max(a, c), min(b, d)
                if hi > lo:
                    best = max(best, (hi - lo) / float(max(b, d) - min(a, c)))
            if best >= iou:
                hit += 1
    return hit, tot


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feat-dir", default="D:/Football Hackathon/stage2/features_h")
    ap.add_argument("--test-feat-dir", default=None,
                    help="features for the held-out fold, if different from --feat-dir. Used to "
                         "train on a clean feed and score on a degraded one")
    ap.add_argument("--feature-set", default="hardened", choices=["base", "hardened"],
                    help="base reproduces the pre-hardening model on identical rows")
    ap.add_argument("--out", default="D:/Football Hackathon/stage2/results_lomo.csv")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--duration-model", default="logkde", choices=["logkde", "hist"],
                    help="hist reproduces the original sparse-histogram duration prior")
    ap.add_argument("--save-pred", default=None,
                    help="directory to write per-frame probabilities and decoded predictions")
    args = ap.parse_args()

    pm = load_scorer()
    print("scorer loaded: SPELL_IOUS={} noise band={}".format(pm.SPELL_IOUS, pm.SPELL_NOISE_BAND))

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from features import is_hardened_column

    # Training always reads --feat-dir. The held-out fold reads --test-feat-dir when one is
    # given, which is how a model trained on a clean feed gets scored on a degraded one.
    def read(d, m):
        t = pd.read_parquet(os.path.join(d, m + ".parquet"))
        t = t[t["label_valid"].astype(bool)].reset_index(drop=True)
        # float32 halves the footprint and costs nothing: these are metres and m/s, not money.
        for c in t.columns:
            if t[c].dtype == "float64":
                t[c] = t[c].astype("float32")
        return t

    data = {m: read(args.feat_dir, m) for m in DEV_MATCHES}
    # Only hold a second copy when the held-out fold really comes from somewhere else.
    test_data = ({m: read(args.test_feat_dir, m) for m in DEV_MATCHES}
                 if args.test_feat_dir else data)

    feat_cols = [c for c in data[DEV_MATCHES[0]].columns if c not in NON_FEATURES]
    if args.feature_set == "base":
        # Reproduce the pre-hardening model exactly, on identical rows.
        feat_cols = [c for c in feat_cols
                     if not is_hardened_column(c) and not c.startswith("n_tracked")]
    print("{} development matches, feature set '{}' -> {} features, no ball features".format(
        len(data), args.feature_set, len(feat_cols)))
    if args.test_feat_dir:
        print("held-out fold scored on: {}".format(args.test_feat_dir))
    print()

    rows = []
    importance = np.zeros(len(feat_cols))

    for held in DEV_MATCHES:
        tr = pd.concat([data[m] for m in DEV_MATCHES if m != held], ignore_index=True)
        te = test_data[held]

        model = lgb.train(
            {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
             "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
             "bagging_freq": 1, "verbosity": -1, "seed": 0, "num_threads": 0},
            lgb.Dataset(tr[feat_cols], label=tr["is_dead"]),
            num_boost_round=args.rounds,
        )
        importance += model.feature_importance("gain")
        prob = model.predict(te[feat_cols])

        train_labels = [data[m]["label"].to_numpy() == "D" for m in DEV_MATCHES if m != held]
        log_a, log_d = fit_durations(train_labels, step=5, max_steps=1000,
                                     mode=args.duration_model)
        prior = float(tr["is_dead"].mean())

        decoders = {
            "raw": decode_raw(prob),
            "smooth": decode_smooth(prob),
            "hsmm": decode_hsmm(prob, log_a, log_d, prior),
        }

        if args.save_pred:
            os.makedirs(args.save_pred, exist_ok=True)
            dump = te[["match_id", "period", "frame_native", "label"]].copy()
            dump["prob"] = prob
            for name, pred in decoders.items():
                dump["pred_" + name] = pred
            dump.to_parquet(os.path.join(args.save_pred, held + ".parquet"), index=False)

        for name, pred in decoders.items():
            gt_c, pr_c = hybrid_codes(pm, te["label"].to_numpy(), pred, te["period"].to_numpy())
            res = pm.evaluate(gt_c, pr_c)
            dh, dt = danger_zone(te["label"].to_numpy(), pred, te["period"].to_numpy())
            ih, it = interval_recall(te["label"].to_numpy(), pred, te["period"].to_numpy())
            rows.append({
                "held_out": held.replace("DFL-MAT-", ""), "decoder": name,
                "frame_f1_dead": res["per_class"]["D"]["f1"],
                "spell_f1_mean": res["spell_f1_mean"],
                "spell_f1_headline": res["spell_f1_at_headline"],
                "fragmentation": res["fragmentation"],
                "n_pred_spells": res["n_pred_spells"],
                "n_gt_spells": res["n_gt_spells"],
                "stoppage_recall_iou50": ih / it if it else np.nan,
                "danger_recall": dh / dt if dt else np.nan,
                "danger_n": dt,
            })
            print("  {}  {:<7} frameF1(D) {:.3f}   spellF1 {:.3f}   stoppages {:.1%}"
                  "   danger {:.1%} of {}".format(
                      held.replace("DFL-MAT-", ""), name, rows[-1]["frame_f1_dead"],
                      rows[-1]["spell_f1_mean"], rows[-1]["stoppage_recall_iou50"],
                      rows[-1]["danger_recall"], dt))

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    print("\n=== leave-one-match-out means over 5 development matches ===")
    agg = out.groupby("decoder").agg(
        frame_f1_dead=("frame_f1_dead", "mean"),
        spell_f1_mean=("spell_f1_mean", "mean"),
        stoppage_recall=("stoppage_recall_iou50", "mean"),
        danger_recall=("danger_recall", "mean"),
        fragmentation=("fragmentation", "mean"),
        pred_spells=("n_pred_spells", "mean"),
        gt_spells=("n_gt_spells", "mean"),
    ).reindex(["raw", "smooth", "hsmm"])
    print(agg.to_string(float_format=lambda v: "{:.3f}".format(v)))

    print("\n=== top 20 features by gain ===")
    imp = pd.Series(importance, index=feat_cols).sort_values(ascending=False)
    for k, (name, v) in enumerate(imp.head(20).items(), 1):
        print("  {:>2}. {:<28} {:>12,.0f}".format(k, name, v))

    print("\nwrote " + args.out)


if __name__ == "__main__":
    main()
