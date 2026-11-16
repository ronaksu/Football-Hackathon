"""Final Stage 3 experiment: the possession path decoded from LEARNED emissions.

Everything else is unchanged from path_model.py. The only difference is that the per-player
emission scores now come from a model trained on event-derived player-level possession, instead of
being hand-set from distance to the ball. That was the last material gap between this and the
published PathCRF method.

Known ceiling, stated up front: the event-derived holder's team agrees with the DFL's own per-frame
BallPossession label on about 90% of live frames, and that is what we are scored against. So a
perfectly learned emission tops out near 0.90 team accuracy, against the roughly 0.856 the
per-frame model already reaches.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
from features_team import wide  # noqa: E402
from path_model import PITCH_X, PITCH_Y, STEP, path_to_team, viterbi_path  # noqa: E402
from train import load_scorer  # noqa: E402
from train_emissions import DEV, FEATURES, player_rows  # noqa: E402
from train_team import score, to_labels  # noqa: E402

NEG = -1e18


def build_emissions(q, tracked, is_home, loose):
    """log-emissions over 22 player states plus two flight states, from learned scores."""
    n, m = q.shape
    qq = np.clip(q, 1e-6, 1.0)
    em = np.full((n, m + 2), NEG)
    em[:, :m] = np.where(tracked, np.log(qq), -1e3)

    # A flight state is favoured when no single player looks like the holder.
    with np.errstate(invalid="ignore"):
        home_mass = np.nansum(np.where(is_home[None, :], qq, 0.0), axis=1)
        away_mass = np.nansum(np.where(~is_home[None, :], qq, 0.0), axis=1)
    tot = np.maximum(home_mass + away_mass, 1e-6)
    em[:, m] = np.log(np.clip(home_mass / tot, 1e-6, 1)) + np.log(0.10) + loose
    em[:, m + 1] = np.log(np.clip(away_mass / tot, 1e-6, 1)) + np.log(0.10) + loose
    return em


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc-dir", default="D:/Football Hackathon/external_data/processed/idsse")
    ap.add_argument("--label-dir", default="D:/Football Hackathon/stage3/player_labels")
    ap.add_argument("--pc-dir", default="D:/Football Hackathon/stage3/pitch_control")
    ap.add_argument("--model-dir", default="D:/Football Hackathon/stage3/emission_models")
    ap.add_argument("--pred-dir", default="D:/Football Hackathon/stage2/predictions_h")
    ap.add_argument("--out", default="D:/Football Hackathon/stage3/results_trained_path.csv")
    args = ap.parse_args()

    import lightgbm as lgb
    pm = load_scorer()
    rows = []

    for held in DEV:
        booster = lgb.Booster(model_file=os.path.join(args.model_dir, held + ".txt"))
        pr = player_rows(held, args.proc_dir, args.label_dir, args.pc_dir,
                         stride=STEP, with_labels=False)
        pr = pr.assign(q=booster.predict(pr[FEATURES]))

        frames = pd.read_parquet(os.path.join(args.proc_dir, held, "frames.parquet"),
                                 columns=["period", "frame_native", "label", "label_valid"])
        s2 = pd.read_parquet(os.path.join(args.pred_dir, held + ".parquet"))
        frames = frames.merge(s2[["period", "frame_native", "pred_hsmm"]],
                             on=["period", "frame_native"], how="left")
        frames = frames[frames["label_valid"].astype(bool)].reset_index(drop=True)
        players = pd.read_parquet(os.path.join(args.proc_dir, held, "players.parquet"))

        home_path = np.zeros(len(frames), dtype=bool)
        for period, g in frames.groupby("period", sort=True):
            w = wide(players, period)
            if w is None:
                continue
            fr, X, Y, is_home = w
            m = len(is_home)

            sub = pr[pr["period"] == period]
            grid = np.sort(sub["frame_native"].unique())
            if not len(grid):
                continue
            q = (sub.pivot_table(index="frame_native", columns="col", values="q",
                                 aggfunc="first")
                    .reindex(index=grid, columns=range(m)).to_numpy())

            idx = pd.Index(fr).get_indexer(grid)
            ok = idx >= 0
            grid, q, idx = grid[ok], q[ok], idx[ok]
            Xs, Ys = X[idx], Y[idx]
            tracked = np.isfinite(Xs)
            q = np.where(np.isfinite(q), q, 0.0)

            # "loose": nobody looks like a holder, so the ball is probably travelling.
            loose = np.clip(1.0 - q.max(axis=1) * 3.0, 0.0, 1.2)
            em = build_emissions(q, tracked, is_home, loose)
            path = viterbi_path(em, Xs, Ys, is_home)
            coarse = path_to_team(path, is_home)

            # spread the decoded grid decision back onto every frame of the period
            pos = pd.Index(g["frame_native"].to_numpy())
            nearest = np.searchsorted(grid, g["frame_native"].to_numpy(), side="right") - 1
            nearest = np.clip(nearest, 0, len(coarse) - 1)
            home_path[g.index.to_numpy()] = coarse[nearest]

        truth = frames["label"].to_numpy()
        dead = frames["pred_hsmm"].fillna(False).to_numpy().astype(bool)
        alive = truth != "D"
        acc = float((home_path[alive] == (truth[alive] == "H")).mean())
        per = frames["period"].to_numpy()

        for name, pred in (("stage3_only", to_labels(truth == "D", home_path)),
                           ("end_to_end", to_labels(dead, home_path))):
            sf, mf, nps, ngs = score(pm, truth, pred, per)
            rows.append({"held_out": held.replace("DFL-MAT-", ""), "which": name,
                         "team_acc": acc, "spellF1": sf, "macroF1": mf,
                         "pred_spells": nps, "gt_spells": ngs})
        print("  {}  team acc {:.3f}   stage3 {:.3f}   end-to-end {:.3f}   spells {:.0f}".format(
            held.replace("DFL-MAT-", ""), acc, rows[-2]["spellF1"], rows[-1]["spellF1"],
            rows[-1]["pred_spells"]))

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print("\n=== TRAINED PATH MODEL, means over 5 development matches ===")
    print(out.groupby("which").agg(
        team_acc=("team_acc", "mean"), spellF1=("spellF1", "mean"),
        macroF1=("macroF1", "mean"), pred_spells=("pred_spells", "mean"),
        gt_spells=("gt_spells", "mean")).to_string(float_format=lambda v: "{:.3f}".format(v)))
    print("\nwrote " + args.out)


if __name__ == "__main__":
    main()
