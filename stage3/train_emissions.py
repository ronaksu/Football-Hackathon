"""Train the possession-path emissions on player-level labels, instead of hand-setting them.

This is the last material difference between our path model and the published PathCRF method:
theirs learns per-frame player-level possession, ours guessed it from distance to the ball.

The model is deliberately PLAYER-RELATIVE, not slot-indexed. Each training row is one
(frame, player) pair described only by quantities that mean the same thing in any match -- how far
that player is from the ball, where they rank on that, how fast they are moving, how close the
nearest opponent is, and how much ground their side controls where they stand. Nothing refers to
slot 7 or to a particular team, so the model transfers to a match it has never seen, and to the
competition's 46 anonymous slots.

The ceiling is known in advance and worth stating. The event-derived holder's team agrees with the
DFL's own per-frame BallPossession label on about 90% of live frames. That is the most a perfectly
trained emission could score against the metric we are judged on; the current models already reach
about 85.6%. So the headroom here is roughly four points of team accuracy, not thirty.
"""

import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
from features_team import wide as _wide_base  # noqa: E402


def wide_labelled(players, period):
    """Like features_team.wide, but also returns the (team, slot) label of every column.

    pivot_table orders columns ALPHABETICALLY, so away_10 sits before away_2. Mapping a slot
    number to a column by position is therefore wrong, and that is what silently pointed the
    holder label at the wrong player.
    """
    g = players[players["period"] == period]
    if g.empty:
        return None
    g = g.copy()
    g["col"] = g["team"].astype(str) + "_" + g["slot"].astype(str)
    x = g.pivot_table(index="frame_native", columns="col", values="x", aggfunc="first")
    y = g.pivot_table(index="frame_native", columns="col", values="y", aggfunc="first")
    x, y = x.sort_index(), y.sort_index().reindex(columns=x.columns)
    labels = [(c.split("_")[0], int(c.split("_")[1])) for c in x.columns]
    is_home = np.array([t == "home" for t, _ in labels])
    return (x.index.to_numpy(), x.to_numpy() * PITCH_X, y.to_numpy() * PITCH_Y,
            is_home, labels)

PITCH_X, PITCH_Y = 105.0, 68.0
FPS = 25
STRIDE = 5          # every 5th frame; 25 Hz is far finer than possession changes
DEV = ["DFL-MAT-J03WOH", "DFL-MAT-J03WOY", "DFL-MAT-J03WPY", "DFL-MAT-J03WQQ",
       "DFL-MAT-J03WR9"]

FEATURES = ["d_ball", "d_ball_rank", "d_ball_margin", "speed", "d_nearest_opp",
            "d_nearest_team", "pc_here", "is_home", "x_norm", "y_norm",
            "d_ball_team_rank", "closing_on_ball"]


def player_rows(match_id, proc_dir, label_dir, pc_dir, stride=STRIDE, with_labels=True):
    """One row per (frame, tracked player), player-relative features only."""
    players = pd.read_parquet(os.path.join(proc_dir, match_id, "players.parquet"))
    frames = pd.read_parquet(os.path.join(proc_dir, match_id, "frames.parquet"),
                             columns=["period", "frame_native", "label", "label_valid",
                                      "ball_x", "ball_y", "ball_observed"])
    pc = pd.read_parquet(os.path.join(pc_dir, match_id + ".parquet"),
                         columns=["period", "frame_native", "pc_home_share"])
    frames = frames.merge(pc, on=["period", "frame_native"], how="left")
    lab = None
    if with_labels:
        lab = pd.read_parquet(os.path.join(label_dir, match_id + ".parquet"))

    out = []
    for period in sorted(frames["period"].unique()):
        w = wide_labelled(players, period)
        if w is None:
            continue
        fr, X, Y, is_home, col_labels = w
        col_index = {lab: i for i, lab in enumerate(col_labels)}
        f = frames[frames["period"] == period].set_index("frame_native").reindex(fr)

        sel = np.arange(0, len(fr), stride)
        keep = f["label_valid"].to_numpy()[sel].astype(bool) & \
            np.isin(f["label"].to_numpy()[sel], ["H", "A"])
        sel = sel[keep]
        if not len(sel):
            continue

        Xs, Ys = X[sel], Y[sel]
        bx = f["ball_x"].to_numpy()[sel] * PITCH_X
        by = f["ball_y"].to_numpy()[sel] * PITCH_Y
        seen = f["ball_observed"].to_numpy()[sel].astype(bool)
        pch = f["pc_home_share"].to_numpy()[sel]

        # velocity, for speed and for whether the player is closing on the ball
        prev = np.clip(sel - 2, 0, len(fr) - 1)
        nxt = np.clip(sel + 2, 0, len(fr) - 1)
        dt = (nxt - prev) / FPS
        dt[dt == 0] = 1.0 / FPS
        vx = (X[nxt] - X[prev]) / dt[:, None]
        vy = (Y[nxt] - Y[prev]) / dt[:, None]
        speed = np.hypot(vx, vy)

        d_ball = np.where(seen[:, None], np.hypot(Xs - bx[:, None], Ys - by[:, None]), np.nan)
        # closing speed on the ball: negative means getting closer
        with np.errstate(invalid="ignore"):
            ux = np.where(seen[:, None], (bx[:, None] - Xs), 0.0)
            uy = np.where(seen[:, None], (by[:, None] - Ys), 0.0)
            nrm = np.maximum(np.hypot(ux, uy), 1e-6)
            closing = -(vx * ux + vy * uy) / nrm

        # pairwise distances for nearest teammate / opponent
        dxx = Xs[:, :, None] - Xs[:, None, :]
        dyy = Ys[:, :, None] - Ys[:, None, :]
        dd = np.sqrt(dxx * dxx + dyy * dyy)
        dd[~np.isfinite(dd)] = np.inf
        np.einsum("ijj->ij", dd)[...] = np.inf
        same = is_home[None, :, None] == is_home[None, None, :]
        d_team = np.where(same, dd, np.inf).min(axis=2)
        d_opp = np.where(~same, dd, np.inf).min(axis=2)

        # rank by distance to the ball, overall and within the player's own side
        order = np.argsort(np.where(np.isfinite(d_ball), d_ball, np.inf), axis=1)
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(d_ball.shape[1])[None, :], axis=1)
        second = np.sort(np.where(np.isfinite(d_ball), d_ball, np.inf), axis=1)[:, 1:2]
        margin = d_ball - second

        team_rank = np.zeros_like(rank)
        for side in (True, False):
            cols = np.flatnonzero(is_home == side)
            sub = np.where(np.isfinite(d_ball[:, cols]), d_ball[:, cols], np.inf)
            o = np.argsort(sub, axis=1)
            r = np.empty_like(o)
            np.put_along_axis(r, o, np.arange(len(cols))[None, :], axis=1)
            team_rank[:, cols] = r

        n_f, n_p = Xs.shape
        frame_ids = np.repeat(fr[sel], n_p)
        slot_ids = np.tile(np.arange(n_p), n_f)
        df = pd.DataFrame({
            "period": period, "frame_native": frame_ids, "col": slot_ids,
            "d_ball": d_ball.ravel(), "d_ball_rank": rank.ravel().astype(float),
            "d_ball_margin": margin.ravel(), "speed": speed.ravel(),
            "d_nearest_opp": np.where(np.isfinite(d_opp), d_opp, np.nan).ravel(),
            "d_nearest_team": np.where(np.isfinite(d_team), d_team, np.nan).ravel(),
            "is_home": np.tile(is_home.astype(float), n_f),
            "x_norm": (Xs / PITCH_X).ravel(), "y_norm": (Ys / PITCH_Y).ravel(),
            "d_ball_team_rank": team_rank.ravel().astype(float),
            "closing_on_ball": closing.ravel(),
        })
        df["pc_here"] = np.where(df["is_home"] > 0,
                                 np.repeat(pch, n_p), 1.0 - np.repeat(pch, n_p))
        df = df[np.isfinite(Xs).ravel()]

        if with_labels:
            l = lab[lab["period"] == period].set_index("frame_native").reindex(fr[sel])
            hs = l["holder_side"].to_numpy()
            hslot = l["holder_slot"].to_numpy()
            # map (side, slot) back to the wide column index
            holder_col = np.full(len(sel), -1)
            for k in range(len(sel)):
                if hs[k] not in ("home", "away"):
                    continue
                holder_col[k] = col_index.get((hs[k], int(hslot[k])), -1)
            df["is_holder"] = (df["col"].to_numpy() ==
                               np.repeat(holder_col, n_p)[np.isfinite(Xs).ravel()]).astype(int)

        out.append(df)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc-dir", default="D:/Football Hackathon/external_data/processed/idsse")
    ap.add_argument("--label-dir", default="D:/Football Hackathon/stage3/player_labels")
    ap.add_argument("--pc-dir", default="D:/Football Hackathon/stage3/pitch_control")
    ap.add_argument("--out-dir", default="D:/Football Hackathon/stage3/emission_models")
    ap.add_argument("--rounds", type=int, default=250)
    args = ap.parse_args()

    cache = {}
    for m in DEV:
        cache[m] = player_rows(m, args.proc_dir, args.label_dir, args.pc_dir)
        print("  {}  {:>9,} player-frames   holder rate {:.3f}".format(
            m.replace("DFL-MAT-", ""), len(cache[m]), cache[m]["is_holder"].mean()))

    os.makedirs(args.out_dir, exist_ok=True)
    print()
    for held in DEV:
        tr = pd.concat([cache[m] for m in DEV if m != held], ignore_index=True)
        te = cache[held]
        model = lgb.train(
            {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
             "min_data_in_leaf": 500, "feature_fraction": 0.9, "verbosity": -1, "seed": 0},
            lgb.Dataset(tr[FEATURES], label=tr["is_holder"]), num_boost_round=args.rounds)
        model.save_model(os.path.join(args.out_dir, held + ".txt"))

        p = model.predict(te[FEATURES])
        te = te.assign(p=p)
        # Does the highest-scoring player per frame identify the right holder, and the right team?
        idx = te.groupby(["period", "frame_native"])["p"].idxmax()
        top = te.loc[idx]
        print("  {}  holder top-1 {:.3f}   team from top-1 {:.3f}".format(
            held.replace("DFL-MAT-", ""), top["is_holder"].mean(),
            (top["is_home"] == te.loc[te["is_holder"] == 1].set_index(
                ["period", "frame_native"]).reindex(
                top.set_index(["period", "frame_native"]).index)["is_home"].to_numpy()).mean()))


if __name__ == "__main__":
    main()
