"""Stage 3, rebuilt as a possession PATH rather than a per-frame team classification.

Why a different model class. Three feature families -- raw geometry, ball proximity, and pitch
control -- all saturate team accuracy at 85-88%, and the HSMM decoder then produces 263 possession
spells where there are 410. More features of the same kind do not move either number. The problem
is the shape of the model, not the inputs.

Inspired by PathCRF (Kim et al., KDD 2026, arXiv:2602.12080,
github.com/hyunsungkim-ds/pathcrf). This is a deliberately simplified re-implementation of the
IDEA, not of their model: they use a Set Attention backbone with learned CRF parameters, and this
uses hand-set transition costs over the same state structure. NOTE: their README states no
licence, so nothing of theirs is copied here -- only the published idea is followed.

The borrowed structure is the one that matters for our label definition: the chain has TWO kinds
of state.

    player i holds the ball        22 states, one per tracked slot
    ball in flight, last held by T  2 states, one per team

The flight state is the point. The competition defines possession by CONTROL, and says a loose
ball stays with the last controlling team until someone else controls it. A pass in the air is
exactly that: nobody holds it, and the label must not flip to the receiving team until they take
it. A per-frame classifier has no way to express this; a path does.

Transitions encode football rather than smoothness:
    stay with the same player               cheap
    player -> flight of their OWN team      allowed; a pass or a clearance begins
    flight -> any player                    allowed; someone receives it
    player -> player directly               only when close together, and costly otherwise
    player -> flight of the OTHER team      forbidden; you cannot pass for the opposition

Possession intervals then fall out of the decoded path instead of being smoothed into existence.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
sys.path.insert(0, HERE)
from features_team import wide  # noqa: E402
from train import load_scorer, runs_of  # noqa: E402

PITCH_X, PITCH_Y = 105.0, 68.0
FPS = 25
STEP = 5                      # decode at 5-frame (0.2 s) resolution
NEG = -1e18

# Transition costs, in log space. Set by hand from the measured possession profile rather than
# learned: median possession 3.9 s, so staying must dominate, and a change must cost something
# comparable to the log-odds the emission can supply.
C_STAY = 0.0
C_TO_FLIGHT = -2.2            # starting a pass
C_FROM_FLIGHT = -1.4          # receiving one
C_FLIGHT_STAY = -0.25         # the ball keeps travelling
C_DIRECT = -4.0               # player to player with no flight between: a close exchange
DIRECT_MAX_M = 6.0            # beyond this, a direct hand-over is not plausible
C_PER_METRE = -0.035          # how much further the ball must travel costs


def emissions(p_home, dist_to_ball, ball_seen, team_of, pc_home, tracked=None):
    """log P(observation | state) for the 22 player states and the 2 flight states.

    Built from what the project already has: the trained per-frame team probability, and a
    per-player plausibility of being the one in control. The point of this script is the path
    structure, so the emission is kept simple and unlearned.
    """
    n, m = dist_to_ball.shape
    ph = np.clip(p_home, 1e-4, 1 - 1e-4)
    log_team = np.where(team_of[None, :], np.log(ph)[:, None], np.log(1 - ph)[:, None])

    # Which player, within the team, is plausibly on the ball.
    with np.errstate(invalid="ignore"):
        near = -dist_to_ball / 4.0                       # softmax over metres when visible
    near = np.where(np.isfinite(near), near, 0.0)
    seen = ball_seen.astype(bool)
    # When the ball cannot be seen, fall back on who holds controlled ground.
    ground = np.where(team_of[None, :], pc_home[:, None], 1.0 - pc_home[:, None])
    ground = np.log(np.clip(ground, 1e-3, 1.0))
    plaus = np.where(seen[:, None], near, ground)

    # A slot that is not being tracked cannot be the one in possession. Without this the NaN
    # distance collapses to plausibility 0.0, which outranks a real player ten metres away at
    # -2.5, and the path hands possession to whichever substitute is sitting unwatched. It only
    # bites once substitutes appear -- which is why it showed up in second halves (16 slots per
    # side) and not in first halves (11 tracked out of 11).
    if tracked is not None:
        plaus = np.where(tracked, plaus, -1e3)

    plaus = plaus - plaus.max(axis=1, keepdims=True)

    em = np.full((n, m + 2), NEG, dtype=np.float64)
    em[:, :m] = log_team + plaus

    # Flight states: favoured when nobody is close to the ball, or it cannot be seen at all.
    with np.errstate(invalid="ignore"):
        closest = np.nanmin(dist_to_ball, axis=1)
    loose = np.where(np.isfinite(closest), np.clip((closest - 2.0) / 6.0, 0.0, 1.5), 0.6)
    em[:, m] = np.log(ph) + np.log(0.12) + loose          # flight, last held by home
    em[:, m + 1] = np.log(1 - ph) + np.log(0.12) + loose  # flight, last held by away
    return em


def viterbi_path(em, X, Y, team_of):
    """Most probable possession path over players and the two flight states."""
    n, S = em.shape
    m = S - 2
    H_FLIGHT, A_FLIGHT = m, m + 1

    V = np.full((n, S), NEG)
    back = np.zeros((n, S), dtype=np.int32)
    V[0] = em[0]

    for t in range(1, n):
        prev = V[t - 1]
        # Distances between every pair of players at this step, for the direct-exchange rule.
        dx = X[t][:, None] - X[t][None, :]
        dy = Y[t][:, None] - Y[t][None, :]
        d = np.sqrt(dx * dx + dy * dy)
        d[~np.isfinite(d)] = np.inf

        best = np.full(S, NEG)
        arg = np.zeros(S, dtype=np.int32)

        # --- into each player state
        # from the same player
        cand = prev[:m] + C_STAY
        # from a flight of that player's own team
        from_flight = np.where(team_of, prev[H_FLIGHT], prev[A_FLIGHT]) + C_FROM_FLIGHT
        # from a nearby player directly
        direct = prev[:m][:, None] + C_DIRECT + C_PER_METRE * d
        direct = np.where(d <= DIRECT_MAX_M, direct, NEG)
        np.fill_diagonal(direct, NEG)
        direct_best = direct.max(axis=0)
        direct_arg = direct.argmax(axis=0)

        stack = np.vstack([cand, from_flight, direct_best])
        which = stack.argmax(axis=0)
        best[:m] = stack.max(axis=0)
        src = np.where(which == 0, np.arange(m),
                       np.where(which == 1,
                                np.where(team_of, H_FLIGHT, A_FLIGHT), direct_arg))
        arg[:m] = src

        # --- into each flight state: from one of that team's own players, or keep travelling
        for fs, is_home_flight in ((H_FLIGHT, True), (A_FLIGHT, False)):
            own = np.flatnonzero(team_of == is_home_flight)
            from_player = prev[own] + C_TO_FLIGHT
            k = int(np.argmax(from_player)) if len(own) else 0
            opts = np.array([from_player[k] if len(own) else NEG,
                             prev[fs] + C_FLIGHT_STAY])
            j = int(np.argmax(opts))
            best[fs] = opts[j]
            arg[fs] = own[k] if j == 0 and len(own) else fs

        V[t] = best + em[t]
        back[t] = arg

    path = np.zeros(n, dtype=np.int32)
    path[-1] = int(np.argmax(V[-1]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def path_to_team(path, team_of):
    """True where the path says HOME, from both player states and flight states."""
    m = len(team_of)
    out = np.zeros(len(path), dtype=bool)
    is_player = path < m
    out[is_player] = team_of[path[is_player]]
    out[path == m] = True        # flight, last held by home
    out[path == m + 1] = False   # flight, last held by away
    return out


def run_match(match_id, feat_dir, pc_dir, pred_dir, proc_dir, model_prob):
    feats = pd.read_parquet(os.path.join(feat_dir, match_id + ".parquet"))
    pc = pd.read_parquet(os.path.join(pc_dir, match_id + ".parquet"))
    s2 = pd.read_parquet(os.path.join(pred_dir, match_id + ".parquet"))
    players = pd.read_parquet(os.path.join(proc_dir, match_id, "players.parquet"))
    # features_team.py already emits ball_y and ball_observed, so the raw columns are renamed
    # rather than allowed to collide into ball_y_x / ball_y_y on the merge.
    raw = pd.read_parquet(os.path.join(proc_dir, match_id, "frames.parquet"),
                          columns=["period", "frame_native", "ball_x", "ball_y", "ball_observed"])
    raw = raw.rename(columns={"ball_x": "raw_ball_x", "ball_y": "raw_ball_y",
                              "ball_observed": "raw_ball_seen"})

    d = (feats.merge(pc, on=["period", "frame_native"], how="left")
              .merge(s2[["period", "frame_native", "pred_hsmm"]],
                     on=["period", "frame_native"], how="left")
              .merge(raw, on=["period", "frame_native"], how="left"))
    d = d[d["label_valid"].astype(bool)].reset_index(drop=True)
    d["prob_home"] = model_prob(d)

    home_path = np.zeros(len(d), dtype=bool)
    for period, g in d.groupby("period", sort=True):
        w = wide(players, period)
        if w is None:
            continue
        fr, X, Y, is_home = w
        idx = pd.Index(fr).get_indexer(g["frame_native"].to_numpy())
        keep = idx >= 0
        rows = g.index.to_numpy()[keep]
        Xg, Yg = X[idx[keep]], Y[idx[keep]]

        sub = slice(None, None, STEP)
        bx = g["raw_ball_x"].to_numpy()[keep][sub] * PITCH_X
        by = g["raw_ball_y"].to_numpy()[keep][sub] * PITCH_Y
        seen = g["raw_ball_seen"].to_numpy()[keep][sub].astype(bool)
        Xs, Ys = Xg[sub], Yg[sub]
        dist = np.sqrt((Xs - bx[:, None]) ** 2 + (Ys - by[:, None]) ** 2)
        dist = np.where(seen[:, None], dist, np.nan)

        em = emissions(g["prob_home"].to_numpy()[keep][sub], dist, seen, is_home,
                       g["pc_home_share"].to_numpy()[keep][sub],
                       tracked=np.isfinite(Xs))
        path = viterbi_path(em, Xs, Ys, is_home)
        coarse = path_to_team(path, is_home)
        home_path[rows] = np.repeat(coarse, STEP)[:len(rows)]

    return d, home_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feat-dir", default="D:/Football Hackathon/stage3/features")
    ap.add_argument("--pc-dir", default="D:/Football Hackathon/stage3/pitch_control")
    ap.add_argument("--pred-dir", default="D:/Football Hackathon/stage2/predictions_h")
    ap.add_argument("--proc-dir", default="D:/Football Hackathon/external_data/processed/idsse")
    ap.add_argument("--out", default="D:/Football Hackathon/stage3/results_path.csv")
    ap.add_argument("--rounds", type=int, default=250)
    ap.add_argument("--rates", type=float, nargs="*", default=None,
                    help="ball-masking rates to score at; default is the full sweep")
    args = ap.parse_args()

    import lightgbm as lgb
    from train_team import DEV, NON_FEATURES, TEST_RATES, mask_ball, score, to_labels

    pm = load_scorer()
    data = {}
    for m in DEV:
        t = pd.read_parquet(os.path.join(args.feat_dir, m + ".parquet"))
        t = t.merge(pd.read_parquet(os.path.join(args.pc_dir, m + ".parquet")),
                    on=["period", "frame_native"], how="left")
        data[m] = t[t["label_valid"].astype(bool)].reset_index(drop=True)
    feat_cols = [c for c in data[DEV[0]].columns if c not in NON_FEATURES]

    rows = []
    for held in DEV:
        tr = pd.concat([data[m][data[m]["is_home"] >= 0] for m in DEV if m != held],
                       ignore_index=True)
        model = lgb.train(
            {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
             "min_data_in_leaf": 200, "feature_fraction": 0.8, "verbosity": -1, "seed": 0},
            lgb.Dataset(tr[feat_cols], label=tr["is_home"]), num_boost_round=args.rounds)

        for rate in (args.rates if args.rates is not None else TEST_RATES):
            def prob(df, rate=rate, model=model):
                return model.predict(mask_ball(df, rate, seed=99)[feat_cols])

            d, home_path = run_match(held, args.feat_dir, args.pc_dir, args.pred_dir,
                                     args.proc_dir, prob)
            truth = d["label"].to_numpy()
            dead = d["pred_hsmm"].fillna(False).to_numpy().astype(bool)
            alive = truth != "D"
            acc = float((home_path[alive] == (truth[alive] == "H")).mean())

            for name, pred in (
                    ("stage3_only", to_labels(truth == "D", home_path)),
                    ("end_to_end", to_labels(dead, home_path))):
                sf, mf, nps, ngs = score(pm, truth, pred, d["period"].to_numpy())
                rows.append({"held_out": held.replace("DFL-MAT-", ""), "ball_rate": rate,
                             "which": name, "team_acc": acc, "spellF1": sf, "macroF1": mf,
                             "pred_spells": nps, "gt_spells": ngs})
            print("  {}  ball hidden {:>3.0f}%   team acc {:.3f}   stage3 {:.3f}   "
                  "end-to-end {:.3f}   spells {:.0f}".format(
                      held.replace("DFL-MAT-", ""), 100 * rate, acc,
                      rows[-2]["spellF1"], rows[-1]["spellF1"], rows[-1]["pred_spells"]))

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print("\n=== PATH MODEL, means over 5 development matches ===")
    for which in ("stage3_only", "end_to_end"):
        sub = out[out["which"] == which]
        print("\n{}:".format(which))
        print(sub.groupby("ball_rate").agg(
            team_acc=("team_acc", "mean"), spellF1=("spellF1", "mean"),
            macroF1=("macroF1", "mean"), pred_spells=("pred_spells", "mean")
        ).to_string(float_format=lambda v: "{:.3f}".format(v)))
    print("\nwrote " + args.out)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- trained emissions

def trained_emissions(match_id, model_path, proc_dir, label_dir, pc_dir, step=STEP):
    """Per-player holder probabilities from the learned emission model.

    Returns a dict keyed by (period, frame_native) -> array over wide-column order, so the path
    decoder can use learned scores in place of the hand-set distance heuristic.
    """
    import lightgbm as lgb
    from train_emissions import FEATURES, player_rows

    booster = lgb.Booster(model_file=model_path)
    rows = player_rows(match_id, proc_dir, label_dir, pc_dir, stride=step, with_labels=False)
    if rows.empty:
        return {}
    rows = rows.assign(q=booster.predict(rows[FEATURES]))
    return {k: g.set_index("col")["q"] for k, g in rows.groupby(["period", "frame_native"])}
