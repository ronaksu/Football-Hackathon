"""Pitch control: who would reach the ball first, computed from player positions alone.

The problem this solves. When the ball is visible, Stage 3 settles possession almost entirely by
asking "who is standing nearest it" -- 76.5% on that one comparison. The shape features built
alongside it are nearly worthless (effect sizes 0.10 to 0.13). So when the ball disappears, which
it does on 41% of the competition's graded frames, the model has nothing to fall back on.

Pitch control is the one quantity that answers the same question *without* the ball: for every
point on the pitch, which team would get there first. It is a stand-in for the ball rather than a
correlate of it.

Implementation follows the simplified time-to-intercept formulation (Spearman; Fernandez & Bornn
2018). For each grid cell:

    t_i   = reaction time applied to the player's current velocity, then remaining distance at a
            realistic top speed
    t_home, t_away = the fastest player of each side
    control_home   = logistic( (t_away - t_home) / tau )

Only the fastest player per side is needed, which keeps this cheap enough to run per frame over a
whole match. Coordinates are direction-normalised exactly as in features_team.py, so "attacking
third" means the same thing in both halves.

Output: one parquet per match under stage3/pitch_control/, to be merged on (period, frame_native).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_team import home_attacks_right, nan_stat, wide  # noqa: E402

PITCH_X, PITCH_Y = 105.0, 68.0
FPS = 25.0

# Movement model. Deliberately plain: a reaction lag, then a realistic sprint.
T_REACT = 0.7      # seconds the player keeps their current velocity before redirecting
V_MAX = 7.0        # m/s, a fast but not freakish sprint
TAU = 0.45         # seconds; how decisively a head start converts into control

# Grid over the pitch. 15 x 10 = 150 cells of 7 x 6.8 m, which is finer than the question needs.
NX, NY = 15, 10
CHUNK = 2000


def grid_centres():
    gx = (np.arange(NX) + 0.5) * (PITCH_X / NX)
    gy = (np.arange(NY) + 0.5) * (PITCH_Y / NY)
    mx, my = np.meshgrid(gx, gy, indexing="ij")
    return mx.ravel(), my.ravel()


def velocities(px, py):
    n = px.shape[0]
    i0 = np.clip(np.arange(n) - 2, 0, n - 1)
    i1 = np.clip(np.arange(n) + 2, 0, n - 1)
    dt = (i1 - i0) / FPS
    dt[dt == 0] = 1.0 / FPS
    return (px[i1] - px[i0]) / dt[:, None], (py[i1] - py[i0]) / dt[:, None]


def control_surface(X, Y, is_home, cells_x, cells_y):
    """(n_frames, n_cells) probability that HOME controls each cell."""
    VX, VY = velocities(X, Y)
    # Where each player can be once they have finished reacting.
    RX = X + VX * T_REACT
    RY = Y + VY * T_REACT

    n = X.shape[0]
    out = np.full((n, len(cells_x)), np.nan, dtype=np.float32)

    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        dx = RX[a:b, :, None] - cells_x[None, None, :]
        dy = RY[a:b, :, None] - cells_y[None, None, :]
        t = T_REACT + np.sqrt(dx * dx + dy * dy) / V_MAX    # (chunk, players, cells)
        t = np.where(np.isfinite(t), t, np.inf)

        t_home = t[:, is_home, :].min(axis=1)
        t_away = t[:, ~is_home, :].min(axis=1)
        # Neither side tracked here: leave it unknown rather than inventing a half.
        both_blind = ~np.isfinite(t_home) & ~np.isfinite(t_away)
        with np.errstate(over="ignore", invalid="ignore"):
            p = 1.0 / (1.0 + np.exp((t_home - t_away) / TAU))
        # One side missing entirely means the other side owns the ground it can reach.
        p = np.where(np.isfinite(t_home) & ~np.isfinite(t_away), 1.0, p)
        p = np.where(~np.isfinite(t_home) & np.isfinite(t_away), 0.0, p)
        out[a:b] = np.where(both_blind, np.nan, p).astype(np.float32)

    return out


def period_features(frames, X, Y, is_home, ball_x, ball_y, ball_seen):
    if not home_attacks_right(X, is_home):
        X = PITCH_X - X
        ball_x = PITCH_X - ball_x

    cx, cy = grid_centres()
    P = control_surface(X, Y, is_home, cx, cy)

    f = {"frame_native": frames}

    # How much of the pitch each side owns, and how decisively.
    f["pc_home_share"] = nan_stat(np.nanmean, P)                       # soft share
    f["pc_home_cells"] = nan_stat(np.nanmean, (P > 0.5).astype(float))  # hard share
    f["pc_decisive"] = nan_stat(np.nanmean, np.abs(P - 0.5) * 2.0)      # how contested overall

    # By third, in normalised coordinates: home attacks +x.
    att = cx > (2.0 * PITCH_X / 3.0)
    mid = (cx >= PITCH_X / 3.0) & (cx <= 2.0 * PITCH_X / 3.0)
    dfn = cx < (PITCH_X / 3.0)
    f["pc_home_att_third"] = nan_stat(np.nanmean, P[:, att])
    f["pc_home_mid_third"] = nan_stat(np.nanmean, P[:, mid])
    f["pc_home_def_third"] = nan_stat(np.nanmean, P[:, dfn])
    f["pc_third_tilt"] = f["pc_home_att_third"] - f["pc_home_def_third"]

    # Control at the ball, when the ball is there. This is the direct substitute for
    # "who is nearest the ball", but it accounts for momentum and for who else is arriving.
    seen = ball_seen.astype(bool)
    ix = np.clip((ball_x / (PITCH_X / NX)).astype(int), 0, NX - 1)
    iy = np.clip((ball_y / (PITCH_Y / NY)).astype(int), 0, NY - 1)
    flat = ix * NY + iy
    at_ball = P[np.arange(len(frames)), flat]
    f["pc_at_ball"] = np.where(seen, at_ball, np.nan)

    # The single most-contested cell, and where the control frontier sits. During settled
    # possession the frontier is deep in one half; in a scramble it is in the middle.
    with np.errstate(invalid="ignore"):
        frontier = np.nansum(P * cx[None, :], axis=1) / np.maximum(
            np.nansum(P, axis=1), 1e-6)
    f["pc_frontier_x"] = frontier

    df = pd.DataFrame(f)
    for col in ("pc_home_share", "pc_third_tilt", "pc_at_ball", "pc_frontier_x", "pc_decisive"):
        s = df[col].astype(float)
        for w in (25, 75):
            df["{}_m{}".format(col, w)] = s.rolling(w, center=True, min_periods=1).mean()
    return df


def build_match(proc_dir, match_id, out_dir):
    frames = pd.read_parquet(
        os.path.join(proc_dir, match_id, "frames.parquet"),
        columns=["period", "frame_native", "ball_x", "ball_y", "ball_observed"])
    players = pd.read_parquet(os.path.join(proc_dir, match_id, "players.parquet"))

    parts = []
    for period in sorted(frames["period"].unique()):
        w = wide(players, period)
        if w is None:
            continue
        fr, X, Y, is_home = w
        fsub = frames[frames["period"] == period].set_index("frame_native").reindex(fr)
        d = period_features(
            fr, X, Y, is_home,
            fsub["ball_x"].to_numpy() * PITCH_X,
            fsub["ball_y"].to_numpy() * PITCH_Y,
            fsub["ball_observed"].to_numpy())
        d.insert(0, "period", period)
        parts.append(d)

    out = pd.concat(parts, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    out.to_parquet(os.path.join(out_dir, match_id + ".parquet"), index=False)
    print("  {}  {:>7,} frames  {} pitch-control features  mean home share {:.3f}".format(
        match_id, len(out), len(out.columns) - 2, out["pc_home_share"].mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc-dir", default="D:/Football Hackathon/external_data/processed/idsse")
    ap.add_argument("--out-dir", default="D:/Football Hackathon/stage3/pitch_control")
    ap.add_argument("--matches", nargs="*", default=None)
    args = ap.parse_args()

    ids = args.matches or sorted(
        d for d in os.listdir(args.proc_dir)
        if os.path.exists(os.path.join(args.proc_dir, d, "frames.parquet")))
    print("building pitch control for {} matches".format(len(ids)))
    for m in ids:
        build_match(args.proc_dir, m, args.out_dir)


if __name__ == "__main__":
    main()
