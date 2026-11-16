"""Stage 2 feature extraction: what a dead ball looks like, using players only.

DELIBERATELY NO BALL FEATURES. The competition loses the ball on ~41% of graded frames, and the
sources we train on do not lose it the same way (IDSSE never loses it at all). A detector that
leans on the ball would learn "ball missing => dead ball", which is the exact failure the
published baseline already makes. Built from the 22 player positions alone, this stage cannot
make that mistake, and it transfers to the competition unchanged whether the ball is visible or
not.

What the features try to capture, in the order a human would notice them:

  everyone stops          -> speed statistics, and the fraction of players below walking pace
  the shape relaxes       -> spread, bounding boxes, how far the two teams' centroids sit apart
  they walk to a spot     -> players near a touchline, a goal line, a corner, a penalty area
  it holds for a while    -> the same quantities averaged over 1, 3 and 5 second windows

Output: one parquet per match under stage2/features/, frame-aligned to processed/idsse/.
"""

import argparse
import os

import numpy as np
import pandas as pd

PITCH_X = 105.0
PITCH_Y = 68.0
FPS = 25.0

# Penalty area, in metres from the left goal line.
BOX_DEPTH = 16.5
BOX_HALF_WIDTH = 20.16  # 40.32 m wide, centred

# Rolling context windows, in frames: 1 s, 3 s, 5 s.
WINDOWS = (25, 75, 125)

# The features that get temporal context. Everything else is used as-is; adding windows to all of
# them triples the column count for very little.
CONTEXT_COLS = (
    "spd_mean", "spd_p90", "frac_walking", "frac_sprinting",
    "spread_all", "centroid_dist", "n_near_touchline", "n_near_goalline",
    "n_outside_pitch", "line_tight_min", "anchor_score",
)

# Leading / trailing windows, in frames: 0.6 s and 2 s. Short on purpose -- their job is to mark
# the instant the picture changes, not to describe a state.
LEAD_WINDOWS = (15, 50)
LEAD_COLS = ("spd_mean", "spd_p90", "n_outside_pitch", "line_tight_min", "spread_all")

# Columns added by the hardening pass. train.py can exclude these to reproduce the original
# model exactly, which is what makes the before/after comparison honest.
HARDENED_PREFIXES = (
    "line_tight_", "line_depth_", "line_width_", "bunch_k3_",
    "lone_isolation_m", "lone_d_line_m", "anchor_score", "n_in_centre_circle",
)


def is_hardened_column(name):
    """True for a column introduced by the hardening pass (direct, context or lead window)."""
    if name.startswith(HARDENED_PREFIXES):
        return True
    # leading / trailing windows are new whatever they are computed on
    return ("_b" in name or "_f" in name or "_step" in name) and any(
        name.startswith(c + "_") for c in LEAD_COLS)


def player_speeds(px, py):
    """Speeds in m/s for an (n_frames, n_players) pair of coordinate arrays.

    Central difference over +/-2 frames. The wider stencil is deliberate: at 25 Hz a 1-frame
    difference is mostly tracking jitter, and we care about whether a player is walking or
    running, not about instantaneous velocity.
    """
    n = px.shape[0]
    i0 = np.clip(np.arange(n) - 2, 0, n - 1)
    i1 = np.clip(np.arange(n) + 2, 0, n - 1)
    dt = (i1 - i0) / FPS
    dt[dt == 0] = 1.0 / FPS
    dx = px[i1] - px[i0]
    dy = py[i1] - py[i0]
    return np.sqrt(dx * dx + dy * dy) / dt[:, None]


def degrade_pan(X, Y, seed=0, half_width=30.0, blackout_rate=0.04, blackout_len=25):
    """Hide players the way a broadcast feed does, in place of a clean 16-camera view.

    The competition's table comes from a single moving camera: it frames the action, so players
    far from the ball simply are not there, and when it pans or cuts to a replay nobody is. The
    IDSSE feed has none of that -- 11.0 of 11.0 players on essentially every frame -- so a model
    tuned on it has never had to cope.

    Two mechanisms, both contiguous in time rather than per-frame coin flips, because that is how
    the real loss behaves:
      framing   only players within `half_width` metres of the camera centre survive, and the
                centre follows a smoothed version of where the players are
      blackout  short runs where nothing is tracked at all, standing in for cuts and replays
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    centre = pd.Series(np.nanmean(X, axis=1)).rolling(25, center=True, min_periods=1).mean()
    centre = centre.to_numpy()[:, None]
    visible = np.abs(X - centre) <= half_width

    n_black = int(n * blackout_rate / blackout_len)
    for start in rng.integers(0, max(1, n - blackout_len), size=max(0, n_black)):
        visible[start:start + blackout_len, :] = False

    Xd, Yd = X.copy(), Y.copy()
    Xd[~visible] = np.nan
    Yd[~visible] = np.nan
    return Xd, Yd


def wide_positions(players, match_id, period):
    """(frames, x_matrix, y_matrix, team_mask) for one period, in metres."""
    g = players[(players["match_id"] == match_id) & (players["period"] == period)]
    if g.empty:
        return None

    g = g.copy()
    g["col"] = g["team"].astype(str) + "_" + g["slot"].astype(str)

    x = g.pivot_table(index="frame_native", columns="col", values="x", aggfunc="first")
    y = g.pivot_table(index="frame_native", columns="col", values="y", aggfunc="first")
    x, y = x.sort_index(), y.sort_index().reindex(columns=x.columns)

    frames = x.index.to_numpy()
    is_home = np.array([c.startswith("home_") for c in x.columns])
    return frames, x.to_numpy() * PITCH_X, y.to_numpy() * PITCH_Y, is_home


def nan_stat(fn, a, axis=1):
    """Apply a nan-aware statistic, tolerating all-NaN rows without warning noise."""
    with np.errstate(invalid="ignore"):
        out = fn(a, axis=axis)
    return np.where(np.isfinite(out), out, np.nan)


def period_features(frames, X, Y, is_home):
    """One row per frame. All inputs in metres."""
    S = player_speeds(X, Y)
    n = len(frames)

    H, A = is_home, ~is_home
    f = {"frame_native": frames}

    # --- everyone stops
    f["spd_mean"] = nan_stat(np.nanmean, S)
    f["spd_med"] = nan_stat(np.nanmedian, S)
    f["spd_p90"] = nan_stat(lambda a, axis: np.nanpercentile(a, 90, axis=axis), S)
    f["spd_max"] = nan_stat(np.nanmax, S)
    f["spd_std"] = nan_stat(np.nanstd, S)
    f["spd_mean_home"] = nan_stat(np.nanmean, S[:, H])
    f["spd_mean_away"] = nan_stat(np.nanmean, S[:, A])
    f["spd_team_gap"] = np.abs(f["spd_mean_home"] - f["spd_mean_away"])

    f["n_tracked"] = np.isfinite(X).sum(axis=1).astype(float)
    f["n_tracked_home"] = np.isfinite(X[:, H]).sum(axis=1).astype(float)
    f["n_tracked_away"] = np.isfinite(X[:, A]).sum(axis=1).astype(float)

    valid = np.isfinite(S).sum(axis=1).astype(float)
    valid[valid == 0] = np.nan
    for name, thr in (("frac_still", 0.5), ("frac_walking", 1.5), ("frac_jogging", 3.0)):
        f[name] = np.nansum(S < thr, axis=1) / valid
    f["frac_sprinting"] = np.nansum(S > 5.5, axis=1) / valid

    # --- the shape
    cxh, cyh = nan_stat(np.nanmean, X[:, H]), nan_stat(np.nanmean, Y[:, H])
    cxa, cya = nan_stat(np.nanmean, X[:, A]), nan_stat(np.nanmean, Y[:, A])
    f["centroid_dist"] = np.hypot(cxh - cxa, cyh - cya)
    f["centroid_dx"] = np.abs(cxh - cxa)
    f["centroid_dy"] = np.abs(cyh - cya)
    f["centroid_x_mid"] = (cxh + cxa) / 2.0

    for tag, M, cx, cy in (("home", H, cxh, cyh), ("away", A, cxa, cya)):
        d = np.hypot(X[:, M] - cx[:, None], Y[:, M] - cy[:, None])
        f["spread_" + tag] = nan_stat(np.nanmean, d)
        f["width_" + tag] = (nan_stat(np.nanmax, Y[:, M]) - nan_stat(np.nanmin, Y[:, M]))
        f["length_" + tag] = (nan_stat(np.nanmax, X[:, M]) - nan_stat(np.nanmin, X[:, M]))
        f["area_" + tag] = f["width_" + tag] * f["length_" + tag]

    cxall, cyall = nan_stat(np.nanmean, X), nan_stat(np.nanmean, Y)
    f["spread_all"] = nan_stat(np.nanmean, np.hypot(X - cxall[:, None], Y - cyall[:, None]))
    f["spread_x"] = nan_stat(np.nanstd, X)
    f["spread_y"] = nan_stat(np.nanstd, Y)
    f["team_overlap"] = (
        np.minimum(nan_stat(np.nanmax, X[:, H]), nan_stat(np.nanmax, X[:, A]))
        - np.maximum(nan_stat(np.nanmin, X[:, H]), nan_stat(np.nanmin, X[:, A]))
    )

    # --- walking to a restart spot
    d_touch = np.minimum(Y, PITCH_Y - Y)
    d_goal = np.minimum(X, PITCH_X - X)
    f["n_near_touchline"] = np.nansum(d_touch < 3.0, axis=1)
    f["n_near_goalline"] = np.nansum(d_goal < 3.0, axis=1)
    f["min_d_touchline"] = nan_stat(np.nanmin, d_touch)
    f["min_d_goalline"] = nan_stat(np.nanmin, d_goal)
    f["n_near_corner"] = np.nansum((d_touch < 6.0) & (d_goal < 6.0), axis=1)

    # Players physically off the field of play. TRACAB tracks beyond the lines (coordinates run
    # to about 3.6 m outside a touchline), so a thrower standing off the pitch is directly
    # observable. This is the clearest available cue for the short same-team stoppages -- quick
    # throw-ins -- that cost a whole possession spell when missed.
    outside_touch = (Y < 0) | (Y > PITCH_Y)
    outside_goal = (X < 0) | (X > PITCH_X)
    f["n_outside_touchline"] = np.nansum(outside_touch, axis=1)
    f["n_outside_goalline"] = np.nansum(outside_goal, axis=1)
    f["n_outside_pitch"] = np.nansum(outside_touch | outside_goal, axis=1)
    f["max_outside_m"] = nan_stat(
        np.nanmax, np.maximum.reduce([-Y, Y - PITCH_Y, -X, X - PITCH_X]))

    in_box = (d_goal < BOX_DEPTH) & (np.abs(Y - PITCH_Y / 2.0) < BOX_HALF_WIDTH)
    f["n_in_box"] = np.nansum(in_box, axis=1)
    f["n_in_box_home"] = np.nansum(in_box[:, H], axis=1)
    f["n_in_box_away"] = np.nansum(in_box[:, A], axis=1)

    # Kick-off signature: both teams sitting either side of the halfway line, nobody over it.
    f["n_left_half"] = np.nansum(X < PITCH_X / 2.0, axis=1)
    f["halfway_split"] = np.abs(f["n_left_half"] - 11.0)

    # --- is the picture contracting or expanding
    for col in ("spread_all", "centroid_dist", "spd_mean"):
        v = np.asarray(f[col], dtype=float)
        f["d_" + col] = np.gradient(np.nan_to_num(v, nan=np.nanmean(v))) * FPS

    # --- CHANGE 1: defensive-line straightness (Law 11 is why the line exists)
    #
    # While the ball is live the defending team holds a flat back line to keep the offside trap.
    # The moment play stops, that line has no reason to exist and it disperses. This is readable
    # from positions alone and is the strongest candidate for separating a lull in live play from
    # a genuine stoppage -- the model's largest error class.
    #
    # Role-free and direction-free on purpose: the competition supplies 46 anonymous slots with
    # no positions and no playing direction, and direction flips at half-time anyway. So for each
    # team we measure the tightness of the four players at EACH end and keep the tighter one;
    # whichever end the team is defending is the one that forms a line.
    for tag, M in (("home", H), ("away", A)):
        xs = np.sort(np.where(np.isfinite(X[:, M]), X[:, M], np.nan), axis=1)
        ys = Y[:, M]
        low, high = xs[:, :4], xs[:, -4:]
        t_low = nan_stat(np.nanstd, low)
        t_high = nan_stat(np.nanstd, high)
        f["line_tight_" + tag] = np.fmin(t_low, t_high)
        f["line_depth_" + tag] = np.fmin(
            nan_stat(np.nanmax, low) - nan_stat(np.nanmin, low),
            nan_stat(np.nanmax, high) - nan_stat(np.nanmin, high))
        # A real defensive line also spans the pitch width rather than bunching.
        f["line_width_" + tag] = nan_stat(np.nanmax, ys) - nan_stat(np.nanmin, ys)

    f["line_tight_min"] = np.fmin(f["line_tight_home"], f["line_tight_away"])
    f["line_tight_sum"] = f["line_tight_home"] + f["line_tight_away"]

    # --- CHANGE 2: restart-anchor convergence
    #
    # A stoppage ends at a fixed point -- a touchline spot, a corner arc, the penalty spot -- so
    # during one, players converge on an anchor and a wall or a box crowd forms. A lull in live
    # play has no anchor. Two cheap, direction-free proxies:
    #   bunch_k3  the tightest group of four teammates anywhere on the pitch (a wall, or bodies
    #             packed into the box for a set piece)
    #   lone_*    the most isolated player and how close they are to a line -- the taker standing
    #             over the ball, away from everyone else
    for tag, M in (("home", H), ("away", A)):
        f["bunch_k3_" + tag] = _kth_neighbour(X[:, M], Y[:, M], k=3)

    iso, iso_line = _isolation(X, Y, d_touch, d_goal)
    f["lone_isolation_m"] = iso
    f["lone_d_line_m"] = iso_line
    f["anchor_score"] = np.where(iso > 0, iso / np.maximum(iso_line, 0.5), 0.0)

    # Kick-off after a goal puts everybody in their own half with the circle nearly empty.
    in_circle = np.hypot(X - PITCH_X / 2.0, Y - PITCH_Y / 2.0) < 9.15
    f["n_in_centre_circle"] = np.nansum(in_circle, axis=1)

    df = pd.DataFrame(f)

    # --- temporal context
    for col in CONTEXT_COLS:
        s = df[col].astype(float)
        for w in WINDOWS:
            roll = s.rolling(w, center=True, min_periods=1)
            df["{}_m{}".format(col, w)] = roll.mean()
            df["{}_s{}".format(col, w)] = roll.std()
        df[col + "_dev"] = s - df["{}_m{}".format(col, WINDOWS[-1])]

    # --- CHANGE 3: leading windows
    #
    # Every context window above is centred, so the model needs about a second of hindsight
    # before it will move -- which is exactly the +1.10 s median start lag measured on the first
    # version. Trailing and leading means of the same quantity, and the step between them, give
    # the classifier an explicit change-point signal instead of making it infer one.
    for col in LEAD_COLS:
        s = df[col].astype(float)
        for w in LEAD_WINDOWS:
            back = s.rolling(w, min_periods=1).mean()
            fwd = s[::-1].rolling(w, min_periods=1).mean()[::-1]
            df["{}_b{}".format(col, w)] = back
            df["{}_f{}".format(col, w)] = fwd
            df["{}_step{}".format(col, w)] = fwd - back

    assert len(df) == n
    return df


def _kth_neighbour(px, py, k=3, chunk=20000):
    """Per frame: the smallest distance from any player to their k-th nearest teammate.

    Small values mean four or more players are packed together -- a defensive wall, or bodies in
    the box at a set piece. Chunked because the full pairwise stack would be a few hundred MB.
    """
    n, m = px.shape
    out = np.full(n, np.nan)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        dx = px[a:b, :, None] - px[a:b, None, :]
        dy = py[a:b, :, None] - py[a:b, None, :]
        d = np.sqrt(dx * dx + dy * dy)
        d[~np.isfinite(d)] = np.inf
        d.sort(axis=2)
        kth = d[:, :, k] if m > k else d[:, :, -1]
        kth[~np.isfinite(kth)] = np.nan
        out[a:b] = nan_stat(np.nanmin, kth)
    return out


def _isolation(X, Y, d_touch, d_goal, chunk=20000):
    """Per frame: how far the most isolated player is from anyone else, and from the nearest line.

    A restart taker stands alone over the ball near a line while everyone else waits. High
    isolation together with a short distance to a line is the signature; neither alone is.
    """
    n = X.shape[0]
    iso = np.full(n, np.nan)
    iso_line = np.full(n, np.nan)
    d_line = np.minimum(d_touch, d_goal)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        dx = X[a:b, :, None] - X[a:b, None, :]
        dy = Y[a:b, :, None] - Y[a:b, None, :]
        d = np.sqrt(dx * dx + dy * dy)
        d[~np.isfinite(d)] = np.inf
        np.einsum("ijj->ij", d)[...] = np.inf  # ignore self
        nearest = d.min(axis=2)
        nearest[~np.isfinite(nearest)] = np.nan
        who = np.nanargmax(np.where(np.isfinite(nearest), nearest, -np.inf), axis=1)
        rows = np.arange(b - a)
        iso[a:b] = nearest[rows, who]
        iso_line[a:b] = d_line[a:b][rows, who]
    return iso, iso_line


def build_match(proc_dir, match_id, out_dir, degrade=None, seed=0, pan_kw=None):
    frames = pd.read_parquet(
        os.path.join(proc_dir, match_id, "frames.parquet"),
        columns=["match_id", "period", "frame_native", "label", "label_valid"],
    )
    players = pd.read_parquet(os.path.join(proc_dir, match_id, "players.parquet"))

    parts = []
    for period in sorted(frames["period"].unique()):
        wide = wide_positions(players, match_id, period)
        if wide is None:
            continue
        frames_i, Xp, Yp, is_home = wide
        if degrade == "pan":
            Xp, Yp = degrade_pan(Xp, Yp, seed=seed + period, **(pan_kw or {}))
        d = period_features(frames_i, Xp, Yp, is_home)
        d.insert(0, "period", period)
        parts.append(d)

    feats = pd.concat(parts, ignore_index=True)
    out = frames.merge(feats, on=["period", "frame_native"], how="inner")
    out["is_dead"] = (out["label"] == "D").astype("int8")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, match_id + ".parquet")
    out.to_parquet(path, index=False)

    n_feat = len([c for c in out.columns if c not in
                  ("match_id", "period", "frame_native", "label", "label_valid", "is_dead")])
    seen = out["n_tracked"].mean() if "n_tracked" in out else float("nan")
    print("  {}  {:>7,} frames  {} features  dead {:.1f}%  mean players seen {:.1f}".format(
        match_id, len(out), n_feat,
        100 * out.loc[out["label_valid"].astype(bool), "is_dead"].mean(), seen))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc-dir", default="D:/Football Hackathon/external_data/processed/idsse")
    ap.add_argument("--out-dir", default="D:/Football Hackathon/stage2/features")
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--degrade", default=None, choices=["pan"],
                    help="hide players the way a single broadcast camera would")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pan-width", type=float, default=30.0,
                    help="half-width in metres of the simulated camera frame")
    ap.add_argument("--blackout-rate", type=float, default=0.04,
                    help="fraction of frames with nothing tracked at all")
    args = ap.parse_args()

    ids = args.matches or sorted(
        d for d in os.listdir(args.proc_dir)
        if os.path.exists(os.path.join(args.proc_dir, d, "frames.parquet"))
    )
    print("building features for {} matches (players only, no ball){}".format(
        len(ids), "  DEGRADED: " + args.degrade if args.degrade else ""))
    for m in ids:
        build_match(args.proc_dir, m, args.out_dir, degrade=args.degrade, seed=args.seed,
                    pan_kw={"half_width": args.pan_width,
                            "blackout_rate": args.blackout_rate})


if __name__ == "__main__":
    main()
