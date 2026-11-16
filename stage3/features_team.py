"""Stage 3 features: which team is in control, from geometry plus the ball when it is there.

Two things make this a different problem from Stage 2.

1. IT NEEDS PLAYING DIRECTION. "Pushed up the pitch" only means something once you know which
   way a team is attacking, and that flips at half-time. No role or identity information is
   available -- the competition supplies 46 anonymous slots -- so direction is inferred from the
   data: over a whole period the team defending the left goal keeps its deepest player (in
   practice the goalkeeper) near x=0, and the other team does not. Coordinates are then flipped
   so that home always attacks +x.

2. IT MAY USE THE BALL, carefully. When the ball is visible, the margin between the nearest home
   player and the nearest away player is overwhelmingly the strongest signal there is, and
   throwing it away would be perverse. But the competition loses the ball on 41% of graded frames
   while IDSSE never loses it, so a model trained here would never learn the fallback. Every ball
   feature is therefore paired with an explicit `ball_observed` flag and set to NaN when the ball
   is not there, and training masks the ball at a range of rates. LightGBM routes NaN down its own
   branch, so "I cannot see the ball" is a state the model can learn, not a number it guesses.

Output: one parquet per match under stage3/features/, frame-aligned to processed/idsse/.
"""

import argparse
import os

import numpy as np
import pandas as pd

PITCH_X, PITCH_Y = 105.0, 68.0
FPS = 25.0
WINDOWS = (25, 75)

BALL_COLS = (
    "ball_x_n", "ball_y", "ball_third", "d_ball_home", "d_ball_away",
    "ball_margin", "ball_nearer_home", "ball_adv_x",
)


def wide(players, period):
    g = players[players["period"] == period]
    if g.empty:
        return None
    g = g.copy()
    g["col"] = g["team"].astype(str) + "_" + g["slot"].astype(str)
    x = g.pivot_table(index="frame_native", columns="col", values="x", aggfunc="first")
    y = g.pivot_table(index="frame_native", columns="col", values="y", aggfunc="first")
    x, y = x.sort_index(), y.sort_index().reindex(columns=x.columns)
    is_home = np.array([c.startswith("home_") for c in x.columns])
    return x.index.to_numpy(), x.to_numpy() * PITCH_X, y.to_numpy() * PITCH_Y, is_home


def home_attacks_right(X, is_home):
    """Infer playing direction for this period.

    The team defending the left goal keeps its deepest player pinned near x=0 for the whole
    period; the other team's deepest player is an outfielder much further up. Comparing the mean
    of each team's minimum x therefore separates them cleanly, with no role labels and no
    assumption about which half of the match this is.
    """
    with np.errstate(invalid="ignore"):
        deep_home = np.nanmean(np.nanmin(X[:, is_home], axis=1))
        deep_away = np.nanmean(np.nanmin(X[:, ~is_home], axis=1))
    # Home defends the left goal -> home attacks right.
    return bool(deep_home < deep_away)


def nan_stat(fn, a, axis=1):
    with np.errstate(invalid="ignore"):
        out = fn(a, axis=axis)
    return np.where(np.isfinite(out), out, np.nan)


def period_features(frames, X, Y, is_home, ball_x, ball_y, ball_seen):
    H, A = is_home, ~is_home

    if not home_attacks_right(X, is_home):
        X = PITCH_X - X
        ball_x = PITCH_X - ball_x

    f = {"frame_native": frames}
    f["home_attacks_right"] = 1.0  # after normalisation this is always true; kept for traceability

    # --- who is pushed up the pitch
    cxh, cxa = nan_stat(np.nanmean, X[:, H]), nan_stat(np.nanmean, X[:, A])
    f["cx_home"], f["cx_away"] = cxh, cxa
    f["depth_diff"] = cxh - cxa
    f["cy_spread_home"] = nan_stat(np.nanstd, Y[:, H])
    f["cy_spread_away"] = nan_stat(np.nanstd, Y[:, A])

    # The attacking team stretches, the defending team squeezes.
    sh = nan_stat(np.nanstd, X[:, H])
    sa = nan_stat(np.nanstd, X[:, A])
    f["spread_x_home"], f["spread_x_away"] = sh, sa
    f["spread_ratio"] = sh / np.maximum(sa, 0.1)

    # Thirds, in normalised coordinates: home attacks +x, so x > 70 is home's attacking third.
    f["n_home_att_third"] = np.nansum(X[:, H] > 70.0, axis=1)
    f["n_away_att_third"] = np.nansum(X[:, A] < 35.0, axis=1)
    f["n_home_own_third"] = np.nansum(X[:, H] < 35.0, axis=1)
    f["n_away_own_third"] = np.nansum(X[:, A] > 70.0, axis=1)
    f["third_balance"] = f["n_home_att_third"] - f["n_away_att_third"]

    # Furthest player forward for each side -- who is holding the high line.
    f["front_home"] = nan_stat(np.nanmax, X[:, H])
    f["front_away"] = nan_stat(np.nanmin, X[:, A])
    f["line_gap"] = f["front_home"] - f["front_away"]

    # --- the ball, when it is there
    seen = ball_seen.astype(bool)
    bx = np.where(seen, ball_x, np.nan)
    by = np.where(seen, ball_y, np.nan)

    dh = np.sqrt((X[:, H] - bx[:, None]) ** 2 + (Y[:, H] - by[:, None]) ** 2)
    da = np.sqrt((X[:, A] - bx[:, None]) ** 2 + (Y[:, A] - by[:, None]) ** 2)
    d_home = nan_stat(np.nanmin, dh)
    d_away = nan_stat(np.nanmin, da)

    f["ball_observed"] = seen.astype(float)
    f["ball_x_n"] = bx
    f["ball_y"] = by
    f["ball_third"] = np.where(seen, np.clip(bx / 35.0, 0, 2.999).astype(float), np.nan)
    f["d_ball_home"] = d_home
    f["d_ball_away"] = d_away
    f["ball_margin"] = d_away - d_home          # positive: home is closer
    f["ball_nearer_home"] = np.where(np.isfinite(d_home) & np.isfinite(d_away),
                                     (d_home < d_away).astype(float), np.nan)
    # How far up the pitch the ball is relative to the midpoint of the two teams.
    f["ball_adv_x"] = bx - (cxh + cxa) / 2.0

    df = pd.DataFrame(f)

    # --- temporal context. Possession is a state that persists, not a per-frame coin flip.
    for col in ("depth_diff", "third_balance", "ball_margin", "ball_nearer_home",
                "spread_ratio", "line_gap"):
        s = df[col].astype(float)
        for w in WINDOWS:
            df["{}_m{}".format(col, w)] = s.rolling(w, center=True, min_periods=1).mean()
    return df


def build_match(proc_dir, match_id, out_dir):
    frames = pd.read_parquet(
        os.path.join(proc_dir, match_id, "frames.parquet"),
        columns=["match_id", "period", "frame_native", "label", "label_valid",
                 "ball_x", "ball_y", "ball_observed"],
    )
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
            fsub["ball_observed"].to_numpy(),
        )
        d.insert(0, "period", period)
        parts.append(d)

    feats = pd.concat(parts, ignore_index=True)
    out = frames.drop(columns=["ball_x", "ball_y", "ball_observed"]).merge(
        feats, on=["period", "frame_native"], how="inner")

    # Stage 3 answers "which team", so it is trained and scored on live frames only.
    out["is_home"] = np.where(out["label"] == "H", 1,
                              np.where(out["label"] == "A", 0, -1)).astype("int8")

    os.makedirs(out_dir, exist_ok=True)
    out.to_parquet(os.path.join(out_dir, match_id + ".parquet"), index=False)

    alive = out[out["is_home"] >= 0]
    print("  {}  {:>7,} frames  {:>6,} alive  home share {:.1f}%  ball seen {:.1f}%".format(
        match_id, len(out), len(alive), 100 * alive["is_home"].mean(),
        100 * out["ball_observed"].mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc-dir", default="D:/Football Hackathon/external_data/processed/idsse")
    ap.add_argument("--out-dir", default="D:/Football Hackathon/stage3/features")
    ap.add_argument("--matches", nargs="*", default=None)
    args = ap.parse_args()

    ids = args.matches or sorted(
        d for d in os.listdir(args.proc_dir)
        if os.path.exists(os.path.join(args.proc_dir, d, "frames.parquet")))
    print("building Stage 3 team features for {} matches".format(len(ids)))
    for m in ids:
        build_match(args.proc_dir, m, args.out_dir)


if __name__ == "__main__":
    main()
