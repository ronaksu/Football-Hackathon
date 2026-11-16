"""Recurring failure patterns across the five external development matches.

Each candidate pattern is a condition computable from the data at any frame -- never a reference
to a particular frame, match or minute -- so it can be carried forward as a reusable flag.

For every flag this reports:
  coverage   share of graded frames where the flag is on
  err_in     error rate on flagged frames
  err_out    error rate on unflagged frames
  lift       err_in / err_out
  share      share of ALL errors that fall on flagged frames

Lift is association, not cause. A flag that marks the frames where the model is unreliable is
useful whether or not it explains anything, and nothing here claims it explains anything. Where a
plausible mechanism exists it is written down as a hypothesis and labelled as one.

Reads only open-access IDSSE development matches. The two sealed matches are never loaded.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
from train import runs_of  # noqa: E402

FPS = 25
PITCH_X, PITCH_Y = 105.0, 68.0
DEV = ["DFL-MAT-J03WOH", "DFL-MAT-J03WOY", "DFL-MAT-J03WPY",
       "DFL-MAT-J03WQQ", "DFL-MAT-J03WR9"]


def build(match, paths):
    """One row per graded frame, with the model's answer, the reference, and the flag inputs."""
    f = pd.read_parquet(os.path.join(paths["proc"], match, "frames.parquet"),
                        columns=["period", "frame_native", "label", "label_valid",
                                 "ball_x", "ball_y", "ball_observed",
                                 "nearest_home_m", "nearest_away_m", "nearest_margin_m"])
    s2 = pd.read_parquet(os.path.join(paths["s2pred"], match + ".parquet"),
                         columns=["period", "frame_native", "prob", "pred_hsmm"])
    s3 = pd.read_parquet(os.path.join(paths["s3pred"], match + ".parquet"),
                         columns=["period", "frame_native", "p_home", "home_end_to_end",
                                  "ball_observed_masked"])
    s2f = pd.read_parquet(os.path.join(paths["s2feat"], match + ".parquet"),
                          columns=["period", "frame_native", "n_tracked", "spd_mean"])
    lab = pd.read_parquet(os.path.join(paths["labels"], match + ".parquet"),
                          columns=["period", "frame_native", "holder_side"])

    d = f
    for other in (s2, s3, s2f, lab):
        d = d.merge(other, on=["period", "frame_native"], how="left")

    # Analyse the model that is actually shipped, which is the restart-hardened one.
    rp = os.path.join(paths["refined"], match + ".parquet")
    if os.path.exists(rp):
        ref = pd.read_parquet(rp, columns=["period", "frame_native", "pred_dead_refined",
                                           "home_refined"])
        d = d.merge(ref, on=["period", "frame_native"], how="left")
        d["pred_hsmm"] = d["pred_dead_refined"].fillna(d["pred_hsmm"])
        d["home_end_to_end"] = d["home_refined"].fillna(d["home_end_to_end"])

    d = d[d["label_valid"].astype(bool)].reset_index(drop=True)
    d["pred_hsmm"] = d["pred_hsmm"].fillna(False).astype(bool)
    d["match"] = match.replace("DFL-MAT-", "")

    parts = []
    for _, g in d.groupby("period", sort=True):
        g = g.reset_index(drop=True)
        truth = g["label"].to_numpy()
        pred = np.where(g["pred_hsmm"].to_numpy(), "D",
                        np.where(g["home_end_to_end"].fillna(False).to_numpy().astype(bool),
                                 "H", "A"))
        g["pred"] = pred
        g["wrong"] = pred != truth
        # team error, restricted to frames both sides agree are live: this isolates stage 3
        g["wrong_team"] = (truth != "D") & (pred != "D") & (pred != truth)
        g["wrong_dead"] = (truth == "D") != (pred == "D")

        # --- flag inputs, all computed from the data ---
        seen = g["ball_observed_masked"].fillna(True).to_numpy().astype(bool)
        # how long the model has been without the ball, in seconds
        gap = np.zeros(len(g))
        run = 0.0
        for i, s in enumerate(seen):
            run = 0.0 if s else run + 1.0 / FPS
            gap[i] = run
        g["ball_gap_s"] = gap

        # distance from the ball to the nearest player of either side
        near = np.fmin(g["nearest_home_m"].to_numpy(), g["nearest_away_m"].to_numpy())
        g["ball_loose_m"] = np.where(seen, near, np.nan)
        g["margin_abs_m"] = np.abs(g["nearest_margin_m"].to_numpy())

        # seconds since the last restart, and until the next one
        dead = truth == "D"
        since = np.full(len(g), 1e6)
        t = 1e6
        for i in range(len(g)):
            t = 0.0 if (i > 0 and dead[i - 1] and not dead[i]) else t + 1.0 / FPS
            since[i] = t
        g["since_restart_s"] = since

        # seconds to the nearest true change of possession
        live = np.where(dead, "D", truth)
        ch = np.flatnonzero((live[1:] != live[:-1]) & (live[1:] != "D") & (live[:-1] != "D")) + 1
        if len(ch):
            idx = np.arange(len(g))
            nearest = np.min(np.abs(idx[:, None] - ch[None, :]), axis=1) / FPS
        else:
            nearest = np.full(len(g), 1e6)
        g["to_turnover_s"] = nearest

        # does the event feed back the reference here?
        side = g["holder_side"].fillna("").to_numpy()
        usable = (truth != "D") & (side != "")
        contested = np.zeros(len(g), dtype=bool)
        contested[usable] = (truth[usable] == "H") != (side[usable] == "home")
        g["ref_contested"] = contested
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def flags(d):
    """The reusable flags. Thresholds are round numbers, not fitted to the error rate."""
    return {
        "ball_unseen_2s": d["ball_gap_s"] >= 2.0,
        "ball_unseen_any": d["ball_gap_s"] > 0.0,
        "loose_ball_5m": d["ball_loose_m"] >= 5.0,
        "contested_margin_1m": d["margin_abs_m"] <= 1.0,
        "partial_tracking": d["n_tracked"] < 22,
        "within_3s_of_restart": d["since_restart_s"] <= 3.0,
        "within_1s_of_turnover": d["to_turnover_s"] <= 1.0,
        "reference_contested": d["ref_contested"],
    }


def table(d, target):
    rows = []
    err = d[target].to_numpy()
    base = float(err.mean())
    for name, mask in flags(d).items():
        m = mask.fillna(False).to_numpy() if hasattr(mask, "fillna") else np.asarray(mask)
        if m.sum() < 500 or (~m).sum() < 500:
            continue
        ein, eout = float(err[m].mean()), float(err[~m].mean())
        rows.append({"flag": name, "coverage": float(m.mean()), "err_in": ein, "err_out": eout,
                     "lift": ein / eout if eout else np.nan,
                     "share_of_errors": float(err[m].sum() / max(err.sum(), 1))})
    out = pd.DataFrame(rows).sort_values("lift", ascending=False)
    out.attrs["base"] = base
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc", default="D:/Football Hackathon/external_data/processed/idsse")
    ap.add_argument("--s2feat", default="D:/Football Hackathon/stage2/features_h")
    ap.add_argument("--refined", default="D:/Football Hackathon/stage3/predictions_refined")
    ap.add_argument("--s2pred", default="D:/Football Hackathon/stage2/predictions_h")
    ap.add_argument("--s3pred", default="D:/Football Hackathon/stage3/predictions_team")
    ap.add_argument("--labels", default="D:/Football Hackathon/stage3/player_labels")
    ap.add_argument("--out", default="D:/Football Hackathon/review/data/patterns.json")
    args = ap.parse_args()
    paths = vars(args)

    parts = [build(m, paths) for m in DEV]
    d = pd.concat(parts, ignore_index=True)
    print("graded frames across {} development matches: {:,}".format(len(DEV), len(d)))
    print()

    result = {"matches": [m.replace("DFL-MAT-", "") for m in DEV],
              "n_frames": int(len(d)), "targets": {}}

    for target, title in (("wrong_team", "TEAM ATTRIBUTION ERRORS (both sides say the ball is live)"),
                          ("wrong_dead", "LIVE/DEAD ERRORS")):
        t = table(d, target)
        print("=== {} ===".format(title))
        print("base error rate: {:.3f}".format(t.attrs["base"]))
        print(t.to_string(index=False, float_format=lambda v: "{:.3f}".format(v)))
        print()
        result["targets"][target] = {"base": t.attrs["base"],
                                     "rows": t.to_dict(orient="records")}

    # --- stability: does each flag behave the same way in every match?
    print("=== PER-MATCH LIFT, team errors (a pattern that only holds in one match is noise) ===")
    per = {}
    for name in flags(d):
        lifts = []
        for m, g in d.groupby("match"):
            fl = flags(g)[name]
            mk = fl.fillna(False).to_numpy() if hasattr(fl, "fillna") else np.asarray(fl)
            if mk.sum() < 200 or (~mk).sum() < 200:
                lifts.append(np.nan)
                continue
            e = g["wrong_team"].to_numpy()
            ein, eout = e[mk].mean(), e[~mk].mean()
            lifts.append(ein / eout if eout else np.nan)
        per[name] = lifts
    pm = pd.DataFrame(per, index=sorted(d["match"].unique())).T
    pm["min"] = pm.min(axis=1)
    pm["max"] = pm.max(axis=1)
    pm["consistent"] = (pm[sorted(d["match"].unique())] > 1.0).all(axis=1) | \
                       (pm[sorted(d["match"].unique())] < 1.0).all(axis=1)
    print(pm.to_string(float_format=lambda v: "{:.2f}".format(v)))
    result["per_match_lift"] = json.loads(pm.reset_index().to_json(orient="records"))

    # --- how much of the team error sits on a contested reference
    live = d[(d["label"] != "D") & (d["pred"] != "D")]
    con = live["ref_contested"].to_numpy()
    wrong = live["wrong_team"].to_numpy()
    result["contested"] = {
        "share_of_live_frames": float(con.mean()),
        "err_on_contested": float(wrong[con].mean()),
        "err_on_corroborated": float(wrong[~con].mean()),
        "share_of_errors_on_contested": float(wrong[con].sum() / max(wrong.sum(), 1)),
    }
    print()
    print("=== THE REFERENCE ITSELF ===")
    print("  live frames where the DFL's two products disagree : {:.1%}".format(con.mean()))
    print("  our team error rate there                         : {:.3f}".format(wrong[con].mean()))
    print("  our team error rate where they agree              : {:.3f}".format(wrong[~con].mean()))
    print("  share of all our team errors sitting there        : {:.1%}".format(
        wrong[con].sum() / max(wrong.sum(), 1)))

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print("\nwrote " + args.out)


if __name__ == "__main__":
    main()
