"""Run the frozen model over the competition match and write the submission file.

    python -I submit/predict_match.py --team-name MyTeam

THE WHOLE CHAIN, IN ORDER
-------------------------
    submit/processed/BAR-ATH/{frames,players}.parquet      from submit/atd_to_canonical.py
      |
      |-- stage2 features (player geometry only, no ball)  stage2/features.py
      |     -> stage2_dead.txt  -> P(play is stopped) per frame
      |     -> HSMM Viterbi with the fitted duration prior -> dead / alive intervals
      |
      |-- stage3 features (geometry + the ball when visible) + pitch control
      |     -> stage3_team.txt  -> P(home is in control) per frame
      |     -> HSMM Viterbi inside each alive stretch     -> H / A
      |
      |-- stage3/restart_logic.py refine_period()          the hardening layer:
      |     restart classification from where the ball rests, the whistle re-read off the
      |     players slowing to a walk, a still ball counted as a stoppage, possession held with
      |     whoever struck a ball still in the air, eleven-a-side to find the restart taker,
      |     short gaps in the ball track bridged, slivers of play between stoppages removed
      |
      -> frame,label for all 199,319 rows: the model's answer on the 158,857 graded frames,
         `X` on the 40,462 that are never scored.

NOTHING IS TRAINED HERE, AND NOTHING IS TUNED HERE. Every model, prior and threshold is loaded
from submit/models/ or read from stage3/restart_logic.py as shipped. No ground truth is opened:
not the private labels, which we do not have, and not the public five minutes, which we do.

DECODING IS PER HALF, NOT PER MATCH
-----------------------------------
The two graded blocks are 20,156 frames -- thirteen and a half minutes -- apart, and those frames
are not in the data at all, so rows 79,466 and 79,467 of this match sit next to each other while
being a quarter of an hour apart in the world. A Viterbi path run straight through that junction
is a path through a gap it cannot see. Both decoders are therefore run on one half at a time.
On the development matches the two halves are adjacent in the file, so this reproduces what was
measured there; here it is the only defensible thing to do.
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
from train import decode_hsmm  # noqa: E402  stage2
from train_team import decode_team  # noqa: E402  stage3
import restart_logic as RL  # noqa: E402  stage3

MATCH_ID = "BAR-ATH"
N_VIDEO_FRAMES = 199319
ABSTAIN = "X"

# The shipped hardening configuration: stage3/restart_logic.py's own argparse defaults, repeated
# here so a submission run cannot change because a default was edited. `reach`, `restart` and
# `gap` are absent on purpose -- each was measured and each made the score worse.
HARDENING_STEPS = ("align", "corner", "flight", "whistle", "still", "lasttouch", "inplay",
                   "outside", "headcount", "bridge", "takeover", "outdead", "island")
HARDENING_MODE = "carry"

# A rule that reads the ball may only run if its premise holds on THIS ball track. The test is
# the three stretches where no football is being played -- before kickoff, halftime, after the
# final whistle -- whose boundaries DATA.md publishes and which carry no possession answer at
# all. A rule that claims play is going on there has been falsified on this input, without a
# single label being read. Measured on the training provider, whose ball track is corrected, the
# same rule fires on 0.00% of reference-labelled dead frames; so anything above a few per cent
# is a different kind of failure, and this threshold is an order of magnitude clear of both.
BALL_RULE_MAX_FP = 0.25


def load_models(model_dir):
    s2 = lgb.Booster(model_file=os.path.join(model_dir, "stage2_dead.txt"))
    s3 = lgb.Booster(model_file=os.path.join(model_dir, "stage3_team.txt"))
    d2 = np.load(os.path.join(model_dir, "stage2_durations.npz"))
    d3 = np.load(os.path.join(model_dir, "stage3_durations.npz"))
    with open(os.path.join(model_dir, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    return s2, s3, d2, d3, manifest


def predict_dead(s2, d2, feats, cols):
    """P(stopped) per frame, then the interval path. Per period."""
    prob = s2.predict(feats[cols])
    dead = np.zeros(len(feats), dtype=bool)
    for period in sorted(feats["period"].unique()):
        m = (feats["period"] == period).to_numpy()
        dead[m] = decode_hsmm(prob[m], d2["log_alive"], d2["log_dead"],
                              float(d2["prior_dead"]))
    return prob, dead


def predict_team(s3, d3, feats, cols, dead):
    """P(home) per frame, then H/A inside each alive stretch. Per period."""
    p_home = s3.predict(feats[cols])
    home = np.zeros(len(feats), dtype=bool)
    for period in sorted(feats["period"].unique()):
        m = (feats["period"] == period).to_numpy()
        home[m] = decode_team(p_home[m], ~dead[m], d3["log_away"], d3["log_home"],
                              float(d3["prior_home"]))
    return p_home, home


def check_ball_rules(ungraded, steps):
    """Drop any ball-reading rule whose premise this ball track falsifies. Returns (steps, log).

    Only `inplay` makes a claim that an ungraded stretch can contradict -- it says play is going
    on. The others say play is stopped or that the ball belongs to nobody, which is true across
    halftime and therefore untestable there; they are left alone and reported instead.
    """
    log, keep = [], list(steps)
    for block, g in ungraded.groupby("block", sort=True):
        if block == "halftime" and len(g) < 1000:
            continue
        bx = np.where(g["ball_observed"], g["ball_x"] * RL.PITCH_X, np.nan).astype(float)
        by = np.where(g["ball_observed"], g["ball_y"] * RL.PITCH_Y, np.nan).astype(float)
        rel = g["ball_observed"].to_numpy().astype(bool)
        if rel.sum() < 500:
            log.append("  {:<16} ball seen on only {:.1f}% of frames, not testable".format(
                block, 100 * rel.mean()))
            continue
        fp = float(RL.ball_in_play(bx, by, rel).mean())
        log.append("  {:<16} {:>7,} frames, ball on {:5.1f}%   inplay claims play on {:5.1f}%"
                   .format(block, len(g), 100 * rel.mean(), 100 * fp))
        if block == "halftime" and fp > BALL_RULE_MAX_FP and "inplay" in keep:
            keep.remove("inplay")
            log.append("  -> `inplay` FALSIFIED on this ball track and dropped: it claims play "
                       "is going on across {:.0f}% of a stretch where none is. The ATD's ball "
                       "is uncorrected computer vision (DATA.md) and its detection moves "
                       "between objects, so 'the ball is moving' is not evidence about play "
                       "here. On the corrected training tracks the same rule fires on 0.00% of "
                       "reference-dead frames and is kept.".format(100 * fp))
    return tuple(keep), log


def harden(frames, players, prob, p_home, dead, home, steps):
    """The restart layer, one period at a time, exactly as stage3/restart_logic.py runs it."""
    d = frames[["period", "frame_native", "time_s", "ball_x", "ball_y",
                "ball_observed"]].copy()
    d["prob"] = prob
    d["p_home"] = p_home
    d["pred_dead"] = dead
    d["home_base"] = home
    # On the development matches the ball was hidden artificially and the hardening layer was
    # only ever allowed to read the ball it could still see. Here nothing is hidden: the ball is
    # missing where the tracker lost it and present where it did not, so the two flags coincide.
    d["ball_observed_masked"] = True

    parts = []
    for period, g in d.groupby("period", sort=True):
        g = g.reset_index(drop=True)
        wp = RL.wide_positions(players, period)
        if wp is None:
            raise SystemExit("no player positions for period {}".format(period))
        pf, X, Y, is_home = wp
        idx = np.clip(np.searchsorted(pf, g["frame_native"].to_numpy()), 0, len(pf) - 1)
        ok = pf[idx] == g["frame_native"].to_numpy()
        Xg = np.where(ok[:, None], X[idx], np.nan)
        Yg = np.where(ok[:, None], Y[idx], np.nan)
        out = RL.refine_period(g, Xg, Yg, is_home, mode=HARDENING_MODE, align=True,
                               steps=steps, late_gate=True)
        res = g[["period", "frame_native", "p_home", "prob"]].copy()
        res["label_base"] = np.where(g["pred_dead"], "D", np.where(g["home_base"], "H", "A"))
        for k, v in out.items():
            res[k] = v
        parts.append(res)
    return pd.concat(parts, ignore_index=True)


def write_submission(path, labels):
    """frame,label for every video frame. The format is four lines of code; the rows are the work."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team-name", required=True,
                    help="the submission is named after the team: <team_name>.csv")
    ap.add_argument("--proc-dir", default=os.path.join(HERE, "processed"))
    ap.add_argument("--work-dir", default=os.path.join(HERE, "work"))
    ap.add_argument("--model-dir", default=os.path.join(HERE, "models"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    ap.add_argument("--report", default=os.path.join(HERE, "reports", "predict_report.txt"))
    args = ap.parse_args()
    started = time.time()

    s2, s3, d2, d3, manifest = load_models(args.model_dir)
    s2_cols = manifest["stage2"]["features"]
    s3_cols = manifest["stage3"]["features"]

    frames = pd.read_parquet(os.path.join(args.proc_dir, MATCH_ID, "frames.parquet"),
                             columns=["period", "frame_native", "time_s", "ball_x", "ball_y",
                                      "ball_observed"])
    players = pd.read_parquet(os.path.join(args.proc_dir, MATCH_ID, "players.parquet"),
                              columns=["period", "frame_native", "team", "slot", "x", "y"])

    f2 = pd.read_parquet(os.path.join(args.work_dir, "stage2_features", MATCH_ID + ".parquet"))
    f3 = pd.read_parquet(os.path.join(args.work_dir, "stage3_features", MATCH_ID + ".parquet"))
    pc = pd.read_parquet(os.path.join(args.work_dir, "pitch_control", MATCH_ID + ".parquet"))
    f3 = f3.merge(pc, on=["period", "frame_native"], how="left")

    # Every table must describe the same frames in the same order, or the model is reading one
    # frame's geometry against another frame's ball. Checked rather than assumed.
    key = ["period", "frame_native"]
    for name, t in (("stage2 features", f2), ("stage3 features", f3)):
        if len(t) != len(frames) or not t[key].reset_index(drop=True).equals(
                frames[key].reset_index(drop=True)):
            raise SystemExit("{} is not frame-aligned to the canonical frames".format(name))
    missing2 = [c for c in s2_cols if c not in f2.columns]
    missing3 = [c for c in s3_cols if c not in f3.columns]
    if missing2 or missing3:
        raise SystemExit("features missing: stage2 {} stage3 {}".format(missing2, missing3))

    print("predicting {} frames of {}".format(len(frames), MATCH_ID))
    prob, dead = predict_dead(s2, d2, f2, s2_cols)
    print("  stage 2   play stopped on {:.1f}% of graded frames".format(100 * dead.mean()))
    p_home, home = predict_team(s3, d3, f3, s3_cols, dead)
    print("  stage 3   home in control on {:.1f}% of live frames".format(
        100 * home[~dead].mean()))

    ungraded = pd.read_parquet(os.path.join(args.proc_dir, MATCH_ID, "ball_ungraded.parquet"))
    steps, rule_log = check_ball_rules(ungraded, HARDENING_STEPS)
    print("  ball rules on stretches with no play (no label read):")
    print("\n".join(rule_log))

    ref = harden(frames, players, prob, p_home, dead, home, steps)
    print("  hardening {} of {} labels changed".format(
        int((ref["label_base"] != ref["label_refined"]).sum()), len(ref)))

    # The refined frame carries every field behind every decision, in the same shape
    # stage3/predictions_refined/ uses for the development matches. Kept so the hardening layer
    # can be audited on this match the same way it was audited on those.
    os.makedirs(args.work_dir, exist_ok=True)
    ref.to_parquet(os.path.join(args.work_dir, "refined_" + MATCH_ID + ".parquet"), index=False)

    labels = np.full(N_VIDEO_FRAMES, ABSTAIN, dtype="<U1")
    labels[ref["frame_native"].to_numpy()] = ref["label_refined"].to_numpy()

    os.makedirs(args.out_dir, exist_ok=True)
    path = write_submission(os.path.join(args.out_dir, args.team_name + ".csv"), labels)

    graded = labels[ref["frame_native"].to_numpy()]
    spells = int((pd.Series(graded) != pd.Series(graded).shift()).sum())
    lines = [
        "frozen model -> submission",
        "",
        "  model          {}  (stage2 {} features, stage3 {} features)".format(
            os.path.basename(args.model_dir), len(s2_cols), len(s3_cols)),
        "  trained on     {}".format(", ".join(
            m.replace("DFL-MAT-", "") for m in manifest["training_matches"])),
        "  rows written   {:,}   required 199,319   {}".format(
            N_VIDEO_FRAMES, "PASS" if len(labels) == N_VIDEO_FRAMES else "MISMATCH"),
        "  graded frames  {:,} predicted, {:,} abstained as X".format(
            len(ref), N_VIDEO_FRAMES - len(ref)),
        "",
        "  label mix over the graded frames",
    ]
    for lab in ("H", "A", "D", "X"):
        n = int((graded == lab).sum())
        lines.append("    {}   {:>7,}   {:5.1f}%".format(lab, n, 100 * n / len(graded)))
    lines += [
        "",
        "  ball rules tested on the stretches with no play, no label read:",
    ] + rule_log + [
        "  hardening steps applied: {}".format(",".join(steps)),
        "",
        "  stage 2 dead   {:.1f}% of graded frames".format(100 * dead.mean()),
        "  stage 3 home   {:.1f}% of live frames".format(100 * home[~dead].mean()),
        "  hardening      {:,} labels changed".format(
            int((ref["label_base"] != ref["label_refined"]).sum())),
        "  runs of one label (possessions and stoppages): {:,}".format(spells),
        "",
        "  submission  -> {}".format(path),
        "",
        "  Validate it before uploading:",
        "    python evaluation/validate_submission.py \"{}\"".format(path),
    ]
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n{:.0f}s   report -> {}".format(time.time() - started, args.report))


if __name__ == "__main__":
    main()
