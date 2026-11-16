"""Build the case bundles behind the local Model Failure Review page.

Every case is one place where the shipped model disagrees with the reference label, exported as a
short replay window with the per-frame data needed to explain it.

TWO BUCKETS, KEPT SEPARATE, because they deserve different treatment:

  confirmed  the reference label is corroborated. The DFL ships two independent products -- a
             per-frame BallPossession field and an event feed naming the player on every touch --
             and across this window they agree. The model is wrong.
  contested  the two DFL products disagree across the window, so the reference itself is not
             settled. Calling these model errors would be measuring our agreement with one of two
             disagreeing scorers. They are surfaced for inspection, not counted as failures.

Sources, all open-access IDSSE (Bassek et al., Nature Scientific Data 2025) and all local:
  external_data/processed/idsse/<match>/{frames,players}.parquet
  stage2/{features_h,predictions_h}/<match>.parquet
  stage3/{features,pitch_control,player_labels,predictions_team}/<match>.parquet
  stage3/predictions_refined/<match>.parquet   the restart-hardened output, and why it decided

The shipped model is the hardened one (stage3/restart_logic.py). Each bundle also carries what
the model said BEFORE hardening, so a reviewer can see what the change did on that case rather
than taking the aggregate on trust.

No competition asset is read, and nothing here is used to train, tune or select a model.
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

# Replay window: context either side of the case, and a hard cap so one long stoppage cannot
# produce a ten-megabyte file.
CONTEXT_S = 8.0
MAX_WINDOW_S = 44.0
STRIDE = 3                      # 8.3 replay steps a second, which is smooth enough to read

# Every case is counted in the index, because the pattern analysis needs the whole population.
# Only a capped sample gets a replay bundle: the common types run to hundreds per match and
# exporting all of them produced 351 MB of JSON for no extra insight. Rare, expensive types are
# never capped -- there are only a few dozen of them and each one matters.
BUNDLE_CAP = 60
NEVER_CAP = {"SPELL_MERGE", "MISSED_STOP", "INVENTED_STOP"}

# A review tool that only shows failures teaches half the lesson: it never shows what the signal
# looks like when the model reads it correctly. These are the same three decisions the failure
# types cover -- a stoppage, a turnover, a long spell -- scored as hits instead of misses.
SUCCESS_KINDS = {"STOPPAGE_CAUGHT", "TURNOVER_CAUGHT", "SPELL_HELD"}
SUCCESS_CAP = 24

# Features shown in the synchronised table. Chosen because they are the ones that actually carry
# the decision -- the top of the gain table for each stage -- not because they look tidy.
S2_FEATURES = ["spd_mean", "spd_p90", "frac_still", "n_outside_pitch", "line_tight_min",
               "n_tracked", "anchor_score", "centroid_dist"]
S3_FEATURES = ["ball_margin", "ball_nearer_home", "d_ball_home", "d_ball_away", "ball_third"]
PC_FEATURES = ["pc_home_share", "pc_at_ball", "pc_decisive"]

# A contested window is one where the two DFL products disagree on more than this share of the
# alive frames in it. 0.25 is three times the match-wide disagreement rate of about 9%, so it
# marks windows that are unusually unsettled rather than ordinary ones.
CONTESTED_SHARE = 0.25

# The HSMM decodes on every 5th frame (stage2/train.py decode_hsmm, step=5), so the model's
# evidence is summed in units of 0.2 s. Inserting a dead segment into a live one costs the duration
# prior a near-constant amount: measured at a median of 9 nats across every duration band, 8.6 to
# 10.1 (stage2/DECODER_FIX.md). That is the bar the per-frame evidence has to clear.
DECODE_STEP = 5
DURATION_COST_NATS = 9.0


def clock(period, time_s_rel):
    base = 0 if period == 1 else 45 * 60
    total = base + time_s_rel
    return "{:d}:{:02d}".format(int(total // 60), int(total % 60))


# The per-frame reasoning behind the hardened output, carried into every bundle.
REFINED_COLS = ["pred_dead_refined", "home_refined", "pred_dead_base", "home_base",
                "last_control", "restart_team", "restart_kind", "first_ball_frame",
                "resume_frame", "knn_nearest_home_m", "knn_nearest_away_m", "knn_margin",
                "ball_move_m", "ball_reliable", "switch_reason", "uncertain", "in_override",
                "ball_speed_ms", "in_flight", "flight_team",
                "team_pace_ms", "team_pace_base", "whistle_shift_s", "ball_at_rest"]


def load_match(match, paths):
    """One frame-aligned table per match, plus the wide player positions per period."""
    f = pd.read_parquet(os.path.join(paths["proc"], match, "frames.parquet"),
                        columns=["period", "frame_native", "time_s", "ball_x", "ball_y",
                                 "ball_observed", "label", "label_valid"])
    s2p = pd.read_parquet(os.path.join(paths["s2pred"], match + ".parquet"),
                          columns=["period", "frame_native", "prob", "pred_hsmm"])
    s3p = pd.read_parquet(os.path.join(paths["s3pred"], match + ".parquet"))
    lab = pd.read_parquet(os.path.join(paths["labels"], match + ".parquet"),
                          columns=["period", "frame_native", "holder_slot", "holder_side"])

    s2f = pd.read_parquet(os.path.join(paths["s2feat"], match + ".parquet"),
                          columns=["period", "frame_native"] + S2_FEATURES)
    s3f = pd.read_parquet(os.path.join(paths["s3feat"], match + ".parquet"),
                          columns=["period", "frame_native"] + S3_FEATURES)
    pc = pd.read_parquet(os.path.join(paths["pc"], match + ".parquet"),
                         columns=["period", "frame_native"] + PC_FEATURES)

    d = f
    for other in (s2p, s3p[["period", "frame_native", "p_home", "home_end_to_end",
                            "pred_end_to_end", "ball_observed_masked"]], lab, s2f, s3f, pc):
        d = d.merge(other, on=["period", "frame_native"], how="left")

    # The hardened output replaces the raw decode as the model under review. Where it is missing
    # for a frame, the raw decode stands, so a missing file degrades to the old behaviour rather
    # than to nonsense.
    rp = os.path.join(paths["refined"], match + ".parquet")
    if os.path.exists(rp):
        ref = pd.read_parquet(rp, columns=["period", "frame_native"] + REFINED_COLS)
        d = d.merge(ref, on=["period", "frame_native"], how="left")
        d["pred_hsmm"] = d["pred_dead_refined"].fillna(d["pred_hsmm"])
        d["home_end_to_end"] = d["home_refined"].fillna(d["home_end_to_end"])

    d = d[d["label_valid"].astype(bool)].reset_index(drop=True)
    d["pred_hsmm"] = d["pred_hsmm"].fillna(False).astype(bool)

    players = pd.read_parquet(os.path.join(paths["proc"], match, "players.parquet"),
                              columns=["period", "frame_native", "team", "slot", "x", "y"])
    return d, players


def wide_positions(players, period):
    """(frame index, X, Y, is_home) with one column per tracked slot, in a stable order."""
    g = players[players["period"] == period]
    if g.empty:
        return None
    g = g.assign(col=g["team"].astype(str) + "_" + g["slot"].astype(str))
    x = g.pivot_table(index="frame_native", columns="col", values="x", aggfunc="first")
    y = g.pivot_table(index="frame_native", columns="col", values="y", aggfunc="first")
    x = x.sort_index()
    y = y.sort_index().reindex(columns=x.columns)
    is_home = np.array([c.startswith("home_") for c in x.columns])
    return x.index.to_numpy(), x.to_numpy(), y.to_numpy(), is_home, list(x.columns)


def find_cases(g):
    """Every disagreement in one period, typed and with a frame span.

    Stage 2 types concern whether the ball is live; stage 3 types concern which team has it. They
    are separated because they are produced by different parts of the model and a reviewer needs
    to know which one to look at.
    """
    out = []
    truth = g["label"].to_numpy()
    true_dead = truth == "D"
    pred_dead = g["pred_hsmm"].to_numpy()
    n = len(g)

    # --- stage 2: stoppages missed, invented, or badly timed
    pred_spans = runs_of(pred_dead)
    for a, b in runs_of(true_dead):
        cov = sum(max(0, min(b, d) - max(a, c)) for c, d in pred_spans) / float(b - a)
        before = truth[a - 1] if a > 0 else "?"
        after = truth[b] if b < n else "?"
        if cov < 0.5:
            kind = "SPELL_MERGE" if before == after and before in "HA" else "MISSED_STOP"
            out.append((kind, a, b, {"covered": round(cov, 3),
                                     "team_before": str(before), "team_after": str(after)}))
        else:
            # found, but check the edges
            best = max(((min(b, d) - max(a, c), c, d) for c, d in pred_spans
                        if min(b, d) > max(a, c)), default=None)
            if best:
                se, ee = (best[1] - a) / FPS, (best[2] - b) / FPS
                if abs(se) > 1.0 or abs(ee) > 1.0:
                    out.append(("BOUNDARY", a, b,
                                {"start_err_s": round(se, 2), "end_err_s": round(ee, 2),
                                 "covered": round(cov, 3)}))
    for c, d in pred_spans:
        ov = sum(max(0, min(b, d) - max(a, c)) for a, b in runs_of(true_dead))
        if ov / float(d - c) < 0.2:
            out.append(("INVENTED_STOP", c, d, {"overlap": round(ov / float(d - c), 3)}))

    # --- stage 3: the ball is live in truth and in prediction, but on the wrong side
    alive = (~true_dead) & (~pred_dead)
    home_pred = g["home_end_to_end"].fillna(False).to_numpy().astype(bool)
    wrong = alive & (home_pred != (truth == "H"))
    for a, b in runs_of(wrong):
        if (b - a) < int(0.6 * FPS):
            continue                      # shorter than 0.6 s is a blink, not a reviewable case
        out.append(("WRONG_TEAM", a, b,
                    {"truth_team": str(truth[a]),
                     "said": "H" if home_pred[a] else "A",
                     "mean_p_home": round(float(np.nanmean(g["p_home"].to_numpy()[a:b])), 3)}))

    # --- stage 3: a possession change invented where the truth has none
    live_truth = np.where(true_dead, "D", truth)
    live_pred = np.where(pred_dead, "D", np.where(home_pred, "H", "A"))
    t_ch = np.flatnonzero((live_truth[1:] != live_truth[:-1]) &
                          (live_truth[1:] != "D") & (live_truth[:-1] != "D")) + 1
    p_ch = np.flatnonzero((live_pred[1:] != live_pred[:-1]) &
                          (live_pred[1:] != "D") & (live_pred[:-1] != "D")) + 1
    tol = int(1.0 * FPS)
    for i in p_ch:
        if not len(t_ch) or np.min(np.abs(t_ch - i)) > tol:
            out.append(("FALSE_CHANGE", max(0, i - 2), min(n, i + 2),
                        {"at": "turnover the reference does not have"}))
    for i in t_ch:
        if not len(p_ch) or np.min(np.abs(p_ch - i)) > tol:
            out.append(("MISSED_CHANGE", max(0, i - 2), min(n, i + 2),
                        {"at": "turnover the model does not make"}))
    return out


def base_cases(g):
    """The same case detection, run against the model as it was BEFORE the hardening pass.

    Needed to answer one question the aggregate cannot: which individual cases did the change
    create? A net count of minus thirty can hide thirty broken ones.
    """
    if "pred_dead_base" not in g.columns:
        return []
    sub = g.copy()
    sub["pred_hsmm"] = g["pred_dead_base"].fillna(g["pred_hsmm"]).astype(bool)
    sub["home_end_to_end"] = g["home_base"].fillna(g["home_end_to_end"])
    return find_cases(sub)


def mark_new(cases, before):
    """Which of `cases` has no counterpart of the same kind overlapping it in `before`."""
    by_kind = {}
    for kind, a, b, _x in before:
        by_kind.setdefault(kind, []).append((a, b))
    out = []
    for kind, a, b, extra in cases:
        prev = by_kind.get(kind, [])
        out.append(not any(min(b, d) > max(a, c) for c, d in prev))
    return out


def regression_reason(g, a, b, kind):
    """Why this case exists now and did not before, measured rather than asserted.

    Everything here is a difference between two runs over the same frames, so it says what the
    hardening pass actually did to this window -- moved an edge, held a restart team, or drove
    the team call itself -- and what the old answer had been.
    """
    n = len(g)
    lo, hi = max(0, a - int(4 * FPS)), min(n, b + int(4 * FPS))
    dead_b = g["pred_dead_base"].fillna(False).to_numpy().astype(bool)
    dead_r = g["pred_hsmm"].to_numpy().astype(bool)
    home_b = g["home_base"].fillna(False).to_numpy().astype(bool)
    home_r = g["home_end_to_end"].fillna(False).to_numpy().astype(bool)
    truth = g["label"].to_numpy()
    over = (g["in_override"].fillna(False).to_numpy().astype(bool)
            if "in_override" in g.columns else np.zeros(n, bool))
    flight = (g["in_flight"].fillna(False).to_numpy().astype(bool)
              if "in_flight" in g.columns else np.zeros(n, bool))
    bspeed = (g["ball_speed_ms"].to_numpy(dtype=float)
              if "ball_speed_ms" in g.columns else np.full(n, np.nan))
    tpace = (g["team_pace_ms"].to_numpy(dtype=float)
             if "team_pace_ms" in g.columns else np.full(n, np.nan))
    tbase = (float(g["team_pace_base"].to_numpy(dtype=float)[0])
             if "team_pace_base" in g.columns and n else float("nan"))

    bits = []
    cause = "unknown"

    # --- did the live/dead edge move, and by how much?
    def span_over(mask, a0, b0):
        best = None
        for c, d in runs_of(mask):
            ov = min(b0, d) - max(a0, c)
            if ov > 0 and (best is None or ov > best[0]):
                best = (ov, c, d)
        return best

    tb = span_over(truth == "D", lo, hi)
    sb, sr = span_over(dead_b, lo, hi), span_over(dead_r, lo, hi)
    edge = None
    if tb and sb and sr:
        se_b, ee_b = (sb[1] - tb[1]) / FPS, (sb[2] - tb[2]) / FPS
        se_r, ee_r = (sr[1] - tb[1]) / FPS, (sr[2] - tb[2]) / FPS
        wmoved = (sr[1] - sb[1]) / FPS
        if abs(wmoved) >= 0.04:
            cause = "whistle"
            def mean_pace(i0, i1):
                seg = tpace[max(0, i0):max(0, i1)]
                seg = seg[np.isfinite(seg)]
                return float(np.mean(seg)) if seg.size else float("nan")

            pre, post = mean_pace(sr[1] - FPS, sr[1]), mean_pace(sr[1], sr[1] + FPS)
            bits.append(
                "The whistle moved <b>{:.2f} s {}</b>, onto the steepest fall in how fast the "
                "players were moving: all 22 of them average <b>{:.2f} m/s</b> in the second "
                "before the new edge and <b>{:.2f} m/s</b> in the second after it, against "
                "<b>{:.2f} m/s</b> for the match as a whole. Against the reference the stoppage "
                "used to start <b>{:.2f} s {}</b>; now it starts <b>{:.2f} s {}</b>.".format(
                    abs(wmoved), "earlier" if wmoved < 0 else "later",
                    pre if pre == pre else 0.0, post if post == post else 0.0,
                    tbase if tbase == tbase else 0.0,
                    abs(se_b), "early" if se_b < 0 else "late",
                    abs(se_r), "early" if se_r < 0 else "late"))
        moved = (sr[2] - sb[2]) / FPS
        if abs(moved) >= 0.04:
            edge = (se_b, ee_b, se_r, ee_r, moved)
            cause = "edge" if cause == "unknown" else cause + "+edge"
            bits.append(
                "The restart edge moved <b>{:.2f} s {}</b>, onto the first frame the ball was "
                "reliably seen to move. Against the reference the stoppage used to end "
                "<b>{:.2f} s {}</b>; now it ends <b>{:.2f} s {}</b>.".format(
                    abs(moved), "earlier" if moved < 0 else "later",
                    abs(ee_b), "early" if ee_b < 0 else "late",
                    abs(ee_r), "early" if ee_r < 0 else "late"))

    # --- was the team call taken over by the touch rule?
    fshare = float(flight[a:b].mean()) if b > a else 0.0
    if fshare > 0.2:
        cause = "touch" if cause == "unknown" else cause + "+touch"
        with np.errstate(invalid="ignore"):
            top = np.nanmax(bspeed[a:b]) if np.isfinite(bspeed[a:b]).any() else float("nan")
        who = ""
        if "flight_team" in g.columns:
            vals = [v for v in g["flight_team"].to_numpy()[a:b] if v in ("H", "A")]
            who = ("home" if vals[0] == "H" else "away") if vals else ""
        bits.append(
            "The ball's own trajectory was read as a touch on <b>{:.0f}%</b> of these frames: it "
            "lost pace or changed course with {} player inside 1.5 m of it while travelling at "
            "up to <b>{:.0f} m/s</b>, so possession went to whoever played it instead of to the "
            "per-frame probability.".format(
                100 * fshare,
                {"home": "a home", "away": "an away"}.get(who, "a"),
                top if top == top else 0))

    # --- was the team call driven by the restart layer rather than the model?
    share = float((over[a:b] & ~flight[a:b]).mean()) if b > a else 0.0
    if share > 0.2:
        cause = "override" if cause == "unknown" else cause + "+override"
        kind_txt = ""
        if "restart_kind" in g.columns:
            k = str(g["restart_kind"].fillna("").to_numpy()[max(0, a - 1)] or "")
            if k:
                kind_txt = " after {}".format(k.replace("_", " "))
        bits.append(
            "The restart layer was driving the team call on <b>{:.0f}%</b> of these frames{} "
            "&mdash; the nearest-player vote and the held restart team, not the per-frame "
            "probability.".format(100 * share, kind_txt))

    # --- a change-of-hands case is four frames long, so the useful question is not what
    #     happened inside it but how far the nearest switch moved
    if kind in ("MISSED_CHANGE", "FALSE_CHANGE"):
        lab_b = np.where(dead_b, "D", np.where(home_b, "H", "A"))
        lab_r = np.where(dead_r, "D", np.where(home_r, "H", "A"))

        def switches(seq):
            return [i for i in range(1, n)
                    if seq[i] != seq[i - 1] and seq[i] != "D" and seq[i - 1] != "D"]

        def nearest(lst, i):
            return None if not lst else min(lst, key=lambda j: abs(j - i))

        mid = (a + b) // 2
        win = int(5.0 * FPS)          # a switch five seconds away is a different switch
        near_b, near_r = nearest(switches(lab_b), mid), nearest(switches(lab_r), mid)
        jb = near_b if near_b is not None and abs(near_b - mid) <= win else None
        jr = near_r if near_r is not None and abs(near_r - mid) <= win else None
        fmt = lambda j: ("none within 5 s" if j is None
                         else "{:.2f} s {}".format(abs(j - mid) / FPS,
                                                   "late" if j > mid else "early"))
        # if the switch vanished, say why: a possession under the 0.8 s floor is merged away
        # which possession disappeared: the one that ended at the old switch, or the one it
        # started. The floor can only be blamed when one of them was under it.
        spell_s = None
        if jb is not None and jr is None:
            cands = []
            for probe in (jb - 1, jb):
                if not (0 <= probe < n) or lab_b[probe] == "D":
                    continue
                for c, d2 in runs_of(lab_b == lab_b[probe]):
                    if c <= probe < d2:
                        cands.append((d2 - c) / FPS)
                        break
            spell_s = min(cands) if cands else None
        if kind == "MISSED_CHANGE":
            if jr is None and jb is not None:
                bits.append(
                    "The reference changes hands here and the model used to change with it, "
                    "<b>{}</b>. It no longer changes at all within five seconds{}.".format(
                        fmt(jb),
                        "" if spell_s is None or spell_s >= 0.8 else
                        ": the possession either side of that switch was only {:.1f} s long, "
                        "and anything under 0.8 s is now merged into its neighbour".format(
                            spell_s)))
                cause = "merged" if (spell_s is not None and spell_s < 0.8) else "redecided"
            else:
                bits.append(
                    "The reference changes hands here. The model's nearest switch used to be "
                    "<b>{}</b>, which the one-second tolerance accepted; it is now <b>{}</b>, "
                    "which it does not.".format(fmt(jb), fmt(jr)))
                if cause == "unknown":
                    cause = "timing"
        else:
            bits.append("The model changes hands here and the reference does not. Before the fix "
                        "its nearest switch to this moment was <b>{}</b>.".format(fmt(jb)))
            if cause == "unknown":
                cause = "timing"

    # --- what the two runs actually said here
    if (home_b[a:b] != home_r[a:b]).any():
        was = "home" if home_b[a:b].mean() >= 0.5 else "away"
        now = "home" if home_r[a:b].mean() >= 0.5 else "away"
        want = "home" if (truth[a:b] == "H").mean() >= 0.5 else "away"
        bits.append("It used to say <b>{}</b> here and now says <b>{}</b>; the reference says "
                    "<b>{}</b>.".format(was, now, want))
    elif (dead_b[a:b] != dead_r[a:b]).any():
        bits.append("The team call did not change &mdash; only where the stoppage begins or "
                    "ends did.")

    if not bits:
        bits.append("The window either side of this case changed, which moved the case rather "
                    "than creating a new error inside it.")

    lesson = {
        "edge": "Trusting the ball's first movement over the decoder's own timing is right on "
                "average and wrong here. The two should have to agree before the edge moves.",
        "override": "A nearest-player vote decided this instead of the model. It is a hard rule "
                    "applied to a soft situation, and the model's probability was the better "
                    "judge of it.",
        "edge+override": "Both halves of the change touched this window at once &mdash; the "
                         "edge moved and the rule then held a team across it.",
        "redecided": "The switch is gone, and not because of the length floor: the possession "
                     "either side of it was re-decided once the restart edge moved. This one is "
                     "worth opening and watching frame by frame.",
        "merged": "The floor on possession length removed a switch the reference does have. "
                  "0.8 s is a blunt instrument: a real touch can be shorter than that, and a "
                  "wobble can be longer. A length floor is the wrong shape of rule for this.",
        "touch": "The ball's trajectory was read as a touch and it was not one &mdash; a "
                 "deflection, a near miss, or the tracked ball breaking course for some other "
                 "reason. This is the price of the touch test, and it is a price worth paying: "
                 "the same rule fixes 325 cases and costs 176 like this one.",
        "whistle": "The whistle was re-read off the players slowing to a walk, and here "
                   "the two readings of it pulled the edge the wrong way. Player movement is a "
                   "strong signal and a slow one: the walk sets in a median 4.6 s after the "
                   "real whistle, so it is only ever averaged with the decoder's own timing, "
                   "never trusted on its own.",
        "whistle+edge": "Both edges of this stoppage moved at once &mdash; the whistle onto the "
                        "players slowing down, the restart onto the ball starting to move "
                        "&mdash; and the window between them is what the reference disagrees "
                        "with.",
        "timing": "Nothing new is wrong here: the same switch is being made, a little further "
                  "from where the reference puts it, and a fixed one-second tolerance turns "
                  "that into a case. Worth seeing, not worth chasing.",
    }.get(cause, "")
    if not lesson:
        parts = []
        if "whistle" in cause:
            parts.append("the whistle re-read off the players slowing down")
        if "edge" in cause:
            parts.append("the restart edge moved onto the ball")
        if "touch" in cause:
            parts.append("a touch read off the ball's trajectory")
        if "override" in cause:
            parts.append("a rule holding a team across the window")
        lesson = ("More than one part of the change reached this window at once: {}. Each is "
                  "measured to be right on average, and this is a window where they are not."
                  .format(", ".join(parts[:-1]) + " and " + parts[-1])
                  if len(parts) > 1 else
                  "This case moved rather than appeared; the error was already nearby.")

    return {"cause": cause, "why": " ".join(bits), "lesson": lesson,
            "edge": None if not edge else {
                "end_err_before_s": round(edge[1], 2), "end_err_after_s": round(edge[3], 2),
                "start_err_before_s": round(edge[0], 2), "start_err_after_s": round(edge[2], 2),
                "moved_s": round(edge[4], 2)},
            "override_share": round(share, 3), "flight_share": round(fshare, 3)}


def find_successes(g):
    """Every place the model agrees with the reference in a way worth looking at.

    Deliberately strict. A near miss is not a success story, so a stoppage has to be covered 80%
    with both edges inside half a second, a turnover has to land within half a second with the
    right team on both sides of it, and a spell has to be held for fifteen seconds at 97%
    agreement. What is left is the model reading the signal cleanly.
    """
    out = []
    truth = g["label"].to_numpy()
    true_dead = truth == "D"
    pred_dead = g["pred_hsmm"].to_numpy()
    home_pred = g["home_end_to_end"].fillna(False).to_numpy().astype(bool)
    pred = np.where(pred_dead, "D", np.where(home_pred, "H", "A"))
    live_truth = np.where(true_dead, "D", truth)
    n = len(g)

    # --- a stoppage found, and found at the right moment
    pred_spans = runs_of(pred_dead)
    for a, b in runs_of(true_dead):
        if (b - a) < int(2.0 * FPS):
            continue
        cov = sum(max(0, min(b, d) - max(a, c)) for c, d in pred_spans) / float(b - a)
        best = max(((min(b, d) - max(a, c), c, d) for c, d in pred_spans
                    if min(b, d) > max(a, c)), default=None)
        if cov < 0.8 or best is None:
            continue
        se, ee = (best[1] - a) / FPS, (best[2] - b) / FPS
        if abs(se) > 0.5 or abs(ee) > 0.5:
            continue
        out.append(("STOPPAGE_CAUGHT", a, b,
                    {"covered": round(cov, 3), "start_err_s": round(se, 2),
                     "end_err_s": round(ee, 2),
                     "team_before": str(truth[a - 1] if a > 0 else "?"),
                     "team_after": str(truth[b] if b < n else "?")}))

    # --- a turnover made on time, with the right team either side of it
    t_ch = np.flatnonzero((live_truth[1:] != live_truth[:-1]) &
                          (live_truth[1:] != "D") & (live_truth[:-1] != "D")) + 1
    p_ch = np.flatnonzero((pred[1:] != pred[:-1]) &
                          (pred[1:] != "D") & (pred[:-1] != "D")) + 1
    half, win = int(0.5 * FPS), int(1.0 * FPS)
    for i in t_ch:
        if not len(p_ch):
            continue
        j = int(p_ch[np.argmin(np.abs(p_ch - i))])
        if abs(j - i) > half:
            continue
        a0, b0 = max(0, i - win), min(n, i + win)
        if not (pred[a0:b0] == live_truth[a0:b0]).all():
            continue                      # on time, but wobbling either side of it
        out.append(("TURNOVER_CAUGHT", max(0, i - 2), min(n, i + 2),
                    {"timing_err_s": round((j - i) / FPS, 2),
                     "from": str(live_truth[i - 1]), "to": str(live_truth[i])}))

    # --- a long possession held without a wobble
    bounds = np.flatnonzero(live_truth[1:] != live_truth[:-1]) + 1
    for a, b in zip(np.concatenate(([0], bounds)), np.concatenate((bounds, [n]))):
        a, b = int(a), int(b)
        if live_truth[a] == "D" or (b - a) < int(15.0 * FPS):
            continue
        acc = float((pred[a:b] == live_truth[a:b]).mean())
        if acc < 0.97:
            continue
        out.append(("SPELL_HELD", a, b,
                    {"agreement": round(acc, 3), "team": str(live_truth[a]),
                     "length_s": round((b - a) / FPS, 1)}))
    return out


# Could the model see the ball here? The model reads the MASKED ball only, so that is what is
# counted -- not what the tracking file happens to contain. A case at or above BALL_VISIBLE is one
# a reviewer can actually judge: the evidence the model had is the evidence a person watching the
# replay has. Below it, the model was working from where play had been, and no amount of looking at
# the clip says what the right answer was.
BALL_WIDEN_S = 1.5
BALL_VISIBLE = 0.8


def ball_visible_share(g, a, b):
    """Share of the case window where the ball was both tracked and visible to the model.

    Widened to BALL_WIDEN_S for short cases, for the same reason contested_share is: a turnover
    case is four frames long by construction and four frames carry no visibility information.
    """
    need = int(BALL_WIDEN_S * FPS)
    if b - a < need:
        mid = (a + b) // 2
        a, b = max(0, mid - need // 2), min(len(g), mid + need // 2)
    sub = g.iloc[a:b]
    if not len(sub):
        return 0.0
    seen = sub["ball_observed"].fillna(False).to_numpy().astype(bool)
    if "ball_observed_masked" in sub:
        seen = seen & sub["ball_observed_masked"].fillna(True).to_numpy().astype(bool)
    return float(seen.mean())


# A case span shorter than this cannot be assessed for contest on its own, so the assessment is
# widened around its centre. A turnover case is four frames long by construction.
MIN_ASSESS_S = 1.5


def contested_share(g, a, b):
    """Share of alive frames around the case where the DFL's two products disagree.

    holder_side comes from the event feed; label comes from the per-frame BallPossession field.
    Frames with no synchronised event nearby carry an empty holder_side and are not counted -- an
    absent second opinion is not a disagreement.

    The span is widened to MIN_ASSESS_S when the case itself is shorter than that. Without this a
    four-frame turnover case always fell through to "confirmed" for want of frames to judge, which
    quietly labelled every change-type case as a settled model error.
    """
    need = int(MIN_ASSESS_S * FPS)
    if b - a < need:
        mid = (a + b) // 2
        a, b = max(0, mid - need // 2), min(len(g), mid + need // 2)
    sub = g.iloc[a:b]
    truth = sub["label"].to_numpy()
    side = sub["holder_side"].fillna("").to_numpy()
    usable = (truth != "D") & (side != "")
    if usable.sum() < 5:
        return None
    ref_home = truth[usable] == "H"
    ev_home = side[usable] == "home"
    return float((ref_home != ev_home).mean())


def max_subarray(v):
    """Largest contiguous sum, and where it is (Kadane). The decoder is free to pick any
    sub-window, so the best window matters more than the average over a span."""
    best = cur = 0.0
    bi = bj = ci = 0
    for k, x in enumerate(v):
        if cur + x < x:
            cur, ci = x, k
        else:
            cur += x
        if cur > best:
            best, bi, bj = cur, ci, k + 1
    return best, bi, bj


def stage2_rationale(g, a, b, prior_dead, model_said_dead):
    """The arithmetic the decoder actually did, in nats.

    Emission gain for calling a step dead instead of alive is logit(p) - logit(prior), exactly as
    decode_hsmm scores it. Sum it over the span, compare it with what the duration prior charges
    for inserting the segment, and the decision explains itself. This is not an interpretation of
    the model -- it is the model's own objective, read back.
    """
    p = np.clip(g["prob"].to_numpy()[a:b:DECODE_STEP].astype(float), 1e-6, 1 - 1e-6)
    if not len(p):
        return None
    prior = float(np.clip(prior_dead, 1e-6, 1 - 1e-6))
    gain = np.log(p / (1 - p)) - np.log(prior / (1 - prior))
    best, bi, bj = max_subarray(gain)
    step_s = DECODE_STEP / FPS
    total = float(gain.sum())
    paid = total - DURATION_COST_NATS
    if model_said_dead:
        why = ("Evidence for dead over this span: {:+.0f} nats, against a duration charge of about "
               "{:.0f}. The switch paid for itself {:.0f} times over, so Viterbi took it."
               ).format(total, DURATION_COST_NATS, max(1.0, total / DURATION_COST_NATS))
    elif best <= DURATION_COST_NATS:
        why = ("Evidence for dead over this span: {:+.0f} nats. The best sub-window in it is worth "
               "{:+.0f} nats over {:.1f} s, and inserting a stoppage costs about {:.0f}. Nothing "
               "here covers the charge, so the model kept play alive. On this objective it was not "
               "a mistake -- the evidence was never there."
               ).format(total, best, (bj - bi) * step_s, DURATION_COST_NATS)
    else:
        why = ("Evidence for dead over this span: {:+.0f} nats overall, but a {:.1f} s sub-window "
               "inside it is worth {:+.0f} -- more than the {:.0f} the duration prior charges. The "
               "evidence was there and the decoder still did not take it, because taking it would "
               "have had to pay for the two alive segments it splits as well."
               ).format(total, (bj - bi) * step_s, best, DURATION_COST_NATS)
    return {"stage": 2,
            "decision": "called the ball dead" if model_said_dead else "kept play alive",
            "emission_nats": round(total, 1),
            "per_step_nats": round(total / len(gain), 2),
            "steps": int(len(gain)),
            "best_window_nats": round(best, 1),
            "best_window_s": round((bj - bi) * step_s, 2),
            "duration_cost_nats": DURATION_COST_NATS,
            "margin_nats": round(paid, 1),
            "prior_dead": round(prior, 3),
            "why": why}


CUE_WIDEN_S = 1.5


def stage3_rationale(g, a, b, said, truth_team):
    """Which inputs pointed which way, averaged over the case.

    Not an attribution method and not claimed as one: these are the cues the stage-3 model is
    given, read off at face value, so a reviewer can see whether the model followed the obvious
    one or overruled it. Where they split, that split IS the explanation of the disagreement.
    """
    need = int(CUE_WIDEN_S * FPS)
    if b - a < need:
        mid = (a + b) // 2
        a, b = max(0, mid - need // 2), min(len(g), mid + need // 2)
    sub = g.iloc[a:b]

    def mean(col):
        if col not in sub:
            return None
        v = sub[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        return None if not len(v) else float(v.mean())

    cues = []

    def add(name, reading, value, points, strength):
        if points is None:
            return
        cues.append({"cue": name, "reading": reading, "value": None if value is None else round(value, 3),
                     "points_to": points, "strength": strength,
                     "with_model": points == said, "with_reference": points == truth_team})

    margin = mean("ball_margin")          # positive: home is closer to the ball
    if margin is not None:
        add("nearest player to the ball", "{:+.2f} m in favour of {}".format(
            abs(margin), "home" if margin > 0 else "away"),
            margin, "H" if margin > 0 else "A",
            "clear" if abs(margin) >= 2.0 else ("slim" if abs(margin) >= 0.5 else "a coin toss"))

    at_ball = mean("pc_at_ball")          # home's control of the ball's own location
    if at_ball is not None:
        add("pitch control where the ball is", "{:.2f} home share".format(at_ball),
            at_ball, "H" if at_ball >= 0.5 else "A",
            "clear" if abs(at_ball - 0.5) >= 0.2 else "slim")

    share = mean("pc_home_share")
    if share is not None:
        add("pitch control over the whole pitch", "{:.2f} home share".format(share),
            share, "H" if share >= 0.5 else "A",
            "clear" if abs(share - 0.5) >= 0.1 else "slim")

    ph = mean("p_home")
    side = sub["holder_side"].fillna("").to_numpy()
    side = side[side != ""]
    ev = None
    if len(side):
        ev = "H" if (side == "home").mean() >= 0.5 else "A"
        add("event feed (second reference)", "names {} player on the ball".format(
            "a home" if ev == "H" else "an away"), float((side == "home").mean()), ev,
            "independent")

    followed = [c["cue"] for c in cues if c["with_model"] and c["strength"] != "a coin toss"]
    against = [c["cue"] for c in cues if c["with_reference"] and not c["with_model"]
               and c["strength"] != "a coin toss"]
    if ph is None:
        why = "No stage-3 probability recorded over this span."
    else:
        head = "The model put P(home) at {:.3f}, so it called {}. ".format(
            ph, "home" if ph >= 0.5 else "away")
        if abs(ph - 0.5) < 0.1:
            head += "That is within 0.1 of the coin-toss line -- it was barely a decision. "
        if followed:
            head += "It followed " + ", ".join(followed) + ". "
        if against:
            head += "The reference is backed by " + ", ".join(against) + ". "
        if not against and said != truth_team:
            head += ("Every cue the model is given pointed the way it went, which is why it was "
                     "confident and wrong: the disagreement is not in these inputs. ")
        why = head.strip()
    return {"stage": 3, "decision": "called {}".format("home" if said == "H" else "away"),
            "p_home": None if ph is None else round(ph, 3),
            "reference": truth_team, "cues": cues, "why": why}


# Cricket equivalents, one per kind of mistake. Not decoration: each one makes the same point,
# which is that these are SCORING mistakes rather than perception ones. The passage of play
# happened; the record written against it is wrong. Cricket is the game whose scoring everybody
# here already knows ball by ball, so it is the shortest way to say which part of the record broke.
CRICKET = {
    "MISSED_STOP": "The umpire has signalled dead ball and the scorer keeps crediting runs off "
                   "the same delivery.",
    "SPELL_MERGE": "Two overs written into the book as one. Every delivery is there; the break "
                   "between them is not.",
    "INVENTED_STOP": "The scorer calls dead ball because the bowler slowed his run-up. Nothing "
                     "stopped &mdash; only the rhythm changed.",
    "BOUNDARY": "The wicket is in the book against the wrong ball number. The event is right, "
                "the ball-by-ball column is not.",
    "WRONG_TEAM": "The catch is credited to the wrong fielder. A dismissal did happen; the name "
                  "beside it belongs to somebody else.",
    "MISSED_CHANGE": "The batsmen crossed and the scorer missed it, so every run after this goes "
                     "against the man at the wrong end.",
    "FALSE_CHANGE": "A change of strike written down on a dot ball. The scorecard swaps ends "
                    "when nobody ran.",
}
BLIND_CRICKET = ("And the scorer never saw this delivery: he is going by where the fielders are "
                 "standing and the noise off the crowd.")

# A case this short says nothing about itself -- a turnover case is four frames long by
# construction -- so the description is widened around its centre.
PLAIN_WIDEN_S = 4.0


def plain_reason(g, a, b, kind, extra=None):
    """Why the model got this case wrong, in one shape for every kind of mistake.

    Four parts, and the shape never changes: what the model predicted against what the reference
    says, the evidence present in these frames, which reading of it decided the call, and the
    cricket equivalent. Everything in the prose is drawn from the readings block beside it, so
    the two can be checked against each other -- there is no claim here that is not a column.
    """
    n = len(g)
    a, b = max(0, a), min(n, max(a + 1, b))
    if (b - a) < int(1.0 * FPS):
        mid = (a + b) // 2
        half = int(PLAIN_WIDEN_S * FPS) // 2
        a, b = max(0, mid - half), min(n, mid + half)
        around = True
    else:
        around = False
    # A boundary case spans the reference stoppage, and the disagreement is outside it: in the
    # seconds the model added or dropped at one edge. The window has to reach them or the case
    # reads as perfect agreement.
    if kind == "BOUNDARY" and extra:
        pad = lambda k: int(min(15.0, max(2.0, abs(extra.get(k, 0.0)))) * FPS)
        a, b = max(0, a - pad("start_err_s")), min(n, b + pad("end_err_s"))
    span = (b - a) / FPS

    truth = g["label"].to_numpy()
    dead = g["pred_hsmm"].to_numpy().astype(bool)
    home = g["home_end_to_end"].fillna(False).to_numpy().astype(bool)
    said = np.where(dead, "D", np.where(home, "H", "A"))
    NAME = {"H": "home", "A": "away", "D": "ball dead"}

    def mean(col):
        if col not in g.columns:
            return None
        v = g[col].to_numpy(dtype=float)[a:b]
        v = v[np.isfinite(v)]
        return None if not len(v) else float(v.mean())

    def share(col):
        if col not in g.columns:
            return 0.0
        return float(g[col].fillna(False).to_numpy().astype(bool)[a:b].mean())

    def phrase(seq):
        """One answer across the window, compressed into its runs."""
        runs = []
        for i in range(a, b):
            if runs and runs[-1][0] == seq[i]:
                runs[-1][1] += 1
            else:
                runs.append([seq[i], 1])
        if len(runs) == 1:
            return "<b>{}</b> throughout".format(NAME[runs[0][0]])
        if len(runs) <= 3:
            return ", then ".join("<b>{}</b> for {:.1f} s".format(NAME[c], k / FPS)
                                  for c, k in runs)
        most = max(runs, key=lambda r: r[1])
        return ("<b>{}</b> for most of it, changing answer <b>{}</b> times and ending on "
                "<b>{}</b>".format(NAME[most[0]], len(runs) - 1, NAME[runs[-1][0]]))

    vis = g["ball_observed"].fillna(False).to_numpy().astype(bool)
    if "ball_observed_masked" in g.columns:
        vis = vis & g["ball_observed_masked"].fillna(True).to_numpy().astype(bool)
    seen = float(vis[a:b].mean())
    dis = float((said[a:b] != truth[a:b]).sum()) / FPS
    fl_arr = (g["in_flight"].fillna(False).to_numpy().astype(bool)
              if "in_flight" in g.columns else np.zeros(n, dtype=bool))
    ov_arr = (g["in_override"].fillna(False).to_numpy().astype(bool)
              if "in_override" in g.columns else np.zeros(n, dtype=bool))
    flight = float(fl_arr[a:b].mean())
    # Reading a touch is itself an override, so the plain override share has to exclude it or the
    # same frames are reported twice under two different names.
    over = float((ov_arr & ~fl_arr)[a:b].mean())
    rest = share("ball_at_rest")
    nh, na = mean("knn_nearest_home_m"), mean("knn_nearest_away_m")
    p_dead, p_home = mean("prob"), mean("p_home")
    bspeed = mean("ball_speed_ms")
    stop_kind = kind in ("MISSED_STOP", "SPELL_MERGE", "INVENTED_STOP", "BOUNDARY")

    # --- 1. what the two answers were, over the same frames
    lead = ("In the {:.1f} s around this moment".format(span) if around else
            "Across this stoppage and the seconds either side of it" if kind == "BOUNDARY" else
            "Over these {:.1f} s".format(span))
    what = "{} the model said {}. The reference says {}. They disagree for <b>{:.1f} s</b>.".format(
        lead, phrase(said), phrase(truth), dis)
    ex = extra or {}
    if kind == "BOUNDARY" and ("start_err_s" in ex or "end_err_s" in ex):
        se, ee = ex.get("start_err_s", 0.0), ex.get("end_err_s", 0.0)
        what += (" The stoppage itself was found &mdash; it is the edges that are out, by "
                 "{:+.1f} s at the start and {:+.1f} s at the end.".format(se, ee))
        # An edge far out that costs little disagreement means the overrun has landed on another
        # stoppage the reference also has. Reporting the two numbers without saying so reads as a
        # contradiction, and the distinction matters: it is a cheap error, not an expensive one.
        if dis < 0.5 * max(abs(se), abs(ee)):
            what += (" Most of that overrun lands on a stoppage the reference also has, which is "
                     "why it costs only {:.1f} s and not {:.1f}.".format(
                         dis, max(abs(se), abs(ee))))
    elif kind in ("MISSED_STOP", "SPELL_MERGE"):
        what += " Play was genuinely stopped for <b>{:.1f} s</b> of this.".format(
            float((truth[a:b] == "D").sum()) / FPS)
    elif kind == "INVENTED_STOP":
        what += " The reference has no stoppage here at all."

    # --- 2. the evidence in these frames, strongest reading first
    if flight >= 0.5:
        ev = ("The ball was in flight on <b>{:.0f}%</b> of these frames and the model read a touch "
              "on it, so possession went to whoever it passed closest to rather than to the "
              "per-frame probability.{}".format(
                  100 * flight,
                  " The ball was then out of sight for <b>{:.0f}%</b> of the window, so that touch "
                  "was being held blind.".format(100 * (1 - seen)) if seen < 0.5 else ""))
    elif seen < 0.5:
        ev = ("For <b>{:.0f}%</b> of these frames there was no ball for the model to read, so the "
              "answer came from the per-frame team model and the last side known to have control "
              "&mdash; not from the ball.".format(100 * (1 - seen)))
    elif flight >= 0.4:
        ev = ("The ball was in flight on <b>{:.0f}%</b> of these frames and the model read a touch "
              "on it, so possession went to whoever it passed closest to rather than to the "
              "per-frame probability.".format(100 * flight))
    elif over >= 0.4:
        rk = ""
        if "restart_kind" in g.columns:
            k = str(g["restart_kind"].fillna("").to_numpy()[max(0, a - 1)] or "")
            rk = " after {}".format(k.replace("_", " ")) if k else ""
        ev = ("The restart layer was holding the team call on <b>{:.0f}%</b> of these frames{} "
              "&mdash; a nearest-player vote and a held restart team, not the model's own "
              "probability.".format(100 * over, rk))
    elif stop_kind and rest >= 0.4:
        ev = ("The ball was in sight for <b>{:.0f}%</b> of these frames and counted as at rest on "
              "<b>{:.0f}%</b> of them, averaging <b>{:.1f} m/s</b> &mdash; a ball lying still is "
              "the strongest single sign of a stoppage this data carries.".format(
                  100 * seen, 100 * rest, bspeed if bspeed is not None else 0.0))
    elif stop_kind:
        ev = ("The ball was in sight for <b>{:.0f}%</b> of these frames and never settled: it "
              "averaged <b>{:.1f} m/s</b> and counted as at rest on only <b>{:.0f}%</b> of them, "
              "so there was nothing in the ball itself to read a stoppage from.".format(
                  100 * seen, bspeed if bspeed is not None else 0.0, 100 * rest))
    elif nh is not None and na is not None and np.isfinite(nh) and np.isfinite(na):
        near = "H" if nh < na else "A"
        want = "H" if (truth[a:b] == "H").mean() >= 0.5 else "A"
        ev = ("The ball was in sight for <b>{:.0f}%</b> of these frames, with the nearest home "
              "player <b>{:.1f} m</b> away and the nearest away player <b>{:.1f} m</b> &mdash; so "
              "the closest body was <b>{}</b>, {}.".format(
                  100 * seen, nh, na, NAME[near],
                  "which is the side the reference has for most of this window" if near == want
                  else "which is the wrong way round" if abs(nh - na) >= 0.3
                  else "by too little to mean anything"))
    else:
        ev = ("The ball was in sight for <b>{:.0f}%</b> of these frames and no single reading "
              "stands out: this is the per-frame model on its own.".format(100 * seen))

    # --- 3. and what the model's own confidence was while it decided it
    if stop_kind and p_dead is not None:
        if (p_dead >= 0.5) != bool(dead[a:b].mean() >= 0.5):
            conf = ("Its own per-frame probability that the ball was dead averaged <b>{:.2f}</b>, "
                    "which points the OTHER way from the answer it gave: the restart layer on top "
                    "of the decoder settled these frames, not the evidence under it.".format(
                        p_dead))
        else:
            conf = ("Its own probability that the ball was dead averaged <b>{:.2f}</b>, so the "
                    "answer it gave is the one its evidence pointed to, {}. The gap here is in "
                    "the evidence, not in the decoding.".format(p_dead,
                                       "and pointed to strongly" if abs(p_dead - 0.5) > 0.3
                                       else "though only just"))
    elif p_home is not None:
        conf = ("P(home) averaged <b>{:.2f}</b>, so the call was {}.".format(
            p_home,
            "a coin toss &mdash; a metre of noise either way would have flipped it"
            if abs(p_home - 0.5) < 0.1 else
            "confident, and confidently wrong" if abs(p_home - 0.5) > 0.3 else
            "made with something to spare, and still wrong"))
    else:
        conf = ""
    why = (ev + " " + conf).strip()

    # --- 4. the numbers the prose is drawn from
    reads = [{"k": "ball in sight", "v": "{:.0f}% of frames".format(100 * seen)}]
    if bspeed is not None:
        reads.append({"k": "ball speed", "v": "{:.1f} m/s".format(bspeed)})
    if rest > 0:
        reads.append({"k": "ball at rest", "v": "{:.0f}% of frames".format(100 * rest)})
    if nh is not None and na is not None:
        reads.append({"k": "nearest player",
                      "v": "home {:.1f} m / away {:.1f} m".format(nh, na)})
    if p_home is not None:
        reads.append({"k": "P(home)", "v": "{:.2f}".format(p_home)})
    if p_dead is not None:
        reads.append({"k": "P(ball dead)", "v": "{:.2f}".format(p_dead)})
    if flight > 0:
        reads.append({"k": "read as a touch", "v": "{:.0f}% of frames".format(100 * flight)})
    if over > 0:
        reads.append({"k": "restart rule driving",
                      "v": "{:.0f}% of frames".format(100 * over)})
    if "restart_kind" in g.columns:
        k = str(g["restart_kind"].fillna("").to_numpy()[max(0, a - 1)] or "")
        if k:
            reads.append({"k": "sits after", "v": k.replace("_", " ")})
    reads.append({"k": "they disagree for", "v": "{:.1f} s of {:.1f} s".format(dis, span)})

    cricket = CRICKET.get(kind, "")
    if seen < 0.5 and cricket:
        cricket += " " + BLIND_CRICKET
    return {"what": what, "why": why, "reads": reads, "cricket": cricket,
            "ball_seen": round(seen, 3)}


def case_rationale(g, a, b, kind, prior_dead):
    truth = g["label"].to_numpy()
    pred_dead = g["pred_hsmm"].to_numpy()
    home_pred = g["home_end_to_end"].fillna(False).to_numpy().astype(bool)
    stage2_kinds = {"SPELL_MERGE", "MISSED_STOP", "INVENTED_STOP", "BOUNDARY", "STOPPAGE_CAUGHT"}
    if kind in stage2_kinds:
        said_dead = bool(pred_dead[a:b].mean() >= 0.5)
        return stage2_rationale(g, a, b, prior_dead, said_dead)
    mid = min(len(g) - 1, (a + b) // 2)
    said = "D" if pred_dead[mid] else ("H" if home_pred[mid] else "A")
    if said == "D":
        return None
    return stage3_rationale(g, a, b, said, str(truth[mid]))


def restart_block(g, lo, a):
    """The restart this case sits after, as the hardening layer read it.

    Looks back from the start of the case for the nearest frame that carries restart fields --
    those are written across each stoppage -- so a case that follows a corner can say so even
    when the stoppage itself is outside the replay window.
    """
    if "restart_kind" not in g.columns:
        return None
    kinds = g["restart_kind"].fillna("").to_numpy()
    j = -1
    for i in range(a, max(-1, lo - int(20 * FPS)), -1):
        if i < 0:
            break
        if str(kinds[i]):
            j = i
            break
    if j < 0:
        return None
    name = {"H": "home", "A": "away"}

    def val(col):
        v = g[col].to_numpy()[j]
        return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v

    return {
        "kind": str(kinds[j]),
        "last_control": name.get(str(val("last_control") or ""), None),
        "restart_team": name.get(str(val("restart_team") or ""), None),
        "first_ball_frame": int(val("first_ball_frame") or -1),
        "resume_frame": int(val("resume_frame") or -1),
        "seconds_before_case": round(float(a - j) / FPS, 2),
    }


def export(match, g, players, period, cases, outdir, index, want_bundle, prior_dead=0.26,
           is_new=None):
    pos = wide_positions(players, period)
    if pos is None:
        return
    frames, X, Y, is_home, _ = pos
    t0 = float(g["time_s"].iloc[0])

    for k, (kind, a, b, extra) in enumerate(cases):
        new_since_fix = bool(is_new[k]) if is_new is not None and k < len(is_new) else False
        pad = int(CONTEXT_S * FPS)
        lo = max(0, a - pad)
        hi = min(len(g), b + pad)
        if (hi - lo) / FPS > MAX_WINDOW_S:
            keep = int(MAX_WINDOW_S * FPS)
            mid = (a + b) // 2
            lo, hi = max(0, mid - keep // 2), min(len(g), mid + keep // 2)

        sub = g.iloc[lo:hi:STRIDE]
        fn = sub["frame_native"].to_numpy()
        idx = pd.Index(frames).get_indexer(fn)
        ok = idx >= 0
        if ok.sum() < 5:
            continue
        sub, fn, idx = sub[ok], fn[ok], idx[ok]

        # positions as integers in decimetres of normalised pitch (0..1000), which keeps the
        # JSON small; -1 marks a slot that is not tracked in this frame.
        px = np.where(np.isfinite(X[idx]), np.round(X[idx] * 1000), -1).astype(int)
        py = np.where(np.isfinite(Y[idx]), np.round(Y[idx] * 1000), -1).astype(int)

        seen = ball_visible_share(g, a, b)
        share = contested_share(g, a, b)
        contested = share is not None and share > CONTESTED_SHARE
        success = kind in SUCCESS_KINDS
        if success and contested:
            continue          # a success story resting on an unsettled reference teaches nothing
        bucket = "corroborated" if success else ("contested" if contested else "confirmed")

        case_id = "{}_P{}_{}_{}".format(match.replace("DFL-MAT-", ""), period, kind,
                                        int(g["frame_native"].iloc[a]))
        meta = {
            "case_id": case_id,
            "match": match.replace("DFL-MAT-", ""),
            "period": int(period),
            "kind": kind,
            "outcome": "correct" if success else "failure",
            "bucket": bucket,
            "new_since_fix": new_since_fix,
            "reference_disagreement": None if share is None else round(share, 3),
            "ball_seen": round(seen, 3),
            "ball_visible": bool(seen >= BALL_VISIBLE),
            "clock": clock(period, float(g["time_s"].iloc[a]) - t0),
            "duration_s": round((b - a) / FPS, 2),
            "frame_start": int(g["frame_native"].iloc[a]),
            "frame_end": int(g["frame_native"].iloc[b - 1]),
            "extra": extra,
        }
        index.append(dict(meta, n_steps=len(sub), bundled=False))
        # A case the latest change created is always worth watching, whatever the cap says:
        # there are a dozen of them and each one is an argument about the change itself.
        if not new_since_fix and not want_bundle(kind, bucket):
            continue
        index[-1]["bundled"] = True

        restart = restart_block(g, lo, a)
        switches = []
        if "switch_reason" in sub:
            for t_rel, why in zip(sub["time_s"].to_numpy() - t0,
                                  sub["switch_reason"].fillna("").to_numpy()):
                if str(why):
                    switches.append({"t": round(float(t_rel), 2), "why": str(why)})

        feat_cols = [c for c in S2_FEATURES + S3_FEATURES + PC_FEATURES if c in sub.columns]
        in_case = np.array([a <= lo + i * STRIDE < b for i in range(len(sub))])
        pred_dead_s = sub["pred_hsmm"].to_numpy().astype(bool)
        home_s = sub["home_end_to_end"].fillna(False).to_numpy().astype(bool)

        def col(name, nd):
            v = sub[name].to_numpy(dtype=float)
            return [None if not np.isfinite(x) else round(float(x), nd) for x in v]

        # Columnar, not one dict per frame. The per-frame form repeated every key name on every
        # row, which was most of the file.
        rec = dict(meta,
                   stride=STRIDE, fps=FPS,
                   is_home=[bool(v) for v in is_home],
                   media={"kind": "tracking_animation",
                          "note": "IDSSE ships tracking only - no video exists for this match. "
                                  "This is a rendered animation of the coordinate file."},
                   ref_src="DFL BallPossession (provider_field, reference quality)",
                   f=[int(v) for v in fn],
                   t=[round(float(v) - t0, 2) for v in sub["time_s"].to_numpy()],
                   in_case="".join("1" if v else "0" for v in in_case),
                   pred="".join("D" if d else ("H" if h else "A")
                                for d, h in zip(pred_dead_s, home_s)),
                   ref="".join(str(v) for v in sub["label"].to_numpy()),
                   holder="".join({"home": "H", "away": "A"}.get(str(v or ""), "-")
                                  for v in sub["holder_side"].to_numpy()),
                   p_dead=col("prob", 4),
                   p_home=col("p_home", 4),
                   ball_seen="".join("1" if v else "0"
                                     for v in sub["ball_observed"].to_numpy().astype(bool)),
                   ball_seen_model="".join(
                       "1" if v else "0"
                       for v in sub["ball_observed_masked"].fillna(True).to_numpy().astype(bool)),
                   bx=col("ball_x", 4), by=col("ball_y", 4),
                   x=[r.tolist() for r in px], y=[r.tolist() for r in py],
                   plain=(plain_reason(g, a, b, kind, extra)
                          if kind not in SUCCESS_KINDS else None),
                   rationale=case_rationale(g, a, b, kind, prior_dead),
                   regression=(regression_reason(g, a, b, kind) if new_since_fix else None),
                   restart=restart,
                   switches=switches,
                   pred_before="".join(
                       "D" if d2 else ("H" if h2 else "A")
                       for d2, h2 in zip(
                           sub["pred_dead_base"].fillna(False).to_numpy().astype(bool)
                           if "pred_dead_base" in sub else pred_dead_s,
                           sub["home_base"].fillna(False).to_numpy().astype(bool)
                           if "home_base" in sub else home_s)),
                   knn_home_m=col("knn_nearest_home_m", 1) if "knn_nearest_home_m" in sub else [],
                   knn_away_m=col("knn_nearest_away_m", 1) if "knn_nearest_away_m" in sub else [],
                   knn_margin=col("knn_margin", 3) if "knn_margin" in sub else [],
                   ball_move_m=col("ball_move_m", 1) if "ball_move_m" in sub else [],
                   overridden="".join(
                       "1" if v else "0"
                       for v in (sub["in_override"].fillna(False).to_numpy().astype(bool)
                                 if "in_override" in sub else np.zeros(len(sub), bool))),
                   unsure="".join(
                       "1" if v else "0"
                       for v in (sub["uncertain"].fillna(False).to_numpy().astype(bool)
                                 if "uncertain" in sub else np.zeros(len(sub), bool))),
                   feat={c: col(c, 3) for c in feat_cols})
        with open(os.path.join(outdir, "cases", case_id + ".json"), "w") as fh:
            json.dump(rec, fh, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc", default="D:/Football Hackathon/external_data/processed/idsse")
    ap.add_argument("--refined", default="D:/Football Hackathon/stage3/predictions_refined")
    ap.add_argument("--s2feat", default="D:/Football Hackathon/stage2/features_h")
    ap.add_argument("--s2pred", default="D:/Football Hackathon/stage2/predictions_h")
    ap.add_argument("--s3feat", default="D:/Football Hackathon/stage3/features")
    ap.add_argument("--s3pred", default="D:/Football Hackathon/stage3/predictions_team")
    ap.add_argument("--pc", default="D:/Football Hackathon/stage3/pitch_control")
    ap.add_argument("--labels", default="D:/Football Hackathon/stage3/player_labels")
    ap.add_argument("--out", default="D:/Football Hackathon/review/data")
    args = ap.parse_args()

    paths = {"proc": args.proc, "s2feat": args.s2feat, "s2pred": args.s2pred,
             "s3feat": args.s3feat, "s3pred": args.s3pred, "pc": args.pc,
             "labels": args.labels, "refined": args.refined}
    os.makedirs(os.path.join(args.out, "cases"), exist_ok=True)

    # The prior the decoder used is the dead-ball rate of its TRAINING folds, so for each match
    # it is the rate over the other four. Recomputed here rather than guessed.
    dead_rate = {}
    for m in DEV:
        lab = pd.read_parquet(os.path.join(args.proc, m, "frames.parquet"),
                              columns=["label", "label_valid"])
        lab = lab[lab["label_valid"].astype(bool)]
        dead_rate[m] = (float((lab["label"] == "D").mean()), int(len(lab)))
    priors = {}
    for m in DEV:
        others = [(r, n) for k, (r, n) in dead_rate.items() if k != m]
        priors[m] = sum(r * n for r, n in others) / sum(n for _, n in others)
    print("dead-ball prior per fold: " +
          ", ".join("{} {:.3f}".format(k.replace("DFL-MAT-", ""), v) for k, v in priors.items()))

    index = []
    bundled = {}

    def want_bundle(kind, bucket):
        """Cap the common case types so the bundle set stays a few tens of megabytes."""
        if kind in NEVER_CAP:
            return True
        key = (kind, bucket)
        bundled[key] = bundled.get(key, 0) + 1
        return bundled[key] <= (SUCCESS_CAP if kind in SUCCESS_KINDS else BUNDLE_CAP)

    for match in DEV:
        d, players = load_match(match, paths)
        total = 0
        for period, g in d.groupby("period", sort=True):
            g = g.reset_index(drop=True)
            fails = find_cases(g)
            is_new = mark_new(fails, base_cases(g))
            cases = fails + find_successes(g)
            export(match, g, players, period, cases, args.out, index, want_bundle,
                   prior_dead=priors[match], is_new=is_new + [False] * (len(cases) - len(fails)))
            total += len(cases)
        print("  {}  {:>4} cases".format(match.replace("DFL-MAT-", ""), total))

    summary = pd.DataFrame(index)
    fl = summary[summary["outcome"] == "failure"]
    print("")
    print("=== of {:,} failures, how much of the ball the model could see ===".format(len(fl)))
    for name, m in (("the whole window", fl["ball_seen"] >= 0.999),
                    ("at least 80% of it (judgeable)", fl["ball_seen"] >= BALL_VISIBLE),
                    ("half to 80%", (fl["ball_seen"] >= 0.5) & (fl["ball_seen"] < BALL_VISIBLE)),
                    ("under half", fl["ball_seen"] < 0.5),
                    ("never at all", fl["ball_seen"] <= 0.001)):
        print("  {:<34}{:>7,}{:>8.0%}".format(name, int(m.sum()), m.mean() if len(fl) else 0))
    vis = fl[fl["ball_visible"]]
    print("\n=== the judgeable failures, by kind ===")
    print(vis.groupby("kind").size().sort_values(ascending=False).to_string())
    if "new_since_fix" in summary:
        nw = summary[summary["new_since_fix"]]
        print("")
        print("=== cases that exist only since the restart hardening: {} ===".format(len(nw)))
        if len(nw):
            print(nw.groupby(["kind", "bucket"]).size().to_string())
    print("")
    print("=== replay bundles written (the rest are counted only) ===")
    print(summary[summary["bundled"]].groupby(["kind", "bucket"]).size().to_string())
    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump({"generated_from": "open-access IDSSE development matches only",
                   "matches": [m.replace("DFL-MAT-", "") for m in DEV],
                   "sealed_matches_excluded": ["J03WMX", "J03WN1"],
                   "cases": index}, fh, separators=(",", ":"))
    print("\n=== cases by type and bucket ===")
    print(pd.crosstab(summary["kind"], summary["bucket"]).to_string())
    print("\nwrote {} cases to {}".format(len(index), args.out))


if __name__ == "__main__":
    main()
