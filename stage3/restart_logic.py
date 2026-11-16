"""Hardening pass over the decoded output: restarts, and carrying possession through ball gaps.

THE PROBLEM THIS SOLVES. Stage 2 decides live or dead from player geometry; stage 3 decides which
team from a per-frame probability. Neither knows anything about *restarts*. So at the moment play
resumes -- a corner swung in, a throw taken, a goal kick launched -- the team call is made by a
classifier looking at one frame of geometry, exactly where geometry is least informative: both
sides are packed into the box, and the only thing that settles possession is who takes the kick.

The Laws settle it instead, and the restart position says which Law applies. A ball placed on the
corner arc is taken by the side attacking that goal; a ball on the six-yard line at the other end
is a goal kick for the side defending it; a ball on the touchline is a throw. None of that needs
the ground truth, and none of it needs the competition's own data -- it is the Laws of the Game
read off the coordinates.

WHAT THIS MODULE DOES, in the order the frames arrive:

  1. For every stoppage, it records the last team in control and the likely restart team as two
     separate fields, classifies the restart from where the ball is resting, and looks for the
     moment play actually resumes -- the ball kicked and clearly moving, or for a throw, entering
     the pitch. The output stays D for the whole stoppage even once the restart team is known.
  2. On the first reliable ball frame it runs a distance-weighted k-NN over the nearest valid
     home and away players, then refuses to act on it until the trajectory and the players'
     movement agree for long enough that a deflection or a defender standing nearby cannot carry
     the decision.
  3. Through a gap in ball tracking it carries the last controlling team rather than letting the
     per-frame probability drift, and re-tests only when the ball comes back.

HONESTY CONSTRAINTS, which are what make the numbers mean anything:

  * The ball is used only where the *masked* ball is observed -- the same ball the model was
    allowed to see at the scored masking rate. Reading the unmasked ball here would be reading
    data the competition will not have.
  * No ground-truth label is read anywhere in the decision path. `label` is loaded for scoring
    only, and the scoring is the organisers' own code.
  * Development matches only. The two sealed matches are never opened.

    python stage3/restart_logic.py --apply --compare
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage2"))
from train import load_scorer, runs_of  # noqa: E402

FPS = 25
PITCH_X, PITCH_Y = 105.0, 68.0
DEV = ["DFL-MAT-J03WOH", "DFL-MAT-J03WOY", "DFL-MAT-J03WPY",
       "DFL-MAT-J03WQQ", "DFL-MAT-J03WR9"]

PROC = "D:/Football Hackathon/external_data/processed/idsse"
S3PRED = "D:/Football Hackathon/stage3/predictions_team"
S2PRED = "D:/Football Hackathon/stage2/predictions_h"
OUTDIR = "D:/Football Hackathon/stage3/predictions_refined"

# --- restart geometry, in metres. These are Laws-of-the-Game distances, not fitted thresholds.
CORNER_R = 4.0          # ball resting this close to a corner flag => corner kick
TOUCH_TOL = 1.8         # this close to a touchline (and not a corner) => throw-in
GOAL_X = 7.0            # within this of a goal line ...
GOAL_Y = 12.0           # ... and this close to the centre of the goal => goal kick
CENTRE_R = 3.5          # the centre spot => kick-off

# --- resumption
RESUME_WIN_S = 1.5      # how far the restart edge may be moved from where stage 2 put it
MOVE_WIN_S = 0.4        # window used to decide "the ball is clearly moving"
MOVE_MIN_M = 1.5        # metres in that window

# --- k-NN and the evidence needed to act on it
KNN_K = 2               # nearest two of each side; one player can be screened, two rarely are
KNN_EPS = 0.5           # metres, so a player standing on the ball does not divide by zero
MARGIN_MIN = 0.25       # (home-away)/(home+away) needed before a switch is even considered
PERSIST_S = 0.6         # the vote has to hold this long, which a deflection does not
PERSIST_FRAC = 0.6
CONTROL_R = 3.0         # a player this close is in contact range

# --- gaps and restart windows
GAP_MIN_S = 0.4         # shorter than this is not a gap, it is a dropped frame
RESTART_MAX_S = 4.0     # how long the restart team is held before the model takes over again
CORNER_HOLD_S = 0.8     # a corner is contested almost at once, so the taker is held briefly
CONFIRM_S = 1.0         # after a gap closes, how long the carried team is held pending evidence
RELEASE_S = 0.4         # the model has to hold its own view this long before it takes back over
LATE_GATE_MODE = "or"   # a later restart edge needs the visible ball OR stage 2 still
                        # calling it dead. Requiring both throws away two thirds of the
                        # fixes; requiring neither breaks nine stoppages that were timed
                        # to within half a second. Measured: 34 fixed / 3 new against
                        # 28 / 3 for "and" and 42 / 12 for no gate at all.
MIN_SPELL_S = 0.8       # a possession shorter than this is a wobble, not a possession

# --- the ball in flight. While a pass is travelling, the player closest to the ball is very
#     often an opponent it is going past, so "nearest player" and "in possession" come apart
#     exactly here. The pass belongs to whoever struck it until somebody else takes it.
SPEED_MAX_DT = 0.2      # seconds; two reliable observations further apart give no velocity
FLIGHT_V_MIN = 5.0      # m/s somewhere in the run before it counts as struck, not dribbled
FLIGHT_V_END = 3.0      # m/s; below this the ball is under somebody's foot again
FLIGHT_MIN_S = 0.2      # shorter than this is a touch
FLIGHT_MAX_S = 2.5      # beyond this the kicker is no longer the best account of possession
LAUNCH_BACK_S = 0.24    # how far back from the launch to look for the player who struck it
LAUNCH_R = 3.5          # and how close that player has to have been; further and we abstain
FLIGHT_HOLD_V = 3.0     # m/s. The rule speaks while the ball is going too fast for the player
                        #   standing nearest it to have it. It used to need 5 m/s, tuned before
                        #   the ball track was bridged; with the gaps joined up there is far more
                        #   continuous flight to follow, and the gate was swept again: 5 / 4 / 3
                        #   m/s give 2,662 / 2,642 / 2,635 failure cases, so it now covers the
                        #   whole of a moving run (FLIGHT_V_END, below). Going lower means
                        #   lowering that too, and 2.5 m/s is both worth only 8 more cases and
                        #   below the speed at which a ball can be under a dribbler's foot.
                        #   Lengthening a flight instead (FLIGHT_MAX_S 2.5 -> 3.5 s) trades
                        #   evenly and is not taken: +11 FALSE_CHANGE, -11 MISSED_CHANGE.
LAUNCH_TIGHT_R = 1.5    # the kicker has to have been this close at the launch ...
LAUNCH_CLEAR_M = 1.5    # ... and this much closer than the nearest opponent, or we abstain

# --- touches inside a flight. A ball can travel for several seconds through several players,
#     so one moving run is not one possession. The ball's own trajectory says where the touches
#     are: it loses pace or changes course, and only a player in contact can do that to it.
TOUCH_R = 1.5           # a player has to be this close to have played it
TOUCH_DECEL = 0.6       # pace dropping to this fraction of the previous frame's is a touch
TOUCH_COS = 0.5         # so is the course turning by more than 60 degrees
SWITCH_MIN_S = 0.0      # Optional: require the ball to stay with the toucher this long and
SWITCH_MIN_M = 0.0      # travel this far before a touch may take possession OFF the other side,
                        # on the brief's reasoning that a deflection is weak evidence. Measured,
                        # it is not worth it: 0.4 s/3 m, 0.6 s/6 m and 0.8 s/8 m fix 257, 243 and
                        # 224 cases against 286 ungated, and every step trades roughly two fixes
                        # for one regression avoided. Left here as a tunable, defaulted off.
FLIGHT_MODE = "both"    # "both": a touch both takes possession and keeps it. "hold": a flight
                        #   may only keep possession where it already is. Which is right depends
                        #   entirely on the touch being real: imposing a team on every moving ball
                        #   from its nearest player broke 538 cases, and imposing it only at a
                        #   detected touch fixes 325 and breaks 176. Holding alone fixes 76.

# --- the whistle, read off the players instead of the ball. A reviewer noticed that at a
#     stoppage all 22 players drop to a walk and stay there, which the data confirms: the mean
#     player speed is 1.10 m/s while the ball is dead against 2.42 m/s while it is live, in
#     every one of the five development matches. It is a strong signal and a slow one -- the
#     walk sets in a median 4.6 s AFTER the whistle, because players keep running until play
#     has plainly stopped -- so it is used only as a SECOND reading of an edge stage 2 already
#     found, never to find a stoppage of its own. See RESTART_HARDENING.md for the measurements
#     that rejected the wider versions.
PACE_WIN_S = 1.0        # team pace is averaged over this long on each side of a candidate frame
PACE_SEARCH_S = 2.0     # how far from stage 2's whistle the steepest drop is looked for
PACE_W = 0.5            # how much of the way to move: 0 keeps stage 2, 1 takes pace's word.
                        #   Measured at the whistle, |error| against the reference is 0.80 s for
                        #   stage 2 alone, 0.76 s for the drop alone and 0.70 s averaged, and
                        #   the share of whistles timed to within a second goes 56% -> 61%.
                        #   Two independent readings of one event; neither wins alone.
PACE_MAX_V = 12.0       # m/s; a player cannot run faster, so this is a tracking glitch
PACE_MIN_PLAYERS = 14   # fewer tracked than this and the average is not the team's

# --- a ball lying still with nobody near it. A reviewer watching J03WOH at 30:10 pointed out
#     that the ball had simply stopped, untouched, which can only mean play had already been
#     stopped. Unlike the players slowing down, this is almost never ambiguous: over the five
#     development matches, 86 such windows cover 4,426 frames the model was calling LIVE, and
#     the reference calls EVERY ONE of them dead. So this rule is allowed to do what the pace
#     rule was not -- find a stoppage of its own.
LAST_TOUCH_S = 3.0      # how far back from a whistle the last touch on the ball is looked for.
                        #   A reviewer watching balls go out of play pointed out that who
                        #   restarts is decided by who touched it last, not by who the decoder
                        #   happened to believe was in possession at the time.
# --- and its mirror: a ball that is plainly moving cannot be a dead ball. Four reviewer notes
#     in one round were the same mistake -- a keeper playing on, a ball being passed around a
#     back four, players and ball both moving -- every one a stoppage the model INVENTED where
#     the ball had gone out of sight. Measured inside the regions the model calls dead, a
#     visible ball moving at 1.5 m/s or more for half a second is live in 198 of 198 windows
#     (5,864 frames). The players say nothing useful here: at 80-100% of match pace they are
#     right 11-15% of the time, because 22 people jog about during a stoppage.
PLAY_V = 1.5            # m/s, measured at 1.5 / 2.0 / 3.0: 198 / 194 / 176 windows, all 100%
PLAY_S = 0.5            # and held this long, so one noisy observation cannot reopen play

# How long a window every ball-speed test measures displacement over. None = decide from the
# ball track itself, which is the default and what ball_speed() documents. A number forces it:
# 0 is a plain frame difference, 1.0 is a one-second displacement.
SPEED_WIN_S = None
SPEED_WIN_AUTO_S = 1.0  # the window used when the track is not a frozen-coordinate one
# A provider that freezes the ball's coordinate while play is stopped repeats the same position
# on consecutive frames. One that re-detects every frame essentially never does. Measured: the
# five IDSSE training matches repeat on 47-51% of consecutive observed pairs, the competition's
# ATD on far less than 1%. Any threshold between those two is the same decision; 10% is an order
# of magnitude clear of both.
FROZEN_REPEAT_SHARE = 0.10

# --- eleven a side. A reviewer's rule, and the best-aimed one yet: "unless there is a red card
#     both teams have eleven players, so if one side is a man short -- especially at a throw-in,
#     a corner or a free kick from outside -- that missing player is the one taking it, and the
#     restart belongs to his team." Measured over 493 reference restarts, the side with a player
#     standing OFF the pitch nearest the ball names the taker correctly 95% of the time where it
#     fires (155 restarts) against the model's own 85% on those same restarts; where the two
#     disagree, the head-count is right 16 times out of 17. Simply counting who has fewer players
#     inside the pitch is the looser reading of the same idea and only reaches 86%, so it is the
#     player standing outside, next to the ball, that the rule actually keys on.
OFF_PITCH_R = 10.0      # how close to the resting ball that off-pitch player has to be

# --- the blind spot. The ball is hidden from the model 41% of the time, and the hidden stretches
#     fall exactly where possession is decided: in the middle of passes. Where the ball is seen
#     on BOTH sides of a short gap, a straight line between the two sightings is an extremely
#     good guess at where it was. Measured against the real track on the five development
#     matches, over 4,410 gaps:
#         gap <= 0.8 s   median error 0.03 m    (1,509 gaps, 11,233 frames)
#         gap <= 1.6 s   median error 0.17 m    (2,124 gaps, 33,592 frames)
#         gap <= 3.2 s   median error 0.91 m, p90 4.08 m
#         gap  > 3.2 s   median error 3.32 m -- no longer a guess worth making
#     End to end the cap is worth sweeping rather than reasoning about: 1.6 s fixes 61
#     cases net, 3.2 s fixes 73, and 6.4 s drops back to 52 because long bridges start
#     to mis-time stoppage edges (BOUNDARY goes +4 to -25). So 3.2 s it is. The bridged
#     ball is used ONLY by the rules that follow the ball itself -- the flight and touch
#     logic, and the moving-ball veto -- and never by the per-frame nearest-player vote,
#     which must keep speaking only about a ball it can actually see.
BRIDGE_MAX_S = 3.2

# --- changes of hands the model never makes. Outside a restart window the refined sequence is
#     the per-frame team model's own output: the nearest-player vote, the one cue that is about
#     where the ball is right now, is never consulted. 45% of the 1,032 missed changes had that
#     vote pointing the right way. Measured over the windows where the vote is firm and
#     disagrees with the model, who is right depends almost entirely on whether anybody is
#     actually NEXT TO the ball:
#         nearest player any distance   vote right 41%, model right 59%
#         nearest player within 3 m     vote 55%, model 45%
#         nearest player within 2 m     vote 64%, model 36%   (0.6 s hold)
#     At arm's length the vote is right two times in three, so it is allowed to take possession
#     there and nowhere else.
#     Swept end to end, the radius is the whole story and 1.5 m -- the same arm's length the
#     touch rule already uses -- is where it settles: 1.5 m fixes 74 cases and creates 45,
#     2.0 m fixes 97 and creates 69 for the same final total, and 3.0 m fixes 147 and creates
#     136, which is a coin toss dressed up as a rule. Holding the vote longer (1.0 s) or asking
#     for a wider margin (0.50) both lose more than they save.
TAKE_R = 1.5            # a player has to be this close before the vote may override the model
TAKE_S = 0.6            # and say the same thing for this long
TAKE_MARGIN = 0.25      # with at least this much daylight between the two sides

# --- and the mirror of it: you cannot take possession of a ball you are nowhere near. Three
#     quarters of the false switches are the per-frame model's own, not a rule's, and the
#     evidence at the moment it switches says plainly which ones to doubt. Of the 977 switches
#     the model makes, 55% are real; of the ones where the side it switches TO has nobody
#     within 3 m of the ball, only 26% are. Blocking those costs 23 real changes of hands and
#     saves 67 false ones.
REACH_R = 3.0           # a team needs a player this close to the ball to be given it
REACH_VOTE = True       # and the nearest-player vote has to disagree with the switch too
STILL_V = 0.5           # m/s. A ball rolling slower than this is at rest, not in play.
STILL_S = 1.0           # and it has to stay there this long, so a ball under a foot for a
                        #   moment, or two noisy observations, do not read as a stoppage.
                        #   Measured at 1.0 / 1.5 / 2.5 s: 86 / 49 / 18 windows, all at 100%.
# --- and the simplest Law of all, which three reviewer notes in one round asked for: a ball
#     sitting outside the lines is nobody's. "Ball is still and also out of boundary areas, so
#     any team can't have the possession." Measured: of every frame where the model can see the
#     ball outside the pitch, 99% are dead in the reference; of the 3,838 such frames the model
#     still calls live, 87% are dead. Adding "and barely moving" -- so a ball being thrown back
#     in does not count -- takes it to 99% over 2,296 frames.
# --- and the note beside it: "between 2 dead phases a team possession is very unlikely,
#     especially when the ball is still". Measured over the 592 stretches of play the model
#     leaves between two stoppages it has already found:
#         under 1 s                         20 stretches, 199 frames, 100% really dead
#         under 2 s, ball barely moving     18 stretches, 495 frames,  92%
#         under 4 s, ball never seen        25 stretches, 836 frames,  89%
#         8 s and over                     439 stretches,            only 3%
#     Swept end to end, the thresholds that read best are wider than the per-window numbers
#     suggest: 1/2 s fixes 20 cases net, 2/4 s fixes 30, 3/6 s drops back to 25.
ISLAND_S = 2.0          # a stretch of play shorter than this between two stoppages is not one
ISLAND_SLOW_S = 4.0     # nor is one shorter than this where the ball never really moved
ISLAND_TRAVEL_M = 2.0   # ... which means it stayed inside this radius
OUT_V = 1.0             # m/s; above this the ball is being put back into play, not lying out
OUT_S = 1.0             # and it has to be out and slow for this long (0.5 s is no better:
                        #   9 cases fixed and 9 created, against 10 and 4 at a second)
STILL_R = 2.0           # and nobody may be within this of it. 3 m (CONTROL_R, what the rest of
                        #   the model calls contact range) is the cleaner window-level number --
                        #   86 windows, all of them truly dead, against 119 windows at 98% for
                        #   2 m -- but end to end the looser radius is worth 25 cases fixed
                        #   against 11 created, so the measurement wins over the tidier figure.


# ----------------------------------------------------------------------------- geometry helpers
def wide_positions(players, period):
    """(frame index, X, Y, is_home) in metres, one column per tracked slot."""
    g = players[players["period"] == period]
    if g.empty:
        return None
    g = g.assign(col=g["team"].astype(str) + "_" + g["slot"].astype(str))
    x = g.pivot_table(index="frame_native", columns="col", values="x", aggfunc="first").sort_index()
    y = g.pivot_table(index="frame_native", columns="col", values="y",
                      aggfunc="first").sort_index().reindex(columns=x.columns)
    is_home = np.array([c.startswith("home_") for c in x.columns])
    return (x.index.to_numpy(), x.to_numpy() * PITCH_X, y.to_numpy() * PITCH_Y, is_home)


def home_attacks_right(X, is_home):
    """Which way home is attacking this period, inferred the same way stage 3 infers it.

    The side defending the left goal keeps its deepest player pinned near x=0 all period; the
    other side's deepest player is an outfielder much further up. No role labels needed.
    """
    with np.errstate(invalid="ignore"):
        deep_home = np.nanmean(np.nanmin(X[:, is_home], axis=1))
        deep_away = np.nanmean(np.nanmin(X[:, ~is_home], axis=1))
    return bool(deep_home < deep_away)


def knn_scores(bx, by, X, Y, is_home):
    """Distance-weighted vote per frame: closer players count for more, k nearest per side.

    Returns (score_home, score_away, nearest_home_m, nearest_away_m). Frames with no ball, or a
    side with no tracked player, come back NaN -- never zero, which would read as evidence.
    """
    n = len(bx)
    d = np.sqrt((X - bx[:, None]) ** 2 + (Y - by[:, None]) ** 2)      # n x slots, NaN-safe below
    out = []
    for side in (is_home, ~is_home):
        ds = np.sort(np.where(np.isfinite(d[:, side]), d[:, side], np.inf), axis=1)[:, :KNN_K]
        w = np.where(np.isfinite(ds), 1.0 / (ds + KNN_EPS) ** 2, np.nan)
        score = np.nansum(np.where(np.isfinite(w), w, 0.0), axis=1)
        score = np.where(np.isfinite(ds[:, 0]), score, np.nan)
        out.append((score, np.where(np.isfinite(ds[:, 0]), ds[:, 0], np.nan)))
    return out[0][0], out[1][0], out[0][1], out[1][1]


def team_pace(X, Y, fnum, pred_dead):
    """How fast everybody is moving, and how sharply that is falling.

    Returns (pace, drop, baseline): the mean speed of every tracked player smoothed over
    PACE_WIN_S, the fall in that mean across each frame (positive = slowing down), and the
    match's own typical pace, taken as the median over the frames stage 2 calls live. The
    baseline is what makes "slowed to a walk" mean the same thing in a fast match and a slow
    one.
    """
    n = len(fnum)
    dt = np.diff(fnum).astype(float) / FPS
    dt = np.where(dt > 0, dt, np.nan)
    with np.errstate(invalid="ignore"):
        sp = np.sqrt(np.diff(X, axis=0) ** 2 + np.diff(Y, axis=0) ** 2) / dt[:, None]
        sp = np.where(np.isfinite(sp) & (sp <= PACE_MAX_V), sp, np.nan)
        pace = np.full(n, np.nan)
        pace[1:] = np.where(np.isfinite(sp).sum(axis=1) >= PACE_MIN_PLAYERS,
                            np.nanmean(sp, axis=1), np.nan)
    if n > 1:
        pace[0] = pace[1]
    k = max(1, int(PACE_WIN_S * FPS))
    before = pd.Series(pace).rolling(k, min_periods=max(1, k // 2)).mean().to_numpy()
    after = np.full(n, np.nan)
    after[:max(0, n - k)] = before[k:]
    live = pace[~pred_dead.astype(bool)]
    live = live[np.isfinite(live)]
    baseline = float(np.median(live)) if live.size else np.nan
    return before, before - after, baseline


def bridge_ball(bx, by, reliable):
    """Fill short gaps in the ball track by joining the sightings either side of them.

    Only gaps with a sighting on BOTH sides are filled: an unseen ball at the start or the end
    of a half, or either side of a long disappearance, is left unseen. Returns the filled track
    and a mask marking which frames are inferred rather than seen.
    """
    n = len(bx)
    fx, fy = bx.copy(), by.copy()
    inferred = np.zeros(n, dtype=bool)
    lim = int(BRIDGE_MAX_S * FPS)
    for a, b in runs_of(~reliable):
        if a == 0 or b >= n or (b - a) > lim:
            continue
        i, j = a - 1, b
        t = (np.arange(a, b) - i) / float(j - i)
        fx[a:b] = bx[i] + t * (bx[j] - bx[i])
        fy[a:b] = by[i] + t * (by[j] - by[i])
        inferred[a:b] = True
    return fx, fy, inferred


def ball_speed(bx, by, reliable):
    """How fast the ball is going, from displacement across a window rather than one frame.

    WHY NOT np.diff. One frame difference is 40 ms of a computer-vision detection, so the speed
    it reports is the detector's own jitter divided by 0.04 s. On a provider that FREEZES the
    ball's coordinate while play is stopped -- which is what every training match here does --
    that jitter is exactly zero, a dead ball reads 0.00 m/s, and every speed test built on
    np.diff works perfectly. On a provider that re-detects the ball every frame instead, a few
    centimetres of detection noise reads as several metres a second, and a ball lying still on
    the turf is indistinguishable from one being passed.

    That is not a hypothetical: the competition's ATD re-detects every frame, and the gap shows
    up in the rules downstream. `ball_in_play` reopened a quarter of the stoppages on the
    competition match and `ball_at_rest` fired on 1.6% of frames against 22% here.

    Displacement over a window divides that noise by the window length and leaves a genuinely
    moving ball's speed alone. Both endpoints must be seen, and most of the window with them, so
    this never measures across a gap in the track.

    WHICH ESTIMATOR, DECIDED FROM THE TRACK AND NOTHING ELSE. On a frozen-coordinate provider a
    frame difference is not merely adequate, it is *exact* -- the ball reads 0.00 m/s when play
    is stopped and the timing of an edge is resolved to 40 ms. Windowing it can only blur that,
    and measured on the five development matches it does: spellF1 0.519 -> 0.517, restart edge
    error 1.04 s -> 1.12 s, 41 more failure cases. So the window is not applied there. The test
    is whether the track repeats a position on consecutive frames, which is the signature of a
    provider that freezes it -- a structural property of the file, with no label, no score and
    nothing about the competition match in it. Pass SPEED_WIN_S explicitly to override.
    """
    n = len(bx)
    sp = np.full(n, np.nan)
    win = SPEED_WIN_S
    if win is None:
        both = reliable & np.concatenate(([False], reliable[:-1]))
        with np.errstate(invalid="ignore"):
            same = both & np.concatenate(([False], (np.diff(bx) == 0) & (np.diff(by) == 0)))
        share = float(same.sum()) / max(1, int(both.sum()))
        win = 0.0 if share >= FROZEN_REPEAT_SHARE else SPEED_WIN_AUTO_S
    if not win:
        with np.errstate(invalid="ignore"):
            sp[1:] = np.sqrt(np.diff(bx) ** 2 + np.diff(by) ** 2) * FPS
        return sp

    half = max(1, int(round(win * FPS / 2.0)))
    i = np.arange(n)
    lo = np.clip(i - half, 0, n - 1)
    hi = np.clip(i + half, 0, n - 1)
    seen = np.concatenate(([0], np.cumsum(reliable.astype(np.int64))))
    span = hi - lo
    enough = (seen[hi + 1] - seen[lo]) >= np.maximum(1, (span + 1) * 2 // 3)
    ok = reliable[lo] & reliable[hi] & (span > 0) & enough
    with np.errstate(invalid="ignore"):
        gone = np.sqrt((bx[hi] - bx[lo]) ** 2 + (by[hi] - by[lo]) ** 2)
    return np.where(ok, gone / (span / FPS), np.nan)


def ball_in_play(bx, by, reliable):
    """Frames where the visible ball is inside the pitch and moving too fast for play to be dead.

    The ball has to be inside the field of play as well as moving: a ball still rolling after it
    has crossed the line is moving and dead at the same time.
    """
    n = len(bx)
    with np.errstate(invalid="ignore"):
        sp = ball_speed(bx, by, reliable)
        inside = (bx >= 0) & (bx <= PITCH_X) & (by >= 0) & (by <= PITCH_Y)
    fast = reliable & inside & np.isfinite(sp) & (sp >= PLAY_V)
    out = np.zeros(n, dtype=bool)
    need = max(1, int(PLAY_S * FPS))
    for a, b in runs_of(fast):
        if b - a >= need:
            out[a:b] = True
    return out


def ball_out_of_play(bx, by, reliable):
    """Frames where the visible ball is lying outside the field of play.

    No distance-to-player test here, and deliberately so: at a throw-in the taker is standing
    right next to it, which is exactly the case the still-ball rule cannot see.
    """
    n = len(bx)
    with np.errstate(invalid="ignore"):
        sp = ball_speed(bx, by, reliable)
        outside = (bx < 0) | (bx > PITCH_X) | (by < 0) | (by > PITCH_Y)
    flag = reliable & outside & np.isfinite(sp) & (sp <= OUT_V)
    out = np.zeros(n, dtype=bool)
    need = max(1, int(OUT_S * FPS))
    for a, b in runs_of(flag):
        if b - a >= need:
            out[a:b] = True
    return out


def ball_at_rest(bx, by, X, Y, reliable):
    """Frames where the visible ball is lying still and no player is within contact range.

    The ball has to be visible throughout -- a ball that is not seen to move is also what a
    masked ball looks like, and that is not evidence of anything.
    """
    n = len(bx)
    with np.errstate(invalid="ignore"):
        sp = ball_speed(bx, by, reliable)
        near = np.nanmin(np.sqrt((X - bx[:, None]) ** 2 + (Y - by[:, None]) ** 2), axis=1)
    still = reliable & np.isfinite(sp) & (sp <= STILL_V) & np.isfinite(near) & (near > STILL_R)
    out = np.zeros(n, dtype=bool)
    need = max(1, int(STILL_S * FPS))
    for a, b in runs_of(still):
        if b - a >= need:
            out[a:b] = True
    return out


def move_whistle(a, b, floor, drop, baseline):
    """Where the whistle went: stage 2's edge averaged with the steepest fall in team pace.

    `a` is stage 2's edge and `floor` the earliest frame this stoppage may claim. The answer may
    be earlier or later than `a`, never outside (floor, b).
    """
    if not np.isfinite(baseline):
        return a
    lo = max(floor, a - int(PACE_SEARCH_S * FPS))
    hi = min(b, a + int(PACE_SEARCH_S * FPS))
    if hi <= lo:
        return a
    seg = drop[lo:hi]
    if not np.isfinite(seg).any():
        return a
    j = lo + int(np.nanargmax(seg))
    k = int(round(a + PACE_W * (j - a)))
    return int(min(max(k, floor), b - 1))


def ball_flight(bx, by, reliable, nh, na):
    """Where the ball is in flight, and which side struck it.

    This exists because of one failure the per-frame classifier cannot talk itself out of: while
    a pass is in the air the nearest player is frequently the opponent the ball is travelling
    past, so every cue built on distance-to-ball reads the receiving end of a pass as a turnover.
    Possession is settled at the other end -- where the ball came from.

    The kicker is found by looking back a few frames from the launch and taking the player who
    got closest to the ball, which is the one who struck it. If nobody was within LAUNCH_R we do
    not know who kicked it and the flight is left alone rather than guessed at.

    One long moving run is not one possession: a ball can travel for seconds through several
    players. So the run is cut at every touch -- a loss of pace or a change of course that only
    a player in contact can have caused -- and each piece is attributed separately.

    Returns (speed_ms, flight_mask, flight_team, flight_firm), where `firm` marks the segments
    that are solid enough to take possession off the other side rather than merely keep it.
    """
    n = len(bx)
    speed = np.full(n, np.nan)
    vx, vy = np.full(n, np.nan), np.full(n, np.nan)
    ridx = np.flatnonzero(reliable)
    if ridx.size > 1:
        i0, i1 = ridx[:-1], ridx[1:]
        dt = (i1 - i0) / FPS
        ok = dt <= SPEED_MAX_DT
        dx, dy = bx[i1] - bx[i0], by[i1] - by[i0]
        with np.errstate(invalid="ignore", divide="ignore"):
            vx[i0[ok]] = dx[ok] / dt[ok]
            vy[i0[ok]] = dy[ok] / dt[ok]
            speed[i0[ok]] = np.sqrt(dx[ok] ** 2 + dy[ok] ** 2) / dt[ok]

    near = np.fmin(nh, na)
    mask = np.zeros(n, dtype=bool)
    firm = np.zeros(n, dtype=bool)
    team = np.full(n, "", dtype=object)

    def owner(k, r):
        """Whose player was on the ball at frame k, or '' where it does not settle it."""
        if not (np.isfinite(nh[k]) and np.isfinite(na[k])):
            return ""
        if near[k] > r or abs(nh[k] - na[k]) < LAUNCH_CLEAR_M:
            return ""                     # nobody on it, or two of them: not a clean touch
        return "H" if nh[k] <= na[k] else "A"

    def touched(i):
        """Did the ball's own trajectory break at frame i, with somebody close enough to do it?"""
        if not (np.isfinite(speed[i]) and np.isfinite(speed[i - 1])) or near[i] > TOUCH_R:
            return False
        if speed[i] <= TOUCH_DECEL * speed[i - 1]:
            return True
        d0, d1 = np.hypot(vx[i - 1], vy[i - 1]), np.hypot(vx[i], vy[i])
        if not (d0 > 0 and d1 > 0):
            return False
        return bool((vx[i] * vx[i - 1] + vy[i] * vy[i - 1]) / (d0 * d1) <= TOUCH_COS)

    moving = np.isfinite(speed) & (speed >= FLIGHT_V_END)
    back = max(1, int(LAUNCH_BACK_S * FPS))
    for a, b in runs_of(moving):
        if np.nanmax(speed[a:b]) < FLIGHT_V_MIN or (b - a) < FLIGHT_MIN_S * FPS:
            continue
        # the strike that began the run: the player who got closest just before it
        w0 = max(0, a - back)
        starts = []
        if np.isfinite(near[w0:a + 1]).any():
            k = w0 + int(np.nanargmin(near[w0:a + 1]))
            if near[k] <= LAUNCH_R:
                starts.append((a, owner(k, LAUNCH_TIGHT_R)))
        # and every touch inside it
        starts += [(i, owner(i, TOUCH_R)) for i in range(a + 1, b) if touched(i)]

        for j, (s, who) in enumerate(starts):
            e = min(b, s + int(FLIGHT_MAX_S * FPS),
                    starts[j + 1][0] if j + 1 < len(starts) else b)
            if not who or e <= s:
                continue
            # Only the part that is genuinely nobody's: a ball at this pace cannot be in another
            # player's possession however close he is standing.
            hold = np.isfinite(speed[s:e]) & (speed[s:e] >= FLIGHT_HOLD_V)
            if not hold.any():
                continue
            hold[0] = True                # the touch itself is the moment possession changes
            mask[s:e] |= hold
            team[s:e] = np.where(hold, who, team[s:e])
            with np.errstate(invalid="ignore"):
                travel = np.nanmax(np.hypot(bx[s:e] - bx[s], by[s:e] - by[s]))
            if (e - s) >= SWITCH_MIN_S * FPS and (not np.isfinite(travel)
                                                  or travel >= SWITCH_MIN_M):
                firm[s:e] |= hold
    return speed, mask, team, firm


def end_geometry(X, Y, is_home, i0, i1, right_end):
    """What the players are doing at one end of the pitch during a stoppage.

    Returns (flag_dist_m, flag_is_home, crowd) -- how close the nearest player gets to a corner
    flag at that end, whose player that is, and how many bodies are in the penalty area. A corner
    puts a taker on the flag and both teams in the box; a goal kick does neither.
    """
    gx = PITCH_X if right_end else 0.0
    sl = slice(i0, i1)
    dx = np.abs(X[sl] - gx)
    dy = np.minimum(Y[sl], PITCH_Y - Y[sl])
    d_flag = np.sqrt(dx ** 2 + dy ** 2)
    if not d_flag.size or not np.isfinite(d_flag).any():
        return np.nan, None, 0.0
    with np.errstate(invalid="ignore"):
        allnan = np.all(~np.isfinite(d_flag), axis=0)
        med = np.where(allnan, np.nan, np.nanmedian(np.where(np.isfinite(d_flag), d_flag,
                                                             np.nan), axis=0))
        inbox = (dx <= 16.5) & (np.abs(Y[sl] - PITCH_Y / 2) <= 20.16)
        crowd = float(np.median(inbox.sum(axis=1)))
    if not np.isfinite(med).any():
        return np.nan, None, crowd
    j = int(np.nanargmin(med))
    return float(med[j]), bool(is_home[j]), crowd


def classify_restart(bx, by, attacks_right, end_geo=None):
    """What kind of restart the resting ball implies, and who the Laws give it to.

    Returns (kind, team) where team is 'H', 'A' or '' when the position does not settle it.
    `attacks_right` is True when home attacks the x=105 goal this period.

    `near_x` can be negative: the ball is tracked beyond the goal line when it is turned behind,
    and that is exactly the case a corner has to be recognised in. Position alone cannot separate
    a corner from a goal kick there -- both leave the ball off the end of the pitch -- so the
    players decide it.
    """
    if not (np.isfinite(bx) and np.isfinite(by)):
        return "unknown", ""
    near_x = min(bx, PITCH_X - bx)
    near_y = min(by, PITCH_Y - by)
    right_end = bx > PITCH_X / 2
    # the side attacking the end the ball is at
    attacker = ("H" if attacks_right else "A") if right_end else ("A" if attacks_right else "H")
    defender = "A" if attacker == "H" else "H"

    if near_x <= CORNER_R and near_y <= CORNER_R:
        return "corner", attacker                     # the ball is on the corner arc itself
    if near_y <= TOUCH_TOL and near_x > CORNER_R:
        return "throw_in", ""                         # the Laws do not say; geometry has to
    if near_x <= GOAL_X:
        flag_d, flag_home, crowd = end_geo if end_geo else (np.nan, None, 0.0)
        if np.isfinite(flag_d) and flag_d <= 3.5 and crowd >= 6:
            return "corner", ("H" if flag_home else "A")   # a taker is standing on the flag
        if crowd >= 7:
            return "corner", attacker                      # the box is loaded; nobody does that
        if abs(by - PITCH_Y / 2) <= GOAL_Y:                # for a goal kick
            return "goal_kick", defender
        return "free_kick", ""
    if (abs(bx - PITCH_X / 2) <= CENTRE_R) and (abs(by - PITCH_Y / 2) <= CENTRE_R):
        return "kick_off", ""
    return "free_kick", ""


# --------------------------------------------------------------------------------- the main pass
def refine_period(df, X, Y, is_home, mode="carry", align=True, steps=("align", "corner"),
                  late_gate=True):
    """Refine one period in place-ish, returning the new labels and every field behind them."""
    n = len(df)
    fnum = df["frame_native"].to_numpy()
    step = 1.0 / FPS
    pred_dead = df["pred_dead"].to_numpy().astype(bool)
    home_dec = df["home_base"].to_numpy().astype(bool)
    # the ball AS THE MODEL WAS ALLOWED TO SEE IT, never the unmasked track
    reliable = (df["ball_observed"].to_numpy().astype(bool)
                & df["ball_observed_masked"].to_numpy().astype(bool))
    p_dead = df["prob"].to_numpy(dtype=float) if "prob" in df else np.full(n, 0.5)
    bx = np.where(reliable, df["ball_x"].to_numpy() * PITCH_X, np.nan)
    by = np.where(reliable, df["ball_y"].to_numpy() * PITCH_Y, np.nan)

    # A ball lying still and untouched is a stoppage stage 2 missed. It is folded into the dead
    # mask here, before anything else runs, so the whistle, the restart alignment and the team
    # carried across the gap all treat it as the ordinary stoppage it is.
    # The ball the ball-following rules get to use: the same track, with short gaps joined up.
    # The per-frame nearest-player vote keeps the strictly-seen ball, so a guess can never
    # become an opinion about who is in possession.
    if "bridge" in steps:
        bxb, byb, inferred = bridge_ball(bx, by, reliable)
        seen_plus = reliable | inferred
    else:
        bxb, byb = bx, by
        inferred = np.zeros(n, dtype=bool)
        seen_plus = reliable

    still = (ball_at_rest(bx, by, X, Y, reliable) if "still" in steps
             else np.zeros(n, dtype=bool))
    if "outdead" in steps:
        still = still | ball_out_of_play(bx, by, reliable)
    # ... and a ball we can see moving inside the pitch means play is going on, whatever stage 2
    # decided from a ball it could not see. The two can never fire together: one needs the ball
    # at rest, the other needs it moving.
    inplay = (ball_in_play(bxb, byb, seen_plus) if "inplay" in steps
              else np.zeros(n, dtype=bool))
    pred_dead = pred_dead | still
    if inplay.any():
        # Cutting the moving frames out would leave a stoppage in pieces, and a piece either
        # side of a pass is two invented stoppages where there was one: measured that way the
        # rule fixed 52 cases and created 99. So a stoppage that contains play survives as its
        # longest surviving stretch, and the rest of it is given back to the match.
        keep = np.zeros(n, dtype=bool)
        for a, b in runs_of(pred_dead):
            pieces = [(a + c, a + d) for c, d in runs_of(pred_dead[a:b] & ~inplay[a:b])]
            if pieces:
                c, d = max(pieces, key=lambda p: p[1] - p[0])
                keep[c:d] = True
        # Which side of the surviving stoppage each handed-back stretch sits on decides who
        # owns it: before the stoppage the ball still belongs to whoever had it, after it the
        # ball belongs to whoever is about to restart.
        veto_runs = []
        for a, b in runs_of(pred_dead):
            kept = [(a + c, a + d) for c, d in runs_of(keep[a:b])]
            mid = kept[0] if kept else (b, b)
            for c, d in runs_of(pred_dead[a:b] & ~keep[a:b]):
                veto_runs.append((a + c, a + d, "pre" if a + d <= mid[0] else "post"))
        pred_dead = keep
    else:
        veto_runs = []

    # A sliver of play between two stoppages is almost never play: the ball is being placed, or
    # walked back, or simply out of sight between a foul and the free kick.
    if "island" in steps:
        for a, b in runs_of(~pred_dead):
            if a == 0 or b >= n:
                continue                      # the start or the end of the half
            span = (b - a) / float(FPS)
            if span >= ISLAND_SLOW_S:
                continue
            if span < ISLAND_S:
                pred_dead[a:b] = True
                continue
            seg = np.flatnonzero(reliable[a:b])
            if not seg.size:
                pred_dead[a:b] = True         # never seen: nothing says play restarted
                continue
            with np.errstate(invalid="ignore"):
                travel = float(np.nanmax(np.hypot(bx[a:b] - np.nanmean(bx[a:b]),
                                                  by[a:b] - np.nanmean(by[a:b]))))
            if np.isfinite(travel) and travel < ISLAND_TRAVEL_M:
                pred_dead[a:b] = True

    attacks_right = home_attacks_right(X, is_home)
    sh, sa, nh, na = knn_scores(bx, by, X, Y, is_home)
    tot = sh + sa
    with np.errstate(invalid="ignore"):
        knn_margin = np.where(tot > 0, (sh - sa) / tot, np.nan)
    knn_pick = np.where(np.isfinite(knn_margin), np.where(knn_margin >= 0, "H", "A"), "")
    # A ball that has left the field of play cannot give possession to the player standing
    # nearest it. A reviewer watching J03WOY at 4:16: "model is checking the nearest player so
    # it changed the possession, but actually the ball went out -- unless the player received
    # the ball, do not change possession, especially when the ball went out."
    if "outside" in steps:
        with np.errstate(invalid="ignore"):
            inside = (bx >= 0) & (bx <= PITCH_X) & (by >= 0) & (by <= PITCH_Y)
        knn_pick = np.where(inside, knn_pick, "")

    # how far the ball travels over the next MOVE_WIN_S, used as "it has been kicked"
    w = max(1, int(MOVE_WIN_S * FPS))
    fx, fy = np.full(n, np.nan), np.full(n, np.nan)
    fx[:n - w], fy[:n - w] = bx[w:], by[w:]
    move = np.sqrt((fx - bx) ** 2 + (fy - by) ** 2)

    pace, pace_drop, pace_base = team_pace(X, Y, fnum, pred_dead)
    if "bridge" in steps:
        _, _, nhb, nab = knn_scores(bxb, byb, X, Y, is_home)
    else:
        nhb, nab = nh, na
    ball_speed, flight_mask, flight_team, flight_firm = ball_flight(bxb, byb, seen_plus,
                                                                    nhb, nab)
    if "flight" not in steps:
        flight_mask = np.zeros(n, dtype=bool)
        flight_firm = np.zeros(n, dtype=bool)
        flight_team = np.full(n, "", dtype=object)

    # --- fields we are asked to make visible
    last_control = np.full(n, "", dtype=object)
    restart_team = np.full(n, "", dtype=object)
    restart_kind = np.full(n, "", dtype=object)
    first_ball = np.full(n, -1, dtype=np.int64)
    resume_at = np.full(n, -1, dtype=np.int64)
    switch_reason = np.full(n, "", dtype=object)
    uncertain = np.zeros(n, dtype=bool)
    in_override = np.zeros(n, dtype=bool)
    flight_held = np.zeros(n, dtype=bool)

    dead_runs = [(a, b) for a, b in runs_of(pred_dead)]
    # The whistle, re-read off the players. Each stoppage may only move inside its own gap, so
    # two of them can never swap order or swallow the possession between them.
    whistle_shift = np.zeros(n, dtype=float)
    whistle_tail = []
    if "whistle" in steps:
        moved, floor = [], 0
        for a, b in dead_runs:
            a2 = move_whistle(a, b, floor, pace_drop, pace_base)
            whistle_shift[min(a, a2):max(a, a2) + 1] = (a2 - a) / float(FPS)
            if a2 > a:
                whistle_tail.append((a, a2))
            moved.append((a2, b))
            floor = b
        dead_runs = moved

    new_dead = pred_dead.copy()
    # restart information, keyed by the alive run that follows each stoppage
    restart_at = {}

    for a, b in dead_runs:
        new_dead[a:b] = True                     # the whistle may have moved earlier than stage 2
        # 1. who had it before the whistle: majority of the decoded team over the second before
        w0 = max(0, a - FPS)
        prev = home_dec[w0:a][~pred_dead[w0:a]]
        last = "" if prev.size == 0 else ("H" if prev.mean() >= 0.5 else "A")
        # The decoded team is what the model believed frame by frame. The last TOUCH is what
        # actually happened to the ball, and when the ball has gone out of play that is what
        # decides who restarts -- a pass deflected out by a defender is the attacker's throw.
        if "lasttouch" in steps:
            seg = flight_team[max(0, a - int(LAST_TOUCH_S * FPS)):a]
            hit = np.flatnonzero(seg != "")
            if hit.size:
                last = str(seg[hit[-1]])

        # 2. where the ball is resting: the median reliable position in the last second of the
        #    stoppage, which is where it is placed for the restart
        lo = max(a, b - FPS)
        rest = np.flatnonzero(reliable[lo:b]) + lo
        if rest.size == 0:                       # nothing reliable late; try the whole stoppage
            rest = np.flatnonzero(reliable[a:b]) + a
        if rest.size:
            rx, ry = float(np.median(bx[rest])), float(np.median(by[rest]))
        else:
            rx = ry = np.nan
        geo = None
        if np.isfinite(rx):
            geo = end_geometry(X, Y, is_home, max(a, b - FPS), b, rx > PITCH_X / 2)
        kind, rule_team = classify_restart(rx, ry, attacks_right, geo)
        # A corner is the one restart where the Laws beat the classifier, and only when the
        # picture is unambiguous: a taker standing on the flag and a loaded box. Measured over
        # the five development matches, that gate is right 91% of the time against the model's
        # 85%; without it the rule drops to 67% and costs more than it wins.
        strict_corner = bool(kind == "corner" and geo and np.isfinite(geo[0])
                             and geo[0] <= 3.5 and geo[2] >= 8)

        # 2b. the head-count. Eleven a side, so a player standing OFF the pitch beside the ball
        #     is not out of position -- he is the one about to put it back into play.
        head_team = ""
        if "headcount" in steps and np.isfinite(rx):
            with np.errstate(invalid="ignore"):
                px, py = X[lo:b], Y[lo:b]
                off = np.isfinite(px) & np.isfinite(py) & (
                    (px < 0) | (px > PITCH_X) | (py < 0) | (py > PITCH_Y))
                dd = np.where(off, np.sqrt((px - rx) ** 2 + (py - ry) ** 2), np.nan)
            if np.isfinite(dd).any():
                flat = int(np.nanargmin(np.where(np.isfinite(dd), dd, np.inf)))
                _, who = np.unravel_index(flat, dd.shape)
                if dd.reshape(-1)[flat] <= OFF_PITCH_R:
                    head_team = "H" if is_home[who] else "A"

        # 3. when does play actually resume: the ball kicked and clearly moving, and for a throw,
        #    back inside the pitch. Searched only within a window of where stage 2 put the edge.
        half = int(RESUME_WIN_S * FPS)
        s0, s1 = max(a + 1, b - half), min(n, b + half)
        resume = -1
        for i in range(s0, s1):
            if not reliable[i] or not np.isfinite(move[i]):
                continue
            if move[i] < MOVE_MIN_M:
                continue
            if kind == "throw_in" and not (0 <= bx[i] <= PITCH_X and 0 <= by[i] <= PITCH_Y):
                continue
            resume = i
            break

        # A LATER move extends the stoppage, and the evidence for it is weaker than it looks:
        # "the ball was not seen to move" is also what a masked ball looks like. Only accept it
        # when the ball was visible the whole way from stage 2's edge -- so we genuinely watched
        # it not move -- and stage 2 still believes the ball is dead there. Without this gate
        # nine stoppages that were timed to within half a second drifted past the one-second
        # tolerance, every one of them where the ball was hidden at the real restart.
        if late_gate and resume > b:
            seen_through = bool(reliable[b:resume].all()) if resume > b else True
            still_dead = float((p_dead[b:resume] >= 0.5).mean()) if resume > b else 1.0
            ok = (seen_through and still_dead >= 0.5) if LATE_GATE_MODE == "and"                 else (seen_through or still_dead >= 0.5)
            if not ok:
                resume = -1
        fb_i = int(np.flatnonzero(reliable[b:min(n, b + 3 * FPS)])[0] + b) \
            if np.any(reliable[b:min(n, b + 3 * FPS)]) else -1
        fb = int(fnum[fb_i]) if fb_i >= 0 else -1

        # 4. the taker. The head-count beats everything else where it speaks: 95% against the
        #    model's 85% on the same restarts, and 16 of the 17 disagreements.
        team = head_team or rule_team
        if not team:
            j = resume if resume >= 0 else fb_i
            if j >= 0 and np.isfinite(knn_margin[j]):
                team = "H" if knn_margin[j] >= 0 else "A"
            elif kind == "throw_in" and last:
                team = "A" if last == "H" else "H"   # the ball went out off the side in control
        if kind == "throw_in" and last and "lasttouch" in steps:
            team = "A" if last == "H" else "H"       # whoever touched it last gave it away
        if kind in ("corner", "goal_kick") and team and resume >= 0 \
                and np.isfinite(knn_margin[resume]) \
                and (("H" if knn_margin[resume] >= 0 else "A") != team) \
                and abs(knn_margin[resume]) > 0.5:
            uncertain[a:b] = True                 # the Law and the geometry disagree loudly

        # 5. move the restart edge onto the evidence, within the window
        end = b
        if align and ("align" in steps) and resume >= 0:
            end = max(a + 1, min(n, resume))
            if end > b:
                new_dead[b:end] = True
            elif end < b:
                new_dead[end:b] = False
        else:
            uncertain[a:b] = True                 # no reliable ball: the edge is stage 2's guess

        for t0, t1 in whistle_tail:
            if t1 == a:
                new_dead[t0:t1] = False          # ... or later, which gives frames back to play
        last_control[a:end] = last
        restart_team[a:end] = team
        restart_kind[a:end] = kind
        first_ball[a:end] = fb
        resume_at[a:end] = int(fnum[min(end, n - 1)])
        restart_at[end] = (team, kind, last, strict_corner, max(0, b - end))

    # ---------------------------------------------------------------- the team sequence itself
    refined = home_dec.copy()
    persist = max(1, int(PERSIST_S * FPS))
    confirm = max(1, int(CONFIRM_S * FPS))
    release = max(1, int(RELEASE_S * FPS))
    min_spell = max(1, int(MIN_SPELL_S * FPS))
    gap_min = max(1, int(GAP_MIN_S * FPS))
    restart_max = int(RESTART_MAX_S * FPS)
    corner_hold = int(CORNER_HOLD_S * FPS)

    def supported(i, who):
        """Does the evidence around frame i support `who` having it, for long enough?"""
        hi = min(n, i + persist)
        votes = knn_pick[i:hi][reliable[i:hi]]
        if votes.size < max(3, int(PERSIST_FRAC * persist * 0.5)):
            return False
        if (votes == who).mean() < PERSIST_FRAC:
            return False
        near = nh[i:hi] if who == "H" else na[i:hi]
        with np.errstate(invalid="ignore"):
            close = np.nanmean(near <= CONTROL_R)
        return bool(np.isfinite(close) and close >= 0.5)

    for c, d in runs_of(~new_dead):
        info = restart_at.get(c)
        # A restart team is only imposed for a strict corner. Imposing the throw-in taker as
        # well -- naming him by who touched the ball last, which is what the Laws do -- was
        # measured and is much worse: 11 cases fixed against 120 created, because the throw is
        # contested within a second and the decoder reads that better than the Law does.
        # The head-count NAMES the taker, and naming is all it does. Imposing that team on
        # the frames after the restart was measured three ways -- at every restart it fires on
        # (16 fixed / 109 created), and only where the model is blind at the restart (4 / 32) --
        # and every one of them is worse. The throw is contested within a second and the
        # per-frame model reads that contest better than any rule about who took it.
        hold_ok = bool(info) and (("restart" in steps)
                                  or ("corner" in steps and info[3]))
        cur = ""
        if info and info[0] and hold_ok:
            cur = info[0]
            span = corner_hold if info[3] and "restart" not in steps else restart_max
            hold_until = min(d, c + span)
            why = "restart: {} taken by {}".format(info[1], "home" if cur == "H" else "away")
        else:
            seg = home_dec[c:d]
            cur = "H" if (seg.size and seg.mean() >= 0.5) else "A"
            hold_until = c
            why = ""
        if why:
            switch_reason[c] = why

        i = c
        while i < d:
            # is this frame inside a region we are entitled to override?
            in_restart = i < hold_until
            gap_end = i
            in_gap = False
            if not reliable[i]:
                j = i
                while j < d and not reliable[j]:
                    j += 1
                if (j - i) >= gap_min and ("gap" in steps):
                    in_gap = True
                    gap_end = min(d, j + confirm)

            if mode == "full":
                in_restart = True                       # k-NN drives every reliable frame

            if not (in_restart or in_gap):
                # The ball is in the air from a known kicker: the pass belongs to the side that
                # struck it until somebody else takes it, whatever the geometry says about who is
                # standing closest to it right now.
                keep = cur in ("", flight_team[i])
                if (flight_mask[i] and flight_team[i]
                        and (keep or (FLIGHT_MODE == "both" and flight_firm[i]))):
                    if (home_dec[i] != (flight_team[i] == "H")
                            and (i == c or not flight_held[i - 1])):
                        who = "home" if flight_team[i] == "H" else "away"
                        switch_reason[i] = (
                            "{} with {}: {} played the ball at {:.1f} m and it is travelling at "
                            "{:.0f} m/s, so the nearest player from here on is the one it is "
                            "going past".format(
                                "held" if cur in ("", flight_team[i]) else "switch", who, who,
                                (nh[i] if flight_team[i] == "H" else na[i])
                                if np.isfinite(nh[i] if flight_team[i] == "H" else na[i])
                                else 0.0,
                                ball_speed[i] if np.isfinite(ball_speed[i]) else 0.0))
                    cur = flight_team[i]
                    flight_held[i] = True
                    in_override[i] = True
                    refined[i] = (cur == "H")
                    i += 1
                    continue
                want = "H" if home_dec[i] else "A"
                if cur == "" or want == cur:
                    cur = want                          # the model has it; follow it
                else:
                    seg = home_dec[i:min(d, i + release)]
                    if seg.size and np.all(seg == (want == "H")):
                        cur = want                      # held its view long enough to take over
                refined[i] = (cur == "H")
                i += 1
                continue

            stop = d if mode == "full" else max(gap_end, hold_until if in_restart else i + 1)
            while i < min(stop, d):
                in_override[i] = True
                if not reliable[i]:
                    uncertain[i] = True
                else:
                    other = "A" if cur == "H" else "H"
                    if (np.isfinite(knn_margin[i]) and abs(knn_margin[i]) >= MARGIN_MIN
                            and (("H" if knn_margin[i] >= 0 else "A") == other)
                            and supported(i, other)):
                        switch_reason[i] = ("switch to {}: nearest {:.1f} m vs {:.1f} m, held "
                                            "{:.1f} s, ball moved {:.1f} m".format(
                                                "home" if other == "H" else "away",
                                                nh[i] if other == "H" else na[i],
                                                na[i] if other == "H" else nh[i],
                                                PERSIST_S,
                                                move[i] if np.isfinite(move[i]) else 0.0))
                        cur = other
                        if in_restart:
                            hold_until = i          # control settled; the model takes over again
                    elif np.isfinite(knn_margin[i]) and abs(knn_margin[i]) < MARGIN_MIN:
                        uncertain[i] = True
                refined[i] = (cur == "H")
                i += 1

    # Where the restart edge moved earlier, the frames it freed were decided by a model that
    # thought the ball was dead. They belong to the possession that follows, not to whatever the
    # probability happened to be doing inside a stoppage.
    for c, d in runs_of(~new_dead):
        info = restart_at.get(c)
        fill = info[4] if info else 0
        if fill and c + fill < d:
            refined[c:c + fill] = refined[c + fill]

    for t0, t1 in whistle_tail:
        if t0 > 0 and t1 > t0:
            refined[t0:t1] = refined[t0 - 1]

    # Frames the moving ball handed back to the match. Stage 2 called them dead because it could
    # not see the ball, so the per-frame classifier has nothing to go on there either and asking
    # it who has possession invents a turnover: measured, letting it decide cost 43 new
    # WRONG_TEAM and 35 new FALSE_CHANGE cases. Whoever had the ball keeps it, unless the ball
    # is in flight with a known kicker, which is better evidence than either.
    # The vote takes the ball off the model where somebody is standing on it.
    if "takeover" in steps:
        base = (~new_dead & reliable & np.isfinite(knn_margin)
                & (np.abs(knn_margin) >= TAKE_MARGIN) & ~flight_held
                & np.isfinite(np.fmin(nh, na)) & (np.fmin(nh, na) <= TAKE_R))
        need = max(1, int(TAKE_S * FPS))
        for who in (True, False):
            want = base & (knn_pick == ("H" if who else "A")) & (refined != who)
            for a, b in runs_of(want):
                if b - a >= need:
                    refined[a:b] = who

    # A switch to a team with nobody near the ball is not a change of possession.
    if "reach" in steps:
        spans = sorted([(a2, b2) for a2, b2 in runs_of(refined)]
                       + [(a2, b2) for a2, b2 in runs_of(~refined)])
        # in time order, so that a blocked switch merges into whatever now precedes it
        for a2, b2 in spans:
            if a2 == 0 or new_dead[a2 - 1]:
                continue                      # the first possession, or a restart: not a switch
            who = bool(refined[a2])
            w0, w1 = max(0, a2 - FPS // 2), min(n, a2 + FPS // 2)
            mine = (nh if who else na)[w0:w1]
            if not np.isfinite(mine).any():
                continue                      # no ball to measure against: say nothing
            if float(np.nanmin(mine)) <= REACH_R:
                continue
            if REACH_VOTE:
                v = knn_pick[w0:w1]
                v = v[v != ""]
                if not v.size or (v == ("H" if who else "A")).mean() >= 0.5:
                    continue                  # the ball itself backs the switch; leave it
            refined[a2:b2] = refined[a2 - 1]

    for t0, t1, side in veto_runs:
        if t1 <= t0:
            continue
        src = t0 - 1 if side == "pre" else min(t1, n - 1)
        if src < 0:
            continue
        hold = ~flight_held[t0:t1]
        refined[t0:t1] = np.where(hold, refined[src], refined[t0:t1])

    for c, d in runs_of(~new_dead):
        spans = ([(a2 + c, b2 + c) for a2, b2 in runs_of(refined[c:d])]
                 + [(a2 + c, b2 + c) for a2, b2 in runs_of(~refined[c:d])])
        spans = sorted(spans)
        for a2, b2 in spans:
            if (b2 - a2) >= min_spell:
                continue
            # A pass that was in the air for half a second is a possession, not a wobble: the
            # length floor is there to suppress a drifting probability, and a struck ball with a
            # known kicker is the opposite of that.
            if flight_held[a2:b2].mean() >= 0.5:
                continue
            if a2 > c:
                refined[a2:b2] = refined[a2 - 1]        # merge back into what came before
            elif b2 < d:
                refined[a2:b2] = refined[b2]            # a sliver at a restart: adopt what
                                                        # the possession actually becomes

    lab = np.where(new_dead, "D", np.where(refined, "H", "A"))
    return {
        "label_refined": lab,
        "pred_dead_refined": new_dead,
        "home_refined": refined,
        "last_control": last_control,
        "restart_team": restart_team,
        "restart_kind": restart_kind,
        "first_ball_frame": first_ball,
        "resume_frame": resume_at,
        "knn_home": sh, "knn_away": sa,
        "knn_nearest_home_m": nh, "knn_nearest_away_m": na,
        "knn_margin": knn_margin,
        "ball_move_m": move,
        "ball_speed_ms": ball_speed,
        "team_pace_ms": pace,
        "team_pace_base": np.full(n, pace_base),
        "whistle_shift_s": whistle_shift,
        "ball_at_rest": still,
        "flight_team": flight_team,
        "in_flight": flight_held,
        "ball_reliable": reliable,
        "switch_reason": switch_reason,
        "uncertain": uncertain,
        "in_override": in_override,
    }


def load_match(match):
    f = pd.read_parquet(os.path.join(PROC, match, "frames.parquet"),
                        columns=["period", "frame_native", "time_s", "ball_x", "ball_y",
                                 "ball_observed", "label", "label_valid"])
    p = pd.read_parquet(os.path.join(S3PRED, match + ".parquet"))
    d = f.merge(p[["period", "frame_native", "p_home", "pred_dead", "home_end_to_end",
                   "ball_observed_masked"]], on=["period", "frame_native"], how="inner")
    # stage 2's own per-frame belief, needed to gate the restart edge: the ball going missing
    # is not evidence that play is still stopped, but stage 2 saying so is.
    s2 = pd.read_parquet(os.path.join(S2PRED, match + ".parquet"),
                         columns=["period", "frame_native", "prob"])
    d = d.merge(s2, on=["period", "frame_native"], how="left")
    d["prob"] = d["prob"].fillna(0.5)

    d = d[d["label_valid"].astype(bool)].reset_index(drop=True)
    d = d.rename(columns={"home_end_to_end": "home_base"})
    d["home_base"] = d["home_base"].fillna(False).astype(bool)
    d["pred_dead"] = d["pred_dead"].fillna(False).astype(bool)
    players = pd.read_parquet(os.path.join(PROC, match, "players.parquet"),
                              columns=["period", "frame_native", "team", "slot", "x", "y"])
    return d, players


def run_match(match, mode, align, steps=("align", "corner"), late_gate=True):
    d, players = load_match(match)
    parts = []
    for period, g in d.groupby("period", sort=True):
        g = g.reset_index(drop=True)
        wp = wide_positions(players, period)
        if wp is None:
            continue
        pf, X, Y, is_home = wp
        idx = np.searchsorted(pf, g["frame_native"].to_numpy())
        idx = np.clip(idx, 0, len(pf) - 1)
        ok = pf[idx] == g["frame_native"].to_numpy()
        Xg = np.where(ok[:, None], X[idx], np.nan)
        Yg = np.where(ok[:, None], Y[idx], np.nan)
        out = refine_period(g, Xg, Yg, is_home, mode=mode, align=align, steps=steps,
                            late_gate=late_gate)
        res = g[["period", "frame_native", "time_s", "label", "p_home",
                 "ball_observed_masked"]].copy()
        res["pred_dead_base"] = g["pred_dead"].to_numpy()
        res["home_base"] = g["home_base"].to_numpy()
        res["label_base"] = np.where(res["pred_dead_base"], "D",
                                     np.where(res["home_base"], "H", "A"))
        for k, v in out.items():
            res[k] = v
        parts.append(res)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------------- measurement
def spell_scores(pm, truth, pred, periods):
    gt, pr = [], []
    for p in sorted(pd.unique(periods)):
        m = periods == p
        if gt:
            gt.append(np.array(["X"]))
            pr.append(np.array(["X"]))
        gt.append(np.asarray(truth[m], dtype=object))
        pr.append(np.asarray(pred[m], dtype=object))
    r = pm.evaluate(pm.label_codes(np.concatenate(gt)), pm.label_codes(np.concatenate(pr)))
    return r["spell_f1_mean"], r["macro_f1"]


def turnovers(seq):
    seq = np.asarray(seq, dtype=object)
    return np.flatnonzero((seq[1:] != seq[:-1]) & (seq[1:] != "D") & (seq[:-1] != "D")) + 1


def restart_report(truth, pred, periods):
    """Restart-edge timing, false and missed switches, and the team right after each restart."""
    out = {"restart_err_s": [], "false_switches": 0, "missed_switches": 0,
           "restart_team_right": 0, "restart_team_total": 0}
    tol = int(1.0 * FPS)
    for p in sorted(pd.unique(periods)):
        m = periods == p
        t, q = np.asarray(truth[m], dtype=object), np.asarray(pred[m], dtype=object)
        n = len(t)
        td, qd = (t == "D"), (q == "D")
        qspans = list(runs_of(qd))
        for a, b in runs_of(td):
            best = max(((min(b, d2) - max(a, c2), c2, d2) for c2, d2 in qspans
                        if min(b, d2) > max(a, c2)), default=None)
            if best:
                out["restart_err_s"].append((best[2] - b) / FPS)
            # the team in the two seconds after the restart, which is what a corner turns on
            if b < n and t[b] != "D":
                k = min(n, b + 2 * FPS)
                want = t[b]
                got = q[b:k]
                got = got[got != "D"]
                if got.size:
                    out["restart_team_total"] += 1
                    out["restart_team_right"] += int((got == want).mean() >= 0.5)
        tc, pc = turnovers(t), turnovers(q)
        for i in pc:
            if not len(tc) or np.min(np.abs(tc - i)) > tol:
                out["false_switches"] += 1
        for i in tc:
            if not len(pc) or np.min(np.abs(pc - i)) > tol:
                out["missed_switches"] += 1
    return out


def case_list(df, which):
    """Failure cases as (period, kind, start, end), using the review tool's own definitions."""
    sys.path.insert(0, os.path.join(HERE, "..", "review"))
    from export_cases import find_cases  # noqa: E402
    out = []
    for period, g in df.groupby("period", sort=True):
        g = g.reset_index(drop=True)
        sub = pd.DataFrame({
            "label": g["label"].to_numpy(),
            "pred_hsmm": g["pred_dead_base" if which == "base" else "pred_dead_refined"],
            "home_end_to_end": g["home_base" if which == "base" else "home_refined"],
            "p_home": g["p_home"].to_numpy(),
        })
        for kind, a, b, _x in find_cases(sub):
            out.append((period, kind, int(a), int(b)))
    return out


def diff_cases(before, after):
    """Which individual cases went away, and which are new.

    Net counts hide the trade. A change that fixes forty cases and breaks forty looks like
    nothing at all in a total, and is not nothing: it is a different model with the same score.
    Two cases are the same case when they are the same kind, in the same period, and overlap.
    """
    fixed, new, kept = {}, {}, {}
    by_after, by_before = {}, {}
    for p, k, a, b in after:
        by_after.setdefault((p, k), []).append((a, b))
    for p, k, a, b in before:
        by_before.setdefault((p, k), []).append((a, b))

    def overlaps(span, others):
        a, b = span
        return any(min(b, d) > max(a, c) for c, d in others)

    for (p, k), spans in by_before.items():
        others = by_after.get((p, k), [])
        for sp in spans:
            if overlaps(sp, others):
                kept[k] = kept.get(k, 0) + 1
            else:
                fixed[k] = fixed.get(k, 0) + 1
    for (p, k), spans in by_after.items():
        others = by_before.get((p, k), [])
        for sp in spans:
            if not overlaps(sp, others):
                new[k] = new.get(k, 0) + 1
    return fixed, new, kept


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="carry", choices=["carry", "full"],
                    help="carry = override only at restarts and through ball gaps (default); "
                         "full = let the k-NN drive every reliable frame")
    ap.add_argument("--no-align", action="store_true",
                    help="leave the restart edge where stage 2 put it")
    ap.add_argument("--late-gate-mode", default="or", choices=["and", "or"],
                    help="and = a later restart edge needs both the visible ball and stage 2; "
                         "or = either on its own")
    ap.add_argument("--no-late-gate", action="store_true",
                    help="allow the restart edge to move later even where the ball was hidden")
    ap.add_argument("--flight-mode", default="both", choices=["hold", "both"],
                    help="both = a detected touch both takes possession and keeps it; hold = a "
                         "ball in flight may only keep possession where it already is "
                         "(measured: 83 cases fixed against 325)")
    ap.add_argument("--corner-hold", type=float, default=None,
                    help="seconds a corner's taker is held before the model takes over")
    ap.add_argument("--resume-win", type=float, default=None,
                    help="seconds the restart edge may be moved from where stage 2 put it")
    ap.add_argument("--speed-win", type=float, default=None,
                    help="seconds every ball-speed test measures displacement over. 0 = one "
                         "frame difference, which reports detection jitter as speed on a "
                         "provider that does not freeze the ball while play is stopped")
    ap.add_argument("--pace-weight", type=float, default=None,
                    help="how much of the way to move the whistle onto the steepest fall "
                         "in team pace (0 = keep stage 2's edge, 1 = take the players' "
                         "word)")
    ap.add_argument("--parts", default="align,corner,flight,whistle,still,lasttouch,inplay,"
                                       "outside,headcount,bridge,takeover,outdead,island",
                    help="which hardening steps to apply: align (move the restart edge onto the "
                         "evidence), corner (hold the restart team for unambiguous corners), "
                         "flight (hold possession with whoever struck a ball that is in the "
                         "air), whistle (re-read the whistle off the players slowing to a "
                         "walk), still (a visible ball lying still with nobody near it is a "
                         "stoppage), lasttouch (name the last player to touch the ball before "
                         "a stoppage, and the throw-in it gives away), inplay (a visible ball "
                         "moving inside the pitch means play is going on), outside (a ball that "
                         "has left the pitch gives possession to nobody), headcount (eleven a "
                         "side, so the player standing off the pitch beside the ball is "
                         "the one taking the restart), bridge (join up short gaps in the "
                         "ball track for the rules that follow the ball), takeover (let "
                         "the nearest-player vote take the ball off the model when "
                         "somebody is standing on it), reach (refuse a switch to a team "
                         "with nobody near the ball -- measured worse), outdead (a ball lying "
                         "outside the lines is nobody's), island (a sliver of play between "
                         "two stoppages is not play), restart (hold "
                         "the team for every restart -- measured worse), gap (carry "
                         "through ball gaps -- measured worse)")
    ap.add_argument("--apply", action="store_true", help="write predictions_refined/")
    ap.add_argument("--compare", action="store_true", help="score before and after")
    ap.add_argument("--out", default=OUTDIR)
    args = ap.parse_args()

    globals()["LATE_GATE_MODE"] = args.late_gate_mode
    globals()["FLIGHT_MODE"] = args.flight_mode
    if args.corner_hold is not None:
        globals()["CORNER_HOLD_S"] = args.corner_hold
    if args.resume_win is not None:
        globals()["RESUME_WIN_S"] = args.resume_win
    if args.pace_weight is not None:
        globals()["PACE_W"] = args.pace_weight
    if args.speed_win is not None:
        globals()["SPEED_WIN_S"] = args.speed_win
    pm = load_scorer() if args.compare else None
    rows, before_cases, after_cases = [], [], []
    for match in DEV:
        df = run_match(match, args.mode, not args.no_align,
                       steps=tuple(x.strip() for x in args.parts.split(",") if x.strip()),
                       late_gate=not args.no_late_gate)
        if args.apply:
            os.makedirs(args.out, exist_ok=True)
            df.to_parquet(os.path.join(args.out, match + ".parquet"), index=False)
        if not args.compare:
            print("{}  {} frames refined".format(match.replace("DFL-MAT-", ""), len(df)))
            continue

        truth = df["label"].to_numpy()
        per = df["period"].to_numpy()
        base, ref = df["label_base"].to_numpy(), df["label_refined"].to_numpy()
        sb, mb = spell_scores(pm, truth, base, per)
        sr, mr = spell_scores(pm, truth, ref, per)
        rb, rr = restart_report(truth, base, per), restart_report(truth, ref, per)
        tag = match.replace("DFL-MAT-", "")
        before_cases += [(tag + "-" + str(p), k, a, b) for p, k, a, b in case_list(df, "base")]
        after_cases += [(tag + "-" + str(p), k, a, b) for p, k, a, b in case_list(df, "refined")]
        rows.append({
            "match": match.replace("DFL-MAT-", ""),
            "spellF1_base": sb, "spellF1_refined": sr,
            "macroF1_base": mb, "macroF1_refined": mr,
            "restart_err_base": float(np.mean(np.abs(rb["restart_err_s"]))),
            "restart_err_refined": float(np.mean(np.abs(rr["restart_err_s"]))),
            "restart_team_base": rb["restart_team_right"],
            "restart_team_refined": rr["restart_team_right"],
            "restart_team_total": rb["restart_team_total"],
            "false_sw_base": rb["false_switches"], "false_sw_refined": rr["false_switches"],
            "missed_sw_base": rb["missed_switches"], "missed_sw_refined": rr["missed_switches"],
        })
        print("  {}  spellF1 {:.3f} -> {:.3f}   restart team {}/{} -> {}/{}".format(
            rows[-1]["match"], sb, sr, rb["restart_team_right"], rb["restart_team_total"],
            rr["restart_team_right"], rr["restart_team_total"]))

    if args.compare:
        out = pd.DataFrame(rows)
        path = os.path.join(HERE, "results_restart_{}.csv".format(args.mode))
        out.to_csv(path, index=False)
        print("\n=== means over 5 development matches (mode={}, parts={}) ===".format(
            args.mode, args.parts))
        for a, b, name in (("spellF1_base", "spellF1_refined", "spellF1 end-to-end"),
                           ("macroF1_base", "macroF1_refined", "macroF1 end-to-end"),
                           ("restart_err_base", "restart_err_refined", "restart edge |err| s")):
            print("  {:<26} {:.3f}  ->  {:.3f}   ({:+.3f})".format(
                name, out[a].mean(), out[b].mean(), out[b].mean() - out[a].mean()))
        for a, b, name in (("restart_team_base", "restart_team_refined", "restart team right"),
                           ("false_sw_base", "false_sw_refined", "false switches"),
                           ("missed_sw_base", "missed_sw_refined", "missed switches")):
            print("  {:<26} {:>6}  ->  {:>6}   ({:+d})".format(
                name, int(out[a].sum()), int(out[b].sum()), int(out[b].sum() - out[a].sum())))
        print("  {:<26} {:>6}".format("restarts judged", int(out["restart_team_total"].sum())))
        fixed, new, kept = diff_cases(before_cases, after_cases)
        print("")
        print("=== individual failure cases: fixed, new, unchanged ===")
        print("  {:<16}{:>8}{:>8}{:>10}{:>10}".format(
            "kind", "fixed", "new", "net", "unchanged"))
        tf = tn = 0
        for k in sorted(set(fixed) | set(new) | set(kept)):
            f, nw = fixed.get(k, 0), new.get(k, 0)
            tf += f
            tn += nw
            print("  {:<16}{:>8}{:>8}{:>+10}{:>10}".format(k, f, nw, f - nw, kept.get(k, 0)))
        print("  {:<16}{:>8}{:>8}{:>+10}{:>10}".format(
            "TOTAL", tf, tn, tf - tn, sum(kept.values())))
        print("  {} cases before, {} after".format(len(before_cases), len(after_cases)))
        print("\nwrote " + path)


if __name__ == "__main__":
    main()
