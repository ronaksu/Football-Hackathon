"""The canonical representation every external source is converted into.

One schema, so that a model trained on Bundesliga TRACAB data and a model trained on Metrica
sample data see the same column names, the same units and the same honesty about what is missing.

DESIGN RULES, and why each one is here
--------------------------------------
1. Native timing is preserved. `frame_native` and `time_s` are whatever the source said, and
   `fps` records its real rate. SkillCorner is 10 fps and is NOT upsampled: resampling a 10 fps
   observation to 25 fps invents 15 observations a second that nobody made.

2. Nothing is interpolated across a period boundary. Halves are separate sequences.

3. Answer fields are kept away from predictor fields. `label*` columns are the thing to be
   predicted; everything else is input. Mixing them is how leakage happens.

4. Player identity is positional, never nominal. The competition gives 46 anonymous track slots
   that start, stop and swap, so a feature that needs "player 7" cannot be computed at
   deployment. Slots here are indices into a team, nothing more.

5. Absence is recorded, never filled. A missing ball is `ball_observed = False` with NaN
   coordinates. It is not interpolated, and it is not silently turned into a dead ball.

COLUMNS
-------
frames.parquet - one row per frame of one match

  identity        source, match_id, period, frame_native, time_s, fps
  ball            ball_x, ball_y            normalised 0-1, NaN when unobserved
                  ball_observed             bool
                  ball_speed_mps            NaN where it cannot be computed
  players         n_home_observed, n_away_observed
  geometry        home_centroid_x/y, away_centroid_x/y, home_spread, away_spread
                  nearest_home_m, nearest_away_m     distance of closest player to the ball
                  nearest_margin_m                   nearest_away - nearest_home, signed
  quality         q_ball_missing, q_ball_motion_unreliable, q_label_missing,
                  q_ball_and_label_missing, q_source_disagreement, q_alignment_problem
  answers         label                     H / A / D, or null where unknown
                  label_valid               bool - False means "do not train or score on this"
                  label_source              provider_field | event_derived | none
                  label_quality             reference | strong | weak | ambiguous | none
  pending         model_uncertainty         always null at this stage, no model exists yet

players.parquet - long format, one row per observed player per frame

  source, match_id, period, frame_native, team ('home'/'away'), slot (int), x, y

UNITS
-----
Positions are normalised 0-1 on both axes, matching the competition ATD and the Metrica
convention: (0,0) top-left, (1,1) bottom-right, (0.5,0.5) the kick-off spot. Distances in metres
are produced by scaling with the match's own pitch size, which is recorded per match in the
match-level manifest rather than assumed to be 105x68 everywhere.
"""

LABEL_H = "H"
LABEL_A = "A"
LABEL_D = "D"

# Ordered weakest to strongest. Only `reference` and `strong` are eligible for independent
# evaluation; `weak` is for training signal; `ambiguous` is masked out entirely.
QUALITY_ORDER = ("none", "ambiguous", "weak", "strong", "reference")

FRAME_COLUMNS = [
    "source", "match_id", "period", "frame_native", "time_s", "fps",
    "ball_x", "ball_y", "ball_observed", "ball_speed_mps",
    "n_home_observed", "n_away_observed",
    "home_centroid_x", "home_centroid_y", "away_centroid_x", "away_centroid_y",
    "home_spread", "away_spread",
    "nearest_home_m", "nearest_away_m", "nearest_margin_m",
    "q_ball_missing", "q_ball_motion_unreliable", "q_label_missing",
    "q_ball_and_label_missing", "q_source_disagreement", "q_alignment_problem",
    "label", "label_valid", "label_source", "label_quality", "model_uncertainty",
]

PLAYER_COLUMNS = ["source", "match_id", "period", "frame_native", "team", "slot", "x", "y"]

# Provisional, and labelled as such everywhere they are used. These are inspection thresholds
# for flagging frames, NOT football truths and NOT model parameters. Each needs justifying
# before it influences anything that is scored.
PROVISIONAL = {
    # A struck ball rarely exceeds ~40 m/s. Above this, a frame-to-frame jump is far more likely
    # to be a detection error than a kick, so the motion is flagged unreliable.
    "max_plausible_ball_speed_mps": 40.0,
    # Below this the ball is effectively stationary. Used only to describe frames, never to
    # decide that play has stopped.
    "ball_still_speed_mps": 0.5,
}
