# Ball possession from tracking — technical document

**Team: Average Guys**

**More Than A Hack 2027 · FC Barcelona vs Athletic Club · tactical recording, 199,319 frames**

Per-frame possession produced by a two-stage model trained entirely on open-access tracking of
other matches, with a rule layer that reads the Laws of the Game off the coordinates.

---

## 1. The approach

### The metric decides the architecture

`spellF1` scores **possessions as intervals**, matched at IoU 0.3 to 0.7. `macroF1` scores
frames. The competition's own baseline shows how far apart those are: across its four smoothing
variants `macroF1` moves by 0.04 while `spellF1` moves by a factor of five. A model optimised
per frame will flatter itself and rank badly.

Possessions are also **short** — the public five-minute window has 17 of mean 14.2 s, and the
training matches average 3.9 s per possession. A per-frame error rate of 15% scattered across
intervals that brief does not degrade them, it shatters them.

So the output has to be intervals *by construction*. Every stage therefore ends in a **hidden
semi-Markov model decoded with Viterbi**, with explicit state-duration distributions fitted from
data. A plain HMM's dwell time is geometric, so its most likely state sequence is one frame long
— precisely the fragmentation that destroys `spellF1`. Explicit durations fix that without a flat
smoothing window, which is the published baseline's only tool and the reason it tops out at
0.2433.

### Three stages, each answering one question

```
46 anonymous track slots + the ball, per frame
          |
   STAGE 2   is play stopped?                 player geometry only, no ball feature at all
          |  LightGBM -> P(dead) -> HSMM -> dead/alive intervals
          |
   STAGE 3   which team is in control?        geometry + the ball when it is visible
          |  LightGBM -> P(home) -> HSMM inside each alive stretch -> H/A
          |
  HARDENING  what do the Laws say?            restart type from where the ball rests,
          |  the whistle re-read off the players, the taker found by counting heads
          |
      H / A / D per frame
```

### Why stage 2 is forbidden the ball

The single most consequential decision in the project. The competition loses the ball on **41%
of graded frames**; the training source never loses it. A detector allowed to see the ball would
learn the shortcut *"ball missing ⇒ dead ball"*, which is roughly true in some providers and
roughly half wrong here — and it is exactly the mistake the published baseline makes, which is
why its `D` class scores 0.32.

Built from the 22 player positions alone, stage 2 **cannot** make that mistake, and it transfers
whether the ball is visible or not. It also meant the hardest unknown in the project — how this
camera loses the ball — never blocked the dead-ball work at all.

### Why stage 3 is allowed the ball, but trained without it

When the ball is visible, the margin between the nearest home player and the nearest away player
settles possession on its own: **76.5%** accuracy from that one comparison. Discarding it would
be perverse. But a model trained where the ball is always present never learns a fallback, so
**every training match is masked at a different rate** — 0%, 25%, 41%, 45%, 65% — with the ball
hidden in contiguous stretches of about two seconds rather than per frame, because a camera loses
the ball for a second at a time and per-frame dropout is a problem nobody has. Every ball feature
is paired with an explicit `ball_observed` flag and set to `NaN` when absent, so LightGBM routes
"I cannot see the ball" down its own branch instead of guessing a number.

Measured cost of blindness, leave-one-match-out:

| ball hidden | team accuracy | end-to-end spellF1 |
|---|---|---|
| 0% | 0.879 | 0.487 |
| 20% | 0.866 | 0.463 |
| **41% (this competition)** | **0.850** | **0.430** |
| 60% | 0.838 | 0.414 |

**0.06 spellF1 between a clean view and the competition's rate.** The masking strategy works;
player geometry carries real signal on its own. The ball was not the main problem.

### Features

All collective, never per-player, because the 46 track slots are **tracklets, not people** —
they appear, vanish and swap identity, so no feature that needs "player 7" is computable at
deployment.

- **Stage 2, 169 features.** Everyone stops (speed means, medians, p90, fraction below walking /
  jogging / sprinting thresholds, per team and overall); the shape relaxes (centroids, spread,
  bounding boxes, how far the two teams' x-ranges overlap); they walk to a spot (players within
  3 m of a line, in a penalty area, near a corner, the halfway split that means a kick-off); and
  **players physically outside the pitch** — a thrower standing off the touchline is directly
  observable, and the clearest cue available for the quick throw-ins that dominate the hard cases.
  Eight core signals also carry rolling mean and standard deviation over 1 s, 3 s and 5 s, because
  a stoppage is not one odd frame, it is a state that holds. Rolling windows are computed inside
  each half separately.
- **Stage 3, 56 features.** Team shape normalised by **playing direction, inferred from the data
  alone** — over a period the side defending the left goal keeps its deepest player near x=0 —
  so no roles or identities are needed and the half-time flip is handled; ball position and the
  nearest-player margins; and 19 **pitch-control** features, the share of space each side would
  reach first, which is what carries possession through the frames where the ball is gone.

### The hardening layer

Neither classifier knows anything about restarts, and a restart is exactly where per-frame
geometry is least informative: both sides are packed into the box and the only thing that settles
possession is who takes the kick. The Laws settle it instead, and **the restart position says
which Law applies** — a ball on the corner arc is a corner for the side attacking that goal; a
ball on the six-yard line at the other end is a goal kick for the side defending it; a ball on
the touchline is a throw. None of that needs a label and none of it needs this match's data: it
is the Laws read off the coordinates.

Thirteen rules are implemented and all thirteen run on the training matches. **Twelve run on the
competition match**, because the guard described in §5 falsifies one of them on this ball track
and drops it. What the full set is worth on the training matches, leave-one-match-out:

| | before | after |
|---|---|---|
| spellF1 end to end | 0.438 | **0.519** |
| macroF1 end to end | 0.855 | **0.884** |
| restart edge, mean absolute error | 1.45 s | **1.04 s** |
| restart team correct | 430 / 460 | **445 / 460** |
| missed possession changes | 1,206 | **1,014** |
| catalogued failure cases | 2,943 | **2,561** |

Three further rules were implemented, measured and **rejected** because they made the score
worse: `reach` (refuse a switch to a team with nobody near the ball), `restart` (hold the team
through every restart) and `gap` (carry possession through every ball gap). They remain in the
code behind flags, with their measurements, because a rejected rule that is documented is more
useful than one that was never tried.

---

## 2. Which assets were used, and which were dropped

| asset | used? | why |
|---|---|---|
| **tactical ATD** — positions, ball, 46 slots | **yes, the only input** | It is on the submission's own timeline, one record per frame, and it is the only asset that carries where everybody is. `video_frame = frame_id - 1`, applied once, in one file |
| **published half boundaries** | **yes** | `DATA.md` publishes them. They are boundaries, not answers: they say where the match is not being played and reveal no label. Used to set `period` exactly rather than inferring it, and to abstain with `X` outside them |
| **tactical homography** | no | It maps pitch coordinates to pixels. Every model here works in pitch coordinates and never looks at a pixel, so there is nothing for it to do. It would be essential for a video model |
| **tactical video** | no | No pixel model was built. Two reasons: the time would have gone into a second perception system rather than into possession, and the ATD already *is* the output of a perception system over the same footage |
| **broadcast ATD, video and homography** | no | **Not frame-aligned to tactical, and no mapping table is provided.** 34% of its frames have no usable homography — replays, close-ups, crowd shots, in 122 stretches, the longest over a minute and a half. Aligning 164,488 broadcast frames to 199,319 tactical ones is its own research problem, and getting it wrong would corrupt every frame it touched rather than just the hard ones |
| **Smart Tagging** — 638 automatic phases | **considered, not integrated** | Honestly: a prior worth having that there was no time to use safely. Three reasons it was deprioritised. It is **automatic and unreviewed**, generated from the same video and tracking we already have, so it carries its own false positives. Its `Team` tag is `N/A` on 36 instances including **every single throw-in** — exactly the restart where knowing the side is worth the most. And its intervals are **phases, not possessions**: a Build Up can span one possession, stop short of one, or cover two, so copying an interval across imports its errors wholesale. The restart logic was built from the Laws and the ball's resting position instead, which needs no second opinion about who had it |

Everything above is permitted as input in any combination. Nothing was labelled or corrected by
hand.

---

## 3. Training data

**None of it comes from the provided game.** No frame of BAR-ATH appears in any training set, no
hand label on it exists, and the competition's ground truth — public or private — is not read by
any code in the training path.

### Prepared, assessed, and documented

30 matches, 2,614,811 frames, converted into a single schema
(`external_data/scripts/common_schema.py`) so a model cannot tell which provider a row came from:

| source | matches | frames | labelled by | label quality | used in the submitted model |
|---|---|---|---|---|---|
| **IDSSE** (Bassek et al., *Nature Scientific Data*, 2025) | 7 | 1,002,644 | the DFL's own per-frame `BallStatus` and `BallPossession` provider fields | **reference** | **yes — 5 matches** |
| **Metrica** sample games | 3 | 429,923 | Metrica's event feed, derived by a documented state machine | 121,934 strong, 302,924 weak, 28 masked | no |
| **SkillCorner** open data | 20 | 1,182,244 | broadcast-tracking derived | 639,048 weak, 543,196 unlabelled | no |

### What the submitted model actually trained on

**Five IDSSE matches: J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9** — all Fortuna Düsseldorf home
fixtures. 715,116 labelled frames for stage 2; 399,359 live frames for stage 3. Open access under
the dataset's own licence, labelled by the provider rather than by us.

Two more IDSSE matches — **J03WMX (1. FC Köln) and J03WN1 (VfL Bochum)** — were sealed at the
start as a final held-out test, from a different competition with no team in common with the
development block. **No model was trained, tuned or selected on them, and their labels have never
been read** — the script that fits the submitted model names them only to state that it does not
load them.

To be exact rather than absolute: one diagnostic, run after the model was frozen and while
preparing this document, measured the ball-track repeat share quoted in §5 across all seven IDSSE
matches. That read their ball coordinates and nothing else. It changed no model and no threshold.
They remain unused as a test set, which is the thing they were kept for.

### Why the other 23 matches were prepared and then not used

Not an oversight, and the reasons are label semantics rather than volume:

- **Metrica defines possession on a TOUCH; this competition requires CONTROL.** Metrica's own
  event definitions begin a possession the moment a player touches the ball, while the
  competition says explicitly that a simple touch, a deflection that does not intercept, and an
  aerial duel won without subsequent control are *not* changes of possession. Those conventions
  are not the same and the difference is not cosmetic — Metrica hands possession over earlier and
  more often. Its **dead-ball** spans are law-based and were graded strong; its **team** spans are
  touch-based and were graded weak. Training stage 3 on them would teach it the wrong convention.
- **SkillCorner is 10 fps and was not upsampled.** Resampling a 10 fps observation to 25 fps
  invents fifteen observations a second that nobody made. Its labels are broadcast-derived and
  graded weak throughout.
- **Challenges with no recorded result were masked, not guessed.** 28 frames across Metrica carry
  an `ambiguous` quality and get no label at all. Filling the gap and calling it truth is the
  thing that quietly destroys an honest evaluation.

**SoccerNet-GAR** was probed for suitability and is **not used, not included and not
redistributed** — it is access-restricted, and the measurements taken from it are reported in
`external_data/reports/SoccerNetGAR_Measurements.txt` without reproducing any of its content.

### Protocol

Leave-one-match-out over the five development matches: train on four, predict the fifth, rotate.
The duration priors are re-fitted on each fold's four training matches, so the held-out match
contributes nothing to the decoder either. Every reported development figure is from that
protocol.

The **submitted** model is then re-fitted once on all five matches, because a fold is a model
that deliberately ignores a fifth of the data and there are five of them with no reason to prefer
one. That model is frozen in `submit/models/` with a manifest recording its feature order, row
counts and hyper-parameters. Nothing about it is estimated on an unseen match, and nothing can
be: the only unseen match left is this one, and its label may not be read.

---

## 4. The possession rules implemented

Where the lines were drawn, and what each one cost or bought.

### Dead or alive

- **Play is stopped when the players say so, not when the ball disappears.** The decision is
  entirely from player geometry: everyone decelerating together, the shape relaxing, players
  converging on a restart spot, and players standing physically outside the touchline.
- **The whistle is re-timed off the players.** Stage 2 puts an edge where its probability crosses;
  the real whistle is where team pace falls off a cliff. The edge is moved half of the way from
  one to the other, never outside the stoppage it belongs to. Worth 0.40 s of mean restart-edge
  error.
- **A visible ball lying still with nobody within 2 m, for a second, is a stoppage** — even one
  stage 2 never saw. A football does not stop on its own in open play. On the training data,
  every one of the 86 windows this finds is dead in the reference. It is the only rule permitted
  to *create* a stoppage rather than re-time one.
- **A sliver of play between two stoppages is not play.** Two stoppages separated by a fraction
  of a second is one stoppage with a hole in it.

### Which team

- **Loose balls go to nobody, not to the nearest body.** A ball outside the lines is `D`. A ball
  in the air is held with whoever struck it rather than handed to whoever it happens to be
  passing, because the nearest player to a ball in flight is usually the player it is flying
  towards.
- **A challenge needs control, not contact.** A switch requires the nearest-player vote to clear
  a margin, and then to **hold** that margin for 0.6 s over 60% of frames. A deflection does not
  survive that test; a genuine turnover does. This is the competition's own definition — control,
  not touch — implemented as a persistence requirement rather than a threshold on one frame.
- **Through a gap in the ball track, the last controlling team is carried** rather than letting a
  per-frame probability drift, and the carry is re-tested the moment the ball comes back. The
  model has to hold its own contrary view for 0.4 s before it takes back over.

### Restarts

- **The restart type is read off where the ball is resting**, in metres, against Laws-of-the-Game
  distances and not fitted thresholds: within 4 m of a corner flag is a corner, within 1.8 m of a
  touchline is a throw, within 7 m of a goal line and 12 m of the centre of the goal is a goal
  kick, the centre spot is a kick-off.
- **The taker is found by counting heads.** Eleven a side: if one side has ten players inside the
  lines and the ball is sitting beside a player who is outside them, that player is taking the
  restart. No identity needed.
- **A corner's taker is held for 0.8 s only.** A corner is contested almost immediately, so
  holding longer asserts something that stopped being true.
- **Possession stays `D` for the whole stoppage even once the restart team is known.** Knowing
  who will restart is not the same as play having resumed.

### Abstention

`X` is written on all 40,462 ungraded frames — before kickoff, half-time, after the final
whistle. Nothing is scored there, so a guess would be noise with no upside.

---

## 5. Limitations

### The finding that most changed the submission

**A rule can be measured as correct on the training provider and be false on this one, and the
cause can be a file-format convention rather than football.**

The training provider **freezes the ball's coordinate while play is stopped**: on **33–55%** of
consecutive observed frames the ball's position is byte-identical to the previous one — measured
per half across the five development matches, lowest 32.9%, highest 54.5% — so a dead ball reads
exactly 0.00 m/s. The competition's ATD is uncorrected computer vision that re-detects the ball
every frame: the same measurement gives **0.03%** and **0.13%** on its two halves.

Two shipped rules read the ball's speed from one frame difference, and both broke:

- `ball_at_rest` fired on **1.6%** of frames here against **22%** on the training matches.
- `ball_in_play` — *a visible ball moving inside the pitch means play is going on* — fired
  catastrophically. Measured on **half-time**, a stretch where the published boundaries say no
  football is being played and which carries no label whatsoever, it claimed **play was going on
  across 76% of it**. The same rule, on the training matches' reference-labelled dead frames,
  fires on **0.00%**.

The ATD's ball detection moves between objects — during half-time its apparent speed has a median
of 10 m/s and a 90th percentile of 39 m/s — so "the ball is moving" is not evidence about play on
this input. Averaging over a one-second window first was tried and does not rescue it: this is
wrong data, not noise.

The pipeline therefore **tests every ball-reading rule against the stretches with no play before
using it**, and drops any rule those stretches falsify. That test reads no label, public or
private; it uses only boundaries `DATA.md` publishes. On this match it drops `ball_in_play` and
keeps everything else, and it says so in its own output. On the training provider it drops
nothing, and the development scores are bit-identical with the guard in place.

Left unguarded, that one rule would have erased a quarter of the match's stoppages.

### What is still wrong with the model

- **Short same-team stoppages are the standing weakness.** 29.7% of stoppages have the same team
  either side, and 6 per match are same-team *and* under ten seconds — 6% of stoppages carrying
  1.5% of dead time. Each one missed merges two possessions into one and destroys a spell, and
  they are nearly invisible to any per-frame score. 1,014 possession changes are still missed
  across the five training matches.
- **The team decoder over-smooths.** It produces far fewer possessions than exist, because the
  duration prior prefers long stable spells to the true short ones. The instinct that took stage 2
  from 0.793 to 0.947 by killing flicker is the instinct that merges real possessions here.
- **85% per-frame team accuracy is not enough for 4-second possessions.** With a median
  possession under four seconds, almost every one contains at least one misattributed frame.
  `macroF1` 0.884 against `spellF1` 0.519 is the same trap, one level up: the per-frame metric
  flatters the model by a wide margin.
- **A deflection still reads as a touch** when the trajectory test cannot distinguish a
  controlled first touch from a ball glancing off a shin.
- **46 anonymous slots mean nothing can be per-player.** No passer, no receiver, no pressing
  player — every feature is collective, which rules out a family of methods that would help.
- **Direction inference is per period and could fail silently.** It is verified on the training
  matches and the half-time flip survives, but there is no label here to confirm it did. The
  home-share check below is the only evidence available that it did not.

### What the score on this match will actually be — unknown

Every development figure is measured on five German second-division matches from one provider,
with the ball hidden **artificially**. This match is a different league, a different tracker, 46
slots instead of 28, 20.5 objects tracked per frame instead of 22, and a ball that is missing for
real. The `ball_in_play` failure is direct evidence that transfer is not automatic. **Expect the
real score below 0.519, and treat the gap as unmeasured rather than small.**

The sealed matches J03WMX and J03WN1 would have given an honest held-out number and were kept for
exactly that. They were not opened, because opening them after the model was frozen would have
produced a number with no time left to act on and the temptation to act on it anyway.

### One check was run, once, and only as a check

The public five-minute window was scored **after** the model was frozen and after the ball-rule
guard was decided — the guard's criterion was the half-time falsification, fixed before any score
was looked at. It is not a tuning set and 17 possessions cannot rank anything.

| on video frames 11,185–18,684 | macroF1 | spellF1 | `D` recall | alive/dead F1 | home share error |
|---|---|---|---|---|---|
| **this model** | **0.886** | **0.533** | 0.961 | 0.975 | 0.0003 |
| published baseline (`smooth50`) | 0.621 | 0.326 | 0.214 | 0.899 | 0.0485 |

What it was used for: confirming the frames line up, that `H` and `A` are not swapped, that the
inferred playing direction is right, and that the possession share is calibrated. It confirmed all
four — team accuracy 0.865 on live frames against 0.855 measured on development data, and a home
share within 0.03% of the reference. Nothing was changed in response to it.

### What more time would go into, in order

1. **Short same-team stoppages**, which is where the ranking metric is actually being lost.
2. **Re-fit the possession-duration prior and weaken it where the probability is confident**, to
   stop the decoder merging real possessions.
3. **Use the Smart Tagging as a prior** on restart timing and team, handling `N/A` explicitly.
4. **Align the broadcast recording to tactical** and use the two cameras together, which is the
   only way to recover the ball on the 41% of frames where the tactical tracker loses it.
5. **Open the two sealed matches** and report the held-out number.

---

## 6. Reproduction

Full instructions, including re-fitting the models from the external data, are in
[README.md](README.md). The short version, from the delivered bundle:

```bash
python -m pip install -r requirements.txt
python local_checks/consolidate_data.py --source <download folder> --data-dir ./data
python submit/run_submission.py --team-name "Average Guys" --data-dir ./data
```

Three minutes, and it runs the organisers' own `validate_submission.py` on what it wrote.

**The repository does not contain the match data or anything derived from it.** `.gitignore` is a
whitelist: everything is excluded unless named, and the bundle, the intermediate files and the
submission itself are excluded by rules that override every whitelist above them.

**Verify a reproduction byte for byte.** The run prints a SHA-256 of the file it wrote.

| | |
|---|---|
| submitted file | `Average Guys.csv`, 199,319 data rows |
| SHA-256 | `8296efaf8cb3374cd24eb02f0cd302944fa1f980b4f88272724abc46a575bd5d` |
| label counts | `H` 46,055 · `A` 34,061 · `D` 78,741 · `X` 40,462 |
| frozen models | `submit/models/`, with `manifest.json` recording features, rows and parameters |
| environment | Python 3.14.2, LightGBM 4.6.0, numpy 2.4.1, pandas 3.0.0 |

Determinism: both models are LightGBM with `seed: 0` and the ball masking uses seeded
`numpy.random.default_rng`, so a re-fit reproduces the same models and a re-run reproduces the
same file.
