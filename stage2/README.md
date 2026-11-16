# Stage 2 — the dead-ball detector

Answer one question for every frame, from player positions alone: **is play stopped?**

Results live in `RESULTS.md`. This file is the method and the reasoning behind it.

---

## The three design decisions, and why

### 1. No ball features. At all.

The competition loses the ball on about 41% of graded frames. None of our training sources loses
it the same way — IDSSE never loses it at all. A detector allowed to look at the ball would learn
the shortcut *"ball missing ⇒ dead ball"*, which is roughly true in Metrica (missing on 96% of
dead frames) and roughly half wrong in the competition. That is the exact failure the published
baseline already makes.

Built from the 22 player positions only, this stage **cannot** make that mistake, and it transfers
to the competition unchanged whether the ball is visible or not. It also means the hardest open
question in the project — how the competition's camera loses the ball — does not block Stage 2 at
all.

### 2. Tuned for boundaries, not for frames.

From `external_data/reports/DeadBallProfile.md`, measured over all 713 stoppages in the seven
IDSSE matches:

- **29.7%** of stoppages have the **same team either side**. Missing one of those merges two
  possession spells into one.
- **42 of them — 6 per match — are same-team *and* under 10 seconds.** They are 6.0% of all
  stoppages and carry **1.5% of total dead time**.

So they are nearly invisible to a per-frame score, and every one that is missed destroys a
possession spell under spellF1 — the metric the competition actually ranks on. The evaluation here
therefore reports **danger-zone recall** as a first-class number, not per-frame accuracy alone.

### 3. Three decoders on identical probabilities.

The strategy deck argued that explicit duration modelling, not the classifier, is what makes
interval output usable. That is a testable claim, so all three decoders consume the *same*
per-frame probabilities and only differ in how they turn them into intervals:

| Decoder | What it does |
|---|---|
| `raw` | threshold at 0.5. No temporal model at all |
| `smooth` | rolling median, then delete runs shorter than ~1.5 s |
| `hsmm` | Viterbi over an explicit-duration semi-Markov model, with Alive and Dead duration distributions fitted on each fold's training matches |

---

## Features

97+ columns, all from player positions, in four families matching what a person would notice:

| Family | Examples |
|---|---|
| **Everyone stops** | mean / median / p90 / max player speed, per team and overall; fraction of players below walking, jogging and sprinting thresholds |
| **The shape relaxes** | team centroids and the distance between them, spread about each centroid, bounding-box width / length / area, how far the two teams' x-ranges overlap |
| **They walk to a spot** | players within 3 m of a touchline or goal line, players in a penalty area, players near a corner, the halfway-line split that signals a kick-off |
| **Off the field** | **players physically outside the lines.** TRACAB tracks to about 3.6 m beyond a touchline, so a thrower standing off the pitch is directly observable — the clearest cue available for the quick throw-ins that dominate the danger zone |

Eight core signals additionally carry **temporal context**: rolling mean and standard deviation
over 1 s, 3 s and 5 s windows, plus the deviation from the 5 s mean. A stoppage is not a single
frame that looks odd; it is a state that holds.

Rolling windows are computed **inside each half separately**, so nothing bleeds across half-time.

---

## Protocol

- **Five development matches**: J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9 (all Fortuna Düsseldorf
  home fixtures, DFL-COM-000002).
- **Leave-one-match-out.** Train on four, predict the fifth, rotate. Duration distributions are
  re-fitted on each fold's four training matches, so the held-out match contributes nothing to
  the decoder either.
- **Two test matches — J03WMX (1. FC Köln) and J03WN1 (VfL Bochum) — are never loaded by this
  code.** They are a different competition with no team shared with the development block, and
  they stay sealed until the end.

## Scoring

Uses the organisers' own `evaluation/possession_metrics.py`, imported directly from the
competition repository. Not a reimplementation.

To isolate what this stage contributes, the sequence scored is a **hybrid**: the true H/A label
wherever the detector says alive, `D` wherever it says dead. A perfect detector therefore scores
`spell_f1_mean = 1.0`, and any shortfall is attributable to dead-ball detection alone. An `X`
frame is inserted between halves so no spell can merge across the half-time gap.

Reported per fold:

- `frame_f1_dead` — per-frame F1 on the D class. The metric that looks good and means little.
- `spell_f1_mean` — the competition's ranking metric, averaged over IoU 0.3–0.7.
- `fragmentation` and `n_pred_spells` vs `n_gt_spells` — whether the decoder invents possessions.
- `stoppage_recall_iou50` — fraction of true stoppages matched at IoU ≥ 0.5.
- `danger_recall` — recall on short same-team stoppages. The one that decides the ranking.

## Running it

```bash
python -I stage2/features.py --matches DFL-MAT-J03WOH DFL-MAT-J03WOY DFL-MAT-J03WPY DFL-MAT-J03WQQ DFL-MAT-J03WR9
```

```bash
python -I stage2/train.py --rounds 400 --out "D:/Football Hackathon/stage2/results_lomo.csv"
```
