# The dead ball: definition, validation and shape

Written 30 October 2026, after reading the IDSSE data descriptor in full and measuring the seven
matches we hold. Raw output: `DeadBallProfile.txt`. Scripts: `scripts/profile_deadball.py`,
`scripts/profile_breakage.py`.

---

## 1. The convention question is settled — and it is the right one

This was the largest open question in the whole data record. `SourceSuitability.csv` previously
said of IDSSE: *"Whether the DFL's possession convention matches the hackathon's CONTROL-based
definition is NOT established and remains an open question."*

It is established. Bassek et al. 2025, *Scientific Data* 12:195, Data Records → Position data:

> "The ball possession is defined by the player in **control** of the ball. The match state is
> inactive when the match is interrupted by the referee, for example after a foul or during a
> substitution."

**The DFL assigns possession on CONTROL, which is the competition's own definition.** Metrica
assigns on TOUCH. That makes IDSSE the only collected source whose possession convention matches
the task, and it promotes IDSSE from "the only reference set" to "the only *correctly defined*
reference set".

### One caveat worth testing later

The sentence defines `BallStatus = 0` as *"interrupted by the referee"*. Under Law 9 the ball is
also out of play the instant it wholly crosses a line, whether or not the referee whistles. The
measurements in §2 below show 100 stoppages per match, which is far too many to be referee
interruptions alone, so in practice ball-out-of-play is clearly included. Treated as settled by
the evidence, but the wording is narrower than the behaviour.

---

## 2. The labels are exactly right — validated against the league's own clock

The match information file carries the DFL's official gross and net playing time per half
(`OtherGameInformation/TotalTime*`, `PlayingTime*`). That is an entirely independent record of how
long the ball was in play, produced by the league rather than derived from the position file.

Dead share from our per-frame labels, against `1 − net/gross`:

| Match | Frames | Dead, from labels | Dead, official | Difference |
|---|---|---|---|---|
| J03WMX | 145,967 | 41.5% | 41.5% | 0.0 pp |
| J03WN1 | 141,561 | 51.7% | 51.7% | 0.0 pp |
| J03WOH | 137,214 | 47.6% | 47.6% | 0.0 pp |
| J03WOY | 142,536 | 46.5% | 46.5% | 0.0 pp |
| J03WPY | 146,211 | 44.3% | 44.3% | 0.0 pp |
| J03WQQ | 142,345 | 40.7% | 40.7% | 0.0 pp |
| J03WR9 | 146,810 | 41.9% | 41.9% | 0.0 pp |

**Mean absolute disagreement: 0.00 percentage points, on all seven.**

This validates three things at once: the position-file parsing, the meaning of `BallStatus`, and
the dead-ball ground truth the whole of Stage 2 will be trained against.

---

## 3. The shape of a stoppage

713 stoppages across seven matches — about **102 per match**.

| | Stoppage | Passage of live play |
|---|---|---|
| p10 | 5.8 s | 4.3 s |
| p25 | 10.6 s | 9.1 s |
| **p50** | **19.6 s** | **20.3 s** |
| p75 | 32.1 s | 41.1 s |
| p90 | 49.4 s | 71.3 s |
| p99 | 102.4 s | 166.1 s |
| max | 206.5 s | 190.4 s |

A match is roughly an alternation of ~20 s live and ~20 s dead, about a hundred times over.

**Why a plain HMM cannot represent this.** An HMM's dwell time is geometric, which makes the
*shortest* stoppage the most likely one. The observed median is 19.6 s — 490 frames. Duration has
to be modelled explicitly, which is the argument for the HSMM in the strategy deck, now with a
measured number behind it.

### Where the dead time sits, versus where the stoppages are

| | Share of stoppages | Share of dead time |
|---|---|---|
| Under 5 s | 8.3% | **1.1%** |
| Over 30 s | 29.5% | **58.6%** |

Long stoppages own the clock. Short ones own the count.

---

## 4. Which stoppages actually break a possession

A dead interval is not itself scored by spellF1 — but it decides where the H and A spells either
side begin and end. The cost of missing one depends entirely on what sits on each side:

- **H → D → H** (same team either side): missing this **merges two spells into one**. Two true
  spells then face a single prediction, so one of them cannot be matched at all. Expensive.
- **H → D → A** (different teams): missing this only **shifts a boundary**. Both spells survive.
  Cheap.

Measured over 696 scored stoppages (17 more sit at the edge of a half):

| | Count | Share | Median | Under 10 s |
|---|---|---|---|---|
| **Same team either side** | 207 | **29.7%** | 22.0 s | 20.3% |
| Different team either side | 489 | 70.3% | 19.0 s | 21.7% |

### The danger zone

**42 stoppages — 6 per match — are both same-team and under 10 seconds.**

They are 6.0% of all stoppages and carry **1.5% of total dead time**.

That is the whole problem in one line. They are nearly invisible to a per-frame score such as
macroF1, and every single one that is missed destroys a possession spell under spellF1 — the
metric the competition actually ranks on.

**Design consequence.** Stage 2 must be tuned for *boundary* recall on short same-team stoppages,
not for per-frame accuracy. A detector that scores well on macroF1 by nailing the long stoppages
and smoothing away the short ones is precisely the detector that loses on spellF1.

This also explains why the SoccerNet-GAR coverage pattern is interesting rather than useless: it
covers the ~2.4 s either side of every stop and restart — the boundaries — and misses the long
interiors, which is the opposite of what a per-frame view would want and the right way round for
this metric.

---

## Sources

- Bassek, M., Rein, R., Weber, H. & Memmert, D. *An integrated dataset of spatiotemporal and
  event data in elite soccer.* Scientific Data **12**, 195 (2025). CC-BY-4.0.
  <https://www.nature.com/articles/s41597-025-04505-y>
- DFL. *Catalogue of Definitions Official Match Data, Version 5.1* (2020) — the formal definition
  document referenced by the above. **Not yet obtained**; worth requesting alongside more matches.
- Our own parsed data at `processed/idsse/*/frames.parquet`, and the DFL match information files
  at `raw/idsse/DFL_02_01_matchinformation_*.xml`.
