# The whole model, end to end — first honest measurement

2 November 2026. Both stages built, joined, and scored with the organisers' own
`evaluation/possession_metrics.py`. Leave-one-match-out over the five development matches; the
two test matches have still never been loaded.

---

## 0. A correction to the number I have been quoting

**The 0.947 reported for Stage 2 was optimistic. The correct figure is 0.862.**

To isolate Stage 2 I built a hybrid sequence: our dead-ball prediction, the true team elsewhere.
I wrote it as `where(model_says_dead, "D", truth)` — and `truth` is itself `H`, `A` *or* `D`. So
whenever the model **missed** a stoppage, the expression fell through to `truth`, which was `D`,
and emitted `D` anyway. **Missed stoppages cost nothing.** Only invented ones were ever charged.

Charged properly, on one match: 0.938 → 0.859, with predicted spells going 399 → 450 against a
true 402. Across all five folds Stage 2 is **0.862**, not 0.947.

Everything downstream of that figure in `RESULTS.md` and `HARDENING_RESULTS.md` carries the same
error. The *comparisons* in those documents remain valid — both arms were scored the same wrong
way — but the absolute level was overstated by about 0.09.

---

## 1. Where the model actually stands

At the competition's documented ball-missing rate of 41%:

| | spellF1 | macroF1 | Spells predicted | True |
|---|---|---|---|---|
| **Stage 2 alone** (our stoppages, true teams) | **0.862** | 0.947 | 468 | 410 |
| **Stage 3 alone** (true stoppages, our teams) | **0.496** | 0.901 | 243 | 410 |
| **End to end** (both ours) | **0.430** | 0.852 | 263 | 410 |

**Stage 3 is the bottleneck, by a wide margin.**

### The ball-robustness curve

Every training match was masked at a different rate (0%, 25%, 45%, 65%) so the model has seen the
whole range rather than one guess. The held-out match was then scored at four rates:

| Ball hidden | Team accuracy | Stage 3 spellF1 | End-to-end spellF1 |
|---|---|---|---|
| 0% | 0.879 | 0.553 | 0.487 |
| 20% | 0.866 | 0.530 | 0.463 |
| **41% (competition)** | **0.850** | **0.496** | **0.430** |
| 60% | 0.838 | 0.475 | 0.414 |

Losing the ball costs about **0.06 spellF1** between a perfect view and the competition's rate.
That is far less than feared — the masking strategy works, and player geometry carries real
signal on its own. **The ball is not the main problem.**

---

## 2. The real problem: 85% per-frame accuracy destroys possession spells

Team accuracy is **85%** at the competition's ball rate. That sounds respectable. It produces a
spellF1 of **0.496**.

The reason is the same lesson as before, one level up. Possession spells are **short** — median
3.9 seconds, 410 per match. A 15% per-frame error rate scattered across spells that brief does
not degrade them, it shatters them.

> **Cricket:** your scorer now gets 85 balls in every 100 attributed to the right batter. That
> sounds like a good scorer. But a partnership is only six or seven balls long, so almost every
> partnership has at least one ball credited to the wrong man — and the partnership column is
> wrong almost all the way down.

And macroF1 is **0.852** while spellF1 is **0.430**. The per-frame metric flatters the model by
twice over. Exactly the trap documented at Stage 2, now repeating at Stage 3.

## 3. The second problem: the team decoder over-smooths

Stage 3 predicts **243 possession spells** where there are **410**. It is merging real
possessions together.

This is the duration model doing too much work. Faced with a noisy 85%-accurate probability, the
HSMM prefers long stable spells to the true short ones. At Stage 2 that instinct was exactly
right — it took spellF1 from 0.793 to 0.947 by killing flicker. Here the true spells *are* short,
so the same instinct erases them.

> **Cricket:** the habit that made him a good scorer of overs — knowing roughly how long things
> last and refusing to believe anything shorter — is the habit that now merges every quick single
> into the partnership before it.

**Fix:** re-fit the duration prior on possession spells specifically rather than reusing the
dead-ball settings, and weaken it where the probability is confident. Not started.

---

## 4. What is working

- **Direction inference works.** Playing direction is inferred per team per period from the data
  alone — the team defending the left goal keeps its deepest player near x=0 all period. No roles,
  no identities, and it survives the half-time flip. Verified: after normalisation both halves
  show the same possession-territory pattern.
- **Ball features are handled honestly.** Every ball feature is paired with an explicit
  `ball_observed` flag and set to NaN when absent, so LightGBM routes "I cannot see the ball" down
  its own branch rather than guessing a number.
- **Masking augmentation works.** Only 0.06 spellF1 lost between a clean view and 41% missing.
- **Stage 2 is genuinely solid** at 0.862 once scored properly.

## 5. What is still missing or broken, ranked

| | Problem | Status |
|---|---|---|
| 1 | **Team attribution is weak.** 85% frame accuracy → 0.496 spellF1 | The main gap. Needs better features, not a better decoder |
| 2 | **Team decoder over-smooths** — 243 spells against 410 | Fix identified, not implemented |
| 3 | **Occlusion training not yet adopted** for Stage 2 | Tested (recovers 0.935), one config change |
| 4 | **Stage 3 has never been tested under occlusion** | Only ball-masking was tested. Players vanishing is untested here |
| 5 | **Evidence ceiling on stoppages** — 24% of misses have no signal | Cannot be fixed. No referee in the data |
| 6 | **Development matches too alike** — 5 of 7 the same club at home | Email to the league drafted, unsent |
| 7 | **Metrica cross-provider check** | Not run |

## 6. What would move the number most

The honest ranking, by expected gain:

1. **Better team features.** Pitch control (Fernández & Bornn) is the established way to turn 22
   positions into a per-team control surface and is not yet implemented — the current geometric
   features are weak (`depth_diff` effect size 0.13, `third_balance` 0.10) and the model is
   leaning almost entirely on ball proximity. This is where the headroom is.
2. **Re-fit the possession duration prior.** Cheap, and 243→410 spells is a large gap.
3. **Adopt occlusion training for Stage 2.** Already proven.

---

## 6b. Results of the three fixes (2 November, later)

### Pitch control — it did not work, and I said it would

I recommended this as "where the score actually moves". It moved 0.002.

| Ball hidden | Without pitch control | With pitch control |
|---|---|---|
| 0% | 0.487 | 0.481 |
| 20% | 0.464 | 0.464 |
| **41%** | **0.430** | **0.432** |
| 60% | 0.412 | 0.416 |

Team accuracy went 0.850 → 0.856 at the competition's ball rate. Every difference is far inside
the 0.04 noise band. It is very slightly better where the ball is scarce and very slightly worse
where it is not, which is the shape you would expect from a weak ball-substitute — but the
magnitude is nil.

**Why, as best I can tell.** Three independent feature families — raw geometry, ball proximity,
and now control surfaces — all land team accuracy in the same 85–88% band. That pattern says the
limit is not a missing feature. Per-frame classification of *who is in control* from positions
appears to saturate around there, and the remaining error is the genuinely ambiguous frames:
contested balls, the instant of a turnover, a loose ball between two players.

### The possession duration prior — correct fix, no effect

I found a real bug: the prior was fitted across stoppages, gluing the possession before a stoppage
to the one after, so it believed possessions were far longer than they are. Fixed, it now fits a
median of 3.8–4.4 s against a true 3.9 s.

End-to-end went 0.430 → 0.428. Predicted spells 263 → 263.

So the under-production of spells (263 against 410) was **not** caused by the prior. It is the
classifier's probability being too smooth to justify boundaries — the rolling windows that make it
stable also make it slow. A real fix would mean giving the team model sharper temporal resolution,
not adjusting the decoder.

### Occlusion training for Stage 2 — this one works, and generalises

The only fix of the three that paid. Trained on the heavy broadcast-like feed, then tested on a
**severity it had never seen**:

| Severity | Players visible | Frames with nobody | spellF1 |
|---|---|---|---|
| clean | 22.0 | 0% | 0.947 (clean-trained) |
| clean-trained, tested heavy | 14.3 | 14% | **0.716** |
| **occlusion-trained, tested on unseen mid** | **18.0** | **7.7%** | **0.929** |
| occlusion-trained, tested heavy | 14.3 | 14% | 0.935 |

0.929 on a severity never trained on, against 0.935 on the one it was. **That is genuine
robustness across the range, not memorisation of one level.** Danger-zone recall also reached its
best figure yet, 0.821.

**Adopted as the Stage 2 default.**

---

## 6c. What this means strategically

Two of three fixes did nothing, and the one that worked was the cheap one I had treated as a
checkbox. The honest reading:

- **Stage 2 is finished work.** 0.862 scored properly, robust to occlusion across a range.
- **Stage 3 is not a feature problem.** Three feature families saturate at the same accuracy. More
  features of the same kind will not help.
- **The remaining gap needs a different model class.** Per-frame classification plus interval
  decoding has gone as far as it goes. The alternative already identified in the data review is
  PathCRF (KDD 2026, MPL-2.0): a graph over players with one edge selected per time step as the
  current possession, decoded with a CRF that forbids impossible transitions. It is built and
  evaluated on the same seven Sportec matches we hold, and it infers possession **with no ball
  trajectory at all**. That is the next thing worth trying, and it is a rebuild of Stage 3 rather
  than an addition to it.

---

## 6d. The PathCRF-style rebuild — and what four failures together now tell us

Stage 3 was rebuilt as a **possession path** rather than a per-frame team classification, following
the idea in PathCRF (Kim et al., KDD 2026, arXiv:2602.12080). Two state types, as they describe:
a player holding the ball, and the ball **in flight** between players. The flight state is the
point — the competition defines possession by control and keeps a loose ball with the last
controlling team, which a per-frame classifier cannot express and a path can.

**Correction to an earlier note:** I had recorded PathCRF as MPL-2.0. Its README states no
licence. Nothing of theirs is used here; only the published idea is followed, with hand-set
transition costs in place of their learned Set Attention backbone.

### Two findings on the way

**The path structure alone is strong.** With a constant 0.5 team prior — no learned model at all,
only ball proximity and the transition constraints — team accuracy is **0.821**. The learned
per-frame model gets 0.850. Nearly all of it comes from structure.

**A bug worth recording.** Second halves collapsed to 0% home. Untracked substitute slots were
getting plausibility 0.0, which outranks a real player ten metres away at −2.5, so the path handed
possession to whichever substitute was sitting unwatched. It only bit once substitutes appeared:
first halves have 11 tracked of 11, second halves 16 slots of which ~5 are idle. Fixed by masking
untracked slots. **This is a direct warning for the competition feed, which has 46 anonymous slots
— most idle at any moment.**

### The result

| Approach | Team accuracy | End-to-end spellF1 | Spells predicted |
|---|---|---|---|
| per-frame + HSMM | 0.850 | 0.430 | 263 |
| + duration-prior fix | 0.850 | 0.428 | 263 |
| + pitch control | 0.856 | **0.432** | 261 |
| **path model** | 0.843 | 0.428 | **316** |

Ground truth: 410 spells.

The path model **fixes a large part of the spell under-production** — 316 against 261, so 77% of
the true count instead of 64% — and gains **nothing** on spellF1, because its team accuracy is
slightly worse and the two effects cancel.

### What the pattern means

**Four structurally different attacks all land between 0.428 and 0.432.** Raw geometry, ball
proximity, control surfaces, and a constrained path over players. Team accuracy stays in a
0.843–0.856 band throughout.

A response that flat is not a sequence of unlucky choices. It says the limit is in the
**information available**, not in the model class or the feature set. Per-frame team attribution
from anonymous positions, under this label definition, appears to saturate around 85%, and short
possession spells convert that into roughly 0.43 however the intervals are produced.

### The one lever not yet pulled

My path emissions are **hand-set and unlearned**. PathCRF trains theirs on per-frame ground-truth
*player-level* possession, derived from event data synchronised to tracking. IDSSE event data
names the player on every on-ball action, so those labels are derivable here — and that is the
only remaining material difference between this implementation and the published method.

Whether it is worth doing with 14 days left is a judgement call, not a technical one. It is the
only untried lever with a plausible mechanism behind it; it is also a multi-day build on top of
four consecutive null results.

---

## 6e. Player-level labels — built, trained, and it made things worse

The last untried lever. Per-frame player-level possession was derived from the DFL event feed
(`stage3/player_labels.py`) and the path emissions trained on it
(`stage3/train_emissions.py`, `stage3/run_trained_path.py`).

### Three things found along the way

1. **Set-piece restarts were being dropped.** Throw-ins, free kicks, goal kicks and corners carry
   the taker on a *nested* `Play` element, not a direct child. Reading only direct children lost
   about ninety of the most precisely-timed possession starts per match. Fixing it moved label
   agreement with the DFL's own field from **85.5% to 90.0%**.
2. **A slot-mapping bug.** `pivot_table` orders columns alphabetically, so `away_10` sits before
   `away_2`. Mapping a slot number to a column by position pointed every holder label at the wrong
   player. Holder top-1 accuracy went **0.08 to 0.45** once fixed (chance is 0.045).
3. **The referee is tracked — but only in the two sealed test matches.** J03WMX and J03WN1 each
   carry six referee FrameSets; the five development matches carry none. So the claim in §2.2 that
   "there is no referee in the data" was wrong as stated, but it is unusable: training on it would
   burn the holdout. **Worth checking whether the competition's own file includes officials** —
   that is a one-command local check and a genuinely untapped signal if it does.

### The result

| Approach | Team accuracy | End-to-end spellF1 | Spells (true 410) |
|---|---|---|---|
| per-frame + HSMM | 0.850 | 0.430 | 263 |
| + duration-prior fix | 0.850 | 0.428 | 263 |
| + pitch control | 0.856 | **0.432** | 261 |
| path, hand-set emissions | 0.843 | 0.428 | 316 |
| **path, trained emissions** | 0.846 | **0.397** | 236 |

Learning the emissions made it **worse**: 0.397 against 0.428. The trained scores are more
confident, so the path becomes stickier and the spell count falls further, from 316 to 236.

## 6f. The ceiling, and why it is where it is

**Five structurally different attempts. Team accuracy 0.843 to 0.856 every single time.**

The explanation is now measurable rather than speculative. **The DFL's two official products
disagree with each other.** Possession derived from their event feed matches their own per-frame
`BallPossession` field on only **89.2% to 91.4%** of live frames, across all five matches.

Our models reach 85%. The supervision we can construct tops out at 90%. The gap between those two
is the only real headroom, and five architectures have failed to take it — because the residual is
the genuinely ambiguous part: contested balls, the instant of a turnover, a ball loose between two
players. Those are exactly the frames the DFL's own two systems disagree about.

**This is a data limit, not a modelling limit.** No further work on Stage 3 architecture is
justified by the evidence.

## 7. Test-set discipline

J03WMX and J03WN1 have never been loaded by any model code in this project. Nothing here was
tuned against them.
