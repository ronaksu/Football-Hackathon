# Stage 2 hardening — before and after

> ## ⚠ CORRECTION, 2 November 2026
>
> **The absolute spellF1 figures in this document are overstated by about 0.09.** The hybrid used
> to isolate Stage 2 was written as `where(model_says_dead, "D", truth)`, and because `truth` is
> itself H, A *or* D, a **missed** stoppage fell through and was emitted as `D` anyway — so missed
> stoppages cost nothing. Only invented ones were charged.
>
> Scored properly, Stage 2 is **0.862**, not 0.947. The *comparisons* here remain valid, since
> both arms were scored the same wrong way, but the level is not. See `stage3/FULL_PICTURE.md`.


Five leave-one-match-out runs over the five development matches. The two test matches
(J03WMX, J03WN1) were never loaded. Scored with the organisers' own
`evaluation/possession_metrics.py`. Reasoning in `LAWS_AUDIT.md`, method in `README.md`.

**Headline: the four changes did not move the ranking metric on clean data, and the robustness
harness found a much bigger problem than the one I set out to fix.**

---

## 1. The five runs

`base` = the 108 pre-hardening features. `hardened` = 169, the same rows plus the 61 new columns,
so the A/B is exact. All with the HSMM decoder.

| | Features | Trained on | Scored on | **spellF1** | frame F1 (D) | Danger recall | Spells predicted |
|---|---|---|---|---|---|---|---|
| **A** | base | clean | clean | **0.948** | 0.971 | 70.6% | 410.2 |
| **B** | hardened | clean | clean | **0.947** | 0.973 | **79.3%** | 410.0 |
| **C** | base | clean | **degraded** | **0.792** | 0.928 | 68.3% | 463.4 |
| **D** | hardened | clean | **degraded** | **0.716** | 0.912 | 74.9% | 501.8 |
| **E** | hardened | **degraded** | **degraded** | **0.935** | 0.965 | 74.2% | 410.6 |

Run A reproduces the published 0.947 to within rounding, which is what makes the rest comparable.

## 2. The metrics the brief asked for

From saved per-frame predictions. Truth is 410 possession spells per match.

| | spellF1 | macroF1 | Dead precision | Dead recall | Dead F1 |
|---|---|---|---|---|---|
| base, clean | 0.947 | 0.974 | 0.941 | 0.935 | 0.938 |
| **hardened, clean** | 0.947 | 0.975 | **0.945** | **0.936** | **0.940** |
| hardened, degraded (trained on degraded) | 0.935 | 0.967 | 0.927 | 0.921 | 0.924 |

## 3. Do the fixes preserve genuine short possessions?

**Yes.** This was worth checking explicitly: a detector can always buy a tidy spell count by
smoothing short possessions away.

| | Spells under 3 s | Median spell |
|---|---|---|
| Ground truth | 826 | 3.92 s |
| base, clean | 851 | 3.84 s |
| hardened, clean | 851 | 3.84 s |
| hardened, degraded | 869 | 3.76 s |

If anything the model produces slightly *more* short possessions than exist. Nothing is being
smoothed out of existence.

## 4. Error counts, base against hardened (clean feed)

| Type | base | hardened | |
|---|---|---|---|
| SPELL_SPLIT (invented inside a live possession) | 50 | **46** | −4 |
| SPELL_MERGE (missed, same team either side) | 12 | **11** | −1 |
| MISSED_STOP | 17 | 17 | — |
| OVER_EXTENDED | 4 | 6 | +2 |
| PHANTOM_STOP | 8 | 11 | +3 |
| BOUNDARY | 281 | **273** | −8 |
| **Total** | **372** | **364** | **−8** |

---

## 5. What this actually says

### The hardening is marginal on clean data

spellF1 0.948 → 0.947. That is nothing. Dead-ball F1 moves 0.938 → 0.940, macroF1 0.974 → 0.975.
Errors fall 372 → 364.

The one real gain is **danger-zone recall, 70.6% → 79.3%** — short same-team stoppages, the class
I built changes 1 and 2 to attack. They worked on their target and the target turned out not to
be worth much, because spellF1 was already dominated by invented stoppages, and those only fell
from 50 to 46.

The raw-threshold decoder improved far more (spellF1 0.800 → 0.837, spells 547 → 508), which says
the new features genuinely sharpen the probabilities — the HSMM was simply already absorbing
most of that.

### The robustness harness is the valuable change

Change 4 was the cheapest to build and found the biggest problem.

On a realistic broadcast feed — mean 14.3 of 22 players visible, **14% of frames with nobody
tracked at all** — a model trained on the clean IDSSE feed drops from **0.948 to 0.792 spellF1**,
while frame F1 falls only 0.971 → 0.928. **A 16-point collapse in the ranking metric that the
per-frame metric almost completely hides.** That is the same blindness as the split/merge finding,
one level up.

### The new geometric features make occlusion worse, not better

Run D against run C: **0.716 against 0.792.** The hardened model is *less* robust.

The reason is plain in hindsight. Line straightness, `bunch_k3`, isolation and anchor score all
need to see most of the players. With 14 of 22 visible they are not noisy — they are wrong, and
confidently so. Plain speed aggregates degrade far more gracefully. Adding `n_tracked` so the
model can see its own blindness was not enough on its own.

### But it is fixable, and cheaply

Run E: train on the degraded feed and score on it, and spellF1 recovers to **0.935** — within
0.012 of the clean-data result, with the spell count back to 410.6 against 410.

**The features are fine. The training distribution was wrong.** Occlusion augmentation is the fix,
and it is a training-set change, not a modelling one.

### `smooth` should not be dropped — I was wrong about that

On a degraded feed `smooth` beats `hsmm` in both untrained-for-occlusion runs:

| | hsmm | smooth |
|---|---|---|
| C: base, degraded | 0.792 | **0.861** |
| D: hardened, degraded | 0.716 | **0.839** |

A duration prior fitted on clean data mis-fires when the probabilities get noisy; a crude fixed
minimum length is blunter but safer. My recommendation to drop `smooth` was made from clean-data
evidence only. **Keep it as the conservative fallback** until the occlusion-augmented model is the
one being shipped.

### One mechanism I got wrong

I predicted the defensive line would be tighter during live play. Per team it behaves that way
(`line_tight_home` 6.54 live vs 8.05 dead); the cross-team minimum does the opposite
(`line_tight_min` 4.17 live vs 3.47 dead). The feature carries signal either way and a tree does
not care about sign, but the mechanism is not the one I claimed in the plan.

---

## 6. What is still outstanding

- **Metrica cross-provider check — not run.** Requested in the brief and not delivered. It needs
  a feature build against a different source layout, and Metrica's dead-ball labels are
  event-derived under law-based rules rather than supplied per frame. It is a fair cross-provider
  check for the D channel specifically and should be done, but it was the lowest-value of the
  requested items next to the occlusion finding, and it is not started.
- **Occlusion augmentation as the shipped configuration.** Run E shows it works; it has not been
  adopted as the default. That is a one-line change to which feature directory trains the model,
  plus a decision on the clean/degraded mixing ratio.
- **The evidence ceiling stands.** 24% of missed stoppages have no signal at all in the player
  picture. No feature reaches those. See `LAWS_AUDIT.md` §2.2.

## 7. Test-set discipline

J03WMX and J03WN1 have never been loaded by any model code in this project. Nothing here was
tuned against them. They stay sealed until the changes are frozen.
