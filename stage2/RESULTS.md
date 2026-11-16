# Stage 2 results — the dead-ball detector

> ## ⚠ CORRECTION, 2 November 2026
>
> **The absolute spellF1 figures in this document are overstated by about 0.09.** The hybrid used
> to isolate Stage 2 was written as `where(model_says_dead, "D", truth)`, and because `truth` is
> itself H, A *or* D, a **missed** stoppage fell through and was emitted as `D` anyway — so missed
> stoppages cost nothing. Only invented ones were charged.
>
> Scored properly, Stage 2 is **0.862**, not 0.947. The *comparisons* here remain valid, since
> both arms were scored the same wrong way, but the level is not. See `stage3/FULL_PICTURE.md`.


Leave-one-match-out over the five development matches. The two test matches were not loaded.
Method in `README.md`. Raw per-fold numbers in `results_lomo.csv`.

Scored with the organisers' own `evaluation/possession_metrics.py`, on a hybrid sequence: the
true H/A label wherever the detector says alive, `D` wherever it says dead. A perfect detector
would score `spell_f1_mean = 1.0`, so every shortfall below is dead-ball detection alone.

---

## Headline

| Decoder | Frame F1 (D) | **spellF1** | Stoppages found | Danger zone | Fragmentation | Spells predicted |
|---|---|---|---|---|---|---|
| `raw` | 0.972 | **0.793** | 88.8% | 73.1% | 1.611 | 558.6 |
| `smooth` | 0.973 | **0.945** | 89.9% | 70.6% | 1.035 | 414.6 |
| **`hsmm`** | 0.971 | **0.947** | 89.7% | **73.1%** | **1.019** | **410.0** |

Ground truth: **410.0** possession spells per match.

### The one line worth remembering

**Frame F1 is 0.971–0.973 for all three. spellF1 runs from 0.793 to 0.947.**

The per-frame metric cannot see a difference of 0.154 in the metric that decides the ranking —
nearly four times the competition's own 0.04 noise band. A model selected on frame accuracy would
pick between these three at random.

This is the strategy deck's central claim, now measured rather than argued: the classifier is not
what matters at this stage. **The decoder is.**

### What the decoder is actually fixing

`raw` thresholding predicts **558.6** spells where there are **410**. It invents roughly 149
possessions per match by flickering in and out of the dead state. Each flicker chops a real
possession into pieces, and spellF1 charges for every one.

The HSMM predicts **410.0** against a truth of **410.0** — a mean error of zero, and within 2.5%
on every individual fold (399/402, 445/456, 420/421, 411/401, 375/370). Giving the model an
explicit duration distribution, rather than letting it change its mind frame by frame, is what
closes the gap.

**Cricket:** three scorers watch the same match and all three get 97% of individual balls right.
The first writes down a wicket every time a batter pauses to adjust his gloves, and ends the day
with 558 partnerships instead of 410. The second and third know roughly how long a partnership
lasts, and hand in the right number. Ball-by-ball accuracy told you nothing about which scorecard
to trust.

---

## Per fold

HSMM only. Full table for all three decoders is in `results_lomo.csv`.

| Held out | Frame F1 (D) | spellF1 | Stoppages | Danger zone | Spells pred / true |
|---|---|---|---|---|---|
| J03WOH | 0.973 | 0.937 | 87.3% | 6/8 | 399 / 402 |
| J03WOY | 0.982 | 0.962 | 91.2% | 9/9 | 445 / 456 |
| J03WPY | 0.976 | 0.951 | 88.7% | 4/7 | 420 / 421 |
| J03WQQ | 0.966 | 0.949 | 89.9% | 2/6 | 411 / 401 |
| J03WR9 | 0.959 | 0.936 | 91.3% | 5/5 | 375 / 370 |

**spellF1 across folds: 0.936 to 0.962, standard deviation 0.011.** Tight — but see the caveats.

---

## Which features carry it

Top of the gain ranking, aggregated over all five folds:

| | Feature | What it is |
|---|---|---|
| 1 | `spd_p90_m125` | 90th-percentile player speed, averaged over 5 s |
| 2 | `spd_mean_m125` | mean player speed, averaged over 5 s |
| 3 | `spd_p90_m75` | the same at 3 s |
| **4** | **`n_outside_pitch_m125`** | **players physically off the field, over 5 s** |
| 5 | `frac_walking_s125` | variability in how many players are walking |
| 6 | `width_home` | how wide the home team is spread |
| 7 | `min_d_goalline` | closest player to a goal line |

The top three are "everyone stopped running", which is the obvious signal and dominates as
expected.

**`n_outside_pitch` is the interesting one.** TRACAB tracks players up to ~3.6 m beyond a
touchline, so a thrower standing off the pitch is directly observable. Adding it moved the HSMM's
stoppage recall from 85.9% to 89.7% and its danger-zone recall from 72.2% to 73.1%. It was added
specifically to target quick throw-ins, which is what the danger zone is mostly made of, and it
landed 4th by gain on its first outing.

**Cricket:** the surest sign the ball is dead is not that the fielders have slowed down — it is
that somebody is standing outside the boundary rope holding it.

---

## The honest weak spot

**Danger-zone recall is 73.1%: 26 of 35 short same-team stoppages across all five matches.**

These are the stoppages where the same team has the ball before and after, and the whole thing
lasts under ten seconds — almost all quick throw-ins and quick free kicks. Miss one and two
possession spells merge into one that never happened.

Three things to be straight about:

1. **The sample is tiny.** 35 cases in total, 5 to 9 per match. J03WQQ's 2-of-6 and J03WOY's
   9-of-9 are not evidence of a real difference between those matches; they are small-number
   noise. The 73% figure has a wide interval around it and should not be quoted to three
   significant figures.
2. **It is the right thing to attack next**, because it is the only failure mode here that
   destroys spells rather than shifting boundaries.
3. **`smooth` is worse than `raw` on exactly this metric** (70.6% vs 73.1%) while looking better
   on stoppage recall. That is the minimum-duration floor deleting genuine short stoppages — a
   clean illustration of why a fixed length cut-off is the wrong tool and an explicit duration
   distribution is the right one.

---

## What these numbers do and do not mean

**Do not read 0.947 as a projected competition score.** Four reasons:

1. **The H/A labels were given.** This is Stage 2 in isolation. Stage 3 has to produce H and A
   from geometry, and its errors compound with these.
2. **All five development matches are Fortuna Düsseldorf home fixtures**, same competition, same
   venue. Leave-one-match-out here is within-domain by construction. The two sealed test matches
   (1. FC Köln, VfL Bochum) are a different competition sharing no team, and are the only
   genuine out-of-domain check we have.
3. **IDSSE tracking is near-perfect** — 11.0 of 11.0 players on essentially every frame. The
   competition's feed is fragmented across 46 anonymous slots. The features here assume a clean
   22-player picture and have not yet been tested against a broken one.
4. **Different dead-ball rates.** These matches are 40.7–51.7% dead. The competition's public
   five-minute window is 19.6%. The HSMM's duration priors are fitted per fold, so they can adapt,
   but they have never been fitted to anything resembling 19.6%.

**What they do establish**, and this is the useful part:

- A dead-ball detector that never looks at the ball works, at 0.947 spellF1 in isolation. The
  missing-ball problem does not block this stage — exactly as argued before building it.
- Explicit duration modelling is worth **+0.154 spellF1** over thresholding, and the per-frame
  metric is blind to all of it.
- The pipeline runs end to end against the organisers' real scorer.

---

## Next

1. **Robustness to a broken tracking picture.** Drop random players, drop whole teams for stretches
   mimicking a camera pan, and re-measure. The SkillCorner matches are already collected as the
   domain-shift probe for this and have never been trained on.
2. **The danger zone.** More restart-specific features — ball-free throw-in and free-kick
   geometry — and a duration model that is less willing to smooth away a 4-second stoppage.
3. **Stage 3**, which is a separate problem: H versus A from player geometry.

The two test matches stay sealed. The right moment to spend them is once Stages 2 and 3 are
joined, not now — they are the only honest out-of-domain measurement available and they can only
be used once.
