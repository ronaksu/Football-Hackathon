# The decoder, investigated and fixed — 3 November 2026

Prompted by a claim in the dry-run report that 79% of the stoppages we miss "had the signal and
the decoder refused it". That claim was **wrong as stated**, and chasing it turned up a worse bug.
Both are recorded here.

All work is leave-one-match-out over the five open-access IDSSE development matches. Nothing was
trained, fitted or tuned on the competition's own data, in line with the organisers' clarification
of 3 November 2026. The two sealed matches remain unopened.

---

## 1. The bug that mattered: the training loop could not see a missed stoppage

`hybrid_codes` in `train.py` built the Stage-2-only sequence as:

```python
p = np.where(pred_dead, "D", true_labels)
```

`true_labels` is `H`, `A` **or `D`**. So wherever the detector *missed* a stoppage, the expression
fell through to the truth and emitted `D` anyway. **Missed stoppages cost nothing. Only invented
ones were ever charged.**

This is the same bug previously found and corrected by hand in `stage3/FULL_PICTURE.md` §0, but
it was never fixed *in the training loop itself*. So every decoder comparison and every tuning
decision made with `train.py` was made against a metric blind to half the error.

Fixed: the team is now carried in from the nearest live frame, so a missed stoppage is charged as
a merged possession, which is what it is.

| Decoder | spellF1, as scored before | spellF1, scored correctly |
|---|---|---|
| raw | 0.837 | **0.699** |
| smooth | 0.946 | **0.864** |
| hsmm | 0.947 | **0.872** |

The decoder ranking survives and the gap between `hsmm` and `smooth` widens from 0.001 to 0.008.
`hsmm` stays the default; `smooth` stays as the degraded-feed fallback.

---

## 2. The duration prior was genuinely broken — and fixing it changed nothing

The Alive/Dead duration distributions were a linear histogram over 1000 bins, box-smoothed over
five of them and floored at `1e-3`. Measured on the five development matches:

- ~500 stoppages spread over 1000 bins
- **76% of the bins inside the observed range are empty**; the largest count is 8

An empty bin lands on the floor at about **−13 nats**; an occupied neighbour costs about **−5**.
Whether a stoppage of a given length was permitted therefore came down to whether some training
match happened to contain one within half a second of it:

| True stoppage | old log P | new log P |
|---|---|---|
| 1.0 s | **−13.13** | −8.38 |
| 2.0 s | −6.04 | −5.82 |
| 10.0 s | −4.88 | −5.11 |
| 90.0 s | **−13.13** | −8.85 |

Replaced with a Gaussian kernel in **log** time — durations span 0.2 s to 200 s, so one linear
window cannot suit both ends — with Silverman's bandwidth computed per fold, so no constant is
hand-tuned. The aggregate shape is unchanged: both models put ~22% of the mass under ten seconds
against a true 21%. Only the holes are gone.

**Effect on the score: none.** spellF1 0.872 → 0.871, stoppage recall 0.904 → 0.910. Adopted
anyway, as a robustness fix: the old prior's behaviour at any given duration was a lottery.

---

## 3. Why the missed stoppages are not the decoder's fault

For every true stoppage, the two terms Viterbi weighs were decomposed: the emission gain summed
over the span, and what the alternating duration model charges to insert it.

| | caught (475) | missed (28) |
|---|---|---|
| median duration | 20.6 s | 8.1 s |
| **median emission gain** | **+595** | **−53** |
| median duration cost | −8.9 | −9.1 |

**The duration cost is effectively constant across every duration band** (−8.6 to −10.1). It is
not what discriminates. The emission is.

Over a missed stoppage the classifier's mean probability is **0.29**, against 0.95 on the ones it
catches, and only **24%** of its frames exceed 0.5, against 96%. The "79% had peak probability
above 0.5" figure was measuring the **peak**, and the peak is a spike of median **1.0 s** inside an
eight-second stoppage. The signal is transient, not present-and-ignored.

Taking the best contiguous sub-window inside each missed stoppage:

| best sub-window worth | count | median length |
|---|---|---|
| > 0 nats | 22 of 28 (79%) | 2.5 s |
| > 10 nats | 13 of 28 (46%) | 3.8 s |
| > 40 nats | 7 of 28 (25%) | 6.2 s |

So between 7 and 13 of the 28 are in reach of a decoder willing to emit short segments — and the
rest are invisible to it.

---

## 4. The decoder is already at its optimum

A per-frame bias on the dead state — the one knob that trades the two error types — swept over
LOMO on external data only:

| bias | spellF1 | stoppage recall | danger recall | invented / match | missed / match |
|---|---|---|---|---|---|
| −1.20 | 0.865 | 0.881 | 0.662 | 7.0 | 5.4 |
| −0.30 | 0.871 | 0.904 | 0.742 | 10.0 | 3.4 |
| **0.00 (shipped)** | **0.871** | 0.910 | 0.793 | 11.6 | 3.0 |
| +0.30 | 0.873 | 0.915 | 0.793 | 13.0 | 2.8 |
| +0.50 | 0.873 | 0.917 | 0.821 | 13.6 | 2.4 |
| +1.00 | 0.871 | 0.915 | 0.875 | 15.8 | 2.4 |
| +1.40 | 0.862 | 0.919 | 0.900 | 19.4 | 1.8 |
| +1.80 | 0.851 | 0.926 | 0.933 | 24.0 | 1.4 |
| +3.00 | 0.807 | 0.885 | 0.933 | 36.2 | 1.2 |

The best attainable is **0.873 against 0.871 shipped — a gain of 0.002, inside the metric's own
0.04 noise band**. Buying the missed stoppages is possible (danger recall reaches 0.933) but costs
more in invented ones than it returns: they triple.

**No bias is adopted.** A tuned constant that buys a sub-noise gain on five matches is
overfitting, not improvement. The curve is recorded here so the choice stays reproducible.

---

## What this means

| Claim | Verdict |
|---|---|
| "The decoder refuses stoppages it can see" | **Wrong.** The classifier does not see them: mean probability 0.29 over the span |
| "79% are fixable in the decoder" | **Wrong as stated.** 79% contain a positive sub-window of median 2.5 s; only 46% are worth more than 10 nats |
| "The duration prior is broken" | **Right** — 76% empty bins — but fixing it moves the score by −0.001 |
| "The training metric was sound" | **Wrong.** It could not see a missed stoppage at all. This is the real find |

The remaining Stage 2 score is in the **classifier**, not the decoder: it needs evidence that a
short same-team restart is happening, and from 22 player positions alone that evidence is a
one-second blip. That is the referee-tracking question, not a decoding question.
