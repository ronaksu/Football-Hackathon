# Laws of the Game, the hackathon labels, and what the tracking can actually see

Written 1 November 2026, as the reasoning behind the Stage 2 hardening pass.

The hackathon's own definitions are authoritative for our labels. The Laws are consulted because
they are what the DFL's human operators were applying when they set `BallStatus`, so they tell us
what each label *means* — and, more usefully, what physical consequence each rule has that a
camera could see.

Throughout: **Stage 2 decides alive vs dead only.** Team attribution (H vs A) is Stage 3.

---

## 1. Why the gaps were not addressed earlier

The brief asks for this split, and it is a real one. Measured over the 87 missed and invented
stoppages in the first version:

| Category | Share | Meaning |
|---|---|---|
| **Decoder discarded a signal that was there** | 21 of 29 missed (**72%**) | The classifier crossed P=0.5 inside the true stoppage and the duration model threw it away. Fixable in the decoder |
| **Classifier confidently wrong** | 40 of 58 invented (**69%** above P=0.9) | Not a threshold problem. The model believes it. Only new evidence changes this |
| **No signal at all** | 7 of 29 missed (**24%**) | Peak P(dead) never reached 0.2 across the whole true stoppage. Nothing in the player picture moved |

So:

- **Missing decision logic** — changes 1, 2 and 3. The evidence was in the data the whole time and
  I never computed it. The speed features worked well enough that I stopped looking. No better
  reason than that.
- **Missing test** — change 4. Robustness to a broken tracking picture was never measured, so it
  was never known to be a problem.
- **Insufficient observable evidence** — the 24%. These are stoppages where no player did anything
  detectably different. The clearest case is J03WQQ 45:28, a 7.3-second stoppage where peak
  P(dead) was **0.008**. No feature engineering reaches it, because **there is no referee in the
  tracking data.**

---

## 2. The audit, scenario by scenario

Confidence is about the *evidence*, not the current model.

### 2.1 Ball out of play — Law 9

> The ball is out of play when it has wholly passed over the goal line or touchline, on the
> ground or in the air, or when the referee stops play.

Label: **D**, from the moment it crosses until the restart is taken.

| Restart | Observable consequence | Evidence | Confidence |
|---|---|---|---|
| Throw-in | **A player steps off the field.** TRACAB tracks ~3.6 m beyond a touchline | `n_outside_pitch`, `max_outside_m`, `n_near_touchline` | **High** — `n_outside_pitch_m125` ranked 4th by gain on its first outing |
| Corner | One player isolated in a corner arc, bodies packing the box | `n_near_corner`, `n_in_box`, `bunch_k3_*` (new) | Medium |
| Goal kick | One player alone deep near a goal line, everyone else pushed up | `min_d_goalline`, `lone_isolation_m` + `lone_d_line_m` (new) | Medium |

**Unobservable:** "wholly passed over" itself. Without the ball there is no way to time the
crossing, only the players' reaction to it — which is why the start boundary runs late.

### 2.2 Referee stoppages — Law 9, second clause

Fouls, offside, injury, substitution, VAR, cautions.

Label: **D**, from the whistle to the restart.

**There is no referee in the data.** No whistle, no official's position, no card. Everything must
be inferred from secondary player reaction — which exists when a foul is obvious (players stop,
appeal, converge on the referee) and does not exist when play simply peters out.

**This is the evidence ceiling.** It accounts for the 24% above. I have not tried to engineer past
it and I do not think it can be.

*Partially recoverable:* a wall forming is a strong downstream signal of a free kick, even though
the whistle was invisible. `bunch_k3_*` targets this — it detects the consequence, not the cause,
and therefore detects it late.

### 2.3 Goals and the kick-off restart — Laws 8 and 10

Label: **D** from the ball crossing the line until the kick-off is taken.

The goal itself is unobservable without the ball. **The restart is extremely observable:** a
kick-off puts all eleven of each team in their own half with the centre circle empty but for the
two taking it. Nothing else in a match looks like that.

Evidence: `n_left_half`, `halfway_split`, `n_in_centre_circle` (new). **Confidence: high.**

Caveat: the same formation occurs at the start of each half, so it must be read together with the
period boundary rather than as "a goal was scored".

### 2.4 A stationary or undetected ball that may still be live

**The single largest error class — 50 invented stoppages, 10 per match.**

The Laws are unambiguous: a stationary ball on the field of play is in play. A goalkeeper holding
the ball is in play, and it is his team's possession. A team passing it around the back with
everybody walking is in play.

The model's problem is that all three look exactly like a stoppage through a speed-only lens:
everyone slows, the picture compresses, nothing sprints.

> **Note the cricket analogy breaks here.** In cricket the ball settling in the keeper's gloves
> makes it *dead*. In football the keeper holding it is *live* and is possession. Opposite
> answers — do not let the intuition carry over.

Evidence that should separate them, and what change 1 and 2 add:

| Signal | Live play | Stoppage |
|---|---|---|
| Defensive line held for the offside trap (Law 11) | Maintained | No reason to exist; disperses |
| A restart anchor — a fixed point everyone orients to | None | Touchline spot, corner arc, penalty spot |
| A wall or a packed box | No | Often |

**Confidence: medium.** These are consequences of live play rather than of the ball being live,
so they are correlates, not proofs.

> **An honest correction.** I predicted the defensive line would be *tighter* during live play.
> Measured across the five development matches, the per-team figure behaves as predicted
> (`line_tight_home` 6.54 live vs 8.05 dead) but the cross-team minimum does the opposite
> (`line_tight_min` 4.17 live vs 3.47 dead) — during a stoppage one team often forms a *very*
> tight line somewhere. The feature carries signal either way and a tree does not care about sign,
> but the mechanism is not the one I claimed.

### 2.5 Controlled possession versus touches, deflections and aerial duels

**Stage 3, not Stage 2** — but the convention matters and is now settled.

The DFL defines possession by *the player in control of the ball* (Bassek et al. 2025). That is
the competition's own definition. Metrica assigns on *touch*. So IDSSE is the only collected
source that is convention-correct, and Metrica's possession labels must be treated as a different
quantity.

Consequences for Stage 3: a deflection is not a change of possession; an aerial duel is contested
with no controlling team until one side settles it; a toe-poked clearance under pressure is a
touch, not control.

**Evidence: partial at best.** Exact ball contacts are not observable — the competition loses the
ball on 41% of graded frames, and Stage 2 uses none of it by design. Control will have to be
inferred from player geometry and the persistence of the possession state.

### 2.6 Loose ball — possession stays with the last controlling team

A ball running loose after a tackle or a blocked pass belongs to nobody, but the competition still
wants H or A.

**This is an architectural property, not a feature.** The HSMM over {H_alive, A_alive, Dead} only
leaves a state when the evidence supports it, so a loose ball does not flip the label — it
persists until the other team demonstrably gains control. Thresholding frame by frame would
flicker; that is exactly the 0.793-vs-0.947 gap already measured.

**Preserved unchanged by this pass.**

### 2.7 Half boundaries and state resets

- Every rolling, leading and trailing window is computed **inside a period** and never across one.
- The decoder runs per period, so no spell or stoppage can span half-time.
- The competition grades blocks 11,185–90,651 and 110,808–190,197. **The second block begins at
  110,808 but the half kicks off at 113,742** — so 2,934 graded frames before kick-off must be
  labelled **D**, not abstained on. A model that treats "before kick-off" as outside the match
  loses all of them.

**Confidence: high, by construction.** No frame numbers are hard-coded anywhere in Stage 2;
periods come from the data.

---

## 3. What is deliberately not inferred

Flagged rather than guessed, per the brief:

| Not inferred | Why |
|---|---|
| Referee decisions | No referee in the data. The 24% ceiling |
| Exact ball contacts | Not observable; Stage 2 uses no ball at all |
| Which team a stoppage restarts | Stage 3's problem; Stage 2 outputs alive/dead |
| Why a stoppage happened | Irrelevant to the label, and unobservable |

## 4. Contract preserved

- **H / A / D / X output contract unchanged.** Stage 2 contributes the D channel; it never emits
  X. Abstention is a statement about team-attribution confidence and belongs to Stage 3, so
  forcing it out of a dead-ball detector would be meaningless.
- **Missing observations stay explicit.** `n_tracked`, `n_tracked_home` and `n_tracked_away` are
  features, so the model can see its own blindness rather than silently treating an absent player
  as a stationary one.
- **No frame-specific hard-coding.** Periods, pitch dimensions and duration distributions are all
  read or fitted from data.
- **No ball features.** The missing-ball problem cannot touch this stage, by construction.
