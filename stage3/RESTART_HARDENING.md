# Restart hardening — what was built, what it changed, and what was rejected

Built 5 November 2026. Code: `stage3/restart_logic.py`. Scored by the organisers' own
`evaluation/possession_metrics.py`, leave-one-match-out over the five open-access IDSSE
development matches. The two sealed test matches were not opened.

---

## The problem

Stage 2 decides live or dead from player geometry. Stage 3 decides which team from a per-frame
probability. Neither knows what a *restart* is. So at the moment play resumes — a corner swung
in, a throw taken, a goal kick launched — the team call is made by a classifier looking at one
frame of geometry, exactly where geometry says least: both sides are packed into the box and the
only thing that settles possession is who takes the kick.

The Laws of the Game settle it instead, and the ball's resting position says which Law applies.
None of that needs the ground truth or the competition's own data.

## What was built

Four steps, in the order the frames arrive.

1. **Every stoppage gets two separate fields** — the last team in control, and the likely restart
   team — plus a restart *kind* classified from where the ball is resting, and a resume frame
   found from the ball being kicked and clearly moving (for a throw, entering the pitch). **The
   output stays `D` for the whole stoppage even once the restart team is known.**
2. **k-NN on the first reliable ball frame** — the nearest two players of each side, weighted
   `1/(d+0.5)²`, with the vote required to hold for 0.6 s across reliable frames and a player
   within 3 m before any switch is made. A defender standing nearby or a deflection does not
   last that long and so cannot carry the decision.
3. **Possession carried through gaps** — when the ball disappears in live play the last
   controlling team is held rather than letting the probability drift; the case is re-tested when
   the ball returns. After a stoppage the carry starts from the restart team.
4. **Every field is exported per frame** and shown in the review tool, and the whole thing is
   measured before and after.

### Honesty constraints, which are what make the numbers mean anything

- The ball is read **only where the masked ball is observed** — the same ball the model was
  allowed to see at the scored 41% masking rate. Reading the unmasked track here would be
  reading data the competition will not have.
- No ground-truth label is touched in the decision path. `label` is loaded for scoring only.
- Development matches only.

---

## What the measurement said

### The restart-team rule is worse than the model, except on corners

For every stoppage followed by live play, the team the rule names against the team the reference
has in the two seconds after the restart:

| restart kind | n | the Laws-and-geometry rule | the model already |
|---|---|---|---|
| goal kick | 68 | **0.985** | 0.985 |
| throw-in | 176 | 0.807 | 0.881 |
| free kick | 194 | 0.737 | 0.985 |
| corner (all) | 61 | 0.672 | 0.852 |
| **corner (taker on the flag, box loaded)** | **34** | **0.912** | **0.853** |
| every kind | 491 | 0.821 | **0.935** |

Read plainly: **the model reads restarts better than a rule can, with one exception.** A free
kick's taker is not determined by anybody's position, and a throw-in's only weakly. The one place
the Laws win is a corner that is unambiguous — a taker standing on the corner flag and a loaded
penalty area — where the rule is right 91% against the model's 85%, and fixes two stoppages
without breaking any.

### So three of the four steps were rejected

End-to-end over the five matches, each step on its own:

| configuration | spellF1 | macroF1 | restart edge \|err\| | false switches | cases |
|---|---|---|---|---|---|
| baseline | 0.438 | 0.855 | 1.446 s | 431 | 2943 |
| restart edge alignment only | 0.439 | 0.856 | 1.355 s | 447 | 2971 |
| + corner rule | 0.445 | 0.858 | 1.355 s | 417 | 2913 |
| + corner rule, **late moves gated** (**shipped**) | **0.446** | **0.858** | **1.311 s** | **419** | **2912** |
| + carry through ball gaps | 0.426 | 0.847 | 1.355 s | 415 | 2954 |
| + restart team for every restart | 0.402 | 0.839 | 1.355 s | 520 | 3108 |

Carrying through gaps costs 0.019 spellF1 and 27 extra missed switches: holding the pre-gap team
is right more often than not, but the model's probability was already doing better than a hold.
Forcing the restart team on every restart costs 0.036 — the table above says why.

### The gate that matters most: a missing ball is not evidence

The first version moved the restart edge onto the first frame the ball was *reliably seen to
move*. At the competition's 41% masking rate that is a trap: "the ball was not seen to move" is
also exactly what a hidden ball looks like, so the edge drifted **later** on stoppages where the
ball happened to be masked at the real restart. Nine stoppages that had been timed to within half
a second drifted past the one-second tolerance and became new `BOUNDARY` cases — every one of
them with an override share of zero, so the nearest-player rule had nothing to do with it.

The fix is to require a second opinion before *extending* a stoppage, and how the two sources are
combined is itself measurable:

| rule for moving the restart edge later | fixed | broken | restart edge |err| |
|---|---|---|---|
| ball evidence alone | 42 | 12 | 1.355 s |
| ball visible **and** stage 2 still calling it dead | 28 | 3 | 1.349 s |
| ball visible **or** stage 2 still calling it dead (**shipped**) | **34** | **3** | **1.311 s** |

Demanding both agree throws away a third of the fixes for no further reduction in breakage.
Demanding either one keeps them and still removes nine of the twelve regressions. Moving the edge
*earlier* needs no gate: seeing the ball move is positive evidence, and its absence is not.

### Three implementation faults found by measuring, not by reading

Each of these made the change look worse than it was, and each was a real defect:

1. **Handing control back to the model the instant an override ended invented a turnover** — the
   carried team and the decoded team differed, so the join itself read as a switch. The model now
   has to hold its own view for 0.4 s before it takes back over.
2. **No floor on possession length.** The base decoder models spell duration explicitly;
   overriding it bypassed that. Possessions under 0.8 s are now merged away.
3. **Frames freed by moving a restart edge earlier kept a team decided while the model thought
   the ball was dead** — a meaningless value. They now adopt the possession that follows. This
   one fault alone accounted for 0.007 spellF1 and 29 cases.

---

## The touch rule: possession belongs to whoever played the ball

Added 5 November 2026, after a reviewer note on J03WOY pointed at a pair of frames where the
model gives the ball to the wrong side while a pass is travelling.

### What the frames actually said

The note read it as a home pass being handed to a closer away player. The frames say the
opposite, and the real mechanism is more useful than the one we assumed:

```
frame   reference   model   nearest home   nearest away   ball speed
12137       A         A         0.8 m          4.5 m        12.6 m/s
12138       H         A         0.4 m          5.0 m         3.4 m/s   <- the touch
12139       H         A         0.5 m          5.0 m         8.0 m/s
...
12168       H         A         8.1 m          0.4 m         7.2 m/s   <- going past an opponent
```

A ball travelling at 12.6 m/s loses three quarters of its pace in one frame with a home player
0.4 m away, and then leaves at 8 m/s on a new course. That is a home player **touching** it, and
the reference changes hands on exactly that frame. Thirty frames later an away player is 0.4 m
from the ball and home is 8 m away, so every cue built on distance-to-ball now points the wrong
way, which is why the model says `away` for the whole second and a half.

So the rule is not "hold possession with the kicker". It is: **cut the ball's track at every
touch, and attribute each piece to whoever played it.** The ball's own trajectory marks the
touches: it loses pace or changes course, and only a player in contact can do that to it.

### What it does

`ball_flight()` computes the ball's velocity between consecutive observations of the *masked*
ball, then:

1. **Finds the touches.** A frame is a touch when pace drops to 60% or less of the previous
   frame's, or the course turns by more than 60 degrees, *and* a player is within 1.5 m. Nothing
   else can change a ball's trajectory.
2. **Names the toucher.** The nearest player, and only if he is 1.5 m clearer than the nearest
   opponent. Two players equally on it is a tackle, not a touch, and the rule abstains.
3. **Holds that team while the ball is nobody's.** From the touch until the next one, for as
   long as the ball is above 5 m/s and at most 2.5 s. Above that pace the nearest player cannot
   be in possession however close he is standing; below it the classifier is the better judge.

### What it cost to get right, measured at each step

| rule | cases fixed | new | spellF1 |
|---|---|---|---|
| nearest player at the launch, imposed on every moving ball | 382 | 538 | 0.445 |
| the same, but only allowed to *keep* possession, never move it | 95 | 101 | 0.442 |
| + a speed floor and a clean-launch test | 65 | 23 | 0.446 |
| + touches found inside the flight, keep only | 83 | 25 | 0.447 |
| + touches allowed to take possession (**shipped**) | **325** | **176** | **0.482** |

The first row is the naive reading of "the closest player passed it": it invents a turnover at
every struck ball whose launch frame names the wrong player, which a tackle or a deflection
regularly does. The jump in the last row is not a looser rule. It is the same rule finally
allowed to do the thing that was needed, once the touch test made the switch evidence-based
rather than proximity-based.

### The deflection gate from the brief, measured and left off

The brief said a brief deflection is weak evidence of a turnover, so a touch was made to earn a
*switch* by keeping the ball for 0.4 to 0.8 s and 3 to 8 m:

| switch gate | fixed | new | spellF1 |
|---|---|---|---|
| none (**shipped**) | **286** | **162** | **0.479** |
| 0.4 s and 3 m | 257 | 146 | 0.474 |
| 0.6 s and 6 m | 243 | 131 | 0.476 |
| 0.8 s and 8 m | 224 | 115 | 0.474 |

(measured at the 7 m/s speed floor, before it was lowered to 5.) Every step of the gate trades
about two fixes for one regression avoided, so it is left off, kept as `SWITCH_MIN_S` and
`SWITCH_MIN_M`. The reasoning in the brief is sound; the data says the deflections it worries
about are rarer than the real touches the gate would suppress.

### What the touch rule did, end to end

| metric | restart layer only | with the touch rule | change |
|---|---|---|---|
| spellF1 | 0.447 | **0.482** | **+0.035** |
| macroF1 | 0.859 | **0.865** | +0.006 |
| missed possession switches | 1200 | **1048** | **-152** |
| false possession switches | 409 | 497 | +88 |
| failure cases | 2885 | **2817** | -68 |
| cases the model gets cleanly right | 283 | **317** | +34 |

Against the pre-hardening baseline: **325 cases fixed, 176 new**, spellF1 0.438 to 0.482 and
macroF1 0.855 to 0.865. All five development matches improve, by 0.025 to 0.061 spellF1.

**This is the first change in this project to move spellF1 outside its own noise band**
(+0.044 against a band of 0.04), and macroF1 lands exactly on its 0.01 band.

The cost is concentrated and visible: 135 of the 176 new cases are `FALSE_CHANGE`, a trajectory
break read as a touch when it was a deflection, a near miss, or tracking noise. Every one of
them carries that explanation in the review page, under the *NEW since the latest change*
filter.

### The two frames that started it

`J03WOY_P1_WRONG_TEAM_12138` and `_12830`, the two with reviewer comments on them, are both
gone. The switch now lands at frame 12139 against the reference's 12138: one frame, 0.04 s, late.

---

## The whistle: reading the stoppage off the players, not the ball

Added 5 November 2026, from a second reviewer note: at a stoppage every player drops to a walk
and stays there, and the model never looks at that.

### The observation is right, and bigger than it looked

Measured over all 715,116 frames of the five development matches, the mean speed of every
tracked player is:

| | mean player speed |
|---|---|
| while the reference calls the ball **dead** | **1.10 m/s** |
| while it calls the ball **live** | **2.42 m/s** |

A walk is about 1.4 m/s, so the reviewer's reading of the video is exactly what the numbers say.
It holds in all ten half-matches, and the per-match typical pace ranges from 2.10 to 2.43 m/s,
which is why every threshold below is expressed as a fraction of the match's own baseline rather
than in m/s.

### But it is a slow signal, and that is what decides how it can be used

The separation is large; the timing is not. Measured against the reference's own stoppages, the
region where team pace sits below 60% of the match baseline **starts a median 4.56 s after the
real whistle**. Players keep running until play has plainly stopped. So the low-pace region is a
lagged, shrunken copy of the stoppage, and anything built on *where it begins* is worse than what
stage 2 already does:

| who times the whistle | median \|error\| | within 1 s |
|---|---|---|
| stage 2, as it ships | 0.80 s | 56% |
| the pace-low region's own start (ratio 0.60) | 4.68 s | — |
| the pace-low region's own start (ratio 0.70) | 2.00 s | — |

### Three wider versions, all measured and all rejected

**Declaring a new stoppage wherever everybody walks.** This is the rule in its literal form. A
sustained walk inside live play is common football — a goalkeeper holding the ball, a back four
passing it around — so it is mostly wrong:

| gate | windows | precision |
|---|---|---|
| ratio ≤ 0.50 held 1.5 s | 35 | 24% |
| ratio ≤ 0.50 held 2.5 s, ball also still | 11 | 36% |
| ratio ≤ 0.60 held 1.5 s | 241 | 14% |
| ratio ≤ 0.60 held 4.0 s, ball also still | 13 | 36% |

At best it is wrong twice for every time it is right. There is also very little left for it to
find: of 503 true stoppages, stage 2 misses only **14 entirely**. The other 378 it finds and
mistimes, which is an edge problem, not a detection problem.

**Extending a stoppage outwards while the pace stays low.** 37–46% precision backwards from the
whistle, 18–26% forwards from the restart. Both add more live frames than dead ones.

**The second half of the rule: a sudden rise in pace as the restart.** The reviewer's symmetry is
natural and the data does not support it — at the restart there is better evidence, namely
somebody kicking the ball, which `align` already uses:

| who times the restart | median \|error\| | within 1 s |
|---|---|---|
| the ball rule, as it ships | **0.36 s** | **75%** |
| the steepest rise in team pace | 0.96 s | 54% |
| the two averaged | 0.58 s | 73% |

Even on the 72 restarts where the ball is masked through the entire window, so the ball rule has
nothing to go on, pace is worse: 0.28 s against 0.86 s. It is not shipped.

### What does ship: a second reading of the whistle

The one place pace is as good as what we have is the **whistle itself**, taken not as a level but
as the *steepest sustained fall* in team pace — the reviewer's "sudden drop", measured as the
mean pace over the second before a frame minus the mean over the second after it. The whistle is
then placed halfway between stage 2's edge and that frame, searched within 2 s of it, and never
outside its own stoppage.

| who times the whistle | median \|error\| | within 1 s |
|---|---|---|
| stage 2 alone (what shipped before) | 0.80 s | 56.2% |
| the steepest fall alone | 0.76 s | 56.9% |
| **the two averaged (shipped)** | **0.70 s** | **61.3%** |
| 25% pace | 0.78 s | 59.9% |
| 75% pace | 0.70 s | 61.3% |

Two independent readings of one event, neither better than the other, and the average better than
both. Gating it — only moving the edge on a large drop, or only where the players really reach
walking pace — makes it *worse*, because then most whistles lose the averaging. The gain is
largest exactly where it should be, on the 152 whistles where the ball was masked (0.88 s
`→` 0.65 s), because there player movement is independent evidence rather than a
second look at the same thing.

Over 547 stoppages the edge moves a median of 0.48 s, earlier 81% of the time, capped at 1.0 s.

### What the whistle rule did, end to end

Against the state that shipped before it (the restart layer plus the touch rule):

| metric | before | after |
|---|---|---|
| spellF1 | 0.482 | 0.485 |
| macroF1 | 0.865 | 0.866 |
| restart edge \|err\| | 1.311 s | **1.292 s** |
| false possession switches | 497 | **476** |
| missed possession switches | 1048 | 1060 |
| failure cases | 2817 | **2780** |

**85 cases fixed, 48 new, net 37.** By kind: `BOUNDARY` +15 (the stoppage-timing kind this rule
aims at), `FALSE_CHANGE` +21, `WRONG_TEAM` +11, `MISSED_CHANGE` −12.

Both score moves are **inside their noise bands** (+0.003 against 0.04 for spellF1, +0.001
against 0.01 for macroF1), and two of the five matches go very slightly down (−0.005 and −0.001).
What is not noise is the thing the rule was built to do: whistle timing over 489 stoppages,
0.80 s `→` 0.70 s and 56% `→` 61% inside a second. This is a small,
well-aimed change, and it is reported as one.

### Against the pre-hardening baseline, with everything on

spellF1 0.438 `→` **0.485**, macroF1 0.855 `→` **0.866**, restart edge
1.446 s `→` **1.292 s**, restart team 430 `→` 434 of 460, missed
switches 1206 `→` **1060**, 2943 failure cases `→` **2780**. Cases:
**399 fixed, 213 new**.

---

## The ball lying still: the one signal that is allowed to find a stoppage

Added 6 November 2026, from a third round of reviewer notes. Watching J03WOH at 30:10 the
reviewer wrote: *"the ball is stopped and seems like nobody is in contact with the ball to stop
it, that means in the last few frames something happened such as a foul, which made the ball
stop — means the ball is dead from that moment."*

### Unlike the players slowing down, this one is never ambiguous

Measured on frames the model currently calls **live**, a visible ball moving slower than 0.5 m/s
with no player inside contact range (`CONTROL_R`, 3 m), held for at least a second:

| gate | windows | frames | the reference calls them dead |
|---|---|---|---|
| held 1.0 s | 86 | 4,426 | **100%** |
| held 1.5 s | 49 | 3,308 | **100%** |
| held 2.5 s | 18 | 1,795 | **100%** |
| contact range relaxed to 2 m, held 1.0 s | 119 | 5,986 | 98% |

A football does not stop on its own in open play. That is why this rule is allowed to do the
thing the pace rule was refused: **find a stoppage stage 2 never saw**, rather than only re-time
one. It is folded into the dead mask before anything else runs, so the whistle, the restart
alignment and the carried team all treat it as the ordinary stoppage it is.

The ball has to be **visible throughout**. A ball that is not seen to move is also exactly what a
masked ball looks like, and that is evidence of nothing.

### What it did

| metric | before | after |
|---|---|---|
| spellF1 | 0.485 | 0.487 |
| macroF1 | 0.866 | **0.871** |
| stoppages found | 489 of 503 | **492 of 503** |
| false possession switches | 476 | **467** |
| failure cases | 2,780 | **2,752** |

**38 cases fixed, 10 new, net 28.** By kind: `BOUNDARY` +14, `FALSE_CHANGE` +9, `MISSED_STOP` +5
(16 → 11), `WRONG_TEAM` +5, `INVENTED_STOP` −5. macroF1 moves +0.005 against a
0.01 band, so it is still inside the noise; the case counts and the three extra stoppages are not.

### Two versions measured beside it and rejected

**Rolling the stoppage back to where the ball started slowing.** The same note says the foul
happened *before* the ball came to rest, so the dead window should start earlier. It cannot: from
the start of an at-rest window, walking back while the ball is merely slowing and still untouched
adds **0 frames** at 2 m/s, 1 frame at 3 m/s and 117 frames at 5 m/s — **none of them dead**.
The ball goes from struck to stopped faster than the tracking resolves, or it is masked on the way
down. There is nothing there to take.

**Naming the throw-in taker by who touched the ball last.** Four notes make the same Laws point:
a pass deflected out by a defender is the attacker's throw. The last touch *is* now read off the
ball's own trajectory (`lasttouch`), and it changes the `last_control` and `restart_team` fields
the review page shows. But **imposing** that team on the frames after the throw is much worse:

| | cases fixed | cases created |
|---|---|---|
| hold the throw-in taker for 0.8 s | 11 | **120** |

spellF1 falls 0.487 → 0.475 and false switches rise 467 → 531. The Law
names the taker correctly; the trouble is that a throw-in is contested inside a second, and the
decoder reads that contest better than the Law reads the second after it. So the last touch is
recorded and shown, and never imposed.

---

## A moving ball means play is going on

Added 6 November 2026, from a fourth round of reviewer notes. Four of them were the same
complaint in different words: a keeper holding the ball and playing on, a ball being rotated
between team-mates, players and ball both moving, a side slowing the game down with everyone
still active — and the model calling all four a stoppage, every time because the ball had gone
out of sight.

### The ball says yes or no; the players say nothing

Measured inside the regions the model currently calls dead:

| evidence of play | windows | frames | the reference calls them live |
|---|---|---|---|
| visible ball moving ≥ 1.5 m/s for 0.5 s | 198 | 5,864 | **100%** |
| visible ball moving ≥ 2.0 m/s for 0.5 s | 194 | 5,495 | **100%** |
| visible ball moving ≥ 3.0 m/s for 0.5 s | 176 | 4,713 | **100%** |
| players at ≥ 80% of match pace for 1 s | 181 | 15,970 | 11% |
| players at ≥ 90% of match pace for 1 s | 111 | 9,198 | 12% |
| players at ≥ 100% of match pace for 1 s | 70 | 5,339 | 15% |

The same asymmetry as before: **the ball is decisive in both directions** — at rest it means
dead, moving it means live, both at 100% — while the players, who were a usable second reading
of the whistle, are worthless here. Twenty-two people jog about during a stoppage.

So a visible ball moving at 1.5 m/s or more **inside the field of play** for half a second vetoes
the stoppage. The ball must be inside the pitch as well as moving: a ball still rolling after it
has crossed the line is moving and dead at the same time.

### Getting the veto right took three attempts, and the first two were worse

| how the vetoed frames are handled | cases fixed | cases created | net |
|---|---|---|---|
| cut the moving frames out of the stoppage | 52 | 99 | **−47** |
| keep the stoppage's longest surviving stretch, carry the team from before it | 52 | 69 | **−17** |
| **the same, but the team comes from the side the frames sit on** | **52** | **37** | **+15** |

Cutting a hole in a stoppage leaves a piece either side of it, and two pieces are two invented
stoppages where there was one. Keeping only the longest stretch fixes that. The last step matters
most: frames handed back **before** a stoppage belong to whoever had the ball, frames handed back
**after** one belong to whoever is about to restart, and treating them alike cost 43 wrong-team
and 35 false-change cases on its own.

### And a ball that has left the pitch gives possession to nobody

From the same round: *"model is checking the nearest player so it changed the possession, but
actually the ball went out — unless the player received the ball, do not change possession."*
The nearest-player vote is now simply not counted while the visible ball is outside the field of
play. One case fixed, none created — small, but it is a Law rather than a threshold, and it
costs nothing.

### Measured and rejected: the keeper holding the ball

One note describes a keeper who has the ball in his hands and is looking to play, which the model
calls dead. "Ball visible, inside the pitch, a player in contact with it" would catch it, but it
is not evidence of play:

| gate | windows | frames | truly live |
|---|---|---|---|
| a player within 1.0 m, held 2 s | 70 | 6,451 | 49% |
| a player within 1.5 m, held 2 s | 124 | 11,960 | 31% |
| a player within 2.0 m, held 2 s | 165 | 16,631 | 25% |

A player standing over a stationary ball is usually about to take a free kick. A coin toss is not
a rule, so this one is left out and the keeper case stays unfixed.

### What the round did, end to end

| metric | before | after |
|---|---|---|
| spellF1 | 0.487 | 0.487 |
| macroF1 | 0.871 | **0.875** |
| right team on live frames | 83.5% | **84.5%** |
| restart edge \|err\| (mean) | 1.26 s | **1.04 s** |
| whistle timing (median) | 0.64 s | **0.56 s** |
| team right after a restart | 434 of 467 | **446 of 482** |
| failure cases | 2,752 | **2,736** |
| cases it gets right | 345 | **373** |

`BOUNDARY` falls 212 → **172**, the biggest single move this layer has produced, and
it is exactly what the notes described: stoppages that ran on through live play.

---

## The blind spot: joining the ball up across the gaps

Added 8 November 2026. The ball is hidden from the model **41% of the time**, and the hidden
stretches fall exactly where possession is decided — in the middle of passes. Every rule built
so far goes quiet there, which is why most of the clips the reviewer flagged were still failing
after four rounds of rules.

### A straight line between two sightings is a very good guess

The development matches carry the real ball on every frame and a mask that hides it the same way
the competition data does. The model only ever reads the masked ball; the hidden positions are
used here **purely as a measuring stick**, to ask how far a straight line between the sighting
before a gap and the sighting after it lands from where the ball actually was. Over 4,410 gaps:

| gap length | gaps | hidden frames | median error | p90 error |
|---|---|---|---|---|
| ≤ 0.2 s | 447 | 1,007 | **0.01 m** | 0.03 m |
| 0.2–0.4 s | 358 | 2,502 | **0.01 m** | 0.12 m |
| 0.4–0.8 s | 544 | 7,724 | **0.03 m** | 0.56 m |
| 0.8–1.6 s | 775 | 22,359 | **0.17 m** | 1.90 m |
| 1.6–3.2 s | 891 | 50,321 | 0.91 m | 4.08 m |
| > 3.2 s | 1,395 | 209,242 | 3.32 m | 10.66 m |

Up to about three seconds the guess is good to within a metre, which is well inside the distance
that separates two players. Beyond that the ball could be anywhere.

### Where the bridged ball is allowed to speak, and where it is not

It is handed to the rules that **follow the ball** — the flight and touch logic, and the
moving-ball veto — and withheld from the per-frame nearest-player vote, which must keep
speaking only about a ball it can actually see. A guess should be allowed to continue a pass; it
should not be allowed to become an opinion about who is in possession.

The cap was swept rather than argued:

| longest gap bridged | cases fixed | cases created | net |
|---|---|---|---|
| 1.6 s | 102 | 41 | +61 |
| **3.2 s** | **137** | **64** | **+73** |
| 6.4 s | 181 | 129 | +52 |

At 6.4 s the extra fixes are real but the stoppage edges start to go: `BOUNDARY` turns from +4 to
−25, because a long straight line through a hidden stretch convinces the still-ball and
moving-ball rules of things that did not happen.

### What it did

This is the largest single change the restart layer has had.

| metric | before | after |
|---|---|---|
| spellF1 | 0.487 | **0.501** |
| macroF1 | 0.875 | **0.877** |
| **false possession switches** | 474 | **444** |
| missed possession switches | 1,053 | **1,038** |
| failure cases | 2,736 | **2,662** |

**137 cases fixed, 64 new.** `FALSE_CHANGE` +30, `WRONG_TEAM` +21, `MISSED_CHANGE` +15 — and it
is the first change in this whole layer that pushes false switches **down**, because a pass that
vanishes mid-flight no longer looks like two possessions with a turnover between them.

---

## Eleven a side: the player standing off the pitch is the one taking the restart

A reviewer's rule, from watching throw-ins: *"unless there is a red card both teams have eleven
players, so if one side is a man short — especially at a throw-in, a corner or a free kick from
outside — that missing player is the one taking it."*

It is the most accurate restart-team signal in the model:

| | fires on | names the right side |
|---|---|---|
| a player standing **off the pitch** within 10 m of the resting ball | 155 of 493 restarts | **95%** |
| the side with **fewer players inside the pitch** | 237 of 493 restarts | 86% |
| what the model itself says, on those same 155 restarts | — | 85% |

Where the head-count and the model disagree, the head-count is right **16 times out of 17**. So
it now names the taker: it sets `restart_team`, and it overrides the corner classifier's own
guess where the two differ.

### What it may not do is impose that team on the play

Naming the taker changes no labels at all, and every attempt to make it change labels was worse:

| | cases fixed | cases created |
|---|---|---|
| hold the named team for 0.8 s at every restart it fires on | 16 | 109 |
| hold it only where the model is blind at the restart | 4 | 32 |
| hold the throw-in taker named by the last touch (5 Nov) | 11 | 120 |

Three separate attempts, the same answer: a restart is contested inside a second, and the
per-frame model reads that contest better than any rule about who took it. The Laws name the
taker; they do not name who has the ball a second later.

---

## Wrong team: where the pile actually is

Added 8 November 2026. Wrong team is the second largest kind and the one with the most
**confirmed** cases, so it was worth profiling before writing another rule. Over the 388,413
frames where the model and the reference both call the ball live, 12.6% get the wrong side. What
makes that rate move:

| when ... | frames | wrong inside | wrong outside | lift |
|---|---|---|---|---|
| **the ball is in flight with a known kicker** | 93,454 | **3.4%** | 15.5% | **0.22×** |
| stage 2 was unsure of the team (0.4 < p < 0.6) | 26,244 | 36.0% | 10.9% | 3.31× |
| the ball is not visible to the model | 157,936 | 16.1% | 10.2% | 1.57× |
| the k-NN margin is thin (< 0.25) | 45,524 | 16.1% | 12.1% | 1.33× |
| within 1 s of a restart | 11,874 | 6.4% | 12.8% | 0.50× |

The first row is the finding. **The model is nearly five times better while it is following a
struck ball than at any other time.** So the question for this pile is not what new signal to
invent, but why the flight rule is silent so much of the time.

### The flight gates were tuned before the ball was bridged

`FLIGHT_HOLD_V` — how fast the ball must be moving for "the nearest player cannot have it" to
hold — was set to 5 m/s back when the ball track was full of holes. With the gaps joined up
there is far more continuous flight to follow, so it was swept again:

| | spellF1 | in flight | failure cases |
|---|---|---|---|
| 5.0 m/s (as it was) | 0.501 | 13.1% | 2,662 |
| 4.0 m/s | 0.507 | 14.5% | 2,642 |
| **3.0 m/s (shipped)** | **0.507** | **15.7%** | **2,635** |
| 2.5 m/s (and `FLIGHT_V_END` with it) | 0.510 | 16.1% | 2,627 |

3.0 m/s is where the hold meets `FLIGHT_V_END`, the speed below which this model already says a
ball can be under somebody's foot — so the rule now speaks for **the whole of a moving run**
rather than only its fast part. Going lower means dropping that line too, and 2.5 m/s buys 8
cases for a claim the rule cannot really make. Lengthening a flight instead (`FLIGHT_MAX_S`
2.5 → 3.5 s) trades evenly and was not taken: +11 `FALSE_CHANGE` against
−11 `MISSED_CHANGE`.

### The still-ball radius, measured end to end instead of window by window

A ball at rest counts as dead when nobody is within `STILL_R` of it. That was 3 m, borrowed from
`CONTROL_R`, because at 3 m every one of the 86 windows it finds is genuinely dead while 2 m
finds 119 at 98%. End to end the looser radius is simply better — **25 cases fixed against 11
created** — so it is now 2 m and `STILL_R` is its own number. The tidier precision figure lost
to the measurement that counts.

### Measured and found to change nothing

Three gates on the possession switch — `MARGIN_MIN` (0.25 → 0.35, 0.45),
`PERSIST_S` (0.6 → 0.9 s) and `PERSIST_FRAC` (0.6 → 0.75) — produce
**byte-identical output**. They only bite inside a restart hold, and the only restart the model
holds is a strict corner. They are live code reached a few dozen times a match, not the knobs
they look like.

### Inside a blind stretch, the model already beats the simple answers

Half of all wrong-team frames are frames with no visible ball, so it is worth knowing what the
alternatives are. Over 2,697 blind gaps with a sighting either side (162,964 live frames):

| gap length | carry the team from BEFORE | take the team seen AFTER | half-and-half | **the model now** |
|---|---|---|---|---|
| 0–1.6 s | 79% | 73% | 76% | **86%** |
| 1.6–3.2 s | 82% | 73% | 79% | **83%** |
| 3.2–6.4 s | 78% | 72% | 78% | **80%** |
| 6.4–12.8 s | 74% | 64% | 74% | **81%** |
| over 12.8 s | 62% | 60% | 64% | **78%** |

Where the two sightings name **different** teams — 713 gaps, 56,043 frames — carrying is right
61% of the time and the model is right 74%. There is no simple policy to swap in here; the
flight, touch and carry machinery is already doing better than any of them.

---

## Missed changes of hands: the model never listens to the ball

Added 8 November 2026. The largest pile, 1,032 of them. The obvious suspect was the
possession-length floor — a change of hands into a possession shorter than 0.8 s is deleted as
a wobble — and the obvious suspect is innocent:

| how long the possession it never joins actually lasts | count | share |
|---|---|---|
| under 0.4 s | 52 | 5% |
| 0.4–0.8 s | 85 | 8% |
| 0.8–2.0 s | 239 | 23% |
| 2–4 s | 226 | 22% |
| **over 4 s** | **430** | **42%** |

The median possession the model fails to join lasts **3.1 s**. These are not slivers; they are
ordinary turnovers, and 42% of them are longer than most possessions in the match.

### What was knowable at the moment it should have switched

| | cases | share |
|---|---|---|
| the ball was visible on at least half the frames | 538 | 52% |
| the ball was not visible at all | 390 | 38% |
| **the nearest-player vote pointed at the NEW team** | **460** | **45%** |
| the ball was in flight | 158 | 15% |

Nearly half the time the one cue that is actually about where the ball is right now was pointing
the right way, and the model went on regardless. The reason is structural: **outside a restart
window the refined sequence is the per-frame team model's own output.** The nearest-player vote
is consulted inside restart holds and in the flight logic, and nowhere else. It is never allowed
to take the ball off the model during open play.

### Letting it take the ball, where it has earned the right

Over every window where the vote is firm, sustained and disagrees with the model, who turns out
to be right depends almost entirely on **whether anybody is actually next to the ball**:

| the nearest player is ... | windows | frames | vote right | model right |
|---|---|---|---|---|
| any distance away | 300 | 7,839 | 41% | **59%** |
| within 3 m | 184 | 4,464 | **55%** | 45% |
| within 2 m | 125 | 2,874 | **64%** | 36% |

At arm's length the vote is right two times in three; at a distance it is worse than the model.
So the vote may override, and only there. Swept end to end:

| | cases fixed | cases created | failure cases |
|---|---|---|---|
| **1.5 m, held 0.6 s (shipped)** | **74** | **45** | **2,597** |
| 2.0 m, held 0.6 s | 97 | 69 | 2,597 |
| 3.0 m, held 0.6 s | 147 | 136 | 2,617 |
| 2.0 m, held 1.0 s | 39 | 43 | 2,625 |
| 2.0 m, margin ≥ 0.50 | 49 | 36 | 2,609 |

1.5 m and 2.0 m land on the same total; 1.5 m gets there with 24 fewer broken cases, and it is
the same arm's length the touch rule already uses for "a player has to be this close to have
played it". 3 m is a coin toss dressed up as a rule.

**74 cases fixed, 45 new.** `MISSED_CHANGE` +19, `WRONG_TEAM` +22, `FALSE_CHANGE` −12 — the
trade this rule makes, openly: it buys changes of hands with a few switches that should not have
happened.

---

## False switches: a round that shipped nothing

Added 8 November 2026. 439 switches the reference does not make. Three separate attacks on them,
all measured, none kept — and the reason is worth more than another rule would have been.

### First: they are mostly not the hardening layer's fault

| | switches | share |
|---|---|---|
| the per-frame team model switched here anyway | 322 | **73%** |
| a hardening rule inserted it | 117 | 27% |

And the possessions they invent are not slivers — median **5.1 s**, 64% of them longer than
3.2 s — so no length floor reaches them. Of the 117 a rule does insert, 71% are the touch rule
reading a deflection off a ball in flight.

### A team cannot take a ball nobody of theirs is near

Of the 977 switches the model makes, 55% are real. Split by how close the side it switches *to*
actually was:

| at the moment of the switch | switches | real |
|---|---|---|
| nobody of the new team within 5 m | 38 | **16%** |
| nobody within 3 m | 90 | **26%** |
| nobody within 2 m | 133 | 32% |
| the nearest-player vote disagrees | 421 | 43% |
| **all switches** | 977 | 55% |

Blocking the 3 m ones saves 67 false switches and costs 23 real ones — a 3:1 trade, which by
the standard this project has used all week is a rule worth having. **It is not**, and this is
the round where the case count and the score came apart:

| | spellF1 | macroF1 | right team | failure cases |
|---|---|---|---|---|
| as it ships | **0.511** | **0.881** | **87.7%** | 2,597 |
| block within 2.5 m | 0.474 | 0.842 | 81.7% | **2,517** |
| block within 3.0 m | 0.483 | 0.853 | 83.4% | 2,523 |
| block within 4.0 m | 0.490 | 0.859 | 84.3% | 2,546 |
| block within 5.0 m | 0.502 | 0.871 | 86.1% | 2,553 |

Eighty fewer failure cases, and **six points of team accuracy gone**. Blocking a switch does not
delete it — it hands those frames to the team that had the ball before, and that team is wrong
for every frame until the model switches again. The case list counts events; the score counts
frames and whole possessions. The score is what is graded.

Gating the block on the ball's own vote disagreeing too narrows it without rescuing it:

| | spellF1 | macroF1 | right team | failure cases |
|---|---|---|---|---|
| block within 3 m, vote disagrees | 0.504 | 0.869 | 85.7% | 2,547 |
| block within 5 m, vote disagrees | 0.510 | 0.878 | 87.2% | 2,573 |
| block within 8 m, vote disagrees | 0.510 | 0.879 | 87.4% | 2,590 |

Every row is worse on all three graded measures. Even the mildest version — refusing a switch
only when the new team is more than **8 m** from the ball, which is nearly absurd on its face
— still loses. The rule is implemented as `--parts reach`, off by default, so the measurement
can be repeated.

### And tightening what counts as a touch makes it worse too

The touch rule is behind 71% of the rule-inserted false switches, so the obvious move is to ask
more of a touch before believing it:

| | spellF1 | failure cases |
|---|---|---|
| **as it ships** (within 1.5 m, pace drops to 0.6, course turns 60°) | **0.511** | **2,597** |
| the player has to be within 1.0 m | 0.509 | 2,617 |
| the pace has to drop to 0.45 | 0.511 | 2,598 |
| the course has to turn 78° | 0.506 | 2,622 |
| within 1.0 m and pace to 0.45 | 0.508 | 2,618 |

All five worse. The touch gates are already at a local optimum.

### What this round actually established

Three quarters of the false switches belong to the per-frame team model, not to anything built
on top of it, and **every attempt to suppress a switch from outside costs more than it saves**.
A switch the model makes carries information even when it is wrong: it is usually right about
*something* happening and wrong about exactly when or to whom, and flattening it loses the right
part along with the wrong one. This pile does not move with another rule in this layer. It moves
by changing the model that produces the switches, trained on external data.

---

## A ball outside the lines is nobody's, and a sliver of play is not play

Added 9 November 2026, from a fifth round of reviewer notes. Four of the eight said the same two
things in different words, and both turned out to be Laws the model had never been told:

> *"Ball is still and also out of boundary areas, so any team can't have the possession."*
> *"Between 2 dead phases a team possession is very unlikely, especially when the ball is still."*

### The ball outside the lines

The model already refused to let the nearest-player vote speak while the ball was outside the
pitch, but it never concluded from that that the ball was **dead**. Measured:

| | frames | the reference calls them dead |
|---|---|---|
| every frame where the visible ball is outside the pitch | 104,376 | **99%** |
| ... of those, the ones the model still calls live | 3,838 | 87% |
| ... and the ball barely moving, held 1 s | 1,932 | **95%** |

The speed test matters: a ball being thrown back in is outside the lines and moving, and that is
the restart, not the stoppage. There is no distance-to-player test, deliberately — at a
throw-in the taker is standing right next to it, which is exactly the case the still-ball rule
(which needs nobody within 2 m) cannot see.

**10 cases fixed, 4 new.** Small, but it is a Law rather than a threshold, and `MISSED_STOP`
falls by 3.

### Slivers of play between two stoppages

The other note is about the gap the model sometimes leaves between two stoppages it has already
found. Measured over all 592 of them:

| length of the stretch | stretches | frames | really dead |
|---|---|---|---|
| under 1 s | 20 | 199 | **100%** |
| 1–2 s | 19 | 765 | 72% |
| 2–4 s | 44 | 3,335 | 47% |
| 4–8 s | 70 | 10,283 | 18% |
| 8 s and over | 439 | 385,573 | 3% |

With the ball barely moving or never seen, the 2–4 s band goes to 84–89%. So a stretch under
**2 s** between two stoppages is closed outright, and one under **4 s** is closed when the ball
never moved more than 2 m or was never seen at all.

Swept: 1/2 s fixes 20 cases net, **2/4 s fixes 30**, 3/6 s drops back to 25. The two rules
together:

| | spellF1 | macroF1 | failure cases |
|---|---|---|---|
| before | 0.511 | 0.881 | 2,597 |
| the ball outside the lines only | 0.512 | 0.882 | 2,591 |
| slivers only (2/4 s) | 0.518 | 0.884 | 2,567 |
| **both (shipped)** | **0.519** | **0.884** | **2,561** |

`BOUNDARY` is where it lands — these are stoppages the model had found and then cut in half.

---

## What shipped, and what it did

`--parts align,corner,flight,whistle,still,lasttouch,inplay,outside,headcount,bridge,takeover,outdead,island --corner-hold 0.8 --flight-mode both` (the defaults). The restart
numbers below are the restart layer on its own; the touch rule's effect on top of it is in the
section above.

| metric | before | after | change |
|---|---|---|---|
| spellF1, end to end | 0.438 | 0.446 | +0.008 (inside the 0.04 noise band) |
| macroF1, end to end | 0.855 | 0.858 | +0.003 (inside the 0.01 band) |
| restart edge, mean absolute error | 1.446 s | **1.311 s** | **−0.134 s, −9.3%** |
| restart team right, of 460 | 430 | **436** | **+6** |
| false possession switches | 431 | **419** | **−12** |
| missed possession switches | 1206 | 1206 | 0 |

All five matches improved on spellF1. Both headline metrics moved inside their noise bands, so
**the honest claim is not "spellF1 improved"** — it is that restart timing improved by a
measurable 6.3% and 30 individual failures went away, at no cost to the headline numbers.

### Individual cases: 34 fixed, 3 new

Net counts hide trades, so cases are matched one to one — same kind, same period, overlapping
span — and counted as fixed or new:

| kind | fixed | new | net | unchanged |
|---|---|---|---|---|
| BOUNDARY | 19 | 0 | **+19** | 241 |
| FALSE_CHANGE | 13 | 1 | **+12** | 418 |
| MISSED_CHANGE | 2 | 2 | 0 | 1204 |
| WRONG_TEAM, INVENTED_STOP, MISSED_STOP, SPELL_MERGE | 0 | 0 | 0 | 1046 |
| **total** | **34** | **3** | **+31** | 2909 |

2,943 failure cases before, 2,912 after. The three new ones are tagged `new_since_fix` in the
case index and have their own filter on the review page, each carrying the measured difference
that produced it.

### The three cases the change created

| case | cause | what happened |
|---|---|---|
| `J03WPY_P2_FALSE_CHANGE_146469` | edge + rule | the edge moved 0.32 s earlier and the restart rule then held a team across half the window; it says home where the reference says away |
| `J03WQQ_P2_MISSED_CHANGE_103310` | possession floor | the model used to switch 0.24 s early; the possession either side was 0.2 s long and the 0.8 s floor merged it away |
| `J03WR9_P2_MISSED_CHANGE_123544` | possession floor | the same, with a 0.6 s possession |

Two of the three are the **length floor**, not the nearest-neighbour rule. 0.8 s is a blunt
instrument — a real touch can be shorter and a wobble can be longer — so it should be replaced by
something that weighs the evidence for a touch rather than counting frames. Both sit on a
contested reference. **Every reduction is in the `confirmed` column** — cases
where the DFL's two reference products agree — which is the only column that counts as a model
error under this project's own rules. Cases the model now gets cleanly right rose from 247 to
276 (`STOPPAGE_CAUGHT` 103 → 120, `SPELL_HELD` 140 → 152).

---

## Known residuals

- **Corners are only classified when the ball is visible at the end of the stoppage.** At the
  competition's 41% masking rate a share of restarts have no reliable ball, and those keep stage
  2's edge and are flagged `uncertain`.
- **The Dry Run corner is now read correctly end to end** — stoppage held at `D`, away given the
  restart, resume detected at frame 12767 against a reference restart at 12776 — but the
  possession 2.3 s later (`J03WOH_P1_WRONG_TEAM_12823`) is still wrong. That case sits on a
  *contested* reference: the per-frame field says home, the event feed does not agree.
- **11 stoppages per five matches are still missed entirely.** The ball-at-rest rule
  found 3 of the 14; the rest are stoppages where the ball is masked or is being held by a
  player, so neither the ball nor the players give the model anything to see.
- **The whistle rule cannot see a stoppage stage 2 missed.** It only re-times edges stage 2
  already found, because every version that looked for a stoppage of its own measured between
  14% and 36% precision. The 14 stoppages per five matches that stage 2 misses entirely are
  still missed.
- **Player pace says nothing about the restart.** The rise back to match pace is both later and
  vaguer than the ball being kicked, so the second half of the reviewer's rule is measured and
  left out. Where the ball is masked through the whole resume window, the model still falls back
  to stage 2's edge and marks the window uncertain.
- **The touch rule abstains wherever the ball is masked.** It reads only the ball the model
  is allowed to see, so at the competition's 41% masking rate a share of touches are simply not
  observable, and those windows keep the classifier's answer.
- **A deflection still reads as a touch.** The trajectory test cannot tell a controlled first
  touch from a block, and the 135 new `FALSE_CHANGE` cases are exactly that. Separating them
  needs something the trajectory does not contain, namely whether the toucher keeps the ball
  afterwards, and the two gates measured for that both cost more than they saved.
- **Throw-ins remain the model's job.** The rule is 80.7% against the model's 88.1%, so the
  "pass to a teammate" assumption from the brief is implemented but not enabled; it is available
  as `--parts align,restart`.

## Reproducing

```bash
python stage3/restart_logic.py --compare --apply
python review/export_cases.py && python review/patterns.py
python review/evaluate.py --compare baseline hardened
```

Ablations: `--parts align`, `--parts align,corner` (no touch rule), `--flight-mode hold`,
`--parts align,corner,gap`, `--parts align,restart,gap`, `--corner-hold 1.5`, `--resume-win 2.5`.
