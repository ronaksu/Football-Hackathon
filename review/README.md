# Model Failure Review — local tool

A local page for watching the model fail, frame by frame, with the data that produced each
decision beside it, and a place to record what you see.

Everything here is built from the **open-access IDSSE development matches** (Bassek et al., Nature
Scientific Data 2025). The two sealed test matches are never loaded, no competition asset is read,
and nothing recorded here feeds training.

---

## Running it

```bash
python review/server.py
```

Then open <http://127.0.0.1:8765/app.html>. The server binds to loopback only, so it is not
reachable from the network.

The page is written to be read without a glossary: cases are named in plain words (*Missed a
stop*, *Wrong team on the ball*), every case opens with one sentence saying what went wrong, and
the detail is folded away until you ask for it.

**Start on the `Watch this first` tab.** It plays `demo/model_demo.mp4` (2 min 48 s) beside a contents
list. Two stoppages from the same half of the same match, almost the same length: one caught to
within a fifth of a second, one missed completely. Every number in it is computed from the case
bundles, and the video shows the same rationale, timing bars, ball strip and features that each
case on the `Cases` tab carries.

To re-render it after a model change:

```bash
python review/make_demo.py
```

To rebuild the underlying data after a model change:

```bash
python stage3/train_team.py --pitch-control "D:/Football Hackathon/stage3/pitch_control" --only-rate 0.41 --save-pred "D:/Football Hackathon/stage3/predictions_team" --out "D:/Football Hackathon/stage3/results_review.csv"
```

```bash
python review/export_cases.py
```

```bash
python review/patterns.py
```

```bash
python review/scoreboard.py
```

---

## What is in it

Each of the three panels on the Cases tab folds away from its own heading — click **Pick a case**,
**Pick a case on the left** or **What happened here** to collapse it and give the room to the
others. Which ones are shut is remembered in your browser. The green “no longer in the list”
notice is a one-line summary you open when you want the quotes, so it never pushes the case
list off the screen.

### 0. How it is doing — one table, and it is the first thing to read

The **How it is doing** tab holds a single table in two halves. The top half is accuracy: spellF1
and macroF1 with their noise bands, the share of live frames given to the right team, how many
stoppages are found at all, the timing of the whistle and of the restart, and the false and missed
possession switches — each one shown **now against the model before any of the hardening work**,
with the change marked better, worse, or *noise* where it falls inside the published band. The
bottom half is every remaining failure, by kind, with the same before/after columns plus how many
are confirmed, how many contested, and how many are new since the latest change.

Both halves are written by `review/scoreboard.py`, which scores the applied predictions and reads
this tool's own case index — so a row in the bottom half **clicks through to exactly the cases it
counts**, and the headline numbers can never drift away from the list.

### 1. Cases — 2,977 of them, and not all are failures

**416 of them are cases the model gets right**, and they come first in the filter for a reason: a
reviewer who has only ever seen failures has no feel for what a clean signal looks like. The bar
for inclusion is deliberately strict.

| Type | What it means | How strict |
|---|---|---|
| `STOPPAGE_CAUGHT` | a stoppage found | ≥80% covered, **both edges inside 0.5 s** |
| `TURNOVER_CAUGHT` | a change of hands made on time | within **0.5 s**, right team for a full second either side |
| `SPELL_HELD` | a long possession held | ≥15 s at **97%** frame agreement |

There are **187** caught stoppages, **171** held spells — and **32** clean turnovers. That last
number is the most useful one on this page. Of 1,549 real changes of possession, **31.6%** now
have a model turnover within a second and **21.4%** within half a second, up from 22.1% and 12.8%
before the hardening work — but only the 32 above clear the bar for a listed success, and a
correct case is listed only where the two reference products agree.

A correct case is only listed when the two reference products **agree** — a success story resting
on an unsettled reference teaches nothing.

### The failures — 2,561 of them

Every place the shipped model disagrees with the reference, typed by what went wrong:

| Type | What it means | Stage |
|---|---|---|
| `SPELL_MERGE` | a stoppage missed, same team either side — two real possessions fuse | 2 |
| `MISSED_STOP` | a stoppage missed, different teams either side — cheaper | 2 |
| `INVENTED_STOP` | a stoppage called where play never stopped | 2 |
| `BOUNDARY` | the stoppage was found but starts or ends more than a second out | 2 |
| `WRONG_TEAM` | both sides agree the ball is live; we put it on the wrong side | 3 |
| `FALSE_CHANGE` | a turnover the reference does not have | 3 |
| `MISSED_CHANGE` | a turnover the model does not make | 3 |

**Confirmed and contested are kept apart**, because they are not the same thing. The DFL ships two
independent products — a per-frame `BallPossession` field and an event feed naming the player on
every touch. We rebuilt the second from the raw feed. Where they **agree** across a case, the
reference is corroborated and the model is simply wrong: *confirmed*. Where they **disagree**, the
reference is not settled, and counting the case as a model error would be measuring our agreement
with one of two disagreeing scorers: *contested*.

| | confirmed | contested |
|---|---|---|
| BOUNDARY | 260 | 0 |
| FALSE_CHANGE | 362 | 69 |
| INVENTED_STOP | 42 | 15 |
| MISSED_CHANGE | 625 | **581** |
| MISSED_STOP | 17 | 0 |
| SPELL_MERGE | 11 | 0 |
| WRONG_TEAM | 655 | **306** |

Nearly half of the "missed turnovers" and a third of the wrong-team cases sit on a contested
reference. That is not a model problem and should not be worked on as one.

All 2,943 are counted. **505 carry a replay bundle** — the common types are capped at 60 so the
export stays at 29 MB; the rare, expensive types (`SPELL_MERGE`, `MISSED_STOP`, `INVENTED_STOP`)
are never capped.

### 1b. In plain words — the same four parts for every failure

Every case the model gets wrong opens with a read-out written by `plain_reason()` in
`export_cases.py` from that case's own frames. The shape never changes, whatever the kind of
mistake, because a reviewer should not have to learn a new format per case type:

| part | what it says |
|---|---|
| **the two answers** | what the model predicted across these frames and what the reference says, each compressed into its runs (*home throughout*, *ball dead for 0.8 s, then away*), and how many seconds of the window they actually disagree on |
| **the evidence** | which reading of these frames decided it — the ball out of sight, a touch read off a ball in flight, the restart layer holding a team, a ball lying still, or simply who was nearest — followed by **the model's own probability**, and whether the answer it gave is the one that probability pointed to |
| **the readings** | the numbers the prose is drawn from, as chips: ball in sight, ball speed, ball at rest, nearest player each side, P(home), P(ball dead), the touch and override shares, the restart it sits after, and the seconds of disagreement |
| **in cricket** | the same mistake in a scorer's terms — a change of strike written on a dot ball, a catch credited to the wrong fielder, runs credited after dead ball. The point of each is that these are **scoring** mistakes, not perception ones: the passage of play happened, the record written against it is wrong |

The third part is there so the second can be checked: every claim in the prose is a column, and
the column is printed beside it. The most useful line it produces is the one that appears when
the decoder's own `P(ball dead)` points away from the answer given — *“which points the OTHER
way from the answer it gave: the restart layer on top of the decoder settled these frames”*.
That is the hardening layer overruling the model, named on the case where it happened.

Reading the first run of this output found four things wrong with it, all now fixed: a boundary
case's disagreement lies **outside** the reference stoppage, so the window has to reach the edge
the model moved; a touch is itself an override, so reporting both shares counted the same frames
twice; an edge 20 s out that costs 2 s of disagreement has run into the **next** stoppage, and
saying both numbers without saying that reads as a contradiction; and a still ball and a ball
that never settles are opposite findings that were sharing one sentence.

The full cue-by-cue breakdown is still a click away under *Show all the numbers*.

### 2. Replay, and the plain read-out beside it

- A rendered animation of the coordinate file, with 8 seconds of context either side and the
  failure window shaded.
- **There is no video.** IDSSE ships tracking only, so every replay is a clearly labelled
  animation. See *Pending* below.
- Play / pause / frame-step / scrub, speeds 0.25×, 0.5×, 1×, 2×. Space, ← and → work as shortcuts.
- The two chips above the pitch say, in words, what each side thinks at that instant: *really*
  and *model*. The three strips below are *what really happened*, *what the model said*, and
  *could the model see the ball*.
- The amber dashed ring on the ball means the model could not see it on that frame.

**The right-hand column is written for a person, not for a reviewer who already knows the
pipeline.** Five blocks, in the order the questions actually get asked:

| Block | Answers |
|---|---|
| the verdict | one sentence: what went wrong, with the seconds that matter |
| the two answers, side by side | two coloured strips and how long they disagree |
| how play restarted | the restart kind, the last team in control and the likely restart team as separate fields, the first frame the ball was visible again, the frame play was judged to resume, and a timed reason for every change of team |
| what was visible | up to three signs that were on the pitch, each with a before/during meter and a plain mark: *the model went with this* / *did not act on this* / *no help either way* |
| how sure was it | one bar: the model's own confidence against the confidence it needed |
| what to teach it | the single lesson this case carries |

Two rules hold that whole column together. **A sign is only called "missed" when it was actually
present and pointed at the right answer** — the absence of a sign is never reported as something
the model ignored. And **a case about which team is never explained with stop signals**, or the
other way round; the signals follow the kind of case, not whatever happens to be in the bundle.

The confidence bar is the decoder's own arithmetic read as a probability instead of in nats: the
duration prior charges a fixed amount to open a stoppage, spread over the steps it would cover, so
`sigmoid(logit(prior) + cost/steps)` is the average confidence that stop had to clear. Nothing is
softened — the nats, the frame table, the timing bars and the ball breakdown are all still there,
folded behind **Show all the numbers** at the bottom of the column, and sections 3 to 5 below
describe them.

### 2b. Before and after the restart fix

The model under review is the hardened one (`stage3/restart_logic.py`). Where the hardening
changed a case, the side-by-side gains a third strip, **before the restart fix**, so the change
is visible on that case rather than taken on trust from an aggregate. 173 of the 557 playable
cases carry one. The frame table adds the layer's own working: nearest home and away player to
the ball, the k-NN margin, how far the ball travelled in the next 0.4 s, whether the restart
logic was driving that frame, and whether it flagged itself uncertain.

What the hardening did, across the five development matches: **34 individual failures fixed, 3
new**, restart edge error 1.446 s → 1.311 s, restart team right 430 → 436 of 460, false switches
431 → 419. Full record, including the three steps that were measured and rejected, in
`stage3/RESTART_HARDENING.md`.

The touch rule on top of that: **325 fixed, 176 new**, spellF1 0.438 → **0.482**,
missed switches 1206 → 1048. 135 of the 176 are a deflection read as a touch.

The whistle rule on top of that: **85 fixed, 48 new**, whistle timing 0.80 s → **0.70 s**
over 489 stoppages, failure cases 2817 → **2780**.

**Whether the ball was visible has its own filter**, and it decides whether a case can be
taught at all. Every case now carries `ball_seen`: the share of its window where the ball was
both tracked and visible to the model &mdash; the masked ball, which is all the model is ever
allowed to read. A case at **80% or above** is one where the reviewer and the model were looking
at the same evidence, so what a person sees in the clip is a fact the model could have used.
Below half, the model was going on where play had been, and nothing in the clip says what the
right answer was.

| of the 2,561 failures | cases | share |
|---|---|---|
| **the ball in sight 80%+ of the window** | **930** | **36%** |
| &hellip; and the reference corroborated (`confirmed`) | 534 | 21% |
| &hellip; and openable on the page right now | 287 | &mdash; |
| the ball hidden more than half the window | 1,178 | 46% |
| the ball never seen at all | 668 | 26% |

So **36% of what is left is reviewable** and 46% is not. The three worst piles split very
differently: `MISSED_CHANGE` 414 of 1,014 and `WRONG_TEAM` 330 of 895 keep a large visible
share, while `BOUNDARY` keeps only 8 of 149 &mdash; its edges are read off a ball that is
usually out of sight, which is why that pile answers to rules and not to review.

The filter sits under the kind and match pickers (*the ball was IN SIGHT*, *half in sight*,
*mostly hidden*), the share is printed on every row in the list, and the **How it is doing**
table has a clickable row that lands on exactly these cases.

**The cases the latest change created have their own filter** — pick *NEW since the restart fix*
in the case list. Each opens with an amber block saying what the old answer was, what the new
one is, which parts of the change touched that window, and what it says about the change. Of
the 213, 114 are the touch rule alone, 69 involve the whistle moving, and the rest are the
restart edge or the possession-length floor.

A comment on a case the model now gets right does not disappear silently. The case list says so
at the top, with the comment, which is the clearest evidence a change worked.

### 3. Why the model decided it — its own rationale

Every case carries a rationale block, and it is the model's arithmetic rather than a story about
it.

**Stage 2 (live or dead) is a Viterbi decode, so the decision *is* a sum.** The emission gain for
calling a step dead is `logit(p) − logit(prior)`, exactly as `decode_hsmm` scores it, summed over
the span in 0.2 s steps. Against it stands what the duration prior charges to insert a stoppage
into live play: a near-constant **9 nats** (measured at 8.6–10.1 across every duration band —
`stage2/DECODER_FIX.md`). The block shows the evidence, the best sub-window inside it, the charge
and the net.

For the two cases in the demo video:

| | caught, 35:56 | missed, 06:46 |
|---|---|---|
| evidence for a stoppage | **+334 nats** | **−56 nats** |
| best sub-window inside it | +334 over 9.2 s | +5 over 2.4 s |
| duration prior charges | −9 | −9 |
| so the decoder | switched | refused |

Read plainly: **on its own objective the model was right both times.** That is not an excuse —
the stoppage was real, we missed it, and spellF1 charges us either way — but it says where a fix
has to go. Into the evidence, not the decoder's threshold. Forcing the decoder to accept weak
evidence was measured: recall rises and invented stoppages triple, 11.6 → 36.2 per match, for a
spellF1 move inside the noise band.

**Stage 3 (which team) has no such closed form**, so the block instead reads off the cues it is
given — nearest player to the ball, pitch control at the ball, pitch control over the pitch, and
the event feed's second opinion — and marks which side each supports. Where they split, that split
is the disagreement. Where they all point the model's way, the honest answer is printed: these
inputs do not explain it.

### 4. Timing — what actually happened against what the model thought

A separate block, because "we found it" and "we found it on time" are different claims. It gives
the reference's dead-ball span and the model's on one axis, both relative to the start of the
case, the **whistle** and **restart** offsets (exact, measured at full rate, not at replay
resolution), the overlap percentage, and every change of hands on each side with the lag between
them. Where the model never called a stoppage at all it says so, because there is then nothing to
compare.

### 5. The ball — in the file, and as the model saw it

Two different things, kept apart. IDSSE tracks the ball on **every** frame; the competition match
loses it on **41%**, so stage 3 is trained and scored with the ball hidden in contiguous bursts at
that rate. The block reports both percentages, the share inside the case window, the longest
unbroken stretch with no ball, and how many steps were held out. The amber dashed ring on the
pitch and the amber in the ball strip mark the frames the model had to work without it.

Stage 2 never reads the ball at all, which is why `ball_unseen_any` lifts the error rate by only
1.36×.

### 6. Your feedback

**The comment box saves itself** — a second after you stop typing, when you click away, and when
the page closes. Switching case cannot lose a note. The server merges field by field, so an
autosave never clears a verdict or a bookmark, and existing notes survive a re-export as long as
the case keeps its id.

Bookmark a case, set a verdict (*model wrong / reference wrong / genuinely ambiguous / not an
error*), propose a label, and leave a comment. Saved to `review/feedback/feedback.json`.

Every record is written with `used_for_training: false`. **Nothing you write here becomes a
training label until it has been read and accepted deliberately.** That is a deliberate stop, not
an oversight.

### 7. Patterns

Flags computed from data available at any frame — never a reference to a particular frame, minute
or match. Thresholds are round numbers chosen in advance, not fitted to the error rate.

Team attribution errors, base rate 0.075:

| flag | coverage | err inside | err outside | lift | share of all errors | consistent across all 5 matches |
|---|---|---|---|---|---|---|
| `within_1s_of_turnover` | 10.1% | 0.410 | 0.037 | **11.0×** | **55%** | yes (7.9–14.5) |
| `reference_contested` | 5.6% | 0.338 | 0.059 | **5.7×** | 25% | yes (4.1–7.9) |
| `contested_margin_1m` | 15.6% | 0.165 | 0.058 | **2.8×** | 34% | yes (2.3–3.3) |
| `ball_unseen_any` | 41.0% | 0.089 | 0.065 | 1.36× | 49% | yes |
| `within_3s_of_restart` | 5.2% | 0.089 | 0.074 | 1.20× | 6% | no |
| `partial_tracking` | 7.4% | 0.073 | 0.075 | 0.97× | 7% | no |
| `loose_ball_5m` | 16.1% | 0.030 | 0.083 | **0.35×** | 6% | yes |

Live/dead errors, base rate 0.052: `within_3s_of_restart` leads at 3.08× but is **not** consistent
across matches, so treat it as unproven. Ball visibility has lift **below 1** — unsurprising, since
Stage 2 never looks at the ball.

Three things worth saying plainly:

- **Turnovers are where the model breaks.** 10% of frames carry 55% of the team errors, and this
  holds in every match.
- **The ball is not the main problem.** 41% of frames have it hidden and the lift is only 1.36×.
- **A loose ball is *easier*, not harder** — lift 0.35×, consistent in all five matches. The
  association is solid; the mechanism is not established, and a plausible guess (a loose ball is
  often a long clearance where the reference is stable) is a hypothesis, not a finding.

Read all of this as **association, not cause**. A flag that marks unreliable frames is useful
whether or not it explains anything, and none of it is evidence that the flagged condition
*causes* the error.

### 8. Measuring a change

```bash
python review/evaluate.py --label my_change --note "what I changed"
python review/evaluate.py --compare baseline my_change
```

Reports spellF1, macroF1, stoppage recall, short same-team recall and the failure counts split
confirmed/contested. Differences below the organisers' own noise band (**0.04** on spellF1, 0.01 on
macroF1) are printed as *noise*, not as improvements. The script refuses to run if a sealed match
appears in the case index.

The architecture is unchanged: dead-ball detection, then possession, then the join.

Baseline, taken 3 November 2026:

| | spellF1 | macroF1 |
|---|---|---|
| Stage 2 alone | 0.862 | 0.947 |
| Stage 3 alone | 0.500 | 0.903 |
| End to end | **0.438** | 0.855 |

---

## Pending — blocked on permission, not on work

| Item | Status |
|---|---|
| **Video replay** | IDSSE has no video, so every replay here is a tracking animation. The only video we hold is the restricted competition match. The organisers' clarification of 3 Nov permits *inspecting* the provided data and video — so a local, inspection-only video pane is possible. **Not built: see the question in chat.** |
| **More matches** | The Sportec/DFL request at `Sportec_IDSSE_data_request_DRAFT.md` is drafted and unsent. Five of our seven matches share one home club. |
| **Reviewer feedback as labels** | Captured and stored, deliberately not wired in. Needs a read-through and an explicit decision first. |
| **Competition-match review** | Not built. The competition match has no reference labels we may train on, so a failure review there could only ever be inspection, never scoring. |

## Rules this tool is built to respect

- Training and tuning use **external data only** — leave-one-match-out over the five IDSSE
  development matches.
- The two sealed test matches, `J03WMX` and `J03WN1`, are never loaded. `evaluate.py` refuses to
  run if they appear.
- No provided competition data is uploaded anywhere; this tool is local and loopback-only.
- No hand labels on the competition match are used for training, tuning or model selection.
- Reviewer feedback is stored, never auto-promoted.
