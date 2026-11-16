# Additional-sources sweep — 30 October 2026

Run in response to three requests: find more data with realistic ball loss (B1), more DFL-labelled
matches (B3), and more matches to widen the test set (B5).

**Outcome: no additional source qualified.** The 7 IDSSE matches are the entire public universe of
frame-level possession *and* ball-status reference labels. Below is what was checked and why each
was rejected, so the search does not get repeated.

> **Revised 30 October 2026, same day.** Access to SoccerNet-GAR was granted and its real files
> were inspected rather than just its dataset card. Section 2 is rewritten: the verdict stands,
> but the earlier one-line reason was wrong and is corrected there.

---

## 1. Are there more open Sportec / DFL matches? — No, and this is definitive

`kloppy` ships an official Sportec open-data loader. Its internal match map
(`kloppy/_providers/sportec.py`, `get_IDSSE_url`) lists exactly seven match ids:

```
J03WMX  J03WN1  J03WOH  J03WOY  J03WPY  J03WQQ  J03WR9
```

Those are precisely the seven already collected. The publishing institution's own announcement
describes the release as two Bundesliga and five 2. Bundesliga matches from 2022/23. There is no
larger open Sportec release, and no 2025–2026 announcement of one.

**B3 and B5 are therefore closed on the data side.** More DFL-labelled matches do not exist to be
found.

---

## 2. SoccerNet-GAR / PFF World Cup 2022 — ACCESS GRANTED, inspected, still rejected

**Updated 30 October 2026, after access was granted.** The first pass on this source judged it
from the dataset card alone and recorded *"no possession or ball-in-play labels — only ten action
labels."* That was too blunt. With access granted I read the actual files, and the picture is
richer than the card implies. The verdict does not change, but the reason does, so the earlier
wording is corrected here rather than left standing.

### What is actually in it

Read from `tracking-full/test/metadata.parquet` through the Hugging Face preview (nothing was
downloaded). Each sample's payload looks like:

```json
{"id": "test_000000",
 "inputs": [{"type": "tracking_parquet", "path": "test/clip_000000.parquet", "fps": 29.97}],
 "labels": {"action": {"label": "PASS"}},
 "metadata": {"game_id": "3850", "game_time": "1 - 00:00",
              "position_ms": 82415, "team": "home"}}
```

and the split header declares `"task": "action_classification"` with exactly ten action names.

So each clip carries **an action, a millisecond timestamp on a per-match clock, and a `team`**.
Ordered by `position_ms` within a `game_id`, that is a continuous touch-level event stream —
the same raw material already used on the Metrica matches. The stoppage pattern is plainly
visible in the first five minutes of game 3850:

| time | event | team |
|---|---|---|
| 00:06 | OUT | away |
| 00:18 | THROW IN | away |
| … | | |
| 02:01 | OUT | away |
| 03:09 | FREE KICK | home |

`OUT → next restart` gives a dead-ball **interval in time**, exactly the construction
`parse_metrica.py` already performs.

### Measured, after the NDA was signed

R. Sudan signed the SoccerNet NDA and authorised downloading. The split index was fetched —
`tracking-full/test/metadata.parquet`, 1,671,596 bytes, SHA-256 verified against the published
value — and **all 13,689 samples across the 10 test matches were measured**. Scripts:
`scripts/probe_soccernet_gar.py` and `scripts/probe_soccernet_gar_team.py`. Full output:
`reports/SoccerNetGAR_Measurements.txt`.

> **This corrects a guess made earlier the same day.** Before measuring, this file estimated that
> the clips "very nearly tile live play" because events run about every 4.4 s. That was wrong.
> Measured coverage is 59.2%, not ~100%.

| Measured over 10 matches, 13,689 clips | Value |
|---|---|
| Elapsed match time covered by any clip (median) | **59.2%** (min 51.3%, max 70.6%) |
| Elapsed match time with **no tracking at all** | **40.8%** |
| Holes between clips | 2,459 holes, 24,214 s total; median 2.4 s, p90 27.3 s, **max 303 s** |
| Stoppages identified (`OUT` → next restart) | 881, totalling 34,928 s |
| Stoppage time that has any tracking | **33.9%** |
| **Stoppage time with no tracking** | **66.1%** |

### Why it is rejected, on the measurements

1. **Two-fifths of every match is simply not there.** 40.8% of elapsed time has no positions at
   all, and the largest single hole is over five minutes. This is not a continuous corpus and
   cannot be made into one.
2. **Two-thirds of dead-ball time is untracked** — the one thing this corpus was wanted for. The
   34% that *is* covered is only the few seconds either side of the `OUT` and the restart, because
   those are themselves events. The middle of every stoppage is blank. A corpus cannot teach what
   a dead ball looks like when it does not contain dead-ball frames.
3. ~~**`team` is not a possession attribution.**~~ **WITHDRAWN — this was wrong, and the test that
   withdrew it is below.** `OUT` really is labelled `away` on 884 of 884 occurrences, but that is
   a defect in one action type, not evidence about the field as a whole. Tested properly
   (`scripts/probe_soccernet_gar_trust.py`), `team` carries genuine possession structure:

   | Test | Result | What noise would give |
   |---|---|---|
   | After a THROW IN / FREE KICK, next on-ball action is the same team | **78.5%** | 50% |
   | Mean length of consecutive same-team on-ball runs | **5.10** | 2.24 (shuffled control) |
   | Runs of 5 or more touches | **34.6%** | 9.3% (shuffled control) |

   And the 40.7–84.1% pass skew is coherent rather than random: in match 3854, away takes 81.9%
   of touches *and* holds the ball for 12.75 touches per run against home's 2.84; in match 3853
   the asymmetry reverses with the skew; in match 3856, at 49.8%, the run lengths are 3.93 and
   3.99. A broken field does not produce that. **These are genuinely lopsided World Cup matches.**
   Conclusion: trust `team` on on-ball actions, ignore it on `OUT`.
4. **No frame-level possession or ball-status field exists anywhere.** The declared task is
   `action_classification`; the only label is one of ten action names.
5. **The NDA is personal, and does not reach the judges.** RULES.md rules out exactly this: *"A
   private in-house model or an unavailable internal dataset is the one thing this rules out."*
   Signing it lets R. Sudan read the data; it does not let Metrica inspect what a submission was
   built on. Licence remains declared only as `other`.

### Revised verdict: split the question

The original single verdict ("rejected") bundled two different questions. Separated, they have
different answers.

| Use | Verdict | Why |
|---|---|---|
| **Teaching the model what a dead ball looks like** | **NO** | 66.1% of dead-ball time has no tracking. The frames do not exist |
| **Teaching the model H vs A from player geometry** | **YES, valuable** | 87,939 touch events over 64 elite matches, with a `team` field now tested and trusted |
| **Measuring how good we are** | **NO** | Touch-based, 30 fps, clip-based; IDSSE remains the only reference set |

And one finding that inverts part of ground 2. The 33.9% of dead-ball time that *is* covered is
not a random third — it is, by construction, the ~2.4 s either side of the ball going out and the
~2.4 s either side of the restart, because those moments are themselves events. **That is the
boundary of every stoppage**, and spellF1 scores interval boundaries, not interval interiors. The
middle of a stoppage (players standing about) is the easy part and the part we already have from
IDSSE. The transition in and out of a stoppage is the hard part, and this corpus covers 881 of
them in the test split alone — roughly 5,600 across all 64 matches.

So the corpus is empty where the problem is easy and populated where it is hard. That is the
opposite of how it was first written up here.

### What still stands

Ground 1 (40.8% of match time untracked) and ground 4 (no frame-level possession or ball-status
field) are unchanged and measured. Ground 5 is unchanged in substance but is now the user's call:
R. Sudan has signed the NDA and, on 30 Oct 2026, directed that the data be used, stating he would
handle the eligibility question himself. **That decision is recorded here, not overridden.** The
risk it carries: a personally-signed NDA does not let Metrica inspect what a submission was built
on, which RULES.md requires.

**Decision: fetch the `tracking-full` shards (7.11 GB) when Stage 3 begins.** The 245 GB of video
remains unnecessary — the model consumes tracking, not pixels. The downloaded index is kept under
`raw/soccernet_gar_probe/` with a README marking it NDA-restricted and not for publication.

### One thing worth keeping from it

The published benchmark on this corpus is independent support for the architecture already chosen:
a tracking model (GIN, 180K parameters) beats a fine-tuned video model (VideoMAEv2-B, 86.3M
parameters) by **16.9 points of balanced accuracy and 6.9 points of macro F1**, training 7× faster
with 479× fewer parameters. Positions beat pixels on this task family. That is a citable reason
why no match video is needed for Phase 1.

Its `tracking_valid` convention is also a useful precedent: missing rows are **preserved and
flagged, never interpolated** — which is what `common_schema.py` already does with `ball_observed`.

This is the same underlying data as the Gradient Sports route previously recorded as pending, so
that route is now closed rather than merely unanswered.

---

## 3. Smaller tracking datasets — all rejected

| Source | What it is | Why rejected |
|---|---|---|
| **SoccerTrack v2** | 10 university-level games, 4K multi-view | Ball-action spotting labels, not possession state. No dead-ball field |
| **Last Row** | 19 Liverpool goals, 2019, player + ball tracking | Goal sequences, not full matches. The repository itself says it *lacks the accuracy needed for research*. No possession labels |
| **Signality / Hammarby** | 3 games, tracking **and video** | Tracking and video are **not synchronised**, no event data, no possession labels. Distributed via a university course page with no clear licence |
| **Alfheim / Tromsø** | ZXY sensor tracking at 20 Hz, >100 GB | Player positions only; ball positions are a small manually extracted set. Sensor-based, not optical — a different domain. Terms bar player profiling |
| **SoccerNet-Tracking** | 200 video sequences, MOT benchmark | Multi-object tracking benchmark; no possession labels, clips not matches |
| **SoccerNet-GSR** | 200 × 30-second clips, player positions + pitch lines | No ball annotations; 30-second clips cannot contain possession intervals |
| **Roboflow soccer sets** | Image object detection | Bounding boxes only, no tracking, no possession |

---

## 3b. Two methods worth borrowing from the Copa América paper

Neither is data, but both are directly usable and both are open.

- **Pitch control** (Fernández & Bornn, 2018), used in that paper as a probabilistic map of the
  space each team controls. This is the canonical way to turn 22 player positions into a per-team
  control surface **without needing the ball at all** — precisely the evidence channel Stage 3
  needs when the ball is missing. Worth adopting as a named feature rather than inventing one.
- **UnravelSports' graph tooling** (same author) converts raw tracking into the player-graph
  representation that both PathCRF (§4) and the strategy deck's architecture assume. It is
  maintained and open, so it is a shortcut to a representation we would otherwise build.

The paper's own subject — visual exploration from head and shoulder pose — is **not** usable here:
the competition supplies x/y only, with no pose, so the vision layer has nothing to run on.

---

## 4. What the sweep did find: a directly relevant method

**PathCRF** — Kim, Lee, Seo, Ko, Yoon, Park (2026), *"Ball-Free Soccer Event Detection via
Possession Path Inference from Player Trajectories"*, KDD 2026. arXiv:2602.12080. Code under
**MPL-2.0** at `github.com/hyunsungkim-ds/pathcrf`.

This is not a data source, but it is the most relevant published work found, and it matters for
B1 specifically:

- It detects on-ball events **from player tracking alone, never using the ball trajectory.** The
  ball is used only as a visual overlay for reference.
- It models players as a fully connected dynamic graph and selects **one edge per time step**
  representing current possession.
- A **CRF forbids impossible transitions** between consecutive edges, and **Viterbi decoding**
  yields the most probable possession path. An event is flagged when the selected edge changes.
- It is built and evaluated on **the same seven Sportec open matches** we hold, and its
  preprocessing produces per-frame ground-truth possession states from them.

Two consequences for our build:

1. **It independently validates the recommended architecture.** A CRF with Viterbi decoding over a
   possession state sequence is the same family as the HSMM-with-Viterbi design in the strategy
   deck — sequence model, explicit transition constraints, interval output by construction.
2. **It reframes the missing-ball problem.** If possession can be inferred from player geometry
   with *no* ball at all, then the competition's 41% missing ball stops being fatal and becomes a
   question of how much the ball adds when present. That argues for making the geometry-only
   evidence channel strong in its own right, not merely a fallback for when the ball is lost.

Its preprocessing code is worth reading before Stage 2, under a licence that permits it.

---

## 5. Consequence for the masking plan (B1)

The sweep does not remove the need to mask. No source reproduces the competition's live-play ball
loss:

| | ball missing during live play |
|---|---|
| IDSSE | 0.0% |
| Metrica | 12.7% |
| SkillCorner | 7.7% |
| *Competition (documented)* | *contributes to 40.7% of graded frames* |

SkillCorner remains the hardest-observation source available and is already collected. Masking the
IDSSE training data therefore stays the plan, and the gap-length distribution from a local run of
`check_assets.py` stays the input that would make it faithful.

---

## 6. Second web sweep — 30 October 2026, afternoon

Run at R. Sudan's request, who offered to obtain access to anything gated. **No new open source
with frame-level possession and ball-status was found.** The searches did confirm, from an
independent direction, that the seven Sportec matches are the whole public supply: a targeted
search for 2026 Bundesliga/Sportec releases on Zenodo and Figshare returned only the Bassek et al.
dataset we already hold (DOI `10.6084/m9.figshare.28196177`, 2 Bundesliga + 5 2.Bundesliga).

### Checked and rejected this round

| Source | What it is | Why it does not help |
|---|---|---|
| **DFL Bundesliga Data Shootout** (AWS/Kaggle 2022) | Broadcast video with Play / Challenge / Throw-In event markers, start and end timestamps | **Video only, no tracking.** Our model consumes tracking tables. The throw-in markers are interesting for dead-ball timing but there are no positions to attach them to |
| **SoccerSum** (Zenodo 2024) | 750 annotated frames, Norwegian Eliteserien | Object detection, not tracking. `CC BY-NC-ND 4.0` bars derivative work anyway |
| **SoccerMon** (Zenodo 2022) | GPS from Norwegian women's elite teams | Player GPS only. No ball, no possession, no opposition |
| **SoccerNet-v3D / SoccerNet-GSR / SoccerNet-Tracking** | Broadcast replay and game-state benchmarks | No possession or ball-in-play labels; clips, not matches |
| **SoccerTrack v2** | 10 university matches, panoramic 4K | Re-checked: ball **action** spotting over 12 classes, and the authors explicitly leave the per-frame ball-status flag out of their baseline |

### Exists, is relevant, but is gated — worth R. Sudan's asking

These are the only leads where an access request could plausibly change the picture. None is
needed for Stage 2.

1. **Copa América 2024 broadcast tracking + pose** — 32 matches, synchronised with on-ball event
   data. Used in *Wide Open Gazes* (Bekkers, arXiv:2602.18519, Sloan 2026). Read in full
   30 Oct 2026. Three corrections to the earlier guess:
   - The provider is **Respo.Vision**, not SkillCorner, and the data was supplied to all teams
     competing at the 2024 Copa América.
   - It is recorded at **25 fps — the competition's own rate**, which makes it a better frame-rate
     match than SoccerNet-GAR's 29.97.
   - What is open-sourced is the **vision-map code** on the U.S. Soccer Federation GitHub, not the
     tracking. The tracking is not public.

   Ask: Joris Bekkers (U.S. Soccer Federation / UnravelSports), or Respo.Vision directly.
   Caveat before asking: the paper gives no indication that this data carries ball-status or
   frame-level possession, so it would be an H/A and geometry resource at best.
2. **SkillCorner beyond the open set** — we hold 20 open matches. SkillCorner has no published
   academic tier, so this is a direct commercial ask.
3. **Sportec Solutions** — they published exactly seven matches. A request for more of the
   2022/23 season is the single highest-value ask on this list, because their data is the only
   kind that carries both `BallPossession` and `BallStatus` per frame, and it is already the
   project's reference set.
4. **StatsBomb/Hudl** — free data for research exists but is event-only; a tracking request would
   be a separate commercial conversation.

**Ranking, if asks are to be made:** Sportec first, by a distance. Copa América second. The rest
would not change what the model can learn.

---

## Sources

- `kloppy/_providers/sportec.py` (installed, v3.18.0) — the authoritative open-match list
- Hugging Face `OpenSportsLab/SoccerNet-GAR` — dataset card for gating and licence; branch listing
  for sizes (`videos` 245 GB, `tracking-full` 7.11 GB); and `tracking-full/test/metadata.parquet`
  read through the Hugging Face preview for the real schema, labels and event stream. Access
  granted to R. Sudan 30 Oct 2026. Nothing was downloaded.
- arXiv:2602.12080 — PathCRF; `github.com/hyunsungkim-ds/pathcrf` — MPL-2.0 code
- `github.com/Friends-of-Tracking-Data-FoTD/Last-Row`; `datasets.simula.no/alfheim`
- Metrica sample-data README — the Signality/Hammarby and Last Row pointers
