# Data Readiness Report — Stage 1

**More Than a Hack 2027, Phase 1: Ball Possession**
Prepared 29 October 2026 · competition repository reviewed at commit `f8f2629` (23 Oct 2026, unchanged since the 28 Oct review)

---

## Verdict, per capability

Not one answer. Each capability is ready, partly ready or not ready on its own terms.

| Capability | Verdict | Why |
|---|---|---|
| Parsing and pipeline tests | **Ready** | 30 matches parse end to end into one schema; every structural check is recorded |
| Weak supervision | **Ready** | 3 Metrica + 20 SkillCorner matches carry usable weak signal |
| Supervised training | **Ready** | 7 IDSSE matches carry provider reference labels on 100% of frames |
| Independent possession evaluation | **Partly ready** | Only 2 genuinely held-out matches, and the DFL's possession convention has not been reconciled with the competition's |
| Dead-ball supervision | **Ready** | IDSSE `BallStatus` is a direct per-frame field; Metrica gives law-based spans |
| **Missing-ball robustness** | **Not ready** | No external source reproduces the competition's live-play ball loss. See the blocking finding below |
| Visual-detector training | **Not ready** | No match video was obtained from any source, and none is offered |

---

## 1. What was collected

All downloads verified by byte size and SHA-256; see `reports/DatasetManifest.csv`.

| Source | Matches | Raw size | Frames | Minutes | Licence |
|---|---|---|---|---|---|
| Metrica Sports sample data | 3 | 175 MB | 429,923 | 287 | No SPDX licence; attribution requested |
| IDSSE (DFL / TRACAB, Bundesliga) | 7 | 2.5 GB | 1,002,644 | 668 | **CC-BY-4.0** |
| SkillCorner Open Data | 20 | 1.9 GB | 1,182,244 | 1,971 | **MIT** |
| StatsBomb Open Data | 0 (docs only) | 1.9 MB | — | — | User agreement, needs reading |
| Gradient Sports (ex-PFF FC) | 0 | — | — | — | **Pending — not collected** |

**Total: 30 matches, 2,614,811 frames, roughly 49 hours of play.**

Deliberately excluded, with reason: SkillCorner `data/bodypose/` and `data/aggregates/` (no role in frame-level possession); the whole StatsBomb `data/` corpus (event data with no continuous tracking cannot produce frame-level H/A/D).

### Matches, by source

- **Metrica** — `Sample_Game_1`, `Sample_Game_2`, `Sample_Game_3`. Anonymised. 25 fps.
- **IDSSE** — `DFL-MAT-J03WMX` (1. FC Köln v Bayern München), `J03WN1` (VfL Bochum), and five Fortuna Düsseldorf home fixtures: `J03WOH`, `J03WOY`, `J03WPY`, `J03WQQ`, `J03WR9`. 25 fps.
- **SkillCorner** — 20 Australian A-League matches, ids `1874553` … `2017461`. 10 fps.

---

## 2. The blocking finding

**The external corpus does not reproduce the competition's central difficulty.**

The competition documents that the ball is absent on **40.7% of graded frames**, and that it goes missing *disproportionately on the interesting ones — crowded boxes, aerial duels, tackles*. That is loss **during live play**.

Measured on the collected corpus:

| | P(ball missing \| play alive) | P(ball missing \| play dead) | Overall |
|---|---|---|---|
| Metrica | **12.7%** | 96.2% | 37.5% |
| SkillCorner | **7.7%** | n/a (no dead-ball field) | 40.4% |
| IDSSE | **0.0%** | 0.0% | 0.0% |
| *Competition (documented)* | — | — | *40.7% of graded frames* |

Metrica's headline rate of 37.5% looks like a match for the competition's 40.7%. It is not the same phenomenon. Metrica simply **stops tracking the ball when play stops** — it is missing on 96.2% of dead-ball frames and only 12.7% of live ones. IDSSE is cleaner still, at zero.

**Consequence.** A model trained on any of this sees a far easier observation regime than the competition match, and would be flattered by it. Worse, a model could learn the shortcut *"ball missing ⇒ dead ball"*, which is 96% true in Metrica and roughly 50% wrong in the competition — and which is exactly the failure the published baseline already makes.

**Cricket parallel.** It is like training a ball-tracking scorer on footage where the camera is switched off between deliveries. It learns "no ball on screen means no delivery in progress" — perfectly true in training, and badly wrong at a ground where the camera also loses the ball every time it is edged to slip.

**Remedy, and it is not optional.** Ball observation must be *masked* on the external training data so that live-play frames go missing at a competition-like rate. The masking pattern cannot be copied from the competition data, which a cloud model may not read. Two permitted routes:

1. Run `local_checks/check_assets.py` on your machine. Its missing-ball section reports the gap-length distribution and the share of gaps under 5, 12, 25, 75 and 250 frames. Those summary statistics, which you choose whether to share, are enough to parameterise a masking model.
2. Failing that, use a documented synthetic model and state it plainly in the technical document.

For comparison, the gap structure already measured on external data: Metrica 487 gaps, median 6.2 s, p90 36.9 s, max 131.7 s; SkillCorner 3,796 gaps, median 5.7 s, p90 30.6 s, max 220.6 s.

---

## 3. What each source can honestly be used for

> **Updated 30 October 2026 — two findings that strengthen IDSSE considerably.**
> 1. **The possession convention is CONTROL, not touch.** Bassek et al. 2025 state it outright:
>    *"The ball possession is defined by the player in control of the ball."* That is the
>    competition's own definition. IDSSE is the only collected source whose convention matches
>    the task; Metrica's is touch-based. This closes the project's largest open question.
> 2. **The dead-ball labels are exact.** Our per-frame dead share agrees with the DFL's own
>    official net playing time (`OtherGameInformation`) to **0.00 percentage points on all seven
>    matches**. See `reports/DeadBallProfile.md`.

### IDSSE — the evaluation set

The only source with genuine frame-level reference labels. The ball's own `FrameSet` carries, on every frame:

```
BallStatus      0 = Dead, 1 = Alive
BallPossession  1 = Home, 2 = Away
```

These are not my reading of the numbers — they are the definitions in `floodlight/io/dfl.py`, written by the group that published the dataset and cited in its paper. Home and away come from an explicit `Role="home"` / `Role="guest"` in the match information XML, so the mapping needs no convention.

Label coverage is **100.0%** of frames on all seven matches, with zero anomalies: no frame without `BallStatus`, no alive frame without a possession team.

Two cautions, both unresolved:

- **The DFL's possession convention is not known to match the competition's.** The competition requires *controlled* possession and says a touch, a non-intercepting deflection, or an aerial duel won without subsequent control are not changes. Whether the DFL feed draws the line in the same place is undocumented and I have not verified it.
- **Dead-ball share is 40.7–51.7%** across the seven matches, against **19.6%** in the competition's public five-minute window. Either that window was an unusually continuous passage or the conventions differ. This cannot be resolved by studying the public window — doing so is the tuning the rules forbid.

### Metrica — the dead-ball teacher

Possession must be derived from events. The official definitions document was read in full, and it is decisive: **Metrica assigns possession the moment a player touches the ball**, while the competition requires control. So the two conventions genuinely differ, and Metrica's possession boundaries arrive earlier and more often.

The derivation therefore splits by quality:

- **Dead-ball spans — `strong`.** Law-based: `BALL OUT` or `FAULT RECEIVED` stops play, a `SET PIECE` or `RECOVERY` restarts it. Both conventions agree about those. Roughly 28% of frames.
- **Team spans — `weak`.** Touch-based, from `RECOVERY`, `BALL LOST`, `PASS`, `SHOT`, `CHALLENGE` with a recorded result.
- **Contested spans — `ambiguous`, and left unlabelled.** A `CHALLENGE` with no recorded result settles nothing, so none is invented.

One correction the definitions forced: `BALL LOST` carrying an `Intervention` subtype of `FORCED`, `VOLUNTARY` or `END HALF` is **the referee or the player stopping play, not a turnover**. An earlier draft treated every `BALL LOST` as a hand-over and was wrong.

### SkillCorner — robustness only

Broadcast tracking at **10 fps**, preserved as such. Upsampling to 25 fps would invent fifteen observations a second that nobody made.

It is a poor label source for three verified reasons:

1. **No ball-in-play field exists.** `possession.group` is null on 45.9% of frames, and a null there could mean dead ball or unknown. This source therefore contributes **no `D` labels at all**, and nulls are left null.
2. **The file is named `tracking_extrapolated` and means it.** A ball coordinate is present on 73.4% of frames while `is_detected` is true on only 60.4%. `ball_observed` is taken from the detection flag, never from the presence of coordinates.
3. **Possession is an algorithmic provider estimate.** Scoring our model against another vendor's model measures agreement, not accuracy.

Its real value is domain shift: on 25.6% of frames the camera is off the pitch and no players are tracked at all — the closest available analogue to the competition's broadcast feed, where a third of frames have no usable homography.

### StatsBomb — documentation only

Event data plus 360 freeze frames, not continuous tracking, so it cannot produce frame-level H/A/D. The six specification PDFs were collected so the event vocabulary can be compared across providers. The `data/` corpus was deliberately not downloaded.

### Gradient Sports — pending

Access appears to require a request form or account, and no dataset-specific licence permitting machine-learning use **and** third-party reproduction was located. No account was registered and no details submitted, per instruction. A free-access offer is not proof of an open licence, and rule 1 of the competition requires the organisers to be able to access whatever we build on.

---

## 4. Videos — none obtained

**No match video was collected from any source, and none of the four collected sources offers
any.** Some *rejected* sources do offer video; none of it was downloaded, and `SourceSuitability.csv`
records that in the column `video_offered_by_source_not_collected` (renamed 30 Oct 2026 from
`raw_video_provided`, which read as though we held the footage).

Of the four sources actually collected:

- **Metrica** — the repository tree contains 11 files: three games of data, a README and the event-definitions PDF. There is no video, and the only video it links to is a tutorial talk, which is not the corresponding match footage.
- **IDSSE** — position, event and match-information XML only.
- **SkillCorner** — tracking JSONL, match JSON and event CSV only.
- **StatsBomb** — event JSON only.

Of the rejected sources, SoccerTrack v2, Alfheim/Tromsø and Signality/Hammarby offer footage
(Signality's is explicitly *not* synchronised to its tracking), and SoccerNet-GAR holds 245 GB of
broadcast clips. All were rejected for reasons unrelated to video, so none was fetched.

What that does and does not block:

| Work | Needs video? | Status |
|---|---|---|
| Tracking → possession model | No | **Unblocked.** This is the recommended architecture, and it runs on tracking tables |
| Evaluating it on other matches | No | **Unblocked** |
| Training a new ball detector | Yes | **Blocked.** No imagery, no annotations |
| Drawing predictions over the competition video | Local only | Possible on your machine; a display step, not a model step |

This is not a serious loss. The recommended architecture consumes tracking, not pixels. A model trained on tracking tables cannot be handed an MP4 — it would need an upstream detection and tracking pipeline, which is option D in the strategy deck and deliberately last.

`ffmpeg` and `ffprobe` 9.0.2 were installed without administrator rights at
`external_data/tools/ffmpeg-9.0.2-essentials_build/bin/`. They are not needed for anything in this stage, but they are required for the day-one decode check on the competition video.

---

## 5. Proposed split

Match-level, and **team-aware**. See `reports/SplitManifest.csv`.

| Split | Matches | Rationale |
|---|---|---|
| train | 4 × IDSSE (Düsseldorf) + 3 × Metrica | reference and weak labels together |
| validation | 1 × IDSSE (`J03WR9`, Düsseldorf) | within-domain held-out |
| **test** | 2 × IDSSE (`J03WMX` Köln, `J03WN1` Bochum) | **different competition, no shared team or venue** |
| robustness | 20 × SkillCorner | never trained on, never scored on |

The seven IDSSE matches are **not seven independent observations**: five are Fortuna Düsseldorf home fixtures in one competition. Splitting those five across train and test would let a model learn Düsseldorf-specific pitch and tracking characteristics and be rewarded for it. So the two competitions are kept whole and placed on opposite sides.

Both halves of a match are always in the same split, because a match is the unit. No windowing has been done yet; when it is, windows must be cut **after** the split, never before, or neighbouring windows will straddle it.

**The honest limit: the test set is two matches.** That is enough to catch a broken model and not enough to separate two good ones. Treat differences on it with the same suspicion the competition's own documentation applies to its five-minute public window.

---

## 6. Feature compatibility

A feature is only allowed to exist if it can also be computed from the competition's own ATD at inference. See `reports/FeatureCompatibility.csv`. Two traps found:

- **Ball height `Z`.** IDSSE/DFL ships a Z coordinate on every ball frame. The competition ATD does not. Any feature using it would train beautifully and be uncomputable at deployment. **Excluded.**
- **Player identity.** DFL and SkillCorner both carry stable player ids. The competition has 46 anonymous track slots that start, stop and swap. Every feature in the canonical schema is therefore collective — centroids, spread, nearest-player distances, counts — and none is keyed to a person.

Also partial: Smart Tagging phases exist only in the competition data (SkillCorner has an analogue, Metrica and DFL have none), so a tagging-dependent feature cannot be trained externally.

---

## 6b. Additional-sources sweep — 30 October 2026

A second search was run specifically to find more data with realistic ball loss, more DFL-labelled
matches, and more matches for the test set. **Nothing qualified.** Full detail in
`reports/AdditionalSourcesSearch.md`.

- **No more open Sportec/DFL matches exist.** `kloppy`'s own open-data loader lists exactly the
  seven match ids already collected. This is definitive, not an absence of evidence.
- **SoccerNet-GAR / PFF World Cup 2022 (64 matches) — ADMITTED as an H/A pre-training set,
  rejected for everything else.** R. Sudan signed the NDA, authorised use, and the index was
  fetched (hash-verified) and all 13,689 test clips measured. Two findings, in opposite
  directions:
  - **40.8% of match time and 66.1% of dead-ball time carry no tracking** (largest hole 303 s),
    so it cannot teach dead-ball interiors. But the third that *is* covered is the ~2.4 s either
    side of every stop and restart — the stoppage **boundaries**, which is what spellF1 scores.
  - The `team` field **is** a real possession attribution, contrary to the first write-up. Tested:
    78.5% restart agreement against 50% chance, and same-team touch runs averaging 5.10 against a
    shuffled control of 2.24. Ignore it on `OUT` only.

  Evidence: `reports/AdditionalSourcesSearch.md` §2, `reports/SoccerNetGAR_Measurements.txt`.
  **Eligibility risk accepted by the user:** a personally-signed NDA does not let the organisers
  inspect it, which RULES.md requires. Recorded, not overridden. Video (245 GB) stays unfetched.
- **SoccerTrack v2, Last Row, Signality/Hammarby, Alfheim/Tromsø, SoccerNet-Tracking and
  SoccerNet-GSR were each checked and rejected** - none carries frame-level possession state.

The sweep did find one thing of real value: **PathCRF** (KDD 2026, MPL-2.0) infers possession from
player trajectories *with no ball trajectory at all*, using a CRF with Viterbi decoding over the
same seven Sportec matches we hold. It independently validates the recommended sequence-model
architecture, and it reframes the missing-ball problem: if possession is inferable from geometry
alone, 41% missing stops being fatal and becomes a question of how much the ball adds when it is
there. Its preprocessing code is worth reading before Stage 2.

---

## 7. What I need from you

1. **Missing-ball masking parameters.** Run `local_checks/check_assets.py` and tell me the gap-length distribution it reports, or authorise a documented synthetic model instead. This is the one gap that affects model quality rather than paperwork.
2. **A decision on Metrica's licence.** It has no SPDX licence — only a request to use the data responsibly and acknowledge the source. That is probably fine for a public repository the organisers can inspect, but it is your call to make and to state in the technical document.
3. **StatsBomb's user agreement.** `raw/statsbomb/LICENSE.pdf` is collected but unread by me. It matters only if we later want their data rather than their schema.
4. ~~Gradient Sports / SoccerNet-GAR.~~ **Closed 30 Oct 2026, after the NDA was signed, the index
   downloaded and all 13,689 test clips measured.** 40.8% of match time and 66.1% of dead-ball
   time carry no tracking at all; the `team` field is not a possession label; no frame-level
   possession or ball-status field exists. The access was worth spending to settle it, and the
   answer is no. Nothing further to decide.

---

## 8. Recommended next step

**Do not start model development yet. Do one thing first: build the ball-masking harness.**

Take the IDSSE training matches — which have perfect observation and perfect labels — and mask the ball during live play to a competition-like rate and gap structure. Then verify that the frame-level and interval-level statistics of the masked corpus resemble what `check_assets.py` reports for the competition match.

The reason is the blocking finding. Every other piece is in place: the labels are reference-grade, the split is honest, the schema is deployment-compatible, and the parsers run. But a model trained on unmasked data would learn that a missing ball means a dead ball, which is the single error the published baseline already makes and the one the whole strategy is built to avoid.

After that, Stage 2 can begin with the dead-ball detector, trained on IDSSE `BallStatus` and Metrica's law-based spans, and evaluated on the two held-out matches.

---

## Reproduction

```bash
python external_data/scripts/download_sources.py --all
python external_data/scripts/parse_metrica.py
python external_data/scripts/parse_dfl.py
python external_data/scripts/parse_skillcorner.py
python external_data/scripts/make_reports.py
```

Environment: Windows 11, Python 3.14.2, numpy 2.4.1, pandas 3.0.0, pyarrow 26.0.0, kloppy 3.18.0, pypdf, ffmpeg/ffprobe 9.0.2 (local, no admin).

## Attribution required

- **IDSSE / DFL** — data provided with authorisation of the *Deutsche Fußball Liga (DFL)*. Cite: Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data, 12(1), 195. https://doi.org/10.1038/s41597-025-04505-y
- **SkillCorner Open Data** — MIT Licence, Copyright (c) 2020 SkillCorner.
- **Metrica Sports** — sample data, acknowledged as the source.

## Competition-data boundary

No competition video, tracking, tagging, homography or extract was opened at any point in this stage. The competition assets are not part of this corpus and are kept in a separate folder. The public five-minute reference was not used for training, tuning or model selection, and nothing in this stage touched it.
