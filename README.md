# Ball possession from tracking — More Than A Hack 2027

Per-frame ball possession (`H` / `A` / `D` / `X`) for the tactical recording of
FC Barcelona vs Athletic Club, produced by an algorithm from the Automatic Tracking Data.

The method, the training data and the limitations are in
**[TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md)**. The architecture is in
**[PROJECT_GUIDE.md](PROJECT_GUIDE.md)**. This file is how to run it.

---

## What is here, and what is deliberately not

**Not here: the match data.** The assets accepted at enrolment are not redistributable, so
`.gitignore` in this repository is a **whitelist** — everything is excluded unless it is named,
and the match bundle, every file derived from it and the submission file itself are excluded by
rules that run last and override every whitelist above them. Nothing derived from the provided
game is published here.

**Here: everything needed to regenerate the submission.** The two trained models are committed
(5 MB), so the file can be reproduced from the data bundle alone without re-downloading the
external training data.

```
submit/                  the submission pipeline
  atd_to_canonical.py      the only file that knows what a Metrica EPTS export is
  fit_final.py             fits the two models on the external training matches and freezes them
  predict_match.py         frozen models -> <team_name>.csv
  run_submission.py        all of the above in one command
  models/                  the frozen models, duration priors and a manifest
stage2/                  the dead-ball detector: is play stopped?
stage3/                  team attribution, and restart_logic.py, the hardening layer
external_data/scripts/   the parsers that built the training data, one per source
review/                  the local failure-review tool used to find the model's mistakes
local_checks/            offline verification of the delivered assets
```

## Reproducing the submission

Needs the data bundle and about three minutes.

```bash
python -m pip install -r requirements.txt
```

Put the bundle in the layout `DATA.md` documents — `data/tactical/BAR-ATH_tactical_atd.txt` and
so on. If the download arrived split across folders, this rebuilds the layout with hard links
rather than copying 19 GB:

```bash
python local_checks/consolidate_data.py --source <download folder> --data-dir ./data
```

Then one command produces and validates the file:

```bash
python submit/run_submission.py --team-name <team_name> --data-dir ./data
```

It prints a SHA-256 of what it wrote. The submitted file's digest is in
[TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md), so a reproduction can be checked byte for byte
rather than approximately.

### The same thing in steps, if a step needs inspecting

```bash
python submit/atd_to_canonical.py --data-dir ./data --out-dir submit/processed
```

```bash
python stage2/features.py --proc-dir submit/processed --matches BAR-ATH --out-dir submit/work/stage2_features
```

```bash
python stage3/features_team.py --proc-dir submit/processed --matches BAR-ATH --out-dir submit/work/stage3_features
```

```bash
python stage3/pitch_control.py --proc-dir submit/processed --matches BAR-ATH --out-dir submit/work/pitch_control
```

```bash
python submit/predict_match.py --team-name <team_name>
```

## Verifying the assets before trusting any of it

Runs offline, makes no network calls, and prints PASS or MISMATCH against every number
`DATA.md` publishes:

```bash
python local_checks/check_assets.py --data-dir ./data
```

The adapter does the same for the figures it depends on. On the delivered bundle it reports
199,319 ATD records, 158,857 graded frames, the ball on 94,214 of them (59.3%) and 20.5 tracked
objects per frame — each matching the published value exactly.

## Re-fitting the models from scratch

Only needed to verify the models themselves; it requires the **external** training data, not the
competition bundle. The sources, their licences and their download steps are in
`external_data/scripts/sources.json` and `external_data/DataReadinessReport.md`.

```bash
python external_data/scripts/download_sources.py
```

```bash
python external_data/scripts/parse_dfl.py
```

```bash
python stage2/features.py --matches DFL-MAT-J03WOH DFL-MAT-J03WOY DFL-MAT-J03WPY DFL-MAT-J03WQQ DFL-MAT-J03WR9 --out-dir stage2/features_h
```

```bash
python stage3/features_team.py && python stage3/pitch_control.py
```

```bash
python submit/fit_final.py
```

To reproduce the reported development scores instead — leave-one-match-out over the five
training matches, scored with the organisers' own `possession_metrics.py`:

```bash
python stage2/train.py --rounds 400
```

```bash
python stage3/train_team.py --pitch-control stage3/pitch_control --save-pred stage3/predictions_team --only-rate 0.41
```

```bash
python stage3/restart_logic.py --compare --apply
```

## The rules this was built under

- **The labels are produced by an algorithm.** Nothing in the provided game was annotated or
  corrected by hand, and no hand label on it exists anywhere in this project.
- **No ground truth was used, public or private.** The models are trained on open-access
  tracking of other matches. The published five-minute window was used once, after the model was
  frozen, to check that the pipeline was right — never to train, tune or select anything. What it
  was used for and what it said are both in the technical document.
- **The two sealed matches stayed sealed.** Two of the seven training-source matches were set
  aside at the start and have never been loaded by any code in this repository.
