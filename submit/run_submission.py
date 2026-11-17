"""One command from the delivered data bundle to a validated submission file.

    python submit/run_submission.py --team-name MyTeam --data-dir ./data

Runs, in order:

    1. submit/atd_to_canonical.py   ATD pair            -> canonical frames + players
    2. stage2/features.py           player geometry      -> 169 dead-ball features
    3. stage3/features_team.py      geometry + the ball  -> 37 team features
    4. stage3/pitch_control.py      space control        -> 19 more team features
    5. submit/predict_match.py      the frozen models    -> <team_name>.csv
    6. evaluation/validate_submission.py                 -> the organisers' own format check

Nothing is trained: step 5 loads submit/models/, which submit/fit_final.py produced from the
open-access training matches. Re-fitting those models is a separate command and is documented in
README.md, because it needs the external training data rather than the competition bundle.

Each step is a normal script with its own arguments; this only chains them with consistent paths
so that a reproduction cannot go wrong by passing one of them a different directory.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MATCH_ID = "BAR-ATH"


def run(label, args):
    print("\n" + "=" * 78)
    print("{}  {}".format(label, " ".join(os.path.basename(a) if os.sep in a else a
                                          for a in args[1:3])))
    print("=" * 78, flush=True)
    done = subprocess.run([sys.executable] + args)
    if done.returncode:
        raise SystemExit("\n{} failed with exit code {}".format(label, done.returncode))


def find_validator():
    """The organisers' validate_submission.py, wherever the challenge repository is.

    It is not vendored here -- it is theirs, and a stale copy of a validator is worse than none.
    So this looks in the places a checkout of it plausibly sits relative to this one: inside this
    repository, beside it, or one level up. MTAH_REPO overrides all of them. Returns None rather
    than guessing, and the caller then prints the command to run by hand.
    """
    env = os.environ.get("MTAH_REPO")
    roots = ([env] if env else []) + [ROOT, os.path.dirname(ROOT)]
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        direct = os.path.join(root, "evaluation", "validate_submission.py")
        if os.path.exists(direct):
            return direct
        for entry in sorted(os.listdir(root)):
            if not entry.lower().startswith("more-than-a-hack"):
                continue
            outer = os.path.join(root, entry)
            # A GitHub zip of the challenge unpacks to a folder containing a folder, and the two
            # are not always named the same once anyone has renamed either, so the inner level is
            # scanned rather than guessed.
            candidates = [outer] + [os.path.join(outer, child)
                                    for child in sorted(os.listdir(outer))
                                    if os.path.isdir(os.path.join(outer, child))]
            for base in candidates:
                found = os.path.join(base, "evaluation", "validate_submission.py")
                if os.path.exists(found):
                    return found
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team-name", required=True)
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--work-dir", default=os.path.join(HERE, "work"))
    ap.add_argument("--proc-dir", default=os.path.join(HERE, "processed"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    ap.add_argument("--skip-adapter", action="store_true",
                    help="reuse an existing canonical parquet instead of re-reading the ATD")
    args = ap.parse_args()
    started = time.time()

    if not args.skip_adapter:
        run("1/6  ATD -> canonical",
            [os.path.join(HERE, "atd_to_canonical.py"),
             "--data-dir", args.data_dir, "--out-dir", args.proc_dir])

    run("2/6  stage 2 features",
        [os.path.join(ROOT, "stage2", "features.py"),
         "--proc-dir", args.proc_dir, "--matches", MATCH_ID,
         "--out-dir", os.path.join(args.work_dir, "stage2_features")])

    run("3/6  stage 3 team features",
        [os.path.join(ROOT, "stage3", "features_team.py"),
         "--proc-dir", args.proc_dir, "--matches", MATCH_ID,
         "--out-dir", os.path.join(args.work_dir, "stage3_features")])

    run("4/6  pitch control",
        [os.path.join(ROOT, "stage3", "pitch_control.py"),
         "--proc-dir", args.proc_dir, "--matches", MATCH_ID,
         "--out-dir", os.path.join(args.work_dir, "pitch_control")])

    run("5/6  predict",
        [os.path.join(HERE, "predict_match.py"),
         "--team-name", args.team_name, "--proc-dir", args.proc_dir,
         "--work-dir", args.work_dir, "--out-dir", args.out_dir])

    csv = os.path.join(args.out_dir, args.team_name + ".csv")
    validator = find_validator()
    if validator:
        run("6/6  validate", [validator, csv])
    else:
        print("\n6/6  validate: could not find the challenge repository's "
              "evaluation/validate_submission.py. Point MTAH_REPO at your checkout of it, or "
              "run it yourself before uploading -- do not skip it:"
              "\n     python evaluation/validate_submission.py \"{}\"".format(csv))

    digest = hashlib.sha256(open(csv, "rb").read()).hexdigest()
    print("\n" + "=" * 78)
    print("submission  {}".format(csv))
    print("sha256      {}".format(digest))
    print("total       {:.0f}s".format(time.time() - started))
    print("=" * 78)


if __name__ == "__main__":
    main()
