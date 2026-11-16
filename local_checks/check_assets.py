"""Verify the downloaded match assets on your own machine, and nowhere else.

WHY THIS EXISTS
---------------
The competition FAQ restates the enrolment terms: participants must not upload, disclose or
otherwise make available to any third-party hosted or cloud LLM any videos, datasets or
restricted materials provided with the competition, "or any extracts thereof".

A cloud-hosted assistant therefore cannot read these files, and neither can it read a screenshot,
an extracted frame, a printed data row or a pasted log from them. This script is the way the
checks still get done: it runs locally, writes its report to a local folder, and MAKES NO NETWORK
CALLS AT ALL. Read the report yourself and decide what, if anything, you share.

WHAT IT CHECKS
--------------
  inventory   the nine documented files exist, with sizes (and optional SHA-256)
  video       container metadata, and optionally an exact decoded frame count
  atd         record count and contiguity, ball-missing rate, tracked-object counts,
              track-slot fragmentation, implausible ball jumps
  tagging     instance and code counts, tag vocabularies, N/A teams, interval overlaps,
              and whether the time base really is "seconds from frame 0"
  homography  row count, valid fraction, reprojection-error quantiles, invalid stretches

Every check prints PASS or MISMATCH against the numbers DATA.md publishes, so a disagreement
between the delivered files and the documentation shows up as a disagreement rather than being
silently absorbed. Where a count differs, DO NOT pad or trim anything to force it -- the FAQ is
explicit that a decode which does not give 199,319 frames is the wrong decode, and the cause is
what matters. Open an issue on the challenge repository.

USAGE
-----
    python local_checks/check_assets.py --data-dir "D:/Football Hackathon/data"
    python local_checks/check_assets.py --data-dir ./data --only atd tagging
    python local_checks/check_assets.py --data-dir ./data --count-frames     # slow, exact decode
    python local_checks/check_assets.py --data-dir ./data --sha256           # slow, 19 GB of video

Needs numpy and pandas (the same two the scorer needs). ffprobe is used for video metadata if it
is on PATH; without it the video section reports "ffprobe not found" and the rest still runs.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pandas as pd

# -- what the documentation says, so every check has something to disagree with ----------------
# Source: DATA.md and games.py at commit f8f2629 of metrica-sports/more-than-a-hack-2027.
DOCUMENTED = {
    "fps": 25,
    "tactical": {
        "n_video_frames": 199319,
        "n_track_slots": 46,
        "ball_present_all": 0.569,          # 113488 / 199319
        "ball_present_graded": 0.593,       # 94214 / 158857
        "homography_valid": 1.000,
        "n_tagging_instances": 638,
        "resolution": (1920, 1080),
    },
    "broadcast": {
        "n_video_frames": 164488,
        "n_track_slots": 51,
        "ball_present_all": 0.505,
        "homography_valid": 0.658,
    },
    # Published graded blocks, inclusive, in tactical video frames. These are boundaries, not
    # labels: DATA.md publishes them precisely because they reveal nothing about any answer.
    "graded_blocks": [(11185, 90651), (110808, 190197)],
    "n_graded": 158857,
    "kickoffs": {"1st Half": 11185, "2nd Half": 113742},
    "pitch_m": (105.3, 68.0),
}

STEM = "BAR-ATH"
SUFFIXES = {
    "video": ".mp4",
    "atd_meta": "_atd.xml",
    "atd_raw": "_atd.txt",
    "homography": "_homography.csv",
    "smart_tagging": "_smart_tagging.xml",
}
ASSETS = {
    "tactical": ("video", "atd_meta", "atd_raw", "homography", "smart_tagging"),
    "broadcast": ("video", "atd_meta", "atd_raw", "homography"),
}

FIELD_SEP, GROUP_SEP, MISSING = ":", ";", "N"

_lines = []


def say(text=""):
    """Print and record, so the console and the written report cannot drift apart."""
    print(text)
    _lines.append(text)


def verdict(label, actual, expected, tolerance=0, fmt="{}"):
    """One PASS/MISMATCH line comparing a measurement against the documented value."""
    if expected is None:
        say("    {:<34} {}".format(label, fmt.format(actual)))
        return None
    ok = abs(actual - expected) <= tolerance
    say("    {:<34} {:<16} doc {:<14} {}".format(
        label, fmt.format(actual), fmt.format(expected), "PASS" if ok else "MISMATCH"))
    return ok


def paths_for(data_dir, source):
    return {name: os.path.join(data_dir, source, "{}_{}{}".format(STEM, source, SUFFIXES[name]))
            for name in ASSETS[source]}


def graded_mask(n_frames):
    """Boolean over video frames, True inside a published graded block."""
    mask = np.zeros(n_frames, dtype=bool)
    for first, last in DOCUMENTED["graded_blocks"]:
        if first < n_frames:
            mask[first:min(last + 1, n_frames)] = True
    return mask


# -- inventory ---------------------------------------------------------------------------------

def sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def check_inventory(data_dir, want_sha):
    say("=" * 86)
    say("INVENTORY  -- are the nine documented files here?")
    say("=" * 86)
    complete = True
    for source in ("tactical", "broadcast"):
        say("  {}".format(source))
        for name, path in sorted(paths_for(data_dir, source).items()):
            if not os.path.isfile(path):
                say("    MISSING  {:<16} {}".format(name, path))
                complete = False
                continue
            size = os.path.getsize(path)
            extra = "  sha256 {}".format(sha256(path)) if want_sha else ""
            say("    ok       {:<16} {:>12,} bytes ({:>7.3f} GB){}".format(
                name, size, size / 1e9, extra))
            if size == 0:
                say("             ZERO BYTES -- the download did not complete")
                complete = False
    say()
    if not complete:
        say("  The bundle is incomplete. Re-download rather than working around the gap.")
    say()
    return complete


# -- video -------------------------------------------------------------------------------------

def check_video(data_dir, exact):
    say("=" * 86)
    say("VIDEO  -- does the decode agree with the 199,319-row submission contract?")
    say("=" * 86)
    if not shutil.which("ffprobe"):
        say("  ffprobe not found on PATH. Install ffmpeg to run this section.")
        say("  This matters: the FAQ says a decode that is not 199,319 frames is the wrong")
        say("  decode, and the row-count check will reject a submission built from it.")
        say()
        return
    for source in ("tactical", "broadcast"):
        path = paths_for(data_dir, source)["video"]
        if not os.path.isfile(path):
            say("  {}: no video file".format(source))
            continue
        say("  {}".format(source))
        command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-print_format", "json",
                   "-show_streams"]
        if exact:
            command.append("-count_frames")
        try:
            raw = subprocess.run(command + [path], capture_output=True, text=True, timeout=14400)
            stream = json.loads(raw.stdout)["streams"][0]
        except Exception as error:                      # noqa: BLE001 - report, never raise
            say("    ffprobe failed: {}".format(error))
            continue

        doc = DOCUMENTED[source]
        rate = stream.get("avg_frame_rate", "0/1")
        numerator, _, denominator = rate.partition("/")
        fps = float(numerator) / float(denominator or 1)
        verdict("fps (avg_frame_rate)", fps, DOCUMENTED["fps"], 0.01, "{:.3f}")
        verdict("width", int(stream.get("width", 0)), doc.get("resolution", (None,))[0])
        verdict("height", int(stream.get("height", 0)),
                doc["resolution"][1] if "resolution" in doc else None)
        container = stream.get("nb_frames")
        if container is not None:
            verdict("nb_frames (container)", int(container), doc["n_video_frames"])
        if exact:
            counted = stream.get("nb_read_frames")
            if counted is not None:
                ok = verdict("decoded frames (exact)", int(counted), doc["n_video_frames"])
                if ok is False:
                    say("      Do NOT pad or trim to fit. Work out why the counts differ and")
                    say("      report it, so every team uses the same frame index.")
        else:
            say("    (container metadata only; --count-frames decodes every frame, slowly)")
        say("    codec {}  duration {} s".format(
            stream.get("codec_name"), stream.get("duration")))
    say()


# -- ATD ---------------------------------------------------------------------------------------

def team_of_each_slot(meta_path):
    """(team_code_per_slot, (home_name, away_name)). First Team in the file is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    names = tuple(team.findtext("Name") for team in root.iter("Team"))
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8), names


def check_atd(data_dir, source):
    say("=" * 86)
    say("ATD ({})  -- alignment, ball coverage, track fragmentation".format(source))
    say("=" * 86)
    paths = paths_for(data_dir, source)
    if not (os.path.isfile(paths["atd_raw"]) and os.path.isfile(paths["atd_meta"])):
        say("  missing atd_raw or atd_meta; skipped\n")
        return
    doc = DOCUMENTED[source]
    expected = doc["n_video_frames"]

    slot_team, names = team_of_each_slot(paths["atd_meta"])
    say("  metadata")
    say("    teams                          H = {}   A = {}".format(*names))
    verdict("track slots", len(slot_team), doc["n_track_slots"])
    say("    slots per team                 home {}  away {}".format(
        int((slot_team == 0).sum()), int((slot_team == 1).sum())))

    n_slots = len(slot_team)
    ball_present = np.zeros(expected + 8, dtype=bool)
    n_tracked = np.zeros(expected + 8, dtype=np.int16)
    slot_present = np.zeros((n_slots, expected + 8), dtype=bool)
    ball_xy = np.full((expected + 8, 2), np.nan, dtype=np.float32)

    first_id = last_id = None
    count = malformed = overflow = 0
    with open(paths["atd_raw"]) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                malformed += 1
                continue
            try:
                frame_id = int(parts[0])
            except ValueError:
                malformed += 1
                continue
            if first_id is None:
                first_id = frame_id
            last_id = frame_id
            count += 1
            row = frame_id - 1                       # video_frame = frame_id - 1
            if not 0 <= row < expected + 8:
                overflow += 1
                continue

            groups = parts[1].split(GROUP_SEP)
            seen = 0
            for slot, group in enumerate(groups):
                if group and group[0] != MISSING:
                    seen += 1
                    if slot < n_slots:
                        slot_present[slot, row] = True
            n_tracked[row] = seen

            ball = parts[2].strip()
            if ball and ball[0] != MISSING:
                ball_present[row] = True
                bits = ball.split(",")
                if len(bits) >= 2:
                    try:
                        ball_xy[row] = (float(bits[0]), float(bits[1]))
                    except ValueError:
                        pass

    say("  alignment")
    verdict("records", count, expected)
    say("    frame_id range                 {} .. {}".format(first_id, last_id))
    contiguous = count > 0 and last_id - first_id + 1 == count
    say("    contiguous ids                 {}".format(contiguous))
    say("    starts at 1 (so row = id - 1)  {}".format(first_id == 1))
    if malformed:
        say("    MALFORMED LINES                {}".format(malformed))
    if overflow:
        say("    ids outside 1..{}      {}".format(expected, overflow))
    if not (count == expected and contiguous and first_id == 1):
        say("    MISMATCH -- this export is not one record per video frame numbered from 1.")
        say("    Do not invent a correction; report it so every team shares one alignment.")

    present = ball_present[:expected]
    tracked = n_tracked[:expected]
    say("  ball coverage")
    verdict("present, whole video", float(present.mean()), doc.get("ball_present_all"),
            0.002, "{:.3f}")
    if source == "tactical":
        mask = graded_mask(expected)
        verdict("graded frames counted", int(mask.sum()), DOCUMENTED["n_graded"])
        verdict("present, graded frames", float(present[mask].mean()),
                doc.get("ball_present_graded"), 0.002, "{:.3f}")
        say("    => missing on {:.1f}% of graded frames".format(100 * (1 - present[mask].mean())))

    # Missing-ball gap structure. This is the single most consequential property for design:
    # a 3-frame gap and a 40-second gap need completely different handling.
    gaps = runs_of(~present)
    if gaps:
        lengths = np.array([b - a for a, b in gaps])
        say("  missing-ball gaps  ({} gaps over the whole video)".format(len(lengths)))
        say("    length in frames               median {}  p90 {}  p99 {}  max {}".format(
            int(np.median(lengths)), int(np.percentile(lengths, 90)),
            int(np.percentile(lengths, 99)), int(lengths.max())))
        for threshold in (5, 12, 25, 75, 250):
            share = float((lengths <= threshold).mean())
            frames = int(lengths[lengths <= threshold].sum())
            say("    gaps <= {:>4} frames            {:>6.1%} of gaps, {:>8,} frames".format(
                threshold, share, frames))
        say("    (25 frames = 1 s. Short gaps are occlusions to bridge; long ones are")
        say("     candidates for a genuine stoppage, but absence is not evidence of a dead ball.)")

    say("  tracked objects per frame")
    say("    mean / median                  {:.1f} / {}".format(
        float(tracked.mean()), int(np.median(tracked))))
    if source == "tactical":
        say("    mean over graded frames        {:.1f}  (DATA.md says 20.5)".format(
            float(tracked[graded_mask(expected)].mean())))
    say("    frames with > 22 tracked       {:,}  (tracker duplicating someone)".format(
        int((tracked > 22).sum())))
    say("    frames with < 20 tracked       {:,}".format(int((tracked < 20).sum())))

    say("  track-slot fragmentation")
    occupancy = slot_present[:, :expected]
    starts = np.logical_and(occupancy[:, 1:], ~occupancy[:, :-1]).sum(axis=1) + occupancy[:, 0]
    coverage = occupancy.mean(axis=1)
    say("    slots present on >90% frames   {} of {}".format(
        int((coverage > 0.9).sum()), n_slots))
    say("    slots present on <10% frames   {}".format(int((coverage < 0.1).sum())))
    say("    appearances per slot           median {}  max {}".format(
        int(np.median(starts)), int(starts.max())))
    say("    total slot appearances         {:,}".format(int(starts.sum())))
    say("    (A slot is a tracklet, not a player: 46 slots for 22 players, and a slot that")
    say("     stops and restarts may well be a different person. Never treat one as an identity.)")

    say("  ball-motion plausibility")
    step = np.linalg.norm(np.diff(ball_xy[:expected], axis=0), axis=1)
    both = present[:-1] & present[1:]
    step = step[both]
    if step.size:
        length, width = DOCUMENTED["pitch_m"]
        metres = step * np.hypot(length, width)       # upper bound on displacement in metres
        speed = metres * DOCUMENTED["fps"]
        say("    frame-to-frame speed (m/s)     median {:.1f}  p99 {:.1f}  max {:.1f}".format(
            float(np.median(speed)), float(np.percentile(speed, 99)), float(speed.max())))
        for limit in (40, 60, 100):
            say("    steps implying > {:>3} m/s       {:,}".format(
                limit, int((speed > limit).sum())))
        say("    (A struck ball rarely exceeds ~40 m/s. Anything far above that is a detection")
        say("     jump, not a kick, and a possession rule reading it as motion will be fooled.)")
    say()


def runs_of(flag):
    """[(start, stop)] half-open runs where the boolean array is True."""
    flag = flag.astype(np.int8)
    edges = np.diff(np.concatenate(([0], flag, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


# -- Smart Tagging -----------------------------------------------------------------------------

def check_tagging(data_dir):
    say("=" * 86)
    say("SMART TAGGING  -- vocabulary, time base, interval structure")
    say("=" * 86)
    path = paths_for(data_dir, "tactical")["smart_tagging"]
    if not os.path.isfile(path):
        say("  missing; skipped\n")
        return
    fps = DOCUMENTED["fps"]
    events = []
    for instance in ET.parse(path).iter("instance"):
        start_s, end_s = float(instance.findtext("start")), float(instance.findtext("end"))
        events.append({
            "code": instance.findtext("code"),
            "start_frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
            "duration_s": end_s - start_s,
            "tags": {label.findtext("group"): label.findtext("text")
                     for label in instance.findall("label")},
        })

    verdict("instances", len(events), DOCUMENTED["tactical"]["n_tagging_instances"])
    say("  codes")
    for code, n in Counter(e["code"] for e in events).most_common():
        say("    {:<34} {:>4}".format(str(code), n))

    groups = Counter(g for e in events for g in e["tags"])
    say("  tag groups present                {}".format(dict(groups)))
    for group in ("Team", "Half", "Type"):
        values = Counter(e["tags"].get(group) for e in events if group in e["tags"])
        if values:
            say("  {} values".format(group))
            for value, n in values.most_common():
                say("    {:<34} {:>4}".format(str(value), n))

    na = [e for e in events if (e["tags"].get("Team") or "").strip().upper() in ("N/A", "NA", "")]
    say("  instances with no usable Team     {}  (DATA.md says 36 are N/A)".format(len(na)))
    if na:
        say("    their codes                    {}".format(dict(Counter(e["code"] for e in na))))
        types = Counter(e["tags"].get("Type") for e in na if e["tags"].get("Type"))
        if types:
            say("    their set-piece types          {}".format(dict(types)))
        say("    Handle these explicitly. A missing team tag is not a team, and must not fall")
        say("    through to a default -- every throw-in is reportedly in this group.")

    say("  time base")
    kickoffs = [e for e in events
                if (e["code"] or "").strip().lower() == "set pieces"
                and (e["tags"].get("Type") or "").strip().lower() == "kick off (start)"]
    say("    'Kick Off (Start)' tags         {}".format(len(kickoffs)))
    for half, frame in sorted(DOCUMENTED["kickoffs"].items()):
        tagged = [e for e in kickoffs if e["tags"].get("Half") == half]
        contains = [e for e in tagged if e["start_frame"] <= frame <= e["end_frame"]]
        say("    {:<10} kickoff frame {:>6}  tagged {}  contains it: {}".format(
            half, frame, [(e["start_frame"], e["end_frame"]) for e in tagged], bool(contains)))
    say("    If a kickoff is not contained, the XML is on a different clock. Do not add a")
    say("    constant to make it fit -- report it.")

    say("  interval structure")
    durations = np.array([e["duration_s"] for e in events])
    say("    duration (s)                   median {:.1f}  p90 {:.1f}  max {:.1f}".format(
        float(np.median(durations)), float(np.percentile(durations, 90)), float(durations.max())))
    say("    zero/negative durations        {}".format(int((durations <= 0).sum())))
    ordered = sorted(events, key=lambda e: e["start_frame"])
    overlaps = sum(1 for a, b in zip(ordered, ordered[1:]) if b["start_frame"] < a["end_frame"])
    say("    overlapping consecutive pairs   {} of {}".format(overlaps, len(ordered) - 1))
    say("    (Phases overlap by design -- several codes describe the same passage of play. An")
    say("     interval is a window around an event, not the instant of it, so a set-piece tag")
    say("     locates a restart to about a second, not to a frame.)")
    covered = np.zeros(DOCUMENTED["tactical"]["n_video_frames"], dtype=bool)
    for e in ordered:
        lo = max(0, e["start_frame"])
        hi = min(len(covered), e["end_frame"] + 1)
        if hi > lo:
            covered[lo:hi] = True
    mask = graded_mask(len(covered))
    say("    graded frames inside some tag   {:.1%}".format(float(covered[mask].mean())))
    say()


# -- homography --------------------------------------------------------------------------------

def check_homography(data_dir, source):
    say("=" * 86)
    say("HOMOGRAPHY ({})".format(source))
    say("=" * 86)
    path = paths_for(data_dir, source)["homography"]
    if not os.path.isfile(path):
        say("  missing; skipped\n")
        return
    doc = DOCUMENTED[source]
    table = pd.read_csv(path)
    matrix_columns = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
    missing = [c for c in ["frame", "reprojection_error", "valid"] + matrix_columns
               if c not in table.columns]
    if missing:
        say("    MISSING COLUMNS {}\n".format(missing))
        return

    verdict("rows", len(table), doc["n_video_frames"])
    frames = table["frame"].values
    say("    frame column is 0..n-1         {}".format(
        bool(np.array_equal(frames, np.arange(len(frames))))))
    valid = table["valid"].values.astype(bool)
    verdict("valid fraction", float(valid.mean()), doc.get("homography_valid"), 0.002, "{:.3f}")

    nan_rows = table[matrix_columns].isna().any(axis=1).values
    say("    rows with NaN in the matrix    {:,}".format(int(nan_rows.sum())))
    say("    valid=0 but matrix not NaN     {:,}  (should be 0)".format(
        int((~valid & ~nan_rows).sum())))
    say("    valid=1 but matrix has NaN     {:,}  (should be 0)".format(
        int((valid & nan_rows).sum())))

    if valid.any():
        error = table["reprojection_error"].values[valid]
        say("    reprojection error             median {:.3f}  p90 {:.3f}  p99 {:.3f}  max {:.3f}"
            .format(float(np.median(error)), float(np.percentile(error, 90)),
                    float(np.percentile(error, 99)), float(error.max())))
    invalid = runs_of(~valid)
    if invalid:
        lengths = np.array([b - a for a, b in invalid])
        say("    invalid stretches              {}  longest {} frames ({:.1f} s)".format(
            len(invalid), int(lengths.max()), lengths.max() / DOCUMENTED["fps"]))
    say("    Check `valid` before using a matrix; `reprojection_error` says how far to trust")
    say("    the one you have. Lower is better and there is no upper bound.")
    say()


# -- main --------------------------------------------------------------------------------------

SECTIONS = ("inventory", "video", "atd", "tagging", "homography")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR", "./data"),
                        help="folder holding tactical/ and broadcast/ (default: $MTAH_DATA_DIR)")
    parser.add_argument("--only", nargs="+", choices=SECTIONS, default=list(SECTIONS),
                        help="run only these sections")
    parser.add_argument("--count-frames", action="store_true",
                        help="decode every video frame for an exact count (slow)")
    parser.add_argument("--sha256", action="store_true",
                        help="hash every file, including 19 GB of video (slow)")
    parser.add_argument("--report", default=None,
                        help="where to write the report (default: local_checks/reports/)")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    say("LOCAL ASSET VERIFICATION -- nothing in this report leaves your machine")
    say("data dir   {}".format(data_dir))
    say("sections   {}".format(", ".join(args.only)))
    say("reference  DATA.md / games.py, metrica-sports/more-than-a-hack-2027 @ f8f2629")
    say()
    if not os.path.isdir(data_dir):
        say("No such folder. Run consolidate_data.py first, or pass --data-dir.")
        return 2

    if "inventory" in args.only:
        check_inventory(data_dir, args.sha256)
    if "video" in args.only:
        check_video(data_dir, args.count_frames)
    if "atd" in args.only:
        for source in ("tactical", "broadcast"):
            check_atd(data_dir, source)
    if "tagging" in args.only:
        check_tagging(data_dir)
    if "homography" in args.only:
        for source in ("tactical", "broadcast"):
            check_homography(data_dir, source)

    say("=" * 86)
    say("Read every MISMATCH above before building anything on these files.")
    say("A disagreement with the documentation is a question for the organisers, not something")
    say("to pad, trim or silently correct. Open an issue on the challenge repository.")
    say("=" * 86)

    out = args.report or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "reports", "asset_check.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(_lines) + "\n")
    print("\nreport written to {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
