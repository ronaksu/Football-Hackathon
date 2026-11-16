"""Rebuild the documented data/ layout from the scattered Google Drive download folders.

WHY THIS EXISTS
---------------
The Drive bundle arrives split across several
"More Than A Hack 2027 - Initial Challenge - Dataset-...-1-0NN" folders, one or two files each.
Every tool in the challenge repository expects ONE folder laid out as DATA.md specifies:

    <data_dir>/
      tactical/   BAR-ATH_tactical.mp4  _atd.xml  _atd.txt  _homography.csv  _smart_tagging.xml
      broadcast/  BAR-ATH_broadcast.mp4 _atd.xml  _atd.txt  _homography.csv

This script finds the nine expected files wherever they are under a source tree and puts them
into that layout. It prefers hard links, so the 19 GB of video is NOT duplicated on disk; it
falls back to a copy only when a hard link is impossible (different volume).

NOTHING LEAVES YOUR MACHINE. This script makes no network calls of any kind. It reads file names
and sizes, and creates links. It never opens the video or data payloads.

USAGE
-----
    python local_checks/consolidate_data.py --source "D:/Football Hackathon" --data-dir "D:/Football Hackathon/data"
    python local_checks/consolidate_data.py --source . --data-dir ./data --dry-run

Then point the challenge repository at it:

    $env:MTAH_DATA_DIR = "D:\\Football Hackathon\\data"      # Windows PowerShell
    export MTAH_DATA_DIR="/d/Football Hackathon/data"         # bash

Standard library only. No dependencies.
"""

import argparse
import os
import shutil
import sys

STEM = "BAR-ATH"

# The documented bundle: source folder -> the file suffixes that belong in it.
# Exactly the nine files DATA.md lists. Smart Tagging is tactical-only; the broadcast
# recording has no equivalent.
EXPECTED = {
    "tactical": (".mp4", "_atd.xml", "_atd.txt", "_homography.csv", "_smart_tagging.xml"),
    "broadcast": (".mp4", "_atd.xml", "_atd.txt", "_homography.csv"),
}


def expected_names():
    """[(source, filename)] for every file the documented bundle should contain."""
    return [(source, "{}_{}{}".format(STEM, source, suffix))
            for source, suffixes in EXPECTED.items() for suffix in suffixes]


def index_source_tree(root):
    """{filename: [full paths]} for every regular file under root.

    Keyed by base name because that is what identifies an asset: the suffix table above and the
    name both carry the source, so a file is identifiable even after being moved out of its
    folder. That redundancy is deliberate in the bundle and this relies on it.
    """
    found = {}
    for directory, _, files in os.walk(root):
        for name in files:
            found.setdefault(name, []).append(os.path.join(directory, name))
    return found


def place(source_path, target_path, dry_run=False):
    """Hard-link source_path to target_path, falling back to a copy. Returns how it was placed.

    A hard link is two directory entries for one set of blocks, so the 14 GB broadcast video is
    not written twice and both paths stay in sync. It only works within one volume, hence the
    fallback.
    """
    if os.path.exists(target_path):
        if os.path.samefile(source_path, target_path):
            return "already linked"
        if os.path.getsize(source_path) == os.path.getsize(target_path):
            return "already present"
        return "PRESENT BUT DIFFERENT SIZE -- left alone, check by hand"

    if dry_run:
        return "would link"

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    try:
        os.link(source_path, target_path)
        return "hard link"
    except OSError:
        shutil.copy2(source_path, target_path)
        return "copied"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=".",
                        help="folder tree holding the downloaded files (searched recursively)")
    parser.add_argument("--data-dir", default="./data",
                        help="where to build the tactical/ and broadcast/ layout")
    parser.add_argument("--dry-run", action="store_true", help="report, change nothing")
    args = parser.parse_args()

    source_root = os.path.abspath(args.source)
    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(source_root):
        print("no such source folder: {}".format(source_root))
        return 2

    print("source   {}".format(source_root))
    print("data dir {}{}".format(data_dir, "   (dry run)" if args.dry_run else ""))
    print()

    found = index_source_tree(source_root)
    missing, ambiguous, placed = [], [], 0

    for source, name in sorted(expected_names()):
        target = os.path.join(data_dir, source, name)
        candidates = [p for p in found.get(name, []) if os.path.abspath(p) != target]
        if not candidates:
            missing.append(name)
            print("  MISSING   {:<44} not found under the source tree".format(name))
            continue
        if len(candidates) > 1:
            # Several copies of the same name. Take the largest, which is the complete download
            # if one of them is a partial file, and say so rather than choosing silently.
            candidates.sort(key=os.path.getsize, reverse=True)
            ambiguous.append(name)
        how = place(candidates[0], target, args.dry_run)
        size_gb = os.path.getsize(candidates[0]) / 1e9
        print("  {:<9} {:<44} {:>8.3f} GB  {}".format(
            "ok", name, size_gb, how))
        if len(candidates) > 1:
            print("            {} copies found; used the largest".format(len(candidates)))
        placed += 1

    print("\n{} of {} expected files placed".format(placed, len(expected_names())))
    if ambiguous:
        print("duplicate names resolved by size: {}".format(", ".join(sorted(ambiguous))))
    if missing:
        print("\nSTILL MISSING: {}".format(", ".join(missing)))
        print("The bundle is incomplete -- check the Drive download rather than working around it.")
        return 1

    print("\nComplete. Point the challenge repository at it:")
    print('  $env:MTAH_DATA_DIR = "{}"'.format(data_dir.replace("/", "\\")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
