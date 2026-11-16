"""Stage 1 acquisition: download the permitted external datasets and record exactly what arrived.

    python external_data/scripts/download_sources.py --list
    python external_data/scripts/download_sources.py --source metrica idsse skillcorner statsbomb
    python external_data/scripts/download_sources.py --all

Behaviour that matters:

  - File lists are DISCOVERED from each host's API, not hard-coded, so the manifest reflects the
    tree as it is today rather than as a tutorial once described it.
  - Downloads resume. A partial file is continued with an HTTP Range request rather than
    restarted, so a dropped connection on a 400 MB XML costs seconds, not minutes.
  - Every finished file is sha256'd and its size recorded. A file already on disk with the
    expected size is skipped, so re-running is cheap and does not re-fetch from a mirror.
  - Raw originals are written under raw/<source>/ and are never modified afterwards.
  - Nothing here touches the competition match. The competition assets stay outside this corpus.

Writes reports/DatasetManifest.csv.

Standard library only.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # external_data/
RAW = os.path.join(ROOT, "raw")
REPORTS = os.path.join(ROOT, "reports")
CONFIG = os.path.join(HERE, "sources.json")

UA = {"User-Agent": "mtah2027-stage1-collector/1.0 (research; contact via project owner)"}
CHUNK = 1 << 20


# -- discovery ---------------------------------------------------------------------------------

def _get_json(url, tries=3):
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:                      # noqa: BLE001 - retry, then report
            if attempt == tries - 1:
                print("    discovery failed: {}".format(error))
                return None
            time.sleep(2 * (attempt + 1))
    return None


def discover(key, spec):
    """[(relative_path, declared_size_or_None)] plus a version string for this source."""
    how = spec.get("discovery", {})
    kind = how.get("kind")
    wanted, version = [], ""

    if kind == "github_tree":
        url = "https://api.github.com/repos/{}/git/trees/{}?recursive=1".format(
            how["repo"], how["ref"])
        tree = _get_json(url)
        if not tree or "tree" not in tree:
            return [], ""
        version = tree.get("sha", "")[:12]
        if tree.get("truncated"):
            print("    WARNING: GitHub truncated the tree listing for {}".format(key))
        for entry in tree["tree"]:
            if entry["type"] != "blob":
                continue
            wanted.append((entry["path"], entry.get("size")))

    elif kind == "github_contents":
        for path in how["paths"]:
            url = "https://api.github.com/repos/{}/contents/{}".format(how["repo"], path)
            listing = _get_json(url)
            if not isinstance(listing, list):
                continue
            for entry in listing:
                if entry.get("type") == "file":
                    wanted.append((entry["path"], entry.get("size")))

    elif kind == "hf_dataset":
        info = _get_json("https://huggingface.co/api/datasets/{}".format(how["repo"]))
        if not info:
            return [], ""
        version = (info.get("sha") or "")[:12]
        for sibling in info.get("siblings", []):
            wanted.append((sibling["rfilename"], None))

    elif kind == "none":
        return [], ""

    return _select(wanted, spec), version


def _select(found, spec):
    """Apply the include/exclude rules from the config."""
    prefixes = tuple(spec.get("include_prefixes", []))
    explicit = set(spec.get("include_files", []))
    banned = spec.get("exclude_substrings", [])
    out = []
    for path, size in found:
        if any(bad in path for bad in banned):
            continue
        if path in explicit or (prefixes and path.startswith(prefixes)):
            out.append((path, size))
    return sorted(set(out))


# -- download ----------------------------------------------------------------------------------

def is_lfs(spec, path):
    """True when this file is stored in Git LFS rather than as an ordinary blob."""
    return bool(spec.get("lfs")) and any(path.endswith(s) for s in spec.get("lfs_suffixes", []))


def url_for(spec, path):
    """LFS-stored files are served from a different host than ordinary blobs.

    The path is percent-encoded because several sources have spaces and parentheses in their
    file names, which urllib rejects outright rather than escaping for you.
    """
    encoded = urllib.parse.quote(path)
    if is_lfs(spec, path):
        return spec["lfs_base"] + encoded
    return spec["raw_base"] + encoded


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, target, expected=None, tries=4):
    """Download with resume. Returns (status, bytes_on_disk)."""
    os.makedirs(os.path.dirname(target), exist_ok=True)

    if os.path.exists(target):
        have = os.path.getsize(target)
        if expected and have == expected:
            return "skipped", have
        if expected is None and have > 0:
            return "present", have

    for attempt in range(tries):
        have = os.path.getsize(target) if os.path.exists(target) else 0
        headers = dict(UA)
        if have:
            headers["Range"] = "bytes={}-".format(have)
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                mode = "ab" if response.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                with open(target, mode) as handle:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
            size = os.path.getsize(target)
            if expected and size != expected:
                if attempt == tries - 1:
                    return "size-mismatch", size
                time.sleep(2 * (attempt + 1))
                continue
            return ("resumed" if have else "downloaded"), size
        except urllib.error.HTTPError as error:
            if error.code == 416 and os.path.exists(target):      # already complete
                return "present", os.path.getsize(target)
            if attempt == tries - 1:
                return "http-{}".format(error.code), (os.path.getsize(target)
                                                      if os.path.exists(target) else 0)
            time.sleep(2 * (attempt + 1))
        except Exception as error:                                # noqa: BLE001
            if attempt == tries - 1:
                return "error:{}".format(type(error).__name__), (os.path.getsize(target)
                                                                 if os.path.exists(target) else 0)
            time.sleep(2 * (attempt + 1))
    return "failed", 0


# -- main --------------------------------------------------------------------------------------

def main():
    config = json.load(open(CONFIG, encoding="utf-8"))
    sources = config["sources"]

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", nargs="+", choices=sorted(sources), default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true", help="discover and report, download nothing")
    args = parser.parse_args()

    chosen = sorted(sources) if (args.all or args.list) else (args.source or [])
    if not chosen:
        parser.error("give --source, --all or --list")

    os.makedirs(REPORTS, exist_ok=True)
    rows = []
    for key in chosen:
        spec = sources[key]
        print("\n=== {}  [{}]  licence: {}".format(key, spec.get("role"), spec.get("licence_status")))
        if spec.get("discovery", {}).get("kind") == "none":
            print("    {}".format(spec.get("status", "no discovery configured")))
            rows.append({
                "source": key, "path": "", "url": spec.get("homepage", ""), "version": "",
                "status": "pending-permission", "bytes": 0, "sha256": "",
                "licence": spec.get("licence", ""), "attribution": spec.get("attribution", ""),
            })
            continue

        files, version = discover(key, spec)
        total = sum(s for _, s in files if s) or 0
        print("    {} files in scope, {:.1f} MB declared, version {}".format(
            len(files), total / 1e6, version or "n/a"))
        if args.list:
            for path, size in files[:200]:
                print("      {:>12}  {}".format("{:,}".format(size) if size else "?", path))
            continue

        for index, (path, size) in enumerate(files, 1):
            target = os.path.join(RAW, key, path.replace("/", os.sep))
            url = url_for(spec, path)
            # The GitHub tree reports the size of the LFS *pointer*, not of the content behind
            # it, so an LFS file must not be size-checked against that number.
            status, got = fetch(url, target, None if is_lfs(spec, path) else size)
            digest = sha256_of(target) if (os.path.exists(target) and got) else ""
            flag = "ok " if status in ("downloaded", "resumed", "skipped", "present") else "!! "
            print("    {}[{:>3}/{}] {:>12,}  {:<10}  {}".format(
                flag, index, len(files), got, status, path))
            rows.append({
                "source": key, "path": path, "url": url, "version": version,
                "status": status, "bytes": got, "sha256": digest,
                "licence": spec.get("licence", ""), "attribution": spec.get("attribution", ""),
            })

    if args.list:
        return 0

    manifest = os.path.join(REPORTS, "DatasetManifest.csv")
    existing = []
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as handle:
            existing = [r for r in csv.DictReader(handle) if r["source"] not in chosen]
    fields = ["source", "path", "url", "version", "status", "bytes", "sha256",
              "licence", "attribution"]
    with open(manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in existing + rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    good = sum(1 for r in rows if r["status"] in ("downloaded", "resumed", "skipped", "present"))
    print("\n{} / {} files in place, {:.2f} GB".format(
        good, len(rows), sum(r["bytes"] for r in rows) / 1e9))
    print("manifest -> {}".format(manifest))
    return 0 if good == len([r for r in rows if r["status"] != "pending-permission"]) else 1


if __name__ == "__main__":
    sys.exit(main())
