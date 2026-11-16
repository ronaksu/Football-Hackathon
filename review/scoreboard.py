"""One table: how accurate the model is now, and what is still wrong.

Reads the APPLIED refined predictions (stage3/predictions_refined) and the review tool's own
case index, so the numbers here and the cases on the page can never drift apart. Scoring is the
official evaluation/possession_metrics.py.

    python -I review/scoreboard.py          # print the table and write review/data/scoreboard.json

Development matches only; the two sealed test matches are never read.
"""

import glob
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage3"))
import restart_logic as R  # noqa: E402

SNAPDIR = os.path.join(HERE, "snapshots")
SEALED = {"J03WMX", "J03WN1"}
SPELL_BAND = 0.04       # the published noise band; smaller moves are not improvements
MACRO_BAND = 0.01
FAIL_ORDER = ["MISSED_CHANGE", "WRONG_TEAM", "FALSE_CHANGE", "BOUNDARY",
              "INVENTED_STOP", "MISSED_STOP", "SPELL_MERGE"]
WHAT = {
    "MISSED_CHANGE": "the ball changed hands and the model stayed with one team",
    "WRONG_TEAM": "the model named the wrong team",
    "FALSE_CHANGE": "the model switched team when nobody took the ball",
    "BOUNDARY": "the stoppage is found, but an edge is more than a second out",
    "INVENTED_STOP": "the model stopped play when play was running",
    "MISSED_STOP": "play stopped and the model never noticed",
    "SPELL_MERGE": "two possessions run together as one",
}


def stoppage_edges(truth, pred, periods):
    """Every stoppage's |error| on the whistle and on the restart, and how many were found at all.

    The errors come back as lists so they can be pooled over the five matches: a median taken over
    all 489 stoppages is a different number from the mean of five per-match medians, and the first
    is the one worth quoting.
    """
    whistle, restart, found, total = [], [], 0, 0
    for p in sorted(pd.unique(periods)):
        m = periods == p
        t, q = np.asarray(truth[m], dtype=object), np.asarray(pred[m], dtype=object)
        spans = list(R.runs_of(q == "D"))
        for a, b in R.runs_of(t == "D"):
            total += 1
            best = max(((min(b, d) - max(a, c), c, d) for c, d in spans if min(b, d) > max(a, c)),
                       default=None)
            if best:
                found += 1
                whistle.append(abs(best[1] - a) / R.FPS)
                restart.append(abs(best[2] - b) / R.FPS)
    return whistle, restart, found, total


def main():
    pm = R.load_scorer()
    files = sorted(glob.glob(os.path.join(HERE, "..", "stage3", "predictions_refined",
                                          "*.parquet")))
    bad = [f for f in files if any(s in f for s in SEALED)]
    if bad:
        raise SystemExit("REFUSING: sealed match in {}".format(bad))

    # Everything is measured twice, on the same frames: the model as it ships (label_refined) and
    # the model before any of the hardening work (label_base). One loop, so the two columns of the
    # table can never be computed differently.
    WHICH = ("now", "before")
    acc = {w: {"spell": [], "macro": [], "whis": [], "rest": [], "found": 0, "stops": 0,
               "live_right": 0, "live_total": 0, "false_switches": 0, "missed_switches": 0,
               "restart_team_right": 0, "restart_team_total": 0} for w in WHICH}
    rows = []
    for f in files:
        d = pd.read_parquet(f)
        truth = d["label"].to_numpy()
        per = d["period"].to_numpy()
        for w, col in zip(WHICH, ("label_refined", "label_base")):
            pred = d[col].to_numpy()
            a = acc[w]
            s, mf = R.spell_scores(pm, truth, pred, per)
            a["spell"].append(s)
            a["macro"].append(mf)
            live = np.isin(truth, ["H", "A"])
            a["live_right"] += int((pred[live] == truth[live]).sum())
            a["live_total"] += int(live.sum())
            r = R.restart_report(truth, pred, per)
            for k in ("false_switches", "missed_switches", "restart_team_right",
                      "restart_team_total"):
                a[k] += r[k]
            wh, rr, fnd, tot = stoppage_edges(truth, pred, per)
            a["whis"] += wh
            a["rest"] += rr
            a["found"] += fnd
            a["stops"] += tot
            if w == "now":
                rows.append({"match": os.path.basename(f).replace("DFL-MAT-", "")
                                                         .replace(".parquet", ""),
                             "spellF1": s, "macroF1": mf})
        print("  scored " + rows[-1]["match"], flush=True)

    with open(os.path.join(HERE, "data", "index.json"), encoding="utf-8") as fh:
        idx = json.load(fh)
    cases = pd.DataFrame(idx["cases"])
    fails = cases[cases["outcome"] == "failure"]
    # The "before" count for each kind comes from the snapshot taken before the hardening work,
    # not from a second pass over the cases, because the case definitions have not changed since.
    with open(os.path.join(SNAPDIR, "baseline.json"), encoding="utf-8") as fh:
        base = json.load(fh)
    was = {k: sum(v.values()) for k, v in base["failures"]["by_kind_bucket"].items()}
    # A failure the model could not see the ball through cannot be taught: the reviewer is
    # looking at the same empty pitch the model was. Counted separately so the worklist is the
    # cases where the evidence was actually there.
    vis = fails[fails["ball_visible"].fillna(False)] if "ball_visible" in fails else fails.iloc[:0]
    by_kind = {k: {"total": int(len(g)),
                   "visible": int(g["ball_visible"].fillna(False).sum())
                   if "ball_visible" in g else 0,
                   "before": int(was.get(k, 0)),
                   "confirmed": int((g["bucket"] == "confirmed").sum()),
                   "contested": int((g["bucket"] == "contested").sum()),
                   "new": int(g["new_since_fix"].fillna(False).sum()),
                   "with_bundle": int(g["bundled"].fillna(False).sum())}
               for k, g in fails.groupby("kind")}

    def block(w):
        a = acc[w]
        return {"spellF1": float(np.mean(a["spell"])),
                "macroF1": float(np.mean(a["macro"])),
                "team_acc_live": a["live_right"] / max(1, a["live_total"]),
                "live_frames": a["live_total"],
                "stoppages_found": a["found"],
                "stoppages_total": a["stops"],
                "whistle_err_s": float(np.median(a["whis"])),
                "restart_err_s": float(np.median(a["rest"])),
                "whistle_err_mean_s": float(np.mean(a["whis"])),
                "restart_err_mean_s": float(np.mean(a["rest"])),
                "edges_timed": len(a["whis"]),
                "restart_team_right": a["restart_team_right"],
                "restart_team_total": a["restart_team_total"],
                "false_switches": a["false_switches"],
                "missed_switches": a["missed_switches"]}

    out = {
        "taken": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "open-access IDSSE development matches only; the sealed test matches untouched",
        "matches": idx["matches"],
        "bands": {"spellF1": SPELL_BAND, "macroF1": MACRO_BAND},
        "accuracy": block("now"),
        "accuracy_before": block("before"),
        "before_note": base.get("note", ""),
        "cases": {
            "failures": int(len(fails)),
            "failures_before": int(base["failures"]["total"]),
            "correct": int((cases["outcome"] == "correct").sum()),
            "new_since_fix": int(cases["new_since_fix"].fillna(False).sum()),
            "with_bundle": int(fails["bundled"].fillna(False).sum()),
            "ball_visible": {
                "total": int(len(vis)),
                "confirmed": int((vis["bucket"] == "confirmed").sum()),
                "contested": int((vis["bucket"] == "contested").sum()),
                "with_bundle": int(vis["bundled"].fillna(False).sum()),
                "hidden": int(len(fails) - len(vis)),
                "half_blind": int((fails["ball_seen"].fillna(0) < 0.5).sum())
                if "ball_seen" in fails else 0,
                "never_seen": int((fails["ball_seen"].fillna(0) <= 0.001).sum())
                if "ball_seen" in fails else 0,
                "threshold": 0.8,
            },
            "by_kind": by_kind,
            "what": WHAT,
        },
        "per_match": rows,
    }
    with open(os.path.join(HERE, "data", "scoreboard.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    a, b = out["accuracy"], out["accuracy_before"]
    print("\n=== how accurate it is, now against before the hardening work ===")
    line = "  {:<34}{:>12}{:>12}"
    print(line.format("measure", "now", "before"))
    print(line.format("spellF1 (decides the ranking)",
                      "{:.3f}".format(a["spellF1"]), "{:.3f}".format(b["spellF1"])))
    print(line.format("macroF1",
                      "{:.3f}".format(a["macroF1"]), "{:.3f}".format(b["macroF1"])))
    print(line.format("right team on live frames",
                      "{:.1%}".format(a["team_acc_live"]), "{:.1%}".format(b["team_acc_live"])))
    print(line.format("stoppages found",
                      "{} / {}".format(a["stoppages_found"], a["stoppages_total"]),
                      "{} / {}".format(b["stoppages_found"], b["stoppages_total"])))
    print(line.format("whistle timing, median |err|",
                      "{:.2f} s".format(a["whistle_err_s"]), "{:.2f} s".format(b["whistle_err_s"])))
    print(line.format("whistle timing, mean |err|",
                      "{:.2f} s".format(a["whistle_err_mean_s"]),
                      "{:.2f} s".format(b["whistle_err_mean_s"])))
    print(line.format("restart timing, median |err|",
                      "{:.2f} s".format(a["restart_err_s"]), "{:.2f} s".format(b["restart_err_s"])))
    print(line.format("restart timing, mean |err|",
                      "{:.2f} s".format(a["restart_err_mean_s"]),
                      "{:.2f} s".format(b["restart_err_mean_s"])))
    print(line.format("team right after a restart",
                      "{} / {}".format(a["restart_team_right"], a["restart_team_total"]),
                      "{} / {}".format(b["restart_team_right"], b["restart_team_total"])))
    print(line.format("false switches", a["false_switches"], b["false_switches"]))
    print(line.format("missed switches", a["missed_switches"], b["missed_switches"]))
    print("\n=== what is still wrong ({} cases, was {}, {} new) ===".format(
        out["cases"]["failures"], out["cases"]["failures_before"],
        out["cases"]["new_since_fix"]))
    print("  {:<16}{:>7}{:>8}{:>11}{:>11}{:>6}{:>8}{:>12}".format(
        "kind", "now", "before", "confirmed", "contested", "new", "bundle", "ball seen"))
    for k in FAIL_ORDER:
        v = by_kind.get(k)
        if v:
            print("  {:<16}{:>7}{:>8}{:>11}{:>11}{:>6}{:>8}{:>12}".format(
                k, v["total"], v["before"], v["confirmed"], v["contested"], v["new"],
                v["with_bundle"], v["visible"]))
    print("  {:<16}{:>7}".format("correct", out["cases"]["correct"]))

    b = out["cases"]["ball_visible"]
    tot = max(1, out["cases"]["failures"])
    print("\n=== of those failures, the ones that can be taught ===")
    print("  {:<40}{:>7,}{:>7.0%}".format(
        "ball in sight 80%+ of the window", b["total"], b["total"] / tot))
    print("  {:<40}{:>7,}{:>7.0%}".format(
        "  ... and the reference corroborated", b["confirmed"], b["confirmed"] / tot))
    print("  {:<40}{:>7,}".format("  ... and openable on the review page", b["with_bundle"]))
    print("  {:<40}{:>7,}{:>7.0%}".format(
        "ball hidden more than half the window", b["half_blind"], b["half_blind"] / tot))
    print("  {:<40}{:>7,}{:>7.0%}".format(
        "ball never seen at all", b["never_seen"], b["never_seen"] / tot))


if __name__ == "__main__":
    main()
