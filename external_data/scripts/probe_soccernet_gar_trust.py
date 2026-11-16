"""Is SoccerNet-GAR's per-event `team` field trustworthy enough to train an H/A model on?

The earlier probe found two warning signs: OUT is labelled "away" on 884 of 884 occurrences, and
the share of passes labelled "away" swings from 40.7% to 84.1% across the ten test matches. Those
could mean the field is broken -- or the OUT default is a separate bug and the swing is real
football (a World Cup group stage genuinely does contain 80/20 matches).

This separates the two with checks that need no external ground truth:

  1. Restart agreement. After a THROW IN or FREE KICK by team X, the next on-ball action should
     usually also be X. If `team` were noise this sits at ~50%.
  2. Chain structure. Real possession comes in runs. If `team` were noise, run lengths would be
     geometric with mean ~2 and no long tail.
  3. Self-consistency of the skew. If a match's pass skew is real dominance, its possession-run
     lengths should be longer for the dominant side too -- a second, independent symptom of the
     same cause.
"""

import json
from collections import Counter

import numpy as np
import pandas as pd

SRC = ("D:/Football Hackathon/external_data/raw/soccernet_gar_probe/"
       "sngar_tracking-full_test_metadata.parquet")

# Actions where `team` should mean "the team doing this".
ON_BALL = {"PASS", "HEADER", "CROSS", "SHOT", "HIGH PASS", "GOAL"}
RESTART = {"THROW IN", "FREE KICK"}


def load():
    df = pd.read_parquet(SRC)
    rows = []
    for p in df["sample_payload"]:
        d = json.loads(p)
        m = d.get("metadata", {})
        rows.append({
            "gid": str(m.get("game_id")),
            "t": m.get("position_ms", 0) / 1000.0,
            "act": d["labels"]["action"]["label"],
            "team": m.get("team"),
        })
    return pd.DataFrame(rows).sort_values(["gid", "t"]).reset_index(drop=True)


def runs(seq):
    """Lengths of consecutive equal values."""
    out, cur, n = [], None, 0
    for v in seq:
        if v == cur:
            n += 1
        else:
            if cur is not None:
                out.append(n)
            cur, n = v, 1
    if cur is not None:
        out.append(n)
    return out


def main():
    df = load()

    # --- 1. restart agreement
    agree = Counter()
    for _, g in df.groupby("gid"):
        acts = g["act"].tolist()
        teams = g["team"].tolist()
        for i, a in enumerate(acts):
            if a not in RESTART:
                continue
            for j in range(i + 1, len(acts)):
                if acts[j] in ON_BALL:
                    agree["same" if teams[j] == teams[i] else "diff"] += 1
                    break
    tot = agree["same"] + agree["diff"]
    print("=== 1. after a THROW IN / FREE KICK, is the next on-ball action the same team? ===")
    print("cases            {:,}".format(tot))
    print("same team        {:.1f}%   (coin flip would be 50%)".format(100 * agree["same"] / tot))
    print()

    # --- 2. possession-run structure, on-ball actions only
    onb = df[df["act"].isin(ON_BALL)]
    all_runs = []
    for _, g in onb.groupby("gid"):
        all_runs += runs(g["team"].tolist())
    r = pd.Series(all_runs)
    print("=== 2. possession runs (consecutive on-ball actions by the same team) ===")
    print("runs             {:,}".format(len(r)))
    print("mean length      {:.2f}".format(r.mean()))
    print("median / p90     {:.0f} / {:.0f}".format(r.median(), r.quantile(0.90)))
    print("longest          {}".format(r.max()))
    print("runs of 1        {:.1f}%".format(100 * (r == 1).mean()))
    print("runs of 5+       {:.1f}%".format(100 * (r >= 5).mean()))
    print()

    # A shuffled control: same team proportions, order destroyed.
    rng = np.random.default_rng(0)
    ctrl = []
    for _, g in onb.groupby("gid"):
        v = g["team"].to_numpy().copy()
        rng.shuffle(v)
        ctrl += runs(list(v))
    c = pd.Series(ctrl)
    print("    shuffled control -> mean {:.2f}, runs of 5+ {:.1f}%".format(
        c.mean(), 100 * (c >= 5).mean()))
    print("    (if `team` were noise, the real numbers would match this control)")
    print()

    # --- 3. does the skew travel with run length?
    print("=== 3. per match: pass skew vs how long each side keeps the ball ===")
    print("{:<8}{:>10}{:>14}{:>14}".format("match", "away %", "away run", "home run"))
    for gid, g in onb.groupby("gid"):
        t = g["team"].tolist()
        rr = runs(t)
        # rebuild which team each run belongs to
        owners, cur = [], None
        for v in t:
            if v != cur:
                owners.append(v)
                cur = v
        aw = [n for n, o in zip(rr, owners) if o == "away"]
        hm = [n for n, o in zip(rr, owners) if o == "home"]
        away_pct = 100 * (pd.Series(t) == "away").mean()
        print("{:<8}{:>9.1f}%{:>14.2f}{:>14.2f}".format(
            gid, away_pct, np.mean(aw) if aw else 0, np.mean(hm) if hm else 0))


if __name__ == "__main__":
    main()
