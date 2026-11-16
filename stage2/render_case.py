"""Draw a failure case as a strip of top-down pitch views, plus the probability trace.

There is no video to watch. The DFL withholds match footage from the open dataset for licensing
reasons, so the only way to look at one of these moments is to draw the tracking -- which is also
exactly what the model sees. If a cue is not visible here, no broadcast footage would have helped
the model find it.

Each figure shows six snapshots across the window, the ground truth band, the predicted band, and
the raw per-frame probability underneath.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

PITCH_X, PITCH_Y = 105.0, 68.0
FPS = 25
PROC = "D:/Football Hackathon/external_data/processed/idsse"
PRED = "D:/Football Hackathon/stage2/predictions"

HOME, AWAY = "#004D98", "#EE2523"
DEAD, ALIVE = "#5B6878", "#0E7C66"


def draw_pitch(ax):
    ax.add_patch(Rectangle((0, 0), PITCH_X, PITCH_Y, fill=False, lw=1.0, ec="#9AA7B6"))
    ax.plot([PITCH_X / 2, PITCH_X / 2], [0, PITCH_Y], lw=0.8, color="#9AA7B6")
    for x0 in (0, PITCH_X - 16.5):
        ax.add_patch(Rectangle((x0, PITCH_Y / 2 - 20.16), 16.5, 40.32,
                               fill=False, lw=0.8, ec="#9AA7B6"))
    c = plt.Circle((PITCH_X / 2, PITCH_Y / 2), 9.15, fill=False, lw=0.8, ec="#9AA7B6")
    ax.add_artist(c)
    ax.set_xlim(-6, PITCH_X + 6)
    ax.set_ylim(-6, PITCH_Y + 6)
    ax.set_aspect("equal")
    ax.axis("off")


def render(match, period, f0, f1, out_path, title="", pad_s=3.0, decoder="hsmm",
           n_panels=6, fig_h=6.4):
    players = pd.read_parquet(os.path.join(PROC, match, "players.parquet"))
    players = players[(players["period"] == period)]

    pred = pd.read_parquet(os.path.join(PRED, match + ".parquet"))
    pred = pred[pred["period"] == period].sort_values("frame_native").reset_index(drop=True)

    pad = int(pad_s * FPS)
    lo, hi = f0 - pad, f1 + pad
    w = pred[(pred["frame_native"] >= lo) & (pred["frame_native"] <= hi)].reset_index(drop=True)
    if w.empty:
        print("  no prediction rows in window for {} {}-{}".format(match, f0, f1))
        return None

    picks = np.linspace(0, len(w) - 1, n_panels).astype(int)

    fig = plt.figure(figsize=(15.5, fig_h))
    gs = fig.add_gridspec(3, n_panels, height_ratios=[3.2, 0.5, 1.0], hspace=0.35, wspace=0.08)

    for k, idx in enumerate(picks):
        fr = int(w["frame_native"].iloc[idx])
        ax = fig.add_subplot(gs[0, k])
        draw_pitch(ax)
        p = players[players["frame_native"] == fr]
        for team, colour in (("home", HOME), ("away", AWAY)):
            q = p[p["team"] == team]
            ax.scatter(q["x"] * PITCH_X, q["y"] * PITCH_Y, s=34, c=colour,
                       edgecolors="white", linewidths=0.6, zorder=3)
        off = p[(p["y"] < 0) | (p["y"] > 1) | (p["x"] < 0) | (p["x"] > 1)]
        if len(off):
            ax.scatter(off["x"] * PITCH_X, off["y"] * PITCH_Y, s=150, facecolors="none",
                       edgecolors="#D98A14", linewidths=1.8, zorder=4)

        truth = w["label"].iloc[idx]
        pr = "D" if bool(w["pred_" + decoder].iloc[idx]) else "alive"
        ok = (truth == "D") == (pr == "D")
        ax.set_title("{:+.1f}s\ntruth {}  |  model {}".format(
            (fr - f0) / FPS, truth, pr), fontsize=9,
            color="#11161F" if ok else "#B01B1B", fontweight="normal" if ok else "bold")

    # --- truth and prediction bands
    axb = fig.add_subplot(gs[1, :])
    t = (w["frame_native"].to_numpy() - f0) / FPS
    truth_d = (w["label"].to_numpy() == "D").astype(float)
    pred_d = w["pred_" + decoder].to_numpy().astype(float)
    axb.fill_between(t, 1.05, 1.95, where=truth_d > 0, color=DEAD, step="mid")
    axb.fill_between(t, 1.05, 1.95, where=truth_d == 0, color=ALIVE, alpha=0.35, step="mid")
    axb.fill_between(t, 0.05, 0.95, where=pred_d > 0, color=DEAD, step="mid")
    axb.fill_between(t, 0.05, 0.95, where=pred_d == 0, color=ALIVE, alpha=0.35, step="mid")
    axb.set_ylim(0, 2)
    axb.set_yticks([0.5, 1.5])
    axb.set_yticklabels(["model", "truth"], fontsize=9)
    axb.set_xlim(t[0], t[-1])
    axb.set_xticks([])
    for s in axb.spines.values():
        s.set_visible(False)

    # --- probability trace
    axp = fig.add_subplot(gs[2, :])
    axp.plot(t, w["prob"].to_numpy(), lw=1.3, color="#1B2A4A")
    axp.axhline(0.5, lw=0.8, ls="--", color="#9AA7B6")
    axp.set_ylim(0, 1)
    axp.set_xlim(t[0], t[-1])
    axp.set_ylabel("P(dead)", fontsize=9)
    axp.set_xlabel("seconds from the start of the true stoppage", fontsize=9)
    axp.tick_params(labelsize=8)
    for s in ("top", "right"):
        axp.spines[s].set_visible(False)

    fig.suptitle(title or "{}  period {}  frames {}-{}".format(match, period, f0, f1),
                 fontsize=12, fontweight="bold", y=0.99)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="D:/Football Hackathon/stage2/failures_hsmm.csv")
    ap.add_argument("--out-dir", default="D:/Football Hackathon/stage2/cases")
    ap.add_argument("--types", nargs="*", default=["SPELL_MERGE", "SPELL_SPLIT"])
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--decoder", default="hsmm")
    args = ap.parse_args()

    cases = pd.read_csv(args.cases)
    cases = cases[cases["type"].isin(args.types)].head(args.limit)
    print("rendering {} cases".format(len(cases)))

    for i, r in enumerate(cases.itertuples(), 1):
        name = "{:02d}_{}_{}_{}_{}.png".format(
            i, r.type, r.match, r.period, str(r.clock).replace(":", "m"))
        title = "{}  {}  period {}  {}   |   {}  lasting {}s   |   {} -> D -> {}".format(
            r.match, r.clock, r.period, "", r.type, r.duration_s,
            r.team_before, r.team_after)
        p = render("DFL-MAT-" + r.match, int(r.period), int(r.frame_start), int(r.frame_end),
                   os.path.join(args.out_dir, name), title=title, decoder=args.decoder)
        if p:
            print("  " + os.path.basename(p))


if __name__ == "__main__":
    main()
