"""Render the guided demo video for the Model Failure Review page.

Two cases, told one after the other, from the same match and the same half:

    J03WOH P1 35:56   a 9.2 s stoppage the model catches to within a fifth of a second
    J03WOH P1 06:46   a 7.4 s stoppage, almost the same length, that it misses completely

The point is not the animation. It is that the two look alike to a human and do not look alike in
the data, and the video shows which numbers differ. Every scene carries the same read-out the page
shows -- model, reference, P(dead), P(home), the features, the ball's visibility, the timing
offsets, and the model's own reasoning in nats -- so watching the video teaches the page.

The comparison table, the rationale line and every number in the read-out are computed from the
exported case bundles, which review/export_cases.py builds from the open-access IDSSE development
matches. Only the narration itself is written by hand.

    python review/make_demo.py
    -> review/demo/model_demo.mp4  plus stills of the key moments
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 25
HERE = os.path.dirname(os.path.abspath(__file__))

INK = (17, 22, 31)
MUTED = (91, 104, 120)
LINE = (215, 222, 232)
TINT = (241, 244, 248)
NAVY = (27, 42, 74)
HOME = (0, 77, 152)
AWAY = (238, 37, 35)
DEAD = (135, 149, 166)
OK = (14, 124, 102)
BAD = (176, 27, 27)
WARN = (217, 138, 20)
WHITE = (255, 255, 255)
COL = {"H": HOME, "A": AWAY, "D": DEAD}

FONTS = "C:/Windows/Fonts"


def font(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), size)


F = {
    "h1": font("cambriab.ttf", 42), "h2": font("cambriab.ttf", 30),
    "h3": font("calibrib.ttf", 22), "b": font("calibrib.ttf", 19),
    "r": font("calibri.ttf", 19), "sm": font("calibri.ttf", 16),
    "smb": font("calibrib.ttf", 16), "tiny": font("calibri.ttf", 13),
    "mono": font("consola.ttf", 16), "monob": font("consola.ttf", 18),
    "cap": font("calibri.ttf", 22), "capb": font("calibrib.ttf", 22),
    "chip": font("calibrib.ttf", 21),
}

# ------------------------------------------------------------------ layout
PITCH = (24, 72, 608)                       # x, y, width -> height follows 105:68
PITCH_H = PITCH[2] * 68.0 / 105.0           # 393.8
CHIPS_Y = int(PITCH[1] + PITCH_H + 12)      # 477
STRIP_Y = CHIPS_Y + 50                      # 527
RIBBON_Y = 590
CAP_Y = 612
PANEL_X, PANEL_W = 680, 576                 # read-out column


# --------------------------------------------------------------------- text


def runs(text):
    """Split '**bold** and plain' into (chunk, is_bold) pairs."""
    out, bold = [], False
    for part in text.split("**"):
        if part:
            out.append((part, bold))
        bold = not bold
    return out


def tokens(text):
    """(word, bold, glue) -- glue means 'no space before me', so '**100%**.' keeps its full stop."""
    out, first_of_chunk = [], False
    tail = ""
    for chunk, bold in runs(text):
        first_of_chunk = True
        for w in chunk.split(" "):
            if not w:
                first_of_chunk = False
                continue
            glue = first_of_chunk and bool(out) and not tail.endswith(" ")
            out.append((w, bold, glue))
            first_of_chunk = False
        tail = chunk
    return out


def wrap(dr, text, fr, fb, width):
    lines, cur, cw = [], [], 0.0
    space = dr.textlength(" ", font=fr)
    for w, bold, glue in tokens(text):
        ww = dr.textlength(w, font=fb if bold else fr)
        gap = 0.0 if (glue and cur) else (space if cur else 0.0)
        if cur and cw + gap + ww > width:
            lines.append(cur)
            cur, cw = [(w, bold, False)], ww
        else:
            cw += gap + ww
            cur.append((w, bold, glue))
    if cur:
        lines.append(cur)
    return lines


def draw_rich(dr, text, x, y, width, fr, fb, fill=INK, leading=29):
    lines = wrap(dr, text, fr, fb, width)
    space = dr.textlength(" ", font=fr)
    for k, line in enumerate(lines):
        cx = x
        for j, (w, bold, glue) in enumerate(line):
            f = fb if bold else fr
            if j and not glue:
                cx += space
            dr.text((cx, y + k * leading), w, font=f, fill=fill)
            cx += dr.textlength(w, font=f)
    return y + len(lines) * leading


# -------------------------------------------------------------------- pitch


def draw_pitch(dr, box):
    x, y, w, h = box
    dr.rectangle([x, y, x + w, y + h], fill=(244, 248, 251), outline=(159, 179, 200), width=2)
    dr.line([x + w / 2, y, x + w / 2, y + h], fill=(159, 179, 200), width=2)
    r = h * 0.137
    dr.ellipse([x + w / 2 - r, y + h / 2 - r, x + w / 2 + r, y + h / 2 + r],
               outline=(159, 179, 200), width=2)
    for side in (0, 1):
        pw, ph = w * 16.5 / 105, h * 40.3 / 68
        sw, sh = w * 5.5 / 105, h * 18.3 / 68
        x0 = x if side == 0 else x + w - pw
        dr.rectangle([x0, y + (h - ph) / 2, x0 + pw, y + (h + ph) / 2],
                     outline=(159, 179, 200), width=2)
        x1 = x if side == 0 else x + w - sw
        dr.rectangle([x1, y + (h - sh) / 2, x1 + sw, y + (h + sh) / 2],
                     outline=(159, 179, 200), width=2)


def draw_players(dr, d, i, box):
    x, y, w, h = box
    X, Y = d["x"][i], d["y"][i]
    for k in range(len(X)):
        if X[k] < 0:
            continue
        px, py = x + X[k] / 1000.0 * w, y + Y[k] / 1000.0 * h
        dr.ellipse([px - 8, py - 8, px + 8, py + 8],
                   fill=HOME if d["is_home"][k] else AWAY, outline=WHITE, width=2)
    if d["bx"][i] is not None:
        px, py = x + d["bx"][i] * w, y + d["by"][i] * h
        hidden = d["ball_seen_model"][i] == "0"
        dr.ellipse([px - 7, py - 7, px + 7, py + 7], fill=WHITE,
                   outline=(WARN if hidden else NAVY), width=4 if hidden else 3)
        if hidden:
            for a in range(0, 360, 30):
                a0, a1 = np.radians(a), np.radians(a + 15)
                dr.line([px + 14 * np.cos(a0), py + 14 * np.sin(a0),
                         px + 14 * np.cos(a1), py + 14 * np.sin(a1)], fill=WARN, width=3)


# ------------------------------------------------------------------ widgets


def meter(dr, x, y, w, value, label, good=None):
    dr.text((x, y), label, font=F["sm"], fill=MUTED)
    yy = y + 20
    dr.rounded_rectangle([x, yy, x + w, yy + 13], 4, fill=(233, 238, 244))
    if value is None:
        dr.text((x + w + 10, y + 16), "  --", font=F["mono"], fill=MUTED)
        return
    dr.rounded_rectangle([x, yy, x + max(4, w * float(value)), yy + 13], 4,
                         fill=OK if (good is None or good) else BAD)
    dr.line([x + w * 0.5, yy - 3, x + w * 0.5, yy + 16], fill=INK, width=2)
    dr.text((x + w + 10, y + 16), "{:.3f}".format(value), font=F["mono"], fill=INK)


def chip(dr, x, y, letter, label):
    dr.rounded_rectangle([x, y, x + 38, y + 36], 5, fill=COL.get(letter, DEAD))
    dr.text((x + 19, y + 18), letter, font=F["chip"], fill=WHITE, anchor="mm")
    dr.text((x + 46, y + 18), label, font=F["sm"], fill=MUTED, anchor="lm")


def strips(dr, img, d, i, x, y, w):
    n = len(d["f"])
    sw = w / float(n)
    for r, (label, seq) in enumerate((("reference", d["ref"]), ("model", d["pred"]))):
        yy = y + r * 18
        for k in range(n):
            dr.rectangle([x + k * sw, yy, x + (k + 1) * sw + 1, yy + 15], fill=COL.get(seq[k], DEAD))
        dr.text((x + 5, yy + 7), label, font=F["tiny"], fill=WHITE, anchor="lm")
    yy = y + 36
    for k in range(n):
        to_model = d["ball_seen_model"][k] == "1"
        in_file = d["ball_seen"][k] == "1"
        dr.rectangle([x + k * sw, yy, x + (k + 1) * sw + 1, yy + 15],
                     fill=OK if to_model else (WARN if in_file else (200, 210, 221)))
    dr.text((x + 5, yy + 7), "ball to model", font=F["tiny"], fill=WHITE, anchor="lm")

    shade = OK if d["outcome"] == "correct" else BAD
    a, b = d["in_case"].find("1"), d["in_case"].rfind("1") + 1
    ov = Image.new("RGBA", (max(1, int((b - a) * sw)), 51), shade + (72,))
    img.paste(ov, (int(x + a * sw), int(y)), ov)
    dr.line([x + i * sw + sw / 2, y - 3, x + i * sw + sw / 2, y + 54], fill=INK, width=3)


def spans_of(seq, ch):
    out, k, n = [], 0, len(seq)
    while k < n:
        if seq[k] == ch:
            j = k
            while j < n and seq[j] == ch:
                j += 1
            out.append((k, j))
            k = j
        else:
            k += 1
    return out


def timing_bars(dr, d, x, y, w):
    """Reference span against model span on one axis, with the offsets written out."""
    n = len(d["f"])
    ref = max(spans_of(d["ref"], "D"), key=lambda s: s[1] - s[0], default=None)
    mod = [s for s in spans_of(d["pred"], "D") if ref and min(s[1], ref[1]) > max(s[0], ref[0])]
    mod = max(mod, key=lambda s: s[1] - s[0], default=None)

    dr.text((x, y), "TIMING   ACTUAL vs MODEL", font=F["smb"], fill=MUTED)
    for r, (label, span, c) in enumerate((("reference", ref, DEAD),
                                          ("model", mod, OK if mod else BAD))):
        yy = y + 24 + r * 30
        dr.text((x, yy), label, font=F["tiny"], fill=MUTED)
        dr.rounded_rectangle([x + 72, yy - 2, x + w, yy + 13], 4, fill=(233, 238, 244))
        if span:
            x0 = x + 72 + span[0] * (w - 72) / n
            x1 = x + 72 + span[1] * (w - 72) / n
            dr.rounded_rectangle([x0, yy - 2, max(x0 + 4, x1), yy + 13], 4, fill=c)
        else:
            dr.text((x + 82, yy - 2), "no stoppage called in this whole window",
                    font=F["tiny"], fill=BAD)
    yy = y + 88
    ex = d.get("extra", {})
    if mod:
        step = d["stride"] / float(d["fps"])
        for label, v in (("whistle", ex.get("start_err_s", (mod[0] - ref[0]) * step)),
                         ("restart", ex.get("end_err_s", (mod[1] - ref[1]) * step))):
            word = ("on time" if abs(v) < 1e-9
                    else "{:.2f} s {}".format(abs(v), "late" if v > 0 else "early"))
            dr.text((x, yy), label, font=F["sm"], fill=MUTED)
            dr.text((x + w, yy), word, font=F["smb"], fill=OK if abs(v) <= 0.25 else WARN,
                    anchor="ra")
            yy += 21
        if ex.get("covered") is not None:
            dr.text((x, yy), "overlap with the reference", font=F["sm"], fill=MUTED)
            dr.text((x + w, yy), "{:.0%}".format(ex["covered"]), font=F["smb"], fill=OK, anchor="ra")
    else:
        dr.text((x, yy), "overlap with the reference", font=F["sm"], fill=MUTED)
        dr.text((x + w, yy), "0%", font=F["smb"], fill=BAD, anchor="ra")
        yy += 21
        dr.text((x, yy), "dead time it played straight through", font=F["sm"], fill=MUTED)
        dr.text((x + w, yy), "{:.2f} s".format(d["duration_s"]), font=F["smb"], fill=BAD,
                anchor="ra")


def ribbon_text(d):
    """The model's own reasoning, in one line, straight out of the bundle."""
    r = d.get("rationale")
    if not r:
        return "MODEL'S REASONING    not recorded for this case", MUTED
    if r["stage"] == 2:
        col = OK if r["margin_nats"] > 0 else BAD
        return ("MODEL'S REASONING   evidence for a stoppage {:+.0f} nats   ·   duration prior "
                "charges -{:.0f}   ·   net {:+.0f}   ->   it {}".format(
                    r["emission_nats"], r["duration_cost_nats"], r["margin_nats"],
                    r["decision"]), col)
    withm = [c["cue"] for c in r["cues"] if c["with_model"] and c["strength"] != "a coin toss"]
    return ("MODEL'S REASONING   P(home) {:.3f}   ·   it {}   ·   following {}".format(
        r["p_home"] or 0.0, r["decision"], ", ".join(withm) or "no single clear cue"), NAVY)


# -------------------------------------------------------------------- scene


FEATS = [("spd_mean", "mean player speed", "m/s"),
         ("frac_still", "share standing still", ""),
         ("n_outside_pitch", "players off the pitch", ""),
         ("ball_margin", "home vs away to ball", "m")]


def scene_frame(d, i, chapter, caption, accent):
    img = Image.new("RGB", (W, H), WHITE)
    dr = ImageDraw.Draw(img)

    dr.rectangle([0, 0, W, 48], fill=NAVY)
    dr.text((24, 24), chapter, font=F["h3"], fill=WHITE, anchor="lm")
    dr.text((W - 24, 24), "{}  P{}  {}   ·   {}".format(d["match"], d["period"], d["clock"],
                                                        d["kind"]),
            font=F["sm"], fill=(185, 198, 219), anchor="rm")
    zero = d["t"][d["in_case"].find("1")]
    dr.text((460, 24), "t = {:+.2f} s   frame {}".format(d["t"][i] - zero, d["f"][i]),
            font=F["mono"], fill=(150, 168, 196), anchor="lm")

    box = (PITCH[0], PITCH[1], PITCH[2], PITCH_H)
    draw_pitch(dr, box)
    draw_players(dr, d, i, box)

    inc = d["in_case"][i] == "1"
    chip(dr, PITCH[0], CHIPS_Y, d["pred"][i], "model")
    chip(dr, PITCH[0] + 176, CHIPS_Y, d["ref"][i], "reference (actual)")
    tag = ("MODEL AGREES HERE" if accent == OK else "INSIDE THE FAILURE") if inc else "context"
    dr.text((PITCH[0] + PITCH[2], CHIPS_Y + 18), tag, font=F["smb"],
            fill=accent if inc else MUTED, anchor="rm")

    strips(dr, img, d, i, PITCH[0], STRIP_Y, PITCH[2])

    # ---- read-out column
    dr.rounded_rectangle([PANEL_X - 14, 62, W - 12, 582], 8, fill=(250, 251, 253), outline=LINE)
    px, pw = PANEL_X, PANEL_W - 40
    y = 76
    pd = d["p_dead"][i]
    meter(dr, px, y, 300, pd, "P(ball is dead)  ·  stage 2, players only",
          good=(pd is not None and (pd >= 0.5) == (d["ref"][i] == "D")))
    y += 56
    ph = d["p_home"][i]
    meter(dr, px, y, 300, ph, "P(home has it)  ·  stage 3, ball + pitch control",
          good=(ph is None or d["ref"][i] not in "HA"
                or ("H" if ph >= 0.5 else "A") == d["ref"][i]))
    y += 62

    dr.text((px, y), "WHERE IS THE BALL", font=F["smb"], fill=MUTED)
    y += 22
    for label, ok in (("in the tracking file", d["ball_seen"][i] == "1"),
                      ("as the model sees it", d["ball_seen_model"][i] == "1")):
        dr.ellipse([px, y + 4, px + 11, y + 15], fill=OK if ok else WARN)
        dr.text((px + 20, y), label, font=F["sm"], fill=MUTED)
        dr.text((px + pw, y), ("tracked" if "file" in label else "visible") if ok
                else ("not tracked" if "file" in label else "hidden"),
                font=F["smb"], fill=OK if ok else WARN, anchor="ra")
        y += 22
    dr.text((px, y), "second opinion (event feed)", font=F["sm"], fill=MUTED)
    dr.text((px + pw, y), {"H": "home", "A": "away", "-": "no synced event"}[d["holder"][i]],
            font=F["smb"], fill=INK, anchor="ra")
    y += 32

    dr.text((px, y), "WHAT THE MODEL IS READING", font=F["smb"], fill=MUTED)
    y += 22
    for key, label, unit in FEATS:
        arr = d["feat"].get(key)
        if arr is None:
            continue
        v = arr[i]
        dr.text((px, y), label, font=F["sm"], fill=MUTED)
        dr.text((px + pw, y), "--" if v is None else
                ("{:.2f} {}".format(v, unit) if unit else "{:.2f}".format(v)),
                font=F["mono"], fill=INK, anchor="ra")
        y += 22
    timing_bars(dr, d, px, y + 10, pw)

    # ---- the model's reasoning, one line, full width
    text, col = ribbon_text(d)
    dr.rounded_rectangle([24, RIBBON_Y, W - 24, RIBBON_Y + 20], 4, fill=(239, 243, 249))
    dr.rectangle([24, RIBBON_Y, 30, RIBBON_Y + 20], fill=col)
    dr.text((40, RIBBON_Y + 10), text, font=F["mono"], fill=col, anchor="lm")

    # ---- caption
    dr.rectangle([0, CAP_Y, W, H], fill=TINT)
    dr.rectangle([0, CAP_Y, 7, H], fill=accent)
    draw_rich(dr, caption, 26, CAP_Y + 14, W - 60, F["cap"], F["capb"], leading=30)
    return img


# --------------------------------------------------------------------- cards


def card(title, kicker, bullets, accent=NAVY, extra=None):
    img = Image.new("RGB", (W, H), WHITE)
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, W, 10], fill=accent)
    dr.text((70, 62), kicker, font=F["smb"], fill=accent)
    dr.text((70, 86), title, font=F["h1"], fill=NAVY)
    y = 168
    for b in bullets:
        if b == "":
            y += 14
            continue
        dr.ellipse([70, y + 9, 80, y + 19], fill=accent)
        y = draw_rich(dr, b, 98, y, W - 180, F["cap"], F["capb"], leading=30) + 14
    if extra:
        extra(dr, img, y)
    return img


def card_iou(dr, img, y):
    x = 140
    y = max(y + 6, 400)
    dr.text((x, y), "one possession spell, measured", font=F["smb"], fill=MUTED)
    ry = y + 32
    dr.rounded_rectangle([x, ry, x + 520, ry + 26], 5, fill=DEAD)
    dr.text((x + 532, ry + 13), "the real spell", font=F["sm"], fill=MUTED, anchor="lm")
    py = ry + 38
    dr.rounded_rectangle([x + 90, py, x + 600, py + 26], 5, fill=HOME)
    dr.text((x + 612, py + 13), "the model's spell", font=F["sm"], fill=MUTED, anchor="lm")
    dr.rectangle([x + 90, ry, x + 520, py + 26], outline=OK, width=3)
    dr.text((x + 305, py + 54), "overlap  /  union   =   IoU", font=F["b"], fill=OK, anchor="mm")
    dr.text((x, py + 86), "A match counts only when IoU clears the bar, and the score is averaged "
                          "over bars at 0.3, 0.4, 0.5, 0.6 and 0.7 -- so 'roughly right' still "
                          "fails the strict ones.", font=F["sm"], fill=MUTED)


def table(dr, rows, x, y, widths, head_cols=(OK, BAD)):
    for r, row in enumerate(rows):
        yy = y + r * 33
        if r == 0:
            dr.line([x, yy + 27, x + sum(widths), yy + 27], fill=LINE, width=2)
        for c, cell in enumerate(row):
            cx = x + sum(widths[:c])
            if c == 0:
                dr.text((cx, yy), cell, font=F["b"] if r == 0 else F["r"],
                        fill=MUTED if r else NAVY)
            else:
                dr.text((cx + widths[c] - 20, yy), cell,
                        font=F["smb"] if r == 0 else F["mono"],
                        fill=head_cols[min(c - 1, len(head_cols) - 1)] if r == 0 else INK,
                        anchor="ra")


def make_card_compare(caught, missed):
    def inner(dr, img, y):
        def rat(d, k):
            return d["rationale"][k]

        def split(d, key, nd=2):
            """mean of a feature outside the case, then inside it."""
            m = np.array([c == "1" for c in d["in_case"]])
            v = np.array([np.nan if x is None else x for x in d["feat"][key]], dtype=float)
            fmt = "{:." + str(nd) + "f}"
            return (fmt.format(np.nanmean(v[~m])), fmt.format(np.nanmean(v[m])))

        def pdead(d):
            m = np.array([c == "1" for c in d["in_case"]])
            v = np.array([np.nan if x is None else x for x in d["p_dead"]], dtype=float)
            return np.nanmean(v[m]), np.nanmax(v[m])

        spd_c, spd_m = split(caught, "spd_mean"), split(missed, "spd_mean")
        stl_c, stl_m = split(caught, "frac_still"), split(missed, "frac_still")
        out_c, out_m = split(caught, "n_outside_pitch"), split(missed, "n_outside_pitch")
        pc, pm = pdead(caught), pdead(missed)
        rows = [("", "CAUGHT  35:56", "MISSED  06:46"),
                ("stoppage length", "{:.2f} s".format(caught["duration_s"]),
                 "{:.2f} s".format(missed["duration_s"])),
                ("mean player speed, live -> dead",
                 "{} -> {} m/s".format(*spd_c), "{} -> {} m/s".format(*spd_m)),
                ("share of players standing still",
                 "{} -> {}".format(*stl_c), "{} -> {}".format(*stl_m)),
                ("players tracked off the pitch",
                 "{} -> {}".format(*out_c), "{} -> {}".format(*out_m)),
                ("P(dead), averaged over the stoppage",
                 "{:.3f}".format(pc[0]), "{:.3f}".format(pm[0])),
                ("P(dead) at its highest",
                 "{:.3f}".format(pc[1]), "{:.3f}".format(pm[1])),
                ("evidence for a stoppage", "{:+.0f} nats".format(rat(caught, "emission_nats")),
                 "{:+.0f} nats".format(rat(missed, "emission_nats"))),
                ("best sub-window inside it", "{:+.0f} nats".format(rat(caught, "best_window_nats")),
                 "{:+.0f} nats".format(rat(missed, "best_window_nats"))),
                ("duration prior charges", "-9 nats", "-9 nats"),
                ("so the decoder", "switched", "refused"),
                ("overlap with the reference", "100%", "0%")]
        table(dr, rows, 110, max(y, 250), [560, 250, 250])
    return inner


def make_card_cues(case):
    """The stage-3 split, straight from a real wrong-team case."""
    def inner(dr, img, y):
        r = case["rationale"]
        x, y = 110, max(y, 250)
        dr.text((x, y), "{}  ·  P{}  {}   the model said {}, the reference says {}".format(
            case["match"], case["period"], case["clock"],
            "HOME" if r["p_home"] >= 0.5 else "AWAY",
            "HOME" if r["reference"] == "H" else "AWAY"), font=F["b"], fill=NAVY)
        y += 36
        rows = [("cue the stage-3 model is given", "reading", "points to", "")]
        for c in r["cues"]:
            mark = "model" if c["with_model"] and not c["with_reference"] else (
                "reference" if c["with_reference"] and not c["with_model"] else "both")
            rows.append((c["cue"] + "   (" + c["strength"] + ")", c["reading"],
                         "HOME" if c["points_to"] == "H" else "AWAY", "backs the " + mark))
        table(dr, rows, x, y, [430, 330, 140, 200], head_cols=(MUTED, MUTED, MUTED))
        y += len(rows) * 33 + 14
        draw_rich(dr, "**The cues split, and that split is the whole disagreement.** Three of four "
                      "point at the reference's answer; the model followed the one that did not.",
                  x, y, W - 220, F["cap"], F["capb"], leading=30)
    return inner


def card_ball(dr, img, y):
    x = 110
    y = max(y, 410)
    rows = [("ball tracked in IDSSE", "every single frame"),
            ("ball tracked in the competition match", "59% of frames - it is lost on 41%"),
            ("so stage 3 is scored here with", "41% of the ball hidden, on purpose"),
            ("error rate with the ball hidden vs visible", "1.36x  -  barely worse"),
            ("error rate within a second of a turnover", "11.0x  -  the real problem")]
    dr.rounded_rectangle([x, y, x + 1060, y + 26 + len(rows) * 33], 8,
                         fill=(250, 251, 253), outline=LINE)
    for r, (k, v) in enumerate(rows):
        yy = y + 16 + r * 33
        dr.text((x + 24, yy), k, font=F["r"], fill=MUTED)
        dr.text((x + 1036, yy), v, font=F["smb"], fill=BAD if "11.0x" in v else INK, anchor="ra")


# ---------------------------------------------------------------- storyboard

CAUGHT = "J03WOH_P1_STOPPAGE_CAUGHT_63917"
MISSED = "J03WOH_P1_SPELL_MERGE_20165"
CUES = "J03WOH_P1_WRONG_TEAM_10260"


def load(cid, data_dir):
    with open(os.path.join(data_dir, "cases", cid + ".json")) as fh:
        return json.load(fh)


def captions_caught(d, onset, end):
    r = d["rationale"]
    return [
        (0, "**Live play.** Twenty-two player tracks at 25 frames a second, and a ball position "
            "that comes and goes. The model has two jobs: **is the ball in play**, and **whose is "
            "it**."),
        (onset - 25, "Watch the **players**, not the ball. Stage 2 never reads the ball position — "
                     "it decides live-or-dead from the 22 tracks alone."),
        (onset - 6, "Something is about to happen: the ball is near the touchline and the mean "
                    "speed is already falling."),
        (onset, "**The ball goes out.** Mean speed falls 2.16 → 1.05 m/s, a fifth of the players "
                "stop dead, and one is tracked **off the pitch** — nobody stands outside the line "
                "while the ball is live."),
        (onset + 4, "**P(dead) = 0.98.** Read the grey line at the bottom: the stoppage is worth "
                    "**{:+.0f} nats** of evidence against a **9-nat** charge for interrupting live "
                    "play. It pays for itself {:.0f} times over.".format(
                        r["emission_nats"], r["emission_nats"] / r["duration_cost_nats"])),
        (onset + 14, "**The model's whistle is 0.08 s early** — two frames before the actual one. "
                     "Inside the tolerance of every IoU bar we are scored on."),
        (onset + 40, "Dead ball held. Both strips agree frame after frame. The amber in the "
                     "ball strip is the ball hidden from the model on purpose — it had a "
                     "position for 43% of this stoppage and did not need one."),
        (end - 10, "The away team lines up the throw. The reference is about to go live again."),
        (end, "**Restart found 0.16 s late** — four frames. Overlap with the actual span: "
              "**100%**. This spell is a hit at every threshold."),
        (end + 14, "And the ball changed hands across the stoppage — **home before, away after** — "
                   "which the model also has right. **This is what a success looks like.**"),
    ]


def captions_missed(d, onset, end):
    r = d["rationale"]
    return [
        (0, "**Same match. Same half. Twenty-nine minutes earlier.** Another stoppage, almost the "
            "same length: 7.4 s against 9.2 s."),
        (onset - 20, "Live play, home in possession. Nothing so far tells this apart from the case "
                     "you just watched."),
        (onset, "**The ball goes dead here.** But the numbers barely move: speed only falls to "
                "1.23 m/s and just **12% of players stand still**. Nobody steps off the pitch."),
        (onset + 5, "**P(dead) peaks at 0.75 and averages 0.25.** The per-frame view is **unsure**, "
                    "not convinced — against 0.98 in the case that worked."),
        (onset + 16, "**Here is the model's own reasoning.** Over the whole stoppage the evidence "
                     "for dead is **{:+.0f} nats**. The best 2.4 s window inside it is worth "
                     "**{:+.0f}**, and interrupting live play costs **9**. Nothing covers the "
                     "charge, so it refuses.".format(r["emission_nats"], r["best_window_nats"])),
        (onset + 34, "**The red band is the failure.** For 7.4 seconds the reference says dead and "
                     "the model says home. Its stoppage never starts, so there is no timing error "
                     "to quote — **there is nothing to compare**."),
        (end, "Play restarts, and the model never noticed it had stopped. Home before, home after: "
              "one unbroken spell where the reference has two."),
        (end + 12, "**That fusion is the expensive part.** Two real possessions become one long "
                   "prediction, and a merged spell fails the overlap test **at both ends**."),
    ]


def build(args):
    import imageio.v2 as imageio

    caught = load(CAUGHT, args.data)
    missed = load(MISSED, args.data)
    cues = load(CUES, args.data)

    frames, stills = [], {}

    def hold(img, seconds, key=None):
        arr = np.asarray(img)
        frames.extend([arr] * int(seconds * FPS))
        if key:
            stills[key] = img

    def replay(d, caps, chapter, accent, start, slow, keys):
        sched = sorted(caps)
        for i in range(start, len(d["f"])):
            cap = [t for s, t in sched if s <= i]
            img = scene_frame(d, i, chapter, cap[-1] if cap else "", accent)
            frames.extend([np.asarray(img)] * slow.get(i, 3))
            if i in keys:
                stills[keys[i]] = img

    hold(card("Does the model know when the ball is dead?",
              "MODEL FAILURE REVIEW — GUIDED DEMO",
              ["Two stoppages from **the same half of the same match**, almost the same length.",
               "The model catches one to within **a fifth of a second**, and misses the other "
               "**completely**.",
               "They look alike on the pitch. They do not look alike **in the data** — and that "
               "is the whole lesson.",
               "",
               "All of it is the **open-access IDSSE** development set. The sealed test matches "
               "are untouched."]), 6.5, "01_title")

    hold(card("What the model actually gets", "THE INPUT",
              ["**22 player tracks** — x, y at 25 Hz. Always there.",
               "**One ball track** — always there in IDSSE, missing on **41%** of the "
               "competition's frames, so we hide it here at that rate to match.",
               "**No video, no audio, no whistle.** There is no referee to listen to.",
               "",
               "**Stage 2 — is the ball live?** Players only: speed, how many stand still, how "
               "many are off the pitch, how tight the lines are.",
               "**Stage 3 — whose ball is it?** Ball position, who is nearest, pitch control.",
               "A **duration model** then decodes the sequence. It knows how long real stoppages "
               "and real possessions last, so a brief flicker has to outweigh that prior to "
               "survive — and that is the arithmetic you will see in the grey line."],
              accent=NAVY), 10, "02_input")

    on, end = caught["in_case"].find("1"), caught["in_case"].rfind("1")
    hold(card("Case 1 — the model gets it right", "J03WOH · FIRST HALF · 35:56",
              ["A **9.16 s** stoppage. Home had the ball before it, away after it.",
               "Watch **P(dead)** on the right, the **timing bars** under it, and the model's own "
               "reasoning along the bottom."], accent=OK), 5)
    replay(caught, captions_caught(caught, on, end), "CASE 1 — MODEL AGREES", OK, 30,
           slow={**{i: 7 for i in range(on - 3, on + 10)},
                 **{i: 6 for i in range(end - 2, end + 6)}},
           keys={on + 5: "03_caught_whistle", end + 3: "04_caught_restart"})

    hold(card("What the clock says", "CASE 1 — ACTUAL vs MODEL",
              ["Actual stoppage **9.16 s**.   Model stoppage **9.40 s**.",
               "**Whistle 0.08 s early** (2 frames).   **Restart 0.16 s late** (4 frames).",
               "Overlap with the actual span: **100%** — a hit at IoU 0.3, 0.4, 0.5, 0.6 and 0.7.",
               "",
               "Across the five development matches the model finds **91% of all stoppages** at "
               "IoU 0.5. Live-or-dead is the part that works."], accent=OK), 9, "05_timing")

    hold(card("How we check it — spellF1", "THE MEASURING STICK",
              ["A **possession spell** is one unbroken run of one team having the ball.",
               "A predicted spell matches a real one only if it is the **same team** and their "
               "overlap in time is big enough."],
              accent=NAVY, extra=card_iou), 10, "06_iou")

    on2, end2 = missed["in_case"].find("1"), missed["in_case"].rfind("1")
    hold(card("Case 2 — the model gets it wrong", "J03WOH · FIRST HALF · 06:46",
              ["A **7.44 s** stoppage. Home had the ball before it — and after it.",
               "Same model, same match, same half. Watch **P(dead)** fail to commit."],
              accent=BAD), 5)
    replay(missed, captions_missed(missed, on2, end2), "CASE 2 — MODEL DISAGREES", BAD, 30,
           slow={**{i: 7 for i in range(on2 - 3, on2 + 12)},
                 **{i: 5 for i in range(end2 - 2, end2 + 4)}},
           keys={on2 + 6: "07_missed_onset", end2 + 2: "08_missed_restart"})

    hold(card("The same event, two different sets of numbers", "WHY ONE WORKED",
              ["Nothing about the model changed between these two. **The evidence did.**"],
              accent=NAVY, extra=make_card_compare(caught, missed)), 13, "09_compare")

    hold(card("So was the model wrong — or was the evidence?", "THE RATIONALE, STATED PLAINLY",
              ["On its own objective the model behaved correctly in **both** cases. It took the "
               "switch that was paid for and refused the one that was not.",
               "That is **not** an excuse. The stoppage was real and we missed it, and spellF1 "
               "charges us for it either way.",
               "But it does tell us **where the fix has to go**: into the evidence — the features "
               "that should have made this stoppage look like a stoppage — and not into the "
               "decoder's threshold.",
               "",
               "We measured the alternative: forcing the decoder to accept weak evidence lifts "
               "recall and **triples invented stoppages**, 11.6 → 36.2 per match, for a spellF1 "
               "move inside the noise band."], accent=WARN), 12, "10_rationale")

    hold(card("When it picks the wrong team", "STAGE 3 — THE OTHER KIND OF DISAGREEMENT",
              ["Live-or-dead is arithmetic. **Whose ball it is** is a judgement between cues that "
               "do not always agree."],
              accent=BAD, extra=make_card_cues(cues)), 13, "11_cues")

    hold(card("Could the model see the ball?", "THE QUESTION EVERYONE ASKS FIRST",
              ["IDSSE tracks the ball on **every frame**. The competition match loses it on "
               "**41%** — so we hide it here at that same rate, in bursts, on purpose.",
               "In case 1 the model had a ball position for **43%** of the stoppage. In case 2, "
               "**55%** — more ball, and it still failed.",
               "That is not a coincidence. **Stage 2 never reads the ball at all**, so a missing "
               "ball cannot explain a missed stoppage.",
               "Stage 3 does depend on it, and falls back to pitch control and player geometry "
               "where it is gone. The **amber ring** in the replay marks those frames."],
              accent=WARN, extra=card_ball), 12, "12_ball")

    hold(card("Where the timing really hurts", "TURNOVERS",
              ["Stoppages: **91%** found at IoU 0.5, with timing errors measured in **frames**.",
               "Turnovers are another story. Of **1,549** real changes of possession:",
               "**22.1%** have a model turnover within **one second**.",
               "**12.8%** within half a second.",
               "**1.0%** — sixteen of them — are clean, with the right team on both sides for a "
               "full second either way.",
               "",
               "**That is the honest headline.** The model knows when the ball stops. It does not "
               "know the moment it changes hands."], accent=BAD), 13, "13_turnovers")

    hold(card("Now go and look for yourself", "THE REVIEW PAGE",
              ["**Start here: cases the model gets RIGHT** — 247 of them, so you know what a clean "
               "signal looks like before judging a failure.",
               "Then **failures: confirmed only** — 1,972 where the reference agrees with itself "
               "and the model is simply wrong.",
               "**Contested** — 971 where the DFL's two products disagree with each other. Not "
               "model defects, and not ours to fix.",
               "",
               "Every case carries the **rationale**, the **timing bars**, the **ball strip** and "
               "the **features** you have just watched.",
               "Bookmark anything that looks wrong. **Nothing you write becomes a training label "
               "until you say so.**"], accent=NAVY), 12, "14_closing")

    os.makedirs(args.out, exist_ok=True)
    stilldir = os.path.join(args.out, "stills")
    os.makedirs(stilldir, exist_ok=True)
    for k, img in stills.items():
        img.save(os.path.join(stilldir, k + ".png"))

    path = os.path.join(args.out, "model_demo.mp4")
    wr = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=8, pixelformat="yuv420p",
                            ffmpeg_log_level="error")
    for f in frames:
        wr.append_data(f)
    wr.close()
    print("frames {}  ({:.0f} s at {} fps)".format(len(frames), len(frames) / FPS, FPS))
    print("wrote  " + path)
    print("stills " + stilldir)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "demo"))
    build(ap.parse_args())


if __name__ == "__main__":
    main()
