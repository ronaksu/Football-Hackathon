"""Parse the seven IDSSE Bundesliga matches (DFL / TRACAB) into the canonical representation.

    python external_data/scripts/parse_dfl.py
    python external_data/scripts/parse_dfl.py --match DFL-MAT-J03WMX

WHY THIS SOURCE MATTERS MOST
----------------------------
Every other source makes us infer possession from events. This one ships it. The ball's own
FrameSet carries two attributes on every frame:

    BallStatus       0 = Dead, 1 = Alive
    BallPossession   1 = Home, 2 = Away

Those definitions are not my reading of the numbers: they are the definitions used by
`floodlight`, the library written by the same group that published the dataset and cited in its
paper (floodlight/io/dfl.py, `definitions={1: "Home", 2: "Away"}` and
`definitions={0: "Dead", 1: "Alive"}`).

That gives a direct frame-level H / A / D label from the provider, which is as close to reference
truth as anything available under an open licence.

WHAT IT STILL IS NOT
--------------------
It is the DFL's annotation convention, not the hackathon's. The hackathon defines a change of
possession as an opponent gaining CONTROLLED possession, and says a touch, a non-intercepting
deflection, or an aerial duel won without subsequent control do not count. Whether the DFL feed
draws the line in the same place is not documented in the dataset, and I have not verified it.
So these labels are graded `reference` - the strongest category available - while the comparison
with the hackathon's convention stays an explicitly open question in the readiness report.

ATTRIBUTION REQUIRED (CC-BY-4.0)
--------------------------------
Data provided with authorisation of the Deutsche Fussball Liga (DFL). Any use must cite:
Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of spatiotemporal and
event data in elite soccer. Scientific Data, 12(1), 195. https://doi.org/10.1038/s41597-025-04505-y
"""

import argparse
import csv
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "raw", "idsse")
OUT = os.path.join(ROOT, "processed", "idsse")
REPORTS = os.path.join(ROOT, "reports")

sys.path.insert(0, HERE)
from common_schema import FRAME_COLUMNS, PLAYER_COLUMNS, PROVISIONAL  # noqa: E402

SEGMENTS = {"firstHalf": 1, "secondHalf": 2}
POSSESSION = {1: "H", 2: "A"}      # floodlight: {1: "Home", 2: "Away"}
BALLSTATUS_ALIVE = 1               # floodlight: {0: "Dead", 1: "Alive"}


def match_files(match_id):
    def one(pattern):
        hits = glob.glob(os.path.join(RAW, pattern.format(m=match_id)))
        return hits[0] if hits else None
    return {
        "info": one("DFL_02_01_matchinformation_*_{m}.xml"),
        "events": one("DFL_03_02_events_raw_*_{m}.xml"),
        "positions": one("DFL_04_03_positions_raw_observed_*_{m}.xml"),
    }


def list_matches():
    ids = set()
    for path in glob.glob(os.path.join(RAW, "DFL_04_03_positions_raw_observed_*.xml")):
        hit = re.search(r"(DFL-MAT-[A-Z0-9]+)\.xml$", os.path.basename(path))
        if hit:
            ids.add(hit.group(1))
    return sorted(ids)


def read_match_info(path):
    """Home/away team ids, pitch size and the match title."""
    root = ET.parse(path).getroot()
    teams = {}
    for team in root.iter("Team"):
        role = (team.get("Role") or "").lower()
        if role in ("home", "guest", "away"):
            teams["home" if role == "home" else "away"] = {
                "id": team.get("TeamId"), "name": team.get("TeamName")}
    general = next(root.iter("General"), None)
    info = {
        "teams": teams,
        "title": general.get("MatchTitle") if general is not None else "",
        "competition": general.get("CompetitionName") if general is not None else "",
        "season": general.get("Season") if general is not None else "",
        "matchday": general.get("MatchDay") if general is not None else "",
    }
    env = next(root.iter("Environment"), None)
    info["pitch"] = None
    if env is not None and env.get("PitchX"):
        try:
            info["pitch"] = (float(env.get("PitchX")), float(env.get("PitchY")))
        except ValueError:
            pass
    return info


def stream_positions(path, home_id, away_id):
    """Walk the 400 MB positions file once, keeping only what is needed.

    iterparse with an explicit clear() keeps peak memory to one FrameSet rather than the whole
    document, which otherwise needs several gigabytes.
    """
    ball = {}          # segment -> dict of arrays
    players = defaultdict(list)   # (segment, team) -> list of (person, frames, xs, ys)
    pitch = None
    anomalies = Counter()

    context = ET.iterparse(path, events=("end",))
    for _, element in context:
        if element.tag == "PitchSize" and pitch is None:
            try:
                pitch = (float(element.get("X")), float(element.get("Y")))
            except (TypeError, ValueError):
                pitch = None
            element.clear()
            continue
        if element.tag != "FrameSet":
            continue

        segment = element.get("GameSection")
        team_id = (element.get("TeamId") or "").upper()
        person = element.get("PersonId")
        frames = element.findall("Frame")
        if not frames or segment not in SEGMENTS:
            element.clear()
            continue

        n = np.empty(len(frames), dtype=np.int64)
        xs = np.empty(len(frames), dtype=np.float32)
        ys = np.empty(len(frames), dtype=np.float32)
        if team_id == "BALL":
            status = np.empty(len(frames), dtype=np.int8)
            possession = np.empty(len(frames), dtype=np.int8)
            zs = np.empty(len(frames), dtype=np.float32)
            times = []
            for i, frame in enumerate(frames):
                n[i] = int(frame.get("N"))
                xs[i] = _f(frame.get("X"))
                ys[i] = _f(frame.get("Y"))
                zs[i] = _f(frame.get("Z"))
                raw_status = frame.get("BallStatus")
                raw_poss = frame.get("BallPossession")
                if raw_status is None:
                    anomalies["ball_frame_without_BallStatus"] += 1
                if raw_poss is None:
                    anomalies["ball_frame_without_BallPossession"] += 1
                status[i] = int(float(raw_status)) if raw_status is not None else -1
                possession[i] = int(float(raw_poss)) if raw_poss is not None else -1
                times.append(frame.get("T"))
            ball[segment] = {"n": n, "x": xs, "y": ys, "z": zs,
                             "status": status, "possession": possession, "t": times}
        elif team_id in (home_id.upper(), away_id.upper()):
            side = "home" if team_id == home_id.upper() else "away"
            for i, frame in enumerate(frames):
                n[i] = int(frame.get("N"))
                xs[i] = _f(frame.get("X"))
                ys[i] = _f(frame.get("Y"))
            players[(segment, side)].append((person, n, xs, ys))
        # referee frame sets are read and discarded on purpose

        element.clear()

    return ball, players, pitch, anomalies


def _f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def build_match(match_id):
    paths = match_files(match_id)
    info = read_match_info(paths["info"])
    home_id = info["teams"]["home"]["id"]
    away_id = info["teams"]["away"]["id"]

    ball, players, pitch, anomalies = stream_positions(paths["positions"], home_id, away_id)
    pitch = info["pitch"] or pitch or (105.0, 68.0)
    length, width = pitch

    frame_rows, player_rows = [], []
    for segment, period in SEGMENTS.items():
        if segment not in ball:
            continue
        b = ball[segment]
        order = np.argsort(b["n"])
        n = b["n"][order]
        bx, by = b["x"][order], b["y"][order]
        status, possession = b["status"][order], b["possession"][order]

        # DFL positions are metres measured from the centre spot. Normalise to the 0-1
        # convention the competition and Metrica both use.
        nx = (bx + length / 2.0) / length
        ny = (by + width / 2.0) / width

        observed = np.isfinite(bx) & np.isfinite(by)
        frame = pd.DataFrame({
            "period": np.int16(period), "frame_native": n,
            "time_s": (n - n[0]) / 25.0,
            "ball_x": np.where(observed, nx, np.nan),
            "ball_y": np.where(observed, ny, np.nan),
            "ball_observed": observed,
            "_status": status, "_possession": possession,
        })

        index = {value: i for i, value in enumerate(n)}
        for side in ("home", "away"):
            count = np.zeros(len(n), dtype=np.int16)
            sum_x = np.zeros(len(n)); sum_y = np.zeros(len(n))
            sum_xx = np.zeros(len(n)); sum_yy = np.zeros(len(n))
            best = np.full(len(n), np.inf)
            for slot, (person, pn, px, py) in enumerate(players[(segment, side)]):
                rows = np.array([index.get(v, -1) for v in pn])
                keep = (rows >= 0) & np.isfinite(px) & np.isfinite(py)
                rows = rows[keep]
                if not len(rows):
                    continue
                gx, gy = px[keep], py[keep]
                count[rows] += 1
                sum_x[rows] += gx; sum_y[rows] += gy
                sum_xx[rows] += gx * gx; sum_yy[rows] += gy * gy
                d = np.hypot(gx - bx[rows], gy - by[rows])
                np.minimum.at(best, rows, np.where(np.isfinite(d), d, np.inf))
                player_rows.append(pd.DataFrame({
                    "period": np.int16(period), "frame_native": n[rows],
                    "team": side, "slot": np.int16(slot),
                    "x": ((gx + length / 2.0) / length).astype(np.float32),
                    "y": ((gy + width / 2.0) / width).astype(np.float32),
                }))

            with np.errstate(invalid="ignore", divide="ignore"):
                mean_x = np.where(count > 0, sum_x / count, np.nan)
                mean_y = np.where(count > 0, sum_y / count, np.nan)
                var_x = np.where(count > 1, sum_xx / count - mean_x ** 2, np.nan)
                var_y = np.where(count > 1, sum_yy / count - mean_y ** 2, np.nan)
            frame["n_{}_observed".format(side)] = count
            frame["{}_centroid_x".format(side)] = (mean_x + length / 2.0) / length
            frame["{}_centroid_y".format(side)] = (mean_y + width / 2.0) / width
            frame["{}_spread".format(side)] = np.sqrt(
                np.clip(var_x, 0, None) + np.clip(var_y, 0, None))
            frame["nearest_{}_m".format(side)] = np.where(np.isfinite(best), best, np.nan)

        frame_rows.append(frame)

    frames = pd.concat(frame_rows, ignore_index=True)
    frames = frames.sort_values(["period", "frame_native"]).reset_index(drop=True)

    frames["source"] = "idsse"
    frames["match_id"] = match_id
    frames["fps"] = 25
    frames["nearest_margin_m"] = frames["nearest_away_m"] - frames["nearest_home_m"]

    step = np.hypot(frames["ball_x"].diff() * length, frames["ball_y"].diff() * width)
    consecutive = (frames["frame_native"].diff() == 1) & (frames["period"].diff() == 0)
    both = frames["ball_observed"] & frames["ball_observed"].shift(1) & consecutive
    frames["ball_speed_mps"] = np.where(both, step * 25, np.nan)

    # --- labels, straight from the provider fields -------------------------------------------
    status = frames.pop("_status")
    possession = frames.pop("_possession")
    alive = status == BALLSTATUS_ALIVE
    team = possession.map(POSSESSION)
    frames["label"] = np.where(~alive, "D", team)
    frames["label"] = frames["label"].where(status >= 0, None)
    unknown_team = alive & team.isna()
    frames.loc[unknown_team, "label"] = None
    frames["label_source"] = "provider_field"
    frames["label_quality"] = np.where(frames["label"].notna(), "reference", "none")
    frames["label_valid"] = frames["label"].notna()

    frames["q_ball_missing"] = ~frames["ball_observed"]
    frames["q_ball_motion_unreliable"] = (
        frames["ball_speed_mps"] > PROVISIONAL["max_plausible_ball_speed_mps"]).fillna(False)
    frames["q_label_missing"] = ~frames["label_valid"]
    frames["q_ball_and_label_missing"] = frames["q_ball_missing"] & frames["q_label_missing"]
    frames["q_source_disagreement"] = False
    frames["q_alignment_problem"] = False
    frames["model_uncertainty"] = np.nan

    players_df = (pd.concat(player_rows, ignore_index=True) if player_rows
                  else pd.DataFrame(columns=["period", "frame_native", "team", "slot", "x", "y"]))
    players_df["source"] = "idsse"
    players_df["match_id"] = match_id

    for column in FRAME_COLUMNS:
        if column not in frames.columns:
            frames[column] = np.nan
    return (frames[FRAME_COLUMNS], players_df[PLAYER_COLUMNS], info, pitch,
            anomalies, int(unknown_team.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", nargs="+", default=None)
    args = parser.parse_args()

    matches = args.match or list_matches()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(REPORTS, exist_ok=True)
    provenance = []

    for match_id in matches:
        frames, players, info, pitch, anomalies, unknown = build_match(match_id)
        n = len(frames)
        counts = Counter(frames["label"].dropna())
        print("\n--- {}  {} ---".format(match_id, info["title"]))
        print("  {} {} MD{}   pitch {}x{} m".format(
            info["competition"], info["season"], info["matchday"], *pitch))
        print("  home {} | away {}".format(info["teams"]["home"]["name"],
                                           info["teams"]["away"]["name"]))
        print("  frames {:,}  periods {}  fps 25".format(
            n, sorted(frames["period"].unique().tolist())))
        for period in sorted(frames["period"].unique()):
            part = frames[frames["period"] == period]
            print("    period {}: frames {:,}..{:,} ({:,} rows), contiguous {}".format(
                period, int(part["frame_native"].min()), int(part["frame_native"].max()),
                len(part), bool((part["frame_native"].diff().dropna() == 1).all())))
        print("  tracked per frame: home {:.1f}, away {:.1f}".format(
            frames["n_home_observed"].mean(), frames["n_away_observed"].mean()))
        print("  ball observed {:,} ({:.1f}%)   missing {:.1f}%".format(
            int(frames["ball_observed"].sum()), 100 * frames["ball_observed"].mean(),
            100 * (~frames["ball_observed"]).mean()))
        print("  labels: " + "  ".join("{} {:,} ({:.1f}%)".format(k, v, 100 * v / n)
                                       for k, v in sorted(counts.items())))
        print("  label_valid {:.2f}%   alive-but-no-team {:,}   anomalies {}".format(
            100 * frames["label_valid"].mean(), unknown, dict(anomalies) or "none"))
        print("  motion unreliable {:,} frames".format(
            int(frames["q_ball_motion_unreliable"].sum())))

        folder = os.path.join(OUT, match_id)
        os.makedirs(folder, exist_ok=True)
        frames.to_parquet(os.path.join(folder, "frames.parquet"), index=False)
        players.to_parquet(os.path.join(folder, "players.parquet"), index=False)

        for quality in sorted(set(frames["label_quality"])):
            mask = frames["label_quality"] == quality
            provenance.append({
                "source": "idsse", "match_id": match_id, "label_quality": quality,
                "label_source": "provider_field" if quality == "reference" else "none",
                "n_frames": int(mask.sum()),
                "pct_of_match": round(100.0 * mask.mean(), 2),
                "eligible_for_training": quality == "reference",
                "eligible_for_evaluation": quality == "reference",
                "definition_note": (
                    "BallStatus {0:Dead,1:Alive} and BallPossession {1:Home,2:Away} per "
                    "floodlight/io/dfl.py, written by the dataset's own authors. Whether the "
                    "DFL's possession convention matches the hackathon's CONTROL-based "
                    "definition is NOT established."),
                "home_mapping": "explicit Role=home/guest in the match information XML",
            })

    path = os.path.join(REPORTS, "LabelProvenance.csv")
    touched = {(r["source"], r["match_id"]) for r in provenance}
    existing, fields = [], None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            existing = [r for r in reader
                        if (r.get("source"), r.get("match_id")) not in touched]
    fields = fields or list(provenance[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in existing + provenance:
            writer.writerow({k: row.get(k, "") for k in fields})
    print("\nprovenance -> {}".format(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
