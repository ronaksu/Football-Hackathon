import json
import pandas as pd

SRC = ("D:/Football Hackathon/external_data/raw/soccernet_gar_probe/"
       "sngar_tracking-full_test_metadata.parquet")

df = pd.read_parquet(SRC)
rows = []
for p in df["sample_payload"]:
    d = json.loads(p)
    m = d.get("metadata", {})
    rows.append({"gid": str(m.get("game_id")), "act": d["labels"]["action"]["label"],
                 "team": m.get("team")})
d = pd.DataFrame(rows)

ct = pd.crosstab(d["act"], d["team"])
ct["away_pct"] = (100 * ct["away"] / (ct["away"] + ct["home"])).round(1)
print("=== team attribution, by action ===")
print(ct.sort_values("away_pct"))
print()
print("=== per match, share of PASS events labelled away ===")
pas = d[d["act"] == "PASS"]
g = pas.groupby("gid")["team"].value_counts().unstack(fill_value=0)
g["away_pct"] = (100 * g["away"] / (g["away"] + g["home"])).round(1)
print(g)
