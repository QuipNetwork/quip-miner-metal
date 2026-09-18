#!/usr/bin/env python3
"""Select the 500 longest rounds from the chain history as a regen input.

    make_hard500.py ALL_QBLOCKS_JSON OUT_JSON META_CSV [TOPOLOGY_PREFIX] [COUNT]

Hardness is the round length: the gap in `submitted_at` between a qblock and
the qblock one id below it. That gap is the network's own time to find a valid
proof for the problem. Blocks keep their own difficulty target and winning
energy, so the output is a drop-in input for `scripts/testnet/regen`.
"""

import csv
import json
import statistics
import sys

ALL, OUT_JSON, META = sys.argv[1], sys.argv[2], sys.argv[3]
TOPO = sys.argv[4] if len(sys.argv) > 4 else "cbec1eb4"
COUNT = int(sys.argv[5]) if len(sys.argv) > 5 else 500

with open(ALL) as f:
    data = json.load(f)
blocks = {b["qblock_id"]: b for b in data["blocks"]}

rows = []
for qid, b in blocks.items():
    prev = blocks.get(qid - 1)
    if prev is None:
        continue
    if not b["topology_hash"].startswith(TOPO):
        continue
    rows.append(
        {
            "qblock_id": qid,
            "round_len": b["submitted_at"] - prev["submitted_at"],
            "target_milli": b["difficulty"]["max_energy_milli"],
            "winner_milli": b["energy_milli"],
            "winner_margin_milli": b["difficulty"]["max_energy_milli"]
            - b["energy_milli"],
            "min_solutions": b["difficulty"]["min_solutions"],
            "submitted_at": b["submitted_at"],
            "device_access_time_us": b["device_access_time_us"],
        }
    )

# Longest round first; the qblock id breaks ties so the set is reproducible.
rows.sort(key=lambda r: (-r["round_len"], -r["qblock_id"]))
picked = rows[:COUNT]

out = {
    "latest_qblock_id": data["latest_qblock_id"],
    "current_difficulty": data["current_difficulty"],
    "blocks": [blocks[r["qblock_id"]] for r in picked],
    "topologies": data["topologies"],
}
with open(OUT_JSON, "w") as f:
    json.dump(out, f)

picked.sort(key=lambda r: r["qblock_id"])
with open(META, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(picked[0].keys()))
    w.writeheader()
    w.writerows(picked)

lens = sorted(r["round_len"] for r in picked)
margins = sorted(r["winner_margin_milli"] for r in picked)
allsorted = sorted(r["round_len"] for r in rows)
print(f"candidates on {TOPO}: {len(rows)}, picked {len(picked)}")
print(
    f"round length picked: min {lens[0]}, median {statistics.median(lens)}, "
    f"max {lens[-1]}; whole chain median {statistics.median(allsorted)}"
)
print(
    f"winner margin below target picked: min {margins[0]}, "
    f"median {statistics.median(margins)}, max {margins[-1]}"
)
print(f"min_solutions values: {sorted({r['min_solutions'] for r in picked})}")
print(f"qblock id range {picked[0]['qblock_id']} to {picked[-1]['qblock_id']}")
print(
    f"overlap with the 60-block set 3191-3250: "
    f"{sum(1 for r in picked if 3191 <= r['qblock_id'] <= 3250)}"
)
print(f"-> {OUT_JSON}, {META}")
