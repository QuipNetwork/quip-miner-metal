#!/usr/bin/env python3
"""Build the regen input: 60 most recent QPU-won qblocks plus matched controls.

The QPU population is miner dd122d6815693... , the only account whose
device_access_time_us ever falls below 550 ms. Controls are the classical-won
qblock nearest in qblock id to each QPU block, without reuse, so the control
sees the same difficulty era and the same topology.

    make_subset.py qblocks-all.json qblocks-subset.json cohorts.json
"""

import json
import sys

QPU_MINER_PREFIX = "dd122d6815690c97190795625d3be61f220fccac507d3df92fd484f006092e9f"
N = 60


def main():
    src, out_path, cohort_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(src) as f:
        d = json.load(f)
    blocks = sorted(d["blocks"], key=lambda b: b["qblock_id"])

    qpu = [b for b in blocks if b["miner"].startswith(QPU_MINER_PREFIX)]
    qpu = sorted(qpu, key=lambda b: -b["qblock_id"])[:N]
    lo = min(b["qblock_id"] for b in qpu)
    hi = max(b["qblock_id"] for b in qpu)

    pool = sorted(
        (b for b in blocks if not b["miner"].startswith(QPU_MINER_PREFIX)),
        key=lambda b: b["qblock_id"],
    )
    used, control = set(), []
    for q in sorted(qpu, key=lambda b: b["qblock_id"]):
        best = min(
            (b for b in pool if b["qblock_id"] not in used),
            key=lambda b: (abs(b["qblock_id"] - q["qblock_id"]), b["qblock_id"]),
        )
        used.add(best["qblock_id"])
        control.append(best)

    subset = qpu + control
    ids = {b["qblock_id"] for b in subset}
    assert len(ids) == len(subset), "duplicate qblock in subset"
    hashes = {b["topology_hash"] for b in subset}
    with open(out_path, "w") as f:
        json.dump(
            {
                "latest_qblock_id": d["latest_qblock_id"],
                "current_difficulty": d["current_difficulty"],
                "blocks": subset,
                "topologies": {h: d["topologies"][h] for h in hashes},
            },
            f,
        )
    with open(cohort_path, "w") as f:
        json.dump(
            {
                "qpu_miner_prefix": QPU_MINER_PREFIX,
                "qpu": sorted(b["qblock_id"] for b in qpu),
                "control": sorted(b["qblock_id"] for b in control),
            },
            f,
            indent=1,
        )
    gaps = sorted(
        abs(c["qblock_id"] - q["qblock_id"])
        for q, c in zip(sorted(qpu, key=lambda b: b["qblock_id"]), control)
    )
    print(f"qpu blocks: {len(qpu)} ids {lo}..{hi}")
    print(
        f"control blocks: {len(control)} ids {control[0]['qblock_id']}..{control[-1]['qblock_id']}"
    )
    print(f"control id offset: median {gaps[len(gaps) // 2]} max {gaps[-1]}")
    print(f"topologies in subset: {[h[:8] for h in hashes]}")
    print(f"wrote {out_path} with {len(subset)} blocks")


if __name__ == "__main__":
    main()
