#!/usr/bin/env python3
"""Summarise reads_concurrency.sh output: aggregate throughput per config.

    probes/reads_concurrency.py reads-concurrency.jsonl

The aggregate for one launch is the total sweeps of all its processes over
the union of their loop windows, so it counts the whole wall time and not
only the overlapped part. `overlap` is the intersection over the union of
those windows and shows how much of each loop ran alongside the others.
"""

import json
import sys
from collections import defaultdict

with open(sys.argv[1]) as f:
    rows = [json.loads(line) for line in f if '"lanes"' in line]
launches = defaultdict(list)
for r in rows:
    launches[(r["lanes"], r["n"], r["round"])].append(r)

print("reads  N  round  overlap  model-sweeps/s  lane-sweeps/s")
medians = defaultdict(list)
for (lanes, n, rnd), rs in sorted(
    launches.items(), key=lambda kv: (-kv[0][0], kv[0][1], kv[0][2])
):
    assert len(rs) == n, (
        f"{n} processes expected at {lanes} reads, round {rnd}, got {len(rs)}"
    )
    start = min(r["loop_start_us"] for r in rs)
    end = max(r["loop_end_us"] for r in rs)
    overlap = max(
        0, min(r["loop_end_us"] for r in rs) - max(r["loop_start_us"] for r in rs)
    ) / (end - start)
    aggregate = sum(r["calls"] for r in rs) / ((end - start) / 1e6)
    medians[(lanes, n)].append(aggregate)
    print(
        f"{lanes:5d} {n:2d} {rnd:6d}  {overlap:7.2f}  {aggregate:14.0f}  {aggregate * lanes:13.0f}"
    )

print("\nmedian over rounds, lane-sweeps/s relative to one process at 128 reads:")
base = sorted(medians[(128, 1)])[len(medians[(128, 1)]) // 2] * 128
for (lanes, n), values in sorted(medians.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
    m = sorted(values)[len(values) // 2]
    print(f"  {lanes:3d} reads x{n}: {m:6.0f} model-sweeps/s, {m * lanes / base:.2f}x")
