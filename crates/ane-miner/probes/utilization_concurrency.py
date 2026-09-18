#!/usr/bin/env python3
"""Summarise utilization_concurrency.sh output: aggregate sweep rate and
power per launch, relative to one process of the same configuration.

    crates/ane-miner/probes/utilization_concurrency.py OUT_DIR [summary.json]

A launch's aggregate rate is the total calls of its processes over the
union of their loop windows, so it charges the whole wall time. Windows
with another client's program creation in the aned log (5-second lead
included) are flagged, and each configuration reports the median over its
clean launches when any exist.
"""

import json
import re
import statistics
import sys
from datetime import datetime

OUT = sys.argv[1]
SAMPLE_MS = 250
LEAD_US = 5_000_000
CREATION = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+).*ProgramCreateNewInstance: procedureName"
)


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.startswith("{")]


def epoch_us(stamp):
    return int(datetime.fromisoformat(stamp).timestamp() * 1_000_000)


results = [r for r in load(f"{OUT}/results.jsonl") if "calls" in r]
power = [
    (epoch_us(s["timestamp"]), s["ane_power"], s["ram_power"])
    for s in load(f"{OUT}/power.jsonl")
]
creations = []
try:
    with open(f"{OUT}/aned.log") as f:
        creations = [epoch_us(m.group(1)) for m in map(CREATION.match, f) if m]
except FileNotFoundError:
    pass

launches = {}
for r in results:
    launches.setdefault((r["round"], r["encoding"], r["h"], r["w"], r["n"]), []).append(
        r
    )

summary = {}
for (rnd, enc, h, w, n), procs in sorted(launches.items()):
    start = min(p["wall_start_us"] for p in procs)
    end = max(p["wall_end_us"] for p in procs)
    union = (end - start) / 1e6
    calls = sum(p["calls"] for p in procs)
    inside = [s for s in power if start + SAMPLE_MS * 1000 <= s[0] <= end]
    external = sum(1 for t in creations if start - LEAD_US <= t <= end)
    overlap_start = max(p["wall_start_us"] for p in procs)
    overlap_end = min(p["wall_end_us"] for p in procs)
    launch = {
        "round": rnd,
        "processes": n,
        "calls": calls,
        "union_s": union,
        "sweeps_per_s": calls / union,
        "lane_sweeps_per_s": calls * h * w / union,
        "ms_per_call_each": [p["ms_per_call"] for p in procs],
        "overlap": max(0, overlap_end - overlap_start) / (end - start),
        "external_creations": external,
        "ane_w": statistics.mean(s[1] for s in inside) if inside else None,
        "ram_w": statistics.mean(s[2] for s in inside) if inside else None,
    }
    summary.setdefault(f"{enc} {h}x{w}", {}).setdefault(n, []).append(launch)

print(
    f"{'config':16}{'n':>3}{'launches':>9}{'clean':>6}{'sweeps/s':>10}{'x n=1':>7}{'ms each':>16}{'overlap':>8}{'ANE W':>7}{'W/n=1':>7}"
)
out = {}
for config, by_n in summary.items():
    base = (1.0, 1.0)
    for n, ls in sorted(by_n.items()):
        clean = [x for x in ls if x["external_creations"] == 0] or ls
        rate = statistics.median(x["sweeps_per_s"] for x in clean)
        watts = statistics.median(x["ane_w"] for x in clean if x["ane_w"] is not None)
        if n == 1:
            base = (rate, watts)
        each = statistics.median(statistics.mean(x["ms_per_call_each"]) for x in clean)
        overlap = statistics.median(x["overlap"] for x in clean)
        print(
            f"{config:16}{n:3}{len(ls):9}{len([x for x in ls if x['external_creations'] == 0]):6}"
            f"{rate:10.0f}{rate / base[0]:7.2f}{each:16.4f}{overlap:8.2f}{watts:7.2f}{watts / base[1]:7.2f}"
        )
        out.setdefault(config, {})[n] = {
            "sweeps_per_s": rate,
            "gain": rate / base[0],
            "ms_per_call_each": each,
            "ane_w": watts,
            "launches": ls,
        }

if len(sys.argv) > 2:
    with open(sys.argv[2], "w") as f:
        json.dump(out, f, indent=1)
