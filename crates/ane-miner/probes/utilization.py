#!/usr/bin/env python3
"""Summarise utilization.sh output: rates against the measured ceilings, and
power per configuration.

    crates/ane-miner/probes/utilization.py OUT_DIR [summary.json]

OUT_DIR holds results.jsonl from the probe, power.jsonl from macmon, and
aned.log from the system log. Each result's wall-clock window selects the
power samples taken inside it, after dropping the first sample, whose
interval straddles the window's start. Program creations by aned inside a
window, or in the five seconds before it, belong to other clients, because
the probe compiles before its loop starts. A client that started earlier
shows as ANE power above zero in the idle gap before the window, and GPU
work on the host shows as GPU power inside it. A window with any of the
three is contaminated. Each configuration's figure is the fastest clean
round, and the spread over clean rounds is reported beside it.
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


results = load(f"{OUT}/results.jsonl")
power = [
    (epoch_us(s["timestamp"]), s["ane_power"], s["ram_power"], s["gpu_power"])
    for s in load(f"{OUT}/power.jsonl")
]
GAP_ANE_W = 0.05
GPU_W = 1.0
creations = []
try:
    with open(f"{OUT}/aned.log") as f:
        for line in f:
            m = CREATION.match(line)
            if m:
                creations.append(epoch_us(m.group(1)))
except FileNotFoundError:
    pass
first_window = min(r["wall_start_us"] for r in results if "calls" in r)
idle = [s for s in power if s[0] < first_window - SAMPLE_MS * 1000]
idle_ram = statistics.mean(s[2] for s in idle) if idle else 0.0


def window_power(r):
    inside = [
        s
        for s in power
        if r["wall_start_us"] + SAMPLE_MS * 1000 <= s[0] <= r["wall_end_us"]
    ]
    if not inside:
        return {}
    return {
        "samples": len(inside),
        "ane_w": statistics.mean(s[1] for s in inside),
        "ram_w": statistics.mean(s[2] for s in inside),
        "gpu_w": statistics.mean(s[3] for s in inside),
    }


def key(r):
    return (r["kind"], r["name"], r["encoding"], r["h"], r["w"])


by_config = {}
timed_results = sorted(
    (r for r in results if "calls" in r), key=lambda r: r["wall_start_us"]
)
previous_end = None
for r in timed_results:
    # A program created up to LEAD_US before the window runs inside it.
    r["external_creations"] = sum(
        1 for t in creations if r["wall_start_us"] - LEAD_US <= t <= r["wall_end_us"]
    )
    gap = [
        s
        for s in power
        if previous_end is not None
        and previous_end + SAMPLE_MS * 1000 <= s[0] <= r["wall_start_us"] - 100_000
    ]
    r["gap_ane_w"] = statistics.mean(s[1] for s in gap) if gap else 0.0
    r["gpu_w"] = window_power(r).get("gpu_w", 0.0)
    r["clean"] = (
        r["external_creations"] == 0
        and r["gap_ane_w"] < GAP_ANE_W
        and r["gpu_w"] < GPU_W
    )
    previous_end = r["wall_end_us"]
for r in results:
    by_config.setdefault(key(r), []).append(r)

rows = []
for rounds in by_config.values():
    timed = [r for r in rounds if "calls" in r]
    if not timed:
        rows.append({**rounds[0], "status": "rejected"})
        continue
    clean = [r for r in timed if r["clean"]] or timed
    best = min(clean, key=lambda r: r["ms_per_call"])
    seconds = best["ms_per_call"] / 1000
    p = window_power(best)
    row = {
        **best,
        "rounds": len(timed),
        "clean_rounds": len([r for r in timed if r["clean"]]),
        "ms_per_call_median": statistics.median(r["ms_per_call"] for r in clean),
        "ms_per_call_max": max(r["ms_per_call"] for r in clean),
        "tmac_s": best["mac_per_call"] / seconds / 1e12,
        "weight_gb_s": best["weight_bytes"] / seconds / 1e9,
        "mac_per_weight_byte": best["mac_per_call"] / best["weight_bytes"],
        **p,
    }
    if p:
        row["mj_per_call"] = p["ane_w"] * best["ms_per_call"]
        row["mj_per_call_with_ram"] = (p["ane_w"] + p["ram_w"] - idle_ram) * best[
            "ms_per_call"
        ]
    rows.append(row)

compute = [
    x
    for x in rows
    if x["kind"] == "roofline" and x["name"].startswith("compute") and "tmac_s" in x
]
stream = [
    x
    for x in rows
    if x["kind"] == "roofline" and x["name"].startswith("stream") and "tmac_s" in x
]
peak_tmac = max(x["tmac_s"] for x in compute)
peak_stream = max(x["weight_gb_s"] for x in stream)
for x in rows:
    if "tmac_s" in x:
        x["mac_utilization"] = x["tmac_s"] / peak_tmac
        x["stream_utilization"] = x["weight_gb_s"] / peak_stream

idle_summary = {
    "samples": len(idle),
    "ane_w": statistics.mean(s[1] for s in idle) if idle else None,
    "ram_w": statistics.mean(s[2] for s in idle) if idle else None,
}

print(f"idle: {idle_summary}")
print(f"ceilings: {peak_tmac:.2f} TMAC/s compute, {peak_stream:.1f} GB/s weight stream")
print(f"external aned program creations in the log: {len(creations)}")
print()
print(
    f"{'kind':9}{'name':24}{'enc':7}{'HxW':7}{'ms/call':>9}{'median':>8}{'clean':>6}{'TMAC/s':>8}{'MAC%':>6}{'GB/s':>7}{'str%':>6}{'ANE W':>7}{'RAM W':>7}{'mJ/call':>9}{'hash':>18}"
)
for x in sorted(
    rows,
    key=lambda x: (x["kind"] != "roofline", x["encoding"], x["h"] * x["w"], x["h"]),
):
    if "tmac_s" not in x:
        print(
            f"{x['kind']:9}{x['name']:24}{x['encoding']:7}{x['h']}x{x['w']:<5} rejected"
        )
        continue
    print(
        f"{x['kind']:9}{x['name']:24}{x['encoding']:7}{x['h']}x{x['w']:<5}"
        f"{x['ms_per_call']:9.4f}{x['ms_per_call_median']:8.4f}{x['clean_rounds']:3}/{x['rounds']:<2}"
        f"{x['tmac_s']:8.2f}{100 * x['mac_utilization']:6.1f}"
        f"{x['weight_gb_s']:7.1f}{100 * x['stream_utilization']:6.1f}"
        f"{x.get('ane_w', float('nan')):7.2f}{x.get('ram_w', float('nan')):7.2f}"
        f"{x.get('mj_per_call', float('nan')):9.3f}{x['output_hash']:>18}"
    )

if len(sys.argv) > 2:
    with open(sys.argv[2], "w") as f:
        json.dump(
            {
                "idle": idle_summary,
                "peak_tmac_s": peak_tmac,
                "peak_weight_gb_s": peak_stream,
                "rows": rows,
            },
            f,
            indent=1,
        )
