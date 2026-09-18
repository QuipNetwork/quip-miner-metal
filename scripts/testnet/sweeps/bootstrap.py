#!/usr/bin/env python3
"""Paired block bootstrap on the ratio of chance per job to the baseline.

    bootstrap.py OUT_DIR BASELINE_CSV

The same 60 blocks carry every cell, so the resample draws blocks, not jobs,
and every cell uses the drawn blocks. That pairing is what makes the interval
on the ratio narrower than two separate intervals would suggest.
"""

import csv
import json
import math
import os
import random
import statistics
import sys

OUT, BASELINE = sys.argv[1], sys.argv[2]
DRAWS = 10000
BASE = (128, 16384)
random.seed(20260918)


def load():
    """{(reads, sweeps): {block: [(valid, beats), ...]}}"""
    cells = {}
    paths = [(s, os.path.join(OUT, f"study-{s}.csv")) for s in (1024, 2048, 4096, 8192)]
    paths.append((16384, BASELINE))
    for sweeps, path in paths:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            reads = int(r["reads"])
            if reads not in (64, 128):
                continue
            cells.setdefault((reads, sweeps), {}).setdefault(r["qblock_id"], []).append(
                (int(r["valid"]), int(r["beats_winner"]))
            )
    return cells


def rate(cell, blocks, field):
    hit = tot = 0
    for b in blocks:
        for row in cell[b]:
            hit += row[field]
            tot += 1
    return hit / tot if tot else 0.0


def main():
    cells = load()
    rates = {}
    with open(os.path.join(OUT, "bench.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            rates.setdefault((row["reads"], row["sweeps"]), []).append(
                row["jobs_per_s"]
            )
    speed = {k: max(v) for k, v in rates.items()}
    blocks = sorted(cells[BASE])
    out = {}
    for cell in sorted(cells):
        if cell == BASE:
            continue
        ratios = {0: [], 1: []}
        for _ in range(DRAWS):
            draw = [random.choice(blocks) for _ in blocks]
            for field in (0, 1):
                den = rate(cells[BASE], draw, field)
                num = rate(cells[cell], draw, field)
                ratios[field].append(
                    (num * speed[cell]) / (den * speed[BASE]) if den else float("nan")
                )
        row = {}
        for field, name in ((0, "valid"), (1, "beats")):
            vals = sorted(v for v in ratios[field] if not math.isnan(v))
            row[name] = (
                statistics.median(vals),
                vals[int(0.05 * len(vals))],
                vals[int(0.95 * len(vals))],
            )
        out[cell] = row
        print(
            f"{cell[0]:>3} reads, {cell[1]:>5} sweeps vs baseline: "
            f"valid/s x{row['valid'][0]:.2f} [{row['valid'][1]:.2f}, {row['valid'][2]:.2f}]  "
            f"winner-beating/s x{row['beats'][0]:.2f} "
            f"[{row['beats'][1]:.2f}, {row['beats'][2]:.2f}]"
        )
    with open(os.path.join(OUT, "bootstrap.json"), "w") as f:
        json.dump({f"{k[0]}r-{k[1]}s": v for k, v in out.items()}, f, indent=1)
    print("\n90% intervals from 10,000 paired block resamples, best-round rates.")


if __name__ == "__main__":
    main()
