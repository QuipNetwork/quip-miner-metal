#!/usr/bin/env python3
"""Combine the chance per job with the measured jobs per second.

    aggregate.py OUT_DIR BASELINE_CSV

Reads `study-<sweeps>.csv` for the low sweep counts, the 16,384-sweep rows
from `BASELINE_CSV`, the earlier reads study's output, and the throughput rows from `bench.jsonl`.
Writes `summary.csv` and prints the table.

Throughput carries two estimators. The machine ran a desktop load throughout,
and interference only slows a run down, so the best round of a configuration
is the closest estimate of the GPU's rate and the median is a downward-biased
floor. Proofs per second appear under both.
"""

import csv
import json
import os
import statistics
import sys

OUT, BASELINE = sys.argv[1], sys.argv[2]
LOW_SWEEPS = [1024, 2048, 4096, 8192]
READS = [64, 128]
MIN_SOLUTIONS = 5


def load_rows():
    """Return job rows keyed by (reads, sweeps), only 64 and 128 reads."""
    by_cell = {}
    for s in LOW_SWEEPS:
        path = os.path.join(OUT, f"study-{s}.csv")
        if not os.path.exists(path):
            print(f"missing {path}, skipping", file=sys.stderr)
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("error"):
                    raise SystemExit(f"error row in {path}: {r}")
                by_cell.setdefault((int(r["reads"]), s), []).append(r)
    with open(BASELINE) as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                raise SystemExit(f"error row in {BASELINE}: {r}")
            if int(r["reads"]) in READS:
                by_cell.setdefault((int(r["reads"]), 16384), []).append(r)
    return by_cell


def load_bench():
    """Return {(reads, sweeps): [jobs/s, ...]} from the harness rows."""
    rates = {}
    path = os.path.join(OUT, "bench.jsonl")
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "error" in row:
                raise SystemExit(f"error row in {path}: {row}")
            rates.setdefault((row["reads"], row["sweeps"]), []).append(
                row["jobs_per_s"]
            )
    return rates


def clustered_se(by_block, pick):
    """Standard error of a per-job rate, clustering on the block."""
    per_block = [sum(pick(r) for r in v) / len(v) for v in by_block.values()]
    if len(per_block) < 2:
        return 0.0
    return statistics.stdev(per_block) / len(per_block) ** 0.5


def cell_stats(rows):
    """Per-cell probabilities, energies, and block-clustered standard errors."""
    n = len(rows)
    bests = [int(r["best_milli"]) for r in rows]
    gaps = [int(r["best_milli"]) - int(r["winner_milli"]) for r in rows]
    rbt_valid = [int(r["reads_below_target"]) for r in rows if int(r["valid"])]
    by_block = {}
    for r in rows:
        by_block.setdefault(r["qblock_id"], []).append(r)
    valid_by_block = {k: [int(r["valid"]) for r in v] for k, v in by_block.items()}
    return {
        "jobs": n,
        "p_valid": sum(int(r["valid"]) for r in rows) / n,
        "p_valid_se": clustered_se(by_block, lambda r: int(r["valid"])),
        "p_beats": sum(int(r["beats_winner"]) for r in rows) / n,
        "p_beats_se": clustered_se(by_block, lambda r: int(r["beats_winner"])),
        # Same proof under a difficulty that asks for five reads below target.
        "p_valid_min5": sum(
            1 for r in rows if int(r["reads_below_target"]) >= MIN_SOLUTIONS
        )
        / n,
        "mean_rbt_valid": (sum(rbt_valid) / len(rbt_valid)) if rbt_valid else 0.0,
        "median_best_milli": statistics.median(bests),
        "mean_gap_milli": sum(gaps) / n,
        "blocks_all5": sum(1 for v in valid_by_block.values() if sum(v) == len(v)),
        "blocks_none": sum(1 for v in valid_by_block.values() if sum(v) == 0),
        "blocks": len(by_block),
        "mean_solve_wall_s": sum(float(r["wall_s"]) for r in rows) / n,
    }


FIELDS = [
    "reads",
    "sweeps",
    "jobs",
    "p_valid",
    "p_valid_se",
    "p_beats",
    "p_beats_se",
    "p_valid_min5",
    "mean_rbt_valid",
    "median_best_milli",
    "mean_gap_milli",
    "blocks_all5",
    "blocks_none",
    "blocks",
    "mean_solve_wall_s",
    "bench_rounds",
    "jobs_per_s_best",
    "jobs_per_s_median",
    "jobs_per_s_min",
    "valid_per_s_best",
    "beats_per_s_best",
    "valid_min5_per_s_best",
    "valid_per_s_median",
    "beats_per_s_median",
]


def main():
    cells, rates = load_rows(), load_bench()
    out_rows = []
    for (reads, sweeps), rows in sorted(cells.items()):
        st = cell_stats(rows)
        st["reads"], st["sweeps"] = reads, sweeps
        rs = rates.get((reads, sweeps), [])
        if not rs:
            raise SystemExit(f"no bench rows for reads={reads} sweeps={sweeps}")
        best, med = max(rs), statistics.median(rs)
        st["bench_rounds"] = len(rs)
        st["jobs_per_s_best"], st["jobs_per_s_median"] = best, med
        st["jobs_per_s_min"] = min(rs)
        st["valid_per_s_best"] = best * st["p_valid"]
        st["beats_per_s_best"] = best * st["p_beats"]
        st["valid_min5_per_s_best"] = best * st["p_valid_min5"]
        st["valid_per_s_median"] = med * st["p_valid"]
        st["beats_per_s_median"] = med * st["p_beats"]
        out_rows.append(st)

    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    print(
        f"{'reads':>5} {'sweeps':>6} {'P(valid)':>16} {'P(beat)':>15} {'P>=5':>6} "
        f"{'rbt':>5} {'medbest':>10} {'gap':>7} {'a5':>3} {'n0':>3} "
        f"{'jobs/s':>13} {'valid/s':>12} {'beat/s':>12}"
    )
    for r in out_rows:
        print(
            f"{r['reads']:>5} {r['sweeps']:>6} "
            f"{r['p_valid']:>9.3f}+/-{r['p_valid_se']:<5.3f} "
            f"{r['p_beats']:>8.3f}+/-{r['p_beats_se']:<5.3f} "
            f"{r['p_valid_min5']:>6.3f} {r['mean_rbt_valid']:>5.2f} "
            f"{r['median_best_milli']:>10.0f} {r['mean_gap_milli']:>7.0f} "
            f"{r['blocks_all5']:>3} {r['blocks_none']:>3} "
            f"{r['jobs_per_s_best']:>6.2f}/{r['jobs_per_s_median']:<6.2f} "
            f"{r['valid_per_s_best']:>5.2f}/{r['valid_per_s_median']:<6.2f} "
            f"{r['beats_per_s_best']:>5.2f}/{r['beats_per_s_median']:<6.2f}"
        )
    base = next(r for r in out_rows if r["reads"] == 128 and r["sweeps"] == 16384)
    print("\nagainst the 128-read, 16,384-sweep baseline (best / median rates):")
    for r in out_rows:
        if r["reads"] != 64:
            continue
        vb = r["valid_per_s_best"] / base["valid_per_s_best"]
        vm = r["valid_per_s_median"] / base["valid_per_s_median"]
        bb = (
            r["beats_per_s_best"] / base["beats_per_s_best"]
            if base["beats_per_s_best"]
            else 0
        )
        bm = (
            r["beats_per_s_median"] / base["beats_per_s_median"]
            if base["beats_per_s_median"]
            else 0
        )
        print(
            f"  64 reads, {r['sweeps']:>5} sweeps: "
            f"valid/s x{vb:.2f}/x{vm:.2f}, winner-beating/s x{bb:.2f}/x{bm:.2f}"
        )
    print(f"\n-> {os.path.join(OUT, 'summary.csv')}")


if __name__ == "__main__":
    main()
