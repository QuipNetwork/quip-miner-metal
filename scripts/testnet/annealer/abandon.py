#!/usr/bin/env python3
"""Does the energy at a checkpoint predict validity at the full budget, and
what does abandoning the poor jobs at that checkpoint buy?

    abandon.py CHECKPOINT_SWEEPS CHECKPOINT_CSV FULL_CSV [READS ...]

Both CSVs come from run_reads_study.py on the same blocks. For each read
count the script reports the Spearman rank correlation between the mean
best energy above target at the checkpoint and the valid fraction at the
full budget, then for every threshold the share of blocks kept, the share
of the full budget's valid outcomes captured, and the valid outcomes per
unit of compute when every job runs to the checkpoint and only the kept
ones continue, against running every job to the full budget.
"""

import csv
import statistics
import sys


def rows(path):
    with open(path) as f:
        return [r for r in csv.DictReader(f) if not r.get("error")]


def spearman(xs, ys):
    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0] * len(values)
        for rank_value, index in enumerate(order):
            ranks[index] = rank_value
        return ranks

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / (sxx * syy) ** 0.5


def main():
    checkpoint = int(sys.argv[1])
    check_rows, full_rows = rows(sys.argv[2]), rows(sys.argv[3])
    reads_list = [int(x) for x in sys.argv[4:]] or [64, 128]
    full_sweeps = (
        int(full_rows[0]["num_sweeps"]) if "num_sweeps" in full_rows[0] else 16384
    )
    fraction = checkpoint / full_sweeps
    for reads in reads_list:
        at_check, at_full, target = {}, {}, {}
        for r in check_rows:
            if int(r["reads"]) == reads:
                at_check.setdefault(r["qblock_id"], []).append(int(r["best_milli"]))
                target[r["qblock_id"]] = int(r["target_milli"])
        for r in full_rows:
            if int(r["reads"]) == reads:
                at_full.setdefault(r["qblock_id"], []).append(int(r["best_milli"]))
        blocks = [q for q in at_check if q in at_full]
        n = len(blocks)
        gap = [statistics.mean(at_check[q]) - target[q] for q in blocks]
        valid = [
            sum(1 for v in at_full[q] if v < target[q]) / len(at_full[q])
            for q in blocks
        ]
        base = sum(valid)
        print(
            f"checkpoint {checkpoint:,} of {full_sweeps:,} sweeps, {reads} reads, {n} blocks: "
            f"Spearman(gap above target at checkpoint, valid fraction at full budget) = {spearman(gap, valid):+.3f}"
        )
        print(f"  baseline: {base / n:.3f} valid outcomes per full-budget job")
        best = None
        for threshold in range(2000, 40001, 2000):
            keep = [i for i in range(n) if gap[i] < threshold]
            captured = sum(valid[i] for i in keep)
            units = n * fraction + len(keep) * (1 - fraction)
            per_unit = captured / units
            gain = per_unit / (base / n)
            if best is None or gain > best[0]:
                best = (gain, threshold, len(keep), captured / base)
            print(
                f"  continue if within {threshold:>6,} of target: keep {len(keep):3}/{n}, "
                f"captures {100 * captured / base:3.0f}% of valid outcomes, {gain:.2f}x valid per unit of compute"
            )
        print(
            f"  best: threshold {best[1]:,}, keep {best[2]}/{n}, captures {100 * best[3]:.0f}%, {best[0]:.2f}x"
        )


if __name__ == "__main__":
    main()
