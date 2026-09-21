#!/usr/bin/env python3
"""What a probe screen yields at scale, from one joint CSV.

    screen_yield.py SCREEN_CSV [RUN_LOG] [--target MILLI]

The CSV comes from `tests/probe_screen.rs`: one row per fresh nonce with a
`best_READSxSWEEPS` column per stage. The stage with the most sweeps is the
full job; every other stage is a probe. The run log, when given, supplies
each stage's measured jobs per second for the equal-compute arithmetic.

For each probe the script reports the rank correlation with the full job,
then for tail thresholds on the full energy (the deepest 5%, 1%, 0.1% and
the deepest ten) the recall inside the deepest fraction by probe, and the
yield of a screened miner relative to the unscreened default at equal
compute: recall x t_full / (t_probe + keep x t_full).
"""

import argparse
import csv
import gzip
import re
import statistics as st

from predict import spearman

KEEP = (2, 4, 8, 16, 32, 64, 128)
TAILS = ((0.05, "deepest 5%"), (0.01, "deepest 1%"), (0.001, "deepest 0.1%"))
DEEPEST = 10
TARGET = -14_625_068


def opened(path):
    """Read a CSV, gzipped or plain. The committed tables are gzipped."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def load(path):
    with opened(path) as f:
        reader = csv.DictReader(f)
        tags = [
            c[len("best_") :] for c in reader.fieldnames or () if c.startswith("best_")
        ]
        rows = [r for r in reader if all(r[f"best_{t}"] for t in tags)]
    best = {t: [int(r[f"best_{t}"]) for r in rows] for t in tags}
    return tags, best


def rates(path):
    """Measured jobs per second per stage tag, from the harness's stderr."""
    out = {}
    if path is None:
        return out
    pat = re.compile(r"stage (\d+x\d+): \d+ of \d+ jobs in [\d.]+ s = ([\d.]+) jobs/s")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


def sweeps(tag):
    return int(tag.split("x")[1])


def deepest_set(xs, count):
    """Indices of the `count` lowest values."""
    return set(sorted(range(len(xs)), key=xs.__getitem__)[:count])


def distribution(full, target):
    n = len(full)
    sd = st.pstdev(full)
    median = st.median(full)
    ordered = sorted(full)
    print(f"{n:,} nonces at the full budget, target {target:,}")
    print(
        f"  median {int(median):,}  sd {sd:,.0f}  p5 {ordered[int(0.05 * n)]:,}"
        f"  p1 {ordered[int(0.01 * n)]:,}  p0.1 {ordered[int(0.001 * n)]:,}"
        f"  min {ordered[0]:,}  max {ordered[-1]:,}"
    )
    print(
        f"  valid {sum(x < target for x in full)}/{n};"
        f" target is {(median - target) / sd:.1f} sd below the median;"
        f" the deepest nonce is {(median - ordered[0]) / sd:.1f} sd below it"
    )


def probe_report(tag, probe, full, keep_rates, full_rate, target):
    n = len(full)
    print(f"\nprobe {tag}: spearman with the full job {spearman(probe, full):+.3f}")
    thresholds = [(int(q * n), label) for q, label in TAILS if int(q * n) > 0]
    thresholds.append((DEEPEST, f"deepest {DEEPEST}"))
    header = "  keep 1 in".ljust(24) + "".join(f"{k:>12}" for k in KEEP)
    print(header)
    print("  median full energy of the kept nonces, milli")
    print(
        "  ".ljust(24)
        + "".join(
            f"{int(st.median([full[i] for i in deepest_set(probe, n // k)])):>12,}"
            for k in KEEP
        )
    )
    for count, label in thresholds:
        deep = deepest_set(full, count)
        recalls = [len(deep & deepest_set(probe, n // k)) / count for k in KEEP]
        print(f"  recall, {label}".ljust(24) + "".join(f"{r:>12.2f}" for r in recalls))
        if keep_rates:
            t_probe, t_full = 1 / keep_rates, 1 / full_rate
            yields = [
                r * t_full / (t_probe + t_full / k)
                for r, k in zip(recalls, KEEP, strict=True)
            ]
            print(
                f"  yield, {label}".ljust(24) + "".join(f"{y:>11.2f}x" for y in yields)
            )
    deep = sorted(deepest_set(full, DEEPEST), key=full.__getitem__)
    order = sorted(range(n), key=probe.__getitem__)
    rank = {i: r for r, i in enumerate(order)}
    print(
        f"  probe rank of the deepest {DEEPEST} by full energy, 1 is deepest: "
        + " ".join(f"{rank[i] + 1:,}" for i in deep)
    )
    margin(probe, full, order, target)


def margin(probe, full, order, target):
    """Where a target-reaching nonce would read on the probe.

    A straight line of probe energy on full energy, fitted on every nonce
    and again on the deepest tenth by full, gives the probe reading a nonce
    at the target would show and the scatter of that reading among nonces
    of one full energy. Each keep fraction's probe cutoff then sits some
    number of residual standard deviations above that reading: the margin
    by which the screen keeps a valid nonce.
    """
    n = len(full)
    by_full = sorted(range(n), key=full.__getitem__)
    fits = [("all nonces", range(n)), ("deepest tenth by full", by_full[: n // 10])]
    for label, idx in fits:
        xs = [full[i] for i in idx]
        ys = [probe[i] for i in idx]
        slope, intercept = st.linear_regression(xs, ys)
        resid = st.pstdev(
            [y - (intercept + slope * x) for x, y in zip(xs, ys, strict=True)]
        )
        at_target = intercept + slope * target
        print(
            f"  fit on {label}: probe = {intercept:,.0f} + {slope:.3f} x full,"
            f" residual sd {resid:,.0f}; a nonce at the target reads {at_target:,.0f} on the probe"
        )
        cutoffs = [probe[order[n // k - 1]] for k in KEEP]
        print(
            "  keep cutoff margin, sd".ljust(24)
            + "".join(f"{(c - at_target) / resid:>12.1f}" for c in cutoffs)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("log", nargs="?")
    parser.add_argument("--target", type=int, default=TARGET)
    args = parser.parse_args()
    tags, best = load(args.csv)
    measured = rates(args.log)
    full_tag = max(tags, key=sweeps)
    full = best[full_tag]
    distribution(full, args.target)
    if measured:
        print(
            "  measured jobs/s: "
            + ", ".join(f"{t} {r:.1f}" for t, r in measured.items())
        )
    for tag in tags:
        if tag != full_tag:
            probe_report(
                tag,
                best[tag],
                full,
                measured.get(tag),
                measured.get(full_tag),
                args.target,
            )


if __name__ == "__main__":
    main()
