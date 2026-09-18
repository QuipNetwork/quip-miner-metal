#!/usr/bin/env python3
"""What a cheap probe buys on fresh nonces: the distribution and the screen.

    fresh_screen.py FULL_CSV PROBE_CSV [PROBE_CSV ...]

The full CSV holds two or more seeds per fresh nonce at the full budget. Each
probe CSV holds one job per nonce at a smaller budget, with its own seed.
The script reports the fresh-nonce distribution of MSA's energy against the
target, how much of a job's energy is the instance rather than the seed, the
probe's rank correlation with the full result, and the direct effect of a
screen that keeps only the nonces the probe ranks deepest.
"""

import csv
import statistics as st
import sys
from collections import defaultdict

from predict import spearman

KEEP = (2, 4, 8, 16)
TAIL = 0.05


def load(path):
    rows, target = defaultdict(list), None
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r["error"]:
                rows[int(r["qblock_id"])].append(int(r["best_milli"]))
                target = int(r["target_milli"])
    return rows, target


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    full, target = load(sys.argv[1])
    nonces = sorted(n for n, v in full.items() if len(v) >= 2)
    one = [full[n][0] for n in nonces]
    two = [full[n][1] for n in nonces]
    best = [min(full[n]) for n in nonces]
    print(f"{len(nonces)} fresh nonces, target {target:,}")
    for label, xs in (("one full job", one), ("best of two full jobs", best)):
        sd = st.pstdev(xs)
        print(
            f"  {label}: median {int(st.median(xs)):,}  sd {sd:,.0f}"
            f"  p5 {quantile(xs, TAIL):,}  min {min(xs):,}  max {max(xs):,}"
            f"  valid {sum(x < target for x in xs)}/{len(xs)}"
            f"  target is {(st.median(xs) - target) / sd:.1f} sd below the median"
        )
    cov = st.covariance(one, two)
    var = st.pvariance(one + two)
    print(
        f"  two full jobs on the same nonce: spearman {spearman(one, two):+.2f},"
        f" instance share of variance {cov / var:.2f}"
        f" (instance sd {max(cov, 0) ** 0.5:,.0f}, seed sd {max(var - cov, 0) ** 0.5:,.0f})"
    )
    tail = set(sorted(nonces, key=lambda n: min(full[n]))[: int(TAIL * len(nonces))])
    for path in sys.argv[2:]:
        probe, _ = load(path)
        p = [probe[n][0] for n in nonces]
        print(
            f"{path}: spearman(probe, one full job) {spearman(p, one):+.2f},"
            f" spearman(probe, best of two) {spearman(p, best):+.2f}"
        )
        by_probe = sorted(nonces, key=lambda n: probe[n][0])
        for keep in KEEP:
            kept = by_probe[: len(nonces) // keep]
            kept_best = [min(full[n]) for n in kept]
            hit = len(tail & set(kept))
            print(
                f"  keep deepest 1/{keep}: {len(kept)} nonces,"
                f" full-job median {int(st.median(kept_best)):,} vs {int(st.median(best)):,} for all,"
                f" holds {hit}/{len(tail)} of the deepest {TAIL:.0%}"
                f" ({hit / len(tail):.0%}; no signal would give {1 / keep:.0%})"
            )


if __name__ == "__main__":
    main()
