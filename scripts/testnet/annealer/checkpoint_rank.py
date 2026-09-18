#!/usr/bin/env python3
"""How well does one short MSA job rank instances against a full MSA job?

    checkpoint_rank.py FULL_CSV PROBE_CSV [PROBE_CSV ...]

The full CSV holds several seeds per block at the full sweep budget. Each
probe CSV holds jobs on the same blocks at a smaller budget, or on the same
fresh nonces. For every read count present in both, the script pairs one
probe job's energy with the full run's best over seeds, then reports the
Spearman correlation. This is the like-for-like figure for a screen that
runs one cheap job per nonce and keeps the nonces it ranks deepest.
"""

import csv
import statistics as st
import sys
from collections import defaultdict

from predict import spearman


def load(path):
    rows = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r["error"]:
                rows[(int(r["reads"]), int(r["qblock_id"]))].append(
                    int(r["best_milli"])
                )
    return rows


def main():
    full = load(sys.argv[1])
    for path in sys.argv[2:]:
        probe = load(path)
        for reads in sorted({r for r, _ in full} & {r for r, _ in probe}):
            keys = sorted(k for k in full if k[0] == reads and k in probe)
            one = [probe[k][0] for k in keys]
            best = [min(full[k]) for k in keys]
            mean = [st.fmean(full[k]) for k in keys]
            probe_mean = [st.fmean(probe[k]) for k in keys]
            print(
                f"{path}: {reads} reads, {len(keys)} blocks,"
                f" {len(probe[keys[0]])} probe seed(s), {len(full[keys[0]])} full seed(s):"
                f" spearman(one probe, full best) {spearman(one, best):+.2f},"
                f" spearman(probe mean, full mean) {spearman(probe_mean, mean):+.2f}"
            )


if __name__ == "__main__":
    main()
