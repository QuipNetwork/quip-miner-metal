#!/usr/bin/env python3
"""Does the winner's energy on a block predict MSA's energy on the same block?

    predict.py cohorts.json LABEL=csv[:reads] ...

Each argument names an effort level and the CSV holding it, as in `gap.py`.
For every block in each cohort the script pairs the energy that won the
block with MSA's best energy over all seeds at that effort level, then
reports Spearman and Pearson correlations and the MSA energy of the blocks
the winner ranked deepest against the ones it ranked shallowest. Spearman is
the figure that matters for a screen, which only has to order nonces.
"""

import csv
import json
import statistics as st
import sys
from collections import defaultdict


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def pearson(xs, ys):
    mx, my = st.fmean(xs), st.fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5


def spearman(xs, ys):
    return pearson(ranks(xs), ranks(ys))


def load(path, reads=None):
    best, winner = defaultdict(list), {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["error"] or (reads and int(r["reads"]) != reads):
                continue
            q = int(r["qblock_id"])
            best[q].append(int(r["best_milli"]))
            winner[q] = int(r["winner_milli"])
    return best, winner


def report(cohort, label, ids, best, winner):
    def med(qs):
        return int(st.median(min(best[q]) for q in qs))

    blocks = [q for q in ids if q in best]
    w = [winner[q] for q in blocks]
    b = [min(best[q]) for q in blocks]
    m = [st.fmean(best[q]) for q in blocks]
    by_winner = sorted(blocks, key=lambda q: winner[q])
    quarter = len(by_winner) // 4
    half = len(by_winner) // 2
    deepest = set(sorted(blocks, key=lambda q: min(best[q]))[:quarter])
    print(f"{cohort} cohort, {label}: {len(blocks)} blocks")
    print(
        f"  spearman(winner, msa best) {spearman(w, b):+.2f}"
        f"  spearman(winner, msa mean) {spearman(w, m):+.2f}"
        f"  pearson(winner, msa best) {pearson(w, b):+.2f}"
    )
    print(
        f"  msa best: median {med(blocks):,}  sd {st.pstdev(b):,.0f}"
        f"  range {min(b):,}..{max(b):,}"
    )
    print(
        f"  winner-deepest quarter -> msa median {med(by_winner[:quarter]):,},"
        f" shallowest quarter -> {med(by_winner[-quarter:]):,}"
        f" (difference {med(by_winner[-quarter:]) - med(by_winner[:quarter]):,})"
    )
    print(
        f"  winner-deepest half -> msa median {med(by_winner[:half]):,},"
        f" shallowest half -> {med(by_winner[half:]):,}"
        f" (difference {med(by_winner[half:]) - med(by_winner[:half]):,})"
    )
    print(
        f"  of msa's {quarter} deepest blocks:"
        f" {len(deepest & set(by_winner[:quarter]))} in the winner-deepest quarter,"
        f" {len(deepest & set(by_winner[:half]))} in the winner-deepest half"
    )


def main():
    with open(sys.argv[1]) as f:
        coh = json.load(f)
    for arg in sys.argv[2:]:
        label, spec = arg.split("=", 1)
        path, _, reads = spec.partition(":")
        best, winner = load(path, int(reads) if reads else None)
        for cohort in ("qpu", "control"):
            if any(q in best for q in coh[cohort]):
                report(cohort, label, coh[cohort], best, winner)


if __name__ == "__main__":
    main()
