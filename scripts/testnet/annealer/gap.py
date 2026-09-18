#!/usr/bin/env python3
"""How much lower than the annealer does MSA go on the annealer-won instances?

    gap.py cohorts.json LABEL=csv[:reads] ...

Each argument names an effort level and the CSV holding it. The optional
`:reads` filter picks one read count out of a multi-read CSV. For every
annealer-won qblock the script takes the best energy MSA reached at that
effort level, over all seeds, and subtracts it from the energy the annealer
put on chain.
"""

import csv
import itertools
import json
import statistics as st
import sys
from collections import defaultdict


def load(path, ids, reads=None):
    best, winner = defaultdict(list), {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["error"]:
                continue
            q = int(r["qblock_id"])
            if q not in ids or (reads and int(r["reads"]) != reads):
                continue
            best[q].append(int(r["best_milli"]))
            winner[q] = int(r["winner_milli"])
    return {q: min(v) for q, v in best.items()}, winner


def main():
    with open(sys.argv[1]) as f:
        coh = json.load(f)
    ids = set(coh["qpu"])
    levels = []
    for arg in sys.argv[2:]:
        label, spec = arg.split("=", 1)
        path, _, reads = spec.partition(":")
        best, winner = load(path, ids, int(reads) if reads else None)
        levels.append((label, best, winner, 0))

    print("## MSA against the annealer on the 60 annealer-won instances")
    print()
    print(
        "| MSA effort | Instances | Seeds each | MSA lower on | Median gap, milli | "
        "Mean gap, milli | Min gap | Max gap | Median gap, percent | Median MSA energy, milli |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    table = {}
    for label, best, winner, _ in levels:
        qs = sorted(best)
        gaps = sorted(winner[q] - best[q] for q in qs)
        pct = sorted(100 * (winner[q] - best[q]) / abs(winner[q]) for q in qs)
        table[label] = {q: best[q] for q in qs}
        print(
            f"| {label} | {len(qs)} | - | {sum(1 for g in gaps if g > 0)}/{len(qs)} "
            f"| {gaps[len(gaps) // 2]:,} | {st.mean(gaps):,.0f} | {gaps[0]:,} | {gaps[-1]:,} "
            f"| {pct[len(pct) // 2]:.3f} | {st.median(best.values()):,.0f} |"
        )

    if len(levels) > 1:
        print()
        print("## Improvement from one effort level to the next")
        print()
        print(
            "| From | To | Instances improved | Median improvement, milli | Max improvement, milli |"
        )
        print("| --- | --- | ---: | ---: | ---: |")
        for (la, ba, _, _), (lb, bb, _, _) in itertools.pairwise(levels):
            common = sorted(set(ba) & set(bb))
            imp = sorted(ba[q] - bb[q] for q in common)
            print(
                f"| {la} | {lb} | {sum(1 for x in imp if x > 0)}/{len(common)} "
                f"| {imp[len(imp) // 2]:,} | {imp[-1]:,} |"
            )

    label, best, winner, _ = levels[-1]
    qs = sorted(best)
    print()
    print(f"## Per instance at the highest effort level ({label})")
    print()
    print(
        "| Qblock | Annealer energy, milli | MSA best, milli | MSA lower by, milli | Percent |"
    )
    print("| ---: | ---: | ---: | ---: | ---: |")
    for q in qs:
        g = winner[q] - best[q]
        print(
            f"| {q} | {winner[q]:,} | {best[q]:,} | {g:,} | {100 * g / abs(winner[q]):.3f} |"
        )


if __name__ == "__main__":
    main()
