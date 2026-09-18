#!/usr/bin/env python3
"""Compare solver outcomes on QPU-won qblocks against matched classical-won ones.

analyze.py study.csv cohorts.json qblocks-all.json
"""

import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    return pearson(rank(xs), rank(ys))


def main():
    study, cohort_path, chain_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(cohort_path) as f:
        coh = json.load(f)
    qpu_ids, ctl_ids = set(coh["qpu"]), set(coh["control"])
    with open(chain_path) as f:
        chain = {b["qblock_id"]: b for b in json.load(f)["blocks"]}

    rows = []
    with open(study) as f:
        for r in csv.DictReader(f):
            if r["error"]:
                print(
                    f"ERROR row: qblock {r['qblock_id']} reads {r['reads']}: {r['error']}"
                )
                continue
            rows.append(
                {
                    "qid": int(r["qblock_id"]),
                    "reads": int(r["reads"]),
                    "best": int(r["best_milli"]),
                    "target": int(r["target_milli"]),
                    "winner": int(r["winner_milli"]),
                    "below": int(r["reads_below_target"]),
                    "valid": int(r["valid"]),
                    "beats": int(r["beats_winner"]),
                    "wall": float(r["wall_s"]),
                }
            )
    print(f"rows: {len(rows)}")
    reads_levels = sorted({r["reads"] for r in rows})

    def cohort(qid):
        return "qpu" if qid in qpu_ids else ("control" if qid in ctl_ids else "?")

    print()
    print("## Table 1: outcome by cohort and read count")
    print()
    print(
        "| Cohort | Reads | Jobs | P(valid) | P(beats winner) | Median best energy, milli "
        "| Mean gap to winner, milli | Mean winner margin, milli | Valid on 5/5 | Valid on 0/5 |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    summary = {}
    for name in ("qpu", "control"):
        for reads in reads_levels:
            g = [r for r in rows if cohort(r["qid"]) == name and r["reads"] == reads]
            if not g:
                continue
            per = defaultdict(list)
            for r in g:
                per[r["qid"]].append(r)
            all5 = sum(1 for v in per.values() if all(x["valid"] for x in v))
            none = sum(1 for v in per.values() if not any(x["valid"] for x in v))
            pv = st.mean(r["valid"] for r in g)
            pb = st.mean(r["beats"] for r in g)
            med = st.median(r["best"] for r in g)
            gap = st.mean(r["best"] - r["winner"] for r in g)
            marg = st.mean(r["target"] - r["winner"] for r in g)
            summary[(name, reads)] = {
                "n": len(g),
                "blocks": len(per),
                "pv": pv,
                "pb": pb,
                "med": med,
                "gap": gap,
                "marg": marg,
                "all5": all5,
                "none": none,
            }
            print(
                f"| {name} | {reads} | {len(g)} | {pv:.2f} | {pb:.2f} | {med:,.0f} "
                f"| {gap:,.0f} | {marg:,.0f} | {all5}/{len(per)} | {none}/{len(per)} |"
            )

    print()
    print("## Table 2: per-block valid count distribution (clustering check)")
    print()
    print(
        "| Cohort | Reads | 0/5 | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | Mean P(valid) "
        "| Blocks at 0 or 5 | Binomial expectation at 0 or 5 |"
    )
    print(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for name in ("qpu", "control"):
        for reads in reads_levels:
            per = defaultdict(list)
            for r in rows:
                if cohort(r["qid"]) == name and r["reads"] == reads:
                    per[r["qid"]].append(r["valid"])
            if not per:
                continue
            counts = [sum(v) for v in per.values()]
            hist = [counts.count(i) for i in range(6)]
            p = sum(counts) / (5 * len(counts))
            exp_extreme = len(counts) * ((1 - p) ** 5 + p**5)
            print(
                f"| {name} | {reads} | "
                + " | ".join(str(h) for h in hist)
                + f" | {p:.2f} | {hist[0] + hist[5]} | {exp_extreme:.1f} |"
            )

    print()
    print("## Table 3: does the winner's margin below target predict reachability?")
    print()
    for reads in reads_levels:
        for name in ("qpu", "control", "both"):
            per = defaultdict(list)
            for r in rows:
                c = cohort(r["qid"])
                if r["reads"] == reads and (name == "both" or c == name):
                    per[r["qid"]].append(r)
            if not per:
                continue
            qids = sorted(per)
            frac = [st.mean(x["valid"] for x in per[q]) for q in qids]
            marg = [per[q][0]["target"] - per[q][0]["winner"] for q in qids]
            wgap = [st.mean(x["best"] - x["winner"] for x in per[q]) for q in qids]
            print(f"reads={reads} cohort={name} blocks={len(qids)}")
            print(
                f"  pearson(winner margin, our valid fraction)  = {pearson(marg, frac):+.3f}"
            )
            print(
                f"  spearman(winner margin, our valid fraction) = {spearman(marg, frac):+.3f}"
            )
            print(
                f"  spearman(winner margin, our gap to winner)  = {spearman(marg, wgap):+.3f}"
            )
            med_m = st.median(marg)
            tab = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
            for m, fr in zip(marg, frac):
                tab[(int(m > med_m), int(fr == 1.0))] += 1
            print(
                f"  2x2 at margin median {med_m:,.0f} milli, rows = winner margin, "
                "cols = we are valid on 5/5 seeds"
            )
            print(f"    narrow margin: 5/5 {tab[(0, 1)]:>3}   not 5/5 {tab[(0, 0)]:>3}")
            print(f"    wide   margin: 5/5 {tab[(1, 1)]:>3}   not 5/5 {tab[(1, 0)]:>3}")
            a, b, c2, dd = tab[(1, 1)], tab[(1, 0)], tab[(0, 1)], tab[(0, 0)]
            if b and c2 and a and dd:
                print(f"    odds ratio = {(a * dd) / (b * c2):.2f}")
            print()

    print("## Table 4: reads below target per job (min_solutions headroom)")
    print()
    print("| Cohort | Reads | Mean reads below target | Mean over valid jobs |")
    print("| --- | ---: | ---: | ---: |")
    for name in ("qpu", "control"):
        for reads in reads_levels:
            g = [r for r in rows if cohort(r["qid"]) == name and r["reads"] == reads]
            if not g:
                continue
            v = [r["below"] for r in g if r["valid"]]
            print(
                f"| {name} | {reads} | {st.mean(r['below'] for r in g):.2f} "
                f"| {st.mean(v) if v else float('nan'):.2f} |"
            )

    print()
    print("## Chain-side cohort check")
    for name, ids in (("qpu", qpu_ids), ("control", ctl_ids)):
        g = [chain[i] for i in sorted(ids)]
        marg = sorted(
            b["difficulty"]["max_energy_milli"] - b["energy_milli"] for b in g
        )
        t = sorted(b["device_access_time_us"] for b in g)
        e = sorted(b["energy_milli"] for b in g)
        print(
            f"  {name}: n={len(g)} ids {min(ids)}..{max(ids)} "
            f"access_time_us med {t[len(t) // 2]:,} ({t[0]:,}..{t[-1]:,})"
        )
        print(
            f"     winner margin below target: mean {st.mean(marg):,.0f} med {marg[len(marg) // 2]:,} "
            f"({marg[0]:,}..{marg[-1]:,})"
        )
        print(f"     winning energy: med {e[len(e) // 2]:,} best {e[0]:,}")
        print(f"     distinct miners: {len({b['miner'] for b in g})}")

    print()
    print("## Wall clock")
    for reads in reads_levels:
        g = [r["wall"] for r in rows if r["reads"] == reads]
        print(
            f"  reads={reads}: {len(g)} jobs, mean {st.mean(g):.2f} s, median {st.median(g):.2f} s"
        )


if __name__ == "__main__":
    main()
