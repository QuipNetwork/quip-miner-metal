#!/usr/bin/env python3
"""Per-block energy delta between 8,192 and 16,384 sweeps.

    hard_delta.py OUT_DIR BASELINE_CSV

Uses `hard-8192.csv` and `hard-16384.csv` for the 500 hardest blocks and
`study-8192.csv` against the baseline CSV for the 60 recent blocks. The delta
is the mean over seeds of the best energy at 8,192 sweeps minus the same at
16,384 sweeps, in milli. A positive delta means 16,384 sweeps reached the
lower energy.
"""

import csv
import os
import statistics
import sys

OUT, BASELINE = sys.argv[1], sys.argv[2]
READS = 64


def rows(path, reads=READS, sweeps=None):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                raise SystemExit(f"error row in {path}: {r}")
            if int(r["reads"]) != reads:
                continue
            out.setdefault(r["qblock_id"], []).append(r)
    return out


def per_block(lo_path, hi_path):
    """{block: stats} for one pair of sweep counts."""
    lo, hi = rows(lo_path), rows(hi_path)
    common = sorted(set(lo) & set(hi), key=int)
    out = {}
    for b in common:
        lo_best = [int(r["best_milli"]) for r in lo[b]]
        hi_best = [int(r["best_milli"]) for r in hi[b]]
        lo_valid = [int(r["valid"]) for r in lo[b]]
        hi_valid = [int(r["valid"]) for r in hi[b]]
        out[b] = {
            "seeds_lo": len(lo_best),
            "seeds_hi": len(hi_best),
            "mean_best_lo": statistics.mean(lo_best),
            "mean_best_hi": statistics.mean(hi_best),
            "delta": statistics.mean(lo_best) - statistics.mean(hi_best),
            "p_valid_lo": sum(lo_valid) / len(lo_valid),
            "p_valid_hi": sum(hi_valid) / len(hi_valid),
            "only_hi": int(any(hi_valid) and not any(lo_valid)),
            "target_milli": int(lo[b][0]["target_milli"]),
            "winner_milli": int(lo[b][0]["winner_milli"]),
        }
    return out


def quantiles(xs):
    s = sorted(xs)
    return (
        s[0],
        s[int(0.25 * len(s))],
        statistics.median(s),
        s[int(0.75 * len(s))],
        s[-1],
    )


def describe(name, blocks):
    d = [b["delta"] for b in blocks.values()]
    q = quantiles(d)
    print(f"\n{name}: {len(blocks)} blocks")
    print(f"  delta mean {statistics.mean(d):,.0f} milli, median {q[2]:,.0f} milli")
    print(
        f"  quartiles: min {q[0]:,.0f}, Q1 {q[1]:,.0f}, Q2 {q[2]:,.0f}, "
        f"Q3 {q[3]:,.0f}, max {q[4]:,.0f}"
    )
    print(
        f"  blocks where 8,192 sweeps reached the lower energy: "
        f"{sum(1 for x in d if x < 0)} of {len(d)}"
    )
    lo = statistics.mean(b["p_valid_lo"] for b in blocks.values())
    hi = statistics.mean(b["p_valid_hi"] for b in blocks.values())
    only = sum(b["only_hi"] for b in blocks.values())
    print(f"  P(valid) {lo:.3f} at 8,192 and {hi:.3f} at 16,384")
    print(f"  blocks valid only at 16,384: {only} ({only / len(blocks):.3f})")
    return d


def quintiles(name, blocks, meta, key, unit):
    """Split the blocks into five equal groups by `key` and describe each."""
    have = [b for b in blocks if b in meta]
    have.sort(key=lambda b: meta[b][key])
    n = len(have)
    print(f"\n{name}, quintiles by {key} ({unit}), lowest first:")
    print(
        f"  {'quintile':>8} {'n':>4} {'range':>16} {'delta mean':>11} "
        f"{'delta med':>10} {'P(v) 8192':>10} {'P(v) 16384':>11} {'only 16384':>11}"
    )
    out = []
    for i in range(5):
        part = have[i * n // 5 : (i + 1) * n // 5]
        d = [blocks[b]["delta"] for b in part]
        lo = statistics.mean(blocks[b]["p_valid_lo"] for b in part)
        hi = statistics.mean(blocks[b]["p_valid_hi"] for b in part)
        only = sum(blocks[b]["only_hi"] for b in part) / len(part)
        rng = f"{meta[part[0]][key]:,} to {meta[part[-1]][key]:,}"
        print(
            f"  {i + 1:>8} {len(part):>4} {rng:>16} {statistics.mean(d):>11,.0f} "
            f"{statistics.median(d):>10,.0f} {lo:>10.3f} {hi:>11.3f} {only:>11.3f}"
        )
        out.append(
            (
                i + 1,
                len(part),
                rng,
                statistics.mean(d),
                statistics.median(d),
                lo,
                hi,
                only,
            )
        )
    return out


def main():
    meta = {}
    with open(os.path.join(OUT, "hard-500-meta.csv")) as f:
        for r in csv.DictReader(f):
            meta[r["qblock_id"]] = {
                "round_len": int(r["round_len"]),
                "winner_margin_milli": int(r["winner_margin_milli"]),
            }

    hard = per_block(
        os.path.join(OUT, "hard-8192.csv"), os.path.join(OUT, "hard-16384.csv")
    )
    recent = per_block(os.path.join(OUT, "study-8192.csv"), BASELINE)
    # The block's own difficulty target explains what round length selects.
    for b, s in hard.items():
        if b in meta:
            meta[b]["target_milli"] = s["target_milli"]

    describe("500 hardest blocks by round length, 64 reads, 3 seeds", hard)
    describe("60 recent blocks, 64 reads, 5 seeds", recent)
    q1 = quintiles("500 hardest blocks", hard, meta, "round_len", "chain blocks")
    q2 = quintiles("500 hardest blocks", hard, meta, "winner_margin_milli", "milli")
    quintiles("500 hardest blocks", hard, meta, "target_milli", "milli")

    path = os.path.join(OUT, "hard-per-block.csv")
    with open(path, "w", newline="") as f:
        cols = [
            "qblock_id",
            "round_len",
            "winner_margin_milli",
            "mean_best_8192",
            "mean_best_16384",
            "delta",
            "p_valid_8192",
            "p_valid_16384",
            "only_16384",
            "target_milli",
            "winner_milli",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for b, s in sorted(hard.items(), key=lambda kv: int(kv[0])):
            w.writerow(
                {
                    "qblock_id": b,
                    "round_len": meta[b]["round_len"],
                    "winner_margin_milli": meta[b]["winner_margin_milli"],
                    "mean_best_8192": round(s["mean_best_lo"]),
                    "mean_best_16384": round(s["mean_best_hi"]),
                    "delta": round(s["delta"]),
                    "p_valid_8192": s["p_valid_lo"],
                    "p_valid_16384": s["p_valid_hi"],
                    "only_16384": s["only_hi"],
                    "target_milli": s["target_milli"],
                    "winner_milli": s["winner_milli"],
                }
            )
    print(f"\n-> {path}")
    return q1, q2


if __name__ == "__main__":
    main()
