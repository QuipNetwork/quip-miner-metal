#!/usr/bin/env python3
"""Does a stronger solver agree with the probe's ranking?

    screen_deep.py select SCREEN_CSV SEEDS_OUT MEMBERS_OUT [--count N] [--seed S]
    screen_deep.py report MEMBERS DEEP_CSV [--target MILLI]

`select` takes the joint CSV from `tests/probe_screen.rs` and picks three
subsets of `count` nonces: the deepest by the 1,024-sweep probe, the
deepest by the full job, and a uniform random control. It writes the
distinct seeds, one per line, for a deep run of the harness
(`QUIP_SCREEN_SEEDS=SEEDS_OUT QUIP_SCREEN_STAGES=256x65536`), and a
members CSV that records which subsets each seed belongs to.

`report` joins the deep run's CSV back by seed and reports, per subset,
the median energy at the full budget and at the deep budget, how far the
deep job moves each nonce, and the rank correlation between the two within
the control, where the range is not restricted. The members CSV carries
each nonce's probe and full energy, so `report` needs no other input.
"""

import argparse
import csv
import gzip
import random
import statistics as st

from predict import spearman

PROBE = "64x1024"
TARGET = -14_625_068
SUBSETS = ("probe", "full", "random")


def opened(path):
    """Read a CSV, gzipped or plain. The committed tables are gzipped."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def load(path):
    with opened(path) as f:
        return list(csv.DictReader(f))


def full_tag(rows):
    tags = [c[len("best_") :] for c in rows[0] if c.startswith("best_")]
    return max(tags, key=lambda t: int(t.split("x")[1]))


def select(args):
    rows = [r for r in load(args.csv) if r[f"best_{PROBE}"]]
    full = full_tag(rows)
    by_probe = sorted(rows, key=lambda r: int(r[f"best_{PROBE}"]))[: args.count]
    by_full = sorted(rows, key=lambda r: int(r[f"best_{full}"]))[: args.count]
    chosen = {r["seed"] for r in by_probe} | {r["seed"] for r in by_full}
    rng = random.Random(args.seed)
    control = rng.sample([r for r in rows if r["seed"] not in chosen], args.count)
    members = {}
    for label, subset in (("probe", by_probe), ("full", by_full), ("random", control)):
        for r in subset:
            members.setdefault(r["seed"], set()).add(label)
    with open(args.seeds_out, "w") as f:
        f.writelines(f"{seed}\n" for seed in members)
    energies = {r["seed"]: (r[f"best_{PROBE}"], r[f"best_{full}"]) for r in rows}
    with open(args.members_out, "w") as f:
        w = csv.writer(f)
        w.writerow(["seed", *SUBSETS, "probe_milli", "full_milli"])
        for seed, labels in members.items():
            w.writerow([seed, *(int(s in labels) for s in SUBSETS), *energies[seed]])
    overlap = len({r["seed"] for r in by_probe} & {r["seed"] for r in by_full})
    print(
        f"{len(members)} seeds: {args.count} deepest by probe, {args.count} deepest"
        f" by full ({overlap} in both), {args.count} random"
    )


def report(args):
    members = load(args.members)
    deep_rows = load(args.deep)
    deep_tag = full_tag(deep_rows)
    deep = {
        r["seed"]: int(r[f"best_{deep_tag}"])
        for r in deep_rows
        if r[f"best_{deep_tag}"]
    }
    print(f"deep stage {deep_tag} on {len(deep)} nonces; target {args.target:,}")
    for label in SUBSETS:
        seeds = [m["seed"] for m in members if m[label] == "1" and m["seed"] in deep]
        by_seed = {m["seed"]: m for m in members}
        probe_e = [int(by_seed[s]["probe_milli"]) for s in seeds]
        full_e = [int(by_seed[s]["full_milli"]) for s in seeds]
        deep_e = [deep[s] for s in seeds]
        gain = [d - f for d, f in zip(deep_e, full_e, strict=True)]
        print(
            f"  {label:>6}: n {len(seeds)}  median probe {int(st.median(probe_e)):,}"
            f"  full {int(st.median(full_e)):,}  deep {int(st.median(deep_e)):,}"
            f"  deep minus full median {int(st.median(gain)):,}"
            f" (min {min(gain):,}, max {max(gain):,})"
            f"  deep below full on {sum(g < 0 for g in gain)}"
            f"  valid {sum(d < args.target for d in deep_e)}"
        )
        print(
            f"          spearman within the subset: probe vs deep {spearman(probe_e, deep_e):+.2f},"
            f" full vs deep {spearman(full_e, deep_e):+.2f}"
        )
    pool = [
        m["seed"]
        for m in members
        if (m["probe"] == "1" or m["random"] == "1") and m["seed"] in deep
    ]
    probe_set = {m["seed"] for m in members if m["probe"] == "1"}
    top = sorted(pool, key=deep.__getitem__)[:20]
    print(
        f"  deepest 20 by the deep job among probe-picked and random nonces:"
        f" {sum(s in probe_set for s in top)} of 20 were probe-picked"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("select")
    s.add_argument("csv")
    s.add_argument("seeds_out")
    s.add_argument("members_out")
    s.add_argument("--count", type=int, default=100)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(run=select)
    r = sub.add_parser("report")
    r.add_argument("members")
    r.add_argument("deep")
    r.add_argument("--target", type=int, default=TARGET)
    r.set_defaults(run=report)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
