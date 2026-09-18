#!/usr/bin/env python3
"""Run every regenerated testnet problem at several read counts.

    scripts/testnet/run_reads_study.py target/release/quip-metal-msa problems/ study.csv

`problems/` is the output directory of `scripts/testnet/regen`. Each job
goes through the miner's `--solve` mode, one process per job, `WIDTH` at a
time. The CSV records each job's best energy against the block's target
(a valid proof needs strictly lower) and against the energy that won.

Knobs: `READS` (default `32,64,128,256`), `SEEDS` per block and read count
(default 5), `SWEEPS` (default 16384), `WIDTH` (default 8). Rows are written
as they finish, so a partial run still has usable data.
"""

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

BIN, PROBLEMS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
READS = [int(x) for x in os.environ.get("READS", "32,64,128,256").split(",")]
SEEDS = int(os.environ.get("SEEDS", "5"))
SWEEPS = int(os.environ.get("SWEEPS", "16384"))
WIDTH = int(os.environ.get("WIDTH", "8"))
FIELDS = [
    "qblock_id",
    "reads",
    "k",
    "seed",
    "best_milli",
    "median_milli",
    "target_milli",
    "winner_milli",
    "reads_below_target",
    "valid",
    "beats_winner",
    "wall_s",
    "error",
]

with open(os.path.join(PROBLEMS, "index.json")) as f:
    INDEX = json.load(f)
CACHE = {}


def problem(path):
    if path not in CACHE:
        with open(path) as f:
            CACHE[path] = json.load(f)
    return CACHE[path]


def run(job):
    entry, reads, k = job
    tag = f"{entry['qblock_id']}:{reads}:{k}".encode()
    seed = int.from_bytes(hashlib.blake2b(tag, digest_size=8).digest(), "little")
    p = dict(problem(entry["path"]))
    p.update(
        {"num_reads": reads, "num_sweeps": SWEEPS, "sweeps_per_beta": 1, "seed": seed}
    )
    t0 = time.monotonic()
    r = subprocess.run(
        [BIN, "--solve"],
        input=json.dumps(p).encode(),
        capture_output=True,
        timeout=600,
        check=False,
    )
    wall = time.monotonic() - t0
    if r.returncode != 0:
        return {
            "qblock_id": entry["qblock_id"],
            "reads": reads,
            "k": k,
            "error": r.stderr.decode()[-200:],
        }
    energies = sorted(s["energy_milli"] for s in json.loads(r.stdout))
    assert len(energies) == reads
    target, winner = entry["target_energy_milli"], entry["winning_energy_milli"]
    return {
        "qblock_id": entry["qblock_id"],
        "reads": reads,
        "k": k,
        "seed": seed,
        "best_milli": energies[0],
        "median_milli": energies[len(energies) // 2],
        "target_milli": target,
        "winner_milli": winner,
        "reads_below_target": sum(1 for e in energies if e < target),
        "valid": int(energies[0] < target),
        "beats_winner": int(energies[0] <= winner),
        "wall_s": round(wall, 3),
    }


def main():
    jobs = [(e, r, k) for e in INDEX for r in READS for k in range(SEEDS)]
    # Interleave read counts so drift in machine state falls on all of them alike.
    jobs.sort(key=lambda j: (j[2], j[0]["qblock_id"], j[1]))
    done = 0
    t0 = time.monotonic()
    with open(OUT, "w", newline="") as f, ThreadPoolExecutor(WIDTH) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in pool.map(run, jobs):
            w.writerow(row)
            f.flush()
            done += 1
            if done % 50 == 0:
                print(
                    f"{done}/{len(jobs)} jobs, {time.monotonic() - t0:.0f} s",
                    flush=True,
                )
    print(f"done: {done} jobs in {time.monotonic() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
