#!/usr/bin/env python3
"""Measure GPU jobs per second at several read and sweep counts.

    run_bench.py BENCH_BIN OUT.jsonl [ROUNDS]

Each configuration keeps 160 replica-words of work: 80 jobs at 64 reads and
40 jobs at 40 reads. Every value goes into the child environment explicitly,
so the zsh word-splitting trap in `docs/perf/2026-09-18-gpu-reads-and-models.md`
cannot apply. The harness's own report line is parsed back and compared with
the requested reads and sweeps; a mismatch aborts the run.
"""

import json
import os
import re
import subprocess
import sys
import time

BIN, OUT = sys.argv[1], sys.argv[2]
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 2

# (reads, jobs): jobs x reads stays at 160 replica-words.
SHAPES = [(64, 80), (128, 40)]
# 1024 is beyond the set the task named. It costs about one second per run and
# it is the only way the 1024-sweep chance per job turns into proofs per second.
SWEEPS = [1024, 2048, 4096, 8192, 16384]

REPORT = re.compile(
    r"^msa: (\d+) jobs x (\d+) reads x (\d+) sweeps in ([\d.]+) s = ([\d.]+) jobs/s;"
    r" best (-?\d+) milli, mean best (-?\d+)"
)


def one(reads, jobs, sweeps, round_index, out):
    env = dict(os.environ)
    env["QUIP_BENCH_JOBS"] = str(jobs)
    env["QUIP_BENCH_READS"] = str(reads)
    env["QUIP_BENCH_SWEEPS"] = str(sweeps)
    env["QUIP_BENCH_KERNEL"] = "msa"
    cmd = [BIN, "--ignored", "--nocapture"]
    shown = (
        f"QUIP_BENCH_JOBS={jobs} QUIP_BENCH_READS={reads} "
        f"QUIP_BENCH_SWEEPS={sweeps} QUIP_BENCH_KERNEL=msa {BIN} --ignored --nocapture"
    )
    t0 = time.monotonic()
    r = subprocess.run(cmd, env=env, capture_output=True, timeout=1800, check=False)
    wall = time.monotonic() - t0
    text = r.stderr.decode() + r.stdout.decode()
    line = next((m for m in (REPORT.match(x) for x in text.splitlines()) if m), None)
    if r.returncode != 0 or line is None:
        row = {
            "round": round_index,
            "reads": reads,
            "jobs": jobs,
            "sweeps": sweeps,
            "command": shown,
            "error": text[-600:],
            "returncode": r.returncode,
        }
        out.write(json.dumps(row) + "\n")
        out.flush()
        raise SystemExit(f"bench failed: reads={reads} sweeps={sweeps}\n{text[-600:]}")
    got_jobs, got_reads, got_sweeps = (int(line.group(i)) for i in (1, 2, 3))
    assert got_reads == reads, f"harness ran {got_reads} reads, asked {reads}"
    assert got_sweeps == sweeps, f"harness ran {got_sweeps} sweeps, asked {sweeps}"
    assert got_jobs == jobs, f"harness ran {got_jobs} jobs, asked {jobs}"
    row = {
        "round": round_index,
        "reads": reads,
        "jobs": got_jobs,
        "sweeps": got_sweeps,
        "stream_s": float(line.group(4)),
        "jobs_per_s": float(line.group(5)),
        "best_milli": int(line.group(6)),
        "mean_best_milli": int(line.group(7)),
        "process_wall_s": round(wall, 3),
        "command": shown,
        "report_line": line.group(0),
    }
    out.write(json.dumps(row) + "\n")
    out.flush()
    print(
        f"round {round_index}: {reads} reads, {sweeps} sweeps -> "
        f"{row['jobs_per_s']:.2f} jobs/s ({wall:.1f} s wall)",
        flush=True,
    )


def main():
    t0 = time.monotonic()
    with open(OUT, "w") as out:
        for round_index in range(1, ROUNDS + 1):
            sweeps = SWEEPS if round_index % 2 else list(reversed(SWEEPS))
            for s in sweeps:
                for reads, jobs in SHAPES:
                    one(reads, jobs, s, round_index, out)
    print(f"done in {time.monotonic() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
