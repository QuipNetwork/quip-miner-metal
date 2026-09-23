#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors
"""Steady-state rate from a quip_miner_metal debug log on stdin.

The rate counts the jobs of every batch after the first, over the time from
the first batch completion to the last. That removes device open, kernel
compile, and the first batch's lead time, which the harness headline counts.
"""
import json
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STAMP = re.compile(r"T(\d+):(\d+):([\d.]+)Z")


def seconds(line):
    h, m, s = STAMP.search(line).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def main():
    lines = [ANSI.sub("", line) for line in sys.stdin]
    sizes = [int(re.search(r"batch=(\d+)", l).group(1)) for l in lines if "prepared batch" in l]
    done = [(seconds(l), int(re.search(r"total_ms=(\d+)", l).group(1)))
            for l in lines if "batch complete" in l]
    if len(done) < 3:
        print(json.dumps({"error": f"need at least 3 completed batches, got {len(done)}"}))
        return 1
    span = done[-1][0] - done[0][0]
    jobs = sum(sizes[1:len(done)])
    gpu = sorted(ms for _, ms in done)
    median = gpu[len(gpu) // 2]
    print(json.dumps({
        "batches": len(done),
        "batch_size": sizes[0],
        "steady_jobs_per_s": jobs / span,
        "gpu_ms_per_batch_median": median,
        "gpu_ms_per_job": median / sizes[0],
        "wall_ms_per_batch": 1000.0 * span / (len(done) - 1),
        "gpu_ms_per_batch_mean": statistics.fmean(ms for _, ms in done),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
