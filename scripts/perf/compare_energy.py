#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors
"""Compare one energy column of two probe_screen CSVs.

Exit 0 when both the mean and the standard deviation of the candidate are
within 0.05 baseline standard deviations of the baseline, else exit 1.
"""

import csv
import statistics
import sys

TOLERANCE_SD = 0.05


def column(path, name):
    with open(path, newline="") as f:
        return [float(row[name]) for row in csv.DictReader(f) if row[name] != ""]


def main():
    base_path, cand_path, name = sys.argv[1:4]
    base = column(base_path, name)
    cand = column(cand_path, name)
    bm, bs = statistics.fmean(base), statistics.stdev(base)
    cm, cs = statistics.fmean(cand), statistics.stdev(cand)
    mean_shift = abs(cm - bm) / bs
    sd_shift = abs(cs - bs) / bs
    print(
        f"{name}: baseline mean {bm:.1f} sd {bs:.1f} (n={len(base)}); "
        f"candidate mean {cm:.1f} sd {cs:.1f} (n={len(cand)}); "
        f"mean shift {mean_shift:.3f} sd, sd shift {sd_shift:.3f} sd"
    )
    return 0 if mean_shift <= TOLERANCE_SD and sd_shift <= TOLERANCE_SD else 1


if __name__ == "__main__":
    sys.exit(main())
