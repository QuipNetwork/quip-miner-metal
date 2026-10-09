# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

LOG = """\
2026-09-22T10:00:00.000000Z DEBUG quip_solver_metal::streaming: prepared batch batch=40 cap=40 chunks=1 seed=Blocking
2026-09-22T10:00:00.010000Z DEBUG quip_solver_metal::streaming: batch complete chunks=1 max_chunk_ms=9 total_ms=9
2026-09-22T10:00:00.011000Z DEBUG quip_solver_metal::streaming: prepared batch batch=40 cap=40 chunks=1 seed=NonBlocking
2026-09-22T10:00:00.020000Z DEBUG quip_solver_metal::streaming: batch complete chunks=1 max_chunk_ms=8 total_ms=8
2026-09-22T10:00:00.021000Z DEBUG quip_solver_metal::streaming: prepared batch batch=40 cap=40 chunks=1 seed=NonBlocking
2026-09-22T10:00:00.030000Z DEBUG quip_solver_metal::streaming: batch complete chunks=1 max_chunk_ms=8 total_ms=8
"""


def test_parse_reports_steady_rate():
    out = subprocess.run(
        [sys.executable, HERE / "parse.py"],
        input=LOG,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    got = json.loads(out)
    assert got["batches"] == 3
    assert got["batch_size"] == 40
    # 80 jobs after the first batch, over 0.020 s between first and last completion.
    assert abs(got["steady_jobs_per_s"] - 4000.0) < 1.0
    assert got["gpu_ms_per_batch_median"] == 8


def test_compare_energy_passes_and_fails(tmp_path):
    base = tmp_path / "a.csv"
    same = tmp_path / "b.csv"
    moved = tmp_path / "c.csv"
    rows = "\n".join(f"{i},x,{-1000 - (i % 10)},0,0" for i in range(1000))
    header = "nonce,seed,best_64x32,median_64x32,below_64x32\n"
    base.write_text(header + rows + "\n")
    same.write_text(header + rows + "\n")
    shifted = "\n".join(f"{i},x,{-1010 - (i % 10)},0,0" for i in range(1000))
    moved.write_text(header + shifted + "\n")
    ok = subprocess.run(
        [sys.executable, HERE / "compare_energy.py", base, same, "best_64x32"],
        check=False,
    )
    bad = subprocess.run(
        [sys.executable, HERE / "compare_energy.py", base, moved, "best_64x32"],
        check=False,
    )
    assert ok.returncode == 0
    assert bad.returncode == 1
