# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors
"""Synthetic checks for the calibration command."""

import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

SCRIPT = Path(__file__).with_name("cascade_calibrate.py")


def write_csv(path, columns, seeds=None):
    count = len(next(iter(columns.values())))
    if seeds is None:
        seeds = [f"{i:064x}" for i in range(count)]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["nonce", "seed", *columns])
        for i in range(count):
            writer.writerow([i, seeds[i], *(values[i] for values in columns.values())])


def run_report(tmp_path, *extra):
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--unfiltered",
            str(tmp_path / "a.csv"),
            "--filtered",
            str(tmp_path / "b.csv"),
            "--out",
            str(output),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(output.read_text())


def test_gaussian_report(tmp_path):
    rng = np.random.default_rng(20260922)
    latent = rng.normal(size=20_000)
    probe = 0.8 * latent + 0.6 * rng.normal(size=20_000)
    middle = 0.9 * latent + np.sqrt(0.19) * rng.normal(size=20_000)
    columns = {"best_64x32": probe, "best_64x256": middle, "best_64x14336": latent}
    write_csv(tmp_path / "a.csv", columns)
    write_csv(tmp_path / "b.csv", {"best_64x32": probe, "best_64x256": middle})
    report = run_report(tmp_path)
    expected = np.corrcoef(
        np.argsort(np.argsort(probe)), np.argsort(np.argsort(latent))
    )[0, 1]
    assert report["stages"] == [32, 256]
    assert abs(report["spearman_vs_full"]["32"] - expected) < 0.02
    assert abs(report["moments"]["32"]["skew"]) < 0.05
    assert 0 <= report["false_negative_rate"]["32->256"]["1000"] <= 1
    kept = np.argsort(probe)[:20]
    rejected = np.argsort(probe)[20:]
    expected_fn = np.mean(middle[rejected] <= np.median(middle[kept]))
    assert report["false_negative_rate"]["32->256"]["1000"] == pytest.approx(
        expected_fn
    )
    for key in ("1e-3", "1e-4", "1e-5"):
        expected_dev = (
            np.quantile(probe, float(key)) - probe.mean()
        ) / probe.std() - NormalDist().inv_cdf(float(key))
        assert report["normal_quantile_dev_sd"][key] == pytest.approx(expected_dev)
    assert report["tail_r_deepest_500"] is None
    assert report["cost_model_us"] == {"a": None, "b": None}


def test_sparse_rows_deep_seed_join_and_cost_units(tmp_path):
    write_csv(
        tmp_path / "a.csv",
        {
            "best_64x32": [0, 1, 2, 3, 4, 5],
            "best_64x256": [3, 0, 1, 4, 2, 5],
            "best_64x14336": [3, 0, 1, 4, 2, 5],
        },
    )
    write_csv(
        tmp_path / "b.csv",
        {
            "best_64x32": [0, 1, 2, 3, 4, 5],
            "best_64x256": [3, "", 1, "", 2, ""],
        },
    )
    # Deep CSV nonce numbers restart at zero. Only seeds identify paired jobs.
    write_csv(
        tmp_path / "c.csv",
        {"best_256x65536": [20, 30, 10]},
        [f"{i:064x}" for i in (4, 0, 2)],
    )
    log = tmp_path / "study.log"
    log.write_text(
        "stage 64x32: 6 of 6 jobs in 0.0 s = 1 jobs/s; lead 0 s; wall_seconds=0.000252\n"
        "stage 64x256: 3 of 3 jobs in 0.0 s = 1 jobs/s; lead 0 s; wall_seconds=0.000798\n"
    )
    report = run_report(tmp_path, "--deep", str(tmp_path / "c.csv"), "--log", str(log))
    assert report["tail_r_deepest_500"] == pytest.approx(1)
    assert report["cost_model_us"] == pytest.approx({"a": 10, "b": 1})
    assert report["false_negative_rate"]["32->256"]["1000"] == pytest.approx(3 / 5)


def test_degenerate_data_is_strict_json(tmp_path):
    columns = {"best_64x32": [1, 1], "best_64x256": ["", ""], "best_64x14336": [2, 2]}
    write_csv(tmp_path / "a.csv", columns)
    write_csv(tmp_path / "b.csv", columns)
    report = run_report(tmp_path)
    assert report["spearman_vs_full"]["32"] is None
    assert report["moments"]["32"]["sd"] == 0
    assert report["moments"]["32"]["skew"] is None
    assert report["moments"]["256"]["mean"] is None
    assert all(value is None for value in report["normal_quantile_dev_sd"].values())
    assert "NaN" not in (tmp_path / "report.json").read_text()


def test_tail_uses_lowest_500_final_probes(tmp_path):
    values = np.arange(600, dtype=float)
    columns = {f"best_64x{sweeps}": values for sweeps in (32, 64, 128, 256)}
    write_csv(tmp_path / "a.csv", {**columns, "best_64x14336": values})
    write_csv(tmp_path / "b.csv", columns)
    deep = values * 2
    deep[500:] = -values[500:] * 100
    write_csv(tmp_path / "c.csv", {"best_64x14336": deep})
    report = run_report(tmp_path, "--deep", str(tmp_path / "c.csv"))
    assert report["stages"] == [32, 64, 128, 256]
    assert report["tail_r_deepest_500"] == pytest.approx(1)
    assert set(report["false_negative_rate"]) == {
        "32->64",
        "32->128",
        "32->256",
        "64->128",
        "64->256",
        "128->256",
    }


@pytest.mark.parametrize(
    "bad_input", ["duplicate_seeds", "nonfinite", "incomplete_log"]
)
def test_invalid_inputs_do_not_write_report(tmp_path, bad_input):
    columns = {"best_64x32": [1, 2], "best_64x256": [2, 3], "best_64x14336": [3, 4]}
    write_csv(tmp_path / "a.csv", columns)
    seeds = ["same", "same"] if bad_input == "duplicate_seeds" else None
    if bad_input == "nonfinite":
        columns["best_64x32"] = [1, float("inf")]
    write_csv(tmp_path / "b.csv", columns, seeds)
    extra = []
    if bad_input == "incomplete_log":
        log = tmp_path / "study.log"
        log.write_text("stage 64x32: 2/2 after 1 s\n")
        extra = ["--log", str(log)]
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--unfiltered",
            str(tmp_path / "a.csv"),
            "--filtered",
            str(tmp_path / "b.csv"),
            "--out",
            str(output),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert not output.exists()
