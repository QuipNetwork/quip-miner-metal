# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors
"""Calibrate short probes from unfiltered, filtered, and optional deep CSVs.

Stage moments and false negatives use the unfiltered arm. Normal quantiles
use the filtered arm's first stage, before selection biases its population.
Tail r is Pearson correlation of the final probe and the deepest reference,
paired by seed on the lowest 500 final probes. Missing or undefined findings
are JSON null. Moments use population SD and standardized central moments.
The timing model is microseconds per completed job: a + b * sweeps.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def load(path: Path) -> tuple[list[str], dict[str, Array]]:
    """Preserve empty stage cells as NaN internally, never as zero energies."""
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = {
            name: [] for name in reader.fieldnames or [] if name.startswith("best_")
        }
        if not columns or "seed" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: expected seed and best_READSxSWEEPS columns")
        seeds = []
        for row in reader:
            seeds.append(row["seed"])
            for name, values in columns.items():
                cell = row[name]
                value = float(cell) if cell else math.nan
                if cell and not math.isfinite(value):
                    raise ValueError(f"{path}: nonfinite energy in {name}")
                values.append(value)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path}: duplicate seeds")
    return seeds, {
        name: np.asarray(values, dtype=float) for name, values in columns.items()
    }


def shape(column: str) -> tuple[int, int]:
    reads, sweeps = column.removeprefix("best_").split("x")
    return int(reads), int(sweeps)


def correlation(x: Array, y: Array, *, ranks: bool = False) -> float | None:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    if ranks:
        return float(
            np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]
        )
    return float(np.corrcoef(x, y)[0, 1])


def moments(values: Array) -> dict[str, float | None]:
    values = values[np.isfinite(values)]
    if not values.size:
        return dict.fromkeys(("mean", "sd", "skew", "excess_kurtosis"))
    mean, sd = float(values.mean()), float(values.std())
    if sd == 0:
        return {"mean": mean, "sd": sd, "skew": None, "excess_kurtosis": None}
    z = (values - mean) / sd
    return {
        "mean": mean,
        "sd": sd,
        "skew": float(np.mean(z**3)),
        "excess_kurtosis": float(np.mean(z**4) - 3),
    }


def false_negatives(source: Array, target: Array) -> dict[str, float | None]:
    valid = np.isfinite(source) & np.isfinite(target)
    source, target = source[valid], target[valid]
    order = np.argsort(source, kind="stable")
    result = {}
    for denominator in (1000, 10000, 30000):
        count = math.ceil(source.size / denominator)
        kept, rejected = target[order[:count]], target[order[count:]]
        result[str(denominator)] = (
            float(np.mean(rejected <= np.median(kept)))
            if kept.size and rejected.size
            else None
        )
    return result


def cost_model(path: Path | None, stages: list[str]) -> dict[str, float | None]:
    if path is None:
        return {"a": None, "b": None}
    observations = []
    pattern = re.compile(
        r"stage (\d+)x(\d+): (\d+) of \d+ jobs.*wall_seconds=([\d.eE+-]+)"
    )
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            reads, sweeps, jobs, seconds = match.groups()
            if f"best_{reads}x{sweeps}" in stages and int(jobs) > 0:
                wall = float(seconds)
                if not math.isfinite(wall) or wall < 0:
                    raise ValueError("wall_seconds must be finite and nonnegative")
                observations.append((int(sweeps), wall * 1e6 / int(jobs)))
    if len({s for s, _ in observations}) < 2:
        raise ValueError(
            "--log needs completed timings for at least two probe sweep budgets"
        )
    sweeps, costs = np.asarray(observations, dtype=float).T
    a, b = np.linalg.lstsq(
        np.column_stack((np.ones_like(sweeps), sweeps)), costs, rcond=None
    )[0]
    return {"a": float(a), "b": float(b)}


def calibrate(
    unfiltered: Path, filtered: Path, deep: Path | None, log: Path | None
) -> dict:
    _, a = load(unfiltered)
    b_seeds, b = load(filtered)
    full = max(a, key=lambda name: shape(name)[1])
    stages = sorted(
        (name for name in a if name != full), key=lambda name: shape(name)[1]
    )
    if not stages or any(name not in b for name in stages):
        raise ValueError(
            "both arms need matching probe columns and an unfiltered full reference"
        )
    if len({shape(name)[1] for name in stages}) != len(stages):
        raise ValueError("probe sweep budgets must be unique")
    if len({shape(name)[0] for name in stages}) != 1:
        raise ValueError("probe read counts must match for the sweep cost model")
    stage_moments = {str(shape(name)[1]): moments(a[name]) for name in stages}
    spearman = {
        str(shape(name)[1]): correlation(a[name], a[full], ranks=True)
        for name in stages
    }
    normal = b[stages[0]]
    normal = normal[np.isfinite(normal)]
    normal_moments = moments(normal)
    mean, sd = normal_moments["mean"], normal_moments["sd"]
    deviations = {
        key: float(
            (
                np.quantile(normal, float(key))
                - (mean + NormalDist().inv_cdf(float(key)) * sd)
            )
            / sd
        )
        if mean is not None and sd is not None and sd > 0
        else None
        for key in ("1e-3", "1e-4", "1e-5")
    }
    tail = None
    if deep is not None:
        c_seeds, c = load(deep)
        reference = c[max(c, key=lambda name: shape(name)[1])]
        by_seed = dict(zip(c_seeds, reference, strict=True))
        final = b[stages[-1]]
        selected = np.flatnonzero(np.isfinite(final))
        selected = selected[np.argsort(final[selected], kind="stable")[:500]]
        tail = correlation(
            final[selected],
            np.asarray([by_seed.get(b_seeds[i], math.nan) for i in selected]),
        )
    rates = {
        f"{shape(source)[1]}->{shape(target)[1]}": false_negatives(a[source], a[target])
        for i, source in enumerate(stages)
        for target in stages[i + 1 :]
    }
    return {
        "stages": [shape(name)[1] for name in stages],
        "spearman_vs_full": spearman,
        "tail_r_deepest_500": tail,
        "normal_quantile_dev_sd": deviations,
        "moments": stage_moments,
        "false_negative_rate": rates,
        "cost_model_us": cost_model(log, stages),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unfiltered", type=Path, required=True)
    parser.add_argument("--filtered", type=Path, required=True)
    parser.add_argument("--deep", type=Path)
    parser.add_argument(
        "--log", type=Path, help="study stderr with wall_seconds= stage timings"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = calibrate(args.unfiltered, args.filtered, args.deep, args.log)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for finding, value in report.items():
        print(f"{finding}: {json.dumps(value, allow_nan=False)}")


if __name__ == "__main__":
    main()
