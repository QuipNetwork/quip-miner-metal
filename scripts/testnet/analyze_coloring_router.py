#!/usr/bin/env python3
"""Compare block-matched production-channel receipts, with a block-cluster bootstrap.

The miner draws independent OS seeds. Replicate IDs match attempts, not RNG
streams. Rate intervals hold observed wall durations fixed and therefore do not
include timing noise. Repeat counterbalanced windows before a speed claim.

python3 scripts/testnet/analyze_coloring_router.py control.jsonl candidate.jsonl
"""

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def validate(rows):
    """Reject incomplete runs and return scored, non-warmup jobs and summary."""
    summaries = [row for row in rows if row["kind"] == "summary"]
    if len(summaries) != 1:
        raise ValueError("receipt needs exactly one completed summary")
    summary = summaries[0]
    if summary["color"] not in {"greedy", "four", "mixed"}:
        raise ValueError("unknown coloring arm")
    jobs = [row for row in rows if row["kind"] == "job" and not row["warmup"]]
    if not jobs or summary["jobs"] != len(jobs) or summary["completed"] != len(jobs):
        raise ValueError("receipt count does not match completed jobs")
    if not summary["rescore_ok"] or any(not row["rescore_ok"] for row in jobs):
        raise ValueError("receipt contains an unverified score")
    wall = summary["window_wall_s"]
    if not math.isfinite(wall) or wall <= 0:
        raise ValueError("receipt needs a finite positive wall duration")
    indexed = {}
    for row in jobs:
        key = (row["block"], row["replicate"])
        if key in indexed:
            raise ValueError(f"duplicate pair {key}")
        if row["reads"] != summary["reads"] or row["sweeps"] != summary["sweeps"]:
            raise ValueError("job budget differs from its summary")
        if not 0 <= row["valid_reads"] <= row["reads"]:
            raise ValueError("invalid count of reads below target")
        valid = row["best_milli"] < row["target_milli"]
        if valid != (row["valid_reads"] > 0):
            raise ValueError("validity must use energy strictly below target")
        if row["engine"] not in {"metal", "ane"}:
            raise ValueError("job has no measured engine assignment")
        if summary["engine"] != "both" and row["engine"] != summary["engine"]:
            raise ValueError("job ran on the wrong engine")
        color = summary["color"]
        if color == "mixed":
            color = summary.get(f"{row['engine']}_color")
        if color not in {"greedy", "four"} or row["color"] != color:
            raise ValueError("job coloring differs from its declared arm")
        if not (0 <= row["dispatch_s"] <= row["completed_s"]):
            raise ValueError("invalid dispatch/completion timestamps")
        indexed[key] = row
    if summary["valid_jobs"] != sum(row["valid_reads"] > 0 for row in jobs):
        raise ValueError("summary valid-job count differs from result records")
    return indexed, summary


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def pool(receipts):
    """Pool independently timed windows, retaining each window's local clock."""
    validated = [validate(rows) for rows in receipts]
    fields = ("engine", "color", "metal_color", "ane_color", "reads", "sweeps")
    first = validated[0][1]
    if any(
        any(summary.get(key) != first.get(key) for key in fields)
        for _, summary in validated[1:]
    ):
        raise ValueError("cannot pool different arms")
    jobs = [
        dict(row, window=window)
        for window, (indexed, _) in enumerate(validated)
        for row in indexed.values()
    ]
    summary = dict(first)
    for key in ("window_wall_s", "jobs", "completed", "valid_jobs"):
        summary[key] = sum(item[key] for _, item in validated)
    summary["windows"] = [item for _, item in validated]
    return jobs + [summary]


def arm_summary(indexed, summary):
    jobs = list(indexed.values())
    wall = summary["window_wall_s"]
    valid_jobs = sum(row["valid_reads"] > 0 for row in jobs)
    blocks = defaultdict(list)
    cohorts = defaultdict(list)
    engines = defaultdict(int)
    for row in jobs:
        blocks[row["block"]].append(row)
        cohorts[row.get("window", 0), row["block"]].append(row)
        engines[row["engine"]] += 1
    first_valid = []
    censored = 0
    for rows in cohorts.values():
        start = min(row["dispatch_s"] for row in rows)
        successes = [row["completed_s"] for row in rows if row["valid_reads"] > 0]
        if successes:
            first_valid.append(min(successes) - start)
        else:
            censored += 1
    latencies = [row["completed_s"] - row["dispatch_s"] for row in jobs]
    return {
        "jobs": len(jobs),
        "valid_jobs": valid_jobs,
        "window_wall_s": wall,
        "jobs_per_s": len(jobs) / wall,
        "valid_jobs_per_s": valid_jobs / wall,
        "seconds_per_valid_job": wall / valid_jobs if valid_jobs else None,
        "winner_matching_or_better_jobs_per_s": sum(
            row["best_milli"] <= row["winner_milli"] for row in jobs
        )
        / wall,
        "valid_reads_per_s": sum(row["valid_reads"] for row in jobs) / wall,
        "latency_median_s": statistics.median(latencies),
        "latency_p95_s": percentile(latencies, 0.95),
        "first_valid_median_s_successful_cohorts": (
            statistics.median(first_valid) if first_valid else None
        ),
        "cohorts_without_valid_result": censored,
        "cohorts": len(cohorts),
        "blocks_without_valid_result": sum(
            not any(row["valid_reads"] > 0 for row in rows) for rows in blocks.values()
        ),
        "engine_jobs": dict(engines),
        "windows": summary.get("windows", [summary]),
    }


def compare(control_rows, candidate_rows, draws=10000, allow_read_change=False):
    control, control_summary = validate(control_rows)
    candidate, candidate_summary = validate(candidate_rows)
    if control.keys() != candidate.keys():
        raise ValueError("control and candidate pairs differ")
    arm_fields = ("engine", "color", "metal_color", "ane_color", "reads", "sweeps")
    if all(
        control_summary.get(key) == candidate_summary.get(key) for key in arm_fields
    ):
        raise ValueError("control and candidate arms are identical")
    blocks = defaultdict(list)
    pairs = []
    for key, before in control.items():
        after = candidate[key]
        if not allow_read_change and before["reads"] != after["reads"]:
            raise ValueError(
                "read budgets differ; use --allow-read-change deliberately"
            )
        for field in ["sweeps", "target_milli", "winner_milli"]:
            if before[field] != after[field]:
                raise ValueError(f"pair {key} differs in {field}")
        delta = after["best_milli"] - before["best_milli"]
        valid_before = int(before["valid_reads"] > 0)
        valid_after = int(after["valid_reads"] > 0)
        blocks[key[0]].append((delta, valid_before, valid_after))
        pairs.append(
            {"block": key[0], "replicate": key[1], "energy_delta_milli": delta}
        )
    if len(blocks) < 2 or draws < 100:
        raise ValueError("inference needs at least two blocks and 100 bootstrap draws")
    # Keep all replicates of each selected block together, preserving both arms.
    totals = [
        (
            sum(x[0] for x in rows),
            sum(x[1] for x in rows),
            sum(x[2] for x in rows),
            len(rows),
        )
        for rows in blocks.values()
    ]
    rng = random.Random(20260921)
    energy_draws, validity_draws, ratio_draws = [], [], []
    wall_ratio = control_summary["window_wall_s"] / candidate_summary["window_wall_s"]
    for _ in range(draws):
        selected = rng.choices(totals, k=len(totals))
        energy, old_valid, new_valid, count = map(sum, zip(*selected, strict=True))
        energy_draws.append(energy / count)
        validity_draws.append((new_valid - old_valid) / count)
        if old_valid:
            ratio_draws.append(new_valid / old_valid * wall_ratio)
    deltas = [row["energy_delta_milli"] for row in pairs]
    old = arm_summary(control, control_summary)
    new = arm_summary(candidate, candidate_summary)
    interval = lambda values: [percentile(values, 0.025), percentile(values, 0.975)]
    return {
        "pairs": len(pairs),
        "sampler_seeds": "independent OS seeds; not recorded by the protocol",
        "blocks": len(blocks),
        "bootstrap_draws": draws,
        "control": old,
        "candidate": new,
        "energy_delta_milli": {
            "mean": statistics.mean(deltas),
            "median": statistics.median(deltas),
            "ci95": interval(energy_draws),
            "candidate_better_fraction": sum(x < 0 for x in deltas) / len(deltas),
            "tied_fraction": sum(x == 0 for x in deltas) / len(deltas),
        },
        "valid_probability_delta": {
            "mean": (new["valid_jobs"] - old["valid_jobs"]) / len(pairs),
            "ci95": interval(validity_draws),
        },
        "valid_rate_ratio": {
            "ratio": new["valid_jobs_per_s"] / old["valid_jobs_per_s"]
            if old["valid_jobs"]
            else None,
            # A zero denominator has no finite ratio. Do not discard those draws.
            "ci95": interval(ratio_draws) if len(ratio_draws) == draws else None,
            "undefined_draws": draws - len(ratio_draws),
            "timing_uncertainty_included": False,
        },
        "paired_energy_deltas": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--control-extra", type=Path, action="append", default=[])
    parser.add_argument("--candidate-extra", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--allow-read-change", action="store_true")
    args = parser.parse_args()
    receipts = []
    for paths in [
        [args.control, *args.control_extra],
        [args.candidate, *args.candidate_extra],
    ]:
        receipts.append(
            pool(
                [
                    [json.loads(line) for line in path.read_text().splitlines() if line]
                    for path in paths
                ]
            )
        )
    result = compare(
        *receipts, draws=args.draws, allow_read_change=args.allow_read_change
    )
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
