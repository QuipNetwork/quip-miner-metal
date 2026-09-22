#!/usr/bin/env python3
"""Run serial production-channel quality or router windows in alternating order."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terminate(signum, _frame):
    raise SystemExit(128 + signum)


def run_window(command, root, env, log, timeout=1500):
    """Terminate the entire device-using process group if the window fails."""
    with subprocess.Popen(
        command,
        cwd=root,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    ) as process:
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            finally:
                # The leader can exit before a descendant that ignores TERM.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["quality", "router"])
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--metal-color", choices=["greedy", "four"], default="greedy")
    parser.add_argument("--ane-color", choices=["greedy", "four"], default="four")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, terminate)
    root = Path(__file__).resolve().parents[2]
    binary, driver, out = (
        args.binary.resolve(),
        args.driver.resolve(),
        args.out.resolve(),
    )
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "quality":
        arms = [
            ("metal", "greedy", 64),
            ("metal", "four", 64),
            ("ane", "greedy", 64),
            ("ane", "four", 64),
            ("ane", "four", 128),
        ]
        seeds, k0, rounds = 1, 10000, 3
        schedule = [arms, list(reversed(arms)), arms[2:] + arms[:2]]
    else:
        arms = [("metal", args.metal_color, 64), ("both", "mixed", 64)]
        seeds, k0, rounds = 5, 20000, 4
        schedule = [arms, list(reversed(arms))] * 2
    metadata = {
        "binary_sha256": sha256(binary),
        "driver_sha256": sha256(driver),
        "fixture_sha256": sha256(
            root / "tests/fixtures/testnet-qblocks-3191-3250.json"
        ),
        "rounds": rounds,
        "blocks": 60,
        "replicates_per_block_per_round": seeds,
        "sweeps": 14336,
        "k0": k0,
        "arms": arms,
        "metal_color": args.metal_color,
        "ane_color": args.ane_color,
        "sampler_seeds": "independent OS seeds, unavailable through protocol",
        "schedule": schedule,
        "position_balance": "incomplete" if args.mode == "quality" else "balanced",
        "environment": {
            key: os.environ.get(key)
            for key in ["QUIP_METAL_TG_PER_CORE", "QUIP_SCORE_THREADS"]
        },
    }
    protocol = out / f"{args.mode}-protocol.json"
    with protocol.open("x") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    for repeat, order in enumerate(schedule):
        for engine, color, reads in order:
            name = f"{args.mode}-{engine}-{color}-{reads}-r{repeat}"
            receipt = out / f"{name}.jsonl"
            if receipt.exists():
                raise FileExistsError(receipt)
            env = dict(
                os.environ,
                QUIP_STUDY_BIN=str(binary),
                QUIP_STUDY_OUT=str(receipt),
                QUIP_STUDY_ENGINE=engine,
                QUIP_STUDY_COLOR=color,
                QUIP_STUDY_METAL_COLOR=args.metal_color,
                QUIP_STUDY_ANE_COLOR=args.ane_color,
                QUIP_STUDY_READS=str(reads),
                QUIP_STUDY_SWEEPS="14336",
                QUIP_STUDY_SEEDS=str(seeds),
                QUIP_STUDY_K0=str(k0 + repeat * seeds),
                QUIP_STUDY_MAX_BLOCKS="60",
            )
            print(f"START {name}", flush=True)
            with (out / f"{name}.driver.log").open("x") as log:
                run_window(
                    [
                        str(root / "scripts/ane-guard"),
                        "--",
                        str(driver),
                        "production_channel_study",
                        "--ignored",
                        "--exact",
                        "--nocapture",
                    ],
                    root,
                    env,
                    log,
                )
            summary = json.loads(receipt.read_text().splitlines()[-1])
            if not summary["rescore_ok"] or summary["completed"] != 60 * seeds:
                raise ValueError(f"invalid receipt: {receipt}")
            print(f"DONE {name} {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
