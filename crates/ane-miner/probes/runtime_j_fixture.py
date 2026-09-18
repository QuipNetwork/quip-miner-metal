"""Build a production-shape runtime-J graph and manifest for profile_runner.

Bead quip-miner-metal-yba asks whether the couplings can move out of the
compiled weight blob into a runtime input, which would let one compiled
program serve many jobs and remove the per-job compile cost.

profile_graphs.py already emits graphs that take J as a runtime input, but
its production path needs a fixture directory that is not checked in, and
that fixture carries fields this question does not need. This script builds
the graph and manifest directly, at production shape, in the same form
single_call.small_fixture uses.

The reference spins come from the same integer rule small_fixture uses, so
profile_runner checks the result rather than only timing it. A run that
reports mismatches greater than zero invalidates its own timings.

Shape matches every other probe in this directory: 4,608 channels and the
eight tile lengths 857, 849, 817, 740, 685, 480, 136, and 13, from
tests/fixtures/advantage2-system1.edges as graph.rs colors it. Couplings are
in {-1, 0, 1} with at most 20 nonzero neighbors per variable, matching the
advertised topology limit. Coupling values do not change the cost of a dense
conv or matmul, which is what this probe times, but they do decide whether
the correctness check is meaningful.

Usage: runtime_j_fixture.py OUTPUT_DIR --mode MODE [--sweeps N] [--eval-repeats N]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "profile_graphs", Path(__file__).with_name("profile_graphs.py")
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to load profile_graphs.py")
profile_graphs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(profile_graphs)

CHANNELS = 4608
LENGTHS = [857, 849, 817, 740, 685, 480, 136, 13]
READS = 128
GROUPS = 4
# At most 20 nonzero neighbors per variable, the limit crates/ane-miner
# advertises for the production topology.
DEGREE = 20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["complete", "hoist", "conv", "state_mm"],
    )
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--eval-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.sweeps < 1:
        parser.error("sweeps must be positive")
    root = args.output
    root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    channels, lengths, sweeps = CHANNELS, LENGTHS, args.sweeps
    if sum(lengths) > channels:
        raise ValueError("tile lengths exceed channels")

    initial = rng.choice([-1, 1], size=(channels, READS)).astype(np.int16)
    initial.astype(np.float16).tofile(root / "state.bin")

    # Four threshold groups, each controlling 32 of the 128 reads, packed the
    # way single_call.small_fixture packs them.
    threshold = rng.integers(0, 4, size=(sweeps, channels, GROUPS), dtype=np.int16)
    width = (sweeps * GROUPS + 31) // 32 * 32
    packed = np.zeros((channels, width), dtype=np.float16)
    packed[:, : sweeps * GROUPS] = threshold.transpose(1, 0, 2).reshape(
        channels, sweeps * GROUPS
    )
    packed.tofile(root / "threshold.bin")

    h = rng.choice([-1, 1], size=(channels, READS)).astype(np.int16)
    h.astype(np.float16).tofile(root / "h.bin")

    # Coupling matrix in {-1, 0, 1} with at most DEGREE nonzero neighbors per
    # row, matching the advertised topology limit.
    #
    # Density has to be real here, not convenient. A dense {-1, 1} matrix
    # would sum 4,608 terms per row, and fp16 represents integers exactly
    # only to 2,048, so the conv accumulator would round and disagree with
    # the integer reference below. That disagreement would be an artifact of
    # the fixture rather than a fault in the graph. Density does not change
    # what a dense conv or matmul costs, which is what this probe times.
    weights = np.zeros((channels, channels), dtype=np.int16)
    for row in range(channels):
        neighbors = rng.choice(channels, size=DEGREE, replace=False)
        weights[row, neighbors] = rng.choice([-1, 1], size=DEGREE)
        weights[row, row] = 0
    tile_files = []
    begin = 0
    for tile, count in enumerate(lengths):
        padded_rows = profile_graphs.pad32(count)
        block = np.zeros((padded_rows, channels), dtype=np.float16)
        block[:count] = weights[begin : begin + count]
        name = f"j{tile}.bin"
        block.tofile(root / name)
        tile_files.append(name)
        begin += count

    # Reference sweep, mirroring the metropolis block profile_graphs.py emits:
    #
    #   signed  = own * (J . spins + h)
    #   margin  = threshold + signed
    #   accept  = clip(margin + 1, 0, 1)
    #   factor  = 1 - 2 * accept
    #   updated = own * factor
    #
    # Every quantity here is an integer, so accept is exactly 0 or 1 and the
    # clip never produces a fraction. A node therefore flips exactly when
    # threshold + own * field is at least zero.
    #
    # Do not substitute the satisfied-terms rule single_call.small_fixture
    # uses. That rule is deliberately independent of the MIL margin, and it
    # agrees with this one only for that fixture's constant h and
    # single-neighbor couplings.
    #
    # The graph reshapes the 128 reads to [count, 4, 32], so threshold group g
    # covers reads 32g through 32g + 31. np.repeat reproduces that grouping.
    spins = initial.copy()
    for sweep in range(sweeps):
        begin = 0
        for count in lengths:
            rows = slice(begin, begin + count)
            own = spins[rows]
            field = weights[rows] @ spins + h[rows]
            margin = np.repeat(threshold[sweep, rows], 32, axis=1) + own * field
            spins[rows] = np.where(margin >= 0, -own, own)
            begin += count
    spins.astype(np.float16).tofile(root / "expected.bin")

    mil, inputs = profile_graphs.build_graph(
        channels, lengths, sweeps, args.mode, zero_h=False
    )
    (root / "program.mil").write_text(mil)

    names = [name for name, _ in inputs]
    supplied = {
        "a_state": ["state.bin"],
        "b_h": ["h.bin"],
        "c_threshold": ["threshold.bin"],
    }
    for tile, name in enumerate(tile_files):
        supplied[f"d_j{tile}"] = [name]
    missing = [name for name in names if name not in supplied]
    if missing:
        raise ValueError(f"graph wants inputs this script does not build: {missing}")

    manifest = {
        "mil": "program.mil",
        "inputs": [
            {"name": name, "elements": int(np.prod(shape)), "files": supplied[name]}
            for name, shape in inputs
        ],
        "output_elements": channels * READS,
        "expected": ["expected.bin"],
        "threshold_input_index": names.index("c_threshold"),
        "h_mode": "runtime",
        "eval_repeats": args.eval_repeats,
        "diagnostics": True,
        "profile_mode": args.mode,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "channels": channels,
                "sweeps": sweeps,
                "inputs": names,
                "mil_bytes": len(mil),
            }
        )
    )


if __name__ == "__main__":
    main()
