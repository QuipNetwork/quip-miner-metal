"""Generate a bounded MIL feasibility program and its runtime input manifest."""

import argparse
import json
import uuid
from pathlib import Path

import numpy as np


def tensor(shape):
    return "tensor<fp16, [" + ",".join(map(str, shape)) + "]>"


def build_mil(channels, lengths, sweeps, zero_h=False):
    lines = []

    def emit(shape, name, expression):
        lines.append(
            f'    {tensor(shape)} {name} = {expression}[name=string("{name}")];'
        )
        return name

    def constant(dtype, name, value):
        lines.append(
            f'    {dtype} {name} = const()[name=string("{name}"), val={dtype}({value})];'
        )

    def slice_tensor(name, source, begin, size):
        for suffix, values in (("begin", begin), ("size", size)):
            constant("tensor<int32, [4]>", name + suffix, str(values))
        return emit(
            size, name, f"slice_by_size(x={source}, begin={name}begin, size={name}size)"
        )

    def reshape(name, source, size):
        constant("tensor<int32, [4]>", name + "shape", str(size))
        return emit(size, name, f"reshape(x={source}, shape={name}shape)")

    def matrix_reshape(name, source, size):
        constant("tensor<int32, [4]>", name + "sh", str(size))
        lines.append(
            f'    tensor<fp16, {size}> {name} = reshape(x={source}, shape={name}sh)[name=string("{name}")];'
        )
        return name

    width = (sweeps * 4 + 31) // 32 * 32
    state_shape = [1, channels, 1, 128]
    inputs = [("a_state", state_shape)]
    if not zero_h:
        inputs.append(("b_h", state_shape))
    inputs.append(("c_threshold", [1, channels, 1, width]))
    for tile, count in enumerate(lengths):
        inputs.append((f"d_j{tile}", [1, (count + 31) // 32 * 32, 1, channels]))
    header = (
        'program(1.3)\n[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}, {"single-call-probe", "'
        + str(uuid.uuid4())
        + '"}})]\n{\n  func main<ios18>('
        + ", ".join(tensor(shape) + " " + name for name, shape in inputs)
        + ") {"
    )
    for dtype, name, value in [
        ("string", "pt", '"valid"'),
        ("tensor<int32, [2]>", "st", "[1,1]"),
        ("tensor<int32, [4]>", "pd", "[0,0,0,0]"),
        ("tensor<int32, [2]>", "dl", "[1,1]"),
        ("int32", "gr", "1"),
        ("int32", "axis", "1"),
        ("bool", "interleave", "false"),
        ("fp16", "zero", "0.0"),
        ("fp16", "one", "1.0"),
        ("fp16", "minusTwo", "-2.0"),
    ]:
        constant(dtype, name, value)
    begin = 0
    for tile, count in enumerate(lengths):
        padded = (count + 31) // 32 * 32
        reshape(f"w{tile}", f"d_j{tile}", [padded, channels, 1, 1])
        if not zero_h:
            slice_tensor(f"h{tile}", "b_h", [0, begin, 0, 0], [1, count, 1, 128])
        begin += count
    state = "a_state"
    for sweep in range(sweeps):
        begin = 0
        for tile, count in enumerate(lengths):
            prefix = f"s{sweep}c{tile}"
            shape = [1, count, 1, 128]
            grouped_shape = [1, count, 4, 32]
            own = slice_tensor(prefix + "own", state, [0, begin, 0, 0], shape)
            threshold = slice_tensor(
                prefix + "threshold",
                "c_threshold",
                [0, begin, 0, sweep * 4],
                [1, count, 1, 4],
            )
            threshold = reshape(prefix + "groupthreshold", threshold, [1, count, 4, 1])
            padded = (count + 31) // 32 * 32
            weights = matrix_reshape(
                prefix + "wm", f"w{tile}", [1, 1, padded, channels]
            )
            spins = matrix_reshape(prefix + "sm", state, [1, 1, channels, 128])
            flag = prefix + "false"
            constant("bool", flag, "false")
            result = emit(
                [1, 1, padded, 128],
                prefix + "mm",
                f"matmul(x={weights}, y={spins}, transpose_x={flag}, transpose_y={flag})",
            )
            raw = matrix_reshape(prefix + "raw", result, [1, padded, 1, 128])
            js = slice_tensor(prefix + "js", raw, [0, 0, 0, 0], shape)
            field = (
                js
                if zero_h
                else emit(shape, prefix + "field", f"add(x={js}, y=h{tile})")
            )
            signed = emit(shape, prefix + "signed", f"mul(x={own}, y={field})")
            signed = reshape(prefix + "groupsigned", signed, grouped_shape)
            margin = emit(
                grouped_shape, prefix + "margin", f"add(x={threshold}, y={signed})"
            )
            shifted = emit(grouped_shape, prefix + "shifted", f"add(x={margin}, y=one)")
            accepted = emit(
                grouped_shape,
                prefix + "accept",
                f"clip(x={shifted}, alpha=zero, beta=one)",
            )
            negative = emit(
                grouped_shape, prefix + "negative", f"mul(x={accepted}, y=minusTwo)"
            )
            factor = emit(grouped_shape, prefix + "factor", f"add(x=one, y={negative})")
            factor = reshape(prefix + "flatfactor", factor, shape)
            updated = emit(shape, prefix + "updated", f"mul(x={own}, y={factor})")
            parts = []
            if begin:
                parts.append(
                    slice_tensor(
                        prefix + "head", state, [0, 0, 0, 0], [1, begin, 1, 128]
                    )
                )
            parts.append(updated)
            if begin + count < channels:
                parts.append(
                    slice_tensor(
                        prefix + "tail",
                        state,
                        [0, begin + count, 0, 0],
                        [1, channels - begin - count, 1, 128],
                    )
                )
            state = (
                emit(
                    state_shape,
                    prefix + "state",
                    f"concat(values=({', '.join(parts)}), axis=axis, interleave=interleave)",
                )
                if len(parts) > 1
                else updated
            )
            begin += count
    return header + "\n" + "\n".join(lines) + f"\n  }} -> ({state});\n}}\n", inputs


def small_fixture(root, sweeps):
    channels, lengths = 32, [16, 16]
    rng = np.random.default_rng(911)
    initial = rng.choice([-1, 1], size=(channels, 128)).astype(np.int16)
    threshold = rng.integers(0, 4, size=(sweeps, channels, 4), dtype=np.int16)
    width = (sweeps * 4 + 31) // 32 * 32
    packed = np.zeros((channels, width), dtype=np.float16)
    packed[:, : sweeps * 4] = threshold.transpose(1, 0, 2).reshape(channels, sweeps * 4)
    packed.tofile(root / "threshold.bin")
    initial.astype(np.float16).tofile(root / "state.bin")
    expected = []
    files = [["state.bin"] * 2, [], ["threshold.bin"] * 2, [], []]
    for job in range(2):
        h = np.full((channels, 128), 1 - 2 * job, dtype=np.int16)
        h.astype(np.float16).tofile(root / f"h{job}.bin")
        files[1].append(f"h{job}.bin")
        weights = np.zeros((channels, channels), dtype=np.int16)
        for node in range(16):
            weights[node, node + 16] = weights[node + 16, node] = 1 - 2 * job
        for tile in range(2):
            padded = np.zeros((32, channels), dtype=np.float16)
            padded[:16] = weights[tile * 16 : (tile + 1) * 16]
            padded.tofile(root / f"j{job}tile{tile}.bin")
            files[3 + tile].append(f"j{job}tile{tile}.bin")
        spins = initial.copy()
        for sweep in range(sweeps):
            for tile in range(2):
                rows = slice(tile * 16, (tile + 1) * 16)
                # Count satisfied incident terms independently of the MIL margin.
                own = spins[rows]
                coupling = weights[rows] @ spins
                satisfied = ((-h[rows] * own + 1) // 2) + ((-coupling * own + 1) // 2)
                limit = (2 + np.repeat(threshold[sweep, rows], 32, axis=1)) // 2
                spins[rows] = np.where(satisfied <= limit, -own, own)
        name = f"expected{job}.bin"
        spins.astype(np.float16).tofile(root / name)
        expected.append(name)
    mil, inputs = build_mil(channels, lengths, sweeps)
    (root / "program.mil").write_text(mil)
    manifest = {
        "mil": "program.mil",
        "inputs": [
            {"name": name, "elements": int(np.prod(shape)), "files": entry}
            for (name, shape), entry in zip(inputs, files, strict=True)
        ],
        "output_elements": channels * 128,
        "expected": expected,
        "threshold_input_index": 2,
        "h_mode": "runtime",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def prepare_fixtures(root, fixtures, block_sweeps=None):
    metas = [json.loads((fixture / "meta.json").read_text()) for fixture in fixtures]
    first = metas[0]
    channels, sweeps = first["channels"], first["sweeps"]
    lengths = [tile["length"] for tile in first["tiles"]]
    block_sweeps = sweeps if block_sweeps is None else block_sweeps
    if block_sweeps <= 0:
        raise ValueError("block sweeps must be positive")
    iterations = (sweeps + block_sweeps - 1) // block_sweeps
    field_paths = [
        fixture / meta["h_file"] for fixture, meta in zip(fixtures, metas, strict=True)
    ]
    if any(path.stat().st_size != channels * 128 * 2 for path in field_paths):
        raise ValueError("h input byte count mismatch")
    fields = [np.fromfile(path, dtype=np.float16) for path in field_paths]
    if any(field.size != channels * 128 for field in fields):
        raise ValueError("h input element count mismatch")
    zero_h = all(not np.any(field) for field in fields)
    threshold_index = 1 if zero_h else 2
    mil, inputs = build_mil(channels, lengths, block_sweeps, zero_h=zero_h)
    files = [[] for _ in inputs]
    expected = []
    width = (block_sweeps * 4 + 31) // 32 * 32
    for job, (fixture, meta) in enumerate(zip(fixtures, metas, strict=True)):
        assert meta["channels"] == channels and meta["sweeps"] == sweeps
        assert meta["order"] == first["order"]
        assert [tile["length"] for tile in meta["tiles"]] == lengths
        files[0].append(str((fixture / meta["state_file"]).resolve()))
        if not zero_h:
            files[1].append(str((fixture / meta["h_file"]).resolve()))
        threshold = np.fromfile(
            fixture / meta["threshold_file"], dtype=np.float16
        ).reshape(sweeps, channels, 4)
        # A -128 threshold skips unused tail sweeps at the maximum local field.
        padded = np.full(
            (iterations * block_sweeps, channels, 4), -128, dtype=np.float16
        )
        padded[:sweeps] = threshold
        packed = np.full((iterations, channels, width), -128, dtype=np.float16)
        packed[:, :, : block_sweeps * 4] = (
            padded.reshape(iterations, block_sweeps, channels, 4)
            .transpose(0, 2, 1, 3)
            .reshape(iterations, channels, block_sweeps * 4)
        )
        threshold_file = root / f"threshold{job}.bin"
        packed.tofile(threshold_file)
        files[threshold_index].append(str(threshold_file.resolve()))
        for tile, tile_meta in enumerate(meta["tiles"]):
            files[threshold_index + 1 + tile].append(
                str((fixture / tile_meta["weight_file"]).resolve())
            )
        expected.append(str((fixture / meta["expected_file"]).resolve()))
    (root / "program.mil").write_text(mil)
    manifest = {
        "mil": "program.mil",
        "inputs": [
            {"name": name, "elements": int(np.prod(shape)), "files": entry}
            for (name, shape), entry in zip(inputs, files, strict=True)
        ],
        "output_elements": channels * 128,
        "expected": expected,
        "iterations": iterations,
        "block_sweeps": block_sweeps,
        "total_sweeps": sweeps,
        "threshold_input_index": threshold_index,
        "h_mode": "zero" if zero_h else "runtime",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "channels": channels,
                "sweeps": sweeps,
                "tiles": len(lengths),
                "mil_bytes": len(mil),
                "iterations": iterations,
                "block_sweeps": block_sweeps,
                "input_bytes": sum(
                    int(np.prod(shape))
                    * 2
                    * (iterations if i == threshold_index else 1)
                    for i, (_, shape) in enumerate(inputs)
                ),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--fixtures", type=Path, nargs="+")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--block-sweeps", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.compile_only:
        if not args.fixtures:
            parser.error("--compile-only requires --fixtures to define the topology")
        meta = json.loads((args.fixtures[0] / "meta.json").read_text())
        mil, _ = build_mil(
            meta["channels"], [tile["length"] for tile in meta["tiles"]], args.sweeps
        )
        (args.output / "program.mil").write_text(mil)
        (args.output / "manifest.json").write_text(
            json.dumps({"mil": "program.mil", "compile_only": True}, indent=2)
        )
        print(
            json.dumps(
                {"sweeps": args.sweeps, "mil_bytes": len(mil), "compile_only": True}
            )
        )
    elif args.fixtures:
        prepare_fixtures(args.output, args.fixtures, args.block_sweeps)
    else:
        small_fixture(args.output, args.sweeps)


if __name__ == "__main__":
    main()
