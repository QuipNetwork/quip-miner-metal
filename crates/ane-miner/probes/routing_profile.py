"""Profile existing B32 routing stages with independent expected values."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sparse_routing as routing


def distinct_rows(rows: int) -> routing.Fp16:
    """Use two exact integer columns to distinguish every source row."""
    indices = np.arange(rows, dtype=np.int32)
    values = ((indices[:, None] + np.arange(128)[None, :] * 17) % 1024).astype(
        np.float16
    )
    values[:, 0] = indices % 1024
    values[:, 1] = indices // 1024
    return values


def direct_field(
    state: routing.Fp16, weights: routing.Fp16, field_h: routing.Fp16
) -> routing.Fp16:
    """Traverse coefficient rows without using the routing map or network."""
    result = field_h.astype(np.int32).copy()
    for row in range(weights.shape[0]):
        for source in np.flatnonzero(weights[row]):
            result[row] += int(weights[row, source]) * state[source].astype(np.int32)
    return result.astype(np.float16)


def count_sweep(
    state: routing.Fp16,
    weights: list[routing.Fp16],
    field_h: routing.Fp16,
    thresholds: routing.Fp16,
) -> routing.Fp16:
    """Count energy-favorable unit terms instead of mirroring the MIL gate."""
    result = state.copy()
    begin = 0
    for tile in weights:
        before = result.copy()
        for row in range(tile.shape[0]):
            node = begin + row
            sources = np.flatnonzero(tile[row])
            terms = tile[row, sources, None] * before[sources]
            if field_h[node, 0] != 0:
                terms = np.concatenate([terms, field_h[node : node + 1]])
            count = np.count_nonzero(terms * before[node] < 0, axis=0)
            limit = (len(terms) + np.repeat(thresholds[node], 32)) // 2
            result[node] = np.where(count <= limit, -before[node], before[node])
        begin += tile.shape[0]
    return result


def component(
    output: Path, route: routing.ColorRoute, mode: str, edge_j: routing.Fp16
) -> tuple[str, list[tuple[str, routing.Fp16]], routing.Fp16]:
    emit = routing.Emitter()
    for dtype, name, value in [
        ("string", "pt", '"valid"'),
        ("tensor<int32, [2]>", "st", "[1,1]"),
        ("tensor<int32, [4]>", "pd", "[0,0,0,0]"),
        ("tensor<int32, [2]>", "dl", "[1,1]"),
        ("int32", "axis", "1"),
        ("int32", "axis2", "2"),
        ("bool", "interleave", "false"),
        ("fp16", "one", "1.0"),
        ("tensor<int32, [1]>", "sum_axes", "[2]"),
        ("bool", "keep_dims", "true"),
    ]:
        emit.constant(dtype, name, value)
    chunks: list[tuple[str, routing.Fp16]] = []
    if mode == "extract":
        chunks.append(("selector", route.selector[:, :, None, None]))
        if route.route_n > route.extract_n:
            chunks.append(
                (
                    "padding",
                    np.zeros((1, route.route_n - route.extract_n, 1, 128), np.float16),
                )
            )
    elif mode == "permute":
        chunks.extend((f"mask{i}", mask) for i, mask in enumerate(route.swap_masks()))
    layout = routing.write_blob(output / "weight_data.bin", chunks)
    for (name, array), chunk in zip(chunks, layout, strict=True):
        emit.constant(
            routing.tensor(list(array.shape)),
            name,
            'BLOBFILE(path=string("@model_path/weights/weight_data.bin"), '
            f"offset=uint64({chunk['header_offset']}))",
        )
    if mode == "extract":
        values = distinct_rows(route.groups * route.block)
        arrays = [("a_state", values.reshape(1, -1, 1, 128))]
        emit.constant("int32", "groups", str(route.groups))
        result = emit.emit(
            [1, route.extract_n, 1, 128],
            "extracted",
            "conv(dilations=dl, groups=groups, pad=pd, pad_type=pt, strides=st, "
            "weight=selector, x=a_state)",
        )
        if route.route_n > route.extract_n:
            result = emit.emit(
                [1, route.route_n, 1, 128],
                "padded",
                f"concat(values=({result}, padding), axis=axis, interleave=interleave)",
            )
        expected = np.zeros((route.route_n, 128), np.float16)
        for group, edges in enumerate(route.group_edges):
            for local, (_, _, source) in enumerate(edges):
                expected[group * route.epad + local] = values[source]
    elif mode == "permute":
        values = distinct_rows(route.route_n)
        arrays = [("a_extracted", values.reshape(1, -1, 1, 128))]
        result = routing.emit_benes(
            emit, "route", "a_extracted", route, [name for name, _ in chunks]
        )
        expected = values[np.asarray(route.src_of)]
    else:
        values = np.where(
            np.arange(route.count * 20 * 128).reshape(route.count, 20, 128) % 3,
            1,
            -1,
        ).astype(np.float16)
        arrays = [
            ("a_neighbors", values[None]),
            ("b_j", routing.packed_j(edge_j).reshape(1, route.count, 1, 32)),
        ]
        flat = emit.slice_tensor("jflat", "b_j", [0, 0, 0, 0], [1, route.count, 1, 20])
        j = emit.reshape("j", flat, [1, route.count, 20, 1])
        products = emit.emit(
            [1, route.count, 20, 128], "products", f"mul(x=a_neighbors, y={j})"
        )
        result = emit.emit(
            [1, route.count, 1, 128],
            "sum",
            f"reduce_sum(x={products}, axes=sum_axes, keep_dims=keep_dims)",
        )
        expected_int = np.zeros((route.count, 128), np.int32)
        for slot in range(20):
            expected_int += values[:, slot].astype(np.int32) * edge_j[
                :, slot, None
            ].astype(np.int32)
        expected = expected_int.astype(np.float16)
    inputs = [(name, list(array.shape)) for name, array in arrays]
    program = (
        routing.program_header(inputs)
        + "\n"
        + "\n".join(emit.lines)
        + f"\n  }} -> ({result});\n}}\n"
    )
    return program, arrays, expected


def prepare(output: Path, fixture: Path, mode: str, repeats: int) -> dict[str, Any]:
    meta, neighbors, lengths = routing.load_union_neighbors([fixture])
    channels = int(meta["channels"])
    routes: list[routing.ColorRoute] = []
    begin = 0
    for tile, count in enumerate(lengths):
        ids, live = routing.pack_ell(neighbors[begin : begin + count])
        routes.append(routing.ColorRoute(tile, begin, count, ids, live, 32, channels))
        begin += count
    color = max(range(len(lengths)), key=lengths.__getitem__)
    route = routes[color]
    state = np.fromfile(fixture / meta["state_file"], np.float16).reshape(channels, 128)
    field_h = np.fromfile(fixture / meta["h_file"], np.float16).reshape(channels, 128)
    thresholds = np.fromfile(fixture / meta["threshold_file"], np.float16).reshape(
        meta["sweeps"], channels, 4
    )
    weights = [
        np.fromfile(fixture / tile["weight_file"], np.float16).reshape(-1, channels)[
            : tile["length"]
        ]
        for tile in meta["tiles"]
    ]
    edges = [
        routing.fill_edge_j(weight, item.sources, item.live)
        for weight, item in zip(weights, routes, strict=True)
    ]
    manifest: dict[str, Any]
    if mode in {"field", "sweep"}:
        jobs = [
            {
                "state": state,
                "h": field_h,
                "thresholds": thresholds,
                "edges": edges,
                "state_path": str(fixture / meta["state_file"]),
                "h_path": str(fixture / meta["h_file"]),
            }
        ]
        routing.prepare_output(
            output,
            channels,
            lengths,
            routes,
            [item.sources for item in routes],
            1,
            1,
            jobs,
            not np.any(field_h),
            color if mode == "field" else None,
            False,
        )
        manifest = json.loads((output / "manifest.json").read_text())
        expected = (
            direct_field(
                state, weights[color], field_h[route.begin : route.begin + route.count]
            )
            if mode == "field"
            else count_sweep(state, weights, field_h, thresholds[0])
        )
    else:
        program, arrays, expected = component(output, route, mode, edges[color])
        identity = hashlib.sha256(program.encode()).hexdigest()
        (output / "program.mil").write_text(program.replace("PROBE_IDENTITY", identity))
        inputs = []
        for index, (name, array) in enumerate(arrays):
            path = output / f"input{index}.bin"
            array.tofile(path)
            inputs.append({"name": name, "elements": array.size, "files": [str(path)]})
        manifest = {"mil": "program.mil", "inputs": inputs}
        if mode != "reduce":
            manifest["weights"] = "weight_data.bin"
    expected.tofile(output / "expected.bin")
    input_specs: list[dict[str, Any]] = manifest["inputs"]
    for index, spec in enumerate(input_specs):
        path = output / f"input{index}.bin"
        source = Path(spec["files"][0])
        if source != path:
            path.write_bytes(source.read_bytes())
        spec["files"] = [str(path)]
    manifest.update(
        expected=["expected.bin"],
        output_elements=expected.size,
        eval_repeats=repeats,
        iterations=1,
        fixed_io=True,
        validate_first=True,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "fixture-meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (output / "neighbors.json").write_text(json.dumps(neighbors) + "\n")
    receipt: dict[str, Any] = {
        "mode": mode,
        "fixture": str(fixture),
        "color": color if mode != "sweep" else None,
        "color_nodes": route.count,
        "channels": channels,
        "groups": route.groups,
        "extract_rows": route.extract_n,
        "route_rows": route.route_n,
        "route_vector_bytes": route.route_n * 128 * 2,
        "destination_neighbor_bytes": route.count * 20 * 128 * 2,
        "expected_elements": expected.size,
        "repeats": repeats,
        "program_bytes": (output / "program.mil").stat().st_size,
        "weight_blob_bytes": (output / "weight_data.bin").stat().st_size,
        "input_bytes": sum(spec["elements"] * 2 for spec in input_specs),
        "output_bytes": expected.nbytes,
        "topology_scope": "Existing nonzero coefficient support only; not full topology coverage.",
        "oracle": "direct indexing / integer edge traversal / count-based Metropolis",
        "status": "generated, not executed",
        "artifact_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir()
            if p.is_file()
        },
    }
    return receipt


def execute(output: Path, runner: Path, guard: Path, receipt: dict[str, Any]) -> int:
    command = [
        sys.executable,
        str(guard),
        str(runner),
        str(output / "manifest.json"),
        "30",
    ]
    started = time.monotonic()
    with (output / "run.log").open("w") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    receipt.update(
        command=command, wall_seconds=time.monotonic() - started, exit=result.returncode
    )
    log = (output / "run.log").read_text()
    samples = [
        int(value) for value in re.findall(r"stage=evaluate .*?evaluate_us=(\d+)", log)
    ]
    receipt["samples_us"] = samples
    for stage, key in [
        ("compiled", "compile_us"),
        ("loaded", "load_us"),
        ("setup", "setup_us"),
        ("setup", "surface_bytes"),
        ("uploaded", "upload_us"),
        ("readback", "readback_us"),
        ("validation", "validation_us"),
        ("completed", "peak_rss_bytes"),
        ("completed", "mismatches"),
    ]:
        match = re.search(rf"stage={stage} .*?{key}=(\d+)", log)
        receipt[key] = int(match[1]) if match else None
    if result.returncode == 0:
        if len(samples) != receipt["repeats"] or receipt["mismatches"] != 0:
            raise ValueError("incomplete successful receipt")
        receipt.update(
            status="exact first and final output, completed dispatches",
            first_us=samples[0],
            warm_median_us=statistics.median(samples[1:]),
            warm_min_us=min(samples[1:]),
            warm_max_us=max(samples[1:]),
        )
        time.sleep(3)
    else:
        receipt["status"] = "failed or blocked; stop hardware trials"
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["extract", "permute", "reduce", "field", "sweep"],
        required=True,
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--guard", type=Path)
    args = parser.parse_args()
    if not 2 <= args.repeats <= 16:
        parser.error("repeats must be between 2 and 16")
    if (args.runner is None) != (args.guard is None):
        parser.error("runner and guard must be supplied together")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = prepare(output, args.fixture.resolve(), args.mode, args.repeats)
    code = 0
    if args.runner is not None:
        code = execute(output, args.runner.resolve(), args.guard.resolve(), receipt)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps({"output": str(output), "status": receipt["status"], "exit": code}),
        flush=True,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
