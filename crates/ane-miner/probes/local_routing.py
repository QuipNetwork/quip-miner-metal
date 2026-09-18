"""Generate exact local fields and resident sweeps from fixed grid shifts and runtime J."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sparse_routing import Emitter, pad32, tensor, write_blob
from topology_layout import Layout, load_layout, pack_j, routing_masks, validate_layout

Int32 = npt.NDArray[np.int32]
Fp16 = npt.NDArray[np.float16]
READS = 128


@dataclass(frozen=True)
class Job:
    spins: Int32
    j: Int32
    h: Int32


def edge_fields(
    edges: Sequence[tuple[int, int]], spins: Int32, j: Int32, h: Int32
) -> Int32:
    """Independent integer oracle in original compact node order."""
    field = h.astype(np.int32).copy()
    for (u, v), coefficient in zip(edges, j, strict=True):
        field[u] += coefficient * spins[v]
        field[v] += coefficient * spins[u]
    return field


def _validate_job(layout: Layout, job: Job, color: int) -> None:
    nodes = len(layout.node_to_slot)
    if type(color) is not int or color not in layout.color_of_node:
        raise ValueError("invalid saved color")
    if job.spins.shape != (nodes, READS) or job.h.shape != job.spins.shape:
        raise ValueError("job needs one spin and h row per node and 128 reads")
    if job.j.shape != (len(layout.edges),):
        raise ValueError("J shape differs from explicit edges")
    if (
        not np.all(np.isin(job.spins, (-1, 1)))
        or not np.all(np.isin(job.h, (-1, 0, 1)))
        or not np.all(np.isin(job.j, (-1, 0, 1)))
    ):
        raise ValueError("spins must be signs and h/J must be -1, 0, or 1")


def _grid(layout: Layout, values: Int32, padding: int = 0) -> Int32:
    grid = np.full((pad32(len(layout.slot_to_node)), READS), padding, dtype=np.int32)
    grid[np.asarray(layout.node_to_slot)] = values
    return grid


def _active_masks(layout: Layout, color: int) -> npt.NDArray[np.bool_]:
    patterns, masks = routing_masks(layout)
    plane = (2 * layout.m + 1) ** 2
    active = np.asarray(masks).reshape(len(patterns), plane) >= 0
    selected = np.zeros(len(layout.slot_to_node), dtype=np.bool_)
    for node, slot in enumerate(layout.node_to_slot):
        selected[slot] = layout.color_of_node[node] == color
    for index, pattern in enumerate(patterns):
        active[index] &= selected[
            pattern.destination_channel * plane : (pattern.destination_channel + 1)
            * plane
        ]
    return active


def routed_fields(layout: Layout, job: Job, color: int) -> Int32:
    """CPU evaluation of masked shifts, independent from the edge-field oracle."""
    _validate_job(layout, job, color)
    side = 2 * layout.m + 1
    plane = side * side
    patterns, _ = routing_masks(layout)
    active = _active_masks(layout, color)
    state = _grid(layout, job.spins, padding=1)
    packed = np.asarray(pack_j(layout, job.j.tolist()), dtype=np.int32).reshape(
        len(patterns), plane
    )
    field = np.zeros_like(state)
    for index, pattern in enumerate(patterns):
        delta = pattern.dy * side + pattern.dx
        shifted = np.zeros((plane, READS), dtype=np.int32)
        begin, end = max(0, -delta), min(plane, plane - delta)
        source = pattern.source_channel * plane
        shifted[begin:end] = state[source + begin + delta : source + end + delta]
        destination = pattern.destination_channel * plane
        field[destination : destination + plane] += (
            shifted * (packed[index] * active[index])[:, None]
        )
    for node, slot in enumerate(layout.node_to_slot):
        if layout.color_of_node[node] == color:
            field[slot] += job.h[node]
    return field


def _validate_thresholds(layout: Layout, thresholds: Int32) -> None:
    if (
        thresholds.ndim != 3
        or thresholds.shape[1:] != (len(layout.node_to_slot), 4)
        or not len(thresholds)
    ):
        raise ValueError("thresholds must have shape [sweeps, nodes, 4]")
    if (
        not np.all(np.isfinite(thresholds))
        or np.any(thresholds != np.rint(thresholds))
        or np.any((thresholds < -128) | (thresholds > 127))
    ):
        raise ValueError("thresholds must be exact integers in [-128, 127]")


def edge_sweeps(layout: Layout, job: Job, thresholds: Int32) -> Int32:
    """Sequential color oracle, using original edges and integer comparisons."""
    _validate_job(layout, job, 0)
    _validate_thresholds(layout, thresholds)
    state = job.spins.copy()
    colors = np.asarray(layout.color_of_node)
    for threshold in thresholds:
        expanded = np.repeat(threshold, 32, axis=1)
        for color in range(max(layout.color_of_node) + 1):
            selected = colors == color
            field = edge_fields(layout.edges, state, job.j, job.h)
            accepted = state[selected] * field[selected] + expanded[selected] >= 0
            state[selected] *= np.where(accepted, -1, 1)
    return state


def routed_sweeps(layout: Layout, job: Job, thresholds: Int32) -> Int32:
    """CPU masked-field sweeps for comparison with the independent oracle."""
    _validate_job(layout, job, 0)
    _validate_thresholds(layout, thresholds)
    state = job.spins.copy()
    colors = np.asarray(layout.color_of_node)
    slots = np.asarray(layout.node_to_slot)
    for threshold in thresholds:
        for color in range(max(layout.color_of_node) + 1):
            field = routed_fields(layout, Job(state, job.j, job.h), color)[slots]
            grouped = (state * field).reshape(-1, 4, 32)
            accepted = np.clip(grouped + threshold[:, :, None] + 1, 0, 1)
            accepted *= (colors == color)[:, None, None]
            state *= (1 - 2 * accepted).reshape(state.shape)
    return state


def load_thresholds(layout: Layout, fixture: Path, sweeps: int) -> Int32:
    meta = json.loads((fixture / "meta.json").read_text())
    if not 0 < sweeps <= meta["sweeps"] or tuple(meta["order"]) != layout.fixture_order:
        raise ValueError(
            "requested sweeps or fixture order differs from saved thresholds"
        )
    raw = np.fromfile(fixture / meta["threshold_file"], dtype=np.float16).reshape(
        meta["sweeps"], meta["channels"], 4
    )
    if np.any(raw != np.rint(raw)) or not np.all(np.isfinite(raw)):
        raise ValueError("saved thresholds must be finite integers")
    thresholds = np.empty((sweeps, len(layout.node_to_slot), 4), dtype=np.int32)
    thresholds[:, np.asarray(layout.fixture_order)] = raw[
        :sweeps, : len(layout.node_to_slot)
    ]
    _validate_thresholds(layout, thresholds)
    return thresholds


def load_job(layout: Layout, fixture: Path) -> Job:
    """Read saved arrays while keeping explicit zero edges and checking support."""
    meta = json.loads((fixture / "meta.json").read_text())
    nodes, channels = meta["nodes"], meta["channels"]
    if (
        nodes != len(layout.node_to_slot)
        or tuple(meta["order"]) != layout.fixture_order
    ):
        raise ValueError("fixture node order differs from layout")
    colors = tuple(
        color for color, tile in enumerate(meta["tiles"]) for _ in tile["nodes"]
    )
    if tuple(layout.color_of_node[node] for node in layout.fixture_order) != colors:
        raise ValueError("fixture color schedule differs from layout")
    if [node for tile in meta["tiles"] for node in tile["nodes"]] != meta["order"]:
        raise ValueError("fixture tile order differs from layout")
    if (meta["reads"], meta["groups"], meta["physical_lanes"]) != (128, 4, 128):
        raise ValueError("fixture must retain four groups of 32 reads")
    order = np.asarray(layout.fixture_order)
    arrays = []
    for key in ("state_file", "h_file"):
        raw = np.fromfile(fixture / meta[key], dtype=np.float16).reshape(
            channels, READS
        )
        if not np.all(np.isin(raw[:nodes], (-1, 0, 1))):
            raise ValueError("fixture values must be exact small integers")
        values = np.zeros((nodes, READS), dtype=np.int32)
        values[order] = raw[:nodes].astype(np.int32)
        arrays.append(values)
    position = np.empty(nodes, dtype=np.int32)
    position[order] = np.arange(nodes)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(nodes)]
    for edge, (u, v) in enumerate(layout.edges):
        adjacency[u].append((v, edge))
        adjacency[v].append((u, edge))
    coefficients = np.zeros(len(layout.edges), dtype=np.int32)
    seen = np.zeros(len(layout.edges), dtype=np.int32)
    begin = 0
    for tile in meta["tiles"]:
        if tile["length"] != len(tile["nodes"]):
            raise ValueError("fixture tile length differs from node list")
        weights = np.fromfile(fixture / tile["weight_file"], dtype=np.float16).reshape(
            tile["padded"], channels
        )
        if np.any(weights[tile["length"] :] != 0) or np.any(weights[:, nodes:] != 0):
            raise ValueError("fixture J padding is nonzero")
        for row in range(tile["length"]):
            node = layout.fixture_order[begin + row]
            nonzero = 0
            for neighbor, edge in adjacency[node]:
                value = weights[row, position[neighbor]]
                if value not in (-1, 0, 1):
                    raise ValueError("fixture J must be -1, 0, or 1")
                if seen[edge] and coefficients[edge] != value:
                    raise ValueError("fixture J is asymmetric")
                coefficients[edge] = int(value)
                seen[edge] += 1
                nonzero += value != 0
            if np.count_nonzero(weights[row, :nodes]) != nonzero:
                raise ValueError("fixture has a coupling outside explicit topology")
        begin += tile["length"]
    if np.any(seen != 2):
        raise ValueError("fixture lacks an explicit edge direction")
    job = Job(arrays[0], coefficients, arrays[1])
    _validate_job(layout, job, 0)
    return job


def _emit_field(
    emit: Emitter,
    layout: Layout,
    prefix: str,
    state: str,
    coefficients: dict[int, str],
    masked_h: str | None,
) -> tuple[str, int]:
    side = 2 * layout.m + 1
    plane, channels = side * side, pad32(len(layout.slot_to_node))
    patterns, _ = routing_masks(layout)
    plane_shape = [1, plane, 1, READS]
    shifted_sources: dict[tuple[int, int], str] = {}
    channel_terms: list[list[str]] = [[] for _ in range(2 * layout.t)]
    for index, coefficient in coefficients.items():
        pattern = patterns[index]
        name = f"{prefix}p{index}"
        delta = pattern.dy * side + pattern.dx
        key = (pattern.source_channel, delta)
        if key not in shifted_sources:
            segment = emit.slice_tensor(
                name + "source",
                state,
                [0, pattern.source_channel * plane + max(delta, 0), 0, 0],
                [1, plane - abs(delta), 1, READS],
            )
            if delta:
                parts = (
                    (segment, f"zero{abs(delta)}")
                    if delta > 0
                    else (f"zero{abs(delta)}", segment)
                )
                segment = emit.emit(
                    plane_shape,
                    name + "shift",
                    f"concat(values=({', '.join(parts)}), axis=axis, interleave=interleave)",
                )
            shifted_sources[key] = segment
        product = emit.emit(
            plane_shape,
            name + "product",
            f"mul(x={shifted_sources[key]}, y={coefficient})",
        )
        channel_terms[pattern.destination_channel].append(product)
    planes = []
    for channel, terms in enumerate(channel_terms):
        total = terms[0] if terms else f"zero{plane}"
        for index, term in enumerate(terms[1:], 1):
            total = emit.emit(
                plane_shape,
                f"{prefix}c{channel}sum{index}",
                f"add(x={total}, y={term})",
            )
        planes.append(total)
    tail = channels - len(layout.slot_to_node)
    if tail:
        planes.append(f"zero{tail}")
    state_shape = [1, channels, 1, READS]
    field = emit.emit(
        state_shape,
        prefix + "field",
        f"concat(values=({', '.join(planes)}), axis=axis, interleave=interleave)",
    )
    if masked_h is not None:
        field = emit.emit(
            state_shape, prefix + "field_with_h", f"add(x={field}, y={masked_h})"
        )
    return field, len(shifted_sources)


def _emit_update(
    emit: Emitter,
    prefix: str,
    state: str,
    field: str,
    threshold: str,
    color_mask: str,
    channels: int,
) -> str:
    shape = [1, channels, 1, READS]
    grouped = [1, channels, 4, 32]
    signed = emit.emit(shape, prefix + "signed", f"mul(x={state}, y={field})")
    signed = emit.reshape(prefix + "signed_groups", signed, grouped)
    threshold = emit.reshape(
        prefix + "threshold_groups", threshold, [1, channels, 4, 1]
    )
    margin = emit.emit(grouped, prefix + "margin", f"add(x={signed}, y={threshold})")
    margin = emit.emit(grouped, prefix + "margin_plus_one", f"add(x={margin}, y=one)")
    accepted = emit.emit(
        grouped, prefix + "accept", f"clip(x={margin}, alpha=zero, beta=one)"
    )
    accepted = emit.emit(
        grouped, prefix + "color_accept", f"mul(x={accepted}, y={color_mask})"
    )
    factor = emit.emit(grouped, prefix + "negative", f"mul(x={accepted}, y=minusTwo)")
    factor = emit.emit(grouped, prefix + "factor", f"add(x=one, y={factor})")
    factor = emit.reshape(prefix + "flat_factor", factor, shape)
    return emit.emit(shape, prefix + "updated", f"mul(x={state}, y={factor})")


def _program(
    output: Path,
    layout: Layout,
    color: int | None,
    zero_h: bool,
    block_sweeps: int = 1,
) -> tuple[str, list[tuple[str, list[int]]], dict[str, int]]:
    side = 2 * layout.m + 1
    plane, channels = side * side, pad32(len(layout.slot_to_node))
    patterns, _ = routing_masks(layout)
    colors = list(range(max(layout.color_of_node) + 1)) if color is None else [color]
    active = {c: _active_masks(layout, c) for c in colors}
    emit = Emitter()
    state_shape = [1, channels, 1, READS]
    jwidth = pad32(plane)
    jshape = [1, len(patterns), 1, jwidth]
    inputs = [("a_state", state_shape)]
    if not zero_h:
        inputs.append(("b_h", state_shape))
    if color is None:
        inputs.append(("c_threshold", [1, channels, 1, pad32(block_sweeps * 4)]))
    inputs.append(("d_j", jshape))
    constants: dict[str, Fp16] = {}
    for c in colors:
        mask = np.zeros((len(patterns), jwidth), dtype=np.float16)
        mask[:, :plane] = active[c]
        constants[f"route_mask{c}"] = mask.reshape(jshape)
        if color is None or not zero_h:
            color_mask = np.zeros((1, channels, 1, 1), dtype=np.float16)
            for node, slot in enumerate(layout.node_to_slot):
                color_mask[0, slot, 0, 0] = layout.color_of_node[node] == c
            constants[f"color_mask{c}"] = color_mask
    selected = {
        c: [i for i in range(len(patterns)) if np.any(active[c][i])] for c in colors
    }
    padding_counts = {plane, channels - len(layout.slot_to_node)}
    for indices in selected.values():
        for index in indices:
            p = patterns[index]
            padding_counts.add(abs(p.dy * side + p.dx))
    for count in sorted(padding_counts):
        if count:
            constants[f"zero{count}"] = np.zeros((1, count, 1, READS), dtype=np.float16)
    blob = write_blob(output / "weight_data.bin", list(constants.items()))
    for (name, array), chunk in zip(constants.items(), blob, strict=True):
        emit.constant(
            tensor(list(array.shape)),
            name,
            'BLOBFILE(path=string("@model_path/weights/weight_data.bin"), '
            + f"offset=uint64({chunk['header_offset']}))",
        )
    emit.constant("int32", "axis", "1")
    emit.constant("bool", "interleave", "false")
    if color is None:
        for name, value in (("zero", "0.0"), ("one", "1.0"), ("minusTwo", "-2.0")):
            emit.constant("fp16", name, value)
    coefficients: dict[int, dict[int, str]] = {}
    h: dict[int, str | None] = {}
    for c in colors:
        coefficients[c] = {}
        for index in selected[c]:
            name = f"c{c}p{index}"
            coefficient = emit.slice_tensor(
                name + "coeff", "d_j", [0, index, 0, 0], [1, 1, 1, plane]
            )
            mask = emit.slice_tensor(
                name + "mask", f"route_mask{c}", [0, index, 0, 0], [1, 1, 1, plane]
            )
            coefficient = emit.emit(
                [1, 1, 1, plane], name + "masked", f"mul(x={coefficient}, y={mask})"
            )
            coefficients[c][index] = emit.reshape(
                name + "j", coefficient, [1, plane, 1, 1]
            )
        h[c] = (
            None
            if zero_h
            else emit.emit(state_shape, f"masked_h{c}", f"mul(x=b_h, y=color_mask{c})")
        )
    state = "a_state"
    shifts = 0
    for sweep in range(block_sweeps if color is None else 1):
        threshold = None
        if color is None:
            threshold = emit.slice_tensor(
                f"s{sweep}threshold",
                "c_threshold",
                [0, 0, 0, sweep * 4],
                [1, channels, 1, 4],
            )
        for c in colors:
            prefix = f"s{sweep}c{c}"
            field, count = _emit_field(
                emit, layout, prefix, state, coefficients[c], h[c]
            )
            shifts += count
            if threshold is None:
                state = field
            else:
                state = _emit_update(
                    emit, prefix, state, field, threshold, f"color_mask{c}", channels
                )
    header = 'program(1.3)\n[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}, {"local-routing-probe", "PROBE_IDENTITY"}})]\n{\n  func main<ios18>('
    program = (
        header
        + ", ".join(tensor(shape) + " " + name for name, shape in inputs)
        + ") {\n"
        + "\n".join(emit.lines)
        + f"\n  }} -> ({state});\n}}\n"
    )
    identity = hashlib.sha256(
        program.encode() + (output / "weight_data.bin").read_bytes()
    ).hexdigest()
    program = program.replace("PROBE_IDENTITY", identity)
    products = sum(len(indices) for indices in selected.values()) * (
        block_sweeps if color is None else 1
    )
    return (
        program,
        inputs,
        {
            "selected_patterns": sum(len(indices) for indices in selected.values()),
            "distinct_shifted_planes": shifts,
            "logical_shifted_planes_bytes": shifts * plane * READS * 2,
            "logical_product_planes_bytes": products * plane * READS * 2,
        },
    )


def prepare_probe(
    output: Path,
    layout: Layout,
    jobs: Sequence[Job],
    *,
    color: int | None,
    thresholds: Sequence[Int32] | None = None,
    block_sweeps: int = 1,
    zero_h: bool = False,
) -> dict[str, Any]:
    validate_layout(layout)
    if not jobs:
        raise ValueError("at least one job is required")
    for job in jobs:
        _validate_job(layout, job, color if color is not None else 0)
    if type(block_sweeps) is not int or not 0 < block_sweeps <= 128:
        raise ValueError("block sweeps must be between 1 and 128")
    sweeps, iterations = 0, 1
    if color is None:
        if thresholds is None or len(thresholds) != len(jobs):
            raise ValueError("sweep mode needs one threshold array per job")
        for threshold in thresholds:
            _validate_thresholds(layout, threshold)
        sweeps = len(thresholds[0])
        if any(len(threshold) != sweeps for threshold in thresholds):
            raise ValueError("all jobs must have the same sweep count")
        iterations = (sweeps + block_sweeps - 1) // block_sweeps
    elif thresholds is not None:
        raise ValueError("field mode does not consume thresholds")
    all_zero_h = all(not np.any(job.h) for job in jobs)
    if zero_h and not all_zero_h:
        raise ValueError("zero-h mode requires every h value to be zero")
    zero_h = all_zero_h
    output.mkdir(parents=True, exist_ok=True)
    program, inputs, graph_sizes = _program(output, layout, color, zero_h, block_sweeps)
    (output / "program.mil").write_text(program)
    files: dict[str, list[str]] = {name: [] for name, _ in inputs}
    expected_files = []
    cpu_mismatches = []
    patterns, _ = routing_masks(layout)
    plane = (2 * layout.m + 1) ** 2
    for index, job in enumerate(jobs):
        state = _grid(layout, job.spins, padding=1)
        values = {"a_state": state.astype(np.float16)}
        if not zero_h:
            values["b_h"] = _grid(layout, job.h).astype(np.float16)
        packed = np.zeros((len(patterns), pad32(plane)), dtype=np.float16)
        packed[:, :plane] = np.asarray(
            pack_j(layout, job.j.tolist()), dtype=np.float16
        ).reshape(len(patterns), plane)
        values["d_j"] = packed
        if thresholds is not None:
            channels = pad32(len(layout.slot_to_node))
            arranged = np.full(
                (iterations * block_sweeps, channels, 4), -128, dtype=np.float16
            )
            arranged[:sweeps, np.asarray(layout.node_to_slot)] = thresholds[index]
            packed_threshold = np.full(
                (iterations, channels, pad32(block_sweeps * 4)), -128, dtype=np.float16
            )
            packed_threshold[:, :, : block_sweeps * 4] = (
                arranged.reshape(iterations, block_sweeps, channels, 4)
                .transpose(0, 2, 1, 3)
                .reshape(iterations, channels, block_sweeps * 4)
            )
            values["c_threshold"] = packed_threshold
        for name, value in values.items():
            path = output / f"{name}-{index}.bin"
            value.tofile(path)
            files[name].append(path.name)
        if color is None:
            assert thresholds is not None
            expected_nodes = edge_sweeps(layout, job, thresholds[index])
            expected = _grid(layout, expected_nodes, padding=1)
            actual = _grid(
                layout, routed_sweeps(layout, job, thresholds[index]), padding=1
            )
        else:
            expected_nodes = edge_fields(layout.edges, job.spins, job.j, job.h)
            for node, node_color in enumerate(layout.color_of_node):
                if node_color != color:
                    expected_nodes[node] = 0
            expected = _grid(layout, expected_nodes)
            actual = routed_fields(layout, job, color)
        mismatches = int(np.count_nonzero(actual != expected))
        if mismatches:
            raise AssertionError(
                f"local field oracle mismatch in job {index}: {mismatches}"
            )
        cpu_mismatches.append(mismatches)
        path = output / f"expected-{index}.bin"
        expected.astype(np.float16).tofile(path)
        expected_files.append(path.name)
        job.j.astype(np.int32).tofile(output / f"edge-j-{index}.i32")
    manifest = {
        "mil": "program.mil",
        "weights": "weight_data.bin",
        "inputs": [
            {"name": name, "elements": int(np.prod(shape)), "files": files[name]}
            for name, shape in inputs
        ],
        "expected": expected_files,
        "output_elements": pad32(len(layout.slot_to_node)) * READS,
        "h_mode": "zero" if zero_h else "runtime",
        "fixed_io": iterations == 1,
        "iterations": iterations,
        "eval_repeats": 10 if iterations == 1 else 1,
        "validate_first": iterations == 1,
    }
    if color is None:
        manifest.update(total_sweeps=sweeps, block_sweeps=block_sweeps)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    proof = {
        "color": color,
        "color_nodes": layout.color_of_node.count(color)
        if color is not None
        else len(layout.node_to_slot),
        "mode": "sweep" if color is None else "field",
        "sweeps": sweeps,
        "block_sweeps": block_sweeps,
        "cpu_mismatches": cpu_mismatches,
        "edges": len(layout.edges),
        "program_bytes": len(program.encode()),
        "constant_blob_bytes": (output / "weight_data.bin").stat().st_size,
        "input_output_logical_bytes": (
            sum(int(np.prod(shape)) for _, shape in inputs)
            + pad32(len(layout.slot_to_node)) * READS
        )
        * 2,
        "program_sha256": hashlib.sha256(program.encode()).hexdigest(),
        **graph_sizes,
    }
    (output / "cpu-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, nargs="+", required=True)
    parser.add_argument("--mode", choices=["field", "sweep"], required=True)
    parser.add_argument("--color", type=int)
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--block-sweeps", type=int, default=1)
    parser.add_argument("--zero-h", action="store_true")
    args = parser.parse_args()
    layout = load_layout(args.layout)
    jobs = [load_job(layout, path) for path in args.fixtures]
    if args.mode == "field" and args.color is None:
        parser.error("field mode requires --color")
    if args.mode == "sweep" and args.color is not None:
        parser.error("sweep mode preserves all colors; omit --color")
    thresholds = (
        [load_thresholds(layout, path, args.sweeps) for path in args.fixtures]
        if args.mode == "sweep"
        else None
    )
    proof = prepare_probe(
        args.output,
        layout,
        jobs,
        color=args.color,
        thresholds=thresholds,
        block_sweeps=args.block_sweeps,
        zero_h=args.zero_h,
    )
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
