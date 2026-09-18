"""Build isolated ANE profiling graphs. Does not change the baseline generator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt


class TileMeta(TypedDict):
    length: int
    weight_file: str


class FixtureMeta(TypedDict):
    channels: int
    sweeps: int
    state_file: str
    h_file: str
    threshold_file: str
    tiles: list[TileMeta]


_SPEC = importlib.util.spec_from_file_location(
    "single_call", Path(__file__).with_name("single_call.py")
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to load single_call.py")
single_call = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(single_call)

Fp16 = npt.NDArray[np.float16]


def tensor(shape: list[int]) -> str:
    return "tensor<fp16, [" + ",".join(map(str, shape)) + "]>"


def pad32(count: int) -> int:
    return (count + 31) // 32 * 32


def round_width(sweeps: int) -> int:
    return pad32(sweeps * 4)


class Emitter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def emit(self, shape: list[int], name: str, expression: str) -> str:
        self.lines.append(
            f'    {tensor(shape)} {name} = {expression}[name=string("{name}")];'
        )
        return name

    def constant(self, dtype: str, name: str, value: str) -> str:
        self.lines.append(
            f'    {dtype} {name} = const()[name=string("{name}"), val={dtype}({value})];'
        )
        return name

    def slice_tensor(
        self, name: str, source: str, begin: list[int], size: list[int]
    ) -> str:
        for suffix, values in (("begin", begin), ("size", size)):
            self.constant("tensor<int32, [4]>", name + suffix, str(values))
        return self.emit(
            size, name, f"slice_by_size(x={source}, begin={name}begin, size={name}size)"
        )

    def reshape(self, name: str, source: str, size: list[int]) -> str:
        self.constant("tensor<int32, [4]>", name + "shape", str(size))
        return self.emit(size, name, f"reshape(x={source}, shape={name}shape)")

    def matrix_reshape(self, name: str, source: str, size: list[int]) -> str:
        self.constant("tensor<int32, [4]>", name + "sh", str(size))
        self.lines.append(
            f'    tensor<fp16, {size}> {name} = reshape(x={source}, shape={name}sh)[name=string("{name}")];'
        )
        return name


def metropolis(
    emit: Emitter,
    prefix: str,
    own: str,
    field: str,
    threshold: str,
    count: int,
) -> str:
    shape = [1, count, 1, 128]
    grouped = [1, count, 4, 32]
    signed = emit.emit(shape, prefix + "signed", f"mul(x={own}, y={field})")
    signed = emit.reshape(prefix + "groupsigned", signed, grouped)
    grouped_threshold = emit.reshape(
        prefix + "groupthreshold", threshold, [1, count, 4, 1]
    )
    margin = emit.emit(
        grouped, prefix + "margin", f"add(x={grouped_threshold}, y={signed})"
    )
    shifted = emit.emit(grouped, prefix + "shifted", f"add(x={margin}, y=one)")
    accepted = emit.emit(
        grouped, prefix + "accept", f"clip(x={shifted}, alpha=zero, beta=one)"
    )
    negative = emit.emit(grouped, prefix + "negative", f"mul(x={accepted}, y=minusTwo)")
    factor = emit.emit(grouped, prefix + "factor", f"add(x=one, y={negative})")
    factor = emit.reshape(prefix + "flatfactor", factor, shape)
    return emit.emit(shape, prefix + "updated", f"mul(x={own}, y={factor})")


def concat_state(
    emit: Emitter,
    prefix: str,
    state: str,
    updated: str,
    begin: int,
    count: int,
    channels: int,
    state_shape: list[int],
    axis: str,
    layout: str,
) -> str:
    parts: list[str] = []
    # Head is the unchanged prefix starting at index 0, matching single_call.build_mil.
    if layout == "nchw":
        if begin:
            parts.append(
                emit.slice_tensor(
                    prefix + "head", state, [0, 0, 0, 0], [1, begin, 1, 128]
                )
            )
        parts.append(updated)
        if begin + count < channels:
            parts.append(
                emit.slice_tensor(
                    prefix + "tail",
                    state,
                    [0, begin + count, 0, 0],
                    [1, channels - begin - count, 1, 128],
                )
            )
    else:
        if begin:
            parts.append(
                emit.slice_tensor(
                    prefix + "head", state, [0, 0, 0, 0], [1, 1, begin, 128]
                )
            )
        parts.append(updated)
        if begin + count < channels:
            parts.append(
                emit.slice_tensor(
                    prefix + "tail",
                    state,
                    [0, 0, begin + count, 0],
                    [1, 1, channels - begin - count, 128],
                )
            )
    if len(parts) == 1:
        return updated
    return emit.emit(
        state_shape,
        prefix + "state",
        f"concat(values=({', '.join(parts)}), axis={axis}, interleave=interleave)",
    )


def tile_matmul(
    emit: Emitter,
    prefix: str,
    weights: str,
    spins: str,
    padded: int,
    channels: int,
    count: int,
) -> str:
    flag = prefix + "false"
    emit.constant("bool", flag, "false")
    result = emit.emit(
        [1, 1, padded, 128],
        prefix + "mm",
        f"matmul(x={weights}, y={spins}, transpose_x={flag}, transpose_y={flag})",
    )
    raw = emit.matrix_reshape(prefix + "raw", result, [1, padded, 1, 128])
    return emit.slice_tensor(prefix + "js", raw, [0, 0, 0, 0], [1, count, 1, 128])


def build_graph(
    channels: int,
    lengths: list[int],
    sweeps: int,
    mode: str,
    zero_h: bool,
    tile_index: int | None = None,
) -> tuple[str, list[tuple[str, list[int]]]]:
    emit = Emitter()
    width = round_width(sweeps)
    state_nchw = [1, channels, 1, 128]
    state_mm = [1, 1, channels, 128]
    selected = (
        list(enumerate(lengths))
        if tile_index is None
        else [(tile_index, lengths[tile_index])]
    )
    inputs: list[tuple[str, list[int]]] = [("a_state", state_nchw)]
    if mode == "identity":
        header_inputs = inputs
        emit.constant("fp16", "one", "1.0")
        state = emit.emit(state_nchw, "out", "mul(x=a_state, y=one)")
        return finish(header_inputs, emit, state, zero_h=True), inputs

    if not zero_h and mode in {
        "complete",
        "hoist",
        "conv",
        "state_mm",
    }:
        inputs.append(("b_h", state_nchw))
    if mode not in {"concat", "identity", "field_tiled", "field_full", "field_tile"}:
        inputs.append(("c_threshold", [1, channels, 1, width]))
    if mode == "metro":
        inputs.append(("e_field", state_nchw))
    if mode not in {"concat", "metro", "identity"}:
        for tile, count in enumerate(lengths):
            if tile_index is not None and tile != tile_index:
                continue
            inputs.append((f"d_j{tile}", [1, pad32(count), 1, channels]))

    emit.constant("string", "pt", '"valid"')
    emit.constant("tensor<int32, [2]>", "st", "[1,1]")
    emit.constant("tensor<int32, [4]>", "pd", "[0,0,0,0]")
    emit.constant("tensor<int32, [2]>", "dl", "[1,1]")
    emit.constant("int32", "gr", "1")
    axis_name = "axis2" if mode == "state_mm" else "axis"
    axis_value = "2" if mode == "state_mm" else "1"
    emit.constant("int32", axis_name, axis_value)
    emit.constant("bool", "interleave", "false")
    emit.constant("fp16", "zero", "0.0")
    emit.constant("fp16", "one", "1.0")
    emit.constant("fp16", "minusTwo", "-2.0")

    begin = 0
    h_names: list[str] = []
    weight_nchw: list[str] = []
    weight_mm: list[str] = []
    for tile, count in enumerate(lengths):
        padded = pad32(count)
        if not zero_h and mode in {"complete", "hoist", "conv", "state_mm"}:
            h_names.append(
                emit.slice_tensor(
                    f"h{tile}", "b_h", [0, begin, 0, 0], [1, count, 1, 128]
                )
            )
        else:
            h_names.append("")
        if mode not in {"concat", "metro", "identity"} and (
            tile_index is None or tile == tile_index
        ):
            nchw = emit.reshape(f"w{tile}", f"d_j{tile}", [padded, channels, 1, 1])
            weight_nchw.append(nchw)
            if mode in {
                "complete",
                "hoist",
                "field_tiled",
                "field_full",
                "field_tile",
                "state_mm",
            }:
                if mode == "hoist":
                    weight_mm.append(
                        emit.matrix_reshape(f"wm{tile}", nchw, [1, 1, padded, channels])
                    )
                else:
                    weight_mm.append("")
            else:
                weight_mm.append("")
        else:
            weight_nchw.append("")
            weight_mm.append("")
        begin += count

    if mode == "field_full":
        emit.constant("int32", "axis0", "0")
        used = [name for name in weight_nchw if name]
        packed_rows = sum(pad32(lengths[tile]) for tile, _ in selected)
        concat_w = (
            used[0]
            if len(used) == 1
            else emit.emit(
                [packed_rows, channels, 1, 1],
                "wcat",
                f"concat(values=({', '.join(used)}), axis=axis0, interleave=interleave)",
            )
        )
        emit.matrix_reshape("wmfull", concat_w, [1, 1, packed_rows, channels])
        emit.matrix_reshape("smfull", "a_state", [1, 1, channels, 128])
        emit.constant("bool", "fullfalse", "false")
        emit.emit(
            [1, 1, packed_rows, 128],
            "mmfull",
            "matmul(x=wmfull, y=smfull, transpose_x=fullfalse, transpose_y=fullfalse)",
        )
        raw = emit.matrix_reshape("rawfull", "mmfull", [1, packed_rows, 1, 128])
        field_parts: list[str] = []
        cursor = 0
        for tile, count in selected:
            field_parts.append(
                emit.slice_tensor(
                    f"js{tile}", raw, [0, cursor, 0, 0], [1, count, 1, 128]
                )
            )
            cursor += pad32(count)
        return finish(
            inputs, emit, pack_fields(emit, field_parts, selected, channels), zero_h
        ), inputs

    state = "a_state"
    layout = "mm" if mode == "state_mm" else "nchw"
    working_shape = state_mm if layout == "mm" else state_nchw
    if mode == "state_mm":
        state = emit.matrix_reshape("state0", "a_state", state_mm)
    if mode in {"field_tiled", "field_tile"}:
        field_parts = []
        begin = 0
        source = "a_state"
        for tile, count in enumerate(lengths):
            if tile_index is not None and tile != tile_index:
                begin += count
                continue
            prefix = f"s0c{tile}"
            padded = pad32(count)
            weights = emit.matrix_reshape(
                prefix + "wm", weight_nchw[tile], [1, 1, padded, channels]
            )
            spins = emit.matrix_reshape(prefix + "sm", source, [1, 1, channels, 128])
            js = tile_matmul(emit, prefix, weights, spins, padded, channels, count)
            field_parts.append(js)
            begin += count
        if mode == "field_tile" and tile_index is not None:
            begin = sum(lengths[:tile_index])
            count = lengths[tile_index]
            js = field_parts[0]
            head = None
            if begin:
                head_src = emit.slice_tensor(
                    "headsrc", "a_state", [0, 0, 0, 0], [1, begin, 1, 128]
                )
                head = emit.emit(
                    [1, begin, 1, 128], "zhead", f"mul(x={head_src}, y=zero)"
                )
            tail_count = channels - begin - count
            tail = None
            if tail_count:
                tail_src = emit.slice_tensor(
                    "tailsrc",
                    "a_state",
                    [0, begin + count, 0, 0],
                    [1, tail_count, 1, 128],
                )
                tail = emit.emit(
                    [1, tail_count, 1, 128],
                    "ztail",
                    f"mul(x={tail_src}, y=zero)",
                )
            parts = [part for part in (head, js, tail) if part]
            out = (
                js
                if len(parts) == 1
                else emit.emit(
                    state_nchw,
                    "fieldout",
                    f"concat(values=({', '.join(parts)}), axis=axis, interleave=interleave)",
                )
            )
            return finish(inputs, emit, out, zero_h), inputs
        return finish(
            inputs, emit, pack_fields(emit, field_parts, selected, channels), zero_h
        ), inputs

    for sweep in range(sweeps):
        begin = 0
        for tile, count in enumerate(lengths):
            if tile_index is not None and tile != tile_index:
                begin += count
                continue
            prefix = f"s{sweep}c{tile}"
            padded = pad32(count)
            if layout == "nchw":
                own = emit.slice_tensor(
                    prefix + "own", state, [0, begin, 0, 0], [1, count, 1, 128]
                )
            else:
                own_mm = emit.slice_tensor(
                    prefix + "ownmm", state, [0, 0, begin, 0], [1, 1, count, 128]
                )
                own = emit.reshape(prefix + "own", own_mm, [1, count, 1, 128])
            if mode == "concat":
                updated = own
            elif mode == "metro":
                threshold = emit.slice_tensor(
                    prefix + "threshold",
                    "c_threshold",
                    [0, begin, 0, sweep * 4],
                    [1, count, 1, 4],
                )
                field = emit.slice_tensor(
                    prefix + "field",
                    "e_field",
                    [0, begin, 0, 0],
                    [1, count, 1, 128],
                )
                updated = metropolis(emit, prefix, own, field, threshold, count)
            else:
                threshold = emit.slice_tensor(
                    prefix + "threshold",
                    "c_threshold",
                    [0, begin, 0, sweep * 4],
                    [1, count, 1, 4],
                )
                if mode == "conv":
                    raw = emit.emit(
                        [1, padded, 1, 128],
                        prefix + "raw",
                        f"conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w{tile}, x={state})",
                    )
                    js = emit.slice_tensor(
                        prefix + "js", raw, [0, 0, 0, 0], [1, count, 1, 128]
                    )
                else:
                    if mode == "hoist":
                        weights = weight_mm[tile]
                    else:
                        source = weight_nchw[tile]
                        weights = emit.matrix_reshape(
                            prefix + "wm", source, [1, 1, padded, channels]
                        )
                    if layout == "mm":
                        spins = state
                    else:
                        spins = emit.matrix_reshape(
                            prefix + "sm", state, [1, 1, channels, 128]
                        )
                    js = tile_matmul(
                        emit, prefix, weights, spins, padded, channels, count
                    )
                field = (
                    js
                    if zero_h
                    else emit.emit(
                        [1, count, 1, 128],
                        prefix + "field",
                        f"add(x={js}, y={h_names[tile]})",
                    )
                )
                updated = metropolis(emit, prefix, own, field, threshold, count)
            if layout == "mm":
                updated = emit.reshape(prefix + "updmm", updated, [1, 1, count, 128])
            state = concat_state(
                emit,
                prefix,
                state,
                updated,
                begin,
                count,
                channels,
                working_shape,
                axis_name,
                layout,
            )
            begin += count

    if layout == "mm":
        state = emit.reshape("outstate", state, state_nchw)
    return finish(inputs, emit, state, zero_h), inputs


def pack_fields(
    emit: Emitter,
    parts: list[str],
    selected: list[tuple[int, int]],
    channels: int,
) -> str:
    packed = sum(count for _, count in selected)
    field = (
        parts[0]
        if len(parts) == 1
        else emit.emit(
            [1, packed, 1, 128],
            "fieldcat",
            f"concat(values=({', '.join(parts)}), axis=axis, interleave=interleave)",
        )
    )
    if packed >= channels:
        return field
    tail = emit.slice_tensor(
        "padsrc",
        "a_state",
        [0, packed, 0, 0],
        [1, channels - packed, 1, 128],
    )
    zpad = emit.emit([1, channels - packed, 1, 128], "zpad", f"mul(x={tail}, y=zero)")
    return emit.emit(
        [1, channels, 1, 128],
        "fieldout",
        f"concat(values=({field}, {zpad}), axis=axis, interleave=interleave)",
    )


def finish(
    inputs: list[tuple[str, list[int]]],
    emit: Emitter,
    state: str,
    zero_h: bool,
) -> str:
    header = (
        'program(1.3)\n[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}, {"profile-probe", "'
        + str(uuid.uuid4())
        + '"}})]\n{\n  func main<ios18>('
        + ", ".join(tensor(shape) + " " + name for name, shape in inputs)
        + ") {"
    )
    del zero_h
    return header + "\n" + "\n".join(emit.lines) + f"\n  }} -> ({state});\n}}\n"


def operator_counts(mil: str) -> dict[str, int]:
    counts = Counter(re.findall(r" = ([a-z_]+)\(", mil))
    return dict(sorted(counts.items()))


def bounds(channels: int, lengths: list[int], sweeps: int) -> dict[str, object]:
    padded = [pad32(count) for count in lengths]
    rows = int(sum(padded))
    mac_per_sweep = rows * channels * 128
    j_bytes = rows * channels * 2
    state_bytes = channels * 128 * 2
    color_bytes = [count * 128 * 2 for count in lengths]
    estimate = {
        "label": "estimate",
        "channels": channels,
        "lengths": lengths,
        "padded_rows": padded,
        "padded_row_sum": rows,
        "sweeps": sweeps,
        "mac_per_sweep": mac_per_sweep,
        "flop_mul_add_per_sweep": 2 * mac_per_sweep,
        "j_fp16_bytes": j_bytes,
        "state_fp16_bytes": state_bytes,
        "own_spin_bytes_per_sweep": int(sum(color_bytes)),
        "matmul_output_bytes_per_sweep": rows * 128 * 2,
        "concat_state_write_bytes_if_unfused": len(lengths) * state_bytes,
        "j_plus_state_read_bytes_per_sweep_if_unfused": j_bytes
        + len(lengths) * state_bytes,
        "full16k_sweeps": 16384,
        "full16k_mac": mac_per_sweep * 16384,
        "full16k_j_read_bytes_if_reused_every_sweep": j_bytes * 16384,
        "notes": [
            "MAC counts one fused multiply-add per inner product term.",
            "flop_mul_add counts multiply and add as two operations.",
            "Memory figures assume no compiler reuse and fp16 storage.",
            "J is one runtime tensor per color and does not change inside a job.",
        ],
    }
    return estimate


def load_meta(fixture: Path) -> FixtureMeta:
    return cast(FixtureMeta, json.loads((fixture / "meta.json").read_text()))


def packed_threshold(
    fixture: Path,
    meta: FixtureMeta,
    block_sweeps: int,
) -> tuple[Fp16, int]:
    channels = meta["channels"]
    sweeps = meta["sweeps"]
    iterations = (sweeps + block_sweeps - 1) // block_sweeps
    threshold = np.fromfile(fixture / meta["threshold_file"], dtype=np.float16).reshape(
        sweeps, channels, 4
    )
    padded = np.full((iterations * block_sweeps, channels, 4), -128, dtype=np.float16)
    padded[:sweeps] = threshold
    width = round_width(block_sweeps)
    packed = np.full((iterations, channels, width), -128, dtype=np.float16)
    packed[:, :, : block_sweeps * 4] = (
        padded.reshape(iterations, block_sweeps, channels, 4)
        .transpose(0, 2, 1, 3)
        .reshape(iterations, channels, block_sweeps * 4)
    )
    return packed, iterations


def load_state(fixture: Path, meta: FixtureMeta) -> Fp16:
    channels = meta["channels"]
    state = np.fromfile(fixture / meta["state_file"], dtype=np.float16)
    if state.size != channels * 128:
        raise ValueError("state element count mismatch")
    return state.reshape(channels, 128)


def load_h(fixture: Path, meta: FixtureMeta) -> Fp16:
    channels = meta["channels"]
    field = np.fromfile(fixture / meta["h_file"], dtype=np.float16)
    if field.size != channels * 128:
        raise ValueError("h element count mismatch")
    return field.reshape(channels, 128)


def load_tiles(fixture: Path, meta: FixtureMeta) -> list[Fp16]:
    channels = meta["channels"]
    tiles = []
    for tile in meta["tiles"]:
        count = tile["length"]
        padded = pad32(count)
        weights = np.fromfile(fixture / tile["weight_file"], dtype=np.float16)
        if weights.size != padded * channels:
            raise ValueError("J tile element count mismatch")
        tiles.append(weights.reshape(padded, channels))
    return tiles


def cpu_field(state: Fp16, tiles: list[Fp16], lengths: list[int]) -> Fp16:
    channels = state.shape[0]
    out = np.zeros((channels, 128), dtype=np.float16)
    begin = 0
    spins = np.rint(state.astype(np.float32)).astype(np.int32)
    for weights, count in zip(tiles, lengths, strict=True):
        coeff = np.rint(weights[:count].astype(np.float32)).astype(np.int32)
        product = coeff @ spins
        out[begin : begin + count] = product.astype(np.float16)
        begin += count
    return out


def cpu_metro_one_sweep(
    state: Fp16,
    field: Fp16,
    threshold: Fp16,
    lengths: list[int],
) -> Fp16:
    spins = state.copy()
    begin = 0
    for count in lengths:
        rows = slice(begin, begin + count)
        own = spins[rows].astype(np.float32)
        local = field[rows].astype(np.float32)
        grouped = threshold[rows].astype(np.float32)
        shifted = np.repeat(grouped, 32, axis=1) + own * local + 1.0
        accepted = np.clip(shifted, 0.0, 1.0)
        spins[rows] = (own * (1.0 + accepted * (-2.0))).astype(np.float16)
        begin += count
    return spins


def write_manifest(
    root: Path,
    mil: str,
    inputs: list[tuple[str, list[int]]],
    files: list[list[str]],
    expected: list[str],
    extra: dict[str, object],
) -> None:
    (root / "program.mil").write_text(mil)
    manifest: dict[str, object] = {
        "mil": "program.mil",
        "inputs": [
            {"name": name, "elements": int(np.prod(shape)), "files": entry}
            for (name, shape), entry in zip(inputs, files, strict=True)
        ],
        "output_elements": extra.get("output_elements", inputs[0][1][1] * 128),
        "expected": expected,
        "diagnostics": True,
        **{key: value for key, value in extra.items() if key != "output_elements"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "operators.json").write_text(
        json.dumps(operator_counts(mil), indent=2) + "\n"
    )


def prepare_complete(
    root: Path,
    fixtures: list[Path],
    block_sweeps: int | None,
    eval_repeats: int,
    mode: str,
) -> None:
    single_call.prepare_fixtures(root, fixtures, block_sweeps)
    meta = load_meta(fixtures[0])
    lengths = [tile["length"] for tile in meta["tiles"]]
    channels = meta["channels"]
    sweeps = block_sweeps or meta["sweeps"]
    zero_h = all(
        not np.any(load_h(fixture, load_meta(fixture))) for fixture in fixtures
    )
    if mode != "complete":
        mil, inputs = build_graph(channels, lengths, sweeps, mode, zero_h)
        (root / "program.mil").write_text(mil)
        (root / "operators.json").write_text(
            json.dumps(operator_counts(mil), indent=2) + "\n"
        )
        manifest = json.loads((root / "manifest.json").read_text())
        if [item["name"] for item in manifest["inputs"]] != [
            name for name, _ in inputs
        ]:
            raise ValueError("input names changed for equivalent mode")
    else:
        mil = (root / "program.mil").read_text()
        (root / "operators.json").write_text(
            json.dumps(operator_counts(mil), indent=2) + "\n"
        )
        manifest = json.loads((root / "manifest.json").read_text())
    manifest["eval_repeats"] = eval_repeats
    manifest["diagnostics"] = True
    manifest["profile_mode"] = mode
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "bounds.json").write_text(
        json.dumps(bounds(channels, lengths, sweeps), indent=2) + "\n"
    )


def prepare_ablation(
    root: Path,
    fixture: Path,
    mode: str,
    eval_repeats: int,
    tile_index: int | None,
) -> None:
    meta = load_meta(fixture)
    channels = meta["channels"]
    lengths = [tile["length"] for tile in meta["tiles"]]
    state = load_state(fixture, meta)
    tiles = load_tiles(fixture, meta)
    packed, _iterations = packed_threshold(fixture, meta, 1)
    zero_h = not np.any(load_h(fixture, meta))
    mil, inputs = build_graph(
        channels, lengths, 1, mode, zero_h=True, tile_index=tile_index
    )
    state_file = str((fixture / meta["state_file"]).resolve())
    file_map: dict[str, list[str]] = {"a_state": [state_file]}
    if mode == "metro":
        threshold_file = root / "threshold0.bin"
        packed[0].tofile(threshold_file)
        file_map["c_threshold"] = [str(threshold_file.resolve())]
        field = cpu_field(state, tiles, lengths)
        field_file = root / "field0.bin"
        field.astype(np.float16).tofile(field_file)
        file_map["e_field"] = [str(field_file.resolve())]
        expected_state = cpu_metro_one_sweep(state, field, packed[0][:, :4], lengths)
    elif mode == "concat":
        expected_state = state
    elif mode in {"field_tiled", "field_full"}:
        expected_state = cpu_field(state, tiles, lengths)
    elif mode == "field_tile":
        if tile_index is None:
            raise ValueError("field_tile requires --tile")
        expected_state = np.zeros_like(state)
        begin = sum(lengths[:tile_index])
        count = lengths[tile_index]
        expected_state[begin : begin + count] = cpu_field(state, tiles, lengths)[
            begin : begin + count
        ]
    elif mode == "identity":
        expected_state = state
    else:
        raise ValueError(f"unsupported ablation {mode}")
    for tile, tile_meta in enumerate(meta["tiles"]):
        name = f"d_j{tile}"
        if any(item[0] == name for item in inputs):
            file_map[name] = [str((fixture / tile_meta["weight_file"]).resolve())]
    files = [file_map[name] for name, _shape in inputs]
    expected_file = root / "expected0.bin"
    expected_state.astype(np.float16).tofile(expected_file)
    extra: dict[str, object] = {
        "iterations": 1,
        "block_sweeps": 1,
        "eval_repeats": eval_repeats,
        "fixed_io": True,
        "h_mode": "zero" if zero_h else "runtime",
        "profile_mode": mode,
        "output_elements": channels * 128,
    }
    names = [name for name, _shape in inputs]
    if "c_threshold" in names:
        extra["threshold_input_index"] = names.index("c_threshold")
    write_manifest(root, mil, inputs, files, [str(expected_file.resolve())], extra)
    (root / "bounds.json").write_text(
        json.dumps(bounds(channels, lengths, 1), indent=2) + "\n"
    )


def prepare_small(root: Path, sweeps: int, mode: str, eval_repeats: int) -> None:
    single_call.small_fixture(root, sweeps)
    if mode == "complete":
        mil = (root / "program.mil").read_text()
    else:
        mil, inputs = build_graph(32, [16, 16], sweeps, mode, zero_h=False)
        (root / "program.mil").write_text(mil)
        manifest = json.loads((root / "manifest.json").read_text())
        if [item["name"] for item in manifest["inputs"]] != [
            name for name, _ in inputs
        ]:
            raise ValueError("small-fixture input names changed")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["eval_repeats"] = eval_repeats
    manifest["diagnostics"] = True
    manifest["profile_mode"] = mode
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "operators.json").write_text(
        json.dumps(operator_counts((root / "program.mil").read_text()), indent=2) + "\n"
    )
    (root / "bounds.json").write_text(
        json.dumps(bounds(32, [16, 16], sweeps), indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "complete",
            "hoist",
            "conv",
            "state_mm",
            "field_tiled",
            "field_full",
            "field_tile",
            "concat",
            "metro",
            "identity",
            "small",
            "bounds",
        ],
    )
    parser.add_argument("--fixtures", type=Path, nargs="+")
    parser.add_argument("--block-sweeps", type=int)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--eval-repeats", type=int, default=1)
    parser.add_argument("--tile", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "bounds":
        if not args.fixtures:
            parser.error("bounds requires --fixtures")
        meta = load_meta(args.fixtures[0])
        lengths = [tile["length"] for tile in meta["tiles"]]
        payload = bounds(meta["channels"], lengths, meta["sweeps"])
        (args.output / "bounds.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload))
        return
    if args.mode == "small" or (
        args.mode in {"complete", "hoist", "conv", "state_mm"} and not args.fixtures
    ):
        mode = "complete" if args.mode == "small" else args.mode
        prepare_small(args.output, args.sweeps, mode, args.eval_repeats)
        return
    if not args.fixtures:
        parser.error("this mode requires --fixtures")
    if args.mode in {"complete", "hoist", "conv", "state_mm"}:
        prepare_complete(
            args.output, args.fixtures, args.block_sweeps, args.eval_repeats, args.mode
        )
        return
    if len(args.fixtures) != 1:
        parser.error("ablations use one fixture")
    prepare_ablation(
        args.output, args.fixtures[0], args.mode, args.eval_repeats, args.tile
    )


if __name__ == "__main__":
    main()
