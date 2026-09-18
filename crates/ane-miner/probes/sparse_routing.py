"""Compact source-block neighbor-slot routing for ANE MSA probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

Fp16 = npt.NDArray[np.float16]
Int32 = npt.NDArray[np.int32]
BenesOp = (
    tuple[Literal["swap"], list[int]] | tuple[Literal["shuffle", "unshuffle"], int]
)

DEGREE = 20
READS = 128


def tensor(shape: list[int]) -> str:
    return "tensor<fp16, [" + ",".join(map(str, shape)) + "]>"


def pad32(count: int) -> int:
    return (count + 31) // 32 * 32


def next_pow2(count: int) -> int:
    if count <= 1:
        return 1
    return 1 << (count - 1).bit_length()


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

    def transpose(
        self, name: str, source: str, perm: list[int], size: list[int]
    ) -> str:
        self.constant("tensor<int32, [4]>", name + "perm", str(perm))
        return self.emit(size, name, f"transpose(x={source}, perm={name}perm)")


def benes_decompose(
    src_of: list[int],
) -> tuple[list[int], list[int], list[int], list[int]]:
    n = len(src_of)
    if n < 2 or n & (n - 1):
        raise ValueError("Beneš size must be a power of two")
    half = n // 2
    perm_dst = [0] * n
    for dest, source in enumerate(src_of):
        perm_dst[source] = dest
    subnet = [-1] * n

    def assign(start: int, color: int) -> None:
        stack = [(start, color)]
        while stack:
            source, bit = stack.pop()
            if subnet[source] == bit:
                continue
            if subnet[source] != -1:
                raise ValueError("Beneš coloring conflict")
            subnet[source] = bit
            stack.append((source ^ 1, bit ^ 1))
            stack.append((src_of[perm_dst[source] ^ 1], bit ^ 1))

    for source in range(n):
        if subnet[source] == -1:
            assign(source, 0)
    in_swap = [0] * half
    out_swap = [0] * half
    upper = [0] * half
    lower = [0] * half
    for source in range(n):
        bit = subnet[source]
        if (source % 2) != bit:
            in_swap[source // 2] = 1
        dest = perm_dst[source]
        if (dest % 2) != bit:
            out_swap[dest // 2] = 1
        inner_src = source // 2
        inner_dst = dest // 2
        if bit == 0:
            upper[inner_dst] = inner_src
        else:
            lower[inner_dst] = inner_src
    return in_swap, out_swap, upper, lower


def benes_ops(src_of: list[int]) -> list[BenesOp]:
    n = len(src_of)
    if n == 1:
        return []
    if n == 2:
        return [("swap", [int(src_of[0] == 1)])]
    in_swap, out_swap, upper, lower = benes_decompose(src_of)
    merged: list[BenesOp] = []
    for up, low in zip(benes_ops(upper), benes_ops(lower), strict=True):
        if up[0] != low[0]:
            raise ValueError("Beneš half mismatch")
        if up[0] == "swap" and low[0] == "swap":
            merged.append(("swap", up[1] + low[1]))
        elif up[0] in {"shuffle", "unshuffle"}:
            merged.append(up)
        else:
            raise ValueError("Beneš half mismatch")
    return (
        [("swap", in_swap), ("shuffle", n)]
        + merged
        + [("unshuffle", n), ("swap", out_swap)]
    )


def apply_benes(
    values: npt.NDArray[np.int32], ops: list[BenesOp]
) -> npt.NDArray[np.int32]:
    data = values.copy()
    length = data.size
    for op in ops:
        if op[0] == "swap":
            bits = np.asarray(op[1], dtype=np.int32)
            even = data[0::2].copy()
            odd = data[1::2].copy()
            swap = bits.astype(bool)
            data[0::2] = np.where(swap, odd, even)
            data[1::2] = np.where(swap, even, odd)
        elif op[0] == "shuffle":
            block = op[1]
            shaped = data.reshape(length // block, block)
            data = np.concatenate([shaped[:, 0::2], shaped[:, 1::2]], axis=1).reshape(
                length
            )
        else:
            block = op[1]
            shaped = data.reshape(length // block, block)
            half = block // 2
            even = shaped[:, :half].copy()
            odd = shaped[:, half:].copy()
            shaped[:, 0::2] = even
            shaped[:, 1::2] = odd
            data = shaped.reshape(length)
    return data


def self_test_benes() -> None:
    rng = np.random.default_rng(20260915)
    for size in (2, 4, 8, 16, 32, 64, 256):
        for _ in range(8):
            src_of = rng.permutation(size).tolist()
            ops = benes_ops(src_of)
            got = apply_benes(np.arange(size, dtype=np.int32), ops)
            assert got.tolist() == src_of, (size, src_of, got.tolist())
        identity = list(range(size))
        assert (
            apply_benes(np.arange(size, dtype=np.int32), benes_ops(identity)).tolist()
            == identity
        )
        reverse = list(range(size - 1, -1, -1))
        assert (
            apply_benes(np.arange(size, dtype=np.int32), benes_ops(reverse)).tolist()
            == reverse
        )


def pack_ell(neighbors: list[list[int]], degree: int = DEGREE) -> tuple[Int32, Int32]:
    count = len(neighbors)
    ids = np.zeros((count, degree), dtype=np.int32)
    live = np.zeros(count, dtype=np.int32)
    for row, srcs in enumerate(neighbors):
        if len(srcs) > degree:
            raise ValueError("degree exceeds slot count")
        ids[row, : len(srcs)] = srcs
        live[row] = len(srcs)
    return ids, live


def oracle_field(spins: Fp16, sources: Int32, edge_j: Fp16, field_h: Fp16) -> Fp16:
    routed = spins[sources]
    products = routed * edge_j[:, :, None]
    return products.sum(axis=1, dtype=np.float16) + field_h


def oracle_metropolis(own: Fp16, field: Fp16, threshold: Fp16) -> Fp16:
    margin = own * field
    accept = np.clip(margin + np.repeat(threshold, 32, axis=1) + 1, 0, 1)
    return own * (1 - 2 * accept)


def oracle_sweeps(
    spins: Fp16,
    field_h: Fp16,
    thresholds: Fp16,
    sources: list[Int32],
    edges: list[Fp16],
    lengths: list[int],
    sweeps: int,
) -> Fp16:
    state = spins.copy()
    for sweep in range(sweeps):
        begin = 0
        for tile, count in enumerate(lengths):
            own = state[begin : begin + count]
            field = oracle_field(
                state, sources[tile], edges[tile], field_h[begin : begin + count]
            )
            state[begin : begin + count] = oracle_metropolis(
                own, field, thresholds[sweep, begin : begin + count]
            )
            begin += count
    return state


def load_union_neighbors(
    fixtures: list[Path],
) -> tuple[dict[str, Any], list[list[int]], list[int]]:
    metas = [json.loads((path / "meta.json").read_text()) for path in fixtures]
    first = metas[0]
    channels = int(first["channels"])
    nodes = int(first["nodes"])
    lengths = [int(tile["length"]) for tile in first["tiles"]]
    union: list[set[int]] = [set() for _ in range(nodes)]
    for fixture, meta in zip(fixtures, metas, strict=True):
        if meta["channels"] != channels or meta["order"] != first["order"]:
            raise ValueError("fixture topology mismatch")
        begin = 0
        for tile_meta in meta["tiles"]:
            weights = np.fromfile(
                fixture / tile_meta["weight_file"], dtype=np.float16
            ).reshape(-1, channels)[: tile_meta["length"]]
            for row in range(weights.shape[0]):
                srcs = np.flatnonzero(weights[row, :nodes] != 0)
                union[begin + row].update(int(src) for src in srcs)
            begin += int(tile_meta["length"])
    neighbors = [sorted(srcs) for srcs in union]
    return first, neighbors, lengths


def fill_edge_j(weights: Fp16, ids: Int32, live_counts: Int32) -> Fp16:
    values = np.zeros((ids.shape[0], DEGREE), dtype=np.float16)
    for row in range(ids.shape[0]):
        for slot in range(int(live_counts[row])):
            values[row, slot] = weights[row, ids[row, slot]]
    return values


class ColorRoute:
    def __init__(
        self,
        tile: int,
        begin: int,
        count: int,
        sources: Int32,
        live: Int32,
        block: int,
        channels: int,
    ) -> None:
        self.tile = tile
        self.begin = begin
        self.count = count
        self.sources = sources
        self.live = live
        self.block = block
        self.groups = channels // block
        self.group_edges: list[list[tuple[int, int, int]]] = [
            [] for _ in range(self.groups)
        ]
        for row in range(count):
            for slot in range(int(live[row])):
                src = int(sources[row, slot])
                self.group_edges[src // block].append((row, slot, src))
        max_e = max((len(edges) for edges in self.group_edges), default=0)
        self.epad = pad32(max(max_e, 1))
        self.extract_n = self.groups * self.epad
        self.dest_n = count * DEGREE
        self.route_n = next_pow2(max(self.extract_n, self.dest_n, 2))
        self.src_of = [-1] * self.route_n
        used_inputs: set[int] = set()
        for group, edges in enumerate(self.group_edges):
            for local, (row, slot, _src) in enumerate(edges):
                extract = group * self.epad + local
                self.src_of[row * DEGREE + slot] = extract
                used_inputs.add(extract)
        unused_in = [index for index in range(self.route_n) if index not in used_inputs]
        unused_out = [index for index, source in enumerate(self.src_of) if source < 0]
        if len(unused_out) != len(unused_in):
            raise ValueError("Beneš pad mismatch")
        for dest, source in zip(unused_out, unused_in, strict=True):
            self.src_of[dest] = source
        if sorted(self.src_of) != list(range(self.route_n)):
            raise ValueError("Beneš mapping is not a permutation")
        self.ops = benes_ops(self.src_of)
        self.selector = np.zeros((self.extract_n, block), dtype=np.float16)
        for group, edges in enumerate(self.group_edges):
            for local, (_row, _slot, src) in enumerate(edges):
                self.selector[group * self.epad + local, src % block] = 1

    def swap_masks(self) -> list[Fp16]:
        masks: list[Fp16] = []
        for op in self.ops:
            if op[0] != "swap":
                continue
            bits = np.asarray(op[1], dtype=np.float16).reshape(
                1, self.route_n // 2, 1, 1
            )
            masks.append(bits)
        return masks


def routing_cpu_field(
    spins: Fp16, route: ColorRoute, edge_j: Fp16, field_h: Fp16, channels: int
) -> Fp16:
    blocked = np.zeros((route.extract_n, READS), dtype=np.float16)
    padded = np.zeros((route.groups * route.block, READS), dtype=np.float16)
    use = min(spins.shape[0], route.groups * route.block)
    padded[:use] = spins[:use]
    grouped = padded.reshape(route.groups, route.block, READS)
    for group in range(route.groups):
        weights = route.selector[group * route.epad : (group + 1) * route.epad]
        if not np.any(weights):
            continue
        with np.errstate(all="ignore"):
            blocked[group * route.epad : (group + 1) * route.epad] = np.matmul(
                weights.astype(np.float32), grouped[group].astype(np.float32)
            ).astype(np.float16)
    routed = np.zeros((route.route_n, READS), dtype=np.float16)
    routed[: route.extract_n] = blocked
    order = apply_benes(np.arange(route.route_n, dtype=np.int32), route.ops)
    dest_major = routed[order][: route.dest_n].reshape(route.count, DEGREE, READS)
    products = dest_major * edge_j[:, :, None]
    return products.sum(axis=1, dtype=np.float16) + field_h


def write_blob(
    path: Path, chunks: list[tuple[str, npt.NDArray[np.float16]]]
) -> list[dict[str, int]]:
    layout: list[dict[str, int]] = []
    with path.open("wb") as blob:
        header = bytearray(64)
        struct.pack_into("<II", header, 0, len(chunks), 2)
        blob.write(header)
        for name, array in chunks:
            payload = np.ascontiguousarray(array, dtype=np.float16).tobytes()
            offset = blob.tell()
            chunk = bytearray(64)
            struct.pack_into(
                "<IIQQ", chunk, 0, 0xDEADBEEF, 1, len(payload), offset + 64
            )
            blob.write(chunk)
            blob.write(payload)
            layout.append(
                {
                    "header_offset": offset,
                    "data_offset": offset + 64,
                    "bytes": len(payload),
                }
            )
            _ = name
    return layout


def emit_pair_swap(
    emit: Emitter, prefix: str, source: str, mask: str, length: int
) -> str:
    paired = emit.reshape(prefix + "pair", source, [1, length // 2, 2, READS])
    even = emit.slice_tensor(
        prefix + "even", paired, [0, 0, 0, 0], [1, length // 2, 1, READS]
    )
    odd = emit.slice_tensor(
        prefix + "odd", paired, [0, 0, 1, 0], [1, length // 2, 1, READS]
    )
    keep = emit.emit([1, length // 2, 1, 1], prefix + "keep", f"sub(x=one, y={mask})")
    ekeep = emit.emit(
        [1, length // 2, 1, READS], prefix + "ek", f"mul(x={even}, y={keep})"
    )
    oswap = emit.emit(
        [1, length // 2, 1, READS], prefix + "os", f"mul(x={odd}, y={mask})"
    )
    okeep = emit.emit(
        [1, length // 2, 1, READS], prefix + "ok", f"mul(x={odd}, y={keep})"
    )
    eswap = emit.emit(
        [1, length // 2, 1, READS], prefix + "es", f"mul(x={even}, y={mask})"
    )
    out0 = emit.emit(
        [1, length // 2, 1, READS], prefix + "o0", f"add(x={ekeep}, y={oswap})"
    )
    out1 = emit.emit(
        [1, length // 2, 1, READS], prefix + "o1", f"add(x={okeep}, y={eswap})"
    )
    joined = emit.emit(
        [1, length // 2, 2, READS],
        prefix + "join",
        f"concat(values=({out0}, {out1}), axis=axis2, interleave=interleave)",
    )
    return emit.reshape(prefix + "swapped", joined, [1, length, 1, READS])


def emit_shuffle(
    emit: Emitter, prefix: str, source: str, length: int, block: int, inverse: bool
) -> str:
    blocks = length // block
    half = block // 2
    if inverse:
        folded = emit.reshape(prefix + "fold", source, [blocks, 2, half, READS])
        swapped = emit.transpose(
            prefix + "tr", folded, [0, 2, 1, 3], [blocks, half, 2, READS]
        )
    else:
        folded = emit.reshape(prefix + "fold", source, [blocks, half, 2, READS])
        swapped = emit.transpose(
            prefix + "tr", folded, [0, 2, 1, 3], [blocks, 2, half, READS]
        )
    return emit.reshape(prefix + "shuf", swapped, [1, length, 1, READS])


def emit_benes(
    emit: Emitter, prefix: str, source: str, route: ColorRoute, mask_names: list[str]
) -> str:
    current = source
    mask_index = 0
    for step, op in enumerate(route.ops):
        name = f"{prefix}b{step}"
        if op[0] == "swap":
            current = emit_pair_swap(
                emit, name, current, mask_names[mask_index], route.route_n
            )
            mask_index += 1
        elif op[0] == "shuffle":
            current = emit_shuffle(emit, name, current, route.route_n, op[1], False)
        else:
            current = emit_shuffle(emit, name, current, route.route_n, op[1], True)
    return current


def emit_metropolis(
    emit: Emitter, prefix: str, own: str, field: str, threshold: str, count: int
) -> str:
    shape = [1, count, 1, READS]
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


def emit_concat_state(
    emit: Emitter,
    prefix: str,
    state: str,
    updated: str,
    begin: int,
    count: int,
    channels: int,
) -> str:
    parts: list[str] = []
    if begin:
        parts.append(
            emit.slice_tensor(
                prefix + "head", state, [0, 0, 0, 0], [1, begin, 1, READS]
            )
        )
    parts.append(updated)
    if begin + count < channels:
        parts.append(
            emit.slice_tensor(
                prefix + "tail",
                state,
                [0, begin + count, 0, 0],
                [1, channels - begin - count, 1, READS],
            )
        )
    if len(parts) == 1:
        return updated
    return emit.emit(
        [1, channels, 1, READS],
        prefix + "state",
        f"concat(values=({', '.join(parts)}), axis=axis, interleave=interleave)",
    )


def build_program(
    channels: int,
    lengths: list[int],
    routes: list[ColorRoute],
    sweeps: int,
    zero_h: bool,
    color_field: int | None,
    blob_layout: list[dict[str, Any]],
) -> tuple[str, list[tuple[str, list[int]]]]:
    emit = Emitter()
    width = pad32(sweeps * 4)
    state_shape = [1, channels, 1, READS]
    inputs: list[tuple[str, list[int]]] = [("a_state", state_shape)]
    if not zero_h:
        inputs.append(("b_h", state_shape))
    selected = (
        [(color_field, lengths[color_field])]
        if color_field is not None
        else list(enumerate(lengths))
    )
    if color_field is None:
        inputs.append(("c_threshold", [1, channels, 1, width]))
    for tile, count in selected:
        inputs.append((f"d_j{tile}", [1, count, 1, 32]))
    emit.constant("string", "pt", '"valid"')
    emit.constant("tensor<int32, [2]>", "st", "[1,1]")
    emit.constant("tensor<int32, [4]>", "pd", "[0,0,0,0]")
    emit.constant("tensor<int32, [2]>", "dl", "[1,1]")
    emit.constant("bool", "interleave", "false")
    emit.constant("int32", "axis", "1")
    emit.constant("int32", "axis2", "2")
    emit.constant("fp16", "zero", "0.0")
    emit.constant("fp16", "one", "1.0")
    emit.constant("fp16", "minusTwo", "-2.0")
    emit.constant("tensor<int32, [1]>", "sum_axes", "[2]")
    emit.constant("bool", "keep_dims", "true")
    chunk = 0
    mask_names: dict[int, list[str]] = {}
    for tile, _count in selected:
        route = routes[tile]
        emit.constant("int32", f"gr{tile}", str(route.groups))
        emit.constant(
            tensor([route.extract_n, route.block, 1, 1]),
            f"sel{tile}",
            'BLOBFILE(path=string("@model_path/weights/weight_data.bin"), '
            f"offset=uint64({blob_layout[chunk]['header_offset']}))",
        )
        chunk += 1
        names: list[str] = []
        for mask_i, _mask in enumerate(route.swap_masks()):
            names.append(
                emit.constant(
                    tensor([1, route.route_n // 2, 1, 1]),
                    f"msk{tile}_{mask_i}",
                    'BLOBFILE(path=string("@model_path/weights/weight_data.bin"), '
                    f"offset=uint64({blob_layout[chunk]['header_offset']}))",
                )
            )
            chunk += 1
        mask_names[tile] = names
        if route.route_n > route.extract_n:
            emit.constant(
                tensor([1, route.route_n - route.extract_n, 1, READS]),
                f"zpad{tile}",
                'BLOBFILE(path=string("@model_path/weights/weight_data.bin"), '
                f"offset=uint64({blob_layout[chunk]['header_offset']}))",
            )
            chunk += 1
    h_names: dict[int, str] = {}
    begin = 0
    for tile, count in enumerate(lengths):
        if not zero_h and (color_field is None or tile == color_field):
            h_names[tile] = emit.slice_tensor(
                f"h{tile}", "b_h", [0, begin, 0, 0], [1, count, 1, READS]
            )
        begin += count
    if color_field is not None:
        tile, count = selected[0]
        route = routes[tile]
        field = emit_color_field(
            emit, "s0", "a_state", route, mask_names[tile], h_names.get(tile), zero_h
        )
        header = program_header(inputs)
        return header + "\n" + "\n".join(
            emit.lines
        ) + f"\n  }} -> ({field});\n}}\n", inputs

    state = "a_state"
    for sweep in range(sweeps):
        begin = 0
        for tile, count in enumerate(lengths):
            route = routes[tile]
            prefix = f"s{sweep}c{tile}"
            own = emit.slice_tensor(
                prefix + "own", state, [0, begin, 0, 0], [1, count, 1, READS]
            )
            threshold = emit.slice_tensor(
                prefix + "threshold",
                "c_threshold",
                [0, begin, 0, sweep * 4],
                [1, count, 1, 4],
            )
            field = emit_color_field(
                emit, prefix, state, route, mask_names[tile], h_names.get(tile), zero_h
            )
            updated = emit_metropolis(emit, prefix, own, field, threshold, count)
            state = emit_concat_state(
                emit, prefix, state, updated, begin, count, channels
            )
            begin += count
    header = program_header(inputs)
    return header + "\n" + "\n".join(emit.lines) + f"\n  }} -> ({state});\n}}\n", inputs


def program_header(inputs: list[tuple[str, list[int]]]) -> str:
    listed = ", ".join(tensor(shape) + " " + name for name, shape in inputs)
    return (
        "program(1.3)\n[buildInfo = dict<string, string>({{"
        '"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, '
        '{"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}, '
        '{"single-call-probe", "PROBE_IDENTITY"}})]\n{\n  func main<ios18>('
        + listed
        + ") {"
    )


def emit_color_field(
    emit: Emitter,
    prefix: str,
    state: str,
    route: ColorRoute,
    mask_names: list[str],
    h_name: str | None,
    zero_h: bool,
) -> str:
    tile = route.tile
    extracted = emit.emit(
        [1, route.extract_n, 1, READS],
        prefix + "ext",
        f"conv(dilations=dl, groups=gr{tile}, pad=pd, pad_type=pt, strides=st, "
        f"weight=sel{tile}, x={state})",
    )
    if route.route_n > route.extract_n:
        extracted = emit.emit(
            [1, route.route_n, 1, READS],
            prefix + "pad",
            f"concat(values=({extracted}, zpad{tile}), axis=axis, interleave=interleave)",
        )
    permuted = emit_benes(emit, prefix, extracted, route, mask_names)
    dest = emit.slice_tensor(
        prefix + "dest",
        permuted,
        [0, 0, 0, 0],
        [1, route.dest_n, 1, READS],
    )
    neighbors = emit.reshape(prefix + "nb", dest, [1, route.count, DEGREE, READS])
    jflat = emit.slice_tensor(
        prefix + "jflat",
        f"d_j{tile}",
        [0, 0, 0, 0],
        [1, route.count, 1, DEGREE],
    )
    jw = emit.reshape(prefix + "j", jflat, [1, route.count, DEGREE, 1])
    products = emit.emit(
        [1, route.count, DEGREE, READS], prefix + "prod", f"mul(x={neighbors}, y={jw})"
    )
    summed = emit.emit(
        [1, route.count, 1, READS],
        prefix + "sum",
        "reduce_sum(x=" + products + ", axes=sum_axes, keep_dims=keep_dims)",
    )
    if zero_h or h_name is None:
        return summed
    return emit.emit(
        [1, route.count, 1, READS], prefix + "field", f"add(x={summed}, y={h_name})"
    )


def packed_j(values: Fp16) -> Fp16:
    packed = np.zeros((values.shape[0], 32), dtype=np.float16)
    packed[:, :DEGREE] = values
    return packed


def small_graph() -> tuple[
    list[int], list[list[int]], list[Fp16], list[Fp16], Fp16, Fp16, Fp16
]:
    lengths = [16, 16]
    neighbors: list[list[int]] = [[] for _ in range(32)]
    for node in range(16):
        for k in range(4):
            left = node
            right = 16 + ((node + k) % 16)
            neighbors[left].append(right)
            neighbors[right].append(left)
    rng = np.random.default_rng(911)
    spins = rng.choice([-1, 1], size=(32, READS)).astype(np.float16)
    thresholds = rng.integers(0, 4, size=(2, 32, 4)).astype(np.float16)
    jobs: list[Fp16] = []
    fields: list[Fp16] = []
    for job in range(2):
        sign = np.float16(1 - 2 * job)
        edges = []
        for srcs in neighbors:
            row = np.zeros(DEGREE, dtype=np.float16)
            row[: len(srcs)] = sign
            edges.append(row)
        jobs.append(np.stack(edges))
        fields.append(np.full((32, READS), sign, dtype=np.float16))
    return lengths, neighbors, jobs, fields, spins, thresholds, fields[0]


def prepare_output(
    output: Path,
    channels: int,
    lengths: list[int],
    routes: list[ColorRoute],
    sources: list[Int32],
    sweeps: int,
    block_sweeps: int,
    jobs: list[dict[str, Any]],
    zero_h: bool,
    color_field: int | None,
    compile_only: bool,
) -> dict[str, Any]:
    chunks: list[tuple[str, npt.NDArray[np.float16]]] = []
    selected = [color_field] if color_field is not None else list(range(len(lengths)))
    for tile in selected:
        route = routes[tile]
        chunks.append(
            (f"sel{tile}", route.selector.reshape(route.extract_n, route.block, 1, 1))
        )
        for index, mask in enumerate(route.swap_masks()):
            chunks.append((f"msk{tile}_{index}", mask))
        if route.route_n > route.extract_n:
            pad = np.zeros(
                (1, route.route_n - route.extract_n, 1, READS), dtype=np.float16
            )
            chunks.append((f"zpad{tile}", pad))
    layout = write_blob(output / "weight_data.bin", chunks)
    program, inputs = build_program(
        channels, lengths, routes, block_sweeps, zero_h, color_field, layout
    )
    identity = hashlib.sha256(program.encode()).hexdigest()
    program = program.replace("PROBE_IDENTITY", identity)
    (output / "program.mil").write_text(program)
    files: list[list[str]] = [[] for _ in inputs]
    expected_files: list[str] = []
    receipts: list[dict[str, Any]] = []
    width = pad32(block_sweeps * 4)
    iterations = (
        1 if color_field is not None else (sweeps + block_sweeps - 1) // block_sweeps
    )
    for job_index, job in enumerate(jobs):
        state = job["state"]
        field_h = job["h"]
        thresholds = job["thresholds"]
        edges = job["edges"]
        for tile, route in enumerate(routes):
            live = route.live
            for row in range(route.count):
                if np.any(edges[tile][row, int(live[row]) :] != 0):
                    raise AssertionError("padding J is not zero")
        if job_index == 1 and np.array_equal(edges[0], jobs[0]["edges"][0]):
            raise AssertionError("distinct jobs reused the same J")
        files[0].append(str(job["state_path"]))
        cursor = 1
        if not zero_h:
            files[cursor].append(str(job["h_path"]))
            cursor += 1
        if color_field is None:
            padded = np.full((iterations, channels, width), -128, dtype=np.float16)
            used = np.full(
                (iterations * block_sweeps, channels, 4), -128, dtype=np.float16
            )
            used[:sweeps] = thresholds[:sweeps]
            packed = (
                used.reshape(iterations, block_sweeps, channels, 4)
                .transpose(0, 2, 1, 3)
                .reshape(iterations, channels, block_sweeps * 4)
            )
            padded[:, :, : block_sweeps * 4] = packed
            threshold_path = output / f"threshold{job_index}.bin"
            padded.tofile(threshold_path)
            files[cursor].append(str(threshold_path))
            cursor += 1
        for tile in selected:
            packed = packed_j(edges[tile])
            path = output / f"edge-j{job_index}-tile{tile}.bin"
            packed.tofile(path)
            files[cursor].append(str(path))
            cursor += 1
        if color_field is not None:
            route = routes[color_field]
            begin = route.begin
            actual = routing_cpu_field(
                state,
                route,
                edges[color_field],
                field_h[begin : begin + route.count],
                channels,
            )
            independent = oracle_field(
                state,
                sources[color_field],
                edges[color_field],
                field_h[begin : begin + route.count],
            )
            mismatches = int(np.count_nonzero(actual != independent))
            if mismatches:
                raise AssertionError(f"routing oracle mismatch job {job_index}")
            path = output / f"expected{job_index}.bin"
            independent.astype(np.float16).tofile(path)
            expected_files.append(str(path))
            receipts.append(
                {"job": job_index, "mismatches": mismatches, "kind": "color_field"}
            )
        else:
            routed = state.copy()
            begin_h = field_h
            for sweep in range(sweeps):
                begin = 0
                for tile, count in enumerate(lengths):
                    own = routed[begin : begin + count]
                    field = routing_cpu_field(
                        routed,
                        routes[tile],
                        edges[tile],
                        begin_h[begin : begin + count],
                        channels,
                    )
                    independent = oracle_field(
                        routed,
                        sources[tile],
                        edges[tile],
                        begin_h[begin : begin + count],
                    )
                    if int(np.count_nonzero(field != independent)):
                        raise AssertionError("per-color routing mismatch")
                    routed[begin : begin + count] = oracle_metropolis(
                        own, independent, thresholds[sweep, begin : begin + count]
                    )
                    begin += count
            independent_state = oracle_sweeps(
                state, field_h, thresholds, sources, edges, lengths, sweeps
            )
            mismatches = int(np.count_nonzero(routed != independent_state))
            if mismatches:
                raise AssertionError(f"sweep oracle mismatch job {job_index}")
            saved = job.get("expected")
            if saved is not None and sweeps == job["meta_sweeps"]:
                errors = int(np.count_nonzero(independent_state != saved))
                if errors:
                    raise AssertionError(
                        f"fixture mismatch job {job_index} errors={errors}"
                    )
                expected_files.append(str(job["expected_path"]))
            else:
                path = output / f"expected{job_index}.bin"
                independent_state.astype(np.float16).tofile(path)
                expected_files.append(str(path))
            receipts.append({"job": job_index, "mismatches": 0, "sweeps": sweeps})
    names = [name for name, _shape in inputs]
    manifest: dict[str, Any] = {
        "mil": "program.mil",
        "weights": "weight_data.bin",
        "inputs": [
            {"name": name, "elements": int(np.prod(shape)), "files": files[index]}
            for index, (name, shape) in enumerate(inputs)
        ],
        "expected": expected_files,
        "output_elements": (
            routes[color_field].count * READS
            if color_field is not None
            else channels * READS
        ),
        "h_mode": "zero" if zero_h else "runtime",
        "routing": "source-block-grouped-benes",
        "block": routes[0].block,
        "packed_replicas": False,
    }
    if color_field is None:
        manifest["iterations"] = iterations
        manifest["block_sweeps"] = block_sweeps
        manifest["total_sweeps"] = sweeps
        manifest["threshold_input_index"] = names.index("c_threshold")
    if compile_only:
        manifest["compile_only"] = True
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    proof = {
        "jobs": receipts,
        "program_bytes": len(program),
        "weight_bytes": (output / "weight_data.bin").stat().st_size,
        "oracle_independent": True,
        "packed_replicas": False,
        "topology": "fixture-union-edges",
        "color_field": color_field,
        "sweeps": sweeps if color_field is None else 0,
        "extract_bytes": int(
            sum(
                route.selector.nbytes
                for tile, route in enumerate(routes)
                if tile in selected
            )
        ),
        "route_sizes": [routes[tile].route_n for tile in selected],
        "groups": [routes[tile].groups for tile in selected],
        "epad": [routes[tile].epad for tile in selected],
    }
    (output / "cpu-proof.json").write_text(json.dumps(proof, indent=2))
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fixtures", type=Path, nargs="+")
    parser.add_argument("--sweeps", type=int)
    parser.add_argument("--block-sweeps", type=int)
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--color-field", type=int)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test_benes()
        print(json.dumps({"benes_self_test": "ok"}))
        return
    self_test_benes()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.fixtures:
        first, neighbors, lengths = load_union_neighbors(args.fixtures)
        channels = int(first["channels"])
        if channels % args.block:
            raise ValueError("block does not divide padded channels")
        sources: list[Int32] = []
        lives: list[Int32] = []
        begin = 0
        routes: list[ColorRoute] = []
        for tile, count in enumerate(lengths):
            ids, live = pack_ell(neighbors[begin : begin + count])
            sources.append(ids)
            lives.append(live)
            routes.append(
                ColorRoute(tile, begin, count, ids, live, args.block, channels)
            )
            begin += count
        jobs = []
        zero_h = True
        for fixture in args.fixtures:
            meta = json.loads((fixture / "meta.json").read_text())
            state = np.fromfile(fixture / meta["state_file"], dtype=np.float16).reshape(
                channels, READS
            )
            field_h = np.fromfile(fixture / meta["h_file"], dtype=np.float16).reshape(
                channels, READS
            )
            if np.any(field_h):
                zero_h = False
            thresholds = np.fromfile(
                fixture / meta["threshold_file"], dtype=np.float16
            ).reshape(meta["sweeps"], channels, 4)
            edges = []
            begin = 0
            for tile, tile_meta in enumerate(meta["tiles"]):
                weights = np.fromfile(
                    fixture / tile_meta["weight_file"], dtype=np.float16
                ).reshape(-1, channels)[: tile_meta["length"]]
                edges.append(fill_edge_j(weights, sources[tile], lives[tile]))
                begin += lengths[tile]
            expected = np.fromfile(
                fixture / meta["expected_file"], dtype=np.float16
            ).reshape(channels, READS)
            jobs.append(
                {
                    "state": state,
                    "h": field_h,
                    "thresholds": thresholds,
                    "edges": edges,
                    "expected": expected,
                    "state_path": str((fixture / meta["state_file"]).resolve()),
                    "h_path": str((fixture / meta["h_file"]).resolve()),
                    "expected_path": str((fixture / meta["expected_file"]).resolve()),
                    "meta_sweeps": int(meta["sweeps"]),
                }
            )
        sweeps = int(first["sweeps"] if args.sweeps is None else args.sweeps)
        block_sweeps = sweeps if args.block_sweeps is None else args.block_sweeps
    else:
        lengths, neighbors, job_edges, job_h, spins, thresholds, _ = small_graph()
        channels = 32
        if channels % args.block:
            args.block = 16
        sources = []
        lives: list[Int32] = []
        routes = []
        begin = 0
        for tile, count in enumerate(lengths):
            ids, live = pack_ell(neighbors[begin : begin + count])
            sources.append(ids)
            lives.append(live)
            routes.append(
                ColorRoute(tile, begin, count, ids, live, args.block, channels)
            )
            begin += count
        jobs = []
        for job in range(2):
            state_path = output / f"state{job}.bin"
            h_path = output / f"h{job}.bin"
            spins.astype(np.float16).tofile(state_path)
            job_h[job].astype(np.float16).tofile(h_path)
            jobs.append(
                {
                    "state": spins,
                    "h": job_h[job],
                    "thresholds": thresholds,
                    "edges": [job_edges[job][:16], job_edges[job][16:]],
                    "state_path": str(state_path),
                    "h_path": str(h_path),
                    "expected": None,
                    "meta_sweeps": 2,
                }
            )
        sweeps = 2 if args.sweeps is None else args.sweeps
        block_sweeps = sweeps if args.block_sweeps is None else args.block_sweeps
        zero_h = False
    proof = prepare_output(
        output,
        channels,
        lengths,
        routes,
        sources,
        sweeps,
        block_sweeps,
        jobs,
        zero_h,
        args.color_field,
        args.compile_only,
    )
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
