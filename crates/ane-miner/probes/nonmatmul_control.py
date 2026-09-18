"""Replace frozen sparse gather with fixed convolution or static slices."""

import argparse
import ast
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np


def tensor(shape):
    return "tensor<fp16, [" + ",".join(map(str, shape)) + "]>"


def constant(dtype, name, value):
    return f'    {dtype} {name} = const()[name=string("{name}"), val={dtype}({value})];'


def emit(shape, name, expression):
    return f'    {tensor(shape)} {name} = {expression}[name=string("{name}")];'


def keep_prefix_sweeps(program, sweeps):
    if sweeps < 1:
        raise ValueError("keep-sweeps must be positive")
    lines = []
    for line in program.splitlines(keepends=True):
        if line.lstrip().startswith("} ->"):
            lines.append(line)
            continue
        match = re.search(r"\bs(\d+)c\d+", line)
        if match and int(match.group(1)) >= sweeps:
            continue
        lines.append(line)
    program = "".join(lines)
    states = re.findall(r"\]> (s\d+c\d+state) = ", program)
    if not states:
        raise ValueError("no remaining color state")
    program, count = re.subn(
        r"\} -> \(s\d+c\d+state\);",
        "} -> (" + states[-1] + ");",
        program,
    )
    if count != 1:
        raise ValueError("return state rewrite failed")
    return program, states[-1]


def oracle_state(state, h, thresholds, selectors, edge_js, sweeps):
    spins = state.copy()
    for sweep in range(sweeps):
        begin = 0
        for tile, ids in selectors.items():
            count = len(ids) // 20
            own = spins[begin : begin + count]
            routed = spins[ids].reshape(count, 20, 128)
            field = (routed * edge_js[tile][:, :, None]).sum(axis=1, dtype=np.float16)
            margin = own * (field + h[begin : begin + count])
            accept = np.clip(
                margin
                + np.repeat(thresholds[sweep, begin : begin + count], 32, axis=1)
                + 1,
                0,
                1,
            )
            spins[begin : begin + count] = own * (1 - 2 * accept)
            begin += count
    return spins


def prepare_color_field(program, manifest, selectors, tile, output, channels):
    ids = selectors[tile]
    count = len(ids) // 20
    begin = sum(len(v) // 20 for k, v in selectors.items() if k < tile)
    first_index = re.search(
        r"^    tensor<int32, \[\d+\]> sparse_c\d+indices =", program, re.MULTILINE
    )
    assert first_index is not None
    head = program[: first_index.start()]
    signature = (
        f"  func main<ios18>({tensor([1, channels, 1, 128])} a_state, "
        f"{tensor([1, count, 1, 128])} b_h, {tensor([1, count, 1, 32])} d_j{tile}) {{"
    )
    head = re.sub(r"  func main<ios18>\([^\n]+\) \{", lambda _: signature, head)
    prefix = f"sparse_c{tile}"
    name = f"s0c{tile}gather"
    lines = [
        constant(f"tensor<int32, [{len(ids)}]>", prefix + "indices", str(ids.tolist())),
        constant("tensor<int32, [4]>", prefix + "shape", str([1, count, 20, 128])),
        constant("tensor<int32, [4]>", prefix + "begin", "[0,0,0,0]"),
        constant("tensor<int32, [4]>", prefix + "jsize", str([1, count, 1, 20])),
        constant("tensor<int32, [4]>", prefix + "jshape", str([1, count, 20, 1])),
        emit(
            [1, count, 1, 20],
            prefix + "jflat",
            f"slice_by_size(x=d_j{tile}, begin={prefix}begin, size={prefix}jsize)",
        ),
        emit(
            [1, count, 20, 1],
            prefix + "j",
            f"reshape(x={prefix}jflat, shape={prefix}jshape)",
        ),
        emit(
            [1, len(ids), 1, 128],
            name,
            f"gather(x=a_state, indices={prefix}indices, axis=axis, "
            "batch_dims=sparse_batch_dims, validate_indices=sparse_validate)",
        ),
        emit(
            [1, count, 20, 128], "neighbors", f"reshape(x={name}, shape={prefix}shape)"
        ),
        emit([1, count, 20, 128], "products", f"mul(x=neighbors, y={prefix}j)"),
        emit(
            [1, count, 1, 128],
            "sum",
            "reduce_sum(x=products, axes=sparse_sum_axes, keep_dims=sparse_keep_dims)",
        ),
        emit([1, count, 1, 128], "field", "add(x=sum, y=b_h)"),
    ]
    program = head + "\n".join(lines) + "\n  } -> (field);\n}\n"
    receipts, h_files, expected_files = [], [], []
    for job in range(len(manifest["expected"])):
        state_path = Path(manifest["inputs"][0]["files"][job])
        meta = json.loads((state_path.parent / "meta.json").read_text())
        state = np.fromfile(state_path, dtype=np.float16).reshape(channels, 128)
        h = np.fromfile(manifest["inputs"][1]["files"][job], dtype=np.float16).reshape(
            channels, 128
        )[begin : begin + count]
        edge_j = np.fromfile(
            manifest["inputs"][3 + tile]["files"][job], dtype=np.float16
        ).reshape(count, 32)[:, :20]
        dense = np.fromfile(
            state_path.parent / meta["tiles"][tile]["weight_file"], dtype=np.float16
        ).reshape(-1, channels)[:count]
        assert np.array_equal(dense, dense.astype(np.int32))
        assert np.array_equal(state, state.astype(np.int32))
        expected = (dense.astype(np.int32) @ state.astype(np.int32) + h).astype(
            np.float16
        )
        actual = (state[ids].reshape(count, 20, 128) * edge_j[:, :, None]).sum(
            axis=1, dtype=np.float16
        ) + h
        mismatches = int(np.count_nonzero(actual != expected))
        assert mismatches == 0
        h_path, expected_path = output / f"h{job}.bin", output / f"expected{job}.bin"
        h.tofile(h_path)
        expected.tofile(expected_path)
        h_files.append(str(h_path))
        expected_files.append(str(expected_path))
        receipts.append({"job": job, "mismatches": mismatches})
    manifest["inputs"] = [
        manifest["inputs"][0],
        {"name": "b_h", "elements": count * 128, "files": h_files},
        manifest["inputs"][3 + tile],
    ]
    manifest["expected"] = expected_files
    manifest["output_elements"] = count * 128
    for key in ("iterations", "block_sweeps", "total_sweeps"):
        manifest.pop(key, None)
    return program, manifest, {tile: ids}, receipts


def write_selector_blob(output, selectors, channels, reuse_weights):
    payload_bytes = sum(
        ((len(ids) + 31) // 32 * 32) * channels * 2 for ids in selectors.values()
    )
    blob_bytes = payload_bytes + 64 * (1 + len(selectors))
    assert blob_bytes <= 1_000_000_000
    layout = []
    setup = []
    blob_path = output / "weight_data.bin"
    if reuse_weights is not None:
        source = reuse_weights.resolve()
        assert source.stat().st_size == blob_bytes
        if blob_path.exists() or blob_path.is_symlink():
            blob_path.unlink()
        blob_path.symlink_to(source)
        offset = 64
        for tile, ids in selectors.items():
            rows = (len(ids) + 31) // 32 * 32
            layout.append(
                {
                    "tile": tile,
                    "header_offset": offset,
                    "data_offset": offset + 64,
                    "rows": rows,
                }
            )
            offset += 64 + rows * channels * 2
    else:
        with blob_path.open("wb") as blob:
            header = bytearray(64)
            struct.pack_into("<II", header, 0, len(selectors), 2)
            blob.write(header)
            for tile, ids in selectors.items():
                rows = (len(ids) + 31) // 32 * 32
                offset = blob.tell()
                header = bytearray(64)
                struct.pack_into(
                    "<IIQQ", header, 0, 0xDEADBEEF, 1, rows * channels * 2, offset + 64
                )
                blob.write(header)
                for begin in range(0, rows, 128):
                    end = min(rows, begin + 128)
                    block = np.zeros((end - begin, channels), dtype=np.float16)
                    live = max(0, min(end, len(ids)) - begin)
                    block[np.arange(live), ids[begin : begin + live]] = 1
                    blob.write(block.tobytes())
                layout.append(
                    {
                        "tile": tile,
                        "header_offset": offset,
                        "data_offset": offset + 64,
                        "rows": rows,
                    }
                )
        for info in layout:
            weights = np.memmap(
                blob_path,
                dtype=np.float16,
                mode="r",
                offset=info["data_offset"],
                shape=(info["rows"], channels),
            )
            ids = selectors[info["tile"]]
            for begin in range(0, info["rows"], 128):
                end = min(info["rows"], begin + 128)
                block = weights[begin:end]
                live = max(0, min(end, len(ids)) - begin)
                assert np.count_nonzero(block) == live
                assert np.all(block[np.arange(live), ids[begin : begin + live]] == 1)
            del weights
    for info in layout:
        tile = info["tile"]
        ids = selectors[tile]
        rows = info["rows"]
        shape = [rows, channels, 1, 1]
        setup.append(
            constant(
                tensor(shape),
                f"selector_w{tile}",
                'BLOBFILE(path=string("@model_path/weights/weight_data.bin"), '
                f"offset=uint64({info['header_offset']}))",
            )
        )
        setup.append(
            constant("tensor<int32, [4]>", f"selector_begin{tile}", "[0,0,0,0]")
        )
        setup.append(
            constant(
                "tensor<int32, [4]>",
                f"selector_size{tile}",
                str([1, len(ids), 1, 128]),
            )
        )
    return setup, layout, payload_bytes


def concat_tree(parts, dest_name, dest_len, prefix, arity):
    lines = []
    current = [(part, 1) for part in parts]
    level = 0
    while len(current) > arity:
        nxt = []
        for begin in range(0, len(current), arity):
            chunk = current[begin : begin + arity]
            if len(chunk) == 1:
                nxt.append(chunk[0])
                continue
            node = f"{prefix}cat{level}_{begin // arity}"
            width = sum(size for _, size in chunk)
            lines.append(
                emit(
                    [1, width, 1, 128],
                    node,
                    "concat(values=("
                    + ", ".join(name for name, _ in chunk)
                    + "), axis=axis, interleave=interleave)",
                )
            )
            nxt.append((node, width))
        current = nxt
        level += 1
    width = sum(size for _, size in current)
    assert width == dest_len
    lines.append(
        emit(
            [1, dest_len, 1, 128],
            dest_name,
            "concat(values=("
            + ", ".join(name for name, _ in current)
            + "), axis=axis, interleave=interleave)",
        )
    )
    return lines


def replace_gathers(program, selectors, selection, concat_arity):
    replaced = 0
    gather_pattern = (
        r"^    tensor<fp16, \[[^\]]+\]> (s(\d+)c(\d+)gather) = "
        r"gather\(x=([^,]+),[^\n]+\];$"
    )

    def replace_gather(match):
        nonlocal replaced
        replaced += 1
        name, tile, state = match[1], int(match[3]), match[4]
        ids = selectors[tile]
        if selection == "conv":
            rows = (len(ids) + 31) // 32 * 32
            return "\n".join(
                [
                    emit(
                        [1, rows, 1, 128],
                        name + "raw",
                        f"conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, "
                        f"strides=st, weight=selector_w{tile}, x={state})",
                    ),
                    emit(
                        [1, len(ids), 1, 128],
                        name,
                        f"slice_by_size(x={name}raw, begin=selector_begin{tile}, "
                        f"size=selector_size{tile})",
                    ),
                ]
            )
        lines = [constant("tensor<int32, [4]>", name + "size", "[1,1,1,128]")]
        for index in np.unique(ids):
            label = name + f"src{index}"
            lines.append(
                constant(
                    "tensor<int32, [4]>", label + "begin", str([0, int(index), 0, 0])
                )
            )
            lines.append(
                emit(
                    [1, 1, 1, 128],
                    label,
                    f"slice_by_size(x={state}, begin={label}begin, size={name}size)",
                )
            )
        parts = [name + f"src{index}" for index in ids]
        if concat_arity is None or len(parts) <= concat_arity:
            lines.append(
                emit(
                    [1, len(ids), 1, 128],
                    name,
                    f"concat(values=({', '.join(parts)}), axis=axis, "
                    "interleave=interleave)",
                )
            )
        else:
            lines.extend(concat_tree(parts, name, len(ids), name, concat_arity))
        return "\n".join(lines)

    program = re.sub(gather_pattern, replace_gather, program, flags=re.MULTILINE)
    assert replaced > 0 and " = gather(" not in program and " = matmul(" not in program
    return program, replaced


def check_jobs(manifest, selectors, channels, sweeps, output):
    receipts = []
    expected_files = []
    for job in range(len(manifest["expected"])):
        state = np.fromfile(
            manifest["inputs"][0]["files"][job], dtype=np.float16
        ).reshape(channels, 128)
        h = np.fromfile(manifest["inputs"][1]["files"][job], dtype=np.float16).reshape(
            channels, 128
        )
        meta = json.loads(
            (Path(manifest["inputs"][0]["files"][job]).parent / "meta.json").read_text()
        )
        thresholds = np.fromfile(
            Path(manifest["inputs"][0]["files"][job]).parent / meta["threshold_file"],
            dtype=np.float16,
        ).reshape(meta["sweeps"], channels, 4)
        edge_js = [
            np.fromfile(item["files"][job], dtype=np.float16).reshape(-1, 32)[:, :20]
            for item in manifest["inputs"][3:]
        ]
        actual = oracle_state(state, h, thresholds, selectors, edge_js, sweeps)
        saved = np.fromfile(manifest["expected"][job], dtype=np.float16).reshape(
            channels, 128
        )
        if sweeps == meta["sweeps"]:
            errors = int(np.count_nonzero(actual != saved))
            assert errors == 0, (job, errors)
            expected_files.append(manifest["expected"][job])
        else:
            path = output / f"expected{job}.bin"
            actual.astype(np.float16).tofile(path)
            expected_files.append(str(path))
            errors = 0
        receipts.append({"job": job, "mismatches": errors, "sweeps": sweeps})
    manifest["expected"] = expected_files
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Frozen sparse-gather fixture directory"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--selection", choices=["conv", "slices"], default="conv")
    parser.add_argument("--color-field", type=int)
    parser.add_argument("--keep-sweeps", type=int)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--reuse-weights", type=Path)
    parser.add_argument("--concat-arity", type=int)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "manifest.json").read_text())
    program = (source / manifest["mil"]).read_text()
    channels = manifest["output_elements"] // 128
    for item in manifest["inputs"]:
        item["files"] = [
            str(path if Path(path).is_absolute() else (source / path).resolve())
            for path in item["files"]
        ]
    manifest["expected"] = [
        str(path if Path(path).is_absolute() else (source / path).resolve())
        for path in manifest["expected"]
    ]
    selectors = {}
    index_pattern = (
        r"^    tensor<int32, \[\d+\]> sparse_c(\d+)indices = const\(\)."
        r"*val=tensor<int32, \[\d+\]>\((\[[^\n]+\])\)\];$"
    )
    for match in re.finditer(index_pattern, program, re.MULTILINE):
        selectors[int(match[1])] = np.array(ast.literal_eval(match[2]), dtype=np.int32)
    assert len(selectors) == len(manifest["inputs"]) - 3
    assert all(len(ids) % 20 == 0 for ids in selectors.values())
    field_receipts = []
    if args.color_field is not None:
        program, manifest, selectors, field_receipts = prepare_color_field(
            program, manifest, selectors, args.color_field, output, channels
        )
    setup, blob_layout = [], []
    payload_bytes = 0
    if args.selection == "conv":
        setup, blob_layout, payload_bytes = write_selector_blob(
            output, selectors, channels, args.reuse_weights
        )
        manifest["weights"] = "weight_data.bin"
    program, replaced = replace_gathers(
        program, selectors, args.selection, args.concat_arity
    )
    program = re.sub(
        r"^    .* sparse_c\d+indices = .*\n|^    .* sparse_(?:batch_dims|validate) = .*\n",
        "",
        program,
        flags=re.MULTILINE,
    )
    anchor = '    fp16 minusTwo = const()[name=string("minusTwo"), val=fp16(-2.0)];'
    assert program.count(anchor) == 1
    program = program.replace(anchor, anchor + "\n" + "\n".join(setup))
    source_meta = json.loads(
        (Path(manifest["inputs"][0]["files"][0]).parent / "meta.json").read_text()
    )
    sweeps = source_meta["sweeps"]
    output_state = None
    if args.keep_sweeps is not None:
        if args.color_field is not None:
            parser.error("--keep-sweeps does not apply to color-field graphs")
        sweeps = args.keep_sweeps
        program, output_state = keep_prefix_sweeps(program, sweeps)
        manifest["iterations"] = 1
        manifest["block_sweeps"] = sweeps
        manifest["total_sweeps"] = sweeps
    identity = hashlib.sha256((args.selection + program).encode()).hexdigest()
    program = re.sub(
        r'\{"single-call-probe", "[^\"]+"\}',
        '{"single-call-probe", "' + identity + '"}',
        program,
    )
    (output / "program.mil").write_text(program)
    if args.compile_only:
        manifest["compile_only"] = True
    manifest["mil"] = "program.mil"
    if args.color_field is not None:
        receipts = field_receipts
    else:
        receipts = check_jobs(manifest, selectors, channels, sweeps, output)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    proof = {
        "selection": args.selection,
        "scope": "CPU checks only",
        "color_field": args.color_field,
        "keep_sweeps": sweeps if args.color_field is None else None,
        "output_state": output_state,
        "jobs": receipts,
        "program_bytes": len(program),
        "replaced_gathers": replaced,
        "constant_payload_bytes": payload_bytes if args.selection == "conv" else 0,
        "unique_source_slices_per_sweep": sum(
            len(np.unique(ids)) for ids in selectors.values()
        ),
        "concat_arity": {int(tile): len(ids) for tile, ids in selectors.items()},
        "concat_tree_arity": args.concat_arity,
        "blob_layout": blob_layout,
        "constant_selectors_verified": args.selection == "conv"
        and args.reuse_weights is None,
        "reused_weights": str(args.reuse_weights.resolve())
        if args.reuse_weights is not None
        else None,
        "compile_only": bool(args.compile_only),
    }
    (output / "cpu-proof.json").write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
