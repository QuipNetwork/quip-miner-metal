"""Make matched diagnostic arms from an existing two-sweep runtime-J fixture."""

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    manifest = json.loads((args.source / "manifest.json").read_text())
    mil = (args.source / manifest["mil"]).read_text()
    for item in manifest["inputs"]:
        path = str((args.source / item["files"][0]).resolve())
        item["files"] = [path] * args.repeats
    manifest["expected"] = [
        str((args.source / manifest["expected"][0]).resolve())
    ] * args.repeats
    manifest["diagnostics"] = True
    runtime = args.output / "runtime"
    constant = args.output / "constant"
    runtime.mkdir(parents=True, exist_ok=True)
    constant.mkdir(parents=True, exist_ok=True)
    (runtime / "program.mil").write_text(mil)
    (runtime / "manifest.json").write_text(json.dumps(manifest, indent=2))
    constants = [item for item in manifest["inputs"] if item["name"].startswith("d_j")]
    blob = bytearray(64)
    struct.pack_into("<II", blob, 0, len(constants), 2)
    identities = []
    for item in constants:
        tile = int(item["name"].removeprefix("d_j"))
        payload = Path(item["files"][0]).read_bytes()
        assert len(payload) == item["elements"] * 2
        offset = len(blob)
        header = bytearray(64)
        struct.pack_into("<IIQQ", header, 0, 0xDEADBEEF, 1, len(payload), offset + 64)
        blob.extend(header)
        blob.extend(payload)
        identities.append(
            {
                "tile": tile,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        pattern = rf'(?m)^(\s*)(tensor<fp16, \[[^\]]+\]>) w{tile} = reshape\(x=d_j{tile}, shape=w{tile}shape\)\[name=string\("w{tile}"\)\];$'

        def replace(match, tile=tile, offset=offset):
            return f'{match[1]}{match[2]} w{tile} = const()[name=string("w{tile}"), val={match[2]}(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64({offset})))];'

        mil, count = re.subn(pattern, replace, mil)
        assert count == 1, f"unexpected generated weight definition {tile}"
    mil, count = re.subn(r", tensor<fp16, \[[^\]]+\]> d_j\d+", "", mil)
    assert count == len(constants)
    manifest["inputs"] = [
        item for item in manifest["inputs"] if not item["name"].startswith("d_j")
    ]
    manifest["weights"] = "weight_data.bin"
    (constant / "weight_data.bin").write_bytes(blob)
    (constant / "program.mil").write_text(mil)
    (constant / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "j-identities.json").write_text(json.dumps(identities, indent=2))


if __name__ == "__main__":
    main()
