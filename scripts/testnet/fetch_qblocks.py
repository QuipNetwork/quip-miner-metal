#!/usr/bin/env python3
"""Fetch the latest N qblocks and their topology from a local Quip node.

The Aglais bootnodes speak libp2p only, so sync a node first and query it
over its local RPC port. Warp sync reaches head in about two minutes:

    docker run -d --name quip-qblock-reader \\
      -v "$PWD/node-data:/data" -v "$PWD/aglais-network.json:/etc/quip/chain-spec.json:ro" \\
      -p 127.0.0.1:9944:9944 \\
      registry.gitlab.com/quip.network/quip-validator/quip-network-node:latest \\
      --chain=/etc/quip/chain-spec.json --base-path=/data --name=qblock-reader \\
      --sync=warp --rpc-port=9944 --unsafe-rpc-external --rpc-cors=all \\
      --rpc-methods=safe --no-mdns --no-prometheus --unsafe-force-node-key-generation

Then:

    scripts/testnet/fetch_qblocks.py 60 qblocks.json

The script calls the `QuantumPowApi` runtime API through the generic
`state_call` RPC method and decodes the SCALE bytes by hand. `QBlock` is a
fixed 180-byte layout followed by a 32-byte nonce, so a wrong layout fails
the length check rather than producing plausible garbage. The nonce comes
back SCALE little-endian and is stored reversed, as the big-endian digest
bytes that seed the problem draw.
"""

import json
import struct
import sys
import urllib.request

RPC = "http://127.0.0.1:9944"


def rpc(method, params):
    body = json.dumps(
        {"id": 1, "jsonrpc": "2.0", "method": method, "params": params}
    ).encode()
    req = urllib.request.Request(
        RPC, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.load(r)
    if "error" in out:
        raise RuntimeError(out["error"])
    return out["result"]


def state_call(method, data=b""):
    return bytes.fromhex(rpc("state_call", [method, "0x" + data.hex()])[2:])


class Cur:
    """Cursor over SCALE-encoded bytes."""

    def __init__(self, b):
        self.b, self.i = b, 0

    def take(self, n):
        v = self.b[self.i : self.i + n]
        assert len(v) == n, f"short read at {self.i}: want {n}"
        self.i += n
        return v

    def u8(self):
        return self.take(1)[0]

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def i32(self):
        return struct.unpack("<i", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def i64(self):
        return struct.unpack("<q", self.take(8))[0]

    def u128(self):
        return int.from_bytes(self.take(16), "little")

    def compact(self):
        b0 = self.b[self.i]
        mode = b0 & 3
        if mode == 0:
            self.i += 1
            return b0 >> 2
        if mode == 1:
            return struct.unpack("<H", self.take(2))[0] >> 2
        if mode == 2:
            return struct.unpack("<I", self.take(4))[0] >> 2
        n = (b0 >> 2) + 4
        self.i += 1
        return int.from_bytes(self.take(n), "little")

    def done(self):
        assert self.i == len(self.b), f"trailing bytes: {len(self.b) - self.i}"


def decode_difficulty(c):
    return {
        "min_solutions": c.u32(),
        "max_energy_milli": c.i64(),
        "min_diversity_milli": c.u32(),
    }


def decode_qblock_with_nonce(raw):
    c = Cur(raw)
    if c.u8() == 0:
        return None
    q = {
        "miner": c.take(32).hex(),
        "salt": c.take(32).hex(),
        "energy_milli": c.i64(),
        "reward": c.u128(),
        "submitted_at": c.u32(),
        "difficulty": decode_difficulty(c),
        "last_proof_block_hash": c.take(32).hex(),
        "topology_hash": c.take(32).hex(),
        "device_access_time_us": c.u64(),
    }
    nonce_le = c.take(32)
    q["nonce_seed"] = nonce_le[::-1].hex()
    c.done()
    return q


def decode_allowed(c):
    v = c.u8()
    if v == 0:
        n = c.compact()
        return {"Set": [c.i32() for _ in range(n)]}
    if v == 1:
        return {"IntegerRange": {"min": c.i32(), "max": c.i32()}}
    if v == 2:
        return {"ContinuousRange": {"min": c.i32(), "max": c.i32()}}
    raise ValueError(f"AllowedValueSpec variant {v}")


def decode_topology(raw):
    c = Cur(raw)
    if c.u8() == 0:
        return None
    n = c.compact()
    nodes = [c.u32() for _ in range(n)]
    m = c.compact()
    edges = [(c.u32(), c.u32()) for _ in range(m)]
    t = {
        "nodes": nodes,
        "edges": edges,
        "allowed_h_values": decode_allowed(c),
        "allowed_j_values": decode_allowed(c),
        "allowed_spin_values": decode_allowed(c),
        "registered_at": c.u32(),
    }
    c.done()
    return t


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    out_path = sys.argv[2] if len(sys.argv) > 2 else "qblocks.json"
    print(
        "chain:", rpc("system_chain", []), "head:", rpc("chain_getHeader", [])["number"]
    )
    c = Cur(state_call("QuantumPowApi_latest_qblock_id"))
    assert c.u8() == 1, "no qblocks yet"
    latest = c.u64()
    c.done()
    print("latest qblock id:", latest)
    difficulty_now = decode_difficulty(
        Cur(state_call("QuantumPowApi_current_difficulty"))
    )
    print("current difficulty:", difficulty_now)

    blocks, topologies = [], {}
    for qid in range(latest, max(latest - count, 0), -1):
        q = decode_qblock_with_nonce(
            state_call("QuantumPowApi_qblock_by_id", struct.pack("<Q", qid))
        )
        if q is None:
            print("missing qblock", qid)
            continue
        q["qblock_id"] = qid
        th = q["topology_hash"]
        if th not in topologies:
            t = decode_topology(
                state_call("QuantumPowApi_topology_meta", bytes.fromhex(th))
            )
            assert t is not None, f"topology {th} not registered"
            topologies[th] = t
            print(
                f"topology {th[:12]}: {len(t['nodes'])} nodes, {len(t['edges'])} edges, "
                f"h={t['allowed_h_values']} j={t['allowed_j_values']}"
            )
        blocks.append(q)
    with open(out_path, "w") as f:
        json.dump(
            {
                "latest_qblock_id": latest,
                "current_difficulty": difficulty_now,
                "blocks": blocks,
                "topologies": topologies,
            },
            f,
        )
    energies = sorted(b["energy_milli"] for b in blocks)
    targets = [b["difficulty"]["max_energy_milli"] for b in blocks]
    print(f"wrote {len(blocks)} qblocks to {out_path}")
    print(
        f"winning energy_milli: min {energies[0]} median {energies[len(energies) // 2]} max {energies[-1]}"
    )
    print(f"target max_energy_milli: min {min(targets)} max {max(targets)}")


if __name__ == "__main__":
    main()
