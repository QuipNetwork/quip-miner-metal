#!/usr/bin/env python3
"""Write N fresh nonces as a regen input, on the chain's current topology.

    scripts/testnet/make_fresh.py qblocks-all.json fresh.json N [RNG_SEED]

A fresh block carries a random 32-byte `nonce_seed` and `"fresh": true`, so
`scripts/testnet/regen` draws the instance from the seed without the nonce
check. The instances follow the same distribution as real nonces, because a
real nonce is a BLAKE3 output and BLAKE3 outputs are uniform. The target is
the chain's current difficulty, so `valid` in the study CSV means a proof at
today's target. `winning_energy_milli` is 0: no one has won these.

Qblock ids start at 1,000,000 so they never collide with real blocks.
"""

import json
import random
import sys

FIRST_ID = 1_000_000


def main():
    src, dst, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
    rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 20260918)
    with open(src) as f:
        chain = json.load(f)
    topology_hash = chain["blocks"][0]["topology_hash"]
    blocks = [
        {
            "qblock_id": FIRST_ID + i,
            "energy_milli": 0,
            "difficulty": chain["current_difficulty"],
            "topology_hash": topology_hash,
            "nonce_seed": rng.getrandbits(256).to_bytes(32, "big").hex(),
            "fresh": True,
        }
        for i in range(count)
    ]
    with open(dst, "w") as f:
        json.dump({"blocks": blocks, "topologies": chain["topologies"]}, f)
    print(
        f"{count} fresh nonces on {topology_hash[:8]}, target {chain['current_difficulty']}"
    )


if __name__ == "__main__":
    main()
