// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Regenerate each fetched qblock's Ising problem from its nonce and write
//! `--solve` inputs with node ids compacted to `0..n`.
//!
//! ```sh
//! cargo run --manifest-path scripts/testnet/regen/Cargo.toml --release -- qblocks.json problems/
//! ```
//!
//! The draw is `quip_protocol::chacha8::draw_ising_milli`, the same call the
//! coordinator makes, so the problem matches what the chain validated. Before
//! drawing, the tool recomputes each nonce as
//! `BLAKE3(last_proof_block_hash || blake2_256(miner) || salt)` and stops on
//! a mismatch, which pins the nonce byte order.
//!
//! A block marked `"fresh": true` skips the nonce check and draws from its
//! `nonce_seed` as given. `scripts/testnet/make_fresh.py` writes such blocks
//! with random seeds, which sample the same instance distribution as real
//! nonces because a BLAKE3 output is uniform.

use blake2::{digest::consts::U32, Blake2b, Digest};
use quip_protocol::chacha8::draw_ising_milli;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct Difficulty {
    max_energy_milli: i64,
}

#[derive(Deserialize)]
struct Block {
    qblock_id: u64,
    #[serde(default)]
    miner: String,
    #[serde(default)]
    salt: String,
    energy_milli: i64,
    difficulty: Difficulty,
    #[serde(default)]
    last_proof_block_hash: String,
    topology_hash: String,
    nonce_seed: String,
    #[serde(default)]
    fresh: bool,
}

#[derive(Deserialize)]
struct Allowed {
    #[serde(rename = "Set")]
    set: Vec<i32>,
}

#[derive(Deserialize)]
struct Topology {
    nodes: Vec<u32>,
    edges: Vec<(u32, u32)>,
    allowed_h_values: Allowed,
    allowed_j_values: Allowed,
}

#[derive(Deserialize)]
struct Input {
    blocks: Vec<Block>,
    topologies: HashMap<String, Topology>,
}

fn hex32(s: &str) -> [u8; 32] {
    hex::decode(s)
        .expect("hex field")
        .try_into()
        .expect("32-byte field")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (input_path, out_dir) = (&args[1], &args[2]);
    let input: Input =
        serde_json::from_reader(std::fs::File::open(input_path).expect("open qblocks json"))
            .expect("parse qblocks json");
    std::fs::create_dir_all(out_dir).expect("create output dir");

    let mut index = Vec::new();
    for b in &input.blocks {
        if !b.fresh {
            let miner32: [u8; 32] = Blake2b::<U32>::digest(hex32(&b.miner)).into();
            let recomputed = quip_protocol::derive::derive_nonce(
                hex32(&b.last_proof_block_hash),
                miner32,
                hex32(&b.salt),
            );
            assert_eq!(
                hex::encode(recomputed),
                b.nonce_seed,
                "nonce mismatch on qblock {}",
                b.qblock_id
            );
        }

        let t = &input.topologies[&b.topology_hash];
        let (h, j) = draw_ising_milli(
            hex32(&b.nonce_seed),
            t.nodes.len(),
            t.edges.len(),
            &t.allowed_h_values.set,
            &t.allowed_j_values.set,
        )
        .expect("draw");
        let pos: HashMap<u32, usize> = t.nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let edges: Vec<[usize; 2]> = t.edges.iter().map(|&(u, v)| [pos[&u], pos[&v]]).collect();
        let hf: Vec<f64> = h.iter().map(|&v| f64::from(v) / 1000.0).collect();
        let jf: Vec<f64> = j.iter().map(|&v| f64::from(v) / 1000.0).collect();

        let path = format!("{out_dir}/problem-{}.json", b.qblock_id);
        let problem = serde_json::json!({ "h": hf, "j": jf, "edges": edges });
        serde_json::to_writer(
            std::fs::File::create(&path).expect("create problem"),
            &problem,
        )
        .expect("write problem");
        index.push(serde_json::json!({
            "qblock_id": b.qblock_id,
            "path": path,
            "winning_energy_milli": b.energy_milli,
            "target_energy_milli": b.difficulty.max_energy_milli,
            "n_nodes": t.nodes.len(),
            "n_edges": t.edges.len(),
        }));
    }
    let index_path = format!("{out_dir}/index.json");
    serde_json::to_writer_pretty(
        std::fs::File::create(&index_path).expect("create index"),
        &index,
    )
    .expect("write index");
    let checked = input.blocks.iter().filter(|b| !b.fresh).count();
    eprintln!(
        "wrote {} problems, nonce check passed on {checked}, fresh {}",
        index.len(),
        index.len() - checked
    );
}

#[cfg(test)]
mod tests {
    use rand_core::{RngCore, SeedableRng};

    /// Nonce seed of Aglais qblock 3250, fetched 2026-09-18.
    const REAL_NONCE: &str = "f72f174fdb5d8d7fd56c6c30c4d03f41cb26fee3d8e1ccf4866009253c82a4bd";

    fn seed32(hex_str: &str) -> [u8; 32] {
        hex::decode(hex_str)
            .expect("hex")
            .try_into()
            .expect("32 bytes")
    }

    /// The chain draws with `rand_chacha::ChaCha8Rng`; the published crate
    /// draws with its own port. A problem consumes 46,091 draws, so 100,000
    /// covers it twice over.
    #[test]
    fn published_draw_matches_the_validators_rng() {
        let seeds = [[0u8; 32], [0x5a; 32], [0xff; 32], seed32(REAL_NONCE)];
        for seed in seeds {
            let mut theirs = rand_chacha::ChaCha8Rng::from_seed(seed);
            let mut ours = quip_protocol::chacha8::ChaCha8Rng::from_seed(seed);
            for draw in 0..100_000 {
                assert_eq!(
                    ours.next_u32(),
                    theirs.next_u32(),
                    "seed {} diverges at draw {draw}",
                    hex::encode(seed)
                );
            }
        }
    }

    /// Positive control for the test above: the comparison must see a
    /// difference when one seed bit flips, or a stub RNG would pass it.
    #[test]
    fn comparison_detects_a_one_bit_seed_change() {
        let mut flipped = seed32(REAL_NONCE);
        flipped[31] ^= 1;
        let mut theirs = rand_chacha::ChaCha8Rng::from_seed(seed32(REAL_NONCE));
        let mut ours = quip_protocol::chacha8::ChaCha8Rng::from_seed(flipped);
        let differs = (0..16).any(|_| ours.next_u32() != theirs.next_u32());
        assert!(differs, "a flipped seed produced the same first block");
    }
}
