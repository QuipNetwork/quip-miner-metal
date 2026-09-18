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
    miner: String,
    salt: String,
    energy_milli: i64,
    difficulty: Difficulty,
    last_proof_block_hash: String,
    topology_hash: String,
    nonce_seed: String,
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
    eprintln!("wrote {} problems, nonce check passed on all", index.len());
}
