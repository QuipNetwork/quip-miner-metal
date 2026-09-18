// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Metal multi-spin coded annealing miner (`quip-metal-msa`). macOS-only at
//! runtime. Same wire protocol and CLI as `quip-metal-sa`; selected in
//! `config.toml` with `[metal.N] binary = ".../quip-metal-msa"`.

// Binary is a separate crate; lib.rs crate-root panic discipline does not apply here.
#![deny(clippy::unwrap_used)]
#![deny(clippy::panic)]
#![warn(clippy::expect_used)]

use clap::Parser;
use quip_miner_metal::{run_metal, MsaTag, METAL_MSA_IDENTITY};
use quip_solver_core::CommonArgs;
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[arg(long, hide = true,
        conflicts_with_all = ["quip_coordinator", "capabilities", "check", "solve"])]
    ane_worker: Option<u32>,
    #[command(flatten)]
    common: CommonArgs,
    /// Metal device index. Default 0 → miner id `metal-0`.
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// Target GPU utilization ceiling percent (1–100). Used by IOKit governor.
    #[arg(long, default_value_t = 100)]
    utilization: u32,
    /// Yield to other GPU users under thermal/display pressure.
    #[arg(long, default_value_t = false)]
    yielding: bool,
}

fn main() -> ExitCode {
    let entry = std::time::Instant::now();
    let mut cli = Cli::parse();
    if let Some(parent_pid) = cli.ane_worker {
        return quip_miner_ane::worker_main(parent_pid, entry);
    }
    if cli.common.miner_id.is_none() {
        cli.common.miner_id = Some(format!("metal-{}", cli.device));
    }
    run_metal::<MsaTag>(
        METAL_MSA_IDENTITY,
        &cli.common,
        cli.device,
        cli.utilization,
        cli.yielding,
    )
}
