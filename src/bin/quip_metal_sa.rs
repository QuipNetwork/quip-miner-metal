//! Metal simulated-annealing miner (`quip-metal-sa`). macOS-only at runtime.

// Binary is a separate crate; lib.rs crate-root panic discipline does not apply here.
#![deny(clippy::unwrap_used)]
#![deny(clippy::panic)]
#![warn(clippy::expect_used)]

use clap::Parser;
use quip_miner_metal::{run_metal, SaTag, METAL_SA_IDENTITY};
use quip_solver_core::CommonArgs;
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
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
    let mut cli = Cli::parse();
    if cli.common.miner_id.is_none() {
        cli.common.miner_id = Some(format!("metal-{}", cli.device));
    }
    run_metal::<SaTag>(
        METAL_SA_IDENTITY,
        &cli.common,
        cli.device,
        cli.utilization,
        cli.yielding,
    )
}
