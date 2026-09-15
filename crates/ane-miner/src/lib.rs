//! Standalone ANE MSA Ising miner.

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
compile_error!("quip-miner-ane requires the aarch64-apple-darwin target");

mod graph;
mod msa;
#[expect(unsafe_code, reason = "checked ANE C ABI boundary")]
mod native;
mod process;
mod solver;
mod worker;
pub use process::AneSampler;
pub use worker::worker_main;

/// Advertised ANE MSA identity. Node and edge caps come from the graph module
/// so capability JSON cannot drift from the host validator.
pub const ANE_MSA_IDENTITY: quip_solver_core::BackendIdentity = quip_solver_core::BackendIdentity {
    backend: "ane",
    algorithm: "msa",
    max_nodes: crate::graph::MAX_NODES as u32,
    max_edges: crate::graph::MAX_EDGES as u32,
    features: &["streaming"],
    adapt: quip_solver_core::adapt::AdaptBounds {
        min_sweeps: 64,
        max_sweeps: 256,
        min_reads: 128,
        max_reads: 128,
        reads_solution_min_factor: 0,
        reads_solution_max_factor: 0,
        reads_solution_floor_factor: 0,
    },
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum AneError {
    #[error("{0}")]
    Capacity(String),
    #[error("{0}")]
    Runtime(String),
}
