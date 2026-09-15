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
        min_sweeps: 2048,
        max_sweeps: 8192,
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

#[cfg(test)]
mod tests {
    use super::ANE_MSA_IDENTITY;
    use quip_solver_core::adapt::adapt_params;

    #[test]
    fn adaptive_budget_matches_zero_field_advantage_topology() {
        let adapted = adapt_params(-14_612_000, 1, 4_577, 41_514, &[0], &ANE_MSA_IDENTITY.adapt);
        assert_eq!(adapted.num_sweeps, 8_049);
        assert_eq!(adapted.num_reads, 128);
    }
}
