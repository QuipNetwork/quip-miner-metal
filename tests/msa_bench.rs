// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Throughput benchmark on the Advantage2 System 1 working graph: 4577 nodes,
//! 41515 edges, and eight greedy colour classes. Couplings and fields use a
//! fixed seed. `#[ignore]`: minutes of GPU time. Run both kernels with
//!
//! ```sh
//! RUST_LOG=quip_miner_metal=debug cargo test --release --test msa_bench -- --ignored --nocapture
//! ```
//!
//! Knobs: `QUIP_BENCH_JOBS` (default 40), `QUIP_BENCH_SWEEPS` (default 7392),
//! `QUIP_BENCH_READS` (default 128), `QUIP_METAL_TG_PER_CORE` (see
//! `streaming::tg_budget`).
//! `QUIP_BENCH_KERNEL` selects `msa`, `sa`, or `both` (default).
//!
//! Step 5 occupancy: run this command with T in 1, 2, 4, 6, 8, one at a time:
//! ```sh
//! QUIP_METAL_TG_PER_CORE=T QUIP_BENCH_JOBS=80 QUIP_BENCH_KERNEL=msa RUST_LOG=quip_miner_metal=debug cargo test --release --test msa_bench -- --ignored --nocapture
//! ```
//! C1 curve: at the chosen T, use 1, 2, 5, 10, 40 jobs and 128 reads, then
//! 1 and 2 jobs with 256 reads. Keep 7392 sweeps. Verify the curve with three
//! rounds of 1, 2, 5, 10, 40 jobs at 128 reads and 7392 sweeps:
//! ```sh
//! QUIP_METAL_TG_PER_CORE=T QUIP_BENCH_JOBS=J QUIP_BENCH_READS=R QUIP_BENCH_SWEEPS=7392 QUIP_BENCH_KERNEL=msa RUST_LOG=quip_miner_metal=debug cargo test --release --test msa_bench -- --ignored --nocapture
//! ```
//! Step 6 envelope: first run the SA reference, then MSA with S in 2048,
//! 4096, 8192, 16384. Set T to the chosen MSA occupancy:
//! ```sh
//! QUIP_BENCH_JOBS=40 QUIP_BENCH_SWEEPS=2048 QUIP_BENCH_READS=256 QUIP_BENCH_KERNEL=sa RUST_LOG=quip_miner_metal=debug cargo test --release --test msa_bench -- --ignored --nocapture
//! QUIP_METAL_TG_PER_CORE=T QUIP_BENCH_JOBS=40 QUIP_BENCH_SWEEPS=S QUIP_BENCH_READS=128 QUIP_BENCH_KERNEL=msa RUST_LOG=quip_miner_metal=debug cargo test --release --test msa_bench -- --ignored --nocapture
//! ```

#![expect(
    clippy::print_stderr,
    reason = "the benchmark reports jobs/s on stderr, as the task brief specifies"
)]

use quip_miner_metal::metal_device::MetalDevice;
use quip_miner_metal::streaming::{run_stream, GpuGovernor};
use quip_miner_metal::topology::SelfFeedingTopology;
use quip_miner_metal::{IsingGraph, Kernel};
use quip_solver_core::quip_protocol::scoring::energy_milli;
use quip_solver_core::{CancelToken, SampleParams, StreamJob, StreamOutcome};
use std::time::Instant;

struct NoGovernor;

impl GpuGovernor for NoGovernor {
    fn should_throttle(&self) -> bool {
        false
    }
    fn budget_scale(&self) -> f64 {
        1.0
    }
    fn record_gpu_busy_us(&self, _us: u64) {}
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Deterministic xorshift64 for fixture generation.
fn xorshift64(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// Load compacted, zero-based fixture edges with seeded `J` in {-1, 1} and
/// `h` in {-1, 0, 1}.
fn advantage2_system1(seed: u64) -> IsingGraph {
    let mut s = seed | 1;
    let mut edges = Vec::with_capacity(41515);
    let mut j = Vec::with_capacity(41515);
    for line in include_str!("fixtures/advantage2-system1.edges").lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut nodes = line.split_whitespace();
        let u = nodes.next().expect("edge start").parse().expect("node id");
        let v = nodes.next().expect("edge end").parse().expect("node id");
        assert!(nodes.next().is_none(), "two node ids per edge");
        assert!(u < 4577 && v < 4577, "fixture node range");
        edges.push((u, v));
        j.push(if xorshift64(&mut s) & 1 == 0 {
            1.0
        } else {
            -1.0
        });
    }
    assert_eq!(edges.len(), 41515);
    let h = (0..4577)
        .map(|_| [-1.0, 0.0, 1.0][(xorshift64(&mut s) % 3) as usize])
        .collect();
    IsingGraph::new(h, j, edges)
}

struct Run {
    jobs: usize,
    wall_s: f64,
    best_energy: i64,
    mean_best: f64,
    bests: Vec<i64>,
}

/// Stream `jobs` copies of one 4577-node problem through `kernel` and time
/// the whole run, wall clock, from first send to last result.
fn drive(
    graph: &IsingGraph,
    kernel: Kernel,
    jobs: usize,
    num_reads: usize,
    num_sweeps: usize,
) -> Run {
    let (job_tx, job_rx) = tokio::sync::mpsc::channel(jobs.max(1));
    let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(jobs.max(1));
    let cancel = CancelToken::default();
    for i in 0..jobs {
        job_tx
            .blocking_send(StreamJob {
                job_id: format!("bench-{i}").into_bytes(),
                graph: graph.clone(),
                params: SampleParams {
                    num_reads,
                    num_sweeps,
                    sweeps_per_beta: 1,
                    beta_range: None,
                    seed: i as u64,
                },
                watermark: None,
            })
            .expect("send job");
    }
    drop(job_tx);

    let start = Instant::now();
    let worker = std::thread::spawn(move || {
        let device = MetalDevice::open(0).expect("Metal device 0");
        run_stream(&device, kernel, job_rx, &out_tx, &NoGovernor, &cancel);
    });
    let mut bests = Vec::with_capacity(jobs);
    while let Some(r) = out_rx.blocking_recv() {
        let StreamOutcome::Completed(Ok(reads)) = r.outcome else {
            panic!(
                "job {:?} did not complete",
                String::from_utf8_lossy(&r.job_id)
            );
        };
        assert_eq!(reads.len(), num_reads);
        for s in &reads {
            assert_eq!(
                s.energy_milli,
                energy_milli(&s.spins, &graph.h, &graph.j, &graph.edges)
            );
        }
        bests.push(reads.iter().map(|s| s.energy_milli).min().unwrap());
    }
    worker.join().expect("stream worker");
    let wall_s = start.elapsed().as_secs_f64();
    Run {
        jobs: bests.len(),
        wall_s,
        best_energy: bests.iter().copied().min().unwrap(),
        mean_best: bests.iter().map(|&e| e as f64).sum::<f64>() / bests.len() as f64,
        bests,
    }
}

fn report(label: &str, r: &Run, num_reads: usize, num_sweeps: usize) {
    eprintln!(
        "{label}: {} jobs x {num_reads} reads x {num_sweeps} sweeps in {:.1} s = {:.2} jobs/s; best {} milli, mean best {:.0} milli",
        r.jobs,
        r.wall_s,
        r.jobs as f64 / r.wall_s,
        r.best_energy,
        r.mean_best
    );
    eprintln!("wall_seconds={:.6}; bests_milli={:?}", r.wall_s, r.bests);
}

#[test]
#[ignore = "GPU benchmark: minutes of device time"]
fn msa_vs_sa_throughput_at_advantage2_scale() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("quip_miner_metal=info")),
        )
        .with_writer(std::io::stderr)
        .try_init();
    let jobs = env_usize("QUIP_BENCH_JOBS", 40);
    let sweeps = env_usize("QUIP_BENCH_SWEEPS", 7392);
    let reads = env_usize("QUIP_BENCH_READS", 128);
    let graph_seed = env_usize("QUIP_BENCH_GRAPH_SEED", 1) as u64;
    let graph = advantage2_system1(graph_seed);
    eprintln!(
        "graph seed {graph_seed}; QUIP_METAL_MSA_FOUR_COLOR={:?}",
        std::env::var("QUIP_METAL_MSA_FOUR_COLOR").ok()
    );
    let colors = SelfFeedingTopology::build(&graph).colors;
    eprintln!(
        "Advantage2 System 1: {} greedy colour classes",
        colors.num_colors
    );
    assert_eq!(colors.num_colors, 8);

    let kernel = std::env::var("QUIP_BENCH_KERNEL").unwrap_or_else(|_| "both".into());
    assert!(matches!(kernel.as_str(), "msa" | "sa" | "both"));
    if kernel == "msa" || kernel == "both" {
        let msa = drive(&graph, Kernel::Msa, jobs, reads, sweeps);
        report("msa", &msa, reads, sweeps);
    }
    if kernel == "sa" || kernel == "both" {
        let sa = drive(&graph, Kernel::Sa, jobs, reads, sweeps);
        report("sa ", &sa, reads, sweeps);
    }
}
