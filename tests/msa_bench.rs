// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Throughput benchmark for the multi-spin kernel, and SA at the same shape
//! for comparison. `#[ignore]`: minutes of GPU time. Run with
//!
//! ```sh
//! RUST_LOG=quip_miner_metal=debug cargo test --release --test msa_bench -- --ignored --nocapture
//! ```
//!
//! Knobs: `QUIP_BENCH_JOBS` (default 40), `QUIP_BENCH_SWEEPS` (default 7392),
//! `QUIP_BENCH_READS` (default 128), `QUIP_METAL_TG_PER_CORE` (see
//! `streaming::tg_budget`).

#![expect(
    clippy::print_stderr,
    reason = "the benchmark reports jobs/s on stderr, as the task brief specifies"
)]

use quip_miner_metal::metal_device::MetalDevice;
use quip_miner_metal::streaming::{run_stream, GpuGovernor};
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

/// A bipartite, 20-regular graph on `n` (even) nodes: left node `a` joins
/// right nodes `half + (a * 20 + k) mod half` for `k` in `0..20`, which gives
/// every right node exactly 20 left neighbours. Random `J` in {-1, 1} and `h`
/// in {-1, 0, 1}. Degree 20 matches Zephyr; the greedy colouring stays small.
fn bipartite_regular(n: usize, seed: u64) -> IsingGraph {
    assert_eq!(n % 2, 0, "n must be even");
    let half = n / 2;
    let mut s = seed | 1;
    let mut edges = Vec::with_capacity(half * 20);
    let mut j = Vec::with_capacity(half * 20);
    for a in 0..half {
        for k in 0..20 {
            edges.push((a, half + (a * 20 + k) % half));
            j.push(if xorshift64(&mut s) & 1 == 0 {
                1.0
            } else {
                -1.0
            });
        }
    }
    let h = (0..n)
        .map(|_| [-1.0, 0.0, 1.0][(xorshift64(&mut s) % 3) as usize])
        .collect();
    IsingGraph::new(h, j, edges)
}

struct Run {
    jobs: usize,
    wall_s: f64,
    best_energy: i64,
    mean_best: f64,
}

/// Stream `jobs` copies of one 4576-node problem through `kernel` and time
/// the whole run, wall clock, from first send to last result.
fn drive(kernel: Kernel, jobs: usize, num_reads: usize, num_sweeps: usize) -> Run {
    let graph = bipartite_regular(4576, 1);
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

    let msa = drive(Kernel::Msa, jobs, reads, sweeps);
    report("msa", &msa, reads, sweeps);
    let sa = drive(Kernel::Sa, jobs, reads, sweeps);
    report("sa ", &sa, reads, sweeps);
}
