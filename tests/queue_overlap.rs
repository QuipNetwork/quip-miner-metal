// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Whether two Metal command queues overlap on the GPU. One device handle runs
//! lone full-budget multi-spin jobs back to back; a second handle, with its own
//! command queue, runs 20-job probe batches. If the probe rate with the long
//! jobs running stays near its rate alone, work from the two queues overlaps.
//!
//! ```sh
//! cargo test --release --test queue_overlap -- --ignored --nocapture
//! ```

#![expect(
    clippy::print_stderr,
    reason = "the experiment reports its rates on stderr"
)]

use quip_miner_metal::metal_device::MetalDevice;
use quip_miner_metal::streaming::{run_stream, GpuGovernor};
use quip_miner_metal::{IsingGraph, Kernel};
use quip_solver_core::{CancelToken, SampleParams, StreamJob, StreamOutcome};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
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

fn job(graph: &IsingGraph, id: usize, num_sweeps: usize) -> StreamJob {
    StreamJob {
        job_id: format!("job-{id}").into_bytes(),
        graph: graph.clone(),
        params: SampleParams {
            num_reads: 64,
            num_sweeps,
            sweeps_per_beta: 1,
            beta_range: None,
            seed: id as u64,
        },
        watermark: None,
    }
}

/// Probe jobs per second over `jobs` 64-read, 32-sweep jobs on a fresh handle.
fn probe_rate(graph: &IsingGraph, jobs: usize) -> f64 {
    let (tx, rx) = tokio::sync::mpsc::channel(jobs);
    let (out, mut results) = tokio::sync::mpsc::channel(jobs);
    for i in 0..jobs {
        tx.blocking_send(job(graph, i, 32)).expect("send probe");
    }
    drop(tx);
    let cancel = CancelToken::default();
    let start = Instant::now();
    let worker = std::thread::spawn(move || {
        let device = MetalDevice::open(0).expect("device");
        run_stream(&device, Kernel::Msa, rx, &out, &NoGovernor, &cancel);
    });
    let mut done = 0;
    while let Some(r) = results.blocking_recv() {
        assert!(matches!(r.outcome, StreamOutcome::Completed(Ok(_))));
        done += 1;
    }
    worker.join().expect("probe worker");
    assert_eq!(done, jobs);
    done as f64 / start.elapsed().as_secs_f64()
}

/// Run lone full-budget jobs, one in flight at a time, until `stop` is set.
/// Returns the count and the mean latency in milliseconds.
fn lone_full_jobs(graph: IsingGraph, stop: Arc<AtomicBool>, count: Arc<AtomicUsize>) -> f64 {
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let (out, mut results) = tokio::sync::mpsc::channel(1);
    let cancel = CancelToken::default();
    let worker = std::thread::spawn(move || {
        let device = MetalDevice::open(0).expect("device");
        run_stream(&device, Kernel::Msa, rx, &out, &NoGovernor, &cancel);
    });
    let mut total_ms = 0.0;
    let mut id = 0;
    while !stop.load(Ordering::Relaxed) {
        let start = Instant::now();
        tx.blocking_send(job(&graph, id, 14_336))
            .expect("send full");
        let r = results.blocking_recv().expect("full result");
        assert!(matches!(r.outcome, StreamOutcome::Completed(Ok(_))));
        total_ms += start.elapsed().as_secs_f64() * 1e3;
        id += 1;
        count.store(id, Ordering::Relaxed);
    }
    drop(tx);
    worker.join().expect("full worker");
    total_ms / id.max(1) as f64
}

#[test]
#[ignore = "GPU experiment: about a minute of device time"]
fn two_queues_overlap_a_lone_long_job_with_probe_batches() {
    if MetalDevice::device_count() == 0 {
        eprintln!("skipping: no Metal device");
        return;
    }
    let graph = advantage2_system1(7);
    let jobs = 4_000;
    let alone = probe_rate(&graph, jobs);

    let stop = Arc::new(AtomicBool::new(false));
    let count = Arc::new(AtomicUsize::new(0));
    let long = {
        let (graph, stop, count) = (graph.clone(), stop.clone(), count.clone());
        std::thread::spawn(move || lone_full_jobs(graph, stop, count))
    };
    // Let the first long job reach the GPU before timing the probes.
    std::thread::sleep(std::time::Duration::from_millis(200));
    let beside = probe_rate(&graph, jobs);
    stop.store(true, Ordering::Relaxed);
    let full_ms = long.join().expect("long thread");
    let full_jobs = count.load(Ordering::Relaxed);

    eprintln!("probe rate alone: {alone:.0} jobs/s");
    eprintln!(
        "probe rate beside lone full jobs: {beside:.0} jobs/s ({:.0} percent)",
        100.0 * beside / alone
    );
    eprintln!("lone full jobs: {full_jobs}, mean latency {full_ms:.0} ms");
}
