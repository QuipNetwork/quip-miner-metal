// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Device tests for the MSA probe cascade.
//!
//! Each test opens its own sampler, tunes the always-on cascade through
//! `Sampler::apply_config`, and drives `Sampler::sample_stream`. A missing
//! Metal device skips the test. The helpers follow `tests/streaming.rs`;
//! integration test binaries do not share modules.

use quip_miner_metal::iokit_gov::UtilGovernor;
use quip_miner_metal::metal_device::MetalDevice;
use quip_miner_metal::{IsingGraph, Kernel, MetalSampler};
use quip_solver_core::quip_protocol::scoring::energy_milli;
use quip_solver_core::{
    CancelToken, SampleParams, Sampler, StreamJob, StreamOutcome, StreamResult,
};
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{Receiver, Sender};

/// Probe-to-full denominators. Each of the two stages keeps the square root.
const CASCADE_TOML: &str = "\
cascade_stages = [8, 32]
cascade_keep = 20
cascade_keep_min = 10
cascade_keep_max = 40";

/// Same read count as `tests/streaming.rs` `light_params`.
const NUM_READS: usize = 4;

const CHAIN_NODES: usize = 64;

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// Ferromagnetic chain. Fields are ternary and couplings are -1, so the
/// device score matches consensus `energy_milli`.
fn chain(n: usize) -> IsingGraph {
    let h: Vec<f64> = (0..n).map(|i| [-1.0, 0.0, 1.0][i % 3]).collect();
    let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
    let j = vec![-1.0; edges.len()];
    IsingGraph::new(h, j, edges)
}

/// Same chain closed into a ring, so the edge list misses the topology cache.
fn ring(n: usize) -> IsingGraph {
    let mut graph = chain(n);
    graph.edges.push((n - 1, 0));
    graph.j.push(-1.0);
    graph
}

fn job_id(prefix: &str, index: usize) -> Vec<u8> {
    format!("{prefix}-{index}").into_bytes()
}

fn make_job(
    job_id: Vec<u8>,
    graph: IsingGraph,
    num_sweeps: usize,
    seed: u64,
    watermark: Option<u64>,
) -> StreamJob {
    StreamJob {
        job_id,
        graph,
        params: SampleParams {
            num_reads: NUM_READS,
            num_sweeps,
            sweeps_per_beta: 1,
            beta_range: None,
            seed,
        },
        watermark,
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => match payload.downcast::<&str>() {
            Ok(message) => (*message).to_owned(),
            Err(_) => "worker panicked".to_owned(),
        },
    }
}

/// Wall-clock guard. The panic payload is forwarded so a failed assertion
/// stays visible on the test thread.
fn with_timeout<F>(secs: u64, label: &str, f: F)
where
    F: FnOnce() + Send + 'static,
{
    let (done_tx, done_rx) = mpsc::channel();
    thread::spawn(move || {
        let result = catch_unwind(AssertUnwindSafe(f));
        let _ = done_tx.send(result);
    });
    match done_rx.recv_timeout(Duration::from_secs(secs)) {
        Ok(Ok(())) => {}
        Ok(Err(payload)) => panic!("{label}: {}", panic_message(payload)),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("{label}: timed out after {secs}s");
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("{label}: worker ended before completing");
        }
    }
}

fn spawn_cascade(
    jobs: Receiver<StreamJob>,
    out: Sender<StreamResult>,
    cancel: CancelToken,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let device = MetalDevice::open(0).unwrap_or_else(|error| {
            panic!("Metal device 0 required for cascade tests: {error}");
        });
        let gov = UtilGovernor::start(0, 100, false);
        let sampler = MetalSampler::new(device, gov, Kernel::Msa);
        sampler.apply_config(CASCADE_TOML);
        sampler.sample_stream(jobs, out, cancel);
    })
}

fn join_with_timeout(handle: thread::JoinHandle<()>, secs: u64, label: &str) {
    let (done_tx, done_rx) = mpsc::channel();
    thread::spawn(move || {
        let result = handle.join();
        let _ = done_tx.send(result);
    });
    match done_rx.recv_timeout(Duration::from_secs(secs)) {
        Ok(Ok(())) => {}
        Ok(Err(payload)) => panic!(
            "{label}: stream worker panicked: {}",
            panic_message(payload)
        ),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("{label}: stream worker join timed out after {secs}s");
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("{label}: join helper disconnected");
        }
    }
}

fn drain_results(mut rx: Receiver<StreamResult>, secs: u64, label: &str) -> Vec<StreamResult> {
    let (done_tx, done_rx) = mpsc::channel();
    thread::spawn(move || {
        let mut out = Vec::new();
        while let Some(result) = rx.blocking_recv() {
            out.push(result);
        }
        let _ = done_tx.send(out);
    });
    match done_rx.recv_timeout(Duration::from_secs(secs)) {
        Ok(results) => results,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("{label}: draining results timed out after {secs}s");
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("{label}: result drain worker panicked");
        }
    }
}

fn run_jobs(jobs: Vec<StreamJob>, secs: u64, label: &str) -> Vec<StreamResult> {
    // The batch is queued before results are drained. Both channels have to
    // hold the whole batch, or a full result queue stalls the relay while the
    // sender is still blocked on the job queue.
    let slots = jobs.len().max(1);
    let (job_tx, job_rx) = tokio::sync::mpsc::channel(slots);
    let (out_tx, out_rx) = tokio::sync::mpsc::channel(slots);
    let worker = spawn_cascade(job_rx, out_tx, CancelToken::default());
    for job in jobs {
        if job_tx.blocking_send(job).is_err() {
            join_with_timeout(worker, 15, label);
            panic!("{label}: job channel closed before the batch was sent");
        }
    }
    drop(job_tx);
    let results = drain_results(out_rx, secs, label);
    join_with_timeout(worker, 30, label);
    results
}

fn assert_one_each(results: &[StreamResult], ids: &[Vec<u8>]) {
    let mut seen: HashMap<Vec<u8>, usize> = HashMap::new();
    for result in results {
        *seen.entry(result.job_id.clone()).or_default() += 1;
    }
    assert_eq!(
        results.len(),
        ids.len(),
        "expected one result per job, got {} results for {} jobs",
        results.len(),
        ids.len()
    );
    assert_eq!(seen.len(), ids.len(), "duplicate job id in cascade results");
    for id in ids {
        assert_eq!(
            seen.get(id).copied().unwrap_or(0),
            1,
            "job {} result count",
            String::from_utf8_lossy(id)
        );
    }
}

fn completed_reads(result: &StreamResult) -> &[quip_miner_metal::SamplerResult] {
    match &result.outcome {
        StreamOutcome::Completed(Ok(reads)) => reads,
        StreamOutcome::Completed(Err(error)) => panic!(
            "job {} failed: {error}",
            String::from_utf8_lossy(&result.job_id)
        ),
        StreamOutcome::Cancelled => panic!(
            "job {} was cancelled",
            String::from_utf8_lossy(&result.job_id)
        ),
    }
}

/// Whether `reads` is the screened-out answer for a job whose probe returned
/// `probe`: the probe's lowest-energy read, first among ties.
fn is_probe_best(
    reads: &[quip_miner_metal::SamplerResult],
    probe: &[quip_miner_metal::SamplerResult],
) -> bool {
    let best = probe
        .iter()
        .enumerate()
        .min_by_key(|&(index, read)| (read.energy_milli, index))
        .map(|(_, read)| read);
    reads.len() == 1 && best == reads.first()
}

/// A kept job returns every read. A screened-out job returns only its best.
fn assert_reads(result: &StreamResult, num_reads: usize, nodes: usize) {
    let reads = completed_reads(result);
    assert!(
        reads.len() == num_reads || reads.len() == 1,
        "job {} read count {} is neither {num_reads} nor 1",
        String::from_utf8_lossy(&result.job_id),
        reads.len()
    );
    for (index, sample) in reads.iter().enumerate() {
        assert_eq!(
            sample.spins.len(),
            nodes,
            "job {} read {index} spin length",
            String::from_utf8_lossy(&result.job_id)
        );
        assert!(
            sample.spins.iter().all(|&spin| spin == 1 || spin == -1),
            "job {} read {index} has a spin outside ±1",
            String::from_utf8_lossy(&result.job_id)
        );
    }
}

fn assert_consensus(graph: &IsingGraph, result: &StreamResult) {
    let reads = completed_reads(result);
    for (index, sample) in reads.iter().enumerate() {
        let want = energy_milli(&sample.spins, &graph.h, &graph.j, &graph.edges);
        assert_eq!(
            sample.energy_milli,
            want,
            "job {} read {index} energy_milli",
            String::from_utf8_lossy(&result.job_id)
        );
    }
}

fn index_results(results: &[StreamResult]) -> HashMap<Vec<u8>, &StreamResult> {
    let mut by_id = HashMap::new();
    for result in results {
        assert!(
            by_id.insert(result.job_id.clone(), result).is_none(),
            "duplicate job id {}",
            String::from_utf8_lossy(&result.job_id)
        );
    }
    by_id
}

#[test]
fn every_job_returns_exactly_one_result() {
    if MetalDevice::device_count() == 0 {
        return;
    }
    let _gpu = gpu_lock();
    with_timeout(300, "cascade_answers_every_job_once", || {
        let graph = chain(CHAIN_NODES);
        let jobs: Vec<_> = (0..400)
            .map(|index| {
                let mut job = make_job(
                    job_id("chain", index),
                    graph.clone(),
                    [8, 128, 256, 64][index / 100],
                    u64::try_from(index).unwrap_or(1) + 1,
                    None,
                );
                if index >= 200 {
                    job.params.num_reads = 8;
                }
                job
            })
            .collect();
        let ids: Vec<_> = jobs.iter().map(|job| job.job_id.clone()).collect();
        let read_counts: HashMap<_, _> = jobs
            .iter()
            .map(|job| (job.job_id.clone(), job.params.num_reads))
            .collect();
        let results = run_jobs(jobs, 240, "cascade_answers_every_job_once");
        assert_one_each(&results, &ids);
        for result in &results {
            assert_reads(result, read_counts[&result.job_id], CHAIN_NODES);
        }
    });
}

#[test]
fn cascade_results_have_consensus_energies() {
    if MetalDevice::device_count() == 0 {
        return;
    }
    let _gpu = gpu_lock();
    with_timeout(300, "cascade_results_have_consensus_energies", || {
        let graph = chain(CHAIN_NODES);
        let jobs: Vec<_> = (0..128)
            .map(|index| {
                make_job(
                    job_id("energy", index),
                    graph.clone(),
                    256,
                    u64::try_from(index).unwrap_or(1) + 1,
                    None,
                )
            })
            .collect();
        let ids: Vec<_> = jobs.iter().map(|job| job.job_id.clone()).collect();
        let results = run_jobs(jobs, 240, "cascade_results_have_consensus_energies");
        assert_one_each(&results, &ids);
        for result in &results {
            assert_reads(result, NUM_READS, CHAIN_NODES);
            assert_consensus(&graph, result);
        }
    });
}

#[test]
fn cascade_completes_mixed_exact_and_non_exact_jobs() {
    if MetalDevice::device_count() == 0 {
        return;
    }
    let _gpu = gpu_lock();
    with_timeout(
        180,
        "cascade_completes_mixed_exact_and_non_exact_jobs",
        || {
            let graphs = [
                ring(8),
                IsingGraph::new(vec![1.0, -1.0], vec![0.5], vec![(0, 1)]),
                ring(8),
                IsingGraph::new(
                    vec![1.0, -1.0, 0.25],
                    vec![0.5, -0.75],
                    vec![(0, 1), (1, 2)],
                ),
                ring(16),
            ];
            let jobs: Vec<_> = graphs
                .iter()
                .enumerate()
                .map(|(index, graph)| {
                    make_job(job_id("mixed", index), graph.clone(), 256, 42, None)
                })
                .collect();
            let ids: Vec<_> = jobs.iter().map(|job| job.job_id.clone()).collect();
            let results = run_jobs(
                jobs,
                120,
                "cascade_completes_mixed_exact_and_non_exact_jobs",
            );
            assert_one_each(&results, &ids);
            let by_id = index_results(&results);
            for (graph, id) in graphs.iter().zip(&ids) {
                let result = by_id[id];
                assert_reads(result, NUM_READS, graph.num_nodes());
                assert_consensus(graph, result);
            }
        },
    );
}

#[test]
fn cascade_passes_short_jobs_through() {
    if MetalDevice::device_count() == 0 {
        return;
    }
    let _gpu = gpu_lock();
    with_timeout(180, "cascade_passes_short_jobs_through", || {
        let graph = chain(CHAIN_NODES);
        let num_reads = NUM_READS;
        let jobs: Vec<_> = (0..64)
            .map(|index| {
                make_job(
                    job_id("short", index),
                    graph.clone(),
                    8,
                    u64::try_from(index).unwrap_or(1) + 1,
                    None,
                )
            })
            .collect();
        let ids: Vec<_> = jobs.iter().map(|job| job.job_id.clone()).collect();
        let results = run_jobs(jobs, 120, "cascade_passes_short_jobs_through");
        assert_one_each(&results, &ids);
        for result in &results {
            assert_reads(result, num_reads, CHAIN_NODES);
        }
    });
}

#[test]
fn cascade_cancel_mid_stream() {
    if MetalDevice::device_count() == 0 {
        return;
    }
    let _gpu = gpu_lock();
    with_timeout(300, "cascade_cancel_mid_stream", || {
        let graph = chain(CHAIN_NODES);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(256);
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(256);
        let cancel = CancelToken::default();
        let worker = spawn_cascade(job_rx, out_tx, cancel.clone());
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let producer = thread::spawn(move || {
            for index in 0..200 {
                let job = make_job(
                    job_id("wm1", index),
                    graph.clone(),
                    256,
                    u64::try_from(index).unwrap_or(1) + 1,
                    Some(1),
                );
                job_tx.blocking_send(job).expect("send watermark 1 job");
            }
            release_rx.recv().expect("release watermark 2");
            for index in 0..50 {
                let job = make_job(
                    job_id("wm2", index),
                    graph.clone(),
                    256,
                    1_000 + u64::try_from(index).unwrap_or(1),
                    Some(2),
                );
                job_tx.blocking_send(job).expect("send watermark 2 job");
            }
            drop(job_tx);
        });

        let deadline = Instant::now() + Duration::from_secs(180);
        let mut results = Vec::new();
        while results.len() < 50 {
            if Instant::now() >= deadline {
                panic!(
                    "cascade_cancel_mid_stream: timed out with {} of 50 results",
                    results.len()
                );
            }
            match out_rx.try_recv() {
                Ok(result) => results.push(result),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                    thread::sleep(Duration::from_millis(1));
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                    panic!(
                        "cascade_cancel_mid_stream: stream closed after {} results",
                        results.len()
                    );
                }
            }
        }
        cancel.cancel_through(1);
        release_tx.send(()).expect("signal watermark 2");
        results.extend(drain_results(out_rx, 180, "cascade_cancel_mid_stream"));
        producer
            .join()
            .unwrap_or_else(|payload| panic!("{}", panic_message(payload)));
        join_with_timeout(worker, 30, "cascade_cancel_mid_stream");

        let mut ids = Vec::new();
        for index in 0..200 {
            ids.push(job_id("wm1", index));
        }
        for index in 0..50 {
            ids.push(job_id("wm2", index));
        }
        assert_one_each(&results, &ids);
        let by_id = index_results(&results);
        for index in 0..50 {
            let id = job_id("wm2", index);
            let result = by_id.get(&id).unwrap_or_else(|| {
                panic!("missing watermark 2 job {}", String::from_utf8_lossy(&id))
            });
            assert_reads(result, NUM_READS, CHAIN_NODES);
        }
    });
}

#[test]
fn topology_change_drains_live_slots() {
    if MetalDevice::device_count() == 0 {
        return;
    }
    let _gpu = gpu_lock();
    with_timeout(360, "cascade_topology_switch", || {
        let chain = chain(CHAIN_NODES);
        let ring = ring(CHAIN_NODES);
        let mut jobs = Vec::with_capacity(600);
        let mut chain_ids = Vec::with_capacity(300);
        let mut ring_ids = Vec::with_capacity(300);
        for index in 0..300 {
            let id = job_id("chain", index);
            chain_ids.push(id.clone());
            jobs.push(make_job(
                id,
                chain.clone(),
                256,
                u64::try_from(index).unwrap_or(1) + 1,
                None,
            ));
        }
        for index in 0..300 {
            let id = job_id("ring", index);
            ring_ids.push(id.clone());
            jobs.push(make_job(
                id,
                ring.clone(),
                256,
                10_000 + u64::try_from(index).unwrap_or(1),
                None,
            ));
        }
        let mut ids = chain_ids.clone();
        ids.extend(ring_ids.iter().cloned());
        let results = run_jobs(jobs, 300, "cascade_topology_switch");
        assert_one_each(&results, &ids);
        let by_id = index_results(&results);
        for id in &chain_ids {
            let result = by_id
                .get(id)
                .unwrap_or_else(|| panic!("missing chain job {}", String::from_utf8_lossy(id)));
            assert_reads(result, NUM_READS, CHAIN_NODES);
            assert_consensus(&chain, result);
        }
        for id in &ring_ids {
            let result = by_id
                .get(id)
                .unwrap_or_else(|| panic!("missing ring job {}", String::from_utf8_lossy(id)));
            assert_reads(result, NUM_READS, CHAIN_NODES);
            assert_consensus(&ring, result);
        }
    });
}

#[test]
fn screened_out_results_carry_probe_reads_and_kept_results_carry_full_reads() {
    if MetalDevice::device_count() == 0 {
        return;
    }
    let _gpu = gpu_lock();
    with_timeout(300, "checkpoint reads", || {
        // A field-free ring leaves domain walls for longer schedules to remove.
        // The short ternary chain often reaches its ground state at the probe.
        let mut graph = ring(256);
        graph.h.fill(0.0);
        let jobs = |sweeps| {
            (0..2000)
                .map(|i| {
                    make_job(
                        job_id("probe", i),
                        graph.clone(),
                        sweeps,
                        i as u64 + 1,
                        None,
                    )
                })
                .collect()
        };
        let probes = run_jobs(jobs(8), 120, "probe baseline");
        let results = run_jobs(jobs(256), 120, "checkpoint reads");
        let probes = index_results(&probes);
        let mut screened = 0;
        let mut continued = 0;
        let mut improvement = 0i64;
        for result in &results {
            assert_consensus(&graph, result);
            let probe = completed_reads(probes[&result.job_id]);
            let reads = completed_reads(result);
            if is_probe_best(reads, probe) {
                screened += 1;
            } else {
                continued += 1;
                improvement += probe.iter().map(|r| r.energy_milli).min().unwrap()
                    - reads.iter().map(|r| r.energy_milli).min().unwrap();
            }
        }
        assert_eq!(results.len(), 2000);
        assert!(screened > 0, "warm-up must return probe reads");
        assert!(continued > 0, "settled controller must continue some jobs");
        assert!(
            improvement >= 0,
            "continued set should improve on average: total {improvement}, jobs {continued}"
        );
    });
}

#[test]
fn closed_output_waits_for_inflight_and_returns() {
    if MetalDevice::device_count() == 0 {
        return;
    }
    let _gpu = gpu_lock();
    with_timeout(60, "closed output", || {
        let (tx, rx) = tokio::sync::mpsc::channel(128);
        let (out, mut results) = tokio::sync::mpsc::channel(1);
        for i in 0..128 {
            tx.blocking_send(make_job(
                job_id("close", i),
                chain(CHAIN_NODES),
                4096,
                i as u64 + 1,
                None,
            ))
            .unwrap();
        }
        let worker = spawn_cascade(rx, out, CancelToken::default());
        assert!(results.blocking_recv().is_some());
        drop(results);
        // Keep the input sender open: output closure must end the runner.
        join_with_timeout(worker, 30, "closed output");
        drop(tx);
    });
}

#[test]
fn default_and_removed_cascade_key_return_screened_reads() {
    if MetalDevice::device_count() == 0 {
        return;
    }
    let _gpu = gpu_lock();
    with_timeout(120, "always-on cascade", || {
        // A field-free ring leaves domain walls for the full budget to remove.
        let mut graph = ring(256);
        graph.h.fill(0.0);
        let execute = |config: &str, sweeps| {
            let (tx, rx) = tokio::sync::mpsc::channel(32);
            let (out, results) = tokio::sync::mpsc::channel(32);
            for i in 0..32 {
                tx.blocking_send(make_job(
                    job_id("always-on", i),
                    graph.clone(),
                    sweeps,
                    i as u64 + 1,
                    None,
                ))
                .unwrap();
            }
            drop(tx);
            let sampler = MetalSampler::new(
                MetalDevice::open(0).unwrap(),
                UtilGovernor::start(0, 100, false),
                Kernel::Msa,
            );
            sampler.apply_config(config);
            sampler.sample_stream(rx, out, CancelToken::default());
            drain_results(results, 30, "always-on cascade")
        };
        // A job whose budget equals the first default stage runs one segment,
        // so its reads are the probe reads of the longer jobs below.
        let probes = execute("", 32);
        let probes = index_results(&probes);
        for config in ["", "cascade = false"] {
            let results = execute(config, 512);
            assert_eq!(results.len(), 32);
            let mut screened = 0;
            for result in results {
                assert_consensus(&graph, &result);
                if is_probe_best(
                    completed_reads(&result),
                    completed_reads(probes[&result.job_id]),
                ) {
                    screened += 1;
                }
            }
            assert!(
                screened > 0,
                "{config:?} must return probe reads before the full budget"
            );
        }
    });
}

#[test]
fn cancelled_continuing_job_answers_once() {
    if MetalDevice::device_count() == 0 {
        return;
    }
    let _gpu = gpu_lock();
    with_timeout(180, "cancel continuing", || {
        let (tx, rx) = tokio::sync::mpsc::channel(65);
        let (out, mut results) = tokio::sync::mpsc::channel(65);
        let cancel = CancelToken::default();
        let worker_cancel = cancel.clone();
        let worker = thread::spawn(move || {
            let sampler = MetalSampler::new(
                MetalDevice::open(0).unwrap(),
                UtilGovernor::start(0, 100, false),
                Kernel::Msa,
            );
            sampler.apply_config("cascade_stages = [8]\ncascade_keep = 2\ncascade_keep_min = 2\ncascade_keep_max = 2");
            // Denominator 2 is the loosest valid keep. The warm-up screens
            // every job for its first 200 observations.
            let (warm_tx, warm_rx) = tokio::sync::mpsc::channel(500);
            let (warm_out, warm_results) = tokio::sync::mpsc::channel(500);
            for i in 0..500 {
                warm_tx
                    .blocking_send(make_job(
                        job_id("warm", i),
                        chain(CHAIN_NODES),
                        16,
                        i as u64 + 1,
                        None,
                    ))
                    .unwrap();
            }
            drop(warm_tx);
            sampler.sample_stream(warm_rx, warm_out, CancelToken::default());
            assert_eq!(drain_results(warm_results, 30, "warm-up").len(), 500);
            sampler.sample_stream(rx, out, worker_cancel);
        });
        for i in 0..64 {
            tx.blocking_send(make_job(
                job_id("continuing", i),
                chain(CHAIN_NODES),
                65_536,
                i as u64 + 1000,
                Some(1),
            ))
            .unwrap();
        }
        // The short job reports after earlier pools have passed their probe.
        tx.blocking_send(make_job(
            b"probe".to_vec(),
            chain(CHAIN_NODES),
            8,
            2000,
            None,
        ))
        .unwrap();
        drop(tx);
        // Jobs screened out at the first gate may answer before the probe.
        let mut answered = Vec::new();
        loop {
            let result = results.blocking_recv().expect("probe result");
            if result.job_id == b"probe" {
                assert_reads(&result, NUM_READS, CHAIN_NODES);
                break;
            }
            answered.push(result);
        }
        cancel.cancel_through(1);
        answered.extend(drain_results(results, 60, "cancel continuing"));
        let ids: Vec<_> = (0..64).map(|i| job_id("continuing", i)).collect();
        assert_one_each(&answered, &ids);
        let cancelled = answered
            .iter()
            .filter(|r| matches!(r.outcome, StreamOutcome::Cancelled))
            .count();
        assert!(cancelled > 0, "no continuing job was cancelled");
        join_with_timeout(worker, 30, "cancel continuing");
    });
}
