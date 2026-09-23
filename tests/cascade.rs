// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Device tests for the MSA probe cascade.
//!
//! Each test opens its own sampler, applies the cascade through
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
cascade = true
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

fn assert_reads(result: &StreamResult, num_reads: usize, nodes: usize) {
    let reads = completed_reads(result);
    assert_eq!(
        reads.len(),
        num_reads,
        "job {} read count",
        String::from_utf8_lossy(&result.job_id)
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
fn cascade_answers_every_job_once() {
    if MetalDevice::device_count() == 0 {
        return;
    }
    let _gpu = gpu_lock();
    with_timeout(300, "cascade_answers_every_job_once", || {
        let graph = chain(CHAIN_NODES);
        let jobs: Vec<_> = (0..400)
            .map(|index| {
                make_job(
                    job_id("chain", index),
                    graph.clone(),
                    256,
                    u64::try_from(index).unwrap_or(1) + 1,
                    None,
                )
            })
            .collect();
        let ids: Vec<_> = jobs.iter().map(|job| job.job_id.clone()).collect();
        let results = run_jobs(jobs, 240, "cascade_answers_every_job_once");
        assert_one_each(&results, &ids);
        for result in &results {
            assert_reads(result, NUM_READS, CHAIN_NODES);
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
fn cascade_topology_switch() {
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
