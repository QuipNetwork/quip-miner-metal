//! Integration tests for `streaming::run_stream` (batched production path).
//!
//! Unit tests in `src/streaming.rs` cover internals (`next_seed`, `fill_batch`
//! keying, `scale_budget`). This file drives the composed loop through the
//! public interface: jobs in → `form_and_commit` / `finish_batch` → results out.
//!
//! Requires a Metal device (Apple Silicon).

use quip_miner_metal::iokit_gov::UtilGovernor;
use quip_miner_metal::metal_device::MetalDevice;
use quip_miner_metal::streaming::{run_stream, GpuGovernor};
use quip_miner_metal::{IsingGraph, Kernel, MetalSampler};
use quip_solver_core::{
    CancelToken, SampleParams, Sampler, StreamJob, StreamOutcome, StreamResult,
};
use std::collections::HashMap;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{Receiver, Sender};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Open device 0 for each test. Same pattern as `tests/golden_parity.rs`.
fn open_device() -> MetalDevice {
    MetalDevice::open(0).unwrap_or_else(|e| {
        panic!("Metal device 0 required for streaming tests: {e}");
    })
}

/// Governor stub: never throttles, never scales. Mirrors the `NoGovernor` in
/// `src/streaming.rs`'s unit-test module (cfg(test) there — not importable).
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

fn ring4() -> IsingGraph {
    IsingGraph::new(
        vec![1.0, -1.0, 0.0, 1.0],
        vec![1.0, -1.0, 1.0, -1.0],
        vec![(0, 1), (1, 2), (2, 3), (3, 0)],
    )
}

fn light_params(seed: u64) -> SampleParams {
    SampleParams {
        num_reads: 4,
        num_sweeps: 16,
        sweeps_per_beta: 1,
        beta_range: None,
        seed,
    }
}

/// `generation` follows the coordinator's own vocabulary; it is converted to
/// the harness's `watermark: Option<u64>` here with the same rule
/// `quip-solver-core`'s `prepare_job` applies on the real path: `0` means the
/// job can never be cancelled (`None`), any other value is a cancellable
/// watermark (`Some`).
fn make_job(job_id: &[u8], generation: u64, seed: u64) -> StreamJob {
    StreamJob {
        job_id: job_id.to_vec(),
        graph: ring4(),
        params: light_params(seed),
        watermark: (generation != 0).then_some(generation),
    }
}

/// Wall-clock guard: a hanging stream test is worse than the gap it fills.
fn with_timeout<F>(secs: u64, label: &str, f: F)
where
    F: FnOnce() + Send + 'static,
{
    let (done_tx, done_rx) = mpsc::channel();
    thread::spawn(move || {
        f();
        let _ = done_tx.send(());
    });
    match done_rx.recv_timeout(Duration::from_secs(secs)) {
        Ok(()) => {}
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("{label}: timed out after {secs}s");
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("{label}: worker panicked before completing");
        }
    }
}

/// Spawn `run_stream` on a dedicated thread (device stays on that thread).
fn spawn_stream(
    kernel: Kernel,
    jobs: Receiver<StreamJob>,
    out: Sender<StreamResult>,
    cancel: CancelToken,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let device = open_device();
        run_stream(&device, kernel, jobs, &out, &NoGovernor, &cancel);
    })
}

/// Join a worker, panicking if it does not finish within `secs`.
fn join_with_timeout(handle: thread::JoinHandle<()>, secs: u64, label: &str) {
    let (done_tx, done_rx) = mpsc::channel();
    thread::spawn(move || {
        let result = handle.join();
        let _ = done_tx.send(result);
    });
    match done_rx.recv_timeout(Duration::from_secs(secs)) {
        Ok(Ok(())) => {}
        Ok(Err(_)) => panic!("{label}: stream worker panicked"),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("{label}: stream worker join timed out after {secs}s");
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("{label}: join helper disconnected");
        }
    }
}

/// Drain the result channel until it closes, with a wall-clock deadline.
fn drain_results(mut rx: Receiver<StreamResult>, secs: u64, label: &str) -> Vec<StreamResult> {
    let (done_tx, done_rx) = mpsc::channel();
    thread::spawn(move || {
        let mut out = Vec::new();
        while let Some(r) = rx.blocking_recv() {
            out.push(r);
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

fn assert_completed_sane(r: &StreamResult, expected_reads: usize, n: usize) {
    let StreamOutcome::Completed(Ok(reads)) = &r.outcome else {
        panic!(
            "expected Completed(Ok) for job {:?}, got {:?}",
            String::from_utf8_lossy(&r.job_id),
            outcome_kind(&r.outcome)
        );
    };
    assert_eq!(
        reads.len(),
        expected_reads,
        "job {:?} read count",
        String::from_utf8_lossy(&r.job_id)
    );
    for (i, sample) in reads.iter().enumerate() {
        assert_eq!(
            sample.spins.len(),
            n,
            "job {:?} read {i} spin length",
            String::from_utf8_lossy(&r.job_id)
        );
        assert!(
            sample.spins.iter().all(|&s| s == 1 || s == -1),
            "job {:?} read {i} has non ±1 spin",
            String::from_utf8_lossy(&r.job_id)
        );
    }
}

fn outcome_kind(o: &StreamOutcome) -> &'static str {
    match o {
        StreamOutcome::Completed(Ok(_)) => "Completed(Ok)",
        StreamOutcome::Completed(Err(_)) => "Completed(Err)",
        StreamOutcome::Cancelled => "Cancelled",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Full batch round trip on `kernel`: several matching jobs → one result each
/// with sane fields.
fn batch_round_trip(kernel: Kernel) {
    let label: &'static str = match kernel {
        Kernel::Sa => "run_stream_batch_round_trip",
        Kernel::Msa => "run_stream_msa_batch_round_trip",
        Kernel::Gibbs => "run_stream_gibbs_batch_round_trip",
    };
    with_timeout(60, label, move || {
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(8);
        let (out_tx, out_rx) = tokio::sync::mpsc::channel(8);
        let cancel = CancelToken::default();

        let worker = spawn_stream(kernel, job_rx, out_tx, cancel);

        let ids: &[&[u8]] = &[b"job-a", b"job-b", b"job-c"];
        for (i, id) in ids.iter().enumerate() {
            job_tx
                .blocking_send(make_job(id, 0, 100 + i as u64))
                .expect("send job");
        }
        drop(job_tx);

        let results = drain_results(out_rx, 45, label);
        join_with_timeout(worker, 15, label);

        assert_eq!(
            results.len(),
            ids.len(),
            "expected one StreamResult per job, got {:?}",
            results
                .iter()
                .map(|r| String::from_utf8_lossy(&r.job_id).into_owned())
                .collect::<Vec<_>>()
        );

        let mut by_id: HashMap<Vec<u8>, &StreamResult> = HashMap::new();
        for r in &results {
            assert!(
                by_id.insert(r.job_id.clone(), r).is_none(),
                "duplicate job_id {:?}",
                String::from_utf8_lossy(&r.job_id)
            );
        }
        for id in ids {
            let r = by_id
                .get(*id)
                .unwrap_or_else(|| panic!("missing result for {}", String::from_utf8_lossy(id)));
            assert_completed_sane(r, 4, 4);
            let _ = r.device_access_time_us;
        }
    });
}

#[test]
fn run_stream_batch_round_trip() {
    batch_round_trip(Kernel::Sa);
}

/// The multi-spin kernel dispatches `words` threadgroups per problem and
/// packs 32 lanes per word; the 4-read job here rounds to one 32-lane word
/// and must still come back truncated to 4 reads of 4 spins.
#[test]
fn run_stream_msa_batch_round_trip() {
    batch_round_trip(Kernel::Msa);
}

/// CancelToken mid-stream: after a live job completes, cancel stale generations
/// and assert Cancelled vs Completed for subsequent jobs.
#[test]
fn run_stream_cancel_mid_stream() {
    with_timeout(60, "run_stream_cancel_mid_stream", || {
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(8);
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(8);
        let cancel = CancelToken::default();
        let cancel_worker = cancel.clone();

        let worker = spawn_stream(Kernel::Sa, job_rx, out_tx, cancel_worker);

        // Establish the stream with a live job (generation 1).
        job_tx
            .blocking_send(make_job(b"live-1", 1, 1))
            .expect("send live-1");

        let first = {
            let deadline = Instant::now() + Duration::from_secs(30);
            loop {
                if Instant::now() >= deadline {
                    panic!("run_stream_cancel_mid_stream: timed out waiting for live-1");
                }
                match out_rx.try_recv() {
                    Ok(r) => break r,
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                        panic!("run_stream_cancel_mid_stream: out channel closed before live-1");
                    }
                }
            }
        };
        assert_eq!(first.job_id, b"live-1");
        assert_completed_sane(&first, 4, 4);

        // Mid-stream cancel: abandon generations <= 10.
        cancel.cancel_through(10);

        job_tx
            .blocking_send(make_job(b"stale", 5, 2))
            .expect("send stale");
        job_tx
            .blocking_send(make_job(b"live-2", 11, 3))
            .expect("send live-2");
        drop(job_tx);

        let rest = drain_results(out_rx, 45, "run_stream_cancel_mid_stream");
        join_with_timeout(worker, 15, "run_stream_cancel_mid_stream");

        assert_eq!(
            rest.len(),
            2,
            "expected stale + live-2, got {:?}",
            rest.iter()
                .map(|r| (
                    String::from_utf8_lossy(&r.job_id).into_owned(),
                    outcome_kind(&r.outcome)
                ))
                .collect::<Vec<_>>()
        );

        let mut by_id: HashMap<Vec<u8>, StreamOutcome> = HashMap::new();
        for r in rest {
            by_id.insert(r.job_id, r.outcome);
        }

        assert!(
            matches!(
                by_id.get(b"stale".as_slice()),
                Some(StreamOutcome::Cancelled)
            ),
            "stale generation should be Cancelled, got {:?}",
            by_id.get(b"stale".as_slice()).map(outcome_kind)
        );

        let live2 = by_id
            .get(b"live-2".as_slice())
            .expect("missing live-2 result");
        match live2 {
            StreamOutcome::Completed(Ok(reads)) => {
                assert_eq!(reads.len(), 4);
                assert!(reads.iter().all(|s| {
                    s.spins.len() == 4 && s.spins.iter().all(|&sp| sp == 1 || sp == -1)
                }));
            }
            other => panic!("live-2 expected Completed(Ok), got {}", outcome_kind(other)),
        }
    });
}

/// Closing an empty job channel must return from `run_stream` (no hang).
#[test]
fn run_stream_exits_on_closed_empty_channel() {
    with_timeout(30, "run_stream_exits_on_closed_empty_channel", || {
        let (job_tx, job_rx) = tokio::sync::mpsc::channel::<StreamJob>(1);
        let (out_tx, out_rx) = tokio::sync::mpsc::channel::<StreamResult>(1);
        drop(job_tx); // close before any job arrives

        let worker = spawn_stream(Kernel::Sa, job_rx, out_tx, CancelToken::default());

        let results = drain_results(out_rx, 20, "run_stream_exits_on_closed_empty_channel");
        join_with_timeout(worker, 10, "run_stream_exits_on_closed_empty_channel");

        assert!(
            results.is_empty(),
            "empty input must emit no results, got {}",
            results.len()
        );
    });
}

/// MetalSampler construction + Sampler-trait smoke (bead item (a), optional).
#[test]
fn metal_sampler_sample_smoke() {
    with_timeout(60, "metal_sampler_sample_smoke", || {
        let device = open_device();
        // Ceiling high, yielding off: smoke path must not park on throttle.
        let gov = UtilGovernor::start(0, 100, false);
        let sampler = MetalSampler::new(device, gov, Kernel::Sa);

        let graph = ring4();
        let params = light_params(42);
        let results = sampler
            .sample(&graph, &params)
            .expect("MetalSampler::sample");

        assert_eq!(results.len(), 4);
        for r in &results {
            assert_eq!(r.spins.len(), 4);
            assert!(r.spins.iter().all(|&s| s == 1 || s == -1));
        }

        // Stream width is a fixed positive budget for this device/kernel.
        assert!(sampler.stream_width() >= 1);
        assert_eq!(
            sampler.max_reads(),
            quip_miner_metal::streaming::max_reads(Kernel::Sa)
        );
    });
}
