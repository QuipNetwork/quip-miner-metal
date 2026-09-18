use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, TryLockError};
use std::time::{Duration, Instant};

use quip_solver_core::{
    CancelToken, IsingGraph, OpenError, SampleError, SampleParams, Sampler, SamplerResult,
    StreamJob, StreamOutcome, StreamResult,
};
use tempfile::TempDir;
use tokio::sync::mpsc;

use crate::graph::prepare;
use crate::msa::validate_params;
use crate::solver::RunOutput;
use crate::worker::{
    read_message, write_message, RawJob, WorkerErrorKind, WorkerReply, WorkerRequest, WorkerResult,
    MAX_MESSAGE_BYTES,
};
use crate::AneError;

fn fault(operation: &str, error: impl std::fmt::Display) -> SampleError {
    SampleError::DeviceFault(format!("{operation}: {error}"))
}

fn sample_error(error: AneError) -> SampleError {
    match error {
        AneError::Capacity(detail) => {
            tracing::warn!(%detail, "ANE job exceeds capacity");
            SampleError::Capacity
        }
        AneError::Runtime(detail) => SampleError::DeviceFault(detail),
    }
}

/// Wall-clock stages of one out-of-process job, in microseconds. Task 8
/// measurement instrumentation only; never read to make a decision.
#[derive(Debug, Default)]
pub(crate) struct WorkerPathStats {
    pub(crate) directory_setup_us: u64,
    pub(crate) request_write_us: u64,
    pub(crate) spawn_us: u64,
    pub(crate) wait_us: u64,
    pub(crate) reply_read_us: u64,
    pub(crate) teardown_us: u64,
}

fn elapsed_us(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)
}

pub(crate) struct WorkerProcess {
    child: Child,
    directory: TempDir,
    reply_path: PathBuf,
    pid: u32,
    reaped: bool,
    stats: WorkerPathStats,
}

impl WorkerProcess {
    pub(crate) fn spawn(executable: &Path, request: &WorkerRequest) -> Result<Self, SampleError> {
        let mut stats = WorkerPathStats::default();
        let directory_setup_started = Instant::now();
        let directory = tempfile::Builder::new()
            .prefix("quip-ane-job-")
            .tempdir()
            .map_err(|error| fault("create job directory", error))?;
        let request_path = directory.path().join("request.json");
        let reply_path = directory.path().join("response.json");
        let file =
            File::create(&request_path).map_err(|error| fault("create request file", error))?;
        stats.directory_setup_us = elapsed_us(directory_setup_started);
        let request_write_started = Instant::now();
        write_message(file, request).map_err(|error| fault("serialize request", error))?;
        stats.request_write_us = elapsed_us(request_write_started);
        let directory_setup_started = Instant::now();
        let request_file =
            File::open(&request_path).map_err(|error| fault("open request file", error))?;
        let reply_file =
            File::create(&reply_path).map_err(|error| fault("create response file", error))?;
        stats.directory_setup_us += elapsed_us(directory_setup_started);
        let spawn_started = Instant::now();
        let child = Command::new(executable)
            .arg("--ane-worker")
            .arg(std::process::id().to_string())
            .env("TMPDIR", directory.path())
            .stdin(Stdio::from(request_file))
            .stdout(Stdio::from(reply_file))
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| fault("spawn worker", error))?;
        stats.spawn_us = elapsed_us(spawn_started);
        let pid = child.id();
        Ok(Self {
            child,
            directory,
            reply_path,
            pid,
            reaped: false,
            stats,
        })
    }

    pub(crate) fn wait(
        mut self,
        should_stop: &dyn Fn() -> bool,
    ) -> Result<Option<WorkerReply>, SampleError> {
        let result = self.wait_inner(should_stop);
        let teardown_started = Instant::now();
        self.stop_and_reap()?;
        // Remove native compiler artifacts only after the process has released them.
        std::fs::remove_dir_all(self.directory.path())
            .map_err(|error| fault("remove job directory", error))?;
        self.stats.teardown_us = elapsed_us(teardown_started);
        tracing::debug!(
            pid = self.pid,
            directory_setup_us = self.stats.directory_setup_us,
            request_write_us = self.stats.request_write_us,
            spawn_us = self.stats.spawn_us,
            wait_us = self.stats.wait_us,
            reply_read_us = self.stats.reply_read_us,
            teardown_us = self.stats.teardown_us,
            "ANE worker process stages"
        );
        result
    }

    fn wait_inner(
        &mut self,
        should_stop: &dyn Fn() -> bool,
    ) -> Result<Option<WorkerReply>, SampleError> {
        let wait_started = Instant::now();
        loop {
            if let Some(status) = self
                .child
                .try_wait()
                .map_err(|error| fault("poll worker", error))?
            {
                self.reaped = true;
                self.stats.wait_us = elapsed_us(wait_started);
                if !status.success() {
                    return Err(fault("worker exited unsuccessfully", status));
                }
                let reply_read_started = Instant::now();
                let file = File::open(&self.reply_path)
                    .map_err(|error| fault("open response file", error))?;
                let reply: WorkerReply =
                    read_message(file).map_err(|error| fault("read worker response", error))?;
                self.stats.reply_read_us = elapsed_us(reply_read_started);
                if reply.pid != self.pid {
                    return Err(fault("worker response PID mismatch", reply.pid));
                }
                return Ok(Some(reply));
            }
            let length = std::fs::metadata(&self.reply_path)
                .map_err(|error| fault("inspect response size", error))?
                .len();
            if length > MAX_MESSAGE_BYTES {
                return Err(fault("worker response", "exceeds size limit"));
            }
            if should_stop() {
                self.stats.wait_us = elapsed_us(wait_started);
                return Ok(None);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn stop_and_reap(&mut self) -> Result<(), SampleError> {
        if self.reaped {
            return Ok(());
        }
        if self
            .child
            .try_wait()
            .map_err(|error| fault("poll worker before cleanup", error))?
            .is_some()
        {
            self.reaped = true;
            return Ok(());
        }
        if let Err(error) = self.child.kill() {
            if self
                .child
                .try_wait()
                .map_err(|error| fault("poll worker after kill failure", error))?
                .is_some()
            {
                self.reaped = true;
                return Ok(());
            }
            return Err(fault("kill worker", error));
        }
        self.child
            .wait()
            .map_err(|error| fault("reap worker", error))?;
        self.reaped = true;
        Ok(())
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        if let Err(error) = self.stop_and_reap() {
            tracing::error!(pid = self.pid, %error, "worker cleanup failed");
        }
    }
}

/// Serial ANE sampler whose native programs live in one disposable child per job.
pub struct AneSampler {
    executable: PathBuf,
    access: Mutex<()>,
}

impl AneSampler {
    /// Verify a completed ANE dispatch before accepting sampling jobs.
    pub fn open(executable: PathBuf) -> Result<Self, OpenError> {
        Self::open_with_cancel(executable, &|| false)
    }

    /// Verify startup while allowing the caller to abandon and reap its worker.
    pub fn open_with_cancel(
        executable: PathBuf,
        should_stop: &dyn Fn() -> bool,
    ) -> Result<Self, OpenError> {
        if should_stop() {
            return Err(OpenError("ANE startup check cancelled".into()));
        }
        let start = Instant::now();
        let worker = WorkerProcess::spawn(&executable, &WorkerRequest::Check)
            .map_err(|error| OpenError(error.to_string()))?;
        let reply = worker
            .wait(&|| should_stop() || start.elapsed() >= Duration::from_secs(30))
            .map_err(|error| OpenError(error.to_string()))?
            .ok_or_else(|| {
                OpenError(if should_stop() {
                    "ANE startup check cancelled".into()
                } else {
                    "ANE startup check exceeded 30 seconds".into()
                })
            })?;
        match reply.result {
            WorkerResult::Checked { dispatches: 1 } => Ok(Self {
                executable,
                access: Mutex::new(()),
            }),
            WorkerResult::Checked { dispatches } => Err(OpenError(format!(
                "invalid startup dispatch receipt: {dispatches}"
            ))),
            WorkerResult::Solved { output: _ } => {
                Err(OpenError("startup check returned a sample reply".into()))
            }
            WorkerResult::Failed { kind: _, detail } => {
                Err(OpenError(format!("ANE startup check failed: {detail}")))
            }
        }
    }

    /// Sample one job, stopping its child when the caller abandons the job.
    pub fn sample_job(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        should_stop: &dyn Fn() -> bool,
    ) -> StreamOutcome {
        let result = self.sample_inner(graph, params, should_stop);
        match result {
            Err(error) => StreamOutcome::Completed(Err(error)),
            Ok(None) => StreamOutcome::Cancelled,
            Ok(Some(samples)) => {
                if should_stop() {
                    StreamOutcome::Cancelled
                } else {
                    StreamOutcome::Completed(Ok(samples))
                }
            }
        }
    }

    fn sample_inner(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
        should_stop: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<SamplerResult>>, SampleError> {
        let _guard = match self.access.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Err(SampleError::DeviceBusy),
            Err(TryLockError::Poisoned(_)) => return Err(fault("sampler mutex", "poisoned")),
        };
        validate_params(params).map_err(sample_error)?;
        let prepared = prepare(graph).map_err(sample_error)?;
        if should_stop() {
            return Ok(None);
        }
        let expected_programs = if params.num_sweeps == 0 || graph.h.is_empty() {
            0
        } else {
            1
        };
        drop(prepared);
        let worker = WorkerProcess::spawn(
            &self.executable,
            &WorkerRequest::Sample(RawJob::from_parts(graph, params)),
        )?;
        let Some(reply) = worker.wait(should_stop)? else {
            return Ok(None);
        };
        let output = match reply.result {
            WorkerResult::Solved { output } => output,
            WorkerResult::Checked { dispatches: _ } => {
                return Err(fault("sampling reply", "received startup receipt"))
            }
            WorkerResult::Failed {
                kind: WorkerErrorKind::Capacity,
                detail,
            } => return Err(sample_error(AneError::Capacity(detail))),
            WorkerResult::Failed {
                kind: WorkerErrorKind::Runtime,
                detail,
            } => return Err(SampleError::DeviceFault(detail)),
        };
        validate_output(&output, graph, params, expected_programs)?;
        tracing::debug!(
            pid = reply.pid,
            programs = output.stats.programs,
            dispatches = output.stats.dispatches,
            setup_us = output.stats.setup_us,
            staging_us = output.stats.staging_us,
            dispatch_us = output.stats.dispatch_us,
            anneal_us = output.stats.anneal_us,
            "ANE worker timing"
        );
        let start = Instant::now();
        let samples = output
            .spins
            .into_iter()
            .map(|spins| {
                let energy_milli = quip_solver_core::quip_protocol::scoring::energy_milli(
                    &spins,
                    &graph.h,
                    &graph.j,
                    &graph.edges,
                );
                SamplerResult {
                    spins,
                    energy_milli,
                }
            })
            .collect();
        tracing::debug!(
            consensus_us = start.elapsed().as_micros(),
            "parent consensus scoring"
        );
        Ok(Some(samples))
    }
}

fn validate_output(
    output: &RunOutput,
    graph: &IsingGraph,
    params: &SampleParams,
    expected_programs: usize,
) -> Result<(), SampleError> {
    if output.spins.len() != params.num_reads {
        return Err(fault("sampling reply", "wrong number of reads"));
    }
    for spins in &output.spins {
        if spins.len() != graph.h.len() {
            return Err(fault("sampling reply", "wrong spin-array length"));
        }
        if spins.iter().any(|&spin| spin != -1 && spin != 1) {
            return Err(fault("sampling reply", "spin outside {-1, 1}"));
        }
    }
    if u64::from(output.stats.programs) != expected_programs as u64 {
        return Err(fault("sampling reply", "wrong program count"));
    }
    let dispatches = if expected_programs == 0 {
        0
    } else {
        params.num_sweeps.div_ceil(crate::native::BLOCK_SWEEPS) as u64
    };
    if output.stats.dispatches != dispatches {
        return Err(fault("sampling reply", "wrong dispatch count"));
    }
    Ok(())
}

impl Sampler for AneSampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        match self.sample_job(graph, params, &|| false) {
            StreamOutcome::Completed(result) => result,
            StreamOutcome::Cancelled => {
                Err(fault("sampling job", "cancelled without stop request"))
            }
        }
    }

    fn sample_stream(
        &self,
        mut jobs: mpsc::Receiver<StreamJob>,
        out: mpsc::Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        while let Some(job) = jobs.blocking_recv() {
            let start = Instant::now();
            let stop = || cancel.is_cancelled(job.watermark) || out.is_closed();
            let outcome = if stop() {
                StreamOutcome::Cancelled
            } else {
                self.sample_job(&job.graph, &job.params, &stop)
            };
            let device_access_time_us =
                u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX);
            if out
                .blocking_send(StreamResult {
                    job_id: job.job_id,
                    outcome,
                    device_access_time_us,
                })
                .is_err()
            {
                break;
            }
        }
    }

    fn stream_width(&self) -> usize {
        1
    }
    fn declared_stream_width() -> u32 {
        1
    }
    fn max_reads(&self) -> u32 {
        128
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn script(body: &str) -> (tempfile::TempDir, PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("worker");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        (directory, path)
    }

    fn assert_gone(pid: u32, directory: &Path) {
        assert!(!directory.exists());
        let status = std::process::Command::new("/bin/kill")
            .args(["-0", &pid.to_string()])
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(!status.success(), "worker {pid} remains alive");
    }

    #[test]
    fn startup_cancellation_reaps_worker_and_cleans_artifacts() {
        let fixture = tempfile::tempdir().unwrap();
        let receipt = fixture.path().join("receipt");
        let body = format!(
            "printf '%s\\n%s\\n' \"$$\" \"$TMPDIR\" > '{}'\nexec /bin/sleep 30",
            receipt.display()
        );
        let (_script_dir, path) = script(&body);
        let start = Instant::now();
        let result = AneSampler::open_with_cancel(path, &|| receipt.exists());
        assert!(matches!(result, Err(OpenError(message)) if message.contains("cancelled")));
        assert!(start.elapsed() < Duration::from_secs(2));
        let text = std::fs::read_to_string(receipt).unwrap();
        let mut lines = text.lines();
        let pid = lines.next().unwrap().parse().unwrap();
        let directory = PathBuf::from(lines.next().unwrap());
        assert_gone(pid, &directory);
    }

    #[test]
    fn cancelled_startup_does_not_launch_an_executable() {
        let result = AneSampler::open_with_cancel(PathBuf::from("/no/such/worker"), &|| true);
        assert!(matches!(result, Err(OpenError(message)) if message.contains("cancelled")));
    }

    #[test]
    fn cancellation_reaps_child_and_removes_directory() {
        let (_fixture, path) = script("exec /bin/sleep 30");
        let worker = WorkerProcess::spawn(&path, &WorkerRequest::Check).unwrap();
        let pid = worker.pid;
        let directory = worker.directory.path().to_path_buf();
        let start = Instant::now();
        assert!(worker
            .wait(&|| start.elapsed() >= Duration::from_millis(50))
            .unwrap()
            .is_none());
        assert_gone(pid, &directory);
    }

    #[test]
    fn dropping_worker_reaps_child_and_removes_directory() {
        let (_fixture, path) = script("exec /bin/sleep 30");
        let worker = WorkerProcess::spawn(&path, &WorkerRequest::Check).unwrap();
        let pid = worker.pid;
        let directory = worker.directory.path().to_path_buf();
        drop(worker);
        assert_gone(pid, &directory);
    }

    #[test]
    fn malformed_successful_children_are_faults_and_cleaned_up() {
        for body in [
            "exit 0",
            "printf '{'",
            "printf '{\"pid\":0,\"result\":{\"status\":\"checked\",\"dispatches\":1}}'",
            "exit 70",
            "exec /bin/dd if=/dev/zero bs=1048576 count=17 2>/dev/null",
        ] {
            let (_fixture, path) = script(body);
            let worker = WorkerProcess::spawn(&path, &WorkerRequest::Check).unwrap();
            let pid = worker.pid;
            let directory = worker.directory.path().to_path_buf();
            assert_fault(worker.wait(&|| false));
            assert_gone(pid, &directory);
        }
    }

    #[test]
    fn completed_failure_wins_over_cancellation() {
        let (_fixture, path) = script("exit 70");
        let mut worker = WorkerProcess::spawn(&path, &WorkerRequest::Check).unwrap();
        worker.child.wait().unwrap();
        assert_fault(worker.wait(&|| true));
    }

    #[test]
    fn valid_reply_checks_pid_and_cleans_directory() {
        let (_fixture, path) = script(
            "printf '{\"pid\":%s,\"result\":{\"status\":\"checked\",\"dispatches\":1}}' \"$$\"",
        );
        let worker = WorkerProcess::spawn(&path, &WorkerRequest::Check).unwrap();
        let pid = worker.pid;
        let directory = worker.directory.path().to_path_buf();
        let reply = worker.wait(&|| false).unwrap().unwrap();
        assert_eq!(reply.pid, pid);
        assert_gone(pid, &directory);
    }

    #[test]
    fn startup_requires_exact_dispatch_receipt() {
        for result in [
            r#"{"status":"checked","dispatches":0}"#,
            r#"{"status":"checked","dispatches":2}"#,
            r#"{"status":"failed","kind":"runtime","detail":"broken device"}"#,
        ] {
            let (_fixture, path) = script(&format!(
                "printf '{{\"pid\":%s,\"result\":{result}}}' \"$$\""
            ));
            assert!(AneSampler::open(path).is_err());
        }
        let (_fixture, path) = script(
            "printf '{\"pid\":%s,\"result\":{\"status\":\"checked\",\"dispatches\":1}}' \"$$\"",
        );
        assert!(AneSampler::open(path).is_ok());
    }

    #[test]
    fn queued_cancellation_does_not_start_worker() {
        let (fixture, path) = script("touch \"$0.started\"; exit 70");
        let sampler = AneSampler {
            executable: path.clone(),
            access: Mutex::new(()),
        };
        let (jobs_tx, jobs_rx) = mpsc::channel(1);
        let (out_tx, mut out_rx) = mpsc::channel(1);
        let cancel = CancelToken::default();
        cancel.cancel_through(1);
        jobs_tx
            .blocking_send(StreamJob {
                job_id: vec![7],
                graph: IsingGraph::new(vec![0.0], vec![], vec![]),
                params: SampleParams::default(),
                watermark: Some(1),
            })
            .unwrap();
        drop(jobs_tx);
        sampler.sample_stream(jobs_rx, out_tx, cancel);
        let reply = out_rx.blocking_recv().unwrap();
        assert_eq!(reply.job_id, vec![7]);
        match reply.outcome {
            StreamOutcome::Cancelled => {}
            StreamOutcome::Completed(_) => panic!("queued job was not cancelled"),
        }
        assert!(!fixture.path().join("worker.started").exists());
    }

    #[test]
    fn concurrent_direct_call_reports_busy() {
        let sampler = AneSampler {
            executable: PathBuf::from("unused"),
            access: Mutex::new(()),
        };
        let _guard = sampler.access.lock().unwrap();
        assert_eq!(
            sampler.sample(
                &IsingGraph::new(vec![], vec![], vec![]),
                &SampleParams::default()
            ),
            Err(SampleError::DeviceBusy)
        );
    }

    #[test]
    fn poisoned_sampler_fault_survives_active_cancellation() {
        let sampler = AneSampler {
            executable: PathBuf::from("unused"),
            access: Mutex::new(()),
        };
        let _panic = std::panic::catch_unwind(|| {
            let _guard = sampler.access.lock().unwrap();
            panic!("poison sampler lock");
        });
        let cancel = CancelToken::default();
        cancel.cancel_through(1);
        let outcome = sampler.sample_job(
            &IsingGraph::new(vec![], vec![], vec![]),
            &SampleParams::default(),
            &|| cancel.is_cancelled(Some(1)),
        );
        match outcome {
            StreamOutcome::Completed(result) => assert_fault(result),
            StreamOutcome::Cancelled => panic!("fatal error was cancelled"),
        }
    }

    #[test]
    fn response_states_and_dispatch_receipts_are_validated() {
        use crate::solver::RunStats;
        let graph = IsingGraph::new(vec![0.0, 1.0], vec![1.0], vec![(0, 1)]);
        let params = SampleParams {
            num_reads: 1,
            num_sweeps: 3,
            ..SampleParams::default()
        };
        // The correct dispatch count is computed from the live constant,
        // not hardcoded, so this test stays correct at every BLOCK_SWEEPS
        // from 1 through 8. At the current value, 1, div_ceil(3, 1) is 3,
        // so this still exercises validate_output's multi-dispatch success
        // path, same as it did against the original num_sweeps: 3,
        // BLOCK_SWEEPS: 2 pairing this replaces.
        let correct = params.num_sweeps.div_ceil(crate::native::BLOCK_SWEEPS) as u64;
        for spins in [
            vec![],
            vec![vec![1]],
            vec![vec![1, 0]],
            vec![vec![1, -1], vec![-1, 1]],
        ] {
            let output = RunOutput {
                spins,
                stats: RunStats {
                    programs: 1,
                    dispatches: correct,
                    ..RunStats::default()
                },
            };
            assert_fault(validate_output(&output, &graph, &params, 1));
        }
        for (programs, dispatches) in [
            (0, 0),
            (1, correct.saturating_sub(1)),
            (1, correct + 1),
            (2, correct),
        ] {
            let output = RunOutput {
                spins: vec![vec![1, -1]],
                stats: RunStats {
                    programs,
                    dispatches,
                    ..RunStats::default()
                },
            };
            assert_fault(validate_output(&output, &graph, &params, 1));
        }
        let output = RunOutput {
            spins: vec![vec![1, -1]],
            stats: RunStats {
                programs: 1,
                dispatches: correct,
                ..RunStats::default()
            },
        };
        assert!(validate_output(&output, &graph, &params, 1).is_ok());
    }

    #[test]
    fn oversize_live_response_reaps_worker_before_cancelling() {
        let (_fixture, path) =
            script("/bin/dd if=/dev/zero bs=1048576 count=17 2>/dev/null; exec /bin/sleep 30");
        let worker = WorkerProcess::spawn(&path, &WorkerRequest::Check).unwrap();
        let pid = worker.pid;
        let directory = worker.directory.path().to_path_buf();
        assert_fault(worker.wait(&|| false));
        assert_gone(pid, &directory);
    }

    #[test]
    fn parent_scores_valid_states_with_original_graph() {
        // num_sweeps is 3, and the expected dispatch count is computed from
        // the live BLOCK_SWEEPS rather than hardcoded, so this test stays
        // correct at every value from 1 through 8. At BLOCK_SWEEPS 1 and 2
        // it still exercises the multi-dispatch success path this test
        // covered before Task 5 (div_ceil(3, 1) is 3, div_ceil(3, 2) is 2).
        let num_sweeps: usize = 3;
        let dispatches = num_sweeps.div_ceil(crate::native::BLOCK_SWEEPS);
        let template = "printf '{\"pid\":%s,\"result\":{\"status\":\"solved\",\"output\":{\"spins\":[[1,-1]],\"stats\":{\"programs\":1,\"dispatches\":DISPATCHES,\"setup_us\":0,\"staging_us\":0,\"dispatch_us\":0,\"anneal_us\":0}}}}' \"$$\"";
        let body = template.replace("DISPATCHES", &dispatches.to_string());
        let (_fixture, path) = script(&body);
        let sampler = AneSampler {
            executable: path,
            access: Mutex::new(()),
        };
        let graph = IsingGraph::new(vec![1.0, -1.0], vec![1.0], vec![(0, 1)]);
        let params = SampleParams {
            num_sweeps,
            ..SampleParams::default()
        };
        assert_eq!(
            sampler.sample(&graph, &params).unwrap(),
            vec![SamplerResult {
                spins: vec![1, -1],
                energy_milli: 1000
            }]
        );
    }

    #[test]
    fn parent_scores_valid_states_with_multiple_dispatches() {
        // The test above, at num_sweeps: 3, only exercises multiple
        // dispatches while BLOCK_SWEEPS is 1 or 2 (div_ceil(3, n) is 1 for
        // every n from 3 through 8). num_sweeps here is 9, more than the
        // largest BLOCK_SWEEPS this crate accepts (8), so this test always
        // exercises more than one dispatch, at every value from 1 through
        // 8, not just 1 and 2. The expected dispatch count is computed from
        // the live constant, not hardcoded, and the assertion below fails
        // loudly rather than silently degrading to single-dispatch coverage
        // if compile_raw's accepted range ever grows past 8.
        let num_sweeps: usize = 9;
        let dispatches = num_sweeps.div_ceil(crate::native::BLOCK_SWEEPS);
        assert!(dispatches > 1, "test no longer exercises multiple dispatches");
        let template = "printf '{\"pid\":%s,\"result\":{\"status\":\"solved\",\"output\":{\"spins\":[[1,-1]],\"stats\":{\"programs\":1,\"dispatches\":DISPATCHES,\"setup_us\":0,\"staging_us\":0,\"dispatch_us\":0,\"anneal_us\":0}}}}' \"$$\"";
        let body = template.replace("DISPATCHES", &dispatches.to_string());
        let (_fixture, path) = script(&body);
        let sampler = AneSampler {
            executable: path,
            access: Mutex::new(()),
        };
        let graph = IsingGraph::new(vec![1.0, -1.0], vec![1.0], vec![(0, 1)]);
        let params = SampleParams {
            num_sweeps,
            ..SampleParams::default()
        };
        assert_eq!(
            sampler.sample(&graph, &params).unwrap(),
            vec![SamplerResult {
                spins: vec![1, -1],
                energy_milli: 1000
            }]
        );
    }

    #[test]
    fn sample_job_rejects_invalid_child_spins() {
        let (_fixture, path) = script("printf '{\"pid\":%s,\"result\":{\"status\":\"solved\",\"output\":{\"spins\":[[0]],\"stats\":{\"programs\":1,\"dispatches\":2,\"setup_us\":0,\"staging_us\":0,\"dispatch_us\":0,\"anneal_us\":0}}}}' \"$$\"");
        let sampler = AneSampler {
            executable: path,
            access: Mutex::new(()),
        };
        assert_fault(sampler.sample(
            &IsingGraph::new(vec![1.0], vec![], vec![]),
            &SampleParams {
                num_sweeps: 3,
                ..SampleParams::default()
            },
        ));
    }

    #[test]
    fn invalid_parent_input_does_not_start_worker() {
        let (fixture, path) = script("touch \"$0.started\"; exit 70");
        let sampler = AneSampler {
            executable: path,
            access: Mutex::new(()),
        };
        let graph = IsingGraph::new(vec![0.5], vec![], vec![]);
        assert_eq!(
            sampler.sample(&graph, &SampleParams::default()),
            Err(SampleError::Capacity)
        );
        let graph = IsingGraph::new(vec![1.0], vec![], vec![]);
        assert_eq!(
            sampler.sample(
                &graph,
                &SampleParams {
                    num_reads: 129,
                    ..SampleParams::default()
                }
            ),
            Err(SampleError::Capacity)
        );
        assert!(!fixture.path().join("worker.started").exists());
    }

    #[test]
    fn output_channel_close_stops_running_child() {
        let (fixture, path) =
            script("printf '%s\\n%s\\n' \"$$\" \"$TMPDIR\" > \"$0.started\"; exec /bin/sleep 30");
        assert_output_close_stops_worker(
            path,
            &fixture.path().join("worker.started"),
            IsingGraph::new(vec![1.0], vec![], vec![]),
            SampleParams::default(),
            false,
        );
    }

    fn assert_output_close_stops_worker(
        path: PathBuf,
        marker: &Path,
        graph: IsingGraph,
        params: SampleParams,
        real_worker: bool,
    ) {
        let sampler = AneSampler {
            executable: path,
            access: Mutex::new(()),
        };
        let (jobs_tx, jobs_rx) = mpsc::channel(1);
        let (out_tx, out_rx) = mpsc::channel(1);
        jobs_tx
            .blocking_send(StreamJob {
                job_id: vec![1],
                graph,
                params,
                watermark: None,
            })
            .unwrap();
        drop(jobs_tx);
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            sampler.sample_stream(jobs_rx, out_tx, CancelToken::default());
            done_tx.send(()).unwrap();
        });
        let start = Instant::now();
        let receipt = loop {
            if let Ok(receipt) = std::fs::read_to_string(marker) {
                if receipt.lines().count() == 2 {
                    break receipt;
                }
            }
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "worker did not start"
            );
            std::thread::sleep(Duration::from_millis(10));
        };
        let mut lines = receipt.lines();
        let pid: u32 = lines.next().unwrap().parse().unwrap();
        let directory = PathBuf::from(lines.next().unwrap());
        if real_worker {
            // The wrapper records the child identity before exec. Verify that
            // the real binary is running before closing its output channel.
            let expected = worker_binary().canonicalize().unwrap();
            loop {
                let output = Command::new("/bin/ps")
                    .args(["-p", &pid.to_string(), "-o", "comm="])
                    .output()
                    .unwrap();
                let observed = String::from_utf8(output.stdout).unwrap();
                if let Ok(path) = Path::new(observed.trim()).canonicalize() {
                    if path == expected {
                        break;
                    }
                }
                assert!(
                    start.elapsed() < Duration::from_secs(5),
                    "sampling worker did not exec: observed {observed:?}"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            std::thread::sleep(Duration::from_millis(50));
            assert_eq!(
                done_rx.try_recv(),
                Err(std::sync::mpsc::TryRecvError::Empty)
            );
        }
        let stop_started = Instant::now();
        drop(out_rx);
        done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        handle.join().unwrap();
        assert_gone(pid, &directory);
        if real_worker {
            eprintln!("stream-close receipt: pid={pid}, shutdown_us={}, directory_removed=true, child_reaped=true", stop_started.elapsed().as_micros());
        }
    }

    #[test]
    fn cancellation_after_scoring_discards_samples() {
        let (_fixture, path) = script("printf '{\"pid\":%s,\"result\":{\"status\":\"solved\",\"output\":{\"spins\":[[1]],\"stats\":{\"programs\":0,\"dispatches\":0,\"setup_us\":0,\"staging_us\":0,\"dispatch_us\":0,\"anneal_us\":0}}}}' \"$$\"");
        let sampler = AneSampler {
            executable: path,
            access: Mutex::new(()),
        };
        // While the child runs, the sampler owns the lock. The final stop check
        // follows scoring and releases that lock, making the race deterministic.
        let stop = || sampler.access.try_lock().is_ok();
        let outcome = sampler.sample_job(
            &IsingGraph::new(vec![1.0], vec![], vec![]),
            &SampleParams {
                num_sweeps: 0,
                ..SampleParams::default()
            },
            &stop,
        );
        match outcome {
            StreamOutcome::Cancelled => {}
            StreamOutcome::Completed(_) => panic!("returned abandoned samples"),
        }
    }

    fn worker_binary() -> PathBuf {
        std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("quip-ane-msa")
    }

    #[test]
    #[ignore = "requires integrated Apple Silicon ANE worker binary"]
    fn hardware_startup_dispatch_receipt() {
        let worker = WorkerProcess::spawn(&worker_binary(), &WorkerRequest::Check).unwrap();
        let pid = worker.pid;
        let directory = worker.directory.path().to_path_buf();
        let start = Instant::now();
        let reply = worker
            .wait(&|| start.elapsed() >= Duration::from_secs(30))
            .unwrap()
            .unwrap();
        let WorkerResult::Checked { dispatches } = reply.result else {
            panic!("native startup check failed");
        };
        assert_eq!(dispatches, 1);
        assert_gone(pid, &directory);
        eprintln!("startup receipt: pid={pid}, dispatches={dispatches}, directory_removed=true");
    }

    /// Task 8 measurement harness: spawns one real `--ane-worker` job on
    /// production's own topology and sweep count, and prints every
    /// worker-path counter to stderr. `wait` already logs its six counters
    /// through `tracing::debug!`; this test installs a `fmt` subscriber so
    /// that event reaches stderr under `--nocapture`. The reply also carries
    /// `RunStats`, Task 7's five
    /// Rust counters plus `setup_us`, `dispatches`, `staging_us`,
    /// `dispatch_us`, and `anneal_us`, printed here too so the report can
    /// separate the child's own compute (already measured by Tasks 1 and 7)
    /// from this task's wrapper counters.
    ///
    /// Same topology as Task 7's harness, `tests/fixtures/advantage2-system1.edges`,
    /// couplings in {-1, 1} from seed 7, zero fields, 128 reads, solved with
    /// seed 123. Sweeps is 2, not 512: the brief asks for the real topology
    /// at 2 sweeps, the value `BLOCK_SWEEPS` held when this test was
    /// written, one dispatch at that value. `BLOCK_SWEEPS` has since
    /// changed (Task 5); this fixture's own `num_sweeps: 2` is a literal,
    /// independent of the constant, so the test still runs, but "one
    /// dispatch" is no longer accurate at the constant's current value. Run
    /// five times, one process at a time with a 3-second sleep between
    /// runs, to collect medians; not itself a benchmark.
    #[test]
    #[ignore = "requires integrated Apple Silicon ANE worker binary"]
    fn hardware_worker_path_stage_medians_advantage2_system1() {
        fn xorshift64(s: &mut u64) -> u64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        }
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_writer(std::io::stderr)
            .try_init();
        let mut s: u64 = 7 | 1;
        let mut edges = Vec::with_capacity(41_515);
        let mut j = Vec::with_capacity(41_515);
        let fixture = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/advantage2-system1.edges"
        ));
        for line in fixture.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut nodes = line.split_whitespace();
            let u = nodes.next().expect("edge start").parse().expect("node id");
            let v = nodes.next().expect("edge end").parse().expect("node id");
            assert!(nodes.next().is_none(), "two node ids per edge");
            edges.push((u, v));
            j.push(if xorshift64(&mut s) & 1 == 0 {
                1.0
            } else {
                -1.0
            });
        }
        assert_eq!(edges.len(), 41_515);
        let graph = IsingGraph::new(vec![0.0; 4_577], j, edges);
        let params = SampleParams {
            num_reads: 128,
            num_sweeps: 2,
            sweeps_per_beta: 1,
            beta_range: None,
            seed: 123,
        };
        let worker = WorkerProcess::spawn(
            &worker_binary(),
            &WorkerRequest::Sample(RawJob::from_parts(&graph, &params)),
        )
        .unwrap();
        let pid = worker.pid;
        let directory = worker.directory.path().to_path_buf();
        let reply = worker.wait(&|| false).unwrap().unwrap();
        let WorkerResult::Solved { output } = reply.result else {
            panic!("job did not solve");
        };
        assert_gone(pid, &directory);
        eprintln!(
            "advantage2-system1 worker job stats: setup_us={} dispatches={} staging_us={} dispatch_us={} anneal_us={}",
            output.stats.setup_us,
            output.stats.dispatches,
            output.stats.staging_us,
            output.stats.dispatch_us,
            output.stats.anneal_us,
        );
    }

    #[test]
    #[ignore = "requires integrated Apple Silicon ANE worker binary"]
    fn hardware_cancellation_reaps_large_job() {
        let graph = IsingGraph::new(vec![0.0; 16_384], vec![], vec![]);
        let params = SampleParams {
            num_reads: 128,
            num_sweeps: 65_536,
            ..SampleParams::default()
        };
        let worker = WorkerProcess::spawn(
            &worker_binary(),
            &WorkerRequest::Sample(RawJob::from_parts(&graph, &params)),
        )
        .unwrap();
        let pid = worker.pid;
        let directory = worker.directory.path().to_path_buf();
        let start = Instant::now();
        assert!(worker
            .wait(&|| start.elapsed() >= Duration::from_millis(50))
            .unwrap()
            .is_none());
        assert_gone(pid, &directory);
        eprintln!("cancellation receipt: pid={pid}, elapsed_us={}, directory_removed=true, child_reaped=true", start.elapsed().as_micros());
    }

    #[test]
    #[ignore = "requires integrated Apple Silicon ANE worker binary"]
    fn hardware_output_channel_close_stops_sampling_worker() {
        let (fixture, path) = script(
            "printf '%s\\n%s\\n' \"$$\" \"$TMPDIR\" > \"$0.started\"; exec \"$0.binary\" \"$@\"",
        );
        std::os::unix::fs::symlink(worker_binary(), fixture.path().join("worker.binary")).unwrap();
        assert_output_close_stops_worker(
            path,
            &fixture.path().join("worker.started"),
            IsingGraph::new(vec![0.0; 16_384], vec![], vec![]),
            SampleParams {
                num_reads: 128,
                num_sweeps: 65_536,
                ..SampleParams::default()
            },
            true,
        );
    }

    fn four_color_oracle(graph: &IsingGraph, params: &SampleParams) -> Vec<Vec<i8>> {
        use crate::graph::LANES;
        use crate::msa::{initial_spins, schedule, ThresholdRows};
        let prepared = prepare(graph).unwrap();
        assert_eq!(prepared.color_count, 4);
        let mut state = initial_spins(prepared.node_count, params.seed);
        let mut rows = ThresholdRows::new(params.seed);
        for (rung_index, rung) in schedule(graph, params).unwrap().iter().enumerate() {
            rows.begin_rung(rung.beta);
            for sweep in 0..rung.sweeps {
                let thresholds = rows.expand(prepared.node_count, rung_index, sweep);
                for color in 0..prepared.color_count {
                    let before = state.clone();
                    for tile in prepared.tiles.iter().filter(|tile| tile.color == color) {
                        for &node in &tile.nodes {
                            for read in 0..LANES {
                                let spin = before[node * LANES + read];
                                let field = prepared.fields[node];
                                let mut degree = usize::from(field != 0);
                                let mut satisfied =
                                    usize::from(i16::from(field) * i16::from(spin) < 0);
                                for &(neighbor, coupling) in &prepared.neighbors[node] {
                                    degree += 1;
                                    satisfied += usize::from(
                                        i16::from(coupling)
                                            * i16::from(spin)
                                            * i16::from(before[neighbor * LANES + read])
                                            < 0,
                                    );
                                }
                                let threshold = usize::from(thresholds[node * LANES + read]);
                                state[node * LANES + read] =
                                    if satisfied <= (degree + threshold) / 2 {
                                        -spin
                                    } else {
                                        spin
                                    };
                            }
                        }
                    }
                }
            }
        }
        (0..params.num_reads)
            .map(|read| {
                (0..prepared.node_count)
                    .map(|node| state[node * LANES + read])
                    .collect()
            })
            .collect()
    }

    #[test]
    #[ignore = "requires integrated Apple Silicon ANE worker binary"]
    fn hardware_64_jobs_use_distinct_children_and_match_oracle() {
        let executable = worker_binary();
        let mut pids = std::collections::HashSet::new();
        let mut programs = 0;
        for job in 0..64 {
            let sign = if job % 2 == 0 { 1.0 } else { -1.0 };
            let graph = IsingGraph::new(
                vec![1.0, 0.0, -1.0, 0.0],
                vec![sign; 6],
                vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            );
            let params = SampleParams {
                num_reads: 128,
                num_sweeps: 16,
                seed: job,
                ..SampleParams::default()
            };
            let expected = four_color_oracle(&graph, &params);
            let worker = WorkerProcess::spawn(
                &executable,
                &WorkerRequest::Sample(RawJob::from_parts(&graph, &params)),
            )
            .unwrap();
            let pid = worker.pid;
            let directory = worker.directory.path().to_path_buf();
            let reply = worker.wait(&|| false).unwrap().unwrap();
            assert!(pids.insert(reply.pid));
            let WorkerResult::Solved { output } = reply.result else {
                panic!("job {job} did not solve");
            };
            validate_output(&output, &graph, &params, 1).unwrap();
            assert_eq!(output.spins, expected, "job {job}");
            programs += output.stats.programs;
            assert_gone(pid, &directory);
            eprintln!("job receipt: job={job}, pid={pid}, programs={}, dispatches={}, oracle_reads={}, directory_removed=true", output.stats.programs, output.stats.dispatches, output.spins.len());
        }
        assert_eq!(pids.len(), 64);
        assert_eq!(programs, 64);
        eprintln!(
            "64-job receipt: distinct_pids={}, programs={programs}, oracle_reads=8192",
            pids.len()
        );
    }

    #[test]
    #[ignore = "requires integrated native parent_pid and worker binary"]
    fn watchdog_exits_when_test_parent_exits() {
        use std::io::Write;
        let directory = tempfile::tempdir().unwrap();
        let fifo = directory.path().join("request.fifo");
        let reply = directory.path().join("response.json");
        let pid_file = directory.path().join("worker.pid");
        assert!(Command::new("/usr/bin/mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success());
        // Keeping both ends open prevents EOF from ending the worker request;
        // only its watchdog can release the blocked worker after parent exit.
        let _fifo = File::options().read(true).write(true).open(&fifo).unwrap();
        let mut parent = Command::new("/bin/sh").arg("-c")
            .arg("\"$1\" --ane-worker \"$$\" < \"$2\" > \"$3\" & worker=$!; printf '%s' \"$worker\" > \"$4\"; read finish; exit 0")
            .arg("test-parent").arg(worker_binary()).arg(&fifo).arg(&reply).arg(&pid_file)
            .stdin(Stdio::piped()).spawn().unwrap();
        let start = Instant::now();
        let pid = loop {
            if let Ok(text) = std::fs::read_to_string(&pid_file) {
                if let Ok(pid) = text.parse::<u32>() {
                    break pid;
                }
            }
            assert!(start.elapsed() < Duration::from_secs(5));
            std::thread::sleep(Duration::from_millis(10));
        };
        std::thread::sleep(Duration::from_millis(250));
        assert!(
            Command::new("/bin/kill")
                .args(["-0", &pid.to_string()])
                .status()
                .unwrap()
                .success(),
            "worker exited before its parent"
        );
        parent.stdin.take().unwrap().write_all(b"exit\n").unwrap();
        assert!(parent.wait().unwrap().success());
        let start = Instant::now();
        while Command::new("/bin/kill")
            .args(["-0", &pid.to_string()])
            .stderr(Stdio::null())
            .status()
            .unwrap()
            .success()
        {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "watchdog left worker {pid} alive"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(std::fs::metadata(reply).unwrap().len(), 0);
    }

    fn assert_fault<T>(result: Result<T, SampleError>) {
        match result {
            Err(SampleError::DeviceFault(_)) => {}
            _ => panic!("expected DeviceFault"),
        }
    }
}
