//! A test coordinator for the ANE solver's unit-coefficient domain.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::process::Stdio;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use quip_solver_conformance::driver::{DriverReport, ObservedReject, ObservedResult, Terminal};
use quip_solver_core::quip_proto::v1::{
    coord_msg, miner_msg,
    miner_service_server::{MinerService, MinerServiceServer},
    CoordMsg, IsingProblem, Job, JobKind, MinerMsg,
};
use quip_solver_core::quip_protocol::{
    scoring::energy_milli,
    wire::{decode_i32_le, decode_spins},
};
use tokio::net::UnixListener;
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::{timeout, timeout_at, Instant};
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tonic::{Request, Response, Status, Streaming};

const PHASE_TIMEOUT: Duration = Duration::from_secs(10);
type Outbound = mpsc::Sender<Result<CoordMsg, Status>>;
type Connection = (Streaming<MinerMsg>, Outbound);

struct Coordinator(Mutex<Option<oneshot::Sender<Connection>>>);

#[tonic::async_trait]
impl MinerService for Coordinator {
    type SessionStream = ReceiverStream<Result<CoordMsg, Status>>;

    async fn session(
        &self,
        request: Request<Streaming<MinerMsg>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let (send, receive) = mpsc::channel(16);
        let connection = self
            .0
            .lock()
            .expect("coordinator lock")
            .take()
            .ok_or_else(|| Status::already_exists("only one miner connection is expected"))?;
        connection
            .send((request.into_inner(), send))
            .map_err(|_| Status::cancelled("test stopped"))?;
        Ok(Response::new(ReceiverStream::new(receive)))
    }
}

struct ServerTask(JoinHandle<()>);

struct DispatchedProblem {
    ising: IsingProblem,
    dense_edges: Vec<(usize, usize)>,
    dispatch_s: f64,
}

#[allow(
    dead_code,
    reason = "study receipts are consumed only by the combined_study integration target"
)]
pub(super) struct StudyResult {
    pub job_id: Vec<u8>,
    pub energies: Vec<i64>,
    pub dispatch_s: f64,
    pub completed_s: f64,
    pub device_us: u64,
    pub rescore_ok: bool,
}

impl Drop for ServerTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Records the complete wire trace as well as the shared conformance report.
pub(super) struct Session {
    pub report: DriverReport,
    pub sent: Vec<CoordMsg>,
    pub received: Vec<MinerMsg>,
    pub study_results: Vec<StudyResult>,
    inbound: Streaming<MinerMsg>,
    outbound: Outbound,
    child: Child,
    jobs: HashMap<Vec<u8>, DispatchedProblem>,
    started: Instant,
    phase_timeout: Duration,
    _server: ServerTask,
    _directory: tempfile::TempDir,
}

impl Session {
    pub(super) async fn start(binary: &str) -> Self {
        Self::start_with(binary, "ane-protocol-test", &[], PHASE_TIMEOUT, None).await
    }

    pub(super) async fn start_with(
        binary: &str,
        miner_id: &str,
        env: &[(String, String)],
        phase_timeout: Duration,
        stderr: Option<&Path>,
    ) -> Self {
        let directory = tempfile::tempdir().expect("socket directory");
        let path = directory.path().join("coord.sock");
        let listener = UnixListener::bind(&path).expect("bind test coordinator");
        let (send, receive) = oneshot::channel();
        let server = ServerTask(tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(MinerServiceServer::new(Coordinator(Mutex::new(Some(send)))))
                .serve_with_incoming(UnixListenerStream::new(listener))
                .await
                .expect("serve test coordinator");
        }));
        let mut command = Command::new(binary);
        command
            .args(["--quip-coordinator", &format!("unix://{}", path.display())])
            .args(["--miner-id", miner_id])
            .env("QUIP_SESSION_TOKEN", "test-token")
            .envs(env.iter().cloned())
            .kill_on_drop(true);
        if let Some(path) = stderr {
            command.stderr(Stdio::from(
                File::create(path).expect("create miner stderr log"),
            ));
        }
        let child = command.spawn().expect("start real ANE miner");
        let (inbound, outbound) = timeout(phase_timeout, receive)
            .await
            .expect("miner must connect within the phase timeout")
            .expect("coordinator connection");
        let started = Instant::now();
        Self {
            report: DriverReport {
                handshake_ok: false,
                hello: None,
                ready_received: false,
                job_request_credits: Vec::new(),
                jobs_dispatched: 0,
                results: Vec::new(),
                rejects: Vec::new(),
                statuses: Vec::new(),
                cancel_acked: false,
                ping_acked: false,
                capabilities_received: None,
                cancelled_watermark: 0,
                fatal: None,
                terminal: Terminal::Open,
                timed_out_phases: Vec::new(),
                exit_code: -1,
            },
            sent: Vec::new(),
            received: Vec::new(),
            study_results: Vec::new(),
            inbound,
            outbound,
            child,
            jobs: HashMap::new(),
            started,
            phase_timeout,
            _server: server,
            _directory: directory,
        }
    }

    pub(super) async fn send(&mut self, message: coord_msg::Msg) {
        let frame = CoordMsg { msg: Some(message) };
        timeout(self.phase_timeout, self.outbound.send(Ok(frame.clone())))
            .await
            .expect("coordinator send timeout")
            .expect("coordinator send");
        self.sent.push(frame);
    }

    pub(super) async fn job(
        &mut self,
        id: &[u8],
        generation: u64,
        ising: IsingProblem,
        dense_edges: &[(usize, usize)],
        kind: JobKind,
        expired: bool,
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let deadline_ms = if expired {
            now - 60_000
        } else {
            now + 3_600_000
        };
        assert!(self
            .jobs
            .insert(
                id.to_vec(),
                DispatchedProblem {
                    ising: ising.clone(),
                    dense_edges: dense_edges.to_vec(),
                    dispatch_s: self.elapsed_s(),
                }
            )
            .is_none());
        self.send(coord_msg::Msg::Job(Job {
            job_id: id.to_vec(),
            kind: kind as i32,
            generation,
            deadline_ms,
            ising: Some(ising),
            provenance: None,
        }))
        .await;
        self.report.jobs_dispatched += 1;
    }

    pub(super) async fn until(&mut self, phase: &str, done: impl Fn(&DriverReport) -> bool) {
        let deadline = Instant::now() + self.phase_timeout;
        while !done(&self.report) {
            match timeout_at(deadline, self.inbound.message()).await {
                Ok(Ok(Some(frame))) => {
                    self.received.push(frame.clone());
                    self.observe(frame);
                }
                Ok(Ok(None)) => self.report.terminal = Terminal::Closed,
                Ok(Err(error)) => self.report.terminal = Terminal::Transport(error.to_string()),
                Err(_) => self.report.timed_out_phases.push(phase.to_owned()),
            }
            if done(&self.report) {
                break;
            }
            assert!(
                self.report.terminal == Terminal::Open && self.report.timed_out_phases.is_empty(),
                "phase {phase} failed: {:#?}; frames={:#?}",
                self.report,
                self.received
            );
        }
    }

    pub(super) async fn refunded(&mut self, phase: &str) {
        let count = self.report.jobs_dispatched as u64;
        self.until(phase, |report| report.credits_refunded() >= count)
            .await;
        assert_eq!(self.report.credits_refunded(), count, "phase {phase}");
    }

    fn observe(&mut self, frame: MinerMsg) {
        let Some(message) = frame.msg else {
            self.report.terminal = Terminal::EmptyMessage;
            return;
        };
        match message {
            miner_msg::Msg::Hello(hello) => {
                assert_eq!(self.received.len(), 1, "Hello must be the first frame");
                self.report.handshake_ok =
                    hello.protocol_version == 1 && hello.session_token == "test-token";
                self.report.hello = Some(hello);
            }
            miner_msg::Msg::Ready(_) => self.report.ready_received = true,
            miner_msg::Msg::JobRequest(request) => {
                self.report.job_request_credits.push(request.credits);
            }
            miner_msg::Msg::Capabilities(capabilities) => {
                self.report.capabilities_received = Some(capabilities);
            }
            miner_msg::Msg::Status(status) => self.report.statuses.push(status),
            miner_msg::Msg::Reject(reject) => self.report.rejects.push(ObservedReject {
                job_id: reject.job_id,
                reason: reject.reason,
            }),
            miner_msg::Msg::Fatal(fatal) => {
                self.report.fatal = Some((fatal.exit_code as i32, fatal.reason));
            }
            miner_msg::Msg::Result(result) => {
                let DispatchedProblem {
                    ising,
                    dense_edges,
                    dispatch_s,
                } = self.jobs.get(&result.job_id).expect("known result job ID");
                let decode = |bytes: &[u8]| {
                    decode_i32_le(bytes)
                        .expect("valid coefficients for a returned result")
                        .into_iter()
                        .map(|value| f64::from(value) / 1000.0)
                        .collect::<Vec<_>>()
                };
                let h = decode(&ising.h_milli_le32);
                let j = decode(&ising.j_milli_le32);
                assert_eq!(result.solutions.len(), ising.num_reads as usize);
                assert_eq!(
                    result.meta.as_ref().expect("SamplerMeta").reads,
                    ising.num_reads
                );
                let scored: Vec<Option<i64>> = result
                    .solutions
                    .iter()
                    .map(|solution| {
                        let spins = decode_spins(&solution.spins_bytes).expect("valid spin bytes");
                        assert_eq!(spins.len(), h.len(), "one spin per variable");
                        Some(energy_milli(&spins, &h, &j, dense_edges))
                    })
                    .collect();
                let energies: Vec<_> = result
                    .solutions
                    .iter()
                    .map(|solution| solution.energy_milli)
                    .collect();
                let rescore_ok = scored
                    .iter()
                    .zip(&energies)
                    .all(|(scored, energy)| *scored == Some(*energy));
                let device_us = result
                    .meta
                    .as_ref()
                    .expect("SamplerMeta")
                    .device_access_time_us;
                self.study_results.push(StudyResult {
                    job_id: result.job_id.clone(),
                    energies: energies.clone(),
                    dispatch_s: *dispatch_s,
                    completed_s: self.elapsed_s(),
                    device_us,
                    rescore_ok,
                });
                self.report.results.push(ObservedResult {
                    job_id: result.job_id,
                    solution_energies_milli: result
                        .solutions
                        .iter()
                        .map(|s| s.energy_milli)
                        .collect(),
                    rescored_energies_milli: Some(scored),
                    meta_present: result.meta.is_some(),
                    meta_sweeps: result.meta.expect("SamplerMeta").sweeps,
                });
            }
        }
    }

    pub(super) async fn finish(&mut self) {
        self.finish_with_trace(true).await;
    }

    #[allow(dead_code, reason = "used only by the production-channel study")]
    pub(super) async fn finish_quiet(&mut self) {
        self.finish_with_trace(false).await;
    }

    async fn finish_with_trace(&mut self, trace: bool) {
        self.until("shutdown", |report| report.terminal == Terminal::Closed)
            .await;
        self.report.exit_code = timeout(self.phase_timeout, self.child.wait())
            .await
            .expect("miner exit within ten seconds")
            .expect("miner exit status")
            .code()
            .unwrap_or(-1);
        if trace || self.report.exit_code != 0 {
            eprintln!(
                "sent={:#?}\nreceived={:#?}\nreport={:#?}",
                self.sent, self.received, self.report
            );
        }
    }

    pub(super) fn elapsed_s(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }
}
