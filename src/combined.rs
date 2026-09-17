//! Route independent jobs to the selected engines under one coordinator identity.

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use quip_miner_ane::AneSampler;
use quip_solver_core::{
    CancelToken, IsingGraph, OpenError, SampleError, SampleParams, Sampler, SamplerResult,
    StreamJob, StreamOutcome, StreamResult,
};
use tokio::sync::mpsc;

use crate::{Kernel, MetalConfig, MetalSampler};

const POLL: Duration = Duration::from_millis(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Settings {
    valid: bool,
    enable_ane: bool,
    enable_metal: bool,
    utilization: u32,
    yielding: bool,
}

fn settings(lock: &Mutex<Settings>) -> Settings {
    *lock.lock().unwrap_or_else(|poison| poison.into_inner())
}

pub(crate) fn stream_width(kernel: Kernel) -> usize {
    crate::streaming::declared_stream_width(kernel) + usize::from(kernel == Kernel::Msa)
}

pub(crate) struct CombinedSampler {
    kernel: Kernel,
    device: usize,
    settings: Mutex<Settings>,
    metal: OnceLock<Result<MetalSampler, String>>,
    ane: Mutex<Option<Result<Arc<AneSampler>, String>>>,
}

impl CombinedSampler {
    pub(crate) fn new(kernel: Kernel, device: usize, utilization: u32, yielding: bool) -> Self {
        Self {
            kernel,
            device,
            settings: Mutex::new(Settings {
                valid: true,
                enable_ane: true,
                enable_metal: true,
                utilization,
                yielding,
            }),
            metal: OnceLock::new(),
            ane: Mutex::new(None),
        }
    }

    fn metal(&self) -> Result<&MetalSampler, SampleError> {
        let result = self.metal.get_or_init(|| {
            let device = crate::metal_device::MetalDevice::open(self.device)
                .map_err(|error| format!("Metal device {}: {error}", self.device))?;
            let cfg = settings(&self.settings);
            let gov = crate::iokit_gov::UtilGovernor::start(
                self.device as u32,
                cfg.utilization,
                cfg.yielding,
            );
            Ok(MetalSampler::new(device, gov, self.kernel))
        });
        let metal = result
            .as_ref()
            .map_err(|error| SampleError::DeviceFault(error.clone()))?;
        // Configuration can arrive while the device is opening.
        let cfg = settings(&self.settings);
        metal.gov.reconfigure(cfg.utilization, cfg.yielding);
        Ok(metal)
    }

    fn ane(&self, should_stop: &dyn Fn() -> bool) -> Result<Arc<AneSampler>, SampleError> {
        let mut cached = self.ane.lock().unwrap_or_else(|poison| poison.into_inner());
        if cached.is_none() {
            let opened = std::env::current_exe()
                .map_err(|error| format!("locate ANE worker executable: {error}"))
                .and_then(|executable| {
                    AneSampler::open_with_cancel(executable, should_stop)
                        .map(Arc::new)
                        .map_err(|error| error.0)
                });
            // A cancelled startup belongs to that job. It must not poison the
            // next live round's attempt to start the engine.
            if should_stop() && opened.is_err() {
                return Err(SampleError::DeviceBusy);
            }
            *cached = Some(opened);
        }
        match cached.as_ref() {
            Some(Ok(ane)) => Ok(Arc::clone(ane)),
            Some(Err(error)) => Err(SampleError::DeviceFault(error.clone())),
            None => Err(SampleError::DeviceFault(
                "missing ANE initialization result".into(),
            )),
        }
    }

    pub(crate) fn check(&self) -> Result<(), OpenError> {
        let cfg = settings(&self.settings);
        if cfg.enable_metal {
            self.metal().map_err(|error| OpenError(error.to_string()))?;
        }
        if cfg.enable_ane && self.kernel == Kernel::Msa {
            self.ane(&|| false)
                .map_err(|error| OpenError(error.to_string()))?;
        }
        Ok(())
    }
}

impl Sampler for CombinedSampler {
    fn sample(
        &self,
        graph: &IsingGraph,
        params: &SampleParams,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        let cfg = settings(&self.settings);
        let eligible = eligibility(self.kernel, graph, params, cfg);
        if eligible[0] {
            self.metal()?.sample(graph, params)
        } else if eligible[1] {
            self.ane(&|| false)?.sample(graph, params)
        } else {
            unavailable(self.kernel, cfg);
            Err(configuration_error(self.kernel, cfg).unwrap_or(SampleError::Capacity))
        }
    }

    fn sample_stream(
        &self,
        jobs: mpsc::Receiver<StreamJob>,
        out: mpsc::Sender<StreamResult>,
        cancel: CancelToken,
    ) {
        Router {
            kernel: self.kernel,
            settings: &self.settings,
            metal_width: crate::streaming::declared_stream_width(self.kernel),
        }
        .run(jobs, out, cancel, &MetalEngine(self), &AneEngine(self));
    }

    fn stream_width(&self) -> usize {
        stream_width(self.kernel)
    }
    fn utilization(&self) -> f64 {
        self.metal
            .get()
            .and_then(|result| result.as_ref().ok())
            .map_or(0.0, Sampler::utilization)
    }
    fn should_throttle(&self) -> bool {
        false
    }
    fn max_reads(&self) -> u32 {
        if settings(&self.settings).enable_metal {
            crate::streaming::max_reads(self.kernel)
        } else {
            128
        }
    }
    fn apply_config(&self, backend_toml: &str) {
        let cfg: MetalConfig = match toml::from_str(backend_toml) {
            Ok(cfg) => cfg,
            Err(error) => {
                tracing::warn!(%error, "invalid engine configuration; rejecting jobs until valid configuration arrives");
                self.settings
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .valid = false;
                return;
            }
        };
        quip_solver_core::config::warn_unknown_fields("metal", cfg.unknown.keys());
        {
            let mut current = self
                .settings
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            current.valid = true;
            if let Some(value) = cfg.enable_ane {
                current.enable_ane = value;
            }
            if let Some(value) = cfg.enable_metal {
                current.enable_metal = value;
            }
            if let Some(value) = cfg.utilization {
                current.utilization = value.clamp(1, 100);
            }
            if let Some(value) = cfg.yielding {
                current.yielding = value;
            }
        }
        let current = settings(&self.settings);
        if !current.enable_metal && (!current.enable_ane || self.kernel != Kernel::Msa) {
            unavailable(self.kernel, current);
        }
        if let Some(Ok(metal)) = self.metal.get() {
            metal.gov.reconfigure(current.utilization, current.yielding);
        }
    }
}

fn configuration_error(kernel: Kernel, cfg: Settings) -> Option<SampleError> {
    let reason = if !cfg.valid {
        "invalid backend TOML; send a valid engine configuration"
    } else if !cfg.enable_metal && !cfg.enable_ane {
        "invalid engine configuration: enable_metal and enable_ane are both false"
    } else if !cfg.enable_metal && kernel != Kernel::Msa {
        "invalid engine configuration: Metal is disabled and ANE supports MSA only; SA and Gibbs require Metal"
    } else {
        return None;
    };
    Some(SampleError::DeviceFault(reason.into()))
}

fn unavailable(kernel: Kernel, cfg: Settings) {
    if !cfg.valid {
        tracing::warn!(
            "invalid engine configuration; no job can run until valid configuration arrives"
        );
        return;
    }
    tracing::warn!(?kernel, enable_ane = cfg.enable_ane, enable_metal = cfg.enable_metal,
        "no enabled engine supports this job; ANE supports MSA only, with unit coefficients and at most 128 reads");
}

/// Linear input scan only. Dense ANE tile preparation belongs to its worker.
fn eligibility(
    kernel: Kernel,
    graph: &IsingGraph,
    params: &SampleParams,
    cfg: Settings,
) -> [bool; 2] {
    if configuration_error(kernel, cfg).is_some() {
        return [false, false];
    }
    let n = graph.h.len();
    let mut metal = cfg.enable_metal
        && n <= crate::sampler::kernel_max_nodes(kernel)
        && graph.edges.len() <= crate::DEFAULT_MAX_EDGES as usize
        && params.num_sweeps <= crate::sampler::MAX_SWEEPS;
    let mut ane = cfg.enable_ane
        && kernel == Kernel::Msa
        && n <= 16_384
        && graph.edges.len() <= 163_840
        && graph.j.len() == graph.edges.len()
        && (1..=128).contains(&params.num_reads)
        && params.num_sweeps <= 65_536
        && params.sweeps_per_beta > 0;
    if !metal && !ane {
        return [false, false];
    }
    if let Some((hot, cold)) = params.beta_range {
        ane &= hot.is_finite() && cold.is_finite() && hot > 0.0 && cold >= hot;
    }
    ane &= graph
        .h
        .iter()
        .all(|&value| value == -1.0 || value == 0.0 || value == 1.0);
    let mut degree = vec![0u32; n];
    let mut ane_degree = vec![0u32; n];
    let mut seen = HashSet::new();
    for (index, &(u, v)) in graph.edges.iter().enumerate() {
        if u >= n || v >= n {
            ane = false;
            continue;
        }
        if kernel == Kernel::Msa {
            degree[u] += 1;
            if u != v {
                degree[v] += 1;
            }
            metal &= degree[u] <= 20 && degree[v] <= 20;
        }
        if ane {
            let coupling = graph.j[index];
            if u == v
                || !seen.insert((u.min(v), u.max(v)))
                || !(coupling == -1.0 || coupling == 0.0 || coupling == 1.0)
            {
                ane = false;
            } else if coupling != 0.0 {
                ane_degree[u] += 1;
                ane_degree[v] += 1;
                ane &= ane_degree[u] <= 20 && ane_degree[v] <= 20;
            }
        }
    }
    [metal, ane]
}

trait StreamEngine: Sync {
    fn run(
        &self,
        jobs: mpsc::Receiver<StreamJob>,
        out: mpsc::Sender<StreamResult>,
        cancel: CancelToken,
        stop: &AtomicBool,
    );
}

struct MetalEngine<'a>(&'a CombinedSampler);
impl StreamEngine for MetalEngine<'_> {
    fn run(
        &self,
        mut jobs: mpsc::Receiver<StreamJob>,
        out: mpsc::Sender<StreamResult>,
        cancel: CancelToken,
        stop: &AtomicBool,
    ) {
        // Wait without consuming the seed: the existing batching loop owns it.
        while jobs.is_empty() {
            if jobs.is_closed() || stop.load(Ordering::Acquire) {
                return;
            }
            std::thread::sleep(POLL);
        }
        match self.0.metal() {
            Ok(metal) => metal.sample_stream(jobs, out, cancel),
            Err(error) => {
                let detail = error.to_string();
                while let Some(job) = jobs.blocking_recv() {
                    if out
                        .blocking_send(result(
                            job,
                            StreamOutcome::Completed(Err(SampleError::DeviceFault(detail.clone()))),
                            0,
                        ))
                        .is_err()
                    {
                        break;
                    }
                }
            }
        }
    }
}

struct AneEngine<'a>(&'a CombinedSampler);
impl StreamEngine for AneEngine<'_> {
    fn run(
        &self,
        mut jobs: mpsc::Receiver<StreamJob>,
        out: mpsc::Sender<StreamResult>,
        cancel: CancelToken,
        stop: &AtomicBool,
    ) {
        while let Some(job) = jobs.blocking_recv() {
            let started = Instant::now();
            let cancelled = || stop.load(Ordering::Acquire) || cancel.is_cancelled(job.watermark);
            let outcome = if cancelled() {
                StreamOutcome::Cancelled
            } else {
                match self.0.ane(&cancelled) {
                    Ok(ane) => ane.sample_job(&job.graph, &job.params, &cancelled),
                    Err(_) if cancelled() => StreamOutcome::Cancelled,
                    Err(error) => StreamOutcome::Completed(Err(error)),
                }
            };
            let micros = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
            if out.blocking_send(result(job, outcome, micros)).is_err() {
                break;
            }
        }
    }
}

fn result(job: StreamJob, outcome: StreamOutcome, device_access_time_us: u64) -> StreamResult {
    StreamResult {
        job_id: job.job_id,
        outcome,
        device_access_time_us,
    }
}

struct QueuedJob {
    job: StreamJob,
    eligibility: Option<(Settings, [bool; 2])>,
}

struct Router<'a> {
    kernel: Kernel,
    settings: &'a Mutex<Settings>,
    metal_width: usize,
}

impl Router<'_> {
    fn run(
        &self,
        mut jobs: mpsc::Receiver<StreamJob>,
        out: mpsc::Sender<StreamResult>,
        cancel: CancelToken,
        metal: &dyn StreamEngine,
        ane: &dyn StreamEngine,
    ) {
        let stop = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let (metal_tx, metal_jobs) = mpsc::channel(self.metal_width.max(1));
            let (ane_tx, ane_jobs) = mpsc::channel(1);
            let (metal_out, metal_results) = mpsc::channel(self.metal_width.max(1));
            let (ane_out, ane_results) = mpsc::channel(1);
            let metal_cancel = cancel.clone();
            let ane_cancel = cancel.clone();
            let stop_ref = &stop;
            let metal_thread = std::thread::Builder::new()
                .name("combined-metal".into())
                .spawn_scoped(scope, move || {
                    metal.run(metal_jobs, metal_out, metal_cancel, stop_ref)
                });
            let ane_thread = std::thread::Builder::new()
                .name("combined-ane".into())
                .spawn_scoped(scope, move || {
                    ane.run(ane_jobs, ane_out, ane_cancel, stop_ref)
                });
            let mut senders = [Some(metal_tx), Some(ane_tx)];
            let mut receivers = [metal_results, ane_results];
            let mut active: [Vec<Vec<u8>>; 2] = [Vec::new(), Vec::new()];
            let limits = [self.metal_width.max(1), 1];
            let mut dead = [false; 2];
            let mut pending = VecDeque::<QueuedJob>::new();
            // Queue plus engine reservations never exceeds advertised credits.
            let capacity = self.metal_width.max(1) + usize::from(self.kernel == Kernel::Msa);
            let mut eof = false;
            loop {
                if out.is_closed() {
                    stop.store(true, Ordering::Release);
                    break;
                }
                for engine in 0..2 {
                    loop {
                        match receivers[engine].try_recv() {
                            Ok(reply) => {
                                if let Some(index) =
                                    active[engine].iter().position(|id| *id == reply.job_id)
                                {
                                    active[engine].swap_remove(index);
                                    if matches!(&reply.outcome, StreamOutcome::Completed(Err(error)) if error.is_fatal())
                                    {
                                        dead[engine] = true;
                                        senders[engine] = None;
                                    }
                                    if out.blocking_send(reply).is_err() {
                                        stop.store(true, Ordering::Release);
                                        break;
                                    }
                                } else {
                                    tracing::error!(
                                        engine,
                                        "engine returned an unassigned or duplicate job"
                                    );
                                }
                            }
                            Err(mpsc::error::TryRecvError::Empty) => break,
                            Err(mpsc::error::TryRecvError::Disconnected) => {
                                dead[engine] = true;
                                for job_id in active[engine].drain(..) {
                                    let reply = StreamResult {
                                        job_id,
                                        outcome: StreamOutcome::Completed(Err(
                                            SampleError::DeviceFault(
                                                "engine stream stopped before completing job"
                                                    .into(),
                                            ),
                                        )),
                                        device_access_time_us: 0,
                                    };
                                    if out.blocking_send(reply).is_err() {
                                        stop.store(true, Ordering::Release);
                                        break;
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
                if stop.load(Ordering::Acquire) {
                    break;
                }
                let owned = pending.len() + active.iter().map(Vec::len).sum::<usize>();
                if owned < capacity && !eof {
                    match jobs.try_recv() {
                        Ok(job) => pending.push_back(QueuedJob {
                            job,
                            eligibility: None,
                        }),
                        Err(mpsc::error::TryRecvError::Empty) => {}
                        Err(mpsc::error::TryRecvError::Disconnected) => eof = true,
                    }
                }
                // Skip a waiting ANE-only job when later work can use Metal.
                // Each graph is scanned once per configuration snapshot.
                for _ in 0..pending.len() {
                    let Some(mut queued) = pending.pop_front() else {
                        break;
                    };
                    let job = queued.job;
                    if cancel.is_cancelled(job.watermark) {
                        if out
                            .blocking_send(result(job, StreamOutcome::Cancelled, 0))
                            .is_err()
                        {
                            stop.store(true, Ordering::Release);
                            break;
                        }
                        continue;
                    }
                    let cfg = settings(self.settings);
                    let eligible = match queued.eligibility {
                        Some((previous, eligible)) if previous == cfg => eligible,
                        _ => {
                            let eligible = eligibility(self.kernel, &job.graph, &job.params, cfg);
                            queued.eligibility = Some((cfg, eligible));
                            eligible
                        }
                    };
                    let selected = [1, 0].into_iter().find(|&engine| {
                        eligible[engine] && !dead[engine] && active[engine].len() < limits[engine]
                    });
                    if let Some(engine) = selected {
                        let job_id = job.job_id.clone();
                        if let Some(sender) = &senders[engine] {
                            match sender.try_send(job) {
                                Ok(()) => active[engine].push(job_id),
                                Err(mpsc::error::TrySendError::Full(job)) => {
                                    pending.push_back(QueuedJob {
                                        job,
                                        eligibility: queued.eligibility,
                                    });
                                }
                                Err(mpsc::error::TrySendError::Closed(job)) => {
                                    dead[engine] = true;
                                    pending.push_back(QueuedJob {
                                        job,
                                        eligibility: queued.eligibility,
                                    });
                                }
                            }
                        } else {
                            dead[engine] = true;
                            pending.push_back(QueuedJob {
                                job,
                                eligibility: queued.eligibility,
                            });
                        }
                    } else if !(0..2).any(|engine| eligible[engine] && !dead[engine]) {
                        unavailable(self.kernel, cfg);
                        let error = if (0..2).any(|engine| eligible[engine] && dead[engine]) {
                            SampleError::DeviceFault("selected engine stream is unavailable".into())
                        } else {
                            configuration_error(self.kernel, cfg).unwrap_or(SampleError::Capacity)
                        };
                        if out
                            .blocking_send(result(job, StreamOutcome::Completed(Err(error)), 0))
                            .is_err()
                        {
                            stop.store(true, Ordering::Release);
                            break;
                        }
                    } else {
                        pending.push_back(QueuedJob {
                            job,
                            eligibility: queued.eligibility,
                        });
                    }
                }
                if eof && pending.is_empty() {
                    senders = [None, None];
                    if active.iter().all(Vec::is_empty) {
                        break;
                    }
                }
                if !pending.is_empty()
                    || jobs.is_empty()
                    || active.iter().map(Vec::len).sum::<usize>() >= capacity
                {
                    std::thread::sleep(POLL);
                }
            }
            stop.store(true, Ordering::Release);
            drop(senders);
            drop(receivers);
            for thread in [metal_thread, ane_thread] {
                match thread {
                    Ok(thread) => {
                        if thread.join().is_err() {
                            tracing::error!("engine thread panicked");
                        }
                    }
                    Err(error) => tracing::error!(%error, "could not start engine thread"),
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc as events;

    fn config(metal: bool, ane: bool) -> Settings {
        Settings {
            valid: true,
            enable_metal: metal,
            enable_ane: ane,
            utilization: 73,
            yielding: true,
        }
    }

    fn job(id: u8) -> StreamJob {
        StreamJob {
            job_id: vec![id],
            graph: IsingGraph::new(vec![0.0; 2], vec![1.0], vec![(0, 1)]),
            params: SampleParams {
                num_reads: 128,
                num_sweeps: 2,
                sweeps_per_beta: 1,
                beta_range: None,
                seed: 7,
            },
            watermark: Some(id as u64 + 1),
        }
    }

    struct Fake {
        index: usize,
        gate: Arc<AtomicBool>,
        started: events::Sender<(usize, u8)>,
        fail: bool,
        exit: bool,
    }

    impl StreamEngine for Fake {
        fn run(
            &self,
            mut jobs: mpsc::Receiver<StreamJob>,
            out: mpsc::Sender<StreamResult>,
            cancel: CancelToken,
            stop: &AtomicBool,
        ) {
            let mut held = Vec::new();
            let mut eof = false;
            loop {
                if stop.load(Ordering::Acquire) || out.is_closed() {
                    return;
                }
                if !eof {
                    match jobs.try_recv() {
                        Ok(job) => {
                            self.started.send((self.index, job.job_id[0])).unwrap();
                            if self.exit {
                                return;
                            }
                            held.push(job);
                        }
                        Err(mpsc::error::TryRecvError::Empty) => {}
                        Err(mpsc::error::TryRecvError::Disconnected) => eof = true,
                    }
                }
                let mut index = 0;
                while index < held.len() {
                    if self.gate.load(Ordering::Acquire)
                        || cancel.is_cancelled(held[index].watermark)
                    {
                        let job = held.remove(index);
                        let outcome = if self.fail {
                            StreamOutcome::Completed(Err(SampleError::DeviceFault(
                                "fake engine failure".into(),
                            )))
                        } else if cancel.is_cancelled(job.watermark) {
                            StreamOutcome::Cancelled
                        } else {
                            StreamOutcome::Completed(Ok(vec![SamplerResult {
                                spins: vec![1, -1],
                                energy_milli: self.index as i64,
                            }]))
                        };
                        if out.blocking_send(result(job, outcome, 1)).is_err() {
                            return;
                        }
                    } else {
                        index += 1;
                    }
                }
                if eof && held.is_empty() {
                    return;
                }
                std::thread::sleep(POLL);
            }
        }
    }

    struct Harness {
        jobs: mpsc::Sender<StreamJob>,
        results: mpsc::Receiver<StreamResult>,
        events: events::Receiver<(usize, u8)>,
        gates: [Arc<AtomicBool>; 2],
        cancel: CancelToken,
        done: events::Receiver<()>,
        settings: Arc<Mutex<Settings>>,
    }

    fn harness(kernel: Kernel, cfg: Settings, ane_fail: bool, ane_exit: bool) -> Harness {
        let (jobs, incoming) = mpsc::channel(16);
        let (out, results) = mpsc::channel(16);
        let (started, events) = events::channel();
        let (done_tx, done) = events::channel();
        let gates = [
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
        ];
        let metal = Fake {
            index: 0,
            gate: Arc::clone(&gates[0]),
            started: started.clone(),
            fail: false,
            exit: false,
        };
        let ane = Fake {
            index: 1,
            gate: Arc::clone(&gates[1]),
            started,
            fail: ane_fail,
            exit: ane_exit,
        };
        let cancel = CancelToken::default();
        let child_cancel = cancel.clone();
        let settings = Arc::new(Mutex::new(cfg));
        let child_settings = Arc::clone(&settings);
        std::thread::spawn(move || {
            Router {
                kernel,
                settings: &child_settings,
                metal_width: 2,
            }
            .run(incoming, out, child_cancel, &metal, &ane);
            done_tx.send(()).unwrap();
        });
        Harness {
            jobs,
            results,
            events,
            gates,
            cancel,
            done,
            settings,
        }
    }

    fn receive(results: &mut mpsc::Receiver<StreamResult>) -> StreamResult {
        let start = Instant::now();
        loop {
            if let Ok(result) = results.try_recv() {
                return result;
            }
            assert!(start.elapsed() < Duration::from_secs(2), "result timed out");
            std::thread::sleep(POLL);
        }
    }

    #[test]
    fn both_engines_overlap_with_separate_admission_limits_and_no_duplicates() {
        let mut h = harness(Kernel::Msa, config(true, true), false, false);
        for id in 0..7 {
            h.jobs.blocking_send(job(id)).unwrap();
        }
        let first: Vec<_> = (0..3)
            .map(|_| h.events.recv_timeout(Duration::from_secs(2)).unwrap())
            .collect();
        assert_eq!(first.iter().filter(|(engine, _)| *engine == 1).count(), 1);
        assert_eq!(first.iter().filter(|(engine, _)| *engine == 0).count(), 2);
        assert_eq!(
            h.events
                .recv_timeout(Duration::from_millis(30))
                .unwrap_err(),
            events::RecvTimeoutError::Timeout
        );
        h.gates[0].store(true, Ordering::Release);
        let first_reply = receive(&mut h.results);
        assert!(
            matches!(first_reply.outcome, StreamOutcome::Completed(Ok(ref samples)) if samples[0].energy_milli == 0)
        );
        // Metal can accept more work while the first ANE job stays blocked.
        assert_eq!(h.events.recv_timeout(Duration::from_secs(2)).unwrap().0, 0);
        h.gates[1].store(true, Ordering::Release);
        drop(h.jobs);
        let mut ids = HashSet::from([first_reply.job_id]);
        for _ in 1..7 {
            assert!(ids.insert(receive(&mut h.results).job_id));
        }
        assert_eq!(ids.len(), 7);
        h.done.recv_timeout(Duration::from_secs(2)).unwrap();
    }

    #[test]
    fn queued_ane_only_job_does_not_hide_ready_metal_work() {
        let mut h = harness(Kernel::Msa, config(true, true), false, false);
        h.jobs.blocking_send(job(0)).unwrap();
        assert_eq!(
            h.events.recv_timeout(Duration::from_secs(2)).unwrap(),
            (1, 0)
        );
        let mut large = job(1);
        large.graph = IsingGraph::new(vec![0.0; crate::sampler::MSA_MAX_NODES + 1], vec![], vec![]);
        h.jobs.blocking_send(large).unwrap();
        h.jobs.blocking_send(job(2)).unwrap();
        // The ANE remains blocked. The later small job must reach Metal.
        assert_eq!(
            h.events.recv_timeout(Duration::from_secs(2)).unwrap(),
            (0, 2)
        );
        h.gates[0].store(true, Ordering::Release);
        assert_eq!(receive(&mut h.results).job_id, [2]);
        h.gates[1].store(true, Ordering::Release);
        drop(h.jobs);
        let mut ids = HashSet::new();
        for _ in 0..2 {
            ids.insert(receive(&mut h.results).job_id);
        }
        assert_eq!(ids, HashSet::from([vec![0], vec![1]]));
        h.done.recv_timeout(Duration::from_secs(2)).unwrap();
    }

    #[test]
    fn individual_engine_modes_never_assign_to_disabled_engine() {
        for (metal, ane, expected) in [(true, false, 0), (false, true, 1)] {
            let mut h = harness(Kernel::Msa, config(metal, ane), false, false);
            h.gates[expected].store(true, Ordering::Release);
            for id in 0..3 {
                h.jobs.blocking_send(job(id)).unwrap();
            }
            drop(h.jobs);
            for _ in 0..3 {
                assert_eq!(
                    h.events.recv_timeout(Duration::from_secs(2)).unwrap().0,
                    expected
                );
                assert!(matches!(
                    receive(&mut h.results).outcome,
                    StreamOutcome::Completed(Ok(_))
                ));
            }
            h.done.recv_timeout(Duration::from_secs(2)).unwrap();
        }
    }

    #[test]
    fn sa_and_gibbs_ignore_ane_and_reject_without_metal() {
        for kernel in [Kernel::Sa, Kernel::Gibbs] {
            for metal in [false, true] {
                let mut h = harness(kernel, config(metal, true), false, false);
                h.gates[0].store(true, Ordering::Release);
                h.jobs.blocking_send(job(0)).unwrap();
                drop(h.jobs);
                let reply = receive(&mut h.results);
                if metal {
                    assert!(matches!(reply.outcome, StreamOutcome::Completed(Ok(_))));
                } else {
                    assert!(matches!(
                        reply.outcome,
                        StreamOutcome::Completed(Err(SampleError::DeviceFault(_)))
                    ));
                }
                h.done.recv_timeout(Duration::from_secs(2)).unwrap();
                assert!(h.events.try_iter().all(|(engine, _)| engine == 0));
            }
        }
    }

    #[test]
    fn cancellation_releases_ane_reservation_for_next_live_round() {
        let mut h = harness(Kernel::Msa, config(false, true), false, false);
        h.jobs.blocking_send(job(0)).unwrap();
        assert_eq!(
            h.events.recv_timeout(Duration::from_secs(2)).unwrap(),
            (1, 0)
        );
        h.jobs.blocking_send(job(1)).unwrap();
        h.cancel.cancel_through(1);
        assert!(matches!(
            receive(&mut h.results).outcome,
            StreamOutcome::Cancelled
        ));
        assert_eq!(
            h.events.recv_timeout(Duration::from_secs(2)).unwrap(),
            (1, 1)
        );
        h.gates[1].store(true, Ordering::Release);
        drop(h.jobs);
        assert!(matches!(
            receive(&mut h.results).outcome,
            StreamOutcome::Completed(Ok(_))
        ));
        h.done.recv_timeout(Duration::from_secs(2)).unwrap();
    }

    #[test]
    fn closed_output_stops_blocked_engines_and_joins_threads() {
        let h = harness(Kernel::Msa, config(true, true), false, false);
        h.jobs.blocking_send(job(0)).unwrap();
        h.jobs.blocking_send(job(1)).unwrap();
        for _ in 0..2 {
            h.events.recv_timeout(Duration::from_secs(2)).unwrap();
        }
        drop(h.results);
        h.done.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(h.jobs.is_closed());
    }

    #[test]
    fn failed_or_stopped_ane_is_not_assigned_more_jobs() {
        for exit in [false, true] {
            let mut h = harness(Kernel::Msa, config(true, true), !exit, exit);
            h.gates[1].store(true, Ordering::Release);
            h.jobs.blocking_send(job(0)).unwrap();
            assert!(matches!(
                receive(&mut h.results).outcome,
                StreamOutcome::Completed(Err(SampleError::DeviceFault(_)))
            ));
            h.gates[0].store(true, Ordering::Release);
            h.jobs.blocking_send(job(1)).unwrap();
            drop(h.jobs);
            assert!(
                matches!(receive(&mut h.results).outcome,StreamOutcome::Completed(Ok(ref samples)) if samples[0].energy_milli == 0)
            );
            h.done.recv_timeout(Duration::from_secs(2)).unwrap();
            let events: Vec<_> = h.events.try_iter().collect();
            assert_eq!(events, [(1, 0), (0, 1)]);
        }
    }

    #[test]
    fn configuration_after_stream_start_controls_new_admission() {
        let mut h = harness(Kernel::Msa, config(true, true), false, false);
        *h.settings.lock().unwrap() = config(true, false);
        h.gates[0].store(true, Ordering::Release);
        h.jobs.blocking_send(job(0)).unwrap();
        assert_eq!(
            h.events.recv_timeout(Duration::from_secs(2)).unwrap(),
            (0, 0)
        );
        receive(&mut h.results);
        *h.settings.lock().unwrap() = config(false, true);
        h.gates[1].store(true, Ordering::Release);
        h.jobs.blocking_send(job(1)).unwrap();
        assert_eq!(
            h.events.recv_timeout(Duration::from_secs(2)).unwrap(),
            (1, 1)
        );
        receive(&mut h.results);
        drop(h.jobs);
        h.done.recv_timeout(Duration::from_secs(2)).unwrap();
    }

    #[test]
    fn cancelled_queued_jobs_do_not_open_either_engine() {
        let sampler = CombinedSampler::new(Kernel::Msa, 0, 100, false);
        let (send, jobs) = mpsc::channel(2);
        let (out, mut replies) = mpsc::channel(2);
        send.blocking_send(job(0)).unwrap();
        send.blocking_send(job(1)).unwrap();
        drop(send);
        let cancel = CancelToken::default();
        cancel.cancel_through(2);
        sampler.sample_stream(jobs, out, cancel);
        for _ in 0..2 {
            assert!(matches!(
                receive(&mut replies).outcome,
                StreamOutcome::Cancelled
            ));
        }
        assert!(sampler.metal.get().is_none());
        assert!(sampler.ane.lock().unwrap().is_none());
    }

    #[test]
    fn single_job_prefers_metal_when_both_engines_are_available() {
        let sampler = CombinedSampler::new(Kernel::Msa, 0, 100, false);
        sampler.metal.set(Err("selected Metal".into())).unwrap();
        *sampler.ane.lock().unwrap() = Some(Err("selected ANE".into()));
        let mut j = job(0);
        assert!(matches!(sampler.sample(&j.graph, &j.params),
            Err(SampleError::DeviceFault(message)) if message == "selected Metal"));
        sampler.apply_config("enable_metal = false");
        assert!(matches!(sampler.sample(&j.graph, &j.params),
            Err(SampleError::DeviceFault(message)) if message == "selected ANE"));
        sampler.apply_config("enable_metal = true");
        j.graph = IsingGraph::new(vec![0.0; crate::sampler::MSA_MAX_NODES + 1], vec![], vec![]);
        assert!(matches!(sampler.sample(&j.graph, &j.params),
            Err(SampleError::DeviceFault(message)) if message == "selected ANE"));
    }

    #[test]
    fn both_disabled_is_an_explicit_configuration_fault_without_opening_devices() {
        for kernel in [Kernel::Sa, Kernel::Gibbs, Kernel::Msa] {
            let sampler = CombinedSampler::new(kernel, 0, 100, false);
            sampler.apply_config("enable_ane = false\nenable_metal = false");
            let j = job(0);
            assert!(matches!(sampler.sample(&j.graph, &j.params),
                Err(SampleError::DeviceFault(message)) if message.contains("both false")));
            assert!(sampler.metal.get().is_none());
            assert!(sampler.ane.lock().unwrap().is_none());
        }
    }

    #[test]
    fn config_is_fail_closed_and_preserves_governor_settings_before_open() {
        let sampler = CombinedSampler::new(Kernel::Msa, 0, 73, true);
        sampler.apply_config(
            "enable_ane = false\nenable_metal = true\nutilization = 44\nyielding = false",
        );
        let cfg = settings(&sampler.settings);
        assert_eq!(
            cfg,
            Settings {
                valid: true,
                enable_ane: false,
                enable_metal: true,
                utilization: 44,
                yielding: false
            }
        );
        assert!(sampler.metal.get().is_none());
        assert!(sampler.ane.lock().unwrap().is_none());
        sampler.apply_config("enable_ane = falze");
        assert!(!settings(&sampler.settings).valid);
        let j = job(0);
        assert!(matches!(
            sampler.sample(&j.graph, &j.params),
            Err(SampleError::DeviceFault(_))
        ));
        sampler.apply_config("enable_metal = false");
        assert_eq!(
            eligibility(
                Kernel::Msa,
                &j.graph,
                &j.params,
                settings(&sampler.settings)
            ),
            [false, false]
        );
        assert!(sampler.metal.get().is_none());
        assert!(sampler.ane.lock().unwrap().is_none());
        sampler.apply_config("enable_metal = true");
        assert_eq!(
            eligibility(
                Kernel::Msa,
                &j.graph,
                &j.params,
                settings(&sampler.settings)
            ),
            [true, false]
        );
    }

    #[test]
    #[ignore = "requires coordinated Metal/ANE hardware access and QUIP_COMBINED_MSA_BIN"]
    fn hardware_all_three_msa_modes_rescore_every_result() {
        use quip_solver_core::quip_protocol::scoring::energy_milli;
        let executable = std::path::PathBuf::from(std::env::var("QUIP_COMBINED_MSA_BIN").unwrap());
        for (metal, ane) in [(true, false), (false, true), (true, true)] {
            let sampler = Arc::new(CombinedSampler::new(Kernel::Msa, 0, 100, false));
            sampler.apply_config(&format!("enable_metal = {metal}\nenable_ane = {ane}"));
            if ane {
                // Unit-test executables do not implement --ane-worker. Use the
                // actual miner binary for its checked child-process transport.
                let opened = AneSampler::open(executable.clone()).unwrap();
                *sampler.ane.lock().unwrap() = Some(Ok(Arc::new(opened)));
            }
            let (send, incoming) = mpsc::channel(8);
            let (out, mut replies) = mpsc::channel(8);
            let mut graphs = Vec::new();
            for id in 0..6 {
                let mut j = job(id);
                j.params.num_reads = 32;
                j.params.num_sweeps = 4;
                j.graph.h = vec![if id % 2 == 0 { 1.0 } else { -1.0 }, 0.0];
                graphs.push(j.graph.clone());
                send.blocking_send(j).unwrap();
            }
            drop(send);
            let running = Arc::clone(&sampler);
            let thread = std::thread::spawn(move || {
                running.sample_stream(incoming, out, CancelToken::default())
            });
            let deadline = Instant::now() + Duration::from_secs(20);
            let mut ids = HashSet::new();
            while ids.len() < 6 {
                let reply = match replies.try_recv() {
                    Ok(reply) => reply,
                    Err(mpsc::error::TryRecvError::Empty) => {
                        assert!(
                            Instant::now() < deadline,
                            "hardware smoke exceeded deadline"
                        );
                        std::thread::sleep(POLL);
                        continue;
                    }
                    Err(error) => panic!("hardware result channel: {error}"),
                };
                assert!(ids.insert(reply.job_id.clone()));
                let graph = &graphs[reply.job_id[0] as usize];
                let StreamOutcome::Completed(Ok(samples)) = reply.outcome else {
                    panic!("hardware job did not complete successfully");
                };
                assert_eq!(samples.len(), 32);
                for sample in samples {
                    assert_eq!(
                        sample.energy_milli,
                        energy_milli(&sample.spins, &graph.h, &graph.j, &graph.edges)
                    );
                }
            }
            thread.join().unwrap();
            assert_eq!(sampler.metal.get().is_some(), metal);
            assert_eq!(sampler.ane.lock().unwrap().is_some(), ane);
        }
    }

    #[test]
    fn admission_checks_each_engines_limits_without_dense_preparation() {
        let mut j = job(0);
        let cfg = config(true, true);
        assert_eq!(
            eligibility(Kernel::Msa, &j.graph, &j.params, cfg),
            [true, true]
        );
        j.graph.j[0] = 0.5;
        assert_eq!(
            eligibility(Kernel::Msa, &j.graph, &j.params, cfg),
            [true, false]
        );
        j.graph.j[0] = 1.0;
        j.params.num_reads = 129;
        assert_eq!(
            eligibility(Kernel::Msa, &j.graph, &j.params, cfg),
            [true, false]
        );
        j.params.num_reads = 128;
        j.graph = IsingGraph::new(vec![0.0; crate::sampler::MSA_MAX_NODES + 1], vec![], vec![]);
        assert_eq!(
            eligibility(Kernel::Msa, &j.graph, &j.params, cfg),
            [false, true]
        );
        j.graph = IsingGraph::new(vec![0.0; 16_385], vec![], vec![]);
        assert_eq!(
            eligibility(Kernel::Msa, &j.graph, &j.params, cfg),
            [false, false]
        );
        j.graph = IsingGraph::new(
            vec![0.0; 22],
            vec![1.0; 21],
            (1..22).map(|v| (0, v)).collect(),
        );
        assert_eq!(
            eligibility(Kernel::Msa, &j.graph, &j.params, cfg),
            [false, false]
        );
        j.graph.j.fill(0.0);
        assert_eq!(
            eligibility(Kernel::Msa, &j.graph, &j.params, cfg),
            [false, true]
        );
        j.graph = IsingGraph::new(vec![0.0; 2], vec![1.0], vec![(0, 2)]);
        assert_eq!(
            eligibility(Kernel::Msa, &j.graph, &j.params, cfg),
            [true, false]
        );
    }
}
