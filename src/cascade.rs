// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Probe jobs before spending their full sweep budget. Each admitted job returns
//! exactly one result: probe reads when screened out, full reads when kept, or
//! its error/cancellation. Budgets at or below the first probe pass through.
//! The relay polls bounded channels so results can release dispatch capacity
//! while new work waits. Metal dispatch remains on the calling thread.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use quip_solver_core::{CancelToken, SampleError, StreamJob, StreamOutcome, StreamResult};
use tokio::sync::mpsc::{self, Receiver, Sender};

use crate::cutoff::{Cutoff, CutoffConfig};

pub(crate) const MAX_STAGES: usize = 3;

/// Gate G4 constants. Times describe a 64-read job: a + b * sweeps.
pub(crate) struct Calibration {
    pub(crate) stages: &'static [usize],
    /// Audit-lane miss rate r0 for each probe-to-next-stage transition.
    #[expect(dead_code, reason = "Task 10 audit-lane model checks")]
    pub(crate) false_negative: &'static [f64],
    #[expect(dead_code, reason = "Task 10 distribution checks")]
    pub(crate) skew: &'static [f64],
    #[expect(dead_code, reason = "Task 10 distribution checks")]
    pub(crate) excess_kurtosis: &'static [f64],
    pub(crate) cost_a_us: f64,
    pub(crate) cost_b_us: f64,
    pub(crate) k0: f64,
}

pub(crate) const CALIBRATION: Calibration = Calibration {
    stages: &[32, 128, 256],
    false_negative: &[0.0581, 0.0496, 0.0804],
    skew: &[-0.138, -0.086, -0.076],
    excess_kurtosis: &[0.059, 0.026, 0.018],
    cost_a_us: 122.3,
    cost_b_us: 2.367,
    k0: 1.0,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CascadeSettings {
    pub(crate) enabled: bool,
    pub(crate) stages: [usize; MAX_STAGES],
    /// Overall probe-to-full denominator, factored across configured probes.
    pub(crate) keep: f64,
    pub(crate) keep_min: f64,
    pub(crate) keep_max: f64,
    pub(crate) audit: u32,
    pub(crate) target_milli: Option<i64>,
    pub(crate) yield_per_million: Option<f64>,
}

impl Default for CascadeSettings {
    fn default() -> Self {
        let mut stages = [0; MAX_STAGES];
        stages.copy_from_slice(CALIBRATION.stages);
        Self {
            enabled: false,
            stages,
            keep: 3_000.0,
            keep_min: 1_000.0,
            keep_max: 30_000.0,
            audit: 200,
            target_milli: None,
            yield_per_million: None,
        }
    }
}

#[derive(serde::Deserialize, Default, Clone)]
pub(crate) struct CascadeToml {
    pub(crate) cascade: Option<bool>,
    pub(crate) cascade_stages: Option<Vec<usize>>,
    pub(crate) cascade_keep: Option<u32>,
    pub(crate) cascade_keep_min: Option<u32>,
    pub(crate) cascade_keep_max: Option<u32>,
    pub(crate) cascade_audit: Option<u32>,
    pub(crate) cascade_target_milli: Option<i64>,
    pub(crate) cascade_yield_per_million: Option<f64>,
}

impl CascadeSettings {
    /// Merge valid keys, retaining previous values for invalid keys. The three
    /// overall keep denominators are validated and accepted as one group.
    pub(crate) fn merge(&mut self, cfg: &CascadeToml) {
        if let Some(value) = cfg.cascade {
            self.enabled = value;
        }
        if let Some(stages) = &cfg.cascade_stages {
            if (1..=MAX_STAGES).contains(&stages.len())
                && stages[0] >= 1
                && stages.windows(2).all(|pair| pair[0] < pair[1])
            {
                self.stages = [0; MAX_STAGES];
                self.stages[..stages.len()].copy_from_slice(stages);
            } else {
                tracing::warn!(
                    ?stages,
                    "invalid cascade_stages; retaining previous budgets"
                );
            }
        }
        let keep = cfg.cascade_keep.map_or(self.keep, f64::from);
        let min = cfg.cascade_keep_min.map_or(self.keep_min, f64::from);
        let max = cfg.cascade_keep_max.map_or(self.keep_max, f64::from);
        if min >= 2.0 && min <= keep && keep <= max {
            self.keep = keep;
            self.keep_min = min;
            self.keep_max = max;
        } else {
            tracing::warn!(
                keep,
                min,
                max,
                "invalid cascade keep denominators; retaining previous values"
            );
        }
        if let Some(audit) = cfg.cascade_audit {
            if audit >= 2 {
                self.audit = audit;
            } else {
                tracing::warn!(audit, "invalid cascade_audit; retaining previous value");
            }
        }
        if let Some(target) = cfg.cascade_target_milli {
            self.target_milli = Some(target);
        }
        if let Some(value) = cfg.cascade_yield_per_million {
            if value.is_finite() && value > 0.0 {
                self.yield_per_million = Some(value);
            } else {
                tracing::warn!(
                    value,
                    "invalid cascade_yield_per_million; retaining previous value"
                );
            }
        }
    }
}

fn stage_seed(seed: u64, stage: usize) -> u64 {
    seed ^ 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(stage as u64 + 1)
}

fn copy_job(job: &StreamJob) -> StreamJob {
    StreamJob {
        job_id: job.job_id.clone(),
        graph: job.graph.clone(),
        params: job.params.clone(),
        watermark: job.watermark,
    }
}

struct Active {
    original: StreamJob,
    stage: usize,
    probes: usize,
    device_us: u64,
    topology_epoch: u64,
}

struct Relay {
    settings: CascadeSettings,
    cutoffs: Vec<Cutoff>,
    active: HashMap<Vec<u8>, Active>,
    ready: VecDeque<StreamJob>,
    topology: Option<(usize, Vec<(usize, usize)>)>,
    topology_epoch: u64,
    width: usize,
    counts: [u64; MAX_STAGES + 1],
    cost_us: f64,
    window: Instant,
    logged: Instant,
}

impl Relay {
    fn new(settings: CascadeSettings, width: usize) -> Self {
        let count = settings.stages.iter().filter(|&&s| s > 0).count();
        // Keep one persistent controller per configured index. Use configured
        // count for the root, even when a job has fewer effective probes, so
        // mixed budgets do not rebuild controllers and lose their observations.
        // Such jobs skip the unused transitions and have a looser overall keep.
        let root = 1.0 / count.max(1) as f64;
        let cfg = CutoffConfig {
            keep: settings.keep.powf(root),
            keep_min: settings.keep_min.powf(root),
            keep_max: settings.keep_max.powf(root),
            k0: CALIBRATION.k0,
            ..CutoffConfig::default()
        };
        Self {
            settings,
            cutoffs: (0..=count).map(|_| Cutoff::new(cfg)).collect(),
            active: HashMap::new(),
            ready: VecDeque::new(),
            topology: None,
            topology_epoch: 0,
            width: width.max(1),
            counts: [0; MAX_STAGES + 1],
            cost_us: 0.0,
            window: Instant::now(),
            logged: Instant::now(),
        }
    }

    fn admit(&mut self, job: StreamJob) {
        let same = self
            .topology
            .as_ref()
            .is_some_and(|(n, edges)| *n == job.graph.num_nodes() && *edges == job.graph.edges);
        if !same {
            for cutoff in &mut self.cutoffs {
                cutoff.reset();
            }
            self.topology = Some((job.graph.num_nodes(), job.graph.edges.clone()));
            self.topology_epoch = self.topology_epoch.wrapping_add(1);
        }
        let probes = self
            .settings
            .stages
            .iter()
            .take_while(|&&s| s > 0 && s < job.params.num_sweeps)
            .count();
        let mut queued = copy_job(&job);
        if probes > 0 {
            queued.params.num_sweeps = self.settings.stages[0];
            queued.params.seed = stage_seed(job.params.seed, 0);
        }
        self.active.insert(
            job.job_id.clone(),
            Active {
                original: job,
                stage: 0,
                probes,
                device_us: 0,
                topology_epoch: self.topology_epoch,
            },
        );
        self.ready.push_back(queued);
    }

    fn complete(&mut self, mut result: StreamResult, out: &Sender<StreamResult>) {
        let Some(mut active) = self.active.remove(&result.job_id) else {
            tracing::error!("cascade received an unassigned or duplicate result");
            return;
        };
        active.device_us = active
            .device_us
            .saturating_add(result.device_access_time_us);
        if let StreamOutcome::Completed(Ok(reads)) = &result.outcome {
            if let Some(best) = reads.iter().map(|r| r.energy_milli).min() {
                let current_topology = active.topology_epoch == self.topology_epoch;
                if active.stage < active.probes {
                    // Old-topology work can finish, but must not train the new
                    // distribution. Screen it out with its available reads.
                    if current_topology && self.cutoffs[active.stage].decide(best as f64) {
                        active.stage += 1;
                        let mut next = copy_job(&active.original);
                        if active.stage < active.probes {
                            next.params.num_sweeps = self.settings.stages[active.stage];
                            next.params.seed = stage_seed(next.params.seed, active.stage);
                        }
                        self.ready.push_front(next);
                        self.active.insert(result.job_id, active);
                        return;
                    }
                } else if current_topology {
                    // The last controller records full-budget observations,
                    // separate from all configured probe distributions.
                    let final_index = self.cutoffs.len() - 1;
                    self.cutoffs[final_index].observe(best as f64);
                }
            }
        }
        result.device_access_time_us = active.device_us;
        let _ = out.blocking_send(result);
    }

    /// Cancellation before dispatch refunds the job once, including time from
    /// any completed probes. In-flight cancellation belongs to the inner loop.
    fn dispatch(
        &mut self,
        inner: &Sender<StreamJob>,
        out: &Sender<StreamResult>,
        cancel: &CancelToken,
    ) -> bool {
        let mut moved = false;
        while let Some(job) = self.ready.pop_front() {
            if cancel.is_cancelled(job.watermark) {
                self.complete(
                    StreamResult {
                        job_id: job.job_id,
                        outcome: StreamOutcome::Cancelled,
                        device_access_time_us: 0,
                    },
                    out,
                );
                moved = true;
                continue;
            }
            let sweeps = job.params.num_sweeps;
            let stage = self.active.get(&job.job_id).map_or(0, |a| {
                if a.stage == a.probes {
                    self.cutoffs.len() - 1
                } else {
                    a.stage
                }
            });
            match inner.try_send(job) {
                Ok(()) => {
                    moved = true;
                    self.counts[stage] += 1;
                    self.cost_us += CALIBRATION.cost_a_us + CALIBRATION.cost_b_us * sweeps as f64;
                }
                Err(mpsc::error::TrySendError::Full(job)) => {
                    self.ready.push_front(job);
                    break;
                }
                Err(mpsc::error::TrySendError::Closed(job)) => {
                    self.complete(
                        StreamResult {
                            job_id: job.job_id,
                            outcome: StreamOutcome::Completed(Err(SampleError::DeviceFault(
                                "cascade inner stream closed".into(),
                            ))),
                            device_access_time_us: 0,
                        },
                        out,
                    );
                    moved = true;
                }
            }
            if out.is_closed() {
                break;
            }
        }
        moved
    }

    fn update_load(&mut self) {
        let elapsed = self.window.elapsed().as_secs_f64();
        if elapsed < 1.0 {
            return;
        }
        let busy = self.cost_us / 1e6 / elapsed;
        let slack = (0.9 - busy).max(0.0) / 0.9;
        for stage in 0..self.cutoffs.len() - 1 {
            let queued_next = self
                .ready
                .iter()
                .filter(|job| {
                    self.active
                        .get(&job.job_id)
                        .is_some_and(|a| a.stage == stage + 1)
                })
                .count();
            let backlog = (queued_next as f64 / (2 * self.width) as f64).max(1.0).ln();
            let cutoff = &mut self.cutoffs[stage];
            let error = if backlog == 0.0 && slack == 0.0 {
                -cutoff.log_adjust()
            } else {
                backlog - slack
            };
            cutoff.load_update(error, 0.1);
        }
        if self.logged.elapsed() >= Duration::from_secs(60) {
            for (stage, cutoff) in self.cutoffs.iter().enumerate() {
                tracing::info!(stage, count = self.counts[stage], denominator = cutoff.denominator(), cutoff = ?cutoff.cutoff(), busy, "cascade load window");
            }
            self.logged = Instant::now();
        }
        self.counts.fill(0);
        self.cost_us = 0.0;
        self.window = Instant::now();
    }

    fn run(
        mut self,
        mut jobs: Receiver<StreamJob>,
        inner: Sender<StreamJob>,
        mut results: Receiver<StreamResult>,
        out: &Sender<StreamResult>,
        cancel: &CancelToken,
    ) {
        let mut eof = false;
        let mut inner_closed = false;
        loop {
            if out.is_closed() || (eof && self.active.is_empty()) {
                return;
            }
            let mut moved = false;
            loop {
                match results.try_recv() {
                    Ok(result) => {
                        self.complete(result, out);
                        moved = true;
                    }
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        inner_closed = true;
                        break;
                    }
                }
            }
            // Bound owned jobs as well as channels; a fast producer must not
            // turn backpressure into an unbounded ready queue.
            if !eof && self.active.len() < 2 * self.width {
                match jobs.try_recv() {
                    Ok(job) => {
                        self.admit(job);
                        moved = true;
                    }
                    Err(mpsc::error::TryRecvError::Empty) => {}
                    Err(mpsc::error::TryRecvError::Disconnected) => eof = true,
                }
            }
            if inner_closed {
                self.ready.clear();
                for (job_id, active) in self.active.drain() {
                    let _ = out.blocking_send(StreamResult {
                        job_id,
                        outcome: StreamOutcome::Completed(Err(SampleError::DeviceFault(
                            "cascade inner stream stopped before completing job".into(),
                        ))),
                        device_access_time_us: active.device_us,
                    });
                    moved = true;
                }
            } else {
                moved |= self.dispatch(&inner, out, cancel);
            }
            self.update_load();
            if !moved {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

/// Run a named scoped relay while dispatch stays on this thread. Dropping its
/// inner sender ends dispatch after upstream EOF and the last tracked result.
pub(crate) fn run(
    settings: CascadeSettings,
    jobs: Receiver<StreamJob>,
    out: &Sender<StreamResult>,
    cancel: &CancelToken,
    width: usize,
    inner: impl FnOnce(Receiver<StreamJob>, Sender<StreamResult>),
) {
    std::thread::scope(|scope| {
        let (tx, rx) = mpsc::channel(width.max(1));
        let (result_tx, result_rx) = mpsc::channel(width.max(1));
        match std::thread::Builder::new()
            .name("metal-cascade".into())
            .spawn_scoped(scope, move || {
                Relay::new(settings, width).run(jobs, tx, result_rx, out, cancel)
            }) {
            Ok(_) => inner(rx, result_tx),
            Err(error) => tracing::error!(%error, "failed to start cascade relay"),
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_solver_core::{IsingGraph, SampleParams, SamplerResult};

    fn job(id: u64, sweeps: usize) -> StreamJob {
        StreamJob {
            job_id: id.to_le_bytes().to_vec(),
            graph: IsingGraph::new(vec![0.0; 2], vec![1.0], vec![(0, 1)]),
            params: SampleParams {
                num_reads: 2,
                num_sweeps: sweeps,
                sweeps_per_beta: 1,
                beta_range: None,
                seed: 42,
            },
            watermark: Some(1),
        }
    }

    fn answer(job: StreamJob) -> StreamResult {
        let id = u64::from_le_bytes(job.job_id.as_slice().try_into().unwrap());
        let energy = -(((id.wrapping_mul(0x9e3779b97f4a7c15) >> 32) % 1000) as i64);
        StreamResult {
            job_id: job.job_id,
            outcome: StreamOutcome::Completed(Ok((0..job.params.num_reads)
                .map(|_| SamplerResult {
                    spins: vec![job.params.num_sweeps as i8],
                    energy_milli: energy * 10_000 - job.params.num_sweeps as i64,
                })
                .collect())),
            device_access_time_us: job.params.num_sweeps as u64,
        }
    }

    fn exercise(
        settings: CascadeSettings,
        input: Vec<StreamJob>,
        cancel: &CancelToken,
        inner: impl FnOnce(Receiver<StreamJob>, Sender<StreamResult>),
    ) -> Vec<StreamResult> {
        let (tx, rx) = mpsc::channel(input.len().max(1));
        let (out, mut results) = mpsc::channel(input.len().max(1));
        for job in input {
            tx.blocking_send(job).unwrap();
        }
        drop(tx);
        let caller = std::thread::current().id();
        run(settings, rx, &out, cancel, 4, |rx, tx| {
            assert_eq!(std::thread::current().id(), caller);
            inner(rx, tx);
        });
        drop(out);
        let mut replies = Vec::new();
        while let Some(reply) = results.blocking_recv() {
            replies.push(reply);
        }
        replies
    }

    fn fake_inner(mut rx: Receiver<StreamJob>, tx: Sender<StreamResult>) {
        while let Some(job) = rx.blocking_recv() {
            tx.blocking_send(answer(job)).unwrap();
        }
    }

    #[test]
    fn every_job_gets_exactly_one_result() {
        let settings = CascadeSettings {
            stages: [32, 256, 0],
            ..CascadeSettings::default()
        };
        let results = exercise(
            settings,
            (0..5000).map(|i| job(i, 1024)).collect(),
            &CancelToken::default(),
            fake_inner,
        );
        let ids: std::collections::HashSet<_> = results.iter().map(|r| &r.job_id).collect();
        assert_eq!(results.len(), 5000);
        assert_eq!(ids.len(), 5000);
    }

    #[test]
    fn screened_out_jobs_return_probe_reads_and_kept_jobs_return_full_reads() {
        // One configured probe: overall keep 20 implies per-stage keep 20.
        let settings = CascadeSettings {
            stages: [32, 0, 0],
            keep: 20.0,
            keep_min: 20.0,
            keep_max: 20.0,
            ..CascadeSettings::default()
        };
        let results = exercise(
            settings,
            (0..5000).map(|i| job(i, 1024)).collect(),
            &CancelToken::default(),
            fake_inner,
        );
        let mut counts = [0; 2];
        for result in results {
            let StreamOutcome::Completed(Ok(reads)) = result.outcome else {
                panic!("unexpected result")
            };
            assert_eq!(reads.len(), 2);
            let sweeps = (-reads[0].energy_milli) % 10_000;
            match sweeps {
                32 => {
                    counts[0] += 1;
                    assert_eq!(result.device_access_time_us, 32);
                }
                1024 => {
                    counts[1] += 1;
                    assert_eq!(result.device_access_time_us, 1056);
                }
                _ => panic!("unexpected budget {sweeps}"),
            }
        }
        assert!(counts.iter().all(|&n| n > 0), "{counts:?}");
    }

    #[test]
    fn a_job_at_or_below_the_first_budget_passes_straight_through() {
        let mut dispatched = Vec::new();
        let results = exercise(
            CascadeSettings::default(),
            vec![job(0, 32), job(1, 16)],
            &CancelToken::default(),
            |mut rx, tx| {
                while let Some(job) = rx.blocking_recv() {
                    dispatched.push((job.params.num_sweeps, job.params.seed, job.watermark));
                    tx.blocking_send(answer(job)).unwrap();
                }
            },
        );
        assert_eq!(dispatched, vec![(32, 42, Some(1)), (16, 42, Some(1))]);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn cancellation_before_dispatch_refunds_completed_probe_time() {
        let settings = CascadeSettings {
            stages: [32, 128, 0],
            ..CascadeSettings::default()
        };
        let mut relay = Relay::new(settings, 1);
        relay.admit(job(0, 1024));
        for _ in 0..200 {
            relay.cutoffs[0].observe(0.0);
        }
        let probe = relay.ready.pop_front().unwrap();
        let (out, mut results) = mpsc::channel(2);
        relay.complete(answer(probe), &out);
        assert_eq!(relay.ready.front().unwrap().params.num_sweeps, 128);
        let cancel = CancelToken::default();
        cancel.cancel_through(1);
        let (inner, mut rx) = mpsc::channel(1);
        relay.dispatch(&inner, &out, &cancel);
        drop(out);
        let result = results.blocking_recv().unwrap();
        assert_eq!(result.job_id, 0u64.to_le_bytes());
        assert!(matches!(result.outcome, StreamOutcome::Cancelled));
        assert_eq!(result.device_access_time_us, 32);
        assert!(results.blocking_recv().is_none());
        assert!(rx.try_recv().is_err());
        assert!(relay.active.is_empty());
    }

    #[test]
    fn cancel_between_stages_yields_one_cancelled_result() {
        let settings = CascadeSettings {
            stages: [32, 128, 0],
            ..CascadeSettings::default()
        };
        let cancel = CancelToken::default();
        let mut held_second_stage = false;
        let results = exercise(
            settings,
            (0..201).map(|i| job(i, 1024)).collect(),
            &cancel,
            |mut rx, tx| {
                while let Some(job) = rx.blocking_recv() {
                    let sweeps = job.params.num_sweeps;
                    let mut reply = answer(job);
                    if sweeps == 128 {
                        // Hold this stage until cancellation. Its stage-0 result
                        // has already reached the relay, otherwise it cannot run.
                        held_second_stage = true;
                        cancel.cancel_through(1);
                        reply.outcome = StreamOutcome::Cancelled;
                    } else if let StreamOutcome::Completed(Ok(reads)) = &mut reply.outcome {
                        // Equal energies make job 200 pass after 200 warm-up jobs.
                        for read in reads {
                            read.energy_milli = -32;
                        }
                    }
                    tx.blocking_send(reply).unwrap();
                }
            },
        );
        assert!(held_second_stage);
        assert_eq!(results.len(), 201);
        let cancelled: Vec<_> = results
            .iter()
            .filter(|r| r.job_id == 200u64.to_le_bytes())
            .collect();
        assert_eq!(cancelled.len(), 1);
        assert!(matches!(cancelled[0].outcome, StreamOutcome::Cancelled));
        assert_eq!(cancelled[0].device_access_time_us, 160);
    }

    #[test]
    fn inner_error_is_forwarded_once() {
        let results = exercise(
            CascadeSettings::default(),
            vec![job(0, 1024)],
            &CancelToken::default(),
            |mut rx, tx| {
                let job = rx.blocking_recv().unwrap();
                tx.blocking_send(StreamResult {
                    job_id: job.job_id,
                    outcome: StreamOutcome::Completed(Err(SampleError::Capacity)),
                    device_access_time_us: 7,
                })
                .unwrap();
                assert!(rx.blocking_recv().is_none());
            },
        );
        assert_eq!(results.len(), 1);
        assert!(matches!(
            results[0].outcome,
            StreamOutcome::Completed(Err(SampleError::Capacity))
        ));
        assert_eq!(results[0].device_access_time_us, 7);
    }

    #[test]
    fn topology_change_resets_the_controllers() {
        let mut relay = Relay::new(CascadeSettings::default(), 1);
        relay.admit(job(0, 1024));
        relay.cutoffs[0].observe(-1.0);
        relay.admit(job(1, 1024));
        assert_eq!(relay.cutoffs[0].moments().count(), 1);
        let mut changed = job(2, 1024);
        changed.graph = IsingGraph::new(vec![0.0; 2], vec![], vec![]);
        relay.admit(changed);
        assert_eq!(relay.cutoffs[0].moments().count(), 0);
    }

    #[test]
    fn stage_seeds_differ_from_the_job_seed_and_each_other() {
        for seed in [0, 42, u64::MAX] {
            assert_ne!(stage_seed(seed, 0), seed);
            assert_ne!(stage_seed(seed, 0), stage_seed(seed, 1));
            assert_ne!(stage_seed(seed, 1), stage_seed(seed, 2));
        }
    }

    #[test]
    fn merge_rejects_invalid_values_and_keeps_the_previous_ones() {
        let mut settings = CascadeSettings::default();
        let previous = settings;
        settings.merge(&CascadeToml {
            cascade_stages: Some(vec![128, 32]),
            cascade_keep_min: Some(4000),
            cascade_audit: Some(0),
            cascade_yield_per_million: Some(f64::NAN),
            ..CascadeToml::default()
        });
        assert_eq!(settings, previous);
        for stages in [vec![], vec![0], vec![32, 32], vec![1, 2, 3, 4]] {
            settings.merge(&CascadeToml {
                cascade_stages: Some(stages),
                ..CascadeToml::default()
            });
            assert_eq!(settings, previous);
        }
    }

    #[test]
    fn backend_toml_routes_cascade_keys_and_leaves_unknown_keys_unknown() {
        let cfg: crate::MetalConfig =
            toml::from_str("cascade = true\ncascade_keep = 5000\nbogus = 1").unwrap();
        assert_eq!(cfg.cascade.cascade, Some(true));
        assert_eq!(cfg.cascade.cascade_keep, Some(5000));
        assert_eq!(cfg.unknown.keys().collect::<Vec<_>>(), vec!["bogus"]);
    }

    #[test]
    fn configured_stage_roots_and_effective_budgets_preserve_the_original_job() {
        let mut relay = Relay::new(CascadeSettings::default(), 1);
        assert!((relay.cutoffs[0].denominator() - 10.0).abs() < 1e-10);
        let mut original = job(0, 128);
        original.params.beta_range = Some((0.5, 4.0));
        original.params.sweeps_per_beta = 2;
        relay.admit(original);
        assert_eq!(relay.active.values().next().unwrap().probes, 1);
        for _ in 0..200 {
            relay.cutoffs[0].observe(0.0);
        }
        let probe = relay.ready.pop_front().unwrap();
        assert_eq!(probe.params.seed, stage_seed(42, 0));
        assert_eq!(probe.params.num_sweeps, 32);
        let (out, mut results) = mpsc::channel(1);
        relay.complete(answer(probe), &out);
        let full = relay.ready.pop_front().unwrap();
        assert_eq!(full.params.seed, 42);
        assert_eq!(full.params.num_sweeps, 128);
        assert_eq!(full.params.num_reads, 2);
        assert_eq!(full.params.sweeps_per_beta, 2);
        assert_eq!(full.params.beta_range, Some((0.5, 4.0)));
        assert_eq!(full.watermark, Some(1));
        assert_eq!(full.graph.edges, vec![(0, 1)]);
        relay.complete(answer(full), &out);
        assert_eq!(results.try_recv().unwrap().device_access_time_us, 160);
        assert_eq!(relay.cutoffs[1].moments().count(), 0);
        assert_eq!(relay.cutoffs[3].moments().count(), 1);
    }

    #[test]
    fn late_results_do_not_train_a_new_topology() {
        let mut relay = Relay::new(CascadeSettings::default(), 1);
        relay.admit(job(0, 1024));
        let old = relay.ready.pop_front().unwrap();
        let mut changed = job(1, 1024);
        changed.graph.h.push(0.0);
        relay.admit(changed);
        let (out, mut results) = mpsc::channel(1);
        relay.complete(answer(old), &out);
        assert_eq!(relay.cutoffs[0].moments().count(), 0);
        assert_eq!(results.try_recv().unwrap().job_id, 0u64.to_le_bytes());
    }

    #[test]
    fn full_inner_queue_preserves_ready_order_and_counts_only_dispatches() {
        let mut relay = Relay::new(CascadeSettings::default(), 1);
        relay.admit(job(0, 1024));
        relay.admit(job(1, 1024));
        let (inner, mut rx) = mpsc::channel(1);
        let (out, _results) = mpsc::channel(2);
        let cancel = CancelToken::default();
        assert!(relay.dispatch(&inner, &out, &cancel));
        assert_eq!(relay.counts[0], 1);
        assert_eq!(relay.ready.front().unwrap().job_id, 1u64.to_le_bytes());
        assert!(!relay.dispatch(&inner, &out, &cancel));
        assert_eq!(rx.try_recv().unwrap().job_id, 0u64.to_le_bytes());
        assert!(relay.dispatch(&inner, &out, &cancel));
        assert_eq!(relay.counts[0], 2);
        assert_eq!(rx.try_recv().unwrap().job_id, 1u64.to_le_bytes());
    }

    #[test]
    fn inner_disconnect_refunds_all_jobs_without_hanging() {
        let results = exercise(
            CascadeSettings::default(),
            (0..20).map(|i| job(i, 1024)).collect(),
            &CancelToken::default(),
            |mut rx, _tx| {
                assert!(rx.blocking_recv().is_some());
            },
        );
        assert_eq!(results.len(), 20);
        assert_eq!(
            results
                .iter()
                .map(|r| &r.job_id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            20
        );
        assert!(results.iter().all(|r| matches!(
            r.outcome,
            StreamOutcome::Completed(Err(SampleError::DeviceFault(_)))
        )));
    }

    #[test]
    fn load_feedback_loosens_tightens_and_decays() {
        let mut relay = Relay::new(CascadeSettings::default(), 1);
        relay.window = Instant::now() - Duration::from_secs(1);
        relay.update_load();
        assert!(relay.cutoffs[0].log_adjust() < 0.0);
        relay.cutoffs[0].reset();
        for id in 0..4 {
            relay.admit(job(id, 1024));
            relay
                .active
                .get_mut(id.to_le_bytes().as_slice())
                .unwrap()
                .stage = 1;
        }
        relay.cost_us = 2e6;
        relay.window = Instant::now() - Duration::from_secs(1);
        relay.update_load();
        let positive = relay.cutoffs[0].log_adjust();
        assert!(positive > 0.0);
        relay.ready.clear();
        relay.cost_us = 2e6;
        relay.window = Instant::now() - Duration::from_secs(1);
        relay.update_load();
        assert!((relay.cutoffs[0].log_adjust() - 0.9 * positive).abs() < 1e-12);
        assert_eq!(relay.counts, [0; MAX_STAGES + 1]);
    }

    #[test]
    fn merge_accepts_all_keys_and_partial_updates() {
        let cfg: crate::MetalConfig = toml::from_str("cascade = true\ncascade_stages = [16, 64]\ncascade_keep = 20\ncascade_keep_min = 2\ncascade_keep_max = 100\ncascade_audit = 2\ncascade_target_milli = -500\ncascade_yield_per_million = 1.5").unwrap();
        assert!(cfg.unknown.is_empty());
        let mut settings = CascadeSettings::default();
        settings.merge(&cfg.cascade);
        assert_eq!(
            settings,
            CascadeSettings {
                enabled: true,
                stages: [16, 64, 0],
                keep: 20.0,
                keep_min: 2.0,
                keep_max: 100.0,
                audit: 2,
                target_milli: Some(-500),
                yield_per_million: Some(1.5)
            }
        );
        let previous = settings;
        settings.merge(&CascadeToml {
            cascade: Some(false),
            ..CascadeToml::default()
        });
        assert_eq!(
            settings,
            CascadeSettings {
                enabled: false,
                ..previous
            }
        );
    }
}
