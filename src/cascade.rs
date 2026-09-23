// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Probe jobs before spending their full sweep budget. Each admitted job returns
//! exactly one result: probe reads when screened out, full reads when kept, or
//! its error/cancellation. Budgets at or below the first probe pass through.
//! The relay polls bounded channels so results can release dispatch capacity
//! while new work waits. Metal dispatch remains on the calling thread.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use quip_solver_core::{CancelToken, SampleError, StreamJob, StreamOutcome, StreamResult};
use tokio::sync::mpsc::{self, Receiver, Sender};

use crate::cutoff::{Cutoff, CutoffConfig};
use crate::model_checks::{Action, AuditLane, Drift, Yield};

pub(crate) const MAX_STAGES: usize = 3;

/// Gate G4 constants. Times describe a 64-read job: a + b * sweeps.
pub(crate) struct Calibration {
    pub(crate) stages: &'static [usize],
    /// Audit-lane miss rate r0 for each probe-to-next-stage transition.
    pub(crate) false_negative: &'static [((usize, usize), f64)],
    pub(crate) skew: &'static [f64],
    pub(crate) excess_kurtosis: &'static [f64],
    pub(crate) cost_a_us: f64,
    pub(crate) cost_b_us: f64,
    pub(crate) k0: f64,
}

pub(crate) const CALIBRATION: Calibration = Calibration {
    stages: &[32, 256],
    false_negative: &[((32, 256), 0.0348), ((256, 14_336), 0.0357)],
    skew: &[-0.147, -0.091],
    excess_kurtosis: &[0.041, 0.016],
    cost_a_us: 124.0,
    cost_b_us: 2.284,
    k0: 1.0,
};

fn calibrated_miss_rate(from: usize, to: usize) -> Option<f64> {
    CALIBRATION
        .false_negative
        .iter()
        .find_map(|&(transition, rate)| (transition == (from, to)).then_some(rate))
}

fn calibrated_rate_from(from: usize) -> Option<f64> {
    CALIBRATION
        .false_negative
        .iter()
        .find_map(|&((start, _), rate)| (start == from).then_some(rate))
}

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
        stages[..CALIBRATION.stages.len()].copy_from_slice(CALIBRATION.stages);
        Self {
            enabled: false,
            stages,
            keep: 2_000.0,
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

fn kept_median(kept: &VecDeque<i64>) -> Option<f64> {
    if kept.is_empty() {
        return None;
    }
    let mut sorted: Vec<_> = kept.iter().copied().collect();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        Some(sorted[mid - 1] as f64 / 2.0 + sorted[mid] as f64 / 2.0)
    } else {
        Some(sorted[mid] as f64)
    }
}

struct Active {
    original: Option<StreamJob>,
    // Keep the admitted controllers alive across configuration changes.
    plan: Rc<RefCell<StagePlan>>,
    stage: usize,
    probes: usize,
    audited: bool,
    device_us: u64,
    topology_epoch: u64,
    yield_epoch: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Check {
    Audit,
    Drift,
    Yield,
}

struct StagePlan {
    settings: CascadeSettings,
    cutoffs: Vec<Cutoff>,
    audit_lanes: Vec<AuditLane>,
    kept: Vec<VecDeque<i64>>,
    audit_rng: Option<u64>,
    audit_loosen: Vec<f64>,
    audit_unavailable_logged: Vec<bool>,
    drift: Drift,
}

struct Relay {
    settings: CascadeSettings,
    plan: Rc<RefCell<StagePlan>>,
    yield_check: Option<Yield>,
    active: HashMap<Vec<u8>, Active>,
    ready: VecDeque<StreamJob>,
    topology: Option<(usize, Vec<(usize, usize)>)>,
    topology_epoch: u64,
    yield_epoch: u64,
    width: usize,
    counts: [u64; MAX_STAGES + 1],
    cost_us: f64,
    window: Instant,
    logged: Instant,
}

impl StagePlan {
    fn new(settings: CascadeSettings) -> Self {
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
            audit_lanes: (0..count)
                .map(|s| AuditLane::new(calibrated_rate_from(settings.stages[s]).unwrap_or(0.0)))
                .collect(),
            kept: (0..count).map(|_| VecDeque::with_capacity(256)).collect(),
            audit_rng: None,
            audit_loosen: vec![1.0; count],
            audit_unavailable_logged: vec![false; count],
            drift: Drift::new(CALIBRATION.skew[0], CALIBRATION.excess_kurtosis[0]),
        }
    }

    fn audit_supported(&mut self, active: &Active, stage: usize) -> bool {
        let from = self.settings.stages[stage];
        let to = if stage + 1 < active.probes {
            self.settings.stages[stage + 1]
        } else {
            active
                .original
                .as_ref()
                .map_or(0, |job| job.params.num_sweeps)
        };
        let supported = calibrated_miss_rate(from, to).is_some();
        if !supported && !self.audit_unavailable_logged[stage] {
            tracing::debug!(stage, from, to, "cascade audit unavailable for transition");
            self.audit_unavailable_logged[stage] = true;
        }
        supported
    }

    fn audit_selected(&mut self, stage: usize) -> bool {
        let Some(state) = &mut self.audit_rng else {
            return false;
        };
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state % u64::from(self.audit_lanes[stage].denominator(self.settings.audit)) == 0
    }

    fn apply_action(
        &mut self,
        stage: usize,
        action: Action,
        check: Check,
        observed: &[f64],
        expected: &[f64],
    ) {
        let cutoff = &mut self.cutoffs[stage];
        match action {
            Action::None => return,
            Action::Loosen { raise_audit } => {
                let factor = cutoff.loosen_step();
                if check == Check::Audit {
                    self.audit_loosen[stage] *= factor;
                }
                if raise_audit {
                    self.audit_lanes[stage].raise();
                }
            }
            Action::Restore => {
                cutoff.restore_step(self.audit_loosen[stage]);
                self.audit_loosen[stage] = 1.0;
            }
            Action::LoosenAndDoubleK0 => {
                cutoff.loosen_step();
                cutoff.double_k0();
            }
        }
        tracing::warn!(
            stage,
            ?check,
            ?action,
            ?observed,
            ?expected,
            denominator = cutoff.denominator(),
            audit_denominator = self.audit_lanes[stage].denominator(self.settings.audit),
            "cascade model check"
        );
    }
}

impl Relay {
    fn new(settings: CascadeSettings, width: usize) -> Self {
        Self {
            settings,
            plan: Rc::new(RefCell::new(StagePlan::new(settings))),
            yield_check: settings
                .yield_per_million
                .zip(settings.target_milli)
                .map(|(rate, target)| Yield::new(rate, target, Duration::from_secs(3600))),
            active: HashMap::new(),
            ready: VecDeque::new(),
            topology: None,
            topology_epoch: 0,
            yield_epoch: 0,
            width: width.max(1),
            counts: [0; MAX_STAGES + 1],
            cost_us: 0.0,
            window: Instant::now(),
            logged: Instant::now(),
        }
    }

    fn refresh_settings(&mut self, shared: &Mutex<CascadeSettings>) {
        let settings = *shared.lock().unwrap_or_else(|p| p.into_inner());
        if settings.stages != self.settings.stages
            || settings.keep != self.settings.keep
            || settings.keep_min != self.settings.keep_min
            || settings.keep_max != self.settings.keep_max
        {
            self.plan = Rc::new(RefCell::new(StagePlan::new(settings)));
        }
        if settings.target_milli != self.settings.target_milli
            || settings.yield_per_million != self.settings.yield_per_million
        {
            self.yield_epoch = self.yield_epoch.wrapping_add(1);
            self.yield_check = settings
                .yield_per_million
                .zip(settings.target_milli)
                .map(|(rate, target)| Yield::new(rate, target, Duration::from_secs(3600)));
        }
        self.settings = settings;
        self.plan.borrow_mut().settings.audit = settings.audit;
    }

    fn admit(&mut self, job: StreamJob) {
        if let Some(check) = &mut self.yield_check {
            check.admit();
        }
        let probes = if self.settings.enabled {
            self.settings
                .stages
                .iter()
                .take_while(|&&s| s > 0 && s < job.params.num_sweeps)
                .count()
        } else {
            0
        };
        let id = job.job_id.clone();
        let (original, queued) = if probes == 0 {
            // Transfer ownership directly: pass-through jobs need no graph copy.
            (None, job)
        } else {
            let mut plan = self.plan.borrow_mut();
            plan.audit_rng.get_or_insert(if job.params.seed == 0 {
                1
            } else {
                job.params.seed
            });
            let same = self
                .topology
                .as_ref()
                .is_some_and(|(n, edges)| *n == job.graph.num_nodes() && *edges == job.graph.edges);
            if !same {
                *plan = StagePlan::new(self.settings);
                plan.audit_rng = Some(job.params.seed.max(1));
                self.topology = Some((job.graph.num_nodes(), job.graph.edges.clone()));
                self.topology_epoch = self.topology_epoch.wrapping_add(1);
            }
            let mut queued = copy_job(&job);
            queued.params.num_sweeps = plan.settings.stages[0];
            queued.params.seed = stage_seed(job.params.seed, 0);
            (Some(job), queued)
        };
        self.active.insert(
            id,
            Active {
                original,
                plan: Rc::clone(&self.plan),
                stage: 0,
                probes,
                audited: false,
                device_us: 0,
                topology_epoch: self.topology_epoch,
                yield_epoch: self.yield_epoch,
            },
        );
        self.ready.push_back(queued);
    }

    fn complete(&mut self, mut result: StreamResult, out: &Sender<StreamResult>) {
        let Some(mut active) = self.active.remove(&result.job_id) else {
            tracing::error!("cascade received an unassigned or duplicate result");
            return;
        };
        let plan_handle = Rc::clone(&active.plan);
        let mut plan = plan_handle.borrow_mut();
        active.device_us = active
            .device_us
            .saturating_add(result.device_access_time_us);
        if let (StreamOutcome::Completed(Ok(reads)), Some(original)) =
            (&result.outcome, &active.original)
        {
            if let Some(best) = reads.iter().map(|r| r.energy_milli).min() {
                let current_topology = active.topology_epoch == self.topology_epoch;
                if current_topology
                    && active.stage > 0
                    && plan.audit_supported(&active, active.stage - 1)
                {
                    let previous = active.stage - 1;
                    if active.audited {
                        if let Some(median) = kept_median(&plan.kept[previous]) {
                            let miss = best as f64 <= median;
                            let (observed, expected) = plan.audit_lanes[previous].observation(miss);
                            let action = plan.audit_lanes[previous].record(miss);
                            plan.apply_action(
                                previous,
                                action,
                                Check::Audit,
                                &[observed],
                                &[expected],
                            );
                        }
                    } else {
                        let kept = &mut plan.kept[previous];
                        if kept.len() == 256 {
                            kept.pop_front();
                        }
                        kept.push_back(best);
                    }
                }
                if active.stage < active.probes {
                    // Old-topology work can finish, but must not train the new
                    // distribution. Screen it out with its available reads.
                    let mut advance = false;
                    if current_topology {
                        let keep = plan.cutoffs[active.stage].decide(best as f64);
                        active.audited = !keep
                            && plan.audit_supported(&active, active.stage)
                            && plan.audit_selected(active.stage);
                        advance = keep || active.audited;
                        if active.stage == 0 && plan.settings.stages[0] == 32 {
                            let m = plan.cutoffs[0].moments().clone();
                            let observed = [m.skew(), m.excess_kurtosis()];
                            let action = plan.drift.check(&m);
                            plan.apply_action(
                                0,
                                action,
                                Check::Drift,
                                &observed,
                                &[CALIBRATION.skew[0], CALIBRATION.excess_kurtosis[0]],
                            );
                        }
                    }
                    if advance {
                        active.stage += 1;
                        let mut next = copy_job(original);
                        if active.stage < active.probes {
                            next.params.num_sweeps = plan.settings.stages[active.stage];
                            next.params.seed = stage_seed(next.params.seed, active.stage);
                        }
                        self.ready.push_front(next);
                        self.active.insert(result.job_id, active);
                        return;
                    }
                } else if current_topology {
                    // The last controller records full-budget observations,
                    // separate from all configured probe distributions.
                    let final_index = plan.cutoffs.len() - 1;
                    plan.cutoffs[final_index].observe(best as f64);
                }
            }
        }
        result.device_access_time_us = active.device_us;
        let best = match &result.outcome {
            StreamOutcome::Completed(Ok(reads)) => reads.iter().map(|r| r.energy_milli).min(),
            StreamOutcome::Completed(Err(_)) | StreamOutcome::Cancelled => None,
        };
        if out.blocking_send(result).is_ok() && active.yield_epoch == self.yield_epoch {
            if let (Some(check), Some(best)) = (&mut self.yield_check, best) {
                check.result(best);
            }
        }
    }

    fn check_yield(&mut self, now: Instant) {
        if let Some(check) = &mut self.yield_check {
            let (observed, expected) = check.observation();
            let action = check.tick(now);
            let mut plan = self.plan.borrow_mut();
            for stage in 0..plan.audit_lanes.len() {
                plan.apply_action(stage, action, Check::Yield, &[observed], &[expected]);
            }
        }
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
                    a.plan.borrow().cutoffs.len() - 1
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
        self.check_yield(Instant::now());
        let elapsed = self.window.elapsed().as_secs_f64();
        if elapsed < 1.0 {
            return;
        }
        let mut plan = self.plan.borrow_mut();
        let busy = self.cost_us / 1e6 / elapsed;
        let slack = (0.9 - busy).max(0.0) / 0.9;
        for stage in 0..plan.cutoffs.len() - 1 {
            // Count the next stage's jobs wherever they are: queued here, in the
            // inner channel, or on the GPU. The relay tracks at most 2 * width
            // jobs and the inner side absorbs about that many, so the ready queue
            // alone stays near empty. A next stage holding more than half a
            // stream width of the tracked jobs is falling behind.
            let at_next = self
                .active
                .values()
                .filter(|a| Rc::ptr_eq(&a.plan, &self.plan) && a.stage == stage + 1)
                .count();
            let reference = (self.width / 2).max(1);
            let backlog = (at_next as f64 / reference as f64).max(1.0).ln();
            let cutoff = &mut plan.cutoffs[stage];
            let error = if backlog == 0.0 && slack == 0.0 {
                -cutoff.log_adjust()
            } else {
                backlog - slack
            };
            cutoff.load_update(error, 0.1);
        }
        if self.logged.elapsed() >= Duration::from_secs(60) {
            for (stage, cutoff) in plan.cutoffs.iter().enumerate() {
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
        shared: &Mutex<CascadeSettings>,
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
                        self.refresh_settings(shared);
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
            if self.window.elapsed() >= Duration::from_secs(1) {
                self.refresh_settings(shared);
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
    settings: &std::sync::Mutex<CascadeSettings>,
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
                let initial = *settings.lock().unwrap_or_else(|p| p.into_inner());
                Relay::new(initial, width).run(settings, jobs, tx, result_rx, out, cancel)
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

    fn enabled_settings() -> CascadeSettings {
        CascadeSettings {
            enabled: true,
            ..CascadeSettings::default()
        }
    }

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
        run(
            &std::sync::Mutex::new(settings),
            rx,
            &out,
            cancel,
            4,
            |rx, tx| {
                assert_eq!(std::thread::current().id(), caller);
                inner(rx, tx);
            },
        );
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
    fn admitted_jobs_retain_their_plan_after_disable_and_stage_change() {
        let mut relay = Relay::new(enabled_settings(), 1);
        let shared = Mutex::new(enabled_settings());
        relay.admit(job(0, 14_336));
        for cutoff in &mut relay.plan.borrow_mut().cutoffs {
            for _ in 0..200 {
                cutoff.observe(0.0);
            }
        }
        let old_plan = Rc::downgrade(&relay.plan);
        {
            let mut settings = shared.lock().unwrap();
            settings.enabled = false;
            settings.stages = [64, 0, 0];
        }
        relay.refresh_settings(&shared);
        assert!(!Rc::ptr_eq(&relay.plan, &old_plan.upgrade().unwrap()));
        let (out, mut results) = mpsc::channel(2);
        let mut sweeps = Vec::new();
        while let Some(next) = relay.ready.pop_front() {
            sweeps.push(next.params.num_sweeps);
            relay.complete(with_energy(next, -1), &out);
        }
        let expected_sweeps = [CALIBRATION.stages, &[14_336]].concat();
        assert_eq!(sweeps, expected_sweeps);
        assert_eq!(
            results.try_recv().unwrap().device_access_time_us,
            expected_sweeps.iter().sum::<usize>() as u64
        );
        assert!(old_plan.upgrade().is_none());
        assert_eq!(relay.plan.borrow().cutoffs[0].moments().count(), 0);
        relay.admit(job(1, 1024));
        assert_eq!(relay.ready.front().unwrap().params.num_sweeps, 1024);
    }

    #[test]
    fn disabled_admission_moves_the_graph_without_a_clone() {
        let mut relay = Relay::new(CascadeSettings::default(), 1);
        let input = job(0, 1024);
        let allocation = input.graph.h.as_ptr();
        relay.admit(input);
        assert!(relay.active.values().next().unwrap().original.is_none());
        assert_eq!(relay.ready.front().unwrap().graph.h.as_ptr(), allocation);
        assert!(relay.topology.is_none());
    }

    #[test]
    fn live_keep_and_yield_changes_reset_only_their_controllers() {
        let settings = CascadeSettings {
            target_milli: Some(-100),
            yield_per_million: Some(1.0),
            ..enabled_settings()
        };
        let mut relay = Relay::new(settings, 1);
        let shared = Mutex::new(settings);
        relay.admit(job(0, 32));
        let old = relay.ready.pop_front().unwrap();
        relay.plan.borrow_mut().cutoffs[0].observe(1.0);
        shared.lock().unwrap().keep = 5000.0;
        relay.refresh_settings(&shared);
        assert_eq!(relay.plan.borrow().cutoffs[0].moments().count(), 0);
        let plan = Rc::clone(&relay.plan);
        {
            let mut settings = shared.lock().unwrap();
            settings.target_milli = Some(-200);
            settings.yield_per_million = Some(10_000_000.0);
        }
        relay.refresh_settings(&shared);
        assert!(Rc::ptr_eq(&plan, &relay.plan));
        let (out, mut results) = mpsc::channel(1);
        relay.complete(with_energy(old, -300), &out);
        results.try_recv().unwrap();
        assert_eq!(
            relay.yield_check.as_ref().unwrap().observation(),
            (0.0, 0.0)
        );
        relay.admit(job(1, 32));
        let next = relay.ready.pop_front().unwrap();
        relay.complete(with_energy(next, -150), &out);
        results.try_recv().unwrap();
        let (hits, bound) = relay.yield_check.as_ref().unwrap().observation();
        assert_eq!(hits, 0.0);
        assert!((bound - (10f64.sqrt() - 0.98).powi(2)).abs() < 1e-10);
    }

    #[test]
    fn live_settings_enable_disable_enable_and_change_stages_on_one_stream() {
        let shared = std::sync::Mutex::new(CascadeSettings::default());
        let (jobs, rx) = mpsc::channel(1);
        let (out, mut results) = mpsc::channel(1);
        let mut first_sweeps = Vec::new();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                for (id, enabled, stages) in [
                    (0, false, [32, 128, 256]),
                    (1, true, [32, 128, 256]),
                    (2, false, [32, 128, 256]),
                    (3, true, [32, 128, 256]),
                    (4, true, [128, 256, 0]),
                ] {
                    {
                        let mut settings = shared.lock().unwrap();
                        settings.enabled = enabled;
                        settings.stages = stages;
                    }
                    jobs.blocking_send(job(id, 1024)).unwrap();
                    let result: StreamResult = results.blocking_recv().unwrap();
                    assert_eq!(result.job_id, id.to_le_bytes());
                }
                drop(jobs);
            });
            run(
                &shared,
                rx,
                &out,
                &CancelToken::default(),
                1,
                |mut rx, tx| {
                    let mut last = None;
                    while let Some(job) = rx.blocking_recv() {
                        if last.as_ref() != Some(&job.job_id) {
                            first_sweeps.push(job.params.num_sweeps);
                            last = Some(job.job_id.clone());
                        }
                        tx.blocking_send(answer(job)).unwrap();
                    }
                },
            );
        });
        assert_eq!(first_sweeps, [1024, 32, 1024, 32, 128]);
    }

    #[test]
    fn custom_stages_use_transition_calibration_and_skip_32_sweep_drift() {
        let ((probe_budget, full_budget), miss_rate) = CALIBRATION.false_negative[1];
        let mut relay = Relay::new(
            CascadeSettings {
                enabled: true,
                stages: [probe_budget, 0, 0],
                ..enabled_settings()
            },
            1,
        );
        relay.admit(job(0, full_budget));
        let expected = AuditLane::new(miss_rate).observation(true);
        assert_eq!(
            relay.plan.borrow_mut().audit_lanes[0].observation(true),
            expected
        );
        for _ in 0..10_000 {
            relay.plan.borrow_mut().cutoffs[0].observe(0.0);
        }
        let mut unchanged = relay.plan.borrow_mut().cutoffs[0].clone();
        unchanged.observe(-1.0);
        let probe = relay.ready.pop_front().unwrap();
        let (out, _results) = mpsc::channel(1);
        relay.complete(with_energy(probe, -1), &out);
        assert_eq!(
            relay.plan.borrow_mut().cutoffs[0].denominator(),
            unchanged.denominator()
        );
    }

    #[test]
    fn short_full_budget_never_trains_another_transition() {
        let mut relay = Relay::new(
            CascadeSettings {
                enabled: true,
                ..enabled_settings()
            },
            1,
        );
        let (out, mut results) = mpsc::channel(1);
        for (id, budget) in [(0, 14_336), (1, 1024)] {
            relay.admit(job(id, budget));
            for cutoff in &mut relay.plan.borrow_mut().cutoffs {
                for _ in 0..200 {
                    cutoff.observe(0.0);
                }
            }
            while let Some(next) = relay.ready.pop_front() {
                relay.complete(with_energy(next, -1), &out);
            }
            results.try_recv().unwrap();
        }
        let last_probe = CALIBRATION.stages.len() - 1;
        assert_eq!(relay.plan.borrow_mut().kept[last_probe].len(), 1);
        // With cold cutoffs and an audit draw at every rejection, only the
        // calibrated transitions may advance a 1,024-sweep job.
        for cutoff in &mut relay.plan.borrow_mut().cutoffs {
            cutoff.reset();
        }
        relay.admit(job(2, 1024));
        while let Some(next) = relay.ready.pop_front() {
            assert_ne!(next.params.num_sweeps, 1024);
            relay.plan.borrow_mut().audit_rng = Some(0);
            relay.complete(with_energy(next, 0), &out);
        }
        assert_eq!(
            results.try_recv().unwrap().device_access_time_us,
            CALIBRATION.stages.iter().sum::<usize>() as u64
        );
    }

    #[test]
    fn audit_restore_preserves_drift_at_the_clamp() {
        let relay = Relay::new(enabled_settings(), 1);
        for _ in 0..100_000 {
            relay.plan.borrow_mut().cutoffs[0].observe(0.0);
        }
        relay.plan.borrow_mut().cutoffs[0].load_update(100.0, 1.0);
        relay
            .plan
            .borrow_mut()
            .apply_action(0, Action::LoosenAndDoubleK0, Check::Drift, &[], &[]);
        let before = relay.plan.borrow_mut().cutoffs[0].denominator();
        relay.plan.borrow_mut().apply_action(
            0,
            Action::Loosen { raise_audit: true },
            Check::Audit,
            &[],
            &[],
        );
        relay
            .plan
            .borrow_mut()
            .apply_action(0, Action::Restore, Check::Audit, &[], &[]);
        assert!((relay.plan.borrow_mut().cutoffs[0].denominator() - before).abs() < 1e-12);
    }

    #[test]
    fn yield_counts_forwarded_results_and_widens_each_probe() {
        for hit in [false, true] {
            let settings = CascadeSettings {
                target_milli: Some(-100),
                yield_per_million: Some(10_000_000.0),
                ..enabled_settings()
            };
            let mut relay = Relay::new(settings, 1);
            relay.admit(job(0, 32));
            let (out, mut results) = mpsc::channel(1);
            let full = relay.ready.pop_front().unwrap();
            relay.complete(with_energy(full, if hit { -100 } else { 0 }), &out);
            results.try_recv().unwrap();
            let (hits, bound) = relay.yield_check.as_ref().unwrap().observation();
            assert_eq!(hits, if hit { 1.0 } else { 0.0 });
            assert!((bound - (10f64.sqrt() - 0.98).powi(2)).abs() < 1e-10);
            relay.check_yield(Instant::now() + Duration::from_secs(3601));
            for lane in &relay.plan.borrow_mut().audit_lanes {
                assert_eq!(lane.denominator(200), 50);
            }
            assert_eq!(relay.yield_check.as_ref().unwrap().observation().0, 0.0);
        }
        let relay = Relay::new(
            CascadeSettings {
                target_milli: Some(-100),
                ..enabled_settings()
            },
            1,
        );
        assert!(relay.yield_check.is_none());
    }

    #[test]
    fn reference_median_and_zero_seed_are_well_defined() {
        assert_eq!(kept_median(&VecDeque::new()), None);
        assert_eq!(kept_median(&VecDeque::from([5, 1, 3])), Some(3.0));
        assert_eq!(kept_median(&VecDeque::from([5, 1, 3, 7])), Some(4.0));
        let mut relay = Relay::new(enabled_settings(), 1);
        let mut first = job(0, 1024);
        first.params.seed = 0;
        relay.admit(first);
        let selected = (0..10_000)
            .filter(|_| relay.plan.borrow_mut().audit_selected(0))
            .count();
        assert!((20..100).contains(&selected), "{selected}");
    }

    #[test]
    fn audit_returns_the_next_probe_when_that_probe_screens_it_out() {
        let ((_, full_budget), _) = CALIBRATION.false_negative[1];
        let mut relay = Relay::new(
            CascadeSettings {
                audit: 2,
                ..enabled_settings()
            },
            1,
        );
        let (out, mut results) = mpsc::channel(1);
        for id in 0..100 {
            relay.admit(job(id, full_budget));
            let probe = relay.ready.pop_front().unwrap();
            relay.complete(with_energy(probe, 0), &out);
            if let Some(next) = relay.ready.pop_front() {
                assert_eq!(next.params.num_sweeps, CALIBRATION.stages[1]);
                // Choose a non-audit draw for the next screening decision.
                relay.plan.borrow_mut().audit_rng = Some(1);
                relay.complete(with_energy(next, 123), &out);
                let result = results.try_recv().unwrap();
                let StreamOutcome::Completed(Ok(reads)) = result.outcome else {
                    panic!("expected reads")
                };
                assert_eq!(reads[0].energy_milli, 123);
                assert_eq!(
                    result.device_access_time_us,
                    CALIBRATION.stages.iter().sum::<usize>() as u64
                );
                assert!(relay.plan.borrow_mut().kept[0].is_empty());
                return;
            }
            results.try_recv().unwrap();
        }
        panic!("expected an audited job");
    }

    #[test]
    fn injected_false_negative_burst_loosens_the_stage() {
        let ((probe_budget, full_budget), _) = CALIBRATION.false_negative[0];
        let settings = CascadeSettings {
            stages: [probe_budget, 0, 0],
            keep: 100.0,
            keep_min: 2.0,
            keep_max: 1000.0,
            audit: 2,
            ..enabled_settings()
        };
        let mut relay = Relay::new(settings, 1);
        relay.admit(job(0, full_budget));
        for _ in 0..100_000 {
            relay.plan.borrow_mut().cutoffs[0].observe(-100.0);
        }
        let (out, mut results) = mpsc::channel(1);
        // Establish a reference from an actually kept job.
        let probe = relay.ready.pop_front().unwrap();
        relay.complete(with_energy(probe, -1000), &out);
        let full = relay.ready.pop_front().unwrap();
        relay.complete(with_energy(full, -1000), &out);
        results.try_recv().unwrap();
        let before = relay.plan.borrow_mut().cutoffs[0].denominator();
        for id in 1..1000 {
            relay.admit(job(id, full_budget));
            let probe = relay.ready.pop_front().unwrap();
            relay.complete(with_energy(probe, 0), &out);
            if let Some(full) = relay.ready.pop_front() {
                relay.complete(with_energy(full, -2000), &out);
                let result = results.try_recv().unwrap();
                let StreamOutcome::Completed(Ok(reads)) = result.outcome else {
                    panic!("expected reads")
                };
                assert_eq!(reads[0].energy_milli, -2000);
                assert_eq!(
                    result.device_access_time_us,
                    (probe_budget + full_budget) as u64
                );
            } else {
                results.try_recv().unwrap();
            }
            if relay.plan.borrow_mut().audit_lanes[0].denominator(200) == 50 {
                assert!(
                    (relay.plan.borrow_mut().cutoffs[0].denominator() - before / 2.0).abs() < 0.01
                );
                assert_eq!(relay.plan.borrow_mut().kept[0].len(), 1);
                return;
            }
        }
        panic!("audit burst did not loosen the screen");
    }

    fn with_energy(job: StreamJob, energy: i64) -> StreamResult {
        let mut result = answer(job);
        if let StreamOutcome::Completed(Ok(reads)) = &mut result.outcome {
            for read in reads {
                read.energy_milli = energy;
            }
        }
        result
    }

    #[test]
    fn every_job_gets_exactly_one_result() {
        let settings = CascadeSettings {
            stages: [32, 256, 0],
            ..enabled_settings()
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
            ..enabled_settings()
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
            enabled_settings(),
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
            ..enabled_settings()
        };
        let mut relay = Relay::new(settings, 1);
        relay.admit(job(0, 1024));
        for _ in 0..200 {
            relay.plan.borrow_mut().cutoffs[0].observe(0.0);
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
        let result: StreamResult = results.blocking_recv().unwrap();
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
            ..enabled_settings()
        };
        let cancel = CancelToken::default();
        let mut held_second_stage = None;
        let results = exercise(
            settings,
            (0..201).map(|i| job(i, 1024)).collect(),
            &cancel,
            |mut rx, tx| {
                while let Some(job) = rx.blocking_recv() {
                    let sweeps = job.params.num_sweeps;
                    let id = job.job_id.clone();
                    let mut reply = answer(job);
                    if sweeps == 128 {
                        // Hold this stage until cancellation. Its stage-0 result
                        // has already reached the relay, otherwise it cannot run.
                        held_second_stage = Some(id);
                        cancel.cancel_through(1);
                        reply.outcome = StreamOutcome::Cancelled;
                    } else if let StreamOutcome::Completed(Ok(reads)) = &mut reply.outcome {
                        // Equal energies permit normal keeps after warm-up.
                        // An audit can reach the second stage before then.
                        for read in reads {
                            read.energy_milli = -32;
                        }
                    }
                    tx.blocking_send(reply).unwrap();
                }
            },
        );
        let held_id = held_second_stage.unwrap();
        assert_eq!(results.len(), 201);
        let cancelled: Vec<_> = results.iter().filter(|r| r.job_id == held_id).collect();
        assert_eq!(cancelled.len(), 1);
        assert!(matches!(cancelled[0].outcome, StreamOutcome::Cancelled));
        assert_eq!(cancelled[0].device_access_time_us, 160);
    }

    #[test]
    fn inner_error_is_forwarded_once() {
        let results = exercise(
            enabled_settings(),
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
        let mut relay = Relay::new(enabled_settings(), 1);
        relay.admit(job(0, 1024));
        relay.plan.borrow_mut().cutoffs[0].observe(-1.0);
        relay.admit(job(1, 1024));
        assert_eq!(relay.plan.borrow_mut().cutoffs[0].moments().count(), 1);
        let mut changed = job(2, 1024);
        changed.graph = IsingGraph::new(vec![0.0; 2], vec![], vec![]);
        relay.admit(changed);
        assert_eq!(relay.plan.borrow_mut().cutoffs[0].moments().count(), 0);
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
        let settings = CascadeSettings {
            stages: [32, 128, 256],
            ..enabled_settings()
        };
        let mut relay = Relay::new(settings, 1);
        let expected_min = settings.keep_min.powf(1.0 / settings.stages.len() as f64);
        assert!((relay.plan.borrow_mut().cutoffs[0].denominator() - expected_min).abs() < 1e-10);
        let mut original = job(0, 128);
        original.params.beta_range = Some((0.5, 4.0));
        original.params.sweeps_per_beta = 2;
        relay.admit(original);
        assert_eq!(relay.active.values().next().unwrap().probes, 1);
        for _ in 0..200 {
            relay.plan.borrow_mut().cutoffs[0].observe(0.0);
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
        assert_eq!(results.try_recv().unwrap().device_access_time_us, 32 + 128);
        assert_eq!(relay.plan.borrow_mut().cutoffs[1].moments().count(), 0);
        assert_eq!(relay.plan.borrow_mut().cutoffs[2].moments().count(), 0);
        assert_eq!(relay.plan.borrow_mut().cutoffs[3].moments().count(), 1);
        assert!(relay.plan.borrow().kept.iter().all(VecDeque::is_empty));
    }

    #[test]
    fn late_results_do_not_train_a_new_topology() {
        let mut relay = Relay::new(enabled_settings(), 1);
        relay.admit(job(0, 1024));
        let old = relay.ready.pop_front().unwrap();
        let mut changed = job(1, 1024);
        changed.graph.h.push(0.0);
        relay.admit(changed);
        let (out, mut results) = mpsc::channel(1);
        relay.complete(answer(old), &out);
        assert_eq!(relay.plan.borrow_mut().cutoffs[0].moments().count(), 0);
        assert_eq!(results.try_recv().unwrap().job_id, 0u64.to_le_bytes());
    }

    #[test]
    fn full_inner_queue_preserves_ready_order_and_counts_only_dispatches() {
        let mut relay = Relay::new(enabled_settings(), 1);
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
            enabled_settings(),
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
        let width = 4;
        let mut relay = Relay::new(enabled_settings(), width);
        relay.window = Instant::now() - Duration::from_secs(1);
        relay.update_load();
        assert!(relay.plan.borrow_mut().cutoffs[0].log_adjust() < 0.0);
        relay.plan.borrow_mut().cutoffs[0].reset();
        // Stay inside the admission cap of 2 * width; three next-stage jobs
        // exceed the reference depth of width / 2.
        for id in 0..width as u64 {
            relay.admit(job(id, 1024));
        }
        assert!(relay.active.len() < 2 * width);
        for id in 0..3u64 {
            relay
                .active
                .get_mut(id.to_le_bytes().as_slice())
                .unwrap()
                .stage = 1;
        }
        relay.cost_us = 2e6;
        relay.window = Instant::now() - Duration::from_secs(1);
        relay.update_load();
        let positive = relay.plan.borrow_mut().cutoffs[0].log_adjust();
        assert!(positive > 0.0);
        for a in relay.active.values_mut() {
            a.stage = 0;
        }
        relay.cost_us = 2e6;
        relay.window = Instant::now() - Duration::from_secs(1);
        relay.update_load();
        assert!((relay.plan.borrow_mut().cutoffs[0].log_adjust() - 0.9 * positive).abs() < 1e-12);
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
