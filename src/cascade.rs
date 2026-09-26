// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Host-only decisions and schedules for the always-on MSA cascade.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::sampler::build_beta_schedule;
use crate::{IsingGraph, SampleParams};
use quip_solver_core::beta::{default_ising_beta_range, geometric_beta_schedule};
use quip_solver_core::StreamJob;

use crate::cutoff::{Cutoff, CutoffConfig};
use crate::model_checks::{Action, AuditLane, Drift, Yield};

pub(crate) const MAX_STAGES: usize = 4;

/// Arm D constants from docs/perf/2026-09-23-seeded-segments.md.
/// Times describe a 64-read job: a + b * sweeps (unchanged from G4).
pub(crate) struct Calibration {
    pub(crate) stages: &'static [usize],
    pub(crate) reheat_beta: f64,
    /// Audit-lane miss rate r0 for each probe-to-next-stage transition.
    pub(crate) false_negative: &'static [((usize, usize), f64)],
    pub(crate) skew: &'static [f64],
    pub(crate) excess_kurtosis: &'static [f64],
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "retained calibration checked by the Arm D regression test"
        )
    )]
    pub(crate) cost_a_us: f64,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "retained calibration checked by the Arm D regression test"
        )
    )]
    pub(crate) cost_b_us: f64,
    pub(crate) k0: f64,
}

pub(crate) const CALIBRATION: Calibration = Calibration {
    stages: &[32, 256],
    reheat_beta: 0.25,
    false_negative: &[((32, 256), 0.0353), ((256, 14_336), 0.0051)],
    skew: &[-0.123, -0.056],
    excess_kurtosis: &[0.014, 0.002],
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
    pub(crate) stages: [usize; MAX_STAGES],
    /// Fixed best-energy gate per checkpoint. A gated stage keeps a job when
    /// its best is at or below the gate and bypasses the adaptive cutoff.
    /// Only [`CascadeSettings::effective`] sets gates, for [`CHAIN_GATES`].
    pub(crate) gates: [Option<i64>; MAX_STAGES],
    /// Keep every job at every checkpoint. For gate measurement only.
    pub(crate) open_gates: bool,
    /// Overall probe-to-full denominator, factored across configured probes.
    pub(crate) keep: f64,
    pub(crate) keep_min: f64,
    pub(crate) keep_max: f64,
    pub(crate) audit: u32,
    pub(crate) reheat_beta: f64,
    pub(crate) target_milli: Option<i64>,
    pub(crate) yield_per_million: Option<f64>,
}

impl Default for CascadeSettings {
    fn default() -> Self {
        let mut stages = [0; MAX_STAGES];
        stages[..CALIBRATION.stages.len()].copy_from_slice(CALIBRATION.stages);
        Self {
            stages,
            gates: [None; MAX_STAGES],
            open_gates: false,
            keep: 2_000.0,
            keep_min: 1_000.0,
            keep_max: 30_000.0,
            audit: 200,
            reheat_beta: CALIBRATION.reheat_beta,
            target_milli: None,
            yield_per_million: None,
        }
    }
}

/// Gates for the chain topology `cbec1eb4` (4,577 nodes, 41,514 edges, h = 0,
/// J = ±1) on the resident schedule at 64 reads, one sweep per beta, and
/// reheat 0.25 (`tests/gate_trace.rs`). The 8 and 16 gates are the shallowest
/// checkpoint best over 30 seeds of the chain's 50 deepest winners. The 64 and
/// 256 gates are the shallowest over those runs and 30 seeds of 100 later
/// winners, plus 10. Population keep, cumulative: 1 in 7, 1 in 28, then about
/// none of 30,000.
pub(crate) struct ChainGates {
    pub(crate) fingerprint: u64,
    pub(crate) stages: [usize; MAX_STAGES],
    pub(crate) gates: [i64; MAX_STAGES],
    pub(crate) min_reads: usize,
    /// Final budget for a gated job, whatever the job asked for. Few models
    /// survive the last gate, so a deep final stage costs little.
    pub(crate) full_sweeps: usize,
}

pub(crate) const CHAIN_GATES: ChainGates = ChainGates {
    fingerprint: 0x38cd_e7d7_931d_f32f,
    stages: [8, 16, 64, 256],
    gates: [-13_244_000, -13_788_000, -14_254_000, -14_444_000],
    min_reads: 64,
    full_sweeps: 131_072,
};

/// FNV-1a over the node count and the sorted, normalized edge list, so edge
/// order and orientation do not change the result.
pub(crate) fn topology_fingerprint(nodes: usize, edges: &[(usize, usize)]) -> u64 {
    let mut sorted: Vec<(usize, usize)> =
        edges.iter().map(|&(u, v)| (u.min(v), u.max(v))).collect();
    sorted.sort_unstable();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let words = std::iter::once(nodes).chain(sorted.into_iter().flat_map(|(u, v)| [u, v]));
    for word in words {
        for byte in (word as u64).to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

impl CascadeSettings {
    /// True when a job on a chain-topology graph runs under the conditions
    /// the chain gates were measured at.
    pub(crate) fn chain_gated(&self, chain_topology: bool, params: &SampleParams) -> bool {
        chain_topology
            && self.reheat_beta.to_bits() == CALIBRATION.reheat_beta.to_bits()
            && params.num_reads >= CHAIN_GATES.min_reads
            && params.sweeps_per_beta == 1
            && params.beta_range.is_none()
    }

    /// The configured settings, or the chain-gated plan when `gated`.
    pub(crate) fn effective(self, gated: bool) -> Self {
        if gated {
            Self {
                stages: CHAIN_GATES.stages,
                gates: CHAIN_GATES.gates.map(Some),
                ..self
            }
        } else {
            self
        }
    }
}

#[derive(serde::Deserialize, Default, Clone)]
pub(crate) struct CascadeToml {
    pub(crate) cascade_stages: Option<Vec<usize>>,
    pub(crate) cascade_open_gates: Option<bool>,
    pub(crate) cascade_keep: Option<u32>,
    pub(crate) cascade_keep_min: Option<u32>,
    pub(crate) cascade_keep_max: Option<u32>,
    pub(crate) cascade_audit: Option<u32>,
    pub(crate) cascade_reheat_beta: Option<f64>,
    pub(crate) cascade_target_milli: Option<i64>,
    pub(crate) cascade_yield_per_million: Option<f64>,
}

impl CascadeSettings {
    /// Merge valid keys, retaining previous values for invalid keys. The three
    /// overall keep denominators are validated and accepted as one group.
    pub(crate) fn merge(&mut self, cfg: &CascadeToml) {
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
        if let Some(open) = cfg.cascade_open_gates {
            self.open_gates = open;
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
        if let Some(beta) = cfg.cascade_reheat_beta {
            self.reheat_beta = beta;
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

pub(crate) struct Ticket {
    pub(crate) gates: usize,
    pub(crate) stage: usize,
    pub(crate) audited: bool,
    pub(crate) topology_epoch: u64,
    pub(crate) yield_epoch: u64,
    // Preserve admitted transitions and controllers across settings changes.
    plan: Arc<Mutex<StagePlan>>,
    final_sweeps: usize,
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

pub(crate) struct Controller {
    settings: CascadeSettings,
    plan: Arc<Mutex<StagePlan>>,
    yield_check: Option<Yield>,
    topology: Option<(usize, Vec<(usize, usize)>)>,
    topology_epoch: u64,
    yield_epoch: u64,
    stats: Stats,
}

/// Best energies seen at one sweep level during a report window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LevelStats {
    pub(crate) sweeps: usize,
    pub(crate) checked: u64,
    pub(crate) kept: u64,
    pub(crate) best: i64,
    pub(crate) worst: i64,
}

/// Per-window counts for the periodic cascade report. Level `k < MAX_STAGES`
/// is checkpoint `k`; the last level is the full budget.
pub(crate) struct Stats {
    started: Instant,
    pub(crate) models: u64,
    pub(crate) cancelled: u64,
    pub(crate) levels: [Option<LevelStats>; MAX_STAGES + 1],
}

impl Stats {
    fn new(now: Instant) -> Self {
        Self {
            started: now,
            models: 0,
            cancelled: 0,
            levels: [None; MAX_STAGES + 1],
        }
    }

    fn record(&mut self, level: usize, sweeps: usize, best: i64, kept: bool) {
        let entry = self.levels[level].get_or_insert(LevelStats {
            sweeps,
            checked: 0,
            kept: 0,
            best,
            worst: best,
        });
        // A settings change mid-window can move a level's budget.
        entry.sweeps = sweeps;
        entry.checked += 1;
        entry.kept += u64::from(kept);
        entry.best = entry.best.min(best);
        entry.worst = entry.worst.max(best);
    }
}

/// Milli energy as a whole-unit string; chain energies are unit integers.
fn energy(milli: i64) -> String {
    if milli % 1000 == 0 {
        (milli / 1000).to_string()
    } else {
        format!("{:.3}", milli as f64 / 1000.0)
    }
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

    fn audit_supported(&mut self, ticket: &Ticket, stage: usize) -> bool {
        let from = self.settings.stages[stage];
        let to = if stage + 1 < ticket.gates {
            self.settings.stages[stage + 1]
        } else {
            ticket.final_sweeps
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

impl Controller {
    #[cfg(test)]
    pub(crate) fn keep_denominators(&self) -> Vec<f64> {
        self.plan
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .cutoffs
            .iter()
            .map(Cutoff::denominator)
            .collect()
    }

    pub(crate) fn new(settings: CascadeSettings) -> Self {
        Self {
            settings,
            plan: Arc::new(Mutex::new(StagePlan::new(settings))),
            yield_check: settings
                .yield_per_million
                .zip(settings.target_milli)
                .map(|(rate, target)| Yield::new(rate, target, Duration::from_secs(3600))),
            topology: None,
            topology_epoch: 0,
            yield_epoch: 0,
            stats: Stats::new(Instant::now()),
        }
    }

    /// Log the window's counts and energy ranges when `period` has passed,
    /// then start a new window.
    pub(crate) fn report(&mut self, now: Instant, period: Duration) {
        let elapsed = now.saturating_duration_since(self.stats.started);
        if elapsed < period {
            return;
        }
        let stats = std::mem::replace(&mut self.stats, Stats::new(now));
        let secs = elapsed.as_secs_f64().max(f64::EPSILON);
        let levels: Vec<String> = stats
            .levels
            .iter()
            .flatten()
            .map(|l| {
                format!(
                    "{}sw checked={} kept={} best={} worst={}",
                    l.sweeps,
                    l.checked,
                    l.kept,
                    energy(l.best),
                    energy(l.worst)
                )
            })
            .collect();
        tracing::info!(
            window_s = secs.round() as u64,
            models = stats.models,
            models_per_s = (stats.models as f64 / secs).round() as u64,
            cancelled = stats.cancelled,
            levels = %levels.join(" | "),
            "cascade report"
        );
    }

    /// Rebuild on a stages/keep change, update audit and yield independently.
    pub(crate) fn refresh(&mut self, settings: CascadeSettings) {
        if settings.stages != self.settings.stages
            || settings.keep != self.settings.keep
            || settings.keep_min != self.settings.keep_min
            || settings.keep_max != self.settings.keep_max
        {
            self.plan = Arc::new(Mutex::new(StagePlan::new(settings)));
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
        let mut plan = self.plan.lock().unwrap_or_else(|p| p.into_inner());
        plan.settings.audit = settings.audit;
        plan.settings.open_gates = settings.open_gates;
    }

    pub(crate) fn matches_topology(&self, graph: &IsingGraph) -> bool {
        self.topology
            .as_ref()
            .is_some_and(|(n, edges)| *n == graph.num_nodes() && *edges == graph.edges)
    }

    /// Reset gate state on a gated topology change. Return the job's schedule plan.
    #[cfg(test)]
    pub(crate) fn admit(&mut self, job: &StreamJob) -> (Ticket, Vec<f32>, Vec<usize>) {
        let chain = topology_fingerprint(job.graph.num_nodes(), &job.graph.edges)
            == CHAIN_GATES.fingerprint;
        let gated = self.settings.chain_gated(chain, &job.params);
        let mut schedule = PreparedSchedule::new(job, self.settings, gated);
        let ticket = self.admit_prepared(job, &mut schedule).unwrap();
        (ticket, schedule.betas.to_vec(), schedule.checkpoints)
    }

    pub(crate) fn admit_prepared(
        &mut self,
        job: &StreamJob,
        schedule: &mut PreparedSchedule,
    ) -> Result<Ticket, crate::sampler::SampleError> {
        // Only settings changes rebuild on the runner. Queued jobs must use
        // the current gate plan.
        if schedule.settings.stages != self.settings.stages
            || schedule.settings.reheat_beta.to_bits() != self.settings.reheat_beta.to_bits()
        {
            let rebuilt = PreparedSchedule::new(job, self.settings, schedule.gated);
            crate::slots::validate_schedule(&rebuilt.betas, &rebuilt.checkpoints)?;
            *schedule = rebuilt;
        }
        if schedule.checkpoints.len() > 1 {
            let wanted = self.settings.effective(schedule.gated);
            if !self.matches_topology(&job.graph) {
                self.plan = Arc::new(Mutex::new(StagePlan::new(wanted)));
                self.topology = Some((job.graph.num_nodes(), job.graph.edges.clone()));
                self.topology_epoch = self.topology_epoch.wrapping_add(1);
            } else {
                let plan = self.plan.lock().unwrap_or_else(|p| p.into_inner());
                let stale =
                    plan.settings.stages != wanted.stages || plan.settings.gates != wanted.gates;
                drop(plan);
                if stale {
                    self.plan = Arc::new(Mutex::new(StagePlan::new(wanted)));
                }
            }
            self.plan
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .audit_rng
                .get_or_insert(job.params.seed.max(1));
        }
        if let Some(check) = &mut self.yield_check {
            check.admit();
        }
        Ok(Ticket {
            gates: schedule.checkpoints.len() - 1,
            stage: 0,
            audited: false,
            topology_epoch: self.topology_epoch,
            yield_epoch: self.yield_epoch,
            plan: Arc::clone(&self.plan),
            final_sweeps: job.params.num_sweeps,
        })
    }

    /// Called at a non-last checkpoint. true = keep running.
    pub(crate) fn checkpoint(&mut self, ticket: &mut Ticket, best: i64) -> bool {
        if ticket.topology_epoch != self.topology_epoch {
            return false;
        }
        let mut plan = ticket.plan.lock().unwrap_or_else(|p| p.into_inner());
        Self::observe_transition(&mut plan, ticket, best);
        let stage = ticket.stage;
        let sweeps = plan.settings.stages[stage];
        if plan.settings.open_gates {
            self.stats.record(stage, sweeps, best, true);
            ticket.stage += 1;
            return true;
        }
        if let Some(gate) = plan.settings.gates[stage] {
            let keep = best <= gate;
            self.stats.record(stage, sweeps, best, keep);
            ticket.audited = false;
            ticket.stage += usize::from(keep);
            return keep;
        }
        let keep = plan.cutoffs[stage].decide(best as f64);
        ticket.audited = !keep && plan.audit_supported(ticket, stage) && plan.audit_selected(stage);
        if stage == 0 && plan.settings.stages[0] == 32 {
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
        self.stats
            .record(stage, sweeps, best, keep || ticket.audited);
        if keep || ticket.audited {
            ticket.stage += 1;
            true
        } else {
            false
        }
    }

    fn observe_transition(plan: &mut StagePlan, ticket: &Ticket, best: i64) {
        if ticket.stage == 0 || !plan.audit_supported(ticket, ticket.stage - 1) {
            return;
        }
        let previous = ticket.stage - 1;
        if ticket.audited {
            if let Some(median) = kept_median(&plan.kept[previous]) {
                let miss = best as f64 <= median;
                let (observed, expected) = plan.audit_lanes[previous].observation(miss);
                let action = plan.audit_lanes[previous].record(miss);
                plan.apply_action(previous, action, Check::Audit, &[observed], &[expected]);
            }
        } else {
            let kept = &mut plan.kept[previous];
            if kept.len() == 256 {
                kept.pop_front();
            }
            kept.push_back(best);
        }
    }

    /// Called once per job with its final best (None for error or cancel).
    pub(crate) fn finish(&mut self, ticket: &Ticket, best: Option<i64>, delivered: bool) {
        match best {
            Some(best) => {
                self.stats.models += 1;
                if ticket.stage == ticket.gates {
                    self.stats
                        .record(MAX_STAGES, ticket.final_sweeps, best, true);
                }
            }
            None => self.stats.cancelled += 1,
        }
        if let Some(best) = best {
            if ticket.topology_epoch == self.topology_epoch
                && ticket.stage == ticket.gates
                && ticket.gates > 0
            {
                let mut plan = ticket.plan.lock().unwrap_or_else(|p| p.into_inner());
                Self::observe_transition(&mut plan, ticket, best);
                let final_index = plan.cutoffs.len() - 1;
                plan.cutoffs[final_index].observe(best as f64);
            }
            if delivered && ticket.yield_epoch == self.yield_epoch {
                if let Some(check) = &mut self.yield_check {
                    check.result(best);
                }
            }
        }
    }

    fn check_yield(&mut self, now: Instant) {
        if let Some(check) = &mut self.yield_check {
            let (observed, expected) = check.observation();
            let action = check.tick(now);
            let mut plan = self.plan.lock().unwrap_or_else(|p| p.into_inner());
            for stage in 0..plan.audit_lanes.len() {
                plan.apply_action(stage, action, Check::Yield, &[observed], &[expected]);
            }
        }
    }

    /// live[k] = live slots whose next checkpoint is k; capacity = total slots.
    pub(crate) fn update_load(&mut self, busy: f64, live: &[usize], capacity: usize) {
        self.check_yield(Instant::now());
        let slack = (0.9 - busy).max(0.0) / 0.9;
        let mut plan = self.plan.lock().unwrap_or_else(|p| p.into_inner());
        for stage in 0..plan.cutoffs.len() - 1 {
            let at_next = live.get(stage + 1).copied().unwrap_or(0);
            let reference = (capacity / 2).max(1);
            let backlog = (at_next as f64 / reference as f64).max(1.0).ln();
            let cutoff = &mut plan.cutoffs[stage];
            let error = if backlog == 0.0 && slack == 0.0 {
                -cutoff.log_adjust()
            } else {
                backlog - slack
            };
            cutoff.load_update(error, 0.1);
        }
    }
}

pub(crate) struct PreparedSchedule {
    settings: CascadeSettings,
    gated: bool,
    pub(crate) betas: Arc<[f32]>,
    pub(crate) checkpoints: Vec<usize>,
}

/// Everything a segment schedule depends on. The graph enters only through
/// its beta range.
#[derive(Clone, Copy, PartialEq)]
struct ScheduleKey {
    hot: u64,
    cold: u64,
    sweeps: usize,
    per_beta: usize,
    stages: [usize; MAX_STAGES],
    reheat: u64,
}

/// The last validated schedule on one preparation worker. Jobs on one lease
/// share it, so later jobs skip the build, the validation, and, through the
/// shared `Arc`, the slot upload.
#[derive(Default)]
pub(crate) struct ScheduleCache {
    entry: Option<(ScheduleKey, Arc<[f32]>, Vec<usize>)>,
}

impl ScheduleCache {
    pub(crate) fn prepare(
        &mut self,
        job: &StreamJob,
        settings: CascadeSettings,
        gated: bool,
    ) -> Result<PreparedSchedule, crate::sampler::SampleError> {
        let (hot, cold) = job
            .params
            .beta_range
            .unwrap_or_else(|| resident_beta_range(&job.graph));
        let key = ScheduleKey {
            hot: hot.to_bits(),
            cold: cold.to_bits(),
            sweeps: job.params.num_sweeps,
            per_beta: job.params.sweeps_per_beta,
            stages: settings.effective(gated).stages,
            reheat: settings.reheat_beta.to_bits(),
        };
        let entry = match self.entry.take() {
            Some(entry) if entry.0 == key => entry,
            _ => {
                let params = SampleParams {
                    beta_range: Some((hot, cold)),
                    ..job.params.clone()
                };
                let (betas, checkpoints) =
                    segment_schedule(&job.graph, &params, &key.stages, settings.reheat_beta);
                crate::slots::validate_schedule(&betas, &checkpoints)?;
                (key, betas.into(), checkpoints)
            }
        };
        let schedule = PreparedSchedule {
            settings,
            gated,
            betas: Arc::clone(&entry.1),
            checkpoints: entry.2.clone(),
        };
        self.entry = Some(entry);
        Ok(schedule)
    }
}

impl PreparedSchedule {
    /// `settings` are the configured settings. `gated` selects the chain plan.
    pub(crate) fn new(job: &StreamJob, settings: CascadeSettings, gated: bool) -> Self {
        let (betas, checkpoints) = segment_schedule(
            &job.graph,
            &job.params,
            &settings.effective(gated).stages,
            settings.reheat_beta,
        );
        Self {
            settings,
            gated,
            betas: betas.into(),
            checkpoints,
        }
    }
}

// Unit-coupling draws have minimum gap one at every nonzero effective field.
// Count incident terms instead of tracking Option<f64> minima per endpoint.
// Keep the library path for every other coefficient domain.
fn resident_beta_range(graph: &IsingGraph) -> (f64, f64) {
    if graph.j.len() != graph.edges.len()
        || graph.j.iter().any(|j| j.abs() != 1.0)
        || graph.h.iter().any(|&h| h != 0.0 && h.abs() != 1.0)
    {
        return default_ising_beta_range(graph);
    }
    let n = graph.num_nodes();
    let mut terms: Vec<usize> = graph.h.iter().map(|&h| usize::from(h != 0.0)).collect();
    for &(u, v) in &graph.edges {
        if u < n && v < n {
            terms[u] += 1;
            terms[v] += 1;
        }
    }
    let max_eff = terms.iter().copied().max().unwrap_or(0) as f64;
    if max_eff == 0.0 {
        return (0.1, 1.0);
    }
    let hot = std::f64::consts::LN_2 / (2.0 * max_eff);
    let gaps = terms.iter().filter(|&&count| count != 0).count() as f64;
    let cold = (gaps / 0.01).ln() / 2.0;
    (hot, cold.max(hot))
}

pub(crate) fn segment_schedule(
    graph: &IsingGraph,
    params: &SampleParams,
    stages: &[usize],
    reheat_beta: f64,
) -> (Vec<f32>, Vec<usize>) {
    let mut checkpoints: Vec<_> = stages
        .iter()
        .copied()
        .take_while(|&s| s > 0 && s < params.num_sweeps)
        .collect();
    checkpoints.push(params.num_sweeps);
    let (hot, cold) = params
        .beta_range
        .unwrap_or_else(|| resident_beta_range(graph));
    let valid_reheat = reheat_beta > hot && reheat_beta < cold;
    if !valid_reheat {
        tracing::warn!(
            reheat_beta,
            hot,
            cold,
            "cascade reheat beta outside graph range; using standard schedule"
        );
    }
    let first = if valid_reheat {
        checkpoints[0]
    } else {
        params.num_sweeps
    };
    let (standard, repeats) =
        build_beta_schedule(graph, first, params.sweeps_per_beta, Some((hot, cold)));
    let mut schedule = Vec::with_capacity(params.num_sweeps);
    for beta in &standard {
        schedule.extend(std::iter::repeat_n(
            *beta,
            repeats.min(first - schedule.len()),
        ));
    }
    // A non-divisible sweep count holds the final standard rung for the remainder.
    schedule.resize(first, standard.last().copied().unwrap_or(cold as f32));
    if valid_reheat {
        for pair in checkpoints.windows(2) {
            let len = pair[1] - pair[0];
            schedule.extend(
                geometric_beta_schedule(reheat_beta, cold, len)
                    .into_iter()
                    .map(|b| b as f32),
            );
        }
    }
    (schedule, checkpoints)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_beta_range_matches_library_bits_on_draws() {
        let edges: Vec<(usize, usize)> = include_str!("../tests/fixtures/advantage2-system1.edges")
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
            .map(|line| {
                let mut nodes = line.split_whitespace();
                (
                    nodes.next().unwrap().parse().unwrap(),
                    nodes.next().unwrap().parse().unwrap(),
                )
            })
            .collect();
        let mut rng = 12345u64;
        let mut draw = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for seed in 0..512 {
            let mut graph = IsingGraph::new(
                (0..4577)
                    .map(|_| [-1.0, 0.0, 1.0][(draw() % 3) as usize])
                    .collect(),
                (0..edges.len())
                    .map(|_| if draw() & 1 == 0 { -1.0 } else { 1.0 })
                    .collect(),
                edges.clone(),
            );
            for nonunit in [false, true] {
                if nonunit {
                    graph.j[seed] = [0.0, 0.5, -2.0, 127.0][seed % 4];
                }
                let actual = resident_beta_range(&graph);
                let expected = default_ising_beta_range(&graph);
                assert_eq!(
                    (actual.0.to_bits(), actual.1.to_bits()),
                    (expected.0.to_bits(), expected.1.to_bits())
                );
            }
        }
        for graph in [
            IsingGraph::new(vec![], vec![], vec![]),
            IsingGraph::new(vec![0.0; 3], vec![], vec![]),
            IsingGraph::new(vec![0.0, -1.0, 0.0], vec![1.0, -1.0], vec![(0, 0), (0, 8)]),
            IsingGraph::new(vec![f64::NAN], vec![], vec![]),
        ] {
            let actual = resident_beta_range(&graph);
            let expected = default_ising_beta_range(&graph);
            assert_eq!(
                (actual.0.to_bits(), actual.1.to_bits()),
                (expected.0.to_bits(), expected.1.to_bits())
            );
        }
    }

    #[test]
    fn settings_change_rejects_invalid_rebuilt_schedule() {
        let old = CascadeSettings {
            stages: [2, 0, 0, 0],
            reheat_beta: -1.5,
            ..Default::default()
        };
        let mut params = params(4, 1);
        params.beta_range = Some((-2.0, -1.0));
        let job = StreamJob {
            job_id: vec![1],
            graph: ring(),
            params,
            watermark: None,
        };
        let mut schedule = PreparedSchedule::new(&job, old, false);
        crate::slots::validate_schedule(&schedule.betas, &schedule.checkpoints).unwrap();
        let mut controller = Controller::new(old);
        controller.refresh(CascadeSettings {
            stages: [1, 0, 0, 0],
            ..old
        });
        assert!(matches!(
            controller.admit_prepared(&job, &mut schedule),
            Err(crate::sampler::SampleError::Driver(_))
        ));
        assert!(
            controller.topology.is_none(),
            "rejection must precede ticket side effects"
        );
    }

    #[test]
    fn segment_schedule_preserves_legacy_bits() {
        let graph = ring();
        for range in [None, Some((0.1, 6.0))] {
            for sweeps in [1, 31, 32, 33, 1000] {
                for repeats in [1, 4, 64] {
                    let mut params = params(sweeps, repeats);
                    params.beta_range = range;
                    for reheat in [0.25, 10.0] {
                        let (actual, checkpoints) =
                            segment_schedule(&graph, &params, &[32, 256], reheat);
                        let (hot, cold) = range.unwrap_or_else(|| default_ising_beta_range(&graph));
                        let reheated = reheat > hot && reheat < cold;
                        let first = if reheated { checkpoints[0] } else { sweeps };
                        // Legacy construction passes the original optional range,
                        // including a second default-range computation for None.
                        let (standard, repeat) = build_beta_schedule(&graph, first, repeats, range);
                        let mut expected: Vec<_> = standard
                            .iter()
                            .flat_map(|b| std::iter::repeat_n(*b, repeat))
                            .take(first)
                            .collect();
                        expected.resize(first, standard.last().copied().unwrap_or(cold as f32));
                        if reheated {
                            for pair in checkpoints.windows(2) {
                                expected.extend(
                                    geometric_beta_schedule(reheat, cold, pair[1] - pair[0])
                                        .into_iter()
                                        .map(|b| b as f32),
                                );
                            }
                        }
                        assert_eq!(
                            actual.iter().map(|b| b.to_bits()).collect::<Vec<_>>(),
                            expected.iter().map(|b| b.to_bits()).collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn prepared_schedule_refreshes_before_admission() {
        let job = StreamJob {
            job_id: vec![1],
            graph: ring(),
            params: params(1000, 1),
            watermark: None,
        };
        let old = CascadeSettings {
            stages: [32, 256, 0, 0],
            ..Default::default()
        };
        for new in [
            CascadeSettings {
                stages: [16, 128, 512, 0],
                ..old
            },
            CascadeSettings {
                reheat_beta: 0.5,
                ..old
            },
        ] {
            let mut prepared = PreparedSchedule::new(&job, old, false);
            let mut controller = Controller::new(old);
            controller.refresh(new);
            let ticket = controller.admit_prepared(&job, &mut prepared).unwrap();
            let mut expected = Controller::new(new);
            let (expected_ticket, betas, checkpoints) = expected.admit(&job);
            assert_eq!(&prepared.betas[..], &betas[..]);
            assert_eq!(prepared.checkpoints, checkpoints);
            assert_eq!(ticket.gates, expected_ticket.gates);
            assert_eq!(ticket.stage, expected_ticket.stage);
            assert_eq!(ticket.final_sweeps, expected_ticket.final_sweeps);
        }
    }
    use crate::sampler::build_beta_schedule;
    use crate::{IsingGraph, SampleParams};
    use quip_solver_core::beta::default_ising_beta_range;

    fn ring() -> IsingGraph {
        IsingGraph::new(
            vec![0.0; 4],
            vec![1.0; 4],
            vec![(0, 1), (1, 2), (2, 3), (0, 3)],
        )
    }
    fn params(num_sweeps: usize, sweeps_per_beta: usize) -> SampleParams {
        SampleParams {
            num_reads: 2,
            num_sweeps,
            sweeps_per_beta,
            beta_range: None,
            seed: 42,
        }
    }
    fn job(id: u64, sweeps: usize) -> StreamJob {
        StreamJob {
            job_id: id.to_le_bytes().to_vec(),
            graph: ring(),
            params: params(sweeps, 1),
            watermark: Some(1),
        }
    }

    #[test]
    fn short_budget_has_one_segment() {
        let (s, c) = segment_schedule(&ring(), &params(32, 1), &[32, 256], 0.25);
        assert_eq!(s.len(), 32);
        assert_eq!(c, vec![32]);
    }

    #[test]
    fn sweeps_per_beta_expands_rungs() {
        let (s, c) = segment_schedule(&ring(), &params(1000, 4), &[32, 256], 0.25);
        assert_eq!(s.len(), 1000);
        assert_eq!(c, vec![32, 256, 1000]);
        let standard = build_beta_schedule(&ring(), 32, 4, None).0;
        for (chunk, beta) in s[..32].chunks(4).zip(standard) {
            assert_eq!(chunk, &[beta; 4]);
        }
    }

    #[test]
    fn every_segment_ends_cold_and_tails_start_at_the_reheat_beta() {
        let (s, _) = segment_schedule(&ring(), &params(1000, 1), &[32, 256], 0.25);
        let cold = default_ising_beta_range(&ring()).1 as f32;
        for i in [31, 255, 999] {
            assert_eq!(s[i], cold);
        }
        for i in [32, 256] {
            assert_eq!(s[i], 0.25);
        }
    }

    #[test]
    fn prefix_equals_the_standard_probe_schedule() {
        let (s, _) = segment_schedule(&ring(), &params(1000, 1), &[32, 256], 0.25);
        assert_eq!(s[..32], build_beta_schedule(&ring(), 32, 1, None).0);
    }

    #[test]
    fn invalid_reheat_uses_one_standard_schedule_with_the_same_checkpoints() {
        let (hot, cold) = default_ising_beta_range(&ring());
        for reheat in [hot, cold, -1.0, f64::NAN, f64::INFINITY] {
            let (s, c) = segment_schedule(&ring(), &params(1000, 4), &[32, 256], reheat);
            let expected: Vec<_> = build_beta_schedule(&ring(), 1000, 4, None)
                .0
                .into_iter()
                .flat_map(|b| [b; 4])
                .collect();
            assert_eq!(s, expected);
            assert_eq!(c, vec![32, 256, 1000]);
        }
    }

    #[test]
    fn warm_up_screens_out_and_the_keep_rate_converges() {
        let settings = CascadeSettings {
            keep: 100.0,
            keep_min: 100.0,
            keep_max: 100.0,
            ..CascadeSettings::default()
        };
        let mut controller = Controller::new(settings);
        let template = job(0, 1000);
        let mut rng = 12345u64;
        let mut uniform = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut counts = [[0usize; 2]; 2];
        for i in 0..50_000 {
            let (mut ticket, _, _) = controller.admit(&template);
            // Disable audit draws to measure only the cutoff keep rate.
            controller.plan.lock().unwrap().audit_rng = None;
            for count in &mut counts {
                let best = (1000.0
                    * (-2.0 * uniform().ln()).sqrt()
                    * (std::f64::consts::TAU * uniform()).cos()) as i64;
                let keep = controller.checkpoint(&mut ticket, best);
                if i == 0 {
                    assert!(!keep);
                }
                if i >= 10_000 {
                    count[0] += 1;
                    count[1] += usize::from(keep);
                }
                if !keep {
                    break;
                }
            }
        }
        for [total, kept] in counts {
            let rate = kept as f64 / total as f64;
            assert!((rate - 0.1).abs() < 0.025, "{rate}");
        }
    }

    #[test]
    fn audited_job_records_one_observation_at_the_next_checkpoint() {
        let mut c = Controller::new(CascadeSettings::default());
        let (mut t, _, _) = c.admit(&job(0, 14_336));
        {
            let mut p = c.plan.lock().unwrap();
            p.kept[0].push_back(-100);
            p.audit_rng = Some(0);
        }
        assert!(c.checkpoint(&mut t, 0));
        assert!(t.audited);
        c.plan.lock().unwrap().audit_rng = None;
        assert!(!c.checkpoint(&mut t, -200));
        let observation = c.plan.lock().unwrap().audit_lanes[0].observation(false);
        let mut expected = AuditLane::new(CALIBRATION.false_negative[0].1);
        expected.record(true);
        assert_eq!(observation, expected.observation(false));
        c.finish(&t, Some(-200), true);
        assert_eq!(
            c.plan.lock().unwrap().audit_lanes[0].observation(false),
            observation
        );
        assert_eq!(c.plan.lock().unwrap().kept[0].len(), 1);
    }

    #[test]
    fn topology_change_resets_the_distribution() {
        let mut c = Controller::new(CascadeSettings::default());
        let (mut old, _, _) = c.admit(&job(0, 1000));
        c.checkpoint(&mut old, 0);
        assert!(c.matches_topology(&ring()));
        let mut same_topology = job(1, 1000);
        same_topology.graph.h[0] = 0.5;
        c.admit(&same_topology);
        assert_eq!(c.plan.lock().unwrap().cutoffs[0].moments().count(), 1);
        let mut changed = job(1, 1000);
        changed.graph.h.push(0.0);
        c.admit(&changed);
        assert!(!c.matches_topology(&ring()));
        assert!(!c.checkpoint(&mut old, -100));
        c.finish(&old, Some(-100), true);
        assert_eq!(c.plan.lock().unwrap().cutoffs[0].moments().count(), 0);
    }

    #[test]
    fn settings_change_to_stages_rebuilds_but_audit_change_does_not() {
        let mut c = Controller::new(CascadeSettings::default());
        let (mut t, _, _) = c.admit(&job(0, 1000));
        c.checkpoint(&mut t, 0);
        c.refresh(CascadeSettings {
            audit: 50,
            ..CascadeSettings::default()
        });
        assert_eq!(c.plan.lock().unwrap().cutoffs[0].moments().count(), 1);
        assert_eq!(c.plan.lock().unwrap().settings.audit, 50);
        c.refresh(CascadeSettings {
            stages: [16, 128, 0, 0],
            ..CascadeSettings::default()
        });
        assert_eq!(c.plan.lock().unwrap().cutoffs[0].moments().count(), 0);
    }

    #[test]
    fn controller_and_ticket_are_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Controller>();
        assert_send::<Ticket>();
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
        for stages in [vec![], vec![0], vec![32, 32], vec![1, 2, 3, 4, 5]] {
            settings.merge(&CascadeToml {
                cascade_stages: Some(stages),
                ..CascadeToml::default()
            });
            assert_eq!(settings, previous);
        }
    }

    #[test]
    fn fixed_gates_keep_at_or_below_the_gate_and_skip_audits() {
        let mut c = Controller::new(CascadeSettings {
            stages: [8, 16, 0, 0],
            gates: [Some(-10), Some(-20), None, None],
            audit: 2,
            ..CascadeSettings::default()
        });
        let (mut kept, _, checkpoints) = c.admit(&job(0, 100));
        assert_eq!(checkpoints, vec![8, 16, 100]);
        assert!(c.checkpoint(&mut kept, -10));
        assert!(c.checkpoint(&mut kept, -20));
        assert_eq!(kept.stage, 2);
        for _ in 0..50 {
            let (mut screened, _, _) = c.admit(&job(0, 100));
            assert!(!c.checkpoint(&mut screened, -9));
            assert!(!screened.audited);
            assert_eq!(screened.stage, 0);
        }
    }

    fn fixture_edges() -> Vec<(usize, usize)> {
        include_str!("../tests/fixtures/advantage2-system1.edges")
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
            .map(|line| {
                let mut nodes = line.split_whitespace();
                (
                    nodes.next().unwrap().parse().unwrap(),
                    nodes.next().unwrap().parse().unwrap(),
                )
            })
            .collect()
    }

    #[test]
    fn chain_fingerprint_is_the_fixture_without_its_removed_edge() {
        let mut edges = fixture_edges();
        assert_ne!(topology_fingerprint(4577, &edges), CHAIN_GATES.fingerprint);
        edges.retain(|&(u, v)| (u.min(v), u.max(v)) != (880, 2695));
        assert_eq!(edges.len(), 41_514);
        assert_eq!(topology_fingerprint(4577, &edges), CHAIN_GATES.fingerprint);
        // Order and orientation do not matter; the node count does.
        edges.reverse();
        edges[0] = (edges[0].1, edges[0].0);
        assert_eq!(topology_fingerprint(4577, &edges), CHAIN_GATES.fingerprint);
        assert_ne!(topology_fingerprint(4578, &edges), CHAIN_GATES.fingerprint);
    }

    #[test]
    fn chain_gates_need_the_measured_conditions() {
        let settings = CascadeSettings::default();
        let measured = SampleParams {
            num_reads: CHAIN_GATES.min_reads,
            ..params(14_336, 1)
        };
        assert!(settings.chain_gated(true, &measured));
        assert!(!settings.chain_gated(false, &measured));
        let fewer = SampleParams {
            num_reads: 32,
            ..measured
        };
        let slower = SampleParams {
            sweeps_per_beta: 2,
            ..measured
        };
        let ranged = SampleParams {
            beta_range: Some((0.1, 4.0)),
            ..measured
        };
        for other in [fewer, slower, ranged] {
            assert!(!settings.chain_gated(true, &other));
        }
        let reheated = CascadeSettings {
            reheat_beta: 0.3,
            ..settings
        };
        assert!(!reheated.chain_gated(true, &measured));
        let gated = settings.effective(true);
        assert_eq!(gated.stages, CHAIN_GATES.stages);
        assert_eq!(gated.gates, CHAIN_GATES.gates.map(Some));
        assert_eq!(settings.effective(false), settings);
    }

    #[test]
    fn schedule_cache_shares_equal_schedules_and_rebuilds_on_change() {
        let settings = CascadeSettings::default();
        let mut cache = ScheduleCache::default();
        let first = cache.prepare(&job(0, 1000), settings, false).unwrap();
        let second = cache.prepare(&job(1, 1000), settings, false).unwrap();
        assert!(Arc::ptr_eq(&first.betas, &second.betas));
        let reference = PreparedSchedule::new(&job(1, 1000), settings, false);
        assert_eq!(&second.betas[..], &reference.betas[..]);
        assert_eq!(second.checkpoints, reference.checkpoints);
        let longer = cache.prepare(&job(2, 2000), settings, false).unwrap();
        assert!(!Arc::ptr_eq(&first.betas, &longer.betas));
        assert_eq!(longer.betas.len(), 2000);
        let gated = cache.prepare(&job(2, 2000), settings, true).unwrap();
        assert_eq!(gated.checkpoints, vec![8, 16, 64, 256, 2000]);
    }

    #[test]
    fn report_counts_each_level_and_resets_the_window() {
        let mut c = Controller::new(CascadeSettings::default().effective(true));
        let start = c.stats.started;
        let (mut kept, _, _) = c.admit(&job(0, 1000));
        for best in [-13_300_000, -13_800_000, -14_300_000, -14_500_000] {
            assert!(c.checkpoint(&mut kept, best));
        }
        c.finish(&kept, Some(-14_600_000), true);
        let (mut screened, _, _) = c.admit(&job(0, 1000));
        assert!(!c.checkpoint(&mut screened, -13_000_000));
        c.finish(&screened, Some(-13_000_000), true);
        let (cancelled, _, _) = c.admit(&job(0, 1000));
        c.finish(&cancelled, None, false);

        assert_eq!((c.stats.models, c.stats.cancelled), (2, 1));
        let first = c.stats.levels[0].unwrap();
        assert_eq!(
            first,
            LevelStats {
                sweeps: 8,
                checked: 2,
                kept: 1,
                best: -13_300_000,
                worst: -13_000_000,
            }
        );
        assert_eq!(c.stats.levels[1].unwrap().checked, 1);
        let full = c.stats.levels[MAX_STAGES].unwrap();
        assert_eq!(
            (full.sweeps, full.checked, full.best),
            (1000, 1, -14_600_000)
        );

        c.report(start + Duration::from_secs(59), Duration::from_secs(60));
        assert_eq!(c.stats.models, 2);
        c.report(start + Duration::from_secs(60), Duration::from_secs(60));
        assert_eq!(c.stats.models, 0);
        assert!(c.stats.levels.iter().all(Option::is_none));
    }

    #[test]
    fn energy_prints_whole_units_and_fractions() {
        assert_eq!(energy(-14_272_000), "-14272");
        assert_eq!(energy(-1_500), "-1.500");
        assert_eq!(energy(250), "0.250");
        assert_eq!(energy(0), "0");
    }

    #[test]
    fn open_gates_keep_every_job_to_the_full_budget() {
        let mut settings = CascadeSettings::default();
        settings.merge(&CascadeToml {
            cascade_open_gates: Some(true),
            ..CascadeToml::default()
        });
        let mut c = Controller::new(CascadeSettings {
            gates: [Some(-10), None, None, None],
            ..settings
        });
        let (mut ticket, _, _) = c.admit(&job(0, 1000));
        assert!(c.checkpoint(&mut ticket, 0));
        assert!(c.checkpoint(&mut ticket, 0));
        assert_eq!(ticket.stage, ticket.gates);
    }

    #[test]
    fn removed_cascade_key_warns_and_keeps_screening() {
        #[derive(Clone)]
        struct LogWriter(Arc<Mutex<Vec<u8>>>);

        impl std::io::Write for LogWriter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let output = Arc::new(Mutex::new(Vec::new()));
        let writer = LogWriter(Arc::clone(&output));
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let backend_toml = "cascade = false\nnum_sweeps = 1024";
        tracing::subscriber::with_default(subscriber, || {
            assert_eq!(
                crate::resolve_governor_config(backend_toml, 73, true),
                (73, true)
            );
        });
        let log = String::from_utf8(output.lock().unwrap().clone()).unwrap();
        assert_eq!(
            log.matches("unknown field 'cascade' for metal (ignored)")
                .count(),
            1
        );
        assert!(!log.contains("unknown field 'num_sweeps'"));
        let cfg: crate::MetalConfig = toml::from_str(backend_toml).unwrap();
        assert_eq!(cfg.unknown["cascade"].as_bool(), Some(false));
        let mut settings = CascadeSettings::default();
        settings.merge(&cfg.cascade);
        let mut controller = Controller::new(settings);
        let (mut ticket, _, checkpoints) = controller.admit(&job(0, 1024));
        assert_eq!(checkpoints, vec![32, 256, 1024]);
        assert!(!controller.checkpoint(&mut ticket, 0));
    }

    #[test]
    fn backend_toml_routes_cascade_keys_and_leaves_unknown_keys_unknown() {
        let cfg: crate::MetalConfig = toml::from_str("cascade_keep = 5000\nbogus = 1").unwrap();
        assert_eq!(cfg.cascade.cascade_keep, Some(5000));
        assert_eq!(cfg.unknown.keys().collect::<Vec<_>>(), vec!["bogus"]);
    }

    #[test]
    fn merge_accepts_all_keys_and_partial_updates() {
        let cfg: crate::MetalConfig = toml::from_str("cascade_stages = [16, 64]\ncascade_keep = 20\ncascade_keep_min = 2\ncascade_keep_max = 100\ncascade_audit = 2\ncascade_target_milli = -500\ncascade_yield_per_million = 1.5\ncascade_reheat_beta = 0.3").unwrap();
        assert!(cfg.unknown.is_empty());
        let mut settings = CascadeSettings::default();
        settings.merge(&cfg.cascade);
        assert_eq!(
            settings,
            CascadeSettings {
                stages: [16, 64, 0, 0],
                gates: [None; MAX_STAGES],
                open_gates: false,
                keep: 20.0,
                keep_min: 2.0,
                keep_max: 100.0,
                audit: 2,
                reheat_beta: 0.3,
                target_milli: Some(-500),
                yield_per_million: Some(1.5)
            }
        );
        let previous = settings;
        settings.merge(&CascadeToml {
            cascade_audit: Some(7),
            ..CascadeToml::default()
        });
        assert_eq!(
            settings,
            CascadeSettings {
                audit: 7,
                ..previous
            }
        );
    }

    #[test]
    fn admitted_jobs_retain_their_plan_after_stage_change() {
        let mut c = Controller::new(CascadeSettings::default());
        let (mut t, _, checkpoints) = c.admit(&job(0, 14_336));
        for cutoff in &mut c.plan.lock().unwrap().cutoffs {
            for _ in 0..200 {
                cutoff.observe(0.0);
            }
        }
        let old = Arc::downgrade(&c.plan);
        c.refresh(CascadeSettings {
            stages: [64, 0, 0, 0],
            ..CascadeSettings::default()
        });
        assert!(!Arc::ptr_eq(&t.plan, &c.plan));
        assert_eq!(checkpoints, vec![32, 256, 14_336]);
        assert!(c.checkpoint(&mut t, -1));
        assert!(c.checkpoint(&mut t, -1));
        c.finish(&t, Some(-1), true);
        assert_eq!(t.plan.lock().unwrap().cutoffs[2].moments().count(), 201);
        drop(t);
        assert!(old.upgrade().is_none());
        assert_eq!(c.plan.lock().unwrap().cutoffs[0].moments().count(), 0);
        let (t, _, checkpoints) = c.admit(&job(1, 1024));
        assert_eq!(t.gates, 1);
        assert_eq!(checkpoints, vec![64, 1024]);
    }

    #[test]
    fn live_keep_and_yield_changes_reset_only_their_controllers() {
        let settings = CascadeSettings {
            target_milli: Some(-100),
            yield_per_million: Some(1.0),
            ..CascadeSettings::default()
        };
        let mut c = Controller::new(settings);
        let (old, _, _) = c.admit(&job(0, 32));
        c.plan.lock().unwrap().cutoffs[0].observe(1.0);
        let settings = CascadeSettings {
            keep: 5000.0,
            ..settings
        };
        c.refresh(settings);
        assert_eq!(c.plan.lock().unwrap().cutoffs[0].moments().count(), 0);
        let plan = Arc::clone(&c.plan);
        c.refresh(CascadeSettings {
            target_milli: Some(-200),
            yield_per_million: Some(10_000_000.0),
            ..settings
        });
        assert!(Arc::ptr_eq(&plan, &c.plan));
        c.finish(&old, Some(-300), true);
        assert_eq!(c.yield_check.as_ref().unwrap().observation(), (0.0, 0.0));
        let (t, _, _) = c.admit(&job(1, 32));
        c.finish(&t, Some(-150), true);
        let (hits, bound) = c.yield_check.as_ref().unwrap().observation();
        assert_eq!(hits, 0.0);
        assert!((bound - (10f64.sqrt() - 0.98).powi(2)).abs() < 1e-10);
    }

    #[test]
    fn live_settings_change_stages_on_one_stream() {
        let mut c = Controller::new(CascadeSettings::default());
        for (stages, expected) in [
            ([32, 256, 0, 0], vec![32, 256, 1000]),
            ([64, 0, 0, 0], vec![64, 1000]),
            ([16, 64, 0, 0], vec![16, 64, 1000]),
        ] {
            c.refresh(CascadeSettings {
                stages,
                ..CascadeSettings::default()
            });
            let (t, _, checkpoints) = c.admit(&job(0, 1000));
            assert_eq!(checkpoints, expected);
            assert_eq!(t.gates, expected.len() - 1);
        }
    }

    #[test]
    fn custom_stages_use_transition_calibration_and_skip_32_sweep_drift() {
        let ((probe, full), rate) = CALIBRATION.false_negative[1];
        let mut c = Controller::new(CascadeSettings {
            stages: [probe, 0, 0, 0],
            ..CascadeSettings::default()
        });
        let (mut t, _, _) = c.admit(&job(0, full));
        let before = {
            let mut p = c.plan.lock().unwrap();
            assert_eq!(
                p.audit_lanes[0].observation(true),
                AuditLane::new(rate).observation(true)
            );
            for _ in 0..10_000 {
                p.cutoffs[0].observe(0.0);
            }
            let mut expected = p.cutoffs[0].clone();
            expected.observe(-1.0);
            expected.denominator()
        };
        c.checkpoint(&mut t, -1);
        assert_eq!(c.plan.lock().unwrap().cutoffs[0].denominator(), before);
    }

    #[test]
    fn short_full_budget_never_trains_another_transition() {
        let mut c = Controller::new(CascadeSettings::default());
        for budget in [14_336, 1024] {
            let (mut t, _, _) = c.admit(&job(0, budget));
            for cutoff in &mut c.plan.lock().unwrap().cutoffs {
                for _ in 0..200 {
                    cutoff.observe(0.0);
                }
            }
            assert!(c.checkpoint(&mut t, -1));
            assert!(c.checkpoint(&mut t, -1));
            c.finish(&t, Some(-1), true);
        }
        assert_eq!(c.plan.lock().unwrap().kept[1].len(), 1);
        for cutoff in &mut c.plan.lock().unwrap().cutoffs {
            cutoff.reset();
        }
        let (mut t, _, _) = c.admit(&job(2, 1024));
        c.plan.lock().unwrap().audit_rng = Some(0);
        assert!(c.checkpoint(&mut t, 0));
        assert!(!c.checkpoint(&mut t, 0));
        assert!(!t.audited);
    }

    #[test]
    fn audit_restore_preserves_drift_at_the_clamp() {
        let c = Controller::new(CascadeSettings::default());
        let mut p = c.plan.lock().unwrap();
        for _ in 0..100_000 {
            p.cutoffs[0].observe(0.0);
        }
        p.cutoffs[0].load_update(100.0, 1.0);
        p.apply_action(0, Action::LoosenAndDoubleK0, Check::Drift, &[], &[]);
        let before = p.cutoffs[0].denominator();
        p.apply_action(
            0,
            Action::Loosen { raise_audit: true },
            Check::Audit,
            &[],
            &[],
        );
        p.apply_action(0, Action::Restore, Check::Audit, &[], &[]);
        assert!((p.cutoffs[0].denominator() - before).abs() < 1e-12);
    }

    #[test]
    fn yield_counts_forwarded_results_and_widens_each_probe() {
        for (best, delivered, hits) in [
            (Some(-100), true, 1.0),
            (Some(0), true, 0.0),
            (Some(-100), false, 0.0),
            (None, true, 0.0),
        ] {
            let mut c = Controller::new(CascadeSettings {
                target_milli: Some(-100),
                yield_per_million: Some(10_000_000.0),
                ..CascadeSettings::default()
            });
            let (t, _, _) = c.admit(&job(0, 32));
            c.finish(&t, best, delivered);
            let (actual, bound) = c.yield_check.as_ref().unwrap().observation();
            assert_eq!(actual, hits);
            assert!((bound - (10f64.sqrt() - 0.98).powi(2)).abs() < 1e-10);
            c.check_yield(Instant::now() + Duration::from_secs(3601));
            for lane in &c.plan.lock().unwrap().audit_lanes {
                assert_eq!(lane.denominator(200), 50);
            }
            assert_eq!(c.yield_check.as_ref().unwrap().observation().0, 0.0);
        }
        assert!(Controller::new(CascadeSettings {
            target_milli: Some(-100),
            ..CascadeSettings::default()
        })
        .yield_check
        .is_none());
    }

    #[test]
    fn reference_median_and_zero_seed_are_well_defined() {
        assert_eq!(kept_median(&VecDeque::new()), None);
        assert_eq!(kept_median(&VecDeque::from([5, 1, 3])), Some(3.0));
        assert_eq!(kept_median(&VecDeque::from([5, 1, 3, 7])), Some(4.0));
        let mut c = Controller::new(CascadeSettings::default());
        let mut first = job(0, 1024);
        first.params.seed = 0;
        c.admit(&first);
        let mut p = c.plan.lock().unwrap();
        let selected = (0..10_000).filter(|_| p.audit_selected(0)).count();
        assert!((20..100).contains(&selected), "{selected}");
    }

    #[test]
    fn injected_false_negative_burst_loosens_the_stage() {
        let mut c = Controller::new(CascadeSettings {
            stages: [32, 0, 0, 0],
            keep: 100.0,
            keep_min: 2.0,
            keep_max: 1000.0,
            audit: 2,
            ..CascadeSettings::default()
        });
        let (mut t, _, _) = c.admit(&job(0, 256));
        for _ in 0..100_000 {
            c.plan.lock().unwrap().cutoffs[0].observe(-100.0);
        }
        assert!(c.checkpoint(&mut t, -1000));
        c.finish(&t, Some(-1000), true);
        let before = c.plan.lock().unwrap().cutoffs[0].denominator();
        for id in 1..1000 {
            let (mut t, _, _) = c.admit(&job(id, 256));
            if c.checkpoint(&mut t, 0) {
                c.finish(&t, Some(-2000), true);
            } else {
                c.finish(&t, Some(0), true);
            }
            let p = c.plan.lock().unwrap();
            if p.audit_lanes[0].denominator(200) == 50 {
                assert!((p.cutoffs[0].denominator() - before / 2.0).abs() < 0.01);
                assert_eq!(p.kept[0].len(), 1);
                return;
            }
        }
        panic!("audit burst did not loosen the screen");
    }

    #[test]
    fn a_job_at_or_below_the_first_budget_passes_straight_through() {
        let mut c = Controller::new(CascadeSettings::default());
        for budget in [16, 32] {
            let (t, s, checkpoints) = c.admit(&job(0, budget));
            assert_eq!(t.gates, 0);
            assert_eq!(checkpoints, vec![budget]);
            assert_eq!(s, build_beta_schedule(&ring(), budget, 1, None).0);
        }
    }

    #[test]
    fn configured_stage_roots_and_effective_budgets_preserve_the_original_job() {
        let settings = CascadeSettings {
            stages: [32, 128, 256, 0],
            reheat_beta: 1.0,
            ..CascadeSettings::default()
        };
        let mut c = Controller::new(settings);
        let mut original = job(0, 128);
        original.params.beta_range = Some((0.5, 4.0));
        original.params.sweeps_per_beta = 2;
        let (mut t, s, checkpoints) = c.admit(&original);
        assert_eq!(t.gates, 1);
        assert_eq!(checkpoints, vec![32, 128]);
        assert_eq!(original.params.seed, 42);
        assert_eq!(original.params.num_sweeps, 128);
        assert_eq!(original.params.beta_range, Some((0.5, 4.0)));
        assert_eq!(original.watermark, Some(1));
        assert_eq!(
            s[..32],
            build_beta_schedule(&original.graph, 32, 2, Some((0.5, 4.0)))
                .0
                .into_iter()
                .flat_map(|b| [b; 2])
                .collect::<Vec<_>>()
        );
        {
            let mut p = c.plan.lock().unwrap();
            assert!((p.cutoffs[0].denominator() - settings.keep_min.powf(1.0 / 3.0)).abs() < 1e-10);
            for _ in 0..200 {
                p.cutoffs[0].observe(0.0);
            }
        }
        assert!(c.checkpoint(&mut t, -1));
        c.finish(&t, Some(-2), true);
        let p = c.plan.lock().unwrap();
        assert_eq!(p.cutoffs[1].moments().count(), 0);
        assert_eq!(p.cutoffs[2].moments().count(), 0);
        assert_eq!(p.cutoffs[3].moments().count(), 1);
        assert!(p.kept.iter().all(VecDeque::is_empty));
    }

    #[test]
    fn load_feedback_loosens_tightens_and_decays() {
        let mut c = Controller::new(CascadeSettings::default());
        c.update_load(0.0, &[], 4);
        assert!(c.plan.lock().unwrap().cutoffs[0].log_adjust() < 0.0);
        c.plan.lock().unwrap().cutoffs[0].reset();
        c.update_load(2.0, &[1, 3, 0], 4);
        let positive = c.plan.lock().unwrap().cutoffs[0].log_adjust();
        assert!((positive - 0.1 * 1.5f64.ln()).abs() < 1e-12);
        c.update_load(2.0, &[4, 0, 0], 4);
        assert!((c.plan.lock().unwrap().cutoffs[0].log_adjust() - 0.9 * positive).abs() < 1e-12);
        c.update_load(0.9, &[], 0);
        assert!(c.plan.lock().unwrap().cutoffs[0].log_adjust().is_finite());
    }

    #[test]
    fn final_audit_and_cancel_accounting() {
        let mut c = Controller::new(CascadeSettings {
            stages: [32, 0, 0, 0],
            ..CascadeSettings::default()
        });
        let (mut t, _, _) = c.admit(&job(0, 256));
        {
            let mut p = c.plan.lock().unwrap();
            p.kept[0].push_back(-100);
            p.audit_rng = Some(0);
        }
        assert!(c.checkpoint(&mut t, 0));
        assert!(t.audited);
        c.finish(&t, Some(-200), true);
        let mut expected = AuditLane::new(CALIBRATION.false_negative[0].1);
        expected.record(true);
        assert_eq!(
            c.plan.lock().unwrap().audit_lanes[0].observation(false),
            expected.observation(false)
        );
        let (mut cancelled, _, _) = c.admit(&job(1, 256));
        assert!(c.checkpoint(&mut cancelled, 0));
        c.finish(&cancelled, None, true);
        let p = c.plan.lock().unwrap();
        assert_eq!(
            p.audit_lanes[0].observation(false),
            expected.observation(false)
        );
        assert_eq!(p.cutoffs[1].moments().count(), 1);
    }

    #[test]
    fn remainder_and_one_sweep_segments_have_exact_lengths() {
        let (s, checkpoints) = segment_schedule(&ring(), &params(34, 3), &[32, 33], 0.25);
        assert_eq!(s.len(), 34);
        assert_eq!(checkpoints, vec![32, 33, 34]);
        let cold = default_ising_beta_range(&ring()).1 as f32;
        assert_eq!(s[31..], [cold; 3]);
        let (s, checkpoints) = segment_schedule(&ring(), &params(0, 0), &[32, 256], 0.25);
        assert!(s.is_empty());
        assert_eq!(checkpoints, vec![0]);
    }

    #[test]
    fn arm_d_constants_and_default_reheat_match_the_report() {
        assert_eq!(CALIBRATION.stages, &[32, 256]);
        assert_eq!(CALIBRATION.skew, &[-0.123, -0.056]);
        assert_eq!(CALIBRATION.excess_kurtosis, &[0.014, 0.002]);
        assert_eq!(
            CALIBRATION.false_negative,
            &[((32, 256), 0.0353), ((256, 14_336), 0.0051)]
        );
        assert_eq!(CALIBRATION.cost_a_us, 124.0);
        assert_eq!(CALIBRATION.cost_b_us, 2.284);
        assert_eq!(CALIBRATION.k0, 1.0);
        assert_eq!(CALIBRATION.reheat_beta, 0.25);
        assert_eq!(CascadeSettings::default().reheat_beta, 0.25);
    }

    #[test]
    fn short_second_topology_preserves_inflight_training() {
        let mut c = Controller::new(CascadeSettings::default());
        let (mut ticket, _, _) = c.admit(&job(0, 1000));
        let plan = Arc::clone(&c.plan);
        let epochs = (c.topology_epoch, c.yield_epoch);
        let rng = c.plan.lock().unwrap().audit_rng;
        let mut short = job(1, 32);
        short.graph.h.push(0.0);
        let (short_ticket, _, _) = c.admit(&short);
        assert_eq!(short_ticket.gates, 0);
        assert!(Arc::ptr_eq(&plan, &c.plan));
        assert_eq!((c.topology_epoch, c.yield_epoch), epochs);
        assert!(c.matches_topology(&ring()));
        assert_eq!(c.plan.lock().unwrap().audit_rng, rng);
        c.checkpoint(&mut ticket, -100);
        assert_eq!(c.plan.lock().unwrap().cutoffs[0].moments().count(), 1);
    }

    #[test]
    fn short_admission_leaves_topology_unset_and_plan_untouched() {
        let mut c = Controller::new(CascadeSettings::default());
        let plan = Arc::clone(&c.plan);
        let (ticket, _, checkpoints) = c.admit(&job(0, 16));
        assert_eq!(ticket.gates, 0);
        assert_eq!(checkpoints, vec![16]);
        assert!(c.topology.is_none());
        assert!(Arc::ptr_eq(&plan, &c.plan));
        assert_eq!((c.topology_epoch, c.yield_epoch), (0, 0));
        assert_eq!(c.plan.lock().unwrap().audit_rng, None);
    }

    #[test]
    fn real_topology_change_keeps_yield_state_and_epoch() {
        let mut c = Controller::new(CascadeSettings {
            target_milli: Some(-100),
            yield_per_million: Some(10_000_000.0),
            ..CascadeSettings::default()
        });
        let (ticket, _, _) = c.admit(&job(0, 32));
        c.finish(&ticket, Some(-100), true);
        let yield_epoch = c.yield_epoch;
        c.admit(&job(1, 1000));
        c.plan.lock().unwrap().cutoffs[0].observe(0.0);
        let topology_epoch = c.topology_epoch;
        let mut changed = job(2, 1000);
        changed.graph.h.push(0.0);
        c.admit(&changed);
        assert_eq!(c.topology_epoch, topology_epoch + 1);
        assert_eq!(c.yield_epoch, yield_epoch);
        assert_eq!(c.plan.lock().unwrap().cutoffs[0].moments().count(), 0);
        let (hits, bound) = c.yield_check.as_ref().unwrap().observation();
        assert_eq!(hits, 1.0);
        assert!((bound - (30f64.sqrt() - 0.98).powi(2)).abs() < 1e-10);
    }

    #[test]
    fn late_topology_result_counts_yield_without_training_either_plan() {
        let mut c = Controller::new(CascadeSettings {
            target_milli: Some(-100),
            yield_per_million: Some(10_000_000.0),
            ..CascadeSettings::default()
        });
        let (mut ticket, _, _) = c.admit(&job(0, 1000));
        ticket.stage = ticket.gates;
        let old_plan = Arc::clone(&ticket.plan);
        let mut changed = job(1, 1000);
        changed.graph.h.push(0.0);
        c.admit(&changed);
        c.finish(&ticket, Some(-100), true);
        let (hits, bound) = c.yield_check.as_ref().unwrap().observation();
        assert_eq!(hits, 1.0);
        assert!((bound - (20f64.sqrt() - 0.98).powi(2)).abs() < 1e-10);
        for plan in [&old_plan, &c.plan] {
            assert!(plan
                .lock()
                .unwrap()
                .cutoffs
                .iter()
                .all(|cutoff| cutoff.moments().count() == 0));
        }
        c.refresh(CascadeSettings {
            target_milli: Some(-200),
            ..c.settings
        });
        c.finish(&ticket, Some(-300), true);
        assert_eq!(c.yield_check.as_ref().unwrap().observation(), (0.0, 0.0));
    }

    fn run_topologies(controller: Controller, jobs: Vec<StreamJob>) -> Controller {
        struct Governor;
        impl crate::streaming::GpuGovernor for Governor {
            fn should_throttle(&self) -> bool {
                false
            }
            fn budget_scale(&self) -> f64 {
                1.0
            }
            fn record_gpu_busy_us(&self, _: u64) {}
        }
        let device = crate::metal_device::MetalDevice::open(0).unwrap();
        let count = jobs.len();
        let (tx, rx) = tokio::sync::mpsc::channel(count);
        let (out, mut results) = tokio::sync::mpsc::channel(count);
        for job in jobs {
            tx.try_send(job).unwrap();
        }
        drop(tx);
        let settings = Mutex::new(controller.settings);
        let store = Mutex::new(Some(controller));
        crate::resident::run(
            &device,
            &settings,
            &store,
            rx,
            &out,
            &Governor,
            &quip_solver_core::CancelToken::default(),
        );
        for _ in 0..count {
            assert!(matches!(
                results.try_recv().unwrap().outcome,
                quip_solver_core::StreamOutcome::Completed(Ok(_))
            ));
        }
        store.into_inner().unwrap().unwrap()
    }

    #[test]
    fn runner_preserves_trained_a_across_gate_free_b_and_back() {
        let mut c = Controller::new(CascadeSettings::default());
        let input = job(0, 64);
        for _ in 0..200 {
            let (mut ticket, _, _) = c.admit(&input);
            c.checkpoint(&mut ticket, 0);
        }
        let plan = Arc::clone(&c.plan);
        let epoch = c.topology_epoch;
        let mut short = job(1, 32);
        short.graph.h.push(0.0);
        let c = run_topologies(c, vec![input, short, job(2, 64)]);
        assert!(Arc::ptr_eq(&plan, &c.plan));
        assert_eq!(c.topology_epoch, epoch);
        assert_eq!(c.plan.lock().unwrap().cutoffs[0].moments().count(), 202);
    }

    #[test]
    fn runner_preserves_yield_across_gated_topology_change() {
        let mut c = Controller::new(CascadeSettings {
            target_milli: Some(i64::MAX),
            yield_per_million: Some(10_000_000.0),
            ..CascadeSettings::default()
        });
        let (ticket, _, _) = c.admit(&job(0, 32));
        c.finish(&ticket, Some(0), true);
        let epoch = c.yield_epoch;
        let mut changed = job(2, 64);
        changed.graph.h.push(0.0);
        let c = run_topologies(c, vec![job(1, 64), changed]);
        assert_eq!(c.yield_epoch, epoch);
        let (hits, bound) = c.yield_check.as_ref().unwrap().observation();
        assert_eq!(hits, 3.0);
        assert!((bound - (30f64.sqrt() - 0.98).powi(2)).abs() < 1e-10);
    }

    #[test]
    fn custom_beta_range_controls_reheat_validation_and_tails() {
        let mut params = params(1000, 2);
        params.beta_range = Some((0.5, 4.0));
        let (fallback, checkpoints) = segment_schedule(&ring(), &params, &[32, 256], 0.25);
        let expected: Vec<_> = build_beta_schedule(&ring(), 1000, 2, params.beta_range)
            .0
            .into_iter()
            .flat_map(|b| [b; 2])
            .collect();
        assert_eq!(fallback, expected);
        assert_eq!(checkpoints, vec![32, 256, 1000]);
        let (reheated, reheated_checkpoints) = segment_schedule(&ring(), &params, &[32, 256], 1.0);
        assert_eq!(reheated_checkpoints, checkpoints);
        assert_eq!(reheated.len(), 1000);
        for i in [32, 256] {
            assert_eq!(reheated[i], 1.0);
        }
        for i in [31, 255, 999] {
            assert_eq!(reheated[i], 4.0);
        }
    }
}
