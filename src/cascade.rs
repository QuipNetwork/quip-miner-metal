// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Host-only cascade decisions and schedules for the resident slot runner.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::sampler::build_beta_schedule;
use crate::{IsingGraph, SampleParams};
use quip_solver_core::beta::{default_ising_beta_range, geometric_beta_schedule};
use quip_solver_core::StreamJob;

use crate::cutoff::{Cutoff, CutoffConfig};
use crate::model_checks::{Action, AuditLane, Drift, Yield};

pub(crate) const MAX_STAGES: usize = 3;

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
    pub(crate) enabled: bool,
    pub(crate) stages: [usize; MAX_STAGES],
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
            enabled: false,
            stages,
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

#[derive(serde::Deserialize, Default, Clone)]
pub(crate) struct CascadeToml {
    pub(crate) cascade: Option<bool>,
    pub(crate) cascade_stages: Option<Vec<usize>>,
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
        }
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
        self.plan
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .settings
            .audit = settings.audit;
    }

    pub(crate) fn matches_topology(&self, graph: &IsingGraph) -> bool {
        self.topology
            .as_ref()
            .is_some_and(|(n, edges)| *n == graph.num_nodes() && *edges == graph.edges)
    }

    /// Reset gate state on a gated topology change. Return the job's schedule plan.
    pub(crate) fn admit(&mut self, job: &StreamJob) -> (Ticket, Vec<f32>, Vec<usize>) {
        let stages = if self.settings.enabled {
            &self.settings.stages[..]
        } else {
            &[]
        };
        let (schedule, checkpoints) =
            segment_schedule(&job.graph, &job.params, stages, self.settings.reheat_beta);
        if checkpoints.len() > 1 {
            if !self.matches_topology(&job.graph) {
                self.plan = Arc::new(Mutex::new(StagePlan::new(self.settings)));
                self.topology = Some((job.graph.num_nodes(), job.graph.edges.clone()));
                self.topology_epoch = self.topology_epoch.wrapping_add(1);
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
        let ticket = Ticket {
            gates: checkpoints.len() - 1,
            stage: 0,
            audited: false,
            topology_epoch: self.topology_epoch,
            yield_epoch: self.yield_epoch,
            plan: Arc::clone(&self.plan),
            final_sweeps: job.params.num_sweeps,
        };
        (ticket, schedule, checkpoints)
    }

    /// Called at a non-last checkpoint. true = keep running.
    pub(crate) fn checkpoint(&mut self, ticket: &mut Ticket, best: i64) -> bool {
        if ticket.topology_epoch != self.topology_epoch {
            return false;
        }
        let mut plan = ticket.plan.lock().unwrap_or_else(|p| p.into_inner());
        Self::observe_transition(&mut plan, ticket, best);
        let stage = ticket.stage;
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
        if ticket.topology_epoch != self.topology_epoch {
            return;
        }
        if let Some(best) = best {
            if ticket.stage == ticket.gates && ticket.gates > 0 {
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
        .unwrap_or_else(|| default_ising_beta_range(graph));
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
        build_beta_schedule(graph, first, params.sweeps_per_beta, params.beta_range);
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
    fn enabled_settings() -> CascadeSettings {
        CascadeSettings {
            enabled: true,
            ..CascadeSettings::default()
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
            ..enabled_settings()
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
        let mut c = Controller::new(enabled_settings());
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
        let mut c = Controller::new(enabled_settings());
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
        let mut c = Controller::new(enabled_settings());
        let (mut t, _, _) = c.admit(&job(0, 1000));
        c.checkpoint(&mut t, 0);
        c.refresh(CascadeSettings {
            audit: 50,
            ..enabled_settings()
        });
        assert_eq!(c.plan.lock().unwrap().cutoffs[0].moments().count(), 1);
        assert_eq!(c.plan.lock().unwrap().settings.audit, 50);
        c.refresh(CascadeSettings {
            stages: [16, 128, 0],
            ..enabled_settings()
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
    fn merge_accepts_all_keys_and_partial_updates() {
        let cfg: crate::MetalConfig = toml::from_str("cascade = true\ncascade_stages = [16, 64]\ncascade_keep = 20\ncascade_keep_min = 2\ncascade_keep_max = 100\ncascade_audit = 2\ncascade_target_milli = -500\ncascade_yield_per_million = 1.5\ncascade_reheat_beta = 0.3").unwrap();
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
                reheat_beta: 0.3,
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

    #[test]
    fn admitted_jobs_retain_their_plan_after_disable_and_stage_change() {
        let mut c = Controller::new(enabled_settings());
        let (mut t, _, checkpoints) = c.admit(&job(0, 14_336));
        for cutoff in &mut c.plan.lock().unwrap().cutoffs {
            for _ in 0..200 {
                cutoff.observe(0.0);
            }
        }
        let old = Arc::downgrade(&c.plan);
        c.refresh(CascadeSettings {
            enabled: false,
            stages: [64, 0, 0],
            ..enabled_settings()
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
        assert_eq!(t.gates, 0);
        assert_eq!(checkpoints, vec![1024]);
    }

    #[test]
    fn live_keep_and_yield_changes_reset_only_their_controllers() {
        let settings = CascadeSettings {
            target_milli: Some(-100),
            yield_per_million: Some(1.0),
            ..enabled_settings()
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
    fn live_settings_enable_disable_enable_and_change_stages_on_one_stream() {
        let mut c = Controller::new(enabled_settings());
        for (enabled, stages, expected) in [
            (true, [32, 256, 0], vec![32, 256, 1000]),
            (false, [32, 256, 0], vec![1000]),
            (true, [16, 64, 0], vec![16, 64, 1000]),
        ] {
            c.refresh(CascadeSettings {
                enabled,
                stages,
                ..enabled_settings()
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
            stages: [probe, 0, 0],
            ..enabled_settings()
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
        let mut c = Controller::new(enabled_settings());
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
        let c = Controller::new(enabled_settings());
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
                ..enabled_settings()
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
            ..enabled_settings()
        })
        .yield_check
        .is_none());
    }

    #[test]
    fn reference_median_and_zero_seed_are_well_defined() {
        assert_eq!(kept_median(&VecDeque::new()), None);
        assert_eq!(kept_median(&VecDeque::from([5, 1, 3])), Some(3.0));
        assert_eq!(kept_median(&VecDeque::from([5, 1, 3, 7])), Some(4.0));
        let mut c = Controller::new(enabled_settings());
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
            stages: [32, 0, 0],
            keep: 100.0,
            keep_min: 2.0,
            keep_max: 1000.0,
            audit: 2,
            ..enabled_settings()
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
        let mut c = Controller::new(enabled_settings());
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
            stages: [32, 128, 256],
            reheat_beta: 1.0,
            ..enabled_settings()
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
        let mut c = Controller::new(enabled_settings());
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
            stages: [32, 0, 0],
            ..enabled_settings()
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
        let mut c = Controller::new(enabled_settings());
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
    fn disabled_admission_leaves_topology_unset_and_plan_untouched() {
        let mut c = Controller::new(CascadeSettings::default());
        let plan = Arc::clone(&c.plan);
        let (ticket, _, checkpoints) = c.admit(&job(0, 1000));
        assert_eq!(ticket.gates, 0);
        assert_eq!(checkpoints, vec![1000]);
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
            ..enabled_settings()
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
