// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Checks that the running cascade still matches its calibration: the audit
//! lane's false-negative rate, the stage-0 energy shape, and the yield of
//! nonces below the chain target. Each returns an [`Action`] that widens the
//! screen when an observation leaves its projection.

use std::time::{Duration, Instant};

use crate::cutoff::Moments;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Action {
    None,
    Loosen { raise_audit: bool },
    Restore,
    LoosenAndDoubleK0,
}

pub(crate) struct AuditLane {
    expected: f64,
    audits: u64,
    misses: u64,
    raised: bool,
}

impl AuditLane {
    pub(crate) fn new(expected_false_negative: f64) -> Self {
        Self {
            expected: expected_false_negative,
            audits: 0,
            misses: 0,
            raised: false,
        }
    }

    /// Current miss count and calibrated upper bound, including the next audit.
    pub(crate) fn observation(&self, miss: bool) -> (f64, f64) {
        let n = (self.audits + 1) as f64;
        (
            (self.misses + u64::from(miss)) as f64,
            n * self.expected + 2.326 * (n * self.expected * (1.0 - self.expected)).sqrt() + 1.0,
        )
    }

    pub(crate) fn record(&mut self, miss: bool) -> Action {
        let (observed, bound) = self.observation(miss);
        self.audits += 1;
        self.misses += u64::from(miss);
        let action = if self.audits >= 20 && observed > bound {
            self.raised = true;
            Action::Loosen { raise_audit: true }
        } else if self.raised && self.audits >= 200 {
            self.raised = false;
            Action::Restore
        } else {
            return Action::None;
        };
        self.audits = 0;
        self.misses = 0;
        action
    }

    pub(crate) fn raise(&mut self) {
        if !self.raised {
            self.audits = 0;
            self.misses = 0;
        }
        self.raised = true;
    }

    pub(crate) fn denominator(&self, base: u32) -> u32 {
        if self.raised {
            base.min(50)
        } else {
            base
        }
    }
}

pub(crate) struct Drift {
    skew: f64,
    kurtosis: f64,
    next_check: u64,
}

impl Drift {
    pub(crate) fn new(skew: f64, kurtosis: f64) -> Self {
        Self {
            skew,
            kurtosis,
            next_check: 10_000,
        }
    }

    pub(crate) fn check(&mut self, m: &Moments) -> Action {
        if m.count() < self.next_check {
            return Action::None;
        }
        self.next_check = (m.count() / 10_000 + 1) * 10_000;
        if (m.skew() - self.skew).abs() > 0.1 || (m.excess_kurtosis() - self.kurtosis).abs() > 0.1 {
            Action::LoosenAndDoubleK0
        } else {
            Action::None
        }
    }
}

pub(crate) struct Yield {
    per_million: f64,
    target: i64,
    admitted: u64,
    hits: u64,
    window_start: Instant,
    window: Duration,
}

impl Yield {
    pub(crate) fn new(per_million: f64, target: i64, window: Duration) -> Self {
        Self {
            per_million,
            target,
            admitted: 0,
            hits: 0,
            window_start: Instant::now(),
            window,
        }
    }

    pub(crate) fn admit(&mut self) {
        self.admitted += 1;
    }

    pub(crate) fn result(&mut self, best: i64) {
        self.hits += u64::from(best <= self.target);
    }

    pub(crate) fn observation(&self) -> (f64, f64) {
        let lambda = self.admitted as f64 * self.per_million / 1e6;
        // One-sided 95 percent Poisson lower bound: (sqrt(lambda) - 0.98)^2,
        // and zero once sqrt(lambda) is below 0.98 rather than the square of a
        // negative difference.
        (self.hits as f64, (lambda.sqrt() - 0.98).max(0.0).powi(2))
    }

    pub(crate) fn tick(&mut self, now: Instant) -> Action {
        if now.saturating_duration_since(self.window_start) < self.window {
            return Action::None;
        }
        let (hits, lower) = self.observation();
        let action = if self.admitted > 0 && hits < lower {
            Action::Loosen { raise_audit: true }
        } else {
            Action::None
        };
        self.admitted = 0;
        self.hits = 0;
        self.window_start = now;
        action
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raised_auditing_never_reduces_frequency() {
        let mut lane = AuditLane::new(0.05);
        lane.raise();
        for base in [10, 50, 200] {
            assert_eq!(lane.denominator(base), base.min(50));
        }
    }

    #[test]
    fn yield_raised_audits_require_200_new_observations_before_restore() {
        let mut lane = AuditLane::new(0.05);
        for _ in 0..1000 {
            assert_eq!(lane.record(false), Action::None);
        }
        lane.raise();
        for _ in 0..199 {
            assert_eq!(lane.record(false), Action::None);
        }
        assert_eq!(lane.record(false), Action::Restore);
    }

    #[test]
    fn a_window_expecting_under_one_hit_never_loosens() {
        // lambda = 0.5: the lower bound is zero, so zero hits is inside it.
        let mut check = Yield::new(1.0, -100, Duration::from_secs(3600));
        for _ in 0..500_000 {
            check.admit();
        }
        assert_eq!(check.observation(), (0.0, 0.0));
        assert_eq!(check.tick(check.window_start + check.window), Action::None);
    }

    #[test]
    fn empty_yield_window_does_not_loosen() {
        let mut check = Yield::new(1.0, -100, Duration::from_secs(3600));
        assert_eq!(check.tick(check.window_start + check.window), Action::None);
    }

    #[test]
    fn audit_lane_holds_at_the_expected_rate() {
        let mut lane = AuditLane::new(0.05);
        let mut s = 12345u64;
        for _ in 0..5_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            assert_eq!(lane.record(s % 100 < 5), Action::None);
        }
    }

    #[test]
    fn audit_lane_loosens_on_a_burst_then_restores() {
        let mut lane = AuditLane::new(0.01);
        for _ in 0..19 {
            assert_eq!(lane.record(true), Action::None);
        }
        assert_eq!(lane.record(true), Action::Loosen { raise_audit: true });
        assert_eq!(lane.denominator(200), 50);
        for _ in 0..199 {
            assert_eq!(lane.record(false), Action::None);
        }
        assert_eq!(lane.record(false), Action::Restore);
        assert_eq!(lane.denominator(200), 200);
    }

    fn uniform(s: &mut u64) -> f64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        ((*s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    #[test]
    fn drift_fires_once_per_window_when_skew_moves() {
        let mut drift = Drift::new(0.0, 0.0);
        let mut m = Moments::default();
        let mut s = 7;
        for n in 1..=20_000 {
            m.push(-uniform(&mut s).ln());
            let expected = if n % 10_000 == 0 {
                Action::LoosenAndDoubleK0
            } else {
                Action::None
            };
            assert_eq!(drift.check(&m), expected);
        }
    }

    #[test]
    fn drift_is_quiet_on_the_calibrated_shape() {
        let mut drift = Drift::new(0.0, 0.0);
        let mut m = Moments::default();
        let mut s = 1;
        for _ in 0..50_000 {
            let x = (-2.0 * uniform(&mut s).ln()).sqrt()
                * (std::f64::consts::TAU * uniform(&mut s)).cos();
            m.push(x);
            assert_eq!(drift.check(&m), Action::None);
        }
    }

    #[test]
    fn yield_below_the_lower_bound_loosens_and_raises_audit() {
        for hits in [0, 10] {
            let mut check = Yield::new(1.0, -100, Duration::from_secs(3600));
            for _ in 0..10_000_000 {
                check.admit();
            }
            for _ in 0..hits {
                check.result(-100);
            }
            check.result(-99);
            let now = check.window_start + check.window;
            assert_eq!(check.tick(now - Duration::from_secs(1)), Action::None);
            assert_eq!(
                check.tick(now),
                if hits == 0 {
                    Action::Loosen { raise_audit: true }
                } else {
                    Action::None
                }
            );
            assert_eq!(check.admitted, 0);
            assert_eq!(check.hits, 0);
        }
    }
}
