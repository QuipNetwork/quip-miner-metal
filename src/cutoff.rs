// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Adaptive keep cutoff for one cascade stage.
//!
//! Three layers set the energy a nonce must reach to go on to the next stage:
//! a Gaussian anchor from running moments, an empirical tail from P² markers
//! blended in as tail observations accumulate, and a slow load term that the
//! cascade drives from its GPU-share estimate. Energies are lower-is-better,
//! so keeping 1 in `D` means keeping energies at or below the `1/D` quantile.

/// Running mean, variance, skew, and excess kurtosis (one-pass update of the
/// second to fourth central moments).
#[derive(Clone, Debug, Default)]
pub(crate) struct Moments {
    n: u64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
}

impl Moments {
    pub(crate) fn push(&mut self, x: f64) {
        let n1 = self.n as f64;
        self.n += 1;
        let n = self.n as f64;
        let delta = x - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1;
        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
    }

    pub(crate) fn count(&self) -> u64 {
        self.n
    }

    pub(crate) fn mean(&self) -> f64 {
        self.mean
    }

    pub(crate) fn sd(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            (self.m2 / (self.n as f64 - 1.0)).sqrt()
        }
    }

    pub(crate) fn skew(&self) -> f64 {
        if self.m2 <= 0.0 {
            0.0
        } else {
            (self.n as f64).sqrt() * self.m3 / self.m2.powf(1.5)
        }
    }

    pub(crate) fn excess_kurtosis(&self) -> f64 {
        if self.m2 <= 0.0 {
            0.0
        } else {
            self.n as f64 * self.m4 / (self.m2 * self.m2) - 3.0
        }
    }
}

/// Streaming estimate of one quantile with five markers (Jain and Chlamtac's
/// P² algorithm). Constant memory, no stored samples.
#[derive(Clone, Debug)]
pub(crate) struct P2 {
    q: [f64; 5],
    pos: [f64; 5],
    want: [f64; 5],
    step: [f64; 5],
    seen: usize,
}

impl P2 {
    pub(crate) fn new(p: f64) -> Self {
        Self {
            q: [0.0; 5],
            pos: [1.0, 2.0, 3.0, 4.0, 5.0],
            want: [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0],
            step: [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0],
            seen: 0,
        }
    }

    pub(crate) fn push(&mut self, x: f64) {
        if self.seen < 5 {
            self.q[self.seen] = x;
            self.seen += 1;
            if self.seen == 5 {
                self.q.sort_by(f64::total_cmp);
            }
            return;
        }
        self.seen += 1;
        let k = if x < self.q[0] {
            self.q[0] = x;
            0
        } else if x >= self.q[4] {
            self.q[4] = x;
            3
        } else {
            (0..4).find(|&i| x < self.q[i + 1]).unwrap_or(3)
        };
        for i in (k + 1)..5 {
            self.pos[i] += 1.0;
        }
        for i in 0..5 {
            self.want[i] += self.step[i];
        }
        for i in 1..4 {
            let d = self.want[i] - self.pos[i];
            let room_up = self.pos[i + 1] - self.pos[i] > 1.0;
            let room_down = self.pos[i - 1] - self.pos[i] < -1.0;
            if (d >= 1.0 && room_up) || (d <= -1.0 && room_down) {
                let s = d.signum();
                let parabolic = self.parabolic(i, s);
                self.q[i] = if self.q[i - 1] < parabolic && parabolic < self.q[i + 1] {
                    parabolic
                } else {
                    self.linear(i, s)
                };
                self.pos[i] += s;
            }
        }
    }

    fn parabolic(&self, i: usize, s: f64) -> f64 {
        let (q, n) = (&self.q, &self.pos);
        q[i] + s / (n[i + 1] - n[i - 1])
            * ((n[i] - n[i - 1] + s) * (q[i + 1] - q[i]) / (n[i + 1] - n[i])
                + (n[i + 1] - n[i] - s) * (q[i] - q[i - 1]) / (n[i] - n[i - 1]))
    }

    fn linear(&self, i: usize, s: f64) -> f64 {
        let j = if s > 0.0 { i + 1 } else { i - 1 };
        self.q[i] + s * (self.q[j] - self.q[i]) / (self.pos[j] - self.pos[i])
    }

    pub(crate) fn estimate(&self) -> Option<f64> {
        (self.seen >= 5).then_some(self.q[2])
    }
}

/// Standard normal quantile (Acklam's rational approximation, relative
/// error under 1.2e-9).
pub(crate) fn inverse_normal_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const LOW: f64 = 0.024_25;
    let p = p.clamp(1e-300, 1.0 - 1e-16);
    let tail = |q: f64| {
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    if p < LOW {
        tail((-2.0 * p.ln()).sqrt())
    } else if p <= 1.0 - LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    }
}

/// Keep denominators the empirical layer tracks: four points spaced evenly in
/// `ln D` across the clamp, so every reachable denominator lies between two
/// markers. A denominator between two markers interpolates in `ln D`. The
/// default clamp gives 1,000, 3,107, 9,655, and 30,000.
fn marker_denominators(cfg: &CutoffConfig) -> [f64; 4] {
    let (lo, hi) = (cfg.keep_min.ln(), cfg.keep_max.ln());
    [0.0, 1.0, 2.0, 3.0].map(|i| (lo + i / 3.0 * (hi - lo)).exp())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CutoffConfig {
    /// Target keep denominator: keep 1 in `keep`.
    pub(crate) keep: f64,
    /// Loosest allowed denominator.
    pub(crate) keep_min: f64,
    /// Tightest allowed denominator.
    pub(crate) keep_max: f64,
    /// Expected tail observations at which the empirical layer and the warm-up
    /// both reach half weight. 2 equals the spec's `n0 = 2,000` samples at the
    /// loosest clamp of 1 in 1,000.
    pub(crate) k0: f64,
    /// Observations before any cutoff exists. Until then every nonce is
    /// screened out.
    pub(crate) min_samples: u64,
}

impl Default for CutoffConfig {
    fn default() -> Self {
        Self {
            keep: 10_000.0,
            keep_min: 1_000.0,
            keep_max: 30_000.0,
            k0: 2.0,
            min_samples: 200,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Cutoff {
    cfg: CutoffConfig,
    moments: Moments,
    markers: [P2; 4],
    /// The denominator each marker estimates, from [`marker_denominators`].
    marker_d: [f64; 4],
    /// Load layer, in `ln D`, clamped so `D` stays inside the clamp.
    log_adjust: f64,
    /// Model-check loosening, a factor at most 1 applied to `D`.
    loosen: f64,
}

impl Cutoff {
    pub(crate) fn new(cfg: CutoffConfig) -> Self {
        let marker_d = marker_denominators(&cfg);
        Self {
            cfg,
            moments: Moments::default(),
            markers: marker_d.map(|d| P2::new(1.0 / d)),
            marker_d,
            log_adjust: 0.0,
            loosen: 1.0,
        }
    }

    #[cfg(test)]
    pub(crate) fn reset(&mut self) {
        *self = Self::new(self.cfg);
    }

    pub(crate) fn observe(&mut self, x: f64) {
        self.moments.push(x);
        for m in &mut self.markers {
            m.push(x);
        }
    }

    pub(crate) fn moments(&self) -> &Moments {
        &self.moments
    }

    /// Keep 1 in this many. Starts at `keep_min` and moves toward `keep` in
    /// `ln D` as expected tail observations at the target accumulate, then
    /// takes the load and model-check adjustments, then the clamp.
    pub(crate) fn denominator(&self) -> f64 {
        let c = &self.cfg;
        let tail = self.moments.count() as f64 / c.keep;
        let warm = tail / (tail + c.k0);
        let ln_d = c.keep_min.ln()
            + warm * (c.keep.ln() - c.keep_min.ln())
            + self.log_adjust
            + self.loosen.ln();
        ln_d.exp().clamp(c.keep_min, c.keep_max)
    }

    pub(crate) fn cutoff(&self) -> Option<f64> {
        self.cutoff_at(self.denominator())
    }

    /// Energy at or below which a nonce is kept at denominator `d`.
    pub(crate) fn cutoff_at(&self, d: f64) -> Option<f64> {
        if self.moments.count() < self.cfg.min_samples {
            return None;
        }
        let p = 1.0 / d;
        let gauss = self.moments.mean() + inverse_normal_cdf(p) * self.moments.sd();
        let Some(emp) = self.empirical(d) else {
            return Some(gauss);
        };
        let tail = self.moments.count() as f64 * p;
        let w = tail / (tail + self.cfg.k0);
        Some((1.0 - w) * gauss + w * emp)
    }

    fn empirical(&self, d: f64) -> Option<f64> {
        let ln_d = d.ln();
        let m = &self.marker_d;
        let last = m.len() - 1;
        if d <= m[0] {
            return self.markers[0].estimate();
        }
        if d >= m[last] {
            return self.markers[last].estimate();
        }
        let i = (0..last).find(|&i| d <= m[i + 1]).unwrap_or(last - 1);
        let (lo, hi) = (self.markers[i].estimate()?, self.markers[i + 1].estimate()?);
        let t = (ln_d - m[i].ln()) / (m[i + 1].ln() - m[i].ln());
        Some(lo + t * (hi - lo))
    }

    /// Decide on `x` with the cutoff as it stood before `x`, then learn from it.
    pub(crate) fn decide(&mut self, x: f64) -> bool {
        let keep = self.cutoff().is_some_and(|c| x <= c);
        self.observe(x);
        keep
    }

    /// Halve the denominator, down to the loosest clamp.
    pub(crate) fn loosen_step(&mut self) -> f64 {
        let before = self.loosen;
        self.loosen = (self.loosen * 0.5).max(self.cfg.keep_min / self.cfg.keep_max);
        self.loosen / before
    }

    /// Undo only the effective factor returned by [`Self::loosen_step`].
    pub(crate) fn restore_step(&mut self, factor: f64) {
        self.loosen = (self.loosen / factor).min(1.0);
    }

    /// Give the Gaussian anchor more weight for longer.
    pub(crate) fn double_k0(&mut self) {
        self.cfg.k0 *= 2.0;
    }

    /// Current load integral, used by the relay's idle-error decay.
    pub(crate) fn log_adjust(&self) -> f64 {
        self.log_adjust
    }

    /// Load layer: positive `error` tightens, negative loosens.
    pub(crate) fn load_update(&mut self, error: f64, gain: f64) {
        let c = &self.cfg;
        let lo = (c.keep_min / c.keep).ln();
        let hi = (c.keep_max / c.keep).ln();
        self.log_adjust = (self.log_adjust + gain * error).clamp(lo, hi);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normal_stream(seed: u64, mean: f64, sd: f64) -> impl Iterator<Item = f64> {
        // Box-Muller on a xorshift stream: deterministic, no dependency.
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        std::iter::from_fn(move || {
            let (u, v) = (next(), next());
            Some(mean + sd * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos())
        })
    }

    #[test]
    fn inverse_normal_matches_known_quantiles() {
        assert!(inverse_normal_cdf(0.5).abs() < 1e-9);
        assert!((inverse_normal_cdf(0.025) + 1.959_963_985).abs() < 1e-6);
        assert!((inverse_normal_cdf(1e-3) + 3.090_232_306).abs() < 1e-6);
        assert!((inverse_normal_cdf(1e-4) + 3.719_016_485).abs() < 1e-6);
        assert!((inverse_normal_cdf(0.975) - 1.959_963_985).abs() < 1e-6);
    }

    #[test]
    fn moments_of_a_normal_stream() {
        let mut m = Moments::default();
        normal_stream(7, -40_000.0, 500.0)
            .take(200_000)
            .for_each(|x| m.push(x));
        assert!((m.mean() + 40_000.0).abs() < 5.0);
        assert!((m.sd() - 500.0).abs() < 5.0);
        assert!(m.skew().abs() < 0.03);
        assert!(m.excess_kurtosis().abs() < 0.06);
    }

    #[test]
    fn p2_tracks_a_low_quantile() {
        let mut q = P2::new(1e-3);
        normal_stream(11, 0.0, 1.0)
            .take(500_000)
            .for_each(|x| q.push(x));
        let got = q.estimate().unwrap();
        assert!((got - inverse_normal_cdf(1e-3)).abs() < 0.1, "{got}");
    }

    #[test]
    fn warm_up_screens_everything_out() {
        let mut c = Cutoff::new(CutoffConfig::default());
        let kept = normal_stream(3, 0.0, 1.0)
            .take(199)
            .filter(|&x| c.decide(x))
            .count();
        assert_eq!(kept, 0);
        assert!(c.cutoff().is_none());
        assert!(!c.decide(-100.0));
        assert_eq!(c.moments().count(), 200);
        assert!(c.cutoff().is_some());
    }

    #[test]
    fn denominator_starts_loose_and_tightens_to_the_target() {
        let mut c = Cutoff::new(CutoffConfig::default());
        c.observe(0.0);
        assert!(c.denominator() < 1_100.0);
        normal_stream(5, 0.0, 1.0)
            .take(2_000_000)
            .for_each(|x| c.observe(x));
        let d = c.denominator();
        assert!(d > 9_000.0 && d <= 10_000.0, "{d}");
    }

    #[test]
    fn markers_span_a_per_stage_clamp_and_hold_its_keep_rate() {
        // One of three probe stages under an overall 1 in 3,000 keep with a
        // [1,000, 30,000] clamp: each bound is the cube root of the overall one.
        let root = |d: f64| d.powf(1.0 / 3.0);
        let cfg = CutoffConfig {
            keep: root(3_000.0),
            keep_min: root(1_000.0),
            keep_max: root(30_000.0),
            ..CutoffConfig::default()
        };
        let m = marker_denominators(&cfg);
        assert!((m[0] - cfg.keep_min).abs() < 1e-9 && (m[3] - cfg.keep_max).abs() < 1e-9);
        let mut c = Cutoff::new(cfg);
        let n = 200_000usize;
        let mut kept = 0usize;
        for (i, x) in normal_stream(11, 0.0, 1.0).take(n).enumerate() {
            if c.decide(x) && i >= n / 2 {
                kept += 1;
            }
        }
        let rate = kept as f64 / (n / 2) as f64;
        let want = 1.0 / cfg.keep;
        assert!((rate - want).abs() < 0.05 * want, "{rate} vs {want}");
    }

    #[test]
    fn keep_rate_converges_near_the_denominator() {
        let mut c = Cutoff::new(CutoffConfig {
            keep: 1_000.0,
            ..CutoffConfig::default()
        });
        let mut kept = 0usize;
        let n = 2_000_000usize;
        for (i, x) in normal_stream(9, 0.0, 1.0).take(n).enumerate() {
            let keep = c.decide(x);
            if i >= n / 2 && keep {
                kept += 1;
            }
        }
        let rate = kept as f64 / (n / 2) as f64;
        assert!((rate - 1e-3).abs() < 2e-4, "{rate}");
    }

    #[test]
    fn clamps_hold_under_load_feedback() {
        let mut c = Cutoff::new(CutoffConfig::default());
        normal_stream(1, 0.0, 1.0)
            .take(100_000)
            .for_each(|x| c.observe(x));
        for _ in 0..1_000 {
            c.load_update(5.0, 1.0);
        }
        // The integral saturates at ln(30_000 / 10_000) on top of the warm-up
        // interpolation; without that clamp the outer clamp would give 30_000.
        let warm = 10.0 / (10.0 + 2.0);
        let saturated = (1_000f64.ln() + warm * 10f64.ln() + 3f64.ln()).exp();
        let d = c.denominator();
        assert!(
            (d - saturated).abs() < 1e-6 * saturated,
            "{d} vs {saturated}"
        );
        for _ in 0..1_000 {
            c.load_update(-5.0, 1.0);
        }
        assert!(c.denominator() >= 1_000.0);
    }

    #[test]
    fn restoring_the_applied_factor_preserves_drift_exactly() {
        let mut c = Cutoff::new(CutoffConfig {
            keep: 3000.0f64.cbrt(),
            keep_min: 1000.0f64.cbrt(),
            keep_max: 30000.0f64.cbrt(),
            ..CutoffConfig::default()
        });
        assert_eq!(c.loosen_step(), 0.5);
        let drift_factor = c.loosen;
        let audit_factor = c.loosen_step();
        assert!(audit_factor > 0.5 && audit_factor < 1.0);
        assert_eq!(c.loosen_step(), 1.0);
        c.restore_step(audit_factor);
        assert_eq!(c.loosen, drift_factor);
    }

    #[test]
    fn loosen_step_halves_the_denominator_and_restore_undoes_it() {
        let mut c = Cutoff::new(CutoffConfig::default());
        normal_stream(2, 0.0, 1.0)
            .take(2_000_000)
            .for_each(|x| c.observe(x));
        let d = c.denominator();
        let factor = c.loosen_step();
        assert!((c.denominator() - d / 2.0).abs() < 1.0);
        c.restore_step(factor);
        assert!((c.denominator() - d).abs() < 1.0);
    }

    #[test]
    fn reset_forgets_the_distribution() {
        let mut c = Cutoff::new(CutoffConfig::default());
        normal_stream(4, 0.0, 1.0)
            .take(10_000)
            .for_each(|x| c.observe(x));
        c.reset();
        assert_eq!(c.moments().count(), 0);
        assert!(c.cutoff().is_none());
    }

    #[test]
    fn doubling_k0_slows_tightening_and_favors_the_gaussian_anchor() {
        let mut c = Cutoff::new(CutoffConfig {
            min_samples: 5,
            ..CutoffConfig::default()
        });
        for x in [0.0, 1.0, 2.0, 3.0, 4.0] {
            c.observe(x);
        }
        let d = c.denominator();
        let before = c.cutoff_at(1_000.0).unwrap();
        // The five samples have mean 2 and sample variance 2.5.
        let gauss = 2.0 - 3.090_232_306 * 2.5_f64.sqrt();
        c.double_k0();
        let after = c.cutoff_at(1_000.0).unwrap();
        assert!(c.denominator() < d);
        assert!((after - gauss).abs() < (before - gauss).abs());
    }
}
