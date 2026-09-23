# Probe cascade (`quip-metal-msa`)

Before a job spends its full sweep budget, the multi-spin kernel runs it
through short probe sweeps. The screen removes a probe that ranks below the
adaptive cutoff, and that probe returns its probe reads instead of the full
reads. A probe that ranks higher than the cutoff advances to the next probe
stage and then to the full budget. The screen keeps the most promising nonces, so
the GPU spends its time on full-budget work most likely to beat the chain
target.

The `quip-metal-msa` miner always screens Metal jobs through the cascade.
No configuration can turn the screen off. The coordinator sends no cascade key.
The simulated annealing and Gibbs binaries do not run the cascade.
The Apple Neural Engine sampler runs full-budget jobs on a separate path.

The keys documented here enter through the `backend_toml` that the
coordinator sends. Invalid keys keep the previous value and log a warning.
The calibration doc `docs/perf/2026-09-23-cascade-calibration.md` records why
the defaults are what they are.

## Keys

| key | default | valid range | notes |
|-----|---------|-------------|-------|
| `cascade_stages` | `[32, 256]` | 1 to 3 values, each greater than the previous, first `>= 1` | sweeps per probe stage |
| `cascade_keep` | `2000` | see the group rule | probe-to-full denominator |
| `cascade_keep_min` | `1000` | see the group rule | floor for the probe-to-full keep |
| `cascade_keep_max` | `30000` | see the group rule | ceiling for the probe-to-full keep |
| `cascade_audit` | `200` | `>= 2` | audit lane denominator |
| `cascade_target_milli` | none | any integer | chain target energy |
| `cascade_yield_per_million` | none | finite and `> 0` | expected nonces per million below target |

`cascade_keep`, `cascade_keep_min`, and `cascade_keep_max` check as one
group. The group passes when `cascade_keep_min >= 2`, `cascade_keep_min
<= cascade_keep`, and `cascade_keep <= cascade_keep_max`. A violation of any
of those three conditions rejects the whole group and keeps the previous
values.

The three keep values are probe-to-full denominators for the whole job.
The keep factor applies across the configured probe stages, so each of
`S` stages keeps the `S`-th root of that factor. At the defaults this is
`1 / 2000^(1/2)`, about 1 in 44.7 per stage. With the default two stages the
screen keeps about 1 in 2000 nonces per job. `cascade_stages` accepts one to
three probe stages.

`cascade_target_milli` and `cascade_yield_per_million` work together. The
yield check is active only when the coordinator sets both. It counts how many
admitted jobs return a best energy at or below `cascade_target_milli` against
the expected rate `cascade_yield_per_million`.

## Screened-out results

A job screened out at a probe stage returns the probe reads with a device
time equal to the sum of the probe stages it ran. The job still reports one
result. `SamplerMeta.sweeps` reports the original requested budget, not the
probe budget. `quip-solver-core` has no `sweeps_done` field yet, so for a
screened-out job the reported sweep count overstates the probes the device
actually ran. The count becomes exact only when the core adds `sweeps_done`.

## Model checks

Three checks widen the screen when the running cascade leaves its
calibration. Each check evaluates after enough observations and, when it
fires, logs a warning line with the stage, the check name, the action, the
observed and expected values, and the new denominators. The line is:
`cascade model check`.

- **Audit misses.** The cascade may audit a probe that fails the screen and
  run it a level deeper. A miss is an audited nonce whose next-stage best is
  at or below the median next-stage best of the kept nonces. When the
  observed miss rate exceeds the calibrated bound for long enough, the audit
  lane loosens the screen and raises the audit rate. The action line reports
  `audit misses`.

- **Distribution drift.** The stage-0 probe energy distribution must match
  the calibrated skew and excess kurtosis. When either moves past 0.1 with
  enough observations, the screen loosens and doubles `k0`. The action line
  reports `distribution drift`.

- **Yield.** The yield check measures how many admitted jobs beat
  `cascade_target_milli`. When the hits stay below a one-sided Poisson lower
  bound for a full window, the screen loosens and raises the audit rate. The
  action line reports `yield hits`.

The relay also logs a `cascade load window` info line once per minute with
the per-stage dispatch counts, the adaptive denominator, and the current
GPU busy estimate. A falling-behind stage and idle full-budget time feed a
controller that tightens and loosens each stage cutoff.
