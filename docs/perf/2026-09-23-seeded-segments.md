# Seeded segments for the resident cascade, 2026-09-23

Arm D of the resident-slot plan. It sets the reheat beta and the stage list
for a cascade whose kept jobs continue from their gate state. Bead
`quip-miner-metal-bh1.3.8.4`.

## Result

- The reheat beta is 0.25. It gives the deepest final energy of the three
  values on both sets.
- The stages are `[32, 256]`. The gate at 256 sweeps ranks the final energy
  better than the gate at 32 sweeps (Spearman 0.912 against 0.774), and the
  second reheat costs no final depth.
- On the 2,000 deepest nonces, schedule `B` at beta 0.25 goes 6,955 milli
  deeper than a cold anneal of the same budget (standard error 224).

## Method

Every job runs 64 reads through the slot pool, 14,336 rungs, one rung per
sweep, in slices of 32 rungs. No gate removes a job, and the harness records
the best energy at every checkpoint. `cold` is the cold end of
`default_ising_beta_range` for the instance.

| schedule | segments | checkpoints |
|---|---|---|
| `A` | standard 32 rungs, then `geometric(reheat, cold, 14,304)` | 32, 14,336 |
| `B` | standard 32 rungs, then `geometric(reheat, cold, 224)`, then `geometric(reheat, cold, 14,080)` | 32, 256, 14,336 |

Each schedule ran at reheat beta 0.10, 0.15, and 0.25 on two sets of 2,000
nonces:

- `random`: the first 2,000 nonces of the `probe_screen` draw with run seed
  20260918, the draw of the 2026-09-23 calibration Arm A.
- `deep`: `docs/perf/data/2026-09-23-calibration/C1-seeds.txt.gz`, the 2,000
  deepest nonces of that calibration's Arm B.

The job seed of nonce `i` is `i`. The instance draw and the Advantage2
topology match `tests/probe_screen.rs`.

The harness is `slots::tests::seeded_segments_calibration` (ignored):

```sh
QUIP_SEGMENTS_NONCES=2000 QUIP_SEGMENTS_REHEAT=0.10,0.15,0.25 \
  QUIP_SEGMENTS_OUT=<dir> [QUIP_SEGMENTS_SEEDS=<seed file>] \
  cargo test --release --lib slots::tests::seeded_segments_calibration \
  -- --exact --ignored --nocapture --test-threads=1
```

Checks:

- For every nonce and reheat, the 32-sweep best of `A` equals that of `B`.
  The two schedules share that prefix.
- The harness ran twice, before and after the fix that adds the `seed`
  column. Every energy in all 12 files was the same in both runs.
- The random set took 612 seconds at a load average of 6.46. The deep set
  took 580 seconds at a load average of 4.52.

The wall times describe a calibration run with no gate and no controller.
They are outside the throughput rules of `AGENTS.md`.

## Reheat beta

The mean of the final best energy, in milli. The difference column pairs each
cell with `A` at beta 0.25 by nonce, with the standard error in brackets. A
positive difference is shallower.

| set | schedule | beta 0.10 | beta 0.15 | beta 0.25 |
|---|---|---:|---:|---:|
| random | `A` | +2,935 (213) | +1,934 (205) | 0 |
| random | `B` | +3,519 (207) | +1,995 (204) | +332 (201) |
| deep | `A` | +2,269 (198) | +1,411 (198) | 0 |
| deep | `B` | +2,392 (206) | +1,467 (202) | −246 (202) |

The mean final best on the random set at beta 0.25 is −14,396,109 milli for
`A` and −14,395,777 for `B`. The next value, 0.15, is 1,934 milli shallower,
more than nine standard errors. The seeded-start study measured beta 0.40
about 9,000 milli shallower than 0.25, so the best value lies near 0.25.

## Stages

Spearman rank correlation of each checkpoint against the final best, schedule
`B`, random set:

| beta | 32 sweeps | 256 sweeps | gain |
|---|---:|---:|---:|
| 0.10 | 0.775 | 0.899 | 0.124 |
| 0.15 | 0.776 | 0.907 | 0.131 |
| 0.25 | 0.774 | 0.912 | 0.138 |

The gain exceeds 0.05 at every beta, so the gate at 256 sweeps stays. On the
deep set, the correlations are 0.507 and 0.731 at beta 0.25. That set is a
narrow slice of the population, so its correlations are lower.

The difference between `B` and `A` at beta 0.25 is within one standard error
on both sets. The gate at 256 sweeps costs no final depth.

## Against a cold anneal

For the deep set, the paired difference of the final best against the cold
column of `docs/perf/data/2026-09-23-calibration/C1.csv.gz`, joined by seed.
A negative value is deeper than cold.

| schedule | beta | difference | deeper / equal / shallower |
|---|---:|---:|---|
| `A` | 0.10 | −4,440 (224) | 1,262 / 149 / 589 |
| `A` | 0.15 | −5,298 (223) | 1,346 / 140 / 514 |
| `A` | 0.25 | −6,709 (216) | 1,466 / 115 / 419 |
| `B` | 0.10 | −4,317 (220) | 1,262 / 160 / 578 |
| `B` | 0.15 | −5,242 (221) | 1,344 / 136 / 520 |
| `B` | 0.25 | −6,955 (224) | 1,467 / 133 / 400 |

The cold column has no nonce at or below the chain target of −14,625,068
milli. Every seeded cell has one.

## Constants for the controller

From schedule `B` at beta 0.25, random set.

| constant | value | before |
|---|---|---|
| reheat beta | 0.25 | none |
| stages | 32, 256 sweeps | 32, 256 sweeps |
| skew at 32 and 256 sweeps | −0.123, −0.056 | −0.147, −0.091 |
| excess kurtosis at 32 and 256 sweeps | 0.014, 0.002 | 0.041, 0.016 |
| audit miss rate `r0`, 32 to 256 | 0.0353 | 0.0348 |
| audit miss rate `r0`, 256 to 14,336 | 0.0051 | 0.0357 |

Both `r0` values use the per-stage keep of 1 in 44.7 from the 2026-09-23
calibration, 45 of 2,000 nonces. They use the definition of
`cascade_calibrate.py`: the share of rejected nonces whose next-stage best is
at or below the median of the kept nonces. The script reports keeps of
1 in 1,000 and tighter, which keep 1 or 2 nonces of 2,000.

The earlier second `r0` came from independent anneals and 26 kept nonces.
Here the final segment continues from the state at 256 sweeps. This run does
not separate that effect from the sampling error of 45 kept nonces.

## Data

`docs/perf/data/2026-09-23-seeded-segments/`:

- `random/` and `deep/`: `A-r0.10.csv.gz` to `B-r0.25.csv.gz`, one row per
  nonce, columns `nonce,seed,best_64x32[,best_64x256],best_64x14336`.
- `calibrate-*.json`: `cascade_calibrate.py` output for each random file.
- `summary.json`: the paired differences and correlations in this report.
- `random.log`, `deep.log`, `random.uptime`, `deep.uptime`: the harness output
  and the load average at the start of each set.
