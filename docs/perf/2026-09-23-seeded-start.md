# Seeded start from probe spins, 2026-09-23

Arm C of the cascade calibration. The test compares a cold full-budget anneal
against a run that keeps the probe's spins and reheats. Bead
`quip-miner-metal-bh1.3.8`.

## Result

A short reheated tail does not reach the depth of a cold anneal. At the full
budget, a reheat to beta 0.25 goes 7,290 milli deeper than cold, with a
standard error of 215 milli. A shorter full stage does not pass gate G5.
The G5 lever is GPU occupancy, which the resident-slot design addresses.

## Method

- Instances: the 2,000 nonces of
  `docs/perf/data/2026-09-23-calibration/C1-seeds.txt.gz`, which are the
  deepest 2,000 of the Arm B probe run. The Aglais topology and instance draw
  match `tests/probe_screen.rs`. The job seed of nonce `i` is `i`.
- Every job has 64 reads. Batches hold 20 problems, one batch in flight.
- Arms per nonce, all with the same job seed:
  - `probe`: 32 sweeps, standard schedule.
  - `cold`: 14,336 sweeps, standard schedule.
  - `seeded`: the standard 32-rung schedule followed by
    `geometric(beta_r, cold, tail)`, one anneal. The state at rung 32 equals
    the probe's final state.
- The default beta range was hot 0.0173, cold 6.517. Every `beta_r` lies
  inside it.
- Before the arms ran, a schedule override equal to the standard 32-rung
  schedule gave the same samples and energies as the standard encode.
- Every job passed the device-energy audit path.

The test is
`sampler::tests::seeded_start_from_probe_spins_against_cold_anneal`
(ignored). The command was:

```sh
QUIP_SEEDED_SEEDS=<C1 seeds, gunzipped> QUIP_SEEDED_OUT=seeded-start.csv \
  cargo test --release --lib \
  sampler::tests::seeded_start_from_probe_spins_against_cold_anneal \
  -- --ignored --exact --nocapture --test-threads=1
```

The run took 688 seconds. The load average was not recorded.

## Paired difference against cold

Each cell is the mean of `best(seeded) - best(cold)` over 2,000 nonces, in
milli, with the standard error in brackets. A negative value means the seeded
run is deeper.

| `beta_r` | tail 2,048 | tail 4,096 | tail 14,304 |
|---|---:|---:|---:|
| 0.25 | +25,261 (275) | +10,303 (249) | –7,290 (215) |
| 0.40 | +50,243 (351) | +28,663 (295) | +1,860 (252) |
| 0.60 | +116,548 (463) | +86,099 (414) | +45,498 (362) |

A tail of 14,304 sweeps plus the 32-sweep probe equals the cold budget.

Counts of nonces where the seeded run is deeper, equal, or shallower than
cold:

| `beta_r` | tail 2,048 | tail 4,096 | tail 14,304 |
|---|---|---|---|
| 0.25 | 36 / 21 / 1,943 | 299 / 98 / 1,603 | 1,487 / 134 / 379 |
| 0.40 | 5 / 0 / 1,995 | 27 / 17 / 1,956 | 798 / 154 / 1,048 |
| 0.60 | 0 / 0 / 2,000 | 0 / 0 / 2,000 | 9 / 3 / 1,988 |

Counts at or below the chain target of –14,625,068 milli: cold reached it on
1 nonce. The seeded run at `beta_r` 0.25 with the full tail reached it on 2.
No other seeded cell reached it.

## Findings

1. A lower reheat point does better at every tail length. Arm D measures
   `beta_r` 0.10 and 0.15 next to 0.25 with the resident-slot kernel.
2. The seeded start is a quality gain at the full budget, not a cost saving.
   The 7,290 milli gain exceeds the 1,000 milli gate of the calibration
   specification.
3. A tail of 2,048 or 4,096 sweeps stays more than 10,000 milli shallower
   than cold. A cascade cannot shorten its full stage this way.

## Data

`docs/perf/data/2026-09-23-seeded-start/`:

- `seeded-start.csv.gz`: one row per nonce and arm, columns
  `nonce,arm,beta_r,tail,sweeps,best,median,device_us`.
- `seeded-start.log`: the test output with the summary lines.
