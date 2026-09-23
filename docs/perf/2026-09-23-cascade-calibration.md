# Cascade calibration, 2026-09-23

This round measures the cascade shape on the part 1 miner before the cascade
is built. It sets the stage list, the keep target, the model-check constants,
and the cost model that the cascade relay reads.

## Result

The screen finds nonces below the chain target. Arm B screened 1,000,000 fresh
nonces through 32, 64, 128, and 256 sweeps. Of the 200 nonces that a
14,336-sweep run then ranked deepest, 3 reach the target of −14,625,068 milli
under a 256-read, 65,536-sweep solve. The deepest is −14,650,000. The 200
random control nonces reach no deeper than −14,536,000.

The cascade uses the stages [32, 128, 256] and then the full budget. It keeps
1 in 3,000 nonces overall, which is 1 in 14.4 at each probe stage.

Two findings miss their projections: the tail correlation and the normality
check. The misses change the stage list, the keep target, and `k0`. They do
not show a weak screen. The section on the tail correlation gives the reason.

## Arms

All runs use 64 reads, six instance-drawing threads, and the Advantage2
topology.

| arm | nonces | stages | purpose |
|---|---:|---|---|
| A | 50,000 | 32, 64, 128, 256 sweeps, every nonce | rank correlation against the committed 14,336-sweep column of the 2026-09-18 study, joined by nonce index |
| B | 1,000,000 | 32, 64, 128, 256 sweeps, keep 0.368 per stage | the tail and the normality check |
| C1 | 2,000 | 14,336 sweeps | the deepest 2,000 of B by the 256-sweep best |
| C2 | 400 | 256 reads × 65,536 sweeps | the deepest 200 of C1, plus 200 random B nonces as a control |

Arm A uses the run seed 20260918, the default of `tests/probe_screen.rs`.
That seed draws the same 50,000 instances as the 2026-09-18 study, so its
14,336-sweep column joins by nonce index. The kernel random stream changed in
part 1, so the probe energies are new measurements of the same instances.

Arm C seeded start did not run. The kernel has no initial-state input. The
seeded start stays under Later for that reason.

## Findings against spec table 3.1

| finding | measured | projection | action |
|---|---:|---:|---|
| Spearman, 32 sweeps against full | 0.768 | above 0.6 | none. The 32-sweep first stage stays |
| tail r on the deepest 500 | 0.372 | above 0.85 | a rung is added and the keep target loosens to 1 in 3,000 |
| normality at 1 in 100,000 (−4.26 sd) | −0.58 sd | within 0.15 sd | `k0` changes from 2 to 1 |
| probe rate | 4,803 to 4,873 jobs/s | the part 1 exit value, 5,128 | none |

Spearman against the full budget rises with the probe budget: 0.768 at 32
sweeps, 0.823 at 64, 0.866 at 128, and 0.899 at 256.

Arm B has 10 observations at 1 in 100,000, so that quantile is the least
certain of the three. At 1 in 1,000 and 1 in 10,000 the observed quantiles
are 0.25 and 0.45 standard deviations below the normal quantile. The lower
tail is heavier than normal. The skew of −0.14 at 32 sweeps agrees.

## Tail correlation

The spec projects a Pearson correlation above 0.85 between the last probe and
the deep reference on the 500 nonces the probe ranks deepest. The measured
value is 0.372.

Those 500 nonces are the deepest 1 in 2,000 by the probe. Selection on the
probe narrows the probe's range inside that group, and a narrow range lowers
Pearson r for any screen at this selectivity. A value near 0.85 was not
reachable. The spec's action for this miss still applies, because the spec is
the binding text. It adds a rung and loosens the keep target.

Two direct measurements show that the tail ranks well:

- In C2, all 20 of the deepest nonces come from the 200 probe picks. None
  come from the 200 random controls. The picks have a median of −14,592,000
  against −14,414,000 for the controls.
- In Arm A, a cascade simulation keeps 6 of the 10 deepest nonces by the full
  budget at 1 in 3,000 overall.

The spec's action says to loosen toward 1 in 1,000. The target is 1 in 3,000,
not 1 in 1,000. At 50,000 nonces the survival counts of the 10 deepest are 7,
6, and 4 at 1 in 1,000, 3,000, and 10,000. Those counts are within their own
noise. The chain target is a 1 in 100,000 to 1 in 400,000 event, and a
tighter keep protects the rarer tail. The clamp stays at 1 in 1,000 to 1 in
30,000, so the load controller can loosen to 1 in 1,000 when the full-budget
stage has idle GPU time.

## Cascade simulation on Arm A

Each row applies the keep to Arm A's 50,000 nonces. It counts how many of the
deepest 10 and 50 by the full budget survive. The cost is GPU time per
screened nonce from the cost model, with a 14,336-sweep full budget.

| overall keep | stages | kept | deepest 10 kept | deepest 50 kept | µs per nonce |
|---|---|---:|---:|---:|---:|
| 1 in 1,000 | 32 | 50 | 4 | 7 | 232 |
| 1 in 1,000 | 32, 256 | 51 | 7 | 21 | 255 |
| 1 in 1,000 | 32, 128, 256 | 51 | 7 | 20 | 282 |
| 1 in 3,000 | 32 | 17 | 1 | 3 | 209 |
| 1 in 3,000 | 32, 256 | 17 | 6 | 12 | 223 |
| 1 in 3,000 | 32, 128, 256 | 17 | 6 | 14 | 242 |
| 1 in 10,000 | 32, 256 | 5 | 3 | 5 | 209 |
| 1 in 10,000 | 32, 128, 256 | 6 | 4 | 6 | 223 |

One 32-sweep stage alone loses most of the deep nonces. A second or third
stage costs 15 to 50 µs per nonce and keeps several times as many.

## Constants for the cascade relay

| constant | value |
|---|---|
| probe stages | 32, 128, 256 sweeps |
| keep target, overall | 1 in 3,000 |
| keep per probe stage | 1 in 3,000^(1/3), about 1 in 14.4 |
| clamp, overall | 1 in 1,000 to 1 in 30,000 |
| `k0` | 1 |
| audit miss rate `r0`, 32 to 128 | 0.0581 |
| audit miss rate `r0`, 128 to 256 | 0.0496 |
| audit miss rate `r0`, 256 to full | 0.0804 |
| skew at 32, 128, 256 sweeps | −0.138, −0.086, −0.076 |
| excess kurtosis at 32, 128, 256 sweeps | 0.059, 0.026, 0.018 |
| cost per 64-read job | 122.3 µs + 2.367 µs × sweeps |

The audit miss rate follows the audit lane's definition. A miss is a rejected
nonce whose next-stage best is at or below the median next-stage best of the
kept nonces. The rates come from Arm A along the cascade path at 1 in 14.4
per stage.

The cost model is a least-squares fit to the timing of Arm A's four stages.
It predicts the measured rates within 5 percent from 32 to 256 sweeps and
within 7 percent at 14,336:

| sweeps | model, jobs/s | measured, jobs/s |
|---:|---:|---:|
| 32 | 5,050 | 4,803 to 4,873 |
| 64 | 3,650 | 3,576 to 3,631 |
| 128 | 2,350 | 2,375 to 2,437 |
| 256 | 1,373 | 1,328 to 1,362 |
| 14,336 | 29.4 | 27.5 |

## Conditions

The load averages at the start of each arm were 2.7 (A), 4.5 (B), 10.3 (C1),
and 5.5 (C2), from desktop applications. The energies do not depend on load.
The rates and the cost model do, and Arm A ran at the lowest load.

## Data

`docs/perf/data/2026-09-23-calibration/`:

- `A-best.csv.gz`, `B-best.csv.gz`: the best energy per nonce at each stage.
  A B nonce that a stage screened out has an empty cell for the later stages.
  The seed of a nonce follows from the run seed and the nonce index.
- `C1-seeds.txt.gz`, `C1.csv.gz`: the 2,000 nonces of arm C1 with their seeds.
- `C2-members.csv`, `C2.csv.gz`: the 400 nonces of arm C2 and their group.
- `report.json`: the output of `scripts/testnet/annealer/cascade_calibrate.py`.
- One log and one `uptime` file per arm.
