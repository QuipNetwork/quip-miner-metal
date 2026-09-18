# Sweep count at 64 reads against the testnet's own blocks, 2026-09-18

## Result

| Reads | Sweeps | Jobs per second | P(valid) | Valid proofs per second | P(beats winner) | Winner-beating proofs per second |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1,024 | 90.61 | 0.000 | 0.00 | 0.000 | 0.00 |
| 64 | 2,048 | 69.67 | 0.000 | 0.00 | 0.000 | 0.00 |
| 64 | 4,096 | 41.50 | 0.060 | 2.49 | 0.030 | 1.24 |
| 64 | 8,192 | 22.49 | 0.230 | 5.17 | 0.097 | 2.17 |
| 64 | 16,384 | 12.19 | 0.403 | 4.92 | 0.203 | 2.48 |
| 128 | 1,024 | 44.96 | 0.000 | 0.00 | 0.000 | 0.00 |
| 128 | 2,048 | 33.94 | 0.023 | 0.79 | 0.013 | 0.45 |
| 128 | 4,096 | 22.05 | 0.110 | 2.43 | 0.057 | 1.25 |
| 128 | 8,192 | 12.64 | 0.293 | 3.71 | 0.150 | 1.90 |
| **128** | **16,384** | **6.24** | **0.540** | **3.37** | **0.320** | **2.00** |

The last row is the current default and the baseline for every comparison.
Each probability cell is 300 jobs: 60 regenerated testnet blocks, qblocks
3,191 to 3,250, five seeds each. Jobs per second is the best of 7 rounds at
each point, and 12 rounds at 16,384 sweeps. The definitions of valid and
beats-winner come from `docs/perf/2026-09-18-testnet-reads-study.md`.

Solution quality at each point:

| Reads | Sweeps | Mean reads below target, valid jobs | Median best energy, milli | Mean gap to winner, milli | Blocks valid on all 5 seeds | Blocks valid on none |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1,024 | 0.00 | -14,540,000 | 84,313 | 0 | 60 |
| 64 | 2,048 | 0.00 | -14,572,000 | 54,240 | 0 | 60 |
| 64 | 4,096 | 1.33 | -14,593,000 | 32,567 | 1 | 50 |
| 64 | 8,192 | 2.22 | -14,608,000 | 19,120 | 5 | 31 |
| 64 | 16,384 | 4.09 | -14,618,000 | 9,127 | 12 | 19 |
| 128 | 1,024 | 0.00 | -14,548,000 | 78,027 | 0 | 60 |
| 128 | 2,048 | 1.14 | -14,578,000 | 48,147 | 0 | 56 |
| 128 | 4,096 | 1.64 | -14,598,000 | 28,187 | 1 | 45 |
| 128 | 8,192 | 3.09 | -14,612,000 | 13,887 | 6 | 25 |
| 128 | 16,384 | 5.82 | -14,622,000 | 3,713 | 18 | 9 |

## The answer

One sweep count below 16,384 matches the baseline at 64 reads, and that is
8,192. It produces 1.53 times the baseline's valid proofs per second and
1.07 times the baseline's winner-beating proofs per second.

| Configuration | Valid proofs per second against baseline | Winner-beating proofs per second against baseline |
| --- | ---: | ---: |
| 64 reads, 4,096 sweeps | 0.72 [0.34, 1.20] | 0.58 [0.10, 1.23] |
| 64 reads, 8,192 sweeps | 1.53 [1.19, 1.87] | 1.07 [0.65, 1.54] |
| 64 reads, 16,384 sweeps | 1.46 [1.27, 1.63] | 1.24 [1.02, 1.44] |

The brackets hold a 90% interval from 10,000 paired resamples of the 60
blocks. The same blocks carry every cell, so the resample draws blocks and
scores every cell on the drawn blocks.

Three findings sit inside that table.

The gain at 8,192 sweeps is a gain in valid proofs only. Its interval for
valid proofs per second, 1.19 to 1.87, stays above parity. Its interval for
winner-beating proofs per second, 0.65 to 1.54, spans parity, so the
measurement does not separate 8,192 sweeps from the baseline on the rate
that decides which proof wins a chain block.

Cutting sweeps is not where the gain comes from. Cutting reads is. At 64
reads and the full 16,384 sweeps the miner already returns 1.46 times the
valid proofs per second and 1.24 times the winner-beating proofs per second,
and the second of those intervals stays above parity. Moving from 16,384 to
8,192 sweeps at 64 reads then adds 5% to the valid rate. The interval cannot
resolve a gain that small. The same move takes 14% off the winner-beating
rate.

Nothing at or below 4,096 sweeps is competitive. At 64 reads, 4,096 sweeps
returns 0.72 of the baseline's valid proofs per second, and 1,024 and 2,048
sweeps return none.

The choice that the measurement supports is 64 reads at 16,384 sweeps, which
beats the baseline on both rates. If the goal is valid proofs per second
alone, 64 reads at 8,192 sweeps is the highest point measured.

## How the chance per job falls

The chance per job falls smoothly across the first halving of sweeps, then
accelerates, then collapses.

| Reads | Sweeps | P(valid) | Share kept from the sweep count above |
| ---: | ---: | ---: | ---: |
| 64 | 16,384 | 0.403 | |
| 64 | 8,192 | 0.230 | 0.57 |
| 64 | 4,096 | 0.060 | 0.26 |
| 64 | 2,048 | 0.000 | 0.00 |
| 128 | 16,384 | 0.540 | |
| 128 | 8,192 | 0.293 | 0.54 |
| 128 | 4,096 | 0.110 | 0.38 |
| 128 | 2,048 | 0.023 | 0.21 |
| 128 | 1,024 | 0.000 | 0.00 |

The first halving costs 43% to 46% of the chance at both read counts. The
second costs 62% to 74%. The third leaves 0 valid proofs in 300 jobs at 64
reads and 7 in 300 at 128 reads. The cliff sits between 4,096 and 2,048
sweeps.

The block counts show the shape behind those numbers. At 64 reads and 8,192
sweeps, 31 of 60 blocks produce no valid proof on any of 5 seeds, against 9
of 60 at the baseline. A block is either within reach at a sweep count or
outside it, and sweeps move that boundary. This matches the reads finding in
the earlier study: the seeds agree with each other far more than the blocks
do.

## Reads below target against a min_solutions of 5

The network ran `min_solutions: 1` when the earlier study fetched the
blocks. The
`DifficultyConfig` default is 5. Under that default a proof needs 5 reads
below the target, not 1, and the ordering of the table inverts for 8,192
sweeps.

| Reads | Sweeps | Mean reads below target, valid jobs | Share of all jobs with 5 or more reads below target | Proofs per second under a min_solutions of 5 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 4,096 | 1.33 | 0.000 | 0.00 |
| 64 | 8,192 | 2.22 | 0.030 | 0.67 |
| 64 | 16,384 | 4.09 | 0.113 | 1.38 |
| 128 | 4,096 | 1.64 | 0.003 | 0.07 |
| 128 | 8,192 | 3.09 | 0.053 | 0.67 |
| 128 | 16,384 | 5.82 | 0.193 | 1.21 |

A valid 64-read job at 8,192 sweeps carries 2.22 reads below target,
against 5.82 for the baseline. Under a `min_solutions` of 5 it
delivers 0.67 proofs per second against the baseline's 1.21, which is 0.55
of the baseline. The 1.53 advantage in the result table turns into a 0.45
deficit. 64 reads at 16,384 sweeps holds up under both settings, at 1.38
proofs per second against 1.21.

## Method

### Chance per job

`scripts/testnet/run_reads_study.py`, unedited, at 64 and 128 reads, five
seeds per block and sweep count, eight processes at a time. Each of the four
runs covered 600 jobs and took 183 to 210 seconds. The 16,384-sweep rows come
from the existing 1,200-row CSV of the earlier study and were not re-run. All
2,400 new rows returned without an error.

```sh
for S in 1024 2048 4096 8192; do
  READS=64,128 SEEDS=5 SWEEPS=$S WIDTH=8 \
    scripts/testnet/run_reads_study.py \
    target/release/quip-metal-msa problems/ sweeps/study-$S.csv
done
```

### Jobs per second

`tests/msa_bench.rs`, unchanged, through its prebuilt test binary under
`target/release/deps/`. Every configuration holds 160 replica-words: 80
jobs at 64 reads and 40 jobs at 128 reads.
`scripts/testnet/sweeps/run_bench.py` puts each value into the child
process environment directly, so the zsh word-splitting trap recorded in
`docs/perf/2026-09-18-gpu-reads-and-models.md` cannot apply. The driver then
parses the harness's own report line and aborts if the reads, sweeps, or jobs
it reports differ from the values requested. No run aborted.

```sh
# one process per configuration, driven by run_bench.py
QUIP_BENCH_JOBS=80 QUIP_BENCH_READS=64  QUIP_BENCH_SWEEPS=S QUIP_BENCH_KERNEL=msa \
  target/release/deps/msa_bench-<hash> --ignored --nocapture
QUIP_BENCH_JOBS=40 QUIP_BENCH_READS=128 QUIP_BENCH_SWEEPS=S QUIP_BENCH_KERNEL=msa \
  target/release/deps/msa_bench-<hash> --ignored --nocapture

scripts/testnet/sweeps/run_bench.py target/release/deps/msa_bench-<hash> sweeps/bench.jsonl 7
```

The run behind this report went in three sets, two rounds, then five, then
five more at 16,384 sweeps alone, concatenated into `bench.jsonl`.

The task named 2,048 to 16,384 sweeps for the bench. This run added 1,024
sweeps, at a cost of about one second per round. That point is the only way
the 1,024-sweep chance per job becomes a rate.

Two rounds were not enough. The machine carried a desktop load throughout,
and the same configuration varied by a factor of 2 to 3 between rounds. Seven
rounds went to every point and twelve to the 16,384-sweep points. Interference
only slows a run down, so the table reports the fastest round at each point.
The fastest rounds agree with the earlier measurement: 6.24 jobs per second at 128
reads and 12.19 at 64 reads, against 6.40 to 7.34 and 12.24 to 12.84 recorded
in `docs/perf/2026-09-18-gpu-reads-and-models.md`.

The median round gives a floor. Under median rates, 64 reads at 8,192 sweeps
returns 1.33 times the baseline's valid proofs per second and 0.94 times its
winner-beating proofs per second, and 64 reads at 16,384 sweeps returns 1.33
and 1.13. The direction of every conclusion stated earlier holds under both
estimators. `summary.csv` carries both.

### Aggregation

```sh
scripts/testnet/sweeps/aggregate.py sweeps/ study.csv   # -> sweeps/summary.csv
scripts/testnet/sweeps/bootstrap.py sweeps/ study.csv   # -> sweeps/bootstrap.json
```

`study.csv` is the earlier reads study's output, which supplies the
16,384-sweep rows. The standard error on each probability clusters on the
block, because the five seeds of one block are not five independent draws.
`summary.csv` carries the clustered standard errors, which run from 0.023
to 0.051 on P(valid).

## What failed

The first bench set ran into an Apple Neural Engine job. The `ane-grid.done`
marker appeared at 09:06:58 and the study runs went ahead. A different
`quip_miner_ane` test process, PID 82068, started later and held
`ANECompilerService` at 98% of a core through the first bench set. That set
reported 4.43 jobs per second at 128 reads and 16,384 sweeps, 29% below the
documented rate, and it showed a super-linear slowdown from 8,192 to 16,384
sweeps that no later set reproduced. That set is kept apart from
`bench.jsonl`, and no number in this report uses it. The neural engine process exited at 09:24:08, and every reported
round ran after it.

Per-job wall time in the study CSVs is flat at 2.37 to 2.78 seconds across
every sweep count, including 1,024 sweeps. Process start dominates the
`--solve` path at this problem size. The rows stay in the CSVs as
instructed, but they carry no throughput signal.

## Blocked and needs decision

The `min_solutions` field decides whether 8,192 sweeps is usable at all. At
the current setting of 1 it returns 1.53 times the baseline's valid proofs
per second. At the `DifficultyConfig` default of 5 it returns 0.55 of the
baseline. A read and sweep count chosen against the current setting needs
that field watched, and the miner has no check on it today.

The machine ran a desktop session, five agent processes, and intermittent
neural engine work throughout the throughput rounds. Rates for one
configuration varied by a factor of 2 to 3 between rounds. The fastest-round
figures match the earlier measurement, so the table stands, but a re-run on
an idle machine would tighten every rate and every interval in this report.

The study covers 60 blocks in one difficulty window, on one machine, at one
beta schedule, with 5 seeds per cell. The target tightened through the window
and keeps moving. These probabilities are a snapshot.
