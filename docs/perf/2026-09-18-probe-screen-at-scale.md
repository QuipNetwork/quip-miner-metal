# A probe screen on 50,000 fresh nonces, 2026-09-18

The 800-nonce study in `2026-09-18-qpu-screen.md` showed that a short job
ranks fresh nonces by the energy a full job reaches. This run puts 50,000
fresh nonces through four budgets to answer the question that matters for a
miner: does the ranking hold for the deepest nonces, the ones that can reach
the target?

## Result

It holds, and the screen is worth 16.2 times the valid proofs per second of
an unscreened miner on one M4 Max, with a spread of 1.4 across ten rounds.

A 512-sweep probe at 64 reads ranks 50,000 fresh nonces against the full
14,336-sweep job at Spearman +0.923. Keeping the 390 nonces the probe ranks
deepest, 1 in 128, holds all 10 of the deepest by the full job and 46 of the
deepest 50. The screened miner examines 211 nonces per second against 13
for the unscreened one, and waits 3.4 seconds for its first answer against
2.8 seconds unscreened.

The probe picks instances whose energy floor is deep for any solver, rather
than instances that suit a 14,336-sweep anneal. On the 100 nonces the probe ranked deepest, a
solver at 256 reads and 65,536 sweeps, 18 times the effort, reaches a median
of −14,555,000 milli against −14,412,000 on 100 random nonces. All 20 of the
deepest nonces under that solver came from the probe's picks and none from
the random control. Its deepest is **−14,614,000**, which is 11,068 milli
short of the chain's target of −14,625,068. The random control's deepest
stops 103,068 short.

The accuracy of the screen is not what limits it. The probe's own rate is.
Keeping 1 in 128 already captures 87% of the speed an infinitely selective
screen would reach, and keeping less than that trades recall for almost
nothing.

## The distribution at 50,000 nonces

Every nonce ran at 64 reads and 14,336 sweeps, the shipped configuration.

| Statistic | Value, milli |
| --- | ---: |
| Median | −14,388,000 |
| Standard deviation | 47,303 |
| Deepest 5% | −14,466,000 |
| Deepest 1% | −14,500,000 |
| Deepest 0.1% | −14,538,000 |
| Deepest of 50,000 | −14,584,000 |
| Target | −14,625,068 |

None of the 50,000 was valid. The target sits 5.0 standard deviations below
the median and the deepest nonce reaches 4.1.

The distribution is normal to the edge of the sample, which is what lets the
next section extrapolate one standard deviation past it. Skew is −0.059 and
excess kurtosis −0.017. The observed quantiles track a normal distribution
within 0.14 standard deviations at every depth measured:

| Quantile | Observed, standard deviations | Normal |
| --- | ---: | ---: |
| 5% | −1.64 | −1.64 |
| 1% | −2.36 | −2.33 |
| 0.1% | −3.16 | −3.09 |
| 0.004% | −3.96 | −3.94 |
| Deepest of 50,000 | −4.13 | −4.26 |

## What a screened miner is worth

At 5.0 standard deviations a normal tail gives one nonce in 3.5 million. On
one M4 Max:

| Miner | Nonces examined per second | Expected time to one valid proof |
| --- | ---: | ---: |
| No screen, 64 reads, 14,336 sweeps | 13 | 74 hours |
| 512-sweep probe, keep 1 in 128 | 211 | 4.6 hours |

Those hours are an extrapolation one standard deviation beyond the deepest
of 50,000 nonces, not a measurement. The ratio between the two rows, 16.2,
rests only on the measured rates and the measured recall.

The composite rate is the series sum of the two stages: one probe for every
nonce, plus one full solve for every one hundred and twenty-eighth nonce. A
screen that keeps a smaller fraction approaches the probe's own rate of 247
nonces per second and no higher.

Dispatch each kept nonce as it arrives rather than waiting for a full solve
batch. The next section measures why.

## Choosing the probe and the keep fraction

Four budgets ran on every nonce, so one table compares them on the same
instances.

| Probe | Rate, nonces per second | Spearman against the full job | Best speedup | Keep fraction at the best speedup |
| --- | ---: | ---: | ---: | ---: |
| 64 reads, 512 sweeps | 247 ± 20 | +0.923 | 16.2 ± 1.4× | 1 in 128 |
| 64 reads, 1,024 sweeps | 152 ± 12 | +0.942 | 10.4 ± 0.6× | 1 in 128 |
| 64 reads, 4,096 sweeps | 44 ± 5 | +0.965 | 3.2 ± 0.4× | 1 in 128 |

The shortest probe wins. A longer probe ranks better and costs more, and the
cost grows faster than the ranking improves. The 800-nonce study could not
see this because it tested no probe below 1,024 sweeps.

Recall as the screen tightens, for the 512-sweep probe:

| Keep | Nonces kept of 50,000 | Probe cutoff, milli | Deepest 50 held | Deepest 10 held | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 in 32 | 1,562 | −14,366,000 | 50 of 50 | 10 of 10 | 11.7× |
| 1 in 64 | 781 | −14,380,000 | 50 of 50 | 10 of 10 | 14.4× |
| 1 in 128 | 390 | −14,396,000 | 46 of 50 | 10 of 10 | 16.2× |
| 1 in 256 | 195 | −14,408,000 | 39 of 50 | 10 of 10 | 17.3× |
| 1 in 512 | 97 | −14,418,000 | 32 of 50 | 9 of 10 | 18.0× |
| 1 in 1,024 | 48 | −14,430,000 | 25 of 50 | 9 of 10 | 18.3× |

Past 1 in 128 the speedup gains 12% in total while the deepest 50 lose more
than half their members. A miner sets the cutoff as an absolute energy, not
a rank, so the cutoff column is the rule to use: at 1 in 128, solve every
nonce whose 512-sweep probe reaches −14,396,000 or lower.

The screen has room for a nonce deeper than any in this sample. Fitting
probe energy on full energy gives a slope of 0.976 and a residual spread of
18,439 milli, so a nonce whose full job would reach the target reads about
−14,503,000 on the probe. That is 5.8 residual standard deviations below the
1-in-128 cutoff. The screen keeps such a nonce with near certainty.

## Does a stronger solver agree

A screen that only predicted what a 14,336-sweep anneal does would be
circular. The deep stage tests the ranking against a solver 18 times
stronger by reads and sweeps: 256 reads and 65,536 sweeps, at 0.79 nonces
per second, run on three sets of 100 nonces drawn from the same 50,000.

| Subset | Median at 14,336 sweeps | Median at 65,536 sweeps, 256 reads | Deepest |
| --- | ---: | ---: | ---: |
| 100 deepest by the probe | −14,532,000 | −14,555,000 | −14,614,000 |
| 100 deepest by the full job | −14,538,000 | −14,558,000 | −14,614,000 |
| 100 random | −14,388,000 | −14,412,000 | −14,522,000 |

The probe's picks and the full job's picks reach the same depth, 143,000
milli below the random control. Of the 20 deepest nonces under the strong
solver, counting the probe's picks and the random control together, all 20
came from the probe.

Within the random control, where the range is not restricted, the probe
predicts the strong solver's energy at Spearman +0.95 and the 14,336-sweep
job predicts it at +0.98. The probe is reading the instance, not the budget.
Inside the probe's own picks the correlation falls to +0.50, which is what
restricting the range to the deepest 0.2% does to a correlation and not a
failure of the screen.

The extra effort buys a median of 20,000 milli over the 14,336-sweep job and
lowers the energy on 98 of 100 nonces.

## Rates on the streaming path

Every rate here comes from one controlled test of five rounds. A round runs
all four budgets back to back in one process. The order of the budgets
rotates each round, so drift and ordering cannot look like a budget effect.
The machine was not quiet. Load averages ran from 3 to 20 on 16 cores, and
so each rate carries a spread.

| Budget | Streaming, nonces per second | Lead to the first solve | One process per nonce |
| --- | ---: | ---: | ---: |
| 64 reads, 512 sweeps | 247 ± 20 | 0.10 ± 0.02 s | |
| 64 reads, 1,024 sweeps | 152 ± 12 | 0.20 ± 0.03 s | 90.6 |
| 64 reads, 4,096 sweeps | 44 ± 5 | 0.80 ± 0.09 s | 41.5 |
| 64 reads, 14,336 sweeps | 13 ± 1 | 2.82 ± 0.30 s | 14.0 |

Rates pool ten rounds across two test sets, lead times the five rounds that
carried the instrument. Lead is the wall time from starting a cold stage to
its first completed job, so it covers the device open, the kernel compile,
and one batch of sweeps.

The ratio of a probe's rate to the full job's, taken inside a single round,
is steadier than either rate on its own, and the screen's speedup depends
only on that ratio. The 1,024-sweep probe runs 11.37 times the full job's
rate with a spread of 0.72 across the ten rounds, the 512-sweep probe 18.61
with a spread of 1.82.

`scripts/testnet/run_reads_study.py` produced the last column. It starts one
process per nonce and runs eight at a time. Fitting each path as a fixed
cost plus a cost per sweep gives 1.98 ms and 5.24 µs per sweep for the
streaming path, against 6.4 ms and 4.54 µs per sweep of pool time for the
process path. The process path pays a device open, a kernel compile, and a
problem parse for every nonce, and streaming removes nearly that whole cost. The
process path in exchange keeps eight independent dispatches on the device
and fills it slightly better than two overlapping batches of 20 jobs at 64
reads. The two curves cross near 6,300 sweeps, so streaming wins on the
probes, draws at 4,096 sweeps, and loses on the full job.

That one fit describes every budget measured:

| Budget | Wall per job, predicted | Observed | Difference |
| --- | ---: | ---: | ---: |
| 64 reads, 512 sweeps | 4.67 ms | 4.32 ms | −8.1% |
| 64 reads, 1,024 sweeps | 7.35 ms | 6.96 ms | −5.7% |
| 64 reads, 4,096 sweeps | 23.46 ms | 24.45 ms | +4.0% |
| 64 reads, 14,336 sweeps | 77.16 ms | 76.92 ms | −0.3% |

One straight line covers a 28-fold range of sweep counts within 8%, which is
the check that each budget runs the sweeps it claims. A budget that skipped
work would sit below the line.

Device time tells the same story from the other side, and it also shows what
holds the probes back. At 64 reads a batch holds 20 jobs, which is 40
threadgroups, one per GPU core.

| Budget | Device time per job | Device busy against wall |
| --- | ---: | ---: |
| 64 reads, 512 sweeps | 3.3 ms | 77% |
| 64 reads, 1,024 sweeps | 7.3 ms | 105% |
| 64 reads, 4,096 sweeps | 38.2 ms | 155% |
| 64 reads, 14,336 sweeps | 139.0 ms | 181% |

Device busy time past 100% of wall time counts two overlapping command
buffers twice, so that column measures overlap rather than work, and device
time per sweep is not comparable between budgets for that reason. Below 100%
it means the opposite: at 512 sweeps the device stands idle for 23% of the
run because the host cannot feed it. The host rescores every job's 64
samples over 41,514 edges, a cost per job that does not shrink with the
sweep count, so the shortest probe is the budget that competes with whatever
else the machine is doing. A miner on a quiet machine should reach the top
of the measured range or better.

## Lead time, and why a screened miner must not wait

A new qblock changes every nonce, so a miner's queue is thrown away and it
starts again. The number that pays is what a miner finishes inside one
qblock's window, not its steady-state rate.

The lead column in the preceding table is the wall time to the first
completed job. A solve
batch dispatched into an idle device is faster still: one job at 64 reads
and 14,336 sweeps returns in 1.01 s, three jobs in 0.99 s, ten in 1.22 s and
twenty in 1.50 s. A single-job batch wastes 38 of the 40 GPU cores, and it
answers sooner than a full one.

That settles how a screened miner should dispatch. At 1 in 128 the kept
nonces arrive 1.6 times a second, so filling a 20-job solve batch takes ten
seconds of probing.

| Dispatch rule | Probes before the first solve | Lead to the first solve |
| --- | ---: | ---: |
| Each kept nonce as it arrives | 128 | 3.4 s |
| Wait for a full 20-job batch | 2,560 | 12.7 s |

Against the chain, the difference is the whole argument for dispatching at
once. The 300 most recent qblocks arrived a median of 17 substrate blocks
apart, a quarter of them within 8 blocks. Reading a substrate block as six
seconds, and charging each miner for its own lead, the screen is worth:

| Qblock window | Dispatch at once | Wait for a batch |
| ---: | ---: | ---: |
| 12 s, the shortest twentieth of windows | 15.2× | nothing |
| 24 s, the shortest tenth | 15.8× | 8.6× |
| 48 s, the lower quarter | 16.0× | 12.7× |
| 99 s, the median | 16.1× | 14.5× |
| 234 s, the upper quarter | 16.2× | 15.5× |

Over the measured distribution of windows the screen keeps 16.1 of its 16.2
times when it dispatches at once, and 15.2 when it waits, with 5% of windows
too short to return any solve at all under the waiting rule. Doubling the
assumed block time to twelve seconds moves both columns up and changes
nothing about the choice.

The chunk planner does not hold its bound on this envelope. Across 95
batches at 64 reads and 14,336 sweeps, two chunks ran past the 400 ms the
GPU watchdog allows, the longest at 444 ms. None passed 500 ms, and an
earlier run on a quieter machine peaked at 366 ms, so host contention
appears to stretch a chunk past what the planner expects. Bead
`quip-miner-metal-fjo.11` carries it.

## Limits

The study covers one machine, one topology, and one target. The machine
carried other work throughout, so every rate is a range rather than a point,
and the probe rates are the ones that suffer most from it. The energies do
not depend on machine load. Every energy is
the best of 64 reads at a fixed budget, not the lowest the solver can reach.
The keep fractions and cutoffs come from a fit on these 50,000 nonces, with
no out-of-sample test. The hours per valid proof extrapolate a normal tail one standard
deviation past the deepest nonce measured.

Both stages anneal. An instance whose floor no anneal of 14,336 sweeps can
find is invisible to the probe and to the full job alike, so this study
cannot rule out a screen that finds such instances. The chain cannot rule
one out either, because it records only winning nonces.

The deep stage covers 241 nonces. Its subsets overlap: 59 of the 100 the
probe ranked deepest are also among the 100 the full job ranked deepest.

## Method

`tests/probe_screen.rs` draws fresh nonces and builds each instance with
`draw_ising_milli` on topology `cbec1eb4`. It then streams every nonce
through each budget. The instance derivation carries a positive control:
`derived_topology_reproduces_a_regenerated_problem` compares the harness's
instance against one `scripts/testnet/regen` wrote from the chain's own
topology record, and the fields, couplings, and edges all match.

```sh
# The joint table: 50,000 nonces, four budgets, 1 h 45 m of GPU time
QUIP_SCREEN_NONCES=50000 QUIP_SCREEN_SEED=20260918 \
  QUIP_SCREEN_STAGES=64x512,64x1024,64x4096,64x14336 \
  QUIP_SCREEN_OUT=screen-50k.csv \
  cargo test --release --test probe_screen probe_then_solve -- --ignored --nocapture
scripts/testnet/annealer/screen_yield.py docs/perf/data/2026-09-18-screen-50k/screen-50k-best.csv.gz run-50k.log

# The deep stage: three subsets of 100, 5 m of GPU time
scripts/testnet/annealer/screen_deep.py select screen-50k.csv deep-seeds.txt deep-members.csv --count 100
QUIP_SCREEN_SEEDS=deep-seeds.txt QUIP_SCREEN_STAGES=256x65536 QUIP_SCREEN_OUT=deep.csv \
  cargo test --release --test probe_screen probe_then_solve -- --ignored --nocapture
scripts/testnet/annealer/screen_deep.py report docs/perf/data/2026-09-18-screen-50k/deep-members.csv docs/perf/data/2026-09-18-screen-50k/deep-256x65536.csv

# Rates and device time: one round of five, budgets rotated between rounds.
# The third field of a stage caps how many nonces it takes, so budgets of
# different cost run for a similar time.
QUIP_SCREEN_NONCES=7000 QUIP_SCREEN_STAGES=64x512x7000,64x1024x4000,64x4096x1100,64x14336x380 \
  RUST_LOG=quip_miner_metal=debug \
  cargo test --release --test probe_screen probe_then_solve -- --ignored --nocapture
```

The harness reports how the producer thread split its time. Drawing one
instance takes 0.25 ms and the producer spends more than 90% of a run
blocked on a full channel, so the device sets the rate rather than the
harness.

`docs/perf/data/2026-09-18-screen-50k/` holds the best energy of all four
budgets for every nonce, gzipped, and the deep stage's energies with the
seeds it ran. The joint table drops the seed column because the harness
regenerates all 50,000 seeds from `QUIP_SCREEN_SEED=20260918`.
