# A probe screen on 50,000 fresh nonces, 2026-09-18

The 800-nonce study in `2026-09-18-qpu-screen.md` showed that a short job
ranks fresh nonces by the energy a full job reaches. This run puts 50,000
fresh nonces through four budgets to answer the question that matters for a
miner: does the ranking hold for the deepest nonces, the ones that can reach
the target?

## Result

It holds, and the screen is worth about 14.6 times the valid proofs per
second of an unscreened miner on one M4 Max.

A 512-sweep probe at 64 reads ranks 50,000 fresh nonces against the full
14,336-sweep job at Spearman +0.923. Keeping the 390 nonces the probe ranks
deepest, 1 in 128, holds all 10 of the deepest by the full job and 46 of the
deepest 50. The screened miner examines 186 nonces per second against 12.8
for the unscreened one.

The probe picks instances whose energy floor is deep for any solver, rather
than instances that suit a 14,336-sweep anneal. On the 100 nonces the probe ranked deepest, a
solver at 256 reads and 65,536 sweeps, 45 times the effort, reaches a median
of −14,555,000 milli against −14,412,000 on 100 random nonces. All 20 of the
deepest nonces under that solver came from the probe's picks and none from
the random control. Its deepest is **−14,614,000**, which is 11,068 milli
short of the chain's target of −14,625,068. The random control's deepest
stops 103,068 short.

The accuracy of the screen is not what limits it. The probe's own rate is.
Keeping 1 in 128 already captures 89% of the speed an infinitely selective
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
| No screen, 64 reads, 14,336 sweeps | 12.8 | 76 hours |
| 512-sweep probe, keep 1 in 128 | 186 | 5.2 hours |

Those hours are an extrapolation one standard deviation beyond the deepest
of 50,000 nonces, not a measurement. The ratio between the two rows, 14.6,
rests only on the measured rates and the measured recall.

The composite rate is the series sum of the two stages: one probe for every
nonce, plus one full solve for every one hundred and twenty-eighth nonce. A screen that keeps a smaller
fraction approaches the probe's own rate of 210 nonces per second and no
higher.

## Choosing the probe and the keep fraction

Four budgets ran on every nonce, so one table compares them on the same
instances.

| Probe | Rate, nonces per second | Spearman against the full job | Best speedup | Keep fraction at the best speedup |
| --- | ---: | ---: | ---: | ---: |
| 64 reads, 512 sweeps | 210.4 | +0.923 | 14.6× | 1 in 128 |
| 64 reads, 1,024 sweeps | 141.1 | +0.942 | 10.2× | 1 in 128 |
| 64 reads, 4,096 sweeps | 38.3 | +0.965 | 2.9× | 1 in 128 |

The shortest probe wins. A longer probe ranks better and costs more, and the
cost grows faster than the ranking improves. The 800-nonce study could not
see this because it tested no probe below 1,024 sweeps.

Recall as the screen tightens, for the 512-sweep probe:

| Keep | Nonces kept of 50,000 | Probe cutoff, milli | Deepest 50 held | Deepest 10 held | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 in 32 | 1,562 | −14,366,000 | 50 of 50 | 10 of 10 | 10.9× |
| 1 in 64 | 781 | −14,380,000 | 50 of 50 | 10 of 10 | 13.1× |
| 1 in 128 | 390 | −14,396,000 | 46 of 50 | 10 of 10 | 14.6× |
| 1 in 256 | 195 | −14,408,000 | 39 of 50 | 10 of 10 | 15.5× |
| 1 in 512 | 97 | −14,418,000 | 32 of 50 | 9 of 10 | 16.0× |
| 1 in 1,024 | 48 | −14,430,000 | 25 of 50 | 9 of 10 | 16.2× |

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
circular. The deep stage tests the ranking against a solver 45 times
stronger: 256 reads and 65,536 sweeps, run on three sets of 100 nonces drawn
from the same 50,000.

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

The preceding rates come from the batched streaming path, which is how a miner
runs. Earlier reports used `scripts/testnet/run_reads_study.py`, which
starts one process per nonce and runs eight at a time. The two paths differ
only in fixed cost per job.

| Budget | Streaming, nonces per second | One process per nonce |
| --- | ---: | ---: |
| 64 reads, 512 sweeps | 210.4 | |
| 64 reads, 1,024 sweeps | 141.1 | 90.6 |
| 64 reads, 4,096 sweeps | 38.3 | 41.5 |
| 64 reads, 14,336 sweeps | 12.8 | 14.0 |

Fitting each path as a fixed cost plus a cost per 1,024 sweeps gives 2.4 ms
fixed and 4.7 ms per 1,024 sweeps for the streaming path, against 6.7 ms and
4.35 ms of wall time per job for the process path at its eight-way width.
The two agree on the cost of a sweep within 8%. The process path pays a
device open, a kernel compile and a problem parse for every nonce, which is
most of a 1,024-sweep job and none of a 14,336-sweep one. That is why
streaming wins on short probes and loses on full jobs.

A debug-logged run confirms the device time directly. At 64 reads the batch
holds 20 jobs, which is 40 threadgroups, one per GPU core.

| Budget | Device time per job | Device time per sweep | Device busy against wall |
| --- | ---: | ---: | ---: |
| 64 reads, 1,024 sweeps | 9.3 ms | 9.08 µs | 109% |
| 64 reads, 14,336 sweeps | 137.1 ms | 9.56 µs | 183% |

Device time per sweep is constant across a 14-fold range of sweep counts,
which is the check that the short budget runs the sweeps it claims. Device
busy time past 100% of wall time is the two batches the streaming loop keeps
in flight.

The largest single chunk in that run was 366 ms, against the 400 ms bound
the GPU watchdog enforces. Bead `quip-miner-metal-fjo.3` closed on the
envelope that produces it.

## Limits

The study covers one machine, one topology, and one target. Every energy is
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

# Device time
QUIP_SCREEN_NONCES=400 QUIP_SCREEN_STAGES=64x1024,64x14336 RUST_LOG=quip_miner_metal=debug \
  cargo test --release --test probe_screen probe_then_solve -- --ignored --nocapture
```

`docs/perf/data/2026-09-18-screen-50k/` holds the best energy of all four
budgets for every nonce, gzipped, and the deep stage's energies with the
seeds it ran. The joint table drops the seed column because the harness
regenerates all 50,000 seeds from `QUIP_SCREEN_SEED=20260918`.
