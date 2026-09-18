# The blocks a quantum annealer won, 2026-09-18

## Result

One account mined the Aglais testnet with a quantum annealer. It won 181
of the chain's 3,278 qblocks, all between ids 1 and 1,308, and it reports a
device access time of 34.8 to 46.1 ms on every one of them, against 0.55 s
to minutes for every other account. Its last win is qblock 1,308. The
difficulty target passed its best energy ever, −14,580,000 milli, at qblock
1,368, and it has not won since. Today's target is −14,624,068.

On its 60 most recent wins, regenerated exactly, the Metal solver at 16,384
sweeps reaches a lower energy than the annealer's winning proof on 60 of 60,
by a median of 30,000 milli (0.2%) at 64 reads and 42,000 at 256 reads and
65,536 sweeps. It produces a valid proof on 97% of jobs at 64 reads and 100%
at 128, and the same on 60 matched classical-won blocks from the same era.
Every one of those blocks is reachable, so the annealer's wins say nothing
about which blocks are hard for this solver.

| Cohort | Reads | Jobs | P(valid) | P(beats winner) | Median best energy, milli | Mean gap to winner, milli | Mean winner margin, milli | Valid on 5/5 | Valid on 0/5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Annealer-won | 64 | 300 | 0.97 | 0.95 | −14,562,000 | −21,473 | 11,318 | 56/60 | 0/60 |
| Annealer-won | 128 | 300 | 1.00 | 0.99 | −14,566,000 | −25,500 | 11,318 | 60/60 | 0/60 |
| Classical control | 64 | 300 | 1.00 | 0.99 | −14,574,000 | −33,860 | 13,604 | 60/60 | 0/60 |
| Classical control | 128 | 300 | 1.00 | 1.00 | −14,578,000 | −38,380 | 13,604 | 60/60 | 0/60 |

Gap to winner is the solver's best energy minus the winning energy, so a
negative gap is a lower energy than the winner. Winner margin is the target
minus the winning energy. Each cell is 300 jobs: 60 blocks, five seeds.

The chain stores only the winner's energy and a self-reported time, so it
cannot say whether an annealer's samples would sort nonces by reachability.
What the data on hand can say is what a filter has to beat: the first
quarter of the solver's own anneal already predicts the outcome, and
abandoning the poor jobs there yields 1.29 times the valid proofs per unit
of compute at 64 reads on the recent 60 blocks.

## The annealer on the chain

The account with prefix `dd122d68…` is the only one whose device access
time ever falls below 0.55 s. Its 181 wins report 34.8 to 46.1 ms, a
median of 34.8, which is an annealer's access time for a fixed schedule.
Its winning energies run from −14,190,000 to −14,580,000 milli, and its
margin below the target has a median of 11,334 milli. It won 13.8% of the
blocks in its era.

The target was −14,531,322 at its last win and tightened past its best
energy 60 blocks later. The Metal solver's median best energy on the recent
60 blocks is −14,622,000 at 128 reads, 42,000 milli below the annealer's
best. An annealer of that era could not produce a valid proof at today's
difficulty, so a filter is the only role the chain history supports for
it.

## The solver against the annealer's own proofs

On the 60 most recent annealer-won blocks, taking the solver's best energy
over all seeds at each effort level:

| Solver effort | Solver lower on | Median gap, milli | Mean gap, milli | Min | Max | Median gap, percent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 reads, 16,384 sweeps | 60/60 | 30,000 | 29,967 | 6,000 | 64,000 | 0.206 |
| 128 reads, 16,384 sweeps | 60/60 | 32,000 | 32,300 | 6,000 | 70,000 | 0.220 |
| 256 reads, 65,536 sweeps | 60/60 | 42,000 | 43,033 | 16,000 | 86,000 | 0.289 |

Going from 128 reads to 256 reads at 65,536 sweeps lowers the energy on
every block, by a median of 10,000 milli. Going from 64 to 128 reads at
16,384 sweeps lowers it on 35 of 60, by a median of 4,000. A run at 512
reads failed: the Metal solver caps reads at 256. A run at 262,144 sweeps
failed on the miner's 65,536-sweep cap.

## What the chain can say about a filter

A filter needs blocks the solver reaches and blocks it does not, and the
annealer's era has none of the second kind. No block of the 120 was
invalid on every seed, and 116 of 120 were valid on every seed. On the
annealer-won blocks at 64 reads the winner's margin below target has a
weak rank correlation with the solver's valid fraction, Spearman +0.29,
with the four blocks that missed a seed all below the median margin. The
control blocks have nothing to correlate, since every seed was valid.

The recent 60 blocks, where the solver reaches 0.40 to 0.54 of its jobs,
are where a filter would matter, and no annealer sample exists for them.
The chain's `device_access_time_us` and `energy_milli` describe the
winner's one proof, not the distribution of samples an annealer would
return on a nonce, so the question of whether those samples rank nonces
by reachability needs an annealer and the recent blocks. The chain cannot
answer it.

## What a filter has to beat

The solver's own early energy is a filter that costs nothing extra. On the
recent 60 blocks, the mean best energy after a fraction of the sweeps
predicts the valid fraction at 16,384 sweeps:

| Checkpoint | Reads | Spearman, gap above target against valid fraction at 16,384 |
| ---: | ---: | ---: |
| 4,096 | 64 | −0.66 |
| 4,096 | 128 | −0.74 |
| 8,192 | 64 | −0.74 |
| 8,192 | 128 | −0.83 |

An abandon rule runs every job to the checkpoint and continues only the
jobs whose energy is within a threshold of the target. Charging every job
for the sweeps it ran:

| Checkpoint | Reads | Threshold above target, milli | Blocks kept | Valid outcomes captured | Valid proofs per unit of compute |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 64 | 26,000 | 28/60 | 78% | 1.29× |
| 4,096 | 128 | 28,000 | 38/60 | 86% | 1.18× |
| 8,192 | 64 | 12,000 | 24/60 | 78% | 1.11× |
| 8,192 | 128 | 12,000 | 39/60 | 89% | 1.08× |

The thresholds are the best on these 60 blocks and are not tested out of
sample, so the gains are upper estimates. The shape holds at every
threshold from 20,000 to 30,000 at the quarter budget: 1.17× to 1.29× at
64 reads and 1.13× to 1.18× at 128.
An annealer filter pays off only if it ranks nonces better than the
quarter-budget energy does, at a cost below 4,096 sweeps of the solver's
time. The 500-block set in `2026-09-18-testnet-sweeps-intermediates.md`
carries loose targets and does not test this: its half-budget energy
correlates with the outcome at Spearman −0.20 because almost every block
is reachable either way.

## Method

The whole qblock history came from `scripts/testnet/fetch_qblocks.py`
against the local node, 3,278 blocks. `scripts/testnet/annealer/make_subset.py`
takes the 60 most recent wins of the annealer account and, for each, the
classical-won block nearest in id without reuse, and writes a regen input.
`scripts/testnet/regen` regenerated the 120 problems with the nonce check
passing on all of them. `scripts/testnet/run_reads_study.py` ran them at
64 and 128 reads, five seeds, 16,384 sweeps, and again at 256 reads,
65,536 sweeps, three seeds, on the annealer-won 60.
`scripts/testnet/annealer/analyze.py` produces the cohort tables and the
margin correlations, `gap.py` the solver-against-annealer table, and
`abandon.py` the checkpoint filter analysis from the sweep-count study's
CSVs.

```sh
scripts/testnet/fetch_qblocks.py 3278 qblocks-all.json
scripts/testnet/annealer/make_subset.py qblocks-all.json subset.json cohorts.json
cargo run --manifest-path scripts/testnet/regen/Cargo.toml --locked --release -- subset.json problems-annealer/
READS=64,128 SEEDS=5 SWEEPS=16384 WIDTH=8 scripts/testnet/run_reads_study.py target/release/quip-metal-msa problems-annealer/ annealer.csv
scripts/testnet/annealer/analyze.py annealer.csv cohorts.json qblocks-all.json
scripts/testnet/annealer/gap.py cohorts.json "64r/16k=annealer.csv:64" "128r/16k=annealer.csv:128" "256r/64k=deep-65536.csv"
scripts/testnet/annealer/abandon.py 4096 sweeps/study-4096.csv study.csv 64 128
```

## Limits

The annealer population is one account identified by its access time, and
the chain does not name the device. The study covers its last 60 wins, in
a difficulty window 60,000 to 90,000 milli looser than today's. The filter
figures are fitted on 60 blocks and one machine. The energies are the
solver's at fixed budgets, not the lowest it can reach.
