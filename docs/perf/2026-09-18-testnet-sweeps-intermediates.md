# Intermediate sweep counts and the 500 longest rounds, 2026-09-18

## Result

Every count from 8,192 to 16,384 sweeps, on the same 60 regenerated testnet
blocks, qblocks 3,191 to 3,250, five seeds each. The baseline is 128 reads at
16,384 sweeps, the current default, marked in bold. Brackets hold 90%
intervals from 10,000 paired resamples of the 60 blocks against that baseline.

| Reads | Sweeps | Jobs per second | P(valid) | Valid proofs per second | Against baseline | P(beats winner) | Winner-beating per second | Against baseline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 8,192 | 22.49 | 0.230 | 5.17 | 1.53 [1.19, 1.87] | 0.097 | 2.17 | 1.07 [0.65, 1.54] |
| 64 | 10,240 | 19.73 | 0.317 | 6.25 | 1.85 [1.54, 2.14] | 0.147 | 2.89 | 1.44 [1.07, 1.80] |
| 64 | 12,288 | 16.14 | 0.337 | 5.43 | 1.61 [1.35, 1.87] | 0.180 | 2.91 | 1.45 [1.13, 1.75] |
| 64 | 14,336 | 13.99 | 0.403 | 5.64 | 1.67 [1.46, 1.88] | 0.217 | 3.03 | 1.51 [1.23, 1.82] |
| 64 | 16,384 | 12.19 | 0.403 | 4.92 | 1.46 [1.27, 1.63] | 0.203 | 2.48 | 1.24 [1.02, 1.45] |
| 128 | 8,192 | 12.64 | 0.293 | 3.71 | 1.10 [0.92, 1.27] | 0.150 | 1.90 | 0.95 [0.70, 1.19] |
| 128 | 10,240 | 10.04 | 0.337 | 3.38 | 1.00 [0.85, 1.14] | 0.187 | 1.87 | 0.94 [0.74, 1.12] |
| 128 | 12,288 | 8.41 | 0.447 | 3.76 | 1.12 [1.00, 1.23] | 0.237 | 1.99 | 1.00 [0.84, 1.16] |
| 128 | 14,336 | 7.73 | 0.557 | 4.30 | 1.28 [1.18, 1.38] | 0.343 | 2.65 | 1.33 [1.17, 1.52] |
| **128** | **16,384** | **6.24** | **0.540** | **3.37** | baseline | **0.320** | **2.00** | baseline |

Counts below 8,192 stay in `2026-09-18-testnet-sweeps-study.md`. At 64 reads they produce 0.72 of the
baseline's valid proofs per second at 4,096 sweeps and none below that.

Jobs per second is the fastest round at each point, measured with the bench
harness. The intermediate counts ran three rounds each at 64 and 128 reads.
No value is interpolated.

## The answer to the intermediate question

Yes. Three intermediate counts match the baseline on both rates where 8,192
sweeps matches on only one.

At 64 reads and 8,192 sweeps the winner-beating interval is 0.65 to 1.54,
which spans parity. At 10,240, 12,288 and 14,336 sweeps every interval on
both rates sits above parity. The gain over the baseline runs from 1.61 to
1.85 on valid proofs per second and from 1.44 to 1.51 on winner-beating
proofs per second.

The three intermediate counts do not separate from each other. Their
intervals overlap on both rates, and 12,288 sits below 14,336 on both, which
a smooth curve would not do. Treat them as one plateau.

A third measure breaks the tie. Under a `min_solutions` of 5, the
`DifficultyConfig` default that `2026-09-18-testnet-sweeps-study.md` flagged as the decision risk,
each point delivers these proofs per second:

| Reads | Sweeps | Share of jobs with 5 or more reads below target | Proofs per second under min_solutions 5 |
| ---: | ---: | ---: | ---: |
| 64 | 8,192 | 0.030 | 0.67 |
| 64 | 10,240 | 0.037 | 0.72 |
| 64 | 12,288 | 0.067 | 1.08 |
| 64 | 14,336 | 0.100 | 1.40 |
| 64 | 16,384 | 0.113 | 1.38 |
| 128 | 16,384 | 0.193 | 1.21 |

64 reads at 14,336 sweeps is the only count that beats the baseline on all
three measures: 1.67 times its valid proofs per second, 1.51 times its
winner-beating proofs per second, and 1.40 against 1.21 proofs per second
under a `min_solutions` of 5. That is the configuration this measurement
supports.

The chance per job stops rising near the top of the range. At 64 reads,
P(valid) is 0.403 at both 14,336 and 16,384 sweeps. At 128 reads the count
is 0.557 at 14,336 against 0.540 at 16,384. Both differences sit inside a standard
error of about 0.05, so the last 2,048 sweeps buy no measurable chance and
cost 13% to 19% of the jobs per second.

## The 500 longest rounds

### What the set is

The set holds the 500 longest rounds among the 3,277 qblocks on topology
`cbec1eb4` that have a predecessor, out of 3,278 in the chain history. Round
length is the gap in `submitted_at` between a qblock and the qblock one id
below it. Round lengths run from 101 to 1,929 chain blocks, median 176,
against a median of 50 across the whole chain. Three of the 500 also appear
in the 60-block set. Every problem has the same 4,577 nodes and 41,514 edges.

Round length does not measure how hard the problem is. It measures how long
the network took, and over chain history that is set by the target and by how
many miners were running. Round length ranks inversely against qblock id,
Spearman minus 0.546, so the longest rounds sit early in the chain: 73% of
the 500 fall in the first half of the history, and the median qblock id is
1,060. Round length ranks with the difficulty target at Spearman plus 0.657,
so the longest rounds carry the looser targets. The median target across the
500 sits 60,538 milli above the loosest target among the recent 60.

The consequence is that the 500 "hardest" blocks are far easier for the
miner than the 60 recent ones. At 64 reads and 8,192 sweeps, P(valid) is
0.819 on the 500 and 0.230 on the 60, and a valid job carries 17.2 reads
below target against 2.22.

### The energy delta

The delta is the mean over seeds of the best energy at 8,192 sweeps minus the
same at 16,384 sweeps. A positive delta means 16,384 sweeps reached the lower
energy. The 500 ran 3 seeds per block at 64 reads, the 60 ran 5.

| Set | Blocks | Mean delta, milli | Median delta, milli | Min | Q1 | Q3 | Max | Blocks where 8,192 won |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 longest rounds | 500 | 10,148 | 10,000 | -10,000 | 5,333 | 14,667 | 34,000 | 23 |
| 60 recent blocks | 60 | 9,993 | 9,200 | -4,400 | 7,200 | 14,000 | 26,800 | 1 |

The two sets agree. Doubling the sweeps from 8,192 to 16,384 buys about
10,000 milli of energy, and it buys the same amount on a block with an easy
target as on a block with a tight one. The delta is a property of the solver
at this problem size, not of the block.

### How the delta relates to hardness

Quintiles of the 500 by round length, 100 blocks each, shortest rounds first:

| Quintile | Round length | Mean delta, milli | Median delta, milli | P(valid) at 8,192 | P(valid) at 16,384 | Share valid only at 16,384 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 101 to 103 | 11,813 | 12,000 | 0.333 | 0.637 | 0.280 |
| 2 | 104 to 139 | 10,573 | 11,333 | 0.923 | 0.977 | 0.040 |
| 3 | 139 to 201 | 9,900 | 10,000 | 0.947 | 0.980 | 0.010 |
| 4 | 201 to 201 | 8,980 | 8,333 | 0.970 | 0.997 | 0.010 |
| 5 | 201 to 1,929 | 9,473 | 9,333 | 0.923 | 0.963 | 0.030

The delta holds inside a spread of 2,833 milli across the five, from 8,980
to 11,813. What
moves is P(valid), and it moves against the hardness label: quintile 1 holds
the shortest rounds in the set and is the hardest for the miner. Quintile 1
is also the most recent, median qblock id 2,545 against 568 to 1,364 in the
others, and it carries the tightest targets in the set, a median 60,488 milli
below the median of the whole 500.

Round lengths cluster on a few values. 180 of the 500 sit at exactly 201 and
90 sit at 101, so quintile 4 spans a single value.

Quintiles by the winner's margin below target, smallest margin first:

| Quintile | Winner margin, milli | Mean delta, milli | P(valid) at 8,192 | P(valid) at 16,384 | Share valid only at 16,384 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 92 to 3,732 | 10,047 | 0.863 | 0.943 | 0.040 |
| 2 | 3,743 to 9,724 | 10,013 | 0.847 | 0.893 | 0.050 |
| 3 | 9,746 to 15,383 | 10,527 | 0.807 | 0.887 | 0.090 |
| 4 | 15,420 to 19,653 | 9,920 | 0.773 | 0.910 | 0.120 |
| 5 | 19,701 to 270,298 | 10,233 | 0.807 | 0.920 | 0.070 |

The margin orders P(valid) in one direction over the five quintiles but not
at every step, and it leaves the delta inside a spread of 607 milli.

The difficulty target orders the set cleanly. Quintiles by target, tightest
first:

| Quintile | Target from, milli | Target to, milli | Mean delta, milli | P(valid) at 8,192 | P(valid) at 16,384 | Share valid only at 16,384 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -14,613,486 | -14,565,982 | 11,080 | 0.320 | 0.623 | 0.300 |
| 2 | -14,564,354 | -14,543,872 | 10,967 | 0.920 | 0.963 | 0.030 |
| 3 | -14,543,741 | -14,534,217 | 9,947 | 0.903 | 0.977 | 0.040 |
| 4 | -14,534,212 | -14,523,851 | 8,827 | 0.963 | 0.993 | 0.000 |
| 5 | -14,523,624 | -14,089,702 | 9,920 | 0.990 | 0.997 | 0.000 |

P(valid) at 8,192 sweeps runs from 0.320 to 0.990 across the five, and the
share of blocks that cross the target only at 16,384 runs from 0.300 to 0.000.
The delta stays inside 8,827 to 11,080.

The picture is one mechanism. The extra 8,192 sweeps deliver a near-constant
10,000 milli. Whether that 10,000 milli matters depends on where the block's
target sits relative to what 8,192 sweeps already reaches. On the recent 60,
which carry the tightest targets in the chain, 13 of 60 blocks cross the
target only at 16,384 sweeps, a share of 0.217. On the 500 the share is
0.074, and on the 500 the tightest-target quintile reaches 0.300.

The delta ranks against every hardness measure tried at Spearman minus 0.129
for round length and minus 0.092 for the target.

## Commands

```sh
# 500 longest rounds, selection and regeneration; qblocks-all.json is the
# whole chain history from scripts/testnet/fetch_qblocks.py
scripts/testnet/sweeps/make_hard500.py qblocks-all.json \
  sweeps/hard-500.json sweeps/hard-500-meta.csv
cargo run --manifest-path scripts/testnet/regen/Cargo.toml --locked --release -- \
  sweeps/hard-500.json sweeps/problems-hard/

# intermediate counts on the 60 blocks, for S in 10240 12288 14336
READS=64,128 SEEDS=5 SWEEPS=$S WIDTH=8 scripts/testnet/run_reads_study.py \
  target/release/quip-metal-msa problems/ sweeps/study-$S.csv

# 500 longest rounds, for S in 8192 16384
READS=64 SEEDS=3 SWEEPS=$S WIDTH=8 scripts/testnet/run_reads_study.py \
  target/release/quip-metal-msa sweeps/problems-hard sweeps/hard-$S.csv

# jobs per second at the intermediate counts, three rounds, appended to
# the earlier bench rows
scripts/testnet/sweeps/run_bench.py target/release/deps/msa_bench-<hash> \
  sweeps/bench-intermediate.jsonl 3 10240,12288,14336
cat sweeps/bench-intermediate.jsonl >> sweeps/bench.jsonl

# aggregation; study.csv is the earlier reads study's output
scripts/testnet/sweeps/aggregate.py sweeps/ study.csv    # -> sweeps/summary.csv
scripts/testnet/sweeps/bootstrap.py sweeps/ study.csv    # -> sweeps/bootstrap.json
scripts/testnet/sweeps/hard_delta.py sweeps/ study.csv   # -> sweeps/hard-per-block.csv
```

`aggregate.py` and `bootstrap.py` pick up every `study-<sweeps>.csv` in
the directory, so the intermediate counts join the table without a
sweep list to edit.

The regeneration checked every nonce against
`BLAKE3(last_proof_block_hash || blake2_256(miner) || salt)` and all 500
matched. Every one of the 4,800 new solver jobs returned without an error:
1,800 on the 60 blocks and 3,000 on the 500.

## What failed

Nothing failed. One timing anomaly is worth recording. The 8,192-sweep run on
the 500 blocks took 608 seconds and the 16,384-sweep run took 507 seconds,
which reverses the expected order. The aggregation and the bootstrap ran on
the same machine inside the 8,192-sweep window and took processor time from
it. Mean per-job wall time was 3.18 seconds at 8,192 sweeps and 2.64 seconds
at 16,384.

That anomaly changes no result. The solver runs its full sweep count whatever
else the machine is doing, so the energies and the validity flags are
unaffected, and every jobs-per-second figure in this report comes from the
bench harness, which ran alone before the studies started.

The lowest per-job wall time is 2.05 seconds in both runs, against 2.64 and
3.18 seconds for the mean. Process start still dominates the `--solve` path,
as `report.md` recorded.

## Blocked and needs decision

Round length measures the network's time to a proof. Its rank correlation
with the difficulty target is plus 0.657, so the longest rounds are early
blocks with loose targets. That makes the set easier for the miner than the
recent window. To build a hard set, select on the target itself, or on the
margin between the target and what a fixed solver configuration reaches. The
tightest-target quintile of the 500 behaves like the recent 60, and it is the
part of this set worth reusing.

The choice between 14,336 and 16,384 sweeps at 64 reads depends on
`min_solutions`. At the current setting of 1, both beat the baseline and
14,336 leads. At the default of 5 they are 1.40 and 1.38 proofs per second,
a difference the measurement cannot resolve. Either is safe on that field,
unlike 8,192 and 10,240 sweeps, which fall to 0.67 and 0.72 against the
baseline's 1.21. The miner still has no check on `min_solutions`.

The plateau between 10,240 and 14,336 sweeps rests on 300 jobs per point and
five seeds per block. Separating those three counts from each other needs
more seeds, not more sweep counts.

All measurements come from one machine carrying a desktop load, one beta
schedule, and one difficulty window for the 60-block set. The 500-block set
spans the whole chain history, across targets 523,784 milli apart.
