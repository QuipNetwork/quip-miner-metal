# A QPU as a nonce screen for the Metal solver, 2026-09-18

Handoff report for the D-Wave research thread. It collects what the
Aglais testnet history and this machine can say about a hybrid in which a
quantum annealer screens nonces and the Metal solver (MSA) spends its sweeps
on the nonces the annealer ranks deepest.

## Result

A screen is the main lever at today's target, and the annealer is one
candidate for it. On 800 fresh nonces the Metal solver at 64 reads and
16,384 sweeps reaches a median of −14,390,000 milli, 235,000 short of the
target of −14,625,068, which sits 4.7 standard deviations out in the tail.
None of the 800 was valid. The instance sets 98% of a job's energy and the
seed 2%, so more seeds or more sweeps on a random nonce buy little, and
finding deep instances is the whole game. A probe job at 1,024 sweeps ranks
fresh nonces against the full job at Spearman +0.95, and keeping the
deepest 1 in 8 by that probe holds 38 of the 40 deepest nonces. On one M4
Max that screen is worth about 3.7 times the valid proofs per second of an
unscreened miner, from the measured rates. The 50,000-nonce follow-up in
`2026-09-18-probe-screen-at-scale.md` measures 14.6 times with a shorter
probe on the streaming path, and supersedes the rate arithmetic here.

The annealer's energy on a nonce ranks nonces the same way. On the 60 most
recent blocks the annealer won, its winning energy ranks the blocks by the
Metal solver's best energy at Spearman +0.74 to +0.77, at every effort
level tested. The quarter of blocks it ranked deepest gives the solver a
median 34,000 milli below the quarter it ranked shallowest, and 14 of the
solver's 15 deepest blocks sit in the half the annealer ranked deepest. The
solver then goes below the annealer's proof on 60 of 60 blocks, by a median
of 30,000 milli at 16,384 sweeps and 42,000 at 65,536. The annealer alone
cannot win at today's target: its best energy ever, −14,580,000, is 45,000
short.

The bar the annealer has to clear is the solver's own probe, and
`2026-09-18-probe-screen-at-scale.md` raised that bar after this report was
written. On 50,000 fresh nonces through the batched streaming path, a
512-sweep probe screens 210 nonces per second at Spearman +0.923 and a
1,024-sweep probe 141 per second at +0.942. At the schedule the chain
records, 34.8 to 46.1 ms per nonce, one annealer screens 22 to 29 nonces per
second. A QPU screen pays only if its call costs well under 5 ms of access
time with the signal intact, or if its ranking beats +0.92 by enough to
offset a rate seven to ten times lower. The chain cannot show either one,
because it records the annealer's energy only on nonces it won.

The experiment for the D-Wave thread is a paired sample. Sample the same
800 fresh nonces on the annealer at the chain's schedule and at 1 and 10
reads, and score each schedule as this report scores the probe. If the
annealer account on the chain was the team's own miner, its logs already
hold samples for every nonce it tried, whether it won or lost.

## What the chain shows

The Aglais testnet has 3,278 qblocks, all on one topology, `cbec1eb4`, the
Advantage2 graph with 4,577 spins and 41,514 edges, fields zero and
couplings in {−1000, +1000}. One account, prefix `dd122d68…`, reports a
device access time of 34.8 to 46.1 ms on every block it won. Every other
account reports 0.55 s or more. That account won 181 blocks, ids 1 to
1,308, and none since. Its access time takes two fixed values, 34.79 ms on
119 blocks and 46.06 ms on 62, each held to 20 µs, which is the signature
of a fixed programming and readout schedule. The team's own 90-minute
mining test of 2025-11-29 ran one Advantage2 system at `num_reads=60` and
`annealing_time=16.13 µs` at 148 nonces per minute, or 2.5 per second. The
chain shows the annealer account at 1.8 to 5.1 nonces per second. The two
match, so the account was plausibly that miner.

| Cohort, 60 blocks each | Reads | P(valid) | P(beats winner) | MSA median best, milli | Valid on 5 of 5 seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Annealer-won, ids 457 to 1,308 | 64 | 0.97 | 0.95 | −14,562,000 | 56 of 60 |
| Annealer-won | 128 | 1.00 | 0.99 | −14,566,000 | 60 of 60 |
| Classical-won, nearest ids | 64 | 1.00 | 0.99 | −14,574,000 | 60 of 60 |
| Classical-won | 128 | 1.00 | 1.00 | −14,578,000 | 60 of 60 |

All at 16,384 sweeps, five seeds per block. Taking MSA's best over all
seeds, MSA goes below the annealer's proof on 60 of 60 blocks:

| MSA effort | Median gap below the annealer, milli | Min | Max |
| --- | ---: | ---: | ---: |
| 64 reads, 16,384 sweeps | 30,000 | 6,000 | 64,000 |
| 128 reads, 16,384 sweeps | 32,000 | 6,000 | 70,000 |
| 256 reads, 65,536 sweeps | 42,000 | 16,000 | 86,000 |

Energy on this graph is −41,514,000 + 2,000 × (unsatisfied couplings), so
42,000 milli is 21 more satisfied couplings out of 41,514. The full
cohort analysis is in `2026-09-18-testnet-annealer-blocks.md`.

## The annealer's energy ranks instances

For each block, pair the energy that won it with MSA's best energy over
all seeds on the regenerated instance. `scripts/testnet/annealer/predict.py`
computes the correlations.

| Winner whose energy is the predictor | MSA effort | Spearman | Pearson |
| --- | --- | ---: | ---: |
| Annealer | 64 reads, 16,384 sweeps | +0.77 | +0.80 |
| Annealer | 128 reads, 16,384 sweeps | +0.74 | +0.78 |
| Annealer | 256 reads, 65,536 sweeps | +0.76 | +0.77 |
| Classical winner, control cohort | 64 reads, 16,384 sweeps | +0.61 | +0.67 |
| Classical winner, control cohort | 128 reads, 16,384 sweeps | +0.66 | +0.70 |

The effect in milli, splitting the 60 annealer-won blocks by the annealer's
energy, is:

| Group by annealer energy | MSA median at 64 reads, 16,384 sweeps | At 256 reads, 65,536 sweeps |
| --- | ---: | ---: |
| Deepest quarter | −14,586,000 | −14,600,000 |
| Shallowest quarter | −14,552,000 | −14,564,000 |
| Difference | 34,000 | 36,000 |
| Deepest half against shallowest half | 24,000 | 29,000 |

Of MSA's 15 deepest blocks, 9 are in the annealer's deepest quarter and 14
in its deepest half. At 256 reads and 65,536 sweeps all 15 are in its
deepest half.

The correlation is a property of the instances. MSA never sees the target
or the winner's energy, so nothing in the regeneration or the run can tie
its energy to them. If instances did not differ, selecting blocks
on the annealer's energy would leave MSA's energy independent of it, and
the correlation would be zero. The 60 blocks span only the energies the
annealer won with, −14,500,000 to −14,580,000. Range restriction lowers a
correlation, so on unselected nonces the figure is more likely above +0.75
than below it, but that is an inference and not a measurement.

## The bar to clear: MSA's own first sweeps

The like-for-like comparison is one cheap job per nonce, with its own
seed, against MSA's best over full jobs on the same nonces.
`scripts/testnet/annealer/checkpoint_rank.py` computes it on the 60 most
recent chain blocks (ids 3,191 to 3,250, best of five seeds), and
`fresh_screen.py` on the 800 fresh nonces (best of two). The sweeps bench
in `2026-09-18-testnet-sweeps-study.md` gives the rates.

| Screen | Spearman on the recent chain blocks | Spearman on fresh nonces | Nonces per second on one M4 Max | Cost per nonce |
| --- | ---: | ---: | ---: | ---: |
| MSA, 1,024 sweeps, 64 reads | +0.62 | +0.95 | 90.6 | 11 ms of GPU time |
| MSA, 1,024 sweeps, 128 reads | +0.63 | not run | 45.0 | 22 ms |
| MSA, 4,096 sweeps, 64 reads | +0.67 | +0.97 | 41.5 | 24 ms |
| MSA, 4,096 sweeps, 128 reads | +0.75 | not run | 22.1 | 45 ms |
| Annealer at the chain's schedule | +0.74 to +0.77, on the blocks it won | no sample | 22 to 29 | 35 to 46 ms of QPU access time |

The chain-block columns run lower than the fresh column for the same
reason on both rows: chain blocks are a selected tail with a standard
deviation of 15,000 to 21,000 milli, against 49,500 for fresh nonces, and
range restriction lowers a rank correlation. The annealer's +0.74 to +0.77
comes from a tail of the same kind, so its figure on fresh nonces is
plausibly higher too. The chain holds no sample to check that.

At the schedule the chain records, the annealer's call costs three to four
1,024-sweep probes and screens a quarter to a third as many nonces per
second as one M4 Max. A QPU screen pays only if one of two things holds.
Either its ranking on fresh nonces beats +0.95, which leaves little room.
Or its call runs well under 11 ms with the signal intact, such as a call
with a handful of reads. Both are measurable in the paired experiment
below.

## Rate arithmetic

| Device | Work | Rate |
| --- | --- | ---: |
| One Advantage2 call at the chain's schedule | screen one nonce | 22 to 29 per second |
| Annealer account, sustained on chain | screen one nonce | 1.8 to 5.1 per second |
| One M4 Max, MSA 64 reads, 16,384 sweeps | solve one nonce | 12.2 per second |
| One M4 Max, MSA 128 reads, 16,384 sweeps | solve one nonce | 6.2 per second |
| One M4 Max, MSA 256 reads, 65,536 sweeps | solve one nonce | 0.44 per second |
| One M4 Max, MSA 1,024-sweep probe, 64 reads | screen one nonce | 90.6 per second |

A screen only matters when it looks at many more nonces than the solver
runs. With one annealer per M4 Max at 64 reads and 16,384 sweeps, the
screen sees two nonces per solve, and the best it can do is discard the
worse half. Deeper MSA effort per kept nonce raises that ratio: at 256
reads and 65,536 sweeps the annealer sees 50 to 65 nonces per solve. The
GPU's own probe at 1,024 sweeps sees 7.4 nonces per full solve on the same
device with no second machine. The fresh-nonce section sizes
the GPU probe. The price of QPU time, which the Leap plan sets, is the
other side of the ledger.

## Fresh nonces

800 nonces drawn at random on the chain's topology, at the chain's current
target of −14,625,068, run at 64 reads and 16,384 sweeps with two seeds
each, plus one probe job at 4,096 sweeps and one at 1,024 sweeps, each
probe with its own seed. `scripts/testnet/annealer/fresh_screen.py` scores
them.

| Job on a fresh nonce | Median, milli | Standard deviation | Deepest of 800 | Valid |
| --- | ---: | ---: | ---: | ---: |
| One full job, 16,384 sweeps | −14,390,000 | 49,500 | −14,544,000 | 0 of 800 |
| Best of two full jobs | −14,394,000 | 49,600 | −14,546,000 | 0 of 800 |
| Probe, 4,096 sweeps | −14,366,000 | 50,100 | −14,514,000 | |
| Probe, 1,024 sweeps | −14,316,000 | 51,300 | −14,476,000 | |

Three facts follow.

The target is far out in the tail. It sits 4.7 standard deviations below
the median fresh nonce, and the deepest of 800 nonces stops 79,000 milli
short of it. The 60 most recent chain blocks reach a median of
−14,622,000 under the same settings, 230,000 deeper than a fresh nonce,
because the network selected them out of about 165,000 nonces per win.

The instance sets the energy, and the seed barely moves it. Two full jobs
on the same nonce agree at Spearman +0.98. The instance accounts for 98% of
the variance, a standard deviation of 49,000 milli against 7,000 for the
seed. A second seed on the same nonce moves the median by 4,000 milli. A
second nonce moves it by 49,000 on average. The chain blocks' standard
deviation of 20,600 is the shape of the selected tail, not of the
population.

A short job ranks fresh nonces almost as well as a full one. That is the
screen, measured directly:

| Probe | Spearman against the best of two full jobs | Keep deepest 1 in 2 | 1 in 4 | 1 in 8 | 1 in 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 sweeps | +0.97 | 40 of 40 | 40 of 40 | 40 of 40 | 33 of 40 |
| 1,024 sweeps | +0.95 | 40 of 40 | 40 of 40 | 38 of 40 | 31 of 40 |
| No signal | 0 | 20 of 40 | 10 of 40 | 5 of 40 | 2.5 of 40 |

The keep columns count how many of the 40 deepest full-job results (the
deepest 5%) survive a screen that keeps only the nonces the probe ranks
deepest. The kept sets are deeper throughout: keeping 1 in 8 by either
probe moves the full-job median from −14,394,000 to −14,472,000, and 1 in
16 to −14,486,000.

A valid nonce lies deeper than any of these 800, and there the probe is
at least as reliable. At Spearman +0.95 the full result given the probe
scatters by about 15,000 milli, so a nonce that MSA would take to
−14,625,000 reads near −14,550,000 on the 1,024-sweep probe, deeper than
any probe value in this sample. A screen that keeps 1 in 16 would keep it.

What the screen is worth on one M4 Max, from the measured rates, keeping
the recall measured on the deepest 5%:

| Miner | Nonces screened per second | Full solves per second | Valid proofs per second, relative to no screen |
| --- | ---: | ---: | ---: |
| No screen, 64 reads, 16,384 sweeps | 12.2 | 12.2 | 1.0× |
| 4,096-sweep probe, keep 1 in 8 | 29 | 3.6 | 2.4× |
| 1,024-sweep probe, keep 1 in 8 | 47 | 5.9 | 3.7× |
| 1,024-sweep probe, keep 1 in 16 | 62 | 3.9 | 3.9× |

Exploratory, not measured on the chain. The last column is nonces screened
per second, times the recall from the deepest 5%, divided by 12.2. The
table divides the GPU's time between probes and full solves at the bench
rates. A chain trial of the screened miner is the test.

## The experiment to run

The chain records only winning nonces, so it holds no annealer sample for
a nonce the annealer lost. The paired sample fills that gap.

1. Take the 800 fresh nonces this report used. `scripts/testnet/make_fresh.py`
   draws them from RNG seed 20260918 on topology `cbec1eb4`, so the D-Wave
   thread regenerates the same instances with the commands in the Method
   section. Each instance is a JSON file with `h`, `j` and `edges`, node
   ids compacted to 0 to 4,576, couplings in ±1.0. MSA's results on them
   are in `docs/perf/data/2026-09-18-fresh/`.
2. Sample every instance on the annealer at the chain's schedule, about 60
   reads at 16 to 20 µs, and record the best and mean sample energy and the
   QPU access time. At 46 ms a nonce, 800 nonces take 37 s of QPU time.
3. Repeat at 1 read and 10 reads per nonce, to find the fastest call that
   keeps the signal.
4. Score each schedule as the preceding tables score the MSA probe. Take
   the Spearman correlation against MSA's best of two full jobs, and the
   share of MSA's deepest 5% of nonces held when the screen keeps the
   deepest 1 in 2, 4, 8, and 16.
   `scripts/testnet/annealer/fresh_screen.py` does that for any CSV with
   one probe row per nonce.
5. Adopt a QPU screen only where it holds more of the deepest 5% than the
   1,024-sweep MSA probe at equal cost per nonce, or the same share at
   lower cost.

If the annealer account was the team's miner, its logs from qblocks 1 to
1,308 hold sample energies for every nonce it tried. Each of those nonces
regenerates from the block's `last_proof_block_hash`, the account, and the
salt, so MSA can run on them and the ranking test needs no new QPU time.

## Limits

The annealer cohort is 60 blocks from one account in one difficulty era,
and every one of them is a nonce that account won. The probe comparison
uses a different set of 60 blocks. The fresh-nonce study is one machine,
one seed per probe, and two per full job. No annealer sample exists for any
fresh nonce yet, so the screen's value for the annealer is an inference
from the chain and not a measurement.

## Method

```sh
# Chain history and the annealer cohort
scripts/testnet/fetch_qblocks.py 3278 qblocks-all.json
scripts/testnet/annealer/make_subset.py qblocks-all.json subset.json cohorts.json
cargo run --manifest-path scripts/testnet/regen/Cargo.toml --locked --release -- subset.json problems-annealer/
READS=64,128 SEEDS=5 SWEEPS=16384 WIDTH=8 scripts/testnet/run_reads_study.py target/release/quip-metal-msa problems-annealer/ annealer.csv
scripts/testnet/annealer/predict.py cohorts.json "64r/16k=annealer.csv:64" "128r/16k=annealer.csv:128" "256r/64k=deep-65536.csv"

# The MSA probe on the recent 60 blocks, from the sweeps study CSVs
scripts/testnet/annealer/checkpoint_rank.py reads-study.csv sweeps/study-1024.csv sweeps/study-4096.csv

# Fresh nonces
scripts/testnet/make_fresh.py qblocks-all.json fresh.json 800
cargo run --manifest-path scripts/testnet/regen/Cargo.toml --locked --release -- fresh.json problems-fresh/
READS=64 SEEDS=2 SWEEPS=16384 WIDTH=8 K0=0 scripts/testnet/run_reads_study.py target/release/quip-metal-msa problems-fresh/ fresh-16384.csv
READS=64 SEEDS=1 SWEEPS=4096  WIDTH=8 K0=2 scripts/testnet/run_reads_study.py target/release/quip-metal-msa problems-fresh/ fresh-4096.csv
READS=64 SEEDS=1 SWEEPS=1024  WIDTH=8 K0=3 scripts/testnet/run_reads_study.py target/release/quip-metal-msa problems-fresh/ fresh-1024.csv
scripts/testnet/annealer/fresh_screen.py fresh-16384.csv fresh-1024.csv fresh-4096.csv
```

The Metal binary is `quip-metal-msa` from `feat/metal-msa`, `--solve` mode,
one process per job, eight at a time, on an M4 Max. Rates in the tables
come from the batched bench in the sweeps study, which is how the miner
runs. The fresh CSVs are committed under `docs/perf/data/2026-09-18-fresh/`.
