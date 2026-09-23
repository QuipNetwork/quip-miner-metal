# Metal multi-spin miner, part 1 gates, 2026-09-22

Part 1 of the cascade design optimises the multi-spin (MSA) miner before any
screening. Every part 1 gate passes. The 32-sweep probe rate is 5,128 jobs
per second, 6.7 times the 770 measured at the start, and the deep workload
takes 817 ms per batch against 2,168 ms.

The miner is still bound by the host. A 20-job batch takes 3.9 ms of wall
time and about 1.8 ms of GPU time. The largest host cost left is `fill_h_j`.

## Gates

Each value is the median of three rounds with the range in parentheses. The
32-sweep workload is 1,000 jobs of 64 reads. The deep workload is one batch
of 40 jobs of 32 reads at 16,384 sweeps. Both use the Advantage2 fixture in
`tests/msa_bench.rs`.

| gate | step | measured | projection | gate |
|---|---|---:|---:|---:|
| G1 host time per 32-sweep job | 2.1.1 to 2.1.3 | 0.205 ms (0.200 to 0.216) | under 0.2 ms | under 0.3 ms |
| G1 deep GPU time per batch | 2.2.1, 2.2.2 | 1,367 ms (1,363 to 1,496) | 1,400 ms | under 1,600 ms |
| G2 32-sweep energy, inline draw | 2.2.3 | mean shift 0.001 sd, sd shift 0.017 sd | within 0.05 sd | within 0.05 sd |
| G2 GPU fixed cost per job | 2.2.3 | 0.005 ms | under 0.12 ms | under 0.2 ms |
| G3 32-sweep jobs per second | part 1 exit | 5,128 (5,117 to 5,180) | 4,500 to 7,000 | at least 4,000 |
| G3 deep GPU time per batch | part 1 exit | 817 ms (809 to 818) | not projected | not gated |

The G2 energy comparison runs 10,000 fixed nonces at 64 reads and 32 sweeps
through the kernel before and after the change. The best-read column and the
median-read column both pass.

The fixed cost is the intercept of a least-squares fit of GPU time per job
against sweeps 1, 32, and 64. The GPU times per job are 0.001, 0.084, and
0.131 ms, so the slope is 0.0021 ms per sweep.

## What changed

| commit | change | effect |
|---|---|---|
| `c97ab32` | The kernel writes the energy of each read. The host rescores 1 job in 1,000 as an audit | host rescore removed |
| `30f2d59` | One compressed sparse row (CSR) structure per batch, 1,024 threads per threadgroup | GPU time down |
| `6fa844e` | Cached topology and pooled batch buffers | encode allocations removed |
| `bb57f67` | GPU core count read once per process | 2,418 to 4,873 jobs per second |
| `870a284` | Inline geometric threshold draw replaces the per-rung row | fixed cost 0.005 ms |
| `cbe37c5` | Advantage2 four-colouring by default | deep batch 1,367 to 817 ms |

## G1 host-time miss and fix

The first G1 round measured 2,418 jobs per second and 0.41 ms of host time
per job, which missed the 0.3 ms gate. The plan formula, wall time minus GPU
time, applies only when the GPU is the bottleneck. Here the GPU was busy 3 to
4 ms of each 8.2 ms batch, so the host time per job equals the wall time per
job.

A `sample` profile of the stream thread found two IORegistry queries per
batch from `iokit_gov::gpu_core_count`, about half of that thread's busy
samples. The hardware sets the core count. Commit `bb57f67` caches it for the
process, and the rate rose to 4,873 jobs per second with 0.205 ms of host
time per job.

The same profile shows where the remaining host time goes. `fill_h_j` takes
about 43 percent of the stream thread, including one large allocation per
batch. `default_ising_beta_range` takes about 7 percent.

## Four-colouring cost

The four-colouring changes the order of spin updates. At 32 sweeps the mean
best energy is 1,469 milli higher than the greedy colouring, a shift of 0.028
standard deviations. That is inside the 0.05 gate. The deep batch is 40
percent faster with it.

## Conditions

The load average was 3.7 to 6.8 during every round because of desktop
applications, over the protocol's quiet-machine limit of 2. The GPU had no
other work. The gates are absolute thresholds and each passes by a margin
larger than the round-to-round range. Each run stores its `uptime` next to
its log.

## Data

`docs/perf/data/2026-09-22-part1/`:

- `energy-baseline.csv.gz`, `energy-inline-draw.csv.gz`,
  `energy-four-color.csv.gz`: 10,000 nonces each, one row per nonce.
- `g1/`: the first G1 rounds and the re-measure after the fix, `probe-fix-r*`.
  The two `sample` profiles are in the same directory.
- `g3/`: the fixed-cost runs (`fixed-s*`) and the exit rounds.

Each rate run has a gzipped debug log, the `scripts/perf/parse.py` output as
JSON, and the load average. The deep runs have no JSON because a one-batch
run has fewer than the three batches the parser needs. Their GPU time is the
`total_ms` of the single `batch complete` line in the log.
