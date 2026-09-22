# Coloring quality and the combined router

## Study results

The study did not show a coloring penalty on either engine. It also did not
show a valid-job rate gain from changing color order. Metal keeps
greedy order by default. The Apple Neural Engine (ANE) keeps four-color order
by default.

The values below use candidate minus control. A negative energy delta favors
the candidate. A positive validity delta favors the candidate. The valid-rate
ratio divides candidate by control. It uses the observed wall times.

| Comparison | Mean energy delta | Energy 95% CI | Validity delta | Validity 95% CI | Rate ratio | Ratio 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Metal four vs greedy, 64 reads | –500 | –2,056 to 1,000 | +1.67 | –6.11 to 9.44 | 1.122 | 0.919 to 1.363 |
| ANE four vs greedy, 64 reads | –544 | –2,333 to 1,256 | +2.78 | –5.56 to 11.11 | 1.087 | 0.875 to 1.371 |
| ANE 64 vs 128 reads, four colors | +4,000 | 2,567 to 5,422 | –13.89 | –21.67 to –6.11 | 1.131 | 0.924 to 1.343 |

The color intervals do not show equal quality or explain the old 250-milli
effect. Their upper bounds of 1,000 and 1,256 milli exclude the old 1,400-milli
penalty on this fixture only. They set no bound for fresh nonces or other graphs.

All read-count jobs used a fixed 14,336-sweep budget. The 64-read ANE jobs used
863.007 seconds of pooled wall time. The 128-read jobs used 1,324.333 seconds.
The 64-read pooled valid-job rate was 13 percent higher. Its 95 percent
rate-ratio interval of 0.924 to 1.343 includes 1.

The 128-read arm's mean per-job energy was 4,000 milli lower. The 95 percent
interval for this gain is 2,567 to 5,422 milli.
It also raised valid-job probability by 13.89 points. The 95 percent interval
is 6.11 to 21.67 points. Standalone ANE keeps 128 reads with its adaptive 2,048
to 8,192 sweep range. The 64-read mode remains supported for jobs at the
measured fixed 14,336-sweep budget and for combined-engine loads. It had lower
observed queue-inclusive completion latency in these windows. This result is
not a general latency claim.
Neither read count wins in all cases.

### Pooled rates and time to a valid job

Each arm has 180 measured jobs from 60 blocks. For each block, it uses three
independent operating-system random draws, but the protocol does not record
their seeds. The
first-valid median includes only cohorts with a valid result. The last column
gives successful cohorts followed by right-censored cohorts with no valid result.

| Arm | Wall, s | Jobs/s | Valid | Valid/s | Seconds/valid | First-valid median, s | Success/censored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Metal greedy, 64 | 16.477 | 10.924 | 73 | 4.430 | 0.226 | 3.499 | 73 / 107 |
| Metal four, 64 | 15.293 | 11.770 | 76 | 4.970 | 0.201 | 3.360 | 76 / 104 |
| ANE greedy, 64 | 871.011 | 0.207 | 65 | 0.0746 | 13.400 | 135.343 | 65 / 115 |
| ANE four, 64 | 863.007 | 0.209 | 70 | 0.0811 | 12.329 | 149.492 | 70 / 110 |
| ANE four, 128 | 1,324.333 | 0.136 | 95 | 0.0717 | 13.940 | 235.173 | 95 / 85 |

The first-valid figures are conditional. The 64-read ANE median uses 70
successful cohorts. The 128-read median uses 95. The censored cohorts remain in
the valid-job rates in the table.

### Quality windows

Each row has 60 measured jobs. Wall time includes the final queue drain but
excludes warmup and shutdown.

| Arm | Round | Wall, s | Valid jobs | Valid jobs/s |
| --- | ---: | ---: | ---: | ---: |
| Metal greedy, 64 | 0 | 5.672 | 24 | 4.231 |
| Metal greedy, 64 | 1 | 5.598 | 24 | 4.287 |
| Metal greedy, 64 | 2 | 5.207 | 25 | 4.802 |
| Metal four, 64 | 0 | 5.105 | 25 | 4.897 |
| Metal four, 64 | 1 | 5.180 | 24 | 4.633 |
| Metal four, 64 | 2 | 5.008 | 27 | 5.391 |
| ANE greedy, 64 | 0 | 290.080 | 18 | 0.0621 |
| ANE greedy, 64 | 1 | 290.689 | 26 | 0.0894 |
| ANE greedy, 64 | 2 | 290.242 | 21 | 0.0724 |
| ANE four, 64 | 0 | 288.577 | 24 | 0.0832 |
| ANE four, 64 | 1 | 287.152 | 21 | 0.0731 |
| ANE four, 64 | 2 | 287.278 | 25 | 0.0870 |
| ANE four, 128 | 0 | 443.051 | 31 | 0.0700 |
| ANE four, 128 | 1 | 437.663 | 33 | 0.0754 |
| ANE four, 128 | 2 | 443.620 | 31 | 0.0699 |

## Router results and policy

The router study used four paired rounds. Each arm had 1,200 measured jobs. A
window had 300 ANE-eligible Advantage2 jobs. Metal-only used greedy order. The
combined arm used greedy order on Metal and four-color order on ANE.

| Arm | Wall, s | Jobs/s | Valid | Valid/s | Seconds/valid | First-valid median, s | Success/censored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Metal only | 98.068 | 12.236 | 452 | 4.609 | 0.217 | 6.368 | 165 / 75 |
| Metal and ANE | 118.344 | 10.140 | 451 | 3.811 | 0.262 | 7.598 | 159 / 81 |

The combined-to-Metal valid-rate ratio was 0.827. Its conditional 95 percent
interval was 0.776 to 0.874. This interval uses the fixed observed window times.
Mean lowest energy changed by +183 milli, with a 95 percent interval of –347 to
+703. Validity changed by –0.08 points, and its interval was –2.25 to +2.08
points. The rate loss came from elapsed time, with no shown change in per-job
validity.

The combined router sent 1,187 jobs to Metal. It sent 13 to ANE. For this finite
300-job mix of one eligible job type, use `enable_metal = true` and
`enable_ane = false`. Keep the current interfaces and default flags. This
result does not call for an added ANE worker.

For a required ANE path or jobs Metal cannot take, keep one ANE slot without
applying this finite result to an infinite stream or an unrelated mix. If both
engines remain enabled, keep Metal greedy and ANE four-color order.

### Router windows

| Arm | Round | Wall, s | Valid jobs | Valid jobs/s |
| --- | ---: | ---: | ---: | ---: |
| Metal only | 0 | 23.822 | 104 | 4.366 |
| Metal only | 1 | 24.143 | 122 | 5.053 |
| Metal only | 2 | 23.979 | 116 | 4.838 |
| Metal only | 3 | 26.124 | 110 | 4.211 |
| Metal and ANE | 0 | 28.077 | 116 | 4.132 |
| Metal and ANE | 1 | 29.789 | 112 | 3.760 |
| Metal and ANE | 2 | 31.221 | 110 | 3.523 |
| Metal and ANE | 3 | 29.258 | 113 | 3.862 |

The [router analysis](data/2026-09-21-coloring-router/router/analysis-router.json)
has the 10,000-draw block bootstrap. The
[router protocol record](data/2026-09-21-coloring-router/router/router-protocol.json)
stores the balanced arm order and hash metadata. The same directory has all
eight receipts and route logs.

### Host and records

The host was an Apple M4 Max, model Mac16,5. It had 128 GiB of memory. It ran
macOS 26.5.2 build 25F84. The actual compiler was Rust 1.98.1. Its commit was
`48a229cea`. The study base was `66471a3de865d941fb43fa4254557708cfe4492e`.

The [environment record](data/2026-09-21-coloring-router/quality/environment.json)
lists the host, toolchain, base revision, and paused-miner state. The
[protocol record](data/2026-09-21-coloring-router/quality/quality-protocol.json)
lists the binary, driver, and fixture hashes plus the arm order. The
three analysis records are [Metal coloring](data/2026-09-21-coloring-router/quality/analysis-metal-color.json),
[ANE coloring](data/2026-09-21-coloring-router/quality/analysis-ane-color.json),
and [ANE read count](data/2026-09-21-coloring-router/quality/analysis-ane-reads.json).
The same directory holds all 15 window receipts and route logs. Across the
quality and router studies, all 3,300 measured channel jobs matched consensus
rescoring exactly. A final production
`--check` exited 0 under `scripts/ane-guard`.

## Study path

The study sends jobs by Unix gRPC to a real `quip-metal-msa` process. The path
includes setup, credits, and routing. It also includes the governor, device
work, and returned results.
The driver scores each returned spin array again. Each consensus score
must match exactly.

The governor uses `utilization = 100` and `yielding = false`. Metal uses its
shipped threadgroup budget. It also uses its shipped safety factor.

Each window starts after warmup ends and the credits return. A combined run
warms both engines at once. It checks the route records for both engines. The
timed window keeps all advertised credits in use until the jobs finish.

The timed window includes the final queue drain. It also includes a separate
score check by the coordinator. It excludes warmup and shutdown. The driver
runs in release mode, which removes debug-loop cost from that score check.

Debug logs name the engine that receives and completes each job. The study
does not guess engine use from elapsed time. It checks the read and sweep counts
in each result. It writes the final receipt after all jobs finish and all
credits return. The miner must exit 0.

## Problems and runs

The fixture has 60 past qblocks, from 3191 through 3250. It stores the
nonce seed, target, and winning energy for each block. The protocol's ChaCha8
generator rebuilds the coefficients.

The topology has 4,577 nodes and 41,514 edges. Its fields are zero. Its
couplings are unit values. It uses `advantage2-system1.edges` without edge
`(880, 2695)`. The fixture test checks all 60 coefficient vector sizes. It
matches archived coefficient fingerprints for block 3250.

The study chose 60 past mining problems with valid proofs.
This success-based choice means that their valid-job rates do not estimate yield
from fresh random nonces. Nonce screening is part of `fjo.10`, set for release 0.3.4, and remains
outside this work.

The protocol does not expose the sampler's operating-system seed. It does not
accept that seed either. Each arm uses independent random draws. A matching
`(block, replicate)`
key means only that the problem and job label match. The random streams remain
independent. The receipts support repeated tests. They do not
support exact replay of each sampler output. The seeded overlap study checks
exact replay on its separate workload.

Quality windows use 64 reads and 14,336 sweeps. On each engine, they compare
greedy order with four-color order. A fifth arm uses four colors and 128 reads on
the Apple Neural Engine (ANE). Its sweep count stays at 14,336.

Each of three rounds sends one job for each problem. This gives 180 measured
jobs per arm. The raw receipt also keeps the excluded warmup from each window.

The arm order is forward in round one and reverse in round two. Round three
rotates the arms left by two places. This order does not give each arm every
time position equally often. The report shows each window's observed wall time
but gives no quality-bootstrap confidence interval for thermal drift.

Router windows alternate four Metal-only and combined-engine rounds. Each
window sends five jobs for each problem, for 300 measured jobs. Each setup runs
first in two rounds. All jobs are ANE-eligible Advantage2 jobs. They use 64
reads and 14,336 sweeps. Each window includes the final queue drain for its
fixed workload.

## Scoring and uncertainty

A job is valid if at least one read is below its historical target. A read that
equals the target is not valid. These historical jobs require one solution.
They have no diversity threshold. A separate measure counts jobs that match or
beat the recorded winning energy.

For each matched job, the energy delta is the candidate's lowest energy minus
the control's lowest energy. A positive delta means that the candidate gave
worse energy.

The analysis samples 60 whole problems with replacement. It repeats this step
10,000 times. Each draw keeps all attempts for a chosen problem together. Each
draw also keeps both arms together. The result is a 95 percent interval for mean
energy and valid-job probability.

A pooled rate divides the total valid jobs by the total measured wall time.
Its interval treats the observed window times as fixed and does not include
timing uncertainty. The report also shows each window rate, so timing changes
remain visible. The inverse rate is the total seconds per valid job.

First-valid latency starts when the coordinator sends a problem's first job.
It ends when that problem returns its first valid result in the window. This measure
includes queue time. Its median includes only cohorts that succeed. Cohorts
with no valid result remain in the failure count and rate denominator. The
conditional median is not an estimate for all problems.

## Run the study

Build the miner and study driver before any timing window:

```sh
cargo build --release --locked --bin quip-metal-msa
cargo test --release --locked --manifest-path crates/ane-miner/Cargo.toml \
  --test combined_study --no-run
```

Use the driver path that Cargo prints:

```sh
python3 scripts/testnet/run_coloring_router.py quality \
  --binary target/release/quip-metal-msa \
  --driver crates/ane-miner/target/release/deps/combined_study-DRIVER_HASH \
  --out /tmp/quality-receipts
python3 scripts/testnet/run_coloring_router.py router \
  --binary target/release/quip-metal-msa \
  --driver crates/ane-miner/target/release/deps/combined_study-DRIVER_HASH \
  --metal-color greedy --ane-color four --out /tmp/router-receipts
```

The runner holds `scripts/ane-guard` for each window and its settling time.
Stop all other mining work first. Do not run builds during the study. The
runner rejects output paths that already exist. It stops if a window fails.
After a timeout, it stops the full process group. It also stops all child
processes.

Pass the first control and candidate files to `analyze_coloring_router.py`.
Add later files with `--control-extra` or `--candidate-extra`. A read-count
comparison also needs `--allow-read-change`.
