# Setup profile for the Apple Neural Engine program, 2026-09-17

## Result

Compile is the largest stage. Its median is 283.887 ms, about 92% of the
native stage sum described below and 75% of the probe's internal total.

The six native stages, the ones this probe times directly against real
`ane_bridge.m` calls, sum to 309.462 ms at the median. `graph_prep_ms` is not
part of that sum. See Method for why it cannot stand in for the real
Rust-side cost. The baseline fixed setup is 520 ms, less about 11.838 ms of
estimated process spawn, for a budget of 508.162 ms. The native stage sum
misses that budget by 198.700 ms, 39.1% of the budget, far over the 15% check
in the task brief. See Gap accounting for the full list of stages this probe
cannot reach.

## Method

The probe, `crates/ane-miner/probes/setup_profile.m`, includes
`crates/ane-miner/native/ane_bridge.m` and calls its `makeMIL`,
`makeWeightBlob`, `shape`, and `slice` functions unchanged. This matches the
convention `single_call.m` already uses in the same directory, and it removes
the risk of a copy drifting from the code that ships.

The probe times seven stages around the same calls `quip_ane_create` makes:
`graph_prep_ms`, `mil_build_ms`, `blob_build_ms`, `blob_write_ms`,
`compile_ms`, `load_ms`, and `surface_setup_ms`. Six of the seven,
`mil_build_ms` through `surface_setup_ms`, keep production's relative order.
The probe also times two stages outside the required schema,
`range_validate_ms` and `dlopen_class_setup_ms`, so the Gap accounting
section can point at real, measured costs.

Placement is not fully identical to production, though. Production builds
the weight blob inline, as the argument to `writeToFile:` at
`ane_bridge.m:331`, immediately before the write. That happens after
`dlopen` loads the private Apple Neural Engine (ANE) runtime library, the
class lookups, plist serialization, descriptor construction, and model
construction. It also happens after staging directory creation and the
`model.mil` write. The
probe instead times `blob_build_ms` before that whole bucket, which it
reports as `dlopen_class_setup_ms`. `blob_build_ms` runs under a different
CPU and cache state in the probe than `makeWeightBlob` runs under in
production, and that state is probably colder in the probe. Read
`blob_build_ms` as an isolated cost of the function, not as a measurement of
its cost in its real position.

`graph_prep_ms` is a synthetic stand-in for the Rust-side weights and fields
build inside `AneProgram::compile`, `crates/ane-miner/src/native.rs:89`
through `native.rs:106`, since the probe has no Rust and cannot run that
code. The stand-in is a poor analog for the function it approximates. The
probe fills the full, about 21.68 million-element, weights buffer with a
`random()` call and a write at every position, `setup_profile.m:44` through
`setup_profile.m:51`. The real function instead zero-fills the same-sized
buffer in bulk with `weights.resize(count, 0)` and then writes only the
about 83,000 entries that have a real neighbor edge, about 100 to 1,000
times less per-element work. Do not read `graph_prep_ms`'s value,
56.065 ms at the median, as an estimate of the real function's cost. The
probe cannot measure the Rust side of setup at all, and `graph_prep_ms` is a
placeholder. Do not read its value as an estimate of anything. Every
comparison against the 520 ms budget in this report excludes it.

The probe uses the topology in `tests/fixtures/advantage2-system1.edges`,
colored by descending degree then ascending index, exactly as `graph.rs`
colors it. This gives eight tiles with lengths 857, 849, 817, 740, 685, 480,
136, and 13, from `docs/perf/2026-09-16-ane-local-routing.md:13`. The probe
asserts these lengths sum to 4,577 and fails otherwise. `channels` is 4,608,
the next multiple of 32 that is at least 4,577. The probe asserts this length
too.
Couplings are `{-1, 1}` from seed 123. Fields are zero. Sweeps is 2, matching
`BLOCK_SWEEPS` in `crates/ane-miner/src/native.rs:10`, the sweep count every
production compile call uses, `--solve` included.

No shared device guard exists in this repository or in `/tmp`. The five runs
below ran one at a time, with a 3-second sleep between runs, in place of a
guard.

### Commands

```bash
clang -O3 -fobjc-arc -fobjc-arc-exceptions -std=c11 -Wall -Wextra -Werror \
  -framework Foundation -framework IOSurface \
  crates/ane-miner/probes/setup_profile.m \
  -o /tmp/quip-ane-throughput-0917/setup-profile
```

```bash
cd /tmp/quip-ane-throughput-0917
for i in 1 2 3 4 5; do
  { /usr/bin/time -p ./setup-profile > run${i}.json; } 2> run${i}.time
  if [ "$i" -lt 5 ]; then sleep 3; fi
done
```

Each run exited 0 and printed one JSON object on standard output, as the
brief requires.

## Stage table

All values are milliseconds, from `mach_absolute_time`, except `blob_bytes`
and `mil_bytes`, which are exact byte counts.

| Stage | Category | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| graph_prep_ms | Placeholder, excluded from budget comparisons | 46.863 | 56.065 | 60.217 | 56.041 | 56.242 | 56.065 |
| mil_build_ms | Native | 3.172 | 2.905 | 2.967 | 2.970 | 2.903 | 2.967 |
| blob_build_ms | Native | 10.399 | 10.094 | 9.870 | 10.046 | 9.955 | 10.046 |
| blob_write_ms | Native | 5.850 | 5.221 | 5.390 | 5.357 | 5.436 | 5.390 |
| compile_ms | Native | 281.772 | 280.266 | 285.039 | 283.887 | 287.671 | 283.887 |
| load_ms | Native | 16.556 | 3.000 | 3.323 | 3.532 | 4.067 | 3.532 |
| surface_setup_ms | Native | 2.362 | 2.089 | 2.621 | 3.670 | 3.013 | 2.621 |
| **Native stage sum, six stages** | Used for Gap accounting | 320.111 | 303.575 | 309.210 | 309.462 | 313.045 | **309.462** |
| All seven required keys summed | Reference only, not used for Gap accounting | 366.974 | 359.640 | 369.426 | 365.503 | 369.289 | 366.974 |
| range_validate_ms | Extra, not a required key | 7.806 | 7.791 | 7.824 | 7.966 | 7.783 | 7.806 |
| dlopen_class_setup_ms | Extra, not a required key | 3.381 | 2.546 | 2.863 | 2.583 | 2.851 | 2.851 |
| Probe internal total, nine measured stages | Extra | 378.162 | 369.977 | 380.113 | 376.052 | 379.923 | 378.162 |
| Wall time, seconds | Extra, whole process | 0.38 | 0.38 | 0.39 | 0.39 | 0.39 | 0.39 |

Both summary rows are the median of the five per-run sums, not the sum of
the displayed per-stage medians. Summing the six native per-stage medians
instead gives 308.443 ms. Summing all seven required-key medians gives
364.508 ms. A third way to reach about the same figure subtracts
`graph_prep_ms`'s own median from the required-key sum, 366.974 minus 56.065,
and gives 310.909 ms. All three native-total figures, 309.462, 308.443, and
310.909 ms, are legitimate, differing only in how each takes the median.
This report uses 309.462, the median of per-run sums, for the rest of Gap
accounting.

Run 1 shows a 16.556 ms `load_ms`, well over the other four runs. This
matches a cold first-load cost against the private ANE runtime. The median
absorbs it. No stage was too fast to measure: the smallest, `mil_build_ms`,
still spans 2.9 to 3.2 ms across runs, well past the timer resolution.

`blob_bytes` and `mil_bytes` held constant across all five runs, as expected
for a deterministic build: 43,352,640 and 70,717.

## Gap accounting

The task brief asks whether the seven required stages sum to within 15% of
the 520 ms fixed setup, less process spawn. `graph_prep_ms` is a placeholder,
not a measurement, so this report answers that question with the native
stage sum, six stages, instead of all seven required keys summed.

The check needs a process-spawn estimate, which the brief does not supply,
so this report derives one instead of assuming a value. Wall time in the
stage table wraps the whole probe process, spawn through exit, using
`/usr/bin/time -p`. Probe internal total is `graph_prep_ms` through
`surface_setup_ms` plus the two extra stages, timed from inside the process.
The median gap between these two, 11.838 ms, is this report's process-spawn
estimate. It covers `dyld` loading Foundation and IOSurface before `main`
runs, plus process exit. This estimate keeps `graph_prep_ms` on the
internal-total side of its own subtraction. It measures how much of wall
time no internal timer captures at all. It does not compare `graph_prep_ms`'s
value to anything.

| Quantity | Value, ms |
| --- | ---: |
| Baseline fixed setup | 520.000 |
| Estimated process spawn | 11.838 |
| Budget for measured stages | 508.162 |
| Native stage sum, median of five runs | 309.462 |
| Gap | 198.700 |
| Gap as a share of budget | 39.1% |

39.1% is a larger miss than an earlier draft of this report showed. That
draft summed `graph_prep_ms` into the figure checked against the budget,
366.974 ms, for a smaller gap of 141.188 ms, 27.8%. That smaller
number is not trustworthy: it credits the budget with 56.065 ms of work the
probe never actually measured. The 39.1% figure is the honest one. Either
way the gap is real and exceeds the 15% check.

`solve_with_block`, `crates/ane-miner/src/solver.rs:45` through
`solver.rs:71`, sets `stats.setup_us`, the metric behind the 520 ms baseline.
Besides `AneProgram::compile`, the function this probe's native stages
measure, that window runs five more functions the probe never touches:

- `validate_params`, `crates/ane-miner/src/msa.rs:21` through `msa.rs:50`,
  called from `solver.rs:46` and again inside `schedule`.
- `prepare`, `crates/ane-miner/src/graph.rs:93`, called from `solver.rs:47`.
  This is the real Rust-side graph coloring and layout step. `graph_prep_ms`
  does not stand in for this function. See Method for what it does stand in
  for.
- `schedule`, `crates/ane-miner/src/msa.rs:52` through `msa.rs:74`, called
  from `solver.rs:48`.
- `initial_spins`, `crates/ane-miner/src/msa.rs:76` through `msa.rs:86`,
  called from `solver.rs:49`, a node-count-by-128-sized buffer built from one
  pseudorandom draw per element.
- `program.reset`, `crates/ane-miner/src/native.rs:184`, wraps
  `quip_ane_reset`, called from `solver.rs:70`. It stages the initial spin
  state into the first IOSurface.

None of these five has a receipt key in this task, and this report does not
estimate any of their costs. A later task should time them directly if full
accounting matters.

## Rust-side setup

This section reports five counters that `crates/ane-miner/src/solver.rs`
now records inside `stats.setup_us`: `validate_us`, `graph_prep_us`,
`schedule_us`, `initial_spins_us`, and `reset_us`. The probe cannot reach
these five functions, because the probe is a C program and these five
functions are Rust. This section measures the functions the probe could not
reach and redoes the preceding Gap accounting section with real numbers in
place of five blank rows.

`graph_prep_us` and `graph_prep_ms` are different measurements of different
code, despite the similar name. `graph_prep_ms`, in the preceding Stage
table, is a synthetic placeholder that fills a weights buffer with
`random()` calls. That does about 100 to 1,000 times more per-element work
than the real code does. See the preceding Method section for that
placeholder. `graph_prep_us` times the real `prepare(graph)` call at
`crates/ane-miner/src/graph.rs:93`, called from
`crates/ane-miner/src/solver.rs:62`. Do not read the two values as
comparable.

### Method

`solve_with_block`, `crates/ane-miner/src/solver.rs:51` through
`solver.rs:131`, now starts an `Instant` before each of the five calls and
stores the elapsed microseconds on `RunStats` right after each call returns.
This adds five field writes and five clock reads around calls the function
already made. It does not change the order of the calls, the arguments
passed to them, or any other control flow.

The `--solve` driver in `quip-solver-core` does not surface `RunStats` on its
own output. `ProblemJson` and `SolutionJson` in that crate's `driver.rs`
carry spins and an energy value, not timing. Wiring `RunStats` into the
worker's tracing output is Task 6's job, not this one. To measure the five
counters, this task added a test-only harness,
`hardware_rust_setup_stage_medians_advantage2_system1` at
`crates/ane-miner/src/solver.rs:515`, marked `#[ignore = "requires Apple
Silicon ANE"]` like the other hardware tests already in that file. It builds
the same topology, coupling seed, and solve seed this section's runs use,
calls `solve_in_process` directly, and prints every `RunStats` field to
standard error.

The topology is `tests/fixtures/advantage2-system1.edges`, the same 4,577
nodes and 41,515 edges the probe uses. Fields are zero. Couplings are `{-1,
1}` from a seeded xorshift64 generator with seed 7, the same generator
`tests/msa_bench.rs` already uses for this fixture in this repository's
top-level test suite. Task 1's probe used coupling seed 123 instead. The two
differ because each follows its own side's convention, and the difference does
not affect timing: coupling values change neither buffer sizes nor branching
anywhere on this path. `num_reads` is 128, `num_sweeps` is 512, matching the
`--solve` sweep count this task's brief specifies, `sweeps_per_beta` is 1,
and `beta_range` is unset. `seed`, the solve-time pseudorandom source, is
123.

No shared device guard exists in this repository or in `/tmp`, the same gap
Task 1 found. The five runs below ran one at a time, with a 3-second sleep
between runs, in place of a guard.

Command, run five times:

```bash
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --release \
  --lib hardware_rust_setup_stage_medians_advantage2_system1 \
  -- --ignored --nocapture
```

### Stage table

All values are milliseconds, converted from the microseconds `RunStats`
records.

| Stage | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validate_us | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| graph_prep_us | 4.432 | 2.460 | 2.512 | 2.545 | 2.563 | 2.545 |
| schedule_us | 0.244 | 0.255 | 0.255 | 0.240 | 0.241 | 0.244 |
| initial_spins_us | 1.114 | 0.544 | 0.554 | 0.515 | 0.544 | 0.544 |
| reset_us | 2.309 | 2.314 | 2.297 | 2.325 | 2.329 | 2.314 |
| setup_us, the whole window | 364.084 | 346.477 | 335.019 | 340.423 | 335.843 | 340.423 |

Run 1 shows the largest `graph_prep_us` and `initial_spins_us` values and the
largest `setup_us` total. This matches the cold-start pattern Task 1 found in
`load_ms`: the first call in a fresh process runs colder caches than the
following four. The median absorbs it here as it did there.

`setup_us` is `AneProgram::compile` plus all five measured counters. Summing
the five counters and subtracting from the `setup_us` median leaves 334.776
ms for `AneProgram::compile` alone, more than the probe's 309.462 ms native
stage sum. That 334.776 ms figure is a composite across the five runs, not
one run's measurement: each term in it is its own per-stage median, and those
medians do not all come from the same run. Run 4 taken alone gives 334.798 ms,
which is close but not identical. See the following Gap accounting section for
what the 25.3 ms difference most likely is.

### Gap accounting

The task brief that requested this section cites a 367 ms native total for
this step. That figure is the all-seven-required-keys sum from the
preceding Stage table, 366.974 ms. It includes the synthetic
`graph_prep_ms` placeholder. The decision that set up this task excludes
`graph_prep_ms` from every budget comparison. This section instead uses
309.462 ms, the native stage sum, six stages, the same figure the earlier
Gap accounting section already uses.

| Quantity | Value, ms |
| --- | ---: |
| Native stage sum, six stages | 309.462 |
| Plus graph_prep_us, median | 2.545 |
| Plus reset_us, median | 2.314 |
| New sum | 314.321 |
| Budget for measured stages | 508.162 |
| Gap | 193.841 |
| Gap as a share of budget | 38.1% |

Adding all five new counters, not only the two the brief names, changes the
sum little. 309.462 plus 5.647 ms, the sum of all five medians, is 315.109
ms. That is a 193.053 ms gap, 38.0% of budget. `validate_us`, `schedule_us`,
and `initial_spins_us` are small next to `graph_prep_us` and `reset_us`, as
the brief predicted.

A third figure uses the `setup_us` median directly, 340.423 ms, instead of
summing figures from two different processes and cache states, the probe
and this task's harness. That gap is 167.739 ms, 33.0% of budget. Trust
this figure over the other two: a single direct measurement beats a sum of
measurements taken in different processes. Even this figure stays far over
the 15% check, 76.224 ms.

The gap does not close under any of the three ways to compute it. Two places
hold the rest, named here rather than estimated by assertion:

- `AneProgram::compile` itself, `crates/ane-miner/src/native.rs:77` through
  `native.rs:176`, spends time neither the probe nor this task's five
  counters measure. The preceding Stage table shows about 25.3 ms of this.
  That is the gap between the measured `setup_us` median and the sum of the
  probe's native total and this task's five counters. Two spots inside that
  function are the likely source. The weights and fields build,
  `native.rs:89` through `native.rs:106`, writes into a weights buffer of
  about 21.68 million elements, the same size the `graph_prep_ms`
  placeholder in the Method section fills. `compile_raw`'s own bounds check
  on that same buffer, `native.rs:137` through `native.rs:145`, walks every
  element again. This report does not measure either loop on its own, so it
  does not assign the 25.3 ms between them.
- A larger, unmeasured cost sits entirely outside `stats.setup_us`. Even the
  fullest figure here, the 340.423 ms `setup_us` median, misses the 508.162
  ms budget by 167.739 ms. Every job in production spawns a second copy of
  the worker binary through `WorkerProcess::spawn`,
  `crates/ane-miner/src/process.rs:63` through `process.rs:104`. That call
  makes a temporary directory, creates two files, and runs a full `fork` and
  `exec` of the ANE worker executable. The new process loads its own copy of
  Metal, IOSurface, and the other linked libraries. `worker_main`,
  `crates/ane-miner/src/worker.rs:169`
  through `worker.rs:196`, starts a watchdog thread, and reads the job off
  standard input before it calls `solve_in_process` at all. None of this
  runs inside `stats.setup_us`, and none of it resembles the single small C
  binary the probe's 11.838 ms process-spawn estimate covers. This report
  does not measure the worker spawn and inter-process communication path, so
  it does not assign the remaining 167.739 ms to it. A later task should
  time `WorkerProcess::spawn` through the first byte `solve_in_process`
  reads, directly, if full accounting matters. Task 8 does this. See
  Worker path below.

## Worker path

Task 7 named the out-of-process worker path as the largest cost outside
`stats.setup_us`, and left it unmeasured. This section measures it. The
pure wrapper, the file and process machinery around the real job, adds
19.082 ms to a job's fixed setup. Most of the earlier gap turns out to sit
inside the child's own compute, which the wrapper counters cannot see, and
inside four file-and-line-named spots this section bounds but does not
directly instrument. The redone total still misses the 508.162 ms budget,
by 110.069 ms, 21.7% of budget, over the 15% check in the task brief. See
Gap accounting below for both ways to compute this, and why they differ.

### Harness cost, bounded

`--solve` reads its whole problem from standard input before any worker
spawns. Production takes jobs over the coordinator protocol and never pays
that read. Five runs each, one at a time with a 3-second sleep, of the
release binary's `--solve` mode, first on the real topology at 2 sweeps,
matching `BLOCK_SWEEPS`, then on the two-node problem in
`crates/ane-miner/README.md:74`:

| Input | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real topology, 2 sweeps, seconds | 1.02 | 0.52 | 0.52 | 0.51 | 0.54 | 0.52 |
| Two-node problem, seconds | 0.22 | 0.22 | 0.21 | 0.21 | 0.22 | 0.22 |

The difference is 0.30 s. It bounds, but does not equal, the cost of the
813 KB standard input read production never pays. The same 0.30 s also
holds a bigger graph preparation step, a bigger inter-process reply, and
the real compile against the full topology instead of a two-node graph.
Do not read 0.30 s as the parse cost alone. Run 1 of the real-topology row
shows the same cold-start pattern every prior section in this document
finds in a first run. The median absorbs it.

### Method

`crates/ane-miner/src/process.rs` now times six stages around
`WorkerProcess::spawn` and `WorkerProcess::wait`: directory and file
creation, the request write, `Command::spawn`, the wait for the child to
exit, the reply read, and teardown. These are new `Instant` calls around
work the function already does, in the same order. No call moved, and no
branch changed. `wait` logs the six counters through one
`tracing::debug!` call once teardown finishes.

`crates/ane-miner/src/worker.rs` and `crates/ane-miner/src/bin/quip_ane_msa.rs`
add a seventh counter, `child_arg_parse_us`. `main` now captures an
`Instant` before it parses arguments, and `worker_main` prints the elapsed
time as its first line, to standard error, which the parent inherits. By
the time any Rust code in `main` runs, `dyld` has already resolved and
loaded every library the binary links, including the private ANE runtime
library, and has already jumped to the compiled binary's entry point.
None of that work happens inside this window. What `child_arg_parse_us`
actually times is `main` entry to `worker_main` entry, which is
`Cli::parse()` parsing one optional flag and nothing else. Its own
0.064 ms median, in the Stage table below, is consistent with that: too
small to be library loading, right for one argument parse. `execve` and
`dyld` both run before `main`, on the child's side, entirely inside the
parent's `wait_us` window. This section does not measure either one, and
does not estimate their cost from this counter. See Gap accounting for a
bound on that cost from a different, direct measurement.

`wait_inner`'s poll loop, `crates/ane-miner/src/process.rs:167`, sleeps
10 ms between checks for the child's exit. This adds up to 10 ms to
`wait_us` beyond the child's real lifetime, about 5 ms on average across
many jobs. This section does not correct for it. It affects every figure
in this section that includes `wait_us`, including the 25.616 ms residual
in Gap accounting below.

Five runs, one at a time with a 3-second sleep, of a real `--ane-worker`
job holding production's own request shape: the topology in
`tests/fixtures/advantage2-system1.edges`, couplings in {-1, 1} from a
seeded xorshift64 generator with seed 7, zero fields, 128 reads, solved
with seed 123, 2 sweeps. This is the same topology and coupling seed
Task 7's Rust-side harness uses, run through `WorkerProcess::spawn` and
`WorkerProcess::wait` directly against the real worker binary, the same
pattern `hardware_rust_setup_stage_medians_advantage2_system1` set. The
new test, `hardware_worker_path_stage_medians_advantage2_system1` in
`crates/ane-miner/src/process.rs`, is `#[ignore = "requires integrated
Apple Silicon ANE worker binary"]` like the file's other hardware tests.
It also prints the reply's `RunStats` fields, so this section can compare
its own counters against Tasks 1 and 7 without a second device run.

No shared device guard exists in this repository or in `/tmp`, the same
gap every prior section in this document finds. The five runs below ran
one at a time, with a 3-second sleep between runs, in place of a guard.

Command, run five times:

```bash
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --release \
  --lib hardware_worker_path_stage_medians_advantage2_system1 \
  -- --ignored --nocapture --exact \
  process::tests::hardware_worker_path_stage_medians_advantage2_system1
```

### Stage table

All values are milliseconds, converted from the microseconds the counters
record.

| Stage | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| directory_setup_us | 0.705 | 0.261 | 0.240 | 0.345 | 0.268 | 0.268 |
| request_write_us | 4.933 | 2.104 | 2.189 | 2.174 | 2.306 | 2.189 |
| spawn_us | 1.416 | 0.331 | 0.313 | 0.327 | 0.308 | 0.327 |
| wait_us | 634.613 | 372.733 | 378.740 | 392.353 | 364.025 | 378.740 |
| reply_read_us | 15.208 | 15.949 | 16.213 | 15.926 | 15.565 | 15.926 |
| teardown_us | 0.326 | 0.366 | 0.398 | 0.423 | 0.372 | 0.372 |
| child_arg_parse_us | 0.858 | 0.062 | 0.064 | 0.061 | 0.065 | 0.064 |
| **Full run total, spawn to reply consumed** | 657.201 | 391.744 | 398.093 | 411.548 | 382.844 | **398.093** |

The full run total row adds `directory_setup_us`, `request_write_us`, and
`spawn_us`, timed before `wait_us` starts, to `wait_us` itself, then to
`reply_read_us` and `teardown_us`, timed after `wait_us` ends. It excludes
`child_arg_parse_us`: that counter times a span inside `wait_us`, on the
child's side, so adding it on top of `wait_us` would count part of that
span twice. The bold median, 398.093 ms, is the median of
the five per-run totals, the same convention Task 1's stage table sets.
Summing the six per-stage medians instead, excluding `child_arg_parse_us`
for the reason just given, gives 397.822 ms, close enough to support
either figure. This report uses 398.093 ms for the rest of Gap accounting.
Summing `directory_setup_us`, `request_write_us`, `spawn_us`,
`reply_read_us`, and `teardown_us` alone, the parts of a job outside
`wait_us`, gives a wrapper cost of 19.082 ms. Run 1 shows the same
cold-start pattern as every earlier stage table in this document. The
median absorbs it.

The same five runs also carried `RunStats` in their reply, the same
fields Task 7's harness measures, confirming this section's job matches
Task 7's fixed-cost stages regardless of sweep count:

| Stage | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| setup_us | 372.037 | 342.903 | 346.355 | 357.544 | 333.469 | 346.355 |
| staging_us | 0.653 | 0.679 | 0.645 | 0.613 | 0.609 | 0.645 |
| dispatch_us | 2.516 | 2.593 | 2.756 | 2.660 | 2.500 | 2.593 |
| anneal_us | 6.619 | 6.709 | 6.836 | 6.705 | 6.481 | 6.705 |

`setup_us` does not depend on sweep count. Validation, graph preparation,
schedule construction, initial spins, compile, and reset all run before
the first dispatch. Its median here, 346.355 ms, is within 1.7% of Task
7's own 340.423 ms median from a 512-sweep job on the same topology and
seed. This cross-check supports both measurements.

### Gap accounting

The task brief's instruction is to add the worker-path total to the
309.462 ms native sum and the 5.647 ms Rust sum, then check the result
against the 508.162 ms budget. Read literally, "worker-path total" is the
preceding 19.082 ms wrapper figure, since the native and Rust sums already
cover the child's compile and setup.

| Quantity | Value, ms |
| --- | ---: |
| Native stage sum (Task 1) | 309.462 |
| Rust setup sum (Task 7) | 5.647 |
| Worker-path wrapper (this task) | 19.082 |
| New total | 334.191 |
| Budget for measured stages | 508.162 |
| Gap | 173.971 |
| Gap as a share of budget | 34.2% |

This does not close to within 15%, and the gap is worse than Task 7's own
33.0% or 38.1% figures. That is a warning sign, not an improvement:
309.462 plus 5.647 is a proxy for the child's setup cost, built from an
independent C probe and a set of Rust counters. This section's own job
carries the real `setup_us`, 346.355 ms, directly in the same reply. The
proxy falls short of the real figure by about 31 ms, the same
`AneProgram::compile` internal cost Task 7 already named at
`crates/ane-miner/src/native.rs:89` through `native.rs:106` and
`native.rs:137` through `native.rs:145`. Adding a wrapper on top of an
already-short proxy widens the gap instead of closing it.

A figure that avoids this proxy is the full run total from the preceding
Stage table, 398.093 ms, the real, directly measured median cost of one
job from `WorkerProcess::spawn` to the parent consuming the reply.
Subtracting the 309.462 ms native sum and the 5.647 ms Rust sum from that
figure gives a worker-path total of 82.984 ms that does not double-count
the child's compute.

| Quantity | Value, ms |
| --- | ---: |
| Native stage sum (Task 1) | 309.462 |
| Rust setup sum (Task 7) | 5.647 |
| Worker-path total, reconciled against the direct 398.093 ms figure | 82.984 |
| New total | 398.093 |
| Budget for measured stages | 508.162 |
| Gap | 110.069 |
| Gap as a share of budget | 21.7% |

Trust this second table over the first. It uses one direct, self-consistent
measurement instead of summing two methods with a known 31 ms disagreement
between them. Either way, the gap stays open past the 15% check, though it
has narrowed from Task 7's 33.0% best figure to 21.7%.

Four spots hold part of the remaining 110.069 ms, named here rather than
estimated by assertion. This section did not instrument any of the four
directly. It bounds the four together with one arithmetic check. Add
`child_arg_parse_us` (0.064 ms), `setup_us` (346.355 ms), and `anneal_us`
(6.705 ms), the child's own compute Tasks 1, 7, and this section's own
`RunStats` figures already name, and subtract that sum from `wait_us`'s
median. 378.740 minus 353.124 leaves 25.616 ms this section cannot
assign to one place:

- The child's own `execve` and `dyld` startup, before `main` runs and
  before `child_arg_parse_us`'s own window starts. See Method above for
  why `child_arg_parse_us` does not cover this. This section does not
  measure it, but bounds it: run five times, one at a time with a
  3-second sleep, `--capabilities` is a complete process lifetime that
  includes `execve`, `dyld`, argument parsing, and exit, with no device
  and no worker child. The team lead measured it directly and reports
  three repetitions at 0.00 s warm against `/usr/bin/time -p`, whose
  resolution is 10 ms, and one cold run at 0.35 s. This cost is under
  roughly 10 ms warm, likely a small part of the 25.616 ms bucket, not a
  large one. This is a bound from a different measurement, not a figure
  this section derived.
- `read_message(std::io::stdin().lock())`, `crates/ane-miner/src/worker.rs:190`,
  deserializes the request the parent already spent 2.189 ms writing.
  Deserializing costs more than writing raw bytes. `serde_json` walks and
  allocates for every field.
- `program.close()`, `crates/ane-miner/src/solver.rs:129`, releases the
  compiled ANE program and its input and output surfaces. It runs after
  `anneal_us` stops timing, so no counter in this document covers it.
- `write_message(std::io::stdout().lock(), &reply)`,
  `crates/ane-miner/src/worker.rs:212`, serializes and writes the reply,
  128 reads across 4,577 nodes as a spin array, before the parent's own
  `reply_read_us` timer starts on its side of the same file.

The 10 ms poll interval in `wait_inner`'s loop, named in Method above,
also inflates this same 25.616 ms bucket, by up to 10 ms and about 5 ms on
average. It is not a fifth occupant so much as noise on top of the four
above, since it does not correspond to any real work on the child's side.

The remaining 84.453 ms, 110.069 minus 25.616, sits outside the span this
section measures altogether: parent process startup before
`WorkerProcess::spawn` runs, and, for jobs sampled rather than checked,
the parent's own consensus scoring after the reply returns,
`crates/ane-miner/src/process.rs:337` through `process.rs:357`. Neither
received a counter in this task. A later task should time both if full
accounting matters.

### Persistent worker

A persistent worker would keep one child process alive across jobs
instead of spawning one per job. It would remove the four counters this
section ties to process lifecycle rather than to message content:
`directory_setup_us` (0.268 ms), `spawn_us` (0.327 ms), `child_arg_parse_us`
(0.064 ms), and `teardown_us` (0.372 ms), for a sum of 1.031 ms per job.
`request_write_us` and `reply_read_us` stay, since they time message
serialization, not process creation, and a persistent worker still has to
serialize and deserialize each job's request and reply.

That 1.031 ms understates the process-lifecycle saving, because
`child_arg_parse_us` does not cover the child's `execve` and `dyld`
startup, per Method above. A persistent worker would also remove that
cost, bounded above at roughly 10 ms warm by the same `--capabilities`
measurement Gap accounting cites. The fuller process-lifecycle saving is
1.031 ms plus up to about 10 ms, not 1.031 ms alone.

Even the fuller figure is small next to the 520 ms fixed-setup budget.
A persistent worker only pays off if the child can also skip recompiling the ANE
program for each job. Task 4's plan is to swap in new weights on one
compiled program instead of recompiling it, but Task 4 has not run, and
this section does not know whether that weight swap works. If it does, a
persistent worker would also remove the 283.887 ms `compile_ms` median
from Task 1's stage table, for a combined saving of 284.918 ms per job. If
it does not, the saving stays at 1.031 ms. State which figure you use, and
why, alongside Task 4's result.

## Weight swap

Task 4 tested whether overwriting `weights/weight_data.bin` in the staging
directory, then calling `unloadWithQoS:` followed by `loadWithQoS:`, can
change a compiled program's couplings without a new `compileWithQoS:` pass.
If it could, a persistent worker could reuse one compiled program across
jobs instead of paying `compile_ms` on every job.

**Outcome 3.** Across three runs, each in its own process with a 3-second
sleep before it, the output after the weight swap and reload equals the
output before it, byte for byte, not the host-computed prediction for the
negated weights. The runtime caches the couplings at compile time. The
283.887 ms `compile_ms` median in the preceding Stage table stays in the
fixed-setup budget. Nothing in this section changes it.

| Quantity | Median across 3 runs, ms |
| --- | ---: |
| unload_ms | 0.777 |
| reload_ms | 2.662 |
| compile_ms this task's outcome fails to remove | 283.887 |

The unload-plus-reload time, 3.439 ms at the median, is far below
`compile_ms`. That speed is moot: outcome 3 means the reload path produces
the wrong (unchanged) result regardless of how fast it runs.

Given outcome 3, the remaining option to remove the per-job compile is to
make the couplings a runtime input rather than a compile-time constant.
`docs/perf/2026-09-16-ane-local-routing.md` already measured one
runtime-input design, the local sweep, at 3.4 to 3.6 times slower than the
matched dense (compiled) control (`docs/perf/2026-09-16-ane-local-routing.md:6`).
Any runtime-input redesign for this bridge starts from that gap, not from
zero.

Full method, receipts, commands, and the host-side integer check, plus a
separate finding on whether production jobs' fields (`h`) can differ from
the fields used to compile a program, are in
`.superpowers/sdd/2026-09-17-ane-throughput/task-4-report.md`.

## Platform

The host is an Apple M4 Max running macOS 26.5.2, build 25F84, Darwin 25.5.0
arm64. The compiler is Apple clang 21.0.0, build clang-2100.1.1.101. The
Rust-side setup runs used the same host, plus rustc 1.98.1, commit
48a229cea, built 2026-09-01, and cargo 1.98.1.

The Method section's receipts are `/tmp/quip-ane-throughput-0917/run1.json`
through `run5.json`, `run1.time` through `run5.time`, and the merged
`/tmp/quip-ane-throughput-0917/setup-profile.json`, which holds all five runs
and the median of each stage. The Rust-side setup section's receipts are
`/tmp/quip-ane-throughput-0917/rust-setup-run1.log` through
`rust-setup-run5.log`, each the full output of one `cargo test` invocation.
The bit-exactness check's receipts are
`/tmp/quip-ane-throughput-0917/solve-input.json`, the problem sent to
`--solve`, `solve-before.json` and `solve-after.json`, its output before and
after this task's change, and `quip-ane-msa-before` and `quip-ane-msa-after`,
the two compiled binaries.

The Worker path section's receipts are
`/tmp/quip-ane-throughput-0917/worker-path-real-run1.log` through
`worker-path-real-run5.log`, each the full output of one `cargo test`
invocation, and, for the Harness cost subsection,
`/tmp/quip-ane-throughput-0917/solve-input-real-2sweeps.json` and
`solve-input-two-node.json`, the two problems sent to `--solve`, with
`solve-real-2sweeps-run1.json` through `run5.json`, `solve-two-node-run1.json`
through `run5.json`, and their matching `.time` files. Its bit-exactness
check reused the same `solve-input.json`, with
`quip-ane-msa-task8-before` and `quip-ane-msa-task8-after` as the two
compiled binaries and `solve-task8-before.json` and `solve-task8-after.json`
as their output.
