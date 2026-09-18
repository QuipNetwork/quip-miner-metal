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

## Platform

The host is an Apple M4 Max running macOS 26.5.2, build 25F84, Darwin 25.5.0
arm64. The compiler is Apple clang 21.0.0, build clang-2100.1.1.101.

The receipts are `/tmp/quip-ane-throughput-0917/run1.json` through
`run5.json`, `run1.time` through `run5.time`, and the merged
`/tmp/quip-ane-throughput-0917/setup-profile.json`, which holds all five runs
and the median of each stage.
