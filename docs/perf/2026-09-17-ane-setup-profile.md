# Setup profile for the Apple Neural Engine program, 2026-09-17

## Result

Compile is the largest stage. Its median is 283.887 ms, about 77% of the
required stage sum and 75% of the total time the probe measures.

The seven required stages sum to 366.974 ms at the median. The baseline fixed
setup is 520 ms, less about 11.838 ms of process spawn, for a budget of
508.162 ms. The required stages miss that budget by 141.188 ms, 27.8% of the
budget. This exceeds the 15% check in the task brief. See Gap accounting below
for the two candidate unmeasured stages.

## Method

The probe, `crates/ane-miner/probes/setup_profile.m`, includes
`crates/ane-miner/native/ane_bridge.m` and calls its `makeMIL`,
`makeWeightBlob`, `shape`, and `slice` functions unchanged. This matches the
convention `single_call.m` already uses in the same directory, and it removes
the risk of a copy drifting from the code that ships.

The probe times seven stages around the same calls `quip_ane_create` makes, in
the same order: `graph_prep_ms` (a synthetic stand-in for the Rust-side
weights and fields build, since the probe has no Rust), `mil_build_ms`,
`blob_build_ms`, `blob_write_ms`, `compile_ms`, `load_ms`, and
`surface_setup_ms`. It also times two stages outside the required schema,
`range_validate_ms` and `dlopen_class_setup_ms`, so the Gap accounting section
can point to those two stages instead of leaving the gap unexplained.

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

| Stage | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| graph_prep_ms | 46.863 | 56.065 | 60.217 | 56.041 | 56.242 | 56.065 |
| mil_build_ms | 3.172 | 2.905 | 2.967 | 2.970 | 2.903 | 2.967 |
| blob_build_ms | 10.399 | 10.094 | 9.870 | 10.046 | 9.955 | 10.046 |
| blob_write_ms | 5.850 | 5.221 | 5.390 | 5.357 | 5.436 | 5.390 |
| compile_ms | 281.772 | 280.266 | 285.039 | 283.887 | 287.671 | 283.887 |
| load_ms | 16.556 | 3.000 | 3.323 | 3.532 | 4.067 | 3.532 |
| surface_setup_ms | 2.362 | 2.089 | 2.621 | 3.670 | 3.013 | 2.621 |
| **Required stage sum** | 366.974 | 359.640 | 369.426 | 365.503 | 369.289 | **366.974** |
| range_validate_ms (extra) | 7.806 | 7.791 | 7.824 | 7.966 | 7.783 | 7.806 |
| dlopen_class_setup_ms (extra) | 3.381 | 2.546 | 2.863 | 2.583 | 2.851 | 2.851 |
| Probe internal total (extra) | 378.162 | 369.977 | 380.113 | 376.052 | 379.923 | 378.162 |
| Wall time, seconds (extra) | 0.38 | 0.38 | 0.39 | 0.39 | 0.39 | 0.39 |

Run 1 shows a 16.556 ms `load_ms`, well over the other four runs. This
matches a cold first-load cost against the private Apple Neural Engine (ANE)
runtime. The median absorbs it. No stage was too fast to measure: the
smallest, `mil_build_ms`, still spans 2.9 to 3.2 ms across runs, well past
the timer resolution.

`blob_bytes` and `mil_bytes` held constant across all five runs, as expected
for a deterministic build: 43,352,640 and 70,717.

## Gap accounting

The task brief asks whether the seven required stages sum to within 15% of
the 520 ms fixed setup, less process spawn. This check needs a process spawn
estimate, which the brief does not supply, so this report derives one instead
of assuming a value.

"Wall time" in the preceding table wraps the whole probe process, spawn
through exit, using `/usr/bin/time -p`. "Probe internal total" is
`graph_prep_ms` through `surface_setup_ms` plus the two extra stages, timed
from inside the process. The median gap between these two, 11.838 ms, is this
report's process spawn estimate. It covers `dyld` loading Foundation and
IOSurface before `main` runs, plus process exit.

| Quantity | Value, ms |
| --- | ---: |
| Baseline fixed setup | 520.000 |
| Estimated process spawn | 11.838 |
| Budget for measured stages | 508.162 |
| Required stage sum (median) | 366.974 |
| Gap | 141.188 |
| Gap as a share of budget | 27.8% |

Using the probe internal total instead of the required stage sum narrows the
gap only slightly: 508.162 minus 378.162 is 130.000 ms, 25.6% of budget.
Either way, the gap exceeds the 15% check.

Two stages inside the baseline's `stats.setup_us` window sit outside this
probe and outside the required schema:

- `prepare(graph)`, `crates/ane-miner/src/graph.rs:93`, called from
  `crates/ane-miner/src/solver.rs:47`. This is the real Rust-side graph
  coloring and layout step that `graph_prep_ms` stands in for with synthetic
  data in this probe. The probe does not run it, so its cost is not in this
  report.
- `program.reset(&packed)`, `crates/ane-miner/src/native.rs:184`, called from
  `crates/ane-miner/src/solver.rs:70`. This stages the initial spin state
  into the first IOSurface through `quip_ane_reset`. The probe never calls
  `quip_ane_reset`.

`stats.setup_us` in `solve_with_block`
(`crates/ane-miner/src/solver.rs:45`–`71`) starts before `prepare(graph)` and
ends after `program.reset(&packed)`, so both stages count toward the 520 ms
baseline. Neither has a receipt key in this task, so this report does not
estimate their cost. A later task should time them directly if full
accounting matters.

## Platform

The host is an Apple M4 Max running macOS 26.5.2, build 25F84, Darwin 25.5.0
arm64. The compiler is Apple clang 21.0.0, build clang-2100.1.1.101.

The receipts are `/tmp/quip-ane-throughput-0917/run1.json` through
`run5.json`, `run1.time` through `run5.time`, and the merged
`/tmp/quip-ane-throughput-0917/setup-profile.json`, which holds all five runs
and the median of each stage.
