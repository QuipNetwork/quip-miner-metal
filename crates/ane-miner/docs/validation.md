# Validation

This file records Apple Neural Engine (ANE) multi-spin simulated annealing results.
The first standalone version is commit `7c2b1b2`.
The earlier measurements remain below.
The sweep redesign has a separate section with new release measurements.

## Host and tools

The host is an Apple M4 Max, model Mac16,5, with 128 GiB memory.
The operating system is macOS 26.5.2, build 25F84.

Actual compiler and Cargo versions are 1.98.1.
The rustc commit is `48a229cea`, dated 2026-09-01.
The local `+1.97.1` toolchain name aliases that compiler.
The crate rust-version field is 1.97.
These checks did not run a Rust 1.97 compiler.

## Host tests

Graph tests:

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib graph::tests
```

All 7 tests passed.

Schedule tests:

```sh
cargo +1.97.1 test --manifest-path crates/ane-miner/Cargo.toml --locked --lib msa::tests
```

All 12 tests passed.

Integrated library tests:

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --lib
```

49 host tests passed.
That run ignored 18 hardware tests.
Do not add those 49 host tests to the hardware counts.
Hardware tests have separate passing receipts below.

Ordinary protocol tests:

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --test protocol
```

4 tests passed. The runner ignored 4 hardware tests.

The four passing tests drive the real `quip-ane-msa` binary:

- `--capabilities` JSON matches `ane`, `msa`, width 1, 16,384 nodes, and 163,840 edges
- `--version` contains `protocol 1`
- `--check`, `--solve`, and `--capabilities` conflict
- `--ane-worker` conflicts with those modes and `--quip-coordinator`

Release `--capabilities` output:

```text
{"backend":"ane","algorithm":"msa","supportedKinds":["ISING_SAMPLE"],"maxNodes":16384,"maxEdges":163840,"features":["streaming"],"protocolVersion":1,"streamWidth":1}
```

Protocol 1 capabilities cannot advertise degree or coefficient limits.
A coordinator must honor the documented degree and coefficient domain.

Format and Clippy after solver integration:

```sh
cargo fmt --manifest-path crates/ane-miner/Cargo.toml --all -- --check
cargo clippy --manifest-path crates/ane-miner/Cargo.toml --locked --all-targets -- -D warnings
```

Both exited 0.
Clippy reported no warnings.

## Hardware tests

The native bridge ran on this host.

Ordinary native tests:

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib native::tests
```

That run passed 5 tests and ignored 6 hardware tests.

Native hardware tests:

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib native::tests -- --ignored --test-threads=1 --nocapture
```

That command passed all 6 hardware tests.

| Hardware test | Successful dispatches | Result |
| --- | ---: | --- |
| Exhaustive degrees 0 through 21, both spin signs | 8 | 32,384 cases, zero mismatches |
| Input binding and acceptance ties | 1 | Distinct threshold changes the expected lane |
| Opposite couplings and different fields | 2 | Distinct outputs, 31 padded channels |
| Weight-only identity and field-only effect | 3 | No coupling cache aliasing |
| Rectangular row-major matrix | 1 | Correct 64-input, 32-output mapping |
| Zero weights and invalid inputs | 1 | Correct flips and eight rejected inputs |

The suite completed 16 dispatches across six tests.
It recorded nine successful explicit closes.
The rectangular test reported staging of 45 microseconds and dispatch of 208 microseconds.
Those single timings exclude compilation.
They do not establish mining throughput.
This file makes no speed claim versus Metal.

Release `--check` opened one startup worker on the same host and exited 0.

## Capacity tests

Every capacity gate passed.
Advertised limits stay 16,384 variables, 163,840 supplied edges, degree 20, 128 reads, and 65,536 sweeps.

The capacity tests used 128 reads and four sweeps.
Each capacity ran in a fresh process under `/usr/bin/time -l`.

```sh
/usr/bin/time -l cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib solver::tests::hardware_capacity_6016 -- --ignored --exact
/usr/bin/time -l cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib solver::tests::hardware_capacity_8192 -- --ignored --exact
/usr/bin/time -l cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib solver::tests::hardware_capacity_16384 -- --ignored --exact
/usr/bin/time -l cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib solver::tests::hardware_isolated_16384 -- --ignored --exact
```

Production setup, staging, dispatch, anneal, and wall values come from `solve_in_process`.
Production wall includes validation, setup, annealing, result conversion, and program close.

| Variables | Graph | Colors | Tiles and shapes | Dispatches | Production setup | Production staging | Production dispatch | Production anneal | Production wall | Peak resident set size |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6,016 | degree 20 | 4 | 4 × 6016 by 1504 | 16 | 1,051,143 µs | 56,678 µs | 6,608 µs | 112,755 µs | 1,179,000 µs | 73,154,560 B, 69.77 MiB |
| 8,192 | degree 20 | 4 | 4 × 8192 by 2048 | 16 | 1,819,935 µs | 76,938 µs | 36,678 µs | 172,981 µs | 2,011,611 µs | 109,854,720 B, 104.77 MiB |
| 16,384 | degree 20 | 4 | 4 × 16384 by 4096 | 16 | 7,009,489 µs | 154,515 µs | 427,784 µs | 683,073 µs | 7,734,679 µs | 319,389,696 B, 304.59 MiB |
| 16,384 | isolated | 1 | 4 × 16384 by 4096 | 16 | 6,740,010 µs | 155,364 µs | 424,704 µs | 676,492 µs | 7,458,534 µs | 295,960,576 B, 282.25 MiB |

Each test then ran a separate CPU oracle and a second ANE run for per-color comparison.
Those validation setup and anneal times were 1,029,000 and 1,016,797 microseconds at 6,016 variables.
The times were 1,793,591 and 1,402,240 at 8,192 variables.
The degree 20 case at 16,384 variables took 6,845,511 and 3,086,040.
The isolated case took 6,703,811 and 1,012,468.
Validation anneal values include CPU oracle comparisons inside the color loop.
They are not production miner timings.

Peak resident set size is for the test process.
That measurement includes the production run and both validation runs.
It excludes separate runtime and driver allocations.
The value is not a total ANE memory budget.
The `/usr/bin/time -l` full-test wall times were 4.20, 6.48, 20.09, and 15.76 seconds in table order.

The largest four-tile shape is four 16,384 by 4,096 matrices.
Raw dense FP16 payload at that shape is 512 MiB.
Padded worst-case payload stays below 544 MiB.

Irregular-color oracles covered complete graphs with 1, 3, and 21 variables.
Those 21-variable complete graphs have degree 20 and use the same path.

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib solver::tests::hardware_irregular_colors -- --ignored --exact
```

Read-count runs matched 1, 31, 32, 33, 127, and 128 reads with zero mismatches.

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib solver::tests::hardware_read_counts -- --ignored --exact
```

Logical reads may be 1 through 128.
Execution always uses 128 physical lanes.
At each node and sweep, each group of 32 replicas shares one threshold value.
The four groups use independent streams.
Nodes do not share one global threshold.

## Lifetime tests

Process gates passed with the real solver.

Build the debug worker before the process hardware tests:

```sh
cargo build --manifest-path crates/ane-miner/Cargo.toml --locked --bin quip-ane-msa
```

Process hardware tests load `crates/ane-miner/target/debug/quip-ane-msa`.
Each test builds that path from `CARGO_MANIFEST_DIR` plus `target/debug/quip-ane-msa`.

Focused library tests:

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib worker::tests
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib process::tests
```

Those commands passed 7 worker tests and 16 process tests.
The host command ignored five hardware tests.

Process hardware tests:

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib process::tests -- --ignored --test-threads=1 --nocapture
```

Those five tests then passed on ANE.
Each hardware test starts a fresh child process.

The hardware filters are:

- `process::tests::hardware_startup_dispatch_receipt`
- `process::tests::hardware_cancellation_reaps_large_job`
- `process::tests::hardware_output_channel_close_stops_sampling_worker`
- `process::tests::hardware_64_jobs_use_distinct_children_and_match_oracle`
- `process::tests::watchdog_exits_when_test_parent_exits`

Startup dispatch used `crates/ane-miner/target/debug/quip-ane-msa`.
Child process 75801 completed one dispatch.
Wait removed its job directory.

The 64-job gate used distinct child processes.
It compiled 256 programs and compared 8,192 reads with the count-based oracle.
Each child completed 64 dispatches and then exited.
Wait removed every job directory.

Large-job cancellation used 16,384 variables, 128 reads, and 65,536 sweeps.
Wait reaped child 91645 in 63,085 microseconds and removed its directory.

Stream closure passed for child 94219.
Cleanup took 17,269 microseconds after the receiver closed.

## Protocol tests

Release build and device probe:

```sh
cargo build --manifest-path crates/ane-miner/Cargo.toml --locked --release --bin quip-ane-msa
crates/ane-miner/target/release/quip-ane-msa --capabilities
crates/ane-miner/target/release/quip-ane-msa --check
```

The release build exited 0.
`--capabilities` printed the identity JSON from Host tests and exited 0.
`--check` opened the real ANE. Native startup completed. The process then exited 0.

Supported-fixture hardware protocol tests must use the release binary:

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --release --test protocol -- --ignored --test-threads=1 --nocapture
```

All four hardware tests passed in 3.89 seconds.

The suite covers the 33-read solve, empty graphs, zero sweeps, and unsupported limits.
The 33-read fixture with seed 123 returned 33 consensus-scored reads.
Each energy was `-1000`.
A second run returned identical bytes.
Empty graphs and zero sweeps succeeded.
Degree 21, 129 reads, 65,537 sweeps, and coefficient `0.5` failed with no JSON array.
The two coefficient jobs emit the production rejection diagnostic.

The supported coordinator session uses a local test-only coordinator.
It keeps the public `DriverReport::is_conformant` checks.
The stock `quip-solver-conformance` 0.0.1 driver is not this gate.
That stock driver still cannot pass. See Stock driver below.

Inline and dense cached jobs use h `[1, -1]` and J `[1]`.
The sparse job uses h `[1, -1, 0]` and J `[1, -1]`.
Sparse native node IDs stay `[0, 12, 2400]` with their original edges.
Separate jobs send fractional h `0.5` and fractional J `0.5` before the supported jobs.

Observed supported-fixture session:

| Check | Result |
| --- | --- |
| Handshake `Hello` | pass. `backend=ane`, `algorithm=msa`, `max_nodes=16384`, `max_edges=163840` |
| Ready after Configure | pass |
| Declared width | pass. `stream_width=1` |
| Credit ledger | pass. initial grant 3, twelve refunds of 1, `jobs_dispatched=12` |
| Supported results | pass. one result each for `job-1`, `job-2`, `job-hash`, and `job-sparse` |
| Consensus energy | pass. `-3000` on the two-spin jobs, `-4000` on the sparse job |
| Configured sweeps 512 | pass. `SamplerMeta` reports 512 on every result |
| Fractional coefficients | pass. `job-fractional-h` and `job-fractional-j` rejected as `TooLarge` |
| Malformed `h` / `j` | pass. `job-bad-h` and `job-bad-j` rejected as `Malformed` |
| Unsupported kind | pass. `job-gate` rejected as `UnsupportedKind` |
| Expired job | pass. `job-old` rejected as `Expired` |
| Ping acknowledgement | pass |
| Live cancellation | pass. no result or reject for the cancelled job, credit returned |
| Stale job | pass. no result or reject for `job-stale` |
| Clean exit | pass. `terminal=Closed`, `exit_code=0`. Fatal is absent. Timed-out phases is empty. |

Six expected rejects occurred.
Each of the twelve dispatched jobs refunded one credit.
The session recovered after the fractional rejects and then scored the four supported jobs.

Unit coefficients remain the only supported domain.
Protocol 1 capabilities cannot advertise degree or coefficient restrictions.
Coordinator routing must honor the documented limits.

### Debug phase budget

The same hardware command without `--release` failed the supported-job phase.
Three tests passed. The coordinator session exceeded the unchanged ten-second budget.

A direct 512-sweep `--solve` diagnostic used this JSON:

```json
{"h":[1.0,-1.0],"j":[1.0],"edges":[[0,1]],"num_reads":1,"num_sweeps":512,"sweeps_per_beta":1,"beta_range":null,"seed":123}
```

Debug took 12.211 seconds. Release took 0.837 seconds.
Debug used 11.93 seconds of user CPU time.
Default sessions use one sweep per inverse-temperature rung.
Each rung fills four 8,192-entry threshold arrays on the CPU.
Those times diagnose the test profile.
They are not a Metal comparison.

### Stock driver

The stock `quip-solver-conformance` 0.0.1 `drive_miner` fixtures still fail.

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --test protocol -- --ignored --test-threads=1
```

That older stock-driver run passed three solve tests and failed the coordinator walk.
No phase timed out. The process exited 0 and closed the stream.

The stock fixtures use `j_milli=500`, which is coupling 0.5, for the two-spin jobs.
The sparse stock job uses `h=[1.0, -1.0, 0.25]` and `j=[0.5, -0.75]`.
The miner rejected `job-1`, `job-2`, `job-hash`, and `job-sparse` as `TooLarge`.
The parent logged `coefficient must be -1, 0, or 1` before each reject.
`--solve` also rejects coefficient `0.5`.

The supported-fixture gate does not change that stock incompatibility.
This miner does not accept fractional coefficients.
It does not claim unrestricted coordinator compatibility.

## Sweep redesign, September 15, 2026

These changes start from merge commit `4ec7ed8746ce5bcd41b2c5e5584cc9d2dba8dbc8`.
Programs share one neighbor input surface and upload changed rows after the first dispatch.
Tile buffers persist across sweeps.
Threshold draws use binary search with the same cutoff comparisons and random streams.

The adaptive range changes from 64–256 to 2,048–8,192 sweeps, with 128 reads.
The hard cap remains 65,536 sweeps.
A zero-field target of `-14612` on 4,577 nodes and 41,514 edges selects 8,049 sweeps.
The new regression test first failed with the old result of 251 sweeps.
Explicit job and target overrides keep their precedence.
The configured sweep count remains a fallback after adaptation.

### Release checks

All commands use the same package and release output directory:

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --target-dir target --locked --release
cargo test --manifest-path crates/ane-miner/Cargo.toml --target-dir target --locked --release --lib native::tests:: -- --include-ignored --test-threads=1
cargo test --manifest-path crates/ane-miner/Cargo.toml --target-dir target --locked --release --lib solver::tests:: -- --include-ignored --test-threads=1
cargo test --manifest-path crates/ane-miner/Cargo.toml --target-dir target --locked --release --lib process::tests::hardware_ -- --ignored --skip hardware_64_jobs_use_distinct_children_and_match_oracle --test-threads=1
cargo test --manifest-path crates/ane-miner/Cargo.toml --target-dir target --locked --release --test protocol -- --ignored --test-threads=1
cargo clippy --manifest-path crates/ane-miner/Cargo.toml --target-dir target --locked --release --all-targets --all-features -- -D warnings
```

| Check | Result |
| --- | --- |
| Ordinary library tests | 52 passed, 20 hardware tests ignored |
| Ordinary command-line tests | 4 passed, 4 hardware tests ignored |
| Native tests, including hardware | 14 passed |
| Solver tests, including hardware | 9 passed |
| Selected worker hardware tests | 3 passed |
| Hardware protocol tests | 4 passed |
| Clippy, formatting, and diff whitespace | pass |

The native and solver totals include host tests from the ordinary run.
The exhaustive native check covered 32,384 integer acceptance cases with zero mismatches.
Shared surface tests covered owner lifetime and different output shapes.
Row update tests checked changed data and invalid row metadata.
Capacity tests passed at 6,016, 8,192, and 16,384 nodes.
Final states and each checked color matched the CPU oracle.

The selected process tests used a local symlink from their fixed debug path to the release executable.
A 16,384-node job with 65,536 requested sweeps stopped after 62.8 milliseconds when cancellation began at 50 milliseconds.
The parent reaped the child and removed its temporary directory.
Closing the output channel also stopped the worker and removed its directory.
The protocol check covered live cancellation and credit refunds.

### Controlled mining replay

The saved live problem has 4,577 variables, 41,514 edges, zero fields, and unit signed couplings.
Each run returned 128 states.
An independent scorer checked every returned energy using the original graph.
The local mining process did not run during these new measurements.

Binary `SHA-256` hashes:

```text
Before: fb24ec381f618de275032841d03788df1b4ccd22b748905e901f5849c76c7a6e
After:  5c93e4f688a008910e2695d0c64fbf09544632266ad7bd230572a1f4ff8f68e1
```

Three paired runs used 1,024 sweeps and consecutive seeds starting at `3302488336868276095`.
The order was before/after, after/before, then before/after.

| Seed offset | Before, seconds | After, seconds | Lowest energy |
| --- | ---: | ---: | ---: |
| 10 | 10.0506 | 7.6401 | -14322 |
| 11 | 9.9765 | 7.5924 | -14328 |
| 12 | 11.2235 | 8.6062 | -14336 |

Each pair produced byte-identical JSON.
Every spin and energy matched.
The median paired speed ratio was 1.314, or 23.9% less elapsed time.
Comparisons at 251 and 4,096 sweeps also produced byte-identical JSON.

The following new-binary runs use seed `3302488336868276085`:

| Sweeps | Whole-process seconds | Lowest energy | Median energy | States at or below `-14612` |
| ---: | ---: | ---: | ---: | ---: |
| 251 | 2.2387 | -14226 | -14122 | 0 |
| 4096 | 28.3303 | -14378 | -14324 | 0 |
| 8049 | 55.1571 | -14386 | -14344 | 0 |

The 8,049-sweep run made 64,392 dispatches through eight color programs.
It spent 0.308 seconds in setup, 19.677 seconds staging data, and 11.032 seconds in device dispatch.
Total annealing time was 54.714 seconds, including host work outside those counters.
More sweeps improved this job, but no returned state reached the target.
These measurements do not establish a qualifying rate across mining jobs.

### Comparison with the reference miner

[MR 27](https://gitlab.com/quip.network/quip-miner-cuda/-/merge_requests/27) sets the `CUDA` miner to 7,392–29,568 adaptive sweeps and 128 reads.
The same target and topology select 29,053 sweeps there.
Both implementations use ordered color updates and Metropolis acceptance.

Source inspection points to a benchmark mismatch in [MR 26](https://gitlab.com/quip.network/quip-miner-cuda/-/merge_requests/26).
The default benchmark preset permits fields in `{-1, 0, 1}` and has 41,515 edges.
The captured live problem has zero fields and 41,514 edges.
The MR does not publish the full command and raw records needed to reconstruct all 24 benchmark jobs.
Its absolute energy values do not establish quality for this live problem.

Same-problem comparisons with Metal and CPU solvers showed similar energy distributions at matched sweep counts.
All returned energies passed independent scoring.
This evidence does not identify an ANE-specific Metropolis or scoring defect.

## Follow-up work

Engine selection is bead `quip-miner-metal-7zk`.
Optimization is bead `quip-miner-metal-djn`.
Both remain open.
