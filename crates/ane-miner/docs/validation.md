# Validation

This file records results for multi-spin simulated annealing on the Apple
Neural Engine (ANE).
The first standalone version is commit `7c2b1b2`.
The earlier results remain below.
The sweep redesign section has new release data.

## Host and tools

The host is an Apple M4 Max, model Mac16,5. It has 128 GiB of memory.
The operating system is macOS 26.5.2. Its build is 25F84.

The compiler is version 1.98.1. Cargo is also version 1.98.1.
The rustc commit is `48a229cea`, dated 2026-09-01.
The local `+1.97.1` toolchain name refers to that compiler.
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
Hardware tests have their own passing receipts below.

Ordinary protocol tests:

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --test protocol
```

4 tests passed. The runner ignored 4 hardware tests.

The four passing tests run the real `quip-ane-msa` binary:

- `--capabilities` JSON matches `ane`, `msa`, width 1, 16,384 nodes, and 163,840 edges
- `--version` contains `protocol 1`
- `--check`, `--solve`, and `--capabilities` conflict
- `--ane-worker` conflicts with those modes and `--quip-coordinator`

Release `--capabilities` output:

```text
{"backend":"ane","algorithm":"msa","supportedKinds":["ISING_SAMPLE"],"maxNodes":16384,"maxEdges":163840,"features":["streaming"],"protocolVersion":1,"streamWidth":1}
```

Protocol 1 capabilities omit degree and coefficient limits.
A coordinator must use the documented limits.

Format and Clippy after solver integration:

```sh
cargo fmt --manifest-path crates/ane-miner/Cargo.toml --all -- --check
cargo clippy --manifest-path crates/ane-miner/Cargo.toml --locked --all-targets -- -D warnings
```

Both exited 0.
Clippy found no warnings.

## Hardware tests

The native bridge ran on this host.

Ordinary native tests:

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib native::tests
```

That run passed 5 tests. It ignored 6 hardware tests.

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

The suite ran 16 dispatches across six tests.
Nine explicit closes succeeded.
The rectangular test took 45 microseconds for staging. It took 208
microseconds for dispatch.
Those single times exclude compilation.
They do not show mining throughput.
This file makes no speed claim versus Metal.

Release `--check` opened one startup worker on the same host. It exited 0.

## Capacity tests

Every capacity gate passed.
The variable limit stays at 16,384. The supplied edge limit stays at 163,840.
The degree limit is 20. The other limits are 128 reads and 65,536 sweeps.

The capacity tests used 128 reads. Each test used four sweeps.
Each capacity ran in a fresh process under `/usr/bin/time -l`.

```sh
/usr/bin/time -l cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib solver::tests::hardware_capacity_6016 -- --ignored --exact
/usr/bin/time -l cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib solver::tests::hardware_capacity_8192 -- --ignored --exact
/usr/bin/time -l cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib solver::tests::hardware_capacity_16384 -- --ignored --exact
/usr/bin/time -l cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib solver::tests::hardware_isolated_16384 -- --ignored --exact
```

The production values come from `solve_in_process`. They cover setup, staging,
dispatch, anneal, and wall time.
Production wall includes validation and setup. It includes annealing and
result conversion. It also includes program close.

| Variables | Graph | Colors | Tiles and shapes | Dispatches | Production setup | Production staging | Production dispatch | Production anneal | Production wall | Peak resident set size |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6,016 | degree 20 | 4 | 4 × 6016 by 1504 | 16 | 1,051,143 µs | 56,678 µs | 6,608 µs | 112,755 µs | 1,179,000 µs | 73,154,560 B, 69.77 MiB |
| 8,192 | degree 20 | 4 | 4 × 8192 by 2048 | 16 | 1,819,935 µs | 76,938 µs | 36,678 µs | 172,981 µs | 2,011,611 µs | 109,854,720 B, 104.77 MiB |
| 16,384 | degree 20 | 4 | 4 × 16384 by 4096 | 16 | 7,009,489 µs | 154,515 µs | 427,784 µs | 683,073 µs | 7,734,679 µs | 319,389,696 B, 304.59 MiB |
| 16,384 | isolated | 1 | 4 × 16384 by 4096 | 16 | 6,740,010 µs | 155,364 µs | 424,704 µs | 676,492 µs | 7,458,534 µs | 295,960,576 B, 282.25 MiB |

Each test then ran a CPU oracle. It also ran ANE a second time for a per-color
check. At 6,016 variables, validation setup took 1,029,000
microseconds. Annealing took 1,016,797 microseconds. At 8,192 variables, the
times were 1,793,591 and 1,402,240 microseconds. The degree 20 case at 16,384
variables took 6,845,511 and 3,086,040 microseconds. The isolated case took
6,703,811 and 1,012,468 microseconds.
Validation anneal values include CPU oracle comparisons inside the color loop.
They are not production miner timings.

Peak resident set size is for the test process.
That value includes the production run. It also includes both validation runs.
It excludes memory that the runtime and driver hold outside the test process.
The value is not a total ANE memory budget.
The `/usr/bin/time -l` full-test wall times follow the table order. They were
4.20, 6.48, 20.09, and 15.76 seconds.

The largest four-tile shape is four 16,384 by 4,096 matrices.
Raw dense FP16 payload at that shape is 512 MiB.
The padded payload stays below 544 MiB in the worst case.

Irregular-color oracles covered complete graphs with 1, 3, and 21 variables.
The 21-variable graphs have degree 20. They use the same path.

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib solver::tests::hardware_irregular_colors -- --ignored --exact
```

Read-count runs found zero mismatches. They covered 1, 31, 32, 33, 127, and
128 reads.

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib solver::tests::hardware_read_counts -- --ignored --exact
```

Logical reads may be 1 through 128.
The first runs used 128 physical lanes for every job.
The current program allocates physical lanes in groups of 32.
Hardware checks on 2026-09-21 matched the oracle. They covered 1, 16, 31, 32,
33, 64, 65, 96, 127, and 128 reads.
At each node and sweep, each group of 32 replicas shares one threshold value.
The active groups use independent streams, up to four at 128 reads.
Nodes do not share one global threshold.

## Lifetime tests

Process gates passed with the real solver.

Build the debug worker before the process hardware tests:

```sh
cargo build --manifest-path crates/ane-miner/Cargo.toml --locked --bin quip-ane-msa
```

Process hardware tests load `crates/ane-miner/target/debug/quip-ane-msa`.
Each test starts that path with `CARGO_MANIFEST_DIR`. It then adds
`target/debug/quip-ane-msa`.

Focused library tests:

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib worker::tests
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --lib process::tests
```

Those commands passed 7 worker tests. They also passed 16 process tests.
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
Child process 75801 finished one dispatch.
Wait removed its job directory.

The 64-job gate used distinct child processes.
It compiled 256 programs. It compared 8,192 reads with the count-based oracle.
Each child finished 64 dispatches. It then exited.
Wait removed every job directory.

Large-job cancellation used 16,384 variables and 128 reads. It used 65,536
sweeps. Wait reaped child 91645 in 63,085 microseconds. It removed the job
directory.

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
`--capabilities` printed the identity JSON from Host tests. It exited 0.
`--check` opened the real ANE. Native startup finished. The process then exited 0.

Supported-fixture hardware protocol tests must use the release binary:

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --release --test protocol -- --ignored --test-threads=1 --nocapture
```

All four hardware tests passed in 3.89 seconds.

The suite covers the 33-read solve. It covers empty graphs and zero sweeps. It
also covers unsupported limits.
The 33-read fixture with seed 123 returned 33 consensus-scored reads.
Each energy was `-1000`.
A second run returned identical bytes.
Empty graphs succeeded. Zero sweeps also succeeded.
Degree 21 failed with no JSON array. A request for 129 reads also failed. The
same result occurred for 65,537 sweeps and coefficient `0.5`.
The two coefficient jobs print the production rejection message.

The supported session uses a local test coordinator.
It keeps the public `DriverReport::is_conformant` checks.
This gate does not use the stock `quip-solver-conformance` 0.0.1 driver.
That stock driver still cannot pass. See Stock driver below.

Inline jobs use h `[1, -1]` and J `[1]`. Dense cached jobs use the same values.
The sparse job uses h `[1, -1, 0]` and J `[1, -1]`.
Sparse native node IDs stay `[0, 12, 2400]` with their original edges.
Two jobs send fractional coefficients before the supported jobs. One sends h
`0.5`, and one sends J `0.5`.

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
The session resumed after the fractional-coefficient rejects. It then scored
the four jobs.

Unit coefficients remain the only supported domain.
Protocol 1 capabilities omit degree and coefficient limits.
Coordinator routing must use the documented limits.

### Debug phase budget

The same hardware command failed the supported-job phase without `--release`.
Three tests passed. The coordinator session exceeded the unchanged ten-second budget.

A direct 512-sweep `--solve` diagnostic used this JSON:

```json
{"h":[1.0,-1.0],"j":[1.0],"edges":[[0,1]],"num_reads":1,"num_sweeps":512,"sweeps_per_beta":1,"beta_range":null,"seed":123}
```

Debug took 12.211 seconds. Release took 0.837 seconds.
Debug used 11.93 seconds of user CPU time.
Default sessions use one sweep per inverse-temperature rung.
Each rung fills four 8,192-entry threshold arrays on the CPU.
Those times describe the test profile.
They are not a Metal comparison.

### Stock driver

The stock `quip-solver-conformance` 0.0.1 `drive_miner` fixtures still fail.

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --test protocol -- --ignored --test-threads=1
```

That older stock-driver run passed three solve tests. It failed the coordinator
walk.
No phase timed out. The process exited 0. It also closed the stream.

The stock fixtures use `j_milli=500` for the two-spin jobs. This value is
coupling 0.5.
The sparse stock job uses `h=[1.0, -1.0, 0.25]` and `j=[0.5, -0.75]`.
The miner rejected four jobs as `TooLarge`. They were `job-1`, `job-2`,
`job-hash`, and `job-sparse`.
The parent logged `coefficient must be -1, 0, or 1` before each reject.
`--solve` also rejects coefficient `0.5`.

The supported-fixture gate does not change the stock mismatch.
This miner accepts only integer unit coefficients.
It does not claim support for all coordinator inputs.

## Sweep redesign, September 15, 2026

These changes start from merge commit `4ec7ed8746ce5bcd41b2c5e5584cc9d2dba8dbc8`.
Programs share one neighbor input surface. They upload changed rows after the
first dispatch.
Tile buffers persist across sweeps.
Threshold draws use binary search. They keep the same cutoff checks. They also
keep the same random streams.

The adaptive range changes from 64–256 to 2,048–8,192 sweeps, with 128 reads.
The hard cap remains 65,536 sweeps.
A zero-field target of `-14612` selects 8,049 sweeps. This problem has 4,577
nodes and 41,514 edges.
The new regression test first failed with the old result of 251 sweeps.
Job overrides come first. Target overrides come next.
The miner still uses the set sweep count as a fallback after adaptation.

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

The native totals include host tests from the normal run. The solver totals
include those tests too.
The exhaustive native check covered 32,384 integer acceptance cases. It found
zero mismatches.
Shared surface tests covered owner lifetime. They also covered different output
shapes. Row update tests checked changed data. They also checked invalid row
metadata.
Capacity tests passed at 6,016 nodes. They also passed at 8,192 and 16,384
nodes. Final states matched the CPU oracle. Each checked color also matched.

The selected process tests used a local symlink. It connected their fixed debug
path to the release executable. A 16,384-node job requested 65,536 sweeps.
Cancellation began at 50 milliseconds. The job stopped after 62.8 milliseconds.
The parent reaped the child. It also removed the temporary directory.
Closing the output channel stopped the worker. It also removed the directory.
The protocol check covered live cancellation. It covered credit refunds too.

### Controlled mining replay

The saved live problem has 4,577 variables and 41,514 edges. It has zero fields
and unit signed couplings.
Each run returned 128 states.
A second scorer checked every returned energy using the original graph.
The local mining process did not run during these new tests.

Binary `SHA-256` hashes:

```text
Before: fb24ec381f618de275032841d03788df1b4ccd22b748905e901f5849c76c7a6e
After:  5c93e4f688a008910e2695d0c64fbf09544632266ad7bd230572a1f4ff8f68e1
```

Three paired runs used 1,024 sweeps. They used consecutive seeds starting at
`3302488336868276095`.
The order was before/after, after/before, then before/after.

| Seed offset | Before, seconds | After, seconds | Lowest energy |
| --- | ---: | ---: | ---: |
| 10 | 10.0506 | 7.6401 | -14322 |
| 11 | 9.9765 | 7.5924 | -14328 |
| 12 | 11.2235 | 8.6062 | -14336 |

Each pair produced byte-identical JSON.
Every spin matched. Every energy also matched.
The median paired speed ratio was 1.314, or 23.9% less elapsed time.
The 251-sweep check produced byte-identical JSON. The 4,096-sweep check did too.

The following new-binary runs use seed `3302488336868276085`:

| Sweeps | Whole-process seconds | Lowest energy | Median energy | States at or below `-14612` |
| ---: | ---: | ---: | ---: | ---: |
| 251 | 2.2387 | -14226 | -14122 | 0 |
| 4096 | 28.3303 | -14378 | -14324 | 0 |
| 8049 | 55.1571 | -14386 | -14344 | 0 |

The 8,049-sweep run made 64,392 dispatches through eight color programs.
It spent 0.308 seconds in setup and 19.677 seconds staging data. Device dispatch
took 11.032 seconds.
Total annealing time was 54.714 seconds. This time includes host work outside
those counters.
More sweeps improved this job, but no returned state reached the target.
These runs do not show a qualifying rate across mining jobs.

### Comparison with the reference miner

[MR 27](https://gitlab.com/quip.network/quip-miner-cuda/-/merge_requests/27)
sets the `CUDA` miner to 7,392–29,568 adaptive sweeps. It uses 128 reads.
That target and topology select 29,053 sweeps there.
Both miners use ordered color updates. Both use Metropolis acceptance.

Source inspection points to a benchmark mismatch in [MR 26](https://gitlab.com/quip.network/quip-miner-cuda/-/merge_requests/26).
The default benchmark preset permits fields in `{-1, 0, 1}`. It has 41,515
edges. The captured live problem has zero fields. It has 41,514 edges.
The MR does not publish the full command. It also omits the raw records needed
to reconstruct all 24 benchmark jobs.
Its absolute energy values do not establish quality for this live problem.

Same-problem comparisons used Metal and CPU solvers. They showed similar energy
distributions at matched sweep counts.
All returned energies passed the second scoring check.
This evidence does not identify an ANE-specific Metropolis or scoring defect.

## Follow-up work

Engine selection is bead `quip-miner-metal-7zk`.
Optimization is bead `quip-miner-metal-djn`.
Both remain open.
