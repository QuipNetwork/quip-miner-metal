# `quip-ane-msa`

`quip-ane-msa` is a standalone Apple Neural Engine (ANE) miner.
It runs multi-spin simulated annealing for protocol 1.
The binary advertises `ane` and `msa`.
Default miner id is `ane-0`.

## Platform

This miner runs on Apple Silicon macOS only.
It uses the private ANE runtime.
The miner has no CPU or GPU fallback.
A failed ANE dispatch returns an explicit error.

The tested host is an Apple M4 Max, model Mac16,5.
The operating system is macOS 26.5.2, build 25F84.
The host has 128 GiB memory.

The crate rust-version field is 1.97.
Checked compiler versions are rustc 1.98.1 and Cargo 1.98.1.
The local `+1.97.1` toolchain name aliases that compiler.
These checks did not run a Rust 1.97 compiler.

## Build

Build the isolated package:

```sh
cargo build --manifest-path crates/ane-miner/Cargo.toml --locked --release
```

## Commands

Print identity JSON without opening the device:

```sh
crates/ane-miner/target/release/quip-ane-msa --capabilities
```

The release binary prints:

```text
{"backend":"ane","algorithm":"msa","supportedKinds":["ISING_SAMPLE"],"maxNodes":16384,"maxEdges":163840,"features":["streaming"],"protocolVersion":1,"streamWidth":1}
```

Protocol 1 capabilities cannot advertise degree or coefficient limits.
A coordinator must honor the Limits section.

Probe the device with one startup worker:

```sh
crates/ane-miner/target/release/quip-ane-msa --check
```

That command exits 0 after one native startup dispatch.
It writes no capabilities JSON.

Connect to a coordinator:

```sh
crates/ane-miner/target/release/quip-ane-msa --quip-coordinator unix:///run/quip/coord.sock
```

`--check`, `--solve`, and `--capabilities` conflict with each other.
Hidden `--ane-worker` starts one child for a single job.
That flag also conflicts with the three core modes and `--quip-coordinator`.

## Solve

`--solve` reads one problem JSON on stdin and writes solutions on stdout.
Use this input:

```json
{"h":[0.0,0.0],"j":[1.0],"edges":[[0,1]],"num_reads":33,"num_sweeps":5,"sweeps_per_beta":2,"beta_range":[0.25,4.0],"seed":123}
```

The release binary returns 33 reads.
Each read has two spins in `{-1, 1}`.
Every reported `energy_milli` is `-1000`.
That value matches `quip_solver_core::quip_protocol::scoring::energy_milli`.
A second run with the same input returns the same bytes.

Empty graphs and zero sweeps are valid.
An empty graph with 4 reads returns 4 empty spin arrays and energy 0.
Zero sweeps return the seeded initial states and do not compile a program.

Unsupported inputs return a non-zero exit and no JSON array:

| Input | Exit | Diagnostic |
| --- | ---: | --- |
| degree 21 | 70 | `variable degree 21 exceeds 20` |
| 129 reads | 64 | `num_reads 129 exceeds this backend's maximum of 128` |
| 65,537 sweeps | 70 | `sweep count exceeds 65536` |
| coefficient `0.5` | 70 | `coefficient must be -1, 0, or 1` |

Those cases do not return a truncated result.

## Coordinator

The miner accepts protocol 1 jobs whose coefficients are in `{-1, 0, 1}`.
Unit coefficients are the only supported domain.
The stock `quip-solver-conformance` 0.0.1 driver sends fractional coefficients.
That stock driver cannot pass for this miner.

The crate protocol suite uses a local test-only coordinator.
It still scores the session with public `DriverReport::is_conformant` checks.

### Mining sweep budget

Mining sessions use 128 reads and 2,048 to 8,192 sweeps.
The target energy determines the sweep count within those bounds.
Failed attempts do not increase the count.
A zero-field job with 4,577 nodes, 41,514 edges, and target `-14612` selects 8,049 sweeps.
The hard cap remains 65,536 sweeps.

An explicit job count takes precedence over the target count.
An explicit target count takes precedence over the adaptive count.
The miner `num_sweeps` setting is a fallback when no target supplies an adaptive count.
Changing `[metal].num_sweeps` alone does not override a normal mining target.
Use `--solve` with an explicit `num_sweeps` to compare budgets on the same problem.

More sweeps give each read a longer search.
Some jobs still miss their energy target.

### Protocol tests

Run ordinary protocol tests in debug:

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --test protocol
```

That command passes 4 tests and ignores 4 hardware tests.

Run hardware protocol tests with the release binary:

```sh
cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --release --test protocol -- --ignored --test-threads=1 --nocapture
```

The hardware command must include `--release`.
The phase budget stays ten seconds.
That budget is too short for debug `--solve`.
A direct 512-sweep diagnostic took 12.211 s in debug and 0.837 s in release.
Those times are whole-process wall times for `--solve`.
Threshold generation is the dominant debug cost.
The diagnostic did not isolate that operation.
They are not a Metal comparison.

## Limits

Advertised limits are:

- 16,384 variables
- 163,840 undirected edges
- at most 20 nonzero neighbors per variable
- coefficients in `{-1, 0, 1}`
- logical reads 1 through 128
- 65,536 sweeps

A complete graph on 21 variables has degree 20.
The miner accepts that graph on the same degree-20 path.
The miner rejects degree 21.

Coloring sorts by descending degree, then ascending index.
The solver walks colors in order.
Each color is an independent set.
A color splits into tiles of at most 4,096 outputs.
Channel counts pad to the next 32-channel boundary.

Requested reads may be 1 through 128.
Execution always uses 128 physical lanes.
At each node and sweep, each group of 32 replicas shares one threshold value.
The four groups use independent streams.
Nodes do not share one global threshold.
A new rung draws new cuts for each group.

The parent keeps the coordinator connection.
It starts one child process per job.
The parent scores returned spins with the consensus scorer.

The worker shares one neighbor input surface across its color programs.
It uploads the full initial spin state once.
Later updates upload only the rows changed by the prior tile.
The worker reuses tile buffers throughout the anneal.

## Memory

Each program carries a dense FP16 weight matrix.
One payload may use at most 128 MiB.
The largest four-tile shape is four matrices of 16,384 by 4,096.
That raw dense FP16 payload is 512 MiB.
Padded worst-case payload stays below 544 MiB.

Those caps bound dense weight bytes.
They are not a total ANE memory budget.
Measured peak resident set size belongs to the test process.
It excludes separate runtime and driver allocations.
IOSurface buffers add one shared neighbor surface and per-program spin, threshold, and output surfaces.
Those surfaces align to 65,536 bytes.

## Follow-up work

Engine selection is bead `quip-miner-metal-7zk`.
Optimization is bead `quip-miner-metal-djn`.
Both are separate work and remain open.

See `docs/validation.md` for host, hardware, capacity, lifetime, and protocol receipts.
