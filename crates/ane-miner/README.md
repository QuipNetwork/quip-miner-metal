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

Run hardware protocol tests with the release binary, through the device
guard described below:

```sh
scripts/ane-guard -- cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --release --test protocol -- --ignored --test-threads=1 --nocapture
```

The hardware command must include `--release`.
The phase budget stays ten seconds.
That budget is too short for debug `--solve`.
A direct 512-sweep diagnostic took 12.211 s in debug and 0.837 s in release.
Those times are whole-process wall times for `--solve`.
Threshold generation is the dominant debug cost.
The diagnostic did not isolate that operation.
They are not a Metal comparison.

### Device guard

The host has one Apple Neural Engine. Concurrent users of it corrupt each
other's measurements, whether that is two probes or a probe and a hardware
test. Concurrency measurements are the worst case, because contention there
is indistinguishable from the result.

Run every command that touches the device through `scripts/ane-guard`:

```sh
scripts/ane-guard -- ./setup-profile
scripts/ane-guard -t 600 -- cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --release --test protocol -- --ignored --test-threads=1
```

The guard takes a lock and runs the command under it. After the command
finishes, the guard keeps the lock through a settling interval and only then
releases it. That ordering is deliberate. A release before the device is
quiet lets the next holder start against a busy engine.

Options are `-t SECONDS` for how long to wait, default 300, and `-s SECONDS`
for the settling interval, default 3. Set `QUIP_ANE_LOCK` to move the lock
directory, which defaults to `/tmp/quip-ane-device.lock`.

The guard exits with the command's own status, 64 for a bad invocation, and
75 when the wait times out. A timeout fails on purpose. A contended run
produces a number nobody can trust, so it must stop rather than publish one.

The guard breaks a lock whose owning process is gone, so a killed probe does
not block later runs.

## Limits

Advertised limits are:

- 16,384 variables
- 163,840 undirected edges
- at most 20 nonzero neighbors per variable
- coefficients in `{-1, 0, 1}`
- logical reads 1 through 128
- 65,536 sweeps

At production shape, a second concurrent ANE worker raises measured throughput by about 17 percent over one worker, and four workers reach about 25 percent, because measurement never showed more than two programs dispatching at once.
Compile scales the same way and does not cap that gain. Four processes compile at once, rather than queueing, but each compile slows enough that four reach only 1.230 times the compile throughput of one. `docs/perf/2026-09-17-ane-compile-contention.md` records that result, which holds for real mining, where every job compiles.

A complete graph on 21 variables has degree 20.
The miner accepts that graph on the same degree-20 path.
The miner rejects degree 21.

The Advantage2 topology uses its own four-colouring, with classes of 1,148,
1,145, 1,145 and 1,139 nodes.
Every other graph colours greedily, by descending degree then ascending index.
The solver walks colors in order.
Each color is an independent set.
A color splits into tiles of at most 4,096 outputs.
The static ANE graph applies all colors in order for one sweep per dispatch.
State rows follow color order, with padding only after the last node.
Each convolution output pads to the next 32-channel boundary.
Static slices remove that padding before spin updates.

Requested reads may be 1 through 128.
Execution always uses 128 physical lanes.

Measurement keeps that count. A sweep costs 0.425 ms plus 1.75 microseconds
per read, and the fixed part is the coupling stream, so 32 reads saves a
quarter of the time for three quarters fewer samples.
`docs/perf/2026-09-18-ane-read-count.md` records the sweep. Against the
testnet's own blocks, 32 reads produces fewer valid proofs per second on
the ANE than 128 does. `docs/perf/2026-09-18-testnet-reads-study.md`
records that result.

At each node and sweep, each group of 32 replicas shares one threshold value.
The four groups use independent streams.
Nodes do not share one global threshold.
A new rung draws new cuts for each group.

The parent keeps the coordinator connection.
It starts one child process per job.
The parent scores returned spins with the consensus scorer.

The worker uploads the full initial spin state once.
Two retained IOSurfaces alternate as the input and output state between fused dispatches.
The host stages deterministic thresholds for each sweep and reads spins once after the final block.
A final unused sweep receives a threshold that disables every flip.
Node reordering preserves the original node and replica assignments for every random value.

## Memory

One program carries all dense FP16 tile matrices in one weight file.
Each matrix is a separate aligned chunk, reused across dispatches.
Each tile matrix may use at most 128 MiB.
The largest four-tile shape is four matrices of 16,384 by 4,096.
That raw dense FP16 payload is 512 MiB.
Padded worst-case payload stays below 544 MiB.

Those caps bound dense weight bytes.
They are not a total ANE memory budget.
Measured peak resident set size belongs to the test process.
It excludes separate runtime and driver allocations.
The program uses two state surfaces and two threshold surfaces, each with the full padded node count.
Those surfaces align to 65,536 bytes.

## Follow-up work

Engine selection closed as bead `quip-miner-metal-7zk`. The combined router in
`src/combined.rs` sends separate jobs to Metal and to the ANE under one miner
identity.
Optimization closed as bead `quip-miner-metal-djn`. The dense path stays in
production. Fused local routing measured 3.4 to 3.6 times slower than matched
dense work. `docs/perf/2026-09-16-ane-local-routing.md` records that result.

Compile serialization under concurrency closed as bead
`quip-miner-metal-6jd`. Compile does not serialize. Four processes compile at
once, and compile throughput scales 1.145 times at two processes and 1.230
times at four, matching the curve measured for `evaluateWithQoS:`.
`docs/perf/2026-09-17-ane-compile-contention.md` records the measurement.

Runtime-input couplings closed as bead `quip-miner-metal-yba`, rejected on
measurement. Moving the couplings and the per-job threshold `h` out of the
compiled weight blob and Model Intermediate Language (MIL) text into runtime
inputs works and gives the right answers, but it costs more than the compile
it removes. The best variant runs a sweep in 1.752 ms against production's
1.13 ms, so the trade turns negative after about 254 sweeps, and production
jobs run 2,048 to 8,192. That variant also cannot fuse sweeps: two sweeps in
one program cost about 700 ms rather than twice 1.752 ms. Every matmul
variant measured slower than the convolution variant.
`docs/perf/2026-09-18-ane-runtime-couplings.md` records the measurement.

That result closes the runtime-input route only. It does not close the
per-job compile. Compiling one program per topology and writing each job's
couplings into the compiled program is a separate route, open as bead
`quip-miner-metal-kyk`. The topology is fixed, so only the values change per
job. `_ANERequest` already takes a `weightsBuffer` argument, typed
`_ANEIOSurfaceObject`, that this crate passes as `nil`.

The compile is not the limit on running more workers, per
`docs/perf/2026-09-17-ane-compile-contention.md`.

Four-coloring chunk balance is open as bead `quip-miner-metal-fjo`. This
task appends the ANE-side argument, that the smallest color classes are too
small to amortize a dispatch, to that bead's existing Metal-side numbers.

A shared device guard closed as bead `quip-miner-metal-c7l`. Every task in
the throughput plan substituted its own serialization for device access,
because no guard existed. `scripts/ane-guard` now provides one. See Device
guard above.

The wall-clock assertion in `crates/ane-miner/src/process.rs:482` closed as
bead `quip-miner-metal-erz`. The test now bounds cancellation against the
child process's own sleep, because the property under test is that
cancellation returns without waiting for the child.

See `docs/validation.md` for host, hardware, capacity, lifetime, and protocol receipts.
