# Neural engine threshold overlap and lane width

Now: median whole-process time falls from 8.406 to 5.706 seconds at 64 reads.
It falls from 14.229 to 8.222 seconds at 128 reads. Every seeded output exactly
matches the serial code. The runs use one past problem. They do not measure
proof rates on other problems.

## Controlled comparison

The host is an Apple M4 Max with 128 GiB. It runs macOS 26.5.2, build 25F84.
The user paused the production miner. No builds ran during these measurements.
No other device tests ran. Each executable uses sparse coupling encoding and
four colors. Each dispatch has one sweep.

The input is historical qblock 3250. It has 4,577 nodes and 41,514 edges.
It has zero fields and signed unit couplings. Each job uses 16,384 sweeps and
sampler seed 123.
The compact nonce fixture is `tests/fixtures/testnet-qblocks-3191-3250.json`.
The study fixture test checks every coefficient against the archived problem.

Three rounds alternate the order of serial and overlap runs at both widths.
Wall time includes all work from process startup through process exit. This
work includes graph preparation, compilation, annealing, readback, and scoring.
The serial executable already has lanes sized to the read count. The comparison
isolates threshold overlap at each width.

| Reads | Serial median wall, s | Overlap median wall, s | Serial / overlap |
| ---: | ---: | ---: | ---: |
| 64 | 8.406 | 5.706 | 1.47 |
| 128 | 14.229 | 8.222 | 1.73 |

At 64 reads, median anneal time falls from 7.255 to 4.448 seconds.
At 128 reads, it falls from 13.177 to 7.201 seconds.
Median dispatch totals remain close. They are 3.069 versus 3.047 seconds at
64 reads. They are 6.237 versus 6.017 seconds at 128 reads.
Staging totals rise from 2.000 to 2.186 seconds at 64 reads. They rise from
3.996 to 4.274 seconds at 128 reads.
Overlap reduces wall time even though staging itself takes more time.
Staging and dispatch durations overlap, so their sum is not wall time.

The [raw receipts](data/2026-09-21-ane-overlap/controlled.jsonl) include every
round and stage time. They include energies and executable hashes. They also
include output hashes.
Every run matches the full seeded sample array for its width.
The 64-read array also matches the first 64 samples of the 128-read array.

## How it works

Each Apple Neural Engine (ANE) program has two spin surfaces. It also has two
threshold banks. The host makes the next threshold block while the current
dispatch runs. Submission converts that block into the inactive bank. It then
joins the prior dispatch before selecting the next state-dependent request.
The native queue keeps its program and request until the run ends.
It does not keep the Rust threshold pointer.

Read, reset, finish, and close join pending evaluation. Each operation passes
through the evaluation error. Arm NEON converts threshold bytes to fp16 values.
It uses a scalar tail.
The skip sentinel remains exactly minus 128 in fp16.

Physical lanes follow the requested read count. The count rounds up to groups
of 32, with a limit of 128. The private runtime requires this alignment.
Exact widths 1 and 16 produced invalid padding in the device checks.
A 33-read job uses 64 lanes.
The returned count remains 33. Random numbers use the original 128-lane
streams, so a narrower job keeps its read prefix.

## Stage probe

Five release-mode runs use the
`hardware_rust_setup_stage_medians_advantage2_system1` test. Each run uses 128
reads and 512 sweeps on the 41,515-edge fixture. Median staging time is 132,270
microseconds. Median dispatch time is 197,000 microseconds. Median anneal wall
time is 227,927 microseconds. The earlier serial receipt recorded 122,870,
203,555, and 387,838 microseconds. That run is a historical reference, not a
second control from this session.

The [five probe receipts](data/2026-09-21-ane-overlap/stage-medians.jsonl)
keep each stage result. Run the test with `--release` and `--ignored`. Run one
process at a time through `scripts/ane-guard`. Debug builds spend much more
time in host loops. They are not comparable to these release measurements.

## Correctness

The [hardware log](data/2026-09-21-ane-overlap/hardware.log) records the checks:

- All 32,384 integer acceptance cases match the scalar calculation.
- Reads 1, 16, 31, 32, 33, 64, 65, 96, 127, and 128 match the scalar oracle.
- Partial blocks, beta boundaries, and block widths 1, 2, 4, and 8 pass.
- Asynchronous read, reset, and close join pending work correctly.
- Cancellation and output-channel closure reap workers and remove staging files.
- Sixty-four sequential jobs use distinct workers and match 8,192 oracle reads.

The native boundary test caught a null timing-pointer regression in the
synchronous entry point. The entry point now rejects it before submission.
The hardware rerun passes. Queued runtime failures and partial staging failures
have code review coverage but no injected-failure test.
