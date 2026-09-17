# Local sweep evidence, 2026-09-16

## Decision

The local sweep gives exact results and reuses inputs in memory. It fails the speed gate.
The dense control is 3.4 to 3.6 times faster in the matched one-sweep control.
The 4,096- and 16,384-sweep runs are **NOT RUN** because Task 4 requires a full-sweep gain first.
The candidate stays in the probe directory. This result changes no production code. It adds no driver for long runs.

## Code

The code keeps 128 reads in the 25 by 25 by 8 grid throughout each chunk.
It keeps the saved eight colors and their sizes: 857, 849, 817, 740, 685, 480, 136, and 13.
Each color reads the updated state from the prior color.
The update accepts a flip when the signed field plus the threshold is at least zero.
Four threshold groups each control 32 reads. Static color masks guard inactive nodes and padding.

The Apple Neural Engine (ANE) program uses Model Intermediate Language (MIL). The graph uses fixed spatial slices, short zero padding, and listed edge masks.
It masks and reshapes runtime J before the sweep loop. It also sets up runtime h before that loop.
An all-zero h mode removes h from the inputs and graph.
The graph hash covers topology, layout, chunk shape, and h mode. J and threshold values remain runtime data.
The saved zero-J edge (880, 2695) stays in the topology.
Task 3 proved that activating this edge after compile changes the exact fields.
The four-color experiment remains separate.

The runner uploads state, J, h, and all threshold chunks before dispatch.
It builds every request before dispatch and swaps two state surfaces across chunks.
Host write counters cover every input memcpy in the runner.
For each real two-chunk job, the before and after dispatch logs record five writes and 3,420,160 bytes.
The chunk loop adds zero host input writes. Each run compiles once and reuses its model across both J jobs.
The one-sweep and two-sweep manifests have the same MIL and constant data.

## Correctness

The edge oracle uses integer math to sum the listed edges.
A second CPU test uses the local shifts and masks.
Both match the saved two-sweep outputs for both saved jobs in the saved node order.
Small tests cover sequential color updates, threshold ties, all four groups, grid boundaries, missing nodes, and zero-J edges.
The tail chunk test uses a `-128` threshold to skip the unused tail sweep.

Seven guarded runs finished with exit code zero. No compile timed out.
They include 58 local dispatches and 24 dense dispatches.
Every logged first and final output check has zero mismatches.
The real tests check all 643,072 local output elements, with 423 grid padding slots plus 24 alignment slots per read.
The tests cover one full sweep and two successive full-sweep chunks for both saved jobs.
Small compiled controls also cover changed runtime J/h, an unused tail sweep, and a zero-h graph.

## Matched full-sweep timings

All times below are milliseconds per full eight-color sweep.
Each one-sweep job uses the same request for ten calls.
The warm times use the last nine calls. The first call has its own column.
Both controls use the same states, J, h, thresholds, color order, and 128 reads.
Layout and padding differ by design.

| Job | Local first | Local warm median | Local warm range | Dense first | Dense warm median | Dense warm range | Local / dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 9.367 | 9.107 | 8.467–10.068 | 3.014 | 2.500 | 2.459–2.585 | 3.64 |
| 1 | 9.662 | 8.836 | 8.524–8.913 | 2.844 | 2.596 | 2.447–2.950 | 3.40 |

The ranges do not overlap. The speed loss is much larger than the spread in these runs.
These short controls do not measure full-solve quality, energy, valid reads, or time per valid solution.
No 4,096- or 16,384-sweep timing or quality claim follows from them.

The real two-chunk local evaluations total 17.192 and 16.990 milliseconds, with loop logs.
Matched dense totals are 6.465 and 5.373 milliseconds.
Each two-chunk process checks both final outputs against the saved two-sweep fixtures.

## Setup and memory

| One-sweep stage | Local | Dense |
| --- | ---: | ---: |
| Compile | 17,182.453 ms | 245.017 ms |
| Load | 19.800 ms | 15.896 ms |
| Request and surface setup | 3.191 ms | 2.600 ms |
| Allocated surface bytes | 4,521,984 | 47,382,528 |
| Job 0 upload | 0.753 ms | 6.561 ms |
| Job 1 upload | 0.637 ms | 4.251 ms |
| Job 0 readback | 0.145 ms | 0.113 ms |
| Job 1 readback | 0.128 ms | 0.098 ms |

Readback includes lock, copy, and unlock. Compile, setup, upload, and checks are outside the per-call timings.
The local MIL has 1,872,419 bytes. Its constant blob has 1,906,368 bytes.
Across eight colors, it emits 441 distinct shifted expressions and 1,235 product expressions.
Their logical output sizes are 70,560,000 and 197,600,000 bytes.
These totals describe graph size, not peak memory. Live values, ANE cache use, and compiler fusion remain unmeasured.
Source reuse does not prove ANE cache residency.
The local graph needs less space for inputs. It uploads less data but takes more time per sweep.

## Checks and platform

The test suite passes 25 tests, with both saved fixtures.
Ruff formatting and lint checks pass. The ty check passes with the installed NumPy search path.
The native runner builds with Clang warnings as errors.
The device is an M4 Max, Mac16,5, with 128 GiB of memory. It runs macOS 26.5.2 build 25F84.
Python is 3.13.15 with NumPy 2.5.3. The platform receipt records exact compiler and platform strings.
Every run uses the shared device guard, a 30-second timeout, and a three-second settling interval.
These timings use the private ANE evaluation path.
They do not add direct device counters or evidence of where the compiler placed each step.
The stage report notes the limits of the device proof.

## Sources and logs

[Task 1 stage profile](/tmp/quip-beads-execution-0916/routing-profile-report.md) records the saved global permute cost.
[Topology audit](/tmp/quip-beads-execution-0916/topology-audit.md) shows the exact graph and source node IDs.
[Task 2 mapping report](/tmp/quip-beads-execution-0916/topology-implementation-report.md) shows edge and color checks.
[Task 3 field report](/tmp/quip-beads-execution-0916/local-field-report.md) records exact fields on the device and changed J.
Those earlier artifacts remain unchanged.

[Full sweep receipts](/tmp/quip-beads-execution-0916/local-sweep/results.json) have all parsed stages and timings.
Each run folder has its raw log, manifest, expected output, actual output, and receipt.
Local run directories also have MIL, constants, runtime inputs, and CPU proofs.
[Artifact hashes](/tmp/quip-beads-execution-0916/local-sweep/artifact-hashes.json) cover those files and source snapshots.
[Platform](/tmp/quip-beads-execution-0916/local-sweep/environment.json) records the test platform.
[Topology sources](/tmp/quip-beads-execution-0916/topology-artifacts/sources.json) record source paths and hashes.
The mapping remains in [layout.json](/tmp/quip-beads-execution-0916/topology-artifacts/layout.json).
Its SHA-256 is `f4b30c9bb7dc6abb5d55a11e46fda5f61425d81e71adf25f5a80a74d152c148c`.

Saved inputs are in `single-call-real-s2-v0` and `single-call-real-s2-v1` under the saved guarded scratch directory.
Dense manifests reference those source arrays. Dense one-sweep fixtures preserve their inputs and select the first threshold sweep.
The separate edge oracle supplies their one-sweep expected states.
Task 4 edits only `local_routing.py`, `test_local_routing.py`, and the probe `profile_runner.m`.
Task 5 records this gated rejection. The parent owns issue closure and integration.
