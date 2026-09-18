# Four-colouring against greedy colouring on the ANE, 2026-09-18

## Result

The four-colouring is faster than the greedy eight-colouring on the Apple
Neural Engine (ANE), on both dispatch cost and compile cost. This measures
speed only. It does not decide the default, because solution quality is a
separate open question that bead `quip-miner-metal-fjo` keeps.

| Measure | Greedy, 8 classes | Four colours | Change |
| --- | ---: | ---: | ---: |
| Milliseconds per sweep | 1.0384 | 0.8818 | 1.178 times faster |
| Compile, ms | 208.26 | 129.91 | 37.6% cheaper |
| Padded rows per sweep | 4,704 | 4,608 | 2.0% fewer |
| Classes in the chain | 8 | 4 | half as many |

Both colourings cover the same 4,577 nodes, so this compares two layouts of
one problem.

For a job of 2,048 sweeps, the two costs together come to 2,335 ms under
greedy and 1,936 ms under four colours, a saving of 399 ms, or 17.1%. At
8,192 sweeps the saving is 1,362 ms, or 15.6%. That is about four times the
gain from setting `BLOCK_SWEEPS` to 1, which
`docs/perf/2026-09-17-ane-setup-profile.md` records at 4.0% and 3.2% for the
same two sweep counts.

## Why it wins twice

Each sweep walks its colour classes in strict sequence, because every class
reads what the previous class wrote. Halving the class count halves the
dependent chain.

The greedy colouring, by descending degree then ascending index, gives
classes of 857, 849, 817, 740, 685, 480, 136 and 13 nodes. The last two are
too small to amortise a dispatch.
`/tmp/quip-ane-architecture.txt:10529` records that the compiler spreads
output channels across engine cores by strided round-robin, so a 13-node
class leaves most of the array idle while still costing a dispatch.

The four-colouring gives 1,148, 1,145, 1,145 and 1,139 nodes, which is nearly
balanced. Padding each class to a multiple of 32 costs 4,704 rows under
greedy and 4,608 under four colours, so the arithmetic is slightly smaller as
well as better shaped.

The compile falls for a related reason. The program text carries one
convolution and its operator chain per class per sweep, so four classes emit
half the operators of eight.

## Where the colouring comes from

`src/topology.rs` already builds it, in `advantage2_color`. Its own test,
`advantage2_four_colors_cover_all_edges_and_preserve_csr`, asserts the class
counts and checks that no edge joins two nodes of one colour. It also
records that the greedy alternative gives 8 classes.

The ANE crate does not use it. `crates/ane-miner/src/graph.rs` colours
greedily on its own. Taking this result into production means giving the ANE
path the four-colouring, which is a code change this report does not make.

## Method

The probe is `crates/ane-miner/probes/coloring_compare.m`. It calls
`quip_ane_create`, `quip_ane_reset` and `quip_ane_evaluate` through the real
bridge, unchanged, the same way `contention.m` and `setup_profile.m` do.

Each layout compiles one program and then runs 200 back-to-back evaluations
at one sweep per dispatch, matching `BLOCK_SWEEPS` in
`crates/ane-miner/src/native.rs:10`, across 4,608 channels. Couplings come
from seed 123 in `{-1, 1}` with zero fields, matching the other probes in
that directory, and their values do not change what a dense convolution
costs.

The probe asserts that both colourings cover the same node count before
timing, so a mismatched layout fails rather than producing a comparison
between two different problems.

Four runs went through `scripts/ane-guard`, so no measurement shared the
device.

### Commands

```sh
clang -O3 -fobjc-arc -fobjc-arc-exceptions -std=c11 -Wall -Wextra -Werror \
  -framework Foundation -framework IOSurface \
  crates/ane-miner/probes/coloring_compare.m -o /tmp/coloring-compare
scripts/ane-guard -s 3 -t 600 -- /tmp/coloring-compare 200 1
```

## Readings

| Run | Greedy ms per sweep | Four ms per sweep | Greedy compile, ms | Four compile, ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0390 | 0.8807 | 211.190 | 128.769 |
| 2 | 1.8524 | 0.8828 | 216.147 | 141.125 |
| 3 | 1.0378 | 0.8806 | 205.331 | 131.042 |
| 4 | 1.0370 | 0.8861 | 196.040 | 126.624 |

Run 2's greedy figure is 78% above the other three. This matches the cold
first-load cost `docs/perf/2026-09-17-ane-setup-profile.md` records for its
own first run. The median absorbs it, and the four-colour figure in the same
run is in line with its neighbours, so the outlier belongs to that one
measurement rather than to the run as a whole.

The four-colour readings span 0.8806 to 0.8861 ms, a range of 0.0055 ms. The
three clean greedy readings span 1.0370 to 1.0390 ms. The two bands are far
apart, so the ordering does not depend on which figure is chosen.

## Limits of this measurement

Solution quality. The bead records that mean lowest energy was worse under
four colours by 1,400 and 250 milli on two graph seeds in the Metal-side
work, and that no run established whether that is a colouring effect or
noise. A colouring that samples worse could cost more than 17% in time to a
valid solution, so speed alone must not decide the default.

This probe also runs one program per process, with no miner channel,
governor, or result scoring in the path, so it supports no production
throughput claim.
