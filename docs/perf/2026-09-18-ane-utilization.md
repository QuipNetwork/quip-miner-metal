# ANE roofline, weight encoding, and power, 2026-09-18

## Result

One fp16 model fills the engine. The production sweep at 128 reads reaches
60% of the engine's measured MAC ceiling and 74% of its measured
weight-stream ceiling at once, so a second model can take only what is
left, and it takes most of it: two processes reach 1.23 times the
sweeps per second of one. The engine draws 0.9 W during that sweep against
6.5 W for a dense kernel at the same MAC rate, because 99.6% of the
coupling weights are zero and the MAC array spends its time multiplying by
them.

The zeros are the lever. Storing the couplings as a one-bit mask plus the
nonzero values, which the program expands with `constexpr_sparse_to_dense`,
shrinks the weight stream from 42.5 MB to 2.8 MB per sweep, and the engine
consumes that form directly. The output is bit-identical to the fp16
program on every layout and read count tested. The sweep at 128 reads falls
from 0.66 ms to 0.46 ms on dispatch alone, and a 16,384-sweep job on the
production path falls from 16.6 s to 13.7 s. The cost per read above 128
reads, which was five times the cost below it, flattens: 1,024 reads in one
model cost 2.7 microseconds per read-sweep against 5.0 for the fp16 program
at 128 reads.

The production bridge now writes the sparse encoding. Every hardware test
passes on it, including the 32,384-case exact integer acceptance and the
64-job oracle match.

The engine runs one program at a time whatever the encoding. Two sparse
models at 128 reads reach 1.22 times one, four reach 1.30, two at 1,024
reads reach 1.04, and two at 64 reads reach 0.80, slower than one. The way
to pack more work into the engine is more reads per model, up to 1,024,
and never more models.

## The ceilings

Two bare 1×1 convolutions, fp16 weights, timed until 3 s elapsed, three
rounds, fastest clean round.

| Kernel | Weights | Positions | ms per call | TMAC/s | GB/s of weights |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048×2048, weights fit on chip | 8.4 MB | 8,192 | 4.843 | 7.09 | 1.7 |
| 2048×2048 | 8.4 MB | 4,096 | 2.559 | 6.71 | 3.3 |
| 1024×1024 | 2.1 MB | 4,096 | 0.727 | 5.91 | 2.9 |
| 4096×4096, weights stream | 33.6 MB | 1,024 | 4.704 | 3.65 | 7.1 |
| 4608×4608, weights stream | 42.5 MB | 128 | 0.498 | 5.46 | 85.3 |
| 4608×4608, weights stream | 42.5 MB | 32 | 0.469 | 1.45 | 90.6 |

The MAC ceiling on this host is 7.09 TMAC/s, or 14.2 TFLOP/s at fp16. The
weight-stream ceiling is 90.6 GB/s. Both are floors on the true hardware
limits, because each kernel also reads its input and writes its output,
but they are the ceilings this program can reach. The ridge, where the two
meet, sits at 78 MAC per weight byte.

## The production sweep against the ceilings

The sweep multiplies the padded 4,608×4,608 coupling matrix by the state
once per sweep, 2.72 GMAC at 128 reads. At fp16 the weights are 42.5 MB,
so the sweep needs 64 MAC per weight byte, just left of the ridge.

| Encoding | Reads | ms per sweep | TMAC/s | MAC ceiling | GB/s | Stream ceiling | ANE W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16 | 32 | 0.483 | 1.41 | 20% | 88.0 | 97% | 0.79 |
| fp16 | 64 | 0.533 | 2.55 | 36% | 79.7 | 88% | 0.65 |
| fp16 | 128 | 0.638 | 4.26 | 60% | 66.6 | 74% | 0.92 |
| fp16 | 256 | 2.567 | 2.12 | 30% | 16.5 | 18% | 0.96 |
| fp16 | 512 | 4.699 | 2.31 | 33% | 9.0 | 10% | 1.05 |
| fp16 | 1,024 | 9.057 | 2.40 | 34% | 4.7 | 5% | 1.07 |
| sparse | 32 | 0.180 | 3.78 | 53% | 15.7 | 17% | 1.00 |
| sparse | 64 | 0.190 | 7.15 | 101% | 14.8 | 16% | 1.10 |
| sparse | 128 | 0.418 | 6.50 | 92% | 6.7 | 7% | 1.00 |
| sparse | 256 | 0.839 | 6.48 | 91% | 3.4 | 4% | 1.08 |
| sparse | 512 | 1.397 | 7.78 | 110% | 2.0 | 2% | 1.17 |
| sparse | 1,024 | 2.805 | 7.75 | 109% | 1.0 | 1% | 1.21 |

TMAC/s counts the dense multiply, so the sparse rows above 100% show the
engine skipping zero weights rather than beating its own ceiling. Each row
is the fastest clean window over the rounds that ran it. The sparse 64 and
128 rows come from the concurrency grid's single-process launches, two
each, because those were the cleanest runs of that point. GB/s is the
weight bytes the program declares over the sweep time.

Below 128 reads the fp16 sweep sits on the stream ceiling, with the weight
path at 97% for 32 reads. At 128 reads the sweep touches both ceilings.
Past 128 reads the compiler splits the activation into 128-position tiles
and re-streams the weights for each tile, which takes the sweep off both
ceilings, and a two-dimensional layout does not avoid the split, since
2×128 costs the same as 1×256 and 4×128 the same as 1×512, to within 1%.

The sparse sweep sits on neither ceiling. What remains at 128 reads is
about 0.15 ms of fixed cost, which is the sixty operations in the program
rather than the weights, and 2 to 3 microseconds per read for the
activation work.

### Run-to-run spread

Fresh compiles of the same program do not run at the same speed. Six
compiles of the sparse 128-read program in one quiet run gave 0.425, 0.433,
0.447, 0.469, 0.473 and 0.515 ms. Six of the fp16 program gave 0.639,
0.643, 0.661, 0.684, 0.722 and 0.782. Across every clean measurement of the
day the sparse point ran between 0.42 and 0.64 ms and the fp16 point
between 0.64 and 0.78. The medians are 0.46 and 0.66 ms, and the 0.418 in
the table is the fastest clean window. The output hash is the same on every
run, so the spread is in the compiler's schedule or the memory placement,
not the arithmetic.

## Models in flight

The concurrency grid launches one, two and four copies of the probe at
once, each timed until 3 s elapsed, and charges the total sweeps to the
union of their loop windows. Two rounds, medians.

| Model | Sweeps per second, one | Two processes | Four processes | ms per sweep each, two |
| --- | ---: | ---: | ---: | ---: |
| fp16, 128 reads | 1,536 | 1.23× | 1.23× | 1.02 |
| sparse, 64 reads | 5,271 | 0.80× | 0.72× | 0.37 |
| sparse, 128 reads | 2,398 | 1.22× | 1.30× | 0.53 |
| sparse, 1,024 reads | 344 | 1.04× | 1.05× | 3.62 |

Two sparse 64-read models each run at 0.37 ms per sweep, twice their solo
0.19 ms, and their combined rate is below one model's. Two 128-read models
each run at 0.53 ms against 0.42, and two 1,024-read models at 3.62 against
2.91. The engine executes one program at a time and overlaps only the
request pipeline around it, and the shorter the program, the more the
second process costs. The 1.23 at fp16 matches the 1.22 measured earlier on
the production reads probe.

The packing dimension is reads per model. Lane-sweeps per second, which is
sweeps per second times reads:

| Model | Lane-sweeps per second | Against fp16 at 128 reads |
| --- | ---: | ---: |
| fp16, 128 reads | 197,000 | 1.00 |
| sparse, 64 reads | 337,000 | 1.71 |
| sparse, 128 reads | 307,000 | 1.56 |
| sparse, 1,024 reads | 352,000 | 1.79 |

From 64 to 1,024 reads the sparse program delivers about the same samples
per second, so the read count is free to follow the mining metric. The
testnet study found the chance of a valid proof per job at 0.40 for 64
reads and 0.54 for 128, and the jobs per second at 64 reads would be 2.2
times those at 128, so 64 reads would give about 1.6 times the valid proofs
per second. The production program pads every job to 128 lanes today, so
that gain needs the lane width to follow the job. Bead
`quip-miner-metal-fjo.8` carries it.

## Power

`macmon` sampled the energy counters at 250 ms through every run. The engine
idles at 0 W.

| Configuration | ANE W | mJ per sweep | J per 16,384-sweep job, dispatch only |
| --- | ---: | ---: | ---: |
| Dense 2048×2048 kernel, MAC-bound | 6.49 | 31.4 per call | |
| Dense 4608×4608 stream kernel, 128 positions | 4.99 | 2.48 per call | |
| fp16 sweep, 128 reads | 0.92 | 0.59 | 9.6 |
| sparse sweep, 128 reads | 1.00 | 0.42 | 6.8 |
| sparse sweep, 1,024 reads | 1.21 | 3.40 | 55.6 |

A model in flight draws about 1 W on the engine whatever its encoding,
because a sweep over a matrix that is 99.6% zeros switches few multiplier
bits. A second process raises the draw to 1.14 to 1.20 times one for 1.22
to 1.30 times the throughput. The engine costs about 1 Wh per hour per
model, models run one at a time, and the encoding decides how many sweeps
each joule buys: 1,700 fp16 sweeps per joule against 2,400 sparse at 128
reads.

On the production path, where the host generates and stages thresholds
between dispatches, the engine is busy for about 11 s of a 16.6 s fp16 job
and 6.6 s of a 13.7 s sparse job, so the engine-side energy per job is
about 10 J for fp16 and 6 J for sparse. CPU energy for the host side is
not counted. DRAM power rose 1.1 to 1.9 W above its 2.1 W idle during fp16
sweeps and stayed at idle during sparse sweeps.

## What changed in production

`crates/ane-miner/native/ane_bridge.m` now writes each tile's couplings as
a uint1 mask over the padded matrix and an fp16 vector of the nonzero
values in mask order, and `makeMIL` emits the two constants and a
`constexpr_sparse_to_dense` in place of the fp16 constant. A tile with no
nonzero weight carries one explicit 0.0 under its first mask bit so that no
tensor is empty. The Rust side hands the bridge the same int8 weights as
before.

Compile grows from 0.10 s to 0.82 s per program at 128 reads, to 1.6 s for
the 2×128, 4×128 and 8×128 layouts, and to 2.7 s for 1×512. At 16,384
sweeps the 0.82 s is 6% of a job. The 27 hardware tests pass, including `hardware_exhaustive_integer_acceptance`
at 32,384 cases with 0 mismatches and `hardware_64_jobs_use_distinct_children_and_match_oracle`.
One test needed a fix that predates this change: `hardware_two_sweeps_exact_oracle`
asserted one dispatch for two sweeps, which stopped being true when
`BLOCK_SWEEPS` went to 1 in 4cfb321. It now derives the count from the
constant.

Whole-process wall time for `--solve` on qblock 3250 at 128 reads and
16,384 sweeps, three runs each. fp16 took 16.46, 16.54 and 16.78 s. Sparse
took 13.59, 13.61 and 13.77 s. The gain is 1.21 on the production path against 1.4 on
dispatch alone, because the host generates thresholds and stages them
serially before each dispatch, about 0.36 ms per sweep against the 0.40 ms
dispatch. Bead `quip-miner-metal-fjo.9` carries the overlap.

## Encodings that failed to compile

`constexpr_lut_to_dense` with two-bit indices over a four-entry palette,
and `constexpr_blockwise_shift_scale` over int8 weights, both fail
`ANECCompile` with `InvalidMILProgram` and no further detail. The sparse
form is the one this compiler accepts, and it is also the smallest of the
three for this matrix.

## Method

The probe is `crates/ane-miner/probes/utilization.m`. It builds the
production sweep text from `ane_bridge.m` with the read layout `[1, C, H, W]`
and the weight encoding lifted out, and bare convolutions for the
ceilings. Couplings are synthetic but shaped like the testnet's: 41,514
edges over 4,577 nodes in the Advantage2 four-colouring, none within a
class, values in {−1, +1}, fields zero. Each timed loop runs until its
budget elapses and hashes the output surface with FNV-1a.

Every layout and encoding at one read count produced the same hash, which
is the correctness check for the encodings and the sign that the layouts
are the same computation. 128 reads hash `9f41c56a68a51b03`. 256 reads hash
`832d241bd2d5be83` for 1×256, 2×128 and 4×64 under both encodings. 512
reads hash `e8bc39b39b696403`.

`crates/ane-miner/probes/utilization.sh` runs the grid ROUNDS times
with `macmon` sampling and dumps the `aned` log.
`utilization_concurrency.sh` launches the concurrency grid.
`utilization.py` and `utilization_concurrency.py` summarise them.

```sh
clang -O3 -fobjc-arc -fobjc-arc-exceptions -std=c11 -Wall -Wextra -Werror \
  -framework Foundation -framework IOSurface \
  crates/ane-miner/probes/utilization.m -o /tmp/utilization
scripts/ane-guard -s 3 -t 1800 -- crates/ane-miner/probes/utilization.sh /tmp/utilization 3000 util 3
crates/ane-miner/probes/utilization.py util util/summary.json
scripts/ane-guard -s 3 -t 1800 -- crates/ane-miner/probes/utilization_concurrency.sh /tmp/utilization 3000 conc 2
crates/ane-miner/probes/utilization_concurrency.py conc conc/summary.json
```

### Contamination

Three signals mark a timed window as contaminated, and a configuration's
figure is its fastest clean window. A program creation by `aned` inside the
window or in the five seconds before it belongs to another client, because
the probe compiles before its loop starts. ANE power above 0.05 W in the
idle gap before the window means a client that started earlier is still
running. GPU power above 1 W inside the window means host work on the GPU,
which shares the memory system.

All three fired during this study. A background service created 27
programs with 60 to 67 MB of weights inside one 8-second span and doubled
the fp16 sweep at 64 and 128 reads in that round. A GPU study ran through
two of the three grid rounds and slowed the dense kernels by up to two
times. Both showed in the power trace before they showed in the log, which
is why the power signals joined the log check.

## Limits

The ceilings are what bare convolutions reach through this runtime, not
the silicon's peak. The dispatch figures exclude the host side of the
production path. The synthetic couplings match the testnet graph's size
and density, not its structure, which does not matter for the weight
stream and did not matter for the hash checks. The run-to-run spread of
about 15% on a fresh compile is larger than most of the differences
between adjacent read counts, so the ladder's shape is reliable and its
individual steps are not.
