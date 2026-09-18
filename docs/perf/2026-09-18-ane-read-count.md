# Read count against sweep cost on the ANE, 2026-09-18

## Result

Keep 128 reads. The count inherited from the CUDA code is the
cheapest per read this engine offers, and it is a clean minimum rather than
a plateau.

Cutting reads saves far less time than it gives up in samples. Dropping from
128 to 32 removes three quarters of the arithmetic and returns only 25% of
the time. Raising reads past 128 costs roughly double per read.

| Reads | Milliseconds per sweep | Microseconds per sweep per read |
| ---: | ---: | ---: |
| 8 | rejected | — |
| 16 | rejected | — |
| 32 | 0.4882 | 15.31 |
| 64 | 0.5107 | 8.00 |
| 96 | 0.6151 | 6.41 |
| **128** | **0.6439** | **5.09** |
| 160 | 1.0618 | 6.64 |
| 256 | 2.5580 | 9.99 |
| 512 | 4.6550 | 9.09 |
| 1,024 | 8.9705 | 8.76 |

The engine refuses fewer than 32 reads. A program at 8 or 16 fails to
compile with `InvalidMILProgram`.

## What the shape says

Cost per read falls from 32 reads to 128, then jumps. Between 128 and 160 it
rises by 31%, and by 256 it has roughly doubled. Past that it settles near
9 microseconds and grows with the count.

The natural reading is that the engine processes a 128-wide activation
vector natively, and a wider program tiles into more than one pass. That
makes 128 the largest read count available at the cheap rate. This report
measures the shape rather than the cause, so treat the explanation as a
hypothesis.

Sublinearity below 128 has a separate cause. Each sweep's convolution reads
the whole weight matrix whatever the read count, because reads multiply only
the activation width. At production shape the couplings are about 44 MB per
program, so weight bandwidth dominates and cutting reads cannot remove it.

## What follows for the read count

Reads are close to free up to 128. Going from 32 to 128 buys four times the
parallel replicas for 32% more time per sweep. On time per sample, 128 wins
outright: 5.09 microseconds against 15.31 at 32 reads, three times better.

Cutting reads to reach a sweep budget is the wrong lever. It gives up
sampling breadth, which is the thing reads exist to provide, in exchange
for a saving the table shows to be small.

## Reaching 16,384 sweeps

At the four-colouring's 0.8818 ms per sweep through the production path,
16,384 sweeps cost 14.45 s, and the compile adds 0.13 s, for about 14.6 s
per job. Under the greedy eight-colouring the same job costs about 17.2 s.

Cutting to 32 reads would remove at most a quarter of the dispatch time
while removing three quarters of the samples, so it does not make 16,384
sweeps meaningfully cheaper in any way that preserves solution quality.

## Method

The probe is `crates/ane-miner/probes/reads_sweep.m`. The read count is not a
parameter of the production builder, which hardcodes 128 in `shape()`, so
the probe generates its own program text with the count lifted out. At 128
reads that text is byte-identical to `makeMIL`'s output apart from the build
identifier, which was checked by diffing the two. Everything else follows
`ane_bridge.m`'s create path, and `makeWeightBlob`, `makeSurface` and
`stageSurface` come from that file unchanged.

Shape is production: 4,608 channels with the Advantage2 four-colouring's
four classes of 1,148, 1,145, 1,145 and 1,139 nodes. One sweep per dispatch,
matching `BLOCK_SWEEPS`. Couplings come from seed 123 in `{-1, 1}` with zero
fields.

Each figure is 200 back-to-back evaluations of one prepared request. Values
are medians of three runs at 32, 64 and 128 reads, and of two runs at 96,
160 and 256. Every run went through `scripts/ane-guard`.

### Commands

```sh
clang -O3 -fobjc-arc -fobjc-arc-exceptions -std=c11 -Wall -Wextra -Werror \
  -framework Foundation -framework IOSurface \
  crates/ane-miner/probes/reads_sweep.m -o /tmp/reads-sweep
scripts/ane-guard -s 3 -t 900 -- /tmp/reads-sweep 200 32 64 96 128 160 256
```

## Limits of this measurement

The probe reuses one request and does not restage thresholds between calls,
which the production path does on every dispatch. That makes its absolute figures lower
than the production path's, 0.6439 against 0.8818 ms per sweep at 128
reads. Staging scales with the read count, so it would widen
rather than narrow the gap between counts. The ordering stands; the absolute
numbers are dispatch cost alone.

Solution quality is untouched here. Reads are independent replicas, so
fewer reads means fewer samples per sweep, and time to a valid solution can
worsen even where time per sweep improves. That question belongs with bead
`quip-miner-metal-fjo.2`.
