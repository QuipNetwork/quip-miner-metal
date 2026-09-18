# Read count against sweep cost on the ANE, 2026-09-18

## Result

Keep 128 reads. Reads cost 1.75 microseconds each at the margin, against a
fixed 0.425 ms per sweep that no read count can remove, so 128 buys four
times the samples of 32 for 32% more time. Above 128 the cost per read
roughly doubles.

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

## The cost model

Up to 128 reads the cost is a fixed charge plus a small charge per read. A
least-squares fit over 32, 64, 96 and 128 reads gives

    sweep = 0.425 ms + 1.75 microseconds x reads

with an R-squared of 0.94.

The fixed 0.425 ms is 65% of a 128-read sweep. It matches the cost of
streaming the couplings: the weight blob is 43,352,640 bytes, and moving it
in 0.425 ms is 102 GB/s, which is a credible memory rate on this host. Each
sweep's convolution reads the whole matrix whatever the read count, because
reads multiply only the activation width.

That is the whole answer to why removing three quarters of the arithmetic
returns only a quarter of the time. Reads scale the 35% of the sweep that is
activation work. They cannot touch the 65% that is weight streaming. The
model predicts a 26% saving from cutting 128 reads to 32, against 25%
measured.

Above 128 the model breaks. It predicts 0.705 ms at 160 reads against 1.062
measured, and 0.873 at 256 against 2.558. The jump is consistent with the
engine processing a 128-wide activation vector natively and tiling a wider
program into more than one pass, which would make 128 the widest count
available before the step. This report measures the step rather than its
cause, so treat that reading as a hypothesis.

The cost-per-read column falls with the count because it is the fixed charge
divided by the reads, plus the marginal 1.75 microseconds. Its minimum at
128 marks where the tiling step falls, not a special property of the number
itself.

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

### Checks against measurement artifacts

The ordering of the runs does not change the result. Sweeping 32, 64, 96,
128 gives 0.4975, 0.5116, 0.5959 and 0.6498 ms. Sweeping the same counts in
reverse gives 0.4973, 0.5084, 0.6003 and 0.6595. Alternating 32 and 128
three times inside one process gives 0.4816, 0.4885 and 0.4781 against
0.6342, 0.6220 and 0.6326, with no overlap between the two bands. Running
each count in its own process gives 0.4952 and 0.7056.

Other processes use the engine during these runs. The system log recorded 85
program-instance creations by `aned` in one two-minute window, none of them
this probe's. The guard serialises this repository's own probes and cannot
exclude Apple's daemons. Two things make that acceptable here. The
alternating runs above hold to about 2% across the session, and background
work of that kind would fall on both read counts alike rather than produce a
consistent ratio between them.

Compile time is not monotonic in the read count. It runs about 95 to 105 ms
at 32 and 128 reads against about 110 to 120 ms at 64 and 96. Reversing the
order leaves that pattern in place. Compile is under 1% of a 16,384-sweep job, so
this report does not chase it, but it is recorded rather than smoothed
away.

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
