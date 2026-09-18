# Runtime couplings against the compiled weight blob, 2026-09-18

## Result

Moving the couplings to a runtime input costs more than the compile it
removes. Bead `quip-miner-metal-yba` proposed the move. The measurement
rejects it at production sweep counts.

The best runtime-J variant runs one sweep in 1.752 ms. Production, which
compiles the couplings into the weight blob, runs one sweep in about 1.13 ms.
The extra 0.6 ms per sweep buys back a 155.016 ms compile, so the trade turns
negative after about 254 sweeps. Production jobs run 2,048 to 8,192 sweeps.

| Sweeps in a job | Compile saved, ms | Extra sweep cost, ms | Net, ms |
| ---: | ---: | ---: | ---: |
| 254 | 155.016 | 155.0 | 0 |
| 2,048 | 155.016 | 1,249.3 | +1,094 worse |
| 8,192 | 155.016 | 4,996.5 | +4,841 worse |

A second finding closes the option more firmly than the arithmetic does. The
runtime-J program cannot fuse sweeps at all. At one sweep per program it
takes 1.752 ms. At two it takes about 700 ms, four hundred times more, and
the cost stays near that level at four and eight sweeps. Production depends
on fusing being cheap, and this variant loses that.

Every run in this report validated against a reference sweep and reported
zero mismatches, so these are timings of a graph that computes the right
answer.

## The variants

All four take J as a runtime input, and differ only in how the sweep consumes
it. Production is not in this table. It compiles J into the weight blob and
reaches about 1.13 ms per sweep.

| Mode | Sweep, ms | Compile, ms | Against production |
| --- | ---: | ---: | ---: |
| conv | 1.752 | 445.7 | 1.55 |
| complete | 2.469 | 247.7 | 2.18 |
| hoist | 2.932 | 243.9 | 2.59 |
| state_mm | 3.013 | 309.4 | 2.67 |

`conv` keeps production's convolution and sources its weight from an input.
The other three are the matmul family: `complete` reshapes each tile to a
matrix product, `hoist` lifts that reshape out of the sweep loop, and
`state_mm` also holds the state in matrix layout.

Every matmul variant is slower than the `conv` variant. The earlier note
putting matmul at about 10 ms per program is not reproduced here, and the
gap between 10 ms and these figures is large enough that the two are
measuring different things. Treat the older figure as superseded.

## Why the compile cannot simply be amortized

The table above already credits the runtime-J variants with a compile cost of
zero per job, which is the most favorable reading available. One compiled
program serves every job, because J and h arrive as data.

That credit does not rescue the option. At 2,048 sweeps the extra per-sweep
cost is eight times the whole compile it removes. The compile is not the
expensive part of a job at production sweep counts, so removing it cannot pay
for a slower inner loop.

## The sweep-fusing collapse

| Sweeps per program | Evaluate, ms | Per sweep, ms | Compile, ms |
| ---: | ---: | ---: | ---: |
| 1 | 1.740 | 1.740 | 434.2 |
| 2 | 631.3 | 315.7 | 1,189.8 |
| 4 | 765.8 | 191.5 | 2,296.1 |
| 8 | 651.5 | 81.4 | 4,535.4 |

The host is idle through the slow evaluations. At two sweeps the process
reports 160 to 197 microseconds of its own processor time against evaluations
of 681,279 to 761,082 microseconds, so the work is not falling back to the
processor. The device itself is that slow.

One reading fits these numbers without being proven by them. A fused
runtime-J program may have to read the whole 44 MB of couplings again for
each sweep it contains. The couplings arrive through input surfaces, not
through a compiled blob that the runtime can lay out once and keep resident.
Production fuses sweeps at a flat cost per sweep from one through eight,
which is what a resident blob would allow. Settling the question would need a
probe that counts device-side reads, and no such probe exists yet.

## Method

Graphs come from `crates/ane-miner/probes/profile_graphs.py`, which already
emits every mode above with J, h, and the thresholds as runtime inputs. The
harness is `crates/ane-miner/probes/profile_runner.m`. Neither is modified
here. Production bridge entry points are unchanged.

`crates/ane-miner/probes/runtime_j_fixture.py` builds the production-shape
inputs and manifest those two need. The production path in
`profile_graphs.py` wants a fixture directory that is not checked in, and
that fixture carries fields this question does not need, so this script
writes the graph and manifest directly in the form
`single_call.small_fixture` uses.

Shape matches the other probes in that directory: 4,608 channels and the
eight tile lengths 857, 849, 817, 740, 685, 480, 136, and 13. Couplings are
in `{-1, 0, 1}` with at most 20 nonzero neighbors per variable, which is the
advertised topology limit.

The density decides whether the correctness check means anything. It does not
affect speed. A dense
`{-1, 1}` matrix sums 4,608 terms per row, and fp16 represents integers
exactly only to 2,048, so the accumulator rounds and disagrees with the
integer reference. A first attempt at this measurement used dense couplings
and reported 33,357 mismatches for that reason. Coupling density does not
change what a dense convolution or matrix product costs.

The reference mirrors the `metropolis` block `profile_graphs.py` emits. A node
flips exactly when its threshold plus its own spin times its field is at
least zero. The satisfied-terms rule in `single_call.small_fixture` is
deliberately independent of that margin and agrees with it only for that
fixture's constant h and single-neighbor couplings. Using it at production
shape produced 72,529 mismatches.

Each run reports the median of ten evaluations of one prepared request. The
figures above are medians across three runs for `conv` at one sweep, which
read 1.695, 1.752, and 1.758 ms, and single runs elsewhere.

Every run went through `scripts/ane-guard`, so no measurement shared the
device.

### Commands

```sh
clang -O3 -fobjc-arc -fobjc-arc-exceptions -std=c11 -Wall -Wextra -Werror \
  -framework Foundation -framework IOSurface \
  crates/ane-miner/probes/profile_runner.m -o /tmp/profile-runner
```

```sh
python3 crates/ane-miner/probes/runtime_j_fixture.py /tmp/rj --mode conv \
  --sweeps 1 --eval-repeats 10
scripts/ane-guard -s 3 -t 600 -- /tmp/profile-runner /tmp/rj/manifest.json 300
```

## The production baseline

Production runs about 1.13 ms per sweep at `BLOCK_SWEEPS` 1, from the two
slope fits in `docs/perf/2026-09-17-ane-setup-profile.md`, which gave 1.0970
to 1.1589 and 1.098 to 1.161 ms per sweep.

A same-day check with `crates/ane-miner/probes/contention.m` at production
shape ran 200 back-to-back evaluations in 395.381 to 396.141 ms at two sweeps
per dispatch, which is 0.99 ms per sweep.

The comparison is not perfectly matched. `contention.m` drives the production
bridge and `profile_runner.m` drives its own harness, and the two differ in
how a request is prepared. Taking the baseline at its slowest published
figure, 1.16 ms, still leaves the `conv` variant 0.59 ms per sweep behind, and
the break-even at 262 sweeps. The conclusion does not turn on which baseline
figure is chosen.

## Limits of this measurement

The probe compiles a program and evaluates one prepared request repeatedly.
It does not stage a fresh J between jobs, which is the operation a real
runtime-J miner would perform once per job. That cost is additional to
everything above and would make the option worse, not better.

This report does not measure a sparse or blocked representation of J. Every
variant here carries the full padded matrix, 44 MB, as dense fp16.
`docs/perf/2026-09-16-ane-local-routing.md` measured a sparse local variant at
3.4 to 3.6 times the dense path, so that direction is already recorded as
slower.
