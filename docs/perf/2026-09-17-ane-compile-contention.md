# Compile contention across concurrent processes, 2026-09-17

## Result

Compile does not serialize across processes. Bead `quip-miner-metal-6jd`
proposed that it does, and the measurement refutes that.

Every run reached an overlap depth equal to the number of processes. Two of
two compiles ran at one instant, and four of four did. No run showed the
depth of 1 that a lock would produce.

Compile still scales poorly, because the processes share a resource without
queueing on it. Each compile gets slower as more run together, and the
throughput gain is modest.

| Processes | Median compile, ms | Median span, ms | Compiles per second | Against one process |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 209.564 | 209.564 | 4.772 | — |
| 2 | 287.201 | 366.043 | 5.464 | 1.145 |
| 4 | 450.463 | 681.723 | 5.868 | 1.230 |

That curve matches the one Task 2 measured for `evaluateWithQoS:` at
production shape, 1.165 and 1.166 at two processes and 1.253 and 1.293 at
four. Compile and evaluate scale alike, so compile does not cap concurrency.

The bead asked whether a second Apple Neural Engine (ANE) worker is worth
running. It raises throughput about 15%, and that figure now holds for real
mining, where every job compiles. A third and fourth worker together add
about 8 points more. Nothing here changes the conclusion drawn from the
evaluate-only figure Task 2 published.

## Why the earlier hypothesis looked right

Task 2 saw its four production loop starts stagger by 258, 288, and 278 ms
against a 283.887 ms compile median. Those numbers are close enough to one
compile each to suggest that the compiles queued.

They did not. The probe below starts every process from one shared deadline,
and the copies enter `quip_ane_create` within a few microseconds of each
other, yet still finish spread out. The stagger Task 2 saw is the compiles
slowing each other down, not waiting in line. The two explanations separate
on one reading. A lock would cap total compile throughput at the rate of one
process. Measured throughput rises with process count instead.

## Method

The probe is `crates/ane-miner/probes/compile_contention.m`. It includes
`crates/ane-miner/native/ane_bridge.m` and calls `quip_ane_create` unchanged,
the same convention `setup_profile.m`, `contention.m`, and `single_call.m`
already use in that directory.

The timed call is `quip_ane_create`, not `compileWithQoS:` alone. Create also
builds the Model Intermediate Language text. It then writes the weight blob
and loads the model. Compile is about 74% of that path. The stage table
in `docs/perf/2026-09-17-ane-setup-profile.md` gives 155.016 ms of compile
inside about 209 ms of create at one process. Read every figure here as a
property of create, dominated by compile.

Each copy allocates its weights and fields first, then parks at a shared
absolute deadline, then calls create. The deadline is a `monotonicUS()`
reading, which `ane_bridge.m` defines as
`clock_gettime_nsec_np(CLOCK_UPTIME_RAW)`, a system-wide monotonic clock.
Timestamps stay comparable across processes, which is what makes overlap
measurable rather than inferred. The runner allows one second of
headroom, and each copy reports `late` if it missed the deadline by more than
5 ms. No run in this report was late.

Shape is production: 4,608 channels and the eight tile lengths 857, 849, 817,
740, 685, 480, 136, and 13 from
`tests/fixtures/advantage2-system1.edges`. Sweeps is 1, tracking `BLOCK_SWEEPS`
in `crates/ane-miner/src/native.rs:10`. Couplings are in `{-1, 1}` from seed
123 and fields are zero, matching the other probes. Real neighbor data is not
needed, because this probe measures compile cost rather than solutions.

Three repetitions ran at each process count. The whole experiment ran under
`scripts/ane-guard`, acquired once around the runner rather than around each
copy. Guarding the copies individually would serialize them and destroy the
measurement.

### Commands

```sh
clang -O3 -fobjc-arc -fobjc-arc-exceptions -std=c11 -Wall -Wextra -Werror \
  -framework Foundation -framework IOSurface \
  crates/ane-miner/probes/compile_contention.m -o /tmp/compile-contention
```

```sh
for n in 1 2 4; do
  for r in 1 2 3; do
    scripts/ane-guard -s 3 -t 600 -- \
      crates/ane-miner/probes/compile_contention.sh /tmp/compile-contention $n production
  done
done
```

## Readings

`span` is the first start to the last end across the copies in one run.
`sum` adds every copy's own compile time.

| Processes | Repetition | Mean compile, ms | Span, ms | Sum, ms | Overlap depth |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 227.004 | 227.004 | 227.004 | 1 |
| 1 | 2 | 209.564 | 209.564 | 209.564 | 1 |
| 1 | 3 | 203.499 | 203.499 | 203.499 | 1 |
| 2 | 1 | 287.201 | 366.043 | 574.401 | 2 |
| 2 | 2 | 288.317 | 367.075 | 576.634 | 2 |
| 2 | 3 | 277.348 | 354.762 | 554.695 | 2 |
| 4 | 1 | 437.505 | 669.412 | 1,750.021 | 4 |
| 4 | 2 | 450.463 | 681.723 | 1,801.853 | 4 |
| 4 | 3 | 452.818 | 686.913 | 1,811.271 | 4 |

Repetition 1 at one process is 17.5 ms slower than repetition 3. This matches
the cold first-load cost `docs/perf/2026-09-17-ane-setup-profile.md` records
for `load_ms` in its own first run. The median absorbs it.

Spread within each process count is small. At two processes the three spans
fall in 354.762 to 367.075 ms, a range of 12.3 ms. At four they fall in
669.412 to 686.913 ms, a range of 17.5 ms. Neither range comes close to the
gaps between process counts, so the ordering is not an artifact of noise.

## Bounding the outcome

Two limits frame the measured span.

Perfect serialization would make the span the sum of the individual compiles.
At four processes that is 1,801.853 ms. The measured span is 681.723 ms, well
under it.

Perfect parallelism would hold the span at one compile, 209.564 ms. The
measured span is more than three times that.

The result sits between, nearer the serial end. Each compile at four
processes takes 2.149 times as long as a lone compile, so most of the added
process time is lost to contention rather than converted into work.

## Limits of this measurement

The probe times create and destroys the program at once. It does not run
`quip_ane_reset` or `quip_ane_evaluate`, so it says nothing about how compile
and evaluate interact when both run across processes. A real miner overlaps
one worker's compile with another's evaluate, and that combination is
unmeasured. The figures here and Task 2's figures agree closely enough that a
large surprise is unlikely, but neither report measures the mixed case.

The probe also compiles one program per process and exits. It does not
measure whether a long-lived process compiling repeatedly behaves the same as
a fresh process, which is the shape bead `quip-miner-metal-yba` would create.
