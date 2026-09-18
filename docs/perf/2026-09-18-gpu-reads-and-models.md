# Read count against models in flight on the GPU, 2026-09-18

## Result

The GPU holds 40 threadgroups at once, one per core, and each holds 32
replicas of one problem. That is the working set, whatever the colouring.
The read count decides how many problems share it. At 128 reads that is
10 problems per dispatch, and 40 at 32 reads. The streaming loop
keeps two batches in flight, so the count of problems in flight is twice
that.

Cutting reads raises jobs per second, but not in proportion below 64 reads.
At 16,384 sweeps and equal total work, 64 reads gives 1.8 times the jobs
per second of 128, and 32 reads gives 2.1 times under the default two-batch
overlap. Running one batch at a time lifts 32 reads to 2.8 times.

| Reads | Jobs | Wall, s | Jobs per second | Against 128 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 40 | 5.4 to 6.2 | 6.40 to 7.34 | 1.0 |
| 64 | 80 | 6.2 to 6.5 | 12.24 to 12.84 | 1.8 |
| 32 | 160 | 10.3 to 11.2 | 14.29 to 15.54 | 2.1 |
| 32, half budget | 160 | 9.2 | 17.36 | 2.5 |
| 32, one batch | 40 | 1.9 | 20.63 | 2.8 |

Every row does the same work: 160 words of 32 replicas, each for 16,384
sweeps. The last row is one batch of 40 problems with nothing overlapping
it, so it excludes the two-batch overlap the other rows carry.

## Why 32 reads falls short of linear

The kernel is linear in words. A single batch of 40 problems at 32 reads
takes 1,757 ms of GPU time against 1,631 ms for 10 problems at 128 reads,
a difference of 8%.

The overlap is where the time goes. With two batches in flight, each
40-problem batch at 32 reads takes 4,636 to 4,759 ms of GPU time. Two
overlapped batches finish in 4.9 s where two serial batches would finish
in 3.5 s. At 128 reads the same overlap helps: each 10-problem batch takes
2,394 to 2,449 ms against 1,631 alone, so 20 problems finish in 2.45 s
where two serial batches would take 3.26 s.

What differs between the two cases is the count of unique problems in
flight, 20 at 128 reads against 80 at 32. Each threadgroup streams its own
problem's couplings on every sweep. Four threadgroups of one problem read
one stream, and one threadgroup per problem cannot share anything. The
three measured points order by that count, but this report did not
measure cache behaviour, so treat the cause as the consistent explanation
rather than a demonstrated one.

Halving the threadgroup budget with `QUIP_METAL_TG_PER_CORE=0.5` keeps 40
unique problems in flight across two batches and recovers part of the
loss, at 17.36 jobs per second. It does not reach the single-batch rate,
because two 20-group command buffers running together are slower than one
40-group buffer.

## The neural engine for comparison

At 16,384 sweeps and 128 reads the GPU runs 7.3 jobs per second. The
Apple Neural Engine (ANE) runs one job in 14.6 s, or 0.068 jobs per
second, and four ANE workers reach 0.086. The ANE adds about 1% to the
GPU. The ANE cannot trade reads for models: its fixed 0.425 ms per sweep
is that model's coupling stream, and every model pays it.
`docs/perf/2026-09-18-ane-read-count.md` records that measurement.

## Solution quality, one signal

The benchmark runs one graph with a different seed per job, so the mean of
each job's best energy compares the best of 128, 64, and 32 replicas on
one problem.

| Reads | Jobs | Mean best, milli | Best of all jobs, milli |
| ---: | ---: | ---: | ---: |
| 128 | 40 | -14,832,850 | -14,838,000 |
| 64 | 80 | -14,830,225 | -14,838,000 |
| 32 | 160 | -14,828,100 | -14,842,000 |

The mean best worsens by 2,625 milli from 128 to 64 reads and by 2,125
from 64 to 32. The best across all jobs is level, because every row draws
5,120 replicas in total. Whether a 32-read job clears a target as often as
a 128-read job depends on where the target sits against that spread. One
graph at one sweep count cannot settle that, so the seeded study in bead
`quip-miner-metal-fjo.2` still owns the decision.

## Method

The harness is `tests/msa_bench.rs`, unchanged. It streams copies of one
4,577-node Advantage2 System 1 problem through `run_stream` with no
governor and times the whole run from first send to last result. Couplings
come from graph seed 1 in `{-1, 1}` with fields in `{-1, 0, 1}`. The
colouring is the greedy eight-class default. Two rounds ran in the order
128, 32, 64 reads.

The `batch complete` debug lines give per-batch GPU time and chunk count.
Every batch ran 26 chunks at the default budget and 13 at half.

### Commands

```sh
cargo test --release --test msa_bench --no-run
BIN=target/release/deps/msa_bench-*
QUIP_BENCH_JOBS=40  QUIP_BENCH_READS=128 QUIP_BENCH_SWEEPS=16384 QUIP_BENCH_KERNEL=msa $BIN --ignored --nocapture
QUIP_BENCH_JOBS=80  QUIP_BENCH_READS=64  QUIP_BENCH_SWEEPS=16384 QUIP_BENCH_KERNEL=msa $BIN --ignored --nocapture
QUIP_BENCH_JOBS=160 QUIP_BENCH_READS=32  QUIP_BENCH_SWEEPS=16384 QUIP_BENCH_KERNEL=msa $BIN --ignored --nocapture
QUIP_METAL_TG_PER_CORE=0.5 QUIP_BENCH_JOBS=160 QUIP_BENCH_READS=32 QUIP_BENCH_SWEEPS=16384 QUIP_BENCH_KERNEL=msa $BIN --ignored --nocapture
RUST_LOG=quip_miner_metal=debug QUIP_BENCH_JOBS=40 QUIP_BENCH_READS=32 QUIP_BENCH_SWEEPS=16384 QUIP_BENCH_KERNEL=msa $BIN --ignored --nocapture
```

The first try at this sweep ran every configuration at 128 reads. The
shell loop split its arguments with an unquoted variable, which zsh does
not split, so `QUIP_BENCH_READS` was empty and took its default. Six
identical result lines gave it away. Check the `jobs x reads` echo on
every line before reading a table from this harness.

## Limits of this measurement

`declared_stream_width` sizes the coordinator credit from 128 nominal
reads, so a production change to 32 reads needs that constant to follow or
the GPU runs under-fed.

The benchmark scores every read on the host and clones the graph once per
job. Those costs scale with total reads and with jobs, so the 160-job rows
carry more host work than the 40-job rows. The per-batch GPU times exclude
that work and show the same ordering.

The measurement covers one graph seed and one sweep count under the
greedy colouring. The four-colouring changes time per sweep and would
move every row by a similar factor. Bead
`quip-miner-metal-fjo.3` owns the Metal knob sweep and should carry the
read count and the overlap policy together, because this report shows
they interact.
