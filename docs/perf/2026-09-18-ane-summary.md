# ANE performance summary, 2026-09-18

Ties together the measurements on this branch. Each section names the report
that carries the evidence.

## Where the ANE stands

Three changes shipped. All are measured, and none depends on a judgment
about solution quality.

| Change | Effect | Report |
| --- | --- | --- |
| `BLOCK_SWEEPS` 2 to 1 | compile 277.6 ms to 155.0 ms | `2026-09-17-ane-setup-profile.md` |
| Greedy eight-colouring to the Advantage2 four-colouring | 1.0384 to 0.8818 ms per sweep, compile 208.3 to 129.9 ms | `2026-09-18-ane-four-coloring.md` |
| Sparse coupling encoding | 0.66 to 0.46 ms per sweep on dispatch, a 16,384-sweep job 16.6 to 13.7 s, compile 0.10 to 0.82 s | `2026-09-18-ane-utilization.md` |

The sparse encoding stores each tile's couplings as a one-bit mask and the
nonzero values, and the engine consumes that form directly. The output is
bit-identical to the dense program.

## Cost of a 16,384-sweep job

16,384 sweeps is the stated target for competitiveness.

| Configuration | Compile, ms | Sweeps, s | Job, s |
| --- | ---: | ---: | ---: |
| Greedy eight colours, one sweep per dispatch | 208.3 | 17.01 | 17.2 |
| Four colours, one sweep per dispatch | 129.9 | 14.45 | 14.6 |
| Four colours, sparse couplings, whole `--solve` process | 820 | | 13.7 |

The first two rows cover dispatch and compile only. The last row is the
whole `--solve` process on a testnet block, against 16.6 s for the same
binary with dense couplings, so it is the production figure. The host now
spends about as long generating and staging thresholds between dispatches
as the engine spends on them. Bead `quip-miner-metal-fjo.9` carries the
overlap.

A second concurrent worker raises throughput about 15%, and that figure now
holds for real mining because compile scales with process count rather than
serialising. Four workers reach about 25%.
`2026-09-17-ane-compile-contention.md` carries both.

## Levers measured and rejected

Four. Each is closed with evidence rather than opinion, so none needs
revisiting without new information.

**Read count below 128, under dense couplings.** A sweep cost 0.425 ms plus
1.75 microseconds per read, fitted over 32 to 128 reads at an R-squared of
0.94. That fixed part was 65% of a 128-read sweep, and it matched streaming
the 43.35 MB coupling blob at 102 GB/s. Every dispatch paid it whatever the
read count, which left cutting to 32 reads giving up three quarters of the
samples to save a quarter of the time. Above 128 the cost per read roughly
doubled. The engine refuses fewer than 32 reads outright.
`2026-09-18-ane-read-count.md`. The sparse encoding removes that fixed
stream, and with it the reason to hold 128 reads: from 64 to 1,024 reads
the sparse sweep costs about 3 microseconds per read-sweep, so 64 reads
would give about 1.6 times the valid proofs per second once the lane width
follows the job. Bead `quip-miner-metal-fjo.8`.

**Couplings as a runtime graph input.** Works and validates, but the best
variant runs 1.752 ms per sweep against 1.13, so the trade turns negative
after about 254 sweeps against jobs of 2,048 to 8,192. It also cannot fuse
sweeps: two sweeps in one program cost about 700 ms rather than twice 1.752.
Every matmul variant measured slower than the convolution variant.
`2026-09-18-ane-runtime-couplings.md`.

**Compile once per topology, then write each job's couplings into the
compiled program.** Three routes, all closed. The `weightsBuffer` argument on
`_ANERequest` is ignored for a program whose weights are compiled in. The
adapter-weight instance path exists and is reachable, but the daemon drops
the connection without the private entitlement. The compiled couplings are
not mapped into the client's address space at all, so patching them in place
cannot be driven from here. Bead `quip-miner-metal-kyk` carries the detail.

**Sweeps fused per dispatch above 1.** Per-sweep cost is flat from 1 to 8
while compile grows from 155 ms to 1,043 ms, so fusing buys nothing.
`2026-09-17-ane-setup-profile.md`.

## What the compile costs now

129.9 ms per job, down from 277.6 ms at the start of this work. No route to
removing it survives: the runtime caches couplings at compile time, and every
way of supplying them afterwards is either slower or entitlement-gated.
Treat it as a fixed per-job cost.

At 16,384 sweeps the compile is 0.9% of the job, so it no longer merits
attention. It mattered when jobs were 2,048 sweeps and it was 277 ms.

## Models in flight, and the GPU next to the ANE

The ANE executes one program at a time under every encoding. One dense
model at 128 reads already runs the engine at 60% of its measured MAC
ceiling and 74% of its weight-stream ceiling, so a second process takes
only what is left, 1.23 times one. Under sparse couplings two 128-read
models reach 1.22, two 1,024-read models 1.04, and two 64-read models
0.80, slower than one. The packing dimension is reads per model, and from
64 to 1,024 reads the sparse program delivers about 340,000 lane-sweeps per
second, 1.7 times the dense program at 128 reads.
`2026-09-18-ane-utilization.md`. The GPU holds 40 threadgroups of 32
replicas each, one per core, so
it holds 10 models at 128 reads and 40 at 32, with two batches in flight
on top. At 16,384 sweeps and 128 reads the GPU runs 6.2 to 7.3 jobs per
second and the ANE 0.073 on the production path. The ANE adds about 1% to
the GPU. `2026-09-18-gpu-reads-and-models.md`.

The engine draws about 1 W with a model in flight and 0 W idle, whatever
the encoding, because a sweep over a 99.6%-zero matrix switches few
multiplier bits. Engine energy per 16,384-sweep job is about 10 J dense
and 6 J sparse.

The GPU trades reads for models linearly down to 64 reads, at 1.8 times
the jobs per second of 128. At 32 reads the default two-batch overlap puts
80 distinct problems in flight and the per-batch GPU time rises 2.7 times,
so 32 reads reaches 2.1 times under the default and 2.8 times with one
batch at a time. Reads and the overlap policy interact.

## Read count against the testnet's own blocks

Sixty recent Aglais blocks, regenerated exactly from their nonces, and run
at 16,384 sweeps. The chance that one job produces a valid proof is 0.30
at 32 reads, 0.40 at 64, 0.54 at 128, and 0.68 at 256. The chance that it
beats the energy that won the block is 0.16, 0.20, 0.32, and 0.47. Reads
raise the chance per job, and the GPU's jobs per second fall faster than
the chance rises, so on the GPU 64 reads gives the most valid proofs per
second under the default streaming, at about 5.0 against 3.9 at 128, and
32 reads with one batch at a time gives 6.1. The ANE keeps 128 until its
lane width can follow the job.

Sweeps trade against the chance per job. At 64 reads, 8,192 sweeps gives
1.53 times the baseline's valid proofs per second but only 1.07 times its
winner-beating proofs, with an interval that spans parity, and under the
chain's default `min_solutions` of 5 it falls to 0.55 of the baseline.
The chance per job collapses between 4,096 and 2,048 sweeps. The chance
stops rising at 14,336 sweeps, and 64 reads at 14,336 beats the baseline
on every measure: 1.67 times the valid proofs per second, 1.51 times the
winner-beating proofs, and 1.40 against 1.21 proofs per second under a
`min_solutions` of 5. Doubling sweeps from 8,192 to 16,384 buys about
10,000 milli of energy on any block, whatever its target. Those studies
ran on the GPU. `2026-09-18-testnet-sweeps-study.md` and
`2026-09-18-testnet-sweeps-intermediates.md`.

One account mined the testnet with a quantum annealer and stopped winning
at qblock 1,308, when the target passed its best energy. The Metal solver
beats its winning proofs on 60 of 60 by a median 30,000 milli, and every
block of that era is reachable. The annealer's winning energy ranks those
blocks by the solver's energy at Spearman +0.74 to +0.77.
`2026-09-18-testnet-annealer-blocks.md`.

On 50,000 fresh nonces the solver at 64 reads and 14,336 sweeps lands 5.0
standard deviations short of today's target, none valid, and the instance
sets 98% of a job's energy. A 512-sweep probe ranks those nonces at Spearman
+0.923, and solving only the 1 in 128 it ranks deepest keeps all 10 of the
deepest and 46 of the deepest 50, worth 16.2 times the valid proofs per
second on one M4 Max, with a spread of 1.4 over ten rounds. The screened
miner answers its first nonce in 3.4 s against 2.8 s unscreened, so it keeps
16.1 of those times over the chain's qblock windows, as long as it solves
each kept nonce on arrival rather than filling a batch first. A solver at 256 reads and 65,536 sweeps agrees with
the probe at +0.95 on an unrestricted sample, and its deepest probe-picked
nonce stops 11,068 milli short of the target against 103,068 for a random
control, so the probe reads the instance rather than the budget. An annealer
at the chain's schedule screens a seventh to a tenth as many nonces per
second as that probe. `2026-09-18-probe-screen-at-scale.md` and
`2026-09-18-qpu-screen.md`.

The testnet mines zero-field problems on the Advantage2 graph with one
read below target as a valid proof. Winners self-report 1.0 to 3.9 s of
compute. `2026-09-18-testnet-reads-study.md` carries the fetch, the
regeneration with its nonce check, and the run.

## Open questions

**Solution quality under four colours.** Mean lowest energy was worse under
four colours by 1,400 and 250 milli on two graph seeds in the Metal work, and
no run has separated that from noise. Two seeds cannot. This gates the
four-colour default on both engines, because a colouring that samples worse
can cost more than the 15% of wall time it saves. Bead
`quip-miner-metal-fjo.2`.

The ANE now runs the four-colouring for its layout, which is a speed change.
That is not the same as making four colours the sampling default, which is
the decision still owed.

**The Metal knobs.** Colouring scheme, `QUIP_METAL_TG_PER_CORE`, jobs per
batch, sweeps per chunk, the 0.4 safety factor, reads, and node cap are
unswept. The chunk bound matters as much as throughput: the largest logged
chunk was 439 ms under greedy against 277 ms under four colours, and greedy
broke the 400 ms watchdog bound. Bead `quip-miner-metal-fjo.3`.

**The router job mix.** The engines couple through the combined router, so
the target is the best pair under a stated mix. Needs the two configurations
first. Bead `quip-miner-metal-fjo.4`.

## Measurement practice this branch established

`scripts/ane-guard` serialises device access across processes. Two agents
held device work at once twice during the earlier plan, once while measuring
concurrency, which is the measurement contention destroys. Acquire it once
around a whole experiment, never around each process of a concurrency test.

Four traps cost real time here and are recorded on their beads: a memory
scan matching its own search pattern; `aned` log lines from other processes
that read exactly like your own request succeeding; a direct dispatch that
reads back the input surface because it did not advance the surface index;
and a correctness reference copied from a fixture whose rule only holds for
that fixture's special structure.

Every probe that reports a null result carries a positive control. An absent
signal means nothing without one.
