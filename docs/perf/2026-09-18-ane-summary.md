# ANE performance summary, 2026-09-18

Ties together the measurements on this branch. Each section names the report
that carries the evidence.

## Where the ANE stands

Two changes shipped. Both are measured, and neither depends on a judgment
about solution quality.

| Change | Effect | Report |
| --- | --- | --- |
| `BLOCK_SWEEPS` 2 to 1 | compile 277.6 ms to 155.0 ms | `2026-09-17-ane-setup-profile.md` |
| Greedy eight-colouring to the Advantage2 four-colouring | 1.0384 to 0.8818 ms per sweep, compile 208.3 to 129.9 ms | `2026-09-18-ane-four-coloring.md` |

The four-colouring is the larger of the two by about four times.

## Cost of a 16,384-sweep job

16,384 sweeps is the stated target for competitiveness.

| Configuration | Compile, ms | Sweeps, s | Job, s |
| --- | ---: | ---: | ---: |
| Greedy eight colours, one sweep per dispatch | 208.3 | 17.01 | 17.2 |
| Four colours, one sweep per dispatch | 129.9 | 14.45 | 14.6 |

The branch takes a 16,384-sweep job from about 17.2 s to about 14.6 s, a
15% reduction. Both figures cover dispatch and compile only. Nothing in them
passes through the miner channel, the governor, or result scoring, so
neither is a production throughput claim.

A second concurrent worker raises throughput about 15%, and that figure now
holds for real mining because compile scales with process count rather than
serialising. Four workers reach about 25%.
`2026-09-17-ane-compile-contention.md` carries both.

## Levers measured and rejected

Four. Each is closed with evidence rather than opinion, so none needs
revisiting without new information.

**Read count below 128.** Cost per read is 5.09 microseconds at 128, against
15.31 at 32 and 9.99 at 256. The inherited count is a minimum rather than a
plateau. Cutting reads to 32 removes three quarters of the arithmetic and
returns only 25% of the time, because each sweep's convolution reads the
whole 44 MB weight matrix whatever the read count. The engine refuses fewer
than 32 reads outright. `2026-09-18-ane-read-count.md`.

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
