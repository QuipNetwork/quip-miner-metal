# Metal multi-spin coded simulated annealing (`quip-metal-msa`)

Design for porting quip-miner-cuda MR 26 (`quip-cuda-msa`, `kernels/msc.cu`)
to this crate, and a comparison of how the CUDA and Metal miners map the same
algorithms onto their hardware. The comparison drives the design choices.

Source MR: https://gitlab.com/quip.network/quip-miner-cuda/-/merge_requests/26
(branch `msa-cuda-kernel`, commit `e7fc201`). It ports `sa_msc.rs` from
quip-miner-cpu v0.3.3 (Isakov, Zintchenko, Ronnow, Troyer, *Optimised
simulated annealing for Ising spin glasses*, Comput. Phys. Commun. 192, 2015).

## What the MR adds, and what transfers

| MR change | Transfers to Metal? | Why |
|-----------|---------------------|-----|
| `kernels/msc.cu`: 64 replicas per `u64` word, integer Metropolis, bit-sliced neighbour count, colour-parallel updates in shared memory | Yes, with a different word width and threadgroup mapping | Apple GPUs cap threadgroup memory at 32 KB and have 32-bit ALUs (see below) |
| `KernelKind::Msc` on the device, `AlgoState::Msc`, `launch_msc` | Yes, as a crate-wide `Kernel` enum | The Metal crate keys width and sizing on a type tag with no device, so the kernel must be a first-class value |
| `cuda_msa_identity` with algorithm `"msa"` and an envelope of 7392 to 29568 sweeps at 128 reads | Yes, with a Metal envelope measured on Apple hardware | Throughput per threadgroup is unknown until measured |
| Stop on cancel: check `EXIT_NOW` inside the sweep loop | No | Metal cannot stop a committed command buffer. It checks cancellation before each chunk and uses a safety margin tuned for measured chunks below 400 ms. |
| Parallel host scoring (`QUIP_SCORE_THREADS`) | Already present | `harvest_batch` scores problems on a rayon pool |
| `QUIP_MSC_DIAG` compile switches | Yes, in tests only | metal-rs 0.33 exposes no preprocessor defines, so `tests/diagnostics.rs` prepends `#define QUIP_MSA_DIAGNOSTICS` to the kernel source. That build renames the entry point to `msa_anneal_diag` and adds a flip counter at buffer 25 and energy parts at buffer 26. No production code compiles it |

## Architecture comparison, CUDA versus Metal

Both crates run the same three algorithms against the same protocol. They map
work onto the GPU in different ways. Each row below names a difference and the
consequence for the multi-spin port.

### Execution model

CUDA runs one persistent, self-feeding kernel per session. Each nonce (block)
owns three rotating slots in device memory. Thread 0 claims a `READY` slot with
`atomicCAS`. The block anneals the slot and publishes `COMPLETE`. Thread 0
then spins with `__nanosleep` until the next slot is `READY`. The host uploads the next model into a free
slot while the kernel runs. `EXIT_NOW` ends the kernel.

Metal has no persistent kernel. The host collects up to `stream_width` jobs
that share a topology and dispatches them as one batch. A batch is one command
buffer per chunk of the beta ladder, committed as encoded. The loop double
buffers: it forms and commits batch `k+1` while batch `k` runs, then waits and
harvests `k`.

Consequence: the Metal multi-spin kernel is a plain compute kernel with the
same buffer layout as the other two Metal kernels. The slot control plane of
`msc.cu` (`s_active_slot`, `EXIT_NOW`) has no Metal counterpart.

### Watchdog and chunking

macOS aborts a command buffer that runs longer than a few seconds and freezes
the machine while the GPU resets. Metal kernels take a
`(beta_start, beta_count)` window and persist their carry-over state in device
buffers between chunks. `chunk_plan` sizes chunks toward a 500 ms planning
ceiling from a measured update rate. The safety margin targets measured
chunks below 400 ms. A compute-only device has no graphics watchdog, so
the CUDA miner runs a whole model in one launch.

Consequence: the multi-spin kernel persists its spin words and per-thread RNG
state every chunk and reloads them when `beta_start > 0`. Chunk boundaries
fall on rung boundaries, so a chunked anneal is bit-identical to an unchunked
one. `chunk_plan` needs a word-update rate for this kernel.

### Problem-to-hardware mapping

| Kernel | CUDA | Metal |
|--------|------|-------|
| SA | A block per nonce with a thread per read. Unpacked state in thread-local memory, `delta_energy` in a global workspace | A threadgroup per problem with a thread per read. Bit-packed state and `delta_energy` in thread-private arrays (`thread int8_t delta_energy[4593]`) |
| Gibbs | `blocks_per_nonce = 4` blocks share one nonce. Reads are handed out through a per-nonce work queue. Unpacked state in `__shared__`, one read at a time per block | One threadgroup per sample, 256 threads split each colour's nodes. Unpacked state in `threadgroup` memory |
| Multi-spin (CUDA MR) | A block per nonce. `N * words` `u64` words in run-time sized shared memory. Thread `tid` owns word `tid & (words - 1)` of the spins it visits | A threadgroup per `(problem, word)`. `N` `uint` words in run-time sized threadgroup memory. See design below |

Consequence: the Metal chromatic Gibbs kernel already uses the mapping the
multi-spin kernel needs. The host side reuses its threadgroup budget logic:
a problem costs `words` threadgroups, so the batch shrinks as reads grow.

### Shared memory

CUDA consumer parts opt in to about 99 KB of shared memory per block.
The MR fits two `u64` words per spin (128 reads) for 4577 spins in that
budget, and raises `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` at load.

Apple GPUs allow 32 KB of threadgroup memory per threadgroup with no opt-in.
Measured on this M4 Max: `maxThreadgroupMemoryLength = 32768`. One `u64` word
per spin for Advantage2's 4577 spins is 36,616 bytes and does not fit. One
`uint` word per spin is 18,308 bytes and fits beside the 8 KB threshold row.

Consequence: 32 replicas per word, one word per threadgroup. This is the
single most important divergence from the CUDA kernel.

### Integer width

NVIDIA ALUs execute 64-bit integer ops natively. Apple GPU ALUs are 32-bit.
MSL supports `ulong`, but every 64-bit bitwise op splits into two 32-bit ops
and doubles register pressure.

Consequence: a `uint` word costs about the same ALU work per replica as a
`ulong` word would, and it halves the register footprint of the 21-input
carry-save tree. The threadgroup memory cap forces the same width.

### Energies

CUDA SA tracks the running energy on the device. CUDA Gibbs reduces it with
warp shuffles. The CUDA multi-spin kernel writes 0 and the host rescores.
Metal never computes an energy on the device (MSL has no `double`), and
scores every read with `energy_milli` on a rayon pool.

Consequence: the Metal multi-spin kernel writes no energies. `harvest_batch`
is unchanged.

### Random numbers

CUDA SA uses xorshift32 per thread. CUDA Gibbs and the multi-spin kernel use
splitmix64 seeding with xorshift64 streams. Metal SA uses xorshift32 and
Metal Gibbs uses xoshiro128** seeded by splitmix32, both persisted per chunk.

Consequence: the Metal multi-spin kernel uses xoshiro128** per thread, four
`uint` in the layout `persistent_rng` already stores for Gibbs. Only the
initial words and the threshold rows consume the stream. Sweep offsets are
a hash of `(seed, problem, word, rung, sweep)`, so resume needs no extra
state.

### Threshold table

The CPU and CUDA kernels compare a 64-bit uniform against
`cut[m] = floor(exp(-2 * beta * m) * 2^64)`. Metal compares a 32-bit uniform
against `cut[m] = floor(exp(-2 * beta * m) * 2^32)` computed with `float`
`exp`. Probabilities below `2^-32` round to a cut of 0 and never draw `M >= m`.
A rung draws 8192 thresholds, so an event with probability under `2^-32` is
not observable at either width. A `float` mantissa carries 24 bits, adequate for
a threshold.

### Kernel compilation

CUDA compiles with NVRTC per architecture and caches the PTX. The compile
passes `-DQUIP_MAX_NODES` and `-DMSC_DIAG`. Metal compiles the `.metal` source at
device open with no defines.

Consequence: the node cap is a host constant checked against the
threadgroup allocation, and the kernel takes `N` at run time.

### Cancellation

CUDA peeks `EXIT_NOW` every eight rungs and leaves mid-model. Metal checks
the cancel token before committing a batch and cannot stop a committed
command buffer. The stream keeps at most two batches in flight, with one
committed chunk per batch. It checks cancellation before encoding and
committing each later chunk. A cancelled batch discards partial results
and returns each job's credit after its committed chunk finishes.

## Design

### Kernel: `kernels/msa.metal`, entry point `msa_anneal`

- **Word.** `uint`, 32 lanes. Bit `r` of `state[i]` is spin `i` of replica
  `r`, `0` meaning `+1` and `1` meaning `-1`. Same convention as the packed
  output of every other kernel.
- **Threadgroup.** One threadgroup per `(problem, word)`.
  `threadgroup_position_in_grid.x = problem * words + word`. 256 threads
  split each colour class: thread `t` updates nodes `t, t + 256, ...` of the
  class, then a `threadgroup_barrier`. Greedy colouring comes from
  `SelfFeedingTopology::colors`, the same buffers Gibbs binds at 16 to 18.
- **Threadgroup memory.** `state` is `N * 4` bytes of run-time sized memory at
  `[[threadgroup(0)]]`, set per dispatch with
  `set_threadgroup_memory_length(0, N * 4)`. Static arrays: the 8192-byte
  threshold row and 64 `uint` cut values. Total at the cap:
  `6016 * 4 + 8192 + 256 = 32,512 <= 32,768`.
- **Update.** For node `i` with own word `b`, neighbour words `s_j`, coupling
  sign `c_j = [J < 0]`: `l_j = c_j ^ b ^ s_j` is set on the replicas where
  bond `j` is satisfied. A field `h != 0` adds `l_h = [h < 0] ^ b`. A Harley
  and Seal carry-save tree counts the 21 inputs into six bit planes. With
  `d` the bond count including the field and `M` the rung's geometric draw
  at `row[(i + off) & 8191]`, the flip mask is `L <= (d + M) / 2`, or all ones
  when `(d + M) / 2 >= d`. `state[i] ^= mask`.
- **Row.** At every rung, `cut[m] = floor(exp(-2 beta m) * 2^32)` for `m` in
  `1..=63`, then 8192 draws of `M = max m : u < cut[m]` spread over the
  threadgroup. Per sweep, `off = splitmix32(seed, problem, word, rung, sweep)
  & 8191`.
- **Chunking.** `beta_start == 0` seeds the RNG from
  `(base_seed, threadgroup, thread)` and draws random words. Otherwise it
  reloads `persistent_state[threadgroup * N + i]` and
  `persistent_rng[(threadgroup * group_size + thread) * 4]`. Every chunk
  writes both back at the end.
- **Output.** Lane `r` of word `w` is read `w * 32 + r`, packed LSB-first
  into `final_samples[(problem * num_reads + read) * packed_size]`, spread
  over `(lane, byte)` pairs across the threadgroup. Reads at or past
  `num_reads` are skipped. No energies are written.
- **Buffer layout.** 0 to 15 as `bind_shared_args`. 16 to 18 colour blocks.
  19 `words`. 20 `num_colors`. 21 `beta_start`, 22 `beta_count`,
  23 `persistent_state` (`uint`), 24 `persistent_rng`. Slots 21 to 24 match
  Gibbs, so `bind_gibbs_chunk` is shared.

### Host

- **`Kernel` enum** (`Sa`, `Msa`, `Gibbs`) replaces `Algorithm` everywhere the
  crate selects a pipeline, a node cap, a chunk rate, or a threadgroup
  budget. `Algorithm` stays only as the re-exported protocol type. Tags become
  `KernelTag { const KERNEL: Kernel }` so `declared_stream_width` can answer
  per kernel with no device.
- **Sizing.** `words = simd_rounded_reads(num_reads) / 32`.
  `num_streams = num_problems * words` is the threadgroup count and the
  `buffer(12)` value. Samples and energies stay sized by
  `num_problems * num_reads`. Persistent: `num_streams * N * 4` bytes of
  state, `num_streams * 256 * 16` bytes of RNG.
- **Validation.** `MSA_MAX_NODES = 6016` (compile-time asserted against the
  32 KB cap), CSR degree at most 20 (`TooLarge`, so the coordinator routes
  the job elsewhere), and `N * 4 + static <= max_threadgroup_memory_length`
  on the opened device.
- **Chunk rate.** `MSA_OCCUPANCY_CURVE` records word-updates/s at each
  occupancy. It shares interpolation with the simulated annealing kernel.
  The curve scales toward zero below its first point. It stays flat for
  occupancies greater than its last point. The 2026-09-15 measurements used
  Apple M4 Max with 40 GPU cores and `tests/fixtures/advantage2-system1.edges`.
  This fixture has 4577 nodes, 41515 edges, and eight greedy classes.
  At T=1 and 7392 sweeps,
  calibration used 1, 2, 5, 10, and 40 jobs with 128 reads.
  It also used 1 and 2 jobs with 256 reads. The curve points are
  `(0.1, 2.6e8)`, `(0.2, 5.2e8)`, `(0.4, 1.0e9)`, `(0.5, 1.2e9)`, and
  `(1.0, 1.2e9)`. Each pair gives threadgroups per core and word-updates/s.
  Safety 0.7 produced a 580 ms chunk. Safety 0.4 reached 606 ms at 16384
  sweeps. The factor is 0.2 to cover the measured sweep range.
  Three verification rounds used T=1, 128 reads, and 7392 sweeps.
  Each round covered 1, 2, 5, 10, and 40 jobs.
  The largest chunk was 261 ms in a 40-job run.
- **Threadgroup budget.** `MSA_TG_PER_CORE` is 1.0. On the same machine and
  fixture, the 2026-09-15 runs used 80 jobs, 128 reads, and 7392 sweeps.
  Rates were 13.64, 14.23, 12.16, 11.57, and 10.85 jobs/s at T=1, 2, 4, 6,
  and 8. T=1 is the smallest within 10% of the peak at T=2.
  `batch_size_for_reads(Kernel::Msa, reads)` divides the budget by `words`.
- **Identity.** `METAL_MSA_IDENTITY`: backend `metal`, algorithm `msa`,
  `max_nodes = MSA_MAX_NODES`, features `streaming` and `governor`. Adapt
  envelope `4096..16384` sweeps, with `128` reads fixed at four words.
  On the same fixture and machine, the 2026-09-15 reference reached 1.02
  jobs/s at 2048 sweeps and 256 reads, using its production T=6.
  The multi-spin kernel used T=1, safety 0.2, 40 jobs, and 128 reads.
  It reached 34.75, 21.94, 12.86, and 7.27 jobs/s at 2048, 4096, 8192, and
  16384 sweeps. The largest tested count meets the reference, so
  `max_sweeps = 16384` and `min_sweeps = 16384 / 4 = 4096`.
  The envelope chunks peaked at 258 ms. `MAX_SWEEPS = 65536` still bounds
  every accepted job.
- **Binary.** `quip-metal-msa`, same CLI as `quip-metal-sa`.

### Precondition on inputs

The kernel reads `J < 0` as `-1`, `J != 0` as a bond, and `h < 0` as `-1`.
Consensus jobs carry `J` in `{-1, 1}` and `h` in `{-1, 0, 1}`, which the
protocol validates before the sampler. Other inputs still produce valid
`+-1` spins scored by `energy_milli`. They are not optimal for that graph.

### Diversity

One threshold `M` is shared by the 32 replicas of a word update, as the CPU
and CUDA kernels share one across 64. Replicas start independent and see
different `L`, so they separate. The CUDA MR measured a median energy within
one unit of the CPU solver. This port keeps that coupling.

## Deferred at port time, built since

The first port deferred the four items below. Each one now has code on this
branch. Two of them stay behind a switch, because measurement did not support a
new default.

- Lazy chunk commit for cancellation. In the production path in
  `src/streaming.rs`. Each batch commits one chunk at a time and checks
  cancellation before it encodes and commits each later chunk. The planner
  accounts for the two batches that overlap.
- Diagnostic compile switches. In tests only. `tests/diagnostics.rs` prepends
  `#define QUIP_MSA_DIAGNOSTICS` to the kernel source, which renames the entry
  point to `msa_anneal_diag` and adds a flip counter and energy parts. No
  production code compiles it.
- A `ulong` word variant with state in device memory. An isolated experiment in
  `kernels/msa64.metal` and `tests/msa64_experiment.rs`. At equal group counts
  the paired speed ratios ranged from 1.20 to 1.50, with a median of 1.44. The
  raw driver excludes the miner channel, the governor, and result scoring, so
  this is not a production speed claim. No file under `src/` refers to it.
- A Zephyr four-colouring. Opt-in behind `QUIP_METAL_MSA_FOUR_COLOR=1` at
  `src/sampler.rs:401`, and off by default while bead `quip-miner-metal-fjo`
  settles whether it becomes the default. Advantage2 System 1 has eight greedy
  classes, with sizes 856, 840, 827, 742, 679, 472, 146, and 15. The four
  classes hold 1148, 1145, 1145, and 1139 nodes. Two graph seeds gave no clear
  throughput result: seed 1 gained 4.18 percent in jobs per second, and seed 3
  took 8.44 percent longer. On seed 1 the largest logged chunk was 439 ms
  under greedy and 277 ms under four colours, so only the four-colour arm
  held the 400 ms bound of success criterion 5. Seed 3 has no logged chunk
  maxima. Mean lowest energy was worse under four colours by 1400 and 250
  milli on the two seeds, and no run separates that from noise. A failed
  edge check selects the greedy scheme.

## Success criteria

1. `cargo clippy --all-targets --all-features -- -D warnings` and
   `cargo fmt --check` are clean.
2. `cargo test` passes on Apple Silicon, including the new conformance,
   golden parity, chunk identity, and streaming tests.
3. `quip-metal-msa --capabilities` advertises `"algorithm":"msa"` and
   `--check` succeeds.
4. Use `tests/fixtures/advantage2-system1.edges`, with 4577 nodes, 41515
   edges, and eight greedy classes. Each read must contain only `+-1`
   spins, with energy equal to `energy_milli`.
   The multi-spin kernel at 128 reads must meet the simulated annealing
   reference rate at 2048 sweeps and 256 reads.
   Measurements on 2026-09-15 used Apple M4 Max with 40 GPU cores and 40 jobs.
   At T=1 and 16384 sweeps, multi-spin reached 7.27 jobs/s.
   The reference reached 1.02 jobs/s at its production T=6.
   With safety 0.2, three rounds at 7392 sweeps peaked at 261 ms.
   Each round covered 1, 2, 5, 10, and 40 jobs at 128 reads.
   The envelope measurements peaked at 258 ms. The README records these results.
5. The largest chunk stays at or below 400 ms at the tuned rate.
