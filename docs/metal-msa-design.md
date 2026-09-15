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
| Abort on cancel (`EXIT_NOW` peeked inside the sweep loop) | No | Metal cannot abort a committed command buffer. The Metal loop already drops cancelled jobs before commit and bounds a chunk to about 500 ms. A lazy chunk commit is a follow-up |
| Parallel host scoring (`QUIP_SCORE_THREADS`) | Already present | `harvest_batch` scores problems on a rayon pool |
| `QUIP_MSC_DIAG` compile switches | No | metal-rs 0.33 exposes no preprocessor defines. A follow-up can splice `#define` lines into the source string |

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
buffers between chunks. `chunk_plan` sizes chunks to land near 500 ms from a
measured update rate. CUDA has no watchdog on a compute-only device and runs a
whole model in one launch.

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
command buffer. `encode_batch` commits every chunk of a batch as it encodes,
so a cancelled batch runs to completion.

Consequence: out of scope for this port. Follow-up: commit one chunk at a
time behind a bounded in-flight window and check the token before each
commit, which bounds wasted work to about one chunk.

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
- **Chunk rate.** `MSA_WORD_UPDATES_PER_SEC` with the same 0.7 safety factor
  as SA. Initial value 0.2e9. **Exploratory**, replaced by measurement.
- **Threadgroup budget.** `MSA_TG_PER_CORE`, initial 2.0. **Exploratory**.
  `batch_size_for_reads(Kernel::Msa, reads)` divides the budget by `words`.
- **Identity.** `METAL_MSA_IDENTITY`: backend `metal`, algorithm `msa`,
  `max_nodes = MSA_MAX_NODES`, features `streaming` and `governor`. Adapt
  envelope initial `2048..8192` sweeps, `128` reads fixed (four words).
  **Exploratory**. `MAX_SWEEPS = 65536` still bounds every accepted job.
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

## Out of scope

- Lazy chunk commit for cancellation.
- Diagnostic compile switches.
- A `ulong` word variant with state in device memory, for a benchmark against
  the `uint` threadgroup design.
- A Zephyr-specific four-colouring. Greedy colouring gives four colours on
  Zephyr already.

## Success criteria

1. `cargo clippy --all-targets --all-features -- -D warnings` and
   `cargo fmt --check` are clean.
2. `cargo test` passes on Apple Silicon, including the new conformance,
   golden parity, chunk identity, and streaming tests.
3. `quip-metal-msa --capabilities` advertises `"algorithm":"msa"` and
   `--check` succeeds.
4. On a 4576-node, degree-20 bipartite graph at 128 reads, the multi-spin
   miner completes more jobs per second than `quip-metal-sa` at the same
   sweep count, with every read a valid `+-1` vector whose reported energy
   equals `energy_milli`. Numbers are recorded in the README.
5. The largest chunk stays under 500 ms at the tuned rate.
