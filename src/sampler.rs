//! Launch the v0.2 Metal SA / Gibbs kernels and score with consensus energy.
//!
//! Kernels are the original v0.2 Metal sources (`GPU/metal_kernels.metal` /
//! `GPU/metal_gibbs.metal`, copied verbatim into `kernels/`): int8-quantized
//! CSR, D-Wave incremental delta-energy SA / color-block Gibbs, bit-packed
//! thread-local state, one thread per read. Solution energies are always
//! scored on the host with
//! [`quip_solver_core::quip_protocol::scoring::energy_milli`] (f64
//! consensus). There is no GPU energy kernel (MSL has no `double`).
//!
//! # Batched dispatch (throughput)
//!
//! The kernel maps **one threadgroup per problem, one thread per read**
//! (`thread_id = threadgroup·num_reads + read`, `problem_id = thread_id /
//! num_reads`). A dispatch of `P` problems is
//! `dispatchThreadgroups(P, num_reads)` → `P` threadgroups occupy `P` GPU
//! cores, so a batch of ≈`gpu_cores` problems fills the whole GPU and hides
//! the kernel's thread-private memory latency. This mirrors v0.2
//! `metal_sa.py::_dispatch_batch`. Driving `P = 1` (one problem per command
//! buffer) leaves all but one core idle — a ~`gpu_cores`× slowdown — which is
//! why the streaming loop batches ([`crate::streaming`]).
//!
//! [`encode_batch`] builds one batch's buffers and encodes the dispatch without
//! committing; [`harvest_batch`] reads the bit-packed samples per problem. The
//! synchronous [`sample_ising`] runs a single-problem batch and waits.

use quip_solver_core::{IsingGraph, SampleParams, SamplerResult};
use thiserror::Error;

/// Which GPU kernel a binary drives.
///
/// Distinct from [`quip_solver_core::Algorithm`]: the coordinator protocol
/// only names `sa` and `gibbs`, and more than one kernel can serve `sa`. The
/// crate keys pipelines, node caps, chunk rates, and threadgroup budgets on
/// the kernel, not the protocol algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// Metropolis simulated annealing, one thread per read
    /// (`kernels/sa.metal`, `pure_simulated_annealing`).
    Sa,
    /// Multi-spin coded simulated annealing: 32 replicas per `uint` word, one
    /// threadgroup per (problem, word) (`kernels/msa.metal`, `msa_anneal`).
    Msa,
    /// Chromatic heat-bath Gibbs, one threadgroup per sample
    /// (`kernels/gibbs.metal`, `block_gibbs_parallel` / `block_gibbs_sampler`).
    Gibbs,
}

use crate::topology::{fill_h_j, SelfFeedingTopology};
use quip_solver_core::beta::{default_ising_beta_range, geometric_beta_schedule};
use quip_solver_core::quip_protocol::scoring::energy_milli;

/// Failure from a Metal sample attempt: capacity refusal or driver fault.
#[derive(Debug, Error)]
pub enum SampleError {
    /// Device open, kernel compile, or pipeline construction failed.
    #[error(transparent)]
    Metal(#[from] crate::metal_device::MetalError),
    /// Command buffer or buffer readback failed after a dispatch.
    #[error("Metal driver: {0}")]
    Driver(String),
    /// The job exceeds a fixed capacity of this backend (kernel node cap,
    /// [`MAX_SWEEPS`]). Distinct from [`Self::Driver`] because it maps to a
    /// different harness condition: `Capacity` tells the coordinator to route
    /// the job elsewhere, while [`Self::Driver`] and [`Self::Metal`] describe
    /// the device itself, not this one job's size.
    #[error("job exceeds Metal backend capacity: {0}")]
    TooLarge(String),
}

impl SampleError {
    /// Map to the harness's device-condition report
    /// ([`quip_solver_core::SampleError`], "Gotchas" in `MIGRATING.md`): a
    /// capacity refusal maps to `Capacity`. Everything else here — a kernel
    /// compile failure, a device reset, a GPU watchdog timeout, or an
    /// internal invariant violation — is a state this backend will not
    /// recover from without a restart, so it maps to `DeviceFault` rather
    /// than `DeviceBusy`. Reporting these as transient load (`OVERLOADED`
    /// pre-migration) is exactly the "wedged GPU forever" failure mode
    /// `MIGRATING.md` warns against: `DeviceBusy` invites the coordinator to
    /// keep sending jobs a broken device can never serve, where `DeviceFault`
    /// ends the session for a supervisor restart. Arms are listed explicitly
    /// so a new variant forces a decision rather than silently inheriting one.
    ///
    /// # Examples
    ///
    /// ```
    /// use quip_miner_metal::sampler::SampleError;
    /// use quip_solver_core::SampleError as HarnessSampleError;
    ///
    /// let err = SampleError::TooLarge("nodes > SA cap".into());
    /// assert_eq!(err.to_sample_error(), HarnessSampleError::Capacity);
    ///
    /// let err = SampleError::Driver("command buffer failed".into());
    /// assert!(matches!(err.to_sample_error(), HarnessSampleError::DeviceFault(_)));
    /// ```
    pub fn to_sample_error(&self) -> quip_solver_core::SampleError {
        match self {
            Self::TooLarge(_) => quip_solver_core::SampleError::Capacity,
            Self::Driver(msg) => quip_solver_core::SampleError::DeviceFault(msg.clone()),
            Self::Metal(e) => quip_solver_core::SampleError::DeviceFault(e.to_string()),
        }
    }
}

/// Largest `num_reads` a dispatch allocates for (mirrors CUDA). Also the
/// per-threadgroup thread count, well under `maxTotalThreadsPerThreadgroup`.
pub(crate) const MAX_READS: usize = 256;

/// Apple GPU SIMD width (threads per simdgroup).
///
/// `num_reads` becomes the thread count of a threadgroup on the SA and
/// sequential-Gibbs kernels, so a count that is not a multiple of this leaves
/// lanes of the final simdgroup permanently idle — they are issued either way.
/// Measured at a constant total sample count, `reads = 16` (half a simdgroup)
/// costs ~20% against `reads = 64`.
pub(crate) const SIMD_WIDTH: usize = 32;

/// Round a job's `num_reads` up to a full simdgroup, capped at [`MAX_READS`].
///
/// The dispatch computes the rounded count; the extra samples are discarded
/// when the result is assembled — `streaming::finish_batch` on the batched path
/// and [`sample_ising`] on the synchronous one both truncate to the count the
/// job asked for — so this never changes what the coordinator sees.
/// It costs nothing: those lanes execute regardless of whether we use them.
/// `MAX_READS` is itself a multiple of `SIMD_WIDTH`, so the cap cannot round
/// back down below the request.
pub(crate) fn simd_rounded_reads(num_reads: usize) -> usize {
    num_reads
        .clamp(1, MAX_READS)
        .div_ceil(SIMD_WIDTH)
        .saturating_mul(SIMD_WIDTH)
        .min(MAX_READS)
}

/// SA kernel `N` cap: `thread int8_t delta_energy[4593]` in `kernels/sa.metal`.
///
/// `crate::METAL_SA_IDENTITY` advertises this same cap, so the identity const
/// and the kernel array have one source.
pub(crate) const SA_MAX_NODES: usize = 4593;
/// Gibbs kernel `N` cap: `thread int8_t packed_state[600]` (600*8) in
/// `kernels/gibbs.metal`.
///
/// Single source of truth with the identity const, as for [`SA_MAX_NODES`].
pub(crate) const GIBBS_MAX_NODES: usize = 4800;

/// Replicas per 32-bit word in `kernels/msa.metal`.
///
/// Equal to [`SIMD_WIDTH`], so [`simd_rounded_reads`] always yields whole
/// words and `words = num_reads / MSA_LANES` is exact.
pub(crate) const MSA_LANES: usize = 32;
const _: () = assert!(
    MSA_LANES == SIMD_WIDTH,
    "simd_rounded_reads must produce whole 32-lane multi-spin words"
);
/// Threads per multi-spin threadgroup: they split each colour class's nodes.
/// Also the per-threadgroup RNG stream count the persistent buffer is sized by.
const MSA_THREADS: usize = 256;
/// Static threadgroup bytes `msa_anneal` declares: an 8192-byte threshold row
/// plus 64 `uint` cut values. `msa_pipeline_compiles_and_admits_256_threads`
/// in `metal_device.rs` pins the compiled figure to this constant.
const MSA_STATIC_TG_BYTES: usize = 8192 + 64 * 4;
/// Threadgroup memory every Apple GPU family offers per threadgroup, bytes.
/// There is no opt-in above it (CUDA's `MAX_DYNAMIC_SHARED_SIZE_BYTES` has no
/// counterpart), which is why the kernel uses 32-bit words: one `u64` per
/// spin does not fit Advantage2's 4577 spins.
const APPLE_TG_MEMORY_BYTES: usize = 32 * 1024;
/// Multi-spin kernel `N` cap: `N * 4` bytes of spin words must fit beside the
/// static arrays under [`APPLE_TG_MEMORY_BYTES`] (6016 * 4 + 8448 = 32,512).
///
/// `crate::METAL_MSA_IDENTITY` advertises this same cap, so the identity
/// const and the dispatch guard have one source.
pub(crate) const MSA_MAX_NODES: usize = 6016;
const _: () = assert!(
    MSA_MAX_NODES * 4 + MSA_STATIC_TG_BYTES <= APPLE_TG_MEMORY_BYTES,
    "MSA_MAX_NODES spin words must fit in Apple threadgroup memory"
);
/// Largest CSR degree the kernel's unrolled 20-neighbour prefetch admits.
/// Zephyr (Advantage2) is degree 20. A nonzero field is a 21st input, which
/// the kernel's `popcount21` covers.
pub(crate) const MSA_MAX_DEG: usize = 20;

/// Largest `num_sweeps` a dispatch accepts.
///
/// `num_sweeps` arrives from the coordinator and nothing upstream bounds it:
/// `quip_solver_core`'s `pick_param` returns the job's value verbatim when
/// non-zero, and unlike `num_reads` (rejected `TooLarge` against `max_reads`)
/// there is no identity-const gate for it. Unbounded, it sizes the beta
/// schedule — `geometric_beta_schedule` collects `num_sweeps / sweeps_per_beta`
/// `f64`s, so `u32::MAX` sweeps is a ~34 GB allocation — and then drives that
/// many kernel sweeps, i.e. both an OOM and a GPU-watchdog denial of service.
///
/// 65536 is 32x the `max_sweeps: 2048` in `METAL_ADAPT` (`lib.rs`).
/// `quip-solver-core` doubles the resolved sweeps for Gibbs
/// (`GIBBS_SWEEP_MULTIPLIER`), so the largest legitimate adapt-driven job
/// reaching here is 4096 — 16x of headroom — while bounding the schedule to
/// 64Ki `f64` + 64Ki `f32` (~768 KiB). Raise this only together with
/// `METAL_ADAPT.max_sweeps`; it must stay >= `2 * METAL_ADAPT.max_sweeps`,
/// which a `const _: () = assert!(..)` in `lib.rs` enforces at compile time.
pub(crate) const MAX_SWEEPS: usize = 65_536;

/// Target GPU time for one command buffer, in milliseconds.
///
/// macOS runs a non-configurable GPU watchdog that aborts a command buffer
/// running more than a few seconds, and a hang freezes the whole machine for
/// ~10 s while the GPU resets. Apple's guidance is to split long compute into
/// tiles well under ~100-250 ms; MLX measured watchdog kills at ~1.2 s per
/// operation with a display attached. Before chunking, this backend ran
/// 23-33 s command buffers at mining settings (82 s at max effort) and did
/// crash the host.
///
/// 500 ms is above Apple's conservative ~100-250 ms tiling guidance, trading
/// margin for efficiency: chunk overhead is ~2.5 ms of dispatch setup per
/// command buffer, so a longer chunk amortizes it to well under 1%. It still
/// keeps ~2x margin under the ~1.2 s at which watchdog kills were actually
/// observed, and is ~50-160x shorter than the 23-82 s this backend ran before
/// chunking — the regime that crashed the host.
///
/// This is a *ceiling*, not a goal: [`estimated_updates_per_sec`] is
/// deliberately pessimistic, so real chunks usually land well under it. Lower
/// this if a machine still stutters; the cost is throughput, not correctness.
const TARGET_DISPATCH_MS: f64 = 500.0;

/// Measured SA spin-update rate as a function of occupancy, in Gupd/s.
///
/// A single constant cannot size chunks correctly: throughput varies ~6x with
/// how full the GPU is, so an estimate tuned for a full batch produces
/// dangerously long chunks for a small one. (Measured: a 20-problem SA batch
/// runs at 0.65 Gupd/s where a 320-problem batch reaches 2.10.)
///
/// Points are `(threadgroups per core, Gupd/s)` from the occupancy sweep on an
/// M4 Max, full Advantage2 topology. SA only — chromatic Gibbs has a different
/// shape and gets its own estimate ([`GIBBS_UPDATES_PER_SEC`]).
const OCCUPANCY_CURVE: [(f64, f64); 8] = [
    (0.2, 0.32),
    (0.5, 0.49),
    (1.0, 0.83),
    (2.0, 1.40),
    (3.0, 1.54),
    (4.0, 1.83),
    (6.0, 2.03),
    (8.0, 2.10),
];

/// Margin applied to [`OCCUPANCY_CURVE`]. Overestimating throughput produces
/// chunks that run *longer* than the target — the failure mode that crashes the
/// host — so the estimate is deliberately pessimistic. Underestimating only
/// costs a few extra command buffers at ~2.5 ms each.
///
/// Measured at 0.7: SA adapt chunks peaked at 360 ms against a 500 ms ceiling.
const SA_THROUGHPUT_SAFETY: f64 = 0.7;

/// Chromatic Gibbs spin-update rate, flat rather than a curve.
///
/// Gibbs is saturated across the whole occupancy range we run it at (2.06-2.12
/// Gupd/s from 16 to 512 threadgroups/core), so occupancy is not the variable
/// that predicts it — the sampling shape is. The same kernel measured 2.21
/// Gupd/s at 64 reads / 256 sweeps but only 1.41 at 160 reads / 2308 sweeps.
/// This is the low end of that range, so chunk sizing is bounded by the
/// slowest configuration rather than the average.
///
/// Kept separate from SA's curve deliberately: sharing one estimate meant
/// tightening it for Gibbs also shrank SA's chunks, which were already well
/// inside the ceiling, and paid for it in extra command buffers.
const GIBBS_UPDATES_PER_SEC: f64 = 1.2e9;

/// See [`SA_THROUGHPUT_SAFETY`]. Gibbs carries its own margin because its
/// per-chunk cost includes work SA's does not — rebuilding the threadgroup spin
/// array from device memory on resume, and recomputing energies every chunk —
/// which the aggregate rate above does not separate out.
const GIBBS_THROUGHPUT_SAFETY: f64 = 0.7;

/// Multi-spin word-update rate, word-updates/s. One word update advances 32
/// replicas of one spin: 21 threadgroup loads, a 21-input carry-save tree,
/// and one store.
///
/// Not yet measured. 0.2e9 is a tenth of SA's saturated spin-update rate,
/// chosen pessimistic so the first chunks cannot approach the watchdog. The
/// benchmark in `tests/msa_bench.rs` replaces it with a measured value.
const MSA_WORD_UPDATES_PER_SEC: f64 = 0.2e9;
/// See [`SA_THROUGHPUT_SAFETY`].
const MSA_THROUGHPUT_SAFETY: f64 = 0.7;

/// Expected updates per second for a dispatch of `groups` threadgroups: spin
/// updates for SA and Gibbs, word updates (32 replicas each) for the
/// multi-spin kernel.
///
/// Below the first measured point the curve is extrapolated linearly toward the
/// origin (a nearly-empty GPU really is proportionally slow); above the last it
/// is held flat, since throughput has saturated by then.
fn estimated_updates_per_sec(kernel: Kernel, groups: usize) -> f64 {
    if kernel == Kernel::Msa {
        return MSA_WORD_UPDATES_PER_SEC * MSA_THROUGHPUT_SAFETY;
    }
    if kernel == Kernel::Gibbs {
        return GIBBS_UPDATES_PER_SEC * GIBBS_THROUGHPUT_SAFETY;
    }
    let cores = crate::iokit_gov::gpu_core_count().unwrap_or(10).max(1);
    let x = groups as f64 / cores as f64;
    let (first_x, first_y) = OCCUPANCY_CURVE[0];
    let (last_x, last_y) = OCCUPANCY_CURVE[OCCUPANCY_CURVE.len() - 1];
    let gups = if x <= first_x {
        // Straight line through the origin, so a tiny batch is never credited
        // with more throughput than it can reach.
        first_y * (x / first_x).max(0.05)
    } else if x >= last_x {
        last_y
    } else {
        let hi = OCCUPANCY_CURVE
            .iter()
            .position(|&(px, _)| px >= x)
            .unwrap_or(OCCUPANCY_CURVE.len() - 1);
        let (x0, y0) = OCCUPANCY_CURVE[hi - 1];
        let (x1, y1) = OCCUPANCY_CURVE[hi];
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    };
    gups * 1e9 * SA_THROUGHPUT_SAFETY
}

/// Split a beta schedule into chunks whose dispatches each land near
/// [`TARGET_DISPATCH_MS`].
///
/// Returns `(beta_start, beta_count)` pairs covering `0..num_betas`. One sweep
/// touches every spin of every sample, so a chunk's cost is
/// `threads * sweeps_per_beta * betas * N` spin updates.
///
/// Both kernels can resume mid-schedule: `beta_start == 0` initializes, any
/// other value restores the carry-over state the previous chunk wrote.
fn chunk_plan(kernel: Kernel, dims: &BatchDims, groups: usize) -> Vec<(i32, i32)> {
    let num_betas = dims.num_betas.max(1);
    let per_beta =
        (dims.num_threads as f64) * (dims.sweeps_per.max(1) as f64) * (dims.n.max(1) as f64);
    let budget = TARGET_DISPATCH_MS / 1000.0 * estimated_updates_per_sec(kernel, groups);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "clamped to 1..=num_betas immediately below"
    )]
    let betas_per_chunk =
        ((budget / per_beta.max(1.0)).floor() as i64).clamp(1, i64::from(num_betas)) as i32;

    let mut plan = Vec::new();
    let mut start = 0;
    while start < num_betas {
        let count = betas_per_chunk.min(num_betas - start);
        plan.push((start, count));
        start += count;
    }
    plan
}

/// Whether Gibbs uses the chromatic (node-parallel) kernel — the default.
///
/// `block_gibbs_parallel` puts one threadgroup per *sample* and lets its
/// threads split each color's nodes over `threadgroup`-shared state, exploiting
/// the chromatic independence of a sweep. `block_gibbs_sampler` instead runs
/// one thread per read and walks all N nodes serially. Measured on an M4 Max
/// over the full Advantage2 topology (4577 nodes): 36.5 vs 13.2 jobs/s — 2.75x
/// — at equal solution quality (mean best energy within 0.01%, same diversity).
///
/// v0.2 shipped the sequential kernel as its default (`parallel=False`), so it
/// has more field exposure; `QUIP_METAL_GIBBS_SEQUENTIAL=1` forces it as an
/// escape hatch. SA is unaffected — its incremental delta-energy chain is
/// inherently serial, so it has no node-parallel variant.
pub(crate) fn gibbs_node_parallel() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("QUIP_METAL_GIBBS_SEQUENTIAL").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// Largest `N` the kernel's fixed-size arrays admit.
pub(crate) fn kernel_max_nodes(kernel: Kernel) -> usize {
    match kernel {
        Kernel::Sa => SA_MAX_NODES,
        Kernel::Msa => MSA_MAX_NODES,
        Kernel::Gibbs => GIBBS_MAX_NODES,
    }
}

fn score_spins(spins: &[i8], graph: &IsingGraph) -> SamplerResult {
    let energy = energy_milli(spins, &graph.h, &graph.j, &graph.edges);
    SamplerResult {
        spins: spins.to_vec(),
        energy_milli: energy,
    }
}

/// Geometric beta schedule cast to f32 for kernel upload, plus sweeps-per-beta.
///
/// Uses the shared f64 schedule and casts each element to f32 — bit-identical
/// to the prior in-crate f32 schedule.
fn build_beta_schedule(
    graph: &IsingGraph,
    num_sweeps: usize,
    sweeps_per_beta: usize,
    beta_range: Option<(f64, f64)>,
) -> (Vec<f32>, usize) {
    let sweeps_per = sweeps_per_beta.max(1);
    let num_betas = (num_sweeps / sweeps_per).max(1);
    let (hot, cold) = beta_range.unwrap_or_else(|| default_ising_beta_range(graph));
    let sched: Vec<f32> = geometric_beta_schedule(hot, cold, num_betas)
        .iter()
        .map(|&b| b as f32)
        .collect();
    (sched, sweeps_per)
}

/// Unpack one read's bit-packed spins (LSB-first per byte, bit=1 -> -1,
/// bit=0 -> +1; matches the kernel's `set_spin_packed`). Identical to the
/// CUDA crate's unpacker — both kernels share the v0.2 packing convention.
///
/// `packed` is expected to hold `n.div_ceil(8)` bytes. A short slice reads as
/// zero past its end (i.e. `+1`, the initialized value) instead of panicking:
/// the caller sizes both the allocation and the harvest count from the same
/// `packed_size`, so a mismatch is a bug, not a reason to abort the miner.
fn unpack_spins(packed: &[i8], n: usize) -> Vec<i8> {
    let mut spins = vec![1i8; n];
    for (i, s) in spins.iter_mut().enumerate() {
        let byte = packed.get(i >> 3).copied().unwrap_or(0) as u8;
        let bit = (byte >> (i & 7)) & 1;
        *s = if bit == 1 { -1 } else { 1 };
    }
    spins
}

/// One encoded-but-uncommitted batch of `num_problems` problems sharing a
/// topology, plus the metadata and device buffers needed to harvest it.
/// Input/scratch buffers are held in `_keep` so they outlive the GPU execution.
pub(crate) struct EncodedBatch {
    /// Command buffers in submission order. A long anneal is split across
    /// several so no single one trips the macOS GPU watchdog; the last carries
    /// the final state. See [`TARGET_DISPATCH_MS`].
    pub(crate) cmds: Vec<metal::CommandBuffer>,
    d_samples: metal::Buffer,
    n: usize,
    num_reads: usize,
    num_problems: usize,
    packed_size: usize,
    _keep: Vec<metal::Buffer>,
}

impl EncodedBatch {
    /// Block until the last chunk retires. Earlier chunks are ordered ahead of
    /// it on the same queue, so waiting on the tail waits for all of them.
    pub(crate) fn wait_until_completed(&self) {
        if let Some(last) = self.cmds.last() {
            last.wait_until_completed();
        }
    }

    /// The first chunk that did not complete, if any. A mid-sequence failure
    /// matters as much as the last one: a watchdog abort or fault partway
    /// through leaves the persistent state garbage, so the whole batch's
    /// samples are suspect.
    pub(crate) fn failed_status(&self) -> Option<metal::MTLCommandBufferStatus> {
        self.cmds
            .iter()
            .map(|c| c.status())
            .find(|s| *s != metal::MTLCommandBufferStatus::Completed)
    }

    /// Total GPU execution time across chunks, microseconds.
    pub(crate) fn gpu_time_us(&self) -> u64 {
        self.cmds.iter().map(|c| gpu_time_us(c)).sum()
    }

    /// Longest single chunk, microseconds — the quantity the macOS GPU watchdog
    /// actually judges. The sum above says how much work ran; this says whether
    /// any one command buffer got close to the abort threshold.
    pub(crate) fn max_chunk_us(&self) -> u64 {
        self.cmds.iter().map(|c| gpu_time_us(c)).max().unwrap_or(0)
    }

    /// Number of command buffers this batch was split into.
    pub(crate) fn chunk_count(&self) -> usize {
        self.cmds.len()
    }
}

/// True GPU execution time of a completed command buffer, in microseconds,
/// from `GPUEndTime - GPUStartTime` (`CFTimeInterval` seconds). metal-rs 0.33
/// exposes no accessor, so read the properties via `objc`. Returns 0 if the
/// timestamps are unavailable / non-positive.
#[expect(
    unexpected_cfgs,
    reason = "objc 0.2 msg_send! expands to cfg(cargo-clippy) the compiler no longer recognizes"
)]
fn gpu_time_us(cmd: &metal::CommandBufferRef) -> u64 {
    use objc::{msg_send, sel, sel_impl};
    // SAFETY: `GPUStartTime`/`GPUEndTime` are `CFTimeInterval` (f64) properties
    // on a completed `MTLCommandBuffer`; `cmd` implements `objc::Message`.
    let (start, end): (f64, f64) =
        unsafe { (msg_send![cmd, GPUStartTime], msg_send![cmd, GPUEndTime]) };
    let dur = end - start;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "guarded finite and positive; GPU spans are far below u64::MAX microseconds"
    )]
    if dur.is_finite() && dur > 0.0 {
        (dur * 1_000_000.0) as u64
    } else {
        0
    }
}

fn set_bytes_i32(enc: &metal::ComputeCommandEncoderRef, index: u64, val: i32) {
    enc.set_bytes(
        index as metal::NSUInteger,
        4,
        &val as *const i32 as *const std::ffi::c_void,
    );
}

fn set_bytes_u32(enc: &metal::ComputeCommandEncoderRef, index: u64, val: u32) {
    enc.set_bytes(
        index as metal::NSUInteger,
        4,
        &val as *const u32 as *const std::ffi::c_void,
    );
}

/// Tile a slice `times` times into one contiguous `Vec`.
fn tile_i32(src: &[i32], times: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(src.len() * times);
    for _ in 0..times {
        out.extend_from_slice(src);
    }
    out
}

/// A 1-D `MTLSize` — every dispatch here is one-dimensional.
fn mtl_size_1d(width: usize) -> metal::MTLSize {
    metal::MTLSize {
        width: width as metal::NSUInteger,
        height: 1,
        depth: 1,
    }
}

/// Scalar kernel arguments plus the sizes the algorithm-specific setup needs.
struct BatchDims {
    n: usize,
    num_betas: i32,
    sweeps_per: usize,
    base_seed: u32,
    num_threads: usize,
    num_problems: usize,
    num_reads: usize,
    packed_size: usize,
}

/// Per-batch input buffers: shared CSR structure plus per-problem `J` / `h`.
struct InputBuffers {
    row: metal::Buffer,
    col: metal::Buffer,
    j: metal::Buffer,
    h: metal::Buffer,
    row_off: metal::Buffer,
    col_off: metal::Buffer,
}

impl InputBuffers {
    /// Consume into an `EncodedBatch::_keep` list. These are bound to the
    /// encoder but never read back on the host; they only have to outlive the
    /// GPU execution.
    fn into_keep(self) -> Vec<metal::Buffer> {
        vec![
            self.row,
            self.col,
            self.j,
            self.h,
            self.row_off,
            self.col_off,
        ]
    }
}

/// Beta ladder plus the kernel's two output buffers.
struct DispatchBuffers {
    beta: metal::Buffer,
    samples: metal::Buffer,
    energies: metal::Buffer,
}

/// Largest CSR row length `SelfFeedingTopology::build` would produce for
/// `graph`: an out-of-range endpoint skips the edge and a self-loop counts
/// once, the same rules the builder applies.
fn max_csr_degree(graph: &IsingGraph) -> usize {
    let n = graph.num_nodes();
    let mut degree = vec![0usize; n];
    for &(u, v) in &graph.edges {
        if u >= n || v >= n {
            continue;
        }
        degree[u] += 1;
        if u != v {
            degree[v] += 1;
        }
    }
    degree.into_iter().max().unwrap_or(0)
}

/// Validate a batch's shared inputs, returning the establishing graph and `N`.
///
/// Split out of [`encode_batch`] so both rejections are reachable without a
/// Metal device.
fn validate_batch<'a>(
    graphs: &[&'a IsingGraph],
    params: &SampleParams,
    kernel: Kernel,
) -> Result<(&'a IsingGraph, usize), SampleError> {
    let Some(&first) = graphs.first() else {
        // Was a `debug_assert!`, which is compiled out in release — an empty
        // batch would have indexed `graphs[0]` and panicked the miner.
        return Err(SampleError::Driver(
            "encode_batch needs at least one graph".into(),
        ));
    };
    let n = first.num_nodes();
    let cap = kernel_max_nodes(kernel);
    if n > cap {
        // Defense in depth: the harness rejects N > max_nodes (identity const)
        // before the sampler; this catches drift before it overruns the
        // kernel's fixed-size thread-local arrays.
        return Err(SampleError::TooLarge(format!(
            "graph N={n} exceeds {kernel:?} kernel limit {cap}"
        )));
    }
    if kernel == Kernel::Msa {
        // The kernel unrolls a fixed 20-neighbour prefetch; a denser node
        // would read past it. Capacity, not a driver fault: the coordinator
        // routes the job to a backend that walks CSR rows of any length.
        let max_deg = max_csr_degree(first);
        if max_deg > MSA_MAX_DEG {
            return Err(SampleError::TooLarge(format!(
                "graph max degree {max_deg} exceeds the multi-spin kernel's \
                 {MSA_MAX_DEG}-neighbour budget"
            )));
        }
    }
    if params.num_sweeps > MAX_SWEEPS {
        // No defense in depth here: nothing upstream bounds `num_sweeps` at
        // all (see `MAX_SWEEPS`), so this is the only gate between a hostile
        // coordinator and an unbounded beta-schedule allocation.
        return Err(SampleError::TooLarge(format!(
            "num_sweeps={} exceeds Metal sampler limit {MAX_SWEEPS}",
            params.num_sweeps
        )));
    }
    Ok((first, n))
}

/// Build and upload one batch's CSR structure and per-problem `h` / `J`.
///
/// The CSR structure is shared by every problem in the batch (same topology),
/// so it is tiled `num_problems` times; the offset arrays give the kernel each
/// problem's slice.
fn upload_inputs(
    device: &crate::metal_device::MetalDevice,
    topo: &SelfFeedingTopology,
    graphs: &[&IsingGraph],
) -> InputBuffers {
    let num_problems = graphs.len();
    let n = topo.n;
    let nnz_alloc = topo.nnz.max(1);
    let rp_len = topo.row_ptr.len().max(1);

    // Shared CSR structure, tiled per problem; per-problem J / h values.
    let zero = [0i32];
    let base_row: &[i32] = if topo.row_ptr.is_empty() {
        &zero
    } else {
        &topo.row_ptr
    };
    let base_col: &[i32] = if topo.nnz == 0 { &zero } else { &topo.col_ind };
    let all_row_ptr = tile_i32(base_row, num_problems);
    let all_col_ind = tile_i32(base_col, num_problems);
    let mut all_j = vec![0i8; num_problems * nnz_alloc];
    let mut all_h = vec![0i8; num_problems * n];
    for (p, graph) in graphs.iter().enumerate() {
        let (j_csr, h_i8) = fill_h_j(topo, graph);
        // `j_csr` has length `topo.nnz`; pad region is the trailing slot when
        // nnz == 0. `h_i8` has length N.
        all_j[p * nnz_alloc..p * nnz_alloc + j_csr.len()].copy_from_slice(&j_csr);
        all_h[p * n..p * n + h_i8.len()].copy_from_slice(&h_i8);
    }
    let row_ptr_offsets: Vec<i32> = (0..=num_problems).map(|p| (p * rp_len) as i32).collect();
    let col_ind_offsets: Vec<i32> = (0..=num_problems).map(|p| (p * nnz_alloc) as i32).collect();

    InputBuffers {
        row: device.new_buffer_from_slice(&all_row_ptr),
        col: device.new_buffer_from_slice(&all_col_ind),
        j: device.new_buffer_from_slice(&all_j),
        h: device.new_buffer_from_slice(&all_h),
        row_off: device.new_buffer_from_slice(&row_ptr_offsets),
        col_off: device.new_buffer_from_slice(&col_ind_offsets),
    }
}

/// Bind the shared buffer layout (indices 0..15) — identical in both kernels.
fn bind_shared_args(
    enc: &metal::ComputeCommandEncoderRef,
    inputs: &InputBuffers,
    out: &DispatchBuffers,
    dims: &BatchDims,
) {
    enc.set_buffer(0, Some(&inputs.row), 0);
    enc.set_buffer(1, Some(&inputs.col), 0);
    enc.set_buffer(2, Some(&inputs.j), 0);
    enc.set_buffer(3, Some(&inputs.row_off), 0);
    enc.set_buffer(4, Some(&inputs.col_off), 0);
    set_bytes_i32(enc, 5, dims.n as i32);
    set_bytes_i32(enc, 6, dims.num_betas);
    set_bytes_i32(enc, 7, dims.sweeps_per as i32);
    set_bytes_u32(enc, 8, dims.base_seed);
    enc.set_buffer(9, Some(&out.beta), 0);
    enc.set_buffer(10, Some(&out.samples), 0);
    enc.set_buffer(11, Some(&out.energies), 0);
    set_bytes_i32(enc, 12, dims.num_threads as i32);
    set_bytes_i32(enc, 13, dims.num_problems as i32);
    set_bytes_i32(enc, 14, dims.num_reads as i32);
    enc.set_buffer(15, Some(&inputs.h), 0);
}

/// SA-only chunked-dispatch persistent buffers (indices 16..21).
///
/// Single-shot run: beta_start = 0, beta_count = num_betas, so these are
/// written but never re-read — allocated as zeroed scratch.
fn new_sa_persistent(
    device: &crate::metal_device::MetalDevice,
    dims: &BatchDims,
) -> [metal::Buffer; 4] {
    [
        device.new_zeroed_buffer((dims.num_threads * dims.packed_size) as u64),
        device.new_zeroed_buffer((dims.num_threads * dims.n.max(1)) as u64),
        device.new_zeroed_buffer((dims.num_threads * 4) as u64), // uint rng
        device.new_zeroed_buffer((dims.num_threads * 4) as u64), // int energy
    ]
}

/// Gibbs carry-over state: spins plus the RNG stream, allocated per anneal.
///
/// Sizes follow the two kernels' differing layouts. The chromatic kernel keeps
/// one *sample* per threadgroup and stores spins **unpacked** (one byte each,
/// mirroring its threadgroup array) to avoid the bit-packing races its own
/// comments warn about; its RNG stream is per *thread*, so the buffer scales
/// with the threadgroup width. The sequential kernel is one thread per read,
/// with bit-packed spins and one stream per thread.
///
/// Unlike SA, Gibbs persists no `delta_energy` or running energy: it recomputes
/// the effective field from the current spins on every node update, and its
/// energy is a pure function of the spins it already carries.
fn new_gibbs_persistent(
    device: &crate::metal_device::MetalDevice,
    dims: &BatchDims,
    node_parallel: bool,
    threads_per_group: usize,
) -> [metal::Buffer; 2] {
    // 4 x u32 of xoshiro128** state per stream.
    const RNG_BYTES_PER_STREAM: usize = 16;
    if node_parallel {
        [
            device.new_zeroed_buffer((dims.num_threads * dims.n.max(1)) as u64),
            device.new_zeroed_buffer(
                (dims.num_threads * threads_per_group.max(1) * RNG_BYTES_PER_STREAM) as u64,
            ),
        ]
    } else {
        [
            device.new_zeroed_buffer((dims.num_threads * dims.packed_size) as u64),
            device.new_zeroed_buffer((dims.num_threads * RNG_BYTES_PER_STREAM) as u64),
        ]
    }
}

/// Multi-spin carry-over state: one `uint` word per (threadgroup, spin) plus
/// one xoshiro128** stream (four `uint`) per (threadgroup, thread). The
/// threshold row is rebuilt at every rung and the sweep offsets are hashed
/// from their coordinates, so nothing else survives a chunk boundary.
fn new_msa_persistent(
    device: &crate::metal_device::MetalDevice,
    num_streams: usize,
    n: usize,
    threads_per_group: usize,
) -> [metal::Buffer; 2] {
    const RNG_BYTES_PER_STREAM: usize = 16;
    [
        device.new_zeroed_buffer((num_streams * n.max(1) * 4) as u64),
        device.new_zeroed_buffer(
            (num_streams * threads_per_group.max(1) * RNG_BYTES_PER_STREAM) as u64,
        ),
    ]
}

/// Per-kernel encode state: the carry-over buffers every chunk binds, plus
/// what the colour-block kernels need at bind time. Replaces the pair of
/// `Option`s the loop used to match on.
enum KernelEncode {
    Sa {
        persist: [metal::Buffer; 4],
    },
    Gibbs {
        persist: [metal::Buffer; 2],
    },
    Msa {
        persist: [metal::Buffer; 2],
        /// Words per problem, bound at slot 19.
        words: i32,
        /// Bytes of `threadgroup uint state[N]` at threadgroup index 0,
        /// rounded up to Metal's 16-byte granularity.
        state_bytes: u64,
    },
}

impl KernelEncode {
    fn into_keep(self) -> Vec<metal::Buffer> {
        match self {
            Self::Sa { persist } => persist.into(),
            Self::Gibbs { persist } | Self::Msa { persist, .. } => persist.into(),
        }
    }
}

/// Bind the chunk window and carry-over state of a colour-block kernel (Gibbs
/// or multi-spin). Indices start at 21 because the colour-block bindings
/// occupy 16..20.
fn bind_color_kernel_chunk(
    enc: &metal::ComputeCommandEncoderRef,
    persist: &[metal::Buffer; 2],
    beta_start: i32,
    beta_count: i32,
) {
    set_bytes_i32(enc, 21, beta_start);
    set_bytes_i32(enc, 22, beta_count);
    enc.set_buffer(23, Some(&persist[0]), 0);
    enc.set_buffer(24, Some(&persist[1]), 0);
}

/// Bind SA's chunk window and its carry-over state.
///
/// `beta_start == 0` tells the kernel to initialize; any other value makes it
/// resume from the persistent buffers, which it rewrites at the end of every
/// chunk. The buffers are shared across a batch's chunks — that carry-over is
/// what lets one anneal span several command buffers.
fn bind_sa_chunk(
    enc: &metal::ComputeCommandEncoderRef,
    persist: &[metal::Buffer; 4],
    beta_start: i32,
    beta_count: i32,
) {
    set_bytes_i32(enc, 16, beta_start);
    set_bytes_i32(enc, 17, beta_count);
    enc.set_buffer(18, Some(&persist[0]), 0);
    enc.set_buffer(19, Some(&persist[1]), 0);
    enc.set_buffer(20, Some(&persist[2]), 0);
    enc.set_buffer(21, Some(&persist[3]), 0);
}

/// Colour-block buffers (indices 16..18) and the two scalars after them, shared
/// by the Gibbs and multi-spin kernels.
///
/// Colour blocks are shared across the batch (same topology → same coloring);
/// the kernel indexes them globally, not per problem. `slot19` is
/// `update_mode` for Gibbs (0 = heat-bath) and `words` for the multi-spin
/// kernel; slot 20 is `num_colors` for both.
fn encode_color_buffers(
    device: &crate::metal_device::MetalDevice,
    enc: &metal::ComputeCommandEncoderRef,
    topo: &SelfFeedingTopology,
    keep: &mut Vec<metal::Buffer>,
    slot19: i32,
) {
    let starts = pad_i32(&topo.colors.starts);
    let counts = pad_i32(&topo.colors.counts);
    let nodes = pad_i32(&topo.colors.nodes);
    let d_cstart = device.new_buffer_from_slice(&starts);
    let d_ccount = device.new_buffer_from_slice(&counts);
    let d_cnodes = device.new_buffer_from_slice(&nodes);
    enc.set_buffer(16, Some(&d_cstart), 0);
    enc.set_buffer(17, Some(&d_ccount), 0);
    enc.set_buffer(18, Some(&d_cnodes), 0);
    set_bytes_i32(enc, 19, slot19);
    set_bytes_i32(enc, 20, topo.colors.num_colors);
    keep.extend([d_cstart, d_ccount, d_cnodes]);
}

/// Build one batch's device buffers and encode its dispatch. Does **not** commit.
///
/// `graphs` must be non-empty and share a topology (same `N` and `edges`) — the
/// caller ([`crate::streaming`]) guarantees this by batch key; the topology is
/// built from `graphs[0]`. `params` (reads, sweeps, beta) is shared across the
/// batch, matching v0.2 (`compute_beta_schedule(h[0], J[0], ...)`).
pub(crate) fn encode_batch(
    device: &crate::metal_device::MetalDevice,
    graphs: &[&IsingGraph],
    params: &SampleParams,
    kernel: Kernel,
) -> Result<EncodedBatch, SampleError> {
    encode_batch_inner(device, graphs, params, kernel, None)
}

/// [`encode_batch`] with an optional chunk-plan override, so a test can split
/// one anneal at chosen rung boundaries and compare it with the unsplit run.
/// Production passes `None` and lets [`chunk_plan`] decide.
fn encode_batch_inner(
    device: &crate::metal_device::MetalDevice,
    graphs: &[&IsingGraph],
    params: &SampleParams,
    kernel: Kernel,
    plan_override: Option<&[(i32, i32)]>,
) -> Result<EncodedBatch, SampleError> {
    let (first, n) = validate_batch(graphs, params, kernel)?;

    let num_problems = graphs.len();
    let num_reads = simd_rounded_reads(params.num_reads);
    let num_samples = num_problems * num_reads;
    let packed_size = n.div_ceil(8).max(1);

    let (beta, sweeps_per) = build_beta_schedule(
        first,
        params.num_sweeps,
        params.sweeps_per_beta,
        params.beta_range,
    );

    // Chromatic Gibbs uses a different pipeline but the *same* buffer layout —
    // only the dispatch geometry below differs.
    let node_parallel = kernel == Kernel::Gibbs && gibbs_node_parallel();
    let pipeline = match kernel {
        Kernel::Sa => &device.sa,
        Kernel::Msa => &device.msa,
        Kernel::Gibbs if node_parallel => &device.gibbs_parallel,
        Kernel::Gibbs => &device.gibbs,
    };

    // Dispatch geometry is constant across a batch's chunks, and must be: the
    // chromatic and multi-spin kernels' RNG stream identity is
    // `(threadgroup, thread_in_group)`, so varying the threadgroup width
    // mid-anneal would resume the wrong streams.
    //
    // Sequential kernels: one threadgroup per problem, `num_reads` threads (one
    // per read) each — `problem_id = thread_id / num_reads` = the threadgroup
    // index, so a model maps to a core and its reads are the threads inside.
    //
    // Chromatic Gibbs: one threadgroup per *sample* (`sample_id =
    // problem*num_reads + read`), and its threads split each color's nodes. The
    // group is capped at 256 by the kernel's `threadgroup int
    // partial_energies[256]` reduction array.
    //
    // Multi-spin: one threadgroup per (problem, 32-replica word), `words`
    // threadgroups per problem, 256 threads splitting each colour class.
    let words = num_reads.div_ceil(MSA_LANES);
    let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
    let (groups, threads_per_group) = match kernel {
        Kernel::Msa => (num_problems * words, MSA_THREADS.min(max_threads).max(1)),
        Kernel::Gibbs if node_parallel => (num_samples, 256.min(max_threads).max(1)),
        Kernel::Sa | Kernel::Gibbs => (num_problems, num_reads),
    };
    // `buffer(12)`: threads for SA / sequential Gibbs (one per read),
    // threadgroups for chromatic Gibbs and multi-spin. Also the count
    // `chunk_plan` multiplies: spin updates per thread, or word updates per
    // threadgroup.
    let num_threads = match kernel {
        Kernel::Msa => groups,
        Kernel::Sa | Kernel::Gibbs => num_samples,
    };
    let dims = BatchDims {
        n,
        num_betas: beta.len() as i32,
        sweeps_per,
        base_seed: (params.seed as u32).wrapping_add(1),
        num_threads,
        num_problems,
        num_reads,
        packed_size,
    };

    // The multi-spin kernel's spin words live in threadgroup memory sized per
    // dispatch. Refuse a job the opened device cannot hold rather than let
    // Metal fail the command buffer, which would end the session as a
    // DeviceFault for a per-job size problem.
    let state_bytes = (n.max(1) * 4).div_ceil(16) * 16;
    if kernel == Kernel::Msa {
        let need = state_bytes + MSA_STATIC_TG_BYTES;
        let have = device.device.max_threadgroup_memory_length() as usize;
        if need > have {
            return Err(SampleError::TooLarge(format!(
                "graph N={n} needs {need} B of threadgroup memory; device offers {have} B"
            )));
        }
    }

    let topo = SelfFeedingTopology::build(first);
    let inputs = upload_inputs(device, &topo, graphs);
    let out = DispatchBuffers {
        beta: device.new_buffer_from_slice(&beta),
        samples: device.new_zeroed_buffer((num_samples * packed_size) as u64),
        energies: device.new_zeroed_buffer((num_samples * 4) as u64), // i32
    };

    let plan = match plan_override {
        Some(p) => p.to_vec(),
        None => chunk_plan(kernel, &dims, groups),
    };
    let kstate = match kernel {
        Kernel::Sa => KernelEncode::Sa {
            persist: new_sa_persistent(device, &dims),
        },
        Kernel::Gibbs => KernelEncode::Gibbs {
            persist: new_gibbs_persistent(device, &dims, node_parallel, threads_per_group),
        },
        Kernel::Msa => KernelEncode::Msa {
            persist: new_msa_persistent(device, groups, n, threads_per_group),
            words: words as i32,
            state_bytes: state_bytes as u64,
        },
    };

    // One command buffer per chunk of the beta schedule. Splitting here rather
    // than encoding several dispatches into one buffer is the whole point: the
    // watchdog measures a *command buffer*, and the driver can only preempt
    // between them, so a single buffer holding every dispatch would be exactly
    // as dangerous as the unchunked version.
    let mut color_keep: Vec<metal::Buffer> = Vec::new();
    let mut cmds = Vec::with_capacity(plan.len());
    for (beta_start, beta_count) in plan {
        let cmd = device.queue.new_command_buffer().to_owned();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        bind_shared_args(encoder, &inputs, &out, &dims);
        match &kstate {
            KernelEncode::Sa { persist } => {
                bind_sa_chunk(encoder, persist, beta_start, beta_count);
            }
            KernelEncode::Gibbs { persist } => {
                encode_color_buffers(device, encoder, &topo, &mut color_keep, 0);
                bind_color_kernel_chunk(encoder, persist, beta_start, beta_count);
            }
            KernelEncode::Msa {
                persist,
                words,
                state_bytes,
            } => {
                encode_color_buffers(device, encoder, &topo, &mut color_keep, *words);
                bind_color_kernel_chunk(encoder, persist, beta_start, beta_count);
                encoder.set_threadgroup_memory_length(0, *state_bytes);
            }
        }
        encoder.dispatch_thread_groups(mtl_size_1d(groups), mtl_size_1d(threads_per_group));
        encoder.end_encoding();
        // Commit as we go. `MTLCommandQueue` holds at most
        // `maxCommandBufferCount` (64 by default) uncommitted buffers, and
        // `new_command_buffer` *blocks* once that many are outstanding — so
        // encoding every chunk before committing any deadlocks the moment an
        // anneal needs more than 64 chunks. (It does: a 1154-beta job at mining
        // settings plans ~165.) Committing here also lets chunk 0 start on the
        // GPU while we encode chunk 1.
        cmd.commit();
        cmds.push(cmd);
    }

    let DispatchBuffers {
        beta,
        samples,
        energies,
    } = out;
    let mut keep = inputs.into_keep();
    keep.extend([beta, energies]);
    keep.append(&mut color_keep);
    keep.extend(kstate.into_keep());

    Ok(EncodedBatch {
        cmds,
        d_samples: samples,
        n,
        num_reads,
        num_problems,
        packed_size,
        _keep: keep,
    })
}

fn pad_i32(v: &[i32]) -> Vec<i32> {
    if v.is_empty() {
        vec![0i32]
    } else {
        v.to_vec()
    }
}

/// Read a completed batch's bit-packed samples and host-score each read,
/// returning one `Vec<SamplerResult>` per problem (in `graphs` order).
///
/// `graphs` must be the exact slice passed to [`encode_batch`] (same order and
/// length) so each problem's spins are scored against its own `h`/`J`. The
/// buffer read is on the caller's thread; unpack + `energy_milli` scoring runs
/// on a rayon pool (one task per problem) — this is the bulk of the per-batch
/// host cost, overlapped with the next batch's GPU compute by the streaming
/// pipeline.
pub(crate) fn harvest_batch(
    batch: &EncodedBatch,
    graphs: &[&IsingGraph],
) -> Result<Vec<Vec<SamplerResult>>, SampleError> {
    use rayon::prelude::*;

    // Was a `debug_assert_eq!`, which is compiled out in release — a length
    // mismatch would index `packed` from `graphs` while sizing it from
    // `batch.num_problems` and panic the miner.
    if graphs.len() != batch.num_problems {
        return Err(SampleError::Driver(format!(
            "harvest_batch graphs.len()={} != batch.num_problems={}",
            graphs.len(),
            batch.num_problems
        )));
    }
    let count = batch.num_problems * batch.num_reads * batch.packed_size;
    let packed = read_i8_buffer(&batch.d_samples, count)?;
    let (num_reads, packed_size, n) = (batch.num_reads, batch.packed_size, batch.n);
    let out = graphs
        .par_iter()
        .enumerate()
        .map(|(p, graph)| {
            (0..num_reads)
                .map(|r| {
                    let start = (p * num_reads + r) * packed_size;
                    let spins = unpack_spins(&packed[start..start + packed_size], n);
                    score_spins(&spins, graph)
                })
                .collect()
        })
        .collect();
    Ok(out)
}

/// Run `num_reads` independent anneals on the GPU for one explicit problem
/// (synchronous single-problem batch; used by [`crate::MetalSampler::sample`]
/// and the golden-parity tests).
///
/// A zero-node graph short-circuits to `num_reads` empty results without
/// touching the GPU.
///
/// # Errors
///
/// Returns [`SampleError::TooLarge`] (harness `SampleError::Capacity`) when:
/// - `graph.num_nodes()` exceeds the algorithm's kernel node cap
///   (`SA_MAX_NODES` / `GIBBS_MAX_NODES`);
/// - `params.num_sweeps` exceeds `MAX_SWEEPS`, the sampler's guard against an
///   unbounded coordinator-supplied beta schedule.
///
/// Returns [`SampleError::Driver`] (harness `SampleError::DeviceFault`) when:
/// - the command buffer finished in a non-`Completed` state (device reset,
///   kernel fault, or GPU watchdog timeout) — the zeroed sample buffer would
///   otherwise score as a real all-`+1` solution;
/// - the samples buffer reads back null or shorter than the harvest expects.
///
/// [`SampleError::Metal`] is not produced by this path — device and pipeline
/// construction errors surface earlier, from [`crate::metal_device`].
///
/// # Examples
///
/// ```no_run
/// use quip_miner_metal::{sample_ising, Kernel, IsingGraph, SampleParams, metal_device::MetalDevice};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let device = MetalDevice::open(0)?;
/// let graph = IsingGraph::new(
///     vec![1.0, -1.0, 0.0, 1.0],
///     vec![1.0, -1.0, 1.0, -1.0],
///     vec![(0, 1), (1, 2), (2, 3), (3, 0)],
/// );
/// let params = SampleParams {
///     num_reads: 4,
///     num_sweeps: 64,
///     ..Default::default()
/// };
/// let samples = sample_ising(&device, &graph, &params, Kernel::Sa)?;
/// assert_eq!(samples.len(), 4);
/// # Ok(())
/// # }
/// ```
pub fn sample_ising(
    device: &crate::metal_device::MetalDevice,
    graph: &IsingGraph,
    params: &SampleParams,
    kernel: Kernel,
) -> Result<Vec<SamplerResult>, SampleError> {
    let n = graph.num_nodes();
    if n == 0 {
        let reads = params.num_reads.max(1);
        return Ok((0..reads)
            .map(|_| SamplerResult {
                spins: vec![],
                energy_milli: 0,
            })
            .collect());
    }

    let batch = encode_batch(device, &[graph], params, kernel)?;
    batch.wait_until_completed();

    // A GPU-side failure (device reset, kernel fault, timeout) leaves d_samples
    // in its allocated-zero state; without this check the unpack would turn
    // those zeros into an all-`+1` config and score it as a real solution.
    if let Some(status) = batch.failed_status() {
        return Err(SampleError::Driver(format!(
            "metal command buffer did not complete: status {status:?}"
        )));
    }

    let mut per_problem = harvest_batch(&batch, &[graph])?;
    let mut samples = per_problem.remove(0);
    // The dispatch computes `simd_rounded_reads(num_reads)` samples to fill the
    // final simdgroup; the caller asked for `num_reads`. Truncate here as
    // `streaming::finish_batch` does on the batched path, so neither path can
    // return more solutions than the job requested.
    samples.truncate(params.num_reads.max(1));
    Ok(samples)
}

fn read_i8_buffer(buf: &metal::Buffer, count: usize) -> Result<Vec<i8>, SampleError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let len = buf.length() as usize;
    if len < count {
        // The `copy_nonoverlapping` below reads `count` bytes out of this
        // buffer; a short buffer would read past its allocation (UB). Sizing
        // is consistent today (allocation and harvest count share
        // `packed_size`), so this only fires on drift.
        return Err(SampleError::Driver(format!(
            "Metal buffer holds {len} bytes, need {count}"
        )));
    }
    let ptr = buf.contents() as *const i8;
    if ptr.is_null() {
        return Err(SampleError::Driver(
            "Metal buffer contents() returned null".into(),
        ));
    }
    let mut out = vec![0i8; count];
    // SAFETY: buffer is StorageModeShared; `contents()` is non-null and
    // `length() >= count` are both *checked* above (not merely assumed); `out`
    // owns `count` initialized bytes and cannot overlap the device allocation;
    // and the command buffer that wrote it completed before this call.
    unsafe {
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), count);
    }
    Ok(out)
}

// Host-side (GPU-free) logic only: everything under test here is `cfg(macos)`,
// so the module carries the same gate. The dispatch itself is covered by
// `tests/golden_parity.rs`, which needs a real device.
#[cfg(test)]
mod tests {
    use super::*;

    /// Small ring: 0-1-2-3-0, unit J, ternary h (same fixture as `topology`).
    fn ring() -> IsingGraph {
        IsingGraph::new(
            vec![1.0, -1.0, 0.0, 1.0],
            vec![1.0, -1.0, 1.0, -1.0],
            vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        )
    }

    fn params(num_sweeps: usize) -> SampleParams {
        SampleParams {
            num_sweeps,
            ..Default::default()
        }
    }

    fn driver_msg(err: SampleError) -> String {
        match err {
            SampleError::Driver(m) => m,
            other => panic!("expected Driver, got {other:?}"),
        }
    }

    /// Message of a capacity refusal, asserting the variant on the way — the
    /// variant is what picks the harness condition (`Capacity`, not
    /// `DeviceFault`), so a size gate degrading to `Driver` is a protocol
    /// regression: `Capacity` is a per-job reject, `DeviceFault` ends the
    /// session.
    fn too_large_msg(err: SampleError) -> String {
        match err {
            SampleError::TooLarge(m) => m,
            other => panic!("expected TooLarge, got {other:?}"),
        }
    }

    #[test]
    fn unpack_spins_is_lsb_first_and_one_means_minus() {
        // 0b0000_0101 -> bits 0 and 2 set -> spins 0 and 2 are -1.
        let spins = unpack_spins(&[0b0000_0101i8], 8);
        assert_eq!(spins, vec![-1, 1, -1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn unpack_spins_reads_the_high_bit_of_a_byte() {
        // 0b1000_0000 is -128 as i8; the cast back to u8 must recover bit 7.
        let spins = unpack_spins(&[-128i8], 8);
        assert_eq!(spins, vec![1, 1, 1, 1, 1, 1, 1, -1]);
    }

    #[test]
    fn unpack_spins_crosses_byte_boundaries() {
        // Byte 0 = 0, byte 1 = bit 0 set -> node 8 is -1, nothing else.
        let spins = unpack_spins(&[0i8, 1i8], 12);
        assert_eq!(spins[..8], [1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(spins[8..], [-1, 1, 1, 1]);
    }

    #[test]
    fn unpack_spins_all_ones_byte_is_all_minus() {
        assert_eq!(unpack_spins(&[-1i8], 8), vec![-1i8; 8]);
    }

    #[test]
    fn unpack_spins_tolerates_a_short_buffer() {
        // Nodes past the slice read as bit 0 -> +1, rather than panicking.
        let spins = unpack_spins(&[0b0000_0011i8], 16);
        assert_eq!(spins.len(), 16);
        assert_eq!(spins[..2], [-1, -1]);
        assert!(spins[2..].iter().all(|&s| s == 1));
    }

    #[test]
    fn unpack_spins_of_zero_nodes_is_empty() {
        assert!(unpack_spins(&[], 0).is_empty());
    }

    #[test]
    fn beta_schedule_length_is_sweeps_over_sweeps_per_beta() {
        let (sched, sweeps_per) = build_beta_schedule(&ring(), 100, 10, Some((0.1, 10.0)));
        assert_eq!(sweeps_per, 10);
        assert_eq!(sched.len(), 10);
    }

    #[test]
    fn beta_schedule_runs_hot_to_cold() {
        let (sched, _) = build_beta_schedule(&ring(), 64, 1, Some((0.1, 10.0)));
        assert_eq!(sched.len(), 64);
        assert!((sched[0] - 0.1).abs() < 1e-6, "starts hot: {}", sched[0]);
        let last = sched[sched.len() - 1];
        assert!((last - 10.0).abs() < 1e-4, "ends cold: {last}");
        assert!(sched.windows(2).all(|w| w[1] >= w[0]), "monotonic");
    }

    #[test]
    fn beta_schedule_floors_at_one_beta() {
        // Zero sweeps and zero sweeps-per-beta must not divide by zero or
        // produce an empty ladder the kernel would read as num_betas = 0.
        let (sched, sweeps_per) = build_beta_schedule(&ring(), 0, 0, Some((0.1, 10.0)));
        assert_eq!(sweeps_per, 1);
        assert_eq!(sched.len(), 1);
    }

    #[test]
    fn beta_schedule_at_the_sweep_cap_is_bounded() {
        // The whole point of MAX_SWEEPS: the largest accepted job still
        // allocates a schedule of a size we can name.
        let (sched, _) = build_beta_schedule(&ring(), MAX_SWEEPS, 1, Some((0.1, 10.0)));
        assert_eq!(sched.len(), MAX_SWEEPS);
    }

    #[test]
    fn validate_batch_rejects_an_empty_batch() {
        // Release builds compiled the old `debug_assert!` out and panicked on
        // `graphs[0]`.
        let err = validate_batch(&[], &params(64), Kernel::Sa).unwrap_err();
        assert!(driver_msg(err).contains("at least one graph"));
    }

    #[test]
    fn validate_batch_accepts_a_normal_job() {
        let g = ring();
        let (first, n) = validate_batch(&[&g], &params(64), Kernel::Sa).unwrap();
        assert_eq!(n, 4);
        assert_eq!(first.num_nodes(), 4);
    }

    #[test]
    fn validate_batch_rejects_n_over_the_kernel_cap() {
        let big = IsingGraph::new(vec![0.0; SA_MAX_NODES + 1], vec![], vec![]);
        let err = validate_batch(&[&big], &params(64), Kernel::Sa).unwrap_err();
        let msg = too_large_msg(err);
        assert!(msg.contains("exceeds"), "{msg}");
        assert!(msg.contains(&SA_MAX_NODES.to_string()), "{msg}");
    }

    #[test]
    fn validate_batch_rejects_num_sweeps_over_the_cap() {
        let g = ring();
        let err = validate_batch(&[&g], &params(MAX_SWEEPS + 1), Kernel::Sa).unwrap_err();
        let msg = too_large_msg(err);
        assert!(msg.contains("num_sweeps"), "{msg}");
    }

    #[test]
    fn validate_batch_rejects_the_unbounded_coordinator_value() {
        // The reported DoS: `num_sweeps = u32::MAX` sized a ~34 GB Vec<f64>.
        let g = ring();
        let sweeps = u32::MAX as usize;
        let err = validate_batch(&[&g], &params(sweeps), Kernel::Sa).unwrap_err();
        assert!(too_large_msg(err).contains("num_sweeps"));
    }

    #[test]
    fn to_sample_error_separates_capacity_refusals_from_device_faults() {
        // A capacity refusal is permanent for this backend: the coordinator
        // must re-route, not retry here.
        assert_eq!(
            SampleError::TooLarge("n".into()).to_sample_error(),
            quip_solver_core::SampleError::Capacity
        );
        // A driver failure is a state this backend cannot recover from on its
        // own — the session must end for a supervisor restart, not reject a
        // job and keep accepting more work a wedged device can never serve.
        assert_eq!(
            SampleError::Driver("device reset".into()).to_sample_error(),
            quip_solver_core::SampleError::DeviceFault("device reset".into())
        );
    }

    #[test]
    fn validate_batch_accepts_the_cap_exactly() {
        let g = ring();
        validate_batch(&[&g], &params(MAX_SWEEPS), Kernel::Sa).unwrap();
    }

    #[test]
    fn sweep_cap_clears_the_gibbs_doubled_adapt_maximum() {
        // `METAL_ADAPT.max_sweeps` is 2048 and quip-solver-core doubles it for
        // Gibbs, so no legitimate adapt-driven job may be rejected. Update
        // this alongside `lib.rs` if either bound moves.
        let adapt_max_sweeps = 2048usize;
        assert!(MAX_SWEEPS >= 2 * adapt_max_sweeps);
        let g = ring();
        let sweeps = 2 * adapt_max_sweeps;
        validate_batch(&[&g], &params(sweeps), Kernel::Gibbs).unwrap();
    }

    #[test]
    fn kernel_max_nodes_matches_the_kernel_arrays() {
        // `delta_energy[4593]` in sa.metal, `packed_state[600]` (600*8) in
        // gibbs.metal.
        assert_eq!(kernel_max_nodes(Kernel::Sa), 4593);
        assert_eq!(kernel_max_nodes(Kernel::Gibbs), 4800);
        assert_eq!(kernel_max_nodes(Kernel::Sa), SA_MAX_NODES);
        assert_eq!(kernel_max_nodes(Kernel::Gibbs), GIBBS_MAX_NODES);
        assert_eq!(kernel_max_nodes(Kernel::Msa), MSA_MAX_NODES);
        assert_eq!(kernel_max_nodes(Kernel::Msa), 6016);
    }

    /// A star whose hub has `leaves` neighbours.
    fn star(leaves: usize) -> IsingGraph {
        let n = leaves + 1;
        let edges: Vec<(usize, usize)> = (1..n).map(|i| (0, i)).collect();
        IsingGraph::new(vec![0.0; n], vec![1.0; edges.len()], edges)
    }

    #[test]
    fn validate_batch_rejects_msa_degree_over_the_prefetch_budget() {
        // The kernel unrolls a fixed 20-neighbour prefetch; a denser node
        // would read past it. Capacity, not DeviceFault: the coordinator
        // routes the job to another backend.
        let g = star(21);
        let err = validate_batch(&[&g], &params(64), Kernel::Msa).unwrap_err();
        let msg = too_large_msg(err);
        assert!(msg.contains("degree 21"), "{msg}");
        // The other kernels walk CSR rows of any length.
        validate_batch(&[&g], &params(64), Kernel::Sa).unwrap();
        validate_batch(&[&g], &params(64), Kernel::Gibbs).unwrap();
    }

    #[test]
    fn validate_batch_accepts_msa_degree_at_the_budget() {
        let g = star(MSA_MAX_DEG);
        validate_batch(&[&g], &params(64), Kernel::Msa).unwrap();
    }

    #[test]
    fn max_csr_degree_follows_the_topology_builder_rules() {
        // A self-loop counts once; an out-of-range endpoint skips the edge.
        let g = IsingGraph::new(
            vec![0.0; 3],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![(0, 1), (1, 1), (1, 2), (2, 9)],
        );
        assert_eq!(max_csr_degree(&g), 3);
        let t = SelfFeedingTopology::build(&g);
        let builder_max = t
            .row_ptr
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap();
        assert_eq!(max_csr_degree(&g), builder_max);
    }

    #[test]
    fn msa_node_cap_fits_apple_threadgroup_memory() {
        assert_eq!(kernel_max_nodes(Kernel::Msa), MSA_MAX_NODES);
        const {
            assert!(MSA_MAX_NODES * 4 + MSA_STATIC_TG_BYTES <= APPLE_TG_MEMORY_BYTES);
            // Larger than either existing cap: the kernel holds 4 bytes per spin
            // where SA holds a delta-energy byte plus a packed bit per spin per
            // thread.
            assert!(MSA_MAX_NODES > SA_MAX_NODES);
            assert!(MSA_MAX_NODES > GIBBS_MAX_NODES);
        }
    }

    #[test]
    fn simd_rounded_reads_are_whole_msa_words() {
        for reads in [1usize, 31, 32, 33, 64, 100, 255, 256, 1000] {
            let r = simd_rounded_reads(reads);
            assert_eq!(r % MSA_LANES, 0, "reads={reads} rounded to {r}");
            assert!((1..=MAX_READS / MSA_LANES).contains(&(r / MSA_LANES)));
        }
    }

    /// Ferromagnetic chain of `n` spins with ternary fields, so every degree
    /// (1 at the ends, 2 inside) and every field value gets exercised.
    fn chain(n: usize) -> IsingGraph {
        let h = (0..n).map(|i| [-1.0, 0.0, 1.0][i % 3]).collect();
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let j = vec![-1.0; edges.len()];
        IsingGraph::new(h, j, edges)
    }

    /// Chunk boundaries fall on rung boundaries, and every carry-over (the
    /// spin words and each thread's RNG stream) round-trips through the
    /// persistent buffers, so splitting an anneal into chunks must not change
    /// one bit of output. This is the resume path's only direct test: small
    /// graphs never plan more than one chunk on their own.
    #[test]
    fn msa_chunked_anneal_is_bit_identical_to_one_chunk() {
        let device = crate::metal_device::MetalDevice::open(0).unwrap();
        let graph = chain(96);
        let params = SampleParams {
            num_reads: 64,
            num_sweeps: 64,
            sweeps_per_beta: 1,
            beta_range: Some((0.1, 4.0)),
            seed: 7,
        };
        let whole =
            encode_batch_inner(&device, &[&graph], &params, Kernel::Msa, Some(&[(0, 64)])).unwrap();
        let split = encode_batch_inner(
            &device,
            &[&graph],
            &params,
            Kernel::Msa,
            Some(&[(0, 16), (16, 16), (32, 16), (48, 16)]),
        )
        .unwrap();
        whole.wait_until_completed();
        split.wait_until_completed();
        assert!(whole.failed_status().is_none());
        assert!(split.failed_status().is_none());
        let count = 64 * whole.packed_size;
        let a = read_i8_buffer(&whole.d_samples, count).unwrap();
        let b = read_i8_buffer(&split.d_samples, count).unwrap();
        assert_eq!(a, b);
        // The anneal did something: not every read is the all-+1 zero state.
        assert!(a.iter().any(|&byte| byte != 0));
    }

    /// Two problems in one batch with two words each: every (problem, word)
    /// threadgroup packs into its own region and nothing overlaps.
    #[test]
    fn msa_batch_packs_each_problem_and_word_into_its_own_region() {
        let device = crate::metal_device::MetalDevice::open(0).unwrap();
        let a = chain(40);
        let b = chain(40);
        let params = SampleParams {
            num_reads: 64,
            num_sweeps: 32,
            sweeps_per_beta: 1,
            beta_range: Some((0.1, 4.0)),
            seed: 3,
        };
        let batch = encode_batch(&device, &[&a, &b], &params, Kernel::Msa).unwrap();
        batch.wait_until_completed();
        assert!(batch.failed_status().is_none());
        let per_problem = harvest_batch(&batch, &[&a, &b]).unwrap();
        assert_eq!(per_problem.len(), 2);
        for reads in &per_problem {
            assert_eq!(reads.len(), 64);
            for r in reads {
                assert_eq!(r.spins.len(), 40);
                assert!(r.spins.iter().all(|&s| s == 1 || s == -1));
            }
        }
        // 64 reads are two 32-lane words drawn from independent streams, so
        // the two words of one problem are not copies of each other.
        let (w0, w1) = per_problem[0].split_at(32);
        assert!(w0.iter().zip(w1).any(|(x, y)| x.spins != y.spins));
    }

    #[test]
    fn tile_i32_repeats_the_slice() {
        assert_eq!(tile_i32(&[1, 2, 3], 3), vec![1, 2, 3, 1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn tile_i32_edge_cases() {
        assert!(tile_i32(&[1, 2], 0).is_empty());
        assert!(tile_i32(&[], 4).is_empty());
        assert_eq!(tile_i32(&[7], 1), vec![7]);
    }

    #[test]
    fn pad_i32_substitutes_one_zero_for_empty() {
        // Metal rejects a zero-length buffer, so empty color blocks upload a
        // single unread slot.
        assert_eq!(pad_i32(&[]), vec![0]);
        assert_eq!(pad_i32(&[4, 5]), vec![4, 5]);
    }

    // -----------------------------------------------------------------------
    // Property tests (proptest is a dev-dep; private fns are reachable here)
    // -----------------------------------------------------------------------

    use proptest::prelude::*;

    /// Reference packer for the kernel bit contract (LSB-first per byte;
    /// bit set → spin -1, bit clear → +1). Inverse of [`unpack_spins`].
    fn pack_spins(spins: &[i8]) -> Vec<i8> {
        let nbytes = spins.len().div_ceil(8);
        let mut packed = vec![0i8; nbytes];
        for (i, &s) in spins.iter().enumerate() {
            if s < 0 {
                let byte_i = i >> 3;
                let bit = (i & 7) as u8;
                packed[byte_i] = (packed[byte_i] as u8 | (1u8 << bit)) as i8;
            }
        }
        packed
    }

    /// ±1 spin vectors; n includes non-byte-aligned sizes (1, 7, 8, 9, 63, 65).
    fn arb_spins() -> impl Strategy<Value = Vec<i8>> {
        prop_oneof![
            Just(1usize),
            Just(7usize),
            Just(8usize),
            Just(9usize),
            Just(63usize),
            Just(65usize),
            0usize..=96,
        ]
        .prop_flat_map(|n| prop::collection::vec(prop_oneof![Just(-1i8), Just(1i8)], n))
    }

    /// Small graphs with consensus-range h/J for scoring properties.
    fn arb_score_input() -> impl Strategy<Value = (Vec<i8>, IsingGraph)> {
        (1usize..=24).prop_flat_map(|n| {
            let max_edges = 32.min(n.saturating_mul(2));
            let spins = prop::collection::vec(prop_oneof![Just(-1i8), Just(1i8)], n);
            let h = prop::collection::vec(prop_oneof![Just(-1.0f64), Just(0.0), Just(1.0)], n);
            let edges = prop::collection::vec((0usize..n, 0usize..n), 0..=max_edges);
            (spins, h, edges).prop_flat_map(move |(spins, h, edges)| {
                let m = edges.len();
                prop::collection::vec(prop_oneof![Just(-1.0f64), Just(1.0)], m).prop_map(move |j| {
                    let graph = IsingGraph::new(h.clone(), j, edges.clone());
                    (spins.clone(), graph)
                })
            })
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            ..ProptestConfig::default()
        })]

        /// unpack(pack(s)) == s for any ±1 configuration.
        #[test]
        fn unpack_spins_round_trips(spins in arb_spins()) {
            let packed = pack_spins(&spins);
            let got = unpack_spins(&packed, spins.len());
            prop_assert_eq!(got, spins);
        }

        /// `score_spins` is bit-identical to consensus `energy_milli` and
        /// copies the input spin vector into `SamplerResult::spins`.
        #[test]
        fn score_spins_matches_energy_milli((spins, graph) in arb_score_input()) {
            let got = score_spins(&spins, &graph);
            let want = energy_milli(&spins, &graph.h, &graph.j, &graph.edges);
            prop_assert_eq!(got.energy_milli, want);
            prop_assert_eq!(got.spins, spins);
        }

        /// `chunk_plan` covers `0..num_betas.max(1)` with positive counts and
        /// no gaps; chunk sizes never under/overflow the schedule.
        ///
        /// Inputs are plain Debug scalars (BatchDims has no Debug derive; we
        /// must not change non-test code).
        #[test]
        fn chunk_plan_covers_schedule_without_zero_counts(
            kernel in prop_oneof![Just(Kernel::Sa), Just(Kernel::Msa), Just(Kernel::Gibbs)],
            num_betas in 0i32..=512,
            n in 0usize..=64,
            sweeps_per in 0usize..=256,
            num_threads in 0usize..=1024,
            groups in 1usize..=512,
        ) {
            let dims = BatchDims {
                n,
                num_betas,
                sweeps_per,
                base_seed: 0,
                num_threads,
                num_problems: 1,
                num_reads: 1,
                packed_size: 0,
            };
            let plan = chunk_plan(kernel, &dims, groups);
            let cover = dims.num_betas.max(1);
            prop_assert!(!plan.is_empty());
            let mut cursor = 0i32;
            for &(start, count) in &plan {
                prop_assert_eq!(start, cursor, "gap or overlap at start={}", start);
                prop_assert!(count > 0, "zero-length chunk at start={}", start);
                prop_assert!(
                    start.checked_add(count).is_some(),
                    "start+count overflows i32: start={} count={}",
                    start,
                    count
                );
                cursor = start + count;
            }
            prop_assert_eq!(cursor, cover, "plan does not cover full schedule");
        }
    }
}
