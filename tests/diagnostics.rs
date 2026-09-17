// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Diagnostic-instrumented MSA kernel checks.
//!
//! `#define QUIP_MSA_DIAGNOSTICS` renames the MSA entry point to
//! `msa_anneal_diag` and appends two per-chunk diagnostics to the buffer ABI
//! (buffer 25, a per-thread accepted-flip counter; buffer 26, per-(thread,
//! lane) energy partials). Without the macro the production `msa_anneal` entry
//! and its 0..24 buffer layout are unchanged.
//!
//! These tests drive both entry points directly with the Metal APIs (no
//! production framework) over the same small graph and seed, then check the
//! diagnostics are correct and that the diagnostic build yields bit-identical
//! packed spins to the production build.
//!
//! They are real-GPU tests, so they are `#[ignore]`d and must be run
//! single-threaded on a device with a free core:
//!
//! ```sh
//! cargo test --release --test diagnostics -- --ignored --test-threads=1
//! ```

#![expect(
    clippy::print_stderr,
    reason = "test reports the device and diagnostic readings on stderr"
)]

use metal::{Buffer, CompileOptions, Device, MTLResourceOptions, MTLSize};
use quip_miner_metal::topology::{fill_h_j, SelfFeedingTopology};
use quip_miner_metal::IsingGraph;
use quip_solver_core::quip_protocol::scoring::energy_milli;

const THREADS: usize = 256;
const LANES: usize = 32;
const MSA_SRC: &str = include_str!("../kernels/msa.metal");

fn diag_source() -> String {
    format!("#define QUIP_MSA_DIAGNOSTICS\n{MSA_SRC}")
}

fn compile(device: &Device, source: &str, entry: &str) -> metal::ComputePipelineState {
    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .unwrap_or_else(|e| panic!("{entry} compile: {e}"));
    let function = library.get_function(entry, None).unwrap();
    device
        .new_compute_pipeline_state_with_function(&function)
        .unwrap()
}

fn typed<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    if data.is_empty() {
        return device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    }
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn zeroed(device: &Device, bytes: usize) -> Buffer {
    device.new_buffer(bytes.max(4) as u64, MTLResourceOptions::StorageModeShared)
}

fn read<T: Copy>(buffer: &Buffer, len: usize) -> Vec<T> {
    assert!(std::mem::size_of::<T>() > 0);
    assert!(len <= buffer.length() as usize / std::mem::size_of::<T>());
    // SAFETY: shared-storage buffer; length covers `len * size_of::<T>()` and
    // the GPU finished all writes before this read (the caller waited).
    let slice = unsafe { std::slice::from_raw_parts(buffer.contents().cast::<T>(), len) };
    slice.to_vec()
}

/// Replica `read`'s spins from `persistent_state` words
/// (`words[tg*n + var]`, bit `lane` set = -1).
fn words_to_spins(words: &[u32], n: usize, read: usize) -> Vec<i8> {
    let word = read / LANES;
    let lane = read % LANES;
    (0..n)
        .map(|v| {
            if (words[word * n + v] >> lane) & 1 == 0 {
                1
            } else {
                -1
            }
        })
        .collect()
}

struct Gpu {
    device: Device,
    queue: metal::CommandQueue,
    prod: metal::ComputePipelineState,
    diag: metal::ComputePipelineState,
}

impl Gpu {
    fn open() -> Self {
        let device = Device::system_default().expect("Metal device");
        let prod = compile(&device, MSA_SRC, "msa_anneal");
        let diag = compile(&device, &diag_source(), "msa_anneal_diag");
        for p in [&prod, &diag] {
            assert!(p.max_total_threads_per_threadgroup() >= THREADS as u64);
        }
        let queue = device.new_command_queue();
        Self {
            device,
            queue,
            prod,
            diag,
        }
    }
}

/// One problem, one word (num_reads = 32 replicas), driven through a chosen
/// entry point with the beta ladder split into `chunk`-sized dispatches.
struct Run {
    gpu: Gpu,
    pipeline: metal::ComputePipelineState,
    n: usize,
    reads: usize,
    num_betas: usize,
    seed: u32,
    num_colors: i32,
    chunk: i32,
    buffers: Vec<(u64, Buffer)>,
    samples: Buffer,
    state: Buffer,
    counts: usize,
    energy_len: usize,
}

impl Run {
    fn new(gpu: Gpu, graph: &IsingGraph, betas: &[f32], seed: u32, chunk: i32, diag: bool) -> Self {
        let n = graph.num_nodes();
        let reads = LANES;
        let topo = SelfFeedingTopology::build(graph);
        let (j_csr, h_i8) = fill_h_j(&topo, graph);
        let col = if topo.col_ind.is_empty() {
            vec![0]
        } else {
            topo.col_ind.clone()
        };
        let j = if j_csr.is_empty() { vec![0] } else { j_csr };
        let groups = 1; // one problem, one word
        let counts = groups * THREADS;
        let energy_len = groups * THREADS * LANES;
        let samples = zeroed(&gpu.device, reads * n.div_ceil(8));
        let state = zeroed(&gpu.device, n * 4);
        let mut buffers = vec![
            (0, typed(&gpu.device, &topo.row_ptr)),
            (1, typed(&gpu.device, &col)),
            (2, typed(&gpu.device, &j)),
            (3, typed(&gpu.device, &[0i32])),
            (4, typed(&gpu.device, &[0i32])),
            (9, typed(&gpu.device, betas)),
            (10, samples.clone()),
            (11, zeroed(&gpu.device, reads * n.div_ceil(8))),
            (15, typed(&gpu.device, &h_i8)),
            (16, typed(&gpu.device, &topo.colors.starts)),
            (17, typed(&gpu.device, &topo.colors.counts)),
            (18, typed(&gpu.device, &topo.colors.nodes)),
            (23, state.clone()),
            (24, zeroed(&gpu.device, THREADS * 16)),
        ];
        if diag {
            buffers.push((25, zeroed(&gpu.device, counts * 8)));
            buffers.push((26, zeroed(&gpu.device, energy_len * 4)));
        }
        let pipeline = if diag {
            gpu.diag.clone()
        } else {
            gpu.prod.clone()
        };
        Self {
            gpu,
            pipeline,
            n,
            reads,
            num_betas: betas.len(),
            seed,
            num_colors: topo.colors.num_colors,
            chunk,
            buffers,
            samples,
            state,
            counts,
            energy_len,
        }
    }

    /// Dispatches `chunk` betas starting at `next`; returns the new `next`.
    fn dispatch(&mut self, next: i32) -> i32 {
        let count = self.chunk.min(self.num_betas as i32 - next);
        let command = self.gpu.queue.new_command_buffer().to_owned();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        for (slot, buffer) in &self.buffers {
            encoder.set_buffer(*slot, Some(buffer), 0);
        }
        let values = [
            (8, self.seed as i32),
            (19, 1),
            (21, next),
            (22, count),
            (5, self.n as i32),
            (6, self.num_betas as i32),
            (7, 1),
            (13, 1),
            (12, 1),
            (14, self.reads as i32),
            (20, self.num_colors),
        ];
        for (slot, value) in values {
            encoder.set_bytes(slot, 4, (&value as *const i32).cast());
        }
        encoder.set_threadgroup_memory_length(0, (self.n * 4).div_ceil(16) as u64 * 16);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: THREADS as u64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
        next + count
    }
}

fn buf_index(buffers: &[(u64, Buffer)], slot: u64) -> usize {
    buffers.iter().position(|(s, _)| *s == slot).unwrap()
}

fn graph_with_selfloop() -> IsingGraph {
    IsingGraph::new(
        vec![1.0, -1.0, 0.0, 1.0, -1.0],
        vec![-1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 0)],
    )
}

#[test]
#[ignore = "real GPU: run single-threaded with a free core"]
fn diagnostics_match_production_spins_and_energy_each_chunk() {
    let graph = graph_with_selfloop();
    let n = graph.num_nodes();
    let betas = [0.1f32, 0.4];
    let seed = 7u32;
    let chunk = 1;

    // Production run (unmodified buffer ABI) over the same graph and seed.
    let mut prod = Run::new(Gpu::open(), &graph, &betas, seed, chunk, false);

    // Diagnostic run with per-chunk read-back of state, energy and flips.
    let mut diag = Run::new(Gpu::open(), &graph, &betas, seed, chunk, true);
    let e_buf = buf_index(&diag.buffers, 26);
    let c_buf = buf_index(&diag.buffers, 25);
    let mut previous: Option<Vec<u32>> = None;
    let mut next = 0;
    while next < betas.len() as i32 {
        let production_end = prod.dispatch(next);
        let processed = diag.dispatch(next);
        assert_eq!(production_end, processed);
        let diag_bytes: Vec<u8> = read(&diag.samples, LANES * n.div_ceil(8));
        let prod_bytes: Vec<u8> = read(&prod.samples, LANES * n.div_ceil(8));
        assert_eq!(diag_bytes, prod_bytes, "packed spins diverged after chunk");
        let chunk_betas = processed - next;
        next = processed;
        // Per-chunk energy: independent `energy_milli` from the persisted
        // state must equal the summed (thread, lane) partials for every lane.
        let words: Vec<u32> = read(&diag.state, n);
        let partials: Vec<i32> = read(&diag.buffers[e_buf].1, diag.energy_len);
        for lane in 0..LANES {
            let spins = words_to_spins(&words, n, lane);
            let independent = energy_milli(&spins, &graph.h, &graph.j, &graph.edges);
            let measured: i64 = (0..THREADS)
                .map(|t| i64::from(partials[t * LANES + lane]))
                .sum();
            assert_eq!(
                measured, independent,
                "after chunk through beta {next}: replica {lane} energy {measured} != {independent}"
            );
        }
        // Per-chunk flips: (0, n * chunk_betas * sweeps * 32].
        let caps: Vec<u64> = read(&diag.buffers[c_buf].1, diag.counts);
        let sum: u64 = caps.iter().sum();
        if let Some(before) = previous.replace(words.clone()) {
            let changed: u64 = before
                .iter()
                .zip(&words)
                .map(|(a, b)| u64::from((a ^ b).count_ones()))
                .sum();
            assert_eq!(sum, changed, "one sweep must count each changed spin once");
        }
        let max = (n * chunk_betas as usize * LANES) as u64;
        assert!(
            sum <= max,
            "chunk {chunk_betas}: accepted flips {sum} outside [0, {max}]"
        );
    }

    eprintln!(
        "device={} nodes={} reads={} diag_spins_matched_prod",
        diag.gpu.device.name(),
        n,
        diag.reads
    );
}

#[test]
#[ignore = "real GPU: zero-beta all-flip accepted-count check"]
fn zero_beta_accepts_every_flip_exactly() {
    // Isolated fieldless nodes: d = 0 at every update and a beta of 0 makes
    // every Metropolis move accept (flip) all 32 replicas regardless of the
    // geometric draw, so the summed accepted count is exactly n * sweeps * 32.
    let graph = IsingGraph::new(vec![0.0; 5], vec![], vec![]);
    let n = graph.num_nodes();
    let betas = [0.0f32, 0.0, 0.0, 0.0];
    let seed = 3u32;
    let chunk = 2;

    let mut run = Run::new(Gpu::open(), &graph, &betas, seed, chunk, true);
    let c_buf = buf_index(&run.buffers, 25);
    let mut next = 0;
    while next < betas.len() as i32 {
        let processed = run.dispatch(next);
        let chunk_betas = processed - next;
        next = processed;
        // This chunk overwrote the counter; every move flips all 32 lanes, so
        // the summed per-thread count is exactly n * chunk_betas * 1 * 32.
        let caps: Vec<u64> = read(&run.buffers[c_buf].1, run.counts);
        let sum: u64 = caps.iter().sum();
        let expected = (n * chunk_betas as usize * LANES) as u64;
        assert_eq!(
            sum, expected,
            "zero-beta chunk {chunk_betas}: got {sum}, expected {expected}"
        );
    }
    eprintln!(
        "device={} zero-beta accepted_every_flip",
        run.gpu.device.name()
    );
}
