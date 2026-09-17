// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Isolated experiment. GPU tests are ignored so shared-device work stays explicit.
//! Run correctness before the release benchmark, with one test thread.

#![expect(
    clippy::print_stderr,
    reason = "isolated experiment emits benchmark records"
)]

use metal::{
    Buffer, CommandBuffer, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
use quip_miner_metal::topology::{fill_h_j, SelfFeedingTopology};
use quip_miner_metal::IsingGraph;
use quip_solver_core::beta::{default_ising_beta_range, geometric_beta_schedule};
use quip_solver_core::quip_protocol::scoring::energy_milli;
use std::collections::VecDeque;
use std::time::Instant;

const THREADS: usize = 256;
const ROW: usize = 8192;

fn splitmix(z: &mut u32) -> u32 {
    *z = z.wrapping_add(0x9e3779b9);
    let mut r = *z;
    r = (r ^ (r >> 16)).wrapping_mul(0x85ebca6b);
    r = (r ^ (r >> 13)).wrapping_mul(0xc2b2ae35);
    r ^ (r >> 16)
}

fn seed_rng(mut seed: u32) -> [u32; 4] {
    std::array::from_fn(|_| splitmix(&mut seed))
}

fn draw(s: &mut [u32; 4]) -> u32 {
    let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
    let t = s[1] << 9;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = s[3].rotate_left(11);
    result
}

// Scalar energy change, independent of the kernel's bit-sliced counters.
fn flip_delta(graph: &IsingGraph, spins: &[i8], node: usize) -> i32 {
    let mut field = graph.h[node] as i32;
    for (&(u, v), &j) in graph.edges.iter().zip(&graph.j) {
        if u == node {
            field += j as i32 * i32::from(spins[v]);
        }
        if v == node {
            field += j as i32 * i32::from(spins[u]);
        }
    }
    -2 * i32::from(spins[node]) * field
}

// Reproduce the random inputs, then update each replica with scalar ΔE <= 2M.
// Only the input RNG and color order match the kernel; no bit-sliced math is used.
fn scalar_oracle(
    graph: &IsingGraph,
    betas: &[f32],
    reads: usize,
    lanes: usize,
    seed: u32,
) -> Vec<Vec<i8>> {
    let topo = SelfFeedingTopology::build(graph);
    let mut samples = vec![vec![1; graph.num_nodes()]; reads];
    for word in 0..reads.div_ceil(lanes) {
        let mut rng: Vec<_> = (0..THREADS)
            .map(|tid| {
                seed_rng(
                    seed.max(1)
                        ^ (word as u32).wrapping_mul(2654435761)
                        ^ (tid as u32).wrapping_mul(2246822519),
                )
            })
            .collect();
        for (tid, state) in rng.iter_mut().enumerate() {
            for node in (tid..graph.num_nodes()).step_by(THREADS) {
                let low = u64::from(draw(state));
                let bits = if lanes == 64 {
                    low | (u64::from(draw(state)) << 32)
                } else {
                    low
                };
                for lane in 0..lanes {
                    if word * lanes + lane < reads {
                        samples[word * lanes + lane][node] =
                            if bits & (1 << lane) == 0 { 1 } else { -1 };
                    }
                }
            }
        }
        for (beta_idx, &beta) in betas.iter().enumerate() {
            let mut thresholds = [0u32; 64];
            for (m, cut) in thresholds.iter_mut().enumerate() {
                let p = (-2.0 * beta * m as f32).exp();
                *cut = if p >= 1.0 {
                    u32::MAX
                } else {
                    (p * 4294967296.0) as u32
                };
            }
            let mut row = [0u8; ROW];
            for (tid, state) in rng.iter_mut().enumerate() {
                for slot in (tid..ROW).step_by(THREADS) {
                    let u = draw(state);
                    row[slot] = (1..=63).take_while(|&m| u < thresholds[m]).count() as u8;
                }
            }
            let mut shift = seed
                ^ (word as u32).wrapping_mul(0x85ebca6b)
                ^ (beta_idx as u32).wrapping_mul(0xc2b2ae35);
            let offset = splitmix(&mut shift) as usize & (ROW - 1);
            for color in 0..topo.colors.num_colors as usize {
                let start = topo.colors.starts[color] as usize;
                let count = topo.colors.counts[color] as usize;
                for &node in &topo.colors.nodes[start..start + count] {
                    let node = node as usize;
                    let m = i32::from(row[(node + offset) & (ROW - 1)]);
                    for sample in samples.iter_mut().skip(word * lanes).take(lanes) {
                        if flip_delta(graph, sample, node) <= 2 * m {
                            sample[node] = -sample[node];
                        }
                    }
                }
            }
        }
    }
    samples
}

#[test]
fn scalar_metropolis_uses_the_actual_energy_change() {
    let graph = IsingGraph::new(vec![1.0, -1.0], vec![-1.0], vec![(0, 1)]);
    for spins in [[1, 1], [1, -1], [-1, 1], [-1, -1]] {
        for node in 0..2 {
            let mut flipped = spins;
            flipped[node] = -flipped[node];
            let measured = energy_milli(&flipped, &graph.h, &graph.j, &graph.edges)
                - energy_milli(&spins, &graph.h, &graph.j, &graph.edges);
            assert_eq!(i64::from(flip_delta(&graph, &spins, node)) * 1000, measured);
        }
    }
}

struct Gpu {
    device: Device,
    queue: metal::CommandQueue,
    p32: ComputePipelineState,
    p64: ComputePipelineState,
}

impl Gpu {
    fn open() -> Self {
        let device = Device::system_default().expect("Metal device");
        let compile = |source: &str, name: &str| {
            let library = device
                .new_library_with_source(source, &metal::CompileOptions::new())
                .expect("compile experiment kernel");
            let function = library.get_function(name, None).unwrap();
            device
                .new_compute_pipeline_state_with_function(&function)
                .unwrap()
        };
        let p32 = compile(include_str!("../kernels/msa.metal"), "msa_anneal");
        let p64 = compile(include_str!("../kernels/msa64.metal"), "msa64_anneal");
        for pipeline in [&p32, &p64] {
            assert!(pipeline.max_total_threads_per_threadgroup() >= THREADS as u64);
        }
        let queue = device.new_command_queue();
        Self {
            device,
            queue,
            p32,
            p64,
        }
    }

    fn buffer<T: Copy>(&self, data: &[T]) -> Buffer {
        if data.is_empty() {
            return self.zeroed(4);
        }
        self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn zeroed(&self, bytes: usize) -> Buffer {
        self.device
            .new_buffer(bytes.max(4) as u64, MTLResourceOptions::StorageModeShared)
    }

    fn pipeline(&self, lanes: usize) -> &ComputePipelineState {
        if lanes == 32 {
            &self.p32
        } else {
            &self.p64
        }
    }
}

struct Batch {
    buffers: Vec<(u64, Buffer)>,
    samples: Buffer,
    commands: Vec<CommandBuffer>,
    n: usize,
    jobs: usize,
    reads: usize,
    lanes: usize,
    betas: usize,
    colors: i32,
    seed: u32,
    next: usize,
    chunk: usize,
}

impl Batch {
    fn new(
        gpu: &Gpu,
        graph: &IsingGraph,
        betas: &[f32],
        jobs: usize,
        lanes: usize,
        seed: u32,
        chunk: usize,
    ) -> Self {
        assert!(lanes == 32 || lanes == 64);
        assert!(chunk > 0 && !betas.is_empty());
        let n = graph.num_nodes();
        let reads: usize = 128;
        let topo = SelfFeedingTopology::build(graph);
        assert!(topo.row_ptr.windows(2).all(|r| r[1] - r[0] <= 20));
        let (j, h) = fill_h_j(&topo, graph);
        let nnz = topo.nnz.max(1);
        let col = if topo.col_ind.is_empty() {
            vec![0]
        } else {
            topo.col_ind.clone()
        };
        let j = if j.is_empty() { vec![0] } else { j };
        let groups = jobs * reads.div_ceil(lanes);
        let samples = gpu.zeroed(jobs * reads * n.div_ceil(8));
        let buffers = vec![
            (0, gpu.buffer(&topo.row_ptr.repeat(jobs))),
            (1, gpu.buffer(&col.repeat(jobs))),
            (2, gpu.buffer(&j.repeat(jobs))),
            (
                3,
                gpu.buffer(&(0..jobs).map(|p| (p * (n + 1)) as i32).collect::<Vec<_>>()),
            ),
            (
                4,
                gpu.buffer(&(0..jobs).map(|p| (p * nnz) as i32).collect::<Vec<_>>()),
            ),
            (9, gpu.buffer(betas)),
            (10, samples.clone()),
            (11, gpu.zeroed(jobs * reads * 4)),
            (15, gpu.buffer(&h.repeat(jobs))),
            (16, gpu.buffer(&topo.colors.starts)),
            (17, gpu.buffer(&topo.colors.counts)),
            (18, gpu.buffer(&topo.colors.nodes)),
            (23, gpu.zeroed(groups * n * (lanes / 8))),
            (24, gpu.zeroed(groups * THREADS * 16)),
        ];
        Self {
            buffers,
            samples,
            commands: Vec::new(),
            n,
            jobs,
            reads,
            lanes,
            betas: betas.len(),
            colors: topo.colors.num_colors,
            seed,
            next: 0,
            chunk,
        }
    }

    fn submit(&mut self, gpu: &Gpu) -> bool {
        if self.next == self.betas {
            return false;
        }
        let count = self.chunk.min(self.betas - self.next);
        let command = gpu.queue.new_command_buffer().to_owned();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(gpu.pipeline(self.lanes));
        for (slot, buffer) in &self.buffers {
            encoder.set_buffer(*slot, Some(buffer), 0);
        }
        let groups = self.jobs * self.reads.div_ceil(self.lanes);
        let values = [
            (5, self.n as i32),
            (6, self.betas as i32),
            (7, 1),
            (8, self.seed as i32),
            (12, groups as i32),
            (13, self.jobs as i32),
            (14, self.reads as i32),
            (19, self.reads.div_ceil(self.lanes) as i32),
            (20, self.colors),
            (21, self.next as i32),
            (22, count as i32),
        ];
        for (slot, value) in values {
            encoder.set_bytes(slot, 4, (&value as *const i32).cast());
        }
        if self.lanes == 32 {
            encoder.set_threadgroup_memory_length(0, (self.n * 4).div_ceil(16) as u64 * 16);
        }
        encoder.dispatch_thread_groups(
            MTLSize {
                width: groups as u64,
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
        self.commands.push(command);
        self.next += count;
        true
    }

    fn wait(&self) {
        let command = self.commands.last().unwrap();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    }

    fn unpack(&self) -> Vec<Vec<i8>> {
        let bytes = self.jobs * self.reads * self.n.div_ceil(8);
        assert!(self.samples.length() as usize >= bytes);
        assert!(!self.samples.contents().is_null());
        // SAFETY: shared buffer length covers `bytes`; all GPU writes completed
        // before this method runs and the owning batch remains alive.
        let packed =
            unsafe { std::slice::from_raw_parts(self.samples.contents().cast::<u8>(), bytes) };
        packed
            .chunks_exact(self.n.div_ceil(8))
            .map(|sample| {
                (0..self.n)
                    .map(|node| {
                        if sample[node / 8] & (1 << (node % 8)) == 0 {
                            1
                        } else {
                            -1
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

#[expect(unexpected_cfgs, reason = "objc 0.2 expands legacy cargo-clippy cfg")]
fn gpu_seconds(command: &metal::CommandBufferRef) -> f64 {
    use objc::{msg_send, sel, sel_impl};
    // SAFETY: the caller has waited for this command to complete; these Metal
    // selectors return double-precision timestamps in seconds.
    let (start, end): (f64, f64) = unsafe {
        (
            msg_send![command, GPUStartTime],
            msg_send![command, GPUEndTime],
        )
    };
    assert!(end >= start && start.is_finite() && end.is_finite());
    end - start
}

#[test]
#[ignore = "requires an exclusive GPU correctness slot"]
fn msa64_matches_scalar_oracle_and_resumes_exactly() {
    let gpu = Gpu::open();
    let graph = IsingGraph::new(
        vec![1.0, -1.0, 0.0, 1.0, -1.0],
        vec![-1.0, 1.0, -1.0, 1.0, 0.0],
        vec![(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)],
    );
    let star = IsingGraph::new(
        (0..21).map(|node| [-1.0, 0.0, 1.0][node % 3]).collect(),
        (0..20)
            .map(|edge| if edge % 2 == 0 { -1.0 } else { 1.0 })
            .collect(),
        (1..21).map(|node| (0, node)).collect(),
    );
    let betas = [0.0, 0.125, 0.25, 0.5];
    for graph in [graph, star] {
        for lanes in [32, 64] {
            let expected = scalar_oracle(&graph, &betas, 128, lanes, 7);
            for chunk in [1, 2, 4] {
                let mut batch = Batch::new(&gpu, &graph, &betas, 1, lanes, 7, chunk);
                while batch.submit(&gpu) {
                    batch.wait();
                }
                assert_eq!(batch.unpack(), expected, "lanes={lanes} chunk={chunk}");
            }
        }
    }
}

#[test]
#[ignore = "requires an exclusive GPU correctness slot"]
fn msa64_device_state_exceeds_threadgroup_capacity() {
    let gpu = Gpu::open();
    assert!(4577 * 8 > gpu.device.max_threadgroup_memory_length() as usize);
    assert_eq!(gpu.p64.static_threadgroup_memory_length(), 8448);
    let graph = IsingGraph::new(vec![1.0; 8193], vec![], vec![]);
    let mut batch = Batch::new(&gpu, &graph, &[4.0, 4.0], 1, 64, 9, 1);
    while batch.submit(&gpu) {
        batch.wait();
    }
    for sample in batch.unpack() {
        assert!(sample.iter().filter(|&&s| s == -1).count() > 8100);
    }
}

fn fixture() -> IsingGraph {
    let edges: Vec<_> = include_str!("fixtures/advantage2-system1.edges")
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let mut fields = line.split_whitespace();
            (
                fields.next().unwrap().parse().unwrap(),
                fields.next().unwrap().parse().unwrap(),
            )
        })
        .collect();
    let mut state = 1u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let j = edges
        .iter()
        .map(|_| if next() & 1 == 0 { 1.0 } else { -1.0 })
        .collect();
    let h = (0..4577)
        .map(|_| [-1.0, 0.0, 1.0][(next() % 3) as usize])
        .collect();
    IsingGraph::new(h, j, edges)
}

struct Measurement {
    wall: f64,
    gpu: f64,
    max_chunk: f64,
    best: i64,
    mean_best: f64,
}

fn measure(
    gpu: &Gpu,
    graph: &IsingGraph,
    betas: &[f32],
    lanes: usize,
    batch_jobs: usize,
) -> Measurement {
    let start = Instant::now();
    let mut remaining = 40;
    let mut window = VecDeque::new();
    let mut complete = Vec::new();
    while remaining > 0 || !window.is_empty() {
        while remaining > 0 && window.len() < 2 {
            let jobs = remaining.min(batch_jobs);
            let mut batch =
                Batch::new(gpu, graph, betas, jobs, lanes, (41 - remaining) as u32, 128);
            assert!(batch.submit(gpu));
            window.push_back(batch);
            remaining -= jobs;
        }
        let mut batch = window.pop_front().unwrap();
        batch.wait();
        if batch.submit(gpu) {
            window.push_back(batch);
        } else {
            complete.push(batch);
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let mut gpu_time = 0.0;
    let mut max_chunk = 0.0f64;
    let mut bests = Vec::new();
    for batch in complete {
        for command in &batch.commands {
            let time = gpu_seconds(command);
            gpu_time += time;
            max_chunk = max_chunk.max(time);
        }
        for samples in batch.unpack().as_chunks::<128>().0 {
            bests.push(
                samples
                    .iter()
                    .map(|sample| energy_milli(sample, &graph.h, &graph.j, &graph.edges))
                    .min()
                    .unwrap(),
            );
        }
    }
    assert_eq!(bests.len(), 40);
    Measurement {
        wall,
        gpu: gpu_time,
        max_chunk,
        best: *bests.iter().min().unwrap(),
        mean_best: bests.iter().map(|&energy| energy as f64).sum::<f64>() / 40.0,
    }
}

#[test]
#[ignore = "exclusive GPU benchmark: run correctness tests first"]
fn benchmark_msa32_against_msa64_at_equal_jobs_and_equal_groups() {
    let gpu = Gpu::open();
    let graph = fixture();
    let (hot, cold) = default_ising_beta_range(&graph);
    let betas: Vec<_> = geometric_beta_schedule(hot, cold, 7392)
        .into_iter()
        .map(|beta| beta as f32)
        .collect();
    eprintln!("device={} threads={} static32={} static64={} nodes={} reads=128 sweeps=7392 jobs=40 chunk=128", gpu.device.name(), THREADS, gpu.p32.static_threadgroup_memory_length(), gpu.p64.static_threadgroup_memory_length(), graph.num_nodes());
    for (lanes, batch_jobs) in [(32, 10), (64, 10), (64, 20)] {
        let _ = measure(&gpu, &graph, &betas[..128], lanes, batch_jobs);
    }
    for pair in 0..3 {
        let arms = if pair % 2 == 0 {
            [(32, 10), (64, 10), (64, 20)]
        } else {
            [(64, 20), (64, 10), (32, 10)]
        };
        for (lanes, batch_jobs) in arms {
            let m = measure(&gpu, &graph, &betas, lanes, batch_jobs);
            let updates = 40.0 * (128 / lanes) as f64 * 4577.0 * 7392.0;
            eprintln!("pair={pair} lanes={lanes} batch_jobs={batch_jobs} groups={} wall_s={:.6} gpu_sum_s={:.6} max_chunk_ms={:.3} jobs_s={:.3} word_updates_s={:.0} replica_updates_s={:.0} best={} mean_best={:.0}", batch_jobs * 128 / lanes, m.wall, m.gpu, m.max_chunk * 1000.0, 40.0 / m.wall, updates / m.wall, updates * lanes as f64 / m.wall, m.best, m.mean_best);
        }
    }
}
