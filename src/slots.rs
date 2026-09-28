// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

use crate::metal_device::MetalDevice;
use crate::sampler::{self, BufferPool, CachedTopology, Kernel, SampleError, MSA_THREADS};
use crate::topology::fill_h_j_matching;
use crate::topology::SelfFeedingTopology;
use crate::{IsingGraph, SampleParams, SamplerResult};
use metal::{MTLCommandBufferStatus, MTLSize};
use std::sync::{Arc, OnceLock};

fn decode_pool() -> Result<&'static rayon::ThreadPool, SampleError> {
    static POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();
    // Bound decode fan-out independently of the global Rayon pool. Four
    // decode workers plus four preparers leave four of the M4 Max's twelve
    // performance cores for the runner, producers and other work. Byte-table
    // expansion reduces the work on the slowest decode worker. Streaming
    // keeps its existing pool and unpacker.
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .thread_name(|index| format!("resident-decode-{index}"))
            .build()
            .map_err(|error| error.to_string())
    })
    .as_ref()
    .map_err(|error| SampleError::Driver(format!("start resident decode pool: {error}")))
}

fn unpack_slot_spins(packed: &[i8], n: usize) -> Vec<i8> {
    const SPINS: [[i8; 8]; 256] = {
        let mut table = [[1; 8]; 256];
        let mut byte = 0;
        while byte < 256 {
            let mut bit = 0;
            while bit < 8 {
                if byte & (1 << bit) != 0 {
                    table[byte][bit] = -1;
                }
                bit += 1;
            }
            byte += 1;
        }
        table
    };
    let mut spins = vec![0; n];
    let (chunks, tail) = spins.as_chunks_mut::<8>();
    for (chunk, &byte) in chunks.iter_mut().zip(packed) {
        *chunk = SPINS[byte as u8 as usize];
    }
    if !tail.is_empty() {
        tail.copy_from_slice(&SPINS[packed[n / 8] as u8 as usize][..n % 8]);
    }
    spins
}

pub(crate) struct SlotJob {
    pub(crate) graph: IsingGraph,
    /// Shared across jobs with the same schedule, so a slot that already
    /// holds it skips the upload.
    pub(crate) schedule: Arc<[f32]>,
    pub(crate) checkpoints: Vec<usize>,
    pub(crate) seed: u64,
}

impl SlotJob {
    #[cfg(test)]
    fn validate(&self, sched_stride: usize) -> Result<(), SampleError> {
        if self.schedule.len() > sched_stride {
            return Err(SampleError::TooLarge("schedule exceeds slot stride".into()));
        }
        if self.graph.j.len() != self.graph.edges.len()
            || !sampler::device_energy_exact(&self.graph)
        {
            return Err(SampleError::TooLarge(
                "slot jobs require exact device-energy coefficients".into(),
            ));
        }
        validate_schedule(&self.schedule, &self.checkpoints)
    }
}

pub(crate) fn validate_schedule(
    schedule: &[f32],
    checkpoints: &[usize],
) -> Result<(), SampleError> {
    if schedule.is_empty()
        || schedule.iter().any(|b| !b.is_finite() || *b < 0.0)
        || checkpoints.first().is_none_or(|&p| p == 0)
        || checkpoints.last() != Some(&schedule.len())
        || checkpoints.windows(2).any(|p| p[0] >= p[1])
    {
        return Err(SampleError::Driver(
            "invalid slot schedule or checkpoints".into(),
        ));
    }
    Ok(())
}

/// A shared edge list. Equal storage identifies a topology without comparing
/// its edges.
pub(crate) type Edges = Arc<[(usize, usize)]>;

/// Host allocations only. Construction binds coefficients to the exact graph.
pub(crate) struct PreparedInputs {
    graph: IsingGraph,
    /// The edge list construction verified `graph` against, edge by edge.
    edges: Edges,
    couplings: Vec<i8>,
    fields: Vec<i8>,
}

impl PreparedInputs {
    pub(crate) fn new(
        graph: &IsingGraph,
        topology: &SelfFeedingTopology,
        edges: &Edges,
    ) -> Option<Self> {
        let (couplings, fields) = fill_h_j_matching(topology, edges, graph)?;
        Some(Self {
            graph: graph.clone(),
            edges: Arc::clone(edges),
            couplings,
            fields,
        })
    }

    pub(crate) fn edges(&self) -> &Edges {
        &self.edges
    }
}

pub(crate) type SlotId = usize;

fn device_time_share(total: u64, count: usize, index: usize) -> u64 {
    total / count as u64 + u64::from((index as u64) < total % count as u64)
}

fn checked_read_pointer<T>(
    contents: *mut std::ffi::c_void,
    length: u64,
    count: usize,
) -> Result<*const T, SampleError> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| SampleError::Driver("slot read size overflow".into()))?;
    if contents.is_null() || length < bytes as u64 {
        return Err(SampleError::Driver(
            "slot read buffer is null or too short".into(),
        ));
    }
    Ok(contents.cast())
}

pub(crate) struct Checkpoint {
    pub(crate) slot: SlotId,
    pub(crate) index: usize,
    pub(crate) last: bool,
    pub(crate) best: i64,
}

struct ResidentJob {
    job: SlotJob,
    position: usize,
    next_checkpoint: usize,
    has_output: bool,
    device_us: u64,
}

pub(crate) struct SlotPool {
    decode: &'static rayon::ThreadPool,
    cached: Arc<CachedTopology>,
    pool: Arc<BufferPool>,
    queue: metal::CommandQueue,
    pipeline: metal::ComputePipelineState,
    // Buffer indices match the slot kernel ABI. Storage is rented only in new.
    buffers: Vec<(u64, metal::Buffer)>,
    slots: Vec<Option<ResidentJob>>,
    /// The schedule each slot's region of buffer 9 holds.
    held: Vec<Option<Arc<[f32]>>>,
    /// Edge storage already found equal to this pool's topology.
    verified: Option<Edges>,
    steps: Vec<SlotStep>,
    command: Option<metal::CommandBuffer>,
    faulted: bool,
    num_reads: usize,
    words: usize,
    threads: usize,
    sched_stride: usize,
}

impl SlotPool {
    pub(crate) fn new(
        device: &MetalDevice,
        graph: &IsingGraph,
        num_reads: usize,
        capacity: usize,
        sched_stride: usize,
    ) -> Result<Self, SampleError> {
        let decode = decode_pool()?;
        let params = SampleParams {
            num_reads,
            num_sweeps: sched_stride,
            ..Default::default()
        };
        sampler::validate_batch(&[graph], &params, Kernel::Msa)?;
        if graph.num_nodes() == 0 || num_reads == 0 || capacity == 0 || sched_stride == 0 {
            return Err(SampleError::TooLarge(
                "slot pool dimensions must be nonzero".into(),
            ));
        }
        let state_bytes = (graph.num_nodes() * 4).div_ceil(16) * 16;
        let need = state_bytes + sampler::MSA_STATIC_TG_BYTES;
        if need > device.device.max_threadgroup_memory_length() as usize {
            return Err(SampleError::TooLarge(format!(
                "slot pool needs {need} B of threadgroup memory"
            )));
        }
        let cached = device.topology_cache.get_or_build(device, graph, true);
        let words = num_reads.div_ceil(sampler::MSA_LANES);
        let threads = MSA_THREADS
            .min(device.msa_slots.max_total_threads_per_threadgroup() as usize)
            .max(1);
        // Bound every dimension and offset used by the kernel's 32-bit ABI.
        let region = |factors: &[usize]| -> Result<u64, SampleError> {
            let bytes = factors
                .iter()
                .try_fold(capacity, |n, &v| n.checked_mul(v))
                .filter(|&n| n <= i32::MAX as usize)
                .ok_or_else(|| {
                    SampleError::TooLarge("slot buffer exceeds 32-bit indexing".into())
                })?;
            Ok(bytes.max(4) as u64)
        };
        let sizes = [
            (2, region(&[cached.topo.nnz])?),
            (9, region(&[sched_stride, 4])?),
            (10, region(&[num_reads, cached.n.div_ceil(8)])?),
            (11, region(&[num_reads, 4])?),
            (15, region(&[cached.n])?),
            (23, region(&[words, cached.n, 4])?),
            (24, region(&[words, threads, 4, 4])?),
            (25, region(&[std::mem::size_of::<SlotStep>()])?),
        ];
        let buffers = sizes
            .into_iter()
            .map(|(index, bytes)| (index, device.buffer_pool.take(&device.device, bytes)))
            .collect();
        Ok(Self {
            decode,
            cached,
            pool: Arc::clone(&device.buffer_pool),
            queue: device.queue.clone(),
            pipeline: device.msa_slots.clone(),
            buffers,
            slots: (0..capacity).map(|_| None).collect(),
            held: (0..capacity).map(|_| None).collect(),
            verified: None,
            steps: Vec::with_capacity(capacity),
            command: None,
            faulted: false,
            num_reads,
            words,
            threads,
            sched_stride,
        })
    }

    /// Whether inputs prepared against `edges` fit this pool. Admission checks
    /// this itself, so callers targeting this pool need not call it first.
    /// Storage already verified skips the edge compare, which reads the whole
    /// edge list and costs more than the rest of admission.
    pub(crate) fn matches_prepared(
        &mut self,
        edges: &Edges,
        nodes: usize,
        num_reads: usize,
    ) -> bool {
        if self.cached.n != nodes || self.num_reads != num_reads {
            return false;
        }
        if self
            .verified
            .as_ref()
            .is_some_and(|v| Arc::ptr_eq(v, edges))
        {
            return true;
        }
        let same = self.cached.edges[..] == edges[..];
        if same {
            self.verified = Some(Arc::clone(edges));
        }
        same
    }

    pub(crate) fn capacity(&self) -> usize {
        self.slots.len()
    }
    pub(crate) fn live(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// A submitted command stays in flight until `take_checkpoints` retires it.
    pub(crate) fn in_flight(&self) -> bool {
        self.command.is_some()
    }

    fn idle(&self) -> Result<(), SampleError> {
        if self.in_flight() || self.faulted {
            return Err(SampleError::Driver(
                "slot pool is in flight or faulted".into(),
            ));
        }
        Ok(())
    }

    fn buffer(&self, index: u64) -> Result<&metal::BufferRef, SampleError> {
        self.buffers
            .iter()
            .find(|(i, _)| *i == index)
            .map(|(_, b)| b.as_ref())
            .ok_or_else(|| SampleError::Driver(format!("missing slot buffer {index}")))
    }

    fn read_pointer<T>(&self, index: u64, count: usize) -> Result<*const T, SampleError> {
        let buffer = self.buffer(index)?;
        checked_read_pointer(buffer.contents(), buffer.length(), count)
    }

    fn write<T: Copy>(&self, index: u64, offset: usize, values: &[T]) -> Result<(), SampleError> {
        let buf = self.buffer(index)?;
        let bytes = std::mem::size_of_val(values);
        let offset = offset
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| SampleError::Driver("slot write offset overflow".into()))?;
        let contents = buf.contents().cast::<u8>();
        if contents.is_null()
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > buf.length() as usize)
        {
            return Err(SampleError::Driver(
                "slot buffer write exceeds storage".into(),
            ));
        }
        // SAFETY: the caller holds the idle pool exclusively. The checked range
        // lies in shared storage and values is a separate host allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                contents.add(offset),
                bytes,
            );
        }
        Ok(())
    }

    /// Validate topology exactly during coefficient construction. Callers can
    /// admit directly without a prior matches query or a topology hash.
    #[cfg(test)]
    pub(crate) fn admit(&mut self, job: SlotJob) -> Result<SlotId, SampleError> {
        self.idle()?;
        job.validate(self.sched_stride)?;
        let (couplings, fields) =
            fill_h_j_matching(&self.cached.topo, &self.cached.edges, &job.graph)
                .ok_or_else(|| SampleError::Driver("job topology differs from slot pool".into()))?;
        self.upload(job, &couplings, &fields)
    }

    /// Validation and quantization have already run on a preparation worker.
    pub(crate) fn admit_prepared(
        &mut self,
        inputs: PreparedInputs,
        schedule: Arc<[f32]>,
        checkpoints: Vec<usize>,
        seed: u64,
    ) -> Result<SlotId, SampleError> {
        self.idle()?;
        if schedule.len() > self.sched_stride {
            return Err(SampleError::TooLarge("schedule exceeds slot stride".into()));
        }
        let nodes = inputs.graph.num_nodes();
        if !self.matches_prepared(&inputs.edges, nodes, self.num_reads) {
            return Err(SampleError::Driver(
                "job topology differs from slot pool".into(),
            ));
        }
        let job = SlotJob {
            graph: inputs.graph,
            schedule,
            checkpoints,
            seed,
        };
        self.upload(job, &inputs.couplings, &inputs.fields)
    }

    fn upload(
        &mut self,
        job: SlotJob,
        couplings: &[i8],
        fields: &[i8],
    ) -> Result<SlotId, SampleError> {
        let slot = self
            .slots
            .iter()
            .position(Option::is_none)
            .ok_or_else(|| SampleError::TooLarge("slot pool is full".into()))?;
        self.write(2, slot * self.cached.topo.nnz, couplings)?;
        self.write(15, slot * self.cached.n, fields)?;
        let held = self.held[slot]
            .as_ref()
            .is_some_and(|held| Arc::ptr_eq(held, &job.schedule));
        if !held {
            self.write(9, slot * self.sched_stride, &job.schedule)?;
            self.held[slot] = Some(Arc::clone(&job.schedule));
        }
        self.slots[slot] = Some(ResidentJob {
            job,
            position: 0,
            next_checkpoint: 0,
            has_output: false,
            device_us: 0,
        });
        Ok(slot)
    }

    pub(crate) fn release(&mut self, slot: SlotId) -> Result<(), SampleError> {
        self.idle()?;
        let entry = self
            .slots
            .get_mut(slot)
            .filter(|s| s.is_some())
            .ok_or_else(|| SampleError::Driver("slot is not live".into()))?;
        *entry = None;
        Ok(())
    }

    /// Advance every live job, stopping each at its next checkpoint. Release
    /// jobs at their final checkpoint before submitting another step.
    pub(crate) fn commit_step(&mut self, slice: usize) -> Result<bool, SampleError> {
        self.idle()?;
        if self.live() == 0 {
            return Ok(false);
        }
        if slice == 0 {
            return Err(SampleError::Driver("slot slice must be nonzero".into()));
        }
        let slice = sampler::msa_step_limit(self.cached.n, self.live() * self.words, slice);
        self.steps.clear();
        for (slot, resident) in self.slots.iter().enumerate() {
            let Some(r) = resident else {
                continue;
            };
            let Some(&checkpoint) = r.job.checkpoints.get(r.next_checkpoint) else {
                return Err(SampleError::Driver(
                    "release final-checkpoint slots before stepping".into(),
                ));
            };
            let count = slice.min(checkpoint - r.position);
            self.steps.push(SlotStep {
                slot: slot as u32,
                beta_start: r.position as i32,
                beta_count: count as i32,
                num_betas: r.job.schedule.len() as i32,
                seed: slot_seed(r.job.seed),
                flags: if r.position + count == checkpoint {
                    SLOT_WRITE_OUTPUT
                } else {
                    0
                },
            });
        }
        self.write(25, 0, &self.steps)?;
        let command = self.queue.new_command_buffer().to_owned();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&self.cached.row), 0);
        encoder.set_buffer(1, Some(&self.cached.col), 0);
        for (index, buffer) in &self.buffers {
            encoder.set_buffer(*index, Some(buffer), 0);
        }
        for (i, buffer) in self.cached.colors.iter().enumerate() {
            encoder.set_buffer(16 + i as u64, Some(buffer), 0);
        }
        for (index, value) in [
            (3, 0),
            (4, self.cached.topo.nnz as i32),
            (5, self.cached.n as i32),
            (6, self.sched_stride as i32),
            (7, 1),
            (8, 0),
            (12, (self.steps.len() * self.words) as i32),
            (13, self.capacity() as i32),
            (14, self.num_reads as i32),
            (19, self.words as i32),
            (20, self.cached.topo.colors.num_colors),
            (21, 0),
            (22, 0),
        ] {
            encoder.set_bytes(index, 4, (&value as *const i32).cast());
        }
        encoder.set_threadgroup_memory_length(0, ((self.cached.n * 4).div_ceil(16) * 16) as u64);
        encoder.dispatch_thread_groups(
            MTLSize::new((self.steps.len() * self.words) as u64, 1, 1),
            MTLSize::new(self.threads as u64, 1, 1),
        );
        encoder.end_encoding();
        command.commit();
        self.command = Some(command);
        Ok(true)
    }

    pub(crate) fn wait(&self) {
        if let Some(command) = &self.command {
            command.wait_until_completed();
        }
    }

    pub(crate) fn take_checkpoints(&mut self) -> Result<Vec<Checkpoint>, SampleError> {
        let Some(command) = &self.command else {
            self.idle()?;
            return Ok(Vec::new());
        };
        let status = command.status();
        if status == MTLCommandBufferStatus::Error {
            self.faulted = true;
            self.command = None;
            return Err(SampleError::Driver("slot command buffer failed".into()));
        }
        if status != MTLCommandBufferStatus::Completed {
            return Err(SampleError::Driver(
                "slot command buffer has not completed".into(),
            ));
        }
        let device_us = sampler::gpu_time_us(command);
        let energies = self.read_pointer::<i32>(11, self.capacity() * self.num_reads)?;
        self.command = None;
        let output_count = self
            .steps
            .iter()
            .filter(|step| step.flags & SLOT_WRITE_OUTPUT != 0)
            .count();
        let mut checkpoints = Vec::with_capacity(output_count);
        for (index, step) in self.steps.iter().enumerate() {
            let slot = step.slot as usize;
            let Some(r) = self.slots[slot].as_mut() else {
                continue;
            };
            r.position += step.beta_count as usize;
            // Attribute equal shares, distributing the integer remainder so
            // per-job totals reconcile exactly with command-buffer time.
            r.device_us =
                r.device_us
                    .saturating_add(device_time_share(device_us, self.steps.len(), index));
            if step.flags & SLOT_WRITE_OUTPUT != 0 {
                let index = r.next_checkpoint;
                r.next_checkpoint += 1;
                r.has_output = true;
                let last = r.next_checkpoint == r.job.checkpoints.len();
                // SAFETY: the completed output step initialized this live
                // slot's region. read_pointer checked the mapping and byte
                // length. No host mutation or GPU write overlaps this copy.
                let slot_energies = unsafe {
                    std::slice::from_raw_parts(energies.add(slot * self.num_reads), self.num_reads)
                }
                .to_vec();
                let best = slot_energies
                    .iter()
                    .copied()
                    .min()
                    .map(i64::from)
                    .ok_or_else(|| SampleError::Driver("slot has no energies".into()))?;
                checkpoints.push(Checkpoint {
                    slot,
                    index,
                    last,
                    best,
                });
            }
        }
        Ok(checkpoints)
    }

    /// Read the latest checkpoint, which remains valid between output steps.
    #[cfg(test)]
    pub(crate) fn reads(
        &self,
        slot: SlotId,
        num_reads: usize,
    ) -> Result<Vec<SamplerResult>, SampleError> {
        self.reads_many(&[slot], num_reads)?
            .pop()
            .ok_or_else(|| SampleError::Driver("missing slot read result".into()))
    }

    /// Harvest checkpoint outputs in requested slot order with one dense
    /// decode across jobs. Repeated slots produce repeated results.
    pub(crate) fn reads_many(
        &self,
        slots: &[SlotId],
        num_reads: usize,
    ) -> Result<Vec<Vec<SamplerResult>>, SampleError> {
        self.idle()?;
        if num_reads > self.num_reads {
            return Err(SampleError::TooLarge(
                "read count exceeds slot capacity".into(),
            ));
        }
        let graphs: Vec<_> = slots
            .iter()
            .map(|&slot| {
                self.slots
                    .get(slot)
                    .and_then(Option::as_ref)
                    .filter(|r| r.has_output)
                    .map(|r| &r.job.graph)
                    .ok_or_else(|| SampleError::Driver("slot has no checkpoint output".into()))
            })
            .collect::<Result<_, _>>()?;
        if slots.is_empty() {
            return Ok(Vec::new());
        }
        let packed_size = self.cached.n.div_ceil(8);
        let count = slots
            .len()
            .checked_mul(num_reads)
            .filter(|&count| {
                count
                    .checked_mul(packed_size.max(4))
                    .is_some_and(|bytes| bytes <= isize::MAX as usize)
            })
            .ok_or_else(|| {
                SampleError::TooLarge("slot read output exceeds host capacity".into())
            })?;
        let mut packed = Vec::with_capacity(count * packed_size);
        let mut energies = Vec::with_capacity(count);
        let sample_ptr =
            self.read_pointer::<i8>(10, self.capacity() * self.num_reads * packed_size)?;
        let energy_ptr = self.read_pointer::<i32>(11, self.capacity() * self.num_reads)?;
        let mut first = 0;
        while first < slots.len() {
            let mut end = first + 1;
            // Full-read contiguous slots use one bulk copy per output buffer.
            // Sparse slots and read prefixes are gathered into the same dense layout.
            if num_reads == self.num_reads {
                while end < slots.len() && slots[end] == slots[end - 1] + 1 {
                    end += 1;
                }
            }
            let read_count = (end - first) * num_reads;
            let read_offset = slots[first] * self.num_reads;
            // SAFETY: all requested slots have completed checkpoint output and
            // idle() excludes GPU writes. Only validated slot regions are read,
            // including when uninitialized or released slots lie between them.
            // read_pointer checked non-null mappings and full buffer lengths.
            unsafe {
                packed.extend_from_slice(std::slice::from_raw_parts(
                    sample_ptr.add(read_offset * packed_size),
                    read_count * packed_size,
                ));
                energies.extend_from_slice(std::slice::from_raw_parts(
                    energy_ptr.add(read_offset),
                    read_count,
                ));
            }
            first = end;
        }
        let n = self.cached.n;
        // Byte expansion avoids per-spin shifts. Keep the same energy audit.
        // Bound both ends of leaf size: a minimum alone can leave 160 reads
        // on one worker, whose remaining sequential work cannot be stolen.
        let decoded = self.decode.install(|| {
            use rayon::prelude::*;
            (0..count)
                .into_par_iter()
                .with_min_len(16)
                .with_max_len(32)
                .map(|index| {
                    let start = index * packed_size;
                    SamplerResult {
                        spins: unpack_slot_spins(&packed[start..start + packed_size], n),
                        energy_milli: i64::from(energies[index]),
                    }
                })
                .collect::<Vec<_>>()
        });
        let mut decoded = decoded.into_iter();
        let reads: Vec<Vec<_>> = graphs
            .iter()
            .map(|_| decoded.by_ref().take(num_reads).collect())
            .collect();
        sampler::audit_device_energies(&reads, &graphs)?;
        Ok(reads)
    }

    pub(crate) fn device_us(&self, slot: SlotId) -> u64 {
        self.slots
            .get(slot)
            .and_then(Option::as_ref)
            .map_or(0, |r| r.device_us)
    }
}

impl Drop for SlotPool {
    fn drop(&mut self) {
        self.wait();
        for (_, buffer) in self.buffers.drain(..) {
            self.pool.give(buffer);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlotStep {
    pub(crate) slot: u32,
    pub(crate) beta_start: i32,
    pub(crate) beta_count: i32,
    pub(crate) num_betas: i32,
    pub(crate) seed: u32,
    pub(crate) flags: u32,
}

pub(crate) const SLOT_WRITE_OUTPUT: u32 = 1;

pub(crate) fn slot_seed(seed: u64) -> u32 {
    ((seed ^ (seed >> 32)) as u32).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cascade::stage_array;

    #[test]
    #[expect(
        clippy::print_stderr,
        reason = "host decode timing checks the byte expansion optimization"
    )]
    fn slot_unpack_matches_every_byte_and_partial_tail() {
        for byte in 0..=255u8 {
            for n in 0usize..=33 {
                let packed = vec![byte as i8; n.div_ceil(8)];
                assert_eq!(
                    unpack_slot_spins(&packed, n),
                    sampler::unpack_spins(&packed, n)
                );
            }
        }
        let packed = vec![0x5a; 4577usize.div_ceil(8)];
        let start = std::time::Instant::now();
        for _ in 0..20000 {
            std::hint::black_box(sampler::unpack_spins(std::hint::black_box(&packed), 4577));
        }
        let old = start.elapsed();
        let start = std::time::Instant::now();
        for _ in 0..20000 {
            std::hint::black_box(unpack_slot_spins(std::hint::black_box(&packed), 4577));
        }
        eprintln!(
            "unpack 4577 spins, 20000 reads: per-bit={old:?}, byte-table={:?}",
            start.elapsed()
        );
    }

    #[test]
    fn device_time_shares_conserve_command_time() {
        for count in [1, 2, 20, 40] {
            for total in [0, 1, 19, 20, 21, 1250243] {
                let shares: Vec<_> = (0..count)
                    .map(|index| device_time_share(total, count, index))
                    .collect();
                assert_eq!(shares.iter().sum::<u64>(), total);
                assert!(shares.iter().max().unwrap() - shares.iter().min().unwrap() <= 1);
            }
        }
    }

    #[test]
    fn prepared_coefficients_match_slot_quantization() {
        let graph = advantage2_system1(7);
        let topology = SelfFeedingTopology::build_with_advantage2_coloring(&graph);
        let host_topology = SelfFeedingTopology::build(&graph);
        let edges = Edges::from(graph.edges.as_slice());
        let inputs = PreparedInputs::new(&graph, &host_topology, &edges).unwrap();
        let (couplings, fields) = fill_h_j_matching(&topology, &graph.edges, &graph).unwrap();
        assert_eq!(inputs.couplings, couplings);
        assert_eq!(inputs.fields, fields);
        let mut other = graph;
        other.edges.swap(0, 1);
        assert!(PreparedInputs::new(&other, &topology, &edges).is_none());
    }
    use crate::metal_device::MetalDevice;
    use crate::sampler::{build_beta_schedule, energy_milli, unpack_spins, MSA_THREADS};
    use crate::topology::{fill_h_j, SelfFeedingTopology};
    use crate::IsingGraph;
    use metal::{MTLCommandBufferStatus, MTLSize};

    const READS: usize = 64;
    const WORDS: usize = 2;

    #[test]
    fn checked_read_pointer_rejects_null_short_and_overflowing_buffers() {
        let mut values = [1i32, 2];
        let ptr = values.as_mut_ptr().cast();
        assert_eq!(
            checked_read_pointer::<i32>(ptr, 8, 2).unwrap(),
            values.as_ptr()
        );
        checked_read_pointer::<i32>(std::ptr::null_mut(), 8, 2).unwrap_err();
        checked_read_pointer::<i32>(ptr, 7, 2).unwrap_err();
        checked_read_pointer::<i32>(ptr, u64::MAX, usize::MAX).unwrap_err();
        checked_read_pointer::<i8>(ptr, u64::MAX, isize::MAX as usize + 1).unwrap_err();
    }

    #[test]
    fn admission_rejects_wrong_edges_and_readback_rejects_short_buffers() {
        let Some(device) = device() else {
            return;
        };
        let mut pool = SlotPool::new(&device, &advantage2_system1(7), READS, 1, 32).unwrap();
        let mut wrong = job(7, 32, vec![32]);
        wrong.graph.edges.swap(0, 1);
        pool.admit(wrong).unwrap_err();
        assert_eq!(pool.live(), 0);
        pool.admit(job(7, 32, vec![32])).unwrap();
        assert!(pool.commit_step(32).unwrap());
        pool.wait();
        let energy_index = pool.buffers.iter().position(|(i, _)| *i == 11).unwrap();
        let original = std::mem::replace(
            &mut pool.buffers[energy_index].1,
            device.new_buffer_from_slice(&[0i32]),
        );
        assert!(matches!(
            pool.take_checkpoints(),
            Err(SampleError::Driver(_))
        ));
        pool.buffers[energy_index].1 = original;
        assert_eq!(pool.take_checkpoints().unwrap().len(), 1);
        pool.buffers[energy_index].1 = device.new_buffer_from_slice(&[0i32]);
        pool.reads_many(&[0], READS).unwrap_err();
        pool.buffers[energy_index].1 = device.new_buffer_from_slice(&vec![0i32; READS]);
        let sample_index = pool.buffers.iter().position(|(i, _)| *i == 10).unwrap();
        pool.buffers[sample_index].1 = device.new_buffer_from_slice(&[0i8]);
        pool.reads_many(&[0], READS).unwrap_err();
    }

    fn job(seed: u64, sweeps: usize, checkpoints: Vec<usize>) -> SlotJob {
        let graph = advantage2_system1(seed);
        let schedule = build_beta_schedule(&graph, sweeps, 1, None).0;
        SlotJob {
            graph,
            schedule: schedule.into(),
            checkpoints,
            seed,
        }
    }

    fn finish_step(pool: &mut SlotPool, slice: usize) -> Vec<Checkpoint> {
        assert!(pool.commit_step(slice).unwrap());
        pool.wait();
        pool.take_checkpoints().unwrap()
    }

    #[test]
    fn pool_matches_the_raw_dispatch() {
        let Some(device) = device() else {
            return;
        };
        let jobs: Vec<_> = [7, 19, 43]
            .into_iter()
            .map(|s| job(s, 64, vec![64]))
            .collect();
        let graphs: Vec<_> = jobs.iter().map(|j| j.graph.clone()).collect();
        let schedules: Vec<_> = jobs.iter().map(|j| j.schedule.to_vec()).collect();
        let steps: Vec<_> = [0, 32]
            .into_iter()
            .map(|start| {
                jobs.iter()
                    .enumerate()
                    .map(|(i, j)| step(i as u32, j.seed, start, 32, 64))
                    .collect()
            })
            .collect();
        let reference = dispatch_slots(&device, &graphs, &schedules, &steps);
        let mut pool = SlotPool::new(&device, &graphs[0], READS, 3, 64).unwrap();
        assert_eq!(pool.capacity(), 3);
        let nodes = graphs[1].num_nodes();
        let edges = Edges::from(graphs[1].edges.as_slice());
        assert!(pool.matches_prepared(&edges, nodes, READS));
        // The second query takes the verified-storage path.
        assert!(pool.matches_prepared(&edges, nodes, READS));
        assert!(!pool.matches_prepared(&edges, nodes, READS + 1));
        let mut swapped = graphs[1].edges.clone();
        swapped.swap(0, 1);
        assert!(!pool.matches_prepared(&Edges::from(swapped.as_slice()), nodes, READS));
        for j in jobs {
            pool.admit(j).unwrap();
        }
        let mut expected_us = [0; 3];
        for index in 0..2 {
            assert!(pool.commit_step(32).unwrap());
            pool.wait();
            let command_us = sampler::gpu_time_us(pool.command.as_ref().unwrap());
            for (slot, us) in expected_us.iter_mut().enumerate() {
                *us += device_time_share(command_us, 3, slot);
            }
            assert_eq!(
                pool.take_checkpoints().unwrap().len(),
                if index == 0 { 0 } else { 3 }
            );
        }
        // SAFETY: both completed steps wrote every live slot's output. The
        // reference lengths are the exact allocated sample and energy counts.
        unsafe {
            let energies = std::slice::from_raw_parts(
                pool.buffer(11).unwrap().contents().cast::<i32>(),
                reference.0.len(),
            );
            let samples = std::slice::from_raw_parts(
                pool.buffer(10).unwrap().contents().cast::<i8>(),
                reference.1.len(),
            );
            assert_eq!(energies, reference.0);
            assert_eq!(samples, reference.1);
        }
        let all_reads = pool.reads_many(&[0, 1, 2], READS).unwrap();
        for (slot, graph) in graphs.iter().enumerate() {
            let packed_size = graph.h.len().div_ceil(8);
            for (r, read) in all_reads[slot].iter().enumerate() {
                let idx = slot * READS + r;
                assert_eq!(read.energy_milli, i64::from(reference.0[idx]));
                assert_eq!(
                    read.spins,
                    unpack_spins(
                        &reference.1[idx * packed_size..(idx + 1) * packed_size],
                        graph.h.len()
                    )
                );
            }
            assert_eq!(pool.device_us(slot), expected_us[slot]);
        }
    }

    #[test]
    fn checkpoints_fire_at_each_boundary_and_last_marks_the_end() {
        let Some(device) = device() else {
            return;
        };
        let mut pool = SlotPool::new(&device, &advantage2_system1(7), READS, 1, 160).unwrap();
        pool.admit(job(7, 160, vec![32, 96, 160])).unwrap();
        for i in 0..5 {
            let checkpoints = finish_step(&mut pool, 32);
            if i % 2 == 0 {
                assert_eq!(checkpoints.len(), 1);
                let cp = &checkpoints[0];
                assert_eq!((cp.slot, cp.index, cp.last), (0, i / 2, i == 4));
                assert_eq!(
                    cp.best,
                    pool.reads(0, READS)
                        .unwrap()
                        .iter()
                        .map(|r| r.energy_milli)
                        .min()
                        .unwrap()
                );
            } else {
                assert!(checkpoints.is_empty());
            }
            assert!(pool.take_checkpoints().unwrap().is_empty());
        }
        pool.release(0).unwrap();
        assert_eq!(pool.live(), 0);
        assert!(!pool.commit_step(32).unwrap());
    }

    #[test]
    fn slot_keeps_a_held_schedule_and_rewrites_a_different_one() {
        let Some(device) = device() else {
            return;
        };
        let graph = advantage2_system1(7);
        let shared: Arc<[f32]> = build_beta_schedule(&graph, 32, 1, None).0.into();
        let halved: Arc<[f32]> = shared.iter().map(|b| b * 0.5).collect();
        let with = |seed: u64, schedule: &Arc<[f32]>| SlotJob {
            graph: graph.clone(),
            schedule: Arc::clone(schedule),
            checkpoints: vec![32],
            seed,
        };
        let run = |pool: &mut SlotPool, job: SlotJob| {
            assert_eq!(pool.admit(job).unwrap(), 0);
            finish_step(pool, 32);
            let reads = pool.reads(0, READS).unwrap();
            pool.release(0).unwrap();
            reads
        };
        let mut pool = SlotPool::new(&device, &graph, READS, 1, 32).unwrap();
        run(&mut pool, with(7, &shared));
        for schedule in [&shared, &halved] {
            let reused = run(&mut pool, with(19, schedule));
            assert!(Arc::ptr_eq(pool.held[0].as_ref().unwrap(), schedule));
            let mut fresh = SlotPool::new(&device, &graph, READS, 1, 32).unwrap();
            let copy: Arc<[f32]> = schedule.to_vec().into();
            let expected = run(&mut fresh, with(19, &copy));
            for (a, b) in reused.iter().zip(&expected) {
                assert_eq!(a.spins, b.spins);
                assert_eq!(a.energy_milli, b.energy_milli);
            }
        }
    }

    #[test]
    fn released_slot_starts_fresh() {
        let Some(device) = device() else {
            return;
        };
        let graph = advantage2_system1(7);
        let mut pool = SlotPool::new(&device, &graph, READS, 1, 64).unwrap();
        pool.admit(job(7, 64, vec![32, 64])).unwrap();
        finish_step(&mut pool, 32);
        pool.release(0).unwrap();
        assert_eq!(pool.admit(job(19, 32, vec![32])).unwrap(), 0);
        pool.reads(0, READS).unwrap_err();
        finish_step(&mut pool, 32);
        let mut fresh = SlotPool::new(&device, &graph, READS, 1, 64).unwrap();
        fresh.admit(job(19, 32, vec![32])).unwrap();
        finish_step(&mut fresh, 32);
        for (a, b) in pool
            .reads(0, READS)
            .unwrap()
            .iter()
            .zip(fresh.reads(0, READS).unwrap())
        {
            assert_eq!(a.spins, b.spins);
            assert_eq!(a.energy_milli, b.energy_milli);
        }
    }

    #[test]
    fn admit_and_release_refuse_while_in_flight() {
        let Some(device) = device() else {
            return;
        };
        let mut pool = SlotPool::new(&device, &advantage2_system1(7), READS, 2, 32).unwrap();
        pool.admit(job(7, 32, vec![32])).unwrap();
        assert!(pool.commit_step(32).unwrap());
        assert!(pool.in_flight());
        pool.admit(job(19, 32, vec![32])).unwrap_err();
        assert!(pool.release(0).is_err());
        pool.reads(0, READS).unwrap_err();
        pool.commit_step(32).unwrap_err();
        assert_eq!(pool.live(), 1);
        pool.wait();
        assert_eq!(pool.take_checkpoints().unwrap().len(), 1);
        assert!(!pool.in_flight());
    }

    #[test]
    fn over_long_schedule_is_too_large() {
        let Some(device) = device() else {
            return;
        };
        let mut pool = SlotPool::new(&device, &advantage2_system1(7), READS, 1, 64).unwrap();
        assert!(matches!(
            pool.admit(job(7, 65, vec![65])),
            Err(crate::sampler::SampleError::TooLarge(_))
        ));
        assert_eq!(pool.live(), 0);
    }

    #[test]
    fn reads_carry_device_energies_equal_to_host_scoring() {
        let Some(device) = device() else {
            return;
        };
        let graph = advantage2_system1(7);
        let mut pool = SlotPool::new(&device, &graph, READS, 1, 32).unwrap();
        pool.admit(job(7, 32, vec![32])).unwrap();
        finish_step(&mut pool, 32);
        for read in pool.reads(0, READS).unwrap() {
            assert_eq!(
                read.energy_milli,
                energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges)
            );
        }
    }

    #[test]
    fn invalid_jobs_fail_validation_without_a_device() {
        let mut j = job(7, 64, vec![32, 64]);
        j.validate(64).unwrap();
        assert!(matches!(j.validate(63), Err(SampleError::TooLarge(_))));
        for checkpoints in [
            vec![],
            vec![0, 64],
            vec![32, 32, 64],
            vec![64, 32],
            vec![32],
        ] {
            j.checkpoints = checkpoints;
            j.validate(64).unwrap_err();
        }
        j.checkpoints = vec![64];
        Arc::make_mut(&mut j.schedule)[0] = f32::NAN;
        j.validate(64).unwrap_err();
        Arc::make_mut(&mut j.schedule)[0] = -1.0;
        j.validate(64).unwrap_err();
        Arc::make_mut(&mut j.schedule)[0] = 0.1;
        j.graph.h[0] = 0.5;
        j.validate(64).unwrap_err();
    }

    #[test]
    fn pool_clamps_steps_and_preserves_checkpoint_reads() {
        let Some(device) = device() else {
            return;
        };
        let graph = advantage2_system1(7);
        let mut pool = SlotPool::new(&device, &graph, 33, 1, 64).unwrap();
        pool.admit(job(7, 64, vec![5, 64])).unwrap();
        pool.admit(job(19, 32, vec![32])).unwrap_err();
        pool.commit_step(0).unwrap_err();
        pool.release(1).unwrap_err();
        let cp = finish_step(&mut pool, 32);
        assert_eq!((cp[0].index, cp[0].last), (0, false));
        let before = pool.reads(0, 33).unwrap();
        pool.reads(0, 34).unwrap_err();
        assert_eq!(pool.reads(0, 1).unwrap().len(), 1);
        assert!(finish_step(&mut pool, 32).is_empty());
        for (a, b) in before.iter().zip(pool.reads(0, 33).unwrap()) {
            assert_eq!(a.spins, b.spins);
            assert_eq!(a.energy_milli, b.energy_milli);
        }
        let cp = finish_step(&mut pool, 32);
        assert_eq!((cp[0].index, cp[0].last), (1, true));
        for read in pool.reads(0, 33).unwrap() {
            assert_eq!(
                read.energy_milli,
                energy_milli(&read.spins, &graph.h, &graph.j, &graph.edges)
            );
        }
        pool.commit_step(32).unwrap_err();
    }

    #[test]
    fn reads_many_preserves_requested_slot_and_read_order() {
        let Some(device) = device() else {
            return;
        };
        let mut pool = SlotPool::new(&device, &advantage2_system1(7), 33, 4, 32).unwrap();
        for seed in [7, 19, 43] {
            pool.admit(job(seed, 32, vec![32])).unwrap();
        }
        pool.reads_many(&[0], 33).unwrap_err();
        assert!(pool.commit_step(32).unwrap());
        pool.reads_many(&[0], 33).unwrap_err();
        pool.wait();
        assert_eq!(pool.take_checkpoints().unwrap().len(), 3);
        let dense = pool.reads_many(&[0, 1, 2], 33).unwrap();
        pool.release(1).unwrap();
        let sparse = pool.reads_many(&[2, 0, 2], 17).unwrap();
        for (reads, slot) in sparse.iter().zip([2, 0, 2]) {
            assert_eq!(reads.len(), 17);
            for (a, b) in reads.iter().zip(&dense[slot]) {
                assert_eq!(a.spins, b.spins);
                assert_eq!(a.energy_milli, b.energy_milli);
            }
        }
        assert!(pool.reads_many(&[], 33).unwrap().is_empty());
        assert!(pool.reads_many(&[0], 0).unwrap()[0].is_empty());
        pool.reads_many(&[0], 34).unwrap_err();
        pool.reads_many(&[0, 1], 33).unwrap_err();
        pool.reads_many(&[3], 33).unwrap_err();
        pool.reads_many(&[4], 33).unwrap_err();
    }

    const AGLAIS_NODES: usize = 4577;
    const AGLAIS_EDGES: usize = 41514;
    const AGLAIS_MISSING_EDGE: (usize, usize) = (880, 2695);
    const AGLAIS_ALLOWED_H: [i32; 1] = [0];
    const AGLAIS_ALLOWED_J: [i32; 2] = [-1000, 1000];

    fn aglais_edges() -> Vec<(usize, usize)> {
        let mut edges = Vec::with_capacity(AGLAIS_EDGES);
        for line in include_str!("../tests/fixtures/advantage2-system1.edges").lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut nodes = line.split_whitespace();
            let u: usize = nodes.next().expect("edge start").parse().expect("node id");
            let v: usize = nodes.next().expect("edge end").parse().expect("node id");
            assert!(nodes.next().is_none(), "two node ids per edge");
            assert!(u < AGLAIS_NODES && v < AGLAIS_NODES, "fixture node range");
            if (u, v) != AGLAIS_MISSING_EDGE {
                edges.push((u, v));
            }
        }
        assert_eq!(edges.len(), AGLAIS_EDGES);
        edges
    }

    /// The exact instance the chain would validate for `seed`.
    fn instance(seed: [u8; 32], edges: &[(usize, usize)]) -> IsingGraph {
        use quip_solver_core::quip_protocol::chacha8::draw_ising_milli;

        let (h, j) = draw_ising_milli(
            seed,
            AGLAIS_NODES,
            edges.len(),
            &AGLAIS_ALLOWED_H,
            &AGLAIS_ALLOWED_J,
        )
        .expect("draw");
        assert!(h.iter().all(|v| AGLAIS_ALLOWED_H.contains(v)));
        assert!(j.iter().all(|v| AGLAIS_ALLOWED_J.contains(v)));
        let milli = |v: &i32| f64::from(*v) / 1000.0;
        IsingGraph::new(
            h.iter().map(milli).collect(),
            j.iter().map(milli).collect(),
            edges.to_vec(),
        )
    }

    fn next_drawn_seed(state: &mut u64) -> [u8; 32] {
        let mut seed = [0u8; 32];
        for word in seed.as_chunks_mut::<8>().0 {
            *word = xorshift64(state).to_le_bytes();
        }
        seed
    }

    fn draw_seeds(run_seed: u64, count: usize) -> Vec<[u8; 32]> {
        let mut state = run_seed | 1;
        (0..count).map(|_| next_drawn_seed(&mut state)).collect()
    }

    fn parse_seed(hex: &str) -> [u8; 32] {
        assert_eq!(hex.len(), 64, "32-byte hex seed");
        assert!(hex.is_ascii(), "ASCII hex seed");
        let mut seed = [0u8; 32];
        for (i, byte) in seed.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("hex seed");
        }
        seed
    }

    fn read_seeds(path: &str) -> Vec<[u8; 32]> {
        std::fs::read_to_string(path)
            .expect("read seeds file")
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(parse_seed)
            .collect()
    }

    fn segment_csv_header(intermediate: bool) -> &'static str {
        if intermediate {
            "nonce,seed,best_64x32,best_64x256,best_64x14336"
        } else {
            "nonce,seed,best_64x32,best_64x14336"
        }
    }

    fn assert_segment_csv_row(header: &str, line: &str) {
        let columns: Vec<_> = header.split(',').collect();
        let cells: Vec<_> = line.split(',').collect();
        assert_eq!(cells.len(), columns.len());
        let seed_index = columns
            .iter()
            .position(|&column| column == "seed")
            .expect("seed column");
        let seed = cells[seed_index];
        assert_eq!(seed.len(), 64, "64-hex instance seed");
        assert!(
            seed.bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "lowercase hex instance seed"
        );
    }

    fn segment_csv_row(nonce: usize, seed: &[u8; 32], energies: &[Option<i64>]) -> String {
        use std::fmt::Write;

        let mut line = format!("{nonce},");
        for byte in seed {
            write!(line, "{byte:02x}").unwrap();
        }
        for energy in energies {
            write!(line, ",{}", energy.expect("checkpoint energy in milli")).unwrap();
        }
        line
    }

    #[test]
    fn seeded_segment_csv_requires_instance_seed() {
        for intermediate in [false, true] {
            let header = segment_csv_header(intermediate);
            assert_eq!(header.split(',').nth(1), Some("seed"));
            let energies = if intermediate { "-1,-2,-3" } else { "-1,-3" };
            let seed = "0123456789abcdef".repeat(4);
            let values = if intermediate {
                vec![Some(-1), Some(-2), Some(-3)]
            } else {
                vec![Some(-1), Some(-3)]
            };
            let line = segment_csv_row(7, &parse_seed(&seed), &values);
            assert_eq!(line, format!("7,{seed},{energies}"));
            assert_segment_csv_row(header, &line);
            for invalid in [
                String::new(),
                "0".repeat(63),
                "0".repeat(65),
                "A".repeat(64),
                "g".repeat(64),
            ] {
                assert!(std::panic::catch_unwind(|| {
                    assert_segment_csv_row(header, &format!("7,{invalid},{energies}"));
                })
                .is_err());
            }
            let missing = header.replace(",seed", "");
            assert!(std::panic::catch_unwind(|| {
                assert_segment_csv_row(&missing, &format!("7,{energies}"));
            })
            .is_err());
        }
    }

    #[test]
    #[ignore = "seeded segment calibration; run only on the controller's GPU"]
    #[expect(
        clippy::print_stderr,
        reason = "calibration progress for the task monitor"
    )]
    fn seeded_segments_calibration() {
        use std::io::{BufWriter, Write};
        use std::time::Instant;

        let Some(device) = device() else {
            return;
        };
        let count: usize = std::env::var("QUIP_SEGMENTS_NONCES")
            .map(|v| v.parse().expect("QUIP_SEGMENTS_NONCES integer"))
            .unwrap_or(2000);
        assert!(count > 0, "at least one nonce");
        let mut seeds = match std::env::var("QUIP_SEGMENTS_SEEDS") {
            Ok(path) => read_seeds(&path),
            Err(std::env::VarError::NotPresent) => draw_seeds(20260918, count),
            Err(error) => panic!("QUIP_SEGMENTS_SEEDS: {error}"),
        };
        seeds.truncate(count);
        assert!(!seeds.is_empty(), "at least one seed");
        let reheats: Vec<f64> = std::env::var("QUIP_SEGMENTS_REHEAT")
            .unwrap_or_else(|_| "0.10,0.15,0.25".into())
            .split(',')
            .map(|v| v.trim().parse().expect("reheat beta"))
            .collect();
        assert!(reheats.iter().all(|r| r.is_finite() && *r > 0.0));
        let mut names = std::collections::HashSet::new();
        for reheat in &reheats {
            assert!(
                names.insert(format!("{reheat:.2}")),
                "duplicate output name"
            );
        }
        let out = std::path::PathBuf::from(
            std::env::var_os("QUIP_SEGMENTS_OUT").expect("QUIP_SEGMENTS_OUT directory"),
        );
        std::fs::create_dir_all(&out).expect("create output directory");
        let edges = aglais_edges();
        let capacity = crate::streaming::declared_stream_width(Kernel::Msa) / 2;
        let mut pool = SlotPool::new(
            &device,
            &instance(seeds[0], &edges),
            READS,
            capacity,
            14_336,
        )
        .unwrap();
        for reheat in reheats {
            let mut prefix = vec![None; seeds.len()];
            for (name, intermediate) in [("A", false), ("B", true)] {
                let start = Instant::now();
                let file_name = format!("{name}-r{reheat:.2}.csv");
                let mut csv = BufWriter::new(
                    std::fs::File::create(out.join(&file_name)).expect("create calibration CSV"),
                );
                let checkpoints = if intermediate {
                    vec![32, 256, 14_336]
                } else {
                    vec![32, 14_336]
                };
                let header = segment_csv_header(intermediate);
                writeln!(csv, "{header}").unwrap();
                let mut completed = 0;
                for (batch, chunk) in seeds.chunks(pool.capacity()).enumerate() {
                    let mut nonces = vec![None; pool.capacity()];
                    let mut rows = vec![vec![None; checkpoints.len()]; pool.capacity()];
                    for (offset, &seed) in chunk.iter().enumerate() {
                        let nonce = batch * pool.capacity() + offset;
                        let graph = instance(seed, &edges);
                        let schedule = segment_schedule(&graph, reheat, intermediate);
                        let slot = pool
                            .admit(SlotJob {
                                graph,
                                schedule: schedule.into(),
                                checkpoints: checkpoints.clone(),
                                seed: nonce as u64,
                            })
                            .unwrap();
                        nonces[slot] = Some(nonce);
                    }
                    while pool.live() > 0 {
                        // Bound command duration while retaining every rung and checkpoint.
                        for checkpoint in finish_step(&mut pool, 32) {
                            let nonce = nonces[checkpoint.slot].expect("admitted nonce");
                            let row = &mut rows[checkpoint.slot];
                            assert!(row[checkpoint.index].replace(checkpoint.best).is_none());
                            if !checkpoint.last {
                                continue;
                            }
                            assert!(
                                row.iter().all(Option::is_some),
                                "missing checkpoint for {nonce}"
                            );
                            if intermediate {
                                assert_eq!(
                                    row[0], prefix[nonce],
                                    "shared prefix: nonce {nonce}, reheat {reheat}"
                                );
                            } else {
                                prefix[nonce] = row[0];
                            }
                            let line = segment_csv_row(nonce, &seeds[nonce], row);
                            assert_segment_csv_row(header, &line);
                            writeln!(csv, "{line}").unwrap();
                            pool.release(checkpoint.slot).unwrap();
                            completed += 1;
                            if completed % 200 == 0 {
                                csv.flush().unwrap();
                                eprintln!(
                                    "{file_name}: {completed}/{} nonces, wall_seconds={:.3}",
                                    seeds.len(),
                                    start.elapsed().as_secs_f64()
                                );
                            }
                        }
                    }
                }
                assert_eq!(completed, seeds.len());
                csv.flush().unwrap();
                eprintln!(
                    "{file_name}: complete {completed} nonces, wall_seconds={:.3}",
                    start.elapsed().as_secs_f64()
                );
            }
        }
    }

    fn segment_schedule(graph: &IsingGraph, reheat: f64, intermediate: bool) -> Vec<f32> {
        use quip_solver_core::beta::{default_ising_beta_range, geometric_beta_schedule};

        let cold = default_ising_beta_range(graph).1;
        let mut schedule = build_beta_schedule(graph, 32, 1, None).0;
        let segments: &[usize] = if intermediate {
            &[224, 14_080]
        } else {
            &[14_304]
        };
        for &length in segments {
            schedule.extend(
                geometric_beta_schedule(reheat, cold, length)
                    .into_iter()
                    .map(|beta| beta as f32),
            );
        }
        schedule
    }

    #[test]
    fn seeded_segment_schedules_preserve_prefix_and_cold_boundaries() {
        use quip_solver_core::beta::default_ising_beta_range;

        let graph = advantage2_system1(7);
        let standard = build_beta_schedule(&graph, 32, 1, None).0;
        let cold = default_ising_beta_range(&graph).1 as f32;
        for reheat in [0.10, 0.15, 0.25] {
            for intermediate in [false, true] {
                let schedule = segment_schedule(&graph, reheat, intermediate);
                assert_eq!(schedule.len(), 14_336);
                assert_eq!(&schedule[..32], standard.as_slice());
                assert_eq!(schedule[31], cold);
                assert_eq!(schedule[32], reheat as f32);
                assert_eq!(schedule[14_335], cold);
                if intermediate {
                    assert_eq!(schedule[255], cold);
                    assert_eq!(schedule[256], reheat as f32);
                }
            }
        }
    }

    #[test]
    #[ignore = "one-gate S4 throughput study; run only on the controller's GPU"]
    #[expect(
        clippy::print_stderr,
        reason = "explicit throughput study reports both measured rates"
    )]
    fn pool_probe_rate_against_run_stream() {
        use crate::cascade::{CascadeSettings, Controller, Ticket};
        use crate::resident::{Preparation, PREP_BOUND, PREP_WORKERS};
        use crate::sampler::{encode_batch, harvest_batch, EncodedBatch};
        use quip_solver_core::StreamJob;
        use std::time::{Duration, Instant};

        fn timed<T>(total: &mut Duration, action: impl FnOnce() -> T) -> T {
            let start = Instant::now();
            let result = action();
            *total += start.elapsed();
            result
        }

        let Some(device) = device() else {
            return;
        };
        const STEPS: usize = 400;
        const SWEEPS: usize = 32;
        // MSA's nominal read count is 64, so half the declared two-batch
        // stream width is exactly batch_size_for_reads(Msa, READS).
        assert_eq!(crate::METAL_MSA_ADAPT.min_reads as usize, READS);
        let capacity = crate::streaming::declared_stream_width(Kernel::Msa) / 2;
        let templates: Vec<_> = (0..capacity)
            .map(|i| advantage2_system1(i as u64 + 1))
            .collect();
        let mut pools = [
            SlotPool::new(&device, &templates[0], READS, capacity, SWEEPS).unwrap(),
            SlotPool::new(&device, &templates[0], READS, capacity, SWEEPS).unwrap(),
        ];
        let settings = CascadeSettings {
            stages: stage_array(&[SWEEPS]),
            ..Default::default()
        };
        let mut controller = Controller::new(settings);
        let mut tickets: [Vec<Option<Ticket>>; 2] =
            std::array::from_fn(|_| (0..capacity).map(|_| None).collect());
        let mut preparation = Preparation::new().unwrap();
        let mut submitted = 0;
        let mut feed = |preparation: &mut Preparation| {
            while preparation.len() < PREP_BOUND && submitted < STEPS * capacity {
                let index = submitted;
                preparation.submit(
                    StreamJob {
                        job_id: index.to_le_bytes().to_vec(),
                        graph: templates[index % capacity].clone(),
                        params: SampleParams {
                            num_reads: READS,
                            num_sweeps: SWEEPS,
                            sweeps_per_beta: 1,
                            seed: index as u64 + 1,
                            ..Default::default()
                        },
                        watermark: None,
                    },
                    settings,
                );
                submitted += 1;
            }
        };
        let mut pool_host = [Duration::ZERO; 7];
        let mut pool_gpu_us = 0u64;
        let mut attributed_gpu_us = 0u64;
        let start = Instant::now();
        timed(&mut pool_host[6], || feed(&mut preparation));
        for batch_index in 0..STEPS + 2 {
            let pool = &mut pools[batch_index % 2];
            let tickets = &mut tickets[batch_index % 2];
            if pool.in_flight() {
                timed(&mut pool_host[2], || pool.wait());
                pool_gpu_us += sampler::gpu_time_us(pool.command.as_ref().unwrap());
                let checkpoints = timed(&mut pool_host[3], || pool.take_checkpoints().unwrap());
                assert_eq!(checkpoints.len(), capacity);
                let slots: Vec<_> = checkpoints
                    .iter()
                    .map(|checkpoint| {
                        assert!(checkpoint.last);
                        attributed_gpu_us += pool.device_us(checkpoint.slot);
                        checkpoint.slot
                    })
                    .collect();
                let reads = timed(&mut pool_host[4], || {
                    pool.reads_many(&slots, READS).unwrap()
                });
                for (slot, reads) in slots.into_iter().zip(reads) {
                    timed(&mut pool_host[5], || {
                        controller.finish(
                            &tickets[slot].take().unwrap(),
                            reads.iter().map(|r| r.energy_milli).min(),
                            true,
                        );
                        pool.release(slot).unwrap();
                    });
                    std::hint::black_box(reads);
                }
            }
            if batch_index < STEPS {
                for _ in 0..capacity {
                    let prepared = timed(&mut pool_host[6], || {
                        let prepared = preparation.next().unwrap();
                        feed(&mut preparation);
                        prepared
                    });
                    let mut data = prepared.data.unwrap().unwrap();
                    timed(&mut pool_host[0], || {
                        let edges = Arc::clone(data.inputs.edges());
                        let ticket = controller
                            .admit_prepared(&prepared.job, &mut data.schedule, &edges)
                            .unwrap();
                        let slot = pool
                            .admit_prepared(
                                data.inputs,
                                data.schedule.betas,
                                data.schedule.checkpoints,
                                prepared.job.params.seed,
                            )
                            .unwrap();
                        tickets[slot] = Some(ticket);
                    });
                }
                assert!(timed(&mut pool_host[1], || pool
                    .commit_step(SWEEPS)
                    .unwrap()));
            }
        }
        let pool_seconds = start.elapsed().as_secs_f64();
        assert_eq!(
            attributed_gpu_us, pool_gpu_us,
            "slot time must conserve command time"
        );
        drop(preparation);
        let mut batches: [Option<(EncodedBatch, Vec<IsingGraph>)>; 2] = [None, None];
        let mut stream_host = [Duration::ZERO; 4];
        let mut stream_gpu_us = 0u64;
        let start = Instant::now();
        for batch_index in 0..STEPS + 2 {
            let pending = &mut batches[batch_index % 2];
            if let Some((batch, graphs)) = pending.take() {
                timed(&mut stream_host[2], || batch.wait_until_completed());
                stream_gpu_us += batch.gpu_time_us();
                assert!(batch.failed_status().is_none());
                let refs: Vec<_> = graphs.iter().collect();
                let reads = timed(&mut stream_host[3], || {
                    harvest_batch(&batch, &refs).unwrap()
                });
                std::hint::black_box(reads);
            }
            if batch_index < STEPS {
                let graphs = templates.clone();
                let refs: Vec<_> = graphs.iter().collect();
                let params = SampleParams {
                    num_reads: READS,
                    num_sweeps: SWEEPS,
                    sweeps_per_beta: 1,
                    seed: (batch_index * capacity + 1) as u64,
                    ..Default::default()
                };
                let mut batch = timed(&mut stream_host[0], || {
                    encode_batch(&device, &refs, &params, Kernel::Msa, 2).unwrap()
                });
                assert_eq!(batch.chunk_count(), 1);
                assert!(timed(&mut stream_host[1], || batch.commit_next(|| false)));
                *pending = Some((batch, graphs));
            }
        }
        let stream_seconds = start.elapsed().as_secs_f64();
        let jobs = (STEPS * capacity) as f64;
        let pool_rate = jobs / pool_seconds;
        let stream_rate = jobs / stream_seconds;
        eprintln!("S4 one gate: {STEPS} steps, capacity={capacity}, reads={READS}, sweeps={SWEEPS}, two in flight, no later gates, controller admission, {PREP_WORKERS} preparation workers, no warmup");
        eprintln!("pool={pool_rate:.2} jobs/s ({pool_seconds:.3}s), run_stream-equivalent={stream_rate:.2} jobs/s ({stream_seconds:.3}s), ratio={:.4}", pool_rate / stream_rate);
        for (phase, elapsed) in [
            "admit",
            "commit_step",
            "wait",
            "take_checkpoints",
            "reads",
            "release",
            "preparation",
        ]
        .into_iter()
        .zip(pool_host)
        {
            eprintln!("pool host {phase}: {:.6}s total", elapsed.as_secs_f64());
        }
        for (phase, elapsed) in [
            "encode_batch",
            "commit_next",
            "wait_until_completed",
            "harvest_batch",
        ]
        .into_iter()
        .zip(stream_host)
        {
            eprintln!("stream host {phase}: {:.6}s total", elapsed.as_secs_f64());
        }
        eprintln!(
            "pool summed GPU time: {pool_gpu_us} us; stream summed GPU time: {stream_gpu_us} us"
        );
        eprintln!(
            "pool attributed slot GPU time: {attributed_gpu_us} us, {:.3} us/job",
            attributed_gpu_us as f64 / jobs
        );
        eprintln!(
            "host other (job copies, cleanup, timer overhead): pool={:.6}s stream={:.6}s",
            pool_seconds - pool_host.iter().sum::<Duration>().as_secs_f64(),
            stream_seconds - stream_host.iter().sum::<Duration>().as_secs_f64()
        );
        eprintln!("Host phases are exclusive wall intervals, including blocking waits. GPU time overlaps host work; do not add it to host totals. Result destruction is in host other. Pool construction is outside timing.");
        eprintln!("Preparation includes submission and ordered result waits. Worker validation, schedule construction, quantization, and graph copies overlap other host phases and GPU work.");
        let read_ratio = pool_host[4].as_secs_f64() / stream_host[3].as_secs_f64();
        eprintln!("decode host time: pool={:.2} us/job, stream={:.2} us/job, ratio={read_ratio:.4}, target<=1.2",
            pool_host[4].as_secs_f64() * 1_000_000.0 / jobs,
            stream_host[3].as_secs_f64() * 1_000_000.0 / jobs);
        assert!(
            read_ratio <= 1.2,
            "pool reads exceed 1.2x batch harvest time per job"
        );
        assert!(
            pool_rate >= 0.95 * stream_rate,
            "S4 missed: profile admit and reads before Task 3"
        );
    }

    fn device() -> Option<MetalDevice> {
        if MetalDevice::device_count() == 0 {
            #[expect(clippy::print_stderr, reason = "device tests report a sandbox skip")]
            {
                eprintln!("skipping slot device test: no Metal device");
            }
            return None;
        }
        Some(MetalDevice::open(0).expect("Metal device and pipelines"))
    }

    fn step(slot: u32, seed: u64, start: i32, count: i32, total: i32) -> SlotStep {
        SlotStep {
            slot,
            beta_start: start,
            beta_count: count,
            num_betas: total,
            seed: slot_seed(seed),
            flags: if start + count >= total {
                SLOT_WRITE_OUTPUT
            } else {
                0
            },
        }
    }

    fn slot_coefficients(topo: &SelfFeedingTopology, graphs: &[IsingGraph]) -> (Vec<i8>, Vec<i8>) {
        let mut h = Vec::new();
        let mut j = Vec::new();
        for graph in graphs {
            assert_eq!(graph.edges, graphs[0].edges);
            let (couplings, fields) = fill_h_j(topo, graph);
            assert_eq!(fields.len(), topo.n);
            assert_eq!(couplings.len(), topo.nnz);
            h.extend(fields);
            j.extend(couplings);
        }
        (h, j)
    }

    #[test]
    fn slot_coefficients_match_field_and_csr_strides() {
        let graphs = [
            IsingGraph::new(vec![1.0, 0.0, -1.0], vec![1.0, -1.0], vec![(0, 1), (1, 2)]),
            IsingGraph::new(vec![0.0, -1.0, 1.0], vec![-1.0, 1.0], vec![(0, 1), (1, 2)]),
        ];
        let topo = SelfFeedingTopology::build(&graphs[0]);
        let (h, j) = slot_coefficients(&topo, &graphs);
        assert_eq!(h.len(), graphs.len() * topo.n);
        assert_eq!(j.len(), graphs.len() * topo.nnz);
        assert_eq!(h, [1, 0, -1, 0, -1, 1]);
        assert_eq!(j, [1, 1, -1, -1, -1, -1, 1, 1]);
    }

    // Each inner slice is one dispatch in a separate command buffer. Storage
    // survives across slices so continuation steps exercise the device state.
    fn dispatch_slots(
        device: &MetalDevice,
        graph_per_slot: &[IsingGraph],
        schedules: &[Vec<f32>],
        steps: &[Vec<SlotStep>],
    ) -> (Vec<i32>, Vec<i8>) {
        let topo = SelfFeedingTopology::build_with_advantage2_coloring(&graph_per_slot[0]);
        let slots = graph_per_slot.len();
        let max_threads = device.msa_slots.max_total_threads_per_threadgroup() as usize;
        let threads_per_group = MSA_THREADS.min(max_threads).max(1);
        let packed_size = topo.n.div_ceil(8);
        let sched_stride = schedules.iter().map(Vec::len).max().unwrap();
        let (h, j) = slot_coefficients(&topo, graph_per_slot);
        let mut schedule = vec![0.0f32; slots * sched_stride];
        for (slot, slot_schedule) in schedules.iter().enumerate() {
            schedule[slot * sched_stride..slot * sched_stride + slot_schedule.len()]
                .copy_from_slice(slot_schedule);
        }
        let samples = device.new_buffer_from_slice(&vec![85i8; slots * READS * packed_size]);
        let energies = device.new_buffer_from_slice(&vec![i32::MIN; slots * READS]);
        let state = device.new_buffer_from_slice(&vec![0u32; slots * WORDS * topo.n]);
        let rng = device.new_buffer_from_slice(&vec![0u32; slots * WORDS * threads_per_group * 4]);
        let buffers = [
            (0, device.new_buffer_from_slice(&topo.row_ptr)),
            (1, device.new_buffer_from_slice(&topo.col_ind)),
            (2, device.new_buffer_from_slice(&j)),
            (3, device.new_buffer_from_slice(&[0i32])),
            (9, device.new_buffer_from_slice(&schedule)),
            (15, device.new_buffer_from_slice(&h)),
            (16, device.new_buffer_from_slice(&topo.colors.starts)),
            (17, device.new_buffer_from_slice(&topo.colors.counts)),
            (18, device.new_buffer_from_slice(&topo.colors.nodes)),
        ];
        for dispatch in steps {
            let table = device.new_buffer_from_slice(dispatch);
            let command = device.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&device.msa_slots);
            for (index, buffer) in &buffers {
                encoder.set_buffer(*index, Some(buffer), 0);
            }
            for (index, buffer) in [
                (10, &samples),
                (11, &energies),
                (23, &state),
                (24, &rng),
                (25, &table),
            ] {
                encoder.set_buffer(index, Some(buffer), 0);
            }
            for (index, value) in [
                (4, topo.nnz as i32),
                (5, topo.n as i32),
                (6, sched_stride as i32),
                (7, 1),
                (8, 0),
                (12, (dispatch.len() * WORDS) as i32),
                (13, slots as i32),
                (14, READS as i32),
                (19, WORDS as i32),
                (20, topo.colors.num_colors),
                (21, 0),
                (22, 0),
            ] {
                encoder.set_bytes(index, 4, (&value as *const i32).cast());
            }
            encoder.set_threadgroup_memory_length(0, ((topo.n * 4).div_ceil(16) * 16) as u64);
            encoder.dispatch_thread_groups(
                MTLSize::new((dispatch.len() * WORDS) as u64, 1, 1),
                MTLSize::new(threads_per_group as u64, 1, 1),
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        }
        // SAFETY: shared buffers hold these initialized element counts, and all
        // commands have completed before the host reads or drops the buffers.
        unsafe {
            (
                std::slice::from_raw_parts(energies.contents().cast::<i32>(), slots * READS)
                    .to_vec(),
                std::slice::from_raw_parts(
                    samples.contents().cast::<i8>(),
                    slots * READS * packed_size,
                )
                .to_vec(),
            )
        }
    }

    #[test]
    fn slot_step_layout_matches_msl() {
        assert_eq!(std::mem::size_of::<SlotStep>(), 24);
        assert_eq!(std::mem::align_of::<SlotStep>(), 4);
        assert_eq!(std::mem::offset_of!(SlotStep, slot), 0);
        assert_eq!(std::mem::offset_of!(SlotStep, beta_start), 4);
        assert_eq!(std::mem::offset_of!(SlotStep, beta_count), 8);
        assert_eq!(std::mem::offset_of!(SlotStep, num_betas), 12);
        assert_eq!(std::mem::offset_of!(SlotStep, seed), 16);
        assert_eq!(std::mem::offset_of!(SlotStep, flags), 20);
    }

    #[test]
    fn slot_seed_is_never_zero() {
        assert_eq!(slot_seed(0), 1);
        assert_eq!(slot_seed(1 << 32 | 1), 1);
        assert_ne!(slot_seed(7), 0);
    }

    #[test]
    fn slot_energies_equal_host_energy_of_packed_samples() {
        let Some(device) = device() else {
            return;
        };
        let graphs: Vec<_> = [7, 19, 43].into_iter().map(advantage2_system1).collect();
        let schedules: Vec<_> = graphs
            .iter()
            .map(|g| build_beta_schedule(g, 32, 1, None).0)
            .collect();
        let steps = vec![vec![
            step(2, 43, 0, 32, 32),
            step(0, 7, 0, 32, 32),
            step(1, 19, 0, 32, 32),
        ]];
        let (energies, samples) = dispatch_slots(&device, &graphs, &schedules, &steps);
        for (slot, graph) in graphs.iter().enumerate() {
            let packed_size = graph.h.len().div_ceil(8);
            for read in 0..READS {
                let idx = slot * READS + read;
                let spins = unpack_spins(
                    &samples[idx * packed_size..(idx + 1) * packed_size],
                    graph.h.len(),
                );
                assert_eq!(
                    i64::from(energies[idx]),
                    energy_milli(&spins, &graph.h, &graph.j, &graph.edges)
                );
            }
        }
    }

    #[test]
    fn a_job_gives_the_same_samples_in_any_slot_and_beside_any_neighbours() {
        let Some(device) = device() else {
            return;
        };
        let x = advantage2_system1(7);
        let sched = build_beta_schedule(&x, 32, 1, None).0;
        let alone = dispatch_slots(
            &device,
            std::slice::from_ref(&x),
            std::slice::from_ref(&sched),
            &[vec![step(0, 99, 0, 32, 32)]],
        );
        let graphs = vec![advantage2_system1(19), advantage2_system1(43), x];
        let schedules = vec![
            build_beta_schedule(&graphs[0], 16, 1, None).0,
            build_beta_schedule(&graphs[1], 48, 1, None).0,
            sched,
        ];
        let beside = dispatch_slots(
            &device,
            &graphs,
            &schedules,
            &[vec![
                step(2, 99, 0, 32, 32),
                step(0, 11, 0, 16, 16),
                step(1, 22, 0, 48, 48),
            ]],
        );
        assert_eq!(alone.0, beside.0[2 * READS..]);
        assert_eq!(alone.1, beside.1[2 * alone.1.len()..]);
    }

    #[test]
    fn slicing_does_not_change_the_result() {
        let Some(device) = device() else {
            return;
        };
        let graph = advantage2_system1(7);
        let schedule = build_beta_schedule(&graph, 64, 1, None).0;
        let run = |count| {
            let steps: Vec<_> = (0..64)
                .step_by(count)
                .map(|start| vec![step(0, 99, start, count as i32, 64)])
                .collect();
            dispatch_slots(
                &device,
                std::slice::from_ref(&graph),
                std::slice::from_ref(&schedule),
                &steps,
            )
        };
        let full = run(64);
        assert_eq!(full, run(32));
        assert_eq!(full, run(16));
        assert_eq!(full, run(4));
    }

    #[test]
    fn large_first_checkpoint_has_bounded_steps_and_identical_reads() {
        let device = MetalDevice::open(0).unwrap();
        let graph = advantage2_system1(7);
        let limit = sampler::msa_step_limit(graph.num_nodes(), WORDS, sampler::MAX_SWEEPS);
        let first = limit + 17;
        let total = first + 31;
        assert!(total <= sampler::MAX_SWEEPS);
        let schedule = build_beta_schedule(&graph, total, 1, None).0;
        let mut pool = SlotPool::new(&device, &graph, READS, 1, total).unwrap();
        pool.admit(SlotJob {
            graph: graph.clone(),
            schedule: schedule.clone().into(),
            checkpoints: vec![first, total],
            seed: 99,
        })
        .unwrap();
        let mut position = 0;
        let mut commands = 0;
        let mut reference_steps = Vec::new();
        for (index, checkpoint) in [first, total].into_iter().enumerate() {
            let start = if index == 0 { 0 } else { first };
            let mut reference_step = step(
                0,
                99,
                start as i32,
                (checkpoint - start) as i32,
                total as i32,
            );
            reference_step.flags = SLOT_WRITE_OUTPUT;
            reference_steps.push(vec![reference_step]);
            loop {
                let checkpoints = finish_step(&mut pool, first);
                let actual = pool.steps[0];
                assert_eq!(actual.beta_start as usize, position);
                assert!((1..=limit).contains(&(actual.beta_count as usize)));
                position += actual.beta_count as usize;
                commands += 1;
                if position < checkpoint {
                    assert!(checkpoints.is_empty());
                    assert_eq!(actual.flags, 0);
                    continue;
                }
                assert_eq!(position, checkpoint);
                assert_eq!(checkpoints.len(), 1);
                assert_eq!(
                    (checkpoints[0].index, checkpoints[0].last),
                    (index, index == 1)
                );
                let (energies, samples) = dispatch_slots(
                    &device,
                    std::slice::from_ref(&graph),
                    std::slice::from_ref(&schedule),
                    &reference_steps,
                );
                let packed_size = graph.num_nodes().div_ceil(8);
                for (read, actual) in pool.reads(0, READS).unwrap().iter().enumerate() {
                    assert_eq!(actual.energy_milli, i64::from(energies[read]));
                    assert_eq!(
                        actual.spins,
                        unpack_spins(
                            &samples[read * packed_size..(read + 1) * packed_size],
                            graph.num_nodes()
                        )
                    );
                }
                break;
            }
        }
        assert!(commands > 2);
        assert_eq!(
            &pool.slots[0].as_ref().unwrap().job.schedule[..],
            &schedule[..]
        );
    }

    #[test]
    fn a_step_without_output_preserves_output_buffers() {
        let Some(device) = device() else {
            return;
        };
        let graph = advantage2_system1(7);
        let schedule = build_beta_schedule(&graph, 32, 1, None).0;
        let mut no_output = step(0, 99, 0, 32, 32);
        no_output.flags = 0;
        let (energies, samples) =
            dispatch_slots(&device, &[graph], &[schedule], &[vec![no_output]]);
        assert!(energies.iter().all(|&e| e == i32::MIN));
        assert!(samples.iter().all(|&b| b == 85));
    }

    /// Deterministic xorshift64 for fixture generation.
    fn xorshift64(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// Load compacted, zero-based fixture edges with seeded `J` in {-1, 1} and
    /// `h` in {-1, 0, 1}.
    fn advantage2_system1(seed: u64) -> IsingGraph {
        let mut s = seed | 1;
        let mut edges = Vec::with_capacity(41515);
        let mut j = Vec::with_capacity(41515);
        for line in include_str!("../tests/fixtures/advantage2-system1.edges").lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut nodes = line.split_whitespace();
            let u = nodes.next().expect("edge start").parse().expect("node id");
            let v = nodes.next().expect("edge end").parse().expect("node id");
            assert!(nodes.next().is_none(), "two node ids per edge");
            assert!(u < 4577 && v < 4577, "fixture node range");
            edges.push((u, v));
            j.push(if xorshift64(&mut s) & 1 == 0 {
                1.0
            } else {
                -1.0
            });
        }
        assert_eq!(edges.len(), 41515);
        let h = (0..4577)
            .map(|_| [-1.0, 0.0, 1.0][(xorshift64(&mut s) % 3) as usize])
            .collect();
        IsingGraph::new(h, j, edges)
    }
}
