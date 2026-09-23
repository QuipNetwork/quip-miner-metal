// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "slot pool dispatch is added in the next task")
)]
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

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "slot pool dispatch is added in the next task")
)]
pub(crate) const SLOT_WRITE_OUTPUT: u32 = 1;

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "slot pool dispatch is added in the next task")
)]
pub(crate) fn slot_seed(seed: u64) -> u32 {
    ((seed ^ (seed >> 32)) as u32).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal_device::MetalDevice;
    use crate::sampler::{build_beta_schedule, energy_milli, unpack_spins, MSA_THREADS};
    use crate::topology::{fill_h_j, SelfFeedingTopology};
    use crate::IsingGraph;
    use metal::{MTLCommandBufferStatus, MTLSize};

    const READS: usize = 64;
    const WORDS: usize = 2;

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
