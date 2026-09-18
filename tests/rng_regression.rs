//! Public seed 12344 becomes base seed 12345. Read 1 then mixes that
//! with 12345 to produce zero, a fixed point of xorshift32. On this ring,
//! the unguarded shader returns a uniform state at maximum energy.
//! Removing the guard makes the trigger test fail while the control passes.

use quip_miner_metal::metal_device::MetalDevice;
use quip_miner_metal::sampler::sample_ising;
use quip_miner_metal::{IsingGraph, Kernel, SampleParams};

fn open_device() -> MetalDevice {
    MetalDevice::open(0).unwrap_or_else(|e| {
        panic!("Metal device 0 required for rng_regression tests: {e}");
    })
}

const FROZEN_THREAD_SEED: u64 = 12344;

const FROZEN_THREAD_READ: usize = 1;

const CONTROL_SEED: u64 = FROZEN_THREAD_SEED - 1;

fn antiferro_ring(n: usize) -> IsingGraph {
    let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    IsingGraph::new(vec![0.0; n], vec![1.0; edges.len()], edges)
}

#[test]
fn sa_trigger_seed_does_not_freeze_read_one() {
    let dev = open_device();
    let n = 6;
    let graph = antiferro_ring(n);
    let params = SampleParams {
        num_reads: 16,
        num_sweeps: 1024,
        seed: FROZEN_THREAD_SEED,
        ..Default::default()
    };
    let results = sample_ising(&dev, &graph, &params, Kernel::Sa).expect("sa");
    assert_eq!(results.len(), 16);

    let read1 = &results[FROZEN_THREAD_READ];
    let max_energy = 1000 * n as i64;

    assert!(
        read1.spins.contains(&-1) && read1.spins.contains(&1),
        "read 1 ({FROZEN_THREAD_READ}) is uniform; a zero xorshift32 state froze this stream: {:?}",
        read1.spins
    );
    assert!(
        read1.energy_milli < max_energy,
        "read 1 ({FROZEN_THREAD_READ}) energy {} is the frozen uniform maximum {}",
        read1.energy_milli,
        max_energy
    );
}

#[test]
fn sa_control_seed_reads_all_live() {
    let dev = open_device();
    let n = 6;
    let graph = antiferro_ring(n);
    let params = SampleParams {
        num_reads: 16,
        num_sweeps: 1024,
        seed: CONTROL_SEED,
        ..Default::default()
    };
    let results = sample_ising(&dev, &graph, &params, Kernel::Sa).expect("sa");
    let max_energy = 1000 * n as i64;
    for (i, r) in results.iter().enumerate() {
        assert!(
            r.energy_milli < max_energy,
            "control read {i} energy {} hit the frozen maximum {} under a non-trigger seed",
            r.energy_milli,
            max_energy
        );
    }
    let ground = -1000 * n as i64;
    assert!(
        results.iter().any(|r| r.energy_milli == ground),
        "no read reached the antiferro ground {ground}: {:?}",
        results.iter().map(|r| r.energy_milli).collect::<Vec<_>>()
    );
}
