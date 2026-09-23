// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Probe-then-solve screening study on fresh nonces from the Aglais testnet's
//! instance distribution. `#[ignore]`: an hour of GPU time at the default
//! nonce count.
//!
//! By default every nonce runs through every stage, so the CSV holds the joint
//! distribution of a short probe and the full job on the same instance. The
//! analysis in `scripts/testnet/annealer/screen_yield.py` then scores any keep
//! fraction and threshold from one run.
//!
//! ```sh
//! QUIP_SCREEN_NONCES=50000 QUIP_SCREEN_OUT=screen.csv \
//!   cargo test --release --test probe_screen -- --ignored --nocapture
//! ```
//!
//! Knobs: `QUIP_SCREEN_NONCES` (default 500), `QUIP_SCREEN_SEED` (run seed
//! for the nonce draw, default 20260918), `QUIP_SCREEN_SEEDS` (a file of
//! 32-byte hex nonce seeds, one per line, which replaces the draw),
//! `QUIP_SCREEN_STAGES` (`READSxSWEEPS[xNONCES]` list, default
//! `64x1024,64x14336`; the optional third field caps how many of the run's
//! nonces that stage takes, so shapes of different cost can run for a
//! similar time),
//! `QUIP_SCREEN_FILTER` (optional keep fraction F, 0 < F <= 1; after the
//! first stage, keep the best ceil(F * completed) from the previous stage,
//! then apply the stage nonce cap; skipped CSV cells stay empty),
//! `QUIP_SCREEN_PRODUCERS` (positive thread count for instance draws, default 1),
//! `QUIP_SCREEN_TARGET` (milli, default the Aglais target of 2026-09-18),
//! `QUIP_SCREEN_OUT` (CSV path, default `probe-screen.csv`).
//!
//! The instance for a nonce is `draw_ising_milli` on the chain's topology
//! `cbec1eb4…`, which is the committed Advantage2 System 1 fixture without
//! its edge `(880, 2695)`. A random 32-byte seed samples the same instance
//! distribution as a real nonce because a BLAKE3 output is uniform.

#![expect(
    clippy::print_stderr,
    reason = "the study reports stage rates on stderr, like the other benches"
)]

use quip_miner_metal::metal_device::MetalDevice;
use quip_miner_metal::streaming::{run_stream, GpuGovernor};
use quip_miner_metal::{IsingGraph, Kernel};
use quip_solver_core::quip_protocol::chacha8::draw_ising_milli;
use quip_solver_core::{CancelToken, SampleParams, StreamJob, StreamOutcome};
use std::io::Write;
use std::time::{Duration, Instant};

struct NoGovernor;

impl GpuGovernor for NoGovernor {
    fn should_throttle(&self) -> bool {
        false
    }
    fn budget_scale(&self) -> f64 {
        1.0
    }
    fn record_gpu_busy_us(&self, _us: u64) {}
}

/// Aglais topology `cbec1eb4e9dc…`: 4,577 nodes, 41,514 edges, fields drawn
/// from `{0}` and couplings from `{-1000, 1000}` milli. Checked 2026-09-18
/// against the chain's topology record: the edge list is the fixture's in
/// the same order and orientation, on the same node compaction, without one
/// edge.
const AGLAIS_NODES: usize = 4577;
const AGLAIS_EDGES: usize = 41514;
const AGLAIS_MISSING_EDGE: (usize, usize) = (880, 2695);
const AGLAIS_ALLOWED_H: [i32; 1] = [0];
const AGLAIS_ALLOWED_J: [i32; 2] = [-1000, 1000];
/// The Aglais chain's `current_difficulty` after qblock 3278, fetched
/// 2026-09-18: the target the next block's proof must reach.
const AGLAIS_TARGET_MILLI: i64 = -14_625_068;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The chain's edge list, derived from the committed fixture.
fn aglais_edges() -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(AGLAIS_EDGES);
    for line in include_str!("fixtures/advantage2-system1.edges").lines() {
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
    let (h, j) = draw_ising_milli(
        seed,
        AGLAIS_NODES,
        edges.len(),
        &AGLAIS_ALLOWED_H,
        &AGLAIS_ALLOWED_J,
    )
    .expect("draw");
    let milli = |v: &i32| f64::from(*v) / 1000.0;
    IsingGraph::new(
        h.iter().map(milli).collect(),
        j.iter().map(milli).collect(),
        edges.to_vec(),
    )
}

/// Deterministic xorshift64 for the nonce draw.
fn xorshift64(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn draw_seeds(run_seed: u64, count: usize) -> Vec<[u8; 32]> {
    let mut s = run_seed | 1;
    (0..count)
        .map(|_| {
            let mut seed = [0u8; 32];
            for word in seed.as_chunks_mut::<8>().0 {
                *word = xorshift64(&mut s).to_le_bytes();
            }
            seed
        })
        .collect()
}

fn parse_seed(hex: &str) -> [u8; 32] {
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

fn hex(seed: &[u8; 32]) -> String {
    seed.iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Clone, Copy)]
struct Stage {
    num_reads: usize,
    num_sweeps: usize,
    /// How many of the run's nonces this stage takes, from the front. `None`
    /// takes all of them. A rate comparison needs a long shape and a short
    /// shape to run for a similar time, which needs different counts.
    nonces: Option<usize>,
}

fn parse_stages(spec: &str) -> Vec<Stage> {
    spec.split(',')
        .map(|s| {
            let mut fields = s.trim().split('x');
            let reads = fields.next().expect("READSxSWEEPS[xNONCES]");
            let sweeps = fields.next().expect("READSxSWEEPS[xNONCES]");
            let stage = Stage {
                num_reads: reads.parse().expect("reads"),
                num_sweeps: sweeps.parse().expect("sweeps"),
                nonces: fields.next().map(|n| n.parse().expect("nonces")),
            };
            assert!(fields.next().is_none(), "READSxSWEEPS[xNONCES]");
            stage
        })
        .collect()
}

/// One job's summary: lowest read, median read, reads strictly below the
/// target, which is what a valid proof needs.
#[derive(Clone, Copy)]
struct Summary {
    best: i64,
    median: i64,
    below_target: usize,
}

/// Select original nonce indices, breaking energy ties by nonce index.
fn select_seeds(
    seeds: &[[u8; 32]],
    previous: Option<&[Option<Summary>]>,
    filter: Option<f64>,
    cap: Option<usize>,
) -> Vec<(usize, [u8; 32])> {
    let mut indices: Vec<usize> = if let (Some(previous), Some(fraction)) = (previous, filter) {
        let mut completed: Vec<_> = previous
            .iter()
            .enumerate()
            .filter_map(|(i, summary)| summary.map(|s| (s.best, i)))
            .collect();
        completed.sort_unstable();
        completed.truncate((fraction * completed.len() as f64).ceil() as usize);
        completed.into_iter().map(|(_, i)| i).collect()
    } else {
        (0..seeds.len()).collect()
    };
    indices.truncate(cap.unwrap_or(indices.len()));
    indices.into_iter().map(|i| (i, seeds[i])).collect()
}

/// Run every seed through `stage`, indexed by its original nonce position.
/// Instances are generated on producer threads as the stream takes
/// them, since 50,000 graphs at a megabyte each do not fit in memory.
fn run_stage(
    edges: &[(usize, usize)],
    seeds: &[(usize, [u8; 32])],
    stage: Stage,
    stage_index: usize,
    target: i64,
    producers: usize,
) -> (Vec<Option<Summary>>, f64, f64) {
    if seeds.is_empty() {
        return (Vec::new(), 0.0, 0.0);
    }
    let (job_tx, job_rx) = tokio::sync::mpsc::channel(128);
    let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(128);
    let cancel = CancelToken::default();
    let count = seeds.len();

    // Each stage draws its own RNG stream: a probe and a full job on the same
    // nonce must not share a seed.
    let seed_base = (stage_index as u64) << 32;
    let mut producer_threads = Vec::new();
    for producer_index in 0..producers.min(count) {
        let edges = edges.to_vec();
        let seeds: Vec<_> = seeds
            .iter()
            .skip(producer_index)
            .step_by(producers)
            .copied()
            .collect();
        let job_tx = job_tx.clone();
        // Report how the producer split its time. Drawing one instance copies
        // about a megabyte, so a short probe can starve on the producer rather
        // than the device. `blocked` is time waiting on a full channel, which
        // is the device holding the producer back and the state we want.
        producer_threads.push(std::thread::spawn(move || {
            let (mut drawing, mut blocked) = (Duration::ZERO, Duration::ZERO);
            for (i, seed) in seeds {
                let t = Instant::now();
                let graph = instance(seed, &edges);
                drawing += t.elapsed();
                let job = StreamJob {
                    job_id: i.to_string().into_bytes(),
                    graph,
                    params: SampleParams {
                        num_reads: stage.num_reads,
                        num_sweeps: stage.num_sweeps,
                        sweeps_per_beta: 1,
                        beta_range: None,
                        seed: seed_base + i as u64,
                    },
                    watermark: None,
                };
                let t = Instant::now();
                let sent = job_tx.blocking_send(job);
                blocked += t.elapsed();
                if sent.is_err() {
                    break;
                }
            }
            (drawing, blocked)
        }));
    }
    drop(job_tx);

    let start = Instant::now();
    let worker = std::thread::spawn(move || {
        let device = MetalDevice::open(0).expect("Metal device 0");
        run_stream(&device, Kernel::Msa, job_rx, &out_tx, &NoGovernor, &cancel);
    });
    let size = seeds.iter().map(|(i, _)| i + 1).max().unwrap_or(0);
    let mut summaries: Vec<Option<Summary>> = vec![None; size];
    let mut done = 0usize;
    let mut last_report = Instant::now();
    // Lead time: how long a cold miner waits for its first answer. It covers
    // the device open, the kernel compile and one batch of sweeps, so it is
    // not the steady-state rate turned upside down.
    let mut first_result: Option<Duration> = None;
    while let Some(r) = out_rx.blocking_recv() {
        first_result.get_or_insert_with(|| start.elapsed());
        let index: usize = String::from_utf8_lossy(&r.job_id)
            .parse()
            .expect("job index");
        let reads = match r.outcome {
            StreamOutcome::Completed(Ok(reads)) => reads,
            StreamOutcome::Completed(Err(e)) => {
                eprintln!("nonce {index} failed: {e}");
                continue;
            }
            StreamOutcome::Cancelled => {
                eprintln!("nonce {index} cancelled");
                continue;
            }
        };
        let mut energies: Vec<i64> = reads.iter().map(|s| s.energy_milli).collect();
        energies.sort_unstable();
        summaries[index] = Some(Summary {
            best: energies[0],
            median: energies[energies.len() / 2],
            below_target: energies.iter().filter(|&&e| e < target).count(),
        });
        done += 1;
        if last_report.elapsed().as_secs() >= 60 {
            let elapsed = start.elapsed().as_secs_f64();
            eprintln!(
                "stage {}x{}: {done}/{count} after {elapsed:.0} s, {:.2} jobs/s",
                stage.num_reads,
                stage.num_sweeps,
                done as f64 / elapsed
            );
            last_report = Instant::now();
        }
    }
    let wall_s = start.elapsed().as_secs_f64();
    worker.join().expect("stream worker");
    let lead_s = first_result.unwrap_or_default().as_secs_f64();
    let (mut drawing, mut blocked) = (Duration::ZERO, Duration::ZERO);
    for producer in producer_threads {
        let (draw_time, block_time) = producer.join().expect("producer");
        drawing += draw_time;
        blocked += block_time;
    }
    eprintln!(
        "  producer: drawing {:.1} s, blocked on the device {:.1} s, of {wall_s:.1} s wall;\
         {:.2} ms per instance",
        drawing.as_secs_f64(),
        blocked.as_secs_f64(),
        1000.0 * drawing.as_secs_f64() / count as f64
    );
    (summaries, wall_s, lead_s)
}

/// Positive control for the fixture derivation. `scripts/testnet/regen`
/// writes a problem from the chain's own topology record; this test draws
/// the same seed on the derived edge list and compares. Set
/// `QUIP_SCREEN_CHECK` to the problem JSON and `QUIP_SCREEN_CHECK_SEED` to
/// its nonce seed.
#[test]
#[ignore = "needs a regenerated problem file and its nonce seed"]
fn derived_topology_reproduces_a_regenerated_problem() {
    #[derive(serde::Deserialize)]
    struct Problem {
        h: Vec<f64>,
        j: Vec<f64>,
        edges: Vec<[usize; 2]>,
    }
    let path = std::env::var("QUIP_SCREEN_CHECK").expect("QUIP_SCREEN_CHECK");
    let seed = parse_seed(&std::env::var("QUIP_SCREEN_CHECK_SEED").expect("nonce seed"));
    let problem: Problem =
        serde_json::from_reader(std::fs::File::open(path).expect("open problem"))
            .expect("parse problem");
    let edges = aglais_edges();
    let ours = instance(seed, &edges);
    let theirs: Vec<(usize, usize)> = problem.edges.iter().map(|e| (e[0], e[1])).collect();
    assert_eq!(ours.edges, theirs);
    assert_eq!(ours.h, problem.h);
    assert_eq!(ours.j, problem.j);
}

#[test]
#[ignore = "GPU study: an hour of device time at the default nonce count"]
fn probe_then_solve_on_fresh_nonces() {
    if MetalDevice::device_count() == 0 {
        eprintln!("skipping probe-screen study: no Metal device");
        return;
    }
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("quip_miner_metal=info")),
        )
        .with_writer(std::io::stderr)
        .try_init();
    let seeds = match std::env::var("QUIP_SCREEN_SEEDS") {
        Ok(path) => read_seeds(&path),
        Err(_) => draw_seeds(
            env_or("QUIP_SCREEN_SEED", 20_260_918),
            env_or("QUIP_SCREEN_NONCES", 500),
        ),
    };
    let stages = parse_stages(
        &std::env::var("QUIP_SCREEN_STAGES").unwrap_or_else(|_| "64x1024,64x14336".into()),
    );
    let target: i64 = env_or("QUIP_SCREEN_TARGET", AGLAIS_TARGET_MILLI);
    let filter = std::env::var("QUIP_SCREEN_FILTER").ok().map(|value| {
        let fraction: f64 = value.parse().expect("QUIP_SCREEN_FILTER must be a number");
        assert!(
            fraction > 0.0 && fraction <= 1.0,
            "QUIP_SCREEN_FILTER must be in (0, 1]"
        );
        fraction
    });
    let producers: usize = std::env::var("QUIP_SCREEN_PRODUCERS")
        .map(|value| {
            value
                .parse()
                .expect("QUIP_SCREEN_PRODUCERS must be an integer")
        })
        .unwrap_or(1);
    assert!(producers > 0, "QUIP_SCREEN_PRODUCERS must be positive");
    let out_path = std::env::var("QUIP_SCREEN_OUT").unwrap_or_else(|_| "probe-screen.csv".into());
    let edges = aglais_edges();
    eprintln!(
        "{} nonces, {} stages, target {target} milli, output {out_path}",
        seeds.len(),
        stages.len()
    );

    let mut columns: Vec<Vec<Option<Summary>>> = Vec::with_capacity(stages.len());
    for (k, stage) in stages.iter().enumerate() {
        let run = select_seeds(
            &seeds,
            columns.last().map(Vec::as_slice),
            filter,
            stage.nonces,
        );
        let (mut summaries, wall_s, lead_s) = run_stage(&edges, &run, *stage, k, target, producers);
        let done = summaries.iter().flatten().count();
        eprintln!(
            "stage {}x{}: {done} of {} jobs in {wall_s:.1} s = {:.2} jobs/s; lead {lead_s:.2} s; wall_seconds={wall_s:.6}",
            stage.num_reads,
            stage.num_sweeps,
            run.len(),
            if wall_s > 0.0 { done as f64 / wall_s } else { 0.0 }
        );
        summaries.resize(seeds.len(), None);
        columns.push(summaries);
    }

    let mut out = std::io::BufWriter::new(std::fs::File::create(&out_path).expect("create csv"));
    write!(out, "nonce,seed").expect("write");
    for stage in &stages {
        let tag = format!("{}x{}", stage.num_reads, stage.num_sweeps);
        write!(out, ",best_{tag},median_{tag},below_{tag}").expect("write");
    }
    writeln!(out).expect("write");
    for (i, seed) in seeds.iter().enumerate() {
        write!(out, "{i},{}", hex(seed)).expect("write");
        for column in &columns {
            match column[i] {
                Some(s) => write!(out, ",{},{},{}", s.best, s.median, s.below_target),
                None => write!(out, ",,,"),
            }
            .expect("write");
        }
        writeln!(out).expect("write");
    }
    out.flush().expect("flush csv");
    eprintln!("wrote {out_path}");
}

#[test]
fn filtered_selection_preserves_indices_and_uses_completed_count() {
    let seeds = draw_seeds(7, 6);
    let summary = |best| {
        Some(Summary {
            best,
            median: best,
            below_target: 0,
        })
    };
    let previous = vec![
        summary(30),
        None,
        summary(10),
        summary(20),
        None,
        summary(10),
    ];
    assert_eq!(
        select_seeds(&seeds, Some(&previous), Some(0.5), None),
        vec![(2, seeds[2]), (5, seeds[5])]
    );
    assert_eq!(
        select_seeds(&seeds, Some(&previous), Some(0.01), None),
        vec![(2, seeds[2])]
    );
    assert_eq!(
        select_seeds(&seeds, Some(&previous), Some(1.0), Some(3)),
        vec![(2, seeds[2]), (5, seeds[5]), (3, seeds[3])]
    );
    assert_eq!(
        select_seeds(&seeds, Some(&previous), None, Some(2)),
        vec![(0, seeds[0]), (1, seeds[1])]
    );
    assert_eq!(
        select_seeds(&seeds, None, Some(0.5), Some(2)),
        vec![(0, seeds[0]), (1, seeds[1])]
    );
    assert!(select_seeds(&seeds, Some(&[None; 6]), Some(0.5), None).is_empty());
}
