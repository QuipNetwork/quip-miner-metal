// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Per-checkpoint best energies on the resident cascade schedule.
//!
//! Ignored: needs a Metal device and a problem directory written by
//! `scripts/testnet/regen` or `scripts/testnet/make_fresh.py`. By default every
//! gate is open, so each job reports its best at every checkpoint and at the
//! end. `QUIP_TRACE_OPEN=0` runs the gates, and a screened job's later cells
//! stay empty. Open gates use `QUIP_TRACE_STAGES` on every topology. With the gates
//! closed, the chain topology uses its compiled stages.
//!
//! ```text
//! QUIP_TRACE_PROBLEMS=dir QUIP_TRACE_OUT=trace.csv \
//!   cargo test --release --test gate_trace -- --ignored --nocapture
//! ```
//!
//! Knobs: `QUIP_TRACE_STAGES` (default `8,16,64,256`, the stages for other
//! topologies and the column labels), `QUIP_TRACE_SWEEPS`
//! (final budget, default 512), `QUIP_TRACE_SEEDS` per problem (default 30),
//! `QUIP_TRACE_READS` (default 64).

#![expect(
    clippy::print_stderr,
    reason = "the trace reports its rate on stderr, like the benches"
)]

use quip_miner_metal::iokit_gov::UtilGovernor;
use quip_miner_metal::metal_device::MetalDevice;
use quip_miner_metal::{IsingGraph, Kernel, MetalSampler};
use quip_solver_core::{CancelToken, SampleParams, Sampler, StreamJob, StreamOutcome};
use std::collections::BTreeMap;
use std::io::Write;
use std::sync::{Arc, Mutex};

#[derive(serde::Deserialize)]
struct Entry {
    path: String,
    qblock_id: u64,
}

#[derive(serde::Deserialize)]
struct Problem {
    h: Vec<f64>,
    j: Vec<f64>,
    edges: Vec<(usize, usize)>,
}

#[derive(Clone, Default)]
struct Buffer(Arc<Mutex<Vec<u8>>>);

impl Write for Buffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn seed(qblock: u64, k: u64) -> u64 {
    let mut z = qblock.wrapping_mul(1_000_003).wrapping_add(k);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (z ^ (z >> 31)).max(1)
}

/// `job=Q:K stage=S best=E` fields from one fmt line.
fn parse(line: &str) -> Option<(String, usize, i64)> {
    let field = |key: &str| {
        line.split_whitespace()
            .find_map(|word| word.strip_prefix(key))
            .map(str::to_owned)
    };
    Some((
        field("job=")?,
        field("stage=")?.parse().ok()?,
        field("best=")?.parse().ok()?,
    ))
}

#[test]
#[ignore = "needs a Metal device and QUIP_TRACE_PROBLEMS"]
fn checkpoint_trace() {
    let dir = std::env::var("QUIP_TRACE_PROBLEMS").expect("QUIP_TRACE_PROBLEMS");
    let out_path = std::env::var("QUIP_TRACE_OUT").expect("QUIP_TRACE_OUT");
    let stages: String = env("QUIP_TRACE_STAGES", "8,16,64,256".to_owned());
    let sweeps: usize = env("QUIP_TRACE_SWEEPS", 512);
    let seeds: u64 = env("QUIP_TRACE_SEEDS", 30);
    let reads: usize = env("QUIP_TRACE_READS", 64);

    let buffer = Buffer::default();
    let writer = buffer.clone();
    tracing_subscriber::fmt()
        .with_env_filter("quip_miner_metal::cascade_trace=debug,quip_miner_metal::cascade=info")
        .with_ansi(false)
        .without_time()
        .with_writer(move || writer.clone())
        .init();

    let index: Vec<Entry> =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/index.json")).unwrap())
            .unwrap();
    let mut jobs = Vec::new();
    for entry in &index {
        let p: Problem =
            serde_json::from_str(&std::fs::read_to_string(&entry.path).unwrap()).unwrap();
        let graph = IsingGraph::new(p.h, p.j, p.edges);
        for k in 0..seeds {
            jobs.push(StreamJob {
                job_id: format!("{}:{k}", entry.qblock_id).into_bytes(),
                graph: graph.clone(),
                params: SampleParams {
                    num_reads: reads,
                    num_sweeps: sweeps,
                    sweeps_per_beta: 1,
                    beta_range: None,
                    seed: seed(entry.qblock_id, k),
                },
                watermark: None,
            });
        }
    }
    let count = jobs.len();
    let open = env("QUIP_TRACE_OPEN", 1u8) != 0;
    let config = format!("cascade_stages = [{stages}]\ncascade_open_gates = {open}");

    let (tx, rx) = tokio::sync::mpsc::channel(64);
    let (out, mut results) = tokio::sync::mpsc::channel(count.max(1));
    let runner = std::thread::spawn(move || {
        let device = MetalDevice::open(0).expect("Metal device 0");
        let sampler = MetalSampler::new(device, UtilGovernor::start(0, 100, false), Kernel::Msa);
        sampler.apply_config(&config);
        sampler.sample_stream(rx, out, CancelToken::default());
    });
    let feeder = std::thread::spawn(move || {
        for job in jobs {
            tx.blocking_send(job).unwrap();
        }
    });
    let started = std::time::Instant::now();
    let mut finals = BTreeMap::new();
    while let Some(result) = results.blocking_recv() {
        let StreamOutcome::Completed(Ok(reads)) = result.outcome else {
            panic!(
                "job {} did not complete",
                String::from_utf8_lossy(&result.job_id)
            );
        };
        let best = reads.iter().map(|r| r.energy_milli).min().unwrap();
        finals.insert(String::from_utf8(result.job_id).unwrap(), best);
        if finals.len() == count {
            break;
        }
    }
    feeder.join().unwrap();
    drop(results);
    runner.join().unwrap();
    let wall = started.elapsed().as_secs_f64();

    let text = String::from_utf8(buffer.0.lock().unwrap().clone()).unwrap();
    for line in text.lines().filter(|line| line.contains("cascade report")) {
        eprintln!("{line}");
    }
    let mut rows: BTreeMap<String, Vec<Option<i64>>> = BTreeMap::new();
    let width = stages.split(',').count();
    for (job, stage, best) in text.lines().filter_map(parse) {
        // The last checkpoint is the full budget, which the result carries.
        if stage < width {
            rows.entry(job).or_insert_with(|| vec![None; width])[stage] = Some(best);
        }
    }
    let mut file = std::fs::File::create(&out_path).unwrap();
    let header: Vec<String> = stages.split(',').map(|s| format!("best_{s}")).collect();
    writeln!(file, "qblock_id,k,{},best_{sweeps}", header.join(",")).unwrap();
    for (job, best) in &finals {
        let (qblock, k) = job.split_once(':').unwrap();
        let cells: Vec<String> = rows
            .get(job)
            .map(|v| {
                v.iter()
                    .map(|c| c.map_or(String::new(), |e| e.to_string()))
                    .collect()
            })
            .unwrap_or_else(|| vec![String::new(); width]);
        writeln!(file, "{qblock},{k},{},{best}", cells.join(",")).unwrap();
    }
    eprintln!(
        "{count} jobs in {wall:.1} s, {:.0} jobs/s",
        count as f64 / wall
    );
}

/// Line-buffered file writer shared by the tracing subscriber.
#[derive(Clone)]
struct FileLog(Arc<Mutex<std::io::BufWriter<std::fs::File>>>);

impl Write for FileLog {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().write(bytes)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.lock().unwrap().flush()
    }
}

/// A fresh population through the compiled chain gates.
///
/// Draws `QUIP_POP_COUNT` models on the topology of `QUIP_POP_EDGES` (a
/// problem file) with h = 0 and J = +-1, from seeds derived from
/// `QUIP_POP_SEED`. Every checkpoint line goes to `QUIP_POP_LOG`, and the
/// final best of each job at or below `QUIP_POP_KEEP` (milli) goes to
/// `QUIP_POP_OUT`.
///
/// ```text
/// QUIP_POP_EDGES=problem.json QUIP_POP_LOG=pop.log QUIP_POP_OUT=pop.csv \
///   cargo test --release --test gate_trace population_trace -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore = "needs a Metal device and QUIP_POP_EDGES"]
fn population_trace() {
    use quip_solver_core::quip_protocol::chacha8::draw_ising_milli;

    let edges_path = std::env::var("QUIP_POP_EDGES").expect("QUIP_POP_EDGES");
    let log_path = std::env::var("QUIP_POP_LOG").expect("QUIP_POP_LOG");
    let out_path = std::env::var("QUIP_POP_OUT").expect("QUIP_POP_OUT");
    let count: u64 = env("QUIP_POP_COUNT", 20_000);
    let run_seed: u64 = env("QUIP_POP_SEED", 1);
    let keep: i64 = env("QUIP_POP_KEEP", -14_400_000);
    let feeders: u64 = env("QUIP_POP_FEEDERS", 8);

    let mut log = FileLog(Arc::new(Mutex::new(std::io::BufWriter::new(
        std::fs::File::create(&log_path).unwrap(),
    ))));
    let writer = log.clone();
    tracing_subscriber::fmt()
        .with_env_filter("quip_miner_metal::cascade_trace=debug,quip_miner_metal::cascade=info")
        .with_ansi(false)
        .without_time()
        .with_writer(move || writer.clone())
        .init();

    let p: Problem = serde_json::from_str(&std::fs::read_to_string(&edges_path).unwrap()).unwrap();
    let edges = Arc::new(p.edges);
    let nodes = p.h.len();

    let (tx, rx) = tokio::sync::mpsc::channel(256);
    let (out, mut results) = tokio::sync::mpsc::channel(1024);
    let runner = std::thread::spawn(move || {
        let device = MetalDevice::open(0).expect("Metal device 0");
        let sampler = MetalSampler::new(device, UtilGovernor::start(0, 100, false), Kernel::Msa);
        sampler.apply_config("cascade_open_gates = false");
        sampler.sample_stream(rx, out, CancelToken::default());
    });
    let feeders: Vec<_> = (0..feeders)
        .map(|f| {
            let tx = tx.clone();
            let edges = Arc::clone(&edges);
            std::thread::spawn(move || {
                for i in (f..count).step_by(feeders as usize) {
                    let mut draw = [0u8; 32];
                    for (w, word) in draw.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                        *word = seed(run_seed, i * 4 + w as u64).to_le_bytes();
                    }
                    let (h, j) = draw_ising_milli(draw, nodes, edges.len(), &[0], &[-1000, 1000])
                        .expect("draw");
                    let milli = |v: &i32| f64::from(*v) / 1000.0;
                    let graph = IsingGraph::new(
                        h.iter().map(milli).collect(),
                        j.iter().map(milli).collect(),
                        edges.to_vec(),
                    );
                    let job = StreamJob {
                        job_id: format!("{run_seed}:{i}").into_bytes(),
                        graph,
                        params: SampleParams {
                            num_reads: 64,
                            num_sweeps: 256,
                            sweeps_per_beta: 1,
                            beta_range: None,
                            seed: seed(run_seed ^ 0x5eed, i),
                        },
                        watermark: None,
                    };
                    if tx.blocking_send(job).is_err() {
                        return;
                    }
                }
            })
        })
        .collect();
    drop(tx);

    let started = std::time::Instant::now();
    let mut file = std::fs::File::create(&out_path).unwrap();
    writeln!(file, "job,best").unwrap();
    let mut done = 0u64;
    while let Some(result) = results.blocking_recv() {
        let StreamOutcome::Completed(Ok(reads)) = result.outcome else {
            panic!(
                "job {} did not complete",
                String::from_utf8_lossy(&result.job_id)
            );
        };
        let best = reads.iter().map(|r| r.energy_milli).min().unwrap();
        if best <= keep {
            let job = String::from_utf8_lossy(&result.job_id);
            writeln!(file, "{job},{best}").unwrap();
        }
        done += 1;
        if done.is_multiple_of(100_000) {
            eprintln!(
                "{done} jobs, {:.0} jobs/s",
                done as f64 / started.elapsed().as_secs_f64()
            );
        }
        if done == count {
            break;
        }
    }
    for feeder in feeders {
        feeder.join().unwrap();
    }
    drop(results);
    runner.join().unwrap();
    log.flush().unwrap();
    let wall = started.elapsed().as_secs_f64();
    eprintln!(
        "{count} jobs in {wall:.1} s, {:.0} jobs/s",
        count as f64 / wall
    );
}
