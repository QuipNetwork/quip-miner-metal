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
//! `QUIP_SCREEN_CASCADE=1` admits one job per nonce to the public MSA sampler
//! with the cascade on, instead of the stage loop. Admission stops after
//! `QUIP_SCREEN_SECONDS` seconds (default 1800) or, when `QUIP_SCREEN_NONCES`
//! is set, after that many jobs, whichever comes first. A `QUIP_SCREEN_SEEDS`
//! file still replaces the draw and caps the run. `QUIP_SCREEN_FULL` is the
//! sweep budget (default 14336) and every job asks for 64 reads. The job seed
//! is `job_seed(0, nonce)`, the same derivation the stage loop uses for its
//! first stage. `QUIP_SCREEN_CASCADE_TOML` supplies extra settings, with
//! `cascade = true` forced for the study. `QUIP_G5_MODE` is `system` (default) or `one-gate`.
//! One-gate mode overrides the backend stages with `[32]` and keep with 2000.
//! Set `QUIP_SCREEN_FULL=32` to measure only the gate at its minimum budget.
//! `QUIP_SCREEN_SECOND_SECONDS` (default 0, off) runs a second stream on the
//! same sampler and producer threads. Nonce indices and drawn seeds continue
//! from the first stream. The nonce cap applies only to the first stream,
//! while a seed file caps both streams at its end. Both streams share the CSV.
//! Rates print every 10 seconds as `cascade:` and `cascade-2:`.
//! The CSV columns are
//! `nonce,seed,best,reads,device_us,ok`.
//!
//! The instance for a nonce is `draw_ising_milli` on the chain's topology
//! `cbec1eb4…`, which is the committed Advantage2 System 1 fixture without
//! its edge `(880, 2695)`. A random 32-byte seed samples the same instance
//! distribution as a real nonce because a BLAKE3 output is uniform.

#![expect(
    clippy::print_stderr,
    reason = "the study reports stage rates on stderr, like the other benches"
)]

use quip_miner_metal::iokit_gov::UtilGovernor;
use quip_miner_metal::metal_device::MetalDevice;
use quip_miner_metal::streaming::{run_stream, GpuGovernor};
use quip_miner_metal::{IsingGraph, Kernel, MetalSampler};
use quip_solver_core::quip_protocol::chacha8::draw_ising_milli;
use quip_solver_core::{CancelToken, SampleParams, Sampler, StreamJob, StreamOutcome};
use std::io::Write;
use std::sync::{Arc, Mutex};
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

/// Keep the stage-0 seed and mix later stages into the device-visible bits.
fn job_seed(stage_index: usize, nonce: usize) -> u64 {
    let seed = nonce as u64;
    if stage_index == 0 {
        seed
    } else {
        seed ^ 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(stage_index as u64 + 1)
    }
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
    /// How many of the run's nonces this stage takes. Unfiltered, that is a
    /// prefix of the nonces. With `QUIP_SCREEN_FILTER`, it is the lowest-energy
    /// nonces of the previous stage's keep. `None` takes all of them. A rate comparison needs a long shape and a short
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
                        seed: job_seed(stage_index, i),
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
        "  producers, summed over threads: drawing {:.1} s, blocked on the device {:.1} s \
         ({wall_s:.1} s wall); {:.2} ms per instance",
        drawing.as_secs_f64(),
        blocked.as_secs_f64(),
        1000.0 * drawing.as_secs_f64() / count as f64
    );
    (summaries, wall_s, lead_s)
}

fn env_parse<T: std::str::FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be an integer")),
        Err(_) => default,
    }
}

fn screen_producers() -> usize {
    let producers: usize = std::env::var("QUIP_SCREEN_PRODUCERS")
        .map(|value| {
            value
                .parse()
                .expect("QUIP_SCREEN_PRODUCERS must be an integer")
        })
        .unwrap_or(1);
    assert!(producers > 0, "QUIP_SCREEN_PRODUCERS must be positive");
    producers
}

/// `ps -o cputime=` on macOS. Minutes may exceed 59 (`158:16.45` is 158
/// minutes plus 16.45 seconds). A day prefix uses `D-HH:MM:SS`.
fn parse_cputime(text: &str) -> Option<f64> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let (days, clock) = if let Some((days, clock)) = text.split_once('-') {
        (days.parse::<f64>().ok()?, clock)
    } else {
        (0.0, text)
    };
    if !days.is_finite() || days < 0.0 {
        return None;
    }
    let mut parts = clock.split(':');
    let first = parts.next()?;
    let second = parts.next()?;
    let third = parts.next();
    if parts.next().is_some() {
        return None;
    }
    let (hours, minutes, seconds) = if let Some(third) = third {
        (
            first.parse::<f64>().ok()?,
            second.parse::<f64>().ok()?,
            third.parse::<f64>().ok()?,
        )
    } else {
        (0.0, first.parse::<f64>().ok()?, second.parse::<f64>().ok()?)
    };
    if [hours, minutes, seconds]
        .iter()
        .any(|part| !part.is_finite() || *part < 0.0)
    {
        return None;
    }
    Some(days * 86_400.0 + hours * 3_600.0 + minutes * 60.0 + seconds)
}

fn process_cpu_seconds() -> f64 {
    let output = std::process::Command::new("ps")
        .args(["-o", "cputime=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps cputime");
    assert!(
        output.status.success(),
        "ps cputime exited {}",
        output.status
    );
    let text = String::from_utf8(output.stdout).expect("ps cputime utf-8");
    parse_cputime(text.trim()).unwrap_or_else(|| panic!("ps cputime format: {text:?}"))
}

fn names_energy_audit_mismatch(text: &str) -> bool {
    text.contains("device energy") && text.contains("host energy")
}

#[expect(
    clippy::print_stderr,
    reason = "study reports an overridden cascade setting"
)]
fn cascade_backend_toml(mode: &str, extra: &str) -> String {
    assert!(
        mode == "system" || mode == "one-gate",
        "QUIP_G5_MODE must be system or one-gate (or unset), got {mode:?}"
    );
    let mut config: toml::Table = extra
        .parse()
        .expect("QUIP_SCREEN_CASCADE_TOML must be TOML");
    if config.insert("cascade".into(), toml::Value::Boolean(true))
        == Some(toml::Value::Boolean(false))
    {
        eprintln!("cascade study overrides cascade = false with cascade = true");
    }
    if mode == "one-gate" {
        config.remove("cascade_stages");
        config.remove("cascade_keep");
    }
    // Emit root settings before tables so appended overrides remain at the root.
    let tables: toml::Table = config
        .iter()
        .filter(|(_, value)| value.is_table())
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    config.retain(|_, value| !value.is_table());
    let mut text = toml::to_string(&config).expect("backend TOML");
    if mode == "one-gate" {
        text.push_str("cascade_stages = [32]\ncascade_keep = 2000\n");
    }
    text.push_str(&toml::to_string(&tables).expect("backend tables"));
    text
}

#[test]
fn cascade_study_overrides_disabled_cascade() {
    for mode in ["system", "one-gate"] {
        let text = cascade_backend_toml(mode, "cascade = false\ncascade_audit = 7");
        let config: toml::Table = toml::from_str(&text).unwrap();
        assert_eq!(config["cascade"].as_bool(), Some(true));
        assert_eq!(config["cascade_audit"].as_integer(), Some(7));
    }
}

#[test]
fn cascade_modes_build_valid_toml() {
    assert_eq!(cascade_backend_toml("system", ""), "cascade = true\n");
    let config = cascade_backend_toml(
        "one-gate",
        "cascade_stages = [16, 64]\ncascade_keep = 10\ncascade_audit = 7",
    );
    let parsed: toml::Table = toml::from_str(&config).expect("valid TOML");
    assert_eq!(
        parsed["cascade_stages"].as_array().expect("stages").len(),
        1
    );
    assert_eq!(parsed["cascade_stages"][0].as_integer(), Some(32));
    assert_eq!(parsed["cascade_keep"].as_integer(), Some(2000));
    assert_eq!(parsed["cascade_audit"].as_integer(), Some(7));
    assert!(config.ends_with("cascade_stages = [32]\ncascade_keep = 2000\n"));
}

#[test]
#[should_panic(expected = "QUIP_G5_MODE must be system or one-gate")]
fn cascade_mode_rejects_unknown_values() {
    cascade_backend_toml("typo", "");
}

fn cascade_job_id(index: usize, seed: &[u8; 32]) -> Vec<u8> {
    format!("{index}:{}", hex(seed)).into_bytes()
}

fn parse_cascade_job_id(job_id: &[u8]) -> (usize, String) {
    let text = String::from_utf8(job_id.to_vec()).expect("job id utf-8");
    let (index, seed) = text.split_once(':').expect("job id");
    let index = index.parse().expect("nonce index");
    (index, seed.to_owned())
}

struct NonceInner {
    cursor: usize,
    rng: u64,
    listed: Option<Vec<[u8; 32]>>,
}

struct NonceSupply {
    inner: Mutex<NonceInner>,
    start: Instant,
    seconds: u64,
    limit: Option<usize>,
}

impl NonceSupply {
    fn drawn(run_seed: u64, limit: Option<usize>, seconds: u64) -> Self {
        Self {
            inner: Mutex::new(NonceInner {
                cursor: 0,
                rng: run_seed | 1,
                listed: None,
            }),
            start: Instant::now(),
            seconds,
            limit,
        }
    }

    fn listed(seeds: Vec<[u8; 32]>, seconds: u64) -> Self {
        let limit = seeds.len();
        Self {
            inner: Mutex::new(NonceInner {
                cursor: 0,
                rng: 0,
                listed: Some(seeds),
            }),
            start: Instant::now(),
            seconds,
            limit: Some(limit),
        }
    }

    fn restart(&mut self, seconds: u64) {
        self.start = Instant::now();
        self.seconds = seconds;
        self.limit = self
            .inner
            .get_mut()
            .expect("nonce supply")
            .listed
            .as_ref()
            .map(Vec::len);
    }

    /// Next nonce, or `None` once the time limit or the job cap is reached.
    /// The lock covers only the counter and the xorshift step. The caller
    /// draws the instance after this returns.
    fn next_nonce(&self) -> Option<(usize, [u8; 32])> {
        let mut inner = self.inner.lock().expect("nonce supply");
        if self.start.elapsed() >= Duration::from_secs(self.seconds) {
            return None;
        }
        if self.limit.is_some_and(|limit| inner.cursor >= limit) {
            return None;
        }
        let index = inner.cursor;
        let seed = if inner.listed.is_some() {
            inner
                .listed
                .as_ref()
                .and_then(|seeds| seeds.get(index).copied())
                .expect("listed nonce")
        } else {
            next_drawn_seed(&mut inner.rng)
        };
        inner.cursor += 1;
        Some((index, seed))
    }
}

fn print_cascade_summary(
    admitted: usize,
    results: usize,
    errors: usize,
    wall_s: f64,
    cpu_s: f64,
    hits: usize,
) {
    let rate = if wall_s > 0.0 {
        results as f64 / wall_s
    } else {
        0.0
    };
    let cpu_ms = if results > 0 {
        cpu_s * 1000.0 / results as f64
    } else {
        0.0
    };
    eprintln!("jobs admitted: {admitted}");
    eprintln!("results: {results}");
    eprintln!("errors: {errors}");
    eprintln!("wall seconds: {wall_s:.3}");
    eprintln!("jobs per second: {rate:.2}");
    eprintln!("process CPU seconds: {cpu_s:.3}");
    eprintln!("CPU ms per job: {cpu_ms:.3}");
    eprintln!("at or below target: {hits}");
}

/// One full-budget job per nonce through `MetalSampler::sample_stream`.
fn run_cascade_study(target: i64, producers: usize, out_path: &str) {
    const NUM_READS: usize = 64;
    let edges = aglais_edges();
    let sweeps = env_parse("QUIP_SCREEN_FULL", 14_336usize);
    let seconds = env_parse("QUIP_SCREEN_SECONDS", 1_800u64);
    let second_seconds = env_parse("QUIP_SCREEN_SECOND_SECONDS", 0u64);
    let mode = match std::env::var("QUIP_G5_MODE") {
        Ok(mode) => mode,
        Err(std::env::VarError::NotPresent) => "system".to_owned(),
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("QUIP_G5_MODE must be system or one-gate (or unset)")
        }
    };
    let backend_toml = cascade_backend_toml(
        &mode,
        &std::env::var("QUIP_SCREEN_CASCADE_TOML").unwrap_or_default(),
    );
    let mut supply = if let Ok(path) = std::env::var("QUIP_SCREEN_SEEDS") {
        Arc::new(NonceSupply::listed(read_seeds(&path), seconds))
    } else {
        let run_seed = env_or("QUIP_SCREEN_SEED", 20_260_918u64);
        let limit = match std::env::var("QUIP_SCREEN_NONCES") {
            Ok(value) => Some(
                value
                    .parse()
                    .unwrap_or_else(|_| panic!("QUIP_SCREEN_NONCES must be an integer")),
            ),
            Err(_) => None,
        };
        Arc::new(NonceSupply::drawn(run_seed, limit, seconds))
    };
    eprintln!(
        "cascade: {NUM_READS} reads, {sweeps} sweeps, {seconds} s, target {target} milli, output {out_path}, mode={mode} backend_toml={backend_toml:?}"
    );

    let mut out = std::io::BufWriter::new(std::fs::File::create(out_path).expect("create csv"));
    writeln!(out, "nonce,seed,best,reads,device_us,ok").expect("write");

    let (streams_tx, streams_rx) = std::sync::mpsc::channel::<(
        tokio::sync::mpsc::Receiver<StreamJob>,
        tokio::sync::mpsc::Sender<quip_solver_core::StreamResult>,
    )>();
    let worker = std::thread::spawn(move || {
        let device = MetalDevice::open(0).expect("Metal device 0");
        let gov = UtilGovernor::start(0, 100, false);
        let sampler = MetalSampler::new(device, gov, Kernel::Msa);
        sampler.apply_config(&backend_toml);
        for (jobs, results) in streams_rx {
            sampler.sample_stream(jobs, results, CancelToken::default());
        }
    });

    let (admitted_tx, admitted_rx) = std::sync::mpsc::channel();
    let mut producer_inputs = Vec::new();
    let mut producer_threads = Vec::new();
    for _ in 0..producers {
        let edges = edges.clone();
        let admitted_tx = admitted_tx.clone();
        let (input_tx, input_rx) =
            std::sync::mpsc::channel::<(Arc<NonceSupply>, tokio::sync::mpsc::Sender<StreamJob>)>();
        producer_inputs.push(input_tx);
        producer_threads.push(std::thread::spawn(move || {
            for (supply, job_tx) in input_rx {
                let mut admitted = 0usize;
                while let Some((index, seed)) = supply.next_nonce() {
                    let graph = instance(seed, &edges);
                    let job = StreamJob {
                        job_id: cascade_job_id(index, &seed),
                        graph,
                        params: SampleParams {
                            num_reads: NUM_READS,
                            num_sweeps: sweeps,
                            sweeps_per_beta: 1,
                            beta_range: None,
                            // Resident slots continue the job's own RNG stream.
                            // Stage 0 is the derivation `run_stage` uses first.
                            seed: job_seed(0, index),
                        },
                        watermark: None,
                    };
                    if job_tx.blocking_send(job).is_err() {
                        break;
                    }
                    admitted += 1;
                }
                drop(job_tx);
                drop(supply);
                admitted_tx.send(admitted).expect("producer count");
            }
        }));
    }
    drop(admitted_tx);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("rate timer");
    for (stream, duration) in [seconds, second_seconds].into_iter().enumerate() {
        if stream == 1 && duration == 0 {
            break;
        }
        let label = if stream == 0 { "cascade" } else { "cascade-2" };
        let cpu_start = process_cpu_seconds();
        let supply_mut = Arc::get_mut(&mut supply).expect("producers released nonce supply");
        if stream == 1 {
            supply_mut.restart(duration);
        } else {
            supply_mut.start = Instant::now();
        }
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(128);
        let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(128);
        streams_tx.send((job_rx, out_tx)).expect("stream worker");
        for input in &producer_inputs {
            input
                .send((Arc::clone(&supply), job_tx.clone()))
                .expect("producer input");
        }
        drop(job_tx);
        let mut done = 0usize;
        let mut errors = 0usize;
        let mut hits = 0usize;
        let mut last_report = supply.start;
        let mut last_done = 0usize;
        loop {
            let deadline = last_report + Duration::from_secs(10);
            let received = runtime
                .block_on(async { tokio::time::timeout_at(deadline.into(), out_rx.recv()).await });
            let now = Instant::now();
            if now >= deadline {
                let elapsed = now.duration_since(supply.start).as_secs_f64();
                let window =
                    (done - last_done) as f64 / now.duration_since(last_report).as_secs_f64();
                let total = done as f64 / elapsed;
                eprintln!("{label}: t={elapsed:.0} s results={done} window={window:.1}/s total={total:.1}/s");
                last_report = now;
                last_done = done;
            }
            let result = match received {
                Ok(Some(result)) => result,
                Ok(None) => break,
                Err(_) => continue,
            };
            let (index, seed_hex) = parse_cascade_job_id(&result.job_id);
            let device_us = result.device_access_time_us;
            let (best, reads_n, ok) = match result.outcome {
                StreamOutcome::Completed(Ok(reads)) => {
                    let best = reads.iter().map(|read| read.energy_milli).min();
                    (best, reads.len(), 1u8)
                }
                StreamOutcome::Completed(Err(error)) => {
                    let text = error.to_string();
                    if names_energy_audit_mismatch(&text) {
                        let _ = out.flush();
                        panic!("nonce {index} device-energy audit mismatch: {text}");
                    }
                    eprintln!("nonce {index} failed: {text}");
                    (None, 0, 0)
                }
                StreamOutcome::Cancelled => {
                    eprintln!("nonce {index} cancelled");
                    (None, 0, 0)
                }
            };
            if ok == 0 {
                errors += 1;
            }
            if let Some(best) = best {
                if best <= target {
                    hits += 1;
                }
                writeln!(out, "{index},{seed_hex},{best},{reads_n},{device_us},{ok}")
            } else {
                writeln!(out, "{index},{seed_hex},,{reads_n},{device_us},{ok}")
            }
            .expect("write");
            done += 1;
        }
        let wall_s = supply.start.elapsed().as_secs_f64();
        out.flush().expect("flush csv");
        let mut admitted = 0usize;
        for _ in 0..producers {
            admitted += admitted_rx.recv().expect("producer count");
        }
        let cpu_s = process_cpu_seconds() - cpu_start;
        eprintln!("{label}: summary");
        print_cascade_summary(admitted, done, errors, wall_s, cpu_s, hits);
    }
    drop(producer_inputs);
    for producer in producer_threads {
        producer.join().expect("producer");
    }
    drop(streams_tx);
    worker.join().expect("stream worker");
    eprintln!("wrote {out_path}");
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
    // An ignored study runs only on request, so a missing device is a failure,
    // not a skip: a skip would exit 0 without writing the CSV.
    assert!(
        MetalDevice::device_count() > 0,
        "probe-screen study requested but no Metal device is available"
    );
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("quip_miner_metal=info")),
        )
        .with_writer(std::io::stderr)
        .try_init();
    let target: i64 = env_or("QUIP_SCREEN_TARGET", AGLAIS_TARGET_MILLI);
    let producers = screen_producers();
    let out_path = std::env::var("QUIP_SCREEN_OUT").unwrap_or_else(|_| "probe-screen.csv".into());
    if std::env::var("QUIP_SCREEN_CASCADE").ok().as_deref() == Some("1") {
        run_cascade_study(target, producers, &out_path);
        return;
    }
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
    let filter = std::env::var("QUIP_SCREEN_FILTER").ok().map(|value| {
        let fraction: f64 = value.parse().expect("QUIP_SCREEN_FILTER must be a number");
        assert!(
            fraction > 0.0 && fraction <= 1.0,
            "QUIP_SCREEN_FILTER must be in (0, 1]"
        );
        fraction
    });
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

#[test]
fn cputime_parses_the_ps_clock() {
    let micros = |text: &str| {
        let seconds = parse_cputime(text).unwrap_or_else(|| panic!("parse {text}"));
        (seconds * 1_000_000.0).round() as i64
    };
    assert_eq!(micros("0:00.02"), 20_000);
    assert_eq!(micros("158:16.45"), (158 * 60 * 1_000_000) + 16_450_000);
    assert_eq!(
        micros("1-02:03:04"),
        86_400_000_000 + 2 * 3_600_000_000 + 3 * 60_000_000 + 4_000_000
    );
    assert_eq!(
        micros("  2:03:04.50 "),
        2 * 3_600_000_000 + 3 * 60_000_000 + 4_500_000
    );
    assert!(parse_cputime("nope").is_none());
    assert!(parse_cputime("").is_none());
}

#[test]
fn cascade_nonce_supply_matches_the_stage_draw() {
    let supply = NonceSupply::drawn(20_260_918, Some(4), 60);
    let expected = draw_seeds(20_260_918, 4);
    for (index, seed) in expected.iter().enumerate() {
        let (got_index, got_seed) = supply.next_nonce().expect("nonce");
        assert_eq!(got_index, index);
        assert_eq!(got_seed, *seed);
        assert_eq!(job_seed(0, index), index as u64);
        let (parsed_index, parsed_seed) = parse_cascade_job_id(&cascade_job_id(index, seed));
        assert_eq!(parsed_index, index);
        assert_eq!(parsed_seed, hex(seed));
    }
    assert!(supply.next_nonce().is_none());
    assert_eq!(job_seed(1, 5), 5 ^ 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(2));

    let listed = draw_seeds(3, 2);
    let supply = NonceSupply::listed(listed.clone(), 60);
    assert_eq!(supply.next_nonce().expect("listed 0").1, listed[0]);
    assert_eq!(supply.next_nonce().expect("listed 1").1, listed[1]);
    assert!(supply.next_nonce().is_none());

    let stopped = NonceSupply::drawn(1, None, 0);
    assert!(stopped.next_nonce().is_none());

    assert!(names_energy_audit_mismatch(
        "device fault: device energy 1 != host energy 2 for problem 0 read 0"
    ));
    assert!(!names_energy_audit_mismatch(
        "device fault: metal command buffer did not complete: status Error"
    ));
}

#[test]
fn cascade_second_stream_continues_nonce_supply() {
    let mut supply = NonceSupply::drawn(3, Some(2), 60);
    let expected = draw_seeds(3, 4);
    assert_eq!(supply.next_nonce(), Some((0, expected[0])));
    assert_eq!(supply.next_nonce(), Some((1, expected[1])));
    assert!(supply.next_nonce().is_none());
    supply.restart(60);
    assert_eq!(supply.next_nonce(), Some((2, expected[2])));
    assert_eq!(supply.next_nonce(), Some((3, expected[3])));
    supply.restart(0);
    assert!(supply.next_nonce().is_none());

    let mut listed = NonceSupply::listed(expected[..2].to_vec(), 60);
    assert_eq!(listed.next_nonce(), Some((0, expected[0])));
    listed.restart(60);
    assert_eq!(listed.next_nonce(), Some((1, expected[1])));
    assert!(listed.next_nonce().is_none());
}

#[test]
fn stage_seeds_have_distinct_device_bits() {
    for nonce in [0, 1, 7, 42, usize::MAX] {
        let seeds: Vec<_> = (0..4).map(|stage| job_seed(stage, nonce) as u32).collect();
        for i in 0..4 {
            for j in i + 1..4 {
                assert_ne!(seeds[i], seeds[j], "nonce {nonce}, stages {i}/{j}");
            }
        }
    }
}
