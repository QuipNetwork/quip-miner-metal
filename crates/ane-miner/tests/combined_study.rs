//! Production-channel quality and combined-router study.

use std::collections::HashMap;
use std::fs::read_to_string;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

#[allow(
    dead_code,
    reason = "shared protocol support has conformance-only helpers"
)]
mod support;

use quip_solver_core::quip_proto::v1::{
    coord_msg, ising_problem, Configure, EdgeList, IsingProblem, JobKind, SetTarget, Shutdown,
    Topology, Welcome,
};
use quip_solver_core::quip_protocol::chacha8::draw_ising_milli;
use quip_solver_core::quip_protocol::wire::encode_i32_le;
use serde::{Deserialize, Serialize};

const NODE_COUNT: usize = 4_577;
const EDGE_COUNT: usize = 41_514;
const TOPOLOGY_HASH: [u8; 32] = [0x51; 32];

#[derive(Clone, Debug, Deserialize)]
struct Block {
    qblock_id: u64,
    nonce_seed: String,
    target_energy_milli: i64,
    winning_energy_milli: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Engine {
    Metal,
    Ane,
    Both,
}

impl Engine {
    fn parse(value: &str) -> Self {
        match value {
            "metal" => Self::Metal,
            "ane" => Self::Ane,
            "both" => Self::Both,
            _ => panic!("QUIP_STUDY_ENGINE must be metal, ane, or both"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Ane => "ane",
            Self::Both => "both",
        }
    }

    fn config(self, sweeps: usize) -> String {
        let (metal, ane) = match self {
            Self::Metal => (true, false),
            Self::Ane => (false, true),
            Self::Both => (true, true),
        };
        format!(
            "num_sweeps = {sweeps}\nenable_metal = {metal}\nenable_ane = {ane}\nutilization = 100\nyielding = false\n"
        )
    }
}

#[derive(Clone)]
struct StudyJob {
    block: Block,
    replicate: usize,
    id: Vec<u8>,
    problem: IsingProblem,
    warmup: bool,
}

#[derive(Serialize)]
struct JobRecord<'a> {
    kind: &'static str,
    block: u64,
    replicate: usize,
    sampler_seed: Option<u64>,
    reads: usize,
    sweeps: usize,
    engine: &'a str,
    color: &'a str,
    best_milli: i64,
    target_milli: i64,
    winner_milli: i64,
    valid_reads: usize,
    dispatch_s: f64,
    completed_s: f64,
    device_us: u64,
    rescore_ok: bool,
    warmup: bool,
}

#[derive(Serialize)]
struct SummaryRecord<'a> {
    kind: &'static str,
    engine: &'a str,
    color: &'a str,
    metal_color: &'a str,
    ane_color: &'a str,
    reads: usize,
    sweeps: usize,
    seeds: usize,
    k0: usize,
    max_blocks: usize,
    startup_s: f64,
    window_wall_s: f64,
    jobs: usize,
    completed: usize,
    valid_jobs: usize,
    rescore_ok: bool,
}

fn blocks() -> Vec<Block> {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/testnet-qblocks-3191-3250.json"
    ))
    .expect("valid compact qblock fixture")
}

fn edges() -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(EDGE_COUNT);
    for line in include_str!("../../../tests/fixtures/advantage2-system1.edges").lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let u = fields.next().expect("edge u").parse().expect("numeric u");
        let v = fields.next().expect("edge v").parse().expect("numeric v");
        assert!(fields.next().is_none(), "two columns per edge");
        if (u, v) != (880, 2695) {
            edges.push((u, v));
        }
    }
    assert_eq!(edges.len(), EDGE_COUNT);
    edges
}

fn hex32(value: &str) -> [u8; 32] {
    assert_eq!(value.len(), 64, "nonce seed must contain 32 bytes");
    let mut bytes = [0; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .expect("nonce seed must be hexadecimal");
    }
    bytes
}

fn problem(block: &Block, reads: usize, sweeps: usize) -> IsingProblem {
    let (h, j) = draw_ising_milli(
        hex32(&block.nonce_seed),
        NODE_COUNT,
        EDGE_COUNT,
        &[0],
        &[-1000, 1000],
    )
    .expect("draw testnet problem");
    IsingProblem {
        graph: Some(ising_problem::Graph::TopologyHash(TOPOLOGY_HASH.to_vec())),
        h_milli_le32: encode_i32_le(&h),
        j_milli_le32: encode_i32_le(&j),
        num_reads: u32::try_from(reads).expect("reads fit u32"),
        num_sweeps: u32::try_from(sweeps).expect("sweeps fit u32"),
        anneal_time_us: 0,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn write_json(out: &mut File, record: &impl Serialize) {
    serde_json::to_writer(&mut *out, record).expect("write JSON record");
    out.write_all(b"\n").expect("terminate JSON record");
    out.flush().expect("flush study receipt");
}

fn route_engines(log: &str) -> HashMap<Vec<u8>, String> {
    let mut routes = HashMap::new();
    for line in log
        .lines()
        .filter(|line| line.contains("combined route completed"))
    {
        let engine = if line.contains("engine=\"metal\"") || line.contains("engine=metal") {
            "metal"
        } else if line.contains("engine=\"ane\"") || line.contains("engine=ane") {
            "ane"
        } else {
            panic!("route completion has no engine: {line}");
        };
        let value = line
            .split("job_id=")
            .nth(1)
            .expect("route completion job_id")
            .split_whitespace()
            .next()
            .expect("route completion job value")
            .trim_matches('"');
        assert!(routes
            .insert(value.as_bytes().to_vec(), engine.into())
            .is_none());
    }
    routes
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    })
}

#[test]
fn compact_fixture_regenerates_the_historical_problem_shape() {
    let fixture = blocks();
    assert_eq!(fixture.len(), 60);
    assert_eq!(fixture.first().expect("first block").qblock_id, 3250);
    assert_eq!(fixture.last().expect("last block").qblock_id, 3191);
    assert_eq!(edges().len(), EDGE_COUNT);
    for (index, block) in fixture.into_iter().enumerate() {
        let generated = problem(&block, 128, 16_384);
        assert_eq!(generated.h_milli_le32.len(), NODE_COUNT * 4);
        assert_eq!(generated.j_milli_le32.len(), EDGE_COUNT * 4);
        if index == 0 {
            // Fingerprints of every coefficient in the archived problem-3250.json.
            assert_eq!(fnv1a64(&generated.h_milli_le32), 0x69af_3cd5_1b79_49f5);
            assert_eq!(fnv1a64(&generated.j_milli_le32), 0x57ab_be35_25c5_f8da);
        }
    }
}

#[test]
fn route_log_attributes_each_completion() {
    let routes = route_engines(
        "DEBUG combined route completed engine=\"metal\" job_id=study:3250:1 outcome=completed\n\
         DEBUG combined route completed engine=ane job_id=study:3249:1 outcome=completed\n",
    );
    assert_eq!(routes[b"study:3250:1".as_slice()], "metal");
    assert_eq!(routes[b"study:3249:1".as_slice()], "ane");
}

#[tokio::test]
#[ignore = "requires exclusive Apple GPU and ANE access"]
async fn production_channel_study() {
    let binary = std::env::var("QUIP_STUDY_BIN").expect("QUIP_STUDY_BIN");
    let output = PathBuf::from(std::env::var("QUIP_STUDY_OUT").expect("QUIP_STUDY_OUT"));
    let route_log = output.with_extension("routes.log");
    let engine =
        Engine::parse(&std::env::var("QUIP_STUDY_ENGINE").unwrap_or_else(|_| "both".into()));
    let color = std::env::var("QUIP_STUDY_COLOR").unwrap_or_else(|_| "four".into());
    assert!(matches!(color.as_str(), "greedy" | "four" | "mixed"));
    let metal_color = if color == "mixed" {
        std::env::var("QUIP_STUDY_METAL_COLOR").expect("QUIP_STUDY_METAL_COLOR")
    } else {
        color.clone()
    };
    let ane_color = if color == "mixed" {
        std::env::var("QUIP_STUDY_ANE_COLOR").expect("QUIP_STUDY_ANE_COLOR")
    } else {
        color.clone()
    };
    assert!(matches!(metal_color.as_str(), "greedy" | "four"));
    assert!(matches!(ane_color.as_str(), "greedy" | "four"));
    let reads = env_usize("QUIP_STUDY_READS", 128);
    let sweeps = env_usize("QUIP_STUDY_SWEEPS", 16_384);
    let seeds = env_usize("QUIP_STUDY_SEEDS", 5);
    let k0 = env_usize("QUIP_STUDY_K0", 0);
    let max_blocks = env_usize("QUIP_STUDY_MAX_BLOCKS", 60).min(60);
    let block_fixture: Vec<_> = blocks().into_iter().take(max_blocks).collect();
    let graph_edges = edges();
    let warmups = if engine == Engine::Both { 2 } else { 1 };
    let mut jobs = Vec::with_capacity(max_blocks * seeds + warmups);
    let warmup_block = block_fixture.first().expect("at least one block").clone();
    for index in 0..warmups {
        jobs.push(StudyJob {
            id: format!("warmup:{}:{k0}:{index}", warmup_block.qblock_id).into_bytes(),
            problem: problem(&warmup_block, reads, sweeps),
            block: warmup_block.clone(),
            replicate: k0,
            warmup: true,
        });
    }
    for block in block_fixture {
        for replicate in k0..k0 + seeds {
            jobs.push(StudyJob {
                id: format!("study:{}:{replicate}", block.qblock_id).into_bytes(),
                problem: problem(&block, reads, sweeps),
                block: block.clone(),
                replicate,
                warmup: false,
            });
        }
    }

    let env = vec![
        (
            "RUST_LOG".to_owned(),
            "quip_miner_metal::combined=debug".to_owned(),
        ),
        (
            "QUIP_METAL_MSA_FOUR_COLOR".to_owned(),
            if metal_color == "four" { "1" } else { "0" }.to_owned(),
        ),
        (
            "QUIP_ANE_MSA_FOUR_COLOR".to_owned(),
            if ane_color == "four" { "1" } else { "0" }.to_owned(),
        ),
    ];
    let mut session = support::Session::start_with(
        &binary,
        "combined-study",
        &env,
        Duration::from_secs(600),
        Some(&route_log),
    )
    .await;
    session
        .until("hello", |report| report.hello.is_some())
        .await;
    session
        .send(coord_msg::Msg::Welcome(Welcome {
            protocol_version: 1,
        }))
        .await;
    session
        .send(coord_msg::Msg::Configure(Configure {
            queue_depth: 512,
            idle_timeout_s: 3_600,
            heartbeat_s: 15,
            reconnect_window_s: 60,
            backend_toml: engine.config(sweeps),
        }))
        .await;
    let (u, v): (Vec<_>, Vec<_>) = graph_edges
        .iter()
        .map(|&(u, v)| (u as u32, v as u32))
        .unzip();
    session
        .send(coord_msg::Msg::Topology(Topology {
            hash: TOPOLOGY_HASH.to_vec(),
            nodes: (0..NODE_COUNT as u32).collect(),
            edges: Some(EdgeList { u, v }),
            allowed_h_milli: vec![0],
        }))
        .await;
    session
        .until("configure", |report| {
            report.ready_received && !report.job_request_credits.is_empty()
        })
        .await;
    let credits = session.report.job_request_credits[0] as usize;
    assert!(credits >= warmups);

    let mut by_id = HashMap::new();
    for job in &jobs[..warmups] {
        dispatch(&mut session, job, &graph_edges).await;
        by_id.insert(job.id.clone(), job.clone());
    }
    session
        .until("warmup results", |report| report.results.len() == warmups)
        .await;
    session.refunded("warmup credit").await;
    let warmup_routes = route_engines(&read_to_string(&route_log).expect("warmup route log"));
    let expected_engines: &[&str] = match engine {
        Engine::Metal => &["metal"],
        Engine::Ane => &["ane"],
        Engine::Both => &["metal", "ane"],
    };
    for expected in expected_engines {
        assert!(
            warmup_routes.values().any(|actual| actual == expected),
            "missing {expected} warmup"
        );
    }

    let startup_s = session.elapsed_s();
    let mut sent = warmups;
    let mut completed = warmups;
    while sent < jobs.len() && sent - completed < credits {
        dispatch(&mut session, &jobs[sent], &graph_edges).await;
        by_id.insert(jobs[sent].id.clone(), jobs[sent].clone());
        sent += 1;
    }
    while completed < jobs.len() {
        let before = session.report.results.len();
        session
            .until("study result and credit", |report| {
                report.results.len() > before
                    && report.credits_refunded()
                        >= u64::try_from(report.results.len()).expect("result count fits u64")
            })
            .await;
        completed += session.report.results.len() - before;
        while sent < jobs.len() && sent - completed < credits {
            dispatch(&mut session, &jobs[sent], &graph_edges).await;
            by_id.insert(jobs[sent].id.clone(), jobs[sent].clone());
            sent += 1;
        }
    }
    let window_wall_s = session.elapsed_s() - startup_s;
    session
        .send(coord_msg::Msg::Shutdown(Shutdown { grace_ms: 1_000 }))
        .await;
    session.finish_quiet().await;
    assert_eq!(session.report.exit_code, 0, "miner must exit successfully");
    assert!(session.report.credit_ledger_balanced());
    assert!(
        session
            .report
            .results
            .iter()
            .all(|result| result.meta_sweeps == sweeps as u32),
        "reported sweep budget must match the study"
    );

    let mut out = File::create(&output).expect("create study output");
    let routes = route_engines(&read_to_string(&route_log).expect("read route log"));
    assert_eq!(routes.len(), session.study_results.len());
    let mut valid_jobs = 0;
    for result in &session.study_results {
        let job = &by_id[&result.job_id];
        let actual_engine = &routes[&result.job_id];
        if engine != Engine::Both {
            assert_eq!(actual_engine, engine.name());
        }
        let actual_color = if actual_engine == "metal" {
            &metal_color
        } else {
            &ane_color
        };
        let best = *result.energies.iter().min().expect("at least one result");
        let valid_reads = result
            .energies
            .iter()
            .filter(|&&energy| energy < job.block.target_energy_milli)
            .count();
        valid_jobs += usize::from(valid_reads > 0 && !job.warmup);
        write_json(
            &mut out,
            &JobRecord {
                kind: "job",
                block: job.block.qblock_id,
                replicate: job.replicate,
                sampler_seed: None,
                reads,
                sweeps,
                engine: actual_engine,
                color: actual_color,
                best_milli: best,
                target_milli: job.block.target_energy_milli,
                winner_milli: job.block.winning_energy_milli,
                valid_reads,
                dispatch_s: result.dispatch_s,
                completed_s: result.completed_s,
                device_us: result.device_us,
                rescore_ok: result.rescore_ok,
                warmup: job.warmup,
            },
        );
    }
    write_json(
        &mut out,
        &SummaryRecord {
            kind: "summary",
            engine: engine.name(),
            color: &color,
            metal_color: &metal_color,
            ane_color: &ane_color,
            reads,
            sweeps,
            seeds,
            k0,
            max_blocks,
            startup_s,
            window_wall_s,
            jobs: jobs.len() - warmups,
            completed: session.study_results.len() - warmups,
            valid_jobs,
            rescore_ok: session.study_results.iter().all(|result| result.rescore_ok),
        },
    );
}

async fn dispatch(session: &mut support::Session, job: &StudyJob, graph_edges: &[(usize, usize)]) {
    session
        .send(coord_msg::Msg::SetTarget(SetTarget {
            max_energy_milli: job.block.target_energy_milli - 1,
            min_solutions: 1,
            min_diversity_milli: 0,
            num_reads: job.problem.num_reads,
            num_sweeps: job.problem.num_sweeps,
            anneal_time_us: 0,
        }))
        .await;
    session
        .job(
            &job.id,
            2,
            job.problem.clone(),
            graph_edges,
            JobKind::IsingSample,
            false,
        )
        .await;
}
