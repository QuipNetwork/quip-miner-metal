//! Command-line and coordinator protocol tests for `quip-ane-msa`.
//!
//! These tests drive the real binary. They do not mock argument parsing.

use std::io::Write;
use std::process::{Command, Stdio};

mod support;

use quip_solver_conformance::driver::CONFIGURED_SWEEPS;
use quip_solver_core::quip_proto::v1::{
    coord_msg, ising_problem, Cancel, Configure, EdgeList, GetCapabilities, IsingProblem, JobKind,
    Ping, RejectReason, Shutdown, Topology, Welcome,
};
use quip_solver_core::quip_protocol::scoring::energy_milli;
use quip_solver_core::quip_protocol::wire::encode_i32_le;

fn miner() -> &'static str {
    env!("CARGO_BIN_EXE_quip-ane-msa")
}

const SOLVE_INPUT: &str = r#"{"h":[0.0,0.0],"j":[1.0],"edges":[[0,1]],"num_reads":33,"num_sweeps":5,"sweeps_per_beta":2,"beta_range":[0.25,4.0],"seed":123}"#;

const SOLVABLE_JOBS: [&[u8]; 4] = [b"job-1", b"job-2", b"job-hash", b"job-sparse"];

fn run_args(args: &[&str]) -> std::process::Output {
    Command::new(miner())
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("run {args:?}: {error}"))
}

fn solve(input: &str) -> std::process::Output {
    let mut child = Command::new(miner())
        .arg("--solve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("spawn --solve: {error}"));
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(input.as_bytes())
        .expect("write problem JSON");
    child
        .wait_with_output()
        .unwrap_or_else(|error| panic!("wait --solve: {error}"))
}

fn parse_solutions(stdout: &[u8]) -> Vec<serde_json::Value> {
    serde_json::from_slice(stdout).unwrap_or_else(|error| {
        panic!(
            "solve stdout is not a JSON array: {error}; stdout={}",
            String::from_utf8_lossy(stdout)
        )
    })
}

fn assert_valid_spins(spins: &[serde_json::Value], nodes: usize) {
    assert_eq!(spins.len(), nodes);
    for spin in spins {
        let value = spin.as_i64().expect("spin must be an integer");
        assert!(
            value == -1 || value == 1,
            "spin {value} is outside {{-1, 1}}"
        );
    }
}

fn assert_solve_error(input: &str) {
    let out = solve(input);
    assert!(
        !out.status.success(),
        "unsupported input must fail, got exit {:?} stdout={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout)
    );
    if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&out.stdout) {
        panic!("unsupported input returned a JSON result instead of an error: {value}");
    }
}

#[test]
fn capabilities_json_matches_ane_msa_identity() {
    let out = run_args(&["--capabilities"]);
    assert!(
        out.status.success(),
        "--capabilities failed: status={:?} stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let value: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("capabilities must be JSON");
    assert_eq!(value["backend"], "ane");
    assert_eq!(value["algorithm"], "msa");
    assert_eq!(value["maxNodes"], 16_384);
    assert_eq!(value["maxEdges"], 163_840);
    assert_eq!(value["streamWidth"], 1);
    assert_eq!(value["protocolVersion"], 1);
    assert_eq!(value["supportedKinds"], serde_json::json!(["ISING_SAMPLE"]));
    let features = value["features"]
        .as_array()
        .expect("features must be an array");
    assert!(
        features.iter().any(|feature| feature == "streaming"),
        "features missing streaming: {features:?}"
    );
}

#[test]
fn version_includes_protocol_one() {
    let out = run_args(&["--version"]);
    assert!(out.status.success(), "--version failed: {:?}", out.status);
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("protocol 1"),
        "--version must name protocol 1: {text}"
    );
}

#[test]
fn check_solve_and_capabilities_are_mutually_exclusive() {
    for pair in [
        ["--check", "--solve"],
        ["--check", "--capabilities"],
        ["--solve", "--capabilities"],
    ] {
        let out = run_args(&pair);
        assert_eq!(
            out.status.code(),
            Some(2),
            "{pair:?} must be a clap conflict, got {:?} stderr={}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("cannot be used with"),
            "{pair:?} stderr must name the conflict: {stderr}"
        );
    }
}

#[test]
fn worker_flag_conflicts_with_core_modes() {
    for flag in ["--capabilities", "--check", "--solve", "--quip-coordinator"] {
        let mut args = vec!["--ane-worker", "1", flag];
        if flag == "--quip-coordinator" {
            args.push("unix:///tmp/quip-ane-unused.sock");
        }
        let out = run_args(&args);
        assert_eq!(
            out.status.code(),
            Some(2),
            "--ane-worker {flag} must conflict, got {:?} stderr={}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
#[ignore = "requires Apple Silicon ANE"]
fn solve_returns_thirty_three_consensus_scored_reads() {
    let first = solve(SOLVE_INPUT);
    assert!(
        first.status.success(),
        "--solve failed: status={:?} stderr={}",
        first.status.code(),
        String::from_utf8_lossy(&first.stderr)
    );
    let solutions = parse_solutions(&first.stdout);
    assert_eq!(solutions.len(), 33);
    let h = [0.0, 0.0];
    let j = [1.0];
    let edges = [(0, 1)];
    for solution in &solutions {
        let spins = solution["spins"].as_array().expect("spins array");
        assert_valid_spins(spins, 2);
        let spin_values: Vec<i8> = spins
            .iter()
            .map(|spin| i8::try_from(spin.as_i64().expect("spin")).expect("spin fits i8"))
            .collect();
        let expected = energy_milli(&spin_values, &h, &j, &edges);
        assert_eq!(
            solution["energy_milli"].as_i64(),
            Some(expected),
            "energy must match the consensus scorer"
        );
    }
    let second = solve(SOLVE_INPUT);
    assert!(second.status.success(), "repeat --solve failed");
    assert_eq!(
        first.stdout, second.stdout,
        "the same --solve input must return identical results"
    );
}

#[test]
#[ignore = "requires Apple Silicon ANE"]
fn solve_accepts_empty_graphs_and_zero_sweeps() {
    for input in [
        r#"{"h":[],"j":[],"edges":[],"num_reads":4,"num_sweeps":5,"sweeps_per_beta":2,"beta_range":[0.25,4.0],"seed":1}"#,
        r#"{"h":[0.0,0.0],"j":[1.0],"edges":[[0,1]],"num_reads":4,"num_sweeps":0,"sweeps_per_beta":2,"beta_range":[0.25,4.0],"seed":1}"#,
    ] {
        let out = solve(input);
        assert!(
            out.status.success(),
            "accepted input failed: {input} status={:?} stderr={}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
        let solutions = parse_solutions(&out.stdout);
        assert_eq!(solutions.len(), 4);
        let nodes = if input.contains(r#""h":[]"#) { 0 } else { 2 };
        for solution in solutions {
            let spins = solution["spins"].as_array().expect("spins array");
            assert_valid_spins(spins, nodes);
        }
    }
}

#[test]
#[ignore = "requires Apple Silicon ANE"]
fn solve_rejects_unsupported_inputs() {
    let degree_21_edges: String = (1..22)
        .map(|v| format!("[0,{v}]"))
        .collect::<Vec<_>>()
        .join(",");
    let degree_21 = format!(
        r#"{{"h":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"j":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"edges":[{degree_21_edges}],"num_reads":1,"num_sweeps":1,"sweeps_per_beta":1,"beta_range":[0.25,4.0],"seed":1}}"#
    );
    assert_solve_error(&degree_21);
    assert_solve_error(
        r#"{"h":[0.0,0.0],"j":[1.0],"edges":[[0,1]],"num_reads":129,"num_sweeps":5,"sweeps_per_beta":2,"beta_range":[0.25,4.0],"seed":1}"#,
    );
    assert_solve_error(
        r#"{"h":[0.0,0.0],"j":[1.0],"edges":[[0,1]],"num_reads":1,"num_sweeps":65537,"sweeps_per_beta":2,"beta_range":[0.25,4.0],"seed":1}"#,
    );
    assert_solve_error(
        r#"{"h":[0.5,0.0],"j":[1.0],"edges":[[0,1]],"num_reads":1,"num_sweeps":5,"sweeps_per_beta":2,"beta_range":[0.25,4.0],"seed":1}"#,
    );
}

#[tokio::test]
#[ignore = "requires Apple Silicon ANE"]
async fn standalone_ane_passes_supported_coordinator_protocol() {
    let inline = IsingProblem {
        graph: Some(ising_problem::Graph::Edges(EdgeList {
            u: vec![0],
            v: vec![1],
        })),
        h_milli_le32: encode_i32_le(&[1000, -1000]),
        j_milli_le32: encode_i32_le(&[1000]),
        num_reads: 1,
        num_sweeps: 0,
        anneal_time_us: 0,
    };
    let dense_hash = vec![0x11; 32];
    let sparse_hash = vec![0x22; 32];
    let mut session = support::Session::start(miner()).await;
    session.until("hello", |r| r.hello.is_some()).await;
    assert!(session.report.handshake_ok);
    session
        .send(coord_msg::Msg::Welcome(Welcome {
            protocol_version: 1,
        }))
        .await;
    session
        .send(coord_msg::Msg::Configure(Configure {
            queue_depth: 3,
            idle_timeout_s: 300,
            heartbeat_s: 15,
            reconnect_window_s: 60,
            backend_toml: format!("num_sweeps = {CONFIGURED_SWEEPS}\n"),
        }))
        .await;
    session
        .send(coord_msg::Msg::Topology(Topology {
            hash: dense_hash.clone(),
            nodes: vec![0, 1],
            edges: Some(EdgeList {
                u: vec![0],
                v: vec![1],
            }),
            allowed_h_milli: vec![-1000, 0, 1000],
        }))
        .await;
    session
        .until("configure", |r| {
            r.ready_received && !r.job_request_credits.is_empty()
        })
        .await;
    assert_eq!(session.report.job_request_credits, [3]);
    session
        .send(coord_msg::Msg::GetCapabilities(GetCapabilities {}))
        .await;
    session
        .until("capabilities", |r| r.capabilities_received.is_some())
        .await;
    let before = session.report.statuses.len();
    session.send(coord_msg::Msg::Ping(Ping {})).await;
    session.until("ping", |r| r.statuses.len() > before).await;
    session.report.ping_acked = true;
    let before = session.report.statuses.len();
    session
        .send(coord_msg::Msg::Cancel(Cancel { max_generation: 1 }))
        .await;
    session
        .until("initial cancel", |r| {
            r.statuses.len() > before && r.abandoned_watermark() >= 1
        })
        .await;
    session.report.cancel_acked = true;
    session.report.cancelled_watermark = 1;

    // These valid wire jobs are outside the approved coefficient domain.
    // Each rejection must refund its credit before supported work resumes.
    let mut fractional_h = inline.clone();
    fractional_h.h_milli_le32 = encode_i32_le(&[500, -1000]);
    let mut fractional_j = inline.clone();
    fractional_j.j_milli_le32 = encode_i32_le(&[500]);
    for (id, problem) in [
        (&b"job-fractional-h"[..], fractional_h),
        (&b"job-fractional-j"[..], fractional_j),
    ] {
        session
            .job(id, 2, problem, &[(0, 1)], JobKind::IsingSample, false)
            .await;
        session.refunded("unsupported coefficient").await;
        assert!(session.report.has_reject(id, RejectReason::TooLarge));
        assert!(!session.report.results.iter().any(|r| r.job_id == id));
    }

    // Required results after both rejections prove the session remains usable.
    for id in [&b"job-1"[..], &b"job-2"[..]] {
        session
            .job(
                id,
                2,
                inline.clone(),
                &[(0, 1)],
                JobKind::IsingSample,
                false,
            )
            .await;
    }
    let mut cached = inline.clone();
    cached.graph = Some(ising_problem::Graph::TopologyHash(dense_hash));
    session
        .job(
            b"job-hash",
            2,
            cached,
            &[(0, 1)],
            JobKind::IsingSample,
            false,
        )
        .await;
    session.refunded("inline and cached jobs").await;

    session
        .send(coord_msg::Msg::Topology(Topology {
            hash: sparse_hash.clone(),
            nodes: vec![0, 12, 2400],
            edges: Some(EdgeList {
                u: vec![0, 12],
                v: vec![12, 2400],
            }),
            allowed_h_milli: vec![-1000, 0, 1000],
        }))
        .await;
    let sparse = IsingProblem {
        graph: Some(ising_problem::Graph::TopologyHash(sparse_hash)),
        h_milli_le32: encode_i32_le(&[1000, -1000, 0]),
        j_milli_le32: encode_i32_le(&[1000, -1000]),
        ..inline.clone()
    };
    session
        .job(
            b"job-sparse",
            2,
            sparse,
            &[(0, 1), (1, 2)],
            JobKind::IsingSample,
            false,
        )
        .await;
    session.refunded("sparse topology job").await;

    let mut malformed_h = inline.clone();
    malformed_h.h_milli_le32 = vec![1, 2, 3];
    let mut malformed_j = inline.clone();
    malformed_j.j_milli_le32 = vec![1, 2, 3];
    for (id, problem, kind, expired) in [
        (&b"job-bad-h"[..], malformed_h, JobKind::IsingSample, false),
        (&b"job-bad-j"[..], malformed_j, JobKind::IsingSample, false),
        (
            &b"job-gate"[..],
            inline.clone(),
            JobKind::GateCircuit,
            false,
        ),
        (&b"job-old"[..], inline.clone(), JobKind::IsingSample, true),
    ] {
        session.job(id, 2, problem, &[(0, 1)], kind, expired).await;
        session.refunded("invalid job").await;
    }

    session
        .job(
            b"job-cancel",
            3,
            inline.clone(),
            &[(0, 1)],
            JobKind::IsingSample,
            false,
        )
        .await;
    let before = session.report.statuses.len();
    let dispatched = session.report.jobs_dispatched as u64;
    session
        .send(coord_msg::Msg::Cancel(Cancel { max_generation: 3 }))
        .await;
    session.report.cancelled_watermark = 3;
    session
        .until("live cancel", |r| {
            r.statuses.len() > before
                && r.abandoned_watermark() >= 3
                && r.credits_refunded() == dispatched
        })
        .await;
    session
        .job(
            b"job-stale",
            3,
            inline,
            &[(0, 1)],
            JobKind::IsingSample,
            false,
        )
        .await;
    session.refunded("stale job").await;
    session
        .send(coord_msg::Msg::Shutdown(Shutdown { grace_ms: 1000 }))
        .await;
    session.finish().await;
    let report = &session.report;
    assert!(report.handshake_ok, "handshake failed: {report:#?}");
    assert!(
        report.ready_received,
        "no Ready after Configure: {report:#?}"
    );
    let hello = report.hello.as_ref().expect("Hello");
    assert_eq!(hello.backend, "ane");
    assert_eq!(hello.algorithm, "msa");
    let capabilities = report.capabilities_received.as_ref().expect("Capabilities");
    assert_eq!(capabilities.stream_width, 1);
    assert!(
        report.credit_ledger_balanced(),
        "credit ledger unbalanced: dispatched {}, refunded {}",
        report.jobs_dispatched,
        report.credits_refunded()
    );
    for id in SOLVABLE_JOBS {
        assert_eq!(
            report
                .results
                .iter()
                .filter(|result| result.job_id == id)
                .count(),
            1,
            "exactly one Result required for {}",
            String::from_utf8_lossy(id)
        );
    }
    assert!(
        report.energies_rescore_clean(),
        "reported energy did not match the consensus scorer"
    );
    assert_eq!(report.expected_meta_sweeps(), CONFIGURED_SWEEPS);
    assert!(
        report.sweeps_honoured(),
        "SamplerMeta.sweeps must be {CONFIGURED_SWEEPS} for algorithm msa"
    );
    for (job_id, reason) in [
        (&b"job-bad-h"[..], RejectReason::Malformed),
        (&b"job-bad-j"[..], RejectReason::Malformed),
        (&b"job-gate"[..], RejectReason::UnsupportedKind),
        (&b"job-old"[..], RejectReason::Expired),
        (&b"job-fractional-h"[..], RejectReason::TooLarge),
        (&b"job-fractional-j"[..], RejectReason::TooLarge),
    ] {
        assert!(
            report.has_reject(job_id, reason),
            "missing {reason:?} reject for {}",
            String::from_utf8_lossy(job_id)
        );
    }
    assert!(report.ping_acked, "Ping not acked");
    assert!(report.cancel_acked, "Cancel not acked");
    assert!(
        report.live_cancel_conformant(),
        "live cancellation not honoured"
    );
    assert!(
        !report
            .results
            .iter()
            .any(|result| result.job_id == b"job-stale"),
        "stale result was returned"
    );
    assert_eq!(
        report.terminal,
        quip_solver_conformance::driver::Terminal::Closed,
        "stream did not end cleanly"
    );
    assert_eq!(report.exit_code, 0, "clean shutdown expected");
    assert_eq!(report.jobs_dispatched, 12);
    assert_eq!(report.rejects.len(), 6, "no duplicate or unexpected Reject");
    assert_eq!(report.job_request_credits.len(), 13);
    assert!(report
        .job_request_credits
        .iter()
        .skip(1)
        .all(|credit| *credit == 1));
    assert!(report.fatal.is_none(), "unexpected Fatal");
    assert!(report
        .results
        .iter()
        .all(|result| result.rescored_energies_milli.is_some()));
    assert!(
        report
            .results
            .iter()
            .filter(|r| r.job_id == b"job-cancel")
            .count()
            <= 1
    );
    assert!(report.is_conformant(), "{report:#?}");
}
