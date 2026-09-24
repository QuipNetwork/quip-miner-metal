// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Salt leases through the real session and the MSA miner.
//!
//! Metal GPU tests: need a real device (Apple Silicon). Run with
//! `--test-threads=1`. `lease_throughput_aglais` is an ignored bench.

#![expect(
    clippy::print_stderr,
    reason = "the bench reports rates on stderr, like the other benches"
)]

mod support;

use quip_solver_core::quip_proto::v1::{
    coord_msg, ising_problem, Cancel, CoefficientEncoding, EdgeList, IsingProblem,
    IsingProblemGenerator, Job, JobKind, RejectReason,
};
use quip_solver_core::quip_protocol::lease::verify_lease_result;
use quip_solver_core::quip_protocol::wire::encode_i32_le;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use support::{aglais, lease, miner_binary, ring, target, wire_target, Session};

const RING_A: [u8; 32] = [0x51; 32];
const RING_B: [u8; 32] = [0x52; 32];
const LONG: Duration = Duration::from_secs(120);

fn msa() -> String {
    miner_binary("quip-metal-msa")
}

fn generator(job: &Job) -> IsingProblemGenerator {
    job.generator.clone().expect("lease job")
}

#[tokio::test(flavor = "multi_thread")]
async fn every_salt_counts_and_every_winner_verifies() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, view) = ring(64, RING_A);
    let t = target(i64::MAX, 32);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(t)).await;
    let job = lease(b"lease-all", 3, RING_A, 10, 50);
    let spec = generator(&job);
    s.send(coord_msg::Msg::Job(job)).await;
    s.until("lease done", LONG, |log| {
        log.done(b"lease-all").is_some() && log.refunds() >= 1
    })
    .await;

    let done = s.log.done(b"lease-all").expect("LeaseDone");
    assert_eq!(done.salts_done, 50);
    let results = s.log.results_for(b"lease-all");
    assert_eq!(results.len(), 50, "an i64::MAX target makes every salt win");
    let mut salts = HashSet::new();
    for result in &results {
        let verified = verify_lease_result(&spec, &view, &wire_target(&t), result)
            .expect("every winner verifies against the lease");
        assert!(salts.insert(verified.salt), "one result per salt");
        assert_eq!(result.meta.as_ref().expect("meta").sweeps, 32);
    }
    let best_reported = results
        .iter()
        .flat_map(|r| r.solutions.iter().map(|x| x.energy_milli))
        .min()
        .expect("solutions");
    assert!(done.best_energy_milli <= best_reported);
    assert_eq!(s.shutdown(2_000).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn unreachable_target_sends_no_results_and_one_summary() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, _) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    // A 64-node ring cannot go below -128,000 milli.
    s.send(coord_msg::Msg::SetTarget(target(-1_000_000, 32)))
        .await;
    s.send(coord_msg::Msg::Job(lease(b"lease-none", 3, RING_A, 0, 200)))
        .await;
    s.until("lease done", LONG, |log| {
        log.done(b"lease-none").is_some() && log.refunds() >= 1
    })
    .await;

    let done = s.log.done(b"lease-none").expect("LeaseDone");
    assert_eq!(done.salts_done, 200);
    assert!(done.best_energy_milli < i64::MAX, "reads finished");
    assert!(s.log.results_for(b"lease-none").is_empty());
    assert_eq!(s.shutdown(2_000).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_ends_a_live_lease_with_partial_progress() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, _) = aglais();
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(
        support::AGLAIS_TARGET_MILLI,
        14_336,
    )))
    .await;
    s.send(coord_msg::Msg::Job(lease(
        b"lease-cancel",
        7,
        support::AGLAIS_HASH,
        0,
        1_000_000,
    )))
    .await;
    s.wait_for_salts(100, LONG).await;
    s.send(coord_msg::Msg::Cancel(Cancel { max_generation: 7 }))
        .await;
    s.until("cancel summary", Duration::from_secs(10), |log| {
        log.done(b"lease-cancel").is_some() && log.refunds() >= 1
    })
    .await;

    let done = s.log.done(b"lease-cancel").expect("LeaseDone");
    assert!(done.salts_done >= 100 && done.salts_done < 1_000_000);
    assert_eq!(s.shutdown(2_000).await, 0);
    let summary = s
        .log
        .order
        .iter()
        .position(|e| e == "done:lease-cancel")
        .expect("summary in order log");
    assert!(
        !s.log.order[summary..]
            .iter()
            .any(|e| e == "result:lease-cancel"),
        "no result after the cancelled lease's summary"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn shutdown_during_a_lease_sends_its_summary_and_exits_cleanly() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, _) = aglais();
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(
        support::AGLAIS_TARGET_MILLI,
        14_336,
    )))
    .await;
    s.send(coord_msg::Msg::Job(lease(
        b"lease-shutdown",
        9,
        support::AGLAIS_HASH,
        0,
        1_000_000,
    )))
    .await;
    s.wait_for_salts(100, LONG).await;

    assert_eq!(s.shutdown(3_000).await, 0);
    let done = s
        .log
        .done(b"lease-shutdown")
        .expect("LeaseDone before close");
    assert!(done.salts_done >= 100 && done.salts_done < 1_000_000);
    assert!(s.log.refunds() >= 1, "the lease refunds its credit");
}

#[tokio::test(flavor = "multi_thread")]
async fn topology_change_between_leases_uses_the_new_graph() {
    let mut s = Session::start(&msa(), 4, "").await;
    let t = target(i64::MAX, 32);
    s.send(coord_msg::Msg::SetTarget(t)).await;
    let (first, _) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(first)).await;
    s.send(coord_msg::Msg::Job(lease(b"lease-a", 3, RING_A, 0, 40)))
        .await;
    s.until("first lease", LONG, |log| log.done(b"lease-a").is_some())
        .await;

    let (second, view) = ring(96, RING_B);
    s.send(coord_msg::Msg::Topology(second)).await;
    let job = lease(b"lease-b", 3, RING_B, 0, 40);
    let spec = generator(&job);
    s.send(coord_msg::Msg::Job(job)).await;
    s.until("second lease", LONG, |log| log.done(b"lease-b").is_some())
        .await;

    assert_eq!(s.log.done(b"lease-b").expect("LeaseDone").salts_done, 40);
    let results = s.log.results_for(b"lease-b");
    assert_eq!(results.len(), 40);
    for result in results {
        verify_lease_result(&spec, &view, &wire_target(&t), result)
            .expect("second-lease winners use the 96-node ring");
    }
    assert_eq!(s.shutdown(2_000).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn plain_job_and_lease_share_the_stream() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, _) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 32)))
        .await;
    s.send(coord_msg::Msg::Job(lease(
        b"lease-mixed",
        3,
        RING_A,
        0,
        100,
    )))
    .await;
    s.send(coord_msg::Msg::Job(Job {
        job_id: b"plain-1".to_vec(),
        kind: JobKind::IsingSample as i32,
        generation: 3,
        deadline_ms: 0,
        ising: Some(IsingProblem {
            encoding: CoefficientEncoding::I32 as i32,
            scale: 1000,
            h: encode_i32_le(&[1000, -1000, 0]),
            j: encode_i32_le(&[1000, -1000]),
            num_reads: 64,
            num_sweeps: 32,
            graph: Some(ising_problem::Graph::Edges(EdgeList {
                u: vec![0, 1],
                v: vec![1, 2],
            })),
            ..Default::default()
        }),
        provenance: None,
        generator: None,
    }))
    .await;
    s.until("both finish", LONG, |log| {
        log.done(b"lease-mixed").is_some()
            && !log.results_for(b"plain-1").is_empty()
            && log.refunds() >= 2
    })
    .await;

    assert_eq!(
        s.log.done(b"lease-mixed").expect("LeaseDone").salts_done,
        100
    );
    assert_eq!(s.log.refunds(), 2, "one refund per job");
    assert_eq!(s.shutdown(2_000).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn invalid_leases_are_rejected_with_refunds() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, _) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::Job(lease(b"no-target", 3, RING_A, 0, 10)))
        .await;
    s.until("target missing", Duration::from_secs(10), |log| {
        !log.rejects.is_empty()
    })
    .await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 32)))
        .await;
    s.send(coord_msg::Msg::Job(lease(b"wrong-hash", 3, RING_B, 0, 10)))
        .await;
    s.until("rejects", Duration::from_secs(10), |log| {
        log.rejects.len() >= 2 && log.refunds() >= 2
    })
    .await;

    let reason = |id: &[u8]| {
        s.log
            .rejects
            .iter()
            .find(|r| r.job_id == id)
            .map(|r| r.reason)
    };
    assert_eq!(
        reason(b"no-target"),
        Some(RejectReason::TargetMissing as i32)
    );
    assert_eq!(
        reason(b"wrong-hash"),
        Some(RejectReason::TopologyMismatch as i32)
    );
    assert!(
        s.log.lease_done.is_empty(),
        "a rejected lease sends no summary"
    );
    assert_eq!(s.shutdown(2_000).await, 0);
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).map_or(default, |v| {
        v.parse()
            .unwrap_or_else(|_| panic!("{name} must be an integer"))
    })
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "GPU benchmark: minutes of device time"]
async fn lease_throughput_aglais() {
    let seconds = env_u64("QUIP_LEASE_SECONDS", 600);
    let per_lease = env_u64("QUIP_LEASE_SALTS", 4_000);
    let sweeps = u32::try_from(env_u64("QUIP_LEASE_SWEEPS", 14_336)).expect("sweeps fit u32");
    let target_milli = std::env::var("QUIP_LEASE_TARGET")
        .map_or(support::AGLAIS_TARGET_MILLI, |v| {
            v.parse().expect("QUIP_LEASE_TARGET must be an integer")
        });
    let backend_toml = std::env::var("QUIP_LEASE_BACKEND_TOML").unwrap_or_default();

    let mut s = Session::start(&msa(), 64, &backend_toml).await;
    let granted = s.log.credits[0];
    let (topology, view) = aglais();
    let t = target(target_milli, sweeps);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(t)).await;
    eprintln!(
        "lease: {granted} credits, {per_lease} salts per lease, {sweeps} sweeps, target {target_milli}, backend_toml={backend_toml:?}"
    );

    let mut specs: HashMap<Vec<u8>, IsingProblemGenerator> = HashMap::new();
    let mut next = 0u64;
    let issue = |next: u64| {
        lease(
            format!("bench-{next}").as_bytes(),
            1,
            support::AGLAIS_HASH,
            next * per_lease,
            per_lease,
        )
    };
    for _ in 0..granted {
        let job = issue(next);
        specs.insert(job.job_id.clone(), generator(&job));
        s.send(coord_msg::Msg::Job(job)).await;
        next += 1;
    }

    let start = Instant::now();
    let cpu_start = support::cpu_seconds(s.pid());
    let mut summaries = 0;
    let mut report_at = Duration::from_secs(10);
    while start.elapsed() < Duration::from_secs(seconds) {
        s.poll(Duration::from_secs(1)).await;
        assert!(!s.log.closed, "miner closed the stream: {:?}", s.log.fatal);
        while s.log.lease_done.len() > summaries {
            summaries += 1;
            let job = issue(next);
            specs.insert(job.job_id.clone(), generator(&job));
            s.send(coord_msg::Msg::Job(job)).await;
            next += 1;
        }
        if start.elapsed() >= report_at {
            let salts = s.jobs_done().await;
            eprintln!(
                "lease: t={:.0} salts={salts} rate={:.2}/s results={}",
                start.elapsed().as_secs_f64(),
                salts as f64 / start.elapsed().as_secs_f64(),
                s.log.results.len()
            );
            report_at += Duration::from_secs(10);
        }
    }
    let salts = s.jobs_done().await;
    let elapsed = start.elapsed().as_secs_f64();
    let cpu = support::cpu_seconds(s.pid()) - cpu_start;
    let mut verified = 0;
    for result in &s.log.results {
        let spec = specs.get(&result.job_id).expect("known lease");
        verify_lease_result(spec, &view, &wire_target(&t), result).expect("winner verifies");
        verified += 1;
    }
    eprintln!(
        "lease summary: salts {salts} in {elapsed:.1} s = {:.2} salts/s; miner CPU ms per salt {:.3}; results {} verified {verified}; leases finished {summaries}",
        salts as f64 / elapsed,
        cpu * 1000.0 / salts.max(1) as f64,
        s.log.results.len()
    );
    assert_eq!(s.shutdown(5_000).await, 0);
}
