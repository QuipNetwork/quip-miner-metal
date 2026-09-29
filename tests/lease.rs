// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Salt leases through the real session and the MSA miner.
//!
//! Metal GPU tests: need a real device (Apple Silicon). Run with
//! `--test-threads=1`.

mod support;

use quip_solver_core::quip_proto::v1::{
    coord_msg, ising_problem, Cancel, CoefficientEncoding, EdgeList, IsingProblem,
    IsingProblemGenerator, Job, JobKind, RejectReason,
};
use quip_solver_core::quip_protocol::lease::verify_lease_result;
use quip_solver_core::quip_protocol::wire::encode_i32_le;
use std::collections::HashSet;
use std::time::Duration;
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
async fn every_salt_is_counted_once() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, view) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 64)))
        .await;
    s.send(coord_msg::Msg::Job(lease(b"lease-measure", 3, RING_A, 0, 200)))
        .await;
    s.until("measured lease done", LONG, |log| {
        log.done(b"lease-measure").is_some()
    })
    .await;
    let mut bests: Vec<i64> = s
        .log
        .results_for(b"lease-measure")
        .iter()
        .map(|r| {
            r.solutions
                .iter()
                .map(|x| x.energy_milli)
                .min()
                .expect("solutions")
        })
        .collect();
    assert_eq!(bests.len(), 200, "an i64::MAX target makes every salt win");
    bests.sort_unstable();
    let median = bests[bests.len() / 2];

    let t = target(median, 64);
    s.send(coord_msg::Msg::SetTarget(t)).await;
    let job = lease(b"lease-median", 5, RING_A, 0, 200);
    let spec = generator(&job);
    s.send(coord_msg::Msg::Job(job)).await;
    s.until("median lease done", LONG, |log| {
        log.done(b"lease-median").is_some()
    })
    .await;

    let done = s.log.done(b"lease-median").expect("LeaseDone");
    assert_eq!(done.salts_done, 200, "every salt is counted once");
    let results = s.log.results_for(b"lease-median");
    assert!(
        results.len() < 200,
        "the median target screens at least one salt"
    );
    let mut salts = HashSet::new();
    for result in &results {
        let verified = verify_lease_result(&spec, &view, &wire_target(&t), result)
            .expect("every winner verifies against the lease");
        assert!(salts.insert(verified.salt), "one result per salt");
    }
    assert_eq!(s.shutdown(2_000).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_reachable_target_screens_no_winner() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, view) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    let t = target(i64::MAX, 32);
    s.send(coord_msg::Msg::SetTarget(t)).await;
    let job = lease(b"lease-all", 3, RING_A, 0, 64);
    let spec = generator(&job);
    s.send(coord_msg::Msg::Job(job)).await;
    s.until("lease done", LONG, |log| log.done(b"lease-all").is_some())
        .await;

    let done = s.log.done(b"lease-all").expect("LeaseDone");
    assert_eq!(done.salts_done, 64);
    let results = s.log.results_for(b"lease-all");
    assert_eq!(results.len(), 64, "an i64::MAX target makes every salt win");
    let mut salts = HashSet::new();
    for result in &results {
        let verified = verify_lease_result(&spec, &view, &wire_target(&t), result)
            .expect("every winner verifies against the lease");
        assert!(salts.insert(verified.salt), "one result per salt");
    }
    assert_eq!(s.shutdown(2_000).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn screened_salts_send_no_reads() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, _) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    // 64 nodes and 64 edges: no spin assignment on unit couplings can go
    // below -(64 + 64) * 1000, so this target screens every salt.
    let bound = -(64i64 + 64) * 1000 - 1;
    s.send(coord_msg::Msg::SetTarget(target(bound, 32))).await;
    let salt_count = 200;
    s.send(coord_msg::Msg::Job(lease(
        b"lease-none",
        3,
        RING_A,
        0,
        salt_count,
    )))
    .await;
    s.until("lease done", LONG, |log| {
        log.done(b"lease-none").is_some()
    })
    .await;

    let done = s.log.done(b"lease-none").expect("LeaseDone");
    assert_eq!(done.salts_done, salt_count);
    assert!(s.log.results_for(b"lease-none").is_empty());
    assert!(done.best_energy_milli < i64::MAX, "reads finished");
    assert!(
        done.best_energy_milli > bound,
        "the unreachable target is never met"
    );
    assert_eq!(s.shutdown(2_000).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_lease_without_plain_jobs_starts_the_device() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, _) = ring(64, RING_A);
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 32)))
        .await;
    s.send(coord_msg::Msg::Job(lease(b"lease-first", 3, RING_A, 0, 50)))
        .await;
    s.until("lease done", Duration::from_secs(60), |log| {
        log.done(b"lease-first").is_some()
    })
    .await;
    assert_eq!(s.shutdown(2_000).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_ends_a_live_lease_with_one_summary() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, _) = aglais();
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(
        support::AGLAIS_TARGET_MILLI,
        16_384,
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
    s.wait_for_salts(1, LONG).await;
    s.send(coord_msg::Msg::Cancel(Cancel { max_generation: 7 }))
        .await;
    s.until("cancel summary", Duration::from_secs(10), |log| {
        log.done(b"lease-cancel").is_some() && log.refunds() >= 1
    })
    .await;

    let done = s.log.done(b"lease-cancel").expect("LeaseDone");
    assert!(done.salts_done < 1_000_000, "the cancel cut the lease short");
    assert_eq!(s.log.done_count(b"lease-cancel"), 1, "one summary");
    assert_eq!(s.log.refunds(), 1, "the cancelled lease refunds its credit");
    assert_eq!(s.shutdown(2_000).await, 0);
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
    assert_eq!(s.log.done_count(b"lease-shutdown"), 1);
    assert_eq!(s.log.refunds(), 1, "the lease refunds its credit");
    let done = s
        .log
        .done(b"lease-shutdown")
        .expect("LeaseDone before close");
    assert!(done.salts_done >= 100 && done.salts_done < 1_000_000);
    let summary = s
        .log
        .order
        .iter()
        .position(|e| e == "done:lease-shutdown")
        .expect("summary in order log");
    assert!(
        !s.log.order[summary..]
            .iter()
            .any(|e| e == "result:lease-shutdown"),
        "no result after the shut-down lease's summary"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_plain_job_during_a_lease_returns_all_its_reads() {
    let mut s = Session::start(&msa(), 4, "").await;
    let (topology, _) = aglais();
    s.send(coord_msg::Msg::Topology(topology)).await;
    s.send(coord_msg::Msg::SetTarget(target(
        support::AGLAIS_TARGET_MILLI,
        14_336,
    )))
    .await;
    s.send(coord_msg::Msg::Job(lease(
        b"lease-long",
        3,
        support::AGLAIS_HASH,
        0,
        1_000_000,
    )))
    .await;
    s.wait_for_salts(1, LONG).await;
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
            num_reads: 16,
            num_sweeps: 64,
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
    s.until("plain job result", LONG, |log| {
        !log.results_for(b"plain-1").is_empty()
    })
    .await;

    let results = s.log.results_for(b"plain-1");
    assert_eq!(results.len(), 1, "a plain job sends one result");
    let result = results[0];
    assert_eq!(result.solutions.len(), 16, "all reads come back");
    assert_eq!(result.meta.as_ref().expect("meta").sweeps, 64);
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
    assert_eq!(s.shutdown(2_000).await, 0);
    assert_eq!(s.log.rejects.len(), 2);
    assert_eq!(s.log.refunds(), 2);
    assert!(
        s.log.lease_done.is_empty(),
        "a rejected lease sends no summary"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_topology_change_between_leases_uses_the_new_graph() {
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
