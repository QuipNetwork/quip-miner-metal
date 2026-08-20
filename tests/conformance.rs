//! Protocol conformance: spawn SA and Gibbs miners against
//! quip-solver-conformance's scripted driver.
//!
//! Metal GPU tests: needs a real device (Apple Silicon).

use quip_solver_conformance::driver::{drive_miner, DriverReport};
use quip_solver_core::quip_proto::v1::RejectReason;
use std::process::Command;

/// Grade one driven session against every axis of the miner protocol.
///
/// `is_conformant()` is the driver's own composite verdict, so this tracks
/// the reference automatically as the contract grows. The per-axis
/// assertions run first purely for diagnosis: a bare composite failure says
/// "not conformant" without saying which rule broke. `report.summary()` (one
/// pass/FAIL line per graded axis) backs the final assertion instead of the
/// full `Debug` dump, which buries the failing axis in spin bytes.
fn assert_conformant(bin: &str, report: &DriverReport) {
    // Handshake: Hello -> Welcome -> Configure -> Ready (SPEC.md
    // "Handshake"). Dispatch stays blocked until `Ready`.
    assert!(report.handshake_ok, "{bin}: handshake failed");
    assert!(report.ready_received, "{bin}: no Ready after Configure");

    // Credits: the miner grants a first batch, then returns one per terminal
    // outcome. A zero grant would deadlock dispatch ("Credits").
    assert!(
        !report.job_request_credits.is_empty(),
        "{bin}: miner granted no credits"
    );
    assert!(
        report.job_request_credits.iter().all(|&c| c > 0),
        "{bin}: zero-credit JobRequest: {:?}",
        report.job_request_credits
    );

    // Results: one per solvable job (job-1, job-2, job-hash, job-sparse),
    // each carrying solutions and a SamplerMeta whose reported energy
    // survives the driver's own re-score, and honouring the configured sweep
    // budget ("Servicing a job").
    assert!(
        report.results_conformant(),
        "{bin}: results not conformant: {:?}",
        report.result_job_ids()
    );
    for r in &report.results {
        assert!(
            !r.solution_energies_milli.is_empty(),
            "{bin}: result {:?} carried no solutions",
            r.job_id
        );
        assert!(r.meta_present, "{bin}: result {:?} had no meta", r.job_id);
    }
    assert!(
        report.energies_rescore_clean(),
        "{bin}: a reported energy did not survive the driver's re-score"
    );
    assert!(
        report.sweeps_honoured(),
        "{bin}: SamplerMeta.sweeps did not echo the configured sweep budget"
    );

    // Reject reasons: each drives a different coordinator response, so the
    // reason must be right, not merely present ("Servicing a job" table).
    for (job_id, reason) in [
        (&b"job-bad-h"[..], RejectReason::Malformed),
        (&b"job-bad-j"[..], RejectReason::Malformed),
        (&b"job-gate"[..], RejectReason::UnsupportedKind),
        (&b"job-old"[..], RejectReason::Expired),
    ] {
        assert!(
            report.has_reject(job_id, reason),
            "{bin}: missing {reason:?} reject for {}: {:?}",
            String::from_utf8_lossy(job_id),
            report.rejects
        );
    }

    // Cancel and Ping are each acknowledged with a Status ("Control-plane
    // pushes"); live cancellation is honoured (no Result/Reject for the
    // cancelled watermark); GetCapabilities answers with the same identity
    // Hello advertised.
    assert!(report.cancel_acked, "{bin}: Cancel not acked with Status");
    assert!(report.ping_acked, "{bin}: Ping not acked with Status");
    assert!(
        report.live_cancel_conformant(),
        "{bin}: live cancellation not honoured"
    );
    assert!(
        report.capabilities_conformant(),
        "{bin}: Capabilities disagrees with the Hello identity"
    );

    // Every dispatched job refunds exactly one credit, however it ends.
    assert!(
        report.credit_ledger_balanced(),
        "{bin}: credit ledger unbalanced: dispatched {}, refunded {}",
        report.jobs_dispatched,
        report.credits_refunded()
    );

    // Shutdown drains in-flight results, closes the stream cleanly, and
    // exits 0 ("Exit codes").
    assert_eq!(report.exit_code, 0, "{bin}: clean shutdown expected");

    assert!(
        report.is_conformant(),
        "{bin}: not conformant per the reference verdict:\n{}",
        report.summary()
    );
}

/// Cross-package binary path (deps/ → profile/ → bin).
fn profile_bin(name: &str) -> String {
    let name = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    let mut p = std::env::current_exe().expect("test exe path");
    p.pop(); // deps/
    p.pop(); // <profile>/
    p.push(&name);
    p.to_string_lossy().into_owned()
}

fn ensure_built(package_bins: &[&str]) {
    let status = Command::new(env!("CARGO"))
        .args(["build", "-p", "quip-miner-metal"])
        .status()
        .expect("cargo build quip-miner-metal");
    assert!(status.success(), "failed to build quip-miner-metal");
    for b in package_bins {
        assert!(
            std::path::Path::new(&profile_bin(b)).exists(),
            "missing binary {b} at {}",
            profile_bin(b)
        );
    }
}

#[tokio::test]
async fn quip_metal_sa_passes_conformance() {
    ensure_built(&["quip-metal-sa"]);
    let miner = profile_bin("quip-metal-sa");
    let socket = format!(
        "/tmp/quip-metal-sa-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner(&miner, &format!("unix://{socket}")).await;
    assert_conformant("quip-metal-sa", &report);
}

#[tokio::test]
async fn quip_metal_gibbs_passes_conformance() {
    ensure_built(&["quip-metal-gibbs"]);
    let miner = profile_bin("quip-metal-gibbs");
    let socket = format!(
        "/tmp/quip-metal-gibbs-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner(&miner, &format!("unix://{socket}")).await;
    assert_conformant("quip-metal-gibbs", &report);
}

#[test]
fn capabilities_and_version_and_check() {
    ensure_built(&["quip-metal-sa", "quip-metal-gibbs"]);

    for (bin, algo) in [("quip-metal-sa", "sa"), ("quip-metal-gibbs", "gibbs")] {
        let path = profile_bin(bin);

        let out = Command::new(&path).arg("--capabilities").output().unwrap();
        assert!(out.status.success(), "{bin} --capabilities failed");
        let s = String::from_utf8(out.stdout).unwrap();
        assert!(s.contains("\"backend\":\"metal\""), "{bin}: {s}");
        assert!(
            s.contains(&format!("\"algorithm\":\"{algo}\"")),
            "{bin}: {s}"
        );

        let out = Command::new(&path).arg("--version").output().unwrap();
        assert!(out.status.success());
        assert!(String::from_utf8(out.stdout).unwrap().contains("protocol"));

        // --check opens the GPU and compiles kernels.
        let status = Command::new(&path)
            .arg("--check")
            .arg("--device")
            .arg("0")
            .status();
        assert!(
            status.unwrap().success(),
            "{bin} --check must succeed when a GPU is present"
        );
    }
}

/// The core must reject an unknown `--log-level` before doing anything else.
/// In `quip-solver-core` 0.0.0 the validation moved from `logging::init`
/// (exit 64) into `CommonArgs` itself — a clap `PossibleValuesParser` — so a
/// bad level is a usage error (exit 2) and capabilities are never printed.
///
/// This crate installs a subscriber of its own, but that one does not validate
/// the level, so the check still detects a stale core: a core without the
/// value parser would print capabilities and exit 0 here.
#[test]
fn invalid_log_level_is_a_usage_error() {
    for bin in [
        env!("CARGO_BIN_EXE_quip-metal-sa"),
        env!("CARGO_BIN_EXE_quip-metal-gibbs"),
    ] {
        let out = Command::new(bin)
            .arg("--capabilities")
            .arg("--log-level")
            .arg("bogus")
            .env("QUIP_SESSION_TOKEN", "tok")
            .output()
            .unwrap();
        assert_eq!(
            out.status.code(),
            Some(2),
            "{bin}: an unknown --log-level must be a clap usage error (got {:?}, stdout={}, stderr={})",
            out.status.code(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("bogus"),
            "{bin}: stderr must name the bad level, got {stderr}"
        );
        assert!(
            String::from_utf8_lossy(&out.stdout).is_empty(),
            "{bin}: capabilities must not print on a bad --log-level"
        );
    }
}
