// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Two alternating pools keep cascade jobs resident between checkpoints.

use crate::cascade::{CascadeSettings, Controller, Ticket, MAX_STAGES};
use crate::metal_device::MetalDevice;
use crate::sampler::{self, Kernel, SampleError};
use crate::slots::{SlotJob, SlotPool};
use crate::streaming::{batch_size_for_reads, scale_budget, send_reject, GpuGovernor};
use quip_solver_core::{CancelToken, StreamJob, StreamOutcome, StreamResult};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::mpsc::{Receiver, Sender};

struct Live {
    job: StreamJob,
    ticket: Ticket,
    accounted_us: u64,
}

struct Pool {
    slots: SlotPool,
    live: Vec<Option<Live>>,
    reads: usize,
    stride: usize,
}

impl Pool {
    fn new(device: &MetalDevice, job: &StreamJob, capacity: usize) -> Result<Self, SampleError> {
        let reads = job.params.num_reads.clamp(1, sampler::MAX_READS);
        Ok(Self {
            slots: SlotPool::new(device, &job.graph, reads, capacity, job.params.num_sweeps)?,
            live: (0..capacity).map(|_| None).collect(),
            reads,
            stride: job.params.num_sweeps,
        })
    }

    fn fits(&self, job: &StreamJob) -> bool {
        self.slots.matches(
            &job.graph,
            job.params.num_reads.clamp(1, sampler::MAX_READS),
        ) && job.params.num_sweeps <= self.stride
    }

    fn harvest(
        &mut self,
        controller: &mut Controller,
        out: &Sender<StreamResult>,
        cancel: &CancelToken,
        gov: &dyn GpuGovernor,
    ) -> Result<u64, SampleError> {
        self.slots.wait();
        let checkpoints = self.slots.take_checkpoints()?;
        let mut busy_us = 0u64;
        for (slot, entry) in self.live.iter_mut().enumerate() {
            let Some(live) = entry else { continue };
            let us = self.slots.device_us(slot);
            busy_us = busy_us.saturating_add(us.saturating_sub(live.accounted_us));
            live.accounted_us = us;
            if cancel.is_cancelled(live.job.watermark) {
                self.slots.release(slot)?;
                let _ = out.blocking_send(StreamResult {
                    job_id: live.job.job_id.clone(),
                    outcome: StreamOutcome::Cancelled,
                    device_access_time_us: us,
                });
                controller.finish(&live.ticket, None, false);
                *entry = None;
            }
        }
        gov.record_gpu_busy_us(busy_us);
        let mut done = Vec::with_capacity(checkpoints.len());
        for checkpoint in checkpoints {
            let Some(live) = self.live[checkpoint.slot].as_mut() else {
                continue;
            };
            debug_assert_eq!(checkpoint.index, live.ticket.stage);
            if checkpoint.last || !controller.checkpoint(&mut live.ticket, checkpoint.best) {
                done.push(checkpoint.slot);
            }
        }
        // One decode across every completed/screened slot in this round.
        let reads = self.slots.reads_many(&done, self.reads)?;
        for (slot, reads) in done.into_iter().zip(reads) {
            self.slots.release(slot)?;
            let Some(live) = self.live[slot].take() else {
                continue;
            };
            let cancelled = cancel.is_cancelled(live.job.watermark);
            let best = if cancelled {
                None
            } else {
                reads.iter().map(|r| r.energy_milli).min()
            };
            // bh1.3.7: StreamResult needs sweeps_done to report the actual checkpoint position.
            let delivered = out
                .blocking_send(StreamResult {
                    job_id: live.job.job_id,
                    outcome: if cancelled {
                        StreamOutcome::Cancelled
                    } else {
                        StreamOutcome::Completed(Ok(reads))
                    },
                    device_access_time_us: live.accounted_us,
                })
                .is_ok();
            controller.finish(&live.ticket, best, delivered && !cancelled);
        }
        Ok(busy_us)
    }
}

fn validate(job: &StreamJob) -> Result<(), SampleError> {
    sampler::validate_batch(&[&job.graph], &job.params, Kernel::Msa)?;
    if job.params.num_sweeps == 0 || job.graph.num_nodes() == 0 {
        return Err(SampleError::TooLarge(
            "resident jobs require nonzero nodes and sweeps".into(),
        ));
    }
    if job.graph.j.len() != job.graph.edges.len() || !sampler::device_energy_exact(&job.graph) {
        return Err(SampleError::TooLarge(
            "slot jobs require exact device-energy coefficients".into(),
        ));
    }
    Ok(())
}

/// Run on the sampler's blocking thread; no Metal object leaves this thread.
pub(crate) fn run(
    device: &MetalDevice,
    settings: &Mutex<CascadeSettings>,
    store: &Mutex<Option<Controller>>,
    mut jobs: Receiver<StreamJob>,
    out: &Sender<StreamResult>,
    gov: &dyn GpuGovernor,
    cancel: &CancelToken,
) {
    let mut config = *settings.lock().unwrap_or_else(|p| p.into_inner());
    let mut controller = store
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .take()
        .unwrap_or_else(|| Controller::new(config));
    controller.refresh(config);
    let mut pools: Option<[Pool; 2]> = None;
    let mut pending = None;
    let mut eof = false;
    let mut turn = 0;
    let mut window = Instant::now();
    let mut busy_us = 0u64;
    let mut fault = None;

    'run: loop {
        if out.is_closed() {
            break;
        }
        if let Some(pools) = &mut pools {
            match pools[turn].harvest(&mut controller, out, cancel, gov) {
                Ok(us) => busy_us = busy_us.saturating_add(us),
                Err(error) => {
                    fault = Some(error);
                    break;
                }
            }
            if window.elapsed() >= Duration::from_secs(1) {
                let mut live_per_stage = [0usize; MAX_STAGES + 1];
                for pool in pools.iter() {
                    for live in pool.live.iter().flatten() {
                        live_per_stage[live.ticket.stage] += 1;
                    }
                }
                controller.update_load(
                    busy_us as f64 / window.elapsed().as_micros().max(1) as f64,
                    &live_per_stage,
                    pools.iter().map(|p| p.slots.capacity()).sum(),
                );
                config = *settings.lock().unwrap_or_else(|p| p.into_inner());
                controller.refresh(config);
                window = Instant::now();
                busy_us = 0;
            }
        }

        loop {
            if out.is_closed() {
                break 'run;
            }
            let empty = pools
                .as_ref()
                .is_none_or(|ps| ps.iter().all(|p| p.slots.live() == 0));
            if pending.is_none() {
                let full = pools.as_ref().is_some_and(|ps| {
                    let pool = &ps[turn];
                    let nominal = batch_size_for_reads(Kernel::Msa, pool.reads);
                    pool.slots.live()
                        >= scale_budget(nominal, gov.budget_scale()).min(pool.slots.capacity())
                });
                if full || eof {
                    break;
                }
                match jobs.try_recv() {
                    Ok(job) => pending = Some(job),
                    Err(TryRecvError::Disconnected) => {
                        eof = true;
                        break;
                    }
                    Err(TryRecvError::Empty) => break,
                }
            }
            let Some(job) = pending.take() else { break };
            if cancel.is_cancelled(job.watermark) {
                let _ = out.blocking_send(StreamResult {
                    job_id: job.job_id,
                    outcome: StreamOutcome::Cancelled,
                    device_access_time_us: 0,
                });
                continue;
            }
            if let Err(error) = validate(&job) {
                send_reject(out, job, error.to_sample_error());
                continue;
            }
            if pools.as_ref().is_none_or(|ps| !ps[turn].fits(&job)) {
                if !empty {
                    pending = Some(job);
                    break;
                }
                let capacity = scale_budget(
                    batch_size_for_reads(
                        Kernel::Msa,
                        job.params.num_reads.clamp(1, sampler::MAX_READS),
                    ),
                    gov.budget_scale(),
                );
                // Both pools are empty, so changing storage cannot discard work.
                let rebuilt = Pool::new(device, &job, capacity)
                    .and_then(|a| Pool::new(device, &job, capacity).map(|b| [a, b]));
                match rebuilt {
                    Ok(rebuilt) => {
                        if !controller.matches_topology(&job.graph) {
                            controller = Controller::new(config);
                        }
                        pools = Some(rebuilt);
                    }
                    Err(error) => {
                        send_reject(out, job, error.to_sample_error());
                        continue;
                    }
                }
            }
            let Some(pools) = &mut pools else { continue };
            let pool = &mut pools[turn];
            let (ticket, schedule, checkpoints) = controller.admit(&job);
            let admitted = pool.slots.admit(SlotJob {
                graph: job.graph.clone(),
                schedule,
                checkpoints,
                seed: job.params.seed,
            });
            match admitted {
                Ok(slot) => {
                    pool.live[slot] = Some(Live {
                        job,
                        ticket,
                        accounted_us: 0,
                    })
                }
                Err(error) => {
                    controller.finish(&ticket, None, false);
                    send_reject(out, job, error.to_sample_error());
                }
            }
        }

        let empty = pools
            .as_ref()
            .is_none_or(|ps| ps.iter().all(|p| p.slots.live() == 0));
        if empty && eof && pending.is_none() {
            break;
        }
        if let Some(pools) = &mut pools {
            if gov.should_throttle() {
                // Drain submitted work before the pause so it frees GPU time.
                for pool in pools.iter() {
                    pool.slots.wait();
                }
                let deadline = Instant::now() + Duration::from_millis(500);
                while !out.is_closed() && gov.should_throttle() && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
            if out.is_closed() {
                break;
            }
            if let Err(error) = pools[turn].slots.commit_step(config.stages[0].max(1)) {
                fault = Some(error);
                break;
            }
        }
        turn ^= 1;
        if empty && pending.is_none() {
            // blocking_recv cannot wake when only the output channel closes.
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    if let Some(pools) = &mut pools {
        for pool in pools {
            pool.slots.wait();
            for live in pool.live.iter_mut().filter_map(Option::take) {
                controller.finish(&live.ticket, None, false);
                if let Some(error) = &fault {
                    send_reject(out, live.job, error.to_sample_error());
                }
            }
        }
    }
    if let Some(error) = &fault {
        tracing::error!(%error, "resident stream failed");
        if let Some(job) = pending {
            send_reject(out, job, error.to_sample_error());
        }
    }
    *store.lock().unwrap_or_else(|p| p.into_inner()) = Some(controller);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IsingGraph, MetalSampler, SampleParams};
    use quip_solver_core::Sampler;

    fn job(id: usize, sweeps: usize) -> StreamJob {
        let n = 64;
        StreamJob {
            job_id: id.to_le_bytes().to_vec(),
            graph: IsingGraph::new(
                (0..n).map(|i| [-1.0, 0.0, 1.0][i % 3]).collect(),
                vec![-1.0; n - 1],
                (0..n - 1).map(|i| (i, i + 1)).collect(),
            ),
            params: SampleParams {
                num_reads: 4,
                num_sweeps: sweeps,
                seed: id as u64 + 1,
                ..Default::default()
            },
            watermark: None,
        }
    }

    #[test]
    fn invalid_admissions_fail_before_allocating_schedules() {
        let mut invalid = job(0, sampler::MAX_SWEEPS + 1);
        assert_eq!(
            validate(&invalid).unwrap_err().to_sample_error(),
            quip_solver_core::SampleError::Capacity
        );
        invalid.params.num_sweeps = 0;
        assert_eq!(
            validate(&invalid).unwrap_err().to_sample_error(),
            quip_solver_core::SampleError::Capacity
        );
        invalid.params.num_sweeps = 32;
        invalid.graph.j.pop();
        assert_eq!(
            validate(&invalid).unwrap_err().to_sample_error(),
            quip_solver_core::SampleError::Capacity
        );
        invalid.graph.j.push(-0.5);
        assert_eq!(
            validate(&invalid).unwrap_err().to_sample_error(),
            quip_solver_core::SampleError::Capacity
        );
        invalid.graph.j.pop();
        invalid.graph.j.push(-1.0);
        validate(&invalid).unwrap();
    }

    #[test]
    fn cancelling_a_kept_slot_returns_once_and_releases_it() {
        if MetalDevice::device_count() == 0 {
            #[expect(clippy::print_stderr, reason = "device tests report a sandbox skip")]
            {
                eprintln!("skipping kept-slot cancellation test: no Metal device");
            }
            return;
        }
        struct Governor;
        impl GpuGovernor for Governor {
            fn should_throttle(&self) -> bool {
                false
            }
            fn budget_scale(&self) -> f64 {
                1.0
            }
            fn record_gpu_busy_us(&self, _: u64) {}
        }
        let device = MetalDevice::open(0).unwrap();
        let settings = CascadeSettings {
            enabled: true,
            stages: [8, 0, 0],
            keep: 1.0,
            keep_min: 1.0,
            keep_max: 1.0,
            ..CascadeSettings::default()
        };
        let mut controller = Controller::new(settings);
        let mut live_job = job(1000, 256);
        live_job.watermark = Some(1);
        // Train the cutoff above this ferromagnetic job's negative energy.
        // Direct settings isolate slot cancellation from TOML's minimum of two.
        for _ in 0..200 {
            let (mut ticket, _, _) = controller.admit(&live_job);
            controller.checkpoint(&mut ticket, 0);
            controller.finish(&ticket, None, false);
        }
        let mut pool = Pool::new(&device, &live_job, 1).unwrap();
        let (ticket, schedule, checkpoints) = controller.admit(&live_job);
        let slot = pool
            .slots
            .admit(SlotJob {
                graph: live_job.graph.clone(),
                schedule,
                checkpoints,
                seed: live_job.params.seed,
            })
            .unwrap();
        pool.live[slot] = Some(Live {
            job: live_job,
            ticket,
            accounted_us: 0,
        });
        let (out, mut results) = tokio::sync::mpsc::channel(2);
        let cancel = CancelToken::default();
        pool.slots.commit_step(8).unwrap();
        pool.harvest(&mut controller, &out, &cancel, &Governor)
            .unwrap();
        assert_eq!(pool.live[slot].as_ref().unwrap().ticket.stage, 1);
        assert_eq!(pool.slots.live(), 1);
        assert!(matches!(results.try_recv(), Err(TryRecvError::Empty)));
        pool.slots.commit_step(8).unwrap();
        cancel.cancel_through(1);
        pool.harvest(&mut controller, &out, &cancel, &Governor)
            .unwrap();
        assert!(matches!(
            results.try_recv().unwrap().outcome,
            StreamOutcome::Cancelled
        ));
        pool.harvest(&mut controller, &out, &cancel, &Governor)
            .unwrap();
        assert_eq!(pool.slots.live(), 0);
        assert!(matches!(results.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn controller_state_survives_a_second_stream() {
        if MetalDevice::device_count() == 0 {
            #[expect(clippy::print_stderr, reason = "device tests report a sandbox skip")]
            {
                eprintln!("skipping controller persistence test: no Metal device");
            }
            return;
        }
        let sampler = MetalSampler::new(
            MetalDevice::open(0).unwrap(),
            crate::iokit_gov::UtilGovernor::start(0, 100, false),
            Kernel::Msa,
        );
        sampler.apply_config("cascade = true\ncascade_stages = [8]\ncascade_keep = 100\ncascade_keep_min = 10\ncascade_keep_max = 100");
        let run = |start, count| {
            let (tx, rx) = tokio::sync::mpsc::channel(count);
            let (out, mut results) = tokio::sync::mpsc::channel(count);
            for id in start..start + count {
                tx.blocking_send(job(id, 16)).unwrap();
            }
            drop(tx);
            sampler.sample_stream(rx, out, CancelToken::default());
            let mut received = 0;
            while let Some(result) = results.blocking_recv() {
                match result.outcome {
                    StreamOutcome::Completed(Ok(reads)) => assert_eq!(reads.len(), 4),
                    StreamOutcome::Completed(Err(error)) => panic!("{error}"),
                    StreamOutcome::Cancelled => panic!("uncancelled stream"),
                }
                received += 1;
            }
            assert_eq!(received, count);
            sampler
                .controller
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .keep_denominators()[0]
        };
        let settled = run(0, 5000);
        assert!(settled > 50.0, "denominator did not settle: {settled}");
        // A reset would have only 100 observations and a denominator near 32.
        let second = run(5000, 100);
        assert!(second > 50.0, "second stream restarted warm-up: {second}");
        assert!(
            second >= settled * 0.9,
            "lost settled denominator: {settled} -> {second}"
        );
    }
}
