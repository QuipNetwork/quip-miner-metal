// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Two alternating pools keep cascade jobs resident between checkpoints.

use crate::cascade::{
    topology_fingerprint, CascadeSettings, Controller, PreparedSchedule, ScheduleCache, Ticket,
    CHAIN_GATES, MAX_STAGES,
};
use crate::metal_device::MetalDevice;
use crate::sampler::{self, Kernel, SampleError};
use crate::slots::{PreparedInputs, SlotPool};
use crate::streaming::{batch_size_for_reads, scale_budget, send_reject, GpuGovernor};
use crate::topology::SelfFeedingTopology;
use quip_solver_core::{CancelToken, StreamJob, StreamOutcome, StreamResult};
use std::collections::VecDeque;
use std::sync::{mpsc, Mutex};
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
            tracing::debug!(
                target: "quip_miner_metal::cascade_trace",
                job = %String::from_utf8_lossy(&live.job.job_id),
                stage = checkpoint.index,
                best = checkpoint.best,
                "checkpoint"
            );
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
    if job.graph.num_nodes() > sampler::kernel_max_nodes(Kernel::Msa)
        || job.params.num_sweeps > sampler::MAX_SWEEPS
    {
        return Err(SampleError::TooLarge(
            "resident job exceeds node or sweep limit".into(),
        ));
    }
    if job.graph.j.len() != job.graph.edges.len() {
        return Err(SampleError::TooLarge(
            "resident jobs require one coefficient per edge".into(),
        ));
    }
    Ok(())
}

pub(crate) struct Prepared {
    pub(crate) job: StreamJob,
    /// None selects the non-slot path, including direct empty-graph answers.
    pub(crate) data: Result<Option<PreparedData>, SampleError>,
}

pub(crate) struct PreparedData {
    pub(crate) schedule: PreparedSchedule,
    pub(crate) inputs: PreparedInputs,
}

#[derive(Default)]
struct Preparer {
    topology: Option<(SelfFeedingTopology, Vec<(usize, usize)>)>,
    /// Whether the cached topology is the chain topology.
    chain: bool,
    schedules: ScheduleCache,
}

impl Preparer {
    fn prepare(
        &mut self,
        job: &mut StreamJob,
        settings: CascadeSettings,
    ) -> Result<Option<PreparedData>, SampleError> {
        if job.graph.num_nodes() == 0 {
            return Ok(None);
        }
        validate(job)?;
        if job.params.num_sweeps == 0 || !sampler::device_energy_exact(&job.graph) {
            sampler::validate_batch(&[&job.graph], &job.params, Kernel::Msa)?;
            return Ok(None);
        }
        let inputs = self
            .topology
            .as_ref()
            .and_then(|(topology, edges)| PreparedInputs::new(&job.graph, topology, edges));
        let inputs = match inputs {
            Some(inputs) => inputs,
            None => {
                // Degree depends only on nodes and ordered edges. The cache is
                // established only after validate_batch, and coefficient filling
                // checks exact topology equality on every hit. Recounting degree
                // would allocate and walk the same 41,515 edges on every job.
                sampler::validate_batch(&[&job.graph], &job.params, Kernel::Msa)?;
                // This cache contains no Metal objects. Coloring does not
                // change the CSR positions used for coefficient order.
                let topology = SelfFeedingTopology::build(&job.graph);
                let inputs = PreparedInputs::new(&job.graph, &topology, &job.graph.edges)
                    .ok_or_else(|| SampleError::Driver("preparation topology mismatch".into()))?;
                self.chain = topology_fingerprint(job.graph.num_nodes(), &job.graph.edges)
                    == CHAIN_GATES.fingerprint;
                tracing::info!(
                    chain = self.chain,
                    gated = settings.chain_gated(self.chain, &job.params),
                    nodes = job.graph.num_nodes(),
                    edges = job.graph.edges.len(),
                    "cascade topology prepared"
                );
                self.topology = Some((topology, job.graph.edges.clone()));
                inputs
            }
        };
        let gated = settings.chain_gated(self.chain, &job.params);
        // Open gates measure the budget the job asked for.
        if gated && !settings.open_gates {
            job.params.num_sweeps = CHAIN_GATES.full_sweeps;
        }
        let schedule = self.schedules.prepare(job, settings, gated)?;
        Ok(Some(PreparedData { schedule, inputs }))
    }
}

struct Work {
    job: StreamJob,
    settings: CascadeSettings,
    reply: mpsc::SyncSender<Prepared>,
}

// Warm Advantage2 preparation measured 251 us/job on one M4 Max host thread
// (preparation_cost_advantage2), down from 436 us before the cache/beta fixes.
// Allow 3x that cost under concurrent load: four workers / 0.753 ms = 5,312
// jobs/s, 15% above the 4,615 jobs/s target. Four decode workers leave four
// of the 12 performance cores, plus four efficiency cores, for the runner,
// producers (~1.3 cores) and other work. Use std threads for explicit queue
// ownership and joining, without borrowing Rayon's decode workers.
// At most 40 jobs (two nominal 20-slot batches) are queued, preparing, or ready
// in total. Per-worker request queues remove shared-receiver contention;
// one bounded reply per job preserves admission order.
/// How often the runner logs the cascade report.
const REPORT_PERIOD: Duration = Duration::from_secs(60);
pub(crate) const PREP_WORKERS: usize = 4;
pub(crate) const PREP_BOUND: usize = 40;

pub(crate) struct Preparation {
    requests: Vec<mpsc::SyncSender<Work>>,
    next_worker: usize,
    ready: VecDeque<mpsc::Receiver<Prepared>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl Preparation {
    pub(crate) fn new() -> Result<Self, SampleError> {
        let mut preparation = Self {
            requests: Vec::new(),
            next_worker: 0,
            ready: VecDeque::new(),
            workers: Vec::new(),
        };
        for index in 0..PREP_WORKERS {
            // Each receiver has one owner. No worker holds a shared lock while
            // waiting for work. PREP_BOUND still bounds all outstanding replies.
            let (tx, rx) = mpsc::sync_channel::<Work>(PREP_BOUND);
            let worker = std::thread::Builder::new()
                .name(format!("resident-prepare-{index}"))
                .spawn(move || {
                    let mut preparer = Preparer::default();
                    loop {
                        let request = rx.recv();
                        let Ok(Work {
                            mut job,
                            settings,
                            reply,
                        }) = request
                        else {
                            break;
                        };
                        // Keep ownership of the job outside unwinding so a failed
                        // worker still returns it to the runner exactly once.
                        let data = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            preparer.prepare(&mut job, settings)
                        }))
                        .unwrap_or_else(|_| {
                            Err(SampleError::Driver("job preparation panicked".into()))
                        });
                        let _ = reply.send(Prepared { job, data });
                    }
                })
                .map_err(|error| {
                    SampleError::Driver(format!("start preparation worker: {error}"))
                })?;
            preparation.workers.push(worker);
            preparation.requests.push(tx);
        }
        Ok(preparation)
    }

    pub(crate) fn len(&self) -> usize {
        self.ready.len()
    }

    fn fill(&mut self, jobs: &mut Receiver<StreamJob>, settings: CascadeSettings, eof: &mut bool) {
        while self.len() < PREP_BOUND && !*eof {
            match jobs.try_recv() {
                Ok(job) => self.submit(job, settings),
                Err(TryRecvError::Disconnected) => *eof = true,
                Err(TryRecvError::Empty) => break,
            }
        }
    }

    pub(crate) fn submit(&mut self, job: StreamJob, settings: CascadeSettings) {
        let (reply, result) = mpsc::sync_channel(1);
        let work = Work {
            job,
            settings,
            reply,
        };
        // The runner submits only below PREP_BOUND, so this queue cannot block.
        // A disconnected queue is returned through the same result owner.
        let tx = &self.requests[self.next_worker];
        self.next_worker = (self.next_worker + 1) % self.requests.len();
        if let Err(mpsc::SendError(work)) = tx.send(work) {
            let _ = work.reply.send(Prepared {
                job: work.job,
                data: Err(SampleError::Driver("preparation workers stopped".into())),
            });
        }
        self.ready.push_back(result);
    }

    pub(crate) fn next(&mut self) -> Option<Prepared> {
        self.ready.pop_front().and_then(|result| result.recv().ok())
    }
}

impl Drop for Preparation {
    fn drop(&mut self) {
        self.requests.clear();
        // Replies have capacity one each. Joining never depends on the runner
        // draining results or on input EOF, including after output closes.
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn reject_or_cancel(
    out: &Sender<StreamResult>,
    job: StreamJob,
    error: &SampleError,
    cancel: &CancelToken,
) {
    let _ = out.blocking_send(StreamResult {
        job_id: job.job_id,
        outcome: if cancel.is_cancelled(job.watermark) {
            StreamOutcome::Cancelled
        } else {
            StreamOutcome::Completed(Err(error.to_sample_error()))
        },
        device_access_time_us: 0,
    });
}

fn reject_tail(
    pending: Option<StreamJob>,
    jobs: &mut Receiver<StreamJob>,
    out: &Sender<StreamResult>,
    error: &SampleError,
    cancel: &CancelToken,
) {
    jobs.close();
    if let Some(job) = pending {
        reject_or_cancel(out, job, error, cancel);
    }
    // close() revokes new reservations, but existing permits may still send.
    // None means both buffered jobs and outstanding permits have drained.
    while let Some(job) = jobs.blocking_recv() {
        reject_or_cancel(out, job, error, cancel);
    }
}

fn reject_preparation(
    preparation: &mut Preparation,
    pending: Option<Prepared>,
    jobs: &mut Receiver<StreamJob>,
    out: &Sender<StreamResult>,
    error: &SampleError,
    cancel: &CancelToken,
) {
    jobs.close();
    while let Some(prepared) = preparation.next() {
        reject_or_cancel(out, prepared.job, error, cancel);
    }
    reject_tail(
        pending.map(|prepared| prepared.job),
        jobs,
        out,
        error,
        cancel,
    );
}

fn run_fallback(
    device: &MetalDevice,
    job: StreamJob,
    out: &Sender<StreamResult>,
    gov: &dyn GpuGovernor,
    cancel: &CancelToken,
) -> Result<(), SampleError> {
    let mut device_access_time_us = 0;
    let result = (|| {
        let mut batch = sampler::encode_batch(device, &[&job.graph], &job.params, Kernel::Msa, 1)?;
        for _ in 0..batch.chunk_count() {
            if out.is_closed() || cancel.is_cancelled(job.watermark) {
                break;
            }
            crate::streaming::yield_gate(out, gov);
            if !batch.commit_next(|| out.is_closed() || cancel.is_cancelled(job.watermark)) {
                break;
            }
            batch.wait_until_completed();
        }
        device_access_time_us = batch.gpu_time_us();
        gov.record_gpu_busy_us(device_access_time_us);
        if let Some(status) = batch.failed_status() {
            return Err(SampleError::Driver(format!(
                "metal command buffer did not complete: status {status:?}"
            )));
        }
        if out.is_closed() || cancel.is_cancelled(job.watermark) {
            return Ok(Vec::new());
        }
        let mut reads = sampler::harvest_batch(&batch, &[&job.graph])?.remove(0);
        reads.truncate(job.params.num_reads.max(1));
        Ok(reads)
    })();
    let fault = match &result {
        Err(SampleError::TooLarge(_)) | Ok(_) => None,
        Err(SampleError::Driver(message)) => Some(SampleError::Driver(message.clone())),
        Err(SampleError::Metal(error)) => Some(SampleError::Driver(error.to_string())),
    };
    let _ = out.blocking_send(StreamResult {
        job_id: job.job_id,
        outcome: if out.is_closed() || cancel.is_cancelled(job.watermark) {
            StreamOutcome::Cancelled
        } else {
            StreamOutcome::Completed(result.map_err(|error| error.to_sample_error()))
        },
        device_access_time_us,
    });
    match fault {
        Some(error) => Err(error),
        None => Ok(()),
    }
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
    let mut preparation = match Preparation::new() {
        Ok(preparation) => preparation,
        Err(error) => {
            jobs.close();
            reject_tail(None, &mut jobs, out, &error, cancel);
            *store.lock().unwrap_or_else(|p| p.into_inner()) = Some(controller);
            return;
        }
    };
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
        preparation.fill(&mut jobs, config, &mut eof);
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
                controller.report(Instant::now(), REPORT_PERIOD);
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
                // Refill as admissions consume replies. The preparation bound
                // must not cap slot occupancy for small read counts or a larger
                // configured threadgroup budget.
                preparation.fill(&mut jobs, config, &mut eof);
                let full = pools.as_ref().is_some_and(|ps| {
                    let pool = &ps[turn];
                    pool.slots.live() >= scale_budget(pool.slots.capacity(), gov.budget_scale())
                });
                if full {
                    break;
                }
                pending = preparation.next();
            }
            let Some(prepared) = pending.take() else {
                break;
            };
            let Prepared { job, data } = prepared;
            if cancel.is_cancelled(job.watermark) {
                let _ = out.blocking_send(StreamResult {
                    job_id: job.job_id,
                    outcome: StreamOutcome::Cancelled,
                    device_access_time_us: 0,
                });
                continue;
            }
            let mut data = match data {
                Ok(Some(data)) => data,
                Ok(None) => {
                    if job.graph.num_nodes() == 0 {
                        crate::streaming::answer_empty(out, job);
                        continue;
                    }
                    // Run on the Metal-owning runner only after live slots drain,
                    // so a rare full-budget fallback cannot stall slot stepping.
                    if !empty {
                        pending = Some(Prepared {
                            job,
                            data: Ok(None),
                        });
                        break;
                    }
                    let started = Instant::now();
                    let result = run_fallback(device, job, out, gov, cancel);
                    // Fallback time belongs to the governor, not cascade load.
                    window += started.elapsed();
                    if let Err(error) = result {
                        fault = Some(error);
                        break 'run;
                    }
                    continue;
                }
                Err(error) => {
                    send_reject(out, job, error.to_sample_error());
                    continue;
                }
            };
            if pools.as_ref().is_none_or(|ps| !ps[turn].fits(&job)) {
                if !empty {
                    pending = Some(Prepared {
                        job,
                        data: Ok(Some(data)),
                    });
                    break;
                }
                let capacity = batch_size_for_reads(
                    Kernel::Msa,
                    job.params.num_reads.clamp(1, sampler::MAX_READS),
                );
                // Both pools are empty, so changing storage cannot discard work.
                let rebuilt = Pool::new(device, &job, capacity)
                    .and_then(|a| Pool::new(device, &job, capacity).map(|b| [a, b]));
                match rebuilt {
                    Ok(rebuilt) => {
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
            let ticket = match controller.admit_prepared(&job, &mut data.schedule) {
                Ok(ticket) => ticket,
                Err(error) => {
                    send_reject(out, job, error.to_sample_error());
                    continue;
                }
            };
            let admitted = pool.slots.admit_prepared(
                data.inputs,
                data.schedule.betas,
                data.schedule.checkpoints,
                job.params.seed,
            );
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
        if empty && eof && pending.is_none() && preparation.len() == 0 {
            break;
        }
        if let Some(pools) = &mut pools {
            if gov.should_throttle() {
                // Drain submitted work before the pause so it frees GPU time.
                for pool in pools.iter() {
                    pool.slots.wait();
                }
                crate::streaming::yield_gate(out, gov);
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

    jobs.close();
    if let Some(pools) = &mut pools {
        for pool in pools {
            pool.slots.wait();
            for live in pool.live.iter_mut().filter_map(Option::take) {
                controller.finish(&live.ticket, None, false);
                if let Some(error) = &fault {
                    send_reject(out, live.job, error.to_sample_error());
                } else {
                    let _ = out.blocking_send(StreamResult {
                        job_id: live.job.job_id,
                        outcome: StreamOutcome::Cancelled,
                        device_access_time_us: live.accounted_us,
                    });
                }
            }
        }
    }
    if let Some(error) = &fault {
        tracing::error!(%error, "resident stream failed");
    }
    // At normal EOF these are empty. On a fault, reject every outstanding job.
    // A closed output cannot receive an answer, but each owned job still has
    // one terminal send attempt, and workers never send stream results.
    let error =
        fault.unwrap_or_else(|| SampleError::Driver("resident result receiver closed".into()));
    reject_preparation(&mut preparation, pending, &mut jobs, out, &error, cancel);
    *store.lock().unwrap_or_else(|p| p.into_inner()) = Some(controller);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slots::SlotJob;
    use crate::{IsingGraph, MetalSampler, SampleParams};
    use quip_solver_core::Sampler;

    #[test]
    fn shutdown_drains_a_permit_held_across_close() {
        let (tx, mut jobs) = tokio::sync::mpsc::channel(2);
        let permit = tx.clone().try_reserve_owned().unwrap();
        let (out, mut results) = tokio::sync::mpsc::channel(2);
        let (finished, completion) = mpsc::channel();
        let cleanup = std::thread::spawn(move || {
            jobs.close();
            reject_tail(
                Some(job(0, 32)),
                &mut jobs,
                &out,
                &SampleError::Driver("injected fault".into()),
                &CancelToken::default(),
            );
            finished.send(()).unwrap();
        });
        assert_eq!(
            results.blocking_recv().unwrap().job_id,
            0usize.to_le_bytes()
        );
        // The first rejection proves that cleanup has closed the receiver.
        // Keep tx alive: draining must wait for permits, not sender EOF.
        let early_completion = completion.recv_timeout(Duration::from_millis(50));
        permit.send(job(1, 32));
        cleanup.join().unwrap();
        assert_eq!(early_completion, Err(mpsc::RecvTimeoutError::Timeout));
        assert_eq!(
            results.blocking_recv().expect("reserved job lost").job_id,
            1usize.to_le_bytes()
        );
        assert!(results.blocking_recv().is_none());
        assert!(tx.is_closed());
    }

    #[test]
    fn chain_jobs_get_the_deep_final_budget_unless_gates_are_open() {
        let edges: Vec<(usize, usize)> = include_str!("../tests/fixtures/advantage2-system1.edges")
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
            .map(|line| {
                let mut words = line.split_whitespace();
                (
                    words.next().unwrap().parse().unwrap(),
                    words.next().unwrap().parse().unwrap(),
                )
            })
            .filter(|&(u, v): &(usize, usize)| (u.min(v), u.max(v)) != (880, 2695))
            .collect();
        let mut chain = job(0, 14_336);
        chain.graph = IsingGraph::new(
            vec![0.0; 4577],
            (0..edges.len())
                .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
                .collect(),
            edges,
        );
        chain.params.num_reads = 64;
        let mut preparer = Preparer::default();
        let data = preparer
            .prepare(&mut chain, CascadeSettings::default())
            .unwrap()
            .unwrap();
        assert_eq!(chain.params.num_sweeps, CHAIN_GATES.full_sweeps);
        assert_eq!(
            data.schedule.checkpoints.last(),
            Some(&CHAIN_GATES.full_sweeps)
        );
        let open = CascadeSettings {
            open_gates: true,
            ..CascadeSettings::default()
        };
        chain.params.num_sweeps = 14_336;
        preparer.prepare(&mut chain, open).unwrap().unwrap();
        assert_eq!(chain.params.num_sweeps, 14_336);
        let mut other = job(1, 14_336);
        preparer
            .prepare(&mut other, CascadeSettings::default())
            .unwrap();
        assert_eq!(other.params.num_sweeps, 14_336);
    }

    #[test]
    #[expect(
        clippy::print_stderr,
        reason = "host preparation cost informs the fixed worker budget"
    )]
    fn preparation_cost_advantage2() {
        let edges = include_str!("../tests/fixtures/advantage2-system1.edges")
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
            .map(|line| {
                let mut words = line.split_whitespace();
                (
                    words.next().unwrap().parse().unwrap(),
                    words.next().unwrap().parse().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let mut job = job(0, 32);
        let mut rng = 12345u64;
        let mut draw = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        job.graph = IsingGraph::new(
            (0..4577)
                .map(|_| [-1.0, 0.0, 1.0][(draw() % 3) as usize])
                .collect(),
            (0..edges.len())
                .map(|_| if draw() & 1 == 0 { -1.0 } else { 1.0 })
                .collect(),
            edges,
        );
        job.params.num_reads = 64;
        let mut preparer = Preparer::default();
        let settings = CascadeSettings::default();
        preparer.prepare(&mut job, settings).unwrap();
        let start = Instant::now();
        for _ in 0..1000 {
            std::hint::black_box(
                preparer
                    .prepare(std::hint::black_box(&mut job), settings)
                    .unwrap(),
            );
        }
        eprintln!(
            "Advantage2 preparation: {:.3} us/job (1000 jobs, warm topology, one host thread)",
            start.elapsed().as_secs_f64() * 1000.0
        );
        let mut preparation = Preparation::new().unwrap();
        let start = Instant::now();
        let mut submitted = 0usize;
        let mut completed = 0usize;
        loop {
            while submitted < 4000 && preparation.len() < PREP_BOUND {
                preparation.submit(
                    StreamJob {
                        job_id: submitted.to_le_bytes().to_vec(),
                        graph: job.graph.clone(),
                        params: job.params.clone(),
                        watermark: None,
                    },
                    settings,
                );
                submitted += 1;
            }
            let Some(prepared) = preparation.next() else {
                break;
            };
            assert_eq!(prepared.job.job_id, completed.to_le_bytes());
            std::hint::black_box(prepared.data.unwrap());
            completed += 1usize;
        }
        assert_eq!(completed, 4000);
        eprintln!("Advantage2 preparation stage: {:.1} jobs/s ({PREP_WORKERS} workers, 4000 jobs, includes graph copies and cache startup)", completed as f64 / start.elapsed().as_secs_f64());
    }

    #[test]
    fn preparation_cache_never_bypasses_changed_topology_or_scalar_limits() {
        let mut preparer = Preparer::default();
        let settings = CascadeSettings::default();
        let mut input = job(0, 32);
        preparer.prepare(&mut input, settings).unwrap();
        input.graph.edges = (1..=sampler::MSA_MAX_DEG + 1).map(|v| (0, v)).collect();
        input.graph.j = vec![1.0; input.graph.edges.len()];
        assert!(matches!(
            preparer.prepare(&mut input, settings),
            Err(SampleError::TooLarge(_))
        ));
        let mut input = job(1, sampler::MAX_SWEEPS + 1);
        assert!(matches!(
            preparer.prepare(&mut input, settings),
            Err(SampleError::TooLarge(_))
        ));
        input.params.num_sweeps = 32;
        input.graph.j[0] = 0.5;
        assert!(preparer.prepare(&mut input, settings).unwrap().is_none());
    }

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
    fn preparation_bounds_prefetch_and_drains_eof() {
        let count = PREP_BOUND * 3;
        let (tx, mut jobs) = tokio::sync::mpsc::channel(count);
        for id in 0..count {
            tx.try_send(job(id, 32)).unwrap();
        }
        drop(tx);
        let mut preparation = Preparation::new().unwrap();
        let mut eof = false;
        preparation.fill(&mut jobs, CascadeSettings::default(), &mut eof);
        assert_eq!(preparation.len(), PREP_BOUND);
        assert_eq!(jobs.len(), count - PREP_BOUND);
        for id in 0..count {
            let prepared = preparation.next().unwrap();
            assert_eq!(prepared.job.job_id, id.to_le_bytes());
            preparation.fill(&mut jobs, CascadeSettings::default(), &mut eof);
            assert!(preparation.len() <= PREP_BOUND);
        }
        assert!(eof);
        assert!(preparation.next().is_none());
    }

    #[test]
    fn closed_output_cleanup_drains_preparation_without_waiting_for_input_eof() {
        let (done, finished) = mpsc::channel();
        let runner = std::thread::spawn(move || {
            let mut preparation = Preparation::new().unwrap();
            for id in 0..PREP_BOUND {
                preparation.submit(job(id, 32), CascadeSettings::default());
            }
            let (tx, mut jobs) = tokio::sync::mpsc::channel(1);
            tx.try_send(job(PREP_BOUND, 32)).unwrap();
            let (out, results) = tokio::sync::mpsc::channel(1);
            drop(results);
            reject_preparation(
                &mut preparation,
                None,
                &mut jobs,
                &out,
                &SampleError::Driver("closed output".into()),
                &CancelToken::default(),
            );
            assert_eq!(preparation.len(), 0);
            assert_eq!(jobs.len(), 0);
            assert!(tx.is_closed());
            drop(preparation);
            done.send(()).unwrap();
        });
        finished
            .recv_timeout(Duration::from_secs(5))
            .expect("closed output cleanup hung");
        runner.join().unwrap();
    }

    #[test]
    fn preparation_returns_valid_and_invalid_jobs_once_in_input_order() {
        let mut preparation = Preparation::new().unwrap();
        assert_eq!(preparation.workers.len(), PREP_WORKERS);
        for worker in &preparation.workers {
            assert_ne!(worker.thread().id(), std::thread::current().id());
        }
        for id in 0..PREP_BOUND {
            let mut job = job(
                id,
                if id % 3 == 0 {
                    sampler::MAX_SWEEPS + 1
                } else {
                    64
                },
            );
            if id % 2 == 0 {
                job.graph.edges.swap(0, 1);
            }
            preparation.submit(job, CascadeSettings::default());
        }
        assert_eq!(preparation.len(), PREP_BOUND);
        for id in 0..PREP_BOUND {
            let prepared = preparation.next().unwrap();
            assert_eq!(prepared.job.job_id, id.to_le_bytes());
            if id % 3 == 0 {
                assert!(matches!(prepared.data, Err(SampleError::TooLarge(_))));
            } else {
                let data = prepared.data.unwrap().unwrap();
                assert_eq!(data.schedule.betas.len(), 64);
                assert_eq!(data.schedule.checkpoints.last(), Some(&64));
            }
        }
        assert_eq!(preparation.len(), 0);
        assert!(preparation.next().is_none());
    }

    #[test]
    fn preparation_shutdown_does_not_need_result_consumption_or_input_eof() {
        let (finished, done) = mpsc::channel();
        let runner = std::thread::spawn(move || {
            let mut preparation = Preparation::new().unwrap();
            for id in 0..PREP_BOUND {
                preparation.submit(job(id, 64), CascadeSettings::default());
            }
            drop(preparation);
            finished.send(()).unwrap();
        });
        done.recv_timeout(Duration::from_secs(5))
            .expect("preparation shutdown hung");
        runner.join().unwrap();
    }

    #[test]
    fn fault_drains_preparation_pending_and_input_once() {
        let mut preparation = Preparation::new().unwrap();
        let (tx, mut jobs) = tokio::sync::mpsc::channel(2);
        let (out, mut results) = tokio::sync::mpsc::channel(PREP_BOUND + 3);
        for id in 0..PREP_BOUND {
            preparation.submit(
                job(id, if id % 2 == 0 { 0 } else { 64 }),
                CascadeSettings::default(),
            );
        }
        tx.try_send(job(PREP_BOUND + 1, 64)).unwrap();
        tx.try_send(job(PREP_BOUND + 2, 64)).unwrap();
        let error = SampleError::Driver("injected device fault".into());
        let pending = Prepared {
            job: job(PREP_BOUND, 64),
            data: Err(SampleError::TooLarge("invalid pending job".into())),
        };
        reject_preparation(
            &mut preparation,
            Some(pending),
            &mut jobs,
            &out,
            &error,
            &CancelToken::default(),
        );
        for id in 0..PREP_BOUND + 3 {
            let result = results.try_recv().unwrap();
            assert_eq!(result.job_id, id.to_le_bytes());
            assert!(matches!(result.outcome, StreamOutcome::Completed(Err(_))));
        }
        assert!(matches!(results.try_recv(), Err(TryRecvError::Empty)));
        drop(tx);
    }

    #[test]
    fn fault_cleanup_answers_non_exact_jobs_once_and_respects_cancellation() {
        let cancel = CancelToken::default();
        cancel.cancel_through(1);
        let non_exact = |id, watermark| {
            let mut job = job(id, 64);
            job.graph.j[0] = 0.5;
            job.watermark = Some(watermark);
            job
        };
        let mut preparation = Preparation::new().unwrap();
        preparation.submit(non_exact(0, 1), CascadeSettings::default());
        preparation.submit(non_exact(1, 2), CascadeSettings::default());
        let pending = Prepared {
            job: non_exact(2, 1),
            data: Ok(None),
        };
        let (tx, mut jobs) = tokio::sync::mpsc::channel(2);
        tx.try_send(non_exact(3, 1)).unwrap();
        tx.try_send(non_exact(4, 2)).unwrap();
        let (out, mut results) = tokio::sync::mpsc::channel(5);
        let error = SampleError::Driver("injected fault".into());
        reject_preparation(
            &mut preparation,
            Some(pending),
            &mut jobs,
            &out,
            &error,
            &cancel,
        );
        for id in 0usize..5 {
            let result = results.try_recv().unwrap();
            assert_eq!(result.job_id, id.to_le_bytes());
            if id == 1 || id == 4 {
                assert!(matches!(result.outcome, StreamOutcome::Completed(Err(_))));
            } else {
                assert!(matches!(result.outcome, StreamOutcome::Cancelled));
            }
        }
        assert!(matches!(results.try_recv(), Err(TryRecvError::Empty)));
        assert!(tx.is_closed());
    }

    #[test]
    fn fault_rejects_pending_and_queued_jobs_without_waiting_for_eof() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(2);
        let (out, mut results) = tokio::sync::mpsc::channel(3);
        tx.try_send(job(1, 8)).unwrap();
        tx.try_send(job(2, 8)).unwrap();
        let error = SampleError::Driver("injected fault".into());
        reject_tail(
            Some(job(0, 8)),
            &mut rx,
            &out,
            &error,
            &CancelToken::default(),
        );
        for id in 0usize..3 {
            let result = results.try_recv().expect("one rejection per submitted job");
            assert_eq!(result.job_id, id.to_le_bytes());
            match result.outcome {
                StreamOutcome::Completed(Err(actual)) => {
                    assert_eq!(actual, error.to_sample_error())
                }
                StreamOutcome::Completed(Ok(_)) | StreamOutcome::Cancelled => {
                    panic!("expected device fault")
                }
            }
        }
        assert!(matches!(results.try_recv(), Err(TryRecvError::Empty)));
        assert!(matches!(rx.try_recv(), Err(TryRecvError::Disconnected)));
        // The sender stays open throughout fault cleanup.
        drop(tx);
    }

    #[test]
    fn admission_grows_when_governor_scale_rises() {
        use std::cell::{Cell, RefCell};

        if MetalDevice::device_count() == 0 {
            #[expect(clippy::print_stderr, reason = "device tests report a sandbox skip")]
            {
                eprintln!("skipping governor growth test: no Metal device");
            }
            return;
        }
        let nominal = batch_size_for_reads(Kernel::Msa, 4);
        if nominal < 2 {
            tracing::warn!("skipping governor growth test: batch capacity is one");
            return;
        }
        struct Governor {
            raised: Cell<bool>,
            completed: RefCell<Vec<usize>>,
            out: Sender<StreamResult>,
        }
        impl GpuGovernor for Governor {
            fn should_throttle(&self) -> bool {
                false
            }
            fn budget_scale(&self) -> f64 {
                if self.raised.get() {
                    1.0
                } else {
                    0.25
                }
            }
            fn record_gpu_busy_us(&self, us: u64) {
                // Results stay buffered, so each delta counts a harvested batch.
                self.completed
                    .borrow_mut()
                    .push(self.out.max_capacity() - self.out.capacity());
                if us > 0 {
                    self.raised.set(true);
                }
            }
        }
        let count = nominal * 6;
        let (tx, rx) = tokio::sync::mpsc::channel(count);
        let (out, mut results) = tokio::sync::mpsc::channel(count);
        for id in 0..count {
            tx.try_send(job(id, 8)).unwrap();
        }
        drop(tx);
        let gov = Governor {
            raised: Cell::new(false),
            completed: RefCell::new(vec![0]),
            out: out.clone(),
        };
        run(
            &MetalDevice::open(0).unwrap(),
            &Mutex::new(CascadeSettings::default()),
            &Mutex::new(None),
            rx,
            &out,
            &gov,
            &CancelToken::default(),
        );
        let mut completed = gov.completed.into_inner();
        completed.push(count - out.capacity());
        let batches: Vec<_> = completed
            .windows(2)
            .map(|p| p[1] - p[0])
            .filter(|&n| n > 0)
            .collect();
        assert_eq!(batches.first(), Some(&scale_budget(nominal, 0.25)));
        assert_eq!(
            batches.iter().max(),
            Some(&nominal),
            "admission stayed at initial scale: {batches:?}"
        );
        for id in 0..count {
            let result = results.try_recv().unwrap();
            match result.outcome {
                StreamOutcome::Completed(Ok(reads)) => assert_eq!(reads.len(), 4, "job {id}"),
                StreamOutcome::Completed(Err(error)) => panic!("{error}"),
                StreamOutcome::Cancelled => panic!("unexpected cancellation"),
            }
        }
        assert!(matches!(results.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn preparation_accepts_non_exact_jobs_with_ordinary_msa_limits() {
        let mut input = job(0, 64);
        input.graph.j[0] = 0.5;
        let mut preparer = Preparer::default();
        assert!(preparer
            .prepare(&mut input, CascadeSettings::default())
            .unwrap()
            .is_none());
        input.params.num_sweeps = sampler::MAX_SWEEPS + 1;
        assert!(matches!(
            preparer.prepare(&mut input, CascadeSettings::default()),
            Err(SampleError::TooLarge(_))
        ));
        input.params.num_sweeps = 0;
        assert!(preparer
            .prepare(&mut input, CascadeSettings::default())
            .unwrap()
            .is_none());
        input.params.num_sweeps = 64;
        input.graph.edges = (1..=sampler::MSA_MAX_DEG + 1).map(|v| (0, v)).collect();
        input.graph.j = vec![0.5; input.graph.edges.len()];
        assert!(matches!(
            preparer.prepare(&mut input, CascadeSettings::default()),
            Err(SampleError::TooLarge(_))
        ));
        input
            .graph
            .h
            .resize(sampler::kernel_max_nodes(Kernel::Msa) + 1, 0.0);
        assert!(matches!(
            preparer.prepare(&mut input, CascadeSettings::default()),
            Err(SampleError::TooLarge(_))
        ));
    }

    #[test]
    fn zero_sweeps_select_batch_in_both_coefficient_domains() {
        let mut preparer = Preparer::default();
        for coefficient in [1.0, 0.5] {
            let mut input = job(0, 0);
            input.graph.j[0] = coefficient;
            assert!(preparer
                .prepare(&mut input, CascadeSettings::default())
                .unwrap()
                .is_none());
        }
    }

    #[test]
    fn public_stream_preserves_empty_and_zero_sweep_results() {
        let device = MetalDevice::open(0).unwrap();
        let mut inputs = Vec::new();
        let mut expected = Vec::new();
        for coefficient in [1.0, 0.5] {
            let mut input = job(inputs.len(), 0);
            input.graph.j[0] = coefficient;
            expected.push(
                sampler::sample_ising(&device, &input.graph, &input.params, Kernel::Msa).unwrap(),
            );
            inputs.push(input);
        }
        for reads in [0, 3] {
            let mut input = job(inputs.len(), 0);
            input.graph = IsingGraph::new(vec![], vec![], vec![]);
            input.params.num_reads = reads;
            expected.push(
                (0..reads.max(1))
                    .map(|_| quip_solver_core::SamplerResult {
                        spins: vec![],
                        energy_milli: 0,
                    })
                    .collect(),
            );
            inputs.push(input);
        }
        let count = inputs.len();
        let sampler = MetalSampler::new(
            device,
            crate::iokit_gov::UtilGovernor::start(0, 100, false),
            Kernel::Msa,
        );
        let (tx, rx) = tokio::sync::mpsc::channel(count);
        let (out, mut results) = tokio::sync::mpsc::channel(count);
        for input in inputs {
            tx.try_send(input).unwrap();
        }
        drop(tx);
        sampler.sample_stream(rx, out, CancelToken::default());
        for (id, reads) in expected.into_iter().enumerate() {
            let result = results.blocking_recv().unwrap();
            assert_eq!(result.job_id, id.to_le_bytes());
            match result.outcome {
                StreamOutcome::Completed(Ok(actual)) => assert_eq!(actual, reads),
                StreamOutcome::Completed(Err(error)) => panic!("{error}"),
                StreamOutcome::Cancelled => panic!("unexpected cancellation"),
            }
            if id >= 2 {
                assert_eq!(result.device_access_time_us, 0);
            }
        }
        assert!(results.blocking_recv().is_none());
    }

    #[test]
    fn fallback_consults_throttling_governor_before_submission() {
        use std::cell::Cell;
        struct Governor {
            checks: Cell<usize>,
            reported: Cell<u64>,
            start: Instant,
        }
        impl GpuGovernor for Governor {
            fn should_throttle(&self) -> bool {
                self.checks.set(self.checks.get() + 1);
                self.checks.get() <= 2
            }
            fn budget_scale(&self) -> f64 {
                1.0
            }
            fn record_gpu_busy_us(&self, us: u64) {
                assert!(self.checks.get() >= 3);
                assert!(self.start.elapsed() >= Duration::from_millis(50));
                self.reported.set(self.reported.get() + us);
            }
        }
        let device = MetalDevice::open(0).unwrap();
        let governor = Governor {
            checks: Cell::new(0),
            reported: Cell::new(0),
            start: Instant::now(),
        };
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let (out, mut results) = tokio::sync::mpsc::channel(1);
        let mut input = job(0, 32);
        input.graph.j[0] = 0.5;
        tx.try_send(input).unwrap();
        drop(tx);
        run(
            &device,
            &Mutex::new(CascadeSettings::default()),
            &Mutex::new(None),
            rx,
            &out,
            &governor,
            &CancelToken::default(),
        );
        let result = results.try_recv().unwrap();
        assert!(matches!(result.outcome, StreamOutcome::Completed(Ok(_))));
        assert!(result.device_access_time_us > 0);
        assert_eq!(governor.reported.get(), result.device_access_time_us);
    }

    #[test]
    fn invalid_admissions_fail_before_allocating_schedules() {
        let mut invalid = job(0, sampler::MAX_SWEEPS + 1);
        assert_eq!(
            validate(&invalid).unwrap_err().to_sample_error(),
            quip_solver_core::SampleError::Capacity
        );
        invalid.params.num_sweeps = 0;
        validate(&invalid).unwrap();
        invalid.params.num_sweeps = 32;
        invalid.graph.j.pop();
        assert_eq!(
            validate(&invalid).unwrap_err().to_sample_error(),
            quip_solver_core::SampleError::Capacity
        );
        invalid.graph.j.push(-0.5);
        validate(&invalid).unwrap();
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
            stages: [8, 0, 0, 0],
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
                schedule: schedule.into(),
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
        sampler.apply_config("cascade_stages = [8]\ncascade_keep = 100\ncascade_keep_min = 10\ncascade_keep_max = 100");
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
