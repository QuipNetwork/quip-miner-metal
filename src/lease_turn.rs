// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! First-come, first-served turns for feeding leases to the resident runner.
//!
//! One lease feeds salts at a time, so each lease finishes as fast as the
//! whole GPU allows instead of sharing it with every other open lease. A lease
//! gives up its turn once its last salt is queued, not when its last unit
//! finishes: the next lease then fills the slots that the previous lease's
//! tail frees, and the GPU never drains at a lease boundary.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, PoisonError};
use std::time::Duration;

/// How often a waiting lease re-checks whether it should stop waiting.
const POLL: Duration = Duration::from_millis(5);

#[derive(Default)]
pub(crate) struct LeaseTurns {
    state: Mutex<Queue>,
    changed: Condvar,
}

#[derive(Default)]
struct Queue {
    next_id: u64,
    waiting: VecDeque<u64>,
}

impl LeaseTurns {
    /// Join the back of the queue.
    pub(crate) fn join(&self) -> Turn<'_> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let id = state.next_id;
        state.next_id += 1;
        state.waiting.push_back(id);
        Turn {
            turns: self,
            id,
            released: false,
        }
    }
}

/// A place in the queue. Dropping it gives up the place, so every exit path
/// lets the next lease run.
pub(crate) struct Turn<'a> {
    turns: &'a LeaseTurns,
    id: u64,
    released: bool,
}

impl Turn<'_> {
    /// Join order: turns are granted in increasing `id`.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Block until this lease is at the front of the queue. Returns `false`,
    /// without the turn, as soon as `stop` returns true.
    pub(crate) fn wait(&self, stop: impl Fn() -> bool) -> bool {
        let mut state = self
            .turns
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        loop {
            if state.waiting.front() == Some(&self.id) {
                return true;
            }
            if stop() {
                return false;
            }
            state = self
                .turns
                .changed
                .wait_timeout(state, POLL)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    /// Whether [`Turn::release`] has run.
    pub(crate) fn released(&self) -> bool {
        self.released
    }

    /// Leave the queue and wake the leases behind this one. Idempotent.
    pub(crate) fn release(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        let mut state = self
            .turns
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        state.waiting.retain(|&id| id != self.id);
        drop(state);
        self.turns.changed.notify_all();
    }
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[test]
    fn turns_are_granted_in_join_order() {
        let turns = LeaseTurns::default();
        let mut first = turns.join();
        let second = turns.join();
        assert!(first.wait(|| false));
        assert!(!second.wait(|| true), "second waits behind first");
        first.release();
        assert!(second.wait(|| false));
    }

    #[test]
    fn a_stopped_waiter_leaves_without_blocking_the_queue() {
        let turns = LeaseTurns::default();
        let first = turns.join();
        let stopped = turns.join();
        let third = turns.join();
        assert!(!stopped.wait(|| true));
        drop(stopped);
        drop(first);
        assert!(third.wait(|| false), "a dropped waiter gives up its place");
    }

    #[test]
    fn release_wakes_a_blocked_waiter() {
        let turns = Arc::new(LeaseTurns::default());
        let mut first = turns.join();
        let started = Arc::new(AtomicBool::new(false));
        let waiter = {
            let turns = Arc::clone(&turns);
            let started = Arc::clone(&started);
            std::thread::spawn(move || {
                let second = turns.join();
                started.store(true, Ordering::Release);
                second.wait(|| false)
            })
        };
        while !started.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        first.release();
        first.release();
        assert!(waiter.join().unwrap());
    }
}
