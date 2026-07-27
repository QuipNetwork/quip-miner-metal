//! IOKit / GPU-util governor (port of `nvml_gov` public interface).
//!
//! Static util ceiling from config; optional background poll when yielding.
//! When yielding and observed util > 90%, the session loop inserts a brief
//! pause so sibling GPU users get time slices.
//!
//! # Sensor source
//!
//! Reads Apple GPU device utilization via IOKit: walks every `IOAccelerator`
//! service, reads its `PerformanceStatistics` CFDictionary, and takes the max
//! `"Device Utilization %"` value across services (a Mac normally has one).
//! This is a direct Rust port of the `ctypes` IOKit binding in
//! `GPU/macos_sensors.py::_query_iokit_gpu_utilization` — same service name,
//! same dictionary keys, same "return 0 on any failure" contract. Keeps the
//! **identical public interface** as `nvml_gov::UtilGovernor` so `session.rs`
//! calls it unchanged.
//!
//! This crate does not use the private/undocumented IOReport channel-sampling
//! path (`GPU/macos_sensors.py`'s `_ioreport_residency`, which is a stub
//! there too) — only the documented IOKit `IOAccelerator` service query.

use core_foundation_sys::base::{CFGetTypeID, CFRelease, CFTypeRef};
use core_foundation_sys::dictionary::{
    CFDictionaryGetTypeID, CFDictionaryGetValue, CFDictionaryRef, CFMutableDictionaryRef,
};
use core_foundation_sys::number::{
    kCFNumberSInt64Type, CFNumberGetTypeID, CFNumberGetValue, CFNumberRef,
};
use core_foundation_sys::string::{kCFStringEncodingUTF8, CFStringCreateWithCString, CFStringRef};
use io_kit_sys::types::io_iterator_t;
use io_kit_sys::{
    kIOMasterPortDefault, IOIteratorNext, IOObjectRelease, IORegistryEntryCreateCFProperties,
    IOServiceGetMatchingServices, IOServiceMatching,
};
use std::ffi::{c_void, CString};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Reconfigurable governor knobs plus the latest util sample, shared with the
/// poll thread.
#[derive(Debug)]
struct Knobs {
    /// Util ceiling 1–100; throttle fires above it when yielding.
    ceiling: AtomicU32,
    yielding: AtomicBool,
    /// Last GPU util percent 0–100 (0 while not yielding).
    last_util: AtomicU32,
    stop: AtomicBool,
}

/// Shared utilization sample and reconfigurable governor knobs.
#[derive(Debug)]
pub struct UtilGovernor {
    knobs: Arc<Knobs>,
    handle: Option<JoinHandle<()>>,
}

impl UtilGovernor {
    /// Start the util poller. Values come from the CLI; `Configure` may later
    /// override them via [`reconfigure`](Self::reconfigure). The poll thread
    /// runs regardless of `yielding` (so a later `false -> true` override starts
    /// sampling with no thread churn) but only records util while yielding.
    ///
    /// Falls back to a silent no-op if sensors are unavailable (miner still
    /// runs; util stays 0 and throttle never fires).
    pub fn start(device_index: u32, utilization_ceiling: u32, yielding: bool) -> Self {
        let knobs = Arc::new(Knobs {
            ceiling: AtomicU32::new(utilization_ceiling.clamp(1, 100)),
            yielding: AtomicBool::new(yielding),
            last_util: AtomicU32::new(0),
            stop: AtomicBool::new(false),
        });
        let knobs_thread = Arc::clone(&knobs);
        let handle = Some(thread::spawn(move || {
            poll_loop(device_index, &knobs_thread)
        }));
        Self { knobs, handle }
    }

    /// Override the ceiling and yielding flag at runtime (config over CLI).
    pub fn reconfigure(&self, utilization_ceiling: u32, yielding: bool) {
        self.knobs
            .ceiling
            .store(utilization_ceiling.clamp(1, 100), Ordering::Relaxed);
        self.knobs.yielding.store(yielding, Ordering::Relaxed);
    }

    /// Current ceiling (CLI value, or the config override once applied).
    pub fn utilization_ceiling(&self) -> u32 {
        self.knobs.ceiling.load(Ordering::Relaxed)
    }

    /// Current yielding flag (CLI value, or the config override once applied).
    pub fn yielding(&self) -> bool {
        self.knobs.yielding.load(Ordering::Relaxed)
    }

    /// Last GPU util percent (0–100), or 0 if not yielding / unavailable.
    pub fn utilization(&self) -> f32 {
        self.knobs.last_util.load(Ordering::Relaxed) as f32
    }

    /// True when yielding and the last util sample exceeds the ceiling.
    pub fn should_throttle(&self) -> bool {
        self.knobs.yielding.load(Ordering::Relaxed)
            && self.knobs.last_util.load(Ordering::Relaxed)
                > self.knobs.ceiling.load(Ordering::Relaxed)
    }

    /// Request the poller to exit and join it.
    pub fn stop(&mut self) {
        self.knobs.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            // A panicking poll thread is not fatal: the governor's whole
            // contract is to degrade to util 0 (see the module docs), which is
            // what a dead thread produces anyway — `last_util` simply stops
            // advancing and `should_throttle` goes quiet. But `poll_loop` has
            // no fallible step (atomic loads, a sleep, and a sensor read that
            // swallows every IOKit error), so a payload here means a real bug
            // in code that is supposed to be panic-free. Report it instead of
            // letting the miner silently mistake it for "no sensor available".
            if let Err(payload) = h.join() {
                // `Box<dyn Any>`'s own `Debug` only ever prints "Any", so dig
                // the message out of the two types `panic!` actually boxes.
                let msg = payload
                    .downcast_ref::<&'static str>()
                    .copied()
                    .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("<non-string panic payload>");
                tracing::warn!(
                    panic = msg,
                    "GPU util poll thread panicked; util stuck at 0"
                );
            }
        }
    }
}

impl Drop for UtilGovernor {
    fn drop(&mut self) {
        self.stop();
    }
}

// `device_index` is unused: IOKit's `IOAccelerator` matching walks every GPU
// service on the host (there is normally exactly one on Apple Silicon) and
// has no per-index selector analogous to NVML's `device_by_index`.
fn poll_loop(_device_index: u32, knobs: &Knobs) {
    while !knobs.stop.load(Ordering::Relaxed) {
        // Sample unconditionally: `utilization()` feeds `Status.utilization`,
        // the miner's health report, which must be truthful whether or not
        // yielding is on. Gating the sample on `yielding` (as this once did)
        // made a fully-busy miner report 0% in the default configuration.
        // `yielding` still gates `should_throttle` — reporting load and acting
        // on it are separate concerns.
        knobs
            .last_util
            .store(query_iokit_gpu_utilization(), Ordering::Relaxed);
        // Yielding needs a fresh sample to act on: the throttle decides whether
        // to hold back the next dispatch, so a 2 s-stale reading would let a
        // whole batch through after pressure appeared, and hold back several
        // after it cleared. Idle-time reporting has no such deadline.
        let interval = if knobs.yielding.load(Ordering::Relaxed) {
            YIELDING_POLL
        } else {
            REPORTING_POLL
        };
        thread::sleep(interval);
    }
}

/// Sensor poll interval while yielding — fast enough that the throttle acts on
/// a current reading, slow enough that the IOKit walk stays negligible.
const YIELDING_POLL: Duration = Duration::from_millis(250);
/// Poll interval when only `Status.utilization` depends on the sample.
const REPORTING_POLL: Duration = Duration::from_secs(2);

/// Wrap a UTF-8 Rust string as a `CFStringRef`, or `None` on any failure.
///
/// Follows CoreFoundation's **Create Rule**: the returned reference is owned by
/// the caller (+1 retain), who must `CFRelease` it. Every call site in this
/// module does so immediately after the one `CFDictionaryGetValue` lookup it
/// was created for.
fn cfstr(s: &str) -> Option<CFStringRef> {
    let c = CString::new(s).ok()?;
    // SAFETY: `c` is a live NUL-terminated buffer that outlives this call, and
    // its contents are valid UTF-8 because `CString::new` was handed a `&str`,
    // matching the `kCFStringEncodingUTF8` we declare. A null allocator selects
    // the default allocator, which is the documented way to spell "no custom
    // allocator". CoreFoundation only reads the buffer — it copies the bytes
    // into the new CFString rather than borrowing them — so `c` is free to drop
    // at the end of this function.
    let r =
        unsafe { CFStringCreateWithCString(std::ptr::null(), c.as_ptr(), kCFStringEncodingUTF8) };
    if r.is_null() {
        None
    } else {
        Some(r)
    }
}

/// Walk every `IOAccelerator` service, handing each one's property dictionary
/// to `visit`.
///
/// This is the shared half of [`query_iokit_gpu_utilization`] and
/// [`gpu_core_count`]; they differ only in which property they pull out of the
/// dictionary and how they fold the results. Both get their "degrade quietly on
/// any failure" contract from this function returning early — a caller that
/// starts from a neutral accumulator and never sees `visit` run reports exactly
/// the same thing as one that found no matching service.
///
/// `visit` receives a **borrowed** `CFDictionaryRef` under the Get Rule: it is
/// live only for the duration of the call, and is released as soon as `visit`
/// returns. A callback must read what it needs and must not retain or stash the
/// pointer without taking its own reference.
fn for_each_accelerator_properties(mut visit: impl FnMut(CFDictionaryRef)) {
    let Ok(service_name) = CString::new("IOAccelerator") else {
        return;
    };
    // SAFETY: `service_name` is a live NUL-terminated buffer that outlives the
    // call, and `IOServiceMatching` only reads it. The returned dictionary
    // comes to us under the Create Rule (+1 owned), but we deliberately never
    // release it — see the ownership note on `IOServiceGetMatchingServices`.
    let matching = unsafe { IOServiceMatching(service_name.as_ptr()) };
    if matching.is_null() {
        return;
    }

    let mut iterator: io_iterator_t = 0;
    // SAFETY: `matching` is a non-null dictionary we own (null-checked above)
    // and `iterator` is a live, initialized out-param.
    //
    // This call *consumes* the `matching` reference: IOKitLib serializes the
    // dictionary and then unconditionally `CFRelease`s it, on the success and
    // the failure paths alike. Releasing `matching` ourselves — here or on the
    // early return below — would therefore be an over-release. The one path
    // that does not consume it is a null dictionary, which we already excluded.
    //
    // The iterator is the reference we *do* own; it is released at the end of
    // the walk. On a non-zero return the out-param is left at its `0`
    // initializer, so the early return below has nothing to clean up.
    let ret = unsafe {
        IOServiceGetMatchingServices(
            kIOMasterPortDefault,
            matching as CFDictionaryRef,
            &mut iterator,
        )
    };
    if ret != 0 {
        return;
    }

    loop {
        // SAFETY: `iterator` came from a successful `IOServiceGetMatchingServices`
        // and is still live. Each non-zero `io_object_t` is returned +1 owned
        // and is released below; 0 marks the end of the sequence.
        let service = unsafe { IOIteratorNext(iterator) };
        if service == 0 {
            break;
        }

        let mut props: CFMutableDictionaryRef = std::ptr::null_mut();
        // SAFETY: `service` is a live object from the iterator and `props` is a
        // live out-param pre-initialized to null; a null allocator selects the
        // default one.
        //
        // On success `props` is a +1 owned dictionary (Create Rule), released
        // after `visit`. A non-zero return never leaves an owned dictionary
        // behind: IOKitLib either returns before writing the out-param at all,
        // or writes it and then returns an error *precisely* when what it wrote
        // was null. So the `pret != 0 || props.is_null()` bail below cannot
        // strand an allocation.
        let pret =
            unsafe { IORegistryEntryCreateCFProperties(service, &mut props, std::ptr::null(), 0) };
        // SAFETY: balances the +1 from `IOIteratorNext`. Releasing the service
        // this early is sound because the property dictionary is a freshly
        // unserialized, independently-owned object rather than a view into the
        // registry entry, so it stays valid after its service goes away.
        unsafe { IOObjectRelease(service) };
        if pret != 0 || props.is_null() {
            continue;
        }

        visit(props as CFDictionaryRef);

        // SAFETY: balances the +1 from `IORegistryEntryCreateCFProperties`.
        // `visit` only borrows the dictionary (documented above), so this drops
        // the last reference.
        unsafe { CFRelease(props as CFTypeRef) };
    }

    // SAFETY: balances the iterator reference retained by
    // `IOServiceGetMatchingServices`. Only reachable when that call succeeded.
    unsafe { IOObjectRelease(iterator) };
}

/// GPU utilization percent (0-100) via IOKit, or 0 on any error.
///
/// Walks the `IOAccelerator` service(s), reading
/// `PerformanceStatistics -> "Device Utilization %"` from the IORegistry.
/// Never panics; a query failure (missing service, unsupported key, ...)
/// degrades to 0, matching the Python reference's `except Exception: return 0`.
fn query_iokit_gpu_utilization() -> u32 {
    let mut best: i64 = 0;
    for_each_accelerator_properties(|props| {
        // SAFETY: `props` is a live, borrowed property dictionary for the whole
        // of this callback, which is exactly `read_device_utilization`'s
        // precondition. It reads without taking ownership.
        if let Some(util) = unsafe { read_device_utilization(props) } {
            best = best.max(util);
        }
    });
    best.clamp(0, 100) as u32
}

/// Best-effort Apple GPU core count via IOKit, or `None` on any failure.
///
/// Reads the `"gpu-core-count"` integer property published on the
/// `IOAccelerator` service (Apple Silicon). Used only to size the streaming
/// command-buffer budget ([`crate::streaming::stream_width`]); callers fall
/// back to a conservative default when this returns `None`, so a miss costs
/// concurrency tuning, never correctness. Never panics — same "return nothing
/// on any error" contract as [`query_iokit_gpu_utilization`].
pub fn gpu_core_count() -> Option<usize> {
    let mut cores: Option<usize> = None;
    for_each_accelerator_properties(|props| {
        // SAFETY: `props` is a live, borrowed property dictionary for the whole
        // of this callback, which is exactly `read_int_property`'s
        // precondition. It reads without taking ownership.
        if let Some(v) = unsafe { read_int_property(props, "gpu-core-count") } {
            if v > 0 {
                cores = Some(cores.map_or(v as usize, |c| c.max(v as usize)));
            }
        }
    });
    cores
}

/// Read a top-level signed-integer property from a service property dict,
/// or `None` if the key is absent / not a `CFNumber`.
///
/// # Safety
///
/// `props` must be a non-null `CFDictionaryRef` that stays live for the whole
/// call — the caller has to hold a reference to it, not merely have seen one.
/// Only borrowed under the Get Rule: this function never releases `props`, and
/// the value it looks up is likewise borrowed, so it must not outlive `props`
/// (it does not — the value is decoded into an owned `i64` before returning).
unsafe fn read_int_property(props: CFDictionaryRef, key: &str) -> Option<i64> {
    let k = cfstr(key)?;
    let val = CFDictionaryGetValue(props, k as *const c_void);
    CFRelease(k as CFTypeRef);
    if val.is_null() || CFGetTypeID(val) != CFNumberGetTypeID() {
        return None;
    }
    let mut out: i64 = 0;
    let ok = CFNumberGetValue(
        val as CFNumberRef,
        kCFNumberSInt64Type,
        &mut out as *mut i64 as *mut c_void,
    );
    ok.then_some(out)
}

/// Read `PerformanceStatistics -> "Device Utilization %"` from one service's
/// property dictionary, or `None` if either key is absent / not a number.
///
/// # Safety
///
/// `props` must be a non-null `CFDictionaryRef` that stays live for the whole
/// call — the caller has to hold a reference to it, not merely have seen one.
/// Only borrowed under the Get Rule: neither `props` nor the nested
/// `PerformanceStatistics` dictionary is released here, and the nested one is
/// only valid because its parent is held alive by the caller for the duration.
/// The dynamic type of every borrowed value is checked before use, so a driver
/// publishing an unexpected CF type yields `None` rather than undefined
/// behavior.
unsafe fn read_device_utilization(props: CFDictionaryRef) -> Option<i64> {
    let perf_key = cfstr("PerformanceStatistics")?;
    let perf_dict = CFDictionaryGetValue(props, perf_key as *const c_void);
    CFRelease(perf_key as CFTypeRef);
    // Guard the borrowed value's dynamic type before treating it as a
    // dictionary: a driver populating an unexpected CF type would otherwise
    // make the CFDictionaryGetValue call below undefined behavior.
    if perf_dict.is_null() || CFGetTypeID(perf_dict) != CFDictionaryGetTypeID() {
        return None;
    }

    let util_key = cfstr("Device Utilization %")?;
    let util_val = CFDictionaryGetValue(perf_dict as CFDictionaryRef, util_key as *const c_void);
    CFRelease(util_key as CFTypeRef);
    if util_val.is_null() || CFGetTypeID(util_val) != CFNumberGetTypeID() {
        return None;
    }

    let mut val: i64 = 0;
    let ok = CFNumberGetValue(
        util_val as CFNumberRef,
        kCFNumberSInt64Type,
        &mut val as *mut i64 as *mut c_void,
    );
    ok.then_some(val)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Live sensor read must never panic and must report a valid percentage.
    /// Exercises the real IOKit path (this crate is macOS-only).
    #[test]
    fn query_iokit_gpu_utilization_is_in_range() {
        let util = query_iokit_gpu_utilization();
        assert!(util <= 100, "util {util} out of 0..=100 range");
    }

    #[test]
    fn governor_without_yielding_never_throttles() {
        let mut gov = UtilGovernor::start(0, 100, false);
        assert!(!gov.should_throttle());
        gov.stop();
    }

    /// Utilization reporting must not depend on `yielding`: `Status.utilization`
    /// is the miner's health report, and a busy non-yielding miner reporting 0%
    /// is a lie the coordinator cannot detect. Only the range is asserted — the
    /// value is a live sensor reading, and a host with no sensor reports 0.
    #[test]
    fn utilization_is_reported_with_yielding_off() {
        let mut gov = UtilGovernor::start(0, 100, false);
        // One poll interval plus slack, so the first sample has landed.
        std::thread::sleep(REPORTING_POLL + Duration::from_millis(500));
        let util = gov.utilization();
        assert!(
            (0.0..=100.0).contains(&util),
            "util {util} out of 0..=100 range"
        );
        gov.stop();
    }
}
