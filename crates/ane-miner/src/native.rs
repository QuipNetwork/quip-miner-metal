use std::ffi::{c_char, c_void};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::AneError;

const LANES: usize = 128;
const ERROR_BYTES: usize = 1024;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct EvalTimes {
    pub(crate) staging_us: u64,
    pub(crate) dispatch_us: u64,
}

unsafe extern "C" {
    fn quip_ane_create(
        input_channels: usize,
        output_channels: usize,
        weights: *const i8,
        weight_count: usize,
        fields: *const i8,
        field_count: usize,
        program: *mut *mut c_void,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn quip_ane_evaluate(
        program: *mut c_void,
        neighbors: *const i8,
        neighbor_count: usize,
        spins: *const i8,
        spin_count: usize,
        thresholds: *const u8,
        threshold_count: usize,
        output: *mut i8,
        output_count: usize,
        times: *mut EvalTimes,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn quip_ane_destroy(program: *mut c_void, error: *mut c_char, error_capacity: usize) -> i32;
    fn quip_ane_parent_pid() -> u32;
}

pub(crate) struct AneProgram {
    input_channels: usize,
    output_channels: usize,
    handle: Option<NonNull<c_void>>,
    _one_thread: PhantomData<Rc<()>>,
}

fn native_error(bytes: &[u8; ERROR_BYTES]) -> String {
    let Some(end) = bytes.iter().position(|&byte| byte == 0) else {
        return "ANE returned an unterminated error buffer".to_owned();
    };
    if end == 0 {
        return "ANE failed without an error message".to_owned();
    }
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn checked_bytes(elements: usize, weight_blob: bool) -> Result<usize, AneError> {
    let payload = elements
        .checked_mul(2)
        .ok_or_else(|| AneError::Capacity("ANE tensor byte size overflow".to_owned()))?;
    if payload > u32::MAX as usize {
        return Err(AneError::Capacity(
            "ANE tensor exceeds its 32-bit byte limit".to_owned(),
        ));
    }
    let bytes = if weight_blob {
        payload.checked_add(128)
    } else {
        payload
            .checked_add(65535)
            .map(|bytes| (bytes & !65535).max(65536))
    };
    bytes
        .filter(|&bytes| {
            bytes <= isize::MAX as usize && (weight_blob || bytes <= u32::MAX as usize)
        })
        .ok_or_else(|| AneError::Capacity("ANE allocation byte size overflow".to_owned()))
}

impl AneProgram {
    pub(crate) fn compile(
        input_channels: usize,
        output_channels: usize,
        weights: &[i8],
        fields: &[i8],
    ) -> Result<Self, AneError> {
        if !(32..=16384).contains(&input_channels)
            || !input_channels.is_multiple_of(32)
            || !(32..=4096).contains(&output_channels)
            || !output_channels.is_multiple_of(32)
        {
            return Err(AneError::Capacity(
                "ANE channels require 32-channel padding, at most 16384 inputs and 4096 outputs"
                    .to_owned(),
            ));
        }
        let weight_count = input_channels
            .checked_mul(output_channels)
            .ok_or_else(|| AneError::Capacity("ANE weight count overflow".to_owned()))?;
        checked_bytes(weight_count, true)?;
        for channels in [input_channels, output_channels] {
            let count = channels
                .checked_mul(LANES)
                .ok_or_else(|| AneError::Capacity("ANE lane count overflow".to_owned()))?;
            checked_bytes(count, false)?;
        }
        if weights.len() != weight_count || fields.len() != output_channels {
            return Err(AneError::Runtime(
                "ANE weight or field length mismatch".to_owned(),
            ));
        }
        if weights
            .iter()
            .chain(fields)
            .any(|&value| !(-1..=1).contains(&value))
        {
            return Err(AneError::Runtime(
                "ANE weights and fields must be -1, 0, or 1".to_owned(),
            ));
        }
        let mut handle = std::ptr::null_mut();
        let mut error = [u8::MAX; ERROR_BYTES];
        // SAFETY: Slice lengths and byte limits were checked above. All arrays
        // and out-pointers remain live for this blocking call. Native code copies
        // constants and transfers exactly one retained handle on success.
        let status = unsafe {
            quip_ane_create(
                input_channels,
                output_channels,
                weights.as_ptr(),
                weights.len(),
                fields.as_ptr(),
                fields.len(),
                &mut handle,
                error.as_mut_ptr().cast(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(AneError::Runtime(native_error(&error)));
        }
        let handle = NonNull::new(handle)
            .ok_or_else(|| AneError::Runtime("ANE creation returned a null handle".to_owned()))?;
        Ok(Self {
            input_channels,
            output_channels,
            handle: Some(handle),
            _one_thread: PhantomData,
        })
    }

    pub(crate) fn evaluate(
        &mut self,
        neighbors: &[i8],
        spins: &[i8],
        thresholds: &[u8],
    ) -> Result<(Vec<i8>, EvalTimes), AneError> {
        // These products and their FP16 allocation bounds were checked at creation.
        let input_count = self.input_channels * LANES;
        let output_count = self.output_channels * LANES;
        if neighbors.len() != input_count
            || spins.len() != output_count
            || thresholds.len() != output_count
        {
            return Err(AneError::Runtime(
                "ANE evaluation slice length mismatch".to_owned(),
            ));
        }
        let handle = self
            .handle
            .ok_or_else(|| AneError::Runtime("ANE program is closed".to_owned()))?;
        let mut output = vec![0; output_count];
        let mut times = EvalTimes::default();
        let mut error = [u8::MAX; ERROR_BYTES];
        // SAFETY: The owned handle is live and confined to one thread. Every
        // buffer has the checked shape, remains live throughout the blocking
        // call, and output/times/error are writable without aliasing inputs.
        let status = unsafe {
            quip_ane_evaluate(
                handle.as_ptr(),
                neighbors.as_ptr(),
                neighbors.len(),
                spins.as_ptr(),
                spins.len(),
                thresholds.as_ptr(),
                thresholds.len(),
                output.as_mut_ptr(),
                output.len(),
                &mut times,
                error.as_mut_ptr().cast(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(AneError::Runtime(native_error(&error)));
        }
        Ok((output, times))
    }

    pub(crate) fn close(mut self) -> Result<(), AneError> {
        self.release()
    }

    fn release(&mut self) -> Result<(), AneError> {
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        let mut error = [u8::MAX; ERROR_BYTES];
        // SAFETY: Taking the sole retained handle ensures destroy consumes it
        // exactly once, including failure. The error buffer is valid for its
        // full capacity, and the native call does not retain its pointer.
        let status =
            unsafe { quip_ane_destroy(handle.as_ptr(), error.as_mut_ptr().cast(), error.len()) };
        if status == 0 {
            Ok(())
        } else {
            Err(AneError::Runtime(native_error(&error)))
        }
    }
}

impl Drop for AneProgram {
    fn drop(&mut self) {
        if let Err(error) = self.release() {
            tracing::error!(?error, "ANE cleanup failed");
        }
    }
}

pub(crate) fn parent_pid() -> u32 {
    // SAFETY: This argument-free native function returns getppid and does not
    // initialize the ANE runtime or access caller-owned memory.
    unsafe { quip_ane_parent_pid() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_malformed_dimensions_and_weights_before_runtime() {
        for (inputs, outputs) in [
            (0, 32),
            (31, 32),
            (32, 33),
            (16416, 32),
            (32, 4128),
            (usize::MAX, 32),
        ] {
            assert!(AneProgram::compile(inputs, outputs, &[], &[]).is_err());
        }
        assert!(AneProgram::compile(32, 32, &[0; 1023], &[0; 32]).is_err());
        assert!(AneProgram::compile(32, 32, &[0; 1024], &[0; 31]).is_err());
        assert!(AneProgram::compile(32, 32, &[2; 1024], &[0; 32]).is_err());
        assert!(AneProgram::compile(32, 32, &[0; 1024], &[2; 32]).is_err());
    }

    #[test]
    fn error_buffer_requires_a_terminator() {
        assert!(native_error(&[b'x'; 1024]).contains("unterminated"));
        let mut bytes = [0; 1024];
        bytes[..4].copy_from_slice(b"oops");
        assert_eq!(native_error(&bytes), "oops");
    }

    #[test]
    fn parent_process_does_not_need_a_program() {
        assert!(parent_pid() > 0);
    }

    #[test]
    fn byte_sizes_reject_overflow_and_round_to_surface_pages() {
        assert!(checked_bytes(usize::MAX, true).is_err());
        assert!(checked_bytes(u32::MAX as usize, false).is_err());
        assert_eq!(checked_bytes(32 * 128, false).unwrap(), 65536);
        assert_eq!(checked_bytes(32769, false).unwrap(), 131072);
        assert_eq!(checked_bytes(1024, true).unwrap(), 2176);
    }

    #[test]
    fn native_boundary_rejects_missing_pointers_and_short_arrays() {
        let weights = [0i8; 1024];
        let fields = [0i8; 32];
        for (inputs, weight_pointer, weight_count, field_pointer, field_count) in [
            (
                usize::MAX,
                weights.as_ptr(),
                weights.len(),
                fields.as_ptr(),
                fields.len(),
            ),
            (
                32,
                std::ptr::null(),
                weights.len(),
                fields.as_ptr(),
                fields.len(),
            ),
            (32, weights.as_ptr(), 1023, fields.as_ptr(), fields.len()),
            (
                32,
                weights.as_ptr(),
                weights.len(),
                std::ptr::null(),
                fields.len(),
            ),
            (32, weights.as_ptr(), weights.len(), fields.as_ptr(), 31),
        ] {
            let mut error = [u8::MAX; ERROR_BYTES];
            let mut handle = std::ptr::null_mut();
            // SAFETY: Non-null arrays are live and at least as long as their
            // supplied counts. Invalid/null arguments must fail before reads;
            // the handle and error output storage are valid for this call.
            let status = unsafe {
                quip_ane_create(
                    inputs,
                    32,
                    weight_pointer,
                    weight_count,
                    field_pointer,
                    field_count,
                    &mut handle,
                    error.as_mut_ptr().cast(),
                    error.len(),
                )
            };
            assert_eq!(status, 1);
            assert!(handle.is_null());
            assert!(error.contains(&0));
        }
        let mut error = [u8::MAX; 1];
        // SAFETY: A missing handle is explicitly rejected without dereference;
        // the one-byte error buffer is writable and tests bounded termination.
        let status = unsafe {
            quip_ane_destroy(std::ptr::null_mut(), error.as_mut_ptr().cast(), error.len())
        };
        assert_eq!(status, 1);
        assert_eq!(error[0], 0);
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_weight_only_identity_and_field_only_effect() {
        let mut positive = [0; 1024];
        positive[0] = 1;
        let mut negative = positive;
        negative[0] = -1;
        let mut fields = [0; 32];
        fields[0] = 1;
        // Same fields and geometry: omitting the unique MIL identity makes
        // the second model reuse the first model's coupling blob.
        let mut first = AneProgram::compile(32, 32, &positive, &fields).unwrap();
        let mut second = AneProgram::compile(32, 32, &negative, &fields).unwrap();
        let neighbors = vec![2; 4096];
        let spins = vec![1; 4096];
        let thresholds = vec![0; 4096];
        let (a, _) = first.evaluate(&neighbors, &spins, &thresholds).unwrap();
        let (b, _) = second.evaluate(&neighbors, &spins, &thresholds).unwrap();
        assert!(a.iter().all(|&spin| spin == -1));
        assert!(b[..128].iter().all(|&spin| spin == 1));
        first.close().unwrap();
        second.close().unwrap();

        fields[0] = -1;
        let mut field_only = AneProgram::compile(32, 32, &[0; 1024], &fields).unwrap();
        let (output, _) = field_only
            .evaluate(&neighbors, &spins, &thresholds)
            .unwrap();
        assert!(output[..128].iter().all(|&spin| spin == 1));
        assert!(output[128..].iter().all(|&spin| spin == -1));
        field_only.close().unwrap();
        eprintln!("weight_only_identity_and_field_only_successful_dispatches=3 explicit_closes=3");
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_exhaustive_integer_acceptance() {
        let mut weights = vec![0; 32 * 32];
        for channel in 0..32 {
            weights[channel * 32 + channel] = 1;
        }
        let mut cases = Vec::new();
        for degree in 0i8..=21 {
            for satisfied in 0..=degree {
                for threshold in 0u8..=63 {
                    for spin in [-1i8, 1] {
                        let neighbor = (degree - 2 * satisfied) * spin;
                        let expected = if i16::from(satisfied)
                            <= (i16::from(degree) + i16::from(threshold)) / 2
                        {
                            -spin
                        } else {
                            spin
                        };
                        cases.push((neighbor, spin, threshold, expected));
                    }
                }
            }
        }
        assert_eq!(cases.len(), 32384);
        let mut program = AneProgram::compile(32, 32, &weights, &[0; 32]).unwrap();
        let mut completed = 0;
        for chunk in cases.chunks(32 * 128) {
            let mut batch = chunk.to_vec();
            batch.resize(32 * 128, *chunk.last().unwrap());
            let neighbors: Vec<_> = batch.iter().map(|case| case.0).collect();
            let spins: Vec<_> = batch.iter().map(|case| case.1).collect();
            let thresholds: Vec<_> = batch.iter().map(|case| case.2).collect();
            let (output, _) = program.evaluate(&neighbors, &spins, &thresholds).unwrap();
            completed += 1;
            for (index, (&actual, case)) in output.iter().zip(&batch).enumerate() {
                assert_eq!(
                    actual, case.3,
                    "batch {completed}, element {index}: {case:?}"
                );
            }
        }
        assert_eq!(completed, 8);
        program.close().unwrap();
        eprintln!("integer_cases=32384 mismatches=0 successful_dispatches={completed}");
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_distinct_weights_fields_and_owned_close() {
        let mut weights = vec![0; 32 * 32];
        weights[0] = 1;
        let mut fields = [0; 32];
        fields[0] = 1;
        let mut first = AneProgram::compile(32, 32, &weights, &fields).unwrap();
        weights[0] = -1;
        fields[0] = -1;
        let mut second = AneProgram::compile(32, 32, &weights, &fields).unwrap();
        let neighbors = vec![1; 32 * 128];
        let spins = vec![1; 32 * 128];
        let thresholds = vec![0; 32 * 128];
        let (positive, _) = first.evaluate(&neighbors, &spins, &thresholds).unwrap();
        let (negative, _) = second.evaluate(&neighbors, &spins, &thresholds).unwrap();
        assert!(positive.iter().all(|&spin| spin == -1));
        assert!(negative[..128].iter().all(|&spin| spin == 1));
        assert!(negative[128..].iter().all(|&spin| spin == -1));
        first.close().unwrap();
        second.close().unwrap();
        eprintln!(
            "distinct_programs=2 singleton_padding=31 successful_dispatches=2 explicit_closes=2"
        );
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_zero_weights_and_invalid_evaluation_inputs() {
        let mut program = AneProgram::compile(64, 32, &[0; 64 * 32], &[0; 32]).unwrap();
        let mut neighbors = vec![-21; 64 * 128];
        let mut spins = vec![1; 32 * 128];
        spins[..128].fill(-1);
        let mut thresholds = vec![0; 32 * 128];
        let mut native_output = vec![0; spins.len()];
        let mut times = EvalTimes::default();
        let mut error = [u8::MAX; ERROR_BYTES];
        // SAFETY: The program owns a live handle. Each non-null buffer has
        // sufficient storage. The deliberately short count and null pointer
        // must be rejected by C before input access or dispatch.
        unsafe {
            let handle = program.handle.unwrap().as_ptr();
            assert_eq!(
                quip_ane_evaluate(
                    handle,
                    neighbors.as_ptr(),
                    neighbors.len() - 1,
                    spins.as_ptr(),
                    spins.len(),
                    thresholds.as_ptr(),
                    thresholds.len(),
                    native_output.as_mut_ptr(),
                    native_output.len(),
                    &mut times,
                    error.as_mut_ptr().cast(),
                    error.len()
                ),
                1
            );
            assert!(error.contains(&0));
            error.fill(u8::MAX);
            assert_eq!(
                quip_ane_evaluate(
                    handle,
                    neighbors.as_ptr(),
                    neighbors.len(),
                    std::ptr::null(),
                    spins.len(),
                    thresholds.as_ptr(),
                    thresholds.len(),
                    native_output.as_mut_ptr(),
                    native_output.len(),
                    &mut times,
                    error.as_mut_ptr().cast(),
                    error.len()
                ),
                1
            );
            assert!(error.contains(&0));
        }
        assert!(program
            .evaluate(&neighbors[..neighbors.len() - 1], &spins, &thresholds)
            .is_err());
        assert!(program
            .evaluate(&neighbors, &spins[1..], &thresholds)
            .is_err());
        assert!(program
            .evaluate(&neighbors, &spins, &thresholds[1..])
            .is_err());
        neighbors[0] = 22;
        assert!(program.evaluate(&neighbors, &spins, &thresholds).is_err());
        neighbors[0] = -21;
        spins[0] = 0;
        assert!(program.evaluate(&neighbors, &spins, &thresholds).is_err());
        spins[0] = -1;
        thresholds[0] = 64;
        assert!(program.evaluate(&neighbors, &spins, &thresholds).is_err());
        thresholds[0] = 0;
        let (output, _) = program.evaluate(&neighbors, &spins, &thresholds).unwrap();
        assert!(output[..128].iter().all(|&spin| spin == 1));
        assert!(output[128..].iter().all(|&spin| spin == -1));
        program.close().unwrap();
        eprintln!("zero_weights_successful_dispatches=1 invalid_inputs_rejected=8");
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_rectangular_row_major_weights() {
        let mut weights = vec![0; 64 * 32];
        weights[2 * 64 + 3] = -1;
        weights[4 * 64 + 5] = 1;
        let mut fields = [0; 32];
        fields[2] = 1;
        let mut program = AneProgram::compile(64, 32, &weights, &fields).unwrap();
        let mut neighbors = vec![0; 64 * 128];
        neighbors[3 * 128..4 * 128].fill(2);
        neighbors[5 * 128..6 * 128].fill(-2);
        let mut spins = vec![1; 32 * 128];
        spins[4 * 128..5 * 128].fill(-1);
        let (output, times) = program
            .evaluate(&neighbors, &spins, &[0; 32 * 128])
            .unwrap();
        for (channel, lanes) in output.chunks(128).enumerate() {
            let expected = if channel == 2 || channel == 4 { 1 } else { -1 };
            assert!(
                lanes.iter().all(|&spin| spin == expected),
                "channel {channel}"
            );
        }
        program.close().unwrap();
        eprintln!(
            "rectangular_successful_dispatches=1 staging_us={} dispatch_us={}",
            times.staging_us, times.dispatch_us
        );
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_input_binding_and_acceptance_ties() {
        let mut weights = vec![0i8; 32 * 32];
        for channel in 0..32 {
            weights[channel * 32 + channel] = 1;
        }
        let mut program = AneProgram::compile(32, 32, &weights, &[0; 32]).unwrap();
        let neighbors = vec![-3; 32 * 128];
        let spins = vec![1; 32 * 128];
        let mut thresholds = vec![2; 32 * 128];
        thresholds[0] = 3;
        let (output, _) = program.evaluate(&neighbors, &spins, &thresholds).unwrap();
        assert_eq!(output[0], -1);
        assert!(output[1..].iter().all(|&spin| spin == 1));
        program.close().unwrap();
    }
}
