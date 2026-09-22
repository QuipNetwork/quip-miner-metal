use std::ffi::{c_char, c_void};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::graph::{PreparedGraph, MAX_LANES};
use crate::AneError;

const ERROR_BYTES: usize = 1024;
pub(crate) const BLOCK_SWEEPS: usize = 1;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct EvalTimes {
    pub(crate) staging_us: u64,
    pub(crate) dispatch_us: u64,
}

unsafe extern "C" {
    fn quip_ane_create(
        channels: usize,
        lanes: usize,
        lengths: *const usize,
        tile_count: usize,
        sweeps: usize,
        weights: *const i8,
        weight_count: usize,
        fields: *const i8,
        field_count: usize,
        program: *mut *mut c_void,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn quip_ane_reset(
        program: *mut c_void,
        spins: *const i8,
        count: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn quip_ane_evaluate(
        program: *mut c_void,
        thresholds: *const u8,
        count: usize,
        times: *mut EvalTimes,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn quip_ane_submit(
        program: *mut c_void,
        thresholds: *const u8,
        count: usize,
        times: *mut EvalTimes,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn quip_ane_finish(
        program: *mut c_void,
        times: *mut EvalTimes,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn quip_ane_read(
        program: *mut c_void,
        output: *mut i8,
        count: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn quip_ane_destroy(program: *mut c_void, error: *mut c_char, error_capacity: usize) -> i32;
    #[cfg(test)]
    fn quip_ane_convert_thresholds(thresholds: *const u8, output_bits: *mut u16, count: usize);
    fn quip_ane_parent_pid() -> u32;
}

pub(crate) struct AneProgram {
    channels: usize,
    lanes: usize,
    sweeps: usize,
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

impl AneProgram {
    pub(crate) fn compile(
        prepared: &PreparedGraph,
        lanes: usize,
        sweeps: usize,
    ) -> Result<Self, AneError> {
        let order = prepared.storage_order();
        let mut position = vec![0; prepared.node_count];
        for (row, &node) in order.iter().enumerate() {
            position[node] = row;
        }
        let lengths: Vec<_> = prepared.tiles.iter().map(|tile| tile.nodes.len()).collect();
        let count = prepared
            .tiles
            .iter()
            .map(|tile| tile.output_channels * prepared.input_channels)
            .sum();
        let mut weights = Vec::new();
        weights
            .try_reserve_exact(count)
            .map_err(|_| AneError::Runtime("ANE weights allocation failed".into()))?;
        weights.resize(count, 0);
        let mut offset = 0;
        for tile in &prepared.tiles {
            for (row, &node) in tile.nodes.iter().enumerate() {
                for &(neighbor, coupling) in &prepared.neighbors[node] {
                    weights[offset + row * prepared.input_channels + position[neighbor]] = coupling;
                }
            }
            offset += tile.output_channels * prepared.input_channels;
        }
        let mut fields = vec![0; prepared.input_channels];
        for (row, &node) in order.iter().enumerate() {
            fields[row] = prepared.fields[node];
        }
        Self::compile_raw(
            prepared.input_channels,
            lanes,
            &lengths,
            sweeps,
            &weights,
            &fields,
        )
    }

    fn compile_raw(
        channels: usize,
        lanes: usize,
        lengths: &[usize],
        sweeps: usize,
        weights: &[i8],
        fields: &[i8],
    ) -> Result<Self, AneError> {
        if !(32..=MAX_LANES).contains(&lanes)
            || !lanes.is_multiple_of(32)
            || !(32..=16384).contains(&channels)
            || !channels.is_multiple_of(32)
            || lengths.is_empty()
            || lengths.len() > 24
            || lengths.iter().any(|&n| n == 0 || n > 4096)
            || !(1..=8).contains(&sweeps)
        {
            return Err(AneError::Capacity(
                "Invalid ANE channel, tile, or sweep dimensions".into(),
            ));
        }
        let count: usize = lengths.iter().map(|n| n.div_ceil(32) * 32 * channels).sum();
        if lengths.iter().sum::<usize>() > channels
            || fields.len() != channels
            || weights.len() != count
        {
            return Err(AneError::Runtime(
                "ANE weight, field, or tile length mismatch".into(),
            ));
        }
        if weights
            .iter()
            .chain(fields)
            .any(|&value| !(-1..=1).contains(&value))
        {
            return Err(AneError::Runtime(
                "ANE weights and fields must be -1, 0, or 1".into(),
            ));
        }
        let mut handle = std::ptr::null_mut();
        let mut error = [u8::MAX; ERROR_BYTES];
        // SAFETY: Dimensions and lengths are bounded above; all slices and output
        // pointers live throughout this blocking call. Native code copies constants.
        let status = unsafe {
            quip_ane_create(
                channels,
                lanes,
                lengths.as_ptr(),
                lengths.len(),
                sweeps,
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
            .ok_or_else(|| AneError::Runtime("ANE creation returned null".into()))?;
        Ok(Self {
            channels,
            lanes,
            sweeps,
            handle: Some(handle),
            _one_thread: PhantomData,
        })
    }

    fn handle(&self) -> Result<*mut c_void, AneError> {
        self.handle
            .map(NonNull::as_ptr)
            .ok_or_else(|| AneError::Runtime("ANE program is closed".into()))
    }

    pub(crate) fn reset(&mut self, spins: &[i8]) -> Result<(), AneError> {
        if spins.len() != self.channels * self.lanes {
            return Err(AneError::Runtime("ANE spin length mismatch".into()));
        }
        let mut error = [u8::MAX; ERROR_BYTES];
        // SAFETY: The live thread-confined handle and checked slice remain valid
        // during the synchronous call. Native code validates every spin.
        let status = unsafe {
            quip_ane_reset(
                self.handle()?,
                spins.as_ptr(),
                spins.len(),
                error.as_mut_ptr().cast(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(AneError::Runtime(native_error(&error)));
        }
        Ok(())
    }

    pub(crate) fn advance(&mut self, thresholds: &[u8]) -> Result<EvalTimes, AneError> {
        if thresholds.len() != self.channels * self.lanes * self.sweeps {
            return Err(AneError::Runtime("ANE threshold length mismatch".into()));
        }
        let mut error = [u8::MAX; ERROR_BYTES];
        let mut times = EvalTimes::default();
        // SAFETY: The handle is live, slice has the compiled shape, and mutable
        // output pointers do not alias the input. Dispatch completes before return.
        let status = unsafe {
            quip_ane_evaluate(
                self.handle()?,
                thresholds.as_ptr(),
                thresholds.len(),
                &mut times,
                error.as_mut_ptr().cast(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(AneError::Runtime(native_error(&error)));
        }
        Ok(times)
    }

    pub(crate) fn submit(&mut self, thresholds: &[u8]) -> Result<EvalTimes, AneError> {
        if thresholds.len() != self.channels * self.lanes * self.sweeps {
            return Err(AneError::Runtime("ANE threshold length mismatch".into()));
        }
        let mut error = [u8::MAX; ERROR_BYTES];
        let mut times = EvalTimes::default();
        // SAFETY: The handle is live and the checked input slice remains valid
        // until this call finishes staging it. Native code does not retain the
        // slice or either output pointer while the dispatch runs asynchronously.
        let status = unsafe {
            quip_ane_submit(
                self.handle()?,
                thresholds.as_ptr(),
                thresholds.len(),
                &mut times,
                error.as_mut_ptr().cast(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(AneError::Runtime(native_error(&error)));
        }
        Ok(times)
    }

    pub(crate) fn finish(&mut self) -> Result<EvalTimes, AneError> {
        let mut error = [u8::MAX; ERROR_BYTES];
        let mut times = EvalTimes::default();
        // SAFETY: The thread-confined handle is live. Native completion joins
        // any queued dispatch before returning and does not retain the outputs.
        let status = unsafe {
            quip_ane_finish(
                self.handle()?,
                &mut times,
                error.as_mut_ptr().cast(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(AneError::Runtime(native_error(&error)));
        }
        Ok(times)
    }

    pub(crate) fn read(&mut self, output: &mut [i8]) -> Result<(), AneError> {
        if output.len() != self.channels * self.lanes {
            return Err(AneError::Runtime("ANE output length mismatch".into()));
        }
        let mut error = [u8::MAX; ERROR_BYTES];
        // SAFETY: The live thread-confined handle and exclusive output buffer
        // remain valid throughout the blocking, shape-checked read.
        let status = unsafe {
            quip_ane_read(
                self.handle()?,
                output.as_mut_ptr(),
                output.len(),
                error.as_mut_ptr().cast(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(AneError::Runtime(native_error(&error)));
        }
        Ok(())
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
    fn rejects_malformed_dimensions_and_constants_before_runtime() {
        for lanes in [0, 1, 16, 31, 33, 129, usize::MAX] {
            assert!(AneProgram::compile_raw(32, lanes, &[32], 2, &[0; 1024], &[0; 32]).is_err());
        }
        for channels in [0, 31, 33, 16416, usize::MAX] {
            assert!(AneProgram::compile_raw(channels, MAX_LANES, &[32], 2, &[], &[]).is_err());
        }
        for lengths in [&[][..], &[0][..], &[4097][..], &[1; 25][..], &[33][..]] {
            assert!(
                AneProgram::compile_raw(32, MAX_LANES, lengths, 2, &[0; 1024], &[0; 32]).is_err()
            );
        }
        for sweeps in [0, 9, usize::MAX] {
            assert!(
                AneProgram::compile_raw(32, MAX_LANES, &[32], sweeps, &[0; 1024], &[0; 32])
                    .is_err()
            );
        }
        assert!(AneProgram::compile_raw(32, MAX_LANES, &[32], 2, &[0; 1023], &[0; 32]).is_err());
        assert!(AneProgram::compile_raw(32, MAX_LANES, &[32], 2, &[0; 1024], &[0; 31]).is_err());
        assert!(AneProgram::compile_raw(32, MAX_LANES, &[32], 2, &[2; 1024], &[0; 32]).is_err());
        assert!(AneProgram::compile_raw(32, MAX_LANES, &[32], 2, &[0; 1024], &[2; 32]).is_err());
    }

    #[test]
    fn error_buffer_and_parent_process() {
        assert!(native_error(&[b'x'; ERROR_BYTES]).contains("unterminated"));
        assert!(native_error(&[0; ERROR_BYTES]).contains("without an error"));
        let mut error = [0; ERROR_BYTES];
        error[..4].copy_from_slice(b"oops");
        assert_eq!(native_error(&error), "oops");
        assert!(parent_pid() > 0);
    }

    #[test]
    fn native_threshold_conversion_preserves_values_and_skip_sentinel() {
        let thresholds = [0, 1, 2, 7, 15, 31, 63, 255, 3, 255, 32];
        let mut output = [0u16; 11];
        // SAFETY: Both arrays are valid for `thresholds.len()` elements and do
        // not overlap. The conversion does not retain either pointer.
        unsafe {
            quip_ane_convert_thresholds(thresholds.as_ptr(), output.as_mut_ptr(), thresholds.len());
        }
        assert_eq!(
            output,
            [
                0x0000, 0x3c00, 0x4000, 0x4700, 0x4b80, 0x4fc0, 0x53e0, 0xd800, 0x4200, 0xd800,
                0x5000,
            ]
        );
    }

    #[test]
    fn native_boundary_rejects_bad_create_buffers() {
        let weights = [0; 1024];
        let fields = [0; 32];
        let lengths = [32];
        for lanes in [0, 1, 16, 31, 33, 129, usize::MAX] {
            let mut error = [u8::MAX; ERROR_BYTES];
            let mut handle = std::ptr::null_mut();
            // SAFETY: All non-null buffers cover the supplied counts. Each
            // unsupported lane width must fail before runtime initialization.
            let status = unsafe {
                quip_ane_create(
                    32,
                    lanes,
                    lengths.as_ptr(),
                    lengths.len(),
                    2,
                    weights.as_ptr(),
                    weights.len(),
                    fields.as_ptr(),
                    fields.len(),
                    &mut handle,
                    error.as_mut_ptr().cast(),
                    error.len(),
                )
            };
            assert_eq!(status, 1);
            assert!(handle.is_null());
            assert!(error.contains(&0));
        }
        for (
            channels,
            length_ptr,
            tiles,
            sweeps,
            weight_ptr,
            weight_count,
            field_ptr,
            field_count,
        ) in [
            (
                usize::MAX,
                lengths.as_ptr(),
                1,
                2,
                weights.as_ptr(),
                1024,
                fields.as_ptr(),
                32,
            ),
            (
                32,
                std::ptr::null(),
                1,
                2,
                weights.as_ptr(),
                1024,
                fields.as_ptr(),
                32,
            ),
            (
                32,
                lengths.as_ptr(),
                0,
                2,
                weights.as_ptr(),
                1024,
                fields.as_ptr(),
                32,
            ),
            (
                32,
                lengths.as_ptr(),
                1,
                9,
                weights.as_ptr(),
                1024,
                fields.as_ptr(),
                32,
            ),
            (
                32,
                lengths.as_ptr(),
                1,
                2,
                std::ptr::null(),
                1024,
                fields.as_ptr(),
                32,
            ),
            (
                32,
                lengths.as_ptr(),
                1,
                2,
                weights.as_ptr(),
                1023,
                fields.as_ptr(),
                32,
            ),
            (
                32,
                lengths.as_ptr(),
                1,
                2,
                weights.as_ptr(),
                1024,
                std::ptr::null(),
                32,
            ),
            (
                32,
                lengths.as_ptr(),
                1,
                2,
                weights.as_ptr(),
                1024,
                fields.as_ptr(),
                31,
            ),
        ] {
            let mut error = [u8::MAX; ERROR_BYTES];
            let mut handle = std::ptr::null_mut();
            // SAFETY: Non-null buffers cover the supplied counts. Invalid
            // dimensions and pointers must fail before runtime initialization.
            let status = unsafe {
                quip_ane_create(
                    channels,
                    MAX_LANES,
                    length_ptr,
                    tiles,
                    sweeps,
                    weight_ptr,
                    weight_count,
                    field_ptr,
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
        let mut error = [255u8];
        // SAFETY: Null is rejected before dereference; the one-byte error buffer
        // checks bounded termination on failure.
        let status = unsafe {
            quip_ane_destroy(std::ptr::null_mut(), error.as_mut_ptr().cast(), error.len())
        };
        assert_eq!(status, 1);
        assert_eq!(error, [0]);
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_exhaustive_integer_acceptance() {
        let mut checked = 0;
        let mut dispatches = 0;
        // Each target has its own 21 fixed auxiliary channels. Even/odd local
        // sums need 20/21 nonzero terms because spin inputs are always +/-1.
        for degree_parity in [0i8, 1] {
            let terms = 20 + degree_parity as usize;
            let channels = 704;
            let mut weights = vec![0; 32 * channels];
            for target in 0..32 {
                for term in 0..terms {
                    weights[target * channels + 32 + target * 21 + term] = 1;
                }
            }
            let mut cases = Vec::new();
            for degree in 0i8..=21 {
                if degree % 2 != degree_parity {
                    continue;
                }
                for satisfied in 0..=degree {
                    for threshold in 0u8..=63 {
                        for spin in [-1i8, 1] {
                            let field = (degree - 2 * satisfied) * spin;
                            let expected = if i16::from(satisfied)
                                <= (i16::from(degree) + i16::from(threshold)) / 2
                            {
                                -spin
                            } else {
                                spin
                            };
                            cases.push((field, spin, threshold, expected));
                        }
                    }
                }
            }
            let mut program = AneProgram::compile_raw(
                channels,
                MAX_LANES,
                &[32],
                2,
                &weights,
                &vec![0; channels],
            )
            .unwrap();
            for chunk in cases.chunks(32 * MAX_LANES) {
                let mut state = vec![1; channels * MAX_LANES];
                let mut thresholds = vec![255; channels * MAX_LANES * 2];
                for (index, &(field, spin, threshold, _)) in chunk.iter().enumerate() {
                    state[index] = spin;
                    thresholds[index] = threshold;
                    let target = index / MAX_LANES;
                    let lane = index % MAX_LANES;
                    let positive = (terms as i16 + i16::from(field)) / 2;
                    for term in 0..terms {
                        state[(32 + target * 21 + term) * MAX_LANES + lane] =
                            if (term as i16) < positive { 1 } else { -1 };
                    }
                }
                program.reset(&state).unwrap();
                program.advance(&thresholds).unwrap();
                let mut output = vec![0; state.len()];
                program.read(&mut output).unwrap();
                for (index, case) in chunk.iter().enumerate() {
                    assert_eq!(output[index], case.3, "case={case:?}");
                }
                assert_eq!(output[32 * MAX_LANES..], state[32 * MAX_LANES..]);
                checked += chunk.len();
                dispatches += 1;
            }
            program.close().unwrap();
        }
        assert_eq!(checked, 32384);
        eprintln!("integer_cases={checked} mismatches=0 successful_dispatches={dispatches}");
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_retained_state_and_skipped_tail() {
        let mut program =
            AneProgram::compile_raw(32, MAX_LANES, &[32], 2, &[0; 1024], &[0; 32]).unwrap();
        let initial: Vec<_> = (0..32 * MAX_LANES)
            .map(|i| if i % 3 == 0 { -1 } else { 1 })
            .collect();
        let mut one_sweep = vec![255; 32 * MAX_LANES * 2];
        one_sweep[..32 * MAX_LANES].fill(0);
        let mut output = vec![0; initial.len()];
        assert!(program.read(&mut output).is_err());
        assert!(program.advance(&one_sweep).is_err());
        program.reset(&initial).unwrap();
        for _ in 0..3 {
            program.advance(&one_sweep).unwrap();
        }
        program.read(&mut output).unwrap();
        assert_eq!(output, initial.iter().map(|spin| -spin).collect::<Vec<_>>());
        program.advance(&vec![255; one_sweep.len()]).unwrap();
        let mut skipped = vec![0; output.len()];
        program.read(&mut skipped).unwrap();
        assert_eq!(skipped, output);
        program.reset(&initial).unwrap();
        program.read(&mut output).unwrap();
        assert_eq!(output, initial);
        program.close().unwrap();
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_distinct_weights_fields_and_owned_close() {
        let mut weights = [0; 1024];
        weights[1] = 1;
        let mut first =
            AneProgram::compile_raw(32, MAX_LANES, &[1], 2, &weights, &[0; 32]).unwrap();
        weights[1] = -1;
        let mut second =
            AneProgram::compile_raw(32, MAX_LANES, &[1], 2, &weights, &[0; 32]).unwrap();
        let mut thresholds = vec![255; 32 * MAX_LANES * 2];
        thresholds[..32 * MAX_LANES].fill(0);
        let initial = vec![1; 32 * MAX_LANES];
        let mut a = vec![0; initial.len()];
        let mut b = a.clone();
        first.reset(&initial).unwrap();
        second.reset(&initial).unwrap();
        first.advance(&thresholds).unwrap();
        second.advance(&thresholds).unwrap();
        first.read(&mut a).unwrap();
        second.read(&mut b).unwrap();
        assert!(a[..MAX_LANES].iter().all(|&spin| spin == -1));
        assert_eq!(b, initial);
        first.close().unwrap();
        second.advance(&thresholds).unwrap();
        second.read(&mut b).unwrap();
        assert_eq!(b, initial);
        second.close().unwrap();
        let mut fields = [0; 32];
        fields[0] = -1;
        let mut field_only =
            AneProgram::compile_raw(32, MAX_LANES, &[32], 2, &[0; 1024], &fields).unwrap();
        field_only.reset(&initial).unwrap();
        field_only.advance(&thresholds).unwrap();
        field_only.read(&mut a).unwrap();
        assert!(a[..MAX_LANES].iter().all(|&spin| spin == 1));
        assert!(a[MAX_LANES..].iter().all(|&spin| spin == -1));
        field_only.close().unwrap();
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_rejects_invalid_buffers_without_mutating_state() {
        let lanes = 64;
        let mut program =
            AneProgram::compile_raw(32, lanes, &[32], 2, &[0; 1024], &[0; 32]).unwrap();
        let initial = vec![1; 32 * lanes];
        let thresholds = vec![0; 32 * lanes * 2];
        let mut output = vec![0; initial.len()];
        program.reset(&initial).unwrap();
        assert!(program.reset(&initial[1..]).is_err());
        assert!(program.reset(&vec![0; initial.len()]).is_err());
        assert!(program.advance(&thresholds[1..]).is_err());
        assert!(program.advance(&vec![64; thresholds.len()]).is_err());
        assert!(program.read(&mut output[1..]).is_err());
        let mut error = [u8::MAX; ERROR_BYTES];
        let mut times = EvalTimes::default();
        let handle = program.handle().unwrap();
        // SAFETY: The live handle and all non-null arrays have sufficient
        // storage. Invalid counts/null pointers must fail before access.
        unsafe {
            assert_eq!(
                quip_ane_reset(
                    handle,
                    std::ptr::null(),
                    initial.len(),
                    error.as_mut_ptr().cast(),
                    error.len()
                ),
                1
            );
            assert_eq!(
                quip_ane_reset(
                    handle,
                    initial.as_ptr(),
                    initial.len() - 1,
                    error.as_mut_ptr().cast(),
                    error.len()
                ),
                1
            );
            assert_eq!(
                quip_ane_evaluate(
                    handle,
                    std::ptr::null(),
                    thresholds.len(),
                    &mut times,
                    error.as_mut_ptr().cast(),
                    error.len()
                ),
                1
            );
            assert_eq!(
                quip_ane_evaluate(
                    handle,
                    thresholds.as_ptr(),
                    thresholds.len() - 1,
                    &mut times,
                    error.as_mut_ptr().cast(),
                    error.len()
                ),
                1
            );
            assert_eq!(
                quip_ane_evaluate(
                    handle,
                    thresholds.as_ptr(),
                    thresholds.len(),
                    std::ptr::null_mut(),
                    error.as_mut_ptr().cast(),
                    error.len()
                ),
                1
            );
            assert_eq!(
                quip_ane_read(
                    handle,
                    std::ptr::null_mut(),
                    output.len(),
                    error.as_mut_ptr().cast(),
                    error.len()
                ),
                1
            );
            assert_eq!(
                quip_ane_read(
                    handle,
                    output.as_mut_ptr(),
                    output.len() - 1,
                    error.as_mut_ptr().cast(),
                    error.len()
                ),
                1
            );
        }
        program.read(&mut output).unwrap();
        assert_eq!(output, initial);
        program.advance(&thresholds).unwrap();
        program.read(&mut output).unwrap();
        assert_eq!(output, initial);
        program.close().unwrap();
    }

    #[test]
    #[ignore = "requires Apple Silicon ANE"]
    fn hardware_async_submission_joins_before_read_reset_and_close() {
        let lanes = 64;
        let mut program =
            AneProgram::compile_raw(32, lanes, &[32], 1, &[0; 1024], &[0; 32]).unwrap();
        let initial = vec![1; 32 * lanes];
        let replacement = vec![-1; initial.len()];
        let thresholds = vec![0; initial.len()];
        let mut output = vec![0; initial.len()];

        program.reset(&initial).unwrap();
        program.submit(&thresholds).unwrap();
        program.submit(&thresholds).unwrap();
        program.read(&mut output).unwrap();
        assert_eq!(output, initial);
        assert_eq!(program.finish().unwrap().dispatch_us, 0);

        program.submit(&thresholds).unwrap();
        program.reset(&replacement).unwrap();
        program.read(&mut output).unwrap();
        assert_eq!(output, replacement);

        program.submit(&thresholds).unwrap();
        program.close().unwrap();
    }
}
