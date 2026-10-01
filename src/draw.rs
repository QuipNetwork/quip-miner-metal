// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! The lease draw in whole device units, tuned for Apple silicon.
//!
//! [`draw_units`] returns exactly what
//! [`quip_protocol::chacha8::draw_ising`] returns for the same arguments:
//! word `w` of the `ChaCha8` stream keyed by the nonce selects
//! `allowed[w % allowed.len()]`, fields from words `0..n_nodes`, couplings
//! from the words after them. Two things make it faster on an M-series core:
//!
//! - A set with one value needs no keystream, since every word selects it.
//! - For a set of 2, 4, 8 or 16 values, NEON computes [`GROUPS`] groups of
//!   four blocks per pass, enough independent work to keep every vector pipe
//!   busy, and selects with one table lookup per 16 words. Only a word's low
//!   bits decide the value, so the words narrow to bytes first.
//!
//! Other set sizes, and other architectures, use the reference draw.

use quip_solver_core::quip_protocol::chacha8::{draw_into, DrawError};

/// Draw a lease model's fields and couplings, each value taken from
/// `allowed_h` or `allowed_j`.
///
/// # Errors
///
/// [`DrawError::EmptyAllowedValues`] when values are needed from an empty set,
/// as the reference draw does.
pub(crate) fn draw_units(
    nonce: [u8; 32],
    n_nodes: usize,
    n_edges: usize,
    allowed_h: &[i8],
    allowed_j: &[i8],
) -> Result<(Vec<i8>, Vec<i8>), DrawError> {
    if (n_nodes > 0 && allowed_h.is_empty()) || (n_edges > 0 && allowed_j.is_empty()) {
        return Err(DrawError::EmptyAllowedValues);
    }
    let mut h = vec![0i8; n_nodes];
    let mut j = vec![0i8; n_edges];
    fill(nonce, 0, allowed_h, &mut h)?;
    fill(nonce, n_nodes as u64, allowed_j, &mut j)?;
    Ok((h, j))
}

/// `out[i] = allowed[w % allowed.len()]` for keystream word
/// `w = first_word + i`.
fn fill(nonce: [u8; 32], first_word: u64, allowed: &[i8], out: &mut [i8]) -> Result<(), DrawError> {
    if out.is_empty() {
        return Ok(());
    }
    if let [only] = allowed {
        out.fill(*only);
        return Ok(());
    }
    #[cfg(target_arch = "aarch64")]
    if allowed.len().is_power_of_two() && allowed.len() <= 16 {
        // SAFETY: NEON is part of the aarch64 baseline, so every aarch64 CPU
        // runs `neon::fill`.
        unsafe { neon::fill(&key(nonce), first_word, allowed, out) };
        return Ok(());
    }
    draw_into(nonce, first_word, allowed, out)
}

/// The `ChaCha8` key: the nonce as eight little-endian words.
#[cfg(target_arch = "aarch64")]
fn key(nonce: [u8; 32]) -> [u32; 8] {
    let mut key = [0u32; 8];
    for (word, bytes) in key.iter_mut().zip(nonce.as_chunks::<4>().0) {
        *word = u32::from_le_bytes(*bytes);
    }
    key
}

/// Groups of four keystream blocks computed together.
#[cfg(target_arch = "aarch64")]
const GROUPS: usize = 2;

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::GROUPS;
    use std::arch::aarch64::{
        uint32x4_t, uint8x16_t, vaddq_u32, vandq_u8, vcombine_u16, vcombine_u8, vdupq_n_u32,
        vdupq_n_u8, veorq_u32, vld1q_u32, vld1q_u8, vmovn_u16, vmovn_u32, vqtbl1q_u8,
        vreinterpretq_s8_u8, vreinterpretq_u16_u32, vreinterpretq_u32_u16, vreinterpretq_u32_u64,
        vreinterpretq_u32_u8, vreinterpretq_u64_u32, vreinterpretq_u8_u32, vrev32q_u16,
        vshlq_n_u32, vsriq_n_u32, vst1q_s8, vtrn1q_u32, vtrn1q_u64, vtrn2q_u32, vtrn2q_u64,
    };

    const CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];
    const BLOCKS: usize = 4 * GROUPS;
    /// Byte order of a 32-bit lane rotated left by 8.
    const ROTATE_8: [u8; 16] = [3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14];

    /// The 16 state words of four blocks, one block per lane.
    type State = [uint32x4_t; 16];

    /// [`super::fill`] for a set of 2, 4, 8 or 16 values.
    #[target_feature(enable = "neon")]
    pub(super) fn fill(key: &[u32; 8], first_word: u64, allowed: &[i8], out: &mut [i8]) {
        debug_assert!(allowed.len().is_power_of_two() && allowed.len() <= 16);
        let mut table = [0u8; 16];
        for (slot, &value) in table.iter_mut().zip(allowed) {
            *slot = value as u8;
        }
        // SAFETY: `table` and `ROTATE_8` are 16 readable bytes each.
        let (table, rotate_8) = unsafe { (vld1q_u8(table.as_ptr()), vld1q_u8(ROTATE_8.as_ptr())) };
        let mask = vdupq_n_u8((allowed.len() - 1) as u8);
        let mut counter = first_word / 16;
        // Below 16, so the cast is lossless.
        let skip = (first_word % 16) as usize;
        let mut selected = [0i8; 16 * BLOCKS];
        let mut rest = out;
        if skip != 0 {
            blocks(key, counter, rotate_8, table, mask, &mut selected);
            counter = counter.wrapping_add(BLOCKS as u64);
            let words = &selected[skip..];
            let count = rest.len().min(words.len());
            let (chunk, tail) = rest.split_at_mut(count);
            chunk.copy_from_slice(&words[..count]);
            rest = tail;
        }
        let (whole, tail) = rest.as_chunks_mut::<{ 16 * BLOCKS }>();
        for chunk in whole {
            blocks(key, counter, rotate_8, table, mask, chunk);
            counter = counter.wrapping_add(BLOCKS as u64);
        }
        if !tail.is_empty() {
            blocks(key, counter, rotate_8, table, mask, &mut selected);
            tail.copy_from_slice(&selected[..tail.len()]);
        }
    }

    /// Select values for keystream blocks `counter .. counter + BLOCKS`,
    /// 16 bytes per block in stream order.
    #[inline]
    #[target_feature(enable = "neon")]
    fn blocks(
        key: &[u32; 8],
        counter: u64,
        rotate_8: uint8x16_t,
        table: uint8x16_t,
        mask: uint8x16_t,
        out: &mut [i8; 16 * BLOCKS],
    ) {
        // Words 0 to 11 of every block's initial state. Words 12 and 13 hold
        // the block counter, and words 14 and 15, the stream id, are zero.
        let initial: [u32; 12] = [
            CONSTANTS[0],
            CONSTANTS[1],
            CONSTANTS[2],
            CONSTANTS[3],
            key[0],
            key[1],
            key[2],
            key[3],
            key[4],
            key[5],
            key[6],
            key[7],
        ];
        let mut low = [0u32; 4 * GROUPS];
        let mut high = [0u32; 4 * GROUPS];
        for (offset, (low, high)) in (0u64..).zip(low.iter_mut().zip(&mut high)) {
            let block = counter.wrapping_add(offset);
            *low = block as u32;
            *high = (block >> 32) as u32;
        }
        let mut counters = [[vdupq_n_u32(0); 2]; GROUPS];
        for ((pair, low), high) in counters
            .iter_mut()
            .zip(low.as_chunks::<4>().0)
            .zip(high.as_chunks::<4>().0)
        {
            // SAFETY: `low` and `high` are four readable words each.
            *pair = unsafe { [vld1q_u32(low.as_ptr()), vld1q_u32(high.as_ptr())] };
        }
        let mut x: [State; GROUPS] = [[vdupq_n_u32(0); 16]; GROUPS];
        for (state, &[low, high]) in x.iter_mut().zip(&counters) {
            for (word, &value) in state.iter_mut().zip(&initial) {
                *word = vdupq_n_u32(value);
            }
            state[12] = low;
            state[13] = high;
        }
        for _ in 0..4 {
            quarter_round(&mut x, [0, 4, 8, 12], rotate_8);
            quarter_round(&mut x, [1, 5, 9, 13], rotate_8);
            quarter_round(&mut x, [2, 6, 10, 14], rotate_8);
            quarter_round(&mut x, [3, 7, 11, 15], rotate_8);
            quarter_round(&mut x, [0, 5, 10, 15], rotate_8);
            quarter_round(&mut x, [1, 6, 11, 12], rotate_8);
            quarter_round(&mut x, [2, 7, 8, 13], rotate_8);
            quarter_round(&mut x, [3, 4, 9, 14], rotate_8);
        }
        // Add the initial state back, rebuilding the constant words rather
        // than keeping a copy of the state live through the rounds.
        for (state, &[low, high]) in x.iter_mut().zip(&counters) {
            for (word, &value) in state.iter_mut().zip(&initial) {
                *word = vaddq_u32(*word, vdupq_n_u32(value));
            }
            state[12] = vaddq_u32(state[12], low);
            state[13] = vaddq_u32(state[13], high);
        }
        for (state, out) in x.iter().zip(out.as_chunks_mut::<64>().0) {
            // `rows[quarter][block]`: words `4 * quarter .. 4 * quarter + 4`
            // of block `block` of this group.
            let rows = [
                transpose(state[0], state[1], state[2], state[3]),
                transpose(state[4], state[5], state[6], state[7]),
                transpose(state[8], state[9], state[10], state[11]),
                transpose(state[12], state[13], state[14], state[15]),
            ];
            for (block, out) in out.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                let low = vcombine_u16(vmovn_u32(rows[0][block]), vmovn_u32(rows[1][block]));
                let high = vcombine_u16(vmovn_u32(rows[2][block]), vmovn_u32(rows[3][block]));
                let bytes = vcombine_u8(vmovn_u16(low), vmovn_u16(high));
                let values = vqtbl1q_u8(table, vandq_u8(bytes, mask));
                // SAFETY: `out` is a 16-byte chunk, the size of the store.
                unsafe { vst1q_s8(out.as_mut_ptr(), vreinterpretq_s8_u8(values)) };
            }
        }
    }

    /// Rows `a`..`d` hold one word of four blocks each. Returns, per block,
    /// those four words in order.
    #[inline]
    #[target_feature(enable = "neon")]
    fn transpose(a: uint32x4_t, b: uint32x4_t, c: uint32x4_t, d: uint32x4_t) -> [uint32x4_t; 4] {
        let ab_even = vreinterpretq_u64_u32(vtrn1q_u32(a, b));
        let ab_odd = vreinterpretq_u64_u32(vtrn2q_u32(a, b));
        let cd_even = vreinterpretq_u64_u32(vtrn1q_u32(c, d));
        let cd_odd = vreinterpretq_u64_u32(vtrn2q_u32(c, d));
        [
            vreinterpretq_u32_u64(vtrn1q_u64(ab_even, cd_even)),
            vreinterpretq_u32_u64(vtrn1q_u64(ab_odd, cd_odd)),
            vreinterpretq_u32_u64(vtrn2q_u64(ab_even, cd_even)),
            vreinterpretq_u32_u64(vtrn2q_u64(ab_odd, cd_odd)),
        ]
    }

    /// The `ChaCha` quarter-round on words `a`, `b`, `c`, `d` of every group.
    #[inline]
    #[target_feature(enable = "neon")]
    fn quarter_round(x: &mut [State; GROUPS], [a, b, c, d]: [usize; 4], rotate_8: uint8x16_t) {
        for state in x.iter_mut() {
            let mut va = state[a];
            let mut vb = state[b];
            let mut vc = state[c];
            let mut vd = state[d];
            va = vaddq_u32(va, vb);
            vd = rotate_16(veorq_u32(vd, va));
            vc = vaddq_u32(vc, vd);
            vb = rotate::<12, 20>(veorq_u32(vb, vc));
            va = vaddq_u32(va, vb);
            vd = vreinterpretq_u32_u8(vqtbl1q_u8(
                vreinterpretq_u8_u32(veorq_u32(vd, va)),
                rotate_8,
            ));
            vc = vaddq_u32(vc, vd);
            vb = rotate::<7, 25>(veorq_u32(vb, vc));
            state[a] = va;
            state[b] = vb;
            state[c] = vc;
            state[d] = vd;
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn rotate_16(value: uint32x4_t) -> uint32x4_t {
        vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(value)))
    }

    /// Rotate left by `LEFT`; `RIGHT` is `32 - LEFT`.
    #[inline]
    #[target_feature(enable = "neon")]
    fn rotate<const LEFT: i32, const RIGHT: i32>(value: uint32x4_t) -> uint32x4_t {
        vsriq_n_u32::<RIGHT>(vshlq_n_u32::<LEFT>(value), value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_solver_core::quip_protocol::chacha8::draw_ising;

    fn nonce(seed: u64) -> [u8; 32] {
        let mut nonce = [0u8; 32];
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        for byte in &mut nonce {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }
        nonce
    }

    /// Every set size, including the ones that fall back, against the
    /// reference draw, with field counts that start the couplings at every
    /// offset inside a block and across a pass of [`GROUPS`] groups.
    #[test]
    fn draw_units_matches_the_reference_draw() {
        let sets: Vec<Vec<i8>> = (1..=17)
            .map(|len| (0..len).map(|i| (i as i8) * 3 - 20).collect())
            .collect();
        for (case, (n_nodes, n_edges)) in [
            (0, 0),
            (1, 0),
            (0, 1),
            (15, 1),
            (16, 16),
            (17, 31),
            (63, 129),
            (130, 257),
            (4577, 41_514),
        ]
        .into_iter()
        .enumerate()
        {
            for allowed_h in &sets {
                for allowed_j in [&sets[1], &sets[3], &sets[2], &sets[15], &sets[16]] {
                    let nonce = nonce(case as u64 * 1000 + allowed_h.len() as u64);
                    assert_eq!(
                        draw_units(nonce, n_nodes, n_edges, allowed_h, allowed_j),
                        draw_ising(nonce, n_nodes, n_edges, allowed_h, allowed_j),
                        "nodes {n_nodes} edges {n_edges} sets {} {}",
                        allowed_h.len(),
                        allowed_j.len()
                    );
                }
            }
        }
    }

    #[test]
    fn draw_units_matches_across_the_block_counter_high_word() {
        // Couplings that start just below 2^36 words cross block 2^32,
        // where the counter's high word first changes.
        let first_word = (1u64 << 36) - 40;
        let allowed = [-1i8, 1];
        let mut ours = vec![0i8; 200];
        let mut reference = vec![0i8; 200];
        fill(nonce(9), first_word, &allowed, &mut ours).unwrap();
        draw_into(nonce(9), first_word, &allowed, &mut reference).unwrap();
        assert_eq!(ours, reference);
    }

    #[test]
    fn empty_sets_are_rejected_like_the_reference() {
        assert_eq!(
            draw_units(nonce(1), 2, 0, &[], &[1]),
            Err(DrawError::EmptyAllowedValues)
        );
        assert_eq!(
            draw_units(nonce(1), 0, 2, &[1], &[]),
            Err(DrawError::EmptyAllowedValues)
        );
        assert_eq!(draw_units(nonce(1), 0, 0, &[], &[]), Ok((vec![], vec![])));
    }

    /// Aglais-sized draws: fields from {0}, couplings from {-1, 1}, and the
    /// same with two-value fields, against the reference draw.
    #[test]
    #[ignore = "measurement; run with --release --ignored --nocapture"]
    #[expect(clippy::print_stderr, reason = "the measurement is the output")]
    fn draw_units_speed() {
        const DRAWS: u64 = 2_000;
        for (label, allowed_h) in [("h {0}", &[0i8][..]), ("h {-1, 1}", &[-1, 1][..])] {
            let time = |draw: &dyn Fn([u8; 32]) -> (Vec<i8>, Vec<i8>)| {
                let start = std::time::Instant::now();
                for seed in 0..DRAWS {
                    std::hint::black_box(draw(nonce(seed)));
                }
                start.elapsed().as_secs_f64() * 1e6 / DRAWS as f64
            };
            let ours = time(&|nonce| draw_units(nonce, 4577, 41_514, allowed_h, &[-1, 1]).unwrap());
            let reference =
                time(&|nonce| draw_ising(nonce, 4577, 41_514, allowed_h, &[-1i8, 1]).unwrap());
            eprintln!(
                "{label}: draw_units {ours:.1} us, reference {reference:.1} us, {:.2}x",
                reference / ours
            );
        }
    }
}
