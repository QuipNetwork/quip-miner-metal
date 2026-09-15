// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// METAL MULTI-SPIN CODED SIMULATED ANNEALING
// ==============================================================================
// Port of quip-miner-cuda's kernels/msc.cu, itself a port of quip-miner-cpu's
// sa_msc.rs (Isakov, Zintchenko, Ronnow, Troyer 2015). 32 replicas share the
// bits of one 32-bit word per spin: bit r of state[i] is spin i of replica r,
// 0 meaning +1 and 1 meaning -1 (the same convention as the packed output of
// every kernel in this crate).
//
// Differences from msc.cu, all forced by Apple GPUs:
// - 32-bit words, not 64-bit. Threadgroup memory is capped at 32 KB per
//   threadgroup with no opt-in, so N * 8 bytes does not fit Advantage2's
//   4577 spins, and Apple ALUs are 32-bit (64-bit integer ops are emulated).
// - One threadgroup per (problem, word). A job of R reads dispatches R / 32
//   independent threadgroups per problem; there is no `words` loop in-kernel.
// - No slot control plane. The host dispatches explicit batches and chunks
//   the beta ladder across command buffers (macOS GPU watchdog), so this
//   kernel resumes from persistent buffers exactly as block_gibbs_parallel
//   in kernels/gibbs.metal does.
// - Thresholds are 32-bit: cut[m] = floor(exp(-2 beta m) * 2^32).
//
// Preconditions the host checks: J in {-1, 0, +1}, |h| <= 1, CSR degree
// <= MSA_MAX_DEG. Energies are not computed here; the host rescores every
// read with energy_milli.

#define MSA_LANES      32
#define MSA_PLANES     6
#define MSA_MAX_COUNT  63
#define MSA_MAX_FIELD  63
#define MSA_ROW        8192
#define MSA_ROW_MASK   8191
#define MSA_MAX_DEG    20

typedef unsigned int uint;
typedef unsigned char uchar;

// ==============================================================================
// RNG - xoshiro128** (same generator and seeding as kernels/gibbs.metal)
// ==============================================================================

struct RngState {
    uint s0, s1, s2, s3;
};

inline uint rotl(uint x, int k) {
    return (x << k) | (x >> (32 - k));
}

inline uint xoshiro128starstar(thread RngState &state) {
    uint result = rotl(state.s1 * 5, 7) * 9;
    uint t = state.s1 << 9;
    state.s2 ^= state.s0;
    state.s3 ^= state.s1;
    state.s1 ^= state.s2;
    state.s0 ^= state.s3;
    state.s2 ^= t;
    state.s3 = rotl(state.s3, 11);
    return result;
}

inline uint splitmix32(thread uint &z) {
    z += 0x9e3779b9u;
    uint r = z;
    r = (r ^ (r >> 16)) * 0x85ebca6bu;
    r = (r ^ (r >> 13)) * 0xc2b2ae35u;
    return r ^ (r >> 16);
}

inline RngState seed_rng(uint seed) {
    uint z = seed;
    RngState state;
    state.s0 = splitmix32(z);
    state.s1 = splitmix32(z);
    state.s2 = splitmix32(z);
    state.s3 = splitmix32(z);
    return state;
}

// Cyclic shift into the threshold row for one sweep. A pure hash of the
// coordinates, so a chunk that resumes mid-ladder computes the same shift
// the unchunked run would.
inline uint sweep_offset(uint base_seed, uint problem_id, uint word, int beta_idx, int sweep) {
    uint z = base_seed
           ^ (problem_id * 0x9E3779B9u)
           ^ (word * 0x85EBCA6Bu)
           ^ (uint(beta_idx) * 0xC2B2AE35u)
           ^ uint(sweep);
    return splitmix32(z) & MSA_ROW_MASK;
}

// ==============================================================================
// Bit-sliced arithmetic (32 lanes)
// ==============================================================================

// Carry-save adder: (h, l) = a + b + c per lane.
inline void csa(thread uint &h, thread uint &l, uint a, uint b, uint c) {
    uint u = a ^ b;
    h = (a & b) | (u & c);
    l = u ^ c;
}

// Per-lane popcount of 21 one-bit inputs (missing inputs are zero words)
// into planes[0..5] = ones, twos, fours, eights, sixteens, 0. Harley-Seal
// tree: about 100 ops instead of 21 x 18 for a ripple add.
inline void popcount21(thread const uint* x, thread uint* planes) {
    uint ones = 0, twos = 0, fours = 0, eights = 0;
    uint tA, tB, fA, fB, eA, eB, sA, sB;
    csa(tA, ones, ones, x[0], x[1]);   csa(tB, ones, ones, x[2], x[3]);   csa(fA, twos, twos, tA, tB);
    csa(tA, ones, ones, x[4], x[5]);   csa(tB, ones, ones, x[6], x[7]);   csa(fB, twos, twos, tA, tB);
    csa(eA, fours, fours, fA, fB);
    csa(tA, ones, ones, x[8], x[9]);   csa(tB, ones, ones, x[10], x[11]); csa(fA, twos, twos, tA, tB);
    csa(tA, ones, ones, x[12], x[13]); csa(tB, ones, ones, x[14], x[15]); csa(fB, twos, twos, tA, tB);
    csa(eB, fours, fours, fA, fB);
    csa(sA, eights, eights, eA, eB);
    csa(tA, ones, ones, x[16], x[17]); csa(tB, ones, ones, x[18], x[19]); csa(fA, twos, twos, tA, tB);
    tA = ones & x[20]; ones ^= x[20];
    fB = twos & tA;    twos ^= tA;
    csa(eA, fours, fours, fA, fB);
    sB = eights & eA;  eights ^= eA;
    planes[0] = ones;
    planes[1] = twos;
    planes[2] = fours;
    planes[3] = eights;
    planes[4] = sA | sB;
    planes[5] = 0u;
}

// Lanes whose 6-bit counter is <= limit (bit-serial compare, LSB first).
// Branchless: the constant changes on every word update.
inline uint le_constant(thread const uint* planes, int limit) {
    int bound = limit + 1;
    if (bound > MSA_MAX_COUNT) return 0xFFFFFFFFu;
    uint ge = 0xFFFFFFFFu;
    for (int k = 0; k < MSA_PLANES; ++k) {
        uint set = 0u - uint((bound >> k) & 1);
        uint p = planes[k];
        uint both = p & ge;
        uint either = p ^ ge;
        ge = both | (either & ~set);
    }
    return ~ge;
}

// ==============================================================================
// Kernel
// ==============================================================================
// One threadgroup per (problem, word): threadgroup_position_in_grid.x =
// problem_id * words + word. Threads split each colour class; a barrier
// closes every class so no thread reads a neighbour mid-update. Buffer
// indices 0..15 match every other kernel in this crate, 16..18 and 20 match
// the Gibbs colour-block bindings, 19 carries `words`, and 21..24 match the
// Gibbs chunk bindings.

kernel void msa_anneal(
    device const int* csr_row_ptr [[buffer(0)]],
    device const int* csr_col_ind [[buffer(1)]],
    device const int8_t* csr_J_vals [[buffer(2)]],
    device const int* row_ptr_offsets [[buffer(3)]],
    device const int* col_ind_offsets [[buffer(4)]],

    constant int& N [[buffer(5)]],
    constant int& num_betas [[buffer(6)]],
    constant int& sweeps_per_beta [[buffer(7)]],
    constant uint& base_seed [[buffer(8)]],

    device const float* beta_schedule [[buffer(9)]],

    device int8_t* final_samples [[buffer(10)]],           // [num_problems * num_reads * packed_size]
    device int* final_energies [[buffer(11)]],             // unused: the host rescores

    constant int& num_threadgroups [[buffer(12)]],         // num_problems * words
    constant int& num_problems [[buffer(13)]],
    constant int& num_reads [[buffer(14)]],

    device const int8_t* csr_h_vals [[buffer(15)]],

    device const int* color_block_starts [[buffer(16)]],
    device const int* color_block_counts [[buffer(17)]],
    device const int* color_node_indices [[buffer(18)]],

    constant int& words [[buffer(19)]],                    // words per problem = num_reads / 32
    constant int& num_colors [[buffer(20)]],

    constant int& beta_start [[buffer(21)]],               // first beta index this chunk (0 = init)
    constant int& beta_count [[buffer(22)]],               // betas to process this chunk
    device uint* persistent_state [[buffer(23)]],          // [num_threadgroups * N] spin words
    device uint* persistent_rng [[buffer(24)]],            // [num_threadgroups * group_size * 4]

    threadgroup uint* state [[threadgroup(0)]],            // [N] spin words, sized by the host

    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_group [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup uchar row[MSA_ROW];              // geometric draws M for the current rung
    threadgroup uint cut[MSA_MAX_FIELD + 1];     // cut[m] = floor(exp(-2 beta m) * 2^32)

    uint tg = threadgroup_pos.x;
    if (tg >= uint(num_threadgroups)) {
        return;
    }
    uint tid = thread_pos_in_group.x;
    uint gsz = threads_per_group.x;
    uint problem_id = tg / uint(words);
    uint w = tg - problem_id * uint(words);

    int row_ptr_start = row_ptr_offsets[problem_id];
    int col_ind_start = col_ind_offsets[problem_id];
    device const int* my_csr_row_ptr = &csr_row_ptr[row_ptr_start];
    device const int* my_csr_col_ind = &csr_col_ind[col_ind_start];
    device const int8_t* my_csr_J_vals = &csr_J_vals[col_ind_start];
    device const int8_t* my_h_vals = &csr_h_vals[problem_id * uint(N)];

    int n = N;
    int packed_size = (n + 7) / 8;

    RngState rng;
    if (beta_start == 0) {
        // First chunk: seed per (threadgroup, thread) and draw random words.
        rng = seed_rng((base_seed ? base_seed : 1u) ^ (tg * 2654435761u) ^ (tid * 2246822519u));
        for (uint var = tid; var < uint(n); var += gsz) {
            state[var] = xoshiro128starstar(rng);
        }
    } else {
        // Continuation chunk: threadgroup memory does not survive dispatches,
        // so rebuild the words and the RNG stream from device memory.
        device const uint* src_rng = &persistent_rng[(tg * gsz + tid) * 4];
        rng.s0 = src_rng[0];
        rng.s1 = src_rng[1];
        rng.s2 = src_rng[2];
        rng.s3 = src_rng[3];
        device const uint* src_state = &persistent_state[tg * uint(n)];
        for (uint var = tid; var < uint(n); var += gsz) {
            state[var] = src_state[var];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int chunk_end = min(beta_start + beta_count, num_betas);
    for (int beta_idx = beta_start; beta_idx < chunk_end; beta_idx++) {
        float beta = beta_schedule[beta_idx];

        // Threshold table for this rung. cut[0] is never read (m starts at 1).
        if (tid <= uint(MSA_MAX_FIELD)) {
            float p = exp(-2.0f * beta * float(tid));
            cut[tid] = (p >= 1.0f) ? 0xFFFFFFFFu : uint(p * 4294967296.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // One geometric draw per row slot: M = max m with u < cut[m].
        for (uint i = tid; i < uint(MSA_ROW); i += gsz) {
            uint u = xoshiro128starstar(rng);
            int m = 0;
            if (u < cut[1]) {
                m = 1;
                while (m < MSA_MAX_FIELD && u < cut[m + 1]) {
                    ++m;
                }
            }
            row[i] = uchar(m);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int sweep = 0; sweep < sweeps_per_beta; sweep++) {
            int off = int(sweep_offset(base_seed, problem_id, w, beta_idx, sweep));

            for (int color = 0; color < num_colors; color++) {
                int block_start = color_block_starts[color];
                int block_count = color_block_counts[color];

                for (uint k = tid; k < uint(block_count); k += gsz) {
                    int var = color_node_indices[block_start + int(k)];
                    int pstart = my_csr_row_ptr[var];
                    int pend = my_csr_row_ptr[var + 1];
                    int h = my_h_vals[var];

                    // Prefetch every neighbour index and coupling before
                    // touching threadgroup memory. Slots past the degree read
                    // node 0 with a zero coupling and contribute nothing.
                    int nb[MSA_MAX_DEG];
                    int jj[MSA_MAX_DEG];
                    #pragma unroll
                    for (int q = 0; q < MSA_MAX_DEG; ++q) {
                        int p = pstart + q;
                        bool ok = p < pend;
                        nb[q] = ok ? my_csr_col_ind[p] : 0;
                        jj[q] = ok ? int(my_csr_J_vals[p]) : 0;
                    }

                    uint bi = state[var];
                    uint x[MSA_MAX_DEG + 1];
                    int d = 0;
                    #pragma unroll
                    for (int q = 0; q < MSA_MAX_DEG; ++q) {
                        int J = jj[q];
                        uint sj = state[nb[q]];
                        // Set on the replicas where bond q is satisfied.
                        uint l = ((J < 0) ? 0xFFFFFFFFu : 0u) ^ bi ^ sj;
                        x[q] = (J != 0) ? l : 0u;
                        d += (J != 0);
                    }
                    // A field is one more bond to a spin pinned at +1.
                    x[MSA_MAX_DEG] = (h != 0) ? ((h < 0) ? ~bi : bi) : 0u;
                    d += (h != 0);

                    uint planes[MSA_PLANES];
                    popcount21(x, planes);

                    // Metropolis: flip where L <= (d + M) / 2.
                    int m = row[(var + off) & MSA_ROW_MASK];
                    int limit = (d + m) >> 1;
                    uint accept = (limit >= d) ? 0xFFFFFFFFu : le_constant(planes, limit);
                    state[var] = bi ^ accept;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    // Persist for the next chunk (always written; the host decides whether
    // one follows). The barrier closing the last colour class already
    // synchronised `state`, and each thread writes a disjoint stride.
    {
        device uint* dst_rng = &persistent_rng[(tg * gsz + tid) * 4];
        dst_rng[0] = rng.s0;
        dst_rng[1] = rng.s1;
        dst_rng[2] = rng.s2;
        dst_rng[3] = rng.s3;
        device uint* dst_state = &persistent_state[tg * uint(n)];
        for (uint var = tid; var < uint(n); var += gsz) {
            dst_state[var] = state[var];
        }
    }

    // Pack lane r of this word as read w * 32 + r (bit 1 == spin -1, LSB
    // first per byte). (lane, byte) pairs are spread over the threadgroup;
    // each pair owns one output byte, so there are no write races.
    int total = MSA_LANES * packed_size;
    for (int idx = int(tid); idx < total; idx += int(gsz)) {
        int lane = idx / packed_size;
        int b = idx - lane * packed_size;
        int read = int(w) * MSA_LANES + lane;
        if (read >= num_reads) {
            continue;
        }
        uint byte = 0u;
        int base = b * 8;
        for (int bit = 0; bit < 8; ++bit) {
            int var = base + bit;
            if (var < n) {
                byte |= ((state[var] >> uint(lane)) & 1u) << uint(bit);
            }
        }
        uint out = (problem_id * uint(num_reads) + uint(read)) * uint(packed_size) + uint(b);
        final_samples[out] = as_type<int8_t>(uchar(byte));
    }
}
