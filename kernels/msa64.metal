// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

#include <metal_stdlib>
using namespace metal;

// Experimental 64-lane multi-spin SA. Spin words stay in device memory so
// N * 8 bytes do not consume the 32 KB threadgroup allocation. Threshold
// rows remain shared by all lanes of a word: 64 correlated replicas here,
// compared with 32 in msa.metal. This is an isolated benchmark pipeline.

#define MSA_LANES      64
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
// Bit-sliced arithmetic (64 lanes)
// ==============================================================================

// Carry-save adder: (h, l) = a + b + c per lane.
inline void csa(thread ulong &h, thread ulong &l, ulong a, ulong b, ulong c) {
    ulong u = a ^ b;
    h = (a & b) | (u & c);
    l = u ^ c;
}

// Per-lane popcount of 21 one-bit inputs (missing inputs are zero words)
// into planes[0..5] = ones, twos, fours, eights, sixteens, 0. Harley-Seal
// tree: about 100 ops instead of 21 x 18 for a ripple add.
inline void popcount21(thread const ulong* x, thread ulong* planes) {
    ulong ones = 0, twos = 0, fours = 0, eights = 0;
    ulong tA, tB, fA, fB, eA, eB, sA, sB;
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
    planes[5] = 0ul;
}

// Lanes whose 6-bit counter is <= limit (bit-serial compare, LSB first).
// Branchless: the constant changes on every word update.
inline ulong le_constant(thread const ulong* planes, int limit) {
    int bound = limit + 1;
    if (bound > MSA_MAX_COUNT) return ~ulong(0);
    ulong ge = ~ulong(0);
    for (int k = 0; k < MSA_PLANES; ++k) {
        ulong set = 0ul - ulong((bound >> k) & 1);
        ulong p = planes[k];
        ulong both = p & ge;
        ulong either = p ^ ge;
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

kernel void msa64_anneal(
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

    constant int& words [[buffer(19)]],                    // words per problem = num_reads / 64
    constant int& num_colors [[buffer(20)]],

    constant int& beta_start [[buffer(21)]],               // first beta index this chunk (0 = init)
    constant int& beta_count [[buffer(22)]],               // betas to process this chunk
    device ulong* persistent_state [[buffer(23)]],          // [num_threadgroups * N] spin words
    device uint* persistent_rng [[buffer(24)]],            // [num_threadgroups * group_size * 4]

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
    device ulong* state = &persistent_state[tg * uint(n)];
    int packed_size = (n + 7) / 8;

    RngState rng;
    if (beta_start == 0) {
        // First chunk: seed per (threadgroup, thread) and draw random words.
        rng = seed_rng((base_seed ? base_seed : 1u) ^ (tg * 2654435761u) ^ (tid * 2246822519u));
        for (uint var = tid; var < uint(n); var += gsz) {
            uint low = xoshiro128starstar(rng);
            uint high = xoshiro128starstar(rng);
            state[var] = ulong(low) | (ulong(high) << 32);
        }
    } else {
        // Spin words already reside in device memory. Restore only the RNG.
        device const uint* src_rng = &persistent_rng[(tg * gsz + tid) * 4];
        rng.s0 = src_rng[0];
        rng.s1 = src_rng[1];
        rng.s2 = src_rng[2];
        rng.s3 = src_rng[3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

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
                    // reading device spin words. Slots past the degree read
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

                    ulong bi = state[var];
                    ulong x[MSA_MAX_DEG + 1];
                    int d = 0;
                    #pragma unroll
                    for (int q = 0; q < MSA_MAX_DEG; ++q) {
                        int J = jj[q];
                        ulong sj = state[nb[q]];
                        // Set on the replicas where bond q is satisfied.
                        ulong l = ((J < 0) ? ~ulong(0) : 0ul) ^ bi ^ sj;
                        x[q] = (J != 0) ? l : 0ul;
                        d += (J != 0);
                    }
                    // A field is one more bond to a spin pinned at +1.
                    x[MSA_MAX_DEG] = (h != 0) ? ((h < 0) ? ~bi : bi) : 0ul;
                    d += (h != 0);

                    ulong planes[MSA_PLANES];
                    popcount21(x, planes);

                    // Metropolis: flip where L <= (d + M) / 2.
                    int m = row[(var + off) & MSA_ROW_MASK];
                    int limit = (d + m) >> 1;
                    ulong accept = (limit >= d) ? ~ulong(0) : le_constant(planes, limit);
                    state[var] = bi ^ accept;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
            }
        }
    }

    // Persist the RNG for the next chunk. Device spin words are already live,
    // and the last color barrier makes their writes visible to output packing.
    {
        device uint* dst_rng = &persistent_rng[(tg * gsz + tid) * 4];
        dst_rng[0] = rng.s0;
        dst_rng[1] = rng.s1;
        dst_rng[2] = rng.s2;
        dst_rng[3] = rng.s3;
    }

    // Pack lane r of this word as read w * 64 + r (bit 1 == spin -1, LSB
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
                byte |= uint((state[var] >> uint(lane)) & 1ul) << uint(bit);
            }
        }
        uint out = (problem_id * uint(num_reads) + uint(read)) * uint(packed_size) + uint(b);
        final_samples[out] = as_type<int8_t>(uchar(byte));
    }
}
