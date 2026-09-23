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
// - Geometric thresholds are drawn inline from a 32-bit RNG per word update.
//
// Preconditions the host checks: J in {-1, 0, +1}, |h| <= 1, CSR degree
// <= MSA_MAX_DEG. On the last chunk the kernel writes each read's energy in
// milli units to final_energies. The value equals energy_milli whenever every
// coefficient is a whole number in int8 range; the host checks that and
// rescores otherwise.

#define MSA_LANES      32
#define MSA_PLANES     6
#define MSA_MAX_COUNT  63
#define MSA_MAX_FIELD  63
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

// One geometric draw M with P(M >= m) = exp(-2 beta m), capped at
// MSA_MAX_FIELD. Same distribution as the old per-rung threshold row
// (M = max m with u < floor(exp(-2 beta m) 2^32)), drawn per word update so
// no row is rebuilt at each rung.
inline int geometric_draw(uint u, float two_beta) {
    if (two_beta <= 0.0f) {
        return MSA_MAX_FIELD;
    }
    float x = min((float(u) + 1.0f) * 2.3283064365386963e-10f, 1.0f);
    float m = floor(-log(x) / two_beta);
    return int(clamp(m, 0.0f, float(MSA_MAX_FIELD)));
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

// Per-lane energy in milli of the spin words in `state` for the nodes this
// thread owns (var = tid, tid + gsz, ...). h*spin*1000 per node plus one
// directed CSR half-edge J*si*sj*500 per slot; a self-loop is stored once,
// so it uses 1000.
inline void lane_energies(
    thread int* ener,
    threadgroup const uint* state,
    device const int* row_ptr,
    device const int* col_ind,
    device const int8_t* j_vals,
    device const int8_t* h_vals,
    int n, uint tid, uint gsz
) {
    for (int r = 0; r < MSA_LANES; ++r) {
        ener[r] = 0;
    }
    for (uint var = tid; var < uint(n); var += gsz) {
        int bi = int(state[var]);
        int pstart = row_ptr[var];
        int pend = row_ptr[var + 1];
        int h = h_vals[var];
        for (int r = 0; r < MSA_LANES; ++r) {
            int spin = ((bi >> r) & 1) ? -1 : 1;
            ener[r] += h * spin * 1000;
        }
        for (int q = 0; q < MSA_MAX_DEG; ++q) {
            int p = pstart + q;
            if (p < pend) {
                int J = int(j_vals[p]);
                if (J != 0) {
                    int nb = col_ind[p];
                    int nbword = int(state[nb]);
                    int coeff = (nb == int(var)) ? 1000 : 500;
                    for (int r = 0; r < MSA_LANES; ++r) {
                        int spin = ((bi >> r) & 1) ? -1 : 1;
                        int sj = ((nbword >> r) & 1) ? -1 : 1;
                        ener[r] += J * spin * sj * coeff;
                    }
                }
            }
        }
    }
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

#ifdef QUIP_MSA_DIAGNOSTICS
kernel void msa_anneal_diag(
#else
kernel void msa_anneal(
#endif
    device const int* csr_row_ptr [[buffer(0)]],
    device const int* csr_col_ind [[buffer(1)]],
    device const int8_t* csr_J_vals [[buffer(2)]],
    device const int* row_ptr_offsets [[buffer(3)]], // unused by msa: the CSR structure is shared
    device const int* col_ind_offsets [[buffer(4)]],

    constant int& N [[buffer(5)]],
    constant int& num_betas [[buffer(6)]],
    constant int& sweeps_per_beta [[buffer(7)]],
    constant uint& base_seed [[buffer(8)]],

    device const float* beta_schedule [[buffer(9)]],

    device int8_t* final_samples [[buffer(10)]],           // [num_problems * num_reads * packed_size]
    device int* final_energies [[buffer(11)]],             // [num_problems * num_reads] milli, last chunk only

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
#ifdef QUIP_MSA_DIAGNOSTICS
    device ulong* diag_accept_counts [[buffer(25)]],       // [num_threadgroups * group_size] per-thread flips
    device int* diag_energy_partials [[buffer(26)]],       // [num_threadgroups * group_size * 32] per-(thread, lane)
#endif
    threadgroup uint* state [[threadgroup(0)]],            // [N] spin words, sized by the host

    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_group [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup atomic_int lane_total[MSA_LANES];
#ifdef QUIP_MSA_DIAGNOSTICS
    ulong accept_count = 0;                      // per-thread flips across this chunk's sweeps
#endif

    uint tg = threadgroup_pos.x;
    if (tg >= uint(num_threadgroups)) {
        return;
    }
    uint tid = thread_pos_in_group.x;
    uint gsz = threads_per_group.x;
    uint problem_id = tg / uint(words);
    uint w = tg - problem_id * uint(words);
    if (tid < uint(MSA_LANES)) {
        atomic_store_explicit(&lane_total[tid], 0, memory_order_relaxed);
    }

    // CSR structure is shared by every problem in a batch (same topology);
    // only the couplings and fields are per problem.
    int col_ind_start = col_ind_offsets[problem_id];
    device const int* my_csr_row_ptr = csr_row_ptr;
    device const int* my_csr_col_ind = csr_col_ind;
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

        float two_beta = 2.0f * beta;

        for (int sweep = 0; sweep < sweeps_per_beta; sweep++) {
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
                    int m = geometric_draw(xoshiro128starstar(rng), two_beta);
                    int limit = (d + m) >> 1;
                    uint accept = (limit >= d) ? 0xFFFFFFFFu : le_constant(planes, limit);
                    state[var] = bi ^ accept;
#ifdef QUIP_MSA_DIAGNOSTICS
                    accept_count += ulong(popcount(accept));
#endif
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

    // Energies. The diagnostic build keeps its per-(thread, lane) partials on
    // every chunk; the production build reduces them on the last chunk only.
    bool last_chunk = chunk_end >= num_betas;
#ifdef QUIP_MSA_DIAGNOSTICS
    bool need_energy = true;
#else
    bool need_energy = last_chunk;
#endif
    if (need_energy) {
        int ener[MSA_LANES];
        lane_energies(ener, state, my_csr_row_ptr, my_csr_col_ind, my_csr_J_vals,
                      my_h_vals, n, tid, gsz);
#ifdef QUIP_MSA_DIAGNOSTICS
        device int* dst = &diag_energy_partials[(tg * gsz + tid) * MSA_LANES];
        for (int r = 0; r < MSA_LANES; ++r) {
            dst[r] = ener[r];
        }
#endif
        if (last_chunk) {
            for (int r = 0; r < MSA_LANES; ++r) {
                int s = simd_sum(ener[r]);
                if (simd_is_first()) {
                    atomic_fetch_add_explicit(&lane_total[r], s, memory_order_relaxed);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (last_chunk && tid < uint(MSA_LANES)) {
        int read = int(w) * MSA_LANES + int(tid);
        if (read < num_reads) {
            final_energies[problem_id * uint(num_reads) + uint(read)] =
                atomic_load_explicit(&lane_total[tid], memory_order_relaxed);
        }
    }

#ifdef QUIP_MSA_DIAGNOSTICS
    // Per-thread accepted-flip counter for this chunk (overwritten per chunk;
    // no atomics, one disjoint write per thread).
    diag_accept_counts[tg * gsz + tid] = accept_count;
#endif

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
