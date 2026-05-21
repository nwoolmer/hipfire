// V4F joint SWA + topK attention — WMMA Q·K^T variant (Phase 1, lever 1).
//
// Strict counterpart of `v4f_attn_swa_topk_batched_f32` with the Phase-2
// Q·K^T scoring loop replaced by WMMA mma_f32_16x16x16_f16 calls. P·V
// (Phase 6) stays in the scalar path; joint softmax + attn_sink unchanged.
//
// Block: 512 threads = 16 waves (wave32). Grid: [n_heads, batch_size].
// LDS: q_lds[head_dim] + scores_lds[MAX_TOTAL] + warp_scratch[16] + qf16_lds.
//
// Per (head, batch) block: load Q into LDS, convert to F16 fragment,
// then each of 16 waves processes a contiguous strip of K positions
// (block-cooperative). For K position p: 16 head_dim chunks of K are
// loaded as F16 fragments, WMMA-multiplied against the Q F16 fragment.
//
// Q·K^T compute cost:
//   - scalar: 512 threads × (head_dim FMAs per position) × (n_total/512
//     positions per thread) = head_dim × n_total FMA ops total
//   - WMMA: (n_total/16 K-tiles) × (head_dim/16 K-steps) × 1 mma per wave
//     × 16 waves cooperative = same total FMAs but fewer wave-cycles
//     since each mma is 256 MACs in 1 wave-cycle.
//
// K access pattern: K layout is `[head_dim, stride]` so per-K-position
// loads are stride accesses (non-coalesced). This is sub-optimal and
// limits the WMMA win. A future variant with transposed K staging would
// remove this constraint.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

#define MAX_TOTAL 1024
#define HEAD_DIM_LDS 512
#define MAX_WARPS 16
#define WAVE_K_TILE 16   // 16 K positions per wave per WMMA step

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

extern "C" __global__ __launch_bounds__(512, 1) void v4f_attn_swa_topk_batched_wmma_f32(
    const float* __restrict__ q,                 // [batch, n_heads, head_dim]
    const float* __restrict__ swa_k,             // [batch, head_dim, swa_window]
    const float* __restrict__ swa_v,             // [batch, head_dim, swa_window]
    const float* __restrict__ topk_k,            // [batch, head_dim, topk_window]
    const float* __restrict__ topk_v,            // [batch, head_dim, topk_window]
    const float* __restrict__ attn_sink,         // [n_heads]
    const int*   __restrict__ n_valid_swa_arr,   // [batch]
    const int*   __restrict__ n_active_topk_arr, // [batch]
    float*       __restrict__ attn_out,          // [batch, n_heads, head_dim]
    int n_heads,
    int head_dim,
    int swa_window,
    int topk_window,
    int batch_size
) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int t = threadIdx.x;
    if (h >= n_heads || b >= batch_size) return;

    const int n_valid_swa  = n_valid_swa_arr[b];
    const int n_active_topk = n_active_topk_arr[b];
    const int n_total      = n_valid_swa + n_active_topk;
    const float inv_scale  = rsqrtf((float)head_dim);

    __shared__ float q_lds[HEAD_DIM_LDS];
    __shared__ _Float16 q_f16_lds[HEAD_DIM_LDS];   // F16 Q for WMMA
    __shared__ float scores_lds[MAX_TOTAL];
    __shared__ float warp_scratch[MAX_WARPS];

    const size_t q_base   = (size_t)(b * n_heads + h) * head_dim;
    const size_t swa_base  = (size_t)b * head_dim * swa_window;
    const size_t topk_base = (size_t)b * head_dim * topk_window;
    const size_t out_base = (size_t)(b * n_heads + h) * head_dim;

    // ── Phase 1: load Q[b, h] into LDS once + F32→F16 staging ─────────
    if (t < head_dim) {
        float qv = q[q_base + t];
        q_lds[t] = qv;
        q_f16_lds[t] = (_Float16)qv;
    }
    __syncthreads();

    // ── Phase 2: WMMA Q·K^T ───────────────────────────────────────────
    // Each wave (16 waves total) handles a contiguous strip of K
    // positions. The strip size is K_TILES_PER_WAVE = ceil(n_total /
    // (16 waves × WAVE_K_TILE)) — wave w covers positions in tile range
    // [w * WAVE_K_TILE, (w+1) * WAVE_K_TILE) within each k_tile_offset
    // group.
    const int warp_id = t >> 5;
    const int lane    = t & 31;
    const int lane_lo = lane & 15;       // 0..15 maps to row of A fragment

    // Iterate over K-position tiles. With 16 waves each owning 16
    // positions per tile, one block-pass covers 256 K positions. For
    // n_total up to 640 (V4F default SWA=128 + topk=512) we need 3
    // block-passes (640/256 = 2.5 → ceil 3).
    const int positions_per_pass = 16 * WAVE_K_TILE;  // = 256
    for (int p_pass_start = 0; p_pass_start < n_total; p_pass_start += positions_per_pass) {
        const int my_tile_start = p_pass_start + warp_id * WAVE_K_TILE;
        if (my_tile_start >= n_total) {
            // This wave has no work this pass.
        } else {
            const int my_tile_end = min(my_tile_start + WAVE_K_TILE, n_total);
            const int tile_len = my_tile_end - my_tile_start;

            // WMMA accumulator: 8 outputs per thread.
            float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

            // Each WMMA mma_f32_16x16x16_f16 processes head_dim=16 K
            // for 16 K-positions against 16 broadcast Q values.
            // Iterate over head_dim/16 = 32 K-axis chunks.
            for (int k_chunk = 0; k_chunk < head_dim; k_chunk += 16) {
                // Load K fragment: 16 K positions × 16 head_dim values.
                // Per thread: load 16 K values from ITS K position.
                // K layout: K[d * stride + col]. For col = my_tile_start
                // + lane_lo (where lane in 0..15), d in [k_chunk, k_chunk+16).
                // Strided access pattern (non-coalesced); future work
                // is a transposed staging variant.
                const int p_in_tile = lane_lo;   // 0..15 → position within tile
                const int p = my_tile_start + p_in_tile;
                _Float16 a_buf[16];
                if (p < my_tile_end) {
                    const float* k_col;
                    int stride;
                    int col;
                    if (p < n_valid_swa) {
                        k_col = swa_k + swa_base;
                        stride = swa_window;
                        col = p;
                    } else {
                        k_col = topk_k + topk_base;
                        stride = topk_window;
                        col = p - n_valid_swa;
                    }
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        a_buf[i] = (_Float16)k_col[(k_chunk + i) * stride + col];
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) a_buf[i] = (_Float16)0.0f;
                }
                half16_t a_frag;
                #pragma unroll
                for (int i = 0; i < 16; ++i) a_frag[i] = a_buf[i];

                // B fragment: broadcast Q across all 16 cols. Each row
                // of B is Q[k_chunk..k_chunk+16] (same for every row).
                half16_t b_frag;
                #pragma unroll
                for (int i = 0; i < 16; ++i) b_frag[i] = q_f16_lds[k_chunk + i];

                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, acc);
            }

            // Write the wave's 16 score outputs. Per RDNA3 wave32 WMMA:
            //   acc[j] = C[2*j + (tid>>4)][tid & 15]
            // We want col-0 outputs (since B-cols all equal). That means
            // only lane=0 (and 16) hold useful col-0 values. Each of
            // lanes 0 and 16 writes 8 rows × 1 col = 8 useful values.
            if (lane_lo == 0) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int p_in_tile = 2 * j + (lane >> 4);
                    int p = my_tile_start + p_in_tile;
                    if (p_in_tile < tile_len && p < n_total) {
                        scores_lds[p] = acc[j] * inv_scale;
                    }
                }
            }
        }
        __syncthreads();
    }

    // ── Phase 3: parallel max-reduction over scores + sink ───────────
    float local_max = -1e30f;
    for (int p = t; p < n_total; p += blockDim.x) {
        local_max = fmaxf(local_max, scores_lds[p]);
    }
    if (t == 0) {
        local_max = fmaxf(local_max, attn_sink[h]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_down(local_max, off));
    }
    const int wid = t >> 5;
    const int lid = t & 31;
    if (lid == 0) {
        warp_scratch[wid] = local_max;
    }
    __syncthreads();

    float max_score;
    if (wid == 0) {
        const int n_warps = (blockDim.x + 31) >> 5;
        float v = (lid < n_warps) ? warp_scratch[lid] : -1e30f;
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_down(v, off));
        }
        if (lid == 0) {
            warp_scratch[0] = v;
        }
    }
    __syncthreads();
    max_score = warp_scratch[0];

    // ── Phase 4: sum_exp reduction ───────────────────────────────────
    float local_sum = 0.0f;
    for (int p = t; p < n_total; p += blockDim.x) {
        float e = expf(scores_lds[p] - max_score);
        scores_lds[p] = e;
        local_sum += e;
    }
    if (t == 0) {
        local_sum += expf(attn_sink[h] - max_score);
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += __shfl_down(local_sum, off);
    }
    if (lid == 0) {
        warp_scratch[wid] = local_sum;
    }
    __syncthreads();

    float sum_exp;
    if (wid == 0) {
        const int n_warps = (blockDim.x + 31) >> 5;
        float v = (lid < n_warps) ? warp_scratch[lid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down(v, off);
        }
        if (lid == 0) {
            warp_scratch[0] = v;
        }
    }
    __syncthreads();
    sum_exp = warp_scratch[0];

    // ── Phase 5: normalise probabilities ─────────────────────────────
    const float inv_sum = 1.0f / sum_exp;
    for (int p = t; p < n_total; p += blockDim.x) {
        scores_lds[p] *= inv_sum;
    }
    __syncthreads();

    // ── Phase 6: V accumulation — UNCHANGED scalar path ──────────────
    if (t >= head_dim) return;
    const int d = t;
    const float* swa_v_b  = swa_v + swa_base;
    const float* topk_v_b = topk_v + topk_base;
    float acc = 0.0f;
    for (int p = 0; p < n_valid_swa; ++p) {
        acc += scores_lds[p] * swa_v_b[d * swa_window + p];
    }
    for (int p = 0; p < n_active_topk; ++p) {
        acc += scores_lds[n_valid_swa + p] * topk_v_b[d * topk_window + p];
    }
    attn_out[out_base + d] = acc;
}
