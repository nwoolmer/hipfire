// V4F indexer-extended SWA attention — PARALLEL softmax version.
//
// Combines the SWA window K/V (last `n_valid_swa` positions, ring
// buffer stride = `window`) with `n_active_topk` slots gathered from
// the indexer's top-K compressed-K cache (linear buffer stride =
// `topk_window`).
//
// Score layout in softmax:
//   index range [0, n_valid_swa)            → SWA cache positions
//   index range [n_valid_swa, n_valid_swa + n_active_topk)
//                                            → top-K gathered positions
//   sink                                     → extra entry beside above
//
// All positions share a single softmax pass (joint normalisation).
//
// V4F: n_kv = 1, head_dim = 512, n_heads = 64, win = 128, topK ≤ 512.
// Combined n_valid ≤ 128 + 512 = 640.
//
// Parallel layout (Phase A perf rewrite, 2026-05-18):
//   Block = head_dim threads (512). Grid = n_heads blocks.
//   Q[h] is loaded into LDS once per block.
//   Scores: each thread handles a stride-blockDim subset of positions
//     (thread t scores positions t, t+512, t+1024, …). Each scoring
//     op is a full head_dim FMA chain over the LDS-resident Q. Memory
//     access on K is column-coalesced within a warp (32 threads → 32
//     adjacent positions → 32 consecutive bytes in row-major K).
//   Softmax max + sum_exp: warp-shuffle reductions, then cross-warp
//     via LDS. attn_sink is folded in by thread 0.
//   V accumulation: each thread d computes one output element, summing
//     `prob[p] * V[d, p]` for p ∈ [0, n_total).

#include <hip/hip_runtime.h>
#include <math.h>

#define MAX_TOTAL 1024
#define HEAD_DIM_LDS 512
#define MAX_WARPS 16    // 512-thread block / 32-thread warp

extern "C" __global__ __launch_bounds__(512, 1) void v4f_attn_swa_topk_f32(
    const float* __restrict__ q,             // [n_heads, head_dim]
    const float* __restrict__ swa_k,         // [n_kv=1, head_dim, swa_window]
    const float* __restrict__ swa_v,         // [n_kv=1, head_dim, swa_window]
    const float* __restrict__ topk_k,        // [n_kv=1, head_dim, topk_window]
    const float* __restrict__ topk_v,        // [n_kv=1, head_dim, topk_window]
    const float* __restrict__ attn_sink,     // [n_heads]
    float*       __restrict__ attn_out,      // [n_heads, head_dim]
    int n_heads,
    int head_dim,
    int swa_window,
    int topk_window,
    int n_valid_swa,
    int n_active_topk
) {
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    if (h >= n_heads) return;

    const int n_total = n_valid_swa + n_active_topk;
    const float inv_scale = rsqrtf((float)head_dim);

    __shared__ float q_lds[HEAD_DIM_LDS];
    __shared__ float scores_lds[MAX_TOTAL];
    __shared__ float warp_scratch[MAX_WARPS];  // for cross-warp reductions

    // Phase 1: load Q[h] into LDS once.
    if (t < head_dim) {
        q_lds[t] = q[h * head_dim + t];
    }
    __syncthreads();

    // Phase 2: each thread computes scores for its stride-blockDim subset
    // of positions. Per position: full head_dim FMA chain over LDS Q.
    for (int p = t; p < n_total; p += blockDim.x) {
        const float* k_col;
        int stride;
        int col;
        if (p < n_valid_swa) {
            k_col = swa_k;
            stride = swa_window;
            col = p;
        } else {
            k_col = topk_k;
            stride = topk_window;
            col = p - n_valid_swa;
        }
        float s = 0.0f;
        // Unroll-friendly inner loop. K access pattern within a warp:
        // 32 threads with adjacent `col` values hit 32 consecutive
        // memory addresses (k_col[i*stride + col_warp + lane]) — coalesced.
        for (int i = 0; i < head_dim; ++i) {
            s += q_lds[i] * k_col[i * stride + col];
        }
        scores_lds[p] = s * inv_scale;
    }
    __syncthreads();

    // Phase 3: parallel max-reduction over scores + sink.
    float local_max = -1e30f;
    for (int p = t; p < n_total; p += blockDim.x) {
        local_max = fmaxf(local_max, scores_lds[p]);
    }
    if (t == 0) {
        local_max = fmaxf(local_max, attn_sink[h]);
    }
    // Warp-shuffle reduction within each 32-thread warp.
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_down(local_max, off));
    }
    const int warp_id = t >> 5;
    const int lane_id = t & 31;
    if (lane_id == 0) {
        warp_scratch[warp_id] = local_max;
    }
    __syncthreads();

    // Cross-warp reduction: first warp reads all warp results.
    float max_score;
    if (warp_id == 0) {
        const int n_warps = (blockDim.x + 31) >> 5;
        float v = (lane_id < n_warps) ? warp_scratch[lane_id] : -1e30f;
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_down(v, off));
        }
        if (lane_id == 0) {
            warp_scratch[0] = v;
        }
    }
    __syncthreads();
    max_score = warp_scratch[0];

    // Phase 4: parallel sum_exp reduction. Convert scores → exp(score - max)
    // in-place in LDS, accumulate sum.
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
    if (lane_id == 0) {
        warp_scratch[warp_id] = local_sum;
    }
    __syncthreads();

    float sum_exp;
    if (warp_id == 0) {
        const int n_warps = (blockDim.x + 31) >> 5;
        float v = (lane_id < n_warps) ? warp_scratch[lane_id] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down(v, off);
        }
        if (lane_id == 0) {
            warp_scratch[0] = v;
        }
    }
    __syncthreads();
    sum_exp = warp_scratch[0];

    // Phase 5: normalise probabilities (parallel write).
    const float inv_sum = 1.0f / sum_exp;
    for (int p = t; p < n_total; p += blockDim.x) {
        scores_lds[p] *= inv_sum;
    }
    __syncthreads();

    // Phase 6: V accumulation. Each thread d computes one output element.
    // attn_out[h, d] = Σ_p prob[p] · V[d, p].
    if (t >= head_dim) return;
    const int d = t;
    float acc = 0.0f;
    for (int p = 0; p < n_valid_swa; ++p) {
        acc += scores_lds[p] * swa_v[d * swa_window + p];
    }
    for (int p = 0; p < n_active_topk; ++p) {
        acc += scores_lds[n_valid_swa + p] * topk_v[d * topk_window + p];
    }
    attn_out[h * head_dim + d] = acc;
}
