// V4F SWA-windowed attention with attn_sink (multi-position).
//
// Output is PER-HEAD (no group reduction). Upstream V4F preserves
// the [n_heads, head_dim] attention output and feeds it into wo_a
// (per-group O-LoRA) + wo_b for the residual contribution.
//
// For each query head h:
//   For each cached position p in [0, n_valid):
//     score[p] = Q[h] · K_cache[p] / sqrt(head_dim)
//   softmax with sink-as-extra-entry:
//     Z = sum_p exp(score[p] - max) + exp(sink_h - max)
//     prob[p] = exp(score[p] - max) / Z
//   attn_out[h, d] = sum_p prob[p] * V_cache[p, d]
//
// V4F: n_kv_heads = 1, head_dim = 512, n_query_heads = 64.
// SWA cache layout: [n_kv_heads, head_dim, window].
//
// Parallel layout (Phase A perf rewrite, 2026-05-18):
//   Block = head_dim threads (512). Grid = n_heads blocks.
//   Q[h] loaded into LDS once per block.
//   Scores: each thread handles a stride-blockDim subset of positions.
//   Softmax max + sum_exp: warp-shuffle reductions, then cross-warp via LDS.
//   V accumulation: each thread d computes one output element.
//
// Twin of v4f_attn_swa_topk for the non-topk path (pre-indexer-warmup or
// pure-SWA layers). Same launch shape (n_heads × head_dim).

#include <hip/hip_runtime.h>
#include <math.h>

#define HEAD_DIM_LDS 512
#define MAX_WINDOW 1024
#define MAX_WARPS 16    // 512-thread block / 32-thread warp

extern "C" __global__ __launch_bounds__(512, 1) void v4f_attn_swa(
    const float* __restrict__ q,           // [n_heads, head_dim]
    const float* __restrict__ k_cache,     // [n_kv_heads, head_dim, window]
    const float* __restrict__ v_cache,     // [n_kv_heads, head_dim, window]
    const float* __restrict__ attn_sink,   // [n_heads]
    float*       __restrict__ attn_out,    // [n_heads, head_dim]
    int n_heads,
    int head_dim,
    int o_groups,                           // unused — kept in signature for ABI compat
    int n_valid,                            // active positions in cache (≤ window)
    int window
) {
    (void)o_groups;
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    if (h >= n_heads) return;

    const float inv_scale = rsqrtf((float)head_dim);

    __shared__ float q_lds[HEAD_DIM_LDS];
    __shared__ float scores_lds[MAX_WINDOW];
    __shared__ float warp_scratch[MAX_WARPS];

    // Phase 1: cooperative load of Q[h] into LDS.
    if (t < head_dim) {
        q_lds[t] = q[h * head_dim + t];
    }
    __syncthreads();

    // Phase 2: parallel score compute. Each thread handles a stride-blockDim
    // subset of positions. Within a warp, 32 threads with adjacent `p` hit
    // 32 consecutive k_cache addresses (k_cache[i*window + p_base+lane]) —
    // coalesced.
    for (int p = t; p < n_valid; p += blockDim.x) {
        float s = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            s += q_lds[i] * k_cache[i * window + p];
        }
        scores_lds[p] = s * inv_scale;
    }
    __syncthreads();

    // Phase 3: parallel max-reduction over scores + sink.
    float local_max = -1e30f;
    for (int p = t; p < n_valid; p += blockDim.x) {
        local_max = fmaxf(local_max, scores_lds[p]);
    }
    if (t == 0) {
        local_max = fmaxf(local_max, attn_sink[h]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_down(local_max, off));
    }
    const int warp_id = t >> 5;
    const int lane_id = t & 31;
    if (lane_id == 0) {
        warp_scratch[warp_id] = local_max;
    }
    __syncthreads();

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

    // Phase 4: parallel sum_exp. Convert scores → exp(score - max) in place.
    float local_sum = 0.0f;
    for (int p = t; p < n_valid; p += blockDim.x) {
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
    for (int p = t; p < n_valid; p += blockDim.x) {
        scores_lds[p] *= inv_sum;
    }
    __syncthreads();

    // Phase 6: V accumulation. Each thread d computes one output element.
    if (t >= head_dim) return;
    const int d = t;
    float acc = 0.0f;
    for (int p = 0; p < n_valid; ++p) {
        acc += scores_lds[p] * v_cache[d * window + p];
    }
    attn_out[h * head_dim + d] = acc;
}
