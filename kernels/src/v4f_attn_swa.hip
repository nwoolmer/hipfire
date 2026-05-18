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

#include <hip/hip_runtime.h>
#include <math.h>

// V4F: 128 SWA + 512 indexer top-K = 640. Round up to 1024 for headroom.
// Increases per-thread-0 stack from 1KB to 4KB; spilled to scratch on
// gfx1151. SWA-only path with n_valid ≤ 128 still uses only ~128 entries
// — extra capacity is dead weight, not a perf hit.
#define MAX_WINDOW 1024

extern "C" __global__ void v4f_attn_swa(
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
    const int h = blockIdx.x;             // head index
    const int d = threadIdx.x;            // dim index
    if (h >= n_heads || d >= head_dim) return;

    __shared__ float shared_probs[MAX_WINDOW];

    const float inv_scale = rsqrtf((float)head_dim);

    // Phase 1: compute probabilities for this head. Thread 0 computes,
    // broadcasts to others via shared memory.
    if (d == 0) {
        float max_score = -1e30f;
        float scores[MAX_WINDOW];
        for (int p = 0; p < n_valid; ++p) {
            float s = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                s += q[h * head_dim + i] * k_cache[i * window + p];
            }
            s *= inv_scale;
            scores[p] = s;
            if (s > max_score) max_score = s;
        }
        float sink = attn_sink[h];
        if (sink > max_score) max_score = sink;
        float sum_exp = expf(sink - max_score);
        for (int p = 0; p < n_valid; ++p) {
            scores[p] = expf(scores[p] - max_score);
            sum_exp += scores[p];
        }
        for (int p = 0; p < n_valid; ++p) {
            shared_probs[p] = scores[p] / sum_exp;
        }
    }
    __syncthreads();

    // Phase 2: attn_out[h, d] = sum_p prob[p] * V[d, p].
    float acc = 0.0f;
    for (int p = 0; p < n_valid; ++p) {
        acc += shared_probs[p] * v_cache[d * window + p];
    }
    attn_out[h * head_dim + d] = acc;
}
