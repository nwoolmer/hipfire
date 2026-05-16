// V4F SWA-windowed attention with attn_sink (multi-position).
//
// Generalizes v4f_attn_pos0 to attend over up to `sliding_window = 128`
// past positions plus the attn_sink. Computes:
//
//   For each query head h:
//     For each cached position p in [0, n_valid):
//       score[p] = (Q[h] · K_cache[p]) + (attn_sink[h] if p == n_valid - sentinel_ix)
//                 actually: scores are Q·K, sink is added as one extra entry.
//     softmax([scores, attn_sink[h]]) → probs[n_valid + 1]
//     contribution = sum over p < n_valid of probs[p] * V_cache[p]
//
//   For each o_group g (8 heads each):
//     attn_out[g, :] = sum over 8 heads in group of contribution
//
// V4F: n_kv_heads = 1, head_dim = 512, n_query_heads = 64,
//      o_groups = 8 → heads_per_group = 8.
//
// SWA cache layout: k_cache, v_cache: [n_kv_heads, head_dim, window].
// Logical ordering for positions: 0..min(n_total, window) where the
// LATEST is at slot (n_total - 1) % window. For simplicity the
// caller passes `n_valid` = min(n_total, window) and we treat the
// first n_valid slots as the cached positions in their ring order.
// (Position numbering details handled at the caller; this kernel
// just dot-products over n_valid slots.)
//
// Output bounded: |attn_out[g, d]| ≤ 8 · max(|V|) per group.

#include <hip/hip_runtime.h>
#include <math.h>

#define MAX_HEADS_PER_GROUP 16  // n_heads / o_groups, V4F=8
#define MAX_WINDOW 256           // sliding_window, V4F=128

extern "C" __global__ void v4f_attn_swa(
    const float* __restrict__ q,           // [n_heads, head_dim]
    const float* __restrict__ k_cache,     // [n_kv_heads, head_dim, window]
    const float* __restrict__ v_cache,     // [n_kv_heads, head_dim, window]
    const float* __restrict__ attn_sink,   // [n_heads]
    float*       __restrict__ attn_out,    // [o_groups, head_dim]
    int n_heads,
    int head_dim,
    int o_groups,
    int n_valid,                            // active positions in cache (≤ window)
    int window
) {
    const int g = blockIdx.x;
    const int d = threadIdx.x;
    if (g >= o_groups || d >= head_dim) return;

    const int heads_per_g = n_heads / o_groups;

    __shared__ float shared_probs[MAX_HEADS_PER_GROUP * MAX_WINDOW];

    // Phase 1: compute probabilities for each head in this group.
    // Done by thread d == 0 only (heads_per_g · n_valid floats fit in
    // shared mem for V4F: 8 · 128 = 1024 floats = 4 KB).
    if (d == 0) {
        for (int h_local = 0; h_local < heads_per_g; ++h_local) {
            int h = g * heads_per_g + h_local;

            // Compute scores[p] = Q[h] · K[p] for p in [0, n_valid).
            // Single thread loops over positions + dims.
            float max_score = -1e30f;
            float scores[MAX_WINDOW];
            for (int p = 0; p < n_valid; ++p) {
                float s = 0.0f;
                for (int i = 0; i < head_dim; ++i) {
                    s += q[h * head_dim + i] * k_cache[i * window + p];
                }
                scores[p] = s;
                if (s > max_score) max_score = s;
            }
            // Include attn_sink as an extra "position" in the softmax.
            float sink = attn_sink[h];
            if (sink > max_score) max_score = sink;

            // Softmax normalization with stability.
            float sum_exp = expf(sink - max_score);
            for (int p = 0; p < n_valid; ++p) {
                scores[p] = expf(scores[p] - max_score);
                sum_exp += scores[p];
            }
            // probs[h_local, p] = scores[p] / sum_exp; sink prob discarded
            // (it contributes nothing to V).
            for (int p = 0; p < n_valid; ++p) {
                shared_probs[h_local * MAX_WINDOW + p] = scores[p] / sum_exp;
            }
        }
    }
    __syncthreads();

    // Phase 2: each thread d computes one element of attn_out[g, d] =
    // sum over heads in group of (sum over positions of probs * V[p, d]).
    float acc = 0.0f;
    for (int h_local = 0; h_local < heads_per_g; ++h_local) {
        for (int p = 0; p < n_valid; ++p) {
            float prob = shared_probs[h_local * MAX_WINDOW + p];
            // V is shared across heads (tied K=V, 1 kv head).
            float v_val = v_cache[d * window + p];
            acc += prob * v_val;
        }
    }
    attn_out[g * head_dim + d] = acc;
}
