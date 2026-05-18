// V4F indexer-extended SWA attention.
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
// (NB: shares the per-head/thread layout of v4f_attn_swa but with
// MAX_TOTAL ≥ 1024 to accommodate the combined window.)

#include <hip/hip_runtime.h>
#include <math.h>

#define MAX_TOTAL 1024

extern "C" __global__ void v4f_attn_swa_topk_f32(
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
    const int d = threadIdx.x;
    if (h >= n_heads || d >= head_dim) return;

    const int n_total = n_valid_swa + n_active_topk;
    __shared__ float shared_probs[MAX_TOTAL];

    const float inv_scale = rsqrtf((float)head_dim);

    if (d == 0) {
        float max_score = -1e30f;
        float scores[MAX_TOTAL];

        // SWA scores.
        for (int p = 0; p < n_valid_swa; ++p) {
            float s = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                s += q[h * head_dim + i] * swa_k[i * swa_window + p];
            }
            s *= inv_scale;
            scores[p] = s;
            if (s > max_score) max_score = s;
        }
        // Top-K scores.
        for (int p = 0; p < n_active_topk; ++p) {
            float s = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                s += q[h * head_dim + i] * topk_k[i * topk_window + p];
            }
            s *= inv_scale;
            scores[n_valid_swa + p] = s;
            if (s > max_score) max_score = s;
        }

        float sink = attn_sink[h];
        if (sink > max_score) max_score = sink;

        float sum_exp = expf(sink - max_score);
        for (int p = 0; p < n_total; ++p) {
            scores[p] = expf(scores[p] - max_score);
            sum_exp += scores[p];
        }
        for (int p = 0; p < n_total; ++p) {
            shared_probs[p] = scores[p] / sum_exp;
        }
    }
    __syncthreads();

    float acc = 0.0f;
    for (int p = 0; p < n_valid_swa; ++p) {
        acc += shared_probs[p] * swa_v[d * swa_window + p];
    }
    for (int p = 0; p < n_active_topk; ++p) {
        acc += shared_probs[n_valid_swa + p] * topk_v[d * topk_window + p];
    }
    attn_out[h * head_dim + d] = acc;
}
