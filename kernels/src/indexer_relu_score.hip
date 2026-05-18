// V4F indexer score kernel — F32, per-(N) combined-across-heads.
//
// Upstream V4F's indexer scores each compressed slot t against the
// current Q via:
//
//   score[t] = sum_h relu(Q[h, :] · K_cache[t, :]) * weights[h]
//
// where Q is [H, D], K_cache is [N, D] (single key per slot, NOT
// per-head — the compressor produces one head_dim vector per slot
// shared across heads), and weights is [H] (output of
// `weights_proj @ x`).
//
// Layout differs from the prior `indexer_compressed_k_score` kernel
// which assumed [H, D, N] per-head K. Here K is genuinely shared.
//
// Grid:  block.x = N (one block per compressed slot)
// Block: H threads (one per head). Each thread does the full
//        D-dim dot for its head, applies relu * weight, then we
//        reduce across heads via LDS.

#include <hip/hip_runtime.h>

#define MAX_HEADS 64  // V4F: index_n_heads = 64

extern "C" __global__ __launch_bounds__(MAX_HEADS, 4)
void indexer_relu_score_f32(
    const float* __restrict__ q,         // [H, D]
    const float* __restrict__ k_cache,   // [N, D]
    const float* __restrict__ weights,   // [H]
    float*       __restrict__ scores,    // [N]
    int H,
    int D,
    int N
) {
    const int n = blockIdx.x;
    if (n >= N) return;
    const int tid = threadIdx.x;
    if (tid >= H) return;

    const float* qh = q + tid * D;
    const float* kt = k_cache + n * D;
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
        dot += qh[d] * kt[d];
    }
    float v = (dot > 0.0f ? dot : 0.0f) * weights[tid];

    __shared__ float reduce[MAX_HEADS];
    reduce[tid] = v;
    __syncthreads();
    for (int s = MAX_HEADS / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    if (tid == 0) scores[n] = reduce[0];
}
