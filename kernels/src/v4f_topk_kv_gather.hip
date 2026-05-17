// V4F indexer-extended attention: gather K/V rows from main_kv_cache
// at indexer's top-K compressed-slot indices, lay out as
// [n_kv, head_dim, K_active] to match v4f_attn_swa's swa_k layout.
//
// V4F has n_kv = 1 and tied K=V, so the gather produces one tensor
// reused as both K and V by the consumer kernel. We still take an
// `n_kv` parameter for future generalization.
//
// Grid:
//   block.x = K_active (one block per gathered slot)
//   block dim = head_dim (one thread per D)
//
// Top-K index -1 is a sentinel meaning "fewer than K valid slots"; we
// write zeros so the downstream softmax can attend to a degenerate
// row without poisoning the result.

#include <hip/hip_runtime.h>

extern "C" __global__ void v4f_topk_kv_gather_f32(
    const float* __restrict__ kv_cache,   // [N_compressed, head_dim]
    const int*   __restrict__ topk_idx,   // [K]
    float*       __restrict__ out,        // [n_kv=1, head_dim, K]
    int K,
    int head_dim,
    int N_compressed
) {
    const int k = blockIdx.x;
    const int d = threadIdx.x;
    if (k >= K || d >= head_dim) return;

    const int idx = topk_idx[k];
    float val = 0.0f;
    if (idx >= 0 && idx < N_compressed) {
        val = kv_cache[idx * head_dim + d];
    }
    // Output layout: [head_dim, K] with K contiguous → out[d * K + k].
    // n_kv = 1, so the (n_kv, ...) prefix is identity.
    out[d * K + k] = val;
}
