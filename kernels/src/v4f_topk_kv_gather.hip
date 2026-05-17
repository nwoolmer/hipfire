// V4F indexer-extended attention: gather K/V rows from main_kv_cache
// at indexer's top-K compressed-slot indices, lay out as a column-slab
// of an [n_kv=1, head_dim, out_stride] tensor — writing columns
// [col_offset, col_offset + K).
//
// This lets the caller stage the gather into an extension of the SWA
// buffer: combined[:, :win] holds raw SWA window K/V; this kernel
// writes combined[:, win:win+K] from main_kv_cache.
//
// V4F has n_kv = 1 and tied K=V, so this kernel emits one buffer and
// the consumer re-uses it as both K and V.
//
// Sentinel topk_idx[k] = -1 (or out-of-range) writes zeros so the
// downstream softmax handles degenerate slots gracefully.

#include <hip/hip_runtime.h>

extern "C" __global__ void v4f_topk_kv_gather_f32(
    const float* __restrict__ kv_cache,   // [N_compressed, head_dim]
    const int*   __restrict__ topk_idx,   // [K]
    float*       __restrict__ out,        // [n_kv=1, head_dim, out_stride]
    int K,
    int head_dim,
    int N_compressed,
    int out_stride,
    int col_offset,
    float scale                            // 1.0 = pass-through; >1 boosts compressed K magnitude
) {
    const int k = blockIdx.x;
    const int d = threadIdx.x;
    if (k >= K || d >= head_dim) return;

    const int idx = topk_idx[k];
    float val = 0.0f;
    if (idx >= 0 && idx < N_compressed) {
        val = kv_cache[idx * head_dim + d] * scale;
    }
    out[d * out_stride + col_offset + k] = val;
}
