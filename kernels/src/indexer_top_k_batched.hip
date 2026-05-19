// Indexer top-K selection — BATCHED variant (Phase A3, 2026-05-18).
//
// Per batch position and per head, select the top-K position indices
// from the score array produced by indexer_compressed_k_score. Layout:
//   scores      : [batch, n_idx_heads, n_compressed]  fp32 in
//   top_indices : [batch, n_idx_heads, k]             i32  out
//
// Grid:  [n_idx_heads, batch, 1]
// Block: same as sequential (single-thread stub; dynamic LDS for `taken`)
//
// At batch_size == 1 this is byte-identical to `indexer_top_k`.

#include <hip/hip_runtime.h>
#include <float.h>

extern "C" __global__ void indexer_top_k_batched(
    const float* __restrict__ scores,
    int*         __restrict__ top_indices,
    int n_idx_heads,
    int n_compressed,
    int k,
    int batch_size
) {
    const int head = blockIdx.x;
    const int b    = blockIdx.y;
    if (head >= n_idx_heads || b >= batch_size) return;
    if (threadIdx.x != 0) return;

    const size_t scores_base = (size_t)b * n_idx_heads * n_compressed
                             + (size_t)head * n_compressed;
    const size_t out_base    = (size_t)b * n_idx_heads * k
                             + (size_t)head * k;

    extern __shared__ unsigned char taken_smem[];
    char* taken = (char*)taken_smem;
    for (int i = 0; i < n_compressed; ++i) taken[i] = 0;

    for (int rank = 0; rank < k; ++rank) {
        float best = -FLT_MAX;
        int best_idx = -1;
        for (int i = 0; i < n_compressed; ++i) {
            if (taken[i]) continue;
            float s = scores[scores_base + i];
            if (s > best) { best = s; best_idx = i; }
        }
        if (best_idx >= 0) {
            top_indices[out_base + rank] = best_idx;
            taken[best_idx] = 1;
        } else {
            top_indices[out_base + rank] = -1;
        }
    }
}
