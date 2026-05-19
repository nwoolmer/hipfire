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

// `n_stride` and `k_stride` are the STORAGE strides (allocated row size
// of `scores` and `top_indices`). `n_iter` and `k_fill` are the actual
// work bounds — typically much smaller for short chunks (n_iter ≈ chunk
// max n_filled; k_fill = min(topk_max, n_iter)). Iterating only n_iter
// slots and filling only k_fill ranks is a 100×+ win at low context.
// Ranks ≥ k_fill get -1 sentinels.
extern "C" __global__ void indexer_top_k_batched(
    const float* __restrict__ scores,
    int*         __restrict__ top_indices,
    int n_idx_heads,
    int n_stride,
    int n_iter,
    int k_stride,
    int k_fill,
    int batch_size
) {
    const int head = blockIdx.x;
    const int b    = blockIdx.y;
    if (head >= n_idx_heads || b >= batch_size) return;
    if (threadIdx.x != 0) return;

    const size_t scores_base = (size_t)b * n_idx_heads * n_stride
                             + (size_t)head * n_stride;
    const size_t out_base    = (size_t)b * n_idx_heads * k_stride
                             + (size_t)head * k_stride;

    extern __shared__ unsigned char taken_smem[];
    char* taken = (char*)taken_smem;
    for (int i = 0; i < n_iter; ++i) taken[i] = 0;

    // Find k_fill best unselected scores.
    for (int rank = 0; rank < k_fill; ++rank) {
        float best = -FLT_MAX;
        int best_idx = -1;
        for (int i = 0; i < n_iter; ++i) {
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
    // Fill remaining ranks (k_fill .. k_stride) with -1 sentinels so
    // downstream gather treats them as "no expert".
    for (int rank = k_fill; rank < k_stride; ++rank) {
        top_indices[out_base + rank] = -1;
    }
}
