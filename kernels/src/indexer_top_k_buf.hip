// HIP-graphs-safe twin of indexer_top_k.
// Reads `n_compressed` (= N) and `k` from device buffers instead of
// i32 kernargs. Body math byte-identical to indexer_top_k for the
// in-range portion; out-of-range top_indices entries are set to -1
// sentinel so downstream gather kernels skip them.

#include <hip/hip_runtime.h>
#include <limits.h>
#include <float.h>

extern "C" __global__ void indexer_top_k_buf(
    const float* __restrict__ scores,
    int*         __restrict__ top_indices,
    const int*   __restrict__ n_compressed_buf,  // [1]
    const int*   __restrict__ k_buf,             // [1]
    int n_idx_heads,
    int max_k                                    // upper bound on k for output stride
) {
    const int head = blockIdx.x;
    if (head >= n_idx_heads) return;
    if (threadIdx.x != 0) return;

    const int n_compressed = n_compressed_buf[0];
    const int k = k_buf[0];

    extern __shared__ unsigned char taken_smem[];
    char* taken = (char*)taken_smem;
    for (int i = 0; i < n_compressed; ++i) taken[i] = 0;

    for (int rank = 0; rank < max_k; ++rank) {
        if (rank >= k) {
            // Beyond the active K → sentinel.
            top_indices[head * max_k + rank] = -1;
            continue;
        }
        float best = -FLT_MAX;
        int best_idx = -1;
        for (int i = 0; i < n_compressed; ++i) {
            if (taken[i]) continue;
            float s = scores[head * n_compressed + i];
            if (s > best) { best = s; best_idx = i; }
        }
        if (best_idx >= 0) {
            top_indices[head * max_k + rank] = best_idx;
            taken[best_idx] = 1;
        } else {
            top_indices[head * max_k + rank] = -1;
        }
    }
}
