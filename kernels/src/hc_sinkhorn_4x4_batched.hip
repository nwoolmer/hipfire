// HC Sinkhorn normaliser — BATCHED variant (Phase B2, 2026-05-18).
//
// Batched twin of `hc_sinkhorn_4x4`. Per batch position b, processes
// the 4×4 matrix at `matrix[b * 16 .. (b+1) * 16]` independently using
// the same row-softmax + alternating col/row normalisation iterations
// as the sequential kernel.
//
// Grid:  [batch_size, 1, 1]
// Block: [1, 1, 1]  (single-thread reference — perf is irrelevant
//                    at 16 elements per position)
//
// Byte-identical to hc_sinkhorn_4x4 at batch_size == 1.

#include <hip/hip_runtime.h>

extern "C" __global__ void hc_sinkhorn_4x4_batched(
    float* __restrict__ matrix,  // [B, 16] row-major 4×4 per batch
    float  eps,
    int    iters,
    int    batch_size
) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    if (threadIdx.x != 0) return;

    float* mat = matrix + (size_t)b * 16;

    // Step 1: row-softmax + eps.
    for (int r = 0; r < 4; ++r) {
        float rm = mat[r * 4 + 0];
        for (int c = 1; c < 4; ++c) {
            float v = mat[r * 4 + c];
            if (v > rm) rm = v;
        }
        float rs = 0.0f;
        for (int c = 0; c < 4; ++c) {
            float v = __expf(mat[r * 4 + c] - rm);
            mat[r * 4 + c] = v;
            rs += v;
        }
        for (int c = 0; c < 4; ++c) {
            mat[r * 4 + c] = mat[r * 4 + c] / rs + eps;
        }
    }

    // Step 2: column normalize.
    for (int c = 0; c < 4; ++c) {
        float cs = 0.0f;
        for (int r = 0; r < 4; ++r) cs += mat[r * 4 + c];
        cs += eps;
        for (int r = 0; r < 4; ++r) mat[r * 4 + c] /= cs;
    }

    // Step 3: (iters - 1) more iterations alternating row and column.
    for (int it = 0; it < iters - 1; ++it) {
        for (int r = 0; r < 4; ++r) {
            float rs = 0.0f;
            for (int c = 0; c < 4; ++c) rs += mat[r * 4 + c];
            rs += eps;
            for (int c = 0; c < 4; ++c) mat[r * 4 + c] /= rs;
        }
        for (int c = 0; c < 4; ++c) {
            float cs = 0.0f;
            for (int r = 0; r < 4; ++r) cs += mat[r * 4 + c];
            cs += eps;
            for (int r = 0; r < 4; ++r) mat[r * 4 + c] /= cs;
        }
    }
}
