// Hyper-Connections Sinkhorn normaliser (V4F Phase 3).
//
// Mirrors upstream `hc_split_sinkhorn_kernel` exactly:
//   1. comb = softmax(comb, dim=-1) + eps    (row-wise softmax + eps)
//   2. comb = comb / (comb.sum(-2) + eps)    (column normalize)
//   3. for sinkhorn_iters - 1 more iters:
//        comb = comb / (comb.sum(-1) + eps)  (row normalize)
//        comb = comb / (comb.sum(-2) + eps)  (column normalize)
//
// Result is approximately doubly stochastic (rows and columns sum to 1).
//
// Previous version used a clamp([-2,2])+exp+alternating-col-row pattern
// which has different initial dynamics than upstream's row-softmax. The
// clamp limited dynamic range too aggressively and the iteration order
// was col-first rather than row-first.

#include <hip/hip_runtime.h>

extern "C" __global__ void hc_sinkhorn_4x4(
    float* __restrict__ matrix,  // [16]: row-major 4×4
    float  eps,
    int    iters
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;  // single-thread reference

    // Step 1: row-softmax + eps
    for (int r = 0; r < 4; ++r) {
        // row max
        float rm = matrix[r * 4 + 0];
        for (int c = 1; c < 4; ++c) {
            float v = matrix[r * 4 + c];
            if (v > rm) rm = v;
        }
        // exp(x - rm)
        float rs = 0.0f;
        for (int c = 0; c < 4; ++c) {
            float v = __expf(matrix[r * 4 + c] - rm);
            matrix[r * 4 + c] = v;
            rs += v;
        }
        // normalize + eps
        for (int c = 0; c < 4; ++c) {
            matrix[r * 4 + c] = matrix[r * 4 + c] / rs + eps;
        }
    }

    // Step 2: column normalize
    for (int c = 0; c < 4; ++c) {
        float cs = 0.0f;
        for (int r = 0; r < 4; ++r) cs += matrix[r * 4 + c];
        cs += eps;
        for (int r = 0; r < 4; ++r) matrix[r * 4 + c] /= cs;
    }

    // Step 3: (iters - 1) more iterations alternating row and column.
    for (int it = 0; it < iters - 1; ++it) {
        // row normalize
        for (int r = 0; r < 4; ++r) {
            float rs = 0.0f;
            for (int c = 0; c < 4; ++c) rs += matrix[r * 4 + c];
            rs += eps;
            for (int c = 0; c < 4; ++c) matrix[r * 4 + c] /= rs;
        }
        // column normalize
        for (int c = 0; c < 4; ++c) {
            float cs = 0.0f;
            for (int r = 0; r < 4; ++r) cs += matrix[r * 4 + c];
            cs += eps;
            for (int r = 0; r < 4; ++r) matrix[r * 4 + c] /= cs;
        }
    }
}
