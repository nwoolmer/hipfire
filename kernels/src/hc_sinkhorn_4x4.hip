// Hyper-Connections Sinkhorn normaliser (V4F Phase 3).
//
// Takes a 4×4 mixing matrix (stored as 16 fp32 values, derived from
// the per-layer hc_*_fn projection + hc_*_base bias) and runs N
// rounds of alternating row/column normalisation. After N iters,
// each row and column sums to 1 (within `eps`).
//
// V4F uses N = hc_sinkhorn_iters = 20 by default.
//
// Signature:
//
//   matrix : [16] fp32, in/out (modified in place)
//   eps    : scalar fp32
//   iters  : scalar int (N)
//
// Strategy: one workgroup, 4 threads (one per row). Communication
// via warp-shuffles or shared memory.
//
// Status: STUB. Single-thread reference implementation for
// correctness; production version uses warp-level reduction.

#include <hip/hip_runtime.h>

extern "C" __global__ void hc_sinkhorn_4x4(
    float* __restrict__ matrix,  // [16]: row-major 4×4
    float  eps,
    int    iters
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;  // stub: single-thread.

    // Sinkhorn requires non-negative input. The HC control vector
    // (c = W_fn @ x + base) can be mixed-sign; abs() it to guarantee
    // bounded output. Paper-canonical preprocessing may be
    // exp()/softplus/etc instead — to verify. abs is a safe lower
    // bound: it preserves magnitude order, kills sign.
    for (int i = 0; i < 16; ++i) {
        float v = matrix[i];
        matrix[i] = v < 0.0f ? -v : v;
    }

    for (int it = 0; it < iters; ++it) {
        // Row normalisation: each row sums to 1.
        for (int r = 0; r < 4; ++r) {
            float sum = 0.0f;
            for (int c = 0; c < 4; ++c) sum += matrix[r * 4 + c];
            sum += eps;
            for (int c = 0; c < 4; ++c) matrix[r * 4 + c] /= sum;
        }
        // Column normalisation: each column sums to 1.
        for (int c = 0; c < 4; ++c) {
            float sum = 0.0f;
            for (int r = 0; r < 4; ++r) sum += matrix[r * 4 + c];
            sum += eps;
            for (int r = 0; r < 4; ++r) matrix[r * 4 + c] /= sum;
        }
    }
}
