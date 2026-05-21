#include <hip/hip_runtime.h>

// Multirow (R=2) variant of gemv_f16_xf32: each block computes TWO
// output rows, sharing one set of x[] reads across both. Same memory
// pattern per row (wave32, lane strides by 32 over K), but `x[i]` is
// loaded once per lane per iteration and reused for both accumulators.
//
// Bandwidth model:
//   single-row:   M × (K bytes weight + K F32 input) reads per launch
//   multirow R=2: ceil(M/2) × (2K bytes weight + K F32 input)
//                 ≈ M × (1K bytes weight + 0.5K F32 input)
//
// At K=2048: input bytes drop from 8 KB/row to 4 KB/row — 33%
// bandwidth reduction even with perfect L1/L2 caching of x[] in the
// scalar variant (cache hits still cost cycles).
//
// Shape contract is the same as gemv_f16_xf32:
//   grid = [ceil(M/2), 1, 1], block = [32, 1, 1]
//
// If M is odd, the last block writes a single output (row M-1) and the
// second-row accumulator is left as 0 (cheap; doesn't write).

__launch_bounds__(32)
extern "C" __global__ void gemv_f16_xf32_multirow_r2(
    const _Float16* __restrict__ W,   // [M, K] F16
    const float*    __restrict__ x,   // [K] F32
    float*          __restrict__ y,   // [M] F32
    int M, int K
) {
    const int base_row = blockIdx.x * 2;
    if (base_row >= M) return;
    const int tid = threadIdx.x;
    const bool have_r1 = (base_row + 1) < M;

    const _Float16* w0 = W + (size_t)base_row * K;
    const _Float16* w1 = have_r1 ? (W + (size_t)(base_row + 1) * K) : w0;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int i = tid; i < K; i += 32) {
        const float xi = x[i];
        sum0 += (float)w0[i] * xi;
        sum1 += (float)w1[i] * xi;
    }

    // Wave-shuffle reduction (wave32).
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
    }
    if (tid == 0) {
        y[base_row] = sum0;
        if (have_r1) y[base_row + 1] = sum1;
    }
}
