// F32 GEMM per-output workgroup with float4 vector loads.
// Same grid (M, B) as gemm_f32_per_output but each thread issues
// ONE 16-byte float4 load instead of 4 separate 4-byte stride-32
// loads. On RDNA wave32 this produces fewer, fatter memory
// transactions (4 cache lines per warp instead of 4 separate
// 128-byte coalesced loads), reducing instruction-issue pressure
// and improving BW.
//
// Each thread handles 4 contiguous K elements per inner step:
//   stride per iter = 32 threads * 4 floats = 128 floats
//   K/128 outer iterations
//
// For K=4096: 32 inner iterations of 4 FMAs each = 128 FMAs/thread.
//
// Inputs/outputs are F32 row-major; A and x must be 16-byte aligned
// at the start of each row (caller's responsibility).

#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(32, 20)
__global__ void gemm_f32_per_output_v4(
    const float* __restrict__ A,    // [M, K]
    const float* __restrict__ x,    // [batch_size, K]
    float* __restrict__ y,          // [batch_size, M]
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    const int b = blockIdx.y;
    if (row >= M || b >= batch_size) return;
    const int tid = threadIdx.x;

    const float4* a4 = reinterpret_cast<const float4*>(A + (size_t)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(x + (size_t)b * K);
    const int K4 = K / 4;
    const int stride4 = 32;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // K4 outer unroll for ILP.
    const int k4_quads = K4 / (stride4 * 4);
    for (int q = 0; q < k4_quads; q++) {
        const int k0 = q * stride4 * 4 + tid;
        float4 w0 = a4[k0];
        float4 v0 = x4[k0];
        float4 w1 = a4[k0 + stride4];
        float4 v1 = x4[k0 + stride4];
        float4 w2 = a4[k0 + 2 * stride4];
        float4 v2 = x4[k0 + 2 * stride4];
        float4 w3 = a4[k0 + 3 * stride4];
        float4 v3 = x4[k0 + 3 * stride4];
        acc0 += w0.x * v0.x + w0.y * v0.y + w0.z * v0.z + w0.w * v0.w;
        acc1 += w1.x * v1.x + w1.y * v1.y + w1.z * v1.z + w1.w * v1.w;
        acc2 += w2.x * v2.x + w2.y * v2.y + w2.z * v2.z + w2.w * v2.w;
        acc3 += w3.x * v3.x + w3.y * v3.y + w3.z * v3.z + w3.w * v3.w;
    }
    // Tail (K not divisible by 32*4*4=512).
    const int tail_start = k4_quads * stride4 * 4;
    for (int k4 = tail_start + tid; k4 < K4; k4 += stride4) {
        float4 w = a4[k4];
        float4 v = x4[k4];
        acc0 += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }
    if (tid == 0) {
        y[(size_t)b * M + row] = acc;
    }
}
