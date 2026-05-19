#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Bulk F32→F16 conversion. Used to stage WMMA-bound activations
// once per layer so the per-call GEMMs can read F16 directly.
//
// Grid: [ceil(n / 128), 1, 1]   Block: [128, 1, 1]

extern "C" __launch_bounds__(128, 8)
__global__ void convert_f32_to_f16(
    const float*    __restrict__ src,
    _Float16*       __restrict__ dst,
    long long n
) {
    const long long i = (long long)blockIdx.x * 128 + threadIdx.x;
    if (i >= n) return;
    dst[i] = (_Float16)src[i];
}
