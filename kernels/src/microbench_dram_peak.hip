#include <hip/hip_runtime.h>

// Bandwidth-saturation test: each workgroup streams a 64KB-aligned
// region as float4 reads and writes. Pure DRAM throughput baseline so
// we know what GEMM kernels are leaving on the table.
//
// Grid: [N_BLOCKS, 1, 1]   block: [256, 1, 1].
// Each workgroup processes BYTES_PER_BLOCK = 256 * 4 * sizeof(float4) = 16 KB
// of reads via 4 outstanding float4 loads per thread.

__launch_bounds__(256, 4)
extern "C" __global__ void microbench_dram_read_copy(
    const float4* __restrict__ src,
    float4*       __restrict__ dst,
    long long n_float4
) {
    const long long block_offset = (long long)blockIdx.x * 256 * 4;
    const long long tid = block_offset + threadIdx.x * 4;
    if (tid + 3 >= n_float4) return;
    float4 a = src[tid];
    float4 b = src[tid + 1];
    float4 c = src[tid + 2];
    float4 d = src[tid + 3];
    dst[tid]     = a;
    dst[tid + 1] = b;
    dst[tid + 2] = c;
    dst[tid + 3] = d;
}
