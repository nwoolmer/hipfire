// Register-tiled F32 batched GEMM with 2 waves (64 threads) per workgroup.
// Same M/B grid pattern as gemm_f32_register_tiled (BATCH_TILE=8) but
// doubles thread count to issue more memory transactions per workgroup.
//
// Rationale: gemm_f32_register_tiled grids at (M, ceil(B/8)) = ~8k
// workgroups for V4F compressor shapes (M=512-1024, B=32). That's only
// ~32 waves per CU on gfx1151 — well below the 16 waves/SIMD × 160
// SIMDs = 2560 waves needed for full occupancy. By doubling to 64
// threads per workgroup we double the in-flight memory requests per
// workgroup, helping saturate DRAM even with the same workgroup count.
//
// Each thread now handles K/64 instead of K/32 weight reads. The
// register-tile accumulators are still BATCH_TILE=8 per thread.
// Warp-reduce happens across all 64 threads via two-stage shuffle:
// intra-wave (16->0) then cross-wave via LDS.

#include <hip/hip_runtime.h>

#define BATCH_TILE 8
#define WG_THREADS 64

extern "C" __launch_bounds__(64, 8)
__global__ void gemm_f32_register_tiled_w2(
    const float* __restrict__ A,    // [M, K]
    const float* __restrict__ x,    // [batch_size, K]
    float* __restrict__ y,          // [batch_size, M]
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int batch_start = blockIdx.y * BATCH_TILE;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;   // 0 or 1

    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    const float* a_row = A + (size_t)row * K;

    float acc[BATCH_TILE];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) acc[b] = 0.0f;

    for (int k = tid; k < K; k += WG_THREADS) {
        const float w = a_row[k];
        #pragma unroll
        for (int b = 0; b < BATCH_TILE; b++) {
            if (b < local_bs) {
                acc[b] += w * x[(size_t)(batch_start + b) * K + k];
            }
        }
    }

    // Two-stage reduce: intra-wave shuffle, then cross-wave via LDS.
    __shared__ float wave_partial[BATCH_TILE * 2];   // 2 waves x 8 batches
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        if (b >= local_bs) continue;
        float val = acc[b];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down(val, offset);
        }
        if (lane == 0) wave_partial[wave * BATCH_TILE + b] = val;
    }
    __syncthreads();
    if (tid < BATCH_TILE) {
        int b = tid;
        if (b < local_bs) {
            float v = wave_partial[b] + wave_partial[BATCH_TILE + b];
            y[(size_t)(batch_start + b) * M + row] = v;
        }
    }
}
