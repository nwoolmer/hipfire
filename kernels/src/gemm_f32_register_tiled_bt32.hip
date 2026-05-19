// Register-tiled F32 batched GEMM with BATCH_TILE=32 — sibling of
// gemm_f32_register_tiled.hip but loads each weight row ONCE for up
// to 32 batch positions instead of looping over ceil(B/8) tiles.
//
// Math contract: identical output (within FMA-order epsilon) to the
// BATCH_TILE=8 variant. Only the per-thread accumulator count and the
// grid Y dimension differ.
//
// Layout:
//   A : [M, K]                   row-major F32 weights
//   x : [batch_size, K]          row-major F32 inputs
//   y : [batch_size, M]          row-major F32 outputs
//
// Grid:  [M, ceil(batch_size / 32), 1]
// Block: [32, 1, 1]    — one wave per (row, batch_tile=32) pair
//
// Why: V4F compressor wkv/wgate (proj_dim ∈ {256, 512, 1024},
// hidden=4096, B=32) was running at ~9 GB/s on gfx1151 — the
// BATCH_TILE=8 variant reads each weight row B/8 = 4 times. At B=32,
// BATCH_TILE=32 collapses that to one read per row, expected 4×
// bandwidth reduction. Acc register pressure: 32 floats per thread
// = 32 VGPR, comfortably within the 1024 VGPR/SIMD budget at
// __launch_bounds__(32, 16).

#include <hip/hip_runtime.h>

#define BATCH_TILE 32

extern "C" __launch_bounds__(32, 16)
__global__ void gemm_f32_register_tiled_bt32(
    const float* __restrict__ A,    // [M, K]
    const float* __restrict__ x,    // [batch_size, K]
    float* __restrict__ y,          // [batch_size, M]
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int batch_start = blockIdx.y * BATCH_TILE;
    const int tid = threadIdx.x;

    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    const float* a_row = A + (size_t)row * K;

    float acc[BATCH_TILE];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) acc[b] = 0.0f;

    for (int k = tid; k < K; k += 32) {
        const float w = a_row[k];
        #pragma unroll
        for (int b = 0; b < BATCH_TILE; b++) {
            if (b < local_bs) {
                acc[b] += w * x[(size_t)(batch_start + b) * K + k];
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        if (b >= local_bs) continue;
        float val = acc[b];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down(val, offset);
        }
        if (tid == 0) {
            y[(size_t)(batch_start + b) * M + row] = val;
        }
    }
}
