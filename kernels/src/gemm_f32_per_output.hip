// F32 GEMM with grid = (M, B) — one workgroup per output element.
// Each workgroup does y[b, m] = sum_k A[m, k] * x[b, k] with 32 threads
// and a warp-shuffle K-reduction. Mirrors the MoE-GEMV grid pattern
// that already saturates DRAM at 192 GB/s on gfx1151.
//
// vs gemm_f32_register_tiled (BATCH_TILE=8):
//   BT=8 grid: (M, ceil(B/8))  → V4F shapes give ~8k workgroups,
//              way below the latency-hiding budget (160 SIMDs × 16 waves
//              = 2560 wave slots, ideally need 25k+ in flight).
//   per-out  : (M, B)          → V4F shapes (M=1024, B=64) give 65k
//              workgroups — well into the saturation regime.
//
// Cost: weight A[m, k] is read once per output element instead of once
// per BATCH_TILE outputs ⇒ ~8× weight bandwidth on paper. But the
// kernel is so close to peak that on-chip caches absorb the repeated
// reads (M consecutive workgroups read the same A row).
//
// Layout:
//   A : [M, K]               row-major F32
//   x : [batch_size, K]      row-major F32
//   y : [batch_size, M]      row-major F32
// Grid:  [M, batch_size, 1]    Block: [32, 1, 1]

#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(32, 20)
__global__ void gemm_f32_per_output(
    const float* __restrict__ A,    // [M, K]
    const float* __restrict__ x,    // [batch_size, K]
    float* __restrict__ y,          // [batch_size, M]
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    const int b = blockIdx.y;
    if (row >= M || b >= batch_size) return;
    const int tid = threadIdx.x;

    const float* a_row = A + (size_t)row * K;
    const float* x_row = x + (size_t)b * K;

    // K4 unroll: 4 independent accumulators per thread for ILP.
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int stride = 32;
    const int k_quads = K / (stride * 4);    // groups of 4 stride-32 chunks

    for (int q = 0; q < k_quads; q++) {
        const int k0 = q * stride * 4 + tid;
        acc0 += a_row[k0]                  * x_row[k0];
        acc1 += a_row[k0 + stride]         * x_row[k0 + stride];
        acc2 += a_row[k0 + 2 * stride]     * x_row[k0 + 2 * stride];
        acc3 += a_row[k0 + 3 * stride]     * x_row[k0 + 3 * stride];
    }
    // Tail.
    const int tail_start = k_quads * stride * 4;
    for (int k = tail_start + tid; k < K; k += stride) {
        acc0 += a_row[k] * x_row[k];
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }
    if (tid == 0) {
        y[(size_t)b * M + row] = acc;
    }
}
