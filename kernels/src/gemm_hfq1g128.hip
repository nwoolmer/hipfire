#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HFQ1G128 batched GEMM. Same wave32 FP-direct shape as gemm_hfq4g128
// but with 1-bit unpack. Each wave32 block computes one output row × all
// BATCH_TILE tokens; the weight scale + bits are loaded once per K-step
// and reused across the batch loop.
//
//   y[b, row] = sum_k A[row, k] * x[b, k]   for b ∈ [0, batch_size)
//
// Grid: [M, ceil(N / BATCH_TILE)]. Block: [32]. LDS: 0.
//
// Replaces the GEMV-per-token fallback used by `weight_gemm` for HFQ1G128
// prefill — single launch instead of N GEMV launches.

#define BATCH_TILE 8

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_hfq1g128(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int batch_start = blockIdx.y * BATCH_TILE;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 18;
    const char* row_ptr = A + (long long)row * row_bytes;

    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    float acc[BATCH_TILE];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) acc[b] = 0.0f;

    const int byte_off = tid >> 1;
    const int bit_shift = (tid & 1) << 2;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 18;
        _Float16 d_h = *(const _Float16*)gptr;
        float d = (float)d_h;
        float neg_d = -d;
        unsigned char byte_val = (unsigned char)gptr[2 + byte_off];
        unsigned char nib = (byte_val >> bit_shift) & 0xF;

        // Pre-dequantize 4 weight signs once per K-step; reused across batch.
        float w0 = (nib & 0x1) ? d : neg_d;
        float w1 = (nib & 0x2) ? d : neg_d;
        float w2 = (nib & 0x4) ? d : neg_d;
        float w3 = (nib & 0x8) ? d : neg_d;

        const int base_k = g * 128 + tid * 4;

        for (int b = 0; b < local_bs; b++) {
            const float* xb = x + (batch_start + b) * K;
            acc[b] += w0 * xb[base_k]
                    + w1 * xb[base_k + 1]
                    + w2 * xb[base_k + 2]
                    + w3 * xb[base_k + 3];
        }
    }

    for (int b = 0; b < local_bs; b++) {
        float val = acc[b];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
        if (tid == 0) y[(long long)(batch_start + b) * M + row] = val;
    }
}
