#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HFQ1G128 multi-row GEMV: one wave32 computes R output rows.
// Mirrors the gemv_hfq4g256_multirow pattern, adapted for 1-bit weights:
//   - Block = 18 B per 128-elt group (FP16 d + 16 B packed bits, LSB-first)
//   - 128 weights / 32 lanes = 4 weights/lane (4 bits = half-byte/lane)
//
// Hoists the x-load (4 floats per lane per K-step) once and reuses across
// all R rows, halving (or quartering) activation traffic per output row.
//
// Memory bookkeeping (per K-step, per wave):
//   - Activations: 4 floats/lane × 32 lanes = 128 FP32 = 512 B (cached)
//   - Weights:     R rows × 18 B/group = 18·R B/group from DRAM
//
// Bandwidth gain at large R is a wash (each row's weights still come from
// DRAM); the win is the per-launch overhead reduction + better wave
// scheduling. R=2 keeps register pressure low; R=4 starts to drop occupancy.

template<int R>
__device__ __forceinline__ void gemv_hfq1g128_multirow_body(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row_base = blockIdx.x * R;
    if (row_base >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 128;
    const long long row_stride = (long long)groups_per_row * 18;

    // Row pointers — clamp out-of-range rows to row_base so reads stay
    // in-bounds; the write is masked separately.
    const char* row_ptrs[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        int row = row_base + r;
        int safe_row = (row < M) ? row : row_base;
        row_ptrs[r] = A + (long long)safe_row * row_stride;
    }

    float acc[R];
    #pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.0f;

    const int byte_off = tid >> 1;          // 2 lanes share a byte
    const int bit_shift = (tid & 1) << 2;   // lower vs upper nibble

    for (int g = 0; g < groups_per_row; g++) {
        // Hoist x: 4 floats per lane (128 floats per wave) — reused across R rows.
        const int base_idx = g * 128 + tid * 4;
        const float xv0 = x[base_idx];
        const float xv1 = x[base_idx + 1];
        const float xv2 = x[base_idx + 2];
        const float xv3 = x[base_idx + 3];

        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gptr = row_ptrs[r] + g * 18;
            _Float16 d_h = *(const _Float16*)gptr;
            float d = (float)d_h;
            float neg_d = -d;
            unsigned char byte_val = (unsigned char)gptr[2 + byte_off];
            unsigned char nib = (byte_val >> bit_shift) & 0xF;
            float w0 = (nib & 0x1) ? d : neg_d;
            float w1 = (nib & 0x2) ? d : neg_d;
            float w2 = (nib & 0x4) ? d : neg_d;
            float w3 = (nib & 0x8) ? d : neg_d;
            acc[r] += w0 * xv0 + w1 * xv1 + w2 * xv2 + w3 * xv3;
        }
    }

    // Per-row warp reduction and write-back.
    #pragma unroll
    for (int r = 0; r < R; r++) {
        float s = acc[r];
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down(s, offset);
        int row = row_base + r;
        if (tid == 0 && row < M) y[row] = s;
    }
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq1g128_multirow_r2(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_hfq1g128_multirow_body<2>(A, x, y, M, K);
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq1g128_multirow_r4(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_hfq1g128_multirow_body<4>(A, x, y, M, K);
}

__launch_bounds__(32, 12)
extern "C" __global__ void gemv_hfq1g128_multirow_r8(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_hfq1g128_multirow_body<8>(A, x, y, M, K);
}
