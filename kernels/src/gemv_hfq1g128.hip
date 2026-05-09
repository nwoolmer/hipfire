#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HFQ1G128 GEMV (PrismML Q1_0 1-bit weights, FP16 scale, group=128).
// Mirrors the gemv_hfq4g128 wave32 template with 1-bit unpack:
//   - Block: 18 B = [FP16 d][16 B packed bits, LSB-first]
//   - Codebook: bit=1 → +d, bit=0 → -d (no zero-point, no offset)
//   - 128 weights / 32 lanes = 4 weights/lane (4 bits = half-byte/lane)
//
// Phase 1: FP-direct path (FP32 activations, dequant inline). dp4a-vs-Q8_1
// path lands in Phase 2 as a perf optimization. See plans/hfq1g128-bonsai.md
// §3.2 and findings/prismml-q1_0-layout.md for full math context.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq1g128(
    const char* __restrict__ A,    // [M, K] HFQ1G128 packed: 18 B per group
    const float* __restrict__ x,   // [K] FP32 activation row
    float* __restrict__ y,         // [M] FP32 output
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 18;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 18;

        // FP16 scale at offset 0..1 (little-endian, 2-byte aligned by row stride).
        _Float16 d_h = *(const _Float16*)gptr;
        float d = (float)d_h;
        float neg_d = -d;

        // Each lane handles weights [tid*4 .. tid*4+3]:
        //   byte_off  = (tid*4) >> 3 = tid >> 1   (2 lanes share one byte)
        //   bit_shift = (tid*4) & 7 = (tid & 1) << 2   (lower nibble vs upper)
        int base_idx = g * 128 + tid * 4;
        int byte_off = tid >> 1;
        int bit_shift = (tid & 1) << 2;
        unsigned char byte = (unsigned char)gptr[2 + byte_off];
        unsigned char nib = (byte >> bit_shift) & 0xF;

        // 4 weight signs from the nibble's 4 bits, LSB-first within nibble.
        float w0 = (nib & 0x1) ? d : neg_d;
        float w1 = (nib & 0x2) ? d : neg_d;
        float w2 = (nib & 0x4) ? d : neg_d;
        float w3 = (nib & 0x8) ? d : neg_d;

        acc += w0 * x[base_idx]
             + w1 * x[base_idx + 1]
             + w2 * x[base_idx + 2]
             + w3 * x[base_idx + 3];
    }

    // Tree reduction across the wave32.
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
