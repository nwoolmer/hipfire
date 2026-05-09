#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Dequant HFQ1G128 packed weights to row-major FP16.
// Input:  A_packed [M × groups_per_row × 18 bytes] (1-bit packed, FP16 scale)
// Output: W_f16 [M × K] FP16 (row-major)
//
// Grid: [M, groups_per_row]. Block: [32].
// Each block dequants one 128-element group for one row.
// Thread t dequants 4 elements (tid*4..tid*4+3) — 4 bits = half a byte/lane.

extern "C" __launch_bounds__(32, 16)
__global__ void dequant_hfq1g128_to_f16(
    const char* __restrict__ A_packed,
    _Float16* __restrict__ W_f16,
    int M, int K
) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 128;

    const char* gp = A_packed + (long long)row * groups_per_row * 18 + group * 18;
    _Float16 d_h = *(const _Float16*)gp;
    _Float16 neg_d_h = -d_h;

    int byte_off = tid >> 1;          // 2 lanes share a byte
    int bit_shift = (tid & 1) << 2;   // lower vs upper nibble
    unsigned char byte_val = (unsigned char)gp[2 + byte_off];
    unsigned char nib = (byte_val >> bit_shift) & 0xF;

    _Float16* out = W_f16 + (long long)row * K + group * 128 + tid * 4;
    out[0] = (nib & 0x1) ? d_h : neg_d_h;
    out[1] = (nib & 0x2) ? d_h : neg_d_h;
    out[2] = (nib & 0x4) ? d_h : neg_d_h;
    out[3] = (nib & 0x8) ? d_h : neg_d_h;
}
