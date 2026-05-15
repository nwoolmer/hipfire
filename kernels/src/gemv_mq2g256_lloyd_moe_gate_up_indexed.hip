#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Index-aware fused MoE gate_up GEMV for MQ2-Lloyd (2-bit + per-block 4-entry
// fp16 codebook, 72 B/group). Sibling of gemv_hfq4g256_moe_gate_up_k8_indexed —
// same grid + expert-ptrs dispatch, but the per-group decode is a 4-entry
// codebook lookup instead of HFQ4's affine sc*q+zp reconstruction.
//
// X must be FWHT-pre-rotated by the caller (MQ-family kernels rotate X
// once per token, then re-use the rotated vector across all top-k experts).
//
// Block layout (72 B / 256 weights):
//   [0..8)   : 4 × fp16 codebook entries (sorted ascending)
//   [8..72)  : 64 bytes packed 2-bit indices (4/byte, little-endian)
//
// Grid: [M, k_top, 1], block [32]. Output split: row<M/2 → y_gate, ≥M/2 → y_up.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256_lloyd_moe_gate_up_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [k_top]
    const float* __restrict__ x_rot,                     // [K] (FWHT-rotated)
    float* __restrict__ y_gate,                          // [k_top × mi]
    float* __restrict__ y_up,                            // [k_top × mi]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 72;

        const __half* cb_h = reinterpret_cast<const __half*>(gptr);
        float cb0 = __half2float(cb_h[0]);
        float cb1 = __half2float(cb_h[1]);
        float cb2 = __half2float(cb_h[2]);
        float cb3 = __half2float(cb_h[3]);

        const unsigned char* data = (const unsigned char*)(gptr + 8);
        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 2;

        unsigned char b0 = data[byte_off];
        unsigned char b1 = data[byte_off + 1];

        #define CB_LOOKUP(q) ((q) == 0 ? cb0 : ((q) == 1 ? cb1 : ((q) == 2 ? cb2 : cb3)))

        acc += CB_LOOKUP(b0 & 3)        * x_rot[base_idx]
             + CB_LOOKUP((b0 >> 2) & 3) * x_rot[base_idx + 1]
             + CB_LOOKUP((b0 >> 4) & 3) * x_rot[base_idx + 2]
             + CB_LOOKUP(b0 >> 6)       * x_rot[base_idx + 3]
             + CB_LOOKUP(b1 & 3)        * x_rot[base_idx + 4]
             + CB_LOOKUP((b1 >> 2) & 3) * x_rot[base_idx + 5]
             + CB_LOOKUP((b1 >> 4) & 3) * x_rot[base_idx + 6]
             + CB_LOOKUP(b1 >> 6)       * x_rot[base_idx + 7];

        #undef CB_LOOKUP
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const int mi = M >> 1;
        if (row < mi) {
            y_gate[krank * mi + row] = acc;
        } else {
            y_up[krank * mi + (row - mi)] = acc;
        }
    }
}
