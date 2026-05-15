#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Index-aware fused MoE gate_up GEMV for MQ3-Lloyd (3-bit + per-block
// 8-entry fp16 codebook, 112 B/group). Sibling of
// gemv_mq2g256_lloyd_moe_gate_up_k8_indexed — same grid + expert-ptrs
// dispatch, but the per-group decode uses 8 centroids and the
// uniform-MQ3 cross-byte 3-bit packing (24 bits = 3 bytes per 8 weights).
//
// X must be FWHT-pre-rotated by the caller (MQ-family kernels rotate X
// once per token, then re-use the rotated vector across all top-k experts).
//
// Block layout (112 B / 256 weights):
//   [0..16)  : 8 × fp16 codebook entries (sorted ascending)
//   [16..112): 96 bytes packed 3-bit indices (cross-byte, matches uniform MQ3)
//
// Grid: [M, k_top, 1], block [32]. Output split: row<M/2 → y_gate, ≥M/2 → y_up.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq3g256_lloyd_moe_gate_up_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,                // [k_top]
    const float* __restrict__ x_rot,                     // [K] FWHT-rotated
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
    const int row_bytes = groups_per_row * 112;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = tid * 3;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 112;

        const __half* cb_h = reinterpret_cast<const __half*>(gptr);
        float cb0 = __half2float(cb_h[0]);
        float cb1 = __half2float(cb_h[1]);
        float cb2 = __half2float(cb_h[2]);
        float cb3 = __half2float(cb_h[3]);
        float cb4 = __half2float(cb_h[4]);
        float cb5 = __half2float(cb_h[5]);
        float cb6 = __half2float(cb_h[6]);
        float cb7 = __half2float(cb_h[7]);

        const unsigned char* data = (const unsigned char*)(gptr + 16);
        unsigned int pk = (unsigned int)data[boff]
                        | ((unsigned int)data[boff + 1] << 8)
                        | ((unsigned int)data[boff + 2] << 16);

        unsigned int q0 = pk & 7u;
        unsigned int q1 = (pk >> 3) & 7u;
        unsigned int q2 = (pk >> 6) & 7u;
        unsigned int q3 = (pk >> 9) & 7u;
        unsigned int q4 = (pk >> 12) & 7u;
        unsigned int q5 = (pk >> 15) & 7u;
        unsigned int q6 = (pk >> 18) & 7u;
        unsigned int q7 = (pk >> 21) & 7u;

        int base_idx = g * 256 + tid * 8;

        #define CB_LOOKUP3(q) (                                          \
            (q) == 0 ? cb0 : (q) == 1 ? cb1 : (q) == 2 ? cb2 : (q) == 3  \
            ? cb3 : (q) == 4 ? cb4 : (q) == 5 ? cb5 : (q) == 6 ? cb6     \
            : cb7                                                         \
        )

        acc += CB_LOOKUP3(q0) * x_rot[base_idx]
             + CB_LOOKUP3(q1) * x_rot[base_idx + 1]
             + CB_LOOKUP3(q2) * x_rot[base_idx + 2]
             + CB_LOOKUP3(q3) * x_rot[base_idx + 3]
             + CB_LOOKUP3(q4) * x_rot[base_idx + 4]
             + CB_LOOKUP3(q5) * x_rot[base_idx + 5]
             + CB_LOOKUP3(q6) * x_rot[base_idx + 6]
             + CB_LOOKUP3(q7) * x_rot[base_idx + 7];

        #undef CB_LOOKUP3
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
