#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Index-aware fused MoE down GEMV with scaled residual for MQ2-Lloyd
// (2-bit + per-block 4-entry fp16 codebook, 72 B/group). Sibling of
// gemv_hfq4g256_moe_down_residual_scaled_k8_indexed — same expert-ptrs
// dispatch + atomicAdd residual, but the per-group decode is the 4-entry
// codebook lookup.
//
// `rot_batch[krank * K .. krank * K + K]` must be FWHT-pre-rotated for the
// down projection (each top-k expert has its own post-gate-up activation;
// each needs an independent rotation before this kernel sees it).
//
// atomicAdd into x_residual accumulates topk_weights[krank] * (W_down · rot_batch[krank])
// across all top-k experts and rows. One launch replaces 8 per-expert
// scaled-gemv-residual launches.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int*   __restrict__ topk_indices,              // [k_top]
    const float* __restrict__ topk_weights,              // [k_top]
    const float* __restrict__ rot_batch,                 // [k_top × K] FWHT-rotated
    float*       __restrict__ x_residual,                // [M]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x = rot_batch + (long long)krank * K;

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

        acc += CB_LOOKUP(b0 & 3)        * x[base_idx]
             + CB_LOOKUP((b0 >> 2) & 3) * x[base_idx + 1]
             + CB_LOOKUP((b0 >> 4) & 3) * x[base_idx + 2]
             + CB_LOOKUP(b0 >> 6)       * x[base_idx + 3]
             + CB_LOOKUP(b1 & 3)        * x[base_idx + 4]
             + CB_LOOKUP((b1 >> 2) & 3) * x[base_idx + 5]
             + CB_LOOKUP((b1 >> 4) & 3) * x[base_idx + 6]
             + CB_LOOKUP(b1 >> 6)       * x[base_idx + 7];

        #undef CB_LOOKUP
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const float scale = topk_weights[krank];
        atomicAdd(&x_residual[row], scale * acc);
    }
}
