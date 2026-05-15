#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Batched N-token indexed MoE down + scaled residual for MQ2-Lloyd.
// Sibling of gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched —
// extends the single-token MQ2-Lloyd MoE down kernel with a batch dimension.
//
// Shapes:
//   expert_ptrs : [n_exp]
//   topk_indices: [N × K_TOP]
//   topk_weights: [N × K_TOP]
//   rot_batch   : [N × K_TOP × K]            (FWHT-rotated per (bid, krank))
//   x_residual  : [N × M]                    accumulated in-place via atomicAdd
//
// Grid: (M, K_TOP, N). Block = 32 threads. atomicAdd across K_TOP for a
// given (bid, row) keeps writes safe.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,              // [N × K_TOP]
    const float* __restrict__ topk_weights,              // [N × K_TOP]
    const float* __restrict__ rot_batch,                 // [N × K_TOP × K]
    float*       __restrict__ x_residual,                // [N × M]
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int routing_base = bid * K_TOP;
    const int expert_id = topk_indices[routing_base + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;

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
        const float scale = topk_weights[routing_base + krank];
        atomicAdd(&x_residual[(long long)bid * M + row], scale * acc);
    }
}
