#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Batched N-token indexed MoE gate_up GEMV for MQ2-Lloyd. Sibling of
// gemv_hfq4g256_moe_gate_up_k8_indexed_batched — extends the single-token
// MQ2-Lloyd MoE kernel with a batch dimension.
//
// Shapes:
//   expert_ptrs : [n_exp] device pointer table, shared across tokens
//   topk_indices: [N × K_TOP]
//   x           : [N × K]                  (FWHT-rotated post-RMSNorm)
//   y_gate      : [N × K_TOP × MI]         (MI = M / 2)
//   y_up        : [N × K_TOP × MI]
//
// Grid: (M, K_TOP, N). Each block reduces one output row for one expert
// for one token. Block = 32 threads.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,                // [N × K_TOP]
    const float* __restrict__ x_rot,                     // [N × K] FWHT-rotated
    float* __restrict__ y_gate,                          // [N × K_TOP × MI]
    float* __restrict__ y_up,                            // [N × K_TOP × MI]
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x_row = x_rot + (long long)bid * K;

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

        acc += CB_LOOKUP(b0 & 3)        * x_row[base_idx]
             + CB_LOOKUP((b0 >> 2) & 3) * x_row[base_idx + 1]
             + CB_LOOKUP((b0 >> 4) & 3) * x_row[base_idx + 2]
             + CB_LOOKUP(b0 >> 6)       * x_row[base_idx + 3]
             + CB_LOOKUP(b1 & 3)        * x_row[base_idx + 4]
             + CB_LOOKUP((b1 >> 2) & 3) * x_row[base_idx + 5]
             + CB_LOOKUP((b1 >> 4) & 3) * x_row[base_idx + 6]
             + CB_LOOKUP(b1 >> 6)       * x_row[base_idx + 7];

        #undef CB_LOOKUP
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const int mi = M >> 1;
        const long long stride = (long long)K_TOP * mi;
        if (row < mi) {
            y_gate[(long long)bid * stride + krank * mi + row] = acc;
        } else {
            y_up[(long long)bid * stride + krank * mi + (row - mi)] = acc;
        }
    }
}
