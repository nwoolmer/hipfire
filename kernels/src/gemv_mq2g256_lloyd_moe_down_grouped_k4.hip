#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Scatter-by-expert K4-unrolled MoE down GEMV + scaled residual atomicAdd
// for MQ2-Lloyd. Sibling of gate_up _grouped_k4. See that kernel for the
// bandwidth-utilization rationale (eliminates cache thrashing from
// indexed gather).
//
// Shapes:
//   expert_ptrs : [n_exp]
//   sorted_b    : [N = B * K_TOP]
//   sorted_krank: [N]
//   sorted_expert:[N]
//   topk_weights: [B, K_TOP]                indexed by (b, krank)
//   rot_batch   : [B, K_TOP, K]             indexed by (b, krank)
//   x_residual  : [B, M]                    atomicAdd-accumulated across K_TOP
//
// Grid: (M, N). Block = 32.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_grouped_k4(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*   __restrict__ sorted_b,
    const int*   __restrict__ sorted_krank,
    const int*   __restrict__ sorted_expert,
    const float* __restrict__ topk_weights,
    const float* __restrict__ rot_batch,
    float*       __restrict__ x_residual,
    int M, int K, int K_TOP, int N
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int routing_idx = blockIdx.y;
    if (routing_idx >= N) return;
    const int tid = threadIdx.x;

    const int b = sorted_b[routing_idx];
    const int krank = sorted_krank[routing_idx];
    const int expert_id = sorted_expert[routing_idx];

    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)b * K_TOP + krank) * K;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = tid * 2;
    const int xoff = tid * 8;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    auto group_fma = [&] (const char* gptr, int base, float& acc) {
        const __half* cb_h = reinterpret_cast<const __half*>(gptr);
        float cb[4] = {
            __half2float(cb_h[0]),
            __half2float(cb_h[1]),
            __half2float(cb_h[2]),
            __half2float(cb_h[3]),
        };
        const unsigned char* data = (const unsigned char*)(gptr + 8);
        unsigned char b0 = data[boff];
        unsigned char b1 = data[boff + 1];
        acc += cb[b0 & 3]        * x[base]
            +  cb[(b0 >> 2) & 3] * x[base + 1]
            +  cb[(b0 >> 4) & 3] * x[base + 2]
            +  cb[b0 >> 6]       * x[base + 3]
            +  cb[b1 & 3]        * x[base + 4]
            +  cb[(b1 >> 2) & 3] * x[base + 5]
            +  cb[(b1 >> 4) & 3] * x[base + 6]
            +  cb[b1 >> 6]       * x[base + 7];
    };

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 72;
        const char* gp1 = gp0 + 72;
        const char* gp2 = gp1 + 72;
        const char* gp3 = gp2 + 72;
        const int base = g * 256 + xoff;
        group_fma(gp0, base,         acc0);
        group_fma(gp1, base + 256,   acc1);
        group_fma(gp2, base + 512,   acc2);
        group_fma(gp3, base + 768,   acc3);
    }
    const int tail_start = quads << 2;
    for (int g = tail_start; g < tail_start + tail; g++) {
        const char* gp = row_ptr + g * 72;
        const int base = g * 256 + xoff;
        group_fma(gp, base, acc0);
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const float scale = topk_weights[(long long)b * K_TOP + krank];
        atomicAdd(&x_residual[(long long)b * M + row], scale * acc);
    }
}
