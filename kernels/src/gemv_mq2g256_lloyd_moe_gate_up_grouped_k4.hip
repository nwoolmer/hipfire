#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Scatter-by-expert K4-unrolled MQ2-Lloyd MoE gate_up GEMV.
//
// Same compute contract as the _indexed_batched_k4 variant but the
// (b, krank, expert_id) for each routing is read from a pre-sorted
// permutation array. When the host (or a prior GPU sort kernel)
// sorts routings by expert_id, adjacent blockIdx.y values share the
// same expert pointer → the SAME 4.5 MB weight slab is read by all
// 2-3 routings that picked that expert (typical V4F density), with
// L2/IC cache absorbing the repeated reads.
//
// Why this matters on bandwidth-bound GPUs (gfx1151 ≈ 256 GB/s):
//   - The original _indexed_batched_k4 kernel had grid =
//     (M, K_TOP, B), so adjacent blocks in (krank, b) read DIFFERENT
//     experts. Measured BW at B=32 chunk: ~3.7 GB/s = 1.4 % of peak.
//   - With this grouped pattern, all blocks for a given expert run
//     close together in execution time → effective BW jumps because
//     the weight slab stays in L2/IC across thread blocks of a given
//     expert routing run.
//
// Shapes:
//   expert_ptrs : [n_exp]                  device-pointer table
//   sorted_b    : [N = B * K_TOP]          permuted batch indices
//   sorted_krank: [N]                      permuted top-K rank indices
//   sorted_expert:[N]                      direct expert id (no extra indirection)
//   x_rot       : [B, K]                   FWHT-rotated input
//   y_gate      : [B, K_TOP, MI]           MI = M / 2
//   y_up        : [B, K_TOP, MI]
//
// Grid : (M, N).      Block : 32.
// Same FMA ordering as _indexed_batched_k4 → output drifts within
// FMA-order epsilon from the single-acc variant.

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_mq2g256_lloyd_moe_gate_up_k8_grouped_k4(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*   __restrict__ sorted_b,
    const int*   __restrict__ sorted_krank,
    const int*   __restrict__ sorted_expert,
    const float* __restrict__ x_rot,
    float*       __restrict__ y_gate,
    float*       __restrict__ y_up,
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
    const float* __restrict__ x_row = x_rot + (long long)b * K;

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
        acc += cb[b0 & 3]        * x_row[base]
            +  cb[(b0 >> 2) & 3] * x_row[base + 1]
            +  cb[(b0 >> 4) & 3] * x_row[base + 2]
            +  cb[b0 >> 6]       * x_row[base + 3]
            +  cb[b1 & 3]        * x_row[base + 4]
            +  cb[(b1 >> 2) & 3] * x_row[base + 5]
            +  cb[(b1 >> 4) & 3] * x_row[base + 6]
            +  cb[b1 >> 6]       * x_row[base + 7];
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
        const int mi = M >> 1;
        const long long stride = (long long)K_TOP * mi;
        if (row < mi) {
            y_gate[(long long)b * stride + krank * mi + row] = acc;
        } else {
            y_up[(long long)b * stride + krank * mi + (row - mi)] = acc;
        }
    }
}
