// V4F MoE down — Option A "single-col WMMA" — MQ2-Lloyd weights.
//
// Counterpart of gemm_mq2g256_lloyd_moe_gate_up_wmma for the down
// projection: atomically adds the scaled (topk_weights × W·rot) result
// into x_residual.
//
// Shape contract (matches gemv_mq2g256_lloyd_moe_down_residual_scaled_
// k8_indexed_batched_k4):
//   expert_ptrs : [n_exp]
//   topk_indices: [B × K_TOP]
//   topk_weights: [B × K_TOP]
//   rot_batch   : [B × K_TOP × K]  F16 (caller stages F32→F16)
//   x_residual  : [B × M]          F32, atomically accumulated
//
// Grid:  [ceil(M / 16), K_TOP, B]
// Block: [32]
// LDS:   128 B (4-entry F16 codebook × 16 tile-rows).
//
// Same single-col WMMA pattern as gate_up: each block produces 16
// M-rows × 1 batch position via broadcast B fragment, only col-0
// carries useful output.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_mq2g256_lloyd_moe_down_residual_scaled_wmma(
    const unsigned long long* __restrict__ expert_ptrs,  // [n_exp]
    const int*      __restrict__ topk_indices,            // [B × K_TOP]
    const float*    __restrict__ topk_weights,            // [B × K_TOP]
    const _Float16* __restrict__ rot_batch_f16,           // [B × K_TOP × K] F16
    float*          __restrict__ x_residual,              // [B × M] F32
    int M, int K, int K_TOP
) {
    const int row_start = blockIdx.x * 16;
    const int krank     = blockIdx.y;
    const int bid       = blockIdx.z;
    const int tid       = threadIdx.x;
    const int lane      = tid & 15;

    if (row_start >= M) return;

    const int routing_base = bid * K_TOP;
    const int expert_id = topk_indices[routing_base + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    const int my_tile_row = lane;
    const int my_m        = row_start + my_tile_row;
    const int safe_m      = (my_m < M) ? my_m : (M - 1);

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 72;
    const char* row_base = A + (long long)safe_m * row_stride;

    const int load_tile_row = tid >> 1;
    const int load_lo       = (tid & 1) * 2;
    const int load_my_row   = row_start + load_tile_row;
    const int load_safe_row = (load_my_row < M) ? load_my_row : (M - 1);
    const char* load_row_base = A + (long long)load_safe_row * row_stride;

    // Input is rot_batch[bid, krank, :] — distinct K-slab per (bid, krank).
    const _Float16* x_row =
        rot_batch_f16 + ((long long)bid * K_TOP + krank) * K;

    __shared__ _Float16 cb_lds[16 * 4];

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int g = 0; g < groups_per_row; g++) {
        const __half* cb_h_load = (const __half*)(load_row_base + g * 72);
        cb_lds[load_tile_row * 4 + load_lo + 0] = cb_h_load[load_lo + 0];
        cb_lds[load_tile_row * 4 + load_lo + 1] = cb_h_load[load_lo + 1];
        __syncthreads();

        const unsigned char* data = (const unsigned char*)(row_base + g * 72 + 8);
        const int cb_base = my_tile_row * 4;

        #pragma unroll 1
        for (int kt = 0; kt < 16; kt++) {
            const unsigned char* d = data + kt * 4;
            unsigned int b0 = d[0], b1 = d[1], b2 = d[2], b3 = d[3];

            half16_t a_frag;
            #define DQ(i, q) a_frag[i] = cb_lds[cb_base + (q)]
            DQ( 0, b0       & 3); DQ( 1, (b0 >> 2) & 3);
            DQ( 2, (b0 >> 4) & 3); DQ( 3, (b0 >> 6)    );
            DQ( 4, b1       & 3); DQ( 5, (b1 >> 2) & 3);
            DQ( 6, (b1 >> 4) & 3); DQ( 7, (b1 >> 6)    );
            DQ( 8, b2       & 3); DQ( 9, (b2 >> 2) & 3);
            DQ(10, (b2 >> 4) & 3); DQ(11, (b2 >> 6)    );
            DQ(12, b3       & 3); DQ(13, (b3 >> 2) & 3);
            DQ(14, (b3 >> 4) & 3); DQ(15, (b3 >> 6)    );
            #undef DQ

            const int k = g * 256 + kt * 16;
            half16_t x_frag = *(const half16_t*)(x_row + k);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, x_frag, acc);
        }
        __syncthreads();
    }

    // Output: atomicAdd `scale * acc[j]` into x_residual[bid * M + out_row].
    // Same lane-0-only filter as gate_up. scale = topk_weights[bid, krank].
    if (lane != 0) return;
    const float scale = topk_weights[routing_base + krank];
    const long long bid_off = (long long)bid * M;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int out_row = row_start + 2 * j + (tid >> 4);
        if (out_row < M) {
            atomicAdd(&x_residual[bid_off + out_row], scale * acc[j]);
        }
    }
}
