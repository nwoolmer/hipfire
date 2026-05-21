// V4F MoE gate_up — Option A "single-col WMMA" — MQ2-Lloyd weights.
//
// Each block handles 16 M-rows × 1 batch position. WMMA still issues full
// 16×16×16 mma per K=16 step, but the B fragment broadcasts ONE X col
// across 16 mma-rows; only col-0 (via lane 0 in the wave32 layout) carries
// a useful output. Hardware is 1/16 utilized for the WMMA output, BUT
// throughput per useful output is still ~2× scalar K4 because the K-loop
// uses one mma per K=16 step (vs 16 FMAs per group).
//
// This design avoids the WMMA tile-homogeneity constraint that breaks for
// MoE (each batch position routes to a different expert). Option B (sort
// + full-tile WMMA) is a separate kernel; this is the safer first cut.
//
// Per-group MQ2-Lloyd format (72 B per 256 weights):
//   bytes 0..7  : 4 × F16 codebook entries
//   bytes 8..71 : 64 B of 2-bit indices (4 per byte, little-endian)
//
// Shape contract (matches gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4):
//   expert_ptrs : [n_exp]   device-pointer table per expert
//   topk_indices: [B × K_TOP]
//   x_rot       : [B × K]   F16 row-major (caller stages F32→F16)
//   y_gate      : [B × K_TOP × MI]   F32
//   y_up        : [B × K_TOP × MI]   F32
//   MI = M / 2 (gate = rows 0..MI-1, up = rows MI..2*MI-1)
//
// Grid:  [ceil(M / 16), K_TOP, B]
// Block: [32]
// LDS:   16 * 4 * 2 = 128 B (16 tile-rows × 4 F16 codebook entries)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_mq2g256_lloyd_moe_gate_up_wmma(
    const unsigned long long* __restrict__ expert_ptrs,  // [n_exp]
    const int*      __restrict__ topk_indices,            // [B × K_TOP]
    const _Float16* __restrict__ x_rot_f16,               // [B × K] F16
    float*          __restrict__ y_gate,                  // [B × K_TOP × MI]
    float*          __restrict__ y_up,                    // [B × K_TOP × MI]
    int M, int K, int K_TOP
) {
    const int row_start = blockIdx.x * 16;
    const int krank     = blockIdx.y;
    const int bid       = blockIdx.z;
    const int tid       = threadIdx.x;
    const int lane      = tid & 15;

    if (row_start >= M) return;

    // Expert routing — per-(b, krank).
    const int expert_id = topk_indices[bid * K_TOP + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    const int my_tile_row = lane;
    const int my_m        = row_start + my_tile_row;
    const int safe_m      = (my_m < M) ? my_m : (M - 1);

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 72;
    const char* row_base = A + (long long)safe_m * row_stride;

    // Cooperative codebook load: 32 threads × 2 entries = 64 entries (16 rows × 4).
    const int load_tile_row = tid >> 1;
    const int load_lo       = (tid & 1) * 2;
    const int load_my_row   = row_start + load_tile_row;
    const int load_safe_row = (load_my_row < M) ? load_my_row : (M - 1);
    const char* load_row_base = A + (long long)load_safe_row * row_stride;

    // X is shared across all 16 batch positions of the WMMA B fragment
    // (which we broadcast). The block handles ONE batch position, so all
    // 16 B-fragment rows read x_rot_f16[bid, :] — identical content.
    const _Float16* x_row = x_rot_f16 + (long long)bid * K;

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
            // Broadcast: every B-fragment row gets the same X[k..k+16].
            half16_t x_frag = *(const half16_t*)(x_row + k);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, x_frag, acc);
        }
        __syncthreads();
    }

    // Output write: only col-0 of the 16×16 WMMA output carries useful data
    // (all B-rows of the input fragment were identical). Per the wave32
    // WMMA layout, col=lane and row=2*j+(tid>>4) means:
    //   threads 0  and 16 hold col-0 outputs at rows m_tile + 2*j + (tid>>4)
    //   threads 1..15 and 17..31 hold col-1..col-15 outputs (garbage — skip).
    //
    // Threads 0 (low half) and 16 (high half) write the same lane=0 col,
    // BUT cover different row sets (different tid>>4 values). Both write.
    if (lane != 0) return;
    const int mi = M >> 1;
    const long long stride = (long long)K_TOP * mi;
    const long long bid_off = (long long)bid * stride + (long long)krank * mi;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int out_row = row_start + 2 * j + (tid >> 4);
        if (out_row < M) {
            if (out_row < mi) {
                y_gate[bid_off + out_row] = acc[j];
            } else {
                y_up[bid_off + (out_row - mi)] = acc[j];
            }
        }
    }
}
