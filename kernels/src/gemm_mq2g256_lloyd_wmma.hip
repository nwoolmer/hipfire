// WMMA-accelerated GEMM for MQ2-Lloyd-G256 weights + F16 input → F32 output.
// Smoke / proof-of-concept for the MQ2 WMMA path (Lever 4 dominant kernel
// is MoE-routed MQ2-Lloyd, currently 54.1% of V4F prefill GPU time at
// post-Option-B).
//
// Strict counterpart of:
//   - `gemm_hfq4g256_wmma.hip` (4-bit affine dequant), and
//   - `gemm_gate_up_mq3g256_lloyd_wmma.hip` (3-bit codebook dequant).
//
// MQ2-Lloyd-G256 per-group layout (72 B per 256 weights):
//   bytes 0..7  : 4 × F16 codebook entries  (sorted ascending at quant time)
//   bytes 8..71 : 64 bytes packed 2-bit indices (4 indices per byte,
//                 little-endian within byte)
//
// Dequant: w = cb[idx] where idx ∈ {0, 1, 2, 3}. NOT scale*idx + zero.
//
// gfx1100+ only (RDNA3 / RDNA3.5 wave32 WMMA, 16×16×16 tiles).
//
// Layout:
//   A : [M, K/256 * 72] bytes (MQ2-Lloyd-packed)
//   X : [B, K] F16 row-major (caller stages F32→F16 upstream)
//   Y : [B, M] F32 row-major
//
// Grid:  [ceil(M / 16), ceil(B / 16)]
// Block: [32]
// LDS:   16 * 4 * 2 = 128 bytes (16 tile-rows × 4 F16 codebook entries)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_mq2g256_lloyd_wmma(
    const char*     __restrict__ A,    // MQ2-Lloyd-G256 packed
    const _Float16* __restrict__ X,    // [B, K] F16
    float*          __restrict__ Y,    // [B, M] F32
    int M, int K, int B
) {
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid  = threadIdx.x;
    const int lane = tid & 15;

    if (row_start >= M || batch_start >= B) return;

    // Compute-side row/batch routing (wave32 WMMA convention: lane = tid&15
    // owns the same row in both halves of the wave).
    const int my_tile_row = lane;
    const int my_m   = row_start + my_tile_row;
    const int my_b   = batch_start + lane;
    const int safe_m = (my_m < M) ? my_m : (M - 1);
    const int safe_b = (my_b < B) ? my_b : (B - 1);

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 72;
    const char* row_base  = A + (long long)safe_m * row_stride;
    const _Float16* x_row = X + (long long)safe_b * K;

    // Cooperative codebook load mapping. Each thread loads 2 of the 4 F16
    // codebook entries for ONE tile-row. 32 threads × 2 entries = 64
    // entries = 16 rows × 4 entries → full coverage.
    //   load_tile_row = tid >> 1  (0..15)
    //   load_lo       = (tid & 1) * 2  (0 or 2)
    const int load_tile_row = tid >> 1;
    const int load_lo       = (tid & 1) * 2;
    const int load_my_row   = row_start + load_tile_row;
    const int load_safe_row = (load_my_row < M) ? load_my_row : (M - 1);
    const char* load_row_base = A + (long long)load_safe_row * row_stride;

    __shared__ _Float16 cb_lds[16 * 4];

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int g = 0; g < groups_per_row; g++) {
        // ── Cooperative codebook load (4 entries × 16 rows) ─────────
        const __half* cb_h_load = (const __half*)(load_row_base + g * 72);
        cb_lds[load_tile_row * 4 + load_lo + 0] = cb_h_load[load_lo + 0];
        cb_lds[load_tile_row * 4 + load_lo + 1] = cb_h_load[load_lo + 1];
        __syncthreads();

        // ── 16 K=16 WMMA sub-tiles per group of 256 ──────────────────
        const unsigned char* data = (const unsigned char*)(row_base + g * 72 + 8);
        const int cb_base = my_tile_row * 4;

        #pragma unroll 1
        for (int kt = 0; kt < 16; kt++) {
            // Each K=16 sub-tile needs 16 2-bit indices = 4 bytes.
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

    // Output write — same RDNA3 wave32 WMMA convention as
    // gemm_hfq4g256_wmma and gemm_f16_x_f16_wmma:
    //   acc[j] = C[2*j + (tid>>4)][tid & 15]
    //   row = 2*j + (tid >> 4), col (= batch) = tid & 15
    const int out_col = batch_start + lane;
    if (out_col < B) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
