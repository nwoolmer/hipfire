// WMMA per-group HFQ4G256 batched GEMV for V4F's wo_a block-diagonal
// projection. F16 input × HFQ4 weight → F32 output, with the per-group
// block structure preserved.
//
// Same shape contract as `wo_per_group_batched_hfq4g256` (scalar):
//   wo_a : [G, M, K] HFQ4G256-packed (each [M, K] block is independent)
//   x_f16: [B, G, K] F16 (caller stages F32→F16)
//   y    : [B, G, M] F32
//
// Per (b, g, m_tile, k): 16-element WMMA fragment of dequant'd weights
// × 16-element F16 fragment of x → 16×16 F32 acc tile.
//
// Grid:  [ceil(M / 16), ceil(B / 16), G]
// Block: [32]

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 8)
extern "C" __global__ void wo_per_group_batched_hfq4g256_wmma(
    const char*     __restrict__ wo_a,    // [G * M * K/256 * 136] bytes
    const _Float16* __restrict__ x_f16,   // [B, G, K] F16
    float*          __restrict__ y_out,   // [B, G, M] F32
    int G, int M, int K, int batch_size
) {
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int g           = blockIdx.z;
    if (row_start >= M || batch_start >= batch_size || g >= G) return;
    const int tid  = threadIdx.x;
    const int lane = tid & 15;

    const int my_m   = row_start + lane;
    const int my_b   = batch_start + lane;
    const int safe_m = (my_m < M) ? my_m : (M - 1);
    const int safe_b = (my_b < batch_size) ? my_b : (batch_size - 1);

    const int groups_per_row = K / 256;
    const long long per_group_bytes = (long long)M * groups_per_row * 136;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_ptr =
        wo_a + (long long)g * per_group_bytes + (long long)safe_m * row_stride;
    const _Float16* x_row =
        x_f16 + ((size_t)safe_b * G + g) * K;

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int gp = 0; gp < groups_per_row; gp++) {
        const char* gpp = row_ptr + gp * 136;
        _Float16 sc = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gpp));
        _Float16 zp = (_Float16)__builtin_bit_cast(float, *(const unsigned int*)(gpp + 4));
        const unsigned int* pk = (const unsigned int*)(gpp + 8);

        #pragma unroll 1
        for (int kt = 0; kt < 16; kt++) {
            unsigned int pk0 = pk[kt * 2];
            unsigned int pk1 = pk[kt * 2 + 1];
            half16_t a_frag;
            #define DQ(i, p, sh) a_frag[i] = sc * (_Float16)(float)(((p) >> (sh)) & 0xFu) + zp
            DQ(0,  pk0,  0); DQ(1,  pk0,  4); DQ(2,  pk0,  8); DQ(3,  pk0, 12);
            DQ(4,  pk0, 16); DQ(5,  pk0, 20); DQ(6,  pk0, 24); DQ(7,  pk0, 28);
            DQ(8,  pk1,  0); DQ(9,  pk1,  4); DQ(10, pk1,  8); DQ(11, pk1, 12);
            DQ(12, pk1, 16); DQ(13, pk1, 20); DQ(14, pk1, 24); DQ(15, pk1, 28);
            #undef DQ

            const int k_off = gp * 256 + kt * 16;
            half16_t x_frag = *(const half16_t*)(x_row + k_off);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, x_frag, acc);
        }
    }

    const int out_col = batch_start + lane;
    if (out_col < batch_size) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                y_out[((size_t)out_col * G + g) * M + out_row] = acc[j];
            }
        }
    }
}
