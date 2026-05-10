#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched HFQ1G128 GEMM with fused residual add.
// gfx1100+ / gfx115x (RDNA 3.5) wave32 WMMA.
//
// Each wave32 block computes a 16×16 output tile:
//   C[16 rows, 16 batch] += W[16 rows, 16 K] * X[16 K, 16 batch]
// iterated over K in chunks of 16, grouped by 128-element weight groups
// (8 K-tiles per group for HFQ1, vs 16 for HFQ4G256).
//
// Grid: [ceil(M/16), ceil(N/16)]. Block: [32]. LDS: 0.
//
// Per K-tile: 16 weight bits packed in 1 uint16 → unpacked to ±d FP16.
//
// RDNA3 wave32 WMMA C layout: lane k holds D[col=k%16, row=2*j+(k>>4)]
// for j=0..7. Output indexing must follow this layout — earlier kernel
// versions (gemm_hfq4g256_residual_wmma) used a transposed layout that
// produced gibberish in production (caught when wiring HFQ1 into
// forward_prefill_chunk).

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_hfq1g128_residual_wmma(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= M) return;
    if (batch_start >= batch_size) return;

    const int safe_row = (row_start + (tid & 15) < M) ? (row_start + (tid & 15)) : (M - 1);
    const int safe_batch = (batch_start + (tid & 15) < batch_size) ? (batch_start + (tid & 15)) : 0;

    const int groups_per_row = K / 128;
    const char* row_base = A + (long long)safe_row * groups_per_row * 18;
    const _Float16* x_base = X + (long long)safe_batch * K;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 18;
        _Float16 d_h = *(const _Float16*)gp;
        _Float16 neg_d_h = -d_h;
        const _Float16* xg = x_base + g * 128;

        #pragma unroll
        for (int kt = 0; kt < 8; kt++) {
            const int k_off = kt * 16;

            // Read 16 weight bits as one uint16. Bit i corresponds to
            // weight at position k_off + i within the group.
            unsigned short pk = *(const unsigned short*)(gp + 2 + (k_off >> 3));

            half16_t a_reg;
            #define DQ(i) a_reg[i] = ((pk >> (i)) & 1u) ? d_h : neg_d_h
            DQ(0);  DQ(1);  DQ(2);  DQ(3);
            DQ(4);  DQ(5);  DQ(6);  DQ(7);
            DQ(8);  DQ(9);  DQ(10); DQ(11);
            DQ(12); DQ(13); DQ(14); DQ(15);
            #undef DQ

            half16_t b_reg = *(const half16_t*)(xg + k_off);

            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }
    }

    // RDNA3 wave32 WMMA C layout: acc[j] = D[row = 2*j + (tid>>4), col = tid & 15].
    const int out_col = batch_start + (tid & 15);
    if (out_col < batch_size) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M)
                Y[(long long)out_col * M + out_row] += acc[j];
        }
    }
}
