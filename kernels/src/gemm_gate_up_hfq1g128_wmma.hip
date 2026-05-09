#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// WMMA-accelerated batched 2-way fused HFQ1G128 GEMM (gate + up).
// gfx1100+ / gfx115x (RDNA 3.5+) wave32 WMMA.
// Grid: [ceil((gate_m + up_m)/16), ceil(N/16)]. Block: [32]. LDS: 0.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_gate_up_hfq1g128_wmma(
    const char*    __restrict__ A_gate,
    const char*    __restrict__ A_up,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y_gate,
    float*         __restrict__ Y_up,
    int gate_m, int up_m,
    int K, int N
) {
    const int total_m = gate_m + up_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    const int my_row = row_start + (tid & 15);
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    const char* A;
    int local_row;
    if (safe_row < gate_m) {
        A = A_gate; local_row = safe_row;
    } else {
        A = A_up; local_row = safe_row - gate_m;
    }

    const int groups_per_row = K / 128;
    const long long row_stride = (long long)groups_per_row * 18;
    const char* row_base = A + local_row * row_stride;

    const int safe_batch = (batch_start + (tid & 15) < N)
                           ? (batch_start + (tid & 15)) : 0;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 18;
        _Float16 d_h = *(const _Float16*)gp;
        _Float16 neg_d_h = -d_h;

        const _Float16* x_base = X + (long long)safe_batch * K + g * 128;

        #pragma unroll
        for (int kt = 0; kt < 8; kt++) {
            const int k_off = kt * 16;
            unsigned short pk = *(const unsigned short*)(gp + 2 + (k_off >> 3));

            half16_t a_reg;
            #define DQ(i) a_reg[i] = ((pk >> (i)) & 1u) ? d_h : neg_d_h
            DQ(0);  DQ(1);  DQ(2);  DQ(3);
            DQ(4);  DQ(5);  DQ(6);  DQ(7);
            DQ(8);  DQ(9);  DQ(10); DQ(11);
            DQ(12); DQ(13); DQ(14); DQ(15);
            #undef DQ

            half16_t b_reg = *(const half16_t*)(x_base + k_off);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }
    }

    const int out_col = batch_start + (tid & 15);
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < total_m) {
                float* Y;
                int proj_row;
                int out_stride;
                if (out_row < gate_m) {
                    Y = Y_gate; proj_row = out_row; out_stride = gate_m;
                } else {
                    Y = Y_up; proj_row = out_row - gate_m; out_stride = up_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
