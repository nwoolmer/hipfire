// V4F ZA-fused WMMA-accelerated GEMM: F16 weight × F16 input → F32 output,
// 4-way M-axis fan-out (Phase D, 2026-05-21).
//
// Strict superset of `gemm_f16_x_f16_wmma`: same K-loop body, but each
// thread in the 16-row tile routes its weight read AND its output write
// to ONE of FOUR `[m_i, K]` weight matrices / `[B, m_i]` output buffers.
// Used by V4F's compressor batched path (`attention_block_batched_mixed`,
// forward.rs:4425-4451), where 4 separate F16-WMMA matmuls
// (comp_wkv, comp_wgate, idx_wkv, idx_wgate) all consume the SAME
// `tmp_plain_batch_f16` input and produce 4 distinct outputs. One launch
// replaces the four sequential calls; the four weight matrices are
// virtually stacked on the M-axis.
//
// Layout:
//   A_0..A_3 : 4× [m_i, K] F16 row-major   (each weight is independent)
//   X        : [B, K] F16 row-major        (shared input, single buffer)
//   Y_0..Y_3 : 4× [B, m_i] F32 row-major   (matches gemm_f16_x_f16_wmma)
//
// Compute (per region i):
//   Y_i[b, m_i_local] = sum_k A_i[m_i_local, k] * X[b, k]
//
// 4-way routing: cumulative M offsets [0, m0, m0+m1, m0+m1+m2,
// m0+m1+m2+m3=total_m]. Each thread checks `my_row` against these
// boundaries to pick (A, local_row, out_stride, Y).
//
// Grid: [ceil(total_m / 16), ceil(B / 16)].  Block: [32].
// LDS:  0.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_f16_x_f16_wmma_za4(
    const _Float16* __restrict__ A_0,    // [m0, K]
    const _Float16* __restrict__ A_1,    // [m1, K]
    const _Float16* __restrict__ A_2,    // [m2, K]
    const _Float16* __restrict__ A_3,    // [m3, K]
    const _Float16* __restrict__ X,      // [B, K]
    float*          __restrict__ Y_0,    // [B, m0]
    float*          __restrict__ Y_1,    // [B, m1]
    float*          __restrict__ Y_2,    // [B, m2]
    float*          __restrict__ Y_3,    // [B, m3]
    int m0, int m1, int m2, int m3,
    int K, int B
) {
    const int total_m = m0 + m1 + m2 + m3;
    const int row_start   = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid  = threadIdx.x;
    const int lane = tid & 15;

    if (row_start >= total_m || batch_start >= B) return;

    // Pick row/batch indices for this thread. Same wave32 WMMA layout as
    // gemm_f16_x_f16_wmma: threads 0..15 and 16..31 hold duplicate
    // fragments (WMMA spec).
    int my_m = row_start + lane;
    int my_b = batch_start + lane;
    int safe_m = (my_m < total_m) ? my_m : (total_m - 1);
    int safe_b = (my_b < B) ? my_b : (B - 1);

    // Route this thread's row to the correct A buffer + local row.
    const _Float16* A;
    int local_row;
    if (safe_m < m0) {
        A = A_0;  local_row = safe_m;
    } else if (safe_m < m0 + m1) {
        A = A_1;  local_row = safe_m - m0;
    } else if (safe_m < m0 + m1 + m2) {
        A = A_2;  local_row = safe_m - (m0 + m1);
    } else {
        A = A_3;  local_row = safe_m - (m0 + m1 + m2);
    }

    const _Float16* w_row = A + (long long)local_row * K;
    const _Float16* x_row = X + (long long)safe_b * K;

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // K loop: one WMMA mma per 16-element step. Identical to the
    // single-output kernel — each thread reads its routed weight row.
    for (int k = 0; k < K; k += 16) {
        half16_t a_frag = *(const half16_t*)(w_row + k);
        half16_t x_frag = *(const half16_t*)(x_row + k);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, x_frag, acc);
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // Each of 8 outputs per thread routes to the right Y buffer.
    const int out_col = batch_start + lane;
    if (out_col < B) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < total_m) {
                float* Y;
                int proj_row;
                int out_stride;
                if (out_row < m0) {
                    Y = Y_0;  proj_row = out_row;                       out_stride = m0;
                } else if (out_row < m0 + m1) {
                    Y = Y_1;  proj_row = out_row - m0;                  out_stride = m1;
                } else if (out_row < m0 + m1 + m2) {
                    Y = Y_2;  proj_row = out_row - (m0 + m1);           out_stride = m2;
                } else {
                    Y = Y_3;  proj_row = out_row - (m0 + m1 + m2);      out_stride = m3;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
