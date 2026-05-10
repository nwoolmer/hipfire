#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Fused HFQ1G128 GEMV with residual add: y[row] += W[row, :] · x[:].
// Eliminates the alloc-tmp + plain-gemv + add_inplace_f32 + free chain
// that `weight_gemv_residual` falls through to for HFQ1 — saves 1 kernel
// launch per residual site (wo and w_down on every decode layer = 72
// per token at Bonsai's 36 layers).
//
// Same multirow-quad inner loop as gemv_hfq1g128_multirow_quad: 4 groups
// per K-step + R rows per block, with the only behavioral diff being
// `y[row] += s` vs `y[row] = s` at write-back.

template<int R>
__device__ __forceinline__ void gemv_hfq1g128_residual_multirow_quad_body(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row_base = blockIdx.x * R;
    if (row_base >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 128;
    const long long row_stride = (long long)groups_per_row * 18;

    const char* row_ptrs[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        int row = row_base + r;
        int safe_row = (row < M) ? row : row_base;
        row_ptrs[r] = A + (long long)safe_row * row_stride;
    }

    float acc[R];
    #pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.0f;

    const int byte_off = tid >> 1;
    const int bit_shift = (tid & 1) << 2;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g_base = q << 2;
        const int x_base = g_base * 128 + tid * 4;

        const float xv00 = x[x_base + 0],   xv01 = x[x_base + 1];
        const float xv02 = x[x_base + 2],   xv03 = x[x_base + 3];
        const float xv10 = x[x_base + 128], xv11 = x[x_base + 129];
        const float xv12 = x[x_base + 130], xv13 = x[x_base + 131];
        const float xv20 = x[x_base + 256], xv21 = x[x_base + 257];
        const float xv22 = x[x_base + 258], xv23 = x[x_base + 259];
        const float xv30 = x[x_base + 384], xv31 = x[x_base + 385];
        const float xv32 = x[x_base + 386], xv33 = x[x_base + 387];

        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gp0 = row_ptrs[r] + (long long)g_base * 18;
            const char* gp1 = gp0 + 18;
            const char* gp2 = gp1 + 18;
            const char* gp3 = gp2 + 18;

            _Float16 d0_h = *(const _Float16*)gp0;
            _Float16 d1_h = *(const _Float16*)gp1;
            _Float16 d2_h = *(const _Float16*)gp2;
            _Float16 d3_h = *(const _Float16*)gp3;
            float d0 = (float)d0_h, nd0 = -d0;
            float d1 = (float)d1_h, nd1 = -d1;
            float d2 = (float)d2_h, nd2 = -d2;
            float d3 = (float)d3_h, nd3 = -d3;

            unsigned char b0 = (unsigned char)gp0[2 + byte_off];
            unsigned char b1 = (unsigned char)gp1[2 + byte_off];
            unsigned char b2 = (unsigned char)gp2[2 + byte_off];
            unsigned char b3 = (unsigned char)gp3[2 + byte_off];
            unsigned char n0 = (b0 >> bit_shift) & 0xF;
            unsigned char n1 = (b1 >> bit_shift) & 0xF;
            unsigned char n2 = (b2 >> bit_shift) & 0xF;
            unsigned char n3 = (b3 >> bit_shift) & 0xF;

            float w00 = (n0 & 0x1) ? d0 : nd0;
            float w01 = (n0 & 0x2) ? d0 : nd0;
            float w02 = (n0 & 0x4) ? d0 : nd0;
            float w03 = (n0 & 0x8) ? d0 : nd0;
            float w10 = (n1 & 0x1) ? d1 : nd1;
            float w11 = (n1 & 0x2) ? d1 : nd1;
            float w12 = (n1 & 0x4) ? d1 : nd1;
            float w13 = (n1 & 0x8) ? d1 : nd1;
            float w20 = (n2 & 0x1) ? d2 : nd2;
            float w21 = (n2 & 0x2) ? d2 : nd2;
            float w22 = (n2 & 0x4) ? d2 : nd2;
            float w23 = (n2 & 0x8) ? d2 : nd2;
            float w30 = (n3 & 0x1) ? d3 : nd3;
            float w31 = (n3 & 0x2) ? d3 : nd3;
            float w32 = (n3 & 0x4) ? d3 : nd3;
            float w33 = (n3 & 0x8) ? d3 : nd3;

            acc[r] += w00 * xv00 + w01 * xv01 + w02 * xv02 + w03 * xv03
                    + w10 * xv10 + w11 * xv11 + w12 * xv12 + w13 * xv13
                    + w20 * xv20 + w21 * xv21 + w22 * xv22 + w23 * xv23
                    + w30 * xv30 + w31 * xv31 + w32 * xv32 + w33 * xv33;
        }
    }

    for (int gi = 0; gi < tail; gi++) {
        const int g = (quads << 2) + gi;
        const int x_base = g * 128 + tid * 4;
        const float xv0 = x[x_base];
        const float xv1 = x[x_base + 1];
        const float xv2 = x[x_base + 2];
        const float xv3 = x[x_base + 3];

        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gp = row_ptrs[r] + (long long)g * 18;
            _Float16 d_h = *(const _Float16*)gp;
            float d = (float)d_h;
            float neg_d = -d;
            unsigned char byte_val = (unsigned char)gp[2 + byte_off];
            unsigned char nib = (byte_val >> bit_shift) & 0xF;
            float w0 = (nib & 0x1) ? d : neg_d;
            float w1 = (nib & 0x2) ? d : neg_d;
            float w2 = (nib & 0x4) ? d : neg_d;
            float w3 = (nib & 0x8) ? d : neg_d;
            acc[r] += w0 * xv0 + w1 * xv1 + w2 * xv2 + w3 * xv3;
        }
    }

    // Per-row warp reduction; lane 0 reads y[row], adds the dot product,
    // writes back. y is also the residual stream — fused.
    #pragma unroll
    for (int r = 0; r < R; r++) {
        float s = acc[r];
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down(s, offset);
        int row = row_base + r;
        if (tid == 0 && row < M) y[row] += s;
    }
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq1g128_residual_multirow_quad_r2(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_hfq1g128_residual_multirow_quad_body<2>(A, x, y, M, K);
}
