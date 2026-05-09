#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HFQ1G128 packed-2-groups GEMV: each lane processes 8 weights of one
// group; 16 lanes serve one group; 32 lanes cover 2 groups per K-step.
// Halves outer-loop count vs single-group GEMV; better DRAM cache-line
// utilization (each lane reads its own byte rather than sharing a byte
// with one partner lane).
//
// Multi-row R=2/R=4 variants combine packed-2-groups with row tiling.
//
// Per K-step bookkeeping:
//   - 16 lanes × 8 floats = 128 activations per group, 2 groups → 256 floats
//   - 16 lanes × 1 byte = 16 bytes per group payload, 2 groups → 32 bytes
//   - Per-row weight scale (FP16) read by all 16 lanes of a group (cached)

template<int R>
__device__ __forceinline__ void gemv_hfq1g128_packed_body(
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

    // Lane → (group_local, byte_within_group)
    //   lanes 0..15 → group_local = 0, byte 0..15 of group g
    //   lanes 16..31 → group_local = 1, byte 0..15 of group g+1
    const int g_local = tid >> 4;       // 0 or 1
    const int byte_off = tid & 15;      // 0..15 (= weight index/8)

    for (int g_outer = 0; g_outer < groups_per_row; g_outer += 2) {
        const int g = g_outer + g_local;
        // Skip lanes whose group index is past the row end (only the last
        // outer step can have this if groups_per_row is odd).
        const bool valid = g < groups_per_row;

        // Hoist 8 x floats per lane (shared across R rows). Use a guard
        // for lanes whose group is invalid — mask via valid flag.
        const int base_idx = g * 128 + byte_off * 8;
        float xv0 = 0.0f, xv1 = 0.0f, xv2 = 0.0f, xv3 = 0.0f;
        float xv4 = 0.0f, xv5 = 0.0f, xv6 = 0.0f, xv7 = 0.0f;
        if (valid) {
            xv0 = x[base_idx];
            xv1 = x[base_idx + 1];
            xv2 = x[base_idx + 2];
            xv3 = x[base_idx + 3];
            xv4 = x[base_idx + 4];
            xv5 = x[base_idx + 5];
            xv6 = x[base_idx + 6];
            xv7 = x[base_idx + 7];
        }

        #pragma unroll
        for (int r = 0; r < R; r++) {
            float row_acc = 0.0f;
            if (valid) {
                const char* gptr = row_ptrs[r] + g * 18;
                _Float16 d_h = *(const _Float16*)gptr;
                float d = (float)d_h;
                float neg_d = -d;
                unsigned char byte_val = (unsigned char)gptr[2 + byte_off];

                float w0 = (byte_val & 0x01) ? d : neg_d;
                float w1 = (byte_val & 0x02) ? d : neg_d;
                float w2 = (byte_val & 0x04) ? d : neg_d;
                float w3 = (byte_val & 0x08) ? d : neg_d;
                float w4 = (byte_val & 0x10) ? d : neg_d;
                float w5 = (byte_val & 0x20) ? d : neg_d;
                float w6 = (byte_val & 0x40) ? d : neg_d;
                float w7 = (byte_val & 0x80) ? d : neg_d;

                row_acc = w0 * xv0 + w1 * xv1 + w2 * xv2 + w3 * xv3
                        + w4 * xv4 + w5 * xv5 + w6 * xv6 + w7 * xv7;
            }
            acc[r] += row_acc;
        }
    }

    // 32-lane tree reduction. Both g_local halves contribute to the same
    // output row, so the full 32-lane sum gives the correct row total.
    #pragma unroll
    for (int r = 0; r < R; r++) {
        float s = acc[r];
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down(s, offset);
        int row = row_base + r;
        if (tid == 0 && row < M) y[row] = s;
    }
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq1g128_packed_r1(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_hfq1g128_packed_body<1>(A, x, y, M, K);
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq1g128_packed_r2(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_hfq1g128_packed_body<2>(A, x, y, M, K);
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq1g128_packed_r4(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_hfq1g128_packed_body<4>(A, x, y, M, K);
}
