// Inverse tail RoPE on attention output — BATCHED (Phase B2, 2026-05-18).
//
// Batched twin of `rope_tail_inverse_f32`. Per batch row b reads its
// absolute position from positions[b] and applies the inverse rotation
// to the last n_rot dims of each head.
//
// Grid:  [ceil(n_rot/2 / block), batch_size, 1]
// Block: [block, 1, 1]
//
// Byte-identical to rope_tail_inverse_f32 at batch_size == 1 with
// positions[0] == pos_buf[0].

#include <hip/hip_runtime.h>

extern "C" __global__ void rope_tail_inverse_batched_f32(
    float* __restrict__ x,           // [batch, n_heads, head_dim]
    const int* __restrict__ positions,
    int n_heads,
    int head_dim,
    int n_rot,
    float freq_base,
    int batch_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    int half = n_rot / 2;
    if (i >= half || b >= batch_size) return;

    int pos = positions[b];
    float freq  = 1.0f / powf(freq_base, (float)(2 * i) / (float)n_rot);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = -sinf(angle);  // INVERSE

    int tail_off = head_dim - n_rot;
    int d_even = tail_off + 2 * i;
    int d_odd  = d_even + 1;

    int row_stride = n_heads * head_dim;
    int row_base = b * row_stride;

    for (int h = 0; h < n_heads; ++h) {
        int base = row_base + h * head_dim;
        float x0 = x[base + d_even];
        float x1 = x[base + d_odd];
        x[base + d_even] = x0 * cos_a - x1 * sin_a;
        x[base + d_odd]  = x0 * sin_a + x1 * cos_a;
    }
}
