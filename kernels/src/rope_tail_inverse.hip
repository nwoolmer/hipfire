// Inverse tail RoPE on attention output (V4F, INTERLEAVED).
//
// Upstream V4F applies `apply_rotary_emb(o[..., -rd:], freqs_cis, True)`
// — the `True` flag uses conjugate frequencies (inverse rotation).
// Pair convention: INTERLEAVED (2i, 2i+1), matching torch.view_as_complex.
//
// Inverse is forward with negated sin:
//   forward: x_e' = x_e*cos - x_o*sin;  x_o' = x_e*sin + x_o*cos
//   inverse: x_e' = x_e*cos + x_o*sin;  x_o' = -x_e*sin + x_o*cos

#include <hip/hip_runtime.h>

extern "C" __global__ void rope_tail_inverse_f32(
    float* __restrict__ x,           // [n_heads, head_dim]
    const int* __restrict__ pos_buf, // [1]
    int n_heads,
    int head_dim,
    int n_rot,
    float freq_base
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = n_rot / 2;
    if (i >= half) return;

    int pos = *pos_buf;
    float freq  = 1.0f / powf(freq_base, (float)(2 * i) / (float)n_rot);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = -sinf(angle);  // INVERSE: negate sin

    int tail_off = head_dim - n_rot;
    int d_even = tail_off + 2 * i;
    int d_odd  = d_even + 1;

    for (int h = 0; h < n_heads; ++h) {
        int base = h * head_dim;
        float x0 = x[base + d_even];
        float x1 = x[base + d_odd];
        x[base + d_even] = x0 * cos_a - x1 * sin_a;
        x[base + d_odd]  = x0 * sin_a + x1 * cos_a;
    }
}
