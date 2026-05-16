// Inverse tail-only partial RoPE on attention output (V4F).
//
// Upstream V4F applies `apply_rotary_emb(o[..., -rd:], freqs_cis, True)`
// — the `True` flag is the inverse rotation. This undoes the RoPE that
// V (tied to K) had baked in when written to the KV cache, so the
// downstream wo_a/wo_b projection sees attention output in a
// position-agnostic phase.
//
// Inverse is forward with negated sin:
//   forward: q' = q*cos - rot_half(q)*sin   →  cos,  sin
//   inverse: q' = q*cos - rot_half(q)*(-sin) →  cos, -sin

#include <hip/hip_runtime.h>

extern "C" __global__ void rope_tail_inverse_f32(
    float* __restrict__ x,           // [n_heads, head_dim]
    const int* __restrict__ pos_buf, // [1]
    int n_heads,
    int head_dim,
    int n_rot,                       // tail rotation dim (= qk_rope_head_dim)
    float freq_base                  // 10000 on V4F main attention
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
    for (int h = 0; h < n_heads; ++h) {
        int base = h * head_dim + tail_off;
        float x0 = x[base + i];
        float x1 = x[base + i + half];
        x[base + i]        = x0 * cos_a - x1 * sin_a;
        x[base + i + half] = x0 * sin_a + x1 * cos_a;
    }
}
