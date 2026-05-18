// Tail-only partial RoPE, half-split convention (V4F Phase 4).
//
// V4F applies rotary positional encoding only to the LAST
// `qk_rope_head_dim = 64` dims of each `head_dim = 512` head. The
// first `head_dim - qk_rope_head_dim = 448` dims are plain Q·K
// matmul without rotation.
//
// This is the tail-counterpart of `rope_partial_halfsplit_f32`,
// which rotates the FIRST `n_rot` dims (Qwen3.5 partial-rotary
// convention). The difference is just an offset on the dim index.
//
// Half-split convention matches HF's `rotate_half`:
//   q_embed = q * cos + rotate_half(q) * sin
// where rotate_half(x) = concat(-x[..., half:], x[..., :half]).
// "half" here refers to half of the rotated region only — not half
// of head_dim. Pairs are at (i, i + n_rot / 2) WITHIN the tail
// region, so absolute dim pairs are
//   (tail_start + i, tail_start + i + n_rot/2).
//
// Layer-specific frequency base is the same as the main RoPE
// (config `rope_theta = 10000` for main attention; for the indexer
// path use `compress_rope_theta = 160000` — see Phase 2 design doc).

#include <hip/hip_runtime.h>

extern "C" __global__ void rope_tail_halfsplit_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ pos_buf,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    int n_rot,           // tail rotation dim (= qk_rope_head_dim on V4F)
    float freq_base      // V4F main: 10000; indexer: 160000
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = n_rot / 2;
    if (i >= half) return;

    int pos = *pos_buf;
    float freq  = 1.0f / powf(freq_base, (float)(2 * i) / (float)n_rot);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // Tail offset: rotation applies to dims [head_dim - n_rot, head_dim).
    int tail_off = head_dim - n_rot;

    for (int h = 0; h < n_heads_q; ++h) {
        int base = h * head_dim + tail_off;
        float q0 = q[base + i];
        float q1 = q[base + i + half];
        q[base + i]        = q0 * cos_a - q1 * sin_a;
        q[base + i + half] = q0 * sin_a + q1 * cos_a;
    }

    for (int h = 0; h < n_heads_k; ++h) {
        int base = h * head_dim + tail_off;
        float k0 = k[base + i];
        float k1 = k[base + i + half];
        k[base + i]        = k0 * cos_a - k1 * sin_a;
        k[base + i + half] = k0 * sin_a + k1 * cos_a;
    }
}
