// V4F position-0 attention with attn_sink (single-position, single KV head).
//
// At position 0 with no past KV history, each of the 64 query heads
// attends only to the current token's K=V vector via the formula:
//
//   score_h    = Q[h, :] · K + attn_sink[h]    (sink is a per-head bias)
//   prob_h     = sigmoid(score_h)
//   per_head_h = prob_h · V                     (V is the same as K — tied)
//
// V4F's o_groups=8 then reduces 64 heads into 8 groups by summing each
// group's 8 contributions, giving an output of shape [o_groups, head_dim]
// = [8, 512] = [4096] = hidden.
//
//   attn_out[g, :] = sum over h in group(g) of per_head_h
//
// Output magnitude bounded: ≤ 8 * |V|. With |V| ~5 from MQ4 GEMV,
// |attn_out| ≤ 40 — tight enough for HC mix to not blow up over
// 43 layers.

#include <hip/hip_runtime.h>

#define HEAD_DIM 512
#define O_GROUPS 8
#define HEADS_PER_GROUP 8   // n_heads / o_groups = 64 / 8

extern "C" __global__ void v4f_attn_pos0(
    const float* __restrict__ q,           // [n_heads, head_dim] = [64, 512]
    const float* __restrict__ kv,          // [head_dim] = [512] (tied K=V)
    const float* __restrict__ attn_sink,   // [n_heads] = [64]
    float*       __restrict__ attn_out,    // [o_groups, head_dim] = [8, 512] = [4096]
    int n_heads,
    int head_dim,
    int o_groups
) {
    const int g = blockIdx.x;             // group index 0..8
    const int d = threadIdx.x;            // dim index within group 0..head_dim
    if (g >= o_groups || d >= head_dim) return;

    // Pre-compute the per-head probabilities for this group.
    float kv_d = kv[d];
    float sum = 0.0f;
    const int heads_per_g = n_heads / o_groups;
    for (int h_local = 0; h_local < heads_per_g; ++h_local) {
        int h = g * heads_per_g + h_local;

        // Q[h] · K dot product across all head_dim. Each thread
        // computes its own component, then we use warp-shuffle to
        // total it. But for simplicity (single-thread per d), each
        // thread does a SEPARATE inner-product per head — wasteful
        // but correct.
        // Better: compute dot once per head and broadcast. Use
        // shared memory.
        __shared__ float shared_prob[HEADS_PER_GROUP];
        if (d == 0) {
            float dot = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                dot += q[h * head_dim + i] * kv[i];
            }
            // Numerically-stable sigmoid: sigmoid(x) = 1 / (1 + exp(-x)).
            float score = dot + attn_sink[h];
            shared_prob[h_local] = 1.0f / (1.0f + __expf(-score));
        }
        __syncthreads();
        sum += shared_prob[h_local] * kv_d;
        __syncthreads();
    }

    attn_out[g * head_dim + d] = sum;
}
