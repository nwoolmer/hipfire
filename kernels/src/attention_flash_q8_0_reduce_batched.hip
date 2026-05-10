// Batched flash attention reduce for Q8_0 V (V in normal/un-rotated space).
// Combines per-tile partials produced by attention_flash_q8_0_tile_batched
// into final attention output via 2-pass online softmax.
//
// 2-pass (find global_max, then accumulate) instead of running corrections —
// matches `attention_flash_q8_0_reduce` numerics so output is byte-equal
// to the per-position single-q flash on identical inputs (modulo legitimate
// FP-reorder differences from the batched accumulator order).
//
// Grid: [n_heads, sub_batch_size]. Block: [32]. Zero LDS.
// Per-thread layout: 4 dims/thread, looped over n_halves = head_dim / 128
// (handles head_dim=128 and 256 with the same kernel).

#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_q8_0_reduce_batched(
    const float* __restrict__ partials,    // [sub_batch × n_heads × max_tiles × stride]
    float* __restrict__ out,               // [sub_batch × n_heads × head_dim]
    const int* __restrict__ positions,     // [total_batch]
    int n_heads,
    int head_dim,
    int tile_size,
    int max_tiles,
    int batch_offset,
    int block_start,                       // 0 = legacy causal, >0 = tree mode
    int block_cols                         // 0 = legacy causal, >0 = tree mode
) {
    const int h = blockIdx.x;
    const int local_bid = blockIdx.y;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const int stride = 2 + head_dim;

    const int global_bid = batch_offset + local_bid;
    const int seq_len = (block_cols > 0) ? (block_start + block_cols)
                                          : (positions[global_bid] + 1);
    const int n_tiles = (seq_len + tile_size - 1) / tile_size;

    const float* my_partials = partials
        + ((long long)local_bid * n_heads + h) * max_tiles * stride;

    // ── Pass 1: global max across tiles ──
    float global_max = -1e30f;
    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = my_partials + tile * stride;
        float tile_sum = p[1];
        if (tile_sum > 0.0f)
            global_max = fmaxf(global_max, p[0]);
    }

    // ── Pass 2: accumulate, normalize ──
    // Loop over halves so head_dim=128 (n_halves=1) and head_dim=256 (n_halves=2)
    // share the same kernel. Each thread covers 4 dims per half.
    const int n_halves = head_dim / 128;
    const int q_dim = n_heads * head_dim;
    float* my_out = out + (long long)local_bid * q_dim + h * head_dim;

    float global_sum = 0.0f;
    for (int half = 0; half < n_halves; half++) {
        const int d_base = half * 128 + tid * 4;
        float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;

        for (int tile = 0; tile < n_tiles; tile++) {
            const float* p = my_partials + tile * stride;
            float tile_max = p[0];
            float tile_sum = p[1];
            if (tile_sum <= 0.0f) continue;

            float corr = expf(tile_max - global_max);
            g0 += p[2 + d_base + 0] * corr;
            g1 += p[2 + d_base + 1] * corr;
            g2 += p[2 + d_base + 2] * corr;
            g3 += p[2 + d_base + 3] * corr;
            // global_sum is scalar; compute on first half only.
            if (half == 0) global_sum += tile_sum * corr;
        }

        float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
        my_out[d_base + 0] = g0 * inv_sum;
        my_out[d_base + 1] = g1 * inv_sum;
        my_out[d_base + 2] = g2 * inv_sum;
        my_out[d_base + 3] = g3 * inv_sum;
    }
}
