// Batched flash attention tile for Q8_0 KV. Mirrors
// `attention_flash_q8_0_tile` (single-q decode) with an outer batch dim
// added via blockIdx.z, plus optional tree-mask bias for DDTree verify.
//
// Per-thread layout: 4 dims per thread (matches the head_dim=128 single-q
// kernel). For head_dim>128 the kernel loops over `n_halves = head_dim/128`
// at write time so the same kernel handles 128 and 256 transparently.
//
// Grid: [n_heads, max_tiles, sub_batch_size]. Block: [32]. LDS: tile_size*4 bytes.
// Partials layout: [sub_batch × n_heads × max_tiles × (2 + head_dim)].
//
// Replaces the per-position `attention_flash_q8_0` loop in
// `forward_prefill_chunk`'s long-context fallback — single launch instead
// of one launch per row.

#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_q8_0_tile_batched(
    const float* __restrict__ q,                 // [sub_batch × n_heads × head_dim]
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,           // [total_batch]
    const float* __restrict__ tree_bias,         // optional [total_batch × block_cols]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles,
    int batch_offset,
    int block_start,                             // ignored when tree_bias == nullptr
    int block_cols                               // ignored when tree_bias == nullptr
) {
    const int h = blockIdx.x;
    const int tile_id = blockIdx.y;
    const int local_bid = blockIdx.z;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;

    const int global_bid = batch_offset + local_bid;
    const bool tree_mode = (tree_bias != nullptr);
    const int seq_len = tree_mode ? (block_start + block_cols)
                                  : (positions[global_bid] + 1);
    const int tile_start = tile_id * tile_size;
    if (tile_start >= seq_len) return;
    const int tile_end = (tile_start + tile_size < seq_len) ? (tile_start + tile_size) : seq_len;
    const int tile_len = tile_end - tile_start;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;

    const int blocks_per_head = head_dim / 32;
    const int total_blocks = n_kv_heads * blocks_per_head;
    const int kv_head_block_start = kv_h * blocks_per_head;
    const int row_stride = total_blocks * 34;
    const int q_dim = n_heads * head_dim;

    // LDS: scores[tile_size]. Q stays in registers (per-thread fragment).
    extern __shared__ float scores[];

    // ═══ Phase A: Q · K^T per tile position ═══
    // Wave-cooperative: each thread covers 4 contiguous dims; wave-shuffles
    // accumulate the partial dot product, lane 0 writes to LDS.
    // Thread assignment: bi_k = tid/8 (which 32-block of dims), offset = (tid%8)*4.
    const int k_bi = tid / 8;
    const int k_off = (tid % 8) * 4;
    const int k_blk_off = (kv_head_block_start + k_bi) * 34;

    // Load Q fragment for this thread (4 fp32 values).
    const float* q_head = q + (long long)local_bid * q_dim + h * head_dim;
    const int q_dim_off = k_bi * 32 + k_off;
    float qf0 = 0.0f, qf1 = 0.0f, qf2 = 0.0f, qf3 = 0.0f;
    if (q_dim_off + 3 < head_dim) {
        qf0 = q_head[q_dim_off + 0];
        qf1 = q_head[q_dim_off + 1];
        qf2 = q_head[q_dim_off + 2];
        qf3 = q_head[q_dim_off + 3];
    }

    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const unsigned char* kb = k_cache + (size_t)t * row_stride + k_blk_off;
        float ks = (float)*((const _Float16*)kb);
        float partial = 0.0f;
        if (q_dim_off + 3 < head_dim) {
            partial = qf0 * (ks * (float)((signed char)kb[2 + k_off + 0]))
                    + qf1 * (ks * (float)((signed char)kb[2 + k_off + 1]))
                    + qf2 * (ks * (float)((signed char)kb[2 + k_off + 2]))
                    + qf3 * (ks * (float)((signed char)kb[2 + k_off + 3]));
        }
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0) {
            float s = partial * scale_attn;
            if (tree_mode && t >= block_start) {
                s += tree_bias[(long long)global_bid * block_cols + (t - block_start)];
            }
            scores[t_local] = s;
        }
    }
    __syncthreads();

    // ═══ Phase B: tile max ═══
    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 32)
        local_max = fmaxf(local_max, scores[i]);
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    float tile_max = local_max;

    // ═══ Phase C: exp + sum ═══
    float local_sum = 0.0f;
    for (int i = tid; i < tile_len; i += 32) {
        float e = expf(scores[i] - tile_max);
        scores[i] = e;
        local_sum += e;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor(local_sum, off);
    float tile_sum = local_sum;
    __syncthreads();

    // ═══ Phase D: V-weighted accumulation ═══
    // Per-thread 4-dim slot. Loop over n_halves so the same kernel handles
    // head_dim=128 (1 half) and head_dim=256 (2 halves).
    const int n_halves = head_dim / 128;
    float* p = partials + ((long long)local_bid * n_heads + h) * max_tiles * (2 + head_dim)
                        + (long long)tile_id * (2 + head_dim);

    for (int half = 0; half < n_halves; half++) {
        const int d_base = half * 128 + tid * 4;
        const int bi = d_base / 32;
        const int bj_base = d_base % 32;
        const int v_blk_off = (kv_head_block_start + bi) * 34;

        float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;
        for (int t_local = 0; t_local < tile_len; t_local++) {
            float w = scores[t_local];
            int t = tile_start + t_local;
            const unsigned char* vb = v_cache + (size_t)t * row_stride + v_blk_off;
            float vs = (float)*((const _Float16*)vb);
            out0 += w * (vs * (float)((signed char)vb[2 + bj_base + 0]));
            out1 += w * (vs * (float)((signed char)vb[2 + bj_base + 1]));
            out2 += w * (vs * (float)((signed char)vb[2 + bj_base + 2]));
            out3 += w * (vs * (float)((signed char)vb[2 + bj_base + 3]));
        }
        p[2 + d_base + 0] = out0;
        p[2 + d_base + 1] = out1;
        p[2 + d_base + 2] = out2;
        p[2 + d_base + 3] = out3;
    }

    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }
}
