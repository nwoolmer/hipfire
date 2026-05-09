#include <hip/hip_runtime.h>

// GQA-N attention for FP32 KV cache: one workgroup processes N consecutive
// Q heads that share the same KV head. K/V values are loaded once and
// dotted against N different Q vectors → N× reduction in DRAM KV traffic
// vs the per-head attention_f32 dispatch.
//
// Bonsai-8B: n_heads=32, n_kv_heads=8 → kv_group=4 → use the R=4 entry.
// Falls back to attention_f32 for other ratios.
//
// Shared memory: N × seq_len floats for scores + nthreads floats for the
// reduction workspace. For the 4-head + 8K seq case that's ~128 KB,
// exceeding gfx1151's 96 KB LDS. Caller decides which kernel to use
// (attention_f32 for very long contexts, this for typical decode).

template<int R>
__device__ __forceinline__ void attention_f32_gqa_body(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale
) {
    const int seq_len = pos_buf[0] + 1;
    const int kv_group = n_heads / n_kv_heads;  // expected = R for the active kernel
    extern __shared__ float sdata[];

    const int kv_h = blockIdx.x;
    if (kv_h >= n_kv_heads) return;

    const int q_head_base = kv_h * kv_group;  // first Q head in this KV group
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Hoist Q vectors for the R Q heads in this KV group into registers /
    // shared memory. For R=4 + head_dim=128 that's 16 floats / lane if we
    // try to register them — instead, point at them in DRAM (cached).
    const float* q_heads[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        q_heads[r] = q + (q_head_base + r) * head_dim;
    }

    // Layout shared mem: [scores_h0[seq_len] | scores_h1 | ... | scores_h{R-1} | workspace[nthreads]]
    float* scores_base = sdata;
    float* workspace = sdata + (long long)R * seq_len;

    // Phase 1: compute Q · K^T for all R Q heads in parallel.
    // Each thread handles a subset of positions. K is loaded ONCE per
    // position and reused for all R Q heads.
    float local_max[R];
    #pragma unroll
    for (int r = 0; r < R; r++) local_max[r] = -1e30f;

    for (int t = tid; t < seq_len; t += nthreads) {
        const float* k_t = k_cache + (long long)t * n_kv_heads * head_dim + kv_h * head_dim;

        float dot[R];
        #pragma unroll
        for (int r = 0; r < R; r++) dot[r] = 0.0f;

        // Single sweep over head_dim, one K load shared across R Q heads.
        for (int d = 0; d < head_dim; d++) {
            const float k_val = k_t[d];
            #pragma unroll
            for (int r = 0; r < R; r++) {
                dot[r] += q_heads[r][d] * k_val;
            }
        }

        #pragma unroll
        for (int r = 0; r < R; r++) {
            float s = dot[r] * scale;
            scores_base[(long long)r * seq_len + t] = s;
            local_max[r] = fmaxf(local_max[r], s);
        }
    }

    // Per-Q-head max reduction. R sequential reductions over the shared
    // workspace — each Q head has independent max.
    float max_val[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        workspace[tid] = local_max[r];
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) workspace[tid] = fmaxf(workspace[tid], workspace[tid + s]);
            __syncthreads();
        }
        max_val[r] = workspace[0];
        __syncthreads();
    }

    // Exp + per-head sum.
    float local_sum[R];
    #pragma unroll
    for (int r = 0; r < R; r++) local_sum[r] = 0.0f;

    for (int t = tid; t < seq_len; t += nthreads) {
        #pragma unroll
        for (int r = 0; r < R; r++) {
            float e = expf(scores_base[(long long)r * seq_len + t] - max_val[r]);
            scores_base[(long long)r * seq_len + t] = e;
            local_sum[r] += e;
        }
    }

    float sum_val[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        workspace[tid] = local_sum[r];
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) workspace[tid] += workspace[tid + s];
            __syncthreads();
        }
        sum_val[r] = workspace[0];
        __syncthreads();
    }

    // Normalize.
    for (int t = tid; t < seq_len; t += nthreads) {
        #pragma unroll
        for (int r = 0; r < R; r++) {
            scores_base[(long long)r * seq_len + t] /= sum_val[r];
        }
    }
    __syncthreads();

    // Phase 2: weighted sum of V. Each thread covers a subset of head_dim
    // for all R Q heads. V is loaded ONCE and used by all R Q heads.
    float* out_heads[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        out_heads[r] = out + (q_head_base + r) * head_dim;
    }

    for (int d = tid; d < head_dim; d += nthreads) {
        float val[R];
        #pragma unroll
        for (int r = 0; r < R; r++) val[r] = 0.0f;

        for (int t = 0; t < seq_len; t++) {
            float v = v_cache[(long long)t * n_kv_heads * head_dim + kv_h * head_dim + d];
            #pragma unroll
            for (int r = 0; r < R; r++) {
                val[r] += scores_base[(long long)r * seq_len + t] * v;
            }
        }

        #pragma unroll
        for (int r = 0; r < R; r++) {
            out_heads[r][d] = val[r];
        }
    }
}

extern "C" __global__ void attention_f32_gqa4(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale
) {
    attention_f32_gqa_body<4>(q, k_cache, v_cache, out, pos_buf,
                              n_heads, n_kv_heads, head_dim, max_seq, scale);
}
