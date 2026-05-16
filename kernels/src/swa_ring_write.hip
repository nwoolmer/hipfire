// Write a single KV vector into the SWA ring buffer at slot pos%window.
//
// Cache layout: [n_kv_heads, head_dim, window].
// For position p, slot = p % window.
// Write k[d] -> cache[0, d, slot] for d in 0..head_dim.
// (V4F: n_kv_heads = 1, head_dim = 512.)

#include <hip/hip_runtime.h>

extern "C" __global__ void swa_ring_write_f32(
    const float* __restrict__ kv,          // [head_dim]
    float*       __restrict__ cache,       // [n_kv_heads, head_dim, window]
    int n_kv_heads,
    int head_dim,
    int window,
    int slot
) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    for (int h = 0; h < n_kv_heads; ++h) {
        // For tied K=V single head: kv is shared across all heads.
        cache[(h * head_dim + d) * window + slot] = kv[d];
    }
}
