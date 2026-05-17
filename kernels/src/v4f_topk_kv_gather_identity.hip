// V4F mixed-attention identity gather (used for ratio=128 compressor layers
// without an indexer). Copies main_kv_cache[0..K, :] into the column-slab
// gathered_k[:, 0..K] without any topk-index indirection.
//
// Input layout : [N_compressed, head_dim] row-major
// Output layout: [head_dim, out_stride] — write columns [0, K).

#include <hip/hip_runtime.h>

extern "C" __global__ void v4f_topk_kv_gather_identity_f32(
    const float* __restrict__ kv_cache,  // [N_compressed, head_dim]
    float*       __restrict__ out,       // [head_dim, out_stride]
    int K,
    int head_dim,
    int out_stride
) {
    const int k = blockIdx.x;
    const int d = threadIdx.x;
    if (k >= K || d >= head_dim) return;
    out[d * out_stride + k] = kv_cache[k * head_dim + d];
}
