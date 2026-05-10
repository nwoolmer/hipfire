#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Batched HFQ1G128 embedding lookup. One launch dequantizes embeddings
// for N tokens into a contiguous [N × dim] FP32 buffer. Grid uses
// blockIdx.x for token index; mirrors embedding_lookup_q8_batched and
// embedding_lookup_hfq4g256_batched. Big-block work (whole row of dim
// elements) — not the small-kernel pattern that regresses on gfx1151.
//
// Saves N-1 launches per prefill chunk for HFQ1G128 embedding format
// (Bonsai: 255 launches saved at chunk_n=256).

extern "C" __global__ void embedding_lookup_hfq1g128_batched(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,           // [n_tokens × dim]
    const int* __restrict__ tokens,        // [n_tokens]
    int n_tokens,
    int dim
) {
    const int b = blockIdx.x;
    if (b >= n_tokens) return;
    const int tid = threadIdx.x;
    const int token_id = tokens[b];

    const int groups_per_row = dim / 128;
    const int bytes_per_row = groups_per_row * 18;
    const unsigned char* row = table + (size_t)token_id * bytes_per_row;
    float* out_row = output + (long long)b * dim;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        int group = elem / 128;
        int within = elem % 128;
        const unsigned char* gptr = row + group * 18;

        _Float16 d_h = *(const _Float16*)gptr;
        float d = (float)d_h;

        unsigned char byte_val = gptr[2 + (within >> 3)];
        unsigned char bit = (byte_val >> (within & 7)) & 1;
        out_row[elem] = bit ? d : -d;
    }
}
