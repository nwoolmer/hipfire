#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HFQ1G128 single-token embedding lookup.
// Block layout: 18 B per 128-elt group (FP16 d + 16 B packed bits, LSB-first).
// bit=1 → +d, bit=0 → -d. Mirror of embedding_hfq4g128.hip with 1-bit unpack.

extern "C" __global__ void embedding_hfq1g128(
    const unsigned char* __restrict__ table,
    float* __restrict__ output,
    int token_id, int dim
) {
    const int tid = threadIdx.x;
    const int groups_per_row = dim / 128;
    const int bytes_per_row = groups_per_row * 18;
    const unsigned char* row = table + (size_t)token_id * bytes_per_row;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        int group = elem / 128;
        int within = elem % 128;
        const unsigned char* gptr = row + group * 18;

        _Float16 d_h = *(const _Float16*)gptr;
        float d = (float)d_h;

        unsigned char byte_val = gptr[2 + (within >> 3)];
        unsigned char bit = (byte_val >> (within & 7)) & 1;
        output[elem] = bit ? d : -d;
    }
}
