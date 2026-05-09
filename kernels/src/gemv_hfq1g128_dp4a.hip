#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// HFQ1G128 dp4a GEMV. Mirrors PrismML's vec_dot_q1_0_q8_1 math exactly:
//   - Activations pre-quantized to Q8_1 (block_q8_1_mmq, 144 B / 128 K-elts).
//   - Weights stay 1-bit packed; lane unpacks 4 weight bits → 4 ±1 INT8s.
//   - dp4a (v_dot4_i32_i8) accumulates INT32 sumi per chunk (32 elements).
//   - Final: total += d_w * d_a[chunk] * (float)sumi  per group, per chunk.
//
// Per-lane mapping (wave32):
//   - lane handles weights at group offsets [tid*4 .. tid*4+3] (4 weights = 4 bits = 1 nibble)
//   - chunk_idx = tid / 8 (4 chunks of 8 lanes per group)
//   - 1 dp4a per K-step → INT32 result, scaled by d_w * d_a[chunk_idx] as FP32
//   - Per-lane FP32 accumulator across K; 32-lane reduce at end → row total
//
// Whether this beats the FP-direct path on this APU is an empirical
// question — it CAN'T be a memory-bandwidth win for decode (activations
// are L2-cached) but might help via register pressure / occupancy /
// compute-overlap. See bench_gemv_hfq1g128.

#define QK8_1 32

struct block_q8_1_mmq {
    half2 ds4[4];        // 4 (d, s) FP16 pairs, one per 32-elt chunk
    int8_t qs[4 * QK8_1]; // 4 chunks × 32 INT8 contiguous along K
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq1g128_dp4a(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,    // K/128 blocks
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 128;
    const long long row_stride = (long long)groups_per_row * 18;
    const char* row_ptr = A + (long long)row * row_stride;

    const int byte_off = tid >> 1;            // 2 lanes share a weight byte
    const int bit_shift = (tid & 1) << 2;     // lower vs upper nibble
    const int chunk_idx = tid >> 3;           // 4 lanes/chunk × 8 chunks? no: 8 lanes/chunk × 4 chunks
    const int pos_in_chunk_bytes = (tid & 7) * 4;  // byte offset within chunk's 32 INT8s

    float row_acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + (long long)g * 18;

        // Weight scale (group-level, broadcast across all 32 lanes).
        _Float16 d_h = *(const _Float16*)gptr;
        float d_w = (float)d_h;

        // Weight bits → 4 ±1 INT8 packed into one int32.
        unsigned char byte_val = (unsigned char)gptr[2 + byte_off];
        unsigned char nib = (byte_val >> bit_shift) & 0xF;
        int b0 = (nib & 0x1) ? 1 : -1;
        int b1 = (nib & 0x2) ? 1 : -1;
        int b2 = (nib & 0x4) ? 1 : -1;
        int b3 = (nib & 0x8) ? 1 : -1;
        int w_packed = (b0 & 0xFF) | ((b1 & 0xFF) << 8)
                     | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);

        // Activation: read 4 INT8s for this lane's chunk position.
        const block_q8_1_mmq* xb = &Xq[g];
        int act_int = ((const int*)&xb->qs[chunk_idx * QK8_1 + pos_in_chunk_bytes])[0];

        // Per-chunk activation scale (FP16 .x of half2).
        _Float16 d_a_h = ((const _Float16*)&xb->ds4[chunk_idx])[0];
        float d_a = (float)d_a_h;

        // INT32 sumi for this lane's 4 weights × 4 acts. Compiler
        // emits v_dot4_i32_i8 on gfx10+ when dot1-insts feature is
        // enabled; otherwise falls back to 4 separate INT muladds.
        // gfx1151 (RDNA 3.5) has the instruction natively.
        int a0 = (int)(int8_t)((act_int      ) & 0xFF);
        int a1 = (int)(int8_t)((act_int >>  8) & 0xFF);
        int a2 = (int)(int8_t)((act_int >> 16) & 0xFF);
        int a3 = (int)(int8_t)((act_int >> 24) & 0xFF);
        int sumi = b0 * a0 + b1 * a1 + b2 * a2 + b3 * a3;

        // Apply group-level d_w × per-chunk d_a, accumulate FP32.
        row_acc += d_w * d_a * (float)sumi;
    }

    // 32-lane reduction → lane 0 has the row dot product.
    for (int offset = 16; offset > 0; offset >>= 1)
        row_acc += __shfl_down(row_acc, offset);

    if (tid == 0) y[row] = row_acc;
}
