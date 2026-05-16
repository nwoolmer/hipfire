// Hyper-Connections control-vector computation (V4F Phase 3).
//
//   c = W_fn · x_flat + base
//
// Shapes:
//   x_flat : [hidden · hc_mult]  (4096 · 4 = 16384 on V4F)  fp16
//   W_fn   : [n_ctrl, hidden · hc_mult]                     fp16
//   base   : [n_ctrl]                                        fp16
//   c      : [n_ctrl]                                        fp32 output
//
// V4F per-layer `hc_attn_fn` / `hc_ffn_fn` are shaped [24, 16384]
// → n_ctrl = 24, x_dim = 16384. The output `c` feeds the
// `sinkhorn_4x4` (after reshape interpretation per V4F paper).
//
// Strategy: one block per output entry (n_ctrl blocks).
// THREADS_PER_BLOCK threads each accumulate a slab of x_dim.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define THREADS_PER_BLOCK 256

__launch_bounds__(THREADS_PER_BLOCK)
extern "C" __global__ void hc_compute_control(
    const float*  __restrict__ x_flat,   // [x_dim] F32 (V4F residual convention)
    const __half* __restrict__ w_fn,     // [n_ctrl, x_dim] F16 (HFQ-converted)
    const __half* __restrict__ base,     // [n_ctrl] F16
    float*        __restrict__ c,         // [n_ctrl] F32
    int n_ctrl,
    int x_dim
) {
    const int ctrl = blockIdx.x;
    if (ctrl >= n_ctrl) return;
    const int tid = threadIdx.x;

    float acc = 0.0f;
    for (int d = tid; d < x_dim; d += THREADS_PER_BLOCK) {
        float xv = x_flat[d];
        float wv = __half2float(w_fn[ctrl * x_dim + d]);
        acc += xv * wv;
    }

    // Wave-shuffle reduction (wave32 on gfx11xx).
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    // Cross-wave reduction via shared memory.
    __shared__ float partial[THREADS_PER_BLOCK / 32];
    if (tid % 32 == 0) partial[tid / 32] = acc;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < THREADS_PER_BLOCK / 32; ++i) total += partial[i];
        c[ctrl] = total + __half2float(base[ctrl]);
    }
}
