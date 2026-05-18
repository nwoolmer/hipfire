// V4F SwiGLU with swiglu_limit clamp.
//
// Per upstream `inference/model.py:Expert.forward`:
//   if swiglu_limit > 0:
//     up   = clamp(up,   -limit, +limit)   (symmetric)
//     gate = clamp(gate,         +limit)   (one-sided, cap from above only)
//   out  = silu(gate) * up
//
// V4F config: swiglu_limit = 10.0. With FP4-quantized routed weights and
// 4096-dim input dots, unclamped values can reach magnitudes that drive
// silu(gate) into a regime where small numerical noise causes attention/
// FFN feedback attractors. The clamp is part of V4F's faithful math, not
// a numerical hack — `original_scores` of the router depend on these
// bounded magnitudes propagating through the residual stream.
//
// Drop-in replacement for `silu_mul_f32` when the layer has swiglu_limit
// > 0. Shared expert and each routed expert both benefit.

#include <hip/hip_runtime.h>

extern "C" __global__ void v4f_silu_mul_clamp_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float*       __restrict__ out,
    int n,
    float swiglu_limit
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    // V4F asymmetric clamp.
    if (g > swiglu_limit) g = swiglu_limit;
    if (u >  swiglu_limit) u =  swiglu_limit;
    if (u < -swiglu_limit) u = -swiglu_limit;
    float silu_g = g / (1.0f + __expf(-g));
    out[i] = silu_g * u;
}
