// V4F mHC α-scaling post-step.
//
// hc_compute_control writes c = (X · W_fn) + base [24]. The paper
// (DeepSeek_V4.pdf §2.2 eqs 3-5) actually specifies:
//   Ã = α^pre  · (X · W^pre)  + S^pre
//   B̃ = α^res  · (X · W^res)  + S^res
//   C̃ = α^post · (X · W^post) + S^post
//
// where α^* are per-block learnable scalars stored in hc_*_scale[3].
// This kernel post-rescales c so each segment has the right α:
//
//   c_new[i] = α[seg(i)] · (c[i] - base[i]) + base[i]
//            = α[seg(i)] · c[i] + (1 - α[seg(i)]) · base[i]
//
// Segments (V4F): [0..4]=pre, [4..20]=res, [20..24]=post.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void hc_apply_alpha(
    float*        __restrict__ c,        // [24] fp32 in/out
    const __half* __restrict__ alpha,    // [3]  fp16 (V4F: stored as F16 in HFQ)
    const __half* __restrict__ base      // [24] fp16
) {
    if (blockIdx.x != 0) return;
    const int i = threadIdx.x;
    if (i >= 24) return;

    int seg;
    if      (i < 4)  seg = 0;
    else if (i < 20) seg = 1;
    else             seg = 2;

    float a = __half2float(alpha[seg]);
    float b = __half2float(base[i]);
    float ci = c[i];
    c[i] = a * ci + (1.0f - a) * b;
}
