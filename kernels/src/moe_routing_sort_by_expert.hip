#include <hip/hip_runtime.h>

// Sort MoE routings by expert_id so that subsequent grouped GEMVs read
// each expert's weight slab once with high L2/IC cache reuse.
//
// Input  : topk_indices [N = B * K_TOP] — each entry is an expert id in [0, n_exp).
// Output : sorted_b, sorted_krank, sorted_expert [N] — permutation of the
//          flat (b, krank) routing pairs grouped by expert id.
//          expert_starts [n_exp + 1] — prefix-sum start offsets.
//
// Algorithm: counting sort. Single block (size = n_exp threads).
//   1. Zero shared counts.
//   2. Atomic histogram of expert_id frequencies into shared counts.
//   3. Exclusive prefix-sum on counts (single-thread sequential scan;
//      n_exp = 256 ≪ 1 wave's reduction cost).
//   4. Reset counts to 0 and scatter each routing into its bucket slot.
//
// Notes:
//   - n_exp must equal block size (or smaller); we assume 256 for V4F.
//   - Total work: O(N + n_exp). N ≤ max_batch * K_TOP = 64 * 6 = 384 for V4F.
//   - Single launch per layer (replaces a d2h + CPU sort + h2d cycle).

__launch_bounds__(256, 4)
extern "C" __global__ void moe_routing_sort_by_expert(
    const int* __restrict__ topk_indices,   // [N]
    int*       __restrict__ sorted_b,       // [N]
    int*       __restrict__ sorted_krank,   // [N]
    int*       __restrict__ sorted_expert,  // [N]
    int*       __restrict__ expert_starts,  // [n_exp + 1]
    int B, int K_TOP, int n_exp
) {
    const int tid = threadIdx.x;
    const int N = B * K_TOP;

    extern __shared__ int counts[];   // [n_exp]

    // Phase 1: zero counts.
    if (tid < n_exp) counts[tid] = 0;
    __syncthreads();

    // Phase 2: atomic histogram.
    for (int i = tid; i < N; i += blockDim.x) {
        int e = topk_indices[i];
        if (e >= 0 && e < n_exp) {
            atomicAdd(&counts[e], 1);
        }
    }
    __syncthreads();

    // Phase 3: exclusive prefix-sum (single-thread sequential).
    // n_exp = 256 ⇒ ~256 adds, trivial vs. the GEMV work that follows.
    if (tid == 0) {
        int sum = 0;
        for (int e = 0; e < n_exp; e++) {
            expert_starts[e] = sum;
            sum += counts[e];
        }
        expert_starts[n_exp] = sum;
    }
    __syncthreads();

    // Phase 4: reset counts (will reuse as per-bucket cursor).
    if (tid < n_exp) counts[tid] = 0;
    __syncthreads();

    // Phase 5: scatter routings into sorted slots.
    for (int i = tid; i < N; i += blockDim.x) {
        int e = topk_indices[i];
        if (e < 0 || e >= n_exp) continue;
        int slot = atomicAdd(&counts[e], 1);
        int dst = expert_starts[e] + slot;
        sorted_b[dst] = i / K_TOP;
        sorted_krank[dst] = i % K_TOP;
        sorted_expert[dst] = e;
    }
}
