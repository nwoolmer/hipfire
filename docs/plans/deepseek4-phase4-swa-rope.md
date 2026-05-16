# V4F Phase 4 — SWA cache + tail-only RoPE

**Status:** design.
**Depends on:** Phase 2 indexer (consumes SWA cache as one of two
attention input sources).

## SWA window (sliding-window attention)

`config.json` → `sliding_window = 128`.

The main attention path attends to the last 128 raw KV positions plus
the indexer's `index_topk = 512` gathered rows. Together: ~640 KV rows
per attention step regardless of how long the context is.

### Implementation

Bounded ring buffer per layer:
```rust
pub struct SwaCache {
    /// `[n_kv_heads = 1, head_dim = 512, window = 128]`
    pub k: GpuTensor,
    pub v: GpuTensor,
    /// Tokens written into the ring (monotonic, wraps modulo window).
    pub n_written: u64,
}
```

On each decode step at position `p`:
```
slot = p % window
k[:, :, slot] = k_raw_current
v[:, :, slot] = v_raw_current
```

The main attention kernel reads K/V as a logical sequence of `min(p+1,
window)` positions. The ring's start is `(slot + 1) mod window` after
the first wrap; before the first wrap it's `0..slot+1`.

### KV cache memory cost

V4F has 43 main layers + 1 MTP = 44. With `n_kv_heads = 1`,
`head_dim = 512`, `window = 128`:

Per layer: `2 (K + V) × 1 × 512 × 128 × 2 (fp16) = 256 KB`
Total: 44 layers × 256 KB = **11 MB for the entire SWA cache.**

Compare to Qwen3.6-35B-A3B's 40 layers × 8 heads × 128 head_dim × 4096
ctx × 2 fp16 × 2 (K+V) = ~840 MB at 4K context. V4F's KV layout is
~75× smaller per layer, ~50× smaller across the model (despite having
~more layers). The compressed-KV indexer's cache eats the difference.

## Tail-only RoPE

V4F applies rotary positional encoding only to the **last 64 dims** of
each 512-dim head (`qk_rope_head_dim = 64`). The first 448 dims are
plain Q·K matmul without rotation.

Standard hipfire RoPE kernels apply rotation to the full head_dim.
Need a `rope_tail` variant that takes a `rope_dim_offset` and applies
the rotation only to `q[:, offset..offset + qk_rope_head_dim]`. The
rest of the head is untouched.

### Variants needed

- Main attention K/V: `rope_tail(theta = rope_theta = 10000,
  offset = head_dim - qk_rope_head_dim = 448, dim = 64)`
- Indexer Q/K: `rope_tail(theta = compress_rope_theta = 160000,
  offset = idx_head_dim - qk_rope_head_dim = ???, dim = 64)`

Wait — `qk_rope_head_dim = 64` is defined relative to `head_dim = 512`
in the config. The indexer head_dim is `index_head_dim = 128`. Does the
same 64-dim tail apply to the indexer, or does the indexer use full-dim
RoPE? Verify against V4F's reference code; the config only documents
one `qk_rope_head_dim` field. Probably the indexer uses full
`compress_rope_theta` rotation over all 128 dims; the "tail-only"
applies to main attention only.

### YaRN scaling

`rope_scaling`:
- `factor = 16` (16× context extension over the original 65536 → 1M)
- `original_max_position_embeddings = 65536`
- `beta_fast = 32`, `beta_slow = 1`

Standard YaRN math: scale the RoPE frequency table by interpolating
between "frequency ÷ factor" (slow) and "frequency" (fast) based on
how many positions fit within original_max. The hipfire codebase
already implements YaRN for Qwen3.5 variants — reuse the same routine,
just feed V4F's params.

### Implementation pattern

```hip
__global__ void rope_tail_apply(
    float* __restrict__ q,        // [n_heads, head_dim]
    int head_dim,
    int rope_offset,              // dims to skip before applying rotation
    int rope_dim,                 // dims to rotate (must be even)
    int position,                 // absolute token position
    float rope_theta,
    // optional YaRN params:
    float yarn_factor,
    int yarn_original_max,
    float yarn_beta_fast,
    float yarn_beta_slow
) {
    int h = blockIdx.x;
    int i = threadIdx.x * 2;
    if (i >= rope_dim) return;
    int base = h * head_dim + rope_offset + i;
    float inv_freq = compute_inv_freq_yarn(
        i, rope_dim, rope_theta, yarn_factor, yarn_original_max,
        yarn_beta_fast, yarn_beta_slow
    );
    float ang = position * inv_freq;
    float c = cosf(ang);
    float s = sinf(ang);
    float a = q[base];
    float b = q[base + 1];
    q[base    ] = a * c - b * s;
    q[base + 1] = a * s + b * c;
}
```

The `head_dim - rope_offset` first dimensions are untouched. The Q·K
matmul naturally combines them as straight dot product.

## Kernel inventory

### New:

1. **`rope_tail_apply.hip`** — partial-dim RoPE with `rope_offset`.
2. **`swa_write_ring.hip`** — write current step's K/V into the ring
   slot. Optimised over 4 KB per layer per step — trivial bandwidth.

### Modified:

3. **Main attention kernel** — read K/V from a logical sequence of
   `min(p+1, window)` rows that wraps. Existing flash-attention-style
   kernels handle non-power-of-two sequence lengths; the wrap math is
   a kernel-launch parameter, not a kernel modification.

## Open questions

- **Wrap-aware FlashAttention:** does the existing FlashAttn kernel
  handle a "start offset" parameter? Probably needs a one-line addition
  to thread the starting row index through to the K/V load loop.
- **Indexer RoPE dim:** confirm against paper code whether the indexer
  uses full-dim or tail-only rotation. The config field
  `qk_rope_head_dim = 64` may apply only to main, not indexer.

## Implementation order

1. **A.** Add `SwaCache` to `DeepseekV4State`. Compile-only.
2. **B.** Write `rope_tail_apply.hip`. Test against existing full-dim
   RoPE on a stub case where `rope_offset = 0` and `rope_dim = head_dim`
   (must match the existing kernel byte-for-byte).
3. **C.** Wrap-aware FlashAttn kernel parameter — add `start_pos` arg.
4. **D.** Forward integration in V4F arch crate.
