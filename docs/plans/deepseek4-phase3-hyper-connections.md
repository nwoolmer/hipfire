# V4F Phase 3 — Hyper-Connections

**Status:** design. Implementation slots into V4F forward.
**Reference:** Zhu et al., "Hyper-Connections" (2024).

## Concept

Standard transformer: one residual stream per token. Each layer reads
the stream, applies a transform (attention or FFN), and adds the result
back. Hyper-Connections (HC) generalises this to `hc_mult = 4` parallel
residual streams that mix at every layer via a Sinkhorn-normalised
gating matrix. The mix lets early-layer signal compete with deep-layer
signal at every depth instead of being squashed by averaging.

## V4F-specific shapes

`config.json`:
- `hc_mult = 4` — number of parallel residual streams
- `hc_sinkhorn_iters = 20` — Sinkhorn iteration count for normalising
  the mixing matrix
- `hc_eps = 1e-6` — Sinkhorn epsilon to prevent division by zero

Per-layer tensor inventory (verified empirically from `model.safetensors`):
```
layers.L.hc_attn_base   [24]        F16   per-block bias
layers.L.hc_attn_fn     [24, 16384] F16   transform-input projection (24 × hidden·hc_mult)
layers.L.hc_attn_scale  [3]         F16   per-stream scaling factors
layers.L.hc_ffn_base    [24]        F16   per-block bias
layers.L.hc_ffn_fn      [24, 16384] F16   transform-input projection
layers.L.hc_ffn_scale   [3]         F16   per-stream scaling factors
```

Two HC blocks per layer: one wrapping the attention transform, one
wrapping the FFN. `hc_attn_fn[24, 16384]` projects the concatenated
4-stream residual (`4 × hidden = 4 × 4096 = 16384`) into a 24-element
control vector that drives the mixing matrix.

The head-level (logits) has its own HC trio:
```
hc_head_base   [4]      F16
hc_head_fn     [4, 16384] F16
hc_head_scale  [1]      F16
```

The `[4]` / `[4, 16384]` / `[1]` shapes there are smaller than the
per-layer `[24]` / `[24, 16384]` / `[3]`. Likely because the head is
collapsing 4 streams down to one for final logits — fewer mixing
degrees of freedom needed.

## Decomposition of the `24` (interpretation, to verify against paper code)

Hypothesis: `24 = 16 + 4 + 4`:
- `16` entries = the 4×4 mixing matrix between input and output streams
- `4` entries = additive bias on the transform's output before mixing
- `4` entries = scale on the transform's output before mixing

Or: `24 = 6 × 4`, six per-stream coefficients controlling Sinkhorn
seed-row weights or similar. The paper has the canonical
decomposition; defer interpretation to implementation.

## Forward sequence (one layer)

```
1. x = layer.residual_streams                # [4, hidden = 4096]
2. x_flat = concat(x, dim = stream)          # [hidden · 4 = 16384]
3. # HC-attn block:
    c_attn = layer.hc_attn_fn @ x_flat        # [24]
    c_attn += layer.hc_attn_base
    # Apply Sinkhorn normalisation to derive the 4×4 mixing matrix from c_attn
    A_attn = sinkhorn(c_attn, iters = 20, eps = 1e-6)  # [4, 4]
    transform_out = attention(rms_norm(x[0]))
                                              # one stream feeds attention
                                              # (which stream? per paper)
    # Mix: each output stream is A_attn[s, :] @ all_streams + scale * transform_out
    x = A_attn @ x  + scale[s] * transform_out  # [4, hidden]
4. # HC-ffn block: same shape, hc_ffn_* tensors
    c_ffn = layer.hc_ffn_fn @ concat(x)
    A_ffn = sinkhorn(c_ffn, 20, 1e-6)
    transform_out = moe_ffn(rms_norm(x[0]))
    x = A_ffn @ x  + scale[s] * transform_out
5. layer.residual_streams = x
```

## Kernel inventory

### New for Phase 3:

1. **`hc_compute_control`** — `c = W_fn @ x_flat + base`. Small dense
   GEMV: `[hidden · 4 = 16384, 24]`. One block, 24 threads, each
   accumulates one output entry. Trivial.

2. **`sinkhorn_normalise_4x4`** — Take a 4×4 (or 16-entry vector
   reshaped) and run 20 alternating row/column normalisation iters.
   - Fits entirely in registers (16 fp32 values).
   - One workgroup, 4 threads (one per row).
   - Communication via shared memory or warp-shuffle for row/col sums.
   - 20 iterations is the V4F default; param via `hc_sinkhorn_iters`.

3. **`hc_mix_4stream`** — `x_out = A @ x_in + scale_diag · transform_out`.
   - Shape: `x_in` is `[4, hidden]`, A is `[4, 4]`, `transform_out` is
     `[hidden]`.
   - Each output stream is a 4-vector dot with a column of A plus
     a single scalar-times-transform_out add.
   - One workgroup per `hidden` element batch, 4 threads (one per output
     stream).
   - Coalesced loads of x_in (cross-stream stride).

### Existing (reuse):
- `rms_norm` (any of the existing variants)
- The transform itself (attention or FFN) — V4F's attention is the
  Q-LoRA + K_idx-gather + main-attention path designed in Phase 2.

## State plumbing

`DeepseekV4State` already allocates per-layer state. Add:
```rust
pub struct ResidualStreams {
    /// `[hc_mult = 4, hidden = 4096]` per token.
    pub streams: GpuTensor,
}
```

Stored on `DeepseekV4State` directly (one per session, not per layer —
the streams ARE the residual, propagated through every layer).

## Cost analysis

Per layer:
- `W_fn @ x_flat`: 24 × 16384 = 393K FMA per layer — negligible.
- Sinkhorn (20 iters × 4×4): 320 FLOPs total — negligible.
- Mix: 4 × 4 × hidden + 4 × hidden = 80K FLOPs per layer — negligible.

The HC overhead is dwarfed by the attention and FFN cost. Memory
pressure is the only concern: 4× the residual storage. At hidden=4096
fp16 per stream per token, 4 streams cost 32 KB per token vs 8 KB for
single-stream. Long context (1M tokens × 4 streams × 4096 fp16) =
32 GB per residual snapshot. Hopefully only the current step is
materialised in full; past steps' residual is consumed by attention and
discarded.

## Open questions

- **Which stream feeds the transform's input?** Paper says one stream
  enters attention/FFN as a "view". Read `hc_*_scale[3]` to find out
  (scale of size 3 = (in_idx, out_idx_a, out_idx_b)? unclear).
- **Sinkhorn output shape mapping.** Does `c_*` (size 24) reshape to
  `[4, 4]` row-major or column-major or via the (16 + 4 + 4)
  decomposition above?
- **First-layer initialisation.** The paper has a special
  initialisation scheme for the 4 streams from the embedding.
  Probably `[embed, 0, 0, 0]` then HC mixes it forward.

## Implementation order

1. **A.** Add `ResidualStreams` to `DeepseekV4State`. Compile-only.
2. **B.** CPU reference impl of `sinkhorn_normalise_4x4` + cross-check
   against a tiny numpy script.
3. **C.** Write HIP kernel `sinkhorn_normalise_4x4.hip`. Test against B.
4. **D.** Write `hc_compute_control.hip` (trivial GEMV).
5. **E.** Write `hc_mix_4stream.hip`. Test against CPU reference.
6. **F.** Forward integration in V4F arch crate.
