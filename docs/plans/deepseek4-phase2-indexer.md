# V4F Phase 2 — Compressed-KV indexer (Lever 3)

**Status:** design + stub. Implementation is part of V4F forward bring-up.
**Depends on:** Phase 1 (FP4 ingest, in progress). Phase 4 MQ3-Lloyd MoE
indexed kernels already exist (re-use for the indexer GEMV).

## What V4F does at attention time

Each main-attention layer attends to:
1. **SWA window** — the last `sliding_window = 128` positions (raw KV).
2. **Indexer-selected positions** — `index_topk = 512` positions surfaced
   by a separate small attention surface that scores ALL past positions
   (after per-layer compression) for relevance to the current query.

This is what lets V4F run at 1M context: only 128 raw KV rows live in
the main attention path, plus 512 indexer-selected rows. The full long-
range attention is mediated through the compressed-KV indexer, which
stores positions at stride `compress_ratios[layer]`.

## Per-layer compress_ratios on the shipped V4F checkpoint

```
[0, 0, 4, 128, 4, 128, 4, 128, ..., 4, 128, 4, 0]
 │  │  ├─ middle 41 layers alternate 4 / 128 ─┤  │
 │  │                                            └── last layer: no compression
 ├──┴────── first 2 layers: no compression
```

- `0` — no compression, no indexer (full attention path).
- `4` — keep 1/4 of positions in compressed cache (~250K rows at 1M ctx).
- `128` — keep 1/128 of positions (~8K rows at 1M ctx).

## Indexer surface shape

- `index_n_heads = 64` (matches `num_attention_heads`)
- `index_head_dim = 128` (smaller than main `head_dim = 512`)
- `index_topk = 512`
- `compress_rope_theta = 160000` (separate RoPE base from main `rope_theta = 10000`)

Per layer with indexer active:
- `W_q_idx`: `[hidden = 4096, n_heads · head_dim = 8192]`  ≈ 33M params
- `W_k_idx`: `[hidden = 4096, n_heads · head_dim = 8192]`  ≈ 33M params

Over 41 active indexer layers: ~2.7B indexer params. At MQ3-Lloyd
(3.5 bpw) → ~1.2 GB of the V4F file is indexer weights. Acceptable.

## Kernel inventory needed

### Existing (reuse):
- `gemv_mq3g256_lloyd_*` — dense GEMV (already in tree, used by Phase 4)

### New for Phase 2:

1. **`indexer_q_gemv`** — `X @ W_q_idx → Q_idx`
   Shape: `[batch, n_idx_heads, idx_head_dim]`.
   Implementation: existing MQ3-Lloyd dense GEMV with output reshape.

2. **`indexer_k_gemv` (per-token)** — `X_token @ W_k_idx → K_idx`
   Computed once per token, appended to compressed cache at
   `position % compress_ratio == 0`. Reuse existing dense GEMV.

3. **`compressed_k_score`** — `Q_idx · K_idx_cache^T`
   For each head, dot-product all currently-cached compressed positions
   against the current Q. Output `scores[n_idx_heads, n_compressed]`.
   - One block per head.
   - Each thread handles a slab of compressed positions.
   - Reduction within warp to accumulate `score[h, p]`.
   - At 1M ctx with `ratio=128`: 64 heads × ~8K compressed = 512K dots.
   - At 1M ctx with `ratio=4`:   64 heads × ~250K = 16M dots.

4. **`indexer_top_k`** — pick top-512 indices per head from scores.
   - Two-pass: histogram + radix select, OR threshold-based partial sort.
   - Output shape `[n_idx_heads, 512]` of i32 position indices.

5. **`kv_gather`** — fetch raw K/V rows from the main KV cache at the
   union (or per-head) of the indexer's top-512 indices.
   - One warp per gather target row.
   - Coalesced read from KV cache + write to a scratch buffer the
     main attention path will read.

## State held by the forward pass

Per layer with indexer active:
```rust
struct IndexerLayerState {
    // Compressed K cache: stride = compress_ratios[layer].
    // Sparse-storage: only positions p where (p % ratio == 0) are stored.
    // Logical capacity: ceil(max_seq / ratio).
    k_idx_compressed: GpuTensor,  // [n_idx_heads, head_dim, n_compressed]

    // The indexer's RoPE applies compress_rope_theta to K_idx_compressed
    // entries at their original (un-divided) position index.

    // Scratch for current-step top-k indices.
    top_k_indices: GpuTensor,     // [n_idx_heads, 512] of i32
}
```

The main attention path:
```rust
struct MainAttentionLayerState {
    // Raw K/V for the SWA window. Bounded ring of last 128 positions.
    k_swa: GpuTensor,             // [n_kv_heads = 1, head_dim = 512, 128]
    v_swa: GpuTensor,             // [n_kv_heads = 1, head_dim = 512, 128]

    // Scratch for K/V rows gathered from indexer top-k positions.
    // Allocated once at max_seq, but indexer fills only n_idx active rows.
    k_gathered: GpuTensor,        // [n_kv_heads = 1, head_dim = 512, max_topk = 512]
    v_gathered: GpuTensor,        // [n_kv_heads = 1, head_dim = 512, max_topk = 512]
}
```

Note `n_kv_heads = 1` for V4F — extreme MQA. The 64 query heads share a
single KV head. That's why head_dim=512 (vs Qwen3.5's 128) — V4F packs
the KV-stream representation into a wider single head.

## Forward sequence (one decode step, one indexer-active layer)

```
1. x = hyper_connections_in(layer.in_residual_streams)
2. h_norm = rms_norm(x)
3. q = h_norm @ W_q_a @ W_q_b           # Q-LoRA bottleneck
   q = reshape(q, [n_heads = 64, head_dim = 512])
   apply tail-only RoPE on q[:, head_dim - qk_rope_head_dim:]
4. kv = h_norm @ W_kv                    # joint K + V projection
   k_raw, v_raw = split(kv, ...)         # both [n_kv_heads = 1, head_dim = 512]
   apply tail-only RoPE on k_raw[:, head_dim - qk_rope_head_dim:]
   swa_k[:, p % 128] = k_raw             # ring-buffer overwrite
   swa_v[:, p % 128] = v_raw
5. # Indexer Q/K
   q_idx = h_norm @ W_q_idx              # [n_idx_heads = 64, idx_head_dim = 128]
   apply compress_rope to q_idx
   if p % ratio == 0:
       k_idx = h_norm @ W_k_idx          # [n_idx_heads = 64, idx_head_dim = 128]
       apply compress_rope at original position p
       k_idx_cache[:, :, p / ratio] = k_idx
6. # Score + top-k
   scores = q_idx @ k_idx_cache^T        # [n_idx_heads, n_compressed]
   top_k_indices = top_k(scores, k = 512)   # may union across heads
   k_gathered = gather(main_k_cache, top_k_indices * ratio)
   v_gathered = gather(main_v_cache, top_k_indices * ratio)
7. # Main attention over (SWA window ∪ gathered top-k)
   attn_in_k = concat(swa_k, k_gathered) # [n_kv_heads, head_dim, 128 + 512]
   attn_in_v = concat(swa_v, v_gathered)
   attn_out = softmax(q @ attn_in_k^T / sqrt(d)) @ attn_in_v
8. # O-LoRA
   x_out = attn_out @ W_o_a @ W_o_b
9. hyper_connections_out(layer.out_residual_streams, x_out)
```

## Open questions

- **Top-k aggregation strategy:** per-head or union? V4F paper says
  union (so all heads share the same gathered KV rows). That's cheaper
  on gather bandwidth and lets the main attention path treat the
  gathered rows uniformly across heads.
- **Compressed-K storage layout:** stride-aware sparse `[n_heads,
  head_dim, n_compressed]` flat, or `[n_compressed, n_heads × head_dim]`
  contiguous? Scoring kernel design depends on this.
- **First-layer treatment:** layers 0, 1, 43 use `ratio = 0` → no
  indexer at all. Those layers do full attention over `swa_k` only.
  Validation: at 1M ctx the first two layers store only 128 positions
  → loss of all but the most recent context. The paper presumably
  compensates by routing global context through the indexer-active
  middle layers' top-k gathered rows.

## Implementation order

1. **A.** Stub `IndexerLayerState` and `MainAttentionLayerState` in
   `crates/hipfire-arch-deepseek4/src/deepseek4.rs`. Compile-only.
2. **B.** Write `compressed_k_score.hip` — the new kernel for step 6.
   Test against a tiny CPU reference.
3. **C.** Write `indexer_top_k.hip`. Test against a sort baseline.
4. **D.** Write `kv_gather.hip`. Test by gathering known indices.
5. **E.** Forward integration in V4F arch crate.
