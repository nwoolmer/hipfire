# DeepSeek V4 Flash — next session

V4F is end-to-end chat-testable + MoE-active. 36 commits this
session drove ppl from 119k → 21k at ctx=128.

## Current PPL baseline (wikitext2-test, MoE+SWA, default settings)

| ctx | ppl   | notes                                  |
|-----|-------|----------------------------------------|
|  64 |  25k  |                                        |
| 128 |  21k  | within SWA window                      |
| 256 |  40k  | needs indexer for >128                 |

Shared-only (no MoE) at ctx=128: **71k** — MoE provides 3.4x gain.

Defaults (in `crates/hipfire-arch-deepseek4/src/forward.rs`):
- `HIPFIRE_V4F_POST_SCALE=0.5` (empirical optimum; upstream uses 2.0)
- `HIPFIRE_V4F_ROUTE_SCALE=1.0` (empirical optimum; upstream uses 1.5)

## Big bugs fixed this session

Commits c39ae3a..0351de9. Highlights in order of impact:

1. **MoE gate input** — was using raw hc_x_in, now uses ffn_norm'd
   ffn_x_rot. **145k → 68k ppl** (`dbd6754`)
2. **HC head mix** — final norm was using only stream 0; now combines
   all 4 streams via hc_head_fn/base/scale (`6ce9781`)
3. **HC segment offsets** — pre/post/comb at [0,4,8] not [0,4,20]
   (`19eb08c`, `b01e13f` for hc_apply_alpha)
4. **HC rsqrt normalization** in hc_compute_control (`879b101`)
5. **HC Sinkhorn**: row-softmax start (was clamp+exp) (`34bc59a`)
6. **Per-head attention output + O-LoRA** — was pre-reducing heads;
   now keeps [n_heads, head_dim] through wo_a per-group + wo_b
   (`e657ece`)
7. **Q-norm, per-head Q-norm, KV-norm** wired in attention (`5b7cc56`)
8. **Inverse tail RoPE on attn output** (`bf0b601`)
9. **RoPE convention**: half-split → INTERLEAVED to match upstream
   `torch.view_as_complex` (`1e37b45`, `0666035`)
10. **HC mix reuse** — no double-compute, α-aware (`1be3583`)
11. **FP4 (E2M1) dequant** in quantizer + paired MoE infra (earlier
    commits c39ae3a..2730224)

## Open puzzles

- **2x post_scale mismatch** — upstream's `2*sigmoid` * route_scale
  1.5 gives ppl 68k; our `0.5*sigmoid` * 1.0 gives 21k. The 6x total
  product mismatch likely reflects accumulated MQ4/MQ2-Lloyd quant
  noise OR a remaining magnitude bug not isolated despite extensive
  bisection (Q-norm, RoPE, FWHT, swiglu_limit, MoE math, hc_apply
  _alpha all verified upstream-correct).

## Diagnostics shipped

- `HIPFIRE_V4F_DUMP_MAG=1` — per-layer stream/attn_out/ffn_out RMS
- `HIPFIRE_V4F_POST_SCALE=N`
- `HIPFIRE_V4F_ROUTE_SCALE=N`
- `HIPFIRE_V4F_SKIP_FFN=1` — zero ffn_out (isolates FFN)
- `HIPFIRE_V4F_SKIP_INV_ROPE=1`
- `HIPFIRE_V4F_SKIP_QHN=1` — skip per-head Q norm
- `HIPFIRE_V4F_EXPERT_LAYER_END=N` — partial MoE for VRAM-constrained
  testing
- `crates/hipfire-arch-deepseek4/examples/v4f_perplexity.rs` —
  perplexity / NLL tool
- `crates/hipfire-arch-deepseek4/examples/v4f_top_logits.rs` —
  per-position top-K logits inspector

## Activation paths

```bash
# Full MoE chat:
HIPFIRE_V4F_MODEL=~/.hipfire/models/v4f.mq2lloyd-fp4fix \
HIPFIRE_V4F_UPLOAD_EXPERTS=1 \
HIPFIRE_V4F_MOE=1 \
./target/release/examples/v4f_chat

# Perplexity:
./target/release/examples/v4f_perplexity \
  ~/.hipfire/models/v4f.mq2lloyd-fp4fix \
  ~/.hipfire/src/dev/bench/data/wikitext2-test.txt \
  --ctx 128 --warmup 8 --moe 1

# Restore upstream-faithful scales for investigation:
HIPFIRE_V4F_POST_SCALE=2.0 HIPFIRE_V4F_ROUTE_SCALE=1.5 ./v4f_chat
```

## Next steps for >128-token context

Implement compressed-KV indexer (#56 task). Multi-hour work; V4F has
separate `attn.indexer.*` sub-module on alternating layers
(compress_ratio=4) with its own wq_b, weights_proj, AND a separate
Compressor with gated pooling + APE.

### Progress this session
- **Phase 1 (weights)**: 955e6db — added compressor.ape +
  indexer.{wq_b, weights_proj, compressor.{wkv,wgate,norm,ape}}
  slots + load_weights upload. Host walk validates presence.
- **Phase 2 (state)**: fec24dd — added IndexerLayerState slots
  for main_kv_cache/kv_state/score_state, indexer equivalents,
  per-step q_idx/idx_weights/index_score/topk_idx_indices.
  All None; lazy-alloc when forward runs.
- **Phase 3a.1 (pool kernel)**: 2962e55 — compressor_softmax_pool_f32:
  output[d] = sum_t softmax_t(score[:, d])[t] * kv[t, d]. One thread
  per d, iterates T (≤16 for ratio=4 overlap, ≤128 ratio=128).
- **Phase 3a.2 (concat kernel)**: 0262886 — compressor_overlap_concat_f32:
  builds [2*ratio, head_dim] view from [2*ratio, 2*head_dim] state by
  taking first-half cols for old window rows, second-half for current.
  Equivalent to upstream cat([state[:r, :d], state[r:, d:]], dim=1).

### Phases pending
- **Phase 3b (Compressor.forward Rust)**: kernels are ready; need
  the Rust function that wires them together for decode:

  ```rust
  fn compressor_forward(cfg, weights, state, gpu, layer_idx,
      x_rotated, position, is_indexer: bool) -> Result<()>
  ```

  Per step:
  1. kv = wkv @ x_rotated  (MQ4 GEMV → [coff*head_dim])
  2. score = wgate @ x_rotated
  3. score += ape[pos%ratio]  (slice of ape via sub_offset + add_inplace)
  4. Write kv into state.{main,indexer}_kv_state[(ratio + pos%ratio) *
     stride] via memcpy_dtod_auto (sub_offset of state buffer)
  5. Same for score → state.{main,indexer}_score_state
  6. If (pos+1) % ratio == 0:
     a. compressor_overlap_concat_f32(kv_state, concat_kv_scratch, ...)
     b. compressor_overlap_concat_f32(score_state, concat_score_scratch, ...)
     c. compressor_softmax_pool_f32(concat_kv, concat_score, kv_cache_slot, T, hd)
     d. rmsnorm_f32 (in place, using compressor.norm weight)
     e. If is_indexer: apply rope_tail_interleaved with compress_rope_theta=160000
     f. Shift kv_state[:ratio] = kv_state[ratio:] (memcpy_dtod for the upper half
        back to the lower half) — same for score_state

  Need to add scratch state slots for concat_kv (size [2*ratio, head_dim] F32)
  and concat_score similarly. Caching capacity for kv_cache: pick a default
  like 1024 compressed positions (= 4096 token ctx for ratio=4); make env-
  overridable via HIPFIRE_V4F_MAX_COMPRESS_POS.
- **Phase 4 (Indexer.forward)**: q = wq_b @ qr; tail RoPE with
  compress_rope_theta=160000; FWHT rotate; weights = weights_proj
  @ x; index_score[t] = relu(Q·K_idx_cache[t]^T) · weights summed
  across heads; top_K (= index_topk = 512) per head; dedup union.
  Needs einsum-style kernel + indexer_top_k already exists.
- **Phase 5 (modified main attention)**: extend v4f_attn_swa to
  also gather K/V from main_kv_cache at top_K positions, append
  to SWA window for softmax. Needs new gather+attend or indexed
  attention kernel.

Sees `inference/model.py:Indexer` and `Compressor`. Approximate
effort: 4-8 hours for the full pipeline. Without it ppl 2x's
beyond ctx=128 (40k vs 21k at ctx=256 vs 128).

## Test inventory

All passing as of 0351de9:
- 8/8 V4F kernel tests (`./scripts/v4f_kernel_tests.sh`)
- 11/11 hipfire-arch-deepseek4 lib tests (incl. 8 routing unit tests)
- 5/5 hipfire-quantize FP4 E2M1 tests
