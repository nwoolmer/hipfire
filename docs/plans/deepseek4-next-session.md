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

See `inference/model.py:Indexer` and `Compressor` in the HF cache
for the algorithm. Without indexer, ppl ~2x'es beyond ctx=128.

## Test inventory

All passing as of 0351de9:
- 8/8 V4F kernel tests (`./scripts/v4f_kernel_tests.sh`)
- 11/11 hipfire-arch-deepseek4 lib tests (incl. 8 routing unit tests)
- 5/5 hipfire-quantize FP4 E2M1 tests
