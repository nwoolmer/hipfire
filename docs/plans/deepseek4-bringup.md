# DeepSeek V4 Flash bring-up plan

**Target:** Run V4F (officially 671B-MoE, 686B params with MTP) under hipfire
at <100 GB so it fits on Strix Halo–class machines (96 GB unified).

**Status (2026-05-15):** scaffold landed (`crates/hipfire-arch-deepseek4`,
`arch_id = 7`, full Config parser). Forward and weight loading
intentionally stubbed.

## Why V4F is interesting

- Largest sparse-MoE on the public side that fits in <100 GB at 2-bit:
  256 routed experts (top-6) + 1 shared, gives ~80 % of params behind
  sparse routing → 2-bit MoE-routed-experts pattern that worked on
  Qwen3.6-35B-A3B applies directly.
- 1M context with YaRN + per-layer compressed-KV (ratio-4 / ratio-128
  alternating) + raw SWA-128. Validates the multi-scale attention
  arithmetic without an enormous training run on our side.
- Hyper-Connections is a new residual-mix pattern (4 streams + Sinkhorn-
  iterated mixing matrix). One of the smaller architectural-novelty
  surfaces to port; small enough to be tractable in a focused PR.

## Levers identified for the size target

Three orthogonal levers to bring V4F under 100 GB while preserving
useful quality:

| Lever | Scope | Status | Estimated saving |
|---|---|---|---|
| **1** | lm_head Q4_K_M vs Q8 | Not yet implemented | ~230 MB (4096 × 129280 × 0.4375 vs 1.0) — small |
| **2** | GPTQ-LDLQ-style Lloyd on gate_up MQ2 | **DONE** (commit 48c1f4c) | Quality, not size — opens path to drop down=MQ3 |
| **3** | Compressed-KV indexer at MQ3-Lloyd | Pending — needs new kernel shape | KV-cache size at long context |

**Lever 1 sizing reality:** the lm_head saving is ~230 MB, not the
~2 GB originally hypothesized. Embeddings (the other ~530 MB Q8 surface
at hidden × vocab) could also drop to Q4_K but that risks input-side
loss of distinction between near-orthography token IDs at greedy decode.
Defer Lever 1 until size target is provably tight.

**Lever 2 follow-up:** `mq4-mq2lloyd-gptq-all` format (this commit)
extends GPTQ to BOTH gate_up AND down — directly tests whether
sequential error feedback closes the quality gap enough to drop the
down=MQ3 antirez recipe. Saving if it works: ~30 % more on routed-
expert size.

## Phases

### Phase 0 (DONE — 2026-05-15): arch crate scaffold

- `hipfire-arch-deepseek4` workspace member.
- `Architecture` trait impl: arch_id=7, name="deepseek4",
  `config_from_hfq` parses the full V4F metadata shape (Hyper-Connections
  params, indexer params, per-layer compress_ratios, YaRN scaling,
  MTP head, hash routing). `load_weights` and forward stubbed.

### Phase 1: FP4 weight ingest in `hipfire-quantize` (~5 days)

V4F ships routed experts as FP4 E4M3 with UE8M0 block scales (128×128).
Conversion path:

1. Add FP4 E4M3 → f32 dequant in `hipfire-quantize/src/main.rs`.
2. Read `quantization_config.weight_block_size` (128×128) from
   V4F config.json; per-(128×128)-tile apply the UE8M0 scale during
   dequant.
3. Walk all 46 safetensors shards; route each tensor by name to
   the right MQ-family quantizer (routed experts → MQ2-Lloyd-GPTQ
   or MQ3-Lloyd depending on Phase 1/2 split decision; attention
   → MQ4-K-map; norms → F16; lm_head → Q8 (or Q4_K if Lever 1
   ships); embeddings → Q8).
4. Emit HFQ file with `arch_id = 7` and a `DeepseekV4`-shaped
   metadata JSON the `DeepseekV4Config::from_hfq` parser accepts.

**Risk:** UE8M0 is an unusual scale format (8-bit exponent, no
mantissa). Confirm dequant math against an upstream `transformers`
forward on a single tensor before quantizing the whole model.

### Phase 2: Indexer kernel (compressed-KV, MQ3-Lloyd, Lever 3) (~3 days)

Compressed-KV indexer is a 64-head × 128-dim attention surface that
scores ALL past tokens (compressed at per-layer stride) and gates
which positions the main attention attends to. Top-512 per layer.

Shape: per layer with `compress_ratio[l] > 0`, build a compressed-KV
cache at stride `compress_ratio[l]` (so a layer with ratio=128 keeps
1/128th of positions as KV-cache rows). Score each row with `Q · K^T`,
take top-512, gather corresponding raw-KV rows from the SWA window
or — for outside-SWA positions — fetch from the compressed cache
itself (read the full V4F paper / antirez code for the exact
gather-vs-scattter pattern).

For 2-bit hipfire-native fit: the K/V projection weights of the
indexer ship as MQ3-Lloyd (3.5 bpw) since the indexer dim (128) is
smaller and the scores feed top-k selection, not the residual stream
— less precision-sensitive than the main attention.

Kernel: GEMV-style score, top-512 selection, gather. The dense
MQ3-Lloyd GEMV (Phase 4 of `project_mq2_lloyd_moe_rescue.md`) covers
the score part; top-512 and gather are new.

### Phase 3: Hyper-Connections (~3 days)

4 residual streams indexed by token. Each layer mixes the streams via
a learned gating matrix that's normalised with Sinkhorn iteration
(20 iters per layer per token in V4F). Replaces the single
`forward_residual` of standard transformers.

Kernel: small dense 4×4 gating matmul + 20-iteration Sinkhorn
normalisation. Both are tiny — probably one kernel covers all
of it. The bigger lift is the residual-state plumbing in
`forward_*` (every layer reads/writes 4 residual buffers, not 1).

### Phase 4: SWA cache + tail-only RoPE (~2 days)

- Bounded ring of the last 128 tokens for main attention. Reuse
  the dense-attention SWA infrastructure if present; otherwise
  port from `crates/hipfire-arch-qwen35`'s sliding-window path.
- Tail-only RoPE: only the last 64 dims (of 512 head_dim) get RoPE;
  the rest is straight Q · K matmul. Adjust the RoPE-application
  kernel to take a slice offset.

### Phase 5: MTP head (~1 day)

Optional. The MTP head can act as a built-in speculative drafter
(predicts the next-next token in parallel with the next one).
Hipfire's existing DFlash spec-decode infrastructure can plug
into it. Defer until base forward works.

## Quality + size validation gates

Before claiming V4F runs at <100 GB usefully, every Phase ends with:
- HFQ file size ≤ target.
- `mq2lloyd_coherence_harness.py` 10-prompt run: ≥ 6 ok, 0 hard
  attractors (matching the Qwen3.6-35B-A3B Lever 2 floor).
- A real long-context probe (256K and 512K tokens; the indexer is
  the main novelty being validated).

## What this plan does NOT cover

- Training-time tricks (load-balance loss, expert-bias adjustment) —
  not relevant for inference.
- FP8 KV-cache for the indexer or main path — antirez doesn't use it;
  we can revisit if memory pressure at 1M context warrants it.
- Cross-arch generalisation. The Hyper-Connection + compressed-KV
  combo is V4F-specific; a follow-up arch (e.g. GPT-5-class if it
  goes sparse) might reuse pieces but not the whole.
