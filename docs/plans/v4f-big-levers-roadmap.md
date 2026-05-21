# V4F prefill/postfill big-lever roadmap

Plan-of-record for the next three V4F perf pushes, ordered by risk × payoff
÷ effort. Synthesized from parallel research over `forward.rs`, `dispatch.rs`,
the `kernels/src/` tree, and qwen35's flash-attention prior art on
2026-05-21.

Execution order: **Lever 2 → Lever 3 → Lever 1**.

## Background — where the bench is today

v4f.mq2lloyd-q8 batched prefill bench (prompt=128 B=64, MoE+experts):
- Single-chunk peak: **36.0 tok/s** at B=64 prompt=64
- 2-chunk: **18.9-19.8 tok/s** at B=64 prompt=128
- Sequential baseline: 13.1-13.3 tok/s (unchanged)
- 1.82× per-token cost in chunk 2 vs chunk 1 — the cross-chunk attention
  decay that motivates Lever 3.

Recent commits on this branch (`feat/deepseek4-v4f`):

- `9637755` perf(v4f): batched Q8_0 wo_a — collapse 64k per-(batch,group)
  `gemv_q8_0` to single launch per layer. +3.1% measured.
- `46e2f81` refactor(v4f-prefill): hoist n_valid_swa_arr upload to
  per-chunk — 42 fewer stream-syncing htods per chunk; perf-flat,
  structural.
- `331533d` perf(v4f): fused `rmsnorm_mq_rotate_plain` — one launch
  writes both rot + plain outputs.
- `3419b69` perf(v4f): batched GPU-side hash-router — removes
  d2h(scores)+CPU+h2d roundtrip per hash-routed layer.

## Lever 2 — WMMA ZA-fused (do first)

### What

In `attention_block_batched_mixed` (`forward.rs:4425-4451`), the F16-WMMA
fast path already stages `tmp_plain_batch_f16` once and fires **four
separate `gemm_f16_x_f16_wmma` calls**:

- `comp_wkv`   — [2*head_dim=256, 4096]
- `comp_wgate` — [256, 4096]
- `idx_wkv`    — [2*index_head_dim=256, 4096]   (ratio==4 only)
- `idx_wgate`  — [256, 4096]                    (ratio==4 only)

All four are F16-native, all four share the same input. A stacked
[1024, 4096] WMMA matmul replaces four [256, 4096] launches, with
better tile saturation at B=64 and 3 fewer kernel launches per ratio=4
layer.

Full QKVZA stacking (`wq_a + wkv + 4× compressor`) is BLOCKED on
v4f.mq2lloyd-q8 because `wq_a/wq_b/wkv` are Q8 while the compressor
weights are F16. Heterogeneous dtypes can't trivially fuse without a
Q8→F16 decode-at-stack cost. Revisit later if a future variant stores
those three as F16-native.

### Prior art

`kernels/src/gemm_qkvza_mq3g256_lloyd_wmma.hip` (qwen35) shows the
multi-output WMMA pattern: thread-row routing by M-axis offset boundaries
picks which output buffer the WMMA accumulator writes into.

### Concrete steps

1. New kernel `kernels/src/gemm_f16_x_f16_wmma_za4.hip` — fork
   `gemm_f16_x_f16_wmma.hip`. Accept 4 weight pointers + 4 output
   pointers + 4 M sizes. Thread-row routing picks the output by
   cumulative M offset (same dispatch as `gemm_qkvza_mq3g256_lloyd_wmma.hip:54-65`).
2. Dispatch wrapper `gpu.gemm_f16_x_f16_wmma_za4_batched(...)` in
   `crates/rdna-compute/src/dispatch.rs`. Mirror the existing
   `gemm_f16_x_f16_wmma` signature, take 4 weight/output pairs.
3. Wire into `forward.rs:4425-4451` — replace the 4 sequential
   `gemm_f16_x_f16_wmma` calls with one fused dispatch. Keep an env
   opt-out `HIPFIRE_V4F_COMP_F16_WMMA_FUSED=0` for A/B.
4. Validate byte-equality at B=1 (against the current per-weight path),
   PPL on wikitext2-{256, 512} ctx, then bench at B=64 prompt=128.

### Effort, risk, payoff

- **Effort:** ~180 LoC. ~1 day including bench + validation.
- **Risk:** Low. Homogeneous F16, no new staging. Byte-equality testable.
- **Payoff:** Modest per-layer compressor speedup. Estimated +2-5% on
  prefill at long context (compressor fires per ratio=4 layer, V4F has
  ~22 such layers). 3 fewer launches × 22 layers × 2 chunks = ~130
  fewer launches per BAT phase.

## Lever 3 — Flash-attention rewrite of v4f_attn_swa_topk_batched (largest payoff)

### What

Current `v4f_attn_swa_topk_batched_f32` (`kernels/src/v4f_attn_swa_topk_batched.hip:31-152`)
runs grid `[n_heads, batch_size]`; each (head, batch) block reads its
own staged K/V slabs `[B, head_dim, swa_window]` and `[B, head_dim,
topk_window]`. The per-batch slabs are built by
`swa_visibility_stage_batched.hip` and have adjacent batch positions
OVERLAPPING by (swa_window − 1) = 127 tokens — adjacent Q positions in
the chunk re-read ~64× redundant K/V data.

Flash-style rewrite: per-(head, tile, sub_batch) grid; each tile holds
one K/V tile and computes Q·K^T against ALL Q rows in the block.
Bandwidth: 1.3 GB/layer/chunk → ~20 MB/layer/chunk (~64× less).

### Prior art

`kernels/src/attention_flash_asym3_tile_batched.hip` + 
`kernels/src/attention_flash_asym_reduce_batched.hip` (qwen35). The
tile kernel emits partials `[batch × n_heads × max_tiles × (2 +
head_dim)]`; the reduce kernel does 2-pass (global_max first, then
exp-corrected sum) for numerically stable cross-tile combine.

### Joint SWA + top-K complications

- Per-batch `n_valid_swa[b]` AND `n_active_topk[b]` are independent
  per row. Tile kernel must apply BOTH masks during score logic.
- `attn_sink[h]` is per-head extra row in the softmax; lands as a
  virtual final tile per head OR folded into first-tile normalizer.
- K/V staging changes: need a unified `[head_dim, position]` slab
  per layer per chunk (no per-batch replication), plus the per-batch
  valid-count arrays we already maintain.

### Concrete steps

1. Read qwen35's `attention_flash_asym3_tile_batched.hip` end-to-end.
   Document the partials layout + 2-pass reduce semantics.
2. New kernel `kernels/src/v4f_attn_swa_topk_flash_tile_batched.hip`.
   Tile-level Q·K^T with per-batch mask logic for the SWA-vs-topK
   split; per-tile online softmax partials.
3. New staging kernel `kernels/src/swa_visibility_stage_unified_batched.hip`.
   Drops the per-batch K/V replication; ONE unified `[head_dim,
   position]` slab per layer per chunk plus per-batch valid-counts.
4. Reuse `attention_flash_asym_reduce_batched.hip` for the reduce
   pass (same partials contract).
5. Handle `attn_sink`: virtual final tile per head OR fold into first
   tile.
6. Validate: byte-equality at B=1 against current kernel, PPL on
   wikitext2-{256, 512, 1024} ctx (target `|ΔPPL| < 1%` per memory
   `project_v4f_wmma_ppl_validation_2026_05_20.md`), then bench.

### Effort, risk, payoff

- **Effort:** ~500-600 LoC total (kernel ~250, staging ~80, dispatch ~120,
  tests ~150). ~3-5 days.
- **Risk:** Medium. Order-of-summation FP shift demands PPL validation
  budget. Joint SWA+topK softmax + per-batch masks needs careful
  testing.
- **Payoff:** **Largest of the three.** ~1.7-2× attention speedup on
  cross-chunk; could recover the 1.82× per-token decay measured at
  2-chunk prompt=128. Translation: bench could jump from 19.8 → ~35
  tok/s on 2-chunk runs.

## Update (2026-05-21 evening)

Following measurement on the actually-batched chunked driver post-Option-B:

- **Lever 2 (ZA-fused WMMA)**: shipped opt-in (`b3a0836`), measured −1.5%
  on v4f.mq2lloyd-q8 at B=64 vs four-call dispatch. Falsified; left as
  opt-in infrastructure for future shape investigations.
- **Pre-Lever-3 silent-fallback fix (Option B)** (`0128cc6`): not on
  the original roadmap. Per-batch device-side `pos_array_device_batch`
  and `attn_state_buf_batch` in PBS, sub-viewed into state during the
  per-position fallback loop. **Real win: prompt=128 B=64 MoE 19.8 →
  42.8 tok/s (+116% / 2.16×)**. The "cross-chunk decay" Lever 3 was
  designed to attack disappeared (3.42× at p=64 → 3.22× at p=512: only
  6% drop across 8× chunks). Lever 3 is no longer worth doing.
- **MTP SKIP_HEAD** (`212b562`): not on the original roadmap. The
  lm_head + d2h at the end of `mtp_forward` was 60% of per-call cost
  during prefill MTP fill (logits unused there). Env-gated. Saves 1.3
  s on a 501-token prompt = −8.9% of prefill stage. Deflated Lever 1's
  remaining payoff to ~3.4% end-to-end.
- **Lever 1 (batched MTP fill)** revised: now ~3.4% end-to-end after
  SKIP_HEAD. ~170 LoC. Modest. Do later when bigger fish are caught.

Remaining open questions:
- What's the dominant kernel cost inside the now-actually-batched main
  forward at B=64? (Per-token at p=512: ~24 ms.) That's likely where
  the next big lever lives.
- Are the small-perf commits (`9637755`/`46e2f81`/`331533d`/`3419b69`)
  worth re-benching now that chunk 2+ actually runs them? They were
  all measured against chunk 1 only.

## Lever 1 — Batched MTP fill (lowest priority)

### What

Current per-position `mtp_forward` loop in
`crates/hipfire-arch-deepseek4/examples/v4f_mtp_smoke.rs:188-230` and
`crates/hipfire-arch-deepseek4/src/spec_decode.rs:144-199` calls
`mtp_forward` once per batch position; **step 6 (the layer block at
lines 1881-1900 of forward.rs) CANNOT be batched** without breaking
per-position RoPE phase + per-position SWA ring slot writes. Only
steps 0-5 (embed, norms, e_proj, h_proj, broadcast-add) are batchable.

### Free version (do first if MTP fill is on the critical path)

`HIPFIRE_V4F_MTP_NO_ROUTED=1` is already plumbed at `forward.rs:120`.
Skips `ffn_routed` inside MTP draft steps. Cost: ~6pp K=2 accept
(81% → 75%); benefit: ~50% MTP step cost. Net: +7% effective tok/s on
spec decode at long prompts (per memory
`project_v4f_mtp_hc_plumbing_fixed.md`).

### Hybrid implementation (if free isn't enough)

1. Add `mtp_e_norm_batch`, `mtp_h_norm_batch`, `mtp_streams_batch` to
   `PrefillBatchScratch`. (3 new GpuTensor fields, ~30 LoC.)
2. New `mtp_forward_batch_chunk(cfg, weights, state, gpu, pbs, h_n_batch,
   next_tokens, start_pos, B)` in `forward.rs`.
3. Hoist steps 0-5 to per-chunk batched (embed lookup, rmsnorm batched,
   per-HC h_proj batched, broadcast-add batched).
4. Step 6 stays as a tight per-position loop — `state.n_tokens` cycles
   each iteration. The raw-pointer trick used in `spec_decode.rs:132-192`
   handles the mtp_last_hidden read/write decoupling.
5. Wire into `v4f_mtp_smoke.rs` prefill (replace the per-position loop
   at lines 206-230).
6. Validate: deterministic-fibonacci smoke retains output; K=2 accept
   stays at 81%.

### Effort, risk, payoff

- **Effort:** ~140 LoC. ~1-2 days.
- **Risk:** Medium. Must preserve SWA ring slot ordering, mtp_last_hidden
  chaining (full-HC `[hc_mult, hidden]`), full-HC plumbing per memory
  `project_v4f_mtp_hc_plumbing_fixed.md`.
- **Payoff:** 40-60% MTP-fill cost reduction at long prompts. Only
  matters for spec-decode users at prompts >> chunk size; the bench
  paths we've been measuring don't exercise MTP fill.

## Prerequisite — fix qwen35 build (coherence-gate dependency)

`coherence-gate.sh` currently fails on this branch due to pre-existing
`hipfire-arch-qwen35` E0599 errors at commit `6725a3a` and earlier
(NOT caused by recent V4F commits). The errors are calls to missing
methods `gemv_mq3g256_lloyd_moe_*` / `gemv_mq2g256_lloyd_moe_*` that
live under the `v4f_gemv_*` prefix in `dispatch.rs`. Without
coherence-gate running, there's no firewall against attractor
regressions (per memory `feedback_attention_precision.md`, 5%
attention error cascades into attractor within ~10 tokens).

**Must fix before Lever 3 PPL validation can be trusted.** Either
rename the qwen35 call sites to the `v4f_gemv_*` methods, or add
non-`v4f_` aliases in dispatch.rs.

## Methodology

For each lever, follow `docs/methodology/perf-benchmarking.md`:

1. Single-shell A/B is noisy on RDNA (±10-15% drift). Always verify
   across a fresh process via `scripts/probe_commits.sh $(git rev-parse
   HEAD~1) HEAD`.
2. Median of 3-5 measures, byte-identical prompt files (prompt md5
   recorded).
3. Δ ≥ 5% triggers investigation per CLAUDE.md (warming first → kernel
   occupancy → rocprof → env state → flag state → code-change bisect).
4. **Real GAIN: coherence MUST be established.** Run
   `scripts/coherence-gate.sh` and (if spec-decode touched)
   `scripts/coherence-gate-dflash.sh` before claiming any win.
5. Top1 match true on bench is necessary but NOT sufficient — verify
   PPL doesn't drift and outputs aren't attractor-ridden.

## Risk-flag inventory

- All three change FP rounding order. PPL validation against committed
  baseline mandatory.
- Lever 2 doesn't activate unless `HIPFIRE_V4F_COMP_F16_WMMA=1`
  (default on for v4f.mq2lloyd-q8).
- Lever 3 attn_sink handling needs care — see kernel comments at
  `v4f_attn_swa_topk_batched.hip:105-141`.
- Lever 1 must keep `mtp_last_hidden` capture as full `[hc_mult,
  hidden]` per `project_v4f_mtp_hc_plumbing_fixed.md` — stream-0-only
  capture is what pinned K=2 accept at ~50% pre-fix.
