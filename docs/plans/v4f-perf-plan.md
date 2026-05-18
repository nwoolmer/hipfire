# V4F performance plan (2026-05-18)

Status: A / B / C / D / E shipped. F (MQ2-GPTQ vs Lloyd) measured negative —
GPTQ-all is 1.9-3.3× worse PPL than Lloyd-f16compress on wikitext2-test, do
not switch defaults. See `project_v4f_mq2_gptq_lloyd_falsified` memory for
the full PPL sweep and the build provenance.

## Session results (2026-05-18, branch feat/deepseek4-v4f)

Cumulative perf delta from baseline at the start of the session:

| ctx  | tok/s pre | tok/s post | tok/s Δ | PPL pre | PPL post | PPL Δ |
|------|-----------|------------|---------|---------|----------|-------|
| 128  |  12.07    |  15.94     |  +32%   |  14.03  |  11.58   | −17%  |
| 1024 |  10.00    |  12.21     |  +22%   |   9.01  |   7.32   | −19%  |
| 2048 |   6.33    |   7.18     |  +13%   |   8.49  |   6.28   | −26%  |

Commits this session (newest first):

- 1050031 — Phase E: GPU bias-aware top-K + scaled normalize (+1.7-1.9%)
- 2158047 — Phase B: drop_mmap before routed-expert upload (load 83s → 56s)
- 1e33ed2 — batched rotate for wo_a per-group loop (+6% ctx=128)
- b3e4332 — batched silu_clamp + rotate in MoE expert loop (+8% ctx=128)
- b5279f2 — MQ2 MoE K4-unroll + LDS codebook port from MQ3 gfx1100 (+7–14%)
- 4d8c757 — **grid bug + parallel softmax rewrite, PPL −17%/−19%/−26%**
- 01eb2a7 — parallel softmax in v4f_attn_swa_topk (prior to grid-fix session)
- 4cf6aff — per-layer batched pread for routed expert upload (150s+ → 75s)

The PPL win in 4d8c757 was a latent dispatch-grid bug (heads 8..63 not
computing) dating to commit e657ece on 2026-05-17 — not a quantization or
training-side improvement. See feedback_v4f_dispatch_grid_axis memory.

Tried-and-rejected this session:
- V-transpose for swa_v cache (commit reverted): bit-identical PPL but a
  wash on tok/s — head_dim=512 already coalesces well from L1 even without
  the layout flip, and the extra write-side transpose cost cancels the gain.
- Batch-aware blockIdx.y in default `gemv_hfq4g256` (reverted): introduced
  enough register pressure to regress the non-batched callers by ~5%.
  Cleaner path forward: separate `gemv_hfq4g256_strided_batched` kernel file
  rather than modifying the shared default. Not pursued this session.

## What we measured (gfx1151 / Strix Halo, 137 GB unified memory, /data NVMe)

### Model load
- Before (mmap + 2-pass thrash): 150+ s, often hit upstream-timeout cutoffs
- After (`tensor_data_pread`, per-layer batched, commit 4cf6aff): 75 s
- Per-layer time grows 0.84 s → 2.04 s as GPU allocations build → unified-memory pressure
- Bottleneck inside the routed-expert pass is `read+concat` (200ms → 1400ms growth), NOT `upload_raw` (consistent 60-180 ms)

### Decode (single-token TG vs context, `bench_decode_vs_ctx`)
| pos | ms | tok/s | notes |
|-----|----|----|---|
| 0   | 141 | 7.09 | cold; one-time lazy-alloc + JIT |
| 16  | 102 | 9.80 | warm, pre-mixed-attn |
| 64  | 169 | 5.92 | SWA grows but still windowed |
| **128** | **253** | **3.95** | **mixed-attn engages** |
| 256 | 298 | 3.36 | gradual scaling |
| 512 | 351 | 2.85 | gradual scaling |
| 1024 | pending | | |
| 2048 | pending | | |

**TG is bimodal**: ~10 tok/s pre-mixed-attn, ~3 tok/s once `v4f_attn_swa_topk` is doing real work on the joint K/V softmax. Earlier "3.5 tok/s" multi-turn average was just (10 + 3 + 3 + …) averaged over 512 tokens.

### Falsified hypotheses (do not chase)
- **D2H syncs per MoE layer dominate**: each measured at 7-9 µs (`HIPFIRE_DTOH_DUMP=1`). Total D2H wall time per token: ~50 µs. Not the bottleneck.
- **Kernel launch overhead dominates**: at pos=16 we already hit 9.8 tok/s = ~10 ms / 41 layers — launches pipeline fine when the kernels themselves are fast.
- **Memory bandwidth dominates**: 6 GB/token at 256 GB/s = 23 ms floor; we're at 102 ms at pos=16 = ~5× above bandwidth ceiling. Bandwidth isn't the issue, dispatch + small kernels + non-MoE work are.

## Plan (ROI-sorted)

### Phase A — `v4f_attn_swa_topk` parallel softmax rewrite **[in progress]**
**Why**: biggest single TG win. Sharp inflection at pos=128 is mixed-attn engaging; current kernel uses thread 0 only for scoring + max + sum_exp serially (511 of 512 threads idle).
**What**:
1. All threads cooperatively compute scores
2. Block-level reduction for max + sum_exp via LDS
3. V accumulation already parallel — leave alone
**Target**: 2-3× on TG @ ctx≥128 (i.e. push the 3 tok/s tier to 6-9 tok/s).
**Validation**: PPL bit-identical @ ctx=128 / 1024 / 2048, multi-turn coherence, bench_decode_vs_ctx delta.

### Phase B — Model-load further speedup
**Why**: faster iteration cycle for every subsequent change.
**What**: reorder uploads so all non-routed weights load first, then `drop_mmap()` *before* the routed-expert pass starts. Frees 78 GB of page-cache before the gate_up uploads, eliminates growing per-layer slowdown.
**Target**: 75 s → 40-50 s.

### Phase C — Low-context TG polish
**Why**: ~10 tok/s pre-mixed-attn is below the 25 tok/s memory ceiling — room to grow.
**What**:
- Pre-alloc per-token MoE scratch (currently `Vec<i32>`/`Vec<u8>` builds per layer per token)
- Disable HIP profile-timer overhead in production decode path
- Fused silu_mul_clamp + FWHT rotate kernel (replaces 12 launches/layer)
**Target**: +20-50 % on pre-mixed-attn TG (10 → 13-15 tok/s).

### Phase D — MQ2-Lloyd MoE kernel LDS-codebook + K4 unroll
**Why**: routed-expert GEMVs are ~70 % of per-token MoE bytes. MQ3-Lloyd gfx1100 sibling already has these optimizations; port to MQ2.
**What**: apply LDS-cooperative-load codebook + K4 group unroll pattern from `gemv_mq3g256_lloyd.gfx1100.hip` to:
- `gemv_mq2g256_lloyd_moe_gate_up_indexed{,_batched}.hip`
- `gemv_mq2g256_lloyd_moe_down_indexed{,_batched}.hip`
**Target**: 5-15 % on the routed-expert portion (both short and long ctx).

### Phase E — HIP graph capture + GPU top-K (deferred)
**Why**: only matters once Phase A-D have cleared bigger bottlenecks. Measurements showed launch overhead is small; HIP graphs would shave µs we don't urgently need.

### Phase F — Quantization quality (parked)
- `v4f.mq2-gptq-all.hfq` exists from the GPTQ-Lloyd quant run (sequential error-feedback, unit imatrix). Validate PPL vs `mq2lloyd-f16compress` baseline once iteration cycle is faster (post-Phase B).
- Mixed MQ3-down sweep only if GPTQ-all-MQ2 doesn't close the gap to antirez IQ2.

## Execution order
**A → B → C → D → E**. Phase A first because it's highest-impact and most uncertain (kernel rewrite needs validation); fast feedback matters. Phase B follows because it speeds every subsequent iteration. C-E are polish on a known-good path.
