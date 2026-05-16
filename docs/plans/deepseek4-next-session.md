# V4F bring-up — handoff for next session

State as of 2026-05-16:

**Done:**
- Phase 0 (scaffold): arch crate, Config parser, schema-drift gate
- Phase 1 (FP8 ingest): byte-exact dequant verified, 40.2 GiB HFQ
  ships under `~/.hipfire/models/v4f.mq2lloyd-gptq-all`
- Phase 1.5 (load_weights walk): all 43 layers' expected tensors
  present in HFQ index, hash-routing on layers 0-2 correctly
  handled
- Phases 2-4 kernels (7 total) — all compile-clean on gfx1151,
  dispatch wrappers landed, GPU-validated against CPU references
  within fp16 tolerance

**Not done — the gap to "first token":**

The kernels are stub-correctness; the forward path doesn't exist.
To get a token out of V4F, the next session needs:

### 1. `load_weights` upload (estimated: 2-3 days)
Replace the host-only walk with actual GPU uploads. For each
tensor the walk currently finds, call `gpu.upload_raw` or
`gpu.upload_f32` and stash the resulting `WeightTensor` handle in
the right `DeepseekV4LayerWeights` slot. Decisions:
- Norms / HC tensors: F16 direct upload
- Embeddings: Q8F16 (existing path)
- Attention LoRAs (Q-LoRA, KV joint, O-LoRA): MQ4G256
- Routed experts × 256 × 3 projections: MQ2G256Lloyd (already there)
- Hash-routing tables: restore I64 raw passthrough (currently
  skipped at quantize time; Phase 5 work)

### 2. Forward integration (estimated: 5-7 days)
Wire the per-layer forward in `arch.rs::forward`. The reference is
`crates/hipfire-arch-qwen35/src/qwen35.rs`'s `forward_scratch`.
V4F-specific differences (each requires kernel calls already
landed):
- 4-stream residual via `hc_compute_control` → `hc_sinkhorn_4x4`
  → `hc_mix_4stream` (twice per layer: once for attn, once for
  ffn)
- Tail-only RoPE via `rope_tail_halfsplit` on main attention Q/K
  (last 64 dims of 512 head_dim)
- Per-layer compressed-KV indexer: `indexer_compressed_k_score`
  → `indexer_top_k` → `indexer_kv_gather` (only for layers with
  compress_ratio > 0)
- SWA-windowed main attention reads gathered K/V (Phase 4 work —
  need wrap-aware FlashAttn parameter)

### 3. Numerical correctness gate (estimated: 1-2 days)
Add `dev/bench/data/v4f_first_token_oracle.txt` — a prompt + the
exact first token V4F should emit at temp=0. Once forward is wired,
this is the single gate: do we produce the expected token? If yes,
V4F works.

### 4. Quality validation (estimated: 1 day)
Run `mq2lloyd_coherence_harness.py` against V4F at all-MQ2-GPTQ.
Expected: similar or better numbers than Qwen3.6-35B-A3B (9/10 ok,
0 attractors per the Lever 2 result). If significantly worse,
the recipe didn't generalise off Qwen3.6 → revisit damping sweep
on V4F-class precision before declaring success.

### 5. Phase 5 (MTP, optional, ~1 day)
Drop the `mtp.` prefix-skip in `hipfire-quantize`, treat MTP layer
as layer 43, wire as DFlash drafter.

## Total estimated time to first token: ~2 weeks focused

## Risks

- **Indexer top-k correctness at long context.** The stub O(N·K)
  top-k works up to N~16K. At 1M ctx with compress_ratio=4, N
  could exceed 250K positions — stub goes O(250K · 512) per head ×
  64 heads × 40 layers ≈ 3e12 ops per step. Replace stub with
  bitonic / radix select before benchmarking at long context.

- **Hyper-Connections decomposition.** The `[24]` control vector
  reshape into Sinkhorn input isn't certain — see open question
  in `docs/plans/deepseek4-phase3-hyper-connections.md`. Cross-
  check against the V4F paper before forward integration.

- **MoE expert dispatch.** V4F's per-expert separate-tensor layout
  means we have 256 expert handles per layer (instead of one
  stacked-3D). Our existing routed-MoE GEMV kernels expect the
  3D layout. Choice: (a) emit re-stacked tensors at quantize
  time, OR (b) extend kernels to take per-expert handles. (a) is
  cheaper.

## Bottom line

V4F at 40.2 GiB is ready to load. The 7 GPU-validated kernels are
the foundation. From here, forward integration is straight Rust
work — no new HIP kernels required (other than possibly an
optimised top-k for long context).
