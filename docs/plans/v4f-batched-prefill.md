# V4F batched-prefill plan (2026-05-19)

Status: active. Currently the **single biggest perf opportunity** on V4F.
Blocks downstream MQ8 work (faster iteration on every subsequent quant test).

## Why this matters

V4F currently has **no batched prefill** — each prompt token is processed
sequentially via the same `decode_step` path. At long context, decode is
bandwidth-bound on weights, so prefill tok/s == decode tok/s at the same
position. This dominates wall time for any long-prompt workflow.

Measured today (mq2lloyd-f16compress, post-session-optimizations):

| Position | tok/s |
|---|---|
| ~128 | 15.94 |
| ~1024 | 12.21 |
| ~2048 | 7.18 |
| ~3800 | ~4 (from `long_code_audit` timing) |

The `long_code_audit` case from antirez's 5-case NLL test (3844 prompt
tokens + 4 target tokens) took ~15 min per quant. Without batched
prefill, every quant comparison run is dominated by this sequential
weight-streaming-per-token cost.

**Expected speedup with batched prefill**: 15-40× at prefill, depending on
batch size and per-layer kernel batching quality. This transforms
downstream iteration:

| Test | Sequential prefill | With batched (est.) |
|---|---|---|
| 5-case NLL scoring | 30 min | ~3 min |
| Long-context fact-recall | 12 min/quant | ~1.5 min |
| Smoke 10-case eval | 5 min/quant | 2-3 min |
| Imatrix collection (4690 prompts, ctx=32768) | days | hours |

## Architecture

V4F's `forward.rs::decode_step` is a per-token state-machine. Batched
prefill processes a batch of `B` prompt positions in parallel through
one forward pass per layer. Weight loads are amortized — load once,
matrix-multiply against `B` input vectors (a small matmul per layer
instead of B separate GEMVs).

### What's already in place (reusable)

| Component | Status | Reuse for V4F? |
|---|---|---|
| `kernels/src/attention_flash_*_batched.hip` (causal, multi-tile) | ✓ | Yes for SWA layers |
| `kernels/src/attention_q8_0_kv_batched.hip` | ✓ | Yes for ratio-128 layers |
| `kernels/src/gemv_mq2g256_lloyd_moe_*_batched.hip` | ✓ | Yes for routed-expert FFN |
| `kernels/src/embedding_*_batched.hip`, `argmax_batched.hip` | ✓ | Yes |
| `qwen35::forward_prefill_batch` (full pipeline driver) | ✓ | **Template to follow** |
| `qwen35::prefill_moe_ffn_body_batched` | ✓ | Direct pattern for V4F MoE |
| `KvCache` batched-fill semantics (`llama.rs`) | ✓ | Mostly transferable |

### What's V4F-specific (new code required)

| Component | Why | Scope |
|---|---|---|
| Indexer top-K (batched) | V4F's compressed-KV indexer scores all compressed rows per query position and picks top-K. Each batch position needs its own top-K. | New kernel `indexer_top_k_batched`. Pattern: `[B, n_compressed]` scores → `[B, K]` indices. ~1 day. |
| Mixed-attention batched (SWA + topk + sink) | Fused softmax over `[SWA + topk_active + sink]` keys, per batch position. | New kernel `v4f_attn_swa_topk_batched`. Parallel softmax across `(B, n_total)`. ~2 days. |
| Compressor commit batched | Writes one compressed-KV row per `ratio` positions. Some batch positions may trigger commit, others not. | Conditional batched write. ~1 day. |
| HC (Hyper-Connection) ops batched | 4-stream HC mix per token. qwen35 doesn't have HC, so no template. | ~2-3 days. |
| `forward_prefill_batch` driver in V4F's `forward.rs` | Orchestrates all batched kernels. | Port pattern from `qwen35.rs:3715`. ~3-5 days. |
| `PrefillBatchScratch` state | Amortize per-call allocations across chunks. | Mirror `qwen35::PrefillBatchScratch`. ~1 day. |
| Test harness | byte-equality with sequential path on representative prompts. | Port `prefill_batch_matches_sequential` from qwen35 tests. ~1 day. |

## Phase breakdown

### Phase A — V4F-specific batched kernels (5-7 days)

The hard part. Once these are working, the rest is mechanical.

* **A1: `v4f_attn_swa_topk_batched.hip`** ✅ **DONE 2026-05-18**
  - Parallel softmax across `(head, batch_position)`
  - Joint softmax over `[swa_n_valid + topk_active]` per query
  - Batched K/V reads from caches (each batch position has its own slot)
  - Per-batch valid counts via `n_valid_swa_arr[B]`, `n_active_topk_arr[B]` i32 device buffers
  - Test: `test_v4f_attn_swa_topk_batched` — byte-equality at B=1/4/32 ✓ (max_abs=0.000e0 at every size)

* **A2: `v4f_attn_swa_batched.hip`** ✅ **DONE 2026-05-18** — pure-SWA twin for non-indexer layers
  - Identical launch shape to A1 minus the topk branch
  - Test: `test_v4f_attn_swa_batched` — byte-equality at B=1/4/32 ✓

* **A3: `indexer_top_k_batched.hip`** ✅ **DONE 2026-05-18**
  - Per-batch-position top-K selection over compressed rows
  - Grid `[n_idx_heads, batch, 1]`; reuses the existing single-thread-per-head stub strategy
  - Test: `test_indexer_top_k_batched` — 0 mismatches at B=1/4/32 ✓

* **A4: `compressor_commit_batched.hip`** — **DEFERRED, low-priority** (2026-05-18)
  - Analysis: the compressor commit path is two operations:
    1. Per-step kv_state write (always) — single proj_dim memcpy into a ring slot
    2. Conditional pool + rmsnorm + rope (every `ratio` positions) — multi-kernel sequence
  - Each commit is small. For B=32, ratio=4: ~8 conditional commits + 32 ring-slot writes
    per layer per side. Total: ~80 small kernel launches per chunk-per-layer for compressor
    work, but no single one is a hot kernel.
  - Decision: don't batch the kernel — instead, loop the compressor sequence sequentially
    in the Phase B driver per (batch_row → commit-boundary) call. Adds modest launch
    overhead but is a tiny fraction of total batched-prefill cost (attention + MoE FFN
    dominate). Revisit only if Phase D profiling shows compressor as a bottleneck.

* **A5: HC ops batched** ✅ **DONE 2026-05-18** (~half a day, smaller than estimated)
  - `hc_mix_4stream_batched`, `hc_input_map_4stream_batched`
  - Per-position 4-stream transforms; batch dim parallelizes cleanly. The plan's
    earlier risk note about per-position state was about `mhc_pre()` (the upstream
    computation that produces `a_vec` / `A` / `scale`), not these leaf kernels.
  - Test: `test_hc_batched` — byte-equality at B=1/4/32 for both ✓

### Phase B — Driver + scratch + wiring (4-6 days)

* **B1: `PrefillBatchScratch` struct** — 🟡 **SCAFFOLD 2026-05-18**
  - Public struct + `new()` constructor in `forward.rs`. Currently a unit
    struct (just `max_batch`) — fields grow incrementally as the Phase B2
    batched chunk forward defines its staging tensor needs.
  - Rationale for the scaffold approach: defining the full tensor list
    upfront without a concrete batched body wastes VRAM on tensors we may
    not need (and risks missing tensors we do need).

* **B2: `forward_prefill_batch_chunk()` function** — pending. Single-chunk
  batched forward pass. Mirrors `decode_step` but for B positions at once.

  **Kernel inventory (2026-05-18 recon):** every per-layer kernel in
  `decode_step` mapped against existing batched twins in dispatch.rs.

  *Already-batched (reuse):*
  - `embedding_lookup_q8_batched`
  - `fused_rmsnorm_rotate_mq_batched`, `rmsnorm_batched`, `rotate_x_mq_batched`
  - `v4f_attn_swa_topk_batched_f32`, `v4f_attn_swa_batched` (Phase A1/A2)
  - `indexer_top_k_batched` (Phase A3)
  - `hc_input_map_4stream_batched`, `hc_mix_4stream_batched` (Phase A5)
  - `hc_streams_init_from_embed_batched` (shipped 2026-05-18)
  - `v4f_silu_mul_clamp_f32_batched` (k_top-batched, reusable per position)
  - `rotate_x_mq_batched` (k_top-batched, reusable per position)

  *High-priority NEW kernels (these block prefill speedup):*
  - **gemv_auto family — mostly WIRING work, not new kernels.** The
    `gemv_auto` dispatcher selects `gemv_f32` / `gemv_q8_0` /
    `gemv_mq4g256_prerotated` per weight dtype. Batched GEMM equivalents
    already exist for the two common V4F dtypes:
    - `gemm_f32_batched` (dispatch.rs:19024) — F16-→F32 attention/compressor
      projections
    - `gemm_q8_0_batched` (dispatch.rs:13013) — Q8 attention projections
    - `gemm_hfq4g256_*` family — MQ4 path (10+ variants tuned per arch)
    Need a new `gemv_auto_batched` dispatcher in V4F's forward.rs that
    routes per-dtype to these GEMM-batched kernels. ~1 day, no new HIP
    kernels required for these dtypes.
    - **NEW kernel still needed for MQ3-Lloyd / MQ2-Lloyd dense weights**
      if any V4F build uses them outside MoE (the antirezQ8 / Q4 builds
      do not; the MQ2-Lloyd build only uses MQ2-Lloyd for routed experts).
      Defer until we have a V4F build that needs it.
  - **V4F MoE position-batched** — `v4f_gemv_mq2g256_lloyd_moe_gate_up_indexed`
    and `..._down_residual_scaled_indexed` are k_top-batched per single
    position. Need new `_position_batched` arms that add a B dim so the
    expert weights load once per (k, expert) pair rather than B × that.
  - **v4f_topk_kv_gather_{f32,identity_f32}_batched** — current gathers
    stage one position's top-K K/V from the main compressed cache. For
    batched attention to consume `[batch, head_dim, topk_window]` (the
    layout v4f_attn_swa_topk_batched expects), this gather needs a B dim.

  *Medium-priority (LOOP ACCEPTABLE for first pass, batch later):*
  - HC algebra: `hc_compute_control`, `hc_apply_alpha`, `hc_sinkhorn_4x4`,
    `hc_head_compute_pre` — operate on small per-position vectors,
    looping per batch row is cheap.
  - `sigmoid_f32`, `scale_f32` — tiny vector ops, loop acceptable.
  - `indexer_relu_score_f32` — per-position score compute against a
    shared compressed-K cache; loop acceptable.
  - `v4f_moe_topk_bias_aware_f32` — top-K + bias + renorm + scale; loop
    acceptable since output is small `[k_top]` per position.

  *Low-priority (defer to Phase D polish):*
  - `rope_tail_interleaved`, `rope_tail_yarn_interleaved`,
    `rope_tail_inverse` — small kernels, loop is fine initially.

  *Per-position sequential (intentional per A4 deferral):*
  - `compressor_overlap_concat_f32`, `compressor_softmax_pool_f32` —
    sparse commits, looped in the chunk forward.

  **Critical-path effort:** ~5-6 days for the high-priority new kernels
  + integration. The plan's original 3-day estimate for B2 was based on
  qwen35's MoE batched kernels being a drop-in; the V4F-specific MoE
  semantics (bias-aware top-K, atomicAdd down with route_scale, MQ2-Lloyd
  format) require V4F-specific batched variants, which add ~2 days.

  **B2 progress checkpoint (2026-05-18, end of working session):**

  *Shipped batched per-layer helpers (all dormant — chunk forward
  errors at attention dispatch before any per-layer stage runs
  end-to-end):*
  - `gemv_auto_batched` — per-dtype dispatcher to gemm_f32_batched /
    gemm_q8_0_batched_chunked / gemm_hfq4g256
  - `mhc_pre_batched` — 5-step pipeline (compute_control → apply_alpha
    → split_finalize → sinkhorn → input_map); honours
    `HIPFIRE_V4F_POST_SCALE`
  - `q_lora_batched` — fused norm+rotate + plain norm + 2 GEMVs +
    q_norm + rotate + per-(B, head) Q norm; honours
    `HIPFIRE_V4F_SKIP_QHN`
  - `kv_joint_batched` — single GEMV + kv_norm in-place
  - `apply_tail_rope_batched` — plain + YaRN variants, per-layer freq
    params, honours `HIPFIRE_V4F_NO_YARN`
  - `hc_attn_mix_batched` / `hc_ffn_mix_batched` — wrappers on top of
    hc_mix_4stream_batched + memcpy back into streams_batch
  - `final_norm_and_head_last_batched` — extracts the last batch
    position via sub_offset, calls the existing per-position chain

  *New HIP kernels shipped this session (10 total):*
  - v4f_attn_swa_topk_batched.hip (Phase A1)
  - v4f_attn_swa_batched.hip (Phase A2)
  - indexer_top_k_batched.hip (Phase A3)
  - hc_mix_4stream_batched.hip, hc_input_map_batched.hip (Phase A5)
  - hc_streams_init_from_embed_batched.hip
  - rope_tail_interleaved_batched.hip, rope_tail_yarn_interleaved_-
    batched.hip
  - hc_compute_control_batched.hip, hc_apply_alpha_batched.hip,
    hc_sinkhorn_4x4_batched.hip, hc_split_finalize_batched.hip

  *forward_prefill_batch_chunk status:*
  - ✓ tokens + positions upload, batched embedding, HC stream init
  - ✓ per-layer mhc_pre + q_lora + kv_joint + rope (attention-side)
  - ☐ per-layer attention dispatch — bails out here
  - ☐ per-layer wo_a/wo_b O-LoRA projection
  - ☐ per-layer hc_attn_mix call
  - ☐ per-layer mhc_pre (ffn-side) + ffn_routed + hc_ffn_mix
  - ☐ final_norm_and_head_last_batched call (function exists)

  *Remaining work (each piece independently complex):*

  1. **Attention block staging** (~2-3 days). Each batch position needs
     a per-row visibility window into the SWA ring buffer + per-row
     indexer top-K K/V gather. Two-step path:
     - new kernel `swa_visibility_stage_batched.hip` — given the layer's
       SWA ring buffer + per-batch absolute positions, build a
       `[B, head_dim, swa_window]` per-row contiguous view (causal-mask
       aware: zero-pad beyond start_pos+b)
     - new kernel `v4f_topk_kv_gather_batched.hip` (Phase A4-equivalent
       deferred from recon) — gather per-batch top-K K/V from the
       main compressed cache via `indexer_top_k_batched` output
     - host-side per-batch `n_valid_swa_arr` and `n_active_topk_arr`
       i32 uploads
     - Then `v4f_attn_swa_topk_batched_f32` runs in one launch.

  2. **wo_a/wo_b O-LoRA batched** (~1 day). Per-group dispatch loops
     in sequential `attn_stub` — 8 groups × 2 GEMVs per group per
     token. Batched: 8 groups × 2 batched-GEMM calls total. Plus
     `rope_tail_inverse_batched` (~half day kernel, mirrors plain
     rope_tail_interleaved_batched).

  3. **ffn_routed_batched + ffn_hash_routed_batched** (~2-3 days). The
     V4F-specific MoE GEMV kernels (`v4f_gemv_mq2g256_lloyd_moe_-
     gate_up_indexed`, `..._down_residual_scaled_indexed`) are
     k_top-batched per single position. Need new
     `_position_batched` variants with an outer B dim that amortize
     the routed expert weight loads across positions. Plus a position-
     batched `v4f_moe_topk_bias_aware_f32_batched`.

  4. **End-to-end integration** (~half day once 1-3 are in place):
     wire all stages in forward_prefill_batch_chunk and replace
     forward_prefill_batch's per-token fallback with a chunked loop
     over forward_prefill_batch_chunk calls. Then byte-equality test
     vs sequential at small B.

  **Total estimated remaining for end-to-end batched prefill:**
  5-7 days of focused work. Roughly equal-effort phases.

  **B2 second-half checkpoint (2026-05-18, continued):**

  After the first checkpoint, another 11 batched kernels + Rust
  launchers shipped — covers the entire attention-block staging
  surface plus the per-group O-LoRA wo_a primitive:

  *Attention block primitives (all shipped):*
  - `swa_visibility_stage_batched` — dual-source [B, head_dim, swa_window]
    staging from PRE-CHUNK ring + within-chunk kv_batch
  - `swa_ring_write_batched` — chunk-end ring advance
  - `indexer_relu_score_batched` — per-batch scoring against shared
    compressed-K cache (H heads, LDS reduction)
  - `v4f_topk_kv_gather_batched` — per-batch top-K K/V gather
  - `v4f_topk_kv_gather_identity_batched` — ratio=128 variant (no top-K)
  - `rope_tail_inverse_batched` — post-attention V de-rotation (non-YaRN)
  - `wo_per_group_batched_f32` — block-diagonal F32 GEMV
    `y[b, g, r] = Σ_k wo_a[g, r, k] · x_in[b, g, k]`; one launch in
    place of B·G separate gemv_f32 calls

  *PrefillBatchScratch is now 21 GPU tensor fields:* embed_batch,
  streams_batch, tokens, tmp_batch, tmp_plain_batch, q_lat_batch,
  q_lat_rot_batch, q_batch, q_head_ones, kv_batch, positions,
  hc_c_batch, hc_pre_batch, hc_post_batch, hc_comb_batch,
  hc_x_in_batch, attn_out_batch, ffn_out_batch, streams_out_batch,
  swa_staged_batch, topk_staged_batch, n_valid_swa_arr,
  n_active_topk_arr, attn_out_raw_batch.

  *forward_prefill_batch_chunk status (end of session):*
  - ✓ token-ids + positions upload, batched embedding, HC stream init
  - ✓ per-layer mhc_pre + q_lora + kv_joint + tail-rope (attention-side)
  - ✗ bails at layer 0 attention dispatch: pure-SWA path needs final
    integration (steps 1-7 listed in the error message)

  **What it would take to land end-to-end batched prefill from here:**
  1. **Pure-SWA path attention dispatch wiring** (~half day): swa_ring
     lazy-alloc per layer, swa_visibility_stage call, n_valid_swa upload,
     v4f_attn_swa_batched dispatch, rope_tail_inverse_batched,
     wo_per_group_batched_f32 + gemv_auto_batched(wo_b),
     hc_attn_mix_batched, swa_ring_write_batched advance.
  2. **Mixed-attention path** (~1-2 days): adds the indexer-Q chain
     (wq_a_idx/wq_b_idx batched + indexer_relu_score_batched +
     indexer_top_k_batched + v4f_topk_kv_gather_batched OR
     v4f_topk_kv_gather_identity_batched) + the compressor commit
     sequential loop (per A4 deferral) + v4f_attn_swa_topk_batched
     dispatch.
  3. **MoE FFN batched** (~2-3 days): V4F-specific position-batched
     MoE GEMV variants (`v4f_gemv_mq2g256_lloyd_moe_gate_up_indexed_-
     position_batched`, `..._down_residual_scaled_indexed_position_-
     batched`) + per-batch top-K-bias-aware router + per-batch silu+rotate.
  4. **Integration + byte-equality test** (~half day) — call from
     `forward_prefill_batch`, verify against per-token decode_step
     at small B.

  **Total: 4-6 days for end-to-end working batched prefill on V4F**, of
  which steps 1+2+4 (≈2-3 days) deliver attention-block batched
  speedup; step 3 unlocks the MoE FFN batched speedup which is the
  bigger absolute win on V4F.

  ## End-of-session bench (2026-05-18, v4f.mq2lloyd-fp4fix)

  Bench: `examples/bench_v4f_batched_prefill.rs`. Sequential
  decode_step loop vs `forward_prefill_batch_chunked`, same prompt,
  fresh `DeepseekV4State` each.

  | prompt_len | max_batch | seq tok/s | bat tok/s | speedup | top1 match | max_abs |
  |---|---|---|---|---|---|---|
  | 16  | 4   | 12.8 | 4.2  | **0.33×** | ✓ | 0.000e0 |
  | 64  | 16  | 19.4 | 20.5 | **1.06×** | ✓ | 0.000e0 |
  | 64  | 32  | 18.3 | 20.4 | **1.12×** | ✓ | 0.000e0 |

  **Math is byte-equal at every size** — the batched chain is
  correct end-to-end. Perf gain is currently modest; the gap from
  the 15-40× plan target reflects:

  - **Per-batch sequential compressor+indexer dominates.** For each
    of V4F's 41 compressed layers, `attention_block_batched_mixed`
    runs a B-iteration sequential loop calling
    `compressor_forward(main)` + `compressor_forward(indexer)` +
    `indexer_forward` + `v4f_topk_kv_gather` per batch position. Each
    inner call is ~5-6 small kernel launches; at B=32 × 41 layers
    × ~17 launches = ~22K launches per chunk, dominating wall time
    at ~5 µs launch overhead each (~110 ms / chunk).
  - The actually-batched parts (attention kernel, wo, FFN) get a
    real but modest win since they were already small fractions of
    decode time.

  **Memory (B=4, v4f.mq2lloyd-fp4fix):**
  - PrefillBatchScratch: **8 MB** (negligible)
  - load_weights jump: 27.7 → 105.5 GiB system-used (~78 GiB delta).
    Likely pre-existing mmap/page-cache behaviour (model ~40 GiB on
    disk; doubled in unified memory until OS reclaims) — independent
    of this work.

  ## Next perf-pass priorities (post-correctness)

  1. **Batch the indexer chain end-to-end.** Replace the per-batch
     sequential `indexer_forward` loop with batched primitives that
     already exist: `gemv_auto_batched(wq_b_idx → q_idx_batch)` +
     `rope_tail_interleaved_batched` + `gemv_auto_batched(weights_-
     proj → idx_w_batch)` + `indexer_relu_score_batched_f32` +
     `indexer_top_k_batched` + `v4f_topk_kv_gather_batched`. The
     subtle correctness constraint: per-batch n_compressed[b] varies
     when B > ratio (within-chunk commits change the cache); handle
     by either limiting chunk B ≤ ratio (= 4 for indexer layers) or
     extending `indexer_relu_score_batched` to take a per-batch
     n_compressed array.

  2. **Batch the per-step compressor kv_state write.** Today's
     `compressor_forward(main)` writes a single position's compressed
     residual into the kv_state ring. Replace the B-iteration loop
     with one launch that writes all B slots. The conditional pool +
     rmsnorm + rope at commit boundaries can stay sequential
     (sparse: ≤ B/ratio commits per chunk).

  3. **Tune the attention/wo block dimensions.** wo_per_group_batched_f32
     uses 32 threads × `M × G × B` blocks — likely under-occupies
     gfx1151 wave32 SIMD. Bump to 64-128 threads with shared-mem
     weight broadcast.

  Each of these is ≈1 day, total ~3 days to push from 1.12× to
  the 5-10× range plausibly. The 15-40× target needs more
  aggressive amortization (e.g. a fused per-layer megakernel).

* **B3: `forward_prefill_batch()` top-level entry** — 🟡 **SCAFFOLD 2026-05-18**
  - Public entry point in `forward.rs` with stable signature
    `(cfg, weights, state, gpu, tokens, start_pos, &mut PrefillBatchScratch)
    → Result<logits, String>`.
  - Body currently loops `decode_step` per-token (byte-identical to the
    existing sequential prefill). Callers (eval harnesses, daemon) can
    wire against this surface while Phase B2 grows the batched body
    behind it.
  - Env opt-out planned: `HIPFIRE_V4F_PREFILL_BATCHED=0` to force the
    per-token path once batched body lands.

* **B4: Integration with existing `decode_step`** — pending. After prefill,
  decode mode takes over at `start_pos + prompt_len`. ~0.5 day.

### Phase C — Correctness validation (2-3 days)

* **C1: Byte-equality test** — `forward_prefill_batch(prompt[0..N])` followed by one `decode_step(prompt[N])` should produce the same logits as N+1 sequential `decode_step`s. Test at multiple ctx points (128, 1024, 4096). ~1 day.

* **C2: PPL sanity** — wikitext-2 PPL via batched prefill should match sequential within FMA-order noise (~0.1%). ~0.5 day.

* **C3: Long-context fact-recall** — `v4f_long_context_test.rs` should pass with batched prefill. ~0.5 day.

* **C4: Quality test alignment** — 5-case NLL scorer should produce identical numbers via batched prefill. ~0.5 day.

### Phase D — Performance polish (2-3 days)

* **D1: Chunk size tuning** — find the max batch that fits in working memory without slowing per-token effective time. ~1 day.

* **D2: KV-cache batched fill optimization** — minimize the per-position writes via vectorized cache-fill kernels. ~1 day.

* **D3: Cleanup + profile pass** — make sure no remaining sequential bottleneck inside the per-chunk path. ~0.5 day.

## Acceptance criteria

1. **Correctness**: byte-equality (or FMA-order ε) between batched and
   sequential prefill on at least 5 representative prompts spanning short /
   medium / long context, both pure-SWA and mixed-attention layers.

2. **Performance**: ≥10× prefill speedup at ctx=2048, ≥15× at ctx=4096.

3. **Quality regression-free**: wikitext-2 PPL via batched prefill within
   ±0.5% of sequential at ctx=128/1024/2048.

4. **Long-context test passes**: `v4f_long_context_test` recovers all 16
   facts using batched prefill.

5. **Quality scoring identical**: 5-case NLL numbers via batched prefill
   match sequential to ε.

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| V4F-specific kernels have subtle SWA-mask / topk-mask bugs at batch=1 vs batch=N | Test byte-equality at batch=1 first (should match sequential exactly); incrementally raise batch size |
| LDS / shared-mem pressure at large batch sizes | Start with batch=32, profile occupancy via `gfx-kernel-metadata` skill, tune downward if needed |
| HC ops don't parallelize cleanly across batch (have per-position state) | Investigate dependencies; may need to keep HC sequential and only batch the GEMVs |
| Compressor commit semantics break with multi-position chunk crossing a commit boundary | Carefully handle per-chunk position arithmetic; test with prompt lengths that cross multiple commit boundaries |
| Tokenizer / chat-template integration assumes per-token decode | Prefill returns logits at LAST position only initially; per-token logits as optional |

## Effort estimate

| Phase | Days |
|---|---|
| A — V4F-specific batched kernels | 5-7 |
| B — Driver + scratch + wiring | 4-6 |
| C — Correctness validation | 2-3 |
| D — Performance polish | 2-3 |
| **Total** | **13-19 days** |

Realistic calendar window: **2-4 weeks** including iteration on bugs.

## Status tracking

This plan is the source of truth for the work. Update phase-status here as
sub-tasks complete. Task tracker entries:
- #91 — Top-level batched prefill (active, blocking #87-90)
- #87 — V4F-MQ8 quant build (blocked on #91)
- #88, #89, #90 — MQ8 kernel work (blocked on #91)
