# V4F to 100 tok/s — proof-based roadmap

**Current state (2026-05-21, post-Option-B + SKIP_HEAD):**
- 41-44 tok/s batched prefill on v4f.mq2lloyd-q8 (prompt=128-706, B=16-64) =
  **1.10 TFLOPs sustained** of an ~10 TFLOPs F16 peak (11%).
- PPL baseline: 10.83 @ ctx=256, 9.21 @ ctx=512.

**Target: 100 tok/s = 2.7 TFLOPs sustained** (matches Qwen3.6-35B-A3B
parity-per-byte on the same hardware).

**Note on numbers WITHOUT MoE:** if you see v4f_chat reporting ~80-90
tok/s prefill, MoE is silently skipping (HIPFIRE_V4F_UPLOAD_EXPERTS=1
is required to upload the expert blobs). Per
`feedback_v4f_chat_experts_silent_skip`, always verify rocprof shows
`gemv_mq2g256_lloyd_moe_*` kernels — otherwise the bench is dropping
the dominant 1.83 GB/token weight read.

## What we know (proven, not assumed)

1. **MoE GEMV is at 80% of its kernel ceiling** on this shape. Apples
   bench (commit pending): MQ2-Lloyd K4 sustains 1.21 TFLOPs at the
   exact V4F shape (M=4096 K=4096 K_TOP=6 B=64); production hits 0.95
   TFLOPs (cache-pressure cost). **No big win available from MoE WMMA
   or kernel re-design.**
2. **MQ2-Lloyd is actually FASTER than HFQ4** at V4F shape on this
   hardware (1.21 vs 0.85 TFLOPs). The "MQ2 is slow because codebook"
   hypothesis is falsified.
3. **The 2.5× sustained-TFLOPs gap to Qwen35** is mostly a
   workload-size difference (V4F 27 GFLOPs/token vs Qwen35 6
   GFLOPs/token = 4.5×) PLUS kernel-suite gaps (fused QKVZA, fused
   silu+mul+rotate, WMMA-on-attention).

## The math

To hit 100 tok/s at V4F's 27 GFLOPs/token, we need:

- 100 × 27 = **2.7 TFLOPs sustained**
- Currently at 1.10 TFLOPs → need **2.45×** improvement

Phase 1 alone (kernel optimizations within current quant) lands at
~50 tok/s (1.35 TFLOPs). **The remaining 1.85× requires a quant
format change** that lets us either compute faster or pack more useful
work per FLOP.

## Phase 1 — kernel optimizations (target: 41 → ~50 tok/s, +22%)

PMC-counter-driven. **Order subject to PMC data — must wait for
hardware-counter evidence before committing.**

### 1.1 WMMA attention kernel (biggest non-MoE lever)
- Current: `v4f_attn_swa_topk_batched_f32` runs in F32, scalar, 11.3%
  of GPU time per rocprof.
- Target: F16-staged Q/K/V, WMMA-accelerated Q·K^T and P·V GEMMs.
  Joint SWA+topK softmax stays sequential (small slice).
- Estimated payoff: 3× attention speedup → 7% prefill gain.
- LoC: ~400 (new kernel + dispatcher + wiring). Risk: medium (FP order
  changes; PPL validation required).

### 1.2 Fused MoE silu+mul+rotate (small but easy)
- Qwen35 has `fused_silu_mul_mq_rotate` for routed-MoE silu+FWHT path.
- V4F has the kernel for shared FFN single-token but not in batched
  MoE path.
- Estimated payoff: 1–2% prefill gain (saves 1 launch per layer per
  k_top × B positions).
- LoC: ~50 + wiring.

### 1.3 Fused MoE down + residual + sigmoid + scale
- Qwen35 has `gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched`.
- V4F's routed-MoE down currently does scaled atomicAdd into ffn_out;
  the post-down sigmoid+scale could fold into the down kernel.
- Estimated payoff: 1–2% prefill gain.
- LoC: ~80.

### 1.4 Fused QKVZA-style WMMA (q_lora + kv_joint)
- Qwen35 has `gemm_qkvza_hfq4g256_wmma` — fuses Q+K+V+Z+α into one
  WMMA launch.
- V4F has separate wq_a, wq_b, wkv calls. Fusing requires matching
  dtypes (currently Q8 mixed) and stacking M-axis.
- Estimated payoff: 3–5% prefill gain.
- LoC: ~250.

### 1.5 Smaller fusions (norm + rotate, etc.)
- Some chains already fused on the decode side; batched twins missing.
- Estimated payoff: 1–2% combined.
- LoC: ~150.

**Phase 1 total estimate: 41 → ~50 tok/s.**

## Phase 2 — quant format research for the remaining ~2× (target: 50 → ~100 tok/s)

The kernel ceiling at current quant is ~50 tok/s. To go further requires
either reducing data volume per token OR increasing arithmetic intensity
per byte. Constraints from the user:

- **PPL must not regress** (current baseline 10.83 @ ctx=256, 9.21 @ ctx=512)
- **KLD quality must not regress** (no measurement yet — need to take
  a baseline)
- Re-quantization is the only mechanism considered (no architectural
  change, no fine-tuning)

### Candidate formats to investigate

1. **MQ-Lloyd with larger codebook** (e.g. MQ3-Lloyd, 8-entry codebook,
   3 bits/weight). Same per-weight cost as HFQ3, more expressiveness
   than MQ2's 4-entry codebook. Existing MQ3-Lloyd WMMA kernels in
   `kernels/src/` from qwen35.
2. **HFQ4 affine on experts** (4 bits/weight, scale + zero). 2× the
   weight bytes of MQ2 but possibly higher TFLOPs sustained via better
   instruction mix.
3. **Mixed-precision per-layer**: keep MQ2 for less-sensitive layers
   (e.g. middle ratio=128 layers), upgrade to HFQ4/MQ3 for sensitive
   layers (compressor, first/last). Costs middle ground on size + PPL.
4. **Codebook-aware Lloyd refinement**: stay at 2-bit, but use GPTQ +
   forward-error-propagation Lloyd (per memory
   `project_gptq_lloyd_mq2_win.md`) to recover quality at smaller size,
   freeing budget for more important layers.
5. **AWQ-aware MQ2-Lloyd**: pre-scale weights by AWQ scales at quant
   time. Existing infrastructure in qwen35's `fused_silu_mul_mq_rotate_awq`.

### Methodology

For each candidate:
1. Quantize a representative subset (e.g. layer 5 and layer 25) of
   V4F's experts into the candidate format.
2. Run the existing `v4f_perplexity` on wikitext2-test at ctx=256, 512.
3. Run KLD vs the F16 reference on a held-out validation set.
4. Bench the kernel speed at production shape.
5. Compute Pareto: (perf gain) × (1 - PPL regression).

The format that wins Pareto becomes the new default. Implementation
LoC depends on whether kernel infrastructure exists — for MQ3-Lloyd
and HFQ4 the kernels are mostly there; for novel formats they'd need
building.

### Expected outcome

If a candidate format delivers:
- 25% smaller data → 1.25× BW relief → ~10% prefill gain
- WMMA-friendly layout → ~1.5× kernel TFLOPs → ~40% prefill gain
- Both compounded → ~1.7× prefill ≈ 50 → ~85 tok/s

100 tok/s is achievable only if a single format gives both BW relief
AND substantially higher TFLOPs at no PPL cost. The HFQ4 candidate is
the most likely path — quality is well-documented and WMMA support
exists.

## Order of work

1. **PMC counter sweep** on current V4F (Task #118 in progress). Proves
   per-kernel bottleneck (compute vs BW vs stall). Required input
   before any kernel changes.
2. **Highest-PMC-evidence Phase 1 lever first.** Don't commit kernel
   work without PMC justifying it.
3. **Measure and commit** after each lever; reject if measured win <
   2% (within bench noise).
4. **Phase 2 quant survey** starts only after Phase 1 lands and we have
   a fresh baseline (current 41 + Phase 1 gains).
5. **PPL + KLD validation MUST pass** before any quant change ships.
