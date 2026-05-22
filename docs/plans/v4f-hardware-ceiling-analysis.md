# V4F prefill — hardware ceiling analysis (validated, not guessed)

**Date:** 2026-05-22
**Hardware:** AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151, RDNA 3.5, Strix Halo)
**Model:** v4f.mq2lloyd-q8.hfq (B=16 batched prefill, 705-token prompt)

## Measurements (production v4f_chat with MoE actually loaded)

| Metric | Value | How measured |
|---|---|---|
| Prefill rate | **43.0 tok/s** | v4f_chat with `HIPFIRE_V4F_{MOE,UPLOAD_EXPERTS}=1` |
| GPU active % | **96.8%** | rocprof kernel time sum / wallclock |
| Sustained TFLOPs | **0.91 TFLOPs** | 43 tok/s × ~21 GFLOPs/token |
| Per-token weight bytes | **2.165 GB** | computed from V4F config |
| Effective DRAM BW | **93 GB/s** | tok/s × bytes/token |
| DRAM peak (measured) | **189.6 GB/s** | `microbench_v4f_kernels dram_peak` |

### Hardware reference ceilings

| Resource | Peak | Production usage | Headroom |
|---|---|---|---|
| DRAM bandwidth | 189.6 GB/s | 93 GB/s | **49% used / 51% free** |
| F32 scalar compute | ~5 TFLOPs | 0.91 TFLOPs | 18% used / 82% free |
| F16 WMMA compute | ~10 TFLOPs | 0.91 TFLOPs | 9% used / 91% free |
| GPU active time | 100% | 96.8% | 3.2% launch overhead |

**Key finding: V4F prefill is COMPUTE-BOUND, not BW-bound.**
The MoE kernels (which dominate) run at 11.5 GB/s = 6% of DRAM peak.
They're capped on arithmetic intensity of the scalar MQ2-Lloyd codebook
lookup, not on memory throughput. There's a 4-15× BW headroom available.

## Per-kernel breakdown (rocprof, sum of 16.33s GPU time)

| % | Kernel | What it is |
|---|---|---|
| **29.2%** | `gemv_mq2g256_lloyd_moe_gate_up_k4` | MoE gate+up (the heaviest) |
| **16.8%** | `gemv_mq2g256_lloyd_moe_down_residual_scaled_k4` | MoE down |
| 14.3% | `v4f_attn_swa_topk_batched_f32` | Joint SWA+topK attention |
| 13.4% | `wo_per_group_batched_q8_0` | wo_a (block-diagonal output proj) |
| 9.6% | `gemm_q8_0_wmma` | Q-LoRA / KV-joint / wo_b Q8-WMMA |
| 5.9% | `v4f_topk_kv_gather_batched_f32` | top-K K/V gather |
| 4.8% | `gemm_f16_x_f16_wmma` | F16 compressor |
| 6.0% | (other small) | norms, ropes, HC mix/control, indexer scoring |

**MoE total: 46.0%** of GPU time — the dominant cost.

## The MoE gate_up kernel — the dominant lever

| Metric | Value |
|---|---|
| Calls | 1935 (45 chunks × 43 layers) |
| Per call | 2.46 ms |
| FLOPs per call | 3.22 GFLOP (2 × M=4096 × K=4096 × K_TOP=6 × B=16) |
| Compute throughput | **1.31 TFLOPs (26% of F32 peak)** |
| Weight bytes per call | 28.3 MB (6 experts × 4.7 MB) |
| BW per call | **11.5 GB/s (6% of DRAM peak)** |
| If BW-bound | Would take 0.29s total instead of 4.76s — 16× theoretical headroom |

The MoE kernel is **scalar codebook-lookup + scalar FMA**. WMMA's
arithmetic intensity (256 MACs per F16 fragment) is what it would
need to saturate the ALU-vs-BW curve, but WMMA on MoE is structurally
blocked by expert-routing (different experts per batch position → can't
fill the 16×16 C-fragment, see prior memory entries
`feedback_v4f_prefill_optimization_dead_ends` and
`v4f-to-100-tps-roadmap`).

## Path to 100 tok/s requires 2.32× speedup

Concrete paths, ranked by realism:

### Path A — Smaller per-token weight bytes
Reduces MoE bytes-per-call proportionally to bpw reduction. If
sub-2-bit on routed experts holds PPL, total bytes/token drops, BW
required drops, and the MoE kernel's per-call time shrinks (it's
compute-bound on the per-byte decode/FMA, so fewer bytes = less work).
- Current: 2.25 bpw → 1.83 GB/token MoE
- Target: 1.5 bpw (MQ1.5-Lloyd mixed) → 1.22 GB/token MoE (-33%)
- Projected prefill: 43 → ~57 tok/s (if proportional)
- Plan: `docs/plans/v4f-sub-mq2-quant-research.md`. Risk: PPL.

### Path B — Re-architect MoE to use WMMA via cooperative LDS staging
Decode K_TOP × 256-weight expert tiles into F16 LDS, then run WMMA
with all 16 batch positions sharing the SAME (b, krank) → can't (each
b has different experts). Sort-by-expert pattern was tried before
(grouped K4) — measured **18% SLOWER** because it loses x-row cache
reuse for marginal expert-slab reuse.
- Concrete fix: **scatter-by-expert grouped GEMM** like qwen35's
  `gemm_hfq4g256_moe_grouped_wmma_k2`. Per Qwen3.5-A3B mq4 the
  grouped-WMMA path lifted prefill 114% (1396 → 2983 tok/s). For
  V4F MQ2-Lloyd would need a new `gemm_mq2g256_lloyd_moe_grouped_wmma`.
  ~600-800 LoC, 4-6 days, depends on whether MQ2-Lloyd's codebook
  decode can ride the same scatter-grouped substrate.

### Path C — Lower-precision attention K/V
Attention reads SWA + topK K/V at F32. Switching to F16 KV halves
those reads. At 14.3% of GPU time, getting half ≈ 7% prefill gain.
Doable; risks attention numerical precision (Q8 KV attempts elsewhere
showed drift).

### Path D — Increase batch B beyond 16
B=16 chosen post-sweep as the plateau optimum. Larger B (32, 64, 128)
showed gradual regression from L2 / Infinity Cache spill on
activations. NOT a path.

## Honest conclusion

- 43 tok/s production, 0.91 TFLOPs sustained at 9% of WMMA peak.
- **NOT BW-bound** (49% DRAM headroom).
- **Compute-bound on scalar MoE kernel** running at 1.31 TFLOPs / 26%
  of F32 peak.
- 100 tok/s requires either smaller weights (Path A, 4 days) or a
  scatter-grouped WMMA MoE kernel (Path B, ~1 week).
- Path B is the cleanest perf lever — same approach Qwen35 used for
  its A3B win, ported to MQ2-Lloyd. Validated win in adjacent codebase.

## Hardware-level absolute ceiling

At 1.5 bpw MoE quant (Path A) + the existing 2.25 bpw on the
non-routed weights:
- Per-token bytes: ~1.6 GB
- BW ceiling: 189.6 / 1.6 = **118 tok/s** (if MoE becomes BW-bound)

At Path B (WMMA MoE at near WMMA peak):
- MoE compute time at 8 TFLOPs sustained: ~0.6s (vs 7.5s today)
- Total wallclock: 16.86 - 7.5 + 0.6 = 10s
- Throughput: 705 / 10 = **~70 tok/s**

The asymptote on this hardware at this quant is **~70-80 tok/s** with
the easy levers. **100 tok/s requires both Path A + Path B** to
compose.
