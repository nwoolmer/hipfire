# V4F sub-MQ2 quant research

**Date:** 2026-05-21
**Baseline:** MQ2-Lloyd-G256 on routed experts (qt=19, 72 B / 256 weights = 2.25 bpw)
+ F16 compressor + F16 indexer + MQ4G256 attention/shared experts. Wikitext2-test PPL:
12.66 @ ctx=256, 9.82 @ ctx=512. KLD vs F16: not measured (baseline must be
established before any candidate can be gated — see §Validation).
**Target:** strictly-less-than-2.25-bpw on routed experts (the 52 % of GPU
time, ≥80 % of weight bytes) with ≤ 5 % PPL and ≤ 5 % KLD regression vs the
baseline above.
**Hardware:** Radeon 8060S / gfx1151 / Strix Halo. 256 GB/s LPDDR5X-8000
theoretical, 189 GB/s measured DRAM ceiling (project_v4f_prefill_kernel_bw_audit).
**Current state:** decode 17.0 tok/s; prefill 41 (long) – 86 (short) tok/s
single-chunk B=16; per-token weight read ≈ 3 GB at MQ2.

## Executive recommendation

**Top candidate: Mixed-precision routed-expert recipe (the "Unsloth Dynamic"
pattern adapted to V4F).** Drop the *least-sensitive* routed-expert
sub-population to MQ1.58 (1-bit-codepoint + group-sign codebook in an
MQ2 byte frame, decode-bit-compatible with the existing MQ2 kernel) while
keeping the *most-sensitive* layers (first 3 + last 3 layers, and `down_proj`
across all layers) at the current MQ2-Lloyd-G256. Drop nothing else.

Expected effective routed-expert bpw: **≈ 1.75 bpw** (1.58 bpw on ~70 % of
routed weights + 2.25 bpw on the ~30 % of sensitive routed-expert
layers/projections). Per-token weight BW: 3.0 GB → 2.35 GB on the MoE
chain = **-22 % MoE bytes / -11 % whole-token bytes**, projected prefill
**+11–13 %** (long-prompt 41 → 46 tok/s; short-prompt 86 → 96 tok/s),
projected decode **+10 %** (17.0 → 18.7 tok/s).

Why this candidate is the lowest-risk pick:

1. **Same recipe shape that Unsloth shipped to production on DeepSeek R1.**
   Same model family (DeepSeek MoE, comparable expert counts, same SwiGLU +
   `down_proj` sensitivity). Unsloth's IQ1_S 131 GB build (≈1.58 bpw
   effective) scored 69 % on a code-gen benchmark where flat-1.58-bit
   collapsed to 0 % ([Unsloth blog](https://unsloth.ai/blog/deepseekr1-dynamic)).
2. **No new GPU kernel.** A 1.58-bpw payload can live inside the existing
   `gemv_mq2g256_lloyd_*` decode and `gemm_mq2g256_lloyd_*_wmma` prefill
   kernels by *re-using a degenerate codebook* — encoder collapses 2 of
   the 4 fp16 codepoints onto identical values, or stores only 3 distinct
   codepoints in the header (4th slot duplicated). Kernel doesn't notice;
   `cb_lds[idx]` lookup still happens. Bit pattern unchanged; bpw on disk
   is the SAME 2.25 — the bandwidth win is **0** on the
   degenerate-codebook variant. Reject this sub-variant (degenerate
   codebook is encoder-side novelty with zero kernel benefit) and instead
   route the low-bpw layers through a real MQ1.5 byte frame (see §
   Candidate 1).
3. **Reuses prior shipped work.** MQ2-Lloyd MoE indexed kernels (6 files)
   already shipped (`gemv_mq2g256_lloyd_moe_{gate_up,down}_indexed*.hip`).
   MQ3-Lloyd MoE indexed kernels are partially built (`gemv_mq2g256_lloyd_moe_*`
   pattern is the template; Phase 3d in `project_mq2_lloyd_moe_rescue.md`
   blocked only on dispatch wiring). Mixed-precision in the V4F arch crate
   is per-layer-class allocation — already factored that way.

**Top secondary: Lloyd-MQ2 + sparse outlier preservation** (FE2-bit /
SqueezeLLM pattern). Adds a per-row "outlier exception list" of 0.3–0.5 %
weights stored at F16, indexed by (row, col_offset). Reuses MQ2-Lloyd-G256
unchanged for the bulk; outlier mask is a separate small tensor. Effective
bpw: ~2.25 + 0.3 × 16 ÷ 100 = **~2.30 bpw** (slightly heavier than
baseline). Direction wrong — increases bytes, doesn't reduce them. Only
worth shipping as a *quality recovery* tool composed *on top of* Candidate
1's MQ1.5 layers to lift them back into the ≤5 % PPL gate. Don't ship
standalone.

## Candidate ranking table

| # | Candidate                                          | Effective bpw routed | Total V4F GiB est. | Proj. prefill (long) | Proj. decode | Quality risk | Eng cost  | Recommend |
|---|----------------------------------------------------|---------------------:|-------------------:|---------------------:|-------------:|-------------:|----------:|-----------|
| 1 | **Mixed-prec: MQ1.5-Lloyd most + MQ2-Lloyd sens.** | ~1.75                | ~70 GiB            | 46–50 tok/s          | 18.7 tok/s   | Medium       | 6–10 d    | **YES — build first** |
| 2 | Lloyd-MQ2 + 0.5 % F16 sparse outliers              | ~2.30                | ~80 GiB            | 40 tok/s (regress)   | 16.5 tok/s   | Low          | 5–7 d     | NO standalone; compose with #1 |
| 3 | Per-row variable codebook (4-entry sens / 2-entry insens) | ~1.85         | ~71 GiB            | 45 tok/s             | 18.3 tok/s   | Medium-high  | 8–12 d    | DEFER — high impl risk for marginal extra win |
| 4 | MQ3-Lloyd (kmap-promote) on top-frequency experts + MQ2-Lloyd on tail | ~2.0 ↑ (size grows!) | ~73 GiB | 42 tok/s    | 17.2 tok/s   | Low          | 4–6 d     | NO — wrong direction (bigger, not smaller) |
| 5 | Uniform LUT-style 1-bit + F16 scale per group      | ~1.13                | ~64 GiB            | 50 tok/s             | 19.5 tok/s   | **Very high** | 7–10 d   | NO — flat 1-bit collapses on DeepSeek per Unsloth; no MoE rescue evidence |
| 6 | GPTQ-Lloyd (real imatrix) on routed experts        | 2.25 (unchanged)     | 77 GiB             | 41 tok/s             | 17.0 tok/s   | Low          | 3 d + imatrix | NO — quality lever, no BW lever; orthogonal to this task |

(Effective routed bpw is the routed-expert sub-population weight average; the
**whole-model** average bpw is dominated by the F16 compressor/indexer (~5 %
of weights) and MQ4 attention (~10–15 %), so total-file shrinkage is
smaller than routed bpw delta implies. Numbers above include the full-file
estimate.)

## Detailed candidates

### Candidate 1: Mixed-precision MQ1.5-Lloyd + MQ2-Lloyd-sensitive (RECOMMENDED)

The Unsloth-style recipe ported to V4F. Pick per-tensor-class precision by
empirical sensitivity, not uniform-low-bpw.

**Recipe (initial, tunable after first PPL measurement):**

| Tensor class                          | bpw   | Format              | Bytes / 256 weights |
|---------------------------------------|------:|---------------------|--------------------:|
| Compressor (wkv, wgate)               | 16    | F16 (unchanged)     | 512                 |
| Indexer (i_kv_a, i_kv_b, i_q_a, i_q_b) | 16   | F16 (unchanged)     | 512                 |
| Attention (q/kv/wo) shared experts    | 4     | MQ4-G256 (unchanged) | 136                |
| MTP head                              | 16    | F16 (unchanged)     | 512                 |
| Routed expert **first 3 layers**      | 2.25  | MQ2-Lloyd (qt=19)   | 72                  |
| Routed expert **last 3 layers**       | 2.25  | MQ2-Lloyd (qt=19)   | 72                  |
| Routed expert **down_proj** (all layers) | 2.25 | MQ2-Lloyd (qt=19)  | 72                  |
| Routed expert **gate_proj, up_proj** middle 55 layers | **1.5** | **MQ1.5-Lloyd (new qt=21)** | **52**     |

**a) Bit budget per group of 256 weights — MQ1.5-Lloyd-G256 (NEW format,
qt=21).** Storage layout:

```
[0..4)    : 2 × fp16 codebook entries (cb[0], cb[1]) — only 2 distinct codepoints
[4..8)    : reserved / padding to keep 8-byte aligned header
[8..40)   : 32 bytes = 256 bits packed 1-bit indices (1/weight, 8/byte)
[40..52)  : 12 bytes = 256 sign-bits encoded in the index stream… NO.
            Cleanest: index is 1 bit (cb[0] or cb[1]). To get TERNARY
            ({-α, 0, +α}, the BitNet b1.58 / Q1.5 shape) you need 2 bits.
            But MQ2-Lloyd already provides that and is 2.25 bpw.
```

The honest re-derivation: **1-bit-per-weight + 1-bit-per-group sign + 4 B
header = 32 B data + 16 B header = 48 B / 256 = 1.5 bpw**. The codebook
has 2 entries (cb[0] = -α, cb[1] = +α); per-group sign-bit OR'd into the
index gives ternary expressiveness (sign + magnitude) at 1.0 bpw data +
1/256 bpw sign overhead. **Total per group: 32 B (data) + 16 B (header:
2 × fp16 codepoints + 4 B group sign mask + 4 B pad) = 48 B.**

Effective bpw: 48 × 8 / 256 = **1.5 bpw** exactly.

(Alternative layout: 1-bit data + 4 fp16 codepoints in header, treat the
1-bit as an index into a *2-element* codebook. Same 48 B/group. The
2-element codebook is non-ternary — pure binary 2-point Lloyd. Empirically
this is the FE2-bit "2-bit groups for non-outliers" payload at 1 bpw data +
1 byte per fp16 scale group. See FE2-bit at [arxiv 2311.16442](https://arxiv.org/abs/2311.16442)
for the precedent.)

**Total V4F file size estimate (current MQ2-Lloyd routed = 77 GiB
baseline):**

V4F has ~284 B params total, ~250 B in routed experts (~88 %). Of those
250 B routed:
- 3 + 3 + 55 (down_proj only) layers worth: ~60 B routed params kept at
  MQ2-Lloyd (2.25 bpw) = ~16.9 GB
- 55 × (gate + up) layers worth: ~190 B routed params at MQ1.5-Lloyd
  (1.5 bpw) = ~35.6 GB
- Non-routed (~34 B params, unchanged): ~12 GB

**Total ≈ 64.5 GB**, down from 77 GB MQ2-Lloyd baseline = **-16 % file size**.
Routed-expert bytes per token: 3.0 GB → ~2.2 GB = **-27 %**.

Actual V4F routed/non-routed split should be measured from `hfq_inventory`
before committing; the 88 % routed-params figure is from the recipe matrix
in `project_v4f_quant_inventory_finding`.

**b) Theoretical BW reduction vs MQ2-Lloyd at V4F MoE shape:**

At V4F MoE shape (M=4096, K=4096, K_TOP=6, B=16), routed-expert bytes per
token at MQ2-Lloyd = 6 × 4096 × 4096 × 72/256 = 28 MB **per expert
projection per token** (gate + up + down → ~84 MB × top_k routings). At
the mixed recipe (gate + up at MQ1.5, down at MQ2):

- Gate at MQ1.5: 6 × 4096 × 4096 × 48/256 = 18.9 MB per call
- Up at MQ1.5: same 18.9 MB per call
- Down at MQ2 (unchanged): 28 MB per call
- Total per top-k routing: 65.8 MB (vs 84 MB at all-MQ2) = **-21.7 %**

Across all 8 routed activations per layer × 58 layers, MoE chain
bytes/token = ~2.2 GB instead of 3.0 GB. The non-MoE chain (~0.3 GB:
attention, compressor, indexer, MTP, norms) is unchanged. Whole-token
total: ~2.5 GB instead of ~3.3 GB = **-24 %**.

**c) Projected prefill tok/s at BW-bound ceiling:**

Measured DRAM ceiling on Strix Halo: 189 GB/s (project_v4f_prefill_kernel_bw_audit).
Current 41–86 tok/s is well below ceiling for long prompts (so other
overheads dominate — chunk-write d2d, attention history reads — see
project_v4f_prefill_idle_gap_analysis_2026_05_20). Short-prompt 86 tok/s
is closer to the BW-bound regime.

Arithmetic:
- BW ceiling at current 3.3 GB/token = 189/3.3 = 57 tok/s if pure BW.
- At new 2.5 GB/token = 189/2.5 = 76 tok/s if pure BW.
- Measured/ceiling ratio at MQ2-Lloyd (long prompt): 41/57 = 72 %.
- Projecting the same ratio: 76 × 0.72 = **55 tok/s long prompt** (+34 %).
- Short prompt: 86 × (3.3/2.5) = **+32 %** projected upper bound; reality
  capped by single-chunk d2d overhead = realistically **+15-25 %**.

**More conservative bandwidth model.** Only the MoE-bound layers are
bandwidth-limited at full DRAM peak. MoE chain is 52 % of GPU time
(project_v4f_sequential_prefill_profile). Reducing the MoE chain by 22 %
yields 52 % × 22 % = **11 % whole-token speedup** = long-prompt 41 → ~46
tok/s; short-prompt 86 → ~96 tok/s. **Use the conservative number for
planning; the optimistic +34 % is a ceiling, not a forecast.**

**d) Quality literature evidence:**

- **Unsloth DeepSeek-R1 dynamic 1.58-bit** ([Unsloth blog](https://unsloth.ai/blog/deepseekr1-dynamic)):
  131 GB total file (full R1 is ~720 GB) with the same recipe shape as
  Candidate 1 — MoE layers at 1.58 bpw via IQ1_S, first 3 dense layers +
  shared experts + MLA attention at 4/6 bpw. Code-gen benchmark Flappy
  Bird score: 69.2 % (vs 91.7 % at IQ2_XXS / 2.22 bpw, vs 0 % at flat
  1.58 bpw). Demonstrates the recipe shape is viable; the exact PPL
  number is not published but Unsloth ship the model in production.
- **APEX-Mini** ([apex-quant Technical Report](https://github.com/mudler/apex-quant/blob/main/paper/APEX_Technical_Report.md)):
  middle-layer experts at IQ2_S (2.0 bpw), shared experts at Q4_K, edge
  layers at Q3_K → 12.2 GB DeepSeek build, PPL 7.088 vs uniform IQ2_M
  (11.3 GB) at PPL 7.303. The mixed-precision *wins* on PPL AND smaller
  size — same recipe family as Candidate 1.
- **MxMoE @ 2.25 bpw weight-only** ([arxiv 2505.05799](https://arxiv.org/abs/2505.05799)):
  DeepSeekV2-Lite at 2.25 bpw mixed: PPL **7.01** vs uniform GPTQ at 2.25
  bpw: PPL 8.49 (= -17.4 %). Wikitext-2. Mixed-precision over uniform
  works at the **same** total bpw — implying mixed at *lower* bpw can
  potentially match uniform at higher bpw.
- **APEX kurtosis observation**: routed expert weights have kurtosis 3.41
  (near-Gaussian, low outliers); shared experts have kurtosis 13.10
  (heavy-tailed). Direct support for keeping shared experts at higher
  precision while pushing routed lower — exactly the V4F recipe shape.

**e) Implementation cost estimate:**

| Component | Where | LoC | Days |
|-----------|-------|----:|-----:|
| `quantize_mq15g256_lloyd` quantizer | crates/hipfire-quantize/src/main.rs (parallel to existing `quantize_mq2g256_lloyd`) | ~140 | 1 |
| `quantize_mq15g256_lloyd_weighted` (imatrix) | same file | ~110 | 0.5 |
| `--format v4f-mixed-q15` recipe | quantizer CLI (per-tensor policy table) | ~80 | 0.5 |
| qt=21 dispatch in engine loader | crates/hipfire-runtime/src/hfq.rs (lines 595–625, follow MQ2/MQ3 lloyd arms) | ~10 | 0.25 |
| `DType::MQ15G256Lloyd` enum + 48 B group size constant | crates/rdna-compute/src/lib.rs (DType enum) | ~10 | 0.25 |
| `gemv_mq15g256_lloyd.hip` (dense decode) | kernels/src/ (copy `gemv_mq2g256_lloyd.hip`, swap decode to single-bit shift+mask) | ~140 | 1 |
| `gemv_mq15g256_lloyd_moe_gate_up_indexed.hip` | kernels/src/ (copy MQ2 sibling) | ~150 | 0.5 |
| `gemv_mq15g256_lloyd_moe_gate_up_indexed_batched_k4.hip` | kernels/src/ | ~180 | 0.5 |
| `gemv_mq15g256_lloyd_moe_down_indexed*.hip` (×2) | kernels/src/ | ~340 | 1 |
| `gemm_mq15g256_lloyd_moe_{gate_up,down}_*_wmma.hip` | kernels/src/ (prefill, lift from MQ2 WMMA) | ~360 | 1.5 |
| Dispatch wiring (dispatch.rs + qwen35/deepseek4 forward arms) | crates/rdna-compute, hipfire-arch-deepseek4 | ~150 | 1 |
| PPL + KLD validation harness extensions | crates/engine/examples/ | ~80 | 0.5 |

**Subtotal: ~1,650 LOC across 14 files, ~8 days end-to-end** with one
engineer. Validation (PPL + KLD sweep) adds ~1 day GPU time. Realistic
calendar window: **6–10 days** including review.

**Reusable from prior shipped work:**
- Lloyd centroid-fit loop (8 iter, rayon parallelization): identical
  structure to `quantize_mq2g256_lloyd`. Just N=2 centroids instead of 4.
  `quantize_mq3g256_lloyd` has the N=8 version too — interpolating to N=2
  is mechanical.
- All 6 MQ2-Lloyd MoE indexed kernels (`gemv_mq2g256_lloyd_moe_*`) — direct
  template. Replace the 2-bit unpacking macro `DOG2_LDS` with a 1-bit
  unpacking variant `DOG1_LDS` that shifts a packed uint32 by 1 instead
  of 2; the codebook lookup becomes `cb_lds[(pk >> i) & 1u]` instead of
  `& 3u`. ~70 % code reuse.
- All 3 MQ2-Lloyd WMMA prefill kernels (`gemm_mq2g256_lloyd_moe_*_wmma.hip`)
  — same template, 1-bit unpack inside K-tile loop.

**f) Risk assessment:**

What would falsify Candidate 1 during implementation:

1. **PPL regression > 5 % on wikitext2-test at ctx=512.** Most likely
   failure mode: MQ1.5's 2-centroid codebook is too coarse for `gate_proj`
   and `up_proj` in middle layers despite SwiGLU's noise filtering.
   Recovery: shrink the MQ1.5 footprint (move more layers back to MQ2),
   re-measure. If MQ1.5 only on the 8 lowest-magnitude-norm experts
   (~10 % of routed weights) still regresses, MQ1.5 is dead on V4F.
2. **Coherence regression — attractor on code-gen prompts (fibonacci_c,
   lru_summary).** Per `project_mq2_lloyd_moe_rescue` Phase 3c, the
   coherence harness must be run alongside PPL. PPL can win while
   attractors break code-gen. Mitigation: include `fibonacci_c` +
   `lru_summary` + `python_classic` in any Candidate 1 sanity run before
   the PPL sweep is even reported.
3. **MQ1.5 kernel register spill.** MQ2-Lloyd is 29 VGPR / 22-24 SGPR / 0
   spills on gfx1100 per `project_mq2_lloyd_moe_rescue` Phase 2. MQ1.5
   has FEWER registers needed (2-entry vs 4-entry codebook → LDS 2 fp32
   × 4 groups = 8 floats instead of 16). Expected: comfortably under
   spill threshold. Verify with `gfx-kernel-metadata` skill on first
   kernel build.
4. **Quantize-time too slow.** Lloyd at N=2 codepoints converges in 3-5
   iterations (faster than N=4) but runs over ~190 B params worth of
   blocks (~740 M groups). At ~3 μs/group on 24-core rayon = ~37 min
   wall. Acceptable for an offline quantize step.
5. **`hfq_inventory` check fails.** The quantizer must NOT compress
   compressor or indexer tensors. Add unit-test gate per
   `project_v4f_compressor_must_stay_f16` — if compressor row in
   inventory ≠ qt=1, fail the build.

**g) PPL/KLD validation methodology — see §Validation methodology below.**

### Candidate 2: MQ2-Lloyd + sparse F16 outlier preservation

Standalone form makes the file BIGGER, not smaller — direction wrong.
**Only ship as a compose-on-top quality-recovery for Candidate 1's MQ1.5
layers if they fall short of the PPL gate.**

**Mechanism:** for each routed-expert tensor, identify the top-K weights
by `|w| × sqrt(act_importance[col])` (where `act_importance` is the
imatrix-derived per-column importance per `project_lloyd_imatrix_fwht_channel_mixing`).
K = 0.5 % of total weights for that tensor. These are NOT quantized; they
are stored as `(row_idx: u16, col_idx: u16, value: f16) = 6 B/outlier`
in a parallel sparse list per tensor. The bulk weights at those (row,col)
positions are quantized as if they were zero (let the codebook absorb the
hole; the outlier slot supplies the true value at runtime).

**Runtime cost:** new tiny kernel `apply_sparse_outliers` that scatter-
writes outlier values into the GEMV output for the affected rows. ~6 B ×
0.005 × tensor_size = 0.003 × tensor bytes overhead. On V4F routed bytes,
that's ~+0.7 GB across the whole model. ~+1 % per-token BW.

**Quality literature evidence:**
- **FE2-bit** ([arxiv 2311.16442](https://arxiv.org/abs/2311.16442)):
  0.5 % sparse outlier cap on Llama2-7B at 2-bit groups gives **+0.5 %
  accuracy** at **<3 % extra bits**. The accuracy delta translates to
  roughly 30–50 % of the gap closure between flat-MQ2 and MQ4 on PPL on
  dense Llama models.
- **SqueezeLLM** ([arxiv 2306.07629](https://arxiv.org/pdf/2306.07629)):
  similar dense-and-sparse decomposition at 0.05 % outlier ratio
  recovers most of the FP16 → 3-bit gap on Llama-7B / Llama-13B.
  Smaller outlier fraction → smaller perplexity benefit.

**Engineering cost:**

| Component | LoC | Days |
|-----------|----:|-----:|
| Outlier-selection pass in quantizer (kurtosis / activation-magnitude top-K) | ~120 | 1 |
| Sparse `.hfq` block layout (new section per tensor) | ~80 | 0.5 |
| Engine loader for sparse-outlier section | ~60 | 0.5 |
| `apply_sparse_outliers_residual` kernel + dispatch | ~150 | 1 |
| PPL+KLD validation | — | 0.5 |
| **Total** | **~410** | **~3.5 d** |

**Risk:** scatter-write into the GEMV output adds a memory-traffic op
that doesn't merge cleanly with the WMMA prefill path. Decode is fine
(it's a small fixup at the end of the row-accumulate). Prefill needs
to be benched.

**When to ship:** only after Candidate 1 lands and its PPL is measured.
If PPL gap > 5 %, layer Candidate 2's outlier preservation on the MQ1.5
sub-population and re-measure. If Candidate 1 already passes the gate,
Candidate 2 is unneeded.

### Candidate 3: Per-row variable codebook (deferred)

**Mechanism:** Header expansion. The current MQ2-Lloyd-G256 header is
8 B (4 × fp16 centroids). Allow the header to be either 8 B (4 entries,
existing format, 2.25 bpw) or 4 B (2 entries, MQ1.5-style, 1.25 bpw
data + 1.5 bpw including header), chosen per-row by a 1-bit flag.

**Bit budget:** flag overhead = 1 bit/row. Per 4096-element row at
M=4096: 4096 × 1 = 4096 bits = 512 B of flags total per layer's
gate_proj. Negligible.

**Theoretical bpw on V4F routed experts:** if 30 % of rows use 4-entry
codebook (the heavy-tailed / outlier rows by L2-norm), 70 % use 2-entry
codebook (the near-zero distributions): 0.30 × 2.25 + 0.70 × 1.5 =
**1.725 bpw**. Roughly tied with Candidate 1, slightly better.

**Why deferred:** the kernel `cb_lds` LDS pre-load doesn't know whether
a row is 2-entry or 4-entry until it reads the row's header. Either:
- (a) Header is read once per row, kernel branches on header type — adds
  per-row branch divergence within a warp. RDNA32-wave warps span 32
  rows; if rows mix 2-entry/4-entry within a wave, divergence cost
  cancels the bpw win.
- (b) Sort rows so all 4-entry rows are first, all 2-entry rows last —
  requires a `row_permute_table` per tensor, adds runtime cost everywhere
  the row index is used (RMSNorm, residual write-back).

Either way, ~+8 days engineering for a ~5 % marginal improvement over
Candidate 1. **Defer.**

### Candidate 4: MQ3-Lloyd on top-frequency experts (REJECT — wrong direction)

The hypothesis: V4F routed experts have heterogeneous activation
frequencies (top-K = 6 out of N=256). High-frequency experts (the ones
hit on >10 % of tokens) deserve higher precision; low-frequency
(<1 % of tokens) tolerate lower precision.

**This is correct as an academic observation but the wrong direction for
this PR.** It makes the model **bigger** (MQ3-Lloyd is 112 B/group = 3.5
bpw; MQ2-Lloyd is 72 B/group = 2.25 bpw), shifting per-token bytes UP.

The right way to use this insight is the *inverse*: Candidate 1's mixed
recipe already protects the *most-sensitive* layer positions (first 3,
last 3, all down_proj). If we have additional headroom, expand the MQ1.5
footprint to lower-frequency *experts* within the middle layers. That's
Candidate 1.5 — a follow-on, not a separate candidate.

### Candidate 5: Uniform 1-bit + F16 scale per group (REJECT)

48 B/group with a 16 B header and 32 B of 1-bit data = 1.5 bpw, same
as Candidate 1's MQ1.5 format. But applied *uniformly* without the
mixed-precision sensitivity sheath, this is the recipe Unsloth showed
collapses to **0 % on Flappy Bird** (vs 69 % for mixed 1.58). Per
`docs/QUANTIZATION.md` line 31-36 and the existing internal sweep, MQ2
*uniform* (without Lloyd codebook) collapses on every model size tested.
Strong prior: pure 1.5 bpw uniformly across all routed experts will
collapse V4F.

**Reject.**

### Candidate 6: GPTQ-Lloyd with real imatrix (REJECT for this PR)

`project_gptq_lloyd_pretendgptq_finding` already shows the current
GPTQ-Lloyd sequential pass adds noise without a real imatrix. With a
real imatrix, expected improvement on the existing MQ2-Lloyd file is
~5–10 % PPL — a quality lever, not a BW lever. **Same bytes per
weight; no prefill/decode speedup.** Orthogonal to this PR.

(Note: a calibrated imatrix is *prerequisite* for Candidate 1's PPL
sweep producing reliable numbers — see §Validation. Build it as the
first task, but ship the recipe change separately.)

## Validation methodology

### Phase A — Establish KLD baseline (prerequisite, ~1 day GPU)

The constraints say "≤ 5 % KLD regression" but no KLD number exists for
the current MQ2-Lloyd build. Establish:

1. **F16 reference run** (no quant): if the V4F bf16 GGUF source weights
   are still on disk, dequantize-to-F16 mock the routed experts (or load
   the bf16 GGUF directly via a llama.cpp comparison run). Sample 200
   prompts from wikitext2-test, capture top-100 next-token logits at
   each of ~50 positions per prompt. ~10K (prompt, position, logits)
   triples.
2. **MQ2-Lloyd reference run** (current build): same 200 prompts, same
   positions, capture same logits.
3. **KLD baseline**: `KL(MQ2-Lloyd || F16)` averaged over all
   (prompt, position) pairs. Record as `KLD_baseline_2026-05-21`.
4. **PPL recalibration** at ctx=512: confirm 9.82 still reproduces on
   current binary (no drift since the playbook number was recorded).

Cost: ~3 hours GPU on F16 reference + ~3 hours on MQ2-Lloyd =
~6 hours. Use `crates/engine/examples/perplexity.rs` extended with a
KLD-collection mode (~100 LoC additional). If the V4F bf16 GGUF is not
accessible, fall back to using the *MQ2-Lloyd* build itself as the
reference and measure KLD of new candidates against MQ2-Lloyd directly
(weaker signal but still useful).

### Phase B — Quantize and PPL-sweep Candidate 1

1. Build Candidate 1's MQ1.5-Lloyd kernels (decode + dense + MoE indexed
   + WMMA prefill). Run `cargo test --release` for hot-path probes.
2. Quantize V4F to `v4f.mixed-q15.hfq`. Run `hfq_inventory` — confirm
   compressor + indexer rows show qt=1 (F16).
3. Smoke-test load: run a coherence-probe single prompt at greedy decode.
   Confirm no panic, fluent output, non-zero distinct tokens.
4. PPL on wikitext2-test at ctx={256, 512, 1024}. Warmup 8 tokens per
   chunk. Same harness as the playbook table.
5. KLD: same 200 prompts × 50 positions sampling. Compute
   `KL(Candidate1 || F16)` and `KL(Candidate1 || MQ2-Lloyd)`. Report
   both.
6. Coherence harness: 10 prompts including `fibonacci_c`, `lru_summary`,
   `python_classic`, greedy decode, attractor/n-gram-density gates per
   CLAUDE.md DFlash gate rules.

### Phase C — Gate decision

| Metric | Threshold | Action if exceeded |
|--------|-----------|---------------------|
| PPL@512 vs MQ2-Lloyd baseline | ≤ +5 %    | shrink MQ1.5 footprint by 1 layer (move L_middle_5 back to MQ2), re-run B |
| PPL@1024 vs baseline           | ≤ +5 %    | same as above (long-ctx attention errors compound) |
| KLD vs F16 reference (or MQ2-Lloyd if no F16) | ≤ +5 % | same as above |
| Coherence ok-count             | ≥ 6/10 (matches current MQ2-Lloyd) | reject candidate; investigate per-prompt attractors |
| `fibonacci_c` per-prompt verdict | ok (not warn, not fail) | reject — code-gen attractor is the historical V4F failure mode |

### Phase D — Iterate or ship

If Phase C passes: measure prefill (long + short) + decode tok/s with
fresh-process protocol per CLAUDE.md (3-5 runs, gpu-tcas-coordinated,
prompt md5 recorded). If tok/s gain ≥ 10 %, ship. If <5 %, audit kernel
register usage and BW utilization before claiming the recipe is fully
extracted.

If Phase C fails on quality: layer Candidate 2 (sparse outlier
preservation) on the MQ1.5 tensors and re-run B+C. If that still fails,
the candidate is rejected — fall back to MQ2-Lloyd-baseline and
investigate other paths (asymmetric attention quant, smaller compressor
basis).

## Recommended first step

**Build the F16 KLD baseline first, then quantize one candidate.** Two
parallel one-day tasks:

1. **Task A (KLD baseline) — 1 day GPU**: extend `crates/engine/examples/perplexity.rs`
   with a `--mode kld --reference <path-to-f16-build-or-bf16-gguf>` flag.
   Run on current `v4f.mq2lloyd-f16compress.hfq` with the V4F BF16
   GGUF as reference (or `v4f.antirezQ8.hfq` as a proxy reference if
   BF16 source not on disk). Record numbers in
   `benchmarks/results/v4f_kld_baseline_2026-05-21.md`.
2. **Task B (Candidate 1 first slice) — 2 days CPU + 4 hours GPU**:
   implement ONLY the MQ1.5 quantizer + dense `gemv_mq15g256_lloyd.hip`
   decode (skip MoE-indexed + WMMA prefill kernels). Quantize a
   *test* V4F with ONLY one routed-expert layer (layer 30, the middle)
   downgraded to MQ1.5 — everything else MQ2-Lloyd baseline. Run PPL at
   ctx=256. If PPL holds to within +2 % of baseline, the format is
   viable and the full mixed-recipe + MoE-indexed kernels become worth
   building. If it regresses > +5 % on a single layer, MQ1.5 is dead
   on V4F.

This sequencing surfaces the kill criterion in 3 days instead of 10
and saves the engineering team from building the full mixed-recipe
stack if MQ1.5 fundamentally doesn't survive on V4F routed experts.

---

## Sources

- Unsloth DeepSeek-R1 Dynamic 1.58-bit (production sub-2-bpw MoE recipe):
  <https://unsloth.ai/blog/deepseekr1-dynamic>
- APEX Mini / APEX Quant (MoE mixed-precision technical report):
  <https://github.com/mudler/apex-quant/blob/main/paper/APEX_Technical_Report.md>
- MxMoE (ICML 2025, mixed-precision MoE quant):
  <https://arxiv.org/abs/2505.05799>
- QuIP# (Hadamard incoherence + lattice codebooks, 2-bit):
  <https://arxiv.org/pdf/2402.04396>
- SqueezeLLM (dense-and-sparse decomposition, outlier preservation):
  <https://arxiv.org/pdf/2306.07629>
- FE2-bit (0.5 % sparse outlier + 2-bit groups on GPU):
  <https://arxiv.org/abs/2311.16442>
- AQLM (additive vector quantization, 2-3 bit Pareto):
  <https://arxiv.org/pdf/2401.06118>
- llama.cpp Tensor Encoding Schemes (IQ1_S = 1.5 bpw, IQ1_M = 1.75 bpw,
  IQ2_XXS = 2.0625 bpw, IQ2_S = 2.5 bpw):
  <https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes>
- Internal: `docs/plans/v4f-quant-optimization-playbook.md`,
  `docs/plans/mq-sub4bit-roadmap.prd`,
  `docs/plans/mq-sub4bit-research-queue.md`,
  `docs/QUANTIZATION.md`.
- Internal memory: `project_mq2_lloyd_moe_rescue.md`,
  `project_v4f_compressor_must_stay_f16.md`,
  `project_v4f_quant_inventory_finding.md`,
  `project_gptq_lloyd_pretendgptq_finding.md`,
  `project_lloyd_imatrix_fwht_channel_mixing.md`,
  `project_v4f_prefill_kernel_bw_audit_2026_05_19.md`,
  `project_v4f_sequential_prefill_profile.md`,
  `project_v4f_prefill_idle_gap_analysis_2026_05_20.md`.
