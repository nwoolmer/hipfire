# V4F scatter-grouped WMMA MoE — validation plan

**Date:** 2026-05-22
**Goal:** Determine whether porting qwen35's `gemm_hfq4g256_moe_grouped_wmma_k2`
to MQ2-Lloyd would speed up V4F prefill, BEFORE writing ~600 LoC of kernel
code that might be falsified post-implementation.

## Premise being tested

Qwen35-A3B saw +114% prefill from a scatter-grouped WMMA MoE kernel
(1396 → 2983 tok/s on gfx1100). Hypothesis: porting that pattern to
V4F's MQ2-Lloyd MoE would give similar uplift at the V4F shape.

## Why the premise may not hold at V4F shape

Two structural risks not present in the qwen35 measurement:

### Risk 1 — Tile fill rate

WMMA's 16-slot C-fragment column dimension means each grouped tile
holds 16 (batch_position, krank) slots routed to the SAME expert.
For grouped-WMMA to be efficient, tiles must be NEARLY FULL.

| Model | K_TOP | n_experts | B | Total slots | Slots/expert avg |
|---|---|---|---|---|---|
| Qwen35-A3B | 8 | 64 | 256 | 2048 | **32** |
| V4F | 6 | 256 | 16 | 96 | **0.4** |

At avg 0.4 slots/expert, MOST tiles will hold 1-2 slots × 14-15 wasted
= ~10% C-fragment fill = the same SLOWER pattern as single-col WMMA
(measured 3× slower earlier this session).

### Risk 2 — Decode cost

Qwen35's win was on HFQ4 (4-bit affine `scale*q + zero`, 1 FMA per
weight decode). MQ2-Lloyd is codebook lookup (3-4 ALU per weight).
Higher decode cost = less ALU budget for the WMMA pipeline to fill.

## Validation gates

Each gate is YES / NO with a HARD THRESHOLD set in advance.

### Gate 1 — Tile fill rate (Day 1)

**Measure**: dump `topk_indices_batch` from a v4f_chat prefill at B=16,
706-token prompt. Compute per-layer:
- Slots-per-expert histogram
- "Tile fill at 16-slot block size" — if 16 slots are sorted by expert,
  how many of the resulting 16-wide blocks would have how many
  same-expert occupants?

**Threshold**:
- If average tile utilization **< 50 %** → Path B is structurally
  blocked at V4F shape. Stop.
- If ≥ 75 % → Path B is structurally viable. Continue.
- If 50-75 % → marginal. Continue to Gate 2 but expect smaller gains.

### Gate 2 — qwen35 grouped WMMA at V4F shape (Day 2)

**Measure**: benchmark the existing qwen35 `gemm_hfq4g256_moe_grouped_wmma_k2`
kernel at V4F shape (M=4096, K=4096, K_TOP=6, B=16, n_experts=256)
against V4F's scalar K4 (`gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4`).

The HFQ4 vs MQ2-Lloyd quant-format difference doesn't matter for
this question — we're testing whether the SCATTER-GROUPED PATTERN
amortizes well at V4F's B=16 / K_TOP=6 / n_exp=256.

Inputs:
- Random HFQ4 weights for 256 experts (any synthetic byte pattern;
  perf isn't sensitive to weight content)
- Random topk_indices matching V4F's actual routing distribution
  (sampled from Day 1)
- Synthetic activation tensor

Repeat N=20 iters with `device_synchronize` after each, take median.

**Threshold**:
- If grouped-WMMA ≥ **1.5×** scalar K4 → port the kernel.
- If grouped-WMMA ≤ **1.0×** → port is doomed by tile-fill;
  Path B dead.
- 1.0× – 1.5× → port might still be worth it because MQ2-Lloyd's
  decode cost is higher than HFQ4's (potentially more upside).

### Decision matrix

| Day 1 fill rate | Day 2 speedup | Decision |
|---|---|---|
| ≥ 75 % | ≥ 1.5× | **GO** — port MQ2-Lloyd grouped WMMA, ~600 LoC, 4-6 days |
| ≥ 75 % | 1.0–1.5× | **MAYBE** — re-derive after considering MQ2 vs HFQ4 decode cost |
| ≥ 75 % | < 1.0× | **STOP** — overhead exceeds win, Path B dead at V4F shape |
| 50–75 % | ≥ 1.5× | **MAYBE** — but expect 0.5× of qwen35's +114%. ~+50% |
| 50–75 % | < 1.5× | **STOP** — tile-fill + scatter overhead dominate |
| < 50 % | any | **STOP** — Path B structurally blocked at V4F shape |

## What this plan AVOIDS

- Writing the full MQ2-Lloyd-grouped-WMMA kernel before knowing if the
  pattern even works at V4F shape. (~600 LoC of risk.)
- Repeating the "looked good in microbench, falsified in production"
  pattern that hit single-col WMMA (3× slower), ZA-fused (-1.5%),
  fused silu+rotate (-1.4%), and others.

## Output

Both gate results saved as JSON + markdown in `docs/plans/`. Decision
documented as a follow-up commit on this same plan file.

## Estimated effort

Day 1: ~80 LoC (instrumentation + analysis script). ~3 hours.
Day 2: ~150 LoC (test harness for the existing qwen35 kernel at V4F
shape + comparison driver). ~5 hours.
Total: 1 day to a clean go/no-go decision.

---

## Verdict (2026-05-22) — Path B is DEAD at V4F shape

### Gate 1 — FAIL (decisive)

Measured by env-gated `HIPFIRE_V4F_DUMP_TOPK` dump of `moe_topk_indices
_batch` from a real v4f_chat prefill (708-token prompt, B=16, all 43
layers, model `v4f.mq2lloyd-q8.hfq`). Analysis script:
`/tmp/analyze_topk.py`.

| Metric | Result |
|---|---|
| Average tile fill (when 96 slots sorted by expert, grouped in 16-wide tiles) | **15.1 %** (threshold: 50 %) |
| Median tile fill | 15.4 % |
| Max tile fill (best layer) | 21.4 % |
| Unique experts active per layer | mean 41.6 of 256 (min 28, max 73) |
| Longest same-expert run per layer | mean 12.2 slots (max 15) |

**Histogram of same-expert run lengths in a 16-slot tile:**

| Run length | Tiles | Share | Cumulative |
|---|---|---|---|
| 1 | 1064 | **59.4 %** | 59.4 % |
| 2 | 306 | 17.1 % | 76.5 % |
| 3 | 142 | 7.9 % | 84.5 % |
| 4 | 71 | 4.0 % | 88.4 % |
| 5–8 | 123 | 6.9 % | 95.3 % |
| 9–15 | 84 | 4.7 % | 100 % |

**Interpretation.** A grouped WMMA kernel that requires 16-slot
tiles would see 59 % of tiles holding a single slot — the WMMA C
fragment would be 15 of 16 columns idle on those tiles. The
expected throughput at 15 % fill ≈ 0.15 × (raw WMMA win). Even if
a hypothetical V4F-shape grouped WMMA kernel doubled the scalar
K4 path at 100 % fill, at 15 % fill it would be 0.3 × scalar —
strictly slower.

This is structurally the SAME failure mode as the previously
measured "single-col WMMA at 3× slower" experiment, just expressed
through the routing-distribution lens.

### Gate 2 — not run (Gate 1 alone is decisive)

The qwen35 family of grouped-WMMA-MoE kernels (e.g.
`gemm_hfq4g256_moe_grouped_mmq.gfx1151.hip`) exists on origin/master
but isn't on this branch. Running Gate 2 by cherry-picking the kernel
would require ~150 LoC of harness work. Skipped: Gate 1's 15.1 % tile
fill makes Gate 2's outcome arithmetically determined regardless of
how good the kernel is.

### Final decision: STOP

Do NOT port qwen35-style grouped-WMMA-MoE to MQ2-Lloyd. The V4F
routing distribution (K_TOP=6, n_experts=256, B=16) gives 0.4 slots
per expert on average — there is nothing for scatter-grouped tiles
to amortize across.

### Routing-distribution arithmetic — what would unblock Path B

To raise tile fill above 50 %, we'd need average slots-per-expert
≥ 8. With V4F's K_TOP=6 and n_experts=256, that requires B such
that `B × K_TOP / n_experts ≥ 8` → `B ≥ 341`. V4F's batched prefill
sweet spot is B=16 (per the earlier sweep — B=128+ regresses from
L2 / Infinity Cache spill). B=341 is not realistic.

### Where the prefill speedup lever actually lives

- **Path A (sub-2-bit MoE quant, per `docs/plans/v4f-sub-mq2-quant-
  research.md`)**: reduces per-call MoE bytes proportional to bpw
  reduction. Doesn't depend on tile-fill; works at any B.
- **Path C (lower-precision attention K/V)**: 14.3 % of GPU time at
  F32; switching to F16 KV ≈ 7 % prefill gain. Quality risk.

Both of these are unblocked by the Gate 1 finding (they don't
require WMMA C-fragment fill on the MoE side).
