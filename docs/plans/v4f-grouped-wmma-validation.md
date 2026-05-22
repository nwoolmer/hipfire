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
