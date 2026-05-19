# antirezQ8 vs mq2lloyd-f16compress perf+quality benchmark
# wikitext-2-test, all session optimizations applied (post-2026-05-19)

ctx | mq2lloyd PPL | antirezQ8 PPL | PPL Δ | mq2lloyd tok/s | antirezQ8 tok/s | tok/s Δ
----|--------------|---------------|-------|----------------|-----------------|--------
128 | 11.58        | 11.67         | +0.78% (Q8 worse) | 15.94 | 12.37 | -22.4%
1024| 7.33         | 7.08          | -3.41%            | 12.21 | 10.20 | -16.5%
2048| 6.28         | 6.06          | -3.51%            | 7.18  | 6.54  | -8.9%

Structural trends:
- Q8 attention perf penalty SHRINKS with ctx (22 → 17 → 9%): decode becomes
  less weight-bandwidth-bound and more attention-compute-bound as ctx grows.
- Q8 PPL benefit GROWS with ctx (-0.8% → +3.4% → +3.5%): more positions
  contributing to softmax means attention precision matters more. Plateaus
  by ctx=1024.

PPL crossover: somewhere between ctx=128 and ctx=1024, Q8 attention switches
from "worse" to "better" than MQ4.

MQ8 (FWHT-rotated symmetric int8) potential:
- Same precision as Q8F16 (8-bit symmetric)
- 5% smaller storage (8.06 bpw vs 8.5)
- dp4a compute path (4× VALU vs fp16 — but attention is bandwidth-bound)
- Current MQ8G256 kernel is BASIC (190 LOC, no K4+LDS/multirow/per-arch)
  vs MQ4 which has 24 kernel files with full optimization tree.

To make MQ8 attention competitive with MQ4 attention perf at 2× precision,
estimate 1.5-2 weeks of kernel work (port K4+LDS pattern from
gemv_hfq4g256.gfx1100.hip, add multirow, gfx-arch tuning).
