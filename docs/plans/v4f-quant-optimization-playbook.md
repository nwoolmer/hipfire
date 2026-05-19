# V4F quant optimization playbook (2026-05-19)

What to rebuild and why, in order of expected payoff. All numbered actions
require access to V4F source weights (BF16 GGUF or safetensors directory)
and an imatrix file generated against them.

## Baseline state on disk

PPL on wikitext2-test, current files at `/data/hipfire-models/` and
`/home/nick/.hipfire/models/`:

| ctx  | mq2lloyd-f16compress | fp4fix (Lloyd+MQ4 compressor) | mq2-gptq-all | antirezQ8 |
|------|----------------------|--------------------------------|--------------|-----------|
| 128  | 11.58                | 20.96                          | 38.74        | 11.67     |
| 1024 |  7.33                | 10.91                          | 14.36        |  7.10     |
| 2048 |  6.28                |  8.76                          | 11.89        |  6.01     |

The fp4fix file (Lloyd-routed + MQ4G256 compressor) was added to the
table 2026-05-19 to give an isolation point: it has the same compressor
quantization as mq2-gptq-all but plain Lloyd on routed experts. The
deltas split the mq2-gptq-all regression into two clean axes:

* **F16 → MQ4 compressor alone**: +81% PPL @ ctx=128, +49% @ 1024,
  +40% @ 2048. Per `project_v4f_compressor_must_stay_f16` — compressor
  + indexer tensors are by far the most quant-sensitive in V4F.
* **GPTQ-Lloyd algorithm on top of MQ4 compressor**: +85% @ ctx=128,
  +32% @ 1024, +36% @ 2048.

So mq2-gptq-all has TWO compounding bugs, each contributing roughly
half of its total regression.

Inventory (from `hfq_inventory` probe in `hfq_block_diag` test module):

| Family            | mq2lloyd        | gptq-all        | antirezQ8       |
|-------------------|------------------|------------------|------------------|
| Attention         | MQ4G256          | MQ4G256          | **Q8F16**        |
| Shared experts    | MQ4G256          | MQ4G256          | **Q8F16**        |
| Routed experts    | MQ2-Lloyd        | MQ2-Lloyd (GPTQ) | MQ2-Lloyd        |
| Compressor        | F16              | MQ4G256          | F16              |
| Total size        | 77 GiB           | 76 GiB           | 80 GiB           |

**Key surprise from the inventory probe**: the antirezQ8 file does NOT
implement the antirez asymmetric routed-expert split (gate_up=MQ2 +
down=MQ3). All routed experts are MQ2-Lloyd. The "antirez" name only
captures Q8 attention. The TRUE antirez recipe has never been built.

## Universal rule for any V4F rebuild

**Compressor + indexer tensors MUST stay F16** (quant_type 1). Inspect
the inventory before shipping any rebuild — if the compressor row
shows qt=13 (MQ4G256) instead of qt=1 (F16), expect +40-81% PPL
regression. The 450 MB savings is never worth it on V4F.

The quantizer's default behaviour preserves F16 for these tensors;
mq2-gptq-all and fp4fix both inadvertently compressed them via some
prior build path that has since been fixed. Always run `hfq_inventory`
on the output and confirm compressor row is F16 before committing the
artifact.

## Action 1 — Build the true antirez recipe (highest expected payoff)

Combines Q8 attention (the antirezQ8 win) with the asymmetric routed-
expert split that the recipe was named after. Per the
`antirez_mq3_to_mq2_downgrade_cost` probe, MQ3-down provides a
+226-265% per-tensor MSE reduction on V4F-realistic distributions vs
MQ2-down. The 13 GB file-size tax buys real precision.

```
cargo run --release -p hipfire-quantize -- \
    --in <path-to-V4F-bf16.gguf> \
    --out /data/hipfire-models/v4f.antirez-true.hfq \
    --format mq4-mqlloyd-antirez \
    --allow-mq3-lloyd \
    --imatrix <path-to-v4f-real.imatrix.gguf>   # optional but recommended
```

Expected size: ~93 GiB (80 + ~13 GiB MQ3 down tax).
Expected PPL at ctx=2048: significantly below antirezQ8's 6.01 — the
asymmetric split protects the residual-write direction, and Q8 attention
keeps the high-ctx attention precision. No empirical PPL number yet —
build and verify.

Validation:
```
target/release/examples/v4f_perplexity \
    /data/hipfire-models/v4f.antirez-true.hfq \
    dev/bench/data/wikitext2-test.txt \
    --ctx 2048 --warmup 8 --moe 1
```

## Action 2 — Build a real imatrix file for V4F

Required for Action 3 and meaningful Lloyd quality wins. Uses
`hipfire-runtime/examples/imatrix_collect.rs` (Tier 2 wrapper around
llama.cpp's `llama-imatrix`).

```
cargo run --release -p hipfire-runtime --example imatrix_collect -- \
    --bf16-gguf <path-to-V4F-bf16.gguf> \
    --corpus    benchmarks/quality-baselines/refs/c4-sample.txt \
    --output    /data/hipfire-models/v4f.imatrix.gguf \
    --n-ctx 2048 --n-batch 512 --chunks -1
```

Expected runtime: several hours on this hardware.

Output is a GGUF file consumable by hipfire-quantize via `--imatrix`.

**Caveat per `project_lloyd_imatrix_fwht_channel_mixing` memory**: the
current `quantize_mq2g256_lloyd_weighted` applies imatrix col_weights
AFTER the inner FWHT, which washes out per-channel signal through
rotation. A real imatrix will produce SOME improvement but less than
the literature suggests until the pre-FWHT Lloyd refactor lands. Still
worth building — it's the prerequisite for any imatrix-aware future
rebuild.

## Action 3 — Rebuild mq2lloyd-f16compress with real imatrix

Once the imatrix file from Action 2 exists:

```
cargo run --release -p hipfire-quantize -- \
    --in <path-to-V4F-bf16.gguf> \
    --out /data/hipfire-models/v4f.mq2lloyd-imatrix.hfq \
    --format mq4-mq2lloyd-imatrix \
    --imatrix /data/hipfire-models/v4f.imatrix.gguf
```

Expected PPL improvement: 5-15% over current mq2lloyd-f16compress at all
contexts. Limited by the FWHT-channel-mixing issue described above —
the full imatrix value will only land after the pre-FWHT Lloyd refactor.

Same file size as current mq2lloyd-f16compress.

## Action 4 — Don't rebuild mq2-gptq-all unless real imatrix lands

The current file is broken: pretend-GPTQ + unit imatrix produces a
strictly-worse-than-Lloyd build (1.9-3.3× worse PPL). Per the
`gptq_damping_probe` test module, GPTQ-Lloyd at any damping > 0 with
unit imatrix is +5-65% MSE worse than plain Lloyd on every V4F-realistic
distribution.

The production defaults shipped 2026-05-19 (HIPFIRE_GPTQ_DAMPING=0.0
with a warning when damping > 0 is used against a unit imatrix) prevent
future broken builds. At d=0 the function is byte-identical to plain
Lloyd, so rebuilding with d=0 + unit imatrix would just duplicate
mq2lloyd-f16compress.

**If** you have a real imatrix (Action 2 complete), GPTQ-Lloyd at d=0.3
with real col_weights might give a modest win:

```
HIPFIRE_GPTQ_DAMPING=0.3 cargo run --release -p hipfire-quantize -- \
    --in <V4F-bf16.gguf> \
    --out /data/hipfire-models/v4f.mq2-gptq-imatrix.hfq \
    --format mq4-mq2lloyd-gptq-all \
    --imatrix /data/hipfire-models/v4f.imatrix.gguf
```

But per the probe, even with real imatrix the d > 0 sequential pass costs
+5-10% MSE; the underlying value is the imatrix-weighted codebook fit,
which the simpler `mq4-mq2lloyd-imatrix` (Action 3) delivers without the
GPTQ noise tax.

## Action 5 — Pre-FWHT Lloyd refactor (research-level, defer)

Would unlock the full imatrix benefit by doing Lloyd-imatrix in the
natural channel basis, then applying FWHT to centroids for the on-disk
format. Substantial pipeline work:

- New `quantize_mq2g256_lloyd_prefwht_imatrix` function
- Runtime kernel changes (centroid math is now in the rotated basis at
  runtime, but on-disk values stay rotated — kernel is unchanged if we
  apply FWHT to centroids at quant time)
- Validation harness comparing pre-FWHT-imatrix-Lloyd vs current
  post-FWHT-imatrix-Lloyd PPL

Expected payoff: another 10-20% MSE improvement on top of Action 3,
which would translate to ~5% PPL improvement at high ctx.

Defer until Actions 1-3 are done and prove value.

## Optimizations already shipped (commits in feat/deepseek4-v4f)

| Commit  | Change | Payoff |
|---------|--------|--------|
| f8cd234 | Lloyd `max_iter` 8 → 16 | +0.4-0.9% MSE on next rebuild |
| f8cd234 | GPTQ `damping` default 0.8 → 0.0 + warning | prevents future broken GPTQ-all builds |

Probes that produced these findings (`crates/hipfire-quantize/src/main.rs`):
- `gptq_damping_probe::sweep_v4f_like_distributions` — GPTQ regression at d>0
- `gptq_damping_probe::lloyd_iteration_headroom` — 8→16 iter is +0.4-0.9%
- `gptq_damping_probe::huber_lloyd_headroom` — Huber strictly worse
- `gptq_damping_probe::gptq_on_correlated_pre_fwht` — GPTQ regression
  holds even on AR(1) correlated inputs
- `gptq_damping_probe::antirez_mq3_to_mq2_downgrade_cost` — MQ3→MQ2
  costs +226-265% MSE
- `gptq_damping_probe::weight_norm_proxy_imatrix_sweep` — calibration-
  free proxy gives ~0% benefit (FWHT washout)
- `gptq_damping_probe::fwht_value_audit` — FWHT is +35% on heavy-tailed,
  −245% on bimodal
- `hfq_block_diag::hfq_inventory` — antirezQ8 is misnamed
- `hfq_block_diag::hfq_dist_sample` — V4F dequant'd weights are
  near-Gaussian (post-CLT blurring; originals likely moderately heavy-
  tailed)

Run any of these:
```
cargo test --release -p hipfire-quantize <name> -- --nocapture
# Ignored tests (need a real HFQ path) — set HIPFIRE_QUANT_DIAG_PATH:
HIPFIRE_QUANT_DIAG_PATH=/data/hipfire-models/v4f.mq2lloyd-f16compress.hfq \
    cargo test --release -p hipfire-quantize hfq_inventory -- --ignored --nocapture
```

## Related memory entries (for future sessions)

- `project_v4f_mq2_gptq_lloyd_falsified.md`
- `project_gptq_lloyd_pretendgptq_finding.md`
- `project_v4f_quant_inventory_finding.md`
- `project_lloyd_imatrix_fwht_channel_mixing.md`
