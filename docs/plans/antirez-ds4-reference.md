# antirez/ds4 reference — quantization recipe, KV cache, and quality testing

**Source**: deep-dive reconnaissance of https://github.com/antirez/ds4 (2026-05-19).
Default branch `main`, ~10.7K stars, MIT license, single-file C/CUDA/Metal codebase.
There is no Python in the inference hot path — only thin Python utilities for the
offline pipeline.

**Key file paths within the repo** (all referenced below):
- `gguf-tools/deepseek4-quantize.c` (1888 lines) — HF safetensors → GGUF quantizer
- `gguf-tools/quants.c` / `quants.h` — the four output quant kernels (Q8_0, Q4_K,
  Q2_K, IQ2_XXS), derived from llama.cpp
- `gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py` — calibration corpus
  builder
- `gguf-tools/imatrix/dataset/manifest.json` — corpus stats
- `gguf-tools/quality-testing/{collect_official.py,score_official.c,compare_scores.py,prompts.jsonl}`
  — regression battery
- `tests/ds4_test.c` + `tests/test-vectors/official.vec` — C regression runner
- `MODEL_CARD.md` — DS4-Flash architecture synopsis (43 layers, indexer top-K)
- `gguf-tools/README.md`, `gguf-tools/imatrix/README.md` — recipe docs
- `ds4.c` (18260 lines) — runtime, including `weights_validate_layout` that
  hard-fails on tensor type mismatches

## 1. Quantization technique

### 1a. The template GGUF is the recipe — not a CLI flag

The defaults in `deepseek4-quantize.c:1731-1738` initialize every quant slot
to `DS4Q_TYPE_COUNT` ("use template type"). Per-tensor formats are dictated by
the template GGUF passed via `--template`, not by CLI flags. The template name
encodes the recipe. Two shipped templates:

- **Q2 recipe**: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`
- **Q4 recipe**: `DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf`

`ds4.c:2285-2353` (`weights_validate_layout`) hard-fails load if the GGUF
doesn't match. This is the byte-exact spec the runtime expects.

### 1b. Per-tensor format table

The format is **identical between Q2 and Q4 recipes except where flagged "DIFFER"**.

| Family | Tensor (GGUF name) | Q2 / Q4 |
|---|---|---|
| Token embedding | `token_embd.weight` | **F16** |
| Output head norm | `output_norm.weight` | F32 (1D) |
| Output projection | `output.weight` | **Q8_0** |
| HC output base/scale | `output_hc_base.weight`, `output_hc_scale.weight` | F32 |
| HC output fn | `output_hc_fn.weight` | **F16** |
| HC attn/ffn base/scale (per-layer) | `hc_attn_base`, `hc_attn_scale`, `hc_ffn_base`, `hc_ffn_scale` | F32 |
| HC attn/ffn fn (per-layer) | `hc_attn_fn`, `hc_ffn_fn` | **F16** |
| Attention norms | `attn_norm`, `attn_q_a_norm`, `attn_kv_a_norm` | F32 (1D) |
| **Q LoRA down** | `attn_q_a` | **Q8_0** |
| **Q LoRA up** | `attn_q_b` | **Q8_0** |
| **KV joint down** | `attn_kv` | **Q8_0** |
| Attn sinks | `attn_sinks` | F32 (1D, `n_head=64`) |
| **O LoRA down** | `attn_output_a` | **Q8_0** |
| **O LoRA up** | `attn_output_b` | **Q8_0** |
| Compressor APE | `attn_compressor_ape` | **F16** |
| Compressor WKV | `attn_compressor_kv` | **F16** |
| Compressor gate | `attn_compressor_gate` | **F16** |
| Compressor norm | `attn_compressor_norm` | F32 (1D) |
| Indexer Q-up | `indexer.attn_q_b` | **F16** |
| Indexer projection | `indexer.proj` | **F16** |
| Indexer compressor APE/KV/gate | `indexer_compressor_{ape,kv,gate}` | **F16** |
| Indexer compressor norm | `indexer_compressor_norm` | F32 (1D) |
| FFN gate (router) | `ffn_gate_inp` | **F16** |
| Router bias (optional) | `exp_probs_b` / `ffn_gate_bias` | F32 |
| Hashed-router lookup | `ffn_gate_tid2eid` (first 3 layers) | I32 |
| **Shared gate (w1)** | `ffn_gate_shexp` | **Q8_0** |
| **Shared up (w3)** | `ffn_up_shexp` | **Q8_0** |
| **Shared down (w2)** | `ffn_down_shexp` | **Q8_0** |
| **Routed gate (w1)** | `ffn_gate_exps` | **IQ2_XXS / Q4_K** ← DIFFER |
| **Routed up (w3)** | `ffn_up_exps` | **IQ2_XXS / Q4_K** ← DIFFER |
| **Routed down (w2)** | `ffn_down_exps` | **Q2_K / Q4_K** ← DIFFER |

Runtime enforces `gate.type == up.type` per layer (`ds4.c:2343-2346`); `down` is
independent. The asymmetric trick in Q2 is just **`down=Q2_K` while `gate/up=IQ2_XXS`**.

bpw: Q8_0 = 8.5; Q4_K = 4.5; Q2_K = 2.625; IQ2_XXS = 2.0625; F16 = 16.
Block sizes from `quants.c:39-74`:
- Q8_0: block 32, type_size 34 B
- Q4_K: block 256, type_size 144 B
- Q2_K: block 256, type_size 84 B
- IQ2_XXS: block 256, type_size 66 B

### 1c. Algorithm — NOT GPTQ / Lloyd / OBS / Hadamard

Explicit grep result from the agent: **no Hadamard, no FWHT, no GPTQ, no
QuaRot/SpinQuant/SmoothQuant, no Lloyd k-means, no outlier mining**. The
asymmetric trick is purely a template-level "use Q2_K instead of IQ2_XXS for
down."

The actual quantizers (`quants.c`):

- **`make_qkx2_quants`** — Q4_K and Q2_K reference paths (no imatrix).
  Hyperparams:
  - Q4_K: `nmax=15, rmin=-1.0, rdelta=0.1, nstep=20, use_mad=false`
  - Q2_K: `nmax=3,  rmin=-0.5, rdelta=0.1, nstep=15, use_mad=true`
- **`make_qkx3_quants`** — imatrix-weighted Q4_K / Q2_K.
  - Q4_K weighted: `nmax=15, rmin=-0.9, rdelta=0.05, nstep=36`
  - Q2_K weighted: `nmax=3,  rmin=-0.9, rdelta=0.05, nstep=36`
  - Per-block weighting: `w[l] = qw[l] * sqrt(sigma2 + x[l]^2)`,
    `sigma2 = 2*Σx²/QK_K` for Q4_K, `Σx²/QK_K` for Q2_K.
- **`make_qp_quants`** — non-negative-only quant for sub-scales + IQ2_XXS group
  scales; ±0.4-range scale sweep then 5 coordinate-descent passes.
- **IQ2_XXS** (`quants.c:822-995`) — 256-element block → 8 groups of 32 →
  4 octets of 8. Each octet quantized to one of 256 fixed published grid indices
  (`kgrid` table verbatim at `quants.c:698-715`), 7-bit sign mask per octet
  (last sign deduced from parity). 4-bit per-group scale nibble + one f16 block
  scale.

`quants.c:10-12` comment: "byte layout compatibility is more important here
than generality" — output is byte-identical to llama.cpp's IQ2_XXS at fixed
imatrix and same compiler flags (assumption, not invariant — `make_qp_quants`
coordinate descent has data-dependent termination).

**Synthetic-fallback imatrix** when none supplied for a quant type that
requires one (`deepseek4-quantize.c:1115-1124`):
```c
for (int64_t r = 0; r < nrows; r++) {
    for (int64_t c = 0; c < ncols; c++)
        synthetic[c] += row[c] * row[c];
}
```
Documented as "weight-energy heuristic, not as good as measuring real
activations."

### 1d. Imatrix calibration

- **Required for IQ2_XXS** (`quants.c:54` `requires_imatrix = true`).
- Optional for Q4_K / Q2_K (used as weighted-MSE hint).
- Q8_0 ignores imatrix (`quants.c:1053-1055`).
- **Format**: legacy llama.cpp `.dat`. Loader at `deepseek4-quantize.c:759-811`.
  Values divided by `ncall` on load.
- **Naming convention** per-expert: `{tensor}.expert.{xid}` or `{tensor}.expert_{xid}`
  or a packed vector of length `n_experts × n_columns`.
- **What's recorded** (`imatrix/README.md:12-17`):
  - Gate/up: squared FFN-normalized input activation per channel
  - Down: squared routed-weighted SwiGLU row
  - **Metal-only collector, hooks layer-major prefill graph, only collects for
    routed MoE path.** Other tensors get the synthetic weight-energy fallback.

### 1e. Calibration corpus

`gguf-tools/imatrix/dataset/manifest.json`:
- 4690 prompts, ~2.92M tokens (bytes/4 estimate)
- 50/50 think/nothink split
- Categories: agent (1106), language (1024), source (2074 — DS4's own
  C/Metal code), eval_reasoning (150), programming (48), long_context (36),
  translation (180), general (40), algorithms (32)
- Built via `build_ds4_imatrix_dataset.py`

Collection command (`imatrix/README.md:52-58`):
```sh
./ds4 -m DeepSeek-V4-Flash-Q4KExperts-...-chat-v2.gguf \
      --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt \
      --imatrix-out routed-moe.dat \
      --ctx 32768
```

Imatrix is collected from the **Q4 model** then applied during requantization
of both Q4 and Q2 outputs.

### 1f. Source preprocessing (HF → F32)

- Routed experts in HF safetensors: **FP4 + E8M0 packed**
  (`deepseek4-quantize.c:709-737`). Custom asymmetric FP4 table:
  ```c
  {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
   0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0}
  ```
  This is NOT OCP MX FP4 (which uses E2M1). Block size 32, one I8 byte = two FP4.
- Non-expert tensors: FP8 E4M3 + E8M0.
- HF shapes are TRANSPOSED vs GGUF; `check_reversed_shape` at `:1151-1163`.

### 1g. Recommended CLI invocation

From `gguf-tools/README.md:71-88`:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --out gguf/DeepSeek-V4-Flash-IQ2XXS-...-imatrix.gguf \
  --imatrix gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat
```

Per-tensor overrides available via `--experts/--routed-w{1,2,3}/--attention-proj/
--attention/--shared/--embedding/--output/--dense/--tensor-type`. Threading via
`--threads N` (default 8); 256 routed experts per layer split across workers.

## 2. KV cache compression

### 2a. Layer pattern (hardcoded)

`ds4.c:411-415`:
```c
static uint32_t ds4_layer_compress_ratio(uint32_t il) {
    if (il < 2) return 0;
    return (il & 1u) == 0 ? 4u : 128u;
}
```

| Layer | Ratio | Indexer? |
|---|---:|---|
| 0, 1 | none | no (raw SWA only) |
| even ≥2 | 4 | yes |
| odd ≥3 | 128 | no |

### 2b. Architectural constants (`ds4.c:86-109`)

- `DS4_N_LAYER` = 43
- `DS4_N_SWA` = 128 (sliding-window size)
- `DS4_N_INDEXER_HEAD` = 64
- `DS4_N_INDEXER_HEAD_DIM` = 128
- `DS4_N_INDEXER_TOP_K` = 512
- `DS4_N_EXPERT` = 256 (routed)
- `DS4_N_FF_EXP` = 2048 (per-expert FFN dim)
- Active experts per token = 6, plus 1 shared
- Attention heads = 64; head_dim = 512; value_dim = 512; RoPE dim = 64
- LoRA: `n_lora_q` = 1024; `n_lora_o` = 1024
- First 3 layers use hashed-router `tid2eid` lookup (`DS4_N_HASH_LAYER` = 3)

### 2c. Sub-component precision at inference

All compressor and indexer weights are **F16** in both Q2 and Q4 recipes
(`ds4.c:2318-2331`):

| Sub-component | Precision |
|---|---|
| Compressor APE (positional) | F16 |
| Compressor WKV | F16 |
| Compressor gate | F16 |
| Compressor norm | F32 (1D) |
| Indexer Q-up | F16 |
| Indexer projection | F16 |
| Indexer compressor APE/WKV/gate | F16 |
| Indexer compressor norm | F32 (1D) |

### 2d. Runtime KV storage uses E4M3

`ds4.c:1632-1634`: the live KV cache for compressed branches stores the
non-RoPE part in **E4M3-style** packed format. This is independent of the
weight quantization — applies only to the activation cache accumulated
during decode.

### 2e. Lookup policy

- Below 512 compressed rows: indexer is a no-op (all visible).
- At ≥512 compressed rows: indexer scores all rows, picks top-512.
- Ratio-128 layers skip the indexer entirely.

### 2f. Persisted KV cache format

48-byte `KVC` header → variable `DSV4` payload with version, prefill-chunk-size,
capacities, per-layer compressed row counts, then payloads in **logical**
position order (not physical ring order). Shippable across restarts.

## 3. Regression testing — what they actually run

### 3a. C regression runner (`ds4_test`)

`make test` runs `--all`. Entries (`tests/ds4_test.c:648-655`):

- **`--logprob-vectors`**: compares local greedy token bytes + top-K logprobs
  against captured DeepSeek API continuations from `tests/test-vectors/official.vec`.
  Captures from `deepseek-v4-flash` with greedy decoding, `thinking=disabled`,
  `top_logprobs=20`, `max_tokens=4`.
- **`--long-context`**: from `tests/long_context_story_prompt.txt`. The model
  must extract 30 hardcoded `(name, number)` facts from prose >30K tokens.
  Hardcoded fact table at `ds4_test.c:186-201` (Bob=34, Alice=52, Clara=71,
  Diego=93, Elena=16, Felix=88, Greta=47, Hugo=29, Iris=64, Jonas=12,
  Kira=81, Leo=39, Marta=76, Nadia=23, ...).
- **`--tool-call-quality`**: DSML tool-call emission in fast + exact paths.
- **`--metal-kernels`**: isolated Metal kernel numeric checks.
- **`--server`**: HTTP server request handling.

CUDA-only smoke test: `tests/cuda_long_context_smoke.c` via `make cuda-regression`.

### 3b. Quantization quality battery (`gguf-tools/quality-testing/`)

Metric: target-token NLL on official continuations, plus first-token-match
and greedy-LCP.

**Pipeline**:
1. `collect_official.py` — 100 prompts × up to 24 target tokens captured
   against `https://api.deepseek.com/chat/completions`:
   - model `deepseek-v4-flash`
   - `temperature=0`, `max_tokens=24`, `thinking={"type":"disabled"}`
   - `logprobs=True`, `top_logprobs=5`
   - Prompts from `prompts.jsonl` (100 prompts; English + Italian; algorithms /
     completion / translation / short-answer)
   - **Outputs NOT checked into repo** (regenerated per quant run)
2. `score_official MODEL.gguf manifest.tsv out.tsv [ctx=4096]` — per case,
   teacher-force the prompt, then for each official continuation token capture
   `-logprob(target)` + greedy argmax.
3. `compare_scores.py` outputs: `avg_nll`, `delta_new_minus_old`,
   `relative_nll_change`, `case_wins_new_old_ties`, `first_token_matches_old_new`,
   `avg_greedy_lcp_old_new`.

**Published reference** (`imatrix/README.md:131-142`):
```
old Q4 avg NLL:         0.177358
Q4 imatrix avg NLL:     0.173895
relative NLL change:   -1.95%
case wins:              54 imatrix / 46 old
first-token matches:    83 imatrix / 81 old
avg greedy LCP:         12.21 imatrix / 11.94 old
```

**This is the only PPL-equivalent number antirez publishes.**
**There is NO published Q2 NLL number** — only the Q4-imatrix improvement.

### 3c. Capability eval (`ds4-eval`)

92 embedded prompts:
- 25 GPQA Diamond
- 25 audited SuperGPQA
- 25 AIME 2025
- 17 audited COMPSEC (single-function C/C++ vulnerability localization)

Defaults: `--tokens 16000`, thinking enabled, soft/hard `</think>` budget cutoff.
Explicit anti-leaderboard intent — "chosen to make local regression testing
useful and visually inspectable."

### 3d. Speed regression (`ds4-bench`)

`speed-bench/` CSVs for M2 Ultra, M3 Max, M4 Max. Methodology in
`CONTRIBUTING.md:107-130`: linear or exponential context sweep, incremental
prefill, greedy non-EOS generation at each frontier, fixed `--gen-tokens 128`.
Driven by `speed-bench/promessi_sposi.txt` (Manzoni's *I Promessi Sposi*
Project Gutenberg #45334, header/footer stripped).

### 3e. What's NOT in their test suite

- No attractor detection (no unique-token-ratio, max-frequency, n-gram density)
- No standard-corpus PPL (Wikitext / C4 / PG19)
- No KLD vs the upstream FP4+FP8 model
- No spec-decode τ gate
- No prompt-md5 logging convention (prompts ARE committed verbatim → byte-stable)

## 4. Implications for hipfire V4F robustness

### 4a. Divergences from antirez, ranked by impact

1. **Our `--format mq4-mqlloyd-antirez` is misnamed.** It implements only the
   routed-expert asymmetric split. The real antirez recipe is the entire
   template: Q8_0 attention, Q8_0 shared, F16 compressor, F16 indexer, Q8_0
   output, IQ2_XXS gate/up, Q2_K down. Our existing `v4f.antirezQ8.hfq`
   covers attention/shared/compressor but uses MQ2-Lloyd for ALL routed
   (no IQ2_XXS, no Q2_K asymmetry).
2. **We FWHT-rotate where they don't.** Our MQ-family is FWHT-rotated; antirez
   ships GGML weighted-MSE block search. This is why our imatrix support is
   structurally broken (per the `prefwht_imatrix_lloyd_value` and
   `weight_norm_proxy_imatrix_sweep` probes — saved as
   `project_lloyd_imatrix_fwht_channel_mixing` memory) while theirs works.
3. **No published reference NLL on hipfire side.** Antirez's 0.173895 Q4-imatrix
   NLL on 100 prompts is the upstream-blessed quality number.
4. **Our regression suite is complementary, not aligned.** We have attractor
   detection (they don't); they have long-context fact-recall, tool-call
   quality, and official logprob fixtures (we don't).
5. **Imatrix collection prerequisite for any IQ2_XXS port.** If we eventually
   port the GGML quantizers, imatrix becomes mandatory for IQ2_XXS.

### 4b. Validated by upstream

- **Compressor F16 must stay F16** (`project_v4f_compressor_must_stay_f16`
  memory): antirez confirms — F16 in BOTH Q2 and Q4 recipes; never quantized
  in any shipped template.
- **Lloyd-imatrix integration broken by FWHT-channel-mixing**
  (`project_lloyd_imatrix_fwht_channel_mixing` memory): antirez bypasses this
  entirely by not rotating. Their imatrix-weighted Lloyd-equivalent
  (`make_qkx3_quants`) works on natural-basis weights.

### 4c. Quick reference for a byte-faithful Rust port

If the eventual goal is "match upstream Q2 imatrix GGUF byte-for-byte":

1. Implement F32 inflation from FP4+E8M0 + FP8 E4M3+E8M0 exactly per
   `deepseek4-quantize.c:670-737`. The asymmetric FP4 table
   `{0,0.5,1,1.5,2,3,4,6}` is load-bearing.
2. Implement the imatrix `.dat` loader including the `values /= ncall`
   step (`:781-783`).
3. For each tensor name, look up the type via `ds4.c:2285-2353`. There's no
   policy logic — the template is ground truth.
4. Port `make_qkx2_quants`, `make_qkx3_quants`, `make_qp_quants`, and the
   IQ2_XXS block packer with the exact hyperparameters above.
5. Embed the 256-entry `kgrid` table from `quants.c:698-715` verbatim.
6. **No rotation step** — antirez does not FWHT.

## 5. Concrete next-step plan for hipfire

In priority order:

- **B**: pull antirez's `prompts.jsonl` and adapt `score_official.c` into a
  hipfire-side `quality_score` binary. Run against our existing best quant
  (`v4f.mq2lloyd-f16compress.hfq`) to produce comparable NLL numbers.
- **C**: adopt their `--long-context` 30-fact-recall test pattern. Adapt the
  test prompt + fact table to run against our daemon.
- **D**: capture our own "official" continuations either via the DeepSeek API
  (if access available) or by running the BF16+FP8 source model locally (the
  cached safetensors in HF cache).
- **A**: port the GGML IQ2_XXS / Q2_K / Q4_K kernels into hipfire-quantize
  as `--format ds4-q2` / `--format ds4-q4`. ~1-2 weeks of work. Unlocks
  byte-for-byte parity with antirez.
- **E**: collect our own imatrix using `gguf-tools/imatrix/dataset/`. Required
  prerequisite for A's IQ2_XXS path; ~hours of compute against the BF16 source.

## Sources

All paths within the antirez/ds4 repo:
- README.md, MODEL_CARD.md, CONTRIBUTING.md (top-level)
- gguf-tools/README.md, gguf-tools/deepseek4-quantize.c, gguf-tools/quants.c
- gguf-tools/imatrix/README.md, gguf-tools/imatrix/dataset/manifest.json
- gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py
- gguf-tools/quality-testing/{README.md,collect_official.py,score_official.c,compare_scores.py,prompts.jsonl}
- tests/{ds4_test.c,test-vectors/{official.vec,README.md,manifest.json},long_context_story_prompt.txt}
- ds4.c (runtime contract at :2285-2353; constants at :86-109)
