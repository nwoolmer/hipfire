# V4F + MTP requant plan (2026-05-20)

## Spec (per user instruction)

Output model: `v4f.mq2lloyd-mtp.hfq` (working name).

Per-tensor target dtype, by source dtype in the safetensors:

| source dtype in safetensors | device dtype in HFQ | applies to |
|---|---|---|
| `I8` + `F8_E8M0` scale (= FP4-packed) | **MQ2-Lloyd** | routed experts only (`*.ffn.experts.*`) |
| `F8_E4M3` + `F8_E8M0` scale | **Q8F16** | attention projections, e_proj, h_proj, compressor wkv/wgate, indexer projections, shared-FFN w1/w2/w3, FFN gate |
| `BF16` | **F16** | all norms, kv/q norms, attn/ffn norms, enorm, hnorm, HC matrices, biases |
| `F32` | **F32** | tiny tensors like `attn_sink` (already 1D F32 on disk) |

No K-map, no imatrix, no kmap promotions — the user's instruction was
"full precision i.e. F16 or Q8, depending on what the original source has",
so we honor that literally.

Includes the MTP layer (`mtp.0.*`) processed with the same rules. The
MTP layer in V4F is a single transformer-style block (no compressor,
no indexer) plus the two MTP-specific input projections `e_proj` and
`h_proj` and per-input norms `enorm` and `hnorm`. It has its own full
set of 256 routed experts + shared expert + router gate.

## Source dtype inventory (verified from safetensors)

Confirmed by reading `model-00046-of-00046.safetensors` header:

- Main layers (0..42) and MTP (`mtp.0.*`):
  - Routed experts (`layers.{L,mtp.0}.ffn.experts.{0..255}.{w1,w2,w3}.weight`) — `I8` + `F8_E8M0`
  - Attention `{wq_a, wq_b, wkv, wo_a, wo_b}` — `F8_E4M3` + `F8_E8M0`
  - `e_proj`, `h_proj` (MTP only) — `F8_E4M3` + `F8_E8M0`
  - Shared expert `w1, w2, w3` — `F8_E4M3` + `F8_E8M0`
  - Compressor `wkv, wgate` (main only) — `F8_E4M3` + `F8_E8M0`
  - Indexer `wq_b, weights_proj, compressor.wkv, compressor.wgate` — `F8_E4M3` + `F8_E8M0`
  - FFN router `gate.weight` — `F8_E4M3` + `F8_E8M0`; `gate.bias` — `F32`
  - Norms (`*norm.weight`, `enorm.weight`, `hnorm.weight`) — `BF16`
  - HC matrices (`hc_attn_*`, `hc_ffn_*`, `hc_head_*`) — `BF16`
  - `attn_sink` — `F32`
- Globals: `embed.weight` — `F8_E4M3` + scale; `head.weight` — `F8_E4M3` + scale; `norm.weight` — `BF16`.

## Implementation steps (quant side)

### Step Q1 — Make the quant tool see MTP

The current `crates/hipfire-quantize/src/main.rs` walks safetensors
tensors and applies routing on `is_v4f && name.contains(".ffn.experts.")`.
That filter ALREADY accepts `mtp.0.ffn.experts.*` (substring match).
Verify by adding a print + dry-run on the safetensors index.

The main concern: tensors that the arch crate's `load_weights` looks
up by name (e.g. `layers.{L}.attn.wq_a.weight`) — those are namespaced
by `layers.L` while the MTP head's are namespaced by `mtp.0`. The
quant tool itself doesn't filter by `layers.` prefix in the V4F path
(verified by reading lines 4574-4645), so MTP tensors go through the
standard tensor pipeline unmodified.

### Step Q2 — Add a new `--format` flag `v4f-source-precision`

New flag that dispatches per tensor:

```rust
// Pseudocode for the per-tensor target dtype.
fn v4f_source_precision_target(name: &str, src_dtype: &str) -> Target {
    if name.contains(".ffn.experts.") && !name.contains("shared_experts") {
        // Routed expert FFN weight (must be 2D for kernel layout).
        return Target::MQ2Lloyd;
    }
    match src_dtype {
        "F8_E4M3" => Target::Q8F16,    // dequant FP8 → F32 → Q8F16
        "BF16"    => Target::F16,      // store as F16 directly
        "F32"     => Target::F32,      // preserve (attn_sink-like)
        "I8"      => Target::Q8F16,    // shouldn't happen except for experts (handled above)
        other     => panic!("v4f-source-precision: unhandled source dtype {other}"),
    }
}
```

The implementation slots into the per-tensor loop after the
`is_v4f && .ffn.experts.` branch (line ~4580); for everything else we
short-circuit to the dtype-specific path before the existing kmap/F16
default fall-through.

### Step Q3 — Build & run

```
cd ~/.hipfire/src
cargo build --release -p hipfire-quantize
./target/release/hipfire-quantize \
    --input /home/nick/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136 \
    --output /data/hipfire-models/v4f.mq2lloyd-mtp.hfq \
    --format v4f-source-precision \
    --no-kmap
```

Background job. Logs go to `/tmp/v4f-requant.log`.

## Implementation steps (arch + spec-decode side, run in parallel)

### Step M1 — Extend `DeepseekV4Weights` for the MTP layer

Add fields to `crates/hipfire-arch-deepseek4/src/deepseek4.rs`:

```rust
pub struct DeepseekV4MtpWeights {
    pub enorm: Option<GpuTensor>,     // [hidden]                BF16→F16
    pub hnorm: Option<GpuTensor>,     // [hidden]                BF16→F16
    pub e_proj: Option<GpuTensor>,    // [hidden, hidden]        FP8 → Q8F16
    pub h_proj: Option<GpuTensor>,    // [hidden, hidden]        FP8 → Q8F16
    pub attn_norm: Option<GpuTensor>,
    pub ffn_norm: Option<GpuTensor>,
    pub norm: Option<GpuTensor>,
    pub wq_a, wq_b, wkv, wo_a, wo_b: Option<GpuTensor>,   // FP8 → Q8F16
    pub q_norm, kv_norm, attn_sink: ...,
    pub gate_weight, gate_bias: ...,
    pub gate_bias_host: Vec<f32>,
    pub shared_w1, shared_w2, shared_w3: Option<GpuTensor>,
    pub expert_w1_blob/_ptrs/_stride etc.: ...,  // mirror main layer
    pub expert_gate_up_blob/_ptrs/_stride: ...,  // mirror main layer
    pub hc_attn_base/_fn/_scale, hc_ffn_*, hc_head_*: ...,
}

// In DeepseekV4Weights, add:
pub mtp: Option<DeepseekV4MtpWeights>,
```

### Step M2 — Ingest in `load_weights`

Mirror the per-layer loader against the `mtp.0.*` prefix. Reuse
`upload_quant_or_f16` etc. — those already handle Q8 (`quant_type=3`)
and F16 (`quant_type=1`) source bytes.

### Step M3 — `mtp_forward` function

In `crates/hipfire-arch-deepseek4/src/forward.rs`:

```rust
pub fn mtp_forward(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    /// Previous-position hidden state (output of main forward at pos N).
    h_n: &GpuTensor,
    /// Token id at position N+1 (the candidate token whose +1 successor we predict).
    next_token: u32,
) -> Result<Vec<f32>, String>  // logits at position N+2
```

Pipeline (per DeepSeek V3 paper §4 + matching the MTP weight names):

```
embed_n1     = embed_lookup(next_token)
h_n_norm     = rmsnorm(h_n,     hnorm)
e_n1_norm    = rmsnorm(embed_n1, enorm)
proj_h       = h_proj  @ h_n_norm
proj_e       = e_proj  @ e_n1_norm
x            = proj_h + proj_e             // (V3-style fusion; verify)
# Standard transformer block:
x            = attention(attn_norm(x))  + x
x            = ffn(ffn_norm(x))         + x
h_n1         = norm(x)
logits_n2    = shared_head(h_n1)
```

The exact composition of `proj_h + proj_e` is what we VERIFY against
upstream Python at validation time (could be concat-then-project or
sum; we test both and pick the one whose logits match upstream).

Reuses existing kernels: `rmsnorm_f32`, `gemv_auto`,
`v4f_attn_swa_topk` (no compressor needed — MTP has no compressor),
the existing MoE pipeline, `final_norm_and_head`.

### Step M4 — Spec-decode wrapper

`speculative_decode_step` in a new file
`crates/hipfire-arch-deepseek4/src/spec_decode.rs`:

1. Given current context, generate K candidates via repeated `mtp_forward`:
   - call 1 produces token N+1 (we already have, so start from existing decoded token)
   - call 2 uses h_N+1 + candidate_N+1 to produce token N+2
   - … iterate K times to produce candidates `[t1, t2, …, tK]`
2. Run main V4F as `forward_prefill_batch_chunk` at B=K on `[committed_token, t1, …, tK-1]`
3. Compare main's top-1 at each position against the next candidate; accept the longest matching prefix
4. Return accepted tokens + the divergence-correcting target token at the boundary

### Step M5 — End-to-end test

Bench `bench_decode_vs_ctx` (or a new `bench_decode_spec`) at ctx=512,
target prompt, measure:
- baseline decode tok/s (existing sequential)
- spec-decode tok/s with K=4, K=8

Validate quality via the existing `coherence_probe`.

## Validation gates

After the requant:
- `inspect_hfq /data/hipfire-models/v4f.mq2lloyd-mtp.hfq` should show
  the MTP layer's classes (expect ~1500 new tensors).
- File size: existing 77 GiB + ~+2 GiB for MTP routed experts + ~+1 GiB
  for MTP dense weights ≈ 80 GiB.

After M3 (MTP forward standalone):
- Compare MTP's predicted logits against a reference Python pass on a
  known token sequence — top-1 must match the upstream MTP output.
- If we don't have a reference Python rig, sanity-check that
  `mtp_forward(h_n, next_token=correct_n+1)` produces a top-1 of
  `correct_n+2` on a known prompt.

After M4 (spec decode):
- PPL parity with sequential decode (spec decode is lossless when the
  acceptance criterion is top-1 match).
- tok/s improvement on the bench.

## Out of scope for this work

- K_TOP reduction (separate orthogonal optimization).
- Layer pruning.
- Lower-bit routed-expert quant.

These remain available as future levers but are not part of this plan.

## Ordering

The user said: run requant in background, do M1-M4 in parallel.

Concretely:
1. (now) Wire Q2 + Q3 in the quant tool. Kick off the requant.
2. (now, in parallel) Land M1, M2, M3 (still tested only by load + a
   small standalone driver, not yet by spec-decode).
3. When the new HFQ file is ready: load it, run M3's standalone check
   (predict the next-next token on a known prompt), then wire M4.
4. Bench + validate.
