# V4F Phase 5 — MTP head (Multi-Token-Prediction)

**Status:** design. Implementation is the last step; defer until
Phases 2-4 produce correct base forward.

## What MTP is

V4F ships with `num_nextn_predict_layers = 1` — one extra transformer
layer trained to predict the *next-next* token (`y_{t+2}`) given the
hidden state for the current token (`y_{t+1}`). The base model
predicts `y_{t+1}` from `y_t`; MTP predicts `y_{t+2}` from `h_t`.

At inference, the MTP layer is a built-in speculative drafter: it
proposes a guess for `y_{t+2}` while the base model is still committing
`y_{t+1}`. If the base model's `y_{t+2}` step agrees with the MTP
proposal, we accept both tokens in one step. Acceptance rate determines
speedup.

## V4F tensor layout (from index)

Walking the V4F safetensors index, MTP tensors live under `mtp.*`:
```
mtp.embed_norm.weight
mtp.input_proj.weight
mtp.layer.attn.{wq_a, wq_b, wo_a, wo_b, wkv}.weight    + .scale
mtp.layer.attn.{q_norm, kv_norm}.weight
mtp.layer.attn.attn_sink
mtp.layer.attn.compressor.{wkv, wgate, norm}.weight
mtp.layer.{attn_norm, ffn_norm}.weight
mtp.layer.ffn.{experts, shared_experts, gate}.*        + .scale
mtp.layer.hc_attn_{base, fn, scale}
mtp.layer.hc_ffn_{base, fn, scale}
```

Same shape as a main layer — MTP is structurally identical to a base
layer plus an `input_proj` that conditions on the base model's
hidden state at the matching position.

The current `hipfire-quantize` main loop SKIPS tensors whose name
starts with `mtp.` (line 3911 in main.rs):

```rust
if name.starts_with("mtp.") { skipped_params += n; continue; }
```

This is the same pattern Qwen3.5/Llama use for skipping MTP/vision
heads from the main quant pipeline. For V4F we'll need to either:
- Remove the skip and quantize the MTP layer alongside main layers
  (one extra layer, treated identically), OR
- Keep the skip and produce a side `model.mtp` HFQ that the runtime
  loads conditionally.

Probably the former — V4F's MTP is small enough (one layer of the
existing layer shape) that conditional loading isn't worth the
complexity.

## DFlash integration

hipfire already has DFlash spec-decode infrastructure: a small drafter
model proposes N tokens, the main model verifies in parallel. The MTP
head fits this slot natively:
- The drafter is `mtp.layer` (one shot, predicts the next token from
  the base model's hidden state).
- The "verify" step is the base model's normal forward.
- No separate drafter weights file needed.

Existing DFlash code paths in `crates/hipfire-runtime/src/dflash.rs`:
- `DFlashState` holds drafter + main state
- `dflash_step` runs draft + verify, returns accepted tokens
- Speculative tree explored via the existing CASK m-fold framework

The V4F integration: implement `mtp_step(h_t)` that returns a
candidate `y_{t+2}`. Hook into DFlash as a one-step drafter.

## Kernel inventory

None new — the MTP layer is structurally identical to a main layer,
so it reuses everything from Phases 2-4 (indexer, HC, SWA, RoPE).
The only V4F-specific piece is `mtp.input_proj` which is a single
GEMV from base hidden → MTP hidden (likely same dim).

## Implementation order

1. **A.** Remove or refine the `mtp.` skip in `hipfire-quantize` —
   include MTP tensors in the output. Treat as layer 43 (the last
   in `compress_ratios`).
2. **B.** Wire the MTP layer into `DeepseekV4Weights` as
   `weights.mtp_layer` (Option<LayerWeights>).
3. **C.** Implement `mtp_step` using the same forward kernels as
   main layers, conditioned on `mtp.input_proj`.
4. **D.** Plug into DFlash drafter slot. Acceptance threshold tuning.

## Pre-condition for skip

If base forward (Phases 2-4) gives correct token IDs end-to-end and
DFlash spec-decode works on Qwen3.5 still, then MTP is a +1 speedup
to layer on top — not on the critical path for "V4F generates correct
text." Defer until base correctness is locked.
