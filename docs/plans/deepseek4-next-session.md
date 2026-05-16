# V4F bring-up — handoff for next session

State as of 2026-05-16 (updated late-session):

**Done:**
- Phase 0 (scaffold): arch crate, Config parser, schema-drift gate
- Phase 1 (FP8 ingest): byte-exact dequant verified, 40.2 GiB HFQ
  ships under `~/.hipfire/models/v4f.mq2lloyd-gptq-all`
- Phase 1.5 (load_weights walk): all 43 layers' expected tensors
  present in HFQ index, hash-routing on layers 0-2 correctly
  handled
- Phase 1.6 (GPU upload): `load_weights` now uploads every non-
  expert tensor — embeddings, norms, Q/O-LoRAs, KV joint, all 6
  HC tensors/layer, router gate.weight (and gate.bias for non-
  hash layers), shared expert × {w1,w2,w3}, compressor (when
  ratio > 0). ~5s wall on gfx1151. Routed experts gated behind
  `HIPFIRE_V4F_UPLOAD_EXPERTS=1` (~38 GB VRAM; defer until
  forward consumes them).
- Phase 1.7 (forward layout): `forward::decode_step` skeleton
  with the complete per-layer call graph documented. Body
  bodies are `unimplemented_step` (return Ok) until impls land.
- Phases 2-4 kernels (7 total) — all compile-clean on gfx1151,
  dispatch wrappers landed, GPU-validated against CPU references
  within fp16 tolerance.
- State: `DeepseekV4State` now carries `residual_streams` and
  `embed_scratch` GpuTensor slots (Option-wrapped for lazy alloc).

**Not done — the gap to "first token":**

The forward bodies are stubs. To get a token out of V4F, the next
session needs:

### 1. Forward bodies (estimated: 5-7 days)
Replace each `unimplemented_step(...)` in
`crates/hipfire-arch-deepseek4/src/forward.rs` with the real
kernel sequence. Order of difficulty (start with easiest):
- **Easiest**: embed → streams init (allocate 4 streams, copy
  embed row into stream 0 via `embedding_lookup_q8`, memset
  streams 1-3).
- **Medium**: RMSNorm + Q/O-LoRA / KV joint GEMVs — existing
  hipfire-runtime kernels.
- **Medium**: HC compute_control → sinkhorn → mix sequence — all
  three kernels are GPU-validated already; just chain the calls.
- **Medium-hard**: Tail-only RoPE — kernel works; need to ensure
  position counter passed correctly.
- **Hard**: Indexer score → top-k → gather + SWA cache update +
  main attention over union — the main attention kernel needs
  the wrap-aware `start_pos` parameter added (Phase 4 doc).
- **Hard**: MoE router (noaux_tc with sqrtsoftplus + topk=6 of
  256) and the hash-routing path for first 3 layers.
- **Hard**: Routed expert dispatch — V4F's per-expert tensor
  layout (gated behind `HIPFIRE_V4F_UPLOAD_EXPERTS=1` for VRAM
  reasons until forward can consume them).

### 2. Numerical correctness gate (estimated: 1-2 days)
Add `dev/bench/data/v4f_first_token_oracle.txt` — a prompt + the
exact first token V4F should emit at temp=0. Once forward is wired,
this is the single gate: do we produce the expected token? If yes,
V4F works.

### 3. Quality validation (estimated: 1 day)
Run `mq2lloyd_coherence_harness.py` against V4F at all-MQ2-GPTQ.
Expected: similar or better numbers than Qwen3.6-35B-A3B (9/10 ok,
0 attractors per the Lever 2 result). If significantly worse,
the recipe didn't generalise off Qwen3.6 → revisit damping sweep
on V4F-class precision before declaring success.

### 4. Phase 5 (MTP, optional, ~1 day)
Drop the `mtp.` prefix-skip in `hipfire-quantize`, treat MTP layer
as layer 43, wire as DFlash drafter.

## Total estimated time to first token: ~1.5 weeks focused

(Revised down from 2 weeks: load_weights upload now done; only
forward bodies + correctness gate remain.)

## Risks

- **Indexer top-k correctness at long context.** The stub O(N·K)
  top-k works up to N~16K. At 1M ctx with compress_ratio=4, N
  could exceed 250K positions — stub goes O(250K · 512) per head ×
  64 heads × 40 layers ≈ 3e12 ops per step. Replace stub with
  bitonic / radix select before benchmarking at long context.

- **Hyper-Connections decomposition.** The `[24]` control vector
  reshape into Sinkhorn input isn't certain — see open question
  in `docs/plans/deepseek4-phase3-hyper-connections.md`. Cross-
  check against the V4F paper before forward integration.

- **MoE expert dispatch.** V4F's per-expert separate-tensor layout
  means we have 256 expert handles per layer (instead of one
  stacked-3D). Our existing routed-MoE GEMV kernels expect the
  3D layout. Choice: (a) emit re-stacked tensors at quantize
  time, OR (b) extend kernels to take per-expert handles. (a) is
  cheaper.

## Bottom line

V4F at 40.2 GiB is ready to load. The 7 GPU-validated kernels are
the foundation. From here, forward integration is straight Rust
work — no new HIP kernels required (other than possibly an
optimised top-k for long context).
