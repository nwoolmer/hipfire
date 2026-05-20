# V4F prefill perf — next-phase implementation plan

**Starting state**: 47 tok/s batched prefill on Radeon 8060S (B=64,
prompt=256), validated by PPL (mean |Δ| 0.5 % vs F32 baseline).

**Diagnostic root cause** (rocprofv3, 2026-05-20):
- GPU kernel-active only 39 % of single-chunk wallclock
- 61 % is launch-gap idle, avg 137 μs/launch
- 10,426 d2d copies + ~3,000 small per-position kernels per chunk
- Source: compressor commit/compress pipeline runs **per-(b, layer)**;
  GEMV phase is batched but ring writes + softmax_pool + RoPE are not.

**Target**: 70–90 tok/s after phase A+B, 80–100 tok/s after phase A–F.

---

## Phase A — Batch compressor commit + compress pipeline (biggest lever)

**Goal**: collapse the ~10k per-(b, layer) launches in
`compressor_forward_impl` into a small set of batched kernels.

### Design

For a chunk of B positions at start_pos, in each layer with
compress_ratio = R (4 or 128 for V4F):

- The wkv/wgate GEMVs are ALREADY batched (committed earlier this
  branch). They produce `kv_batch[B, proj_dim]` and
  `score_batch[B, proj_dim]`.
- The current per-position loop does:
  - `kv_state[slot] ← kv_batch[b]` (d2d memcpy)
  - `score_state[slot] ← score_batch[b]`
  - if `(pos+1) % R == 0`: overlap_concat → softmax_pool → rmsnorm
    → tail-rope → write to `kv_cache[pos / R]`

For chunks where `B ≥ R` and aligned at ring boundaries:
- The compress events fire at positions `b ∈ {R-1, 2R-1, ...}` —
  every R-th position within the chunk.
- The compress input for event k is `kv_batch[(k-1)*R+1 .. k*R+1]`
  (plus the overlap from the previous ring state for ratio=4).
- **We can skip the ring writes entirely** for those events: the
  compress operates directly on the kv_batch slice.
- For the tail (positions that don't complete a compress event), we
  still need to write them to the ring for the next chunk's compress.

### Steps

1. **Write new batched kernels**:
   - `compressor_compress_batched_f32`: input `[N_events, 2R,
     head_dim]` kv concat and `[N_events, 2R, head_dim]` score concat,
     output `[N_events, head_dim]` kv_cache rows. Handles overlap=true
     (ratio=4) and overlap=false (ratio=128) via a flag.
   - `compressor_overlap_concat_batched_f32`: applies the overlap
     concatenation step across N_events.
   - Keep `compressor_softmax_pool_f32` and `compressor_overlap_concat_f32`
     as the per-event scalar versions for the tail path.
2. **Compute compress-event boundaries** in `attention_block_batched_mixed`:
   - `n_aligned_events = floor((B + start_pos % R) / R)` for current
     chunk
   - `tail_start = aligned_events * R - (start_pos % R)`
   - Tail positions [tail_start..B) get written to ring as before.
3. **Refactor `compressor_forward_impl`**:
   - When called for an *aligned* event (compress fires this step),
     accept a `kv_concat_view` and `score_concat_view` directly
     instead of building from ring.
   - Skip the d2d copy to ring_state for aligned positions.
4. **Wire into `attention_block_batched_mixed`** (and the indexer
   variant):
   - After batched wkv/wgate GEMVs: run batched overlap_concat across
     all aligned events.
   - Then batched softmax_pool → batched rmsnorm → batched RoPE.
   - Then if tail exists: per-tail-position ring writes (small loop).
5. **Same for indexer compressor** (ratio=4 only).
6. **Validation gates**:
   - Bisect at B=1: max_abs must stay ≤ 1e-3 (we currently see ~1e-4).
   - Bisect at B=R (smallest case with one aligned event): top-1 match.
   - Bisect at B=64: top-1 match preserved.
7. **Profile**: rocprof kernel-trace; verify `__amd_rocclr_copyBuffer`
   count drops from ~10k to under ~500 (tail only).
8. **Bench**: target ≥ 65 tok/s at B=64 prompt=256.

### Risk register

- Numerical: F32 accumulation in compress_softmax_pool — keep batched
  kernel using F32 internals.
- Tail handling at start_pos > 0 (multi-chunk runs): the ring state
  carries over from previous chunks. Need careful boundary math.
- ratio=128 layers: B=64 < R, so NO aligned events. Whole chunk goes
  through the tail path — no win for these layers individually, but
  also no regression.

**Expected gain**: −1.0 to −1.5 s/chunk wallclock → **65–90 tok/s**.

---

## Phase B — hipGraph the chunk forward

**Goal**: capture the chunk's kernel sequence once, replay per chunk
with only the per-chunk inputs updated.

### Prerequisites
- Phase A landed (reduces graph size from 20k → ~4k kernels).
- HIP bridge `hipGraph` API exposure check.

### Steps

1. **Audit hip-bridge crate** for hipGraph wrappers:
   - `hipGraphCreate`, `hipStreamBeginCapture`, `hipStreamEndCapture`
   - `hipGraphInstantiate`, `hipGraphLaunch`
   - `hipGraphExecKernelNodeSetParams` for updating per-launch params
2. **If missing, add to hip-bridge**:
   - Match the existing FFI style (raw bindings + safe Rust wrapper).
3. **Identify per-chunk inputs that vary**:
   - `pbs.tokens`, `pbs.positions` (uploaded each chunk)
   - `start_pos` (used for ring slot math — but this is built into
     compress event positions, not a direct kernel param)
4. **Wrap `forward_prefill_batch_chunk`**:
   - First call with new (B, layer_count): record into graph.
   - Subsequent calls: update only the varying input nodes, then
     `hipGraphLaunch`.
   - Cache the graph keyed by `(batch_size, n_layers, dtype)`.
5. **Handle multi-chunk runs**:
   - Each chunk = one graph replay.
   - SWA ring / KV cache state must persist across chunks (already
     does, but verify graph capture doesn't break it).
6. **Validation**:
   - Bisect: byte-eq.
   - Bench at B=64 prompt=256: target ≥ 70 tok/s on top of Phase A.

### Risk register

- HSA queue stalls during graph capture: profile to confirm.
- Dynamic kernel selection (env flags) inside the chunk forward: the
  graph captures the *current* code path; opt-out env vars become
  ignored after first capture. Document this.

**Expected gain**: −0.2 to −0.5 s/chunk → +5–10 % on top of Phase A.

---

## Phase C — Async h2d uploads with active stream

**Goal**: hide the small h2d uploads (n_valid_swa_arr,
n_active_topk_arr, moe_topk_indices_batch, etc.) behind compute.

### Steps

1. **Identify all blocking h2d uploads in V4F forward**:
   - `pbs.n_valid_swa_arr` upload (line 2716, 3249)
   - `pbs.n_active_topk_arr` upload (line 3154, 3254)
   - `pbs.moe_topk_indices_batch`, `pbs.moe_topk_weights_batch`
     (line 3533, 3536)
   - `state.comp_pos_buf` (line 519, 2027)
   - `pbs.positions` (per-chunk upload in chunk forward)
2. **Migrate to async path**:
   - Use `gpu.hip.memcpy_htod_async(&buf, &bytes, stream)` instead of
     blocking memcpy.
   - Ensure consumer kernels are dispatched on the same stream.
3. **Identify any FORCED syncs** (`device_synchronize`, sync memcpy):
   - Should be only at chunk boundary (end of `forward_prefill_batch_chunk`).
4. **Validation**: byte-eq, bench.

### Risk register

- Async h2d needs pinned host memory for best perf; check if our
  upload paths use pinned.
- Order dependency: kernel that reads the buffer must wait on the
  upload completion.

**Expected gain**: small (a few hundred μs per chunk), but free.

---

## Phase D — Async streams for indexer ⟂ main attention

**Goal**: overlap independent compute branches.

### Prerequisites
- Phase C landed (async basics in place).

### Steps

1. **Identify independent work** in `attention_block_batched_mixed`:
   - Main compressor (kv/score for main attention) vs indexer
     compressor (kv/score for top-K selection) are independent in
     their GEMV/commit/compress phases (both consume tmp_batch).
   - After both finish, indexer top-K is computed, then gather, then
     attention.
2. **Create a second HIP stream** in Gpu (or PBS-owned).
3. **Submit indexer compressor on stream B** while main compressor
   runs on stream A.
4. **Sync at indexer_top_k** (it consumes idx_q + idx_w from both
   compressors' outputs).
5. **Validation**: byte-eq, bench.

### Risk register

- HIP stream synchronization correctness — easy to introduce subtle
  data races. Add asserts in debug mode.
- Cache contention between streams (both hitting DRAM): could be
  neutral or net-negative if memory bus is already saturated.

**Expected gain**: +5–10 % for ratio=4 layers (where indexer fires).

---

## Phase E — F16 residual streams

**Goal**: halve memory traffic on residual stream reads/writes; remove
the 592 `convert_f32_to_f16` launches per chunk.

### Prerequisites
- Full understanding of which kernels touch residual streams.
- PPL validation budget (full sweep at ctx={256, 512, 1024}).

### Steps

1. **Inventory residual stream consumers**:
   - `DeepseekV4State.residual_streams` (or `pbs.streams_batch`)
   - hc_mix_4stream, hc_input_map_4stream, residual_add, etc.
2. **Decide accumulator strategy**: F16 storage, F32 accumulator
   inside each kernel.
3. **Convert each consumer kernel** to take F16 streams.
4. **Eliminate convert_f32_to_f16** for any input that's now F16
   on-storage.
5. **PPL re-validation** at ctx={256, 512, 1024}. Δppl > 2 % is a
   fail → revert.
6. **Bench**: target +5–10 %.

### Risk register

- **F16 overflow at depth**: residual streams accumulate over 43
  layers. With F16 max = 65504, sum of many norm'd-but-not-clamped
  values could hit infinity. Measure carefully.
- Some kernels' internals may need F32 staging anyway. Net win
  diminishes.

**Expected gain**: +5–10 %, conditional on PPL validation.

---

## Phase F — B=128 batched chunks

**Goal**: amortize all per-chunk overhead over 2× more tokens.

### Prerequisites
- Audit PBS allocation at B=128.

### Steps

1. **Increase max_batch** in PBS allocation (currently 64 in
   bench_v4f_batched_prefill).
2. **Verify VRAM headroom**: PBS scales ~linearly with max_batch.
   Current ~100 MB at B=64 → ~200 MB at B=128. Fine on Strix Halo.
3. **Confirm wmma_x_scratch_f16 sizing** still covers max(G *
   per_group_in) at B=128.
4. **Bench at B=128** to measure speedup.
5. **Investigate plateau**: at very high B, MoE may have so many
   routings that some kernels hit per-launch limits or saturation
   on different bottlenecks. Measure to find sweet spot.

### Risk register

- KV cache state per-layer is sized for `cfg.sliding_window`, not B.
  Should be fine.
- At B=128, the MoE GEMV grid becomes (M, K_TOP, 128) = 3M
  workgroups. May approach scheduler limits.

**Expected gain**: +10–15 %, free.

---

## Phase G — DP4A / WMMA INT8 for *non-MoE* HFQ4 paths (deferred)

**Goal**: 2× compute on attention + shared FFN GEMMs.

### Why deferred
Currently those kernels are only ~13 % of total kernel time. Even a
2× speedup on the kernel caps end-to-end gain at ~6 %. Phases A–F
should land first.

### Sketch
- Cast HFQ4 dequant output to INT8 with per-group scale.
- Quantize F32 input to INT8 per-group.
- Use `v_wmma_i32_16x16x16_iu8` for the matmul.
- Multiply (scale_w × scale_x) at output write.

**Expected gain**: +3–5 %, mid-effort.

---

## Phase H — Persistent kernel for per-layer GEMM chain (deferred)

**Goal**: eliminate per-GEMM launch tax via single-launch multi-op
pipeline.

### Why deferred
Big architectural change. After hipGraph (phase B), the launch tax
may already be substantially reduced. Profile *after* Phase B to
re-evaluate whether this is worth it.

---

## Phase I — LDS-staged grouped MoE GEMM (deferred)

**Goal**: process all routings for one expert against an LDS-cached
weight tile.

### Why deferred
The MoE kernels are already at 100 % of measured DRAM peak. Improving
them needs *reducing bytes read*, which means either lower-bit quant
(user disallowed) or expert weight tile re-use across multiple
routings inside one workgroup.

LDS holds ~64 KB per workgroup. An expert's weight slab is 4.5 MB —
can't fit fully in LDS. Would need K-tiling: load K_TILE columns of
the expert weight into LDS, do partial dots across multiple routings,
accumulate.

**Expected gain**: 1.5–2× on MoE GEMV kernels at high B, but high
implementation cost. Re-evaluate after Phases A–F.

---

## Cross-phase validation gates

| stage          | byte-eq bisect (B=1)         | top-1 match (B=64) | PPL re-sweep            |
|----------------|-------------------------------|--------------------|--------------------------|
| after Phase A  | required                      | required           | spot-check ctx=512       |
| after Phase B  | required                      | required           | skip (graph is bit-eq)   |
| after Phase C  | required                      | required           | skip                     |
| after Phase D  | required                      | required           | spot-check ctx=512       |
| after Phase E  | required                      | required           | **full sweep required**  |
| after Phase F  | n/a (config change)           | required           | skip                     |

PPL acceptance: |Δppl| < 2 % vs current baseline (12.245 @ ctx=256,
10.278 @ ctx=512, 7.337 @ ctx=1024). Larger drift → revert.

## Execution order

1. **Phase A** (5–8 hours) — biggest impact, no PPL risk
2. **Phase C** (1–2 hours) — small, free, lays async groundwork
3. **Phase B** (3–5 hours) — stacks on top of A and C
4. **Phase F** (1 hour) — quick free win
5. **Phase D** (3–4 hours) — async streams
6. **Phase E** (4–6 hours) — F16 residual streams, includes PPL sweep
7. *Re-profile and re-rank G/H/I after the above lands.*

## Out-of-scope reminders

- ✗ Lowering routed-expert quant below MQ2-Lloyd — user-disallowed
  on this branch. Re-evaluate later if performance still insufficient.
- ✗ Touching the compressor's F16-native upload (validated at PPL
  parity; do not regress).
- ✗ Vulkan / cross-vendor backend (out of scope per CLAUDE.md).
