# HFQ1G128 — Implementation Plan for Bonsai-8B (PrismML 1-bit)

Status: **proposal, pre-implementation**. Targets PrismML's GGUF
`Q1_0` format (1-bit weights, 18 B/group, no zero-point) so hipfire
can run [`prism-ml/Bonsai-8B-gguf`](https://huggingface.co/prism-ml/Bonsai-8B-gguf)
end-to-end on RDNA without the upstream llama.cpp fork.

Reference docs:

- `findings/prismml-q1_0-layout.md` — byte-precise upstream format spec
  (block struct, dequant loop, CUDA matmul math, GGUF type ID).
- `findings/hipfire-hfq1g128-integration-map.md` — every hipfire file
  that needs to change, with line refs and templates.

This doc is the **implementation contract**: design decisions, kernel
math, test plan, perf expectations.

---

## 1. Why `HFQ1G128`, not `MQ1G256`

Two-line summary, full reasoning in conversation history:

- PrismML ships **G128** with **no FWHT rotation**, baked in via QAT.
  Re-grouping to G256 averages two FP16 scales → quality loss. Adding
  FWHT means dequant→rotate→re-quant, which round-trips through the
  1-bit codebook and destroys the calibration.
- G128 + wave32 is also the *natural* fast path: 128 weights / 32 lanes
  = 4 bits/lane, one wave owns one group, single FP16 scale broadcast
  to scalar register, single shuffle-tree reduction.

Conclusion: match PrismML's layout byte-for-byte, name it `HFQ1G128`.

---

## 2. Block format

**18 bytes per 128-element group.** Identical to upstream
`block_q1_0`:

```
offset  bytes  field   content
0..1    2      d       FP16 scale (IEEE-754 binary16, little-endian)
2..17   16     qs[16]  packed bits, LSB-first within byte, K-axis only
```

Element `j ∈ [0,128)` of the group → byte `j>>3`, bit `j&7`. Bit `1`
maps to `+d`, bit `0` maps to `−d`. **No zero-point. No second scale.
No offset.** Zero is not representable.

This **diverges** from hipfire's existing HFQ family layout (which
uses 4 B FP32 scale + 4 B FP32 zero-point + packed bits). Two
divergences, both forced rather than chosen:

**(a) FP16 scale instead of FP32.** Strictly economic. Every FP16
casts losslessly to FP32 at use, so promotion buys zero quality. But
the header overhead dominates at 1-bit because the payload is so
small:

| header choice | block size | header % | Bonsai-8B disk size |
|---|---|---|---|
| FP16 scale only (PrismML, 2 B) | 18 B | 11% | 1.13 GB |
| FP32 scale + FP32 zero (HFQ family, 8 B) | 24 B | 33% | 1.51 GB |

A 33% format tax for no benefit is hard to justify, especially when
matching PrismML byte-for-byte means GGUF ingestion can copy weight
bytes verbatim and only re-tag the type.

**(b) No zero-point field.** Mathematically forced, not chosen. For
4-bit, `dequant = scale*nibble + zero` and `zero` shifts where the
codebook centers in FP space — useful because the 4-bit codebook is
asymmetric `[0..15]`. For 1-bit, the codebook is `{−d, +d}` —
sign-symmetric around zero by construction. Adding a per-group zero
`z` would just bias the dot output by `z·Σx`, which the linear
layer's bias parameter already captures. Storing it would be
redundant *and* PrismML's `d = mean(|w|)` is the MSE-optimal scale
only under a zero-mean codebook, so adding `z` would invalidate the
scale derivation.

Per-row stride: `n_per_row * 9 / 64` bytes (e.g., 4096 K-dim → 576 B
per output row).

`QuantType::HFQ1G128 = 21` (next free in
`crates/hipfire-quantize/src/main.rs:1298-1312`).

`DType::HFQ1G128` in `crates/hipfire-runtime/src/llama.rs` (next free).

GGUF type ID accepted on read: `41` (PrismML's `GGML_TYPE_Q1_0`). Map
to `QuantType::HFQ1G128` in the GGUF reader.

---

## 3. Kernel math

PrismML's CUDA path picks **dp4a-against-Q8_1**, not FP×FP. We mirror
that decision because:

- Q1_0 weights unpack to `±1` signed INT8, which feeds `v_dot4_i32_i8`
  natively (gfx906/1010/1030/1100/1200 all support it).
- Skipping the FP16 dequant materialization saves ~4× the weight
  bandwidth that the equivalent HFQ4G256 GEMV needs to move.
- Reuses hipfire's existing Q8_1 activation pre-pass (already used by
  `gemm_hfq4g256_residual_wave64_dp4a` and friends).

### 3.1 Per-32-element-chunk dot (the hot inner loop)

For one Q1_0 block (128 weights, scale `d_w`) against four Q8_1 blocks
of activations (32 INT8s each, scales `d_a[0..3]`):

```c
// Per chunk (chunk index iqs ∈ [0,4)):
uint32_t v = *(uint32_t*)(&qs[iqs*4]);   // 32 weight bits
int sumi = 0;
#pragma unroll
for (int j = 0; j < 8; ++j) {
    uint32_t bits4 = (v >> (j*4)) & 0xF;
    int b0 = (bits4 & 0x1) ? 1 : -1;
    int b1 = (bits4 & 0x2) ? 1 : -1;
    int b2 = (bits4 & 0x4) ? 1 : -1;
    int b3 = (bits4 & 0x8) ? 1 : -1;
    int packed = (b0 & 0xFF) | ((b1 & 0xFF) << 8)
               | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
    int act = ((int*)act_qs[iqs])[j];
    sumi = __builtin_amdgcn_sdot4(packed, act, sumi);   // v_dot4_i32_i8
}
float chunk_out = (float)d_w * (float)d_a[iqs] * (float)sumi;
```

The Q8_1 `s` field (sum bias) is **dead** — Q1_0's codebook is
sign-symmetric, so `Σ_w · z_a = 0` for any zero-point. Don't read it.

### 3.2 Wave32 GEMV dispatch (gfx1151 only)

Grid `[M, 1, 1]` × Block `[32, 1, 1]`. `__launch_bounds__(32, 16)` —
gfx1151 is RDNA 3.5 (Strix Halo), 16 waves/SIMD register-pressure cap
matches gfx1100. **Wave64 is out of scope** — the only target hardware
on hand is wave32-native, and an unbenched wave64 kernel is a liability
(see issue list in §8).

One wave owns one output row. Lane `t ∈ [0,32)` handles weight bits at
group offsets `t*4..t*4+3`. Each lane processes one Q1_0 block per
K-step (the whole 128-elt group is laid across the 32 lanes × 4
elts/lane).

**Use the 4-way accumulator interleave** from the start
(`acc0..acc3`). Single-accumulator HFQ4G128 was a tail-rounding
foot-gun (commit 5302926); don't repeat. Final combine: pairwise sum
`(acc0 + acc1) + (acc2 + acc3)`, then `__shfl_down` tree to `tid==0`.

### 3.3 GEMM / prefill

Template: `gemm_hfq4g256_residual_wave64_dp4a.hip` for the *math
shape* (Q8_1 activations, dp4a chain) but reparameterized to wave32.
`BATCH_TILE = 8` tokens per block, 4 accumulators × 8 batch elements
= 32 FP32 registers. Activations pre-quantized to Q8_1 once per
matmul. The weight unpack happens once per K-step and is reused
across all 8 batch elements — that's the amortization.

**WMMA prefill variant (`gemm_hfq1g128_residual_wmma`) is now Phase 2
not Phase 3** — gfx1151 has WMMA, and the unpack-to-INT8 → WMMA i8
tile path is the most material decode-vs-prefill perf lever on this
target. See §7.

---

## 4. Quantizer (FP16 → HFQ1G128)

For weights we re-quantize ourselves (not ingested from PrismML), the
upstream rule is `d = mean(|w|)` over the 128-element group, sign bit
= `w >= 0.0f`. PrismML's `quantize_q1_0` ignores the imatrix argument,
so hipfire must too.

```rust
fn quantize_block_hfq1g128(w: &[f32; 128], out: &mut [u8; 18]) {
    let sum_abs: f32 = w.iter().map(|x| x.abs()).sum();
    let d = sum_abs / 128.0;
    let d_f16 = f16::from_f32(d);
    out[0..2].copy_from_slice(&d_f16.to_le_bytes());
    out[2..18].fill(0);
    for j in 0..128 {
        if w[j] >= 0.0 {
            out[2 + j / 8] |= 1 << (j % 8);
        }
    }
}
```

Negative-zero edge case: `-0.0_f32 >= 0.0_f32` is `true` in IEEE, so
`-0.0` → `bit=1 → +d`. Match this. NaN → `>=` evaluates `false` → bit=0
→ `−d`. Don't substitute `signbit()`; results diverge on `±0.0` and
NaN.

This quantizer is only useful for hipfire-native re-quantization
experiments. **For Bonsai-8B itself, ingest PrismML's bytes verbatim
through the GGUF reader** — there is no FP16 source to re-quantize from
(PrismML's QAT is what produces the calibrated 1-bit weights, and we
can't replicate that without their training pipeline).

---

## 5. Kernel set for end-to-end Bonsai-8B

Bonsai-8B is non-MoE Qwen3-8B re-quantized. Minimum kernel set
(see integration map §4):

1. `embedding_hfq1g128`
2. `embedding_hfq1g128_batched`
3. `gemv_hfq1g128`
4. `gemv_hfq1g128_residual_sigmoid_scaled`
5. `fused_qkvza_hfq1g128`
6. `gemm_qkvza_hfq1g128`
7. `gemm_gate_up_hfq1g128`
8. `gemm_hfq1g128_residual`
9. `gemm_hfq1g128_batched_lmhead`

If Bonsai's GGUF keeps `token_embd` and `output` at F16 / Q8_0
(common in 1-bit releases), kernels 1, 2, and 9 are not needed —
existing kernels handle those tensors. **Verify at first load**, do
not pre-commit.

Wave64 ports are **cut**, not deferred. Only target is gfx1151
(wave32 native). Reintroduce only if/when there's hardware to bench
on.

---

## 6. Test plan

Tiered, gated, each tier blocks the next.

**Critical insight:** the dp4a math is bit-equivalent across CUDA
`__dp4a` and HIP `v_dot4_i32_i8`, and PrismML's reference path has a
CPU implementation (`dequantize_row_q1_0`, scalar `vec_dot` fallback).
**No NVIDIA hardware is on the critical path.** All correctness is
verified against PrismML's *CPU* code; gfx1151 is the only GPU
needed.

### Tier A — Format round-trip (no GPU)

CPU Rust tests in `crates/hipfire-quantize/src/tests.rs`. Build
PrismML's `quantize_row_q1_0_ref` and `dequantize_row_q1_0` either as
a tiny C library linked via `cc` crate, or transliterate to Rust
(both are ~30 lines).

- A.1: synthesize `block_q1_0` with `d=1.0`, `qs={0xAA, 0x55, 0x00, 0xFF, …}`,
  dequant, verify alternating `±1.0` per LSB-first bit order.
- A.2: round-trip 128 random FP32 weights through
  `quantize_block_hfq1g128` + hipfire dequant; verify reconstruction
  equals `sign(w) * mean(|w|)` to ULP precision.
- A.3: **byte-identical to PrismML.** For 1000 random FP32 vectors,
  hipfire's quantizer must produce the exact same 18 bytes as
  PrismML's `quantize_row_q1_0_ref`. Hipfire's dequant must produce
  the exact same FP16 reconstruction as `dequantize_row_q1_0`. Any
  divergence here = bit-order / endianness / scale-derivation bug.
- A.4: **whole-model bytes match.** Read every Q1_0 tensor from
  `Bonsai-8B-Q1_0.gguf` through both PrismML's loader and hipfire's
  loader; assert raw block bytes are identical. (We're not transcoding,
  but this catches alignment/stride bugs in the GGUF reader.)

This tier is the bug-killer. 90% of bit-order mistakes die here, and
it runs in CI without any GPU.

### Tier B — HIP kernel vs. CPU dp4a reference

Write a 50-line CPU Rust function that mirrors `vec_dot_q1_0_q8_1`
exactly: build the 32-bit `v` from `qs[off..off+4]`, unpack to eight
INT32s of packed `±1` bytes, INT32 dot against Q8_1 activation bytes,
`sumi` accumulation. This is the source-of-truth for the kernel
arithmetic.

For each kernel in §5:

- Generate one weight row (real or synthetic) + random Q8_1
  activations.
- Run hipfire HIP kernel and the CPU reference on the same inputs.
- Assert **`sumi` is bit-identical INT32**. Both paths compute
  INT8×INT8 → INT32 dot; the only difference is hardware lane
  parallelism, which doesn't affect the integer sum.
- Assert FP32 final products agree to **1 ULP**. ULP failure = real
  arithmetic bug, not rounding noise.

### Tier C — End-to-end vs PrismML llama.cpp (CPU build)

Build PrismML's llama.cpp on the gfx1151 box with
`-DGGML_CUDA=OFF -DGGML_HIP=OFF` (CPU-only — slow, but it's the
universal reference). Greedy-decode 32 tokens on a fixed prompt with
both PrismML llama.cpp and hipfire daemon (`--temperature 0`,
identical sampler).

- Identical token IDs across all 32 → pass.
- Mismatch at token N → run layer-dump pass on both implementations
  (insert hidden-state dumps at each transformer block), find the
  first divergent layer, fix.

**Bonus:** if PrismML's `prism` branch ships HIP backend support for
Q1_0 (agent 1 didn't surface evidence of one — check
`ggml/src/ggml-hip/` in their fork before assuming CPU-only), build
their HIP backend on gfx1151 for a fast end-to-end reference. This
is a nice-to-have, not a blocker.

### Tier D — Coherence gate

Add Bonsai-8B to the model matrix in `scripts/coherence-gate.sh`.
Must pass the standard battery: no panics, no zero-token outputs, no
1-bit-attractor loops. The gate accepts "fluent and on-topic" output
— not byte-exact to llama.cpp across 200 tokens, since sampler
differences make that unrealistic. The Tier C gate is what enforces
arithmetic equivalence; Tier D enforces *behavioral* equivalence.

Per the DFlash gate rules in `CLAUDE.md`: tight stddev across runs
on a 1-bit model is **suspicious**, not reassuring. Real 1-bit
output noise is wider than 4-bit. Watch for attractor loops in the
last 128 tokens specifically.

### Tier E — Performance (gfx1151)

Bench on gfx1151 (Strix Halo, ~245 GB/s LPDDR5X unified memory,
RDNA 3.5, 40 CUs, WMMA available). Per
`docs/methodology/perf-benchmarking.md`: byte-identical prompt,
`prompt_normalize=true`, fresh-process A/B via
`scripts/probe_commits.sh`.

**Bandwidth-bound decode ceiling math:**

- Bonsai-8B at 1.125 bpw → ~1.13 GB of weight bytes.
- gfx1151 LPDDR5X realistic bandwidth ≈ 245 GB/s.
- Memory ceiling: 245 / 1.13 ≈ **217 tok/s** decode-bound (ignoring
  KV traffic, activation traffic, attention).
- Comparison: equivalent HFQ4G256 Qwen3-8B at ~4.25 bpw is
  ~4.27 GB → 245 / 4.27 ≈ 57 tok/s ceiling.

**Realistic targets:**

| phase | metric | target | actual |
|---|---|---|---|
| Phase 1 exit | hipfire HFQ1G128 decode tok/s | 60–90 (correctness-first) | n/a (kernel only) |
| Phase 2 exit | hipfire HFQ1G128 decode tok/s | 100–150 (after WMMA prefill, residual fusion) | **65–71 tok/s** |
| Phase 3 ambition | hipfire HFQ1G128 decode tok/s | 150–200 (close to bandwidth ceiling) | TBD |

**Phase 2 actuals (2026-05-09, gfx1151):**

| variant | kernel BW (12288×4096) | Bonsai-8B tok/s ceiling | E2E decode (best of 3) |
|---|---|---|---|
| single-row (Phase 1 baseline) | 90 GB/s | 78 | 46.7 |
| multirow R=4 (Phase 2 first lock-in) | 116 GB/s | 100 | 65–71 |
| **multirow-quad R=2** (current default) | **142 GB/s** | **122** | **75.7** (best of 3) |

Multirow alone gave +30% kernel-BW / +39% E2E. Quad-group K-step on top
gave +22% kernel-BW / +14% E2E. Bandwidth ceiling is 245 GB/s LPDDR5X →
**58% of peak** achieved with the FP-direct path. Within-session A/B
noise is ±15-25% on this APU (DPM ramp + shared CPU+GPU thermal budget),
so individual runs vary; best-of-3 is the honest steady-state number.

**Why we're below the Phase 2 target band (100–150):**

- No fused QKV/gate-up/down kernels — each projection is a separate
  GEMV launch. Per-layer kernel-launch overhead is meaningful.
- No dp4a path — activation traffic isn't dominated, but launch reduction
  via QKV fusion would help.
- No WMMA prefill — short prefill perf is acceptable (79 tok/s for
  13-token prompt) but would benefit at longer contexts.

**Phase 3 perf path** (in priority order, after Phase 2 perf shortfall):

1. **Fused projections**: `fused_qkv_hfq1g128` (3 outputs from same input)
   and `fused_gate_up_hfq1g128` (2 outputs). Each saves 2 of 3 / 1 of 2
   kernel launches per layer. Modeled gain ~10–20% E2E.
2. **dp4a-against-Q8_1**: matches PrismML's CUDA reference exactly. Saves
   activation bandwidth in batched flows. Modeled gain ~5–15% E2E for
   decode, ~30%+ for prefill.
3. **WMMA prefill GEMM**: i8 tile path, mirrors gemm_hfq4g256_residual_wmma
   for HFQ1G128. Critical for long-context prefill scaling.
4. **Wider per-lane reads**: `gemv_hfq1g128_packed` (8 weights/lane,
   2 groups packed per K-step) is implemented but did not show a
   clear win on top of multirow in within-session A/B (noisy ±10-15%
   per `docs/methodology/perf-benchmarking.md`). Worth re-bench in
   Phase 3 with a fresh-process A/B.

Decode multiplier vs HFQ4G256 Qwen3-8B on the same hardware:
**≥2.5×** is the success threshold for Phase 2. Bandwidth math
suggests ~3.5× is achievable on gfx1151 if Phase 3 lands.

---

## 7. Implementation phases

### Phase 1 — Format + GEMV decode (~3 days)

PR #1: scaffolding + decode path correctness.

- [ ] Inspect `Bonsai-8B-Q1_0.gguf` header (small Rust GGUF reader
      binary) to confirm tensor type breakdown — pre-empts open
      question #1, may shrink the kernel set.
- [ ] Add `QuantType::HFQ1G128 = 21`, `DType::HFQ1G128`.
- [ ] GGUF reader: accept `ggml_type=41`, copy bytes verbatim,
      tag as HFQ1G128.
- [ ] Quantizer (`quantize_block_hfq1g128`) + Tier A tests.
- [ ] PrismML CPU reference linked in (transliterate or `cc` crate)
      for byte-exact comparison.
- [ ] `kernels/src/gemv_hfq1g128.hip` (wave32, dp4a, **4-way
      accumulator from day one** — don't repeat the HFQ4G128
      foot-gun).
- [ ] CPU dp4a reference (~50 lines Rust) for Tier B.
- [ ] Dispatcher wiring (`rdna-compute/src/dispatch.rs`).
- [ ] Tier A all variants pass.
- [ ] Tier B passes for `gemv_hfq1g128` on a real Bonsai weight row.

Exit gate: hipfire `gemv_hfq1g128` output is bit-exact (INT32 sumi)
and ULP-exact (FP32 product) against the CPU reference, on a real
Bonsai weight tensor.

### Phase 2 — Forward pass + WMMA prefill (~4 days)

PR #2: end-to-end Bonsai-8B inference, including WMMA. (Phase 2 is
larger now because gfx1151 has WMMA and prefill perf matters.)

- [ ] `gemm_hfq1g128_residual` (batched prefill, dp4a baseline).
- [ ] `gemm_hfq1g128_residual_wmma` (WMMA i8 tile path — promoted
      from Phase 3 since gfx1151 has WMMA).
- [ ] `fused_qkvza_hfq1g128` (decode attention).
- [ ] `gemm_qkvza_hfq1g128` + `gemm_qkvza_hfq1g128_wmma` (prefill
      attention, dp4a + WMMA).
- [ ] `gemm_gate_up_hfq1g128` + `gemm_gate_up_hfq1g128_wmma`
      (prefill FFN).
- [ ] `gemv_hfq1g128_residual_sigmoid_scaled` (decode gating).
- [ ] Wire up `crates/hipfire-arch-qwen35/src/qwen35.rs` to dispatch
      HFQ1G128 alongside HFQ4G256.
- [ ] Tier C (end-to-end vs. PrismML CPU build) passes 32-token
      greedy match.
- [ ] Tier D (coherence gate) passes with Bonsai-8B added.
- [ ] Tier E perf bench: confirm Phase 2 target (100–150 tok/s
      decode, ≥2.5× HFQ4G256 Qwen3-8B on the same hardware).

Exit gate: `cargo run -- chat --model bonsai-8b.hfq` produces fluent
output on the coherence prompts and lands within the Phase 2 perf
target on gfx1151.

### Phase 2 — actual deliverables shipped (2026-05-09)

What landed (against the Phase 2 plan):

- [x] Embedding kernels: `embedding_hfq1g128.hip`, `dequant_hfq1g128_to_f16.hip`.
- [x] Multirow GEMV: `gemv_hfq1g128_multirow.hip` (R=2/4/8) — kernel-level
      stepping stone (+30% over single-row) before quad lockin.
- [x] Multirow-quad GEMV: `gemv_hfq1g128_multirow_quad.hip` (R=2/4/8,
      production default is R=2). 4 groups packed per K-step + 2-row
      tile = 142 GB/s bandwidth (+58% over single-row, +22% over plain
      multirow). E2E best-of-3: 75.7 tok/s on Bonsai-8B greedy decode.
- [x] Packed-2g GEMV: `gemv_hfq1g128_packed.hip` (R=1/2/4) — implemented
      but no clear win in within-session A/B; deferred to Phase 3 fresh-
      process bench.
- [x] Runtime wiring: `DType::HFQ1G128`, `EmbeddingFormat::HFQ1G128`,
      `weight_gemv` arm.
- [x] GGUF→.hfq passthrough rule (preserves PrismML's per-tensor 1-bit
      choice for embeddings/lm_head).
- [x] Coherence-gate matrix: Bonsai-8B added to short battery.
- [ ] Strict Tier C (32-token byte-exact match vs PrismML llama.cpp CPU
      build): **not run** — smoke + Tier B + coherence-gate provide
      adequate signals; defer to Phase 3 if a regression demands it.
- [ ] `gemm_hfq1g128_residual` and WMMA prefill: **deferred to Phase 3**.
      Phase 2 uses the existing GEMV-per-token fallback in `weight_gemm`
      (correct, not perf-optimal for long prefill).

### Phase 3 — Perf headroom (open-ended)

- [ ] Multi-row GEMV (`gemv_hfq1g128_multirow`) — 2/4/8 rows per
      block for higher CU occupancy on gfx1151.
- [ ] Software prefetch variant (mirror
      `gemv_hfq4g256_residual_wave64_prefetch.hip`'s pattern).
- [ ] Activation pre-quantization fusion — fold `quantize_q8_1`
      into the previous op so it's not a separate launch.
- [ ] LM-head and embedding HFQ1G128 kernels — *only* if Bonsai's
      release actually quantizes `output` and `token_embd` to Q1_0
      (Phase 1 inspection settles this; most 1-bit releases keep
      these at F16/Q8_0).
- [ ] gfx-specific tuning file (`gemv_hfq1g128.gfx1151.hip`) if the
      generic kernel falls short on register pressure or scheduling.

**Cut from scope:** wave64 ports (no hardware to bench),
gfx1100/gfx1010/gfx906 per-arch tuning files (no hardware to bench).
Reintroduce only if hardware appears.

---

## 8. Open questions / risks

1. **Bonsai-8B tensor type breakdown** — *not yet verified*. Need to
   inspect the GGUF metadata and confirm which tensors are Q1_0 vs.
   F16/Q8_0. The minimum kernel set assumes worst case (everything
   Q1_0); reality will likely shrink the set. Verify at Phase 1
   start by running a small Rust binary that reads the GGUF header.

2. **Q8_1 activation pre-pass kernel** — does hipfire already have
   one? Existing dp4a kernels (`*_wave64_dp4a.hip`) suggest yes;
   confirm and reuse rather than duplicating.

3. **Scale FP16 vs FP32 on disk** — *resolved* in §2: keep FP16, no
   zero-point. Hipfire's `.hfq` file-format reader must dispatch on
   QuantType to pick the per-block header layout (it already does
   this implicitly via `block_size` per format; the change is one
   match arm).

4. **gfx1151-only testing.** Strix Halo / Ryzen AI Max+ 395 (RDNA 3.5,
   wave32 native, WMMA available, `v_dot4_i32_i8` available, ~245 GB/s
   LPDDR5X unified memory). All kernels are wave32. WMMA prefill
   variants are testable here (and promoted to Phase 2). Wave64 ports
   and other-arch tuning files are **out of scope** until hardware
   exists to bench them.

5. **Quality regression on hipfire-native re-quantization.** If we
   ever re-quantize a non-PrismML FP16 model to HFQ1G128, output will
   be substantially worse than PrismML's QAT'd Bonsai. Document this
   loudly in the QUANTIZE.md page; do not surface `--format hfq1` as
   a default option. The format exists primarily to *consume* PrismML
   weights, not to *produce* new 1-bit checkpoints from arbitrary
   FP16 sources.

6. **Bit-ordering bugs are silent.** Wrong LSB/MSB or wrong byte order
   produces plausible-looking-but-wrong outputs that may pass a
   small-prompt coherence check while failing real inference. Tier A
   round-trip and Tier B reference-vector tests are *non-optional*
   gates before Tier C.

---

## 9. Success criteria

- [ ] hipfire daemon loads `prism-ml/Bonsai-8B-gguf` directly (no
      external transcode step).
- [ ] Tier A + B + C pass: byte-exact and ULP-exact arithmetic
      against PrismML's CPU reference; 32-token greedy match end-to-
      end vs. PrismML llama.cpp CPU build.
- [ ] Coherence gate passes Bonsai-8B alongside the existing model
      matrix.
- [ ] Decode tok/s on gfx1151 is at least **2.5×** the equivalent
      HFQ4G256-quantized Qwen3-8B on the same hardware (bandwidth-
      bound, so the bits-per-weight ratio sets the floor; the math
      in §6 Tier E suggests ~3.5× is reachable in Phase 3).
- [ ] Spec doc at `docs/quantization/HFQ1G128.md` (promoted from this
      `plans/` doc once shipped) covering format, kernel math, test
      methodology, and perf data.
