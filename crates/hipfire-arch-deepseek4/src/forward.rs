//! V4F forward pass — skeleton.
//!
//! Layout-only: the function signatures and per-layer call sequence
//! are locked in; the bodies are `unimplemented!` until each piece
//! gets wired. Reading this file gives a future implementer (or
//! reviewer) the entire decode-step flow at a glance.
//!
//! The seven GPU-validated kernels referenced here:
//!   - `gpu.hc_compute_control`       (Phase 3)
//!   - `gpu.hc_sinkhorn_4x4`          (Phase 3)
//!   - `gpu.hc_mix_4stream`           (Phase 3)
//!   - `gpu.indexer_compressed_k_score`  (Phase 2)
//!   - `gpu.indexer_top_k`            (Phase 2)
//!   - `gpu.indexer_kv_gather`        (Phase 2)
//!   - `gpu.rope_tail_halfsplit`      (Phase 4)
//!
//! Existing hipfire-runtime kernels reused (no V4F-specific impl):
//!   - RMSNorm
//!   - Quantized GEMV (MQ-family) for Q-LoRA, KV, O-LoRA, experts
//!   - Embedding lookup, lm_head matmul, sampler

use crate::{DeepseekV4Config, DeepseekV4State, DeepseekV4Weights};
use rdna_compute::{DType, Gpu, GpuTensor};

/// V4F GEMV dispatch: switch kernel based on weight dtype.
///
/// - `DType::MQ4G256` (default V4F non-expert quant): consume FWHT-rotated
///   input via `gemv_mq4g256_prerotated`. This is the existing fast path.
/// - `DType::F32` (set by `--non-expert-f16` quantizer flag, F16 source
///   converted to F32 on upload): consume plain RMSNorm'd input (no FWHT)
///   via `gemv_f32`. Used to faithfully reproduce antirez/ds4's PROVEN
///   recipe of keeping compressor / indexer / attn projections at F16
///   precision.
///
/// Caller passes BOTH the FWHT-rotated and plain inputs; helper picks
/// whichever the weight needs. `m` and `k` are passed-through for the
/// MQ4 path only — gemv_f32 derives them from the weight's shape.
fn gemv_auto(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x_rotated: &GpuTensor,
    x_plain: &GpuTensor,
    y: &GpuTensor,
    m: usize, k: usize,
) -> Result<(), String> {
    // Dispatch by GpuTensor.dtype (set by upload_quant_or_f16):
    //   F32  — F16-source weight, decoded at upload (legacy decode-only path).
    //   F16  — F16-source kept native. Uses gemm_f16_x_f16_wmma at B=1.
    //   Q8_0 — Q8F16-source (antirez attn/shared quant). Use plain input.
    //   else (Raw) — MQ4G256 default. Use FWHT-rotated input.
    match weight.dtype {
        DType::F32 => gpu.gemv_f32(weight, x_plain, y)
            .map_err(|e| format!("gemv_f32: {e:?}")),
        // F16: gemv_f16_xf32 keeps F32 input precision (reads F16 weight,
        // casts in-loop, F32 multiply-accumulate). The legacy
        // gemv_f16_x_decode path converts F32→F16 input before WMMA,
        // losing ~13 mantissa bits — that made F16 measure worse than Q8
        // for downstream tasks. Opt back to lossy WMMA via
        // HIPFIRE_V4F_F16_WMMA=1 for perf comparison.
        DType::F16 => {
            let use_wmma = std::env::var("HIPFIRE_V4F_F16_WMMA")
                .map(|s| s == "1").unwrap_or(false);
            if use_wmma {
                gemv_f16_x_decode(gpu, weight, x_plain, y, m, k)
            } else {
                gpu.gemv_f16_xf32(weight, x_plain, y, m, k)
                    .map_err(|e| format!("gemv_f16_xf32: {e:?}"))
            }
        }
        // Q8 decode (B=1) stays on the scalar `gemv_q8_0` kernel. Empirically
        // measured: the WMMA path at B=1 wastes 15/16 of the 16×16 output
        // tile on unused N-axis columns, while scalar gemv_q8_0 produces
        // exactly 1 output per thread with no wasted work. Result: WMMA at
        // B=1 is 30% SLOWER than scalar for Q8 on gfx1151 (measured
        // 2026-05-20). The WMMA path stays opt-in via the decode helper
        // below for future regression checks; the batched path
        // (gemv_auto_batched_wmma) does benefit and IS WMMA by default.
        DType::Q8_0 => gpu.gemv_q8_0(weight, x_plain, y, m, k)
            .map_err(|e| format!("gemv_q8_0: {e:?}")),
        _ => gpu.gemv_mq4g256_prerotated(weight, x_rotated, y, m, k)
            .map_err(|e| format!("gemv_mq4g256_prerotated: {e:?}")),
    }
}

/// F16-weight single-token decode: F32 input → F16 (small scratch),
/// then `gemm_f16_x_f16_wmma` at B=1. The WMMA tile shape is 16×16
/// so even at B=1 the grid is M/16; not as bandwidth-saturating as
/// the prefill path but still strictly faster than gemv_f32 on
/// F16-source weights because we read half the weight bytes.
///
/// Allocates a per-call scratch — fine for decode latency since
/// decode-step inputs are O(K) floats.
fn gemv_f16_x_decode(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x_plain: &GpuTensor,
    y: &GpuTensor,
    m: usize, k: usize,
) -> Result<(), String> {
    // Scratch lives for this call only — k floats.
    let scratch = gpu.alloc_tensor(&[k], DType::F16)
        .map_err(|e| format!("gemv_f16 scratch alloc: {e:?}"))?;
    gpu.convert_f32_to_f16(x_plain, &scratch, k as i64)
        .map_err(|e| format!("gemv_f16 convert: {e:?}"))?;
    gpu.gemm_f16_x_f16_wmma(weight, &scratch, y, m, k, 1)
        .map_err(|e| format!("gemv_f16 wmma: {e:?}"))
}

/// Q8_0-weight single-token decode via WMMA-Q8: F32 input → F16
/// (small scratch), then `gemm_q8_0_wmma` at B=1. Symmetric with
/// `gemv_f16_x_decode` — the WMMA tile is 16×16 so at B=1 the
/// N-dimension is underutilised, but the M-dimension is fully
/// covered and the matrix-multiply hardware still beats the scalar
/// `gemv_q8_0` per-FMA throughput.
///
/// Opt out with `HIPFIRE_V4F_Q8_WMMA=0` (same env knob as the
/// batched path) — falls back to the scalar kernel.
fn gemv_q8_0_via_wmma_decode(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x_plain: &GpuTensor,
    y: &GpuTensor,
    m: usize, k: usize,
) -> Result<(), String> {
    let wmma_on = std::env::var("HIPFIRE_V4F_Q8_WMMA")
        .map(|s| s != "0").unwrap_or(true);
    if !wmma_on {
        return gpu.gemv_q8_0(weight, x_plain, y, m, k)
            .map_err(|e| format!("gemv_q8_0: {e:?}"));
    }
    let scratch = gpu.alloc_tensor(&[k], DType::F16)
        .map_err(|e| format!("gemv_q8 wmma scratch alloc: {e:?}"))?;
    gpu.convert_f32_to_f16(x_plain, &scratch, k as i64)
        .map_err(|e| format!("gemv_q8 wmma convert: {e:?}"))?;
    gpu.gemm_q8_0_wmma(weight, &scratch, y, m, k, 1)
        .map_err(|e| format!("gemm_q8_0_wmma decode: {e:?}"))
}

/// Batched twin of `gemv_auto` for Phase B2 chunk forward.
///
/// Same dispatch shape but each call processes `batch_size` inputs against
/// a single weight matrix. Output `y` is row-major `[batch_size, m]` —
/// matches what concatenating `batch_size` sequential gemv_auto outputs
/// would produce.
///
/// Inputs:
///   - `x_rotated_batch`: `[batch_size, k]` FWHT-rotated (consumed by the
///     MQ4 path only)
///   - `x_plain_batch`:   `[batch_size, k]` plain RMSNorm'd (consumed by
///     the F32 and Q8 paths)
///
/// Backed by the existing GEMM-batched kernels:
///   - F32  → `gemm_f32_batched` (M_kernel=batch, N_kernel=output_dim)
///   - Q8_0 → `gemm_q8_0_batched_chunked` (handles batch > 64 via internal
///            sub-batching; same MAX_BATCH=64 as the underlying kernel)
///   - Raw (MQ4G256) → `gemm_hfq4g256` (consumes pre-rotated x)
///
/// At batch_size == 1 each path reduces to the equivalent of one
/// sequential gemv_auto call against the same weight; per-row outputs
/// match within FMA-order ε.
#[allow(dead_code, clippy::too_many_arguments)]
fn gemv_auto_batched(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x_rotated_batch: &GpuTensor,
    x_plain_batch: &GpuTensor,
    y: &GpuTensor,
    m: usize, k: usize,
    batch_size: usize,
) -> Result<(), String> {
    gemv_auto_batched_wmma(
        gpu, weight, x_rotated_batch, x_plain_batch, y,
        m, k, batch_size, /*x_f16_scratch=*/ None,
    )
}

/// `gemv_auto_batched` plus an opt-in WMMA path. When `x_f16_scratch`
/// is `Some` and the weight is HFQ4/MQ4 (Raw) on a WMMA-capable arch,
/// stages the F32 input → F16 once and dispatches `gemm_hfq4g256_wmma`.
/// Falls back to the scalar path on other dtypes / when scratch is None.
fn gemv_auto_batched_wmma(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x_rotated_batch: &GpuTensor,
    x_plain_batch: &GpuTensor,
    y: &GpuTensor,
    m: usize, k: usize,
    batch_size: usize,
    x_f16_scratch: Option<&GpuTensor>,
) -> Result<(), String> {
    match weight.dtype {
        DType::F32 => {
            if std::env::var("HIPFIRE_V4F_F32_TRACE").is_ok() {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static N: AtomicUsize = AtomicUsize::new(0);
                let c = N.fetch_add(1, Ordering::Relaxed);
                if c < 8 {
                    eprintln!("[F32_TRACE #{c}] m={m} k={k} B={batch_size} weight.shape={:?}",
                        weight.shape);
                }
            }
            gpu.gemm_f32_register_tiled(
                weight, x_plain_batch, y, m, k, batch_size,
            ).map_err(|e| format!("gemm_f32_register_tiled: {e:?}"))
        },
        DType::Q8_0 => {
            // WMMA-Q8 route mirrors the HFQ4 WMMA path: stage F32 input
            // → F16 once and dispatch `gemm_q8_0_wmma`. 11–30× microbench
            // speedup over the scalar `gemm_q8_0_batched_chunked` per
            // bench_q8_wmma_variants. Opt-out via HIPFIRE_V4F_Q8_WMMA=0
            // for diagnosis or if a future kernel regression surfaces.
            let wmma_on = std::env::var("HIPFIRE_V4F_Q8_WMMA")
                .map(|s| s != "0").unwrap_or(true);
            if wmma_on {
                if let Some(scratch) = x_f16_scratch {
                    let n = (batch_size * k) as i64;
                    gpu.convert_f32_to_f16(x_plain_batch, scratch, n)
                        .map_err(|e| format!("convert_f32_to_f16 (Q8 WMMA): {e:?}"))?;
                    return gpu.gemm_q8_0_wmma(weight, scratch, y, m, k, batch_size)
                        .map_err(|e| format!("gemm_q8_0_wmma: {e:?}"));
                }
            }
            gpu.gemm_q8_0_batched_chunked(
                weight, x_plain_batch, y, m, k, batch_size,
            ).map_err(|e| format!("gemm_q8_0_batched_chunked: {e:?}"))
        },
        DType::F16 => {
            // F16 weight: need to stage F32 input to F16 too, then WMMA.
            if let Some(scratch) = x_f16_scratch {
                let n = (batch_size * k) as i64;
                gpu.convert_f32_to_f16(x_plain_batch, scratch, n)
                    .map_err(|e| format!("convert_f32_to_f16 (F16 weight): {e:?}"))?;
                gpu.gemm_f16_x_f16_wmma(weight, scratch, y, m, k, batch_size)
                    .map_err(|e| format!("gemm_f16_x_f16_wmma: {e:?}"))
            } else {
                Err("F16 weight requires WMMA path with x_f16_scratch".to_string())
            }
        }
        _ => {
            // HFQ4G256/Raw. WMMA route requires F16 input staging.
            // Note: HFQ4 expects FWHT-rotated input.
            let wmma_on = std::env::var("HIPFIRE_V4F_HFQ4_WMMA")
                .map(|s| s != "0").unwrap_or(true);
            if wmma_on {
                if let Some(scratch) = x_f16_scratch {
                    let n = (batch_size * k) as i64;
                    gpu.convert_f32_to_f16(x_rotated_batch, scratch, n)
                        .map_err(|e| format!("convert_f32_to_f16 (HFQ4 WMMA): {e:?}"))?;
                    return gpu.gemm_hfq4g256_wmma(weight, scratch, y, m, k, batch_size)
                        .map_err(|e| format!("gemm_hfq4g256_wmma: {e:?}"));
                }
            }
            gpu.gemm_hfq4g256(weight, x_rotated_batch, y, m, k, batch_size)
                .map_err(|e| format!("gemm_hfq4g256: {e:?}"))
        }
    }
}

/// V4F Compressor decode step (phase 3b scaffold — not yet wired).
///
/// Implements the upstream `Compressor.forward` decode case
/// (start_pos != 0):
///
///   kv = wkv @ x_rotated     [coff * head_dim]
///   score = wgate @ x_rotated [coff * head_dim]
///   score += ape[pos % ratio]
///   kv_state[ratio + pos%ratio]    = kv     (overlap=true)
///   score_state[ratio + pos%ratio] = score
///   if (pos+1) % ratio == 0:
///     overlap_concat → [2*ratio, head_dim]  for kv and score
///     softmax_pool   → [head_dim] compressed
///     rmsnorm (compressor.norm)
///     if is_indexer: tail RoPE (compress_rope_theta = 160000)
///     kv_cache[pos // ratio] = compressed
///     shift kv_state[:ratio] = kv_state[ratio:]  (and score_state)
///
/// Parameterized by `is_indexer`:
///   - false → main attn compressor; head_dim = cfg.head_dim = 512;
///     no RoPE on output; targets `state._indexer[l].main_*`
///   - true  → indexer's sub-compressor; head_dim = idx_head_dim = 128;
///     applies tail RoPE with cfg.compress_rope_theta;
///     targets `state._indexer[l].indexer_*`
///
/// TODO: implement (kernels ready: compressor_softmax_pool_f32 +
/// compressor_overlap_concat_f32). See `docs/plans/deepseek4-next-
/// session.md` for the precise step-by-step.
#[allow(dead_code, clippy::too_many_arguments)]
fn compressor_forward(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    x_rotated: &GpuTensor,
    position: u32,
    is_indexer: bool,
) -> Result<(), String> {
    compressor_forward_impl(
        cfg, weights, state, gpu, layer_idx, x_rotated, position, is_indexer,
        /*pre_batched=*/None,
    )
}

/// Variant of `compressor_forward` that uses pre-batched wkv/wgate
/// outputs computed once per (layer, compressor) for all B positions
/// in a chunk. Skips the per-position GEMVs entirely; the caller is
/// responsible for running gemv_auto_batched on the full tmp/tmp_plain
/// batch and providing the resulting (kv, score) buffers with a
/// per-position offset into the [B, proj_dim] view.
#[allow(dead_code, clippy::too_many_arguments)]
fn compressor_forward_prebatched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    position: u32,
    is_indexer: bool,
    kv_batch: &GpuTensor,
    score_batch: &GpuTensor,
    batch_offset: usize,
) -> Result<(), String> {
    let null_x = state.tmp.as_ref()
        .ok_or_else(|| format!("compressor_forward_prebatched: state.tmp missing l{layer_idx}"))?
        .sub_offset(0, cfg.hidden_size);
    compressor_forward_impl(
        cfg, weights, state, gpu, layer_idx, &null_x, position, is_indexer,
        Some((kv_batch, score_batch, batch_offset)),
    )
}

#[allow(dead_code, clippy::too_many_arguments)]
fn compressor_forward_impl(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    x_rotated: &GpuTensor,
    position: u32,
    is_indexer: bool,
    pre_batched: Option<(&GpuTensor, &GpuTensor, usize)>,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let ratio = layer.compress_ratio as usize;
    if ratio == 0 { return Ok(()); }
    if is_indexer && ratio != 4 { return Ok(()); }

    let overlap = ratio == 4;
    let coff: usize = if overlap { 2 } else { 1 };
    let head_dim = if is_indexer { cfg.index_head_dim } else { cfg.head_dim };
    let proj_dim = coff * head_dim;
    let state_rows = coff * ratio;  // 8 for ratio=4 overlap, 128 for ratio=128

    // Pick weights based on which compressor (main vs indexer).
    let (wkv, wgate, norm, ape) = if is_indexer {
        (
            layer.indexer_compressor_wkv.as_ref()
                .ok_or_else(|| format!("idx_comp_wkv l{layer_idx}"))?,
            layer.indexer_compressor_wgate.as_ref()
                .ok_or_else(|| format!("idx_comp_wgate l{layer_idx}"))?,
            layer.indexer_compressor_norm.as_ref()
                .ok_or_else(|| format!("idx_comp_norm l{layer_idx}"))?,
            layer.indexer_compressor_ape.as_ref()
                .ok_or_else(|| format!("idx_comp_ape l{layer_idx}"))?,
        )
    } else {
        (
            layer.compressor_wkv.as_ref()
                .ok_or_else(|| format!("comp_wkv l{layer_idx}"))?,
            layer.compressor_wgate.as_ref()
                .ok_or_else(|| format!("comp_wgate l{layer_idx}"))?,
            layer.compressor_norm.as_ref()
                .ok_or_else(|| format!("comp_norm l{layer_idx}"))?,
            layer.compressor_ape.as_ref()
                .ok_or_else(|| format!("comp_ape l{layer_idx}"))?,
        )
    };

    let max_compressed: usize = std::env::var("HIPFIRE_V4F_MAX_COMPRESS_POS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2048);

    // Lazy-allocate state buffers per (layer, compressor-type).
    {
        let l_state = &mut state._indexer[layer_idx];
        if is_indexer {
            if l_state.indexer_kv_state.is_none() {
                l_state.indexer_kv_state = Some(gpu.zeros(&[state_rows, proj_dim], DType::F32)
                    .map_err(|e| format!("alloc idx kv_state l{layer_idx}: {e:?}"))?);
            }
            if l_state.indexer_score_state.is_none() {
                l_state.indexer_score_state = Some(gpu.zeros(&[state_rows, proj_dim], DType::F32)
                    .map_err(|e| format!("alloc idx score_state l{layer_idx}: {e:?}"))?);
            }
            if l_state.indexer_kv_cache.is_none() {
                l_state.indexer_kv_cache = Some(gpu.zeros(&[max_compressed, head_dim], DType::F32)
                    .map_err(|e| format!("alloc idx kv_cache l{layer_idx}: {e:?}"))?);
            }
        } else {
            if l_state.main_kv_state.is_none() {
                l_state.main_kv_state = Some(gpu.zeros(&[state_rows, proj_dim], DType::F32)
                    .map_err(|e| format!("alloc main kv_state l{layer_idx}: {e:?}"))?);
            }
            if l_state.main_score_state.is_none() {
                l_state.main_score_state = Some(gpu.zeros(&[state_rows, proj_dim], DType::F32)
                    .map_err(|e| format!("alloc main score_state l{layer_idx}: {e:?}"))?);
            }
            if l_state.main_kv_cache.is_none() {
                l_state.main_kv_cache = Some(gpu.zeros(&[max_compressed, head_dim], DType::F32)
                    .map_err(|e| format!("alloc main kv_cache l{layer_idx}: {e:?}"))?);
            }
        }
    }

    // Per-step scratch — lazy-alloc on layer's IndexerLayerState.
    {
        let l_state = &mut state._indexer[layer_idx];
        if l_state.comp_kv_buf.is_none() {
            l_state.comp_kv_buf = Some(gpu.alloc_tensor(&[proj_dim], DType::F32)
                .map_err(|e| format!("alloc comp_kv_buf l{layer_idx}: {e:?}"))?);
        }
        if l_state.comp_score_buf.is_none() {
            l_state.comp_score_buf = Some(gpu.alloc_tensor(&[proj_dim], DType::F32)
                .map_err(|e| format!("alloc comp_score_buf l{layer_idx}: {e:?}"))?);
        }
        if overlap && l_state.comp_concat_kv.is_none() {
            l_state.comp_concat_kv = Some(gpu.alloc_tensor(
                &[2 * ratio, head_dim], DType::F32)
                .map_err(|e| format!("alloc comp_concat_kv l{layer_idx}: {e:?}"))?);
        }
        if overlap && l_state.comp_concat_score.is_none() {
            l_state.comp_concat_score = Some(gpu.alloc_tensor(
                &[2 * ratio, head_dim], DType::F32)
                .map_err(|e| format!("alloc comp_concat_score l{layer_idx}: {e:?}"))?);
        }
    }

    let hidden = cfg.hidden_size;
    let pos = position as usize;
    let slot = if overlap { ratio + pos % ratio } else { pos % ratio };

    // 1. kv = wkv @ x_rotated; score = wgate @ x_rotated
    //    Dispatch: MQ4 path uses x_rotated (FWHT'd); F16 path uses
    //    tmp_plain (plain RMSNorm, no FWHT — see q_lora step 1b).
    // If pre_batched is Some, the caller has already run the GEMVs
    // for all B positions; we just point kv/score at the b-th slice.
    let owned_kv_buf;
    let owned_score_buf;
    let (kv_buf, score_buf) = if let Some((kv_b, score_b, b_off)) = pre_batched {
        owned_kv_buf = kv_b.sub_offset(b_off * proj_dim, proj_dim);
        owned_score_buf = score_b.sub_offset(b_off * proj_dim, proj_dim);
        (&owned_kv_buf, &owned_score_buf)
    } else {
        let kvb = state._indexer[layer_idx].comp_kv_buf.as_ref().unwrap();
        let scb = state._indexer[layer_idx].comp_score_buf.as_ref().unwrap();
        let tmp_plain = state.tmp_plain.as_ref()
            .ok_or_else(|| format!("comp l{layer_idx}: tmp_plain missing (q_lora must run first)"))?;
        gemv_auto(gpu, wkv, x_rotated, tmp_plain, kvb, proj_dim, hidden)?;
        gemv_auto(gpu, wgate, x_rotated, tmp_plain, scb, proj_dim, hidden)?;
        (kvb, scb)
    };

    // 2. score += ape[pos % ratio]
    // ape is shape [ratio, proj_dim] F16; row pos%ratio is proj_dim consecutive F16s.
    // We need an F16→F32 add-in-place. Simplest: convert ape row to F32 once at load.
    // For now, treat ape as a raw F16 view and do an add via a tiny dispatch wrapper.
    // PUNT: this needs an f16-add-to-f32 kernel we don't have; skip the ape add
    // in this commit (the ape is small positional encoding — quality impact bounded).
    // TODO: write add_f16_to_f32_inplace kernel and call:
    //   gpu.add_f16_to_f32_inplace(score_buf, ape_row_view, proj_dim)
    let _ = ape;

    // 3. Store kv_buf at kv_state[slot, :], score_buf at score_state[slot, :].
    {
        let l_state = &state._indexer[layer_idx];
        let kv_state = if is_indexer {
            l_state.indexer_kv_state.as_ref().unwrap()
        } else {
            l_state.main_kv_state.as_ref().unwrap()
        };
        let score_state = if is_indexer {
            l_state.indexer_score_state.as_ref().unwrap()
        } else {
            l_state.main_score_state.as_ref().unwrap()
        };
        let kv_dst = kv_state.sub_offset(slot * proj_dim, proj_dim);
        let score_dst = score_state.sub_offset(slot * proj_dim, proj_dim);
        gpu.memcpy_dtod_auto(&kv_dst.buf, &kv_buf.buf, proj_dim * 4)
            .map_err(|e| format!("comp kv-store l{layer_idx}: {e:?}"))?;
        gpu.memcpy_dtod_auto(&score_dst.buf, &score_buf.buf, proj_dim * 4)
            .map_err(|e| format!("comp score-store l{layer_idx}: {e:?}"))?;
    }

    // 4. Compression every `ratio` steps.
    let should_compress = (pos + 1) % ratio == 0;
    if !should_compress { return Ok(()); }

    let compressed_slot = pos / ratio;
    if compressed_slot >= max_compressed {
        // Cache full; skip — TODO: ring or panic when long context exceeds cap.
        return Ok(());
    }

    let l_state = &state._indexer[layer_idx];
    let kv_state = if is_indexer {
        l_state.indexer_kv_state.as_ref().unwrap()
    } else {
        l_state.main_kv_state.as_ref().unwrap()
    };
    let score_state = if is_indexer {
        l_state.indexer_score_state.as_ref().unwrap()
    } else {
        l_state.main_score_state.as_ref().unwrap()
    };
    let kv_cache = if is_indexer {
        l_state.indexer_kv_cache.as_ref().unwrap()
    } else {
        l_state.main_kv_cache.as_ref().unwrap()
    };

    let kv_cache_slot = kv_cache.sub_offset(compressed_slot * head_dim, head_dim);

    if overlap {
        let concat_kv = l_state.comp_concat_kv.as_ref().unwrap();
        let concat_score = l_state.comp_concat_score.as_ref().unwrap();

        // Build concat views from [2*ratio, 2*head_dim] state → [2*ratio, head_dim].
        gpu.compressor_overlap_concat_f32(kv_state, concat_kv, ratio as i32, head_dim as i32)
            .map_err(|e| format!("comp concat_kv l{layer_idx}: {e:?}"))?;
        gpu.compressor_overlap_concat_f32(score_state, concat_score, ratio as i32, head_dim as i32)
            .map_err(|e| format!("comp concat_score l{layer_idx}: {e:?}"))?;

        // Pool with softmax weights → [head_dim] at kv_cache slot.
        gpu.compressor_softmax_pool_f32(concat_kv, concat_score, &kv_cache_slot,
            (2 * ratio) as i32, head_dim as i32)
            .map_err(|e| format!("comp pool l{layer_idx}: {e:?}"))?;
    } else {
        // overlap=false (ratio=128): state IS already [ratio, head_dim] since
        // coff=1 → proj_dim=head_dim. Pool directly.
        gpu.compressor_softmax_pool_f32(kv_state, score_state, &kv_cache_slot,
            ratio as i32, head_dim as i32)
            .map_err(|e| format!("comp pool no-overlap l{layer_idx}: {e:?}"))?;
    }

    // RMSNorm in place on the compressed kv_cache slot.
    gpu.rmsnorm_f32(&kv_cache_slot, norm, &kv_cache_slot, cfg.rms_norm_eps)
        .map_err(|e| format!("comp rmsnorm l{layer_idx}: {e:?}"))?;

    // Tail RoPE on the compressed entry.
    //
    // Indexer compressor: plain rope_tail_interleaved with
    // compress_rope_theta=160000 at start-of-window position. Q used in
    // indexer scoring also uses plain rope_tail_interleaved with the
    // same theta, so Q·K is consistent in indexer scoring.
    //
    // Main compressor: YaRN-aware tail RoPE so the K-space matches Q
    // (which has YaRN tail-RoPE applied for compressed layers via
    // apply_tail_rope). Without this, mixed attention computes Q·K with
    // Q rotated at absolute pos but K unrotated — breaking the RoPE
    // relative-position invariant. Long-context wins (ctx=2048: 14.12 →
    // 8.30 ppl, ctx=1024: 10.38 → 8.76 ppl) outweigh the modest
    // short-context regression (ctx=128: 14.69 → 16.83 ppl).
    // Env opt-out: HIPFIRE_V4F_NO_MAIN_ROPE=1.
    // Per-layer compressor rope pos comes from the pre-computed pos_array.
    // Slot 1 = main_comp_rope_pos (mid-of-window by default; respects the
    // HIPFIRE_V4F_COMP_ROPE_POS env var read once in precompute_positions).
    // Slot 2 = indexer_comp_rope_pos (always start-of-window).
    let slot = if is_indexer { 2 } else { 1 };
    let pos_slice = pos_slot(state, layer_idx, slot)?;
    // Keep state.comp_pos_buf populated for any external reader; the value
    // here is the same slice we're about to pass into the rope kernels.
    state.comp_pos_buf = Some(pos_slice);
    let pos_buf = state.comp_pos_buf.as_ref().unwrap();
    let _ = (position, ratio); // values now consumed in precompute_positions

    if is_indexer {
        gpu.rope_tail_interleaved(
            &kv_cache_slot, &kv_cache_slot, pos_buf,
            1, 0,
            head_dim as i32,
            cfg.qk_rope_head_dim as i32,
            cfg.compress_rope_theta,
        ).map_err(|e| format!("comp rope l{layer_idx}: {e:?}"))?;
    } else if std::env::var("HIPFIRE_V4F_NO_MAIN_ROPE").ok().as_deref() != Some("1") {
        // YaRN-aware tail RoPE on main compressor (single-tensor via
        // n_heads_q=1, n_heads_k=0) — matches Q's apply_tail_rope.
        let (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high) =
            layer_rope_params(cfg, layer.compress_ratio);
        gpu.rope_tail_yarn_interleaved(
            &kv_cache_slot, &kv_cache_slot, pos_buf,
            1, 0,
            head_dim as i32,
            cfg.qk_rope_head_dim as i32,
            freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high,
            /*inverse=*/0,
        ).map_err(|e| format!("comp main rope l{layer_idx}: {e:?}"))?;
    }

    // State shift for overlap: kv_state[:ratio] = kv_state[ratio:].
    if overlap {
        let shift_bytes = ratio * proj_dim * 4;
        let src_view = kv_state.sub_offset(ratio * proj_dim, ratio * proj_dim);
        let dst_view = kv_state.sub_offset(0, ratio * proj_dim);
        gpu.memcpy_dtod_auto(&dst_view.buf, &src_view.buf, shift_bytes)
            .map_err(|e| format!("comp kv_state shift l{layer_idx}: {e:?}"))?;

        let src_view = score_state.sub_offset(ratio * proj_dim, ratio * proj_dim);
        let dst_view = score_state.sub_offset(0, ratio * proj_dim);
        gpu.memcpy_dtod_auto(&dst_view.buf, &src_view.buf, shift_bytes)
            .map_err(|e| format!("comp score_state shift l{layer_idx}: {e:?}"))?;
    }

    Ok(())
}

/// Batched compressor commit + compress for a whole chunk of B
/// positions in a single layer (Phase A, 2026-05-20).
///
/// Replaces the per-batch-position loop of `compressor_forward_prebatched`
/// when `start_pos % R == 0` (aligned chunks). For ratio=4 layers
/// at B=64, this collapses ~256 launches/layer (64 × 2 ring writes
/// + 16 × 6 compress-event kernels) into ~6 batched launches:
///   - 1× compressor_compress_aligned_batched_f32
///   - 1× rmsnorm_batched (on N_events × head_dim)
///   - 1× rope_tail_(yarn_)interleaved_batched
///   - 1× memcpy_dtod to update ring state for next chunk
///
/// For the no-event case (B < R, e.g. ratio=128 layers at B=64):
///   - 1× compressor_ring_write_batched_f32
///
/// Bisect at B=1 must remain byte-eq vs the per-position path.
#[allow(dead_code, clippy::too_many_arguments)]
fn compressor_forward_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    layer_idx: usize,
    start_pos: u32,
    batch_size: usize,
    is_indexer: bool,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let ratio = layer.compress_ratio as usize;
    if ratio == 0 { return Ok(()); }
    if is_indexer && ratio != 4 { return Ok(()); }

    let overlap = ratio == 4;
    let coff: usize = if overlap { 2 } else { 1 };
    let head_dim = if is_indexer { cfg.index_head_dim } else { cfg.head_dim };
    let proj_dim = coff * head_dim;
    let state_rows = coff * ratio;

    let max_compressed: usize = std::env::var("HIPFIRE_V4F_MAX_COMPRESS_POS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2048);

    // Lazy-alloc state buffers (mirror compressor_forward_impl exactly).
    {
        let l_state = &mut state._indexer[layer_idx];
        if is_indexer {
            if l_state.indexer_kv_state.is_none() {
                l_state.indexer_kv_state = Some(gpu.zeros(&[state_rows, proj_dim], DType::F32)
                    .map_err(|e| format!("alloc idx kv_state l{layer_idx}: {e:?}"))?);
            }
            if l_state.indexer_score_state.is_none() {
                l_state.indexer_score_state = Some(gpu.zeros(&[state_rows, proj_dim], DType::F32)
                    .map_err(|e| format!("alloc idx score_state l{layer_idx}: {e:?}"))?);
            }
            if l_state.indexer_kv_cache.is_none() {
                l_state.indexer_kv_cache = Some(gpu.zeros(&[max_compressed, head_dim], DType::F32)
                    .map_err(|e| format!("alloc idx kv_cache l{layer_idx}: {e:?}"))?);
            }
        } else {
            if l_state.main_kv_state.is_none() {
                l_state.main_kv_state = Some(gpu.zeros(&[state_rows, proj_dim], DType::F32)
                    .map_err(|e| format!("alloc main kv_state l{layer_idx}: {e:?}"))?);
            }
            if l_state.main_score_state.is_none() {
                l_state.main_score_state = Some(gpu.zeros(&[state_rows, proj_dim], DType::F32)
                    .map_err(|e| format!("alloc main score_state l{layer_idx}: {e:?}"))?);
            }
            if l_state.main_kv_cache.is_none() {
                l_state.main_kv_cache = Some(gpu.zeros(&[max_compressed, head_dim], DType::F32)
                    .map_err(|e| format!("alloc main kv_cache l{layer_idx}: {e:?}"))?);
            }
        }
    }

    // Select the right wkv/wgate-output buffer + ring state + cache.
    let (kv_batch_full, score_batch_full) = if is_indexer {
        (&pbs.comp_idx_kv_batch, &pbs.comp_idx_score_batch)
    } else {
        (&pbs.comp_main_kv_batch, &pbs.comp_main_score_batch)
    };
    let norm = if is_indexer {
        layer.indexer_compressor_norm.as_ref()
            .ok_or_else(|| format!("idx_comp_norm l{layer_idx}"))?
    } else {
        layer.compressor_norm.as_ref()
            .ok_or_else(|| format!("comp_norm l{layer_idx}"))?
    };

    let slot_base = (start_pos as usize) % ratio;
    // first chunk position whose absolute (p+1) % R == 0:
    // first_event_chunk_pos = R - 1 - slot_base.
    let first_event_chunk_pos = if slot_base == 0 {
        ratio - 1
    } else {
        ratio - slot_base - 1
    };
    let n_events = if first_event_chunk_pos < batch_size {
        (batch_size - first_event_chunk_pos + ratio - 1) / ratio
    } else {
        0
    };

    let aligned = slot_base == 0;
    let compressed_slot_base = (start_pos as usize) / ratio;

    // Check kv_cache capacity for this chunk's events.
    let n_events_capped = if compressed_slot_base + n_events > max_compressed {
        max_compressed.saturating_sub(compressed_slot_base)
    } else {
        n_events
    };

    // ALIGNED PATH: B*R-aligned chunk start, do batched compress.
    if aligned && n_events_capped > 0 {
        let kv_state = if is_indexer {
            state._indexer[layer_idx].indexer_kv_state.as_ref().unwrap().clone()
        } else {
            state._indexer[layer_idx].main_kv_state.as_ref().unwrap().clone()
        };
        let score_state = if is_indexer {
            state._indexer[layer_idx].indexer_score_state.as_ref().unwrap().clone()
        } else {
            state._indexer[layer_idx].main_score_state.as_ref().unwrap().clone()
        };
        let kv_cache = if is_indexer {
            state._indexer[layer_idx].indexer_kv_cache.as_ref().unwrap().clone()
        } else {
            state._indexer[layer_idx].main_kv_cache.as_ref().unwrap().clone()
        };

        // `prev_kv` / `prev_score` for event 0 = first R rows of ring state.
        // For overlap=1: ring rows 0..R hold the prior chunk's last NEW window
        //   (FIRST half is the OLD-contribution; SECOND half unused).
        // For chunk 0 (start_pos=0): ring state is zeros — correct: OLD == 0.
        let prev_kv = kv_state.sub_offset(0, ratio * proj_dim);
        let prev_score = score_state.sub_offset(0, ratio * proj_dim);

        let kv_cache_out =
            kv_cache.sub_offset(compressed_slot_base * head_dim,
                                n_events_capped * head_dim);

        gpu.compressor_compress_aligned_batched_f32(
            &prev_kv, &prev_score,
            kv_batch_full, score_batch_full,
            &kv_cache_out,
            ratio as i32, head_dim as i32,
            n_events_capped as i32,
            if overlap { 1 } else { 0 },
            batch_size as i32,
        ).map_err(|e| format!("compressor_compress_aligned_batched l{layer_idx}: {e:?}"))?;

        // RMSNorm batched over n_events × head_dim.
        gpu.rmsnorm_batched(
            &kv_cache_out, norm, &kv_cache_out,
            n_events_capped, head_dim, cfg.rms_norm_eps,
        ).map_err(|e| format!("comp rmsnorm batched l{layer_idx}: {e:?}"))?;

        // Tail RoPE batched. Per event we want a per-event position.
        // Build the position array on host and upload once.
        let rope_pos_mode = std::env::var("HIPFIRE_V4F_COMP_ROPE_POS")
            .ok().unwrap_or_else(|| "mid".to_string());
        let positions_host: Vec<i32> = (0..n_events_capped).map(|k| {
            let absolute_event_pos = first_event_chunk_pos + k * ratio + (start_pos as usize);
            if is_indexer {
                // Indexer always uses start-of-window.
                (absolute_event_pos / ratio * ratio) as i32
            } else {
                match rope_pos_mode.as_str() {
                    "end"   => absolute_event_pos as i32,
                    "start" => (absolute_event_pos / ratio * ratio) as i32,
                    _       => ((absolute_event_pos / ratio * ratio) + ratio / 2) as i32,
                }
            }
        }).collect();
        // Use the existing pbs.positions field as scratch (it's [max_batch] F32).
        // We need at least n_events_capped slots. n_events_capped <= max_batch
        // because each event consumes R positions of input. Safe.
        let pos_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(positions_host.as_ptr() as *const u8,
                                       n_events_capped * 4)
        };
        gpu.memcpy_htod_auto(&pbs.comp_positions.buf, pos_bytes)
            .map_err(|e| format!("htod comp positions l{layer_idx}: {e:?}"))?;

        let no_main_rope = std::env::var("HIPFIRE_V4F_NO_MAIN_ROPE")
            .ok().as_deref() == Some("1");
        if is_indexer {
            gpu.rope_tail_interleaved_batched(
                &kv_cache_out, &kv_cache_out, &pbs.comp_positions,
                1, 0, head_dim as i32, cfg.qk_rope_head_dim as i32,
                cfg.compress_rope_theta, n_events_capped as i32,
            ).map_err(|e| format!("comp idx rope batched l{layer_idx}: {e:?}"))?;
        } else if !no_main_rope {
            let (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high) =
                layer_rope_params(cfg, layer.compress_ratio);
            gpu.rope_tail_yarn_interleaved_batched(
                &kv_cache_out, &kv_cache_out, &pbs.comp_positions,
                1, 0, head_dim as i32, cfg.qk_rope_head_dim as i32,
                freq_base, freq_scale, ext_factor, attn_factor,
                corr_low, corr_high, /*inverse=*/0,
                n_events_capped as i32,
            ).map_err(|e| format!("comp main rope batched l{layer_idx}: {e:?}"))?;
        }

        // Update ring state for next chunk: kv_state[0..R] ← last NEW window's
        // positions from kv_batch_full. For overlap=1 the last NEW window is
        // chunk positions [(n_events - 1) * R + first_event_chunk_pos - R + 1
        // .. n_events * R + first_event_chunk_pos]. With aligned (slot_base=0,
        // first_event_chunk_pos = R-1), that simplifies to
        // chunk positions [(n_events - 1) * R .. n_events * R - 1].
        //
        // For overlap=0 (ratio=128), no shift-state needed for the next chunk:
        // the ring still holds in-progress NEW positions (which the no-event
        // path will scatter). But here n_events > 0 only happens for
        // overlap=1 at our typical B=64 ratio=4 case.
        if overlap {
            let last_new_start_b = (n_events_capped - 1) * ratio;
            // Source slice: kv_batch_full[last_new_start_b..last_new_start_b + R]
            let src_kv = kv_batch_full
                .sub_offset(last_new_start_b * proj_dim, ratio * proj_dim);
            let src_score = score_batch_full
                .sub_offset(last_new_start_b * proj_dim, ratio * proj_dim);
            let dst_kv = kv_state.sub_offset(0, ratio * proj_dim);
            let dst_score = score_state.sub_offset(0, ratio * proj_dim);
            let bytes = ratio * proj_dim * 4;
            gpu.memcpy_dtod_auto(&dst_kv.buf, &src_kv.buf, bytes)
                .map_err(|e| format!("comp state update kv l{layer_idx}: {e:?}"))?;
            gpu.memcpy_dtod_auto(&dst_score.buf, &src_score.buf, bytes)
                .map_err(|e| format!("comp state update score l{layer_idx}: {e:?}"))?;
        }

        return Ok(());
    }

    // NO-EVENT PATH (n_events == 0): just scatter all B positions into the
    // ring state for the next chunk to pick up.
    // Also covers the non-aligned case as a safe fallback for now.
    if !aligned || n_events_capped == 0 {
        let kv_state = if is_indexer {
            state._indexer[layer_idx].indexer_kv_state.as_ref().unwrap()
        } else {
            state._indexer[layer_idx].main_kv_state.as_ref().unwrap()
        };
        let score_state = if is_indexer {
            state._indexer[layer_idx].indexer_score_state.as_ref().unwrap()
        } else {
            state._indexer[layer_idx].main_score_state.as_ref().unwrap()
        };

        gpu.compressor_ring_write_batched_f32(
            kv_batch_full, score_batch_full,
            kv_state, score_state,
            batch_size as i32, proj_dim as i32, ratio as i32,
            slot_base as i32, if overlap { 1 } else { 0 },
        ).map_err(|e| format!("comp ring write batched l{layer_idx}: {e:?}"))?;

        // If aligned but n_events==0 (impossible by construction), or
        // non-aligned (we should add per-position compress-event handling
        // for any events that DO fire in this chunk). For our V4F bench
        // start_pos is always a multiple of B which is a multiple of 4,
        // so this path is hit only for ratio=128 layers at B<128. No
        // compress events to handle.
        if !aligned && n_events_capped > 0 {
            return Err(format!(
                "compressor_forward_batched: non-aligned chunks with compress events \
                 not yet supported (l{layer_idx}, start_pos={start_pos}, B={batch_size}, ratio={ratio})"
            ));
        }
    }

    Ok(())
}

/// V4F indexer scoring + top-K selection (phase 4b).
///
/// Run after `compressor_forward(is_indexer=true)` for layers with
/// `compress_ratio == 4`. Produces `state._indexer[l].topk_idx_indices`,
/// the indices into `indexer_kv_cache` that the modified main attention
/// (phase 5) will gather K/V from.
///
/// Pipeline:
///   q_idx     = indexer_wq_b @ q_lat_rot                  → [H, D]
///   tail-rope on q_idx (compress_rope_theta, current pos) → [H, D]
///   idx_w     = indexer_weights_proj @ state.tmp          → [H]
///   scores[n] = sum_h relu(q_idx[h] · K_cache[n]) * idx_w[h]
///   topk      = top-K(scores) — combined, not per-head
///
/// Returns the actual number of compressed slots scored (0 means no
/// scoring possible because the cache is still empty at this pos).
#[allow(dead_code, clippy::too_many_arguments)]
fn indexer_forward(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    position: u32,
) -> Result<usize, String> {
    let layer = weights.resolve_layer(layer_idx);
    if layer.compress_ratio != 4 { return Ok(0); }

    let h = cfg.index_n_heads;
    let d = cfg.index_head_dim;
    let k = cfg.index_topk;
    let pos = position as usize;
    let ratio = 4usize;

    // Compressed-slot count = number of writes already committed.
    // Writes happen when `(pos+1) % ratio == 0`. Just-finished pos:
    //   n_filled = (pos + 1) / ratio  (integer)
    let n_filled = (pos + 1) / ratio;
    if n_filled == 0 { return Ok(0); }
    let max_compressed: usize = std::env::var("HIPFIRE_V4F_MAX_COMPRESS_POS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let n = n_filled.min(max_compressed);

    let wq_b = layer.indexer_wq_b.as_ref()
        .ok_or_else(|| format!("idx_wq_b l{layer_idx}"))?;
    let weights_proj = layer.indexer_weights_proj.as_ref()
        .ok_or_else(|| format!("idx_weights_proj l{layer_idx}"))?;

    // Lazy-alloc scratch on this layer's indexer state.
    {
        let l_state = &mut state._indexer[layer_idx];
        if l_state.q_idx.is_none() {
            l_state.q_idx = Some(gpu.alloc_tensor(&[h, d], DType::F32)
                .map_err(|e| format!("alloc q_idx l{layer_idx}: {e:?}"))?);
        }
        if l_state.idx_weights.is_none() {
            l_state.idx_weights = Some(gpu.alloc_tensor(&[h], DType::F32)
                .map_err(|e| format!("alloc idx_weights l{layer_idx}: {e:?}"))?);
        }
        if l_state.index_score.is_none() {
            l_state.index_score = Some(gpu.alloc_tensor(&[max_compressed], DType::F32)
                .map_err(|e| format!("alloc index_score l{layer_idx}: {e:?}"))?);
        }
        if l_state.topk_idx_indices.is_none() {
            l_state.topk_idx_indices = Some(gpu.alloc_tensor(&[k], DType::F32)
                .map_err(|e| format!("alloc topk_idx l{layer_idx}: {e:?}"))?);
        }
    }

    // 1. q_idx = wq_b @ q_lat_rot   (MQ4 prerotated GEMV: M = H*D, K = q_lora_rank)
    let q_lat = state.q_lat.as_ref()
        .ok_or_else(|| "indexer: q_lat not allocated".to_string())?;
    let q_lat_rot = state.q_lat_rot.as_ref()
        .ok_or_else(|| "indexer: q_lat_rot not allocated".to_string())?;
    let q_idx = state._indexer[layer_idx].q_idx.as_ref().unwrap();
    gemv_auto(gpu, wq_b, q_lat_rot, q_lat, q_idx, h * d, cfg.q_lora_rank)?;

    // 2. Tail RoPE on q_idx with compress_rope_theta (matching is_indexer=true
    //    K-side compressor's RoPE). Use main `pos_buf` (already holds current
    //    position from apply_tail_rope). qk_rope_head_dim applies on each head.
    let pos_buf = state.pos_buf.as_ref()
        .ok_or_else(|| "indexer: pos_buf missing".to_string())?;
    gpu.rope_tail_interleaved(
        q_idx, q_idx, pos_buf,
        h as i32, 0,
        d as i32,
        cfg.qk_rope_head_dim as i32,
        cfg.compress_rope_theta,
    ).map_err(|e| format!("idx rope l{layer_idx}: {e:?}"))?;

    // 3. idx_w = weights_proj @ state.tmp  → [H]
    let tmp = state.tmp.as_ref()
        .ok_or_else(|| "indexer: state.tmp missing".to_string())?;
    let tmp_plain = state.tmp_plain.as_ref()
        .ok_or_else(|| "indexer: tmp_plain missing".to_string())?;
    let idx_w = state._indexer[layer_idx].idx_weights.as_ref().unwrap();
    gemv_auto(gpu, weights_proj, tmp, tmp_plain, idx_w, h, cfg.hidden_size)?;

    // 4. Score: combined relu-weighted dot products.
    let kv_cache = state._indexer[layer_idx].indexer_kv_cache.as_ref()
        .ok_or_else(|| "indexer: kv_cache missing".to_string())?;
    // Sub-view K_cache to just the filled slots.
    let k_cache_view = kv_cache.sub_offset(0, n * d);
    let scores = state._indexer[layer_idx].index_score.as_ref().unwrap();
    let scores_view = scores.sub_offset(0, n);
    gpu.indexer_relu_score_f32(q_idx, &k_cache_view, idx_w, &scores_view,
        h as i32, d as i32, n as i32)
        .map_err(|e| format!("idx score l{layer_idx}: {e:?}"))?;

    // 5. Top-K: combined scores [N], single "head".
    let topk = state._indexer[layer_idx].topk_idx_indices.as_ref().unwrap();
    let k_take = k.min(n);
    gpu.indexer_top_k(&scores_view, topk,
        /*n_idx_heads=*/1, n as i32, k_take as i32)
        .map_err(|e| format!("idx top_k l{layer_idx}: {e:?}"))?;

    Ok(n)
}

/// Single-token decode step. Takes the token id of the previous
/// position, returns the logits over `vocab_size`.
///
/// Caller is responsible for sampler integration and KV-state
/// advancement.
#[allow(unused_variables, dead_code)]
pub fn decode_step(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    // HIP-graphs prerequisite: lift the ~130 per-token pos_buf
    // `memcpy_htod` calls out of the per-layer code into a single
    // bulk write at decode-step entry. Per-layer kernels then read
    // their slot via `pos_slot(state, layer_idx, slot)`.
    precompute_positions(cfg, state, gpu, position)?;

    // 1. Token embedding → initial residual streams.
    //    V4F uses `hc_mult = 4` parallel streams. Init pattern is
    //    [embed, 0, 0, 0] (paper-specified; verify against the V4F
    //    reference code before optimising).
    init_residual_streams(cfg, weights, state, gpu, token_id)?;

    // Optional early-stop for bisection: env HIPFIRE_V4F_FORWARD_LAYER_END=N
    // halts after layer N-1 (exclusive bound) — leaves residual_streams in
    // their just-after-layer-(N-1) state for cross-path comparison.
    let layer_end: usize = std::env::var("HIPFIRE_V4F_FORWARD_LAYER_END")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(cfg.num_hidden_layers);

    // 2. Per-layer forward.
    for layer_idx in 0..cfg.num_hidden_layers.min(layer_end) {
        let layer = weights.resolve_layer(layer_idx);
        let l_state = &mut state._indexer[layer_idx];
        let l_attn  = &mut state._attention[layer_idx];

        // ── 2a. Attention block ───────────────────────────────────────
        //
        // mHC pre-step + full mHC mix (paper-faithful) — DISABLED.
        // Even with proper sigmoid/exp+Sinkhorn/2σ/input-mapping
        // implementation, 43 layers cumulative additions overflow f32
        // because we don't apply the small-init learnable α scalars
        // (hc_*_scale [3] in the paper, initialised to small values).
        // Wire those into hc_compute_control as `α · (X · W) + base`
        // and retry. For now: pipeline runs HC-disabled producing
        // bounded but architecturally-trivial logits.
        // Real mHC with corrected kernels (F32 throughout for residuals).
        mhc_pre(cfg, weights, state, gpu, layer_idx, /*is_attn=*/true)?;
        q_lora(cfg, weights, state, gpu, layer_idx)?;

        // (Q-LoRA call moved above into the fused RMSNorm + GEMV step.)

        // iii. Joint KV: wkv @ tmp → kv [head_dim = 512] (tied K=V).
        kv_joint(cfg, weights, state, gpu, layer_idx)?;

        // iv. Tail-only RoPE on Q and KV.
        //     Apply rotation on last `qk_rope_head_dim = 64` of each
        //     head's 512 dims.
        //     SWA ring write deferred (needs swa state alloc per layer).
        apply_tail_rope(cfg, weights, state, gpu, position, layer_idx)?;

        // iv. Indexer path (only when compress_ratio > 0):
        //     a. Compressor: x @ compressor.wkv → idx_qk
        //        x @ compressor.wgate → idx_v (per V4F structure)
        //        x normalised by compressor.norm
        //     b. Apply compress_rope on idx_q (freq_base = compress_rope_theta = 160000)
        //     c. If position % compress_ratio == 0: append idx_k to k_idx_compressed cache
        //     d. `gpu.indexer_compressed_k_score(q_idx, k_idx_cache, scores, ...)`
        //     e. `gpu.indexer_top_k(scores, top_indices, ..., k = index_topk = 512)`
        //     f. dedup top_indices across heads (UNION strategy per paper)
        //     g. `gpu.indexer_kv_gather(k_main_cache, v_main_cache, unique_indices, ...)`
        //
        //     When compress_ratio == 0: skip; attention reads SWA only.
        //
        // Phase 3c: run main + indexer compressors. Both consume the FWHT-
        // rotated post-attn_norm input held in `state.tmp` (populated by
        // q_lora's fused_rmsnorm_rotate_mq step above). Gated on
        // HIPFIRE_V4F_RUN_COMPRESSOR (default off until phases 4-5 land,
        // since the cache fill alone does not affect attention output yet
        // but does consume VRAM + GEMV cycles per layer per token).
        // V4F compressor + indexer (antirez-faithful default behavior):
        // Always run for ratio>0 layers (no env gate). Antirez ds4 runs
        // compressor unconditionally for compressed layers and the
        // indexer for ratio==4 layers (ds4.c:7505-7555).
        // Env opt-out: HIPFIRE_V4F_NO_COMPRESSOR=1 for diagnosis.
        if layer.compress_ratio > 0
            && std::env::var("HIPFIRE_V4F_NO_COMPRESSOR").ok().as_deref() != Some("1")
        {
            let tmp_view = {
                let t = state.tmp.as_ref().unwrap();
                t.sub_offset(0, t.numel())
            };
            compressor_forward(cfg, weights, state, gpu, layer_idx,
                &tmp_view, position, /*is_indexer=*/false)?;
            if layer.compress_ratio == 4 {
                compressor_forward(cfg, weights, state, gpu, layer_idx,
                    &tmp_view, position, /*is_indexer=*/true)?;
                let _n = indexer_forward(cfg, weights, state, gpu, layer_idx, position)?;
            }
        }

        // v + vi. Main attention + O-LoRA — STUB.
        attn_stub(cfg, weights, state, gpu, layer_idx)?;

        hc_attn_mix(cfg, weights, state, gpu, layer_idx)?;

        // Bisection break: stop after hc_attn_mix (= attn-side residual update
        // applied, FFN side not yet). Useful for isolating divergence in the
        // FFN-side stages from divergence in the attention-side stages.
        if layer_idx + 1 == cfg.num_hidden_layers.min(layer_end)
            && std::env::var("HIPFIRE_V4F_BISECT_BREAK").ok().as_deref()
                == Some("after_attn_mix")
        {
            return Ok(Vec::new());
        }

        // ── 2b. FFN block ─────────────────────────────────────────────
        mhc_pre(cfg, weights, state, gpu, layer_idx, /*is_attn=*/false)?;
        if layer_idx + 1 == cfg.num_hidden_layers.min(layer_end)
            && std::env::var("HIPFIRE_V4F_BISECT_BREAK").ok().as_deref()
                == Some("after_mhc_pre_ffn")
        {
            return Ok(Vec::new());
        }
        if std::env::var("HIPFIRE_V4F_SKIP_FFN").ok().as_deref() != Some("1") {
            ffn_stub(cfg, weights, state, gpu, layer_idx)?;
            if layer_idx < cfg.num_hash_layers {
                ffn_hash_routed(cfg, weights, state, gpu, layer_idx, token_id)?;
            } else {
                ffn_routed(cfg, weights, state, gpu, layer_idx)?;
            }
        } else {
            // Diagnostic: zero ffn_out to isolate attn contribution to growth.
            if state.ffn_out.is_none() {
                state.ffn_out = Some(gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                    .map_err(|e| format!("alloc ffn_out: {e:?}"))?);
            }
            let ffn_out = state.ffn_out.as_ref().unwrap();
            gpu.hip.memset(&ffn_out.buf, 0, ffn_out.byte_size())
                .map_err(|e| format!("memset ffn_out: {e:?}"))?;
        }
        hc_ffn_mix(cfg, weights, state, gpu, layer_idx)?;

        // Optional magnitude diagnostic, gated on HIPFIRE_V4F_DUMP_MAG.
        if std::env::var("HIPFIRE_V4F_DUMP_MAG").ok().as_deref() == Some("1") {
            let streams = state.residual_streams.as_ref().unwrap();
            let attn_out = state.attn_out.as_ref().unwrap();
            let ffn_out = state.ffn_out.as_ref().unwrap();
            let host = gpu.download_f32(streams).unwrap_or_default();
            let host_attn = gpu.download_f32(attn_out).unwrap_or_default();
            let host_ffn = gpu.download_f32(ffn_out).unwrap_or_default();
            let rms = |v: &[f32]| (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
            let max = |v: &[f32]| v.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
            eprintln!("[layer {layer_idx:>2}] streams rms={:.4} max={:.4} | attn_out rms={:.4} max={:.4} | ffn_out rms={:.4} max={:.4}",
                rms(&host), max(&host), rms(&host_attn), max(&host_attn), rms(&host_ffn), max(&host_ffn));
        }

        // Phase 5 debug: dump SWA K vs main_kv_cache K magnitudes for the
        // first indexer-active layer. Helps next session diagnose the
        // K-space mismatch causing phase 5 regression at ctx>128.
        if layer.compress_ratio == 4
            && std::env::var("HIPFIRE_V4F_DUMP_PHASE5_K").ok().as_deref() == Some("1")
            && state._attention[layer_idx].swa_k.is_some()
            && state._indexer[layer_idx].main_kv_cache.is_some()
        {
            let swa_k = state._attention[layer_idx].swa_k.as_ref().unwrap();
            let main_kv = state._indexer[layer_idx].main_kv_cache.as_ref().unwrap();
            let swa_host = gpu.download_f32(swa_k).unwrap_or_default();
            let main_host = gpu.download_f32(main_kv).unwrap_or_default();
            let rms = |v: &[f32]| {
                let n = v.iter().filter(|x| **x != 0.0).count().max(1);
                (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / n as f64).sqrt()
            };
            let max = |v: &[f32]| v.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
            let n_compressed = (state.n_tokens as usize + 1) / 4;
            let main_view = &main_host[0..n_compressed.max(1) * cfg.head_dim];
            eprintln!("[L{layer_idx:>2} K-spaces] swa_k rms={:.4} max={:.4} | main_kv_cache[0..{}] rms={:.4} max={:.4}",
                rms(&swa_host), max(&swa_host), n_compressed, rms(main_view), max(main_view));
        }
    }

    // 3. Final norm + LM head. The head-HC mix INSIDE final_norm_and_head
    //    now ALSO captures head_hc_out into state.mtp_last_hidden — that's
    //    the value V4F MTP expects as h_n (post-head-HC-mix, pre-output-norm).
    //    The previous "capture stream 0 before final_norm_and_head" pattern
    //    was wrong on HC models — MTP saw 1 of 4 streams instead of the
    //    actual hidden the main model uses for its own prediction.
    //    Note: V4F has head-level HC (hc_head_base/fn/scale).
    //    For minimal forward: skip the head-HC mix (TODO: head HC
    //    likely projects 4 streams → 1 then applies head_weight)
    //    and just run final norm + standard lm_head.
    final_norm_and_head(cfg, weights, state, gpu)?;

    // Download logits to host and return.
    let logits = state.logits.as_ref().unwrap();
    let logits_host = gpu.download_f32(logits)
        .map_err(|e| format!("download logits: {e:?}"))?;
    state.n_tokens += 1;
    Ok(logits_host)
}

fn unimplemented_step(name: &str) -> Result<(), String> {
    let _ = name;  // silence unused; kept for stack-trace clarity later.
    Ok(())
}

/// V4F Multi-Token Prediction (MTP) forward step — DeepSeek V3 §4.
///
/// Predicts the **next-next** token given:
///   - `h_n`         : hidden state at absolute position N (the output of
///                     the main forward at that position, before the head)
///   - `next_token`  : the token that was emitted at position N+1
///   - `position`    : absolute position N+1 (used by tail-RoPE)
///
/// Output: logits over the vocab for position N+2.
///
/// Architecture (from `mtp.0.*` weights in V4F-MTP HFQ files):
/// ```text
/// e_norm     = enorm(embed_lookup(next_token))
/// h_norm     = hnorm(h_n)
/// x_in       = e_proj @ e_norm + h_proj @ h_norm         (Q8F16 GEMVs)
/// x_attn     = attention(attn_norm(x_in))   + x_in        (SWA-only — no compressor)
/// x_ffn      = ffn(ffn_norm(x_attn))        + x_attn      (shared + routed MoE)
/// h_n_plus_1 = mtp_final_norm(x_ffn)
/// logits     = shared_head @ h_n_plus_1                   (reuses main lm_head)
/// ```
///
/// The MTP layer has NO compressor and NO indexer (verified against the
/// safetensors tensor table: only standard attn + FFN weights). Its
/// attention block is the SWA-only path (same as a hash-routed main
/// layer's attention).
///
/// **Status**: M1+M2 (weights ingest) are landed; the standard layer
/// block (attn + FFN with MTP weights) is still pending — the existing
/// per-layer helpers (`q_lora`, `kv_joint`, `attn_stub`, ...) all read
/// `weights.layers[layer_idx]` and need refactoring to accept a
/// `&DeepseekV4LayerWeights` parameter so they can run against
/// `weights.mtp_layer`. Filling in that refactor is M3-complete; it
/// will land alongside validation against the new HFQ that contains
/// the MTP layer.
///
/// Until then this function returns a clear error so callers can stub
/// out the spec-decode path without false-positive bring-up.
pub fn mtp_forward(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    h_n: &GpuTensor,
    next_token: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    // ── 0. Validate MTP weights are present ────────────────────────────
    let mtp = weights.mtp_layer.as_ref()
        .ok_or_else(|| "mtp_forward: weights.mtp_layer is None — \
            re-quantize V4F with --format v4f-q8-mtp to include the \
            mtp.0.* tensors, then HIPFIRE_V4F_LOAD_MTP=1 at load time. \
            Existing v4f.mq2lloyd-f16compress.hfq and antirezQ8.hfq were \
            quantized without MTP and cannot run spec-decode.".to_string())?;
    let mtp_enorm  = mtp.mtp_enorm.as_ref().ok_or("mtp_forward: mtp_enorm missing")?;
    let mtp_hnorm  = mtp.mtp_hnorm.as_ref().ok_or("mtp_forward: mtp_hnorm missing")?;
    let mtp_e_proj = mtp.mtp_e_proj.as_ref().ok_or("mtp_forward: mtp_e_proj missing")?;
    let mtp_h_proj = mtp.mtp_h_proj.as_ref().ok_or("mtp_forward: mtp_h_proj missing")?;
    let mtp_final  = mtp.mtp_final_norm.as_ref().ok_or("mtp_forward: mtp_final_norm missing")?;

    // Defensive: step 4 below passes `dummy_rotated` aliasing the OTHER
    // norm scratch (not a real FWHT rotation). That's safe for Q8_0 /
    // F16 / F32 dtypes since gemv_auto reads only x_plain on those
    // paths. For MQ4 (Raw dtype) gemv_auto reads x_rotated → we'd feed
    // garbage and produce silent NaN cascades. Reject upfront with a
    // clear message; if someone wants MQ4 MTP they need to plumb proper
    // rotated buffers through step 4.
    for (name, t) in [("mtp_e_proj", mtp_e_proj), ("mtp_h_proj", mtp_h_proj)] {
        match t.dtype {
            DType::F32 | DType::F16 | DType::Q8_0 => {}
            other => return Err(format!(
                "mtp_forward: {name} dtype {other:?} unsupported — step 4 \
                 only plumbs plain input (no FWHT rotation). Add rotated \
                 buffers or re-quant MTP at Q8F16 / F16.")),
        }
    }

    if h_n.shape != [cfg.hidden_size] && h_n.shape != [1, cfg.hidden_size] {
        return Err(format!(
            "mtp_forward: h_n shape {:?} != [hidden_size={}]",
            h_n.shape, cfg.hidden_size,
        ));
    }
    if cfg.num_nextn_predict_layers == 0 {
        return Err("mtp_forward: cfg.num_nextn_predict_layers == 0; MTP not enabled".to_string());
    }

    let hidden = cfg.hidden_size;
    let hc_mult = cfg.hc_mult;
    // MTP layer occupies the slot just past the main layers; resolve_layer
    // routes the per-layer helpers below to `weights.mtp_layer`.
    let mtp_layer_idx = cfg.num_hidden_layers;

    // ── 1. Lazy state allocation ───────────────────────────────────────
    if state.embed_scratch.is_none() {
        state.embed_scratch = Some(gpu.alloc_tensor(&[hidden], DType::F32)
            .map_err(|e| format!("alloc embed_scratch: {e:?}"))?);
    }
    if state.residual_streams.is_none() {
        let t = gpu.zeros(&[hc_mult, hidden], DType::F32)
            .map_err(|e| format!("alloc residual_streams: {e:?}"))?;
        state.residual_streams = Some(t);
    }
    if state.tmp.is_none() {
        state.tmp = Some(gpu.alloc_tensor(&[hidden], DType::F32)
            .map_err(|e| format!("alloc tmp: {e:?}"))?);
    }
    if state.mtp_e_norm_scratch.is_none() {
        state.mtp_e_norm_scratch = Some(gpu.alloc_tensor(&[hidden], DType::F32)
            .map_err(|e| format!("alloc mtp_e_norm_scratch: {e:?}"))?);
    }
    if state.mtp_h_norm_scratch.is_none() {
        state.mtp_h_norm_scratch = Some(gpu.alloc_tensor(&[hidden], DType::F32)
            .map_err(|e| format!("alloc mtp_h_norm_scratch: {e:?}"))?);
    }
    if state.logits.is_none() {
        state.logits = Some(gpu.alloc_tensor(&[cfg.vocab_size], DType::F32)
            .map_err(|e| format!("alloc logits: {e:?}"))?);
    }

    let token_embd = weights.token_embd.as_ref()
        .ok_or("mtp_forward: token_embd not uploaded")?;
    let head = weights.head.as_ref()
        .ok_or("mtp_forward: head not uploaded")?;

    // ── 2. Embed next_token → embed_scratch [hidden] ───────────────────
    {
        let embed_scratch = state.embed_scratch.as_ref().unwrap();
        gpu.embedding_lookup_q8(token_embd, embed_scratch, next_token, hidden)
            .map_err(|e| format!("mtp embedding_lookup_q8: {e:?}"))?;
    }

    // ── 3. RMSNorm both inputs ─────────────────────────────────────────
    // e_norm = mtp_enorm(embed)  → mtp_e_norm_scratch
    // h_norm = mtp_hnorm(h_n)    → mtp_h_norm_scratch
    {
        let embed_scratch = state.embed_scratch.as_ref().unwrap();
        let e_out = state.mtp_e_norm_scratch.as_ref().unwrap();
        gpu.rmsnorm_f32(embed_scratch, mtp_enorm, e_out, cfg.rms_norm_eps)
            .map_err(|e| format!("mtp rmsnorm_e: {e:?}"))?;
    }
    {
        let h_out = state.mtp_h_norm_scratch.as_ref().unwrap();
        gpu.rmsnorm_f32(h_n, mtp_hnorm, h_out, cfg.rms_norm_eps)
            .map_err(|e| format!("mtp rmsnorm_h: {e:?}"))?;
    }

    // ── 4. x_in = mtp_e_proj @ e_norm + mtp_h_proj @ h_norm ────────────
    // gemv_auto dispatches on the weight's GpuTensor.dtype:
    //   - Q8F16 → gemv_q8_0   (plain input)         ← v4f-q8-mtp uses this
    //   - F16   → gemm_f16_x_f16_wmma at B=1        (plain input)
    //   - MQ4   → gemv_mq4g256_prerotated           (rotated input)
    //
    // For the MQ4 fallback path we'd need a FWHT-rotated copy of the norm
    // outputs; in v4f-q8-mtp these matrices are Q8F16 so the `x_rotated`
    // argument is unused. We pass `mtp_h_norm_scratch` itself as the dummy
    // rotated alias — it's the right size and content doesn't matter for
    // Q8/F16 paths.
    {
        let e_norm = state.mtp_e_norm_scratch.as_ref().unwrap();
        let dummy_rotated = state.mtp_h_norm_scratch.as_ref().unwrap();
        let tmp = state.tmp.as_ref().unwrap();
        gemv_auto(gpu, mtp_e_proj, dummy_rotated, e_norm, tmp, hidden, hidden)?;
    }
    {
        let h_norm = state.mtp_h_norm_scratch.as_ref().unwrap();
        let dummy_rotated = state.mtp_e_norm_scratch.as_ref().unwrap();
        let embed_scratch = state.embed_scratch.as_ref().unwrap();
        // Reuse embed_scratch as the h_proj output buffer — its previous
        // content (raw embed of next_token) is no longer needed.
        gemv_auto(gpu, mtp_h_proj, dummy_rotated, h_norm, embed_scratch, hidden, hidden)?;
        let tmp = state.tmp.as_ref().unwrap();
        gpu.add_inplace_f32(tmp, embed_scratch)
            .map_err(|e| format!("mtp x_in add: {e:?}"))?;
    }

    // ── 5. Broadcast x_in (now in state.tmp) to all hc_mult streams ────
    // Matches `init_residual_streams`' HC init (per antirez ds4
    // `hc_from_plain_embedding`): all streams start equal, not [x, 0,0,0].
    {
        let streams = state.residual_streams.as_ref().unwrap();
        let src = state.tmp.as_ref().unwrap();
        let bytes = hidden * 4;
        for h in 0..hc_mult {
            let dst = streams.sub_offset(h * hidden, hidden);
            gpu.memcpy_dtod_auto(&dst.buf, &src.buf, bytes)
                .map_err(|e| format!("mtp d2d copy stream {h}: {e:?}"))?;
        }
    }

    // ── 6. Standard layer block at layer_idx = num_hidden_layers ───────
    // All per-layer helpers below call `weights.resolve_layer(layer_idx)`
    // internally, which routes to `weights.mtp_layer`. The MTP layer has
    // NO compressor/indexer (compress_ratio = 0 by construction), and is
    // NOT a hash layer (mtp_layer_idx >= num_hash_layers), so we use the
    // standard MoE router (`ffn_routed`) rather than `ffn_hash_routed`.
    mhc_pre(cfg, weights, state, gpu, mtp_layer_idx, /*is_attn=*/true)?;
    q_lora(cfg, weights, state, gpu, mtp_layer_idx)?;
    kv_joint(cfg, weights, state, gpu, mtp_layer_idx)?;
    apply_tail_rope(cfg, weights, state, gpu, position, mtp_layer_idx)?;
    // (No compressor / indexer for MTP — compress_ratio == 0.)
    attn_stub(cfg, weights, state, gpu, mtp_layer_idx)?;
    hc_attn_mix(cfg, weights, state, gpu, mtp_layer_idx)?;
    mhc_pre(cfg, weights, state, gpu, mtp_layer_idx, /*is_attn=*/false)?;
    ffn_stub(cfg, weights, state, gpu, mtp_layer_idx)?;
    ffn_routed(cfg, weights, state, gpu, mtp_layer_idx)?;
    hc_ffn_mix(cfg, weights, state, gpu, mtp_layer_idx)?;

    // ── 7. Capture stream 0 → mtp_last_hidden for chaining ────────────
    {
        if state.mtp_last_hidden.is_none() {
            state.mtp_last_hidden = Some(gpu.alloc_tensor(&[hidden], DType::F32)
                .map_err(|e| format!("alloc mtp_last_hidden: {e:?}"))?);
        }
        let streams = state.residual_streams.as_ref().unwrap();
        let stream0 = streams.sub_offset(0, hidden);
        let dst = state.mtp_last_hidden.as_ref().unwrap();
        gpu.memcpy_dtod_auto(&dst.buf, &stream0.buf, hidden * 4)
            .map_err(|e| format!("capture stream0 → mtp_last_hidden: {e:?}"))?;
    }

    // ── 8. mtp_final_norm + shared lm_head → logits ───────────────────
    // Stream 0 → mtp_final_norm → lm_head. The MTP head-HC tensors
    // (mtp.0.hc_head_*) are intentionally unused here — see memory
    // entry; using them measured WORSE acceptance than skipping.
    if state.final_norm.is_none() {
        state.final_norm = Some(gpu.alloc_tensor(&[hidden], DType::F32)
            .map_err(|e| format!("alloc final_norm: {e:?}"))?);
    }
    if state.final_norm_rot.is_none() {
        state.final_norm_rot = Some(gpu.alloc_tensor(&[hidden], DType::F32)
            .map_err(|e| format!("alloc final_norm_rot: {e:?}"))?);
    }
    let final_norm = state.final_norm.as_ref().unwrap();
    let final_norm_rot = state.final_norm_rot.as_ref().unwrap();
    {
        let streams = state.residual_streams.as_ref().unwrap();
        let stream0 = streams.sub_offset(0, hidden);
        gpu.rmsnorm_f32(&stream0, mtp_final, final_norm, cfg.rms_norm_eps)
            .map_err(|e| format!("mtp final rmsnorm: {e:?}"))?;
        gpu.rotate_x_mq(final_norm, final_norm_rot, hidden)
            .map_err(|e| format!("mtp rotate final: {e:?}"))?;
        let logits = state.logits.as_ref().unwrap();
        gemv_auto(gpu, head, final_norm_rot, final_norm, logits,
            cfg.vocab_size, hidden)?;
    }

    // ── 9. Download logits ─────────────────────────────────────────────
    let logits = state.logits.as_ref().unwrap();
    let logits_host = gpu.download_f32(logits)
        .map_err(|e| format!("mtp download logits: {e:?}"))?;
    Ok(logits_host)
}

/// FFN block (partial — shared expert only; routed experts pending).
///
/// V4F has one shared expert + 256 routed experts (top-6 selected
/// per token). The shared expert is a standard SwiGLU:
///   gate = x @ shared_w1   [moe_intermediate=2048]
///   up   = x @ shared_w3   [moe_intermediate]
///   silu_gated = silu(gate) * up
///   out  = silu_gated @ shared_w2   [hidden]
///
/// Then x_ffn = shared_out + routed_scaling_factor * routed_out.
/// Routed_out is currently 0 (router/expert dispatch pending), so
/// ffn_out = shared_out.
fn ffn_stub(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let ffn_norm  = layer.ffn_norm.as_ref().unwrap();
    let shared_w1 = layer.shared_w1.as_ref().unwrap();
    let shared_w2 = layer.shared_w2.as_ref().unwrap();
    let shared_w3 = layer.shared_w3.as_ref().unwrap();
    let hc_x_in = state.hc_x_in.as_ref().unwrap();

    let im = cfg.moe_intermediate_size;
    if state.ffn_out.is_none() {
        state.ffn_out = Some(gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
            .map_err(|e| format!("alloc ffn_out: {e:?}"))?);
    }
    if state.ffn_x_rot.is_none() {
        state.ffn_x_rot = Some(gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
            .map_err(|e| format!("alloc ffn_x_rot: {e:?}"))?);
    }
    if state.ffn_gate.is_none() {
        state.ffn_gate = Some(gpu.alloc_tensor(&[im], DType::F32)
            .map_err(|e| format!("alloc ffn_gate: {e:?}"))?);
    }
    if state.ffn_up.is_none() {
        state.ffn_up = Some(gpu.alloc_tensor(&[im], DType::F32)
            .map_err(|e| format!("alloc ffn_up: {e:?}"))?);
    }
    if state.ffn_silu_rot.is_none() {
        state.ffn_silu_rot = Some(gpu.alloc_tensor(&[im], DType::F32)
            .map_err(|e| format!("alloc ffn_silu_rot: {e:?}"))?);
    }
    if state.ffn_x_plain.is_none() {
        state.ffn_x_plain = Some(gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
            .map_err(|e| format!("alloc ffn_x_plain: {e:?}"))?);
    }

    let ffn_x_rot = state.ffn_x_rot.as_ref().unwrap();
    let ffn_x_plain = state.ffn_x_plain.as_ref().unwrap();
    let gate = state.ffn_gate.as_ref().unwrap();
    let up   = state.ffn_up.as_ref().unwrap();
    let silu_rot = state.ffn_silu_rot.as_ref().unwrap();
    let ffn_out = state.ffn_out.as_ref().unwrap();

    // 1. Fused RMSNorm + FWHT rotate for the two MQ4 GEMVs.
    gpu.fused_rmsnorm_rotate_mq(hc_x_in, ffn_norm, ffn_x_rot,
        cfg.hidden_size, cfg.rms_norm_eps)
        .map_err(|e| format!("fused_rmsnorm_rotate_mq ffn layer {layer_idx}: {e:?}"))?;
    // 1b. Plain RMSNorm (no FWHT) for F16 non-expert GEMVs.
    gpu.rmsnorm_f32(hc_x_in, ffn_norm, ffn_x_plain, cfg.rms_norm_eps)
        .map_err(|e| format!("rmsnorm_f32 ffn-side plain l{layer_idx}: {e:?}"))?;

    // 2. gate = x @ shared_w1
    gemv_auto(gpu, shared_w1, ffn_x_rot, ffn_x_plain, gate, im, cfg.hidden_size)?;

    // 3. up = x @ shared_w3
    gemv_auto(gpu, shared_w3, ffn_x_rot, ffn_x_plain, up, im, cfg.hidden_size)?;

    // 4. V4F SwiGLU with swiglu_limit clamp (cfg.swiglu_limit = 10.0
    //    on V4F). Same Expert class used for shared and routed in
    //    upstream model.py — both apply this clamp before silu_mul.
    gpu.v4f_silu_mul_clamp_f32(gate, up, gate, cfg.swiglu_limit)
        .map_err(|e| format!("v4f_silu_mul_clamp layer {layer_idx}: {e:?}"))?;

    // 5. FWHT-rotate the silu-gated vector for the down GEMV.
    gpu.rotate_x_mq(gate, silu_rot, im)
        .map_err(|e| format!("rotate_x_mq silu layer {layer_idx}: {e:?}"))?;

    // 6. ffn_out = silu_rot @ shared_w2 (down: [hidden, im])
    // shared_w2: rotated path uses silu_rot (FWHT'd), plain path uses
    // `gate` itself (post-silu_mul, no FWHT).
    gemv_auto(gpu, shared_w2, silu_rot, gate, ffn_out, cfg.hidden_size, im)?;

    Ok(())
}

/// Routed-expert dispatch (V4F top-6 MoE). Accumulates `routed_scaling
/// _factor · Σ_k w_k · expert_{idx_k}(ffn_x_rot)` into `ffn_out`
/// (which already holds the shared-expert output from `ffn_stub`).
///
/// Gated on HIPFIRE_V4F_MOE=1 AND expert blobs present (uploaded via
/// HIPFIRE_V4F_UPLOAD_EXPERTS=1) AND layer is score-routed
/// (layer_idx >= num_hash_layers). Hash-routed layers 0..3 fall back
/// to shared-only (tid2eid lookup table is skipped at quant time).
///
/// Math (per upstream `inference/model.py:Gate.forward` and `Expert.
/// forward`):
///   scores = sqrt(softplus(gate.weight @ x))             [n_exp]
///   indices = topk(scores + bias, k=6)[1]                [k]   ← +bias for selection
///   weights = scores[indices]                            [k]   ← unbiased scores for weights
///   weights /= weights.sum(); weights *= route_scale     [k]
///   for each (idx, w) in (indices, weights):
///     gate_e = w1[idx] @ x                  ← clamp to swiglu_limit (skipped)
///     up_e   = w3[idx] @ x                  ← clamp to ±swiglu_limit (skipped)
///     e_out  = w2[idx] @ (silu(gate_e) * up_e * w)
///     ffn_out += e_out * routed_scaling_factor
fn ffn_routed(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    if std::env::var("HIPFIRE_V4F_MOE").ok().as_deref() != Some("1") {
        return Ok(());
    }
    if layer_idx < cfg.num_hash_layers {
        // Hash routing — tid2eid table skipped at quant time, no expert
        // selection possible. Shared expert alone for these layers.
        return Ok(());
    }
    let layer = weights.resolve_layer(layer_idx);
    if layer.expert_gate_up_blob.is_none() || layer.expert_w2_blob.is_none()
    {
        return Ok(());  // experts not uploaded; nothing to dispatch
    }

    // 1. Run router: compute unbiased scores on-device. V4F's selection
    //    uses BIASED scores while the routing weights use UNBIASED scores
    //    (per upstream model.py: Gate.forward). The GPU top-K kernel
    //    `v4f_moe_topk_bias_aware_f32` handles this two-score semantic in
    //    one launch, eliminating the per-layer D2H/CPU/H2D round-trip
    //    used by the legacy fallback below (HIPFIRE_V4F_CPU_TOPK=1).
    moe_route(cfg, weights, state, gpu, layer_idx)?;

    let k = cfg.num_experts_per_tok;
    let n_exp = cfg.n_routed_experts;
    let im = cfg.moe_intermediate_size;
    let ffn_x_rot = state.ffn_x_rot.as_ref().unwrap();
    let ffn_out = state.ffn_out.as_ref().unwrap();
    let route_scale_override: f32 = std::env::var("HIPFIRE_V4F_ROUTE_SCALE")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2.2);
    let cpu_topk = std::env::var("HIPFIRE_V4F_CPU_TOPK")
        .ok().as_deref() == Some("1");

    // Legacy CPU top-K (kept for parity testing under HIPFIRE_V4F_CPU_TOPK=1).
    let (topk_ids, wts): (Vec<u32>, Vec<f32>) = if cpu_topk {
        let scores_dev = state.router_scores.as_ref().unwrap();
        let scores_host = gpu.download_f32(scores_dev)
            .map_err(|e| format!("d2h scores l{layer_idx}: {e:?}"))?;
        match bias_aware_topk_weights(&scores_host[..n_exp], &layer.gate_bias_host, k)
        {
            Some(x) => x,
            None => return Ok(()),  // degenerate router output (sum <= 0)
        }
    } else {
        (Vec::new(), Vec::new())
    };

    if std::env::var("HIPFIRE_V4F_NO_FUSED_MOE").ok().as_deref() != Some("1")
        && layer.expert_gate_up_blob.is_some()
    {
        // Fused MoE dispatch: 2 indexed kernels (gate_up + down) plus
        // k_top per-expert silu_clamp+rotate. Replaces the per-expert
        // k=0..6 × 3 GEMV loop (18 launches → 14 launches per layer).
        // The bigger win is GPU utilisation: grid Y dim spans all k_top
        // experts so the GEMVs run in parallel rather than serially.
        let k_top = k;
        // Lazy-alloc scratch.
        if state.moe_topk_indices.is_none() {
            state.moe_topk_indices = Some(gpu.alloc_tensor(&[k_top], DType::F32)
                .map_err(|e| format!("alloc moe_topk_indices: {e:?}"))?);
        }
        if state.moe_topk_weights.is_none() {
            state.moe_topk_weights = Some(gpu.alloc_tensor(&[k_top], DType::F32)
                .map_err(|e| format!("alloc moe_topk_weights: {e:?}"))?);
        }
        if state.moe_gate_batch.is_none() {
            state.moe_gate_batch = Some(gpu.alloc_tensor(&[k_top, im], DType::F32)
                .map_err(|e| format!("alloc moe_gate_batch: {e:?}"))?);
        }
        if state.moe_up_batch.is_none() {
            state.moe_up_batch = Some(gpu.alloc_tensor(&[k_top, im], DType::F32)
                .map_err(|e| format!("alloc moe_up_batch: {e:?}"))?);
        }
        if state.moe_rot_batch.is_none() {
            state.moe_rot_batch = Some(gpu.alloc_tensor(&[k_top, im], DType::F32)
                .map_err(|e| format!("alloc moe_rot_batch: {e:?}"))?);
        }
        let topk_idx_dev = state.moe_topk_indices.as_ref().unwrap();
        let topk_w_dev = state.moe_topk_weights.as_ref().unwrap();
        if cpu_topk {
            // Legacy CPU/H2D path. topk_ids and wts populated above.
            let idx_i32: Vec<i32> = topk_ids.iter().map(|&x| x as i32).collect();
            let idx_bytes: Vec<u8> = idx_i32.iter().flat_map(|i| i.to_le_bytes()).collect();
            gpu.memcpy_htod_auto(&topk_idx_dev.buf, &idx_bytes)
                .map_err(|e| format!("htod topk_indices l{layer_idx}: {e:?}"))?;
            let w_scaled: Vec<f32> = wts.iter().map(|&w| w * route_scale_override).collect();
            let w_bytes: Vec<u8> = w_scaled.iter().flat_map(|w| w.to_le_bytes()).collect();
            gpu.memcpy_htod_auto(&topk_w_dev.buf, &w_bytes)
                .map_err(|e| format!("htod topk_weights l{layer_idx}: {e:?}"))?;
        } else {
            // GPU top-K: bias-aware select + normalize + route_scale in one
            // launch, outputs straight into topk_idx_dev / topk_w_dev.
            let scores_dev = state.router_scores.as_ref().unwrap();
            let bias_dev = layer.gate_bias.as_ref()
                .ok_or_else(|| format!("ffn_routed l{layer_idx}: gate_bias missing"))?;
            gpu.v4f_moe_topk_bias_aware_f32(
                scores_dev, bias_dev, topk_idx_dev, topk_w_dev,
                n_exp as i32, k_top as i32, route_scale_override,
            ).map_err(|e| format!("v4f_moe_topk_bias_aware l{layer_idx}: {e:?}"))?;
        }

        let gate_up_ptrs = layer.expert_gate_up_ptrs.as_ref().unwrap();
        let w2_ptrs = layer.expert_w2_ptrs.as_ref().unwrap();
        let gate_batch = state.moe_gate_batch.as_ref().unwrap();
        let up_batch = state.moe_up_batch.as_ref().unwrap();
        let rot_batch = state.moe_rot_batch.as_ref().unwrap();

        // 1. Fused gate_up GEMV: one launch dispatches all k_top experts'
        //    gate and up halves in parallel. M = 2*intermediate; the
        //    kernel splits output rows by r<im → gate, r>=im → up.
        gpu.v4f_gemv_mq2g256_lloyd_moe_gate_up_indexed(
            gate_up_ptrs, topk_idx_dev,
            ffn_x_rot, gate_batch, up_batch,
            2 * im, cfg.hidden_size, k_top,
        ).map_err(|e| format!("fused gate_up l{layer_idx}: {e:?}"))?;

        // 2. Batched silu_clamp + batched FWHT rotate. Each kernel handles
        //    all k_top streams in one launch (grid.y = k_top), replacing
        //    2*k_top = 12 small launches with 2.
        gpu.v4f_silu_mul_clamp_f32_batched(
            gate_batch, up_batch, gate_batch,
            im, k_top, cfg.swiglu_limit,
        ).map_err(|e| format!("v4f_silu_mul_clamp batched l{layer_idx}: {e:?}"))?;
        gpu.rotate_x_mq_batched(gate_batch, rot_batch, im, k_top)
            .map_err(|e| format!("rotate batched l{layer_idx}: {e:?}"))?;

        // 3. Fused down GEMV: one launch atomicAdds
        //      Σ_k topk_weights[k] * (W_down[expert_k] · rot_batch[k])
        //    into ffn_out. Replaces k_top per-expert GEMV + scaled_add
        //    pairs. The route_scale_override is baked into topk_weights.
        gpu.v4f_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
            w2_ptrs, topk_idx_dev, topk_w_dev,
            rot_batch, ffn_out,
            cfg.hidden_size, im, k_top,
        ).map_err(|e| format!("fused down l{layer_idx}: {e:?}"))?;

        return Ok(());
    }

    // Per-expert fallback path is no longer reachable: separate w1/w3
    // blobs are no longer uploaded (only the combined gate_up blob).
    // HIPFIRE_V4F_NO_FUSED_MOE=1 yields a hard error rather than silent
    // shared-only fallback.
    let _ = (wts, topk_ids, route_scale_override);
    Err(format!(
        "deepseek4: HIPFIRE_V4F_NO_FUSED_MOE=1 but layer {layer_idx} \
         has no separate w1/w3 blobs (only combined gate_up). Unset the \
         env var or rebuild the loader with separate-blob uploads."))
}

/// Hash-routed FFN dispatch (V4F layers 0..num_hash_layers = 0..3).
///
/// Per upstream V4F (model.py:Gate.forward, model.py:587-606):
///   if self.hash:
///     indices = self.tid2eid[input_ids]          [k]   ← static lookup
///   else:
///     indices = scores.topk(k)[1]
///   weights = original_scores.gather(1, indices) [k]   ← from unbiased scores
///   weights /= weights.sum();  weights *= route_scale
///
/// So we still need the gate.weight GEMV to get scores for the weight
/// values — only the SELECTION is static. The dispatch loop is otherwise
/// identical to `ffn_routed`.
///
/// Same env gate (`HIPFIRE_V4F_MOE=1`) and blob-presence guard.
fn ffn_hash_routed(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    token_id: u32,
) -> Result<(), String> {
    if std::env::var("HIPFIRE_V4F_MOE").ok().as_deref() != Some("1") {
        return Ok(());
    }
    // Bisection knob: disable hash routing on layers 0..num_hash_layers.
    // Existing v4f.mq2lloyd-fp4fix shipped without tid2eid → hash routing
    // was silently no-op (returned at tid2eid empty check). The new HFQ
    // (v4f.mq2lloyd-f16compress.hfq) includes tid2eid so the path runs
    // for the first time. If the static-routing math has a bug (e.g.
    // wrong score normalisation vs upstream's pre-softplus gather), this
    // flag bisects: HIPFIRE_V4F_NO_HASH=1 reproduces the old shared-only
    // behaviour on hash layers.
    if std::env::var("HIPFIRE_V4F_NO_HASH").ok().as_deref() == Some("1") {
        return Ok(());
    }
    let layer = weights.resolve_layer(layer_idx);
    if layer.expert_gate_up_blob.is_none() || layer.expert_w2_blob.is_none()
    {
        return Ok(());
    }
    if layer.tid2eid_host.is_empty() {
        // tid2eid not in the HFQ (pre-FP4-fix quant skipped it). Fall back
        // to shared-only on this layer.
        return Ok(());
    }

    // Compute scores (unbiased) on-device for the weight values.
    moe_route(cfg, weights, state, gpu, layer_idx)?;
    let scores = state.router_scores.as_ref().unwrap();
    let scores_host = gpu.download_f32(scores)
        .map_err(|e| format!("d2h scores hash l{layer_idx}: {e:?}"))?;

    let k = cfg.num_experts_per_tok;
    let n_exp = cfg.n_routed_experts;

    // Static expert IDs from tid2eid[token_id, 0..k].
    let row = (token_id as usize) * k;
    if row + k > layer.tid2eid_host.len() {
        return Err(format!(
            "hash l{layer_idx}: token_id {token_id} out of tid2eid range \
             ({} entries)", layer.tid2eid_host.len()));
    }
    let topk_ids: Vec<u32> = layer.tid2eid_host[row..row + k].iter()
        .map(|&i| i.min((n_exp - 1) as u32))
        .collect();

    let wts = match gather_normalized_weights(&scores_host, &topk_ids) {
        Some(w) => w,
        None => return Ok(()),
    };

    // Fused MoE dispatch — same body as ffn_routed but with static
    // tid2eid-derived top-K indices.
    let im = cfg.moe_intermediate_size;
    let ffn_x_rot = state.ffn_x_rot.as_ref().unwrap();
    let ffn_out = state.ffn_out.as_ref().unwrap();
    let route_scale_override: f32 = std::env::var("HIPFIRE_V4F_ROUTE_SCALE")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2.2);
    let k_top = topk_ids.len();

    // Lazy-alloc moe scratch (shared with ffn_routed via state).
    if state.moe_topk_indices.is_none() {
        state.moe_topk_indices = Some(gpu.alloc_tensor(&[k_top], DType::F32)
            .map_err(|e| format!("alloc moe_topk_indices hash: {e:?}"))?);
    }
    if state.moe_topk_weights.is_none() {
        state.moe_topk_weights = Some(gpu.alloc_tensor(&[k_top], DType::F32)
            .map_err(|e| format!("alloc moe_topk_weights hash: {e:?}"))?);
    }
    if state.moe_gate_batch.is_none() {
        state.moe_gate_batch = Some(gpu.alloc_tensor(&[k_top, im], DType::F32)
            .map_err(|e| format!("alloc moe_gate_batch hash: {e:?}"))?);
    }
    if state.moe_up_batch.is_none() {
        state.moe_up_batch = Some(gpu.alloc_tensor(&[k_top, im], DType::F32)
            .map_err(|e| format!("alloc moe_up_batch hash: {e:?}"))?);
    }
    if state.moe_rot_batch.is_none() {
        state.moe_rot_batch = Some(gpu.alloc_tensor(&[k_top, im], DType::F32)
            .map_err(|e| format!("alloc moe_rot_batch hash: {e:?}"))?);
    }

    let topk_idx_dev = state.moe_topk_indices.as_ref().unwrap();
    let topk_w_dev = state.moe_topk_weights.as_ref().unwrap();
    let idx_i32: Vec<i32> = topk_ids.iter().map(|&x| x as i32).collect();
    let idx_bytes: Vec<u8> = idx_i32.iter().flat_map(|i| i.to_le_bytes()).collect();
    gpu.memcpy_htod_auto(&topk_idx_dev.buf, &idx_bytes)
        .map_err(|e| format!("htod topk_indices hash l{layer_idx}: {e:?}"))?;
    let w_scaled: Vec<f32> = wts.iter().map(|&w| w * route_scale_override).collect();
    let w_bytes: Vec<u8> = w_scaled.iter().flat_map(|w| w.to_le_bytes()).collect();
    gpu.memcpy_htod_auto(&topk_w_dev.buf, &w_bytes)
        .map_err(|e| format!("htod topk_weights hash l{layer_idx}: {e:?}"))?;

    let gate_up_ptrs = layer.expert_gate_up_ptrs.as_ref().unwrap();
    let w2_ptrs = layer.expert_w2_ptrs.as_ref().unwrap();
    let gate_batch = state.moe_gate_batch.as_ref().unwrap();
    let up_batch = state.moe_up_batch.as_ref().unwrap();
    let rot_batch = state.moe_rot_batch.as_ref().unwrap();

    gpu.v4f_gemv_mq2g256_lloyd_moe_gate_up_indexed(
        gate_up_ptrs, topk_idx_dev,
        ffn_x_rot, gate_batch, up_batch,
        2 * im, cfg.hidden_size, k_top,
    ).map_err(|e| format!("fused gate_up hash l{layer_idx}: {e:?}"))?;

    gpu.v4f_silu_mul_clamp_f32_batched(
        gate_batch, up_batch, gate_batch,
        im, k_top, cfg.swiglu_limit,
    ).map_err(|e| format!("v4f_silu_mul_clamp batched hash l{layer_idx}: {e:?}"))?;
    gpu.rotate_x_mq_batched(gate_batch, rot_batch, im, k_top)
        .map_err(|e| format!("rotate batched hash l{layer_idx}: {e:?}"))?;

    gpu.v4f_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
        w2_ptrs, topk_idx_dev, topk_w_dev,
        rot_batch, ffn_out,
        cfg.hidden_size, im, k_top,
    ).map_err(|e| format!("fused down hash l{layer_idx}: {e:?}"))?;

    Ok(())
}

/// HC FFN mix — same pattern as `hc_attn_mix` but with `hc_ffn_*`
/// tensors and `ffn_out` as transform_out.
fn hc_ffn_mix(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let _ = (weights, layer_idx);
    let streams = state.residual_streams.as_ref().unwrap();
    let ffn_out = state.ffn_out.as_ref().unwrap();

    // Same reasoning as hc_attn_mix: mhc_pre(is_attn=false) has
    // already populated state.hc_c with the FFN block's post and comb
    // (α-scaled, sigmoid'd, sinkhorn'd). Just consume them.
    let post_view = state.hc_c.as_ref().unwrap().sub_offset(4, 4);
    let comb_view = state.hc_c.as_ref().unwrap().sub_offset(8, 16);

    let streams_out = state.q.as_ref().unwrap();
    gpu.hc_mix_4stream(streams, &comb_view, &post_view, ffn_out, streams_out,
        cfg.hidden_size as i32)
        .map_err(|e| format!("hc_mix_4stream ffn: {e:?}"))?;

    let bytes = cfg.hc_mult * cfg.hidden_size * 4;
    gpu.memcpy_dtod_auto(&streams.buf, &streams_out.buf, bytes)
        .map_err(|e| format!("d2d hc_ffn_mix → streams: {e:?}"))?;
    Ok(())
}

/// Final head HC mix + norm + lm_head.
///
/// Upstream V4F ParallelHead.hc_head:
///   x_flat = streams.flatten()  # [hc_mult * hidden]
///   rsqrt = rsqrt(mean(x_flat^2) + eps)
///   mixes = (hc_head_fn @ x_flat) * rsqrt        # [hc_mult]
///   pre   = sigmoid(mixes * hc_head_scale + hc_head_base) + hc_eps
///   y[d]  = sum_h pre[h] * streams[h, d]         # [hidden]
///   final = rmsnorm(y, output_norm)              # [hidden]
///   logits = head @ final                        # [vocab_size]
///
/// Save stream 0 of `state.residual_streams` to `state.mtp_last_hidden`.
///
/// Captures the post-layer-block hidden state for downstream MTP
/// speculative decoding (V3 paper §4 `h_n`). Both `decode_step` and
/// `mtp_forward` call this just before applying their respective final
/// RMSNorm + lm_head, so the caller of `speculative_decode_step` can
/// always read `state.mtp_last_hidden` regardless of which function
/// produced it.
///
/// Allocates the destination on first call. d2d copy of `hidden` F32
/// floats — negligible cost relative to a layer block.
fn capture_mtp_hidden(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    if state.mtp_last_hidden.is_none() {
        state.mtp_last_hidden = Some(gpu.alloc_tensor(&[hidden], DType::F32)
            .map_err(|e| format!("alloc mtp_last_hidden: {e:?}"))?);
    }
    let streams = state.residual_streams.as_ref()
        .ok_or_else(|| "capture_mtp_hidden: residual_streams missing".to_string())?;
    let stream0 = streams.sub_offset(0, hidden);
    let dst = state.mtp_last_hidden.as_ref().unwrap();
    gpu.memcpy_dtod_auto(&dst.buf, &stream0.buf, hidden * 4)
        .map_err(|e| format!("capture_mtp_hidden d2d: {e:?}"))
}

/// We were previously taking ONLY stream 0 for the head — discarding 75%
/// of the model's output state. This wires the full HC mix.
fn final_norm_and_head(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    let output_norm = weights.output_norm.as_ref()
        .ok_or_else(|| "output_norm not uploaded".to_string())?;
    let head = weights.head.as_ref()
        .ok_or_else(|| "head not uploaded".to_string())?;
    let hc_head_fn = weights.hc_head_fn.as_ref()
        .ok_or_else(|| "hc_head_fn not uploaded".to_string())?;
    let hc_head_base = weights.hc_head_base.as_ref()
        .ok_or_else(|| "hc_head_base not uploaded".to_string())?;
    let streams = state.residual_streams.as_ref().unwrap();

    if state.final_norm.is_none() {
        state.final_norm = Some(gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
            .map_err(|e| format!("alloc final_norm: {e:?}"))?);
    }
    if state.final_norm_rot.is_none() {
        state.final_norm_rot = Some(gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
            .map_err(|e| format!("alloc final_norm_rot: {e:?}"))?);
    }
    if state.logits.is_none() {
        state.logits = Some(gpu.alloc_tensor(&[cfg.vocab_size], DType::F32)
            .map_err(|e| format!("alloc logits: {e:?}"))?);
    }
    if state.head_hc_pre.is_none() {
        state.head_hc_pre = Some(gpu.alloc_tensor(&[cfg.hc_mult], DType::F32)
            .map_err(|e| format!("alloc head_hc_pre: {e:?}"))?);
    }
    if state.head_hc_out.is_none() {
        state.head_hc_out = Some(gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
            .map_err(|e| format!("alloc head_hc_out: {e:?}"))?);
    }

    let final_norm = state.final_norm.as_ref().unwrap();
    let final_norm_rot = state.final_norm_rot.as_ref().unwrap();
    let logits = state.logits.as_ref().unwrap();
    let head_hc_pre = state.head_hc_pre.as_ref().unwrap();
    let head_hc_out = state.head_hc_out.as_ref().unwrap();

    // 1. Head HC: compute pre[hc_mult] = sigmoid((hc_head_fn @ x_flat * rsqrt) * scale + base) + eps
    let x_dim = cfg.hidden_size * cfg.hc_mult;
    gpu.hc_head_compute_pre(streams, hc_head_fn, hc_head_base, head_hc_pre,
        cfg.hc_mult as i32, x_dim as i32,
        weights.hc_head_scale, cfg.rms_norm_eps, cfg.hc_eps,
    ).map_err(|e| format!("hc_head_compute_pre: {e:?}"))?;

    // 2. Head HC combine: head_hc_out[d] = sum_h pre[h] * streams[h, d]
    gpu.hc_input_map_4stream(head_hc_pre, streams, head_hc_out, cfg.hidden_size as i32)
        .map_err(|e| format!("hc_input_map (head): {e:?}"))?;

    // 2.5. Capture h_n for downstream MTP / spec-decode.
    //
    // OPEN QUESTION (empirical): which value does V4F's MTP head_proj
    // expect as its h-input?
    //   (A) main's stream 0 (residual_streams[0, :])
    //   (B) main's post-head-HC-mixed (head_hc_out)
    //   (C) main's post-output-norm (final_norm)
    //
    // Measured K=2 short-prompt acceptance:
    //   (A): ~50%  ← used to be our convention
    //   (B): ~25%  ← worse, surprised since MTP ships hc_head_* tensors
    // (C) untested.
    //
    // Sticking with (A) until a definitive signal (e.g. byte-equal
    // against an antirez ds4 reference) says otherwise. MTP's own
    // hc_head_* matrices are still used in mtp_forward step 7 for its
    // OUTPUT pipeline (lm_head input), not for processing main's input.
    if state.mtp_last_hidden.is_none() {
        state.mtp_last_hidden = Some(gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
            .map_err(|e| format!("alloc mtp_last_hidden in final_norm_and_head: {e:?}"))?);
    }
    {
        let stream0 = streams.sub_offset(0, cfg.hidden_size);
        let dst = state.mtp_last_hidden.as_ref().unwrap();
        gpu.memcpy_dtod_auto(&dst.buf, &stream0.buf, cfg.hidden_size * 4)
            .map_err(|e| format!("capture stream0 → mtp_last_hidden: {e:?}"))?;
    }

    // 3. RMSNorm of the combined stream output.
    gpu.rmsnorm_f32(head_hc_out, output_norm, final_norm, cfg.rms_norm_eps)
        .map_err(|e| format!("final rmsnorm_f32: {e:?}"))?;

    // 4. FWHT-rotate for MQ4 GEMV.
    gpu.rotate_x_mq(final_norm, final_norm_rot, cfg.hidden_size)
        .map_err(|e| format!("rotate_x_mq final_norm: {e:?}"))?;

    // 5. lm_head GEMV. F16 path uses un-rotated final_norm.
    gemv_auto(gpu, head, final_norm_rot, final_norm, logits,
        cfg.vocab_size, cfg.hidden_size)?;

    Ok(())
}

/// Step 6: Single-position attention (position-0 degenerate case).
///
/// V4F's attention with `o_groups = 8` means the 64 query heads
/// are reduced over groups of 8 heads → 8 grouped outputs each of
/// `head_dim = 512`, yielding `[8 * 512 = 4096]` = hidden directly.
/// No separate O-projection needed (wo_a/wo_b's role TBD per paper).
///
/// For position-0 (no past KV history), each query head attends
/// only to the current token's K/V. softmax over 1 position = 1.0,
/// so attn_per_head = V. With o_groups grouping: each of 8 groups
/// sums 8 identical V vectors → attn_per_group = 8 * V.
///
/// Output `[hidden = o_groups * head_dim]`: 8 copies of V (each
/// scaled by 8 due to the in-group sum), giving [8*V, 8*V, ..., 8*V].
///
/// This handles position 0. For position > 0 we need SWA cache +
/// real Q·K·V over history — pending.
fn attn_stub(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    // Final attention contribution: shape [hidden]. Consumed by hc_attn_mix.
    if state.attn_out.is_none() {
        state.attn_out = Some(gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
            .map_err(|e| format!("alloc attn_out: {e:?}"))?);
    }
    // Raw attention output [n_heads, head_dim] — kernel writes here.
    let n_heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim;
    let n_heads_head_dim = n_heads * head_dim;
    if state.attn_out_raw.is_none() {
        state.attn_out_raw = Some(gpu.alloc_tensor(&[n_heads, head_dim], DType::F32)
            .map_err(|e| format!("alloc attn_out_raw: {e:?}"))?);
    }
    if state.attn_out_raw_rot.is_none() {
        state.attn_out_raw_rot = Some(gpu.alloc_tensor(&[n_heads_head_dim], DType::F32)
            .map_err(|e| format!("alloc attn_out_raw_rot: {e:?}"))?);
    }
    let n_groups = cfg.o_groups;
    let o_lora_rank = cfg.o_lora_rank;
    let groups_o_lora = n_groups * o_lora_rank;
    if state.wo_a_out.is_none() {
        state.wo_a_out = Some(gpu.alloc_tensor(&[groups_o_lora], DType::F32)
            .map_err(|e| format!("alloc wo_a_out: {e:?}"))?);
    }
    if state.wo_a_out_rot.is_none() {
        state.wo_a_out_rot = Some(gpu.alloc_tensor(&[groups_o_lora], DType::F32)
            .map_err(|e| format!("alloc wo_a_out_rot: {e:?}"))?);
    }

    // SWA is now the production default. Pos-0 path retained only as a
    // diagnostic/regression-check escape hatch via HIPFIRE_V4F_ATTN=pos0.
    let use_swa = std::env::var("HIPFIRE_V4F_ATTN").ok().as_deref() != Some("pos0");

    let q = state.q.as_ref().unwrap();
    let kv = state.kv.as_ref().unwrap();
    let attn_out_raw = state.attn_out_raw.as_ref().unwrap();
    let layer = weights.resolve_layer(layer_idx);
    let attn_sink = layer.attn_sink.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} attn_sink not uploaded"))?;

    if !use_swa {
        // Pos-0 attention (default). Each step independent.
        gpu.v4f_attn_pos0(q, kv, attn_sink, attn_out_raw,
            n_heads as i32, head_dim as i32, n_groups as i32,
        ).map_err(|e| format!("v4f_attn_pos0: {e:?}"))?;
    } else {
        // SWA path.
        let n_kv = cfg.num_key_value_heads;
        let win = cfg.sliding_window;
        {
            let attn = &mut state._attention[layer_idx];
            if attn.swa_k.is_none() {
                attn.swa_k = Some(gpu.zeros(&[n_kv, head_dim, win], DType::F32)
                    .map_err(|e| format!("alloc swa_k l{layer_idx}: {e:?}"))?);
            }
            if attn.swa_v.is_none() {
                attn.swa_v = Some(gpu.zeros(&[n_kv, head_dim, win], DType::F32)
                    .map_err(|e| format!("alloc swa_v l{layer_idx}: {e:?}"))?);
            }
        }
        let pos = state.n_tokens as usize;
        let slot = pos % win;
        {
            let swa_k = state._attention[layer_idx].swa_k.as_ref().unwrap();
            let swa_v = state._attention[layer_idx].swa_v.as_ref().unwrap();
            gpu.swa_ring_write_f32(kv, swa_k, n_kv as i32, head_dim as i32, win as i32, slot as i32)
                .map_err(|e| format!("swa_k write: {e:?}"))?;
            gpu.swa_ring_write_f32(kv, swa_v, n_kv as i32, head_dim as i32, win as i32, slot as i32)
                .map_err(|e| format!("swa_v write: {e:?}"))?;
        }
        let n_valid = (pos + 1).min(win) as i32;

        // Antirez-faithful mixed attention (ds4.c:7559-7566):
        //   ratio == 0 (dense): plain SWA attention over raw_kv
        //   ratio  > 0 (compressed): JOINT softmax over raw_kv + main_kv_cache
        //     ratio == 4: indexer top-K selects which compressor entries
        //     ratio == 128: no indexer, attend to ALL compressor entries
        //
        // Both compressor and raw entries share ONE softmax with the
        // attn_sink as an extra implicit drain entry. The compressed
        // cache contains the model's "coarse memory" — even at small pos
        // (within SWA window) the compressor cache provides DIFFERENT
        // signal than raw KV (compressed entries are softmax-pooled
        // wkv outputs with compressor.norm + RoPE applied; raw KV is the
        // per-position post-kv_norm post-RoPE K=V).
        //
        // Env opt-out: HIPFIRE_V4F_NO_MIXED=1 falls back to SWA-only.
        let no_mixed = std::env::var("HIPFIRE_V4F_NO_MIXED").ok().as_deref() == Some("1");
        let do_mixed = !no_mixed
            && layer.compress_ratio > 0
            && state._indexer[layer_idx].main_kv_cache.is_some();

        if do_mixed {
            let topk_max = cfg.index_topk;
            if state._attention[layer_idx].gathered_k.is_none() {
                state._attention[layer_idx].gathered_k = Some(
                    gpu.zeros(&[n_kv, head_dim, topk_max], DType::F32)
                        .map_err(|e| format!("alloc gathered_k l{layer_idx}: {e:?}"))?
                );
            }
            let ratio = layer.compress_ratio as usize;
            // n_compressed: number of compressed slots committed so far.
            // Compressor commits a slot every `ratio` steps when
            // (pos+1) % ratio == 0. With state.n_tokens being the
            // just-incremented position, n_committed = (state.n_tokens) / ratio.
            // Actually since we're at the END of attn_stub's pos (= state.n_tokens
            // before increment), and the compressor ran BEFORE attn_stub:
            //   at pos p with (p+1)%ratio==0, compressor wrote slot p/ratio.
            //   n_committed after compressor = (p+1) / ratio.
            let n_compressed = ((pos + 1) / ratio).min(topk_max);

            let k_active = if n_compressed == 0 {
                0
            } else if layer.compress_ratio == 4
                && state._indexer[layer_idx].topk_idx_indices.is_some()
            {
                // ratio=4: gather using indexer top-K. The topk_idx_indices
                // was populated by indexer_forward — it has up to index_topk
                // entries, padded with -1 sentinels beyond n_compressed.
                let topk_idx = state._indexer[layer_idx].topk_idx_indices.as_ref().unwrap();
                let main_kv_cache = state._indexer[layer_idx].main_kv_cache.as_ref().unwrap();
                let gathered_k = state._attention[layer_idx].gathered_k.as_ref().unwrap();
                let k = cfg.index_topk.min(n_compressed);
                gpu.v4f_topk_kv_gather_f32(
                    main_kv_cache, topk_idx, gathered_k,
                    k as i32, head_dim as i32, n_compressed as i32,
                    topk_max as i32, 0, /*scale=*/1.0,
                ).map_err(|e| format!("mixed gather (idx) l{layer_idx}: {e:?}"))?;
                k
            } else {
                // ratio=128 (or fallback): no indexer, attend to all
                // n_compressed entries directly. Copy main_kv_cache[0..n]
                // into gathered_k[0..n] (transposed layout).
                let main_kv_cache = state._indexer[layer_idx].main_kv_cache.as_ref().unwrap();
                let gathered_k = state._attention[layer_idx].gathered_k.as_ref().unwrap();
                gpu.v4f_topk_kv_gather_identity_f32(
                    main_kv_cache, gathered_k,
                    n_compressed as i32, head_dim as i32, topk_max as i32,
                ).map_err(|e| format!("mixed gather (all) l{layer_idx}: {e:?}"))?;
                n_compressed
            };

            let swa_k = state._attention[layer_idx].swa_k.as_ref().unwrap();
            let swa_v = state._attention[layer_idx].swa_v.as_ref().unwrap();
            let gathered_k = state._attention[layer_idx].gathered_k.as_ref().unwrap();
            // Joint softmax: scores = Q·K for [swa_k, gathered_k, attn_sink],
            // single normalization, V = swa_v + gathered_v (K=V tied, so
            // we pass gathered_k as V too).
            gpu.v4f_attn_swa_topk_f32(
                q, swa_k, swa_v, gathered_k, gathered_k,
                attn_sink, attn_out_raw,
                n_heads as i32, head_dim as i32,
                win as i32, topk_max as i32,
                n_valid, k_active as i32,
            ).map_err(|e| format!("v4f_attn_swa_topk l{layer_idx}: {e:?}"))?;
        } else {
            let swa_k = state._attention[layer_idx].swa_k.as_ref().unwrap();
            let swa_v = state._attention[layer_idx].swa_v.as_ref().unwrap();
            gpu.v4f_attn_swa(q, swa_k, swa_v, attn_sink, attn_out_raw,
                n_heads as i32, head_dim as i32, n_groups as i32,
                n_valid, win as i32,
            ).map_err(|e| format!("v4f_attn_swa: {e:?}"))?;
        }
    }

    // Inverse tail RoPE on attn_out_raw. Same YaRN params as the forward
    // apply_tail_rope so the rotation cancels correctly across attention.
    // Antirez `layer_forward_self_one` does the matching:
    //   rope_tail_layer_inplace(q,     ..., pos, il, false)  // forward
    //   rope_tail_layer_inplace(heads, ..., pos, il, true)   // inverse
    // (ds4.c:7868, 7874)
    let pos_buf = state.pos_buf.as_ref()
        .ok_or_else(|| "pos_buf not allocated".to_string())?;
    if std::env::var("HIPFIRE_V4F_SKIP_INV_ROPE").ok().as_deref() != Some("1") {
        if std::env::var("HIPFIRE_V4F_NO_YARN").ok().as_deref() == Some("1") {
            gpu.rope_tail_inverse(attn_out_raw, pos_buf,
                n_heads as i32, head_dim as i32,
                cfg.qk_rope_head_dim as i32, cfg.rope_theta,
            ).map_err(|e| format!("rope_tail_inverse (no-yarn) l{layer_idx}: {e:?}"))?;
        } else {
            let layer = weights.resolve_layer(layer_idx);
            let (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high) =
                layer_rope_params(cfg, layer.compress_ratio);
            gpu.rope_tail_yarn_interleaved(
                attn_out_raw, attn_out_raw, pos_buf,
                n_heads as i32, 0,
                head_dim as i32,
                cfg.qk_rope_head_dim as i32,
                freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high,
                /*inverse=*/1,
            ).map_err(|e| format!("rope_tail_yarn_interleaved (inverse) l{layer_idx}: {e:?}"))?;
        }
    }

    // O-LoRA projection: wo_a per-group + wo_b.
    //   wo_a: [n_groups * o_lora_rank, heads_per_group * head_dim] MQ4
    //         = [8 * 1024, 8 * 512] = [8192, 4096]
    //   Per group g: y_g [o_lora_rank=1024] = wo_a_g [1024, 4096] @ x_g [4096]
    //   wo_b: [hidden, n_groups * o_lora_rank] MQ4 = [4096, 8192]
    //   y [hidden=4096] = wo_b @ wo_a_out_rot [8192]
    let wo_a = layer.wo_a.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wo_a missing"))?;
    let wo_b = layer.wo_b.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wo_b missing"))?;
    let attn_out_raw_rot = state.attn_out_raw_rot.as_ref().unwrap();
    let wo_a_out = state.wo_a_out.as_ref().unwrap();
    let wo_a_out_rot = state.wo_a_out_rot.as_ref().unwrap();
    let final_attn_out = state.attn_out.as_ref().unwrap();

    // FWHT-rotate per-group slices of attn_out_raw (k=heads_per_group*head_dim).
    let heads_per_group = n_heads / n_groups;
    let per_group_in = heads_per_group * head_dim;
    let per_group_elems = o_lora_rank * per_group_in;
    // Per-group byte stride depends on wo_a's dtype:
    //   MQ4G256 (Raw):     136 bytes per 256 elements
    //   Q8_0:               34 bytes per 32 elements
    //   F32 (F16-source):   4 bytes per element (handled via sub_offset's
    //                       built-in size scaling — pass elem count)
    let per_group_wa_bytes_raw = (per_group_elems / 256) * 136;
    let per_group_wa_bytes_q8  = (per_group_elems / 32) * 34;

    // FWHT-rotate all 8 group slices in one batched launch. attn_out_raw
    // is contiguous [n_groups, per_group_in] so grid.y=n_groups indexes
    // each group at stride per_group_in.
    gpu.rotate_x_mq_batched(attn_out_raw, attn_out_raw_rot, per_group_in, n_groups)
        .map_err(|e| format!("rotate attn_out batched l{layer_idx}: {e:?}"))?;

    for g in 0..n_groups {
        let raw_view = attn_out_raw.sub_offset(g * per_group_in, per_group_in);
        let rot_view = attn_out_raw_rot.sub_offset(g * per_group_in, per_group_in);
        // Dtype-aware sub-view for wo_a's per-group slice.
        let wo_a_view = match wo_a.dtype {
            DType::F32 => {
                // sub_offset handles size scaling for F32 (size=4). Result
                // is 1D; gemv_f32 expects 2D [m, k] so we mutate the shape.
                let mut v = wo_a.sub_offset(g * per_group_elems, per_group_elems);
                v.shape = vec![o_lora_rank, per_group_in];
                v
            }
            DType::Q8_0 => {
                wo_a.sub_offset(g * per_group_wa_bytes_q8, per_group_wa_bytes_q8)
            }
            _ => {
                wo_a.sub_offset(g * per_group_wa_bytes_raw, per_group_wa_bytes_raw)
            }
        };
        let out_view = wo_a_out.sub_offset(g * o_lora_rank, o_lora_rank);
        // Dispatch per dtype. F32/Q8 use plain raw_view; MQ4 uses rot_view.
        gemv_auto(gpu, &wo_a_view, &rot_view, &raw_view, &out_view,
                  o_lora_rank, per_group_in)?;
    }

    // FWHT-rotate wo_a_out then wo_b GEMV → final_attn_out [hidden].
    // wo_b path: F32/Q8 use plain wo_a_out; MQ4 uses wo_a_out_rot.
    gpu.rotate_x_mq(wo_a_out, wo_a_out_rot, groups_o_lora)
        .map_err(|e| format!("rotate wo_a_out l{layer_idx}: {e:?}"))?;
    gemv_auto(gpu, wo_b, wo_a_out_rot, wo_a_out, final_attn_out,
              cfg.hidden_size, groups_o_lora)?;

    Ok(())
}

/// V4F MoE router: scores and top-K expert selection.
///
/// For score-routed layers (l >= num_hash_layers = 3 on V4F):
///   1. logits = gate.weight @ ffn_input  [256]  (MQ4G256 GEMV, M=256, K=hidden)
///   2. logits += gate.bias
///   3. scores = sqrt(softplus(logits))   [256]  (V4F affinity)
///   4. topk_indices = top_k(scores, k=6)        (reuses indexer_top_k)
///
/// For hash-routed layers (l < 3): use the static `tid2eid` lookup
/// table. Currently SKIPPED at quantize time, so hash-routed layers
/// fall back to shared expert only.
///
/// Output lives in state.router_scores and state.topk_indices. The
/// expert-dispatch step reads topk_indices, fetches per-expert weights
/// from `layer.expert_w{1,2,3}` (requires `HIPFIRE_V4F_UPLOAD_EXPERTS=1`),
/// and accumulates weighted expert outputs into ffn_out.
/// Gather routing weights at the given indices from the (unbiased) scores,
/// then normalize to sum to 1. Returns `None` if the sum is non-positive.
fn gather_normalized_weights(scores: &[f32], indices: &[u32]) -> Option<Vec<f32>> {
    let mut wts: Vec<f32> = indices.iter()
        .map(|&i| *scores.get(i as usize).unwrap_or(&0.0))
        .collect();
    let s: f32 = wts.iter().sum();
    if s <= 0.0 { return None; }
    for w in wts.iter_mut() { *w /= s; }
    Some(wts)
}

/// V4F bias-aware top-K routing on host. Pure function — no GPU types.
///
/// Per upstream `inference/model.py:Gate.forward`:
///   biased = scores + bias                  (zero-pad bias if shorter)
///   indices = argsort_desc(biased)[..k]     (greedy top-K)
///   weights = scores[indices]               (UNBIASED scores)
///   weights /= weights.sum()                (normalize)
///
/// Returns `Some((indices, weights))` on success, or `None` if the
/// unbiased weight-sum is non-positive (degenerate router output).
///
/// The `routed_scaling_factor` multiply happens later at the per-expert
/// accumulation step in `ffn_routed` / `ffn_hash_routed` (avoids double
/// counting).
fn bias_aware_topk_weights(
    scores: &[f32],
    bias: &[f32],
    k: usize,
) -> Option<(Vec<u32>, Vec<f32>)> {
    let n = scores.len();
    if k == 0 || n == 0 { return None; }
    let mut biased: Vec<f32> = (0..n)
        .map(|i| scores[i] + bias.get(i).copied().unwrap_or(0.0))
        .collect();
    let k = k.min(n);
    let mut indices: Vec<u32> = Vec::with_capacity(k);
    for _ in 0..k {
        let (best_i, _) = biased.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        indices.push(best_i as u32);
        biased[best_i] = f32::NEG_INFINITY;
    }
    let mut wts: Vec<f32> = indices.iter().map(|&i| scores[i as usize]).collect();
    let w_sum: f32 = wts.iter().sum();
    if w_sum <= 0.0 { return None; }
    for w in wts.iter_mut() { *w /= w_sum; }
    Some((indices, wts))
}

fn moe_route(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    // Hash-routed and score-routed layers BOTH need router_scores for the
    // per-token expert weights (upstream V4F gathers unbiased scores at
    // tid2eid indices for hash layers, top-K for score layers). The split
    // was: score layers ALSO use gate.bias for bias-aware selection. So
    // gate.weight + sqrt_softplus is shared; gate.bias is optional.
    let layer = weights.resolve_layer(layer_idx);
    let gate_w = layer.gate_weight.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} gate.weight missing"))?;
    let _gate_b = layer.gate_bias.as_ref();  // None for hash layers; unused here

    let n_exp = cfg.n_routed_experts;
    let k = cfg.num_experts_per_tok;
    if state.router_scores.is_none() {
        state.router_scores = Some(gpu.alloc_tensor(&[n_exp], DType::F32)
            .map_err(|e| format!("alloc router_scores: {e:?}"))?);
    }
    if state.topk_indices.is_none() {
        state.topk_indices = Some(gpu.alloc_tensor(&[k], DType::F32)
            .map_err(|e| format!("alloc topk_indices: {e:?}"))?);
    }
    let scores = state.router_scores.as_ref().unwrap();
    let topk = state.topk_indices.as_ref().unwrap();

    // Upstream V4F gates on the POST-ffn_norm input (same x that
    // shared/routed experts see). ffn_x_rot is FWHT(ffn_norm(hc_x_in));
    // ffn_x_plain is the un-rotated version. Both are populated in
    // ffn_stub which runs before us.
    //
    // gemv_auto dispatches on the gate weight's dtype: MQ4 path consumes
    // ffn_x_rot, Q8_0 / F16 paths consume ffn_x_plain. Switching from the
    // hardcoded gemv_mq4g256_prerotated call lets the router work with
    // any quant of `gate.weight` — needed by v4f-q8-mtp (Q8F16) and
    // future formats. Using raw hc_x_in (as before ffn_stub landed)
    // caused scores to scale with stream magnitude, biasing selection.
    let ffn_x_rot = state.ffn_x_rot.as_ref()
        .ok_or_else(|| "ffn_x_rot not allocated — moe_route must run after ffn_stub".to_string())?;
    let ffn_x_plain = state.ffn_x_plain.as_ref()
        .ok_or_else(|| "ffn_x_plain not allocated — moe_route must run after ffn_stub".to_string())?;

    // logits = gate.weight @ x  (dispatch on gate.weight dtype)
    gemv_auto(gpu, gate_w, ffn_x_rot, ffn_x_plain, scores, n_exp, cfg.hidden_size)?;

    // logits += gate.bias (bias is F16, scores is F32 — need a kernel
    // for f16-bias-add. Skip for now; bias is small magnitude).
    let _ = _gate_b;

    // scores = sqrt(softplus(logits))
    gpu.sqrt_softplus_f32(scores)
        .map_err(|e| format!("sqrt_softplus layer {layer_idx}: {e:?}"))?;

    // top-K via indexer_top_k. H=1, N=n_exp, K=k.
    gpu.indexer_top_k(scores, topk, 1, n_exp as i32, k as i32)
        .map_err(|e| format!("indexer_top_k router layer {layer_idx}: {e:?}"))?;

    Ok(())
}

/// mHC pre-step: compute c = X · W_fn + base [24], split into
/// Ã/B̃/C̃, apply sigmoid/exp+Sinkhorn/2σ, then compute
/// state.hc_x_in = A_l · streams (the input mapping).
///
/// After this runs, the layer's transform (attn or FFN) reads
/// hc_x_in as its [hidden]-shaped input.
fn mhc_pre(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    is_attn: bool,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let (hc_fn, hc_base) = if is_attn {
        (layer.hc_attn_fn.as_ref().unwrap(),
         layer.hc_attn_base.as_ref().unwrap())
    } else {
        (layer.hc_ffn_fn.as_ref().unwrap(),
         layer.hc_ffn_base.as_ref().unwrap())
    };
    let streams = state.residual_streams.as_ref().unwrap();

    if state.hc_x_in.is_none() {
        state.hc_x_in = Some(gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
            .map_err(|e| format!("alloc hc_x_in: {e:?}"))?);
    }
    if state.hc_c.is_none() {
        state.hc_c = Some(gpu.alloc_tensor(&[24], DType::F32)
            .map_err(|e| format!("alloc hc_c: {e:?}"))?);
    }

    let n_ctrl = 24;
    let x_dim = cfg.hidden_size * cfg.hc_mult;
    let c_view = state.hc_c.as_ref().unwrap().sub_offset(0, n_ctrl);

    // c = streams · W_fn + base
    gpu.hc_compute_control(streams, hc_fn, hc_base, &c_view,
        n_ctrl as i32, x_dim as i32)
        .map_err(|e| format!("hc_compute_control layer {layer_idx}: {e:?}"))?;

    // Apply α^pre/res/post scaling (paper eqs 3-5): rescales c so
    // c[i] = α[seg(i)] · (X · W) + (1 - α[seg(i)]) · base[i].
    // α small → static-bias-dominated (initial training behavior).
    let hc_scale = if is_attn {
        layer.hc_attn_scale.as_ref().unwrap()
    } else {
        layer.hc_ffn_scale.as_ref().unwrap()
    };
    gpu.hc_apply_alpha(&c_view, hc_scale, hc_base)
        .map_err(|e| format!("hc_apply_alpha layer {layer_idx}: {e:?}"))?;

    // Upstream V4F mixes layout: [pre(4), post(4), comb(16)] at
    // offsets [0, 4, 8]. The 24-element c[] follows the same ordering
    // since c = α·(hc_fn @ x · rsqrt) + base maintains row order.
    //
    // PRE (4-dim, sigmoid + eps): per-stream INPUT-mapping weights;
    //   y[d] = sum_h pre[h] * x[h, d]. Used by hc_input_map_4stream.
    //
    // Antirez ds4 (ds4.c:4202): `pre[i] = sigmoid(...) + DS4_HC_EPS`
    // where DS4_HC_EPS = 1e-6 (matches our cfg.hc_eps). The eps is tiny
    // but applied uniformly across all 4 streams — its omission shifts
    // every stream by zero in the limit so quality is unchanged here,
    // kept aligned for clarity.
    let pre_view = state.hc_c.as_ref().unwrap().sub_offset(0, 4);
    gpu.sigmoid_f32(&pre_view)
        .map_err(|e| format!("sigmoid pre layer {layer_idx}: {e:?}"))?;

    // POST (4-dim, 2·sigmoid): per-stream OUTPUT scaling. Antirez ds4
    // (hc_split_sinkhorn_one, ds4.c:4205-4208): `out[off] = 2.0 / (1.0 +
    // exp(-z))` where z = mix[off] * scale[1] + base[off]. The factor 2
    // is hardcoded; the learned per-layer scale[1] was already applied
    // via hc_apply_alpha above (c[4..8] = α[1]*mix + base[1]). So our
    // sigmoid(c) * post_scale matches antirez when post_scale=2.0.
    // Env override kept for diagnostics.
    let post_view = state.hc_c.as_ref().unwrap().sub_offset(4, 4);
    gpu.sigmoid_f32(&post_view)
        .map_err(|e| format!("sigmoid post layer {layer_idx}: {e:?}"))?;
    // Default 1.5: empirical optimum under mixed attention + YaRN. Antirez
    // hardcodes 2.0; the 0.5 delta is plausibly MQ2-Lloyd vs IQ2_XXS+Q2_K
    // quantization noise compensation. Env override kept for tuning.
    let post_scale: f32 = std::env::var("HIPFIRE_V4F_POST_SCALE")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1.5);
    gpu.scale_f32(&post_view, post_scale)
        .map_err(|e| format!("scale post layer {layer_idx}: {e:?}"))?;

    // COMB (16-dim → 4x4): cross-stream combining matrix, Sinkhorn-
    //   normalized to be doubly stochastic.
    let comb_view = state.hc_c.as_ref().unwrap().sub_offset(8, 16);
    gpu.hc_sinkhorn_4x4(&comb_view, cfg.hc_eps, cfg.hc_sinkhorn_iters as i32)
        .map_err(|e| format!("hc_sinkhorn_4x4 layer {layer_idx}: {e:?}"))?;

    // Input mapping: hc_x_in = sum_h pre[h] · streams[h, :]
    let hc_x_in = state.hc_x_in.as_ref().unwrap();
    gpu.hc_input_map_4stream(&pre_view, streams, hc_x_in, cfg.hidden_size as i32)
        .map_err(|e| format!("hc_input_map layer {layer_idx}: {e:?}"))?;

    Ok(())
}

/// Step 8 (attention block): full manifold-constrained Hyper-Connection mix.
///
/// Per DeepSeek_V4.pdf §2.2:
///   c     = α · (X · W_fn) + base                  [24]
///   Ã,B̃,C̃ = c[0..4], c[4..20], c[20..24]
///   A_l   = σ(Ã_l)                                  [4]    (input mapping)
///   B_l   = Sinkhorn(exp(B̃_l))                     [4,4]  (residual matrix)
///   C_l   = 2σ(C̃_l)                                 [4]    (output mapping)
///   x_in  = A_l · X_l                               [hidden]   (NOT YET — uses stream0)
///   y     = F_l(x_in)
///   X_l+1 = B_l · X_l + C_l · y
///
/// Currently `α · X · W_fn` is computed without the α scaling, and
/// the input mapping `A·X` is stubbed (transform input = stream 0 not
/// the weighted-sum across streams). These approximations make HC
/// numerically non-canonical but kept bounded by the doubly-stochastic
/// B and bounded-magnitude C.
fn hc_attn_mix(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let _ = weights;  // post/comb already in state.hc_c from mhc_pre
    let _ = layer_idx;
    let streams = state.residual_streams.as_ref().unwrap();
    let attn_out = state.attn_out.as_ref().unwrap();

    // Reuse the post and comb values that mhc_pre already computed
    // and saved into state.hc_c (with the correct α scaling applied
    // via hc_apply_alpha + sigmoid + sinkhorn). No need to recompute
    // — same input, same weights, no intervening writes to hc_c.
    let post_view = state.hc_c.as_ref().unwrap().sub_offset(4, 4);
    let comb_view = state.hc_c.as_ref().unwrap().sub_offset(8, 16);

    // X_{l+1} = comb · X_l + post · attn_out
    let streams_out = state.q.as_ref().unwrap();
    gpu.hc_mix_4stream(streams, &comb_view, &post_view, attn_out, streams_out,
        cfg.hidden_size as i32)
        .map_err(|e| format!("hc_mix_4stream layer: {e:?}"))?;

    let bytes = cfg.hc_mult * cfg.hidden_size * 4;
    gpu.memcpy_dtod_auto(&streams.buf, &streams_out.buf, bytes)
        .map_err(|e| format!("d2d hc_mix → streams: {e:?}"))?;
    Ok(())
}

/// Step 5 (attention block): Tail-only RoPE on Q and KV.
///
/// V4F's `qk_rope_head_dim = 64` of `head_dim = 512`. Only the last
/// 64 dims of each head's 512-dim vector get rotated; the first 448
/// are pass-through. Same rotation applies to KV's 512-dim vector
/// (treated as 1 head).
///
/// Uses `rope_tail_halfsplit_f32` with V4F's `rope_theta = 10000`.
/// YaRN correction dim: per-dim-pair index at which the high-vs-low
/// frequency split happens. Matches antirez ds4's `rope_yarn_corr_dim`.
fn rope_yarn_corr_dim(n_dims: u32, n_ctx_orig: u64, n_rot: f32, base: f32) -> f32 {
    n_dims as f32 * ((n_ctx_orig as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln())
        / (2.0 * base.ln())
}

/// Per-layer RoPE parameters: returns (freq_base, freq_scale, ext_factor,
/// attn_factor, corr_low, corr_high). Mirrors antirez's
/// `layer_rope_freq_base` / `layer_rope_freq_scale` + the attn_factor
/// cancellation in `rope_tail_layer_inplace`.
fn layer_rope_params(
    cfg: &DeepseekV4Config,
    compress_ratio: u32,
) -> (f32, f32, f32, f32, f32, f32) {
    let compressed = compress_ratio != 0;
    let freq_base = if compressed { cfg.compress_rope_theta } else { cfg.rope_theta };
    let scale_factor = cfg.rope_scaling_factor;
    let freq_scale = if compressed && scale_factor > 1.0 { 1.0 / scale_factor } else { 1.0 };
    let ext_factor = if compressed && scale_factor > 1.0 { 1.0 } else { 0.0 };
    // attn_factor: antirez pre-divides by (1+0.1*log(1/fs)) here so the
    // kernel's inner `mscale *= (1+0.1*log(1/fs))` cancels it back to 1.0
    // (see ds4.c:4769-4778). For dense (ext_factor=0) the kernel skips the
    // log multiplication, so attn_factor stays 1.0.
    let attn_factor = if ext_factor != 0.0 && freq_scale > 0.0 {
        1.0 / (1.0 + 0.1 * (1.0_f32 / freq_scale).ln())
    } else {
        1.0
    };
    let n_rot = cfg.qk_rope_head_dim as u32;
    let n_ctx_orig = cfg.rope_scaling_original_max_position_embeddings as u64;
    let beta_fast = cfg.rope_scaling_beta_fast as f32;
    let beta_slow = cfg.rope_scaling_beta_slow as f32;
    let (corr_low, corr_high) = if ext_factor != 0.0 {
        let lo = rope_yarn_corr_dim(n_rot, n_ctx_orig, beta_fast, freq_base).floor().max(0.0);
        let hi = rope_yarn_corr_dim(n_rot, n_ctx_orig, beta_slow, freq_base).ceil().min((n_rot - 1) as f32);
        (lo, hi)
    } else {
        (0.0, 0.0)
    };
    (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high)
}

fn apply_tail_rope(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    position: u32,
    layer_idx: usize,
) -> Result<(), String> {
    // Position is pre-loaded into `state.pos_array_device` at decode_step
    // entry (single htod for all layers). Slice the qk_pos slot for this
    // layer. Also seed the legacy `state.pos_buf` field so other code
    // paths that still read it (inverse RoPE on attn_out, indexer) work
    // unchanged — they get the SAME slice. The per-layer memcpy_htod is
    // gone, lifting it out of any future HIP-graph captured region.
    let pos_slice = pos_slot(state, layer_idx, 0)?;
    state.pos_buf = Some(pos_slice);
    let pos_buf = state.pos_buf.as_ref().unwrap();
    let _ = position; // silence unused; precompute_positions already used it

    let q  = state.q.as_ref().unwrap();
    let kv = state.kv.as_ref().unwrap();

    // V4F upstream (per antirez ds4 reference):
    //   compress_ratio == 0 (layers 0, 1, MTP): rope_theta = 10000, no YaRN
    //   compress_ratio  > 0 (layers 2..42):      compress_rope_theta = 160000,
    //                                            YaRN with scale_factor = 16
    // Pre-YaRN escape hatch: HIPFIRE_V4F_NO_YARN=1 reverts to the old
    // single-theta path (rope_theta=10000 everywhere) for direct A/B
    // comparison with prior tuning data.
    if std::env::var("HIPFIRE_V4F_NO_YARN").ok().as_deref() == Some("1") {
        gpu.rope_tail_interleaved(
            q, kv, pos_buf,
            cfg.num_attention_heads as i32,
            cfg.num_key_value_heads as i32,
            cfg.head_dim as i32,
            cfg.qk_rope_head_dim as i32,
            cfg.rope_theta,
        ).map_err(|e| format!("rope_tail_interleaved (no-yarn): {e:?}"))?;
        return Ok(());
    }

    let layer = weights.resolve_layer(layer_idx);
    let (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high) =
        layer_rope_params(cfg, layer.compress_ratio);

    gpu.rope_tail_yarn_interleaved(
        q, kv, pos_buf,
        cfg.num_attention_heads as i32,
        cfg.num_key_value_heads as i32,
        cfg.head_dim as i32,
        cfg.qk_rope_head_dim as i32,
        freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high,
        /*inverse=*/0,
    ).map_err(|e| format!("rope_tail_yarn_interleaved: {e:?}"))?;

    Ok(())
}

/// Step 4 (attention block): Joint KV projection.
///
/// V4F has `n_kv_heads = 1`, `head_dim = 512`, so the entire KV
/// stream per token is one 512-dim vector. `wkv` shape on disk is
/// `[512, 4096]` — a standard small GEMV producing 512 outputs from
/// 4096 hidden inputs.
///
/// Tail-only RoPE applies to the last `qk_rope_head_dim = 64` dims.
/// The leading 448 dims are pass-through.
///
/// Caller assumes `state.tmp` is still the FWHT-rotated post-RMSNorm
/// input from `q_lora` (gemv_mq4g256_prerotated doesn't modify x).
fn kv_joint(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let wkv = layer.wkv.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wkv missing"))?;
    let kv_norm = layer.kv_norm.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} kv_norm missing"))?;

    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    if state.kv.is_none() {
        state.kv = Some(gpu.alloc_tensor(&[kv_dim], DType::F32)
            .map_err(|e| format!("alloc kv: {e:?}"))?);
    }
    let tmp = state.tmp.as_ref().unwrap();
    let tmp_plain = state.tmp_plain.as_ref()
        .ok_or_else(|| "kv_joint: tmp_plain missing (q_lora must run first)".to_string())?;
    let kv  = state.kv.as_ref().unwrap();

    // wkv @ tmp → kv.  Dispatch on weight dtype (MQ4G256 / F32-from-F16).
    gemv_auto(gpu, wkv, tmp, tmp_plain, kv, kv_dim, cfg.hidden_size)?;

    // kv_norm RMSNorm in place (upstream V4F: `kv = self.kv_norm(kv)`
    // after wkv, before apply_rotary_emb). Was missing — likely
    // contributed to the SWA attractor since Q is rmsnormed but K=V
    // had arbitrary magnitudes.
    gpu.rmsnorm_f32(kv, kv_norm, kv, cfg.rms_norm_eps)
        .map_err(|e| format!("kv_norm rmsnorm layer {layer_idx}: {e:?}"))?;

    Ok(())
}

/// Step 3 (attention block): Q via Q-LoRA + tail-only RoPE.
///
///   x = state.tmp (post-RMSNorm)  -- but actually we should re-do
///       RMSNorm here with the fused-rotate variant so x is in the
///       FWHT-rotated domain that MQ4 expects.
///
///   Algorithm:
///     1. fused_rmsnorm_rotate_mq(stream0, attn_norm, x_rot, hidden, eps)
///        → x_rot [hidden] in MQ-rotated domain
///     2. gemv_mq4g256_prerotated(wq_a, x_rot, q_lat, q_lora_rank, hidden)
///        → q_lat [q_lora_rank=1024]
///     3. rotate_x_mq(q_lat, q_lat_rot, q_lora_rank)
///        → q_lat_rot [q_lora_rank]
///     4. gemv_mq4g256_prerotated(wq_b, q_lat_rot, q, n_heads*head_dim, q_lora_rank)
///        → q [n_heads*head_dim = 32768]
///     5. rope_tail_halfsplit on q (only last qk_rope_head_dim=64 of each
///        head's 512 dims)
///
/// Reuses `state.tmp` as the rotated post-RMSNorm input. Reuses
/// `state.q_lat`, `state.q_lat_rot`, `state.q`.
fn q_lora(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let attn_norm = layer.attn_norm.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} attn_norm missing"))?;
    let q_norm = layer.q_norm.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} q_norm missing"))?;
    let wq_a = layer.wq_a.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wq_a missing"))?;
    let wq_b = layer.wq_b.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wq_b missing"))?;
    let streams = state.residual_streams.as_ref().unwrap();

    // Allocate Q-LoRA state slots once.
    if state.q_lat.is_none() {
        state.q_lat = Some(gpu.alloc_tensor(&[cfg.q_lora_rank], DType::F32)
            .map_err(|e| format!("alloc q_lat: {e:?}"))?);
    }
    if state.q_lat_rot.is_none() {
        state.q_lat_rot = Some(gpu.alloc_tensor(&[cfg.q_lora_rank], DType::F32)
            .map_err(|e| format!("alloc q_lat_rot: {e:?}"))?);
    }
    if state.q.is_none() {
        // 2D shape so rmsnorm_f32 does per-head normalization.
        state.q = Some(gpu.alloc_tensor(
            &[cfg.num_attention_heads, cfg.head_dim], DType::F32)
            .map_err(|e| format!("alloc q: {e:?}"))?);
    }
    if state.q_head_ones.is_none() {
        let ones = vec![1.0f32; cfg.head_dim];
        state.q_head_ones = Some(gpu.upload_f32(&ones, &[cfg.head_dim])
            .map_err(|e| format!("upload q_head_ones: {e:?}"))?);
    }
    // Plain rmsnorm output for F16 non-expert GEMVs (antirez recipe).
    if state.tmp_plain.is_none() {
        state.tmp_plain = Some(gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
            .map_err(|e| format!("alloc tmp_plain: {e:?}"))?);
    }

    let hc_x_in = state.hc_x_in.as_ref().unwrap();
    let tmp = state.tmp.as_ref().unwrap();
    let tmp_plain = state.tmp_plain.as_ref().unwrap();
    let q_lat = state.q_lat.as_ref().unwrap();
    let q_lat_rot = state.q_lat_rot.as_ref().unwrap();
    let q = state.q.as_ref().unwrap();
    let q_head_ones = state.q_head_ones.as_ref().unwrap();
    let _ = streams;  // streams not used directly anymore; transform reads hc_x_in

    // 1. Fused RMSNorm + FWHT-rotate hc_x_in → tmp.
    gpu.fused_rmsnorm_rotate_mq(hc_x_in, attn_norm, tmp, cfg.hidden_size, cfg.rms_norm_eps)
        .map_err(|e| format!("fused_rmsnorm_rotate_mq layer {layer_idx}: {e:?}"))?;
    // 1b. Plain RMSNorm (no FWHT) → tmp_plain for F16 non-expert GEMVs.
    gpu.rmsnorm_f32(hc_x_in, attn_norm, tmp_plain, cfg.rms_norm_eps)
        .map_err(|e| format!("rmsnorm_f32 attn-side plain l{layer_idx}: {e:?}"))?;

    // 2. wq_a @ tmp → q_lat. M = q_lora_rank, K = hidden.
    gemv_auto(gpu, wq_a, tmp, tmp_plain, q_lat, cfg.q_lora_rank, cfg.hidden_size)?;

    // 2.5. Apply q_norm to the q-LoRA bottleneck (upstream V4F:
    //     `q = self.q_norm(self.wq_a(x))`). RMSNorm with q_norm weight.
    //     In-place: read q_lat, write q_lat.
    gpu.rmsnorm_f32(q_lat, q_norm, q_lat, cfg.rms_norm_eps)
        .map_err(|e| format!("q_norm rmsnorm layer {layer_idx}: {e:?}"))?;

    // 3. Rotate q_lat for the second GEMV.
    gpu.rotate_x_mq(q_lat, q_lat_rot, cfg.q_lora_rank)
        .map_err(|e| format!("rotate_x_mq q_lat layer {layer_idx}: {e:?}"))?;

    // 4. wq_b @ q_lat_rot → q. M = n_heads * head_dim, K = q_lora_rank.
    //    Use q_lat (un-rotated) for F16 path; q_lat_rot for MQ4 path.
    let q_total = cfg.num_attention_heads * cfg.head_dim;
    gemv_auto(gpu, wq_b, q_lat_rot, q_lat, q, q_total, cfg.q_lora_rank)?;

    // 4.5. Per-head RMSNorm of Q (upstream V4F:
    //     `q *= rsqrt(q.square().mean(-1, keepdim=True) + eps)`).
    //     Skip via HIPFIRE_V4F_SKIP_QHN=1 for bisecting.
    if std::env::var("HIPFIRE_V4F_SKIP_QHN").ok().as_deref() != Some("1") {
        gpu.rmsnorm_f32(q, q_head_ones, q, cfg.rms_norm_eps)
            .map_err(|e| format!("q per-head rmsnorm layer {layer_idx}: {e:?}"))?;
    }

    Ok(())
}

/// Step 1 of forward: embedding lookup + 4-stream residual init.
///
/// V4F's HC pattern starts with `[embed, 0, 0, 0]` — stream 0 gets
/// the embedding, streams 1-3 zero-initialised. Subsequent layers'
/// HC mixes propagate signal across all four streams.
///
/// Allocates `state.residual_streams` and `state.embed_scratch`
/// lazily on first call.
/// Per-layer slot count in `pos_array_*`. Layout per layer:
///   [0] qk_pos              = position
///   [1] main_comp_rope_pos  = mid-of-window  (depends on ratio + COMP_ROPE_POS env)
///   [2] indexer_comp_rope_pos = start-of-window
/// Used by the HIP-graphs-friendly position-array path (default in
/// `decode_step` since 2026-05-21). Direct-dispatch path uses the same
/// array but doesn't strictly need the stable host source.
pub(crate) const POS_SLOTS_PER_LAYER: usize = 3;

/// Compute per-layer derived positions and update `state.pos_array_*`.
///
/// Single host-to-device copy of the entire `[(num_layers + 1) * 3]` i32
/// array, with `pos_array_host` as the stable source pointer (required so
/// captured graph nodes re-read valid values on replay).
///
/// Reads env vars HIPFIRE_V4F_COMP_ROPE_POS once into a cache (TODO:
/// migrate to OnceLock once we settle on a fixed default).
pub(crate) fn precompute_positions(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    let total_slots = (cfg.num_hidden_layers + 1) * POS_SLOTS_PER_LAYER;

    // Lazy-alloc device buffer + stable host source (Box<[i32]>).
    if state.pos_array_device.is_none() {
        state.pos_array_device = Some(
            gpu.alloc_tensor(&[total_slots], DType::F32)
                .map_err(|e| format!("alloc pos_array_device: {e:?}"))?,
        );
    }
    if state.pos_array_host.is_none() {
        state.pos_array_host = Some(vec![0i32; total_slots].into_boxed_slice());
    }

    // Read env vars once per call (compile-time const + env cache would be
    // better; this is intentionally a single read, not 43× per-layer reads).
    let comp_rope_mode = std::env::var("HIPFIRE_V4F_COMP_ROPE_POS").ok();
    let comp_rope_mode = comp_rope_mode.as_deref();

    let pos_array_host = state.pos_array_host.as_mut().unwrap();
    for layer_idx in 0..=cfg.num_hidden_layers {
        // Per-layer compress_ratio; MTP layer (idx == num_hidden_layers)
        // has compress_ratio = 0 by V4F construction (no compressor).
        let ratio = if layer_idx < cfg.num_hidden_layers {
            cfg.compress_ratios[layer_idx] as usize
        } else {
            0
        };
        let base = layer_idx * POS_SLOTS_PER_LAYER;
        pos_array_host[base + 0] = position as i32;
        if ratio > 0 {
            let main_rope_pos: i32 = match comp_rope_mode {
                Some("end") => position as i32,
                Some("start") => ((position as usize) / ratio * ratio) as i32,
                _ => (((position as usize) / ratio * ratio) + ratio / 2) as i32,
            };
            // Indexer always uses start-of-window (matches the indexer Q
            // rotation derivation in compressor_forward).
            let indexer_rope_pos = ((position as usize) / ratio * ratio) as i32;
            pos_array_host[base + 1] = main_rope_pos;
            pos_array_host[base + 2] = indexer_rope_pos;
        } else {
            pos_array_host[base + 1] = 0;
            pos_array_host[base + 2] = 0;
        }
    }

    // ONE htod for the whole array. Source is the stable Box<[i32]> on
    // the heap, so captured graph nodes can re-read it on replay.
    let pos_array_device = state.pos_array_device.as_ref().unwrap();
    let bytes = unsafe {
        std::slice::from_raw_parts(
            pos_array_host.as_ptr() as *const u8,
            pos_array_host.len() * 4,
        )
    };
    gpu.memcpy_htod_auto(&pos_array_device.buf, bytes)
        .map_err(|e| format!("htod pos_array: {e:?}"))?;
    Ok(())
}

/// Slice the pos_array for a given layer's slot. Caller passes the slot
/// constant (0=qk_pos, 1=main_comp_rope, 2=indexer_comp_rope).
pub(crate) fn pos_slot(
    state: &DeepseekV4State,
    layer_idx: usize,
    slot: usize,
) -> Result<rdna_compute::GpuTensor, String> {
    let arr = state.pos_array_device.as_ref()
        .ok_or_else(|| "pos_array_device not initialised".to_string())?;
    let offset = layer_idx * POS_SLOTS_PER_LAYER + slot;
    Ok(arr.sub_offset(offset, 1))
}

fn init_residual_streams(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    token_id: u32,
) -> Result<(), String> {
    let token_embd = weights.token_embd.as_ref()
        .ok_or_else(|| "init_residual_streams: token_embd not uploaded".to_string())?;
    let hidden = cfg.hidden_size;
    let hc_mult = cfg.hc_mult;

    if state.embed_scratch.is_none() {
        state.embed_scratch = Some(
            gpu.alloc_tensor(&[hidden], DType::F32)
                .map_err(|e| format!("alloc embed_scratch: {e:?}"))?
        );
    }
    if state.residual_streams.is_none() {
        // Zero-init: alloc_tensor leaves memory uninitialized, but the
        // [embed, 0, 0, 0] init pattern relies on streams 1..hc_mult
        // being zero. `gpu.zeros` is the right primitive.
        let t = gpu.zeros(&[hc_mult, hidden], DType::F32)
            .map_err(|e| format!("alloc residual_streams: {e:?}"))?;
        state.residual_streams = Some(t);
    }
    if state.tmp.is_none() {
        state.tmp = Some(
            gpu.alloc_tensor(&[hidden], DType::F32)
                .map_err(|e| format!("alloc tmp: {e:?}"))?
        );
    }

    // Dequant + lookup token row → embed_scratch [hidden].
    let embed_scratch = state.embed_scratch.as_ref().unwrap();
    gpu.embedding_lookup_q8(token_embd, embed_scratch, token_id, hidden)
        .map_err(|e| format!("embedding_lookup_q8: {e:?}"))?;

    // **HC init**: per antirez ds4 `hc_from_plain_embedding` (ds4.c:4358),
    // ALL `hc_mult` streams are initialised with a COPY of the embedding,
    // NOT `[embed, 0, 0, 0]` as our prior comment claimed. The "0 streams"
    // pattern would have forced HC pre/post/comb to propagate signal from
    // stream 0 across layers, producing wrong magnitudes throughout the
    // forward.
    let streams = state.residual_streams.as_ref().unwrap();
    let bytes_per_stream = hidden * 4;  // F32 = 4 bytes
    for h in 0..hc_mult {
        let dst_view = streams.sub_offset(h * hidden, hidden);
        gpu.memcpy_dtod_auto(&dst_view.buf, &embed_scratch.buf, bytes_per_stream)
            .map_err(|e| format!("d2d copy stream {h}: {e:?}"))?;
    }

    Ok(())
}

/// Reusable per-call scratch for the batched-prefill driver.
///
/// **Phase B status (2026-05-18):** growing. Currently holds the
/// per-layer batched intermediates needed by `q_lora_batched`. Future
/// per-stage batched helpers (kv_joint_batched, attn_batched,
/// ffn_batched, hc_mix_batched) extend this struct as they land.
///
/// Sized to `max_batch` rows everywhere; tensors are reused across
/// per-chunk layer iterations.
pub struct PrefillBatchScratch {
    pub max_batch: usize,
    /// Embedding-lookup output `[max_batch, hidden]`. Source for the
    /// HC stream-broadcast init at chunk start.
    pub embed_batch: GpuTensor,
    /// HC residual streams `[max_batch, hc_mult, hidden]`. Lives across
    /// the full per-layer loop within a chunk.
    pub streams_batch: GpuTensor,
    /// Token-ids buffer feeding `embedding_lookup_q8_batched`.
    /// `[max_batch]` stored as F32 (same i32-in-F32-slots dtype-cosmetic
    /// pattern as qwen35's `pbs.tokens`).
    pub tokens: GpuTensor,
    /// FWHT-rotated attn_norm output `[max_batch, hidden]` feeding MQ4
    /// non-expert GEMMs.
    pub tmp_batch: GpuTensor,
    /// Plain attn_norm output `[max_batch, hidden]` feeding F32/Q8
    /// non-expert GEMMs.
    pub tmp_plain_batch: GpuTensor,
    /// Q-LoRA bottleneck `[max_batch, q_lora_rank]`. Reused: wq_a output
    /// → q_norm in place → fed to wq_b (after rotate into q_lat_rot_batch).
    pub q_lat_batch: GpuTensor,
    /// FWHT-rotated q_lat for the MQ4 wq_b path `[max_batch, q_lora_rank]`.
    pub q_lat_rot_batch: GpuTensor,
    /// Q output `[max_batch, n_heads, head_dim]`. wq_b output, then
    /// per-(batch, head) RMSNormed by `q_head_ones`.
    pub q_batch: GpuTensor,
    /// Per-head ones vector `[head_dim]` reused as the rmsnorm weight
    /// for the per-(batch, head) Q normalisation. Shared across batch.
    pub q_head_ones: GpuTensor,
    /// Joint KV `[max_batch, kv_dim]` where `kv_dim = n_kv_heads * head_dim`.
    /// wkv output, then kv_norm RMSNormed in place.
    pub kv_batch: GpuTensor,
    /// Per-batch absolute KV positions `[max_batch]` stored as F32 (the
    /// rope_tail_*_batched kernels read it as i32). Uploaded once per
    /// chunk: positions[b] = start_pos + b.
    pub positions: GpuTensor,
    /// HC control vector `[max_batch, 24]` — output of hc_compute_control
    /// _batched, in-place rescaled by hc_apply_alpha_batched, then split
    /// into pre/post/comb by hc_split_finalize_batched.
    pub hc_c_batch: GpuTensor,
    /// HC `pre` weights `[max_batch, hc_mult=4]`. Used by
    /// hc_input_map_4stream_batched and hc_mix_4stream_batched.
    pub hc_pre_batch: GpuTensor,
    /// HC `post` weights `[max_batch, hc_mult=4]`. Scale-multiplied
    /// sigmoid output. Feeds hc_mix_4stream_batched as the per-stream
    /// scale.
    pub hc_post_batch: GpuTensor,
    /// HC `comb` matrix `[max_batch, 4, 4]` — Sinkhorn-normalised to be
    /// doubly stochastic per batch row.
    pub hc_comb_batch: GpuTensor,
    /// HC transform input `[max_batch, hidden]` — output of mhc_pre's
    /// hc_input_map_4stream_batched. Feeds q_lora_batched / kv_joint
    /// _batched on the attention side, and the FFN gate/up on the FFN
    /// side.
    pub hc_x_in_batch: GpuTensor,
    /// Attention contribution `[max_batch, hidden]` produced by the
    /// attention block (Q · K → softmax → V → wo). Consumed by
    /// hc_attn_mix_batched as the `transform_out` argument.
    pub attn_out_batch: GpuTensor,
    /// FFN contribution `[max_batch, hidden]` produced by the routed
    /// MoE FFN. Consumed by hc_ffn_mix_batched as `transform_out`.
    pub ffn_out_batch: GpuTensor,
    /// Temporary `[max_batch, hc_mult, hidden]` for the hc_mix output
    /// before it's memcpy'd back into streams_batch. Mirrors the
    /// sequential path's reuse of `state.q` as the mix-output buffer.
    pub streams_out_batch: GpuTensor,
    /// Per-row visible SWA window `[max_batch, head_dim, swa_window]`
    /// produced by swa_visibility_stage_batched. V4F has K=V tied so
    /// one buffer feeds both the K and V args of the attention kernel.
    pub swa_staged_batch: GpuTensor,
    /// Per-row top-K K/V gather buffer `[max_batch, head_dim, topk_max]`
    /// produced by v4f_topk_kv_gather_batched (or the identity variant
    /// for ratio=128). Same K=V tied semantics.
    pub topk_staged_batch: GpuTensor,
    /// Per-row n_valid_swa array `[max_batch]` (i32-in-F32 slots).
    /// Tells v4f_attn_swa_topk_batched_f32 how many SWA entries are
    /// valid for each batch row.
    pub n_valid_swa_arr: GpuTensor,
    /// Per-row n_active_topk array `[max_batch]` (i32-in-F32 slots).
    pub n_active_topk_arr: GpuTensor,
    /// Raw attention output `[max_batch, n_heads, head_dim]`. Output of
    /// v4f_attn_swa_topk_batched_f32; consumed by inverse RoPE + the
    /// O-LoRA wo_a/wo_b projection chain.
    pub attn_out_raw_batch: GpuTensor,
    /// FWHT-rotated attn_out_raw `[max_batch, n_heads * head_dim]`.
    /// Input to per-group wo_a batched GEMV (MQ4 weight path).
    pub attn_out_raw_rot_batch: GpuTensor,
    /// wo_a output `[max_batch, n_groups, o_lora_rank]`.
    pub wo_a_out_batch: GpuTensor,
    /// FWHT-rotated wo_a output `[max_batch, n_groups * o_lora_rank]`.
    /// Input to wo_b batched GEMV (MQ4 weight path).
    pub wo_a_out_rot_batch: GpuTensor,
    // ── FFN-side scratch ──
    pub ffn_x_rot_batch: GpuTensor,        // [B, hidden]
    pub ffn_x_plain_batch: GpuTensor,      // [B, hidden]
    pub ffn_shared_gate_batch: GpuTensor,  // [B, IM]
    pub ffn_shared_up_batch: GpuTensor,    // [B, IM]
    pub ffn_shared_rot_batch: GpuTensor,   // [B, IM]
    pub moe_scores_batch: GpuTensor,       // [B, n_exp]
    pub moe_topk_indices_batch: GpuTensor, // [B, k_top]  i32-in-F32
    pub moe_topk_weights_batch: GpuTensor, // [B, k_top]
    pub moe_gate_batch: GpuTensor,         // [B, k_top, IM]
    pub moe_up_batch: GpuTensor,           // [B, k_top, IM]
    pub moe_rot_batch: GpuTensor,          // [B, k_top, IM]
    // ── Indexer chain scratch (Step-1 perf pass) ──
    pub idx_q_batch: GpuTensor,            // [B, idx_n_heads, idx_head_dim]
    pub idx_w_batch: GpuTensor,            // [B, idx_n_heads]
    pub idx_scores_batch: GpuTensor,       // [B, max_compressed]
    pub idx_topk_indices_batch: GpuTensor, // [B, index_topk]  i32-in-F32
    // ── Compressor batched-GEMV scratch (Phase 2.5 perf pass) ──
    // Holds the wkv / wgate compressor outputs across all B positions
    // so the GEMVs can be batched out of the per-position loop. Main
    // and indexer compressors get separate buffers because the proj_dim
    // differs (main=2*head_dim=1024, idx=2*idx_head_dim=256 for V4F).
    pub comp_main_kv_batch: GpuTensor,     // [B, 2*head_dim]
    pub comp_main_score_batch: GpuTensor,  // [B, 2*head_dim]
    pub comp_idx_kv_batch: GpuTensor,      // [B, 2*idx_head_dim]
    pub comp_idx_score_batch: GpuTensor,   // [B, 2*idx_head_dim]
    // ── Scatter-by-expert MoE sort outputs ──
    // Single counting-sort produces these per layer; the grouped MoE
    // GEMVs then read each expert weight slab once with cache reuse.
    pub moe_sorted_b: GpuTensor,           // [B * K_TOP] i32
    pub moe_sorted_krank: GpuTensor,       // [B * K_TOP] i32
    pub moe_sorted_expert: GpuTensor,      // [B * K_TOP] i32
    pub moe_expert_starts: GpuTensor,      // [n_exp + 1] i32
    // ── F16 staging for WMMA compressor GEMMs ──
    // F32 attention-norm output gets converted once per layer into
    // these buffers, then the four compressor GEMMs (wkv/wgate ×
    // main/idx) consume F16 inputs directly. Sized at [max_batch,
    // hidden] like tmp_batch; 1/2 the per-element bytes of F32.
    pub tmp_batch_f16:        GpuTensor,   // [B, hidden] F16 (stored as Raw)
    pub tmp_plain_batch_f16:  GpuTensor,   // [B, hidden] F16 (stored as Raw)
    /// Generic F16 staging buffer for WMMA HFQ4 GEMMs. Sized at
    /// `max_batch * max_dim * 2 bytes` so any batched GEMM input can
    /// be converted F32→F16 in place before dispatch. max_dim is the
    /// largest K dim across all V4F batched GEMM call sites — wo_b's
    /// K = groups × o_lora_rank for V4F (= 8 × 1024 = 8192).
    pub wmma_x_scratch_f16: GpuTensor,
    /// Per-compress-event RoPE positions buffer for the Phase A
    /// batched compressor pipeline. Sized [max_batch] F32 (i32-in-F32)
    /// since at B=64 ratio=4 we have at most 16 events per layer and
    /// ratio=128 has at most 1; total fits well under max_batch slots.
    /// Separate from `pbs.positions` (which holds the chunk's
    /// [batch_size] absolute positions and is read by the indexer).
    pub comp_positions: GpuTensor,
}

impl PrefillBatchScratch {
    /// Allocate scratch for prefill chunks of up to `max_batch` tokens.
    /// Sizes track the V4F config's hidden_size / q_lora_rank /
    /// num_attention_heads × head_dim. Prints per-field VRAM cost +
    /// running total to stderr — gated on `HIPFIRE_V4F_PBS_VRAM=1`
    /// (default off; on by default for the bench example).
    pub fn new(gpu: &mut Gpu, cfg: &DeepseekV4Config, max_batch: usize) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_rank = cfg.q_lora_rank;
        let n_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        let hc_mult = cfg.hc_mult;

        let log_vram = std::env::var("HIPFIRE_V4F_PBS_VRAM").ok().as_deref() == Some("1");
        let mut running_bytes: u64 = 0;
        if log_vram {
            eprintln!("PrefillBatchScratch::new max_batch={max_batch}");
            eprintln!("  V4F shape: hidden={hidden}, head_dim={head_dim}, n_heads={n_heads}, q_rank={q_rank}, hc_mult={hc_mult}, n_kv={}, swa={}, idx_topk={}, n_exp={}, k_top={}, IM={}",
                cfg.num_key_value_heads, cfg.sliding_window, cfg.index_topk,
                cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size);
        }
        let alloc = |gpu: &mut Gpu, shape: &[usize], label: &str, running: &mut u64, log: bool| -> Result<GpuTensor, String> {
            let nelem: usize = shape.iter().product();
            let bytes = (nelem * 4) as u64;
            *running += bytes;
            if log {
                eprintln!("  + {label:<28} {shape:?} = {} MB (cum {} MB)",
                    bytes / (1024 * 1024), *running / (1024 * 1024));
            }
            gpu.alloc_tensor(shape, DType::F32)
                .map_err(|e| format!("PrefillBatchScratch alloc {label}: {e:?}"))
        };
        let zeros = |gpu: &mut Gpu, shape: &[usize], label: &str, running: &mut u64, log: bool| -> Result<GpuTensor, String> {
            let nelem: usize = shape.iter().product();
            let bytes = (nelem * 4) as u64;
            *running += bytes;
            if log {
                eprintln!("  + {label:<28} {shape:?} = {} MB (cum {} MB)",
                    bytes / (1024 * 1024), *running / (1024 * 1024));
            }
            gpu.zeros(shape, DType::F32)
                .map_err(|e| format!("PrefillBatchScratch zeros {label}: {e:?}"))
        };

        let ones_host = vec![1.0f32; head_dim];
        let q_head_ones = gpu.upload_f32(&ones_host, &[head_dim])
            .map_err(|e| format!("PrefillBatchScratch upload q_head_ones: {e:?}"))?;

        let kv_dim = cfg.num_key_value_heads * head_dim;

        let r = &mut running_bytes;
        let out = Ok(Self {
            max_batch,
            embed_batch:     alloc(gpu, &[max_batch, hidden], "embed_batch", r, log_vram)?,
            streams_batch:   zeros(gpu, &[max_batch, hc_mult, hidden], "streams_batch", r, log_vram)?,
            tokens:          alloc(gpu, &[max_batch], "tokens", r, log_vram)?,
            tmp_batch:       alloc(gpu, &[max_batch, hidden], "tmp_batch", r, log_vram)?,
            tmp_plain_batch: alloc(gpu, &[max_batch, hidden], "tmp_plain_batch", r, log_vram)?,
            q_lat_batch:     alloc(gpu, &[max_batch, q_rank], "q_lat_batch", r, log_vram)?,
            q_lat_rot_batch: alloc(gpu, &[max_batch, q_rank], "q_lat_rot_batch", r, log_vram)?,
            q_batch:         alloc(gpu, &[max_batch, n_heads, head_dim], "q_batch", r, log_vram)?,
            q_head_ones,
            kv_batch:        alloc(gpu, &[max_batch, kv_dim], "kv_batch", r, log_vram)?,
            positions:       alloc(gpu, &[max_batch], "positions", r, log_vram)?,
            hc_c_batch:      alloc(gpu, &[max_batch, 24], "hc_c_batch", r, log_vram)?,
            hc_pre_batch:    alloc(gpu, &[max_batch, hc_mult], "hc_pre_batch", r, log_vram)?,
            hc_post_batch:   alloc(gpu, &[max_batch, hc_mult], "hc_post_batch", r, log_vram)?,
            hc_comb_batch:   alloc(gpu, &[max_batch, hc_mult, hc_mult], "hc_comb_batch", r, log_vram)?,
            hc_x_in_batch:   alloc(gpu, &[max_batch, hidden], "hc_x_in_batch", r, log_vram)?,
            attn_out_batch:  alloc(gpu, &[max_batch, hidden], "attn_out_batch", r, log_vram)?,
            ffn_out_batch:   alloc(gpu, &[max_batch, hidden], "ffn_out_batch", r, log_vram)?,
            streams_out_batch: alloc(gpu, &[max_batch, hc_mult, hidden], "streams_out_batch", r, log_vram)?,
            swa_staged_batch: alloc(gpu, &[max_batch, head_dim, cfg.sliding_window], "swa_staged_batch", r, log_vram)?,
            topk_staged_batch: alloc(gpu, &[max_batch, head_dim, cfg.index_topk], "topk_staged_batch", r, log_vram)?,
            n_valid_swa_arr: alloc(gpu, &[max_batch], "n_valid_swa_arr", r, log_vram)?,
            n_active_topk_arr: alloc(gpu, &[max_batch], "n_active_topk_arr", r, log_vram)?,
            attn_out_raw_batch: alloc(gpu, &[max_batch, n_heads, head_dim], "attn_out_raw_batch", r, log_vram)?,
            attn_out_raw_rot_batch: alloc(gpu, &[max_batch, n_heads * head_dim], "attn_out_raw_rot_batch", r, log_vram)?,
            wo_a_out_batch: alloc(gpu, &[max_batch, cfg.o_groups, cfg.o_lora_rank], "wo_a_out_batch", r, log_vram)?,
            wo_a_out_rot_batch: alloc(gpu, &[max_batch, cfg.o_groups * cfg.o_lora_rank], "wo_a_out_rot_batch", r, log_vram)?,
            ffn_x_rot_batch: alloc(gpu, &[max_batch, hidden], "ffn_x_rot_batch", r, log_vram)?,
            ffn_x_plain_batch: alloc(gpu, &[max_batch, hidden], "ffn_x_plain_batch", r, log_vram)?,
            ffn_shared_gate_batch: alloc(gpu, &[max_batch, cfg.moe_intermediate_size], "ffn_shared_gate_batch", r, log_vram)?,
            ffn_shared_up_batch: alloc(gpu, &[max_batch, cfg.moe_intermediate_size], "ffn_shared_up_batch", r, log_vram)?,
            ffn_shared_rot_batch: alloc(gpu, &[max_batch, cfg.moe_intermediate_size], "ffn_shared_rot_batch", r, log_vram)?,
            moe_scores_batch: alloc(gpu, &[max_batch, cfg.n_routed_experts], "moe_scores_batch", r, log_vram)?,
            moe_topk_indices_batch: alloc(gpu, &[max_batch, cfg.num_experts_per_tok], "moe_topk_indices_batch", r, log_vram)?,
            moe_topk_weights_batch: alloc(gpu, &[max_batch, cfg.num_experts_per_tok], "moe_topk_weights_batch", r, log_vram)?,
            moe_gate_batch: alloc(gpu, &[max_batch, cfg.num_experts_per_tok, cfg.moe_intermediate_size], "moe_gate_batch", r, log_vram)?,
            moe_up_batch: alloc(gpu, &[max_batch, cfg.num_experts_per_tok, cfg.moe_intermediate_size], "moe_up_batch", r, log_vram)?,
            moe_rot_batch: alloc(gpu, &[max_batch, cfg.num_experts_per_tok, cfg.moe_intermediate_size], "moe_rot_batch", r, log_vram)?,
            // Indexer-chain scratch. max_compressed default 2048 unless overridden via env.
            idx_q_batch: alloc(gpu, &[max_batch, cfg.index_n_heads, cfg.index_head_dim], "idx_q_batch", r, log_vram)?,
            idx_w_batch: alloc(gpu, &[max_batch, cfg.index_n_heads], "idx_w_batch", r, log_vram)?,
            idx_scores_batch: alloc(gpu, &[max_batch, 2048], "idx_scores_batch", r, log_vram)?,
            idx_topk_indices_batch: alloc(gpu, &[max_batch, cfg.index_topk], "idx_topk_indices_batch", r, log_vram)?,
            // Compressor batched-GEMV scratch — main coff=2, idx coff=2.
            comp_main_kv_batch: alloc(gpu, &[max_batch, 2 * head_dim], "comp_main_kv_batch", r, log_vram)?,
            comp_main_score_batch: alloc(gpu, &[max_batch, 2 * head_dim], "comp_main_score_batch", r, log_vram)?,
            comp_idx_kv_batch: alloc(gpu, &[max_batch, 2 * cfg.index_head_dim], "comp_idx_kv_batch", r, log_vram)?,
            comp_idx_score_batch: alloc(gpu, &[max_batch, 2 * cfg.index_head_dim], "comp_idx_score_batch", r, log_vram)?,
            // Scatter-by-expert MoE sort scratch.
            moe_sorted_b: alloc(gpu, &[max_batch * cfg.num_experts_per_tok], "moe_sorted_b", r, log_vram)?,
            moe_sorted_krank: alloc(gpu, &[max_batch * cfg.num_experts_per_tok], "moe_sorted_krank", r, log_vram)?,
            moe_sorted_expert: alloc(gpu, &[max_batch * cfg.num_experts_per_tok], "moe_sorted_expert", r, log_vram)?,
            moe_expert_starts: alloc(gpu, &[cfg.n_routed_experts + 1], "moe_expert_starts", r, log_vram)?,
            // F16 staging buffers: 2 bytes per element. Allocate as Raw
            // with byte-count shape so DType::size() == 1 stays consistent.
            tmp_batch_f16: {
                let nbytes = max_batch * hidden * 2;
                *r += nbytes as u64;
                if log_vram {
                    eprintln!("  + {:<28} [{}] = {} MB (cum {} MB) (F16 raw)",
                        "tmp_batch_f16", nbytes,
                        nbytes / (1024 * 1024), *r / (1024 * 1024));
                }
                let mut t = gpu.zeros(&[nbytes], DType::Raw)
                    .map_err(|e| format!("PBS alloc tmp_batch_f16: {e:?}"))?;
                t.dtype = DType::F16;
                t.shape = vec![max_batch, hidden];
                t
            },
            tmp_plain_batch_f16: {
                let nbytes = max_batch * hidden * 2;
                *r += nbytes as u64;
                if log_vram {
                    eprintln!("  + {:<28} [{}] = {} MB (cum {} MB) (F16 raw)",
                        "tmp_plain_batch_f16", nbytes,
                        nbytes / (1024 * 1024), *r / (1024 * 1024));
                }
                let mut t = gpu.zeros(&[nbytes], DType::Raw)
                    .map_err(|e| format!("PBS alloc tmp_plain_batch_f16: {e:?}"))?;
                t.dtype = DType::F16;
                t.shape = vec![max_batch, hidden];
                t
            },
            comp_positions: alloc(gpu, &[max_batch], "comp_positions", r, log_vram)?,
            wmma_x_scratch_f16: {
                // Cover the largest x-tensor size across all batched
                // WMMA call sites. wo_a's input is [B, G, per_group_in]
                // where per_group_in = (n_heads/n_groups) * head_dim —
                // can exceed `hidden` for V4F (G=8, per_group_in=4096
                // ⇒ G*per_group_in = 32768).
                let per_group_in = (n_heads / cfg.o_groups) * head_dim;
                let max_dim = cfg.o_groups * cfg.o_lora_rank;
                let max_dim = max_dim
                    .max(hidden)
                    .max(cfg.q_lora_rank)
                    .max(cfg.o_groups * per_group_in);
                let nbytes = max_batch * max_dim * 2;
                *r += nbytes as u64;
                if log_vram {
                    eprintln!("  + {:<28} [{}] = {} MB (cum {} MB) (F16 raw)",
                        "wmma_x_scratch_f16", nbytes,
                        nbytes / (1024 * 1024), *r / (1024 * 1024));
                }
                let mut t = gpu.zeros(&[nbytes], DType::Raw)
                    .map_err(|e| format!("PBS alloc wmma_x_scratch_f16: {e:?}"))?;
                t.dtype = DType::F16;
                t.shape = vec![max_batch, max_dim];
                t
            },
        });
        if log_vram {
            eprintln!("PrefillBatchScratch total: {} MB ({:.2} GB)",
                running_bytes / (1024 * 1024),
                (running_bytes as f64) / (1024.0 * 1024.0 * 1024.0));
        }
        out
    }
}

/// Batched twin of `hc_attn_mix` for Phase B2 chunk forward.
///
/// X_{l+1}[b] = comb[b] · X_l[b] + post[b] · attn_out[b]
/// where comb, post are from the latest mhc_pre_batched(is_attn=true) call.
/// The mix output is written into pbs.streams_out_batch, then copied
/// back into pbs.streams_batch (mirrors the sequential pattern of
/// staging into state.q before the d2d memcpy).
#[allow(dead_code)]
fn hc_attn_mix_batched(
    cfg: &DeepseekV4Config,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    batch_size: usize,
) -> Result<(), String> {
    gpu.hc_mix_4stream_batched(
        &pbs.streams_batch, &pbs.hc_comb_batch, &pbs.hc_post_batch,
        &pbs.attn_out_batch, &pbs.streams_out_batch,
        cfg.hidden_size as i32, batch_size as i32,
    ).map_err(|e| format!("hc_mix_4stream_batched (attn): {e:?}"))?;

    let bytes = batch_size * cfg.hc_mult * cfg.hidden_size * 4;
    gpu.memcpy_dtod_auto(&pbs.streams_batch.buf, &pbs.streams_out_batch.buf, bytes)
        .map_err(|e| format!("d2d streams_out → streams: {e:?}"))?;
    Ok(())
}

/// Pure-SWA batched attention block (compress_ratio == 0 layers).
///
/// Stages:
///   1. Lazy-alloc state._attention[L].swa_k / swa_v rings (per layer)
///   2. swa_visibility_stage_batched: pre-chunk ring + within-chunk
///      kv_batch → pbs.swa_staged_batch [B, head_dim, swa_window]
///   3. Upload per-batch n_valid_swa_arr
///   4. v4f_attn_swa_batched (K=V tied: pass swa_staged for both args)
///      → pbs.attn_out_raw_batch
///   5. Inverse tail RoPE (plain or YaRN per HIPFIRE_V4F_NO_YARN)
///   6. FWHT rotate attn_out_raw_batch → attn_out_raw_rot_batch
///   7. wo_per_group_batched_f32 → pbs.wo_a_out_batch (F32 wo_a only)
///   8. FWHT rotate wo_a_out_batch → wo_a_out_rot_batch
///   9. gemv_auto_batched_wmma(wo_b, ..., pbs.attn_out_batch, Some(&pbs.wmma_x_scratch_f16))
///   10. swa_ring_write_batched: advance ring with chunk's KVs
///
/// hc_attn_mix_batched is called by the chunk-forward caller after
/// this returns (mirrors the sequential ordering).
#[allow(dead_code)]
fn attention_block_batched_swa_only(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    layer_idx: usize,
    start_pos: u32,
    batch_size: usize,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let attn_sink = layer.attn_sink.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} attn_sink missing"))?;
    let wo_a = layer.wo_a.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wo_a missing"))?;
    let wo_b = layer.wo_b.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wo_b missing"))?;

    let n_kv = cfg.num_key_value_heads;
    let win = cfg.sliding_window;
    let n_heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim;
    let n_groups = cfg.o_groups;
    let o_lora_rank = cfg.o_lora_rank;
    let groups_o_lora = n_groups * o_lora_rank;

    // 1. Lazy-alloc the per-layer SWA ring (zero-init: pre-chunk
    //    visibility for early positions reads zero history correctly).
    {
        let attn = &mut state._attention[layer_idx];
        if attn.swa_k.is_none() {
            attn.swa_k = Some(gpu.zeros(&[n_kv, head_dim, win], DType::F32)
                .map_err(|e| format!("alloc swa_k l{layer_idx}: {e:?}"))?);
        }
        if attn.swa_v.is_none() {
            attn.swa_v = Some(gpu.zeros(&[n_kv, head_dim, win], DType::F32)
                .map_err(|e| format!("alloc swa_v l{layer_idx}: {e:?}"))?);
        }
    }
    let swa_k_ref = state._attention[layer_idx].swa_k.as_ref().unwrap().buf.as_ptr();
    let swa_v_ref = state._attention[layer_idx].swa_v.as_ref().unwrap().buf.as_ptr();
    let _ = (swa_k_ref, swa_v_ref); // borrow workaround handled below

    // 2. Stage per-batch SWA visibility window from pre-chunk ring +
    //    within-chunk kv_batch. V4F K=V tied so we only stage once and
    //    pass swa_staged_batch as both K and V args.
    {
        let swa_k = state._attention[layer_idx].swa_k.as_ref().unwrap();
        gpu.swa_visibility_stage_batched(
            swa_k, &pbs.kv_batch, &pbs.swa_staged_batch,
            start_pos as i32, win as i32, head_dim as i32, batch_size as i32,
        ).map_err(|e| format!("swa_visibility_stage_batched l{layer_idx}: {e:?}"))?;
    }

    // 3. Compute and upload per-batch n_valid_swa_arr.
    //    n_valid_swa[b] = min(start_pos + b + 1, swa_window).
    let n_valid_host: Vec<i32> = (0..batch_size)
        .map(|b| ((start_pos as usize + b + 1).min(win)) as i32)
        .collect();
    let n_valid_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(n_valid_host.as_ptr() as *const u8, batch_size * 4)
    };
    gpu.memcpy_htod_auto(&pbs.n_valid_swa_arr.buf, n_valid_bytes)
        .map_err(|e| format!("htod n_valid_swa_arr: {e:?}"))?;

    // 4. v4f_attn_swa_batched. o_groups passed through for ABI parity
    //    (unused inside the kernel).
    gpu.v4f_attn_swa_batched(
        &pbs.q_batch, &pbs.swa_staged_batch, &pbs.swa_staged_batch,
        attn_sink, &pbs.n_valid_swa_arr, &pbs.attn_out_raw_batch,
        n_heads as i32, head_dim as i32, n_groups as i32, win as i32,
        batch_size as i32,
    ).map_err(|e| format!("v4f_attn_swa_batched l{layer_idx}: {e:?}"))?;

    // 5. Inverse tail RoPE on attn_out_raw_batch.
    if std::env::var("HIPFIRE_V4F_SKIP_INV_ROPE").ok().as_deref() != Some("1") {
        if std::env::var("HIPFIRE_V4F_NO_YARN").ok().as_deref() == Some("1") {
            gpu.rope_tail_inverse_batched(
                &pbs.attn_out_raw_batch, &pbs.positions,
                n_heads as i32, head_dim as i32,
                cfg.qk_rope_head_dim as i32, cfg.rope_theta,
                batch_size as i32,
            ).map_err(|e| format!("rope_tail_inverse_batched l{layer_idx}: {e:?}"))?;
        } else {
            let (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high) =
                layer_rope_params(cfg, layer.compress_ratio);
            // n_heads_k=0: K already written + tail-rope'd at kv_joint
            // time; only un-rotate Q-tail-equivalents in attn_out.
            gpu.rope_tail_yarn_interleaved_batched(
                &pbs.attn_out_raw_batch, &pbs.attn_out_raw_batch, &pbs.positions,
                n_heads as i32, 0,
                head_dim as i32, cfg.qk_rope_head_dim as i32,
                freq_base, freq_scale, ext_factor, attn_factor,
                corr_low, corr_high,
                /*inverse=*/1, batch_size as i32,
            ).map_err(|e| format!("rope_tail_yarn_interleaved_batched (inv) l{layer_idx}: {e:?}"))?;
        }
    }

    // 6. FWHT rotate attn_out_raw_batch → attn_out_raw_rot_batch.
    //    The full [B, n_heads * head_dim] vector at once.
    gpu.rotate_x_mq_batched(
        &pbs.attn_out_raw_batch, &pbs.attn_out_raw_rot_batch,
        n_heads * head_dim, batch_size,
    ).map_err(|e| format!("rotate attn_out_raw_batch l{layer_idx}: {e:?}"))?;

    // 7. wo_a per-group batched.
    //    F32     → wo_per_group_batched_f32 (single launch).
    //    HFQ4G256→ wo_per_group_batched_hfq4g256 (single launch, MQ4 prerotated).
    //    Q8_0    → per-(B, G) sequential gemv_auto loop (still TODO).
    // Opt out via HIPFIRE_V4F_WO_A_BATCHED=0.
    let per_group_in = (n_heads / n_groups) * head_dim;
    let wo_a_batched = std::env::var("HIPFIRE_V4F_WO_A_BATCHED")
        .map(|s| s != "0").unwrap_or(true);
    match wo_a.dtype {
        DType::F32 => {
            gpu.wo_per_group_batched_f32(
                wo_a, &pbs.attn_out_raw_batch, &pbs.wo_a_out_batch,
                n_groups as i32, o_lora_rank as i32, per_group_in as i32,
                batch_size as i32,
            ).map_err(|e| format!("wo_per_group_batched_f32 l{layer_idx}: {e:?}"))?;
        }
        DType::Raw if wo_a_batched => {
            // MQ4G256 (HFQ4-packed weights, FWHT-rotated input).
            // WMMA route was measured slightly SLOWER (44.7 vs 45.6 tok/s
            // on Radeon 8060S) because the 16×16 WMMA tile shrinks the
            // workgroup count to (M/16, B/16, G) = ~2k workgroups, well
            // under the wave-saturation budget that the scalar per-output
            // grid hits. Keep WMMA wired as opt-in for other arches.
            let wo_a_wmma = std::env::var("HIPFIRE_V4F_WO_A_WMMA")
                .map(|s| s == "1").unwrap_or(false);
            if wo_a_wmma {
                let n_inputs = (batch_size * n_groups * per_group_in) as i64;
                gpu.convert_f32_to_f16(
                    &pbs.attn_out_raw_rot_batch, &pbs.wmma_x_scratch_f16, n_inputs,
                ).map_err(|e| format!("convert_f32_to_f16 wo_a l{layer_idx}: {e:?}"))?;
                gpu.wo_per_group_batched_hfq4g256_wmma(
                    wo_a, &pbs.wmma_x_scratch_f16, &pbs.wo_a_out_batch,
                    n_groups as i32, o_lora_rank as i32, per_group_in as i32,
                    batch_size as i32,
                ).map_err(|e| format!("wo_per_group_batched_hfq4g256_wmma l{layer_idx}: {e:?}"))?;
            } else {
                gpu.wo_per_group_batched_hfq4g256(
                    wo_a, &pbs.attn_out_raw_rot_batch, &pbs.wo_a_out_batch,
                    n_groups as i32, o_lora_rank as i32, per_group_in as i32,
                    batch_size as i32,
                ).map_err(|e| format!("wo_per_group_batched_hfq4g256 l{layer_idx}: {e:?}"))?;
            }
        }
        _ => {
            // Per-(batch, group) sequential fallback for Q8 and the
            // opt-out path. attn_out_raw_batch and attn_out_raw_rot_batch
            // are both [B, n_heads * head_dim] viewable as
            // [B, G, per_group_in]. wo_a is [G * o_lora_rank, per_group_in]
            // — per-group slice rows [g*o_lora_rank..].
            for b in 0..batch_size {
                for g in 0..n_groups {
                    let in_off = b * n_heads * head_dim + g * per_group_in;
                    let rot_view = pbs.attn_out_raw_rot_batch.sub_offset(in_off, per_group_in);
                    let raw_view = pbs.attn_out_raw_batch.sub_offset(in_off, per_group_in);
                    let out_off = b * n_groups * o_lora_rank + g * o_lora_rank;
                    let out_view = pbs.wo_a_out_batch.sub_offset(out_off, o_lora_rank);
                    let wo_a_view = match wo_a.dtype {
                        DType::Q8_0 => {
                            let per_g_bytes = (o_lora_rank * per_group_in / 32) * 34;
                            let mut v = wo_a.sub_offset(g * per_g_bytes, per_g_bytes);
                            v.dtype = DType::Q8_0;
                            v.shape = vec![o_lora_rank, per_group_in];
                            v
                        }
                        _ => {
                            // Treat as MQ4G256 (Raw). 136 bytes per 256 elements.
                            let per_g_bytes = (o_lora_rank * per_group_in / 256) * 136;
                            let mut v = wo_a.sub_offset(g * per_g_bytes, per_g_bytes);
                            v.shape = vec![o_lora_rank, per_group_in];
                            v
                        }
                    };
                    gemv_auto(gpu, &wo_a_view, &rot_view, &raw_view, &out_view,
                        o_lora_rank, per_group_in)?;
                }
            }
        }
    }

    // 8. FWHT rotate wo_a_out_batch → wo_a_out_rot_batch.
    gpu.rotate_x_mq_batched(
        &pbs.wo_a_out_batch, &pbs.wo_a_out_rot_batch,
        groups_o_lora, batch_size,
    ).map_err(|e| format!("rotate wo_a_out l{layer_idx}: {e:?}"))?;

    // 9. wo_b GEMV batched: wo_a_out_rot_batch → attn_out_batch.
    //    Standard non-block-diagonal GEMV; gemv_auto_batched handles
    //    F32/Q8/MQ4 dispatch.
    gemv_auto_batched_wmma(
        gpu, wo_b, &pbs.wo_a_out_rot_batch, &pbs.wo_a_out_batch,
        &pbs.attn_out_batch, cfg.hidden_size, groups_o_lora, batch_size, Some(&pbs.wmma_x_scratch_f16))?;

    // 10. Advance the SWA ring with this chunk's KVs for future steps.
    {
        let swa_k = state._attention[layer_idx].swa_k.as_ref().unwrap();
        let swa_v = state._attention[layer_idx].swa_v.as_ref().unwrap();
        gpu.swa_ring_write_batched_f32(
            &pbs.kv_batch, swa_k, n_kv as i32, head_dim as i32, win as i32,
            start_pos as i32, batch_size as i32,
        ).map_err(|e| format!("swa_ring_write_batched (k) l{layer_idx}: {e:?}"))?;
        gpu.swa_ring_write_batched_f32(
            &pbs.kv_batch, swa_v, n_kv as i32, head_dim as i32, win as i32,
            start_pos as i32, batch_size as i32,
        ).map_err(|e| format!("swa_ring_write_batched (v) l{layer_idx}: {e:?}"))?;
    }

    Ok(())
}

/// Mixed-attention batched dispatch (compress_ratio > 0 layers).
///
/// V4F's compressed layers attend jointly to (SWA window K/V) +
/// (top-K of compressed-K cache, gated by the indexer for ratio=4 or
/// the identity gather for ratio=128). The compressor + indexer
/// pipelines per-position are stateful (writes to kv_state ring,
/// conditional pool to main/indexer_kv_cache); we loop those
/// sequentially per batch position by temporarily swapping the
/// per-position state.* fields with sub_offset views into the
/// batched scratch buffers. The big-fish attention kernel still runs
/// in one batched launch.
///
/// Stages:
///   1. SWA visibility staging from pre-chunk ring + within-chunk kv_batch
///   2. For each batch position b:
///      a. Swap state.tmp / tmp_plain / q_lat / q_lat_rot to b's slice
///      b. compressor_forward(main, position=start_pos+b)
///      c. compressor_forward(indexer, position=start_pos+b) for ratio=4
///      d. indexer_forward → state._indexer[L].topk_idx_indices
///      e. Gather top-K K/V into pbs.topk_staged_batch[b] slot OR
///         identity-gather for ratio=128
///      f. Compute n_active_topk[b] = min(n_compressed, index_topk)
///   3. Upload n_valid_swa_arr + n_active_topk_arr
///   4. v4f_attn_swa_topk_batched_f32 (single launch over all batch rows)
///   5. Inverse RoPE batched
///   6. FWHT rotate attn_out_raw → attn_out_raw_rot
///   7. wo_per_group_batched_f32 (F32 wo_a only)
///   8. FWHT rotate wo_a_out → wo_a_out_rot
///   9. gemv_auto_batched_wmma(wo_b → attn_out_batch, Some(&pbs.wmma_x_scratch_f16))
///   10. swa_ring_write_batched
///
/// Errors out cleanly on non-F32 wo_a (Q8/MQ4 need separate per-group
/// batched kernels) or when the compressor/indexer state isn't
/// allocated.
#[allow(dead_code)]
fn attention_block_batched_mixed(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    layer_idx: usize,
    start_pos: u32,
    batch_size: usize,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let ratio = layer.compress_ratio as usize;
    assert!(ratio > 0, "attention_block_batched_mixed called on dense layer");

    let attn_sink = layer.attn_sink.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} attn_sink missing"))?;
    let wo_a = layer.wo_a.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wo_a missing"))?;
    let wo_b = layer.wo_b.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wo_b missing"))?;

    let n_kv = cfg.num_key_value_heads;
    let win = cfg.sliding_window;
    let n_heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim;
    let n_groups = cfg.o_groups;
    let o_lora_rank = cfg.o_lora_rank;
    let groups_o_lora = n_groups * o_lora_rank;
    let topk_max = cfg.index_topk;

    // Lazy-alloc SWA rings.
    {
        let attn = &mut state._attention[layer_idx];
        if attn.swa_k.is_none() {
            attn.swa_k = Some(gpu.zeros(&[n_kv, head_dim, win], DType::F32)
                .map_err(|e| format!("alloc swa_k l{layer_idx}: {e:?}"))?);
        }
        if attn.swa_v.is_none() {
            attn.swa_v = Some(gpu.zeros(&[n_kv, head_dim, win], DType::F32)
                .map_err(|e| format!("alloc swa_v l{layer_idx}: {e:?}"))?);
        }
        if attn.gathered_k.is_none() {
            attn.gathered_k = Some(gpu.zeros(&[n_kv, head_dim, topk_max], DType::F32)
                .map_err(|e| format!("alloc gathered_k l{layer_idx}: {e:?}"))?);
        }
    }

    // 1. Stage per-batch SWA visibility window.
    {
        let swa_k = state._attention[layer_idx].swa_k.as_ref().unwrap();
        gpu.swa_visibility_stage_batched(
            swa_k, &pbs.kv_batch, &pbs.swa_staged_batch,
            start_pos as i32, win as i32, head_dim as i32, batch_size as i32,
        ).map_err(|e| format!("swa_visibility_stage_batched l{layer_idx}: {e:?}"))?;
    }

    // 2a. Compressor commits (sequential per batch — stateful ring writes
    //     and conditional pools to indexer/main_kv_cache). MUST run before
    //     the batched indexer chain so n_filled[b] reflects all relevant
    //     commits. We swap state.* fields to point at per-row sub-views.
    let n_valid_host: Vec<i32> = (0..batch_size)
        .map(|b| ((start_pos as usize + b + 1).min(win)) as i32)
        .collect();

    // Snapshot the per-token state fields so we can restore after the loop.
    let orig_tmp = state.tmp.take();
    let orig_tmp_plain = state.tmp_plain.take();
    let orig_q_lat = state.q_lat.take();
    let orig_q_lat_rot = state.q_lat_rot.take();

    let hidden = cfg.hidden_size;
    let q_rank = cfg.q_lora_rank;
    let mut loop_err: Option<String> = None;

    // 2a-pre. Batched compressor GEMVs for the whole chunk. Collapses
    // 2 × batch_size sequential gemv_auto calls into ONE batched GEMM
    // per (wkv|wgate) × (main|indexer). Wires through to
    // compressor_forward_prebatched in the per-position loop below.
    // Opt out via HIPFIRE_V4F_COMP_BATCHED_GEMV=0.
    //
    // WMMA fast path: when all four compressor weights have F16-native
    // copies (`compressor_w{kv,gate}_f16` etc.), convert the F32 inputs
    // to F16 once and run gemm_f16_x_f16_wmma — measured 26× faster
    // than the F32 register-tiled path on V4F shapes (microbench).
    // Opt out via HIPFIRE_V4F_COMP_F16_WMMA=0.
    let comp_batched = std::env::var("HIPFIRE_V4F_COMP_BATCHED_GEMV")
        .map(|s| s != "0").unwrap_or(true);
    let comp_f16_wmma = std::env::var("HIPFIRE_V4F_COMP_F16_WMMA")
        .map(|s| s != "0").unwrap_or(true);
    let main_coff = 2; // ratio=4 has overlap=true; ratio=128 has coff=1 → wastes half the buf.
    let main_proj_dim = main_coff * head_dim;
    let idx_coff = 2;
    let idx_proj_dim = idx_coff * cfg.index_head_dim;
    if comp_batched {
        let comp_wkv = layer.compressor_wkv.as_ref()
            .ok_or_else(|| format!("comp_wkv l{layer_idx}"))?;
        let comp_wgate = layer.compressor_wgate.as_ref()
            .ok_or_else(|| format!("comp_wgate l{layer_idx}"))?;
        let real_main_proj = if ratio == 4 { 2 * head_dim } else { head_dim };
        // WMMA route requires both main + idx (when ratio=4) F16 weights
        // and works on F16 inputs.
        let wkv_f16 = layer.compressor_wkv_f16.as_ref();
        let wgate_f16 = layer.compressor_wgate_f16.as_ref();
        let idx_wkv_f16 = layer.indexer_compressor_wkv_f16.as_ref();
        let idx_wgate_f16 = layer.indexer_compressor_wgate_f16.as_ref();
        let have_idx_f16 = ratio != 4 || (idx_wkv_f16.is_some() && idx_wgate_f16.is_some());
        let use_wmma = comp_f16_wmma
            && wkv_f16.is_some() && wgate_f16.is_some() && have_idx_f16;
        if use_wmma {
            // Stage F32 → F16 inputs once per layer.
            let n_inputs = (batch_size * hidden) as i64;
            gpu.convert_f32_to_f16(&pbs.tmp_batch, &pbs.tmp_batch_f16, n_inputs)
                .map_err(|e| format!("convert_f32_to_f16 tmp l{layer_idx}: {e:?}"))?;
            gpu.convert_f32_to_f16(&pbs.tmp_plain_batch, &pbs.tmp_plain_batch_f16, n_inputs)
                .map_err(|e| format!("convert_f32_to_f16 tmp_plain l{layer_idx}: {e:?}"))?;
            // V4F compressor uses FWHT-rotated input (tmp_batch) when the
            // weight is MQ4-style, and plain input (tmp_plain_batch) when
            // F16/F32. We're on the F16 path → tmp_plain_batch_f16.
            gpu.gemm_f16_x_f16_wmma(
                wkv_f16.unwrap(), &pbs.tmp_plain_batch_f16, &pbs.comp_main_kv_batch,
                real_main_proj, hidden, batch_size,
            ).map_err(|e| format!("gemm_f16_wmma comp_wkv l{layer_idx}: {e:?}"))?;
            gpu.gemm_f16_x_f16_wmma(
                wgate_f16.unwrap(), &pbs.tmp_plain_batch_f16, &pbs.comp_main_score_batch,
                real_main_proj, hidden, batch_size,
            ).map_err(|e| format!("gemm_f16_wmma comp_wgate l{layer_idx}: {e:?}"))?;
            if ratio == 4 {
                gpu.gemm_f16_x_f16_wmma(
                    idx_wkv_f16.unwrap(), &pbs.tmp_plain_batch_f16, &pbs.comp_idx_kv_batch,
                    idx_proj_dim, hidden, batch_size,
                ).map_err(|e| format!("gemm_f16_wmma idx_wkv l{layer_idx}: {e:?}"))?;
                gpu.gemm_f16_x_f16_wmma(
                    idx_wgate_f16.unwrap(), &pbs.tmp_plain_batch_f16, &pbs.comp_idx_score_batch,
                    idx_proj_dim, hidden, batch_size,
                ).map_err(|e| format!("gemm_f16_wmma idx_wgate l{layer_idx}: {e:?}"))?;
            }
        } else {
            gemv_auto_batched_wmma(
                gpu, comp_wkv, &pbs.tmp_batch, &pbs.tmp_plain_batch,
                &pbs.comp_main_kv_batch, real_main_proj, hidden, batch_size, Some(&pbs.wmma_x_scratch_f16))?;
            gemv_auto_batched_wmma(
                gpu, comp_wgate, &pbs.tmp_batch, &pbs.tmp_plain_batch,
                &pbs.comp_main_score_batch, real_main_proj, hidden, batch_size, Some(&pbs.wmma_x_scratch_f16))?;
            if ratio == 4 {
                let idx_wkv = layer.indexer_compressor_wkv.as_ref()
                    .ok_or_else(|| format!("idx_comp_wkv l{layer_idx}"))?;
                let idx_wgate = layer.indexer_compressor_wgate.as_ref()
                    .ok_or_else(|| format!("idx_comp_wgate l{layer_idx}"))?;
                gemv_auto_batched_wmma(
                    gpu, idx_wkv, &pbs.tmp_batch, &pbs.tmp_plain_batch,
                    &pbs.comp_idx_kv_batch, idx_proj_dim, hidden, batch_size, Some(&pbs.wmma_x_scratch_f16))?;
                gemv_auto_batched_wmma(
                    gpu, idx_wgate, &pbs.tmp_batch, &pbs.tmp_plain_batch,
                    &pbs.comp_idx_score_batch, idx_proj_dim, hidden, batch_size, Some(&pbs.wmma_x_scratch_f16))?;
            }
        }
    }
    // The pre-batched buffers are stored at stride main_proj_dim (=1024)
    // even when ratio=128 (proj_dim=512). For ratio=128 the second half
    // of each [B, 1024] slot is unused but still strided. That matches
    // the alloc but means the per-position offset uses the real proj_dim.
    let main_view_proj = if ratio == 4 { main_proj_dim } else { head_dim };

    // PHASE A: batched commit/compress for the whole chunk in one call
    // per (main, indexer) per layer. Replaces the per-batch loop when
    // start_pos % ratio == 0 (aligned chunk). Opt out via
    // HIPFIRE_V4F_COMP_FULLY_BATCHED=0.
    let comp_fully_batched = comp_batched
        && (start_pos as usize) % ratio == 0
        && std::env::var("HIPFIRE_V4F_COMP_FULLY_BATCHED")
            .map(|s| s != "0").unwrap_or(true);

    if comp_fully_batched {
        if let Err(e) = compressor_forward_batched(
            cfg, weights, state, pbs, gpu, layer_idx, start_pos,
            batch_size, /*is_indexer=*/false,
        ) {
            loop_err = Some(format!("compressor_forward_batched(main) l{layer_idx}: {e}"));
        }
        if loop_err.is_none() && ratio == 4 {
            if let Err(e) = compressor_forward_batched(
                cfg, weights, state, pbs, gpu, layer_idx, start_pos,
                batch_size, /*is_indexer=*/true,
            ) {
                loop_err = Some(format!("compressor_forward_batched(idx) l{layer_idx}: {e}"));
            }
        }
    } else {
        for b in 0..batch_size {
            let pos = start_pos + b as u32;
            state.tmp = Some(pbs.tmp_batch.sub_offset(b * hidden, hidden));
            state.tmp_plain = Some(pbs.tmp_plain_batch.sub_offset(b * hidden, hidden));
            state.q_lat = Some(pbs.q_lat_batch.sub_offset(b * q_rank, q_rank));
            state.q_lat_rot = Some(pbs.q_lat_rot_batch.sub_offset(b * q_rank, q_rank));

            let cf_res = if comp_batched {
                // The pre-batched [B, main_view_proj] buffer is laid out
                // contiguous-per-row with stride main_view_proj at the
                // wkv/wgate gemv_auto_batched call.
                let _ = main_proj_dim; // silence warning when ratio=128
                compressor_forward_prebatched(
                    cfg, weights, state, gpu, layer_idx, pos,
                    /*is_indexer=*/false,
                    &pbs.comp_main_kv_batch, &pbs.comp_main_score_batch, b,
                )
            } else {
                let tmp_view = state.tmp.as_ref().unwrap().sub_offset(0, hidden);
                compressor_forward(
                    cfg, weights, state, gpu, layer_idx,
                    &tmp_view, pos, /*is_indexer=*/false,
                )
            };
            if let Err(e) = cf_res {
                loop_err = Some(format!("compressor_forward(main) b={b} l{layer_idx}: {e}"));
                break;
            }
            if ratio == 4 {
                let cf_res2 = if comp_batched {
                    compressor_forward_prebatched(
                        cfg, weights, state, gpu, layer_idx, pos,
                        /*is_indexer=*/true,
                        &pbs.comp_idx_kv_batch, &pbs.comp_idx_score_batch, b,
                    )
                } else {
                    let tmp_view2 = state.tmp.as_ref().unwrap().sub_offset(0, hidden);
                    compressor_forward(
                        cfg, weights, state, gpu, layer_idx,
                        &tmp_view2, pos, /*is_indexer=*/true,
                    )
                };
                if let Err(e) = cf_res2 {
                    loop_err = Some(format!("compressor_forward(idx) b={b} l{layer_idx}: {e}"));
                    break;
                }
            }
        }
    }
    let _ = main_view_proj;

    // Restore per-token state fields before any potential early-return.
    state.tmp = orig_tmp;
    state.tmp_plain = orig_tmp_plain;
    state.q_lat = orig_q_lat;
    state.q_lat_rot = orig_q_lat_rot;
    if let Some(e) = loop_err {
        return Err(e);
    }

    // 2b. Batched indexer chain (ratio == 4 only) OR batched identity gather
    //     (ratio == 128). Replaces the per-batch indexer_forward + gather
    //     loop with one batched call per stage.
    let mut n_active_host: Vec<i32> = vec![0; batch_size];
    if ratio == 4 {
        let wq_b_idx = layer.indexer_wq_b.as_ref()
            .ok_or_else(|| format!("idx wq_b l{layer_idx}"))?;
        let weights_proj = layer.indexer_weights_proj.as_ref()
            .ok_or_else(|| format!("idx weights_proj l{layer_idx}"))?;
        let h_idx = cfg.index_n_heads;
        let d_idx = cfg.index_head_dim;

        // Per-batch n_filled = (start_pos+b+1)/ratio, clamped.
        // n_max across batch = max value, used as kernel's per-batch cap.
        let max_compressed: usize = std::env::var("HIPFIRE_V4F_MAX_COMPRESS_POS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
        let n_per_batch_host: Vec<i32> = (0..batch_size).map(|b| {
            (((start_pos as usize) + b + 1) / ratio).min(max_compressed) as i32
        }).collect();
        let n_max_chunk = *n_per_batch_host.iter().max().unwrap_or(&0) as usize;
        if n_max_chunk == 0 {
            // No commits yet — nothing to score/gather. n_active_topk stays 0.
        } else {
            // Upload n_per_batch via the existing n_active_topk_arr buffer
            // (repurposed temporarily — we'll overwrite it below with the
            // actual k_active values).
            let np_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(n_per_batch_host.as_ptr() as *const u8, batch_size * 4)
            };
            gpu.memcpy_htod_auto(&pbs.n_active_topk_arr.buf, np_bytes)
                .map_err(|e| format!("htod n_per_batch: {e:?}"))?;

            // wq_b_idx GEMV batched: q_lat_rot_batch → q_idx_batch.
            gemv_auto_batched_wmma(
                gpu, wq_b_idx, &pbs.q_lat_rot_batch, &pbs.q_lat_batch,
                &pbs.idx_q_batch, h_idx * d_idx, q_rank, batch_size, Some(&pbs.wmma_x_scratch_f16))?;

            // Tail RoPE on q_idx_batch with compress_rope_theta.
            gpu.rope_tail_interleaved_batched(
                &pbs.idx_q_batch, &pbs.idx_q_batch, &pbs.positions,
                h_idx as i32, 0, d_idx as i32,
                cfg.qk_rope_head_dim as i32, cfg.compress_rope_theta,
                batch_size as i32,
            ).map_err(|e| format!("rope_tail_batched idx l{layer_idx}: {e:?}"))?;

            // weights_proj GEMV batched: tmp_batch → idx_w_batch.
            gemv_auto_batched_wmma(
                gpu, weights_proj, &pbs.tmp_batch, &pbs.tmp_plain_batch,
                &pbs.idx_w_batch, h_idx, hidden, batch_size, Some(&pbs.wmma_x_scratch_f16))?;

            // Batched scoring. Pass the SCORE BUFFER STRIDE (max_compressed,
            // = the allocated row stride of pbs.idx_scores_batch), not the
            // chunk's n_max_chunk. The kernel writes scores[b * stride + n];
            // slots with n >= n_per_batch[b] get -inf and slots with
            // n >= n_max_chunk read uninit K_cache data but also get -inf
            // (since n_per_batch[b] ≤ n_max_chunk ≤ n).
            let kv_cache = state._indexer[layer_idx].indexer_kv_cache.as_ref()
                .ok_or_else(|| "indexer_kv_cache missing".to_string())?;
            gpu.indexer_relu_score_batched_f32(
                &pbs.idx_q_batch, kv_cache, &pbs.idx_w_batch,
                &pbs.n_active_topk_arr,  // reuse buffer: holds n_per_batch right now
                &pbs.idx_scores_batch,
                h_idx as i32, d_idx as i32, max_compressed as i32, batch_size as i32,
            ).map_err(|e| format!("indexer_relu_score_batched l{layer_idx}: {e:?}"))?;

            // Batched top-K. n_stride = max_compressed (storage),
            // n_iter = n_max_chunk (actual range with valid scores),
            // k_stride = topk_max (storage), k_fill = min(topk_max,
            // n_max_chunk). The bound matters a LOT — at low context
            // n_max_chunk ≈ 8 vs max_compressed = 2048, which is
            // ~100× iteration savings.
            let k_fill = topk_max.min(n_max_chunk);
            gpu.indexer_top_k_batched(
                &pbs.idx_scores_batch, &pbs.idx_topk_indices_batch,
                /*n_idx_heads=*/1,
                max_compressed as i32, n_max_chunk as i32,
                topk_max as i32, k_fill as i32,
                batch_size as i32,
            ).map_err(|e| format!("indexer_top_k_batched l{layer_idx}: {e:?}"))?;

            // Batched gather: top-K K/V → pbs.topk_staged_batch. Pass
            // K=topk_max (storage stride); -1 indices write zeros.
            let main_kv_cache = state._indexer[layer_idx].main_kv_cache.as_ref()
                .ok_or_else(|| "main_kv_cache missing".to_string())?;
            gpu.v4f_topk_kv_gather_batched_f32(
                main_kv_cache, &pbs.idx_topk_indices_batch, &pbs.topk_staged_batch,
                topk_max as i32, head_dim as i32, n_max_chunk as i32,
                topk_max as i32, 0, /*scale=*/1.0, batch_size as i32,
            ).map_err(|e| format!("v4f_topk_kv_gather_batched l{layer_idx}: {e:?}"))?;

            // n_active_topk[b] = min(topk_max, n_per_batch[b]) — top-K
            // returned -1 sentinels past n_per_batch[b], and gather wrote
            // zeros there. Cap attention's visible-slot count to the
            // actual valid range per batch row.
            for b in 0..batch_size {
                n_active_host[b] = topk_max.min(n_per_batch_host[b] as usize) as i32;
            }
        }
    } else {
        // ratio == 128: identity gather, no indexer. Per-batch n_compressed.
        let max_compressed: usize = std::env::var("HIPFIRE_V4F_MAX_COMPRESS_POS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
        let max_n_compressed = (((start_pos as usize) + batch_size) / ratio)
            .min(max_compressed).min(topk_max);
        if max_n_compressed > 0 {
            let main_kv_cache = state._indexer[layer_idx].main_kv_cache.as_ref()
                .ok_or_else(|| "main_kv_cache missing".to_string())?;
            gpu.v4f_topk_kv_gather_identity_batched_f32(
                main_kv_cache, &pbs.topk_staged_batch,
                max_n_compressed as i32, head_dim as i32, topk_max as i32,
                batch_size as i32,
            ).map_err(|e| format!("v4f_topk_kv_gather_identity_batched l{layer_idx}: {e:?}"))?;
            for b in 0..batch_size {
                let n_b = (((start_pos as usize) + b + 1) / ratio)
                    .min(max_compressed).min(topk_max);
                n_active_host[b] = n_b as i32;
            }
        }
    }

    // 3. Upload per-batch valid-counts.
    let n_valid_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(n_valid_host.as_ptr() as *const u8, batch_size * 4)
    };
    gpu.memcpy_htod_auto(&pbs.n_valid_swa_arr.buf, n_valid_bytes)
        .map_err(|e| format!("htod n_valid_swa_arr: {e:?}"))?;
    let n_active_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(n_active_host.as_ptr() as *const u8, batch_size * 4)
    };
    gpu.memcpy_htod_auto(&pbs.n_active_topk_arr.buf, n_active_bytes)
        .map_err(|e| format!("htod n_active_topk_arr: {e:?}"))?;

    // 4. Batched joint-softmax attention over SWA + topK + sink.
    gpu.v4f_attn_swa_topk_batched_f32(
        &pbs.q_batch,
        &pbs.swa_staged_batch, &pbs.swa_staged_batch,    // K=V tied
        &pbs.topk_staged_batch, &pbs.topk_staged_batch,
        attn_sink,
        &pbs.n_valid_swa_arr, &pbs.n_active_topk_arr,
        &pbs.attn_out_raw_batch,
        n_heads as i32, head_dim as i32,
        win as i32, topk_max as i32,
        batch_size as i32,
    ).map_err(|e| format!("v4f_attn_swa_topk_batched l{layer_idx}: {e:?}"))?;

    // 5. Inverse RoPE.
    if std::env::var("HIPFIRE_V4F_SKIP_INV_ROPE").ok().as_deref() != Some("1") {
        if std::env::var("HIPFIRE_V4F_NO_YARN").ok().as_deref() == Some("1") {
            gpu.rope_tail_inverse_batched(
                &pbs.attn_out_raw_batch, &pbs.positions,
                n_heads as i32, head_dim as i32,
                cfg.qk_rope_head_dim as i32, cfg.rope_theta,
                batch_size as i32,
            ).map_err(|e| format!("rope_tail_inverse_batched l{layer_idx}: {e:?}"))?;
        } else {
            let (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high) =
                layer_rope_params(cfg, layer.compress_ratio);
            gpu.rope_tail_yarn_interleaved_batched(
                &pbs.attn_out_raw_batch, &pbs.attn_out_raw_batch, &pbs.positions,
                n_heads as i32, 0,
                head_dim as i32, cfg.qk_rope_head_dim as i32,
                freq_base, freq_scale, ext_factor, attn_factor,
                corr_low, corr_high,
                /*inverse=*/1, batch_size as i32,
            ).map_err(|e| format!("rope_tail_yarn_inv_batched l{layer_idx}: {e:?}"))?;
        }
    }

    // 6. FWHT rotate attn_out_raw_batch → attn_out_raw_rot_batch.
    gpu.rotate_x_mq_batched(
        &pbs.attn_out_raw_batch, &pbs.attn_out_raw_rot_batch,
        n_heads * head_dim, batch_size,
    ).map_err(|e| format!("rotate attn_out_raw l{layer_idx}: {e:?}"))?;

    // 7. wo_a per-group batched.
    //    F32     → wo_per_group_batched_f32 (single launch).
    //    HFQ4G256→ wo_per_group_batched_hfq4g256 (single launch).
    //    Q8_0    → per-(B, G) sequential gemv_auto loop.
    // Opt out via HIPFIRE_V4F_WO_A_BATCHED=0.
    let per_group_in = (n_heads / n_groups) * head_dim;
    let wo_a_batched = std::env::var("HIPFIRE_V4F_WO_A_BATCHED")
        .map(|s| s != "0").unwrap_or(true);
    match wo_a.dtype {
        DType::F32 => {
            gpu.wo_per_group_batched_f32(
                wo_a, &pbs.attn_out_raw_batch, &pbs.wo_a_out_batch,
                n_groups as i32, o_lora_rank as i32, per_group_in as i32,
                batch_size as i32,
            ).map_err(|e| format!("wo_per_group_batched_f32 l{layer_idx}: {e:?}"))?;
        }
        DType::Raw if wo_a_batched => {
            gpu.wo_per_group_batched_hfq4g256(
                wo_a, &pbs.attn_out_raw_rot_batch, &pbs.wo_a_out_batch,
                n_groups as i32, o_lora_rank as i32, per_group_in as i32,
                batch_size as i32,
            ).map_err(|e| format!("wo_per_group_batched_hfq4g256 l{layer_idx}: {e:?}"))?;
        }
        _ => {
            for b in 0..batch_size {
                for g in 0..n_groups {
                    let in_off = b * n_heads * head_dim + g * per_group_in;
                    let rot_view = pbs.attn_out_raw_rot_batch.sub_offset(in_off, per_group_in);
                    let raw_view = pbs.attn_out_raw_batch.sub_offset(in_off, per_group_in);
                    let out_off = b * n_groups * o_lora_rank + g * o_lora_rank;
                    let out_view = pbs.wo_a_out_batch.sub_offset(out_off, o_lora_rank);
                    let wo_a_view = match wo_a.dtype {
                        DType::Q8_0 => {
                            let per_g_bytes = (o_lora_rank * per_group_in / 32) * 34;
                            let mut v = wo_a.sub_offset(g * per_g_bytes, per_g_bytes);
                            v.dtype = DType::Q8_0;
                            v.shape = vec![o_lora_rank, per_group_in];
                            v
                        }
                        _ => {
                            let per_g_bytes = (o_lora_rank * per_group_in / 256) * 136;
                            let mut v = wo_a.sub_offset(g * per_g_bytes, per_g_bytes);
                            v.shape = vec![o_lora_rank, per_group_in];
                            v
                        }
                    };
                    gemv_auto(gpu, &wo_a_view, &rot_view, &raw_view, &out_view,
                        o_lora_rank, per_group_in)?;
                }
            }
        }
    }

    // 8. FWHT rotate wo_a_out → wo_a_out_rot.
    gpu.rotate_x_mq_batched(
        &pbs.wo_a_out_batch, &pbs.wo_a_out_rot_batch,
        groups_o_lora, batch_size,
    ).map_err(|e| format!("rotate wo_a_out l{layer_idx}: {e:?}"))?;

    // 9. wo_b GEMV batched.
    gemv_auto_batched_wmma(
        gpu, wo_b, &pbs.wo_a_out_rot_batch, &pbs.wo_a_out_batch,
        &pbs.attn_out_batch, cfg.hidden_size, groups_o_lora, batch_size, Some(&pbs.wmma_x_scratch_f16))?;

    // 10. Advance the SWA ring.
    {
        let swa_k = state._attention[layer_idx].swa_k.as_ref().unwrap();
        let swa_v = state._attention[layer_idx].swa_v.as_ref().unwrap();
        gpu.swa_ring_write_batched_f32(
            &pbs.kv_batch, swa_k, n_kv as i32, head_dim as i32, win as i32,
            start_pos as i32, batch_size as i32,
        ).map_err(|e| format!("swa_ring_write_batched (k) l{layer_idx}: {e:?}"))?;
        gpu.swa_ring_write_batched_f32(
            &pbs.kv_batch, swa_v, n_kv as i32, head_dim as i32, win as i32,
            start_pos as i32, batch_size as i32,
        ).map_err(|e| format!("swa_ring_write_batched (v) l{layer_idx}: {e:?}"))?;
    }

    Ok(())
}

/// Batched FFN: shared expert + routed-expert MoE, end-to-end.
///
/// Computes per-batch ffn_out_batch[b, :] = shared_expert(hc_x_in[b])
/// + (if score-routed) Σ_k topk_w[b,k] · routed_expert_{topk_idx[b,k]}(hc_x_in[b])
///
/// Stages:
///   1. fused_rmsnorm_rotate_mq_batched(hc_x_in → ffn_x_rot)
///   2. rmsnorm_batched(hc_x_in → ffn_x_plain)
///   3. gemv_auto_batched_wmma(shared_w1, → shared_gate, Some(&pbs.wmma_x_scratch_f16))
///   4. gemv_auto_batched_wmma(shared_w3, → shared_up, Some(&pbs.wmma_x_scratch_f16))
///   5. v4f_silu_mul_clamp_f32_batched(shared_gate, shared_up → shared_gate)
///   6. rotate_x_mq_batched(shared_gate → shared_rot)
///   7. gemv_auto_batched_wmma(shared_w2, shared_rot → ffn_out_batch, Some(&pbs.wmma_x_scratch_f16))
///   8. (score-routed only) gemv_auto_batched_wmma(gate.weight, ffn_x_rot → moe_scores, Some(&pbs.wmma_x_scratch_f16))
///   9. sqrt_softplus_f32 on moe_scores (operates on full [B*n_exp] numel)
///   10. v4f_moe_topk_bias_aware_batched_f32 → topk_indices, topk_weights
///   11. v4f_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched → moe_gate, moe_up
///   12. v4f_silu_mul_clamp_f32_batched(B*k_top streams of MI) → moe_gate
///   13. rotate_x_mq_batched(B*k_top FWHT rotations) → moe_rot
///   14. v4f_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched
///       (atomicAdds routed expert outputs into ffn_out_batch with scale)
///
/// Hash-routed layers (layer_idx < num_hash_layers) skip steps 8-14.
/// V4F's hash routing uses static tid2eid lookup which is skipped at
/// quant time per the load_weights logic; falls back to shared-only.
#[allow(dead_code)]
fn ffn_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    layer_idx: usize,
    batch_size: usize,
    tokens: &[u32],
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let ffn_norm  = layer.ffn_norm.as_ref().unwrap();
    let shared_w1 = layer.shared_w1.as_ref().unwrap();
    let shared_w2 = layer.shared_w2.as_ref().unwrap();
    let shared_w3 = layer.shared_w3.as_ref().unwrap();

    let hidden = cfg.hidden_size;
    let im = cfg.moe_intermediate_size;

    // 1. Fused RMSNorm + FWHT rotate of hc_x_in_batch → ffn_x_rot_batch.
    gpu.fused_rmsnorm_rotate_mq_batched(
        &pbs.hc_x_in_batch, ffn_norm, &pbs.ffn_x_rot_batch,
        hidden, cfg.rms_norm_eps, batch_size,
    ).map_err(|e| format!("fused_rmsnorm_rotate_mq_batched ffn l{layer_idx}: {e:?}"))?;

    // 1b. Plain RMSNorm → ffn_x_plain_batch.
    gpu.rmsnorm_batched(
        &pbs.hc_x_in_batch, ffn_norm, &pbs.ffn_x_plain_batch,
        batch_size, hidden, cfg.rms_norm_eps,
    ).map_err(|e| format!("rmsnorm_batched ffn-side l{layer_idx}: {e:?}"))?;

    // 2-3. Shared expert gate + up GEMVs.
    gemv_auto_batched_wmma(
        gpu, shared_w1, &pbs.ffn_x_rot_batch, &pbs.ffn_x_plain_batch,
        &pbs.ffn_shared_gate_batch, im, hidden, batch_size, Some(&pbs.wmma_x_scratch_f16))?;
    gemv_auto_batched_wmma(
        gpu, shared_w3, &pbs.ffn_x_rot_batch, &pbs.ffn_x_plain_batch,
        &pbs.ffn_shared_up_batch, im, hidden, batch_size, Some(&pbs.wmma_x_scratch_f16))?;

    // 4. SwiGLU + clamp. The kernel batches `B` streams of length `n`.
    gpu.v4f_silu_mul_clamp_f32_batched(
        &pbs.ffn_shared_gate_batch, &pbs.ffn_shared_up_batch, &pbs.ffn_shared_gate_batch,
        im, batch_size, cfg.swiglu_limit,
    ).map_err(|e| format!("v4f_silu_mul_clamp_f32_batched shared l{layer_idx}: {e:?}"))?;

    // 5. FWHT rotate silu output.
    gpu.rotate_x_mq_batched(
        &pbs.ffn_shared_gate_batch, &pbs.ffn_shared_rot_batch, im, batch_size,
    ).map_err(|e| format!("rotate_x_mq_batched shared silu l{layer_idx}: {e:?}"))?;

    // 6. Shared down GEMV → ffn_out_batch.
    gemv_auto_batched_wmma(
        gpu, shared_w2, &pbs.ffn_shared_rot_batch, &pbs.ffn_shared_gate_batch,
        &pbs.ffn_out_batch, hidden, im, batch_size, Some(&pbs.wmma_x_scratch_f16))?;

    // ── Routed-expert MoE ───────────────────────────────────────────
    let do_routed = std::env::var("HIPFIRE_V4F_MOE").ok().as_deref() == Some("1")
        && layer.expert_gate_up_blob.is_some()
        && layer.expert_w2_blob.is_some();
    if !do_routed {
        return Ok(());
    }
    // Layers 0..num_hash_layers use STATIC tid2eid routing per upstream
    // V4F. Disabled via HIPFIRE_V4F_NO_HASH=1 (mirrors ffn_hash_routed).
    let hash_routing = layer_idx < cfg.num_hash_layers;
    if hash_routing
        && (std::env::var("HIPFIRE_V4F_NO_HASH").ok().as_deref() == Some("1")
            || layer.tid2eid_host.is_empty())
    {
        return Ok(());
    }

    let gate_w = layer.gate_weight.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} gate.weight missing"))?;
    let gate_up_ptrs = layer.expert_gate_up_ptrs.as_ref().unwrap();
    let w2_ptrs = layer.expert_w2_ptrs.as_ref().unwrap();
    let n_exp = cfg.n_routed_experts;
    let k_top = cfg.num_experts_per_tok;
    let route_scale: f32 = std::env::var("HIPFIRE_V4F_ROUTE_SCALE")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2.2);

    // 8. Router GEMV: gate.weight @ ffn_x_rot_batch → moe_scores [B, n_exp].
    gemv_auto_batched_wmma(
        gpu, gate_w, &pbs.ffn_x_rot_batch, &pbs.ffn_x_plain_batch,
        &pbs.moe_scores_batch, n_exp, hidden, batch_size, Some(&pbs.wmma_x_scratch_f16))?;

    // 9. sqrt_softplus over the full [B, n_exp] buffer.
    gpu.sqrt_softplus_f32(&pbs.moe_scores_batch)
        .map_err(|e| format!("sqrt_softplus_f32 moe scores l{layer_idx}: {e:?}"))?;

    if hash_routing {
        // Hash routing: per-batch CPU lookup of static tid2eid → topk_ids,
        // gather + normalise weights from the (already sqrt_softplus'd)
        // scores, then upload to the same topk_indices/weights buffers
        // the score-routed branch uses. Mirrors ffn_hash_routed
        // line-for-line, applied per batch row.
        if tokens.len() < batch_size {
            return Err(format!(
                "ffn_batched l{layer_idx}: tokens len {} < batch_size {}",
                tokens.len(), batch_size,
            ));
        }
        let scores_host = gpu.download_f32(&pbs.moe_scores_batch)
            .map_err(|e| format!("d2h moe_scores l{layer_idx}: {e:?}"))?;
        let mut topk_idx_host: Vec<i32> = Vec::with_capacity(batch_size * k_top);
        let mut topk_w_host:   Vec<f32> = Vec::with_capacity(batch_size * k_top);
        for b in 0..batch_size {
            let token_id = tokens[b] as usize;
            let row = token_id * k_top;
            if row + k_top > layer.tid2eid_host.len() {
                return Err(format!(
                    "ffn_batched hash l{layer_idx}: token_id {token_id} \
                     out of tid2eid range ({} entries)",
                    layer.tid2eid_host.len()
                ));
            }
            let topk_ids: Vec<u32> = layer.tid2eid_host[row..row + k_top].iter()
                .map(|&i| i.min((n_exp - 1) as u32))
                .collect();
            let scores_b = &scores_host[b * n_exp..(b + 1) * n_exp];
            let wts = match gather_normalized_weights(scores_b, &topk_ids) {
                Some(w) => w,
                None => vec![0.0; k_top],
            };
            topk_idx_host.extend(topk_ids.iter().map(|&i| i as i32));
            topk_w_host.extend(wts.iter().map(|&w| w * route_scale));
        }
        let idx_bytes: Vec<u8> = topk_idx_host.iter().flat_map(|i| i.to_le_bytes()).collect();
        gpu.memcpy_htod_auto(&pbs.moe_topk_indices_batch.buf, &idx_bytes)
            .map_err(|e| format!("htod hash topk_idx l{layer_idx}: {e:?}"))?;
        let w_bytes: Vec<u8> = topk_w_host.iter().flat_map(|w| w.to_le_bytes()).collect();
        gpu.memcpy_htod_auto(&pbs.moe_topk_weights_batch.buf, &w_bytes)
            .map_err(|e| format!("htod hash topk_w l{layer_idx}: {e:?}"))?;
    } else {
        let gate_bias = layer.gate_bias.as_ref()
            .ok_or_else(|| format!("layer {layer_idx} gate.bias missing"))?;
        // 10. Bias-aware top-K per batch row.
        gpu.v4f_moe_topk_bias_aware_batched_f32(
            &pbs.moe_scores_batch, gate_bias,
            &pbs.moe_topk_indices_batch, &pbs.moe_topk_weights_batch,
            n_exp as i32, k_top as i32, route_scale, batch_size as i32,
        ).map_err(|e| format!("v4f_moe_topk_bias_aware_batched l{layer_idx}: {e:?}"))?;
    }

    // 10.5. Scatter-by-expert routing sort + grouped GEMV. Opt-in via
    //       HIPFIRE_V4F_MOE_GROUPED=1. Default OFF: measured 18 % SLOWER
    //       than the indexed-batched K4 path on V4F+gfx1151 because the
    //       grouped layout trades x-row cache reuse (M=4096 blocks share
    //       x[b] in the indexed grid) for expert-slab reuse — and the
    //       indexed kernel was already at ~62 % of peak DRAM bandwidth.
    //       Keep wired for larger batches / archs where the trade flips.
    let moe_grouped = std::env::var("HIPFIRE_V4F_MOE_GROUPED")
        .map(|s| s == "1").unwrap_or(false);
    if moe_grouped {
        gpu.moe_routing_sort_by_expert(
            &pbs.moe_topk_indices_batch,
            &pbs.moe_sorted_b, &pbs.moe_sorted_krank,
            &pbs.moe_sorted_expert, &pbs.moe_expert_starts,
            batch_size as i32, k_top as i32, cfg.n_routed_experts as i32,
        ).map_err(|e| format!("moe_routing_sort_by_expert l{layer_idx}: {e:?}"))?;
    }

    // 11. Routed expert gate_up (MQ2-Lloyd, scatter-by-expert grouped K4).
    // K4-unrolled variant (4 independent accumulators per thread for ILP).
    // Opt out via HIPFIRE_V4F_GATEUP_K4=0 (default: on).
    let gate_up_k4 = std::env::var("HIPFIRE_V4F_GATEUP_K4")
        .map(|s| s != "0").unwrap_or(true);
    if moe_grouped && gate_up_k4 {
        gpu.v4f_gemv_mq2g256_lloyd_moe_gate_up_grouped_k4(
            gate_up_ptrs,
            &pbs.moe_sorted_b, &pbs.moe_sorted_krank, &pbs.moe_sorted_expert,
            &pbs.ffn_x_rot_batch,
            &pbs.moe_gate_batch, &pbs.moe_up_batch,
            2 * im, hidden, k_top, batch_size,
        ).map_err(|e| format!("v4f_gemv_gate_up_grouped_k4 l{layer_idx}: {e:?}"))?;
    } else if gate_up_k4 {
        gpu.v4f_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4(
            gate_up_ptrs, &pbs.moe_topk_indices_batch, &pbs.ffn_x_rot_batch,
            &pbs.moe_gate_batch, &pbs.moe_up_batch,
            2 * im, hidden, k_top, batch_size,
        ).map_err(|e| format!("v4f_gemv_gate_up_batched_k4 l{layer_idx}: {e:?}"))?;
    } else {
        gpu.v4f_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched(
            gate_up_ptrs, &pbs.moe_topk_indices_batch, &pbs.ffn_x_rot_batch,
            &pbs.moe_gate_batch, &pbs.moe_up_batch,
            2 * im, hidden, k_top, batch_size,
        ).map_err(|e| format!("v4f_gemv_gate_up_batched l{layer_idx}: {e:?}"))?;
    }

    // 12. SwiGLU + clamp over B * k_top streams of length IM.
    gpu.v4f_silu_mul_clamp_f32_batched(
        &pbs.moe_gate_batch, &pbs.moe_up_batch, &pbs.moe_gate_batch,
        im, batch_size * k_top, cfg.swiglu_limit,
    ).map_err(|e| format!("v4f_silu_mul_clamp_f32_batched routed l{layer_idx}: {e:?}"))?;

    // 13. FWHT rotate B * k_top vectors of length IM.
    gpu.rotate_x_mq_batched(
        &pbs.moe_gate_batch, &pbs.moe_rot_batch, im, batch_size * k_top,
    ).map_err(|e| format!("rotate_x_mq_batched routed l{layer_idx}: {e:?}"))?;

    // 14. Routed expert down with scaled atomicAdd into ffn_out_batch.
    // K4-unrolled scatter-by-expert grouped variant (or fallbacks).
    // Opt out via HIPFIRE_V4F_DOWN_K4=0 (default: on).
    let down_k4 = std::env::var("HIPFIRE_V4F_DOWN_K4")
        .map(|s| s != "0").unwrap_or(true);
    if moe_grouped && down_k4 {
        gpu.v4f_gemv_mq2g256_lloyd_moe_down_residual_scaled_grouped_k4(
            w2_ptrs,
            &pbs.moe_sorted_b, &pbs.moe_sorted_krank, &pbs.moe_sorted_expert,
            &pbs.moe_topk_weights_batch,
            &pbs.moe_rot_batch, &pbs.ffn_out_batch,
            hidden, im, k_top, batch_size,
        ).map_err(|e| format!("v4f_gemv_down_grouped_k4 l{layer_idx}: {e:?}"))?;
    } else if down_k4 {
        gpu.v4f_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched_k4(
            w2_ptrs, &pbs.moe_topk_indices_batch, &pbs.moe_topk_weights_batch,
            &pbs.moe_rot_batch, &pbs.ffn_out_batch,
            hidden, im, k_top, batch_size,
        ).map_err(|e| format!("v4f_gemv_down_batched_k4 l{layer_idx}: {e:?}"))?;
    } else {
        gpu.v4f_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched(
            w2_ptrs, &pbs.moe_topk_indices_batch, &pbs.moe_topk_weights_batch,
            &pbs.moe_rot_batch, &pbs.ffn_out_batch,
            hidden, im, k_top, batch_size,
        ).map_err(|e| format!("v4f_gemv_down_batched l{layer_idx}: {e:?}"))?;
    }

    Ok(())
}

/// Batched-aware twin of `final_norm_and_head` — extracts the LAST
/// position's residual streams from pbs.streams_batch and runs the
/// existing per-position head pipeline against it.
///
/// Phase B2 chunk forward only needs logits at the last position
/// (matches qwen35::forward_prefill_batch's contract). All upstream
/// state.* scratch fields used by `final_norm_and_head` are sized for
/// one position and get reused unchanged.
///
/// Returns the logits at the last position. Caller is responsible for
/// any sampler integration.
#[allow(dead_code)]
fn final_norm_and_head_last_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    batch_size: usize,
) -> Result<Vec<f32>, String> {
    if batch_size == 0 {
        return Err("final_norm_and_head_last_batched: empty batch".to_string());
    }
    // Snapshot the original residual_streams so we can restore it (the
    // sequential function reads/keeps state.residual_streams; we point
    // it temporarily at the last position's slice).
    let last_off = (batch_size - 1) * cfg.hc_mult * cfg.hidden_size;
    let last_len = cfg.hc_mult * cfg.hidden_size;
    let last_streams = pbs.streams_batch.sub_offset(last_off, last_len);

    let orig = state.residual_streams.take();
    state.residual_streams = Some(last_streams);

    let result = final_norm_and_head(cfg, weights, state, gpu);

    // Restore. Drop the temporary view (it shares the pbs buffer; the
    // underlying buffer is owned by pbs, so leaking the view is fine —
    // it's a thin GpuTensor wrapper, not a fresh allocation).
    state.residual_streams = orig;

    result?;
    let logits_tensor = state.logits.as_ref()
        .ok_or_else(|| "logits not allocated".to_string())?;
    gpu.download_f32(logits_tensor)
        .map_err(|e| format!("download logits: {e:?}"))
}

/// Run final_norm + head on EVERY position of the batched chunk.
///
/// `final_norm_and_head_last_batched` only produces logits for the last
/// position (the only position whose token is sampled in normal prefill).
/// Speculative-decode verification needs per-position logits so each
/// draft can be compared against the verifier's preferred token —
/// that's what this helper provides.
///
/// Cost: K invocations of the per-position final_norm_and_head pipeline
/// (head HC + RMSNorm + rotate + lm_head GEMV + d2h). The lm_head is
/// [vocab=129280, hidden=4096]; per-position it's well under 5 ms on
/// gfx1151, so K=8 takes <40 ms. Acceptable for spec-decode windows.
///
/// Returns Vec<Vec<f32>> of length `batch_size`, each inner Vec sized
/// `vocab_size`.
pub fn final_norm_and_head_all_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    batch_size: usize,
) -> Result<Vec<Vec<f32>>, String> {
    if batch_size == 0 {
        return Err("final_norm_and_head_all_batched: empty batch".to_string());
    }
    let stream_len = cfg.hc_mult * cfg.hidden_size;
    let orig = state.residual_streams.take();
    let mut all_logits: Vec<Vec<f32>> = Vec::with_capacity(batch_size);
    let result: Result<(), String> = (|| {
        for i in 0..batch_size {
            let off = i * stream_len;
            let streams_i = pbs.streams_batch.sub_offset(off, stream_len);
            state.residual_streams = Some(streams_i);
            final_norm_and_head(cfg, weights, state, gpu)?;
            let logits_tensor = state.logits.as_ref()
                .ok_or_else(|| "logits not allocated".to_string())?;
            let logits_host = gpu.download_f32(logits_tensor)
                .map_err(|e| format!("download logits @pos {i}: {e:?}"))?;
            all_logits.push(logits_host);
        }
        Ok(())
    })();
    // Restore original residual_streams regardless of success / failure.
    state.residual_streams = orig;
    result?;
    Ok(all_logits)
}

/// Batched twin of `hc_ffn_mix`. Same shape as `hc_attn_mix_batched`
/// but mixes the FFN-side post/comb (produced by the second
/// mhc_pre_batched call with is_attn=false) and the FFN's transform
/// output `pbs.ffn_out_batch`.
#[allow(dead_code)]
fn hc_ffn_mix_batched(
    cfg: &DeepseekV4Config,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    batch_size: usize,
) -> Result<(), String> {
    gpu.hc_mix_4stream_batched(
        &pbs.streams_batch, &pbs.hc_comb_batch, &pbs.hc_post_batch,
        &pbs.ffn_out_batch, &pbs.streams_out_batch,
        cfg.hidden_size as i32, batch_size as i32,
    ).map_err(|e| format!("hc_mix_4stream_batched (ffn): {e:?}"))?;

    let bytes = batch_size * cfg.hc_mult * cfg.hidden_size * 4;
    gpu.memcpy_dtod_auto(&pbs.streams_batch.buf, &pbs.streams_out_batch.buf, bytes)
        .map_err(|e| format!("d2d streams_out → streams: {e:?}"))?;
    Ok(())
}

/// Batched twin of `mhc_pre` for Phase B2 chunk forward.
///
/// Per batch position b, after this returns:
///   pbs.hc_pre_batch[b, :]  = sigmoid(c[b, 0..4])
///   pbs.hc_post_batch[b, :] = post_scale * sigmoid(c[b, 4..8])
///   pbs.hc_comb_batch[b, :, :] = Sinkhorn(c[b, 8..24])
///   pbs.hc_x_in_batch[b, :] = sum_h hc_pre_batch[b, h] · streams[b, h, :]
///
/// where c is the post-α-rescale control vector. The split into separate
/// pre/post/comb buffers avoids strided sigmoid_f32 calls on the [B, 24]
/// layout (per-row segments are not memory-contiguous).
///
/// `is_attn` selects attn-side vs FFN-side W_fn / base / scale.
/// `HIPFIRE_V4F_POST_SCALE` env override (default 1.5) is honoured.
#[allow(dead_code)]
fn mhc_pre_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    layer_idx: usize,
    is_attn: bool,
    batch_size: usize,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let (hc_fn, hc_base, hc_scale) = if is_attn {
        (
            layer.hc_attn_fn.as_ref().unwrap(),
            layer.hc_attn_base.as_ref().unwrap(),
            layer.hc_attn_scale.as_ref().unwrap(),
        )
    } else {
        (
            layer.hc_ffn_fn.as_ref().unwrap(),
            layer.hc_ffn_base.as_ref().unwrap(),
            layer.hc_ffn_scale.as_ref().unwrap(),
        )
    };

    let n_ctrl = 24usize;
    let x_dim = cfg.hidden_size * cfg.hc_mult;
    let post_scale: f32 = std::env::var("HIPFIRE_V4F_POST_SCALE")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1.5);

    // 1. c = streams · W_fn · rsqrt(mean) + base. Per-batch.
    gpu.hc_compute_control_batched(
        &pbs.streams_batch, hc_fn, hc_base, &pbs.hc_c_batch,
        n_ctrl as i32, x_dim as i32, batch_size as i32,
    ).map_err(|e| format!("hc_compute_control_batched l{layer_idx}: {e:?}"))?;

    // 2. α-rescale c in place per batch.
    gpu.hc_apply_alpha_batched(
        &pbs.hc_c_batch, hc_scale, hc_base, batch_size as i32,
    ).map_err(|e| format!("hc_apply_alpha_batched l{layer_idx}: {e:?}"))?;

    // 3. Split c[B, 24] → contiguous pre[B, 4] / post[B, 4] / comb[B, 16]
    //    with sigmoid on pre, post_scale·sigmoid on post.
    gpu.hc_split_finalize_batched(
        &pbs.hc_c_batch, &pbs.hc_pre_batch, &pbs.hc_post_batch, &pbs.hc_comb_batch,
        post_scale, batch_size as i32,
    ).map_err(|e| format!("hc_split_finalize_batched l{layer_idx}: {e:?}"))?;

    // 4. Sinkhorn-normalize comb[B, 4, 4] in place per batch.
    gpu.hc_sinkhorn_4x4_batched(
        &pbs.hc_comb_batch, cfg.hc_eps, cfg.hc_sinkhorn_iters as i32,
        batch_size as i32,
    ).map_err(|e| format!("hc_sinkhorn_4x4_batched l{layer_idx}: {e:?}"))?;

    // 5. Input mapping: hc_x_in[b, d] = sum_h pre[b, h] · streams[b, h, d].
    gpu.hc_input_map_4stream_batched(
        &pbs.hc_pre_batch, &pbs.streams_batch, &pbs.hc_x_in_batch,
        cfg.hidden_size as i32, batch_size as i32,
    ).map_err(|e| format!("hc_input_map_4stream_batched l{layer_idx}: {e:?}"))?;

    Ok(())
}

/// Batched twin of `apply_tail_rope` for Phase B2 chunk forward.
///
/// Per batch position b: applies V4F's tail-only RoPE on the last
/// `qk_rope_head_dim` dims of each head in pbs.q_batch and pbs.kv_batch.
/// Reads positions[b] from `pbs.positions` (caller responsible for
/// pre-uploading `start_pos + b` per batch row at chunk start).
///
/// Per-layer YaRN parameters resolved via `layer_rope_params` exactly as
/// in the sequential path. Honours `HIPFIRE_V4F_NO_YARN=1` (single-theta
/// path via `rope_tail_interleaved_batched`).
#[allow(dead_code)]
fn apply_tail_rope_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    layer_idx: usize,
    batch_size: usize,
) -> Result<(), String> {
    if std::env::var("HIPFIRE_V4F_NO_YARN").ok().as_deref() == Some("1") {
        gpu.rope_tail_interleaved_batched(
            &pbs.q_batch, &pbs.kv_batch, &pbs.positions,
            cfg.num_attention_heads as i32,
            cfg.num_key_value_heads as i32,
            cfg.head_dim as i32,
            cfg.qk_rope_head_dim as i32,
            cfg.rope_theta,
            batch_size as i32,
        ).map_err(|e| format!("rope_tail_interleaved_batched (no-yarn): {e:?}"))?;
        return Ok(());
    }

    let layer = weights.resolve_layer(layer_idx);
    let (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high) =
        layer_rope_params(cfg, layer.compress_ratio);

    gpu.rope_tail_yarn_interleaved_batched(
        &pbs.q_batch, &pbs.kv_batch, &pbs.positions,
        cfg.num_attention_heads as i32,
        cfg.num_key_value_heads as i32,
        cfg.head_dim as i32,
        cfg.qk_rope_head_dim as i32,
        freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high,
        /*inverse=*/0,
        batch_size as i32,
    ).map_err(|e| format!("rope_tail_yarn_interleaved_batched l{layer_idx}: {e:?}"))?;

    Ok(())
}

/// Batched twin of `kv_joint` for Phase B2 chunk forward.
///
/// Per batch position b:
///   kv[b] = wkv @ {tmp[b] or tmp_plain[b]}   (gemv_auto_batched)
///   kv[b] = RMSNorm(kv[b], kv_norm)          (in-place)
///
/// Reuses pbs.tmp_batch / pbs.tmp_plain_batch produced by q_lora_batched
/// in the same layer iteration. Writes pbs.kv_batch.
#[allow(dead_code)]
fn kv_joint_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    layer_idx: usize,
    batch_size: usize,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let wkv = layer.wkv.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wkv missing"))?;
    let kv_norm = layer.kv_norm.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} kv_norm missing"))?;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

    // wkv @ tmp → kv.
    gemv_auto_batched_wmma(
        gpu, wkv, &pbs.tmp_batch, &pbs.tmp_plain_batch, &pbs.kv_batch,
        kv_dim, cfg.hidden_size, batch_size, Some(&pbs.wmma_x_scratch_f16))?;

    // kv_norm RMSNorm in-place: batch x [kv_dim].
    gpu.rmsnorm_batched(
        &pbs.kv_batch, kv_norm, &pbs.kv_batch,
        batch_size, kv_dim, cfg.rms_norm_eps,
    ).map_err(|e| format!("kv_norm rmsnorm_batched l{layer_idx}: {e:?}"))?;

    Ok(())
}

/// Batched twin of `q_lora` for Phase B2 chunk forward.
///
/// Per batch position b:
///   tmp[b] = FWHT(RMSNorm(hc_x_in[b], attn_norm))
///   tmp_plain[b] = RMSNorm(hc_x_in[b], attn_norm)
///   q_lat[b] = wq_a @ {tmp[b] or tmp_plain[b]}  (gemv_auto_batched)
///   q_lat[b] = RMSNorm(q_lat[b], q_norm)        (in-place per row)
///   q_lat_rot[b] = FWHT(q_lat[b])
///   q[b] = wq_b @ {q_lat_rot[b] or q_lat[b]}    (gemv_auto_batched)
///   q[b, head] = RMSNorm(q[b, head], q_head_ones) for each head  (per-head)
///
/// All seven steps stay in lockstep across the B positions by riding the
/// existing `*_batched` kernels. The per-head Q normalisation at the end
/// flattens `[B, n_heads, head_dim]` into `B * n_heads` rows of head_dim
/// elements before calling `rmsnorm_batched`.
///
/// Honours `HIPFIRE_V4F_SKIP_QHN=1` for the per-head Q rmsnorm (matches
/// the sequential bisect-escape hatch).
#[allow(dead_code, clippy::too_many_arguments)]
fn q_lora_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    pbs: &PrefillBatchScratch,
    hc_x_in_batch: &GpuTensor,   // [B, hidden]
    gpu: &mut Gpu,
    layer_idx: usize,
    batch_size: usize,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let attn_norm = layer.attn_norm.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} attn_norm missing"))?;
    let q_norm = layer.q_norm.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} q_norm missing"))?;
    let wq_a = layer.wq_a.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wq_a missing"))?;
    let wq_b = layer.wq_b.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wq_b missing"))?;

    let hidden = cfg.hidden_size;
    let q_rank = cfg.q_lora_rank;
    let n_heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim;

    // 1. Fused RMSNorm + FWHT-rotate batched: hc_x_in_batch → tmp_batch.
    gpu.fused_rmsnorm_rotate_mq_batched(
        hc_x_in_batch, attn_norm, &pbs.tmp_batch,
        hidden, cfg.rms_norm_eps, batch_size,
    ).map_err(|e| format!("fused_rmsnorm_rotate_mq_batched l{layer_idx}: {e:?}"))?;

    // 1b. Plain RMSNorm batched: hc_x_in_batch → tmp_plain_batch.
    gpu.rmsnorm_batched(
        hc_x_in_batch, attn_norm, &pbs.tmp_plain_batch,
        batch_size, hidden, cfg.rms_norm_eps,
    ).map_err(|e| format!("rmsnorm_batched attn-side plain l{layer_idx}: {e:?}"))?;

    // 2. wq_a GEMV batched: tmp* → q_lat_batch. M = q_lora_rank, K = hidden.
    gemv_auto_batched_wmma(
        gpu, wq_a, &pbs.tmp_batch, &pbs.tmp_plain_batch, &pbs.q_lat_batch,
        q_rank, hidden, batch_size, Some(&pbs.wmma_x_scratch_f16))?;

    // 3. q_norm RMSNorm batched (in-place): batch x [q_lora_rank].
    gpu.rmsnorm_batched(
        &pbs.q_lat_batch, q_norm, &pbs.q_lat_batch,
        batch_size, q_rank, cfg.rms_norm_eps,
    ).map_err(|e| format!("q_norm rmsnorm_batched l{layer_idx}: {e:?}"))?;

    // 4. FWHT rotate q_lat → q_lat_rot for the MQ4 wq_b path.
    gpu.rotate_x_mq_batched(&pbs.q_lat_batch, &pbs.q_lat_rot_batch, q_rank, batch_size)
        .map_err(|e| format!("rotate_x_mq_batched q_lat l{layer_idx}: {e:?}"))?;

    // 5. wq_b GEMV batched: q_lat_rot* → q_batch. M = n_heads*head_dim, K = q_lora_rank.
    let q_total = n_heads * head_dim;
    gemv_auto_batched_wmma(
        gpu, wq_b, &pbs.q_lat_rot_batch, &pbs.q_lat_batch, &pbs.q_batch,
        q_total, q_rank, batch_size, Some(&pbs.wmma_x_scratch_f16))?;

    // 6. Per-(batch, head) RMSNorm of Q using q_head_ones as weight.
    //    [B, n_heads, head_dim] viewed as [B*n_heads, head_dim].
    if std::env::var("HIPFIRE_V4F_SKIP_QHN").ok().as_deref() != Some("1") {
        gpu.rmsnorm_batched(
            &pbs.q_batch, &pbs.q_head_ones, &pbs.q_batch,
            batch_size * n_heads, head_dim, cfg.rms_norm_eps,
        ).map_err(|e| format!("q per-head rmsnorm_batched l{layer_idx}: {e:?}"))?;
    }

    Ok(())
}

/// Batched-prefill entry point for V4F.
///
/// Processes the `tokens` slice starting at absolute KV position
/// `start_pos`. Returns the logits at the LAST position only (matches
/// the qwen35 forward_prefill_batch contract).
///
/// **Phase B status (2026-05-18):** scaffold. The body falls back to a
/// per-token `decode_step` loop — byte-identical to the existing
/// sequential prefill semantics. Phase B2 will replace the loop body
/// with a `forward_prefill_batch_chunk` call that processes `max_batch`
/// positions at once using the Phase A batched kernels (A1: SWA-topK,
/// A2: SWA, A3: indexer top-K, A5: HC mix).
///
/// The entry-point shape is finalised now so callers (eval harnesses,
/// daemon, eventual prefill API) can wire against the stable signature
/// while the inner batched body grows behind it. `HIPFIRE_V4F_PREFILL_BATCHED=0`
/// will force the per-token fallback path once batching lands.
pub fn forward_prefill_batch(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
    _scratch: &mut PrefillBatchScratch,
) -> Result<Vec<f32>, String> {
    if tokens.is_empty() {
        return Err("forward_prefill_batch: empty tokens slice".to_string());
    }
    // Per-token fallback until forward_prefill_batch_chunk is end-to-end.
    let mut last_logits = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        last_logits = decode_step(cfg, weights, state, gpu, tok, start_pos + i as u32)?;
    }
    Ok(last_logits)
}

/// Single-chunk batched forward pass — Phase B2 work in progress.
///
/// Processes a chunk of `tokens.len()` ≤ `pbs.max_batch` positions
/// starting at `start_pos` through one batched forward. Mirrors
/// `decode_step` but with each per-layer stage swapped for its batched
/// twin. Returns the logits at the LAST position only.
///
/// Currently a partial wiring — runs through the stages that have
/// shipped batched bodies (embedding, HC stream init, q_lora,
/// kv_joint, tail RoPE) then errors out at the first unbatched stage
/// (the indexer + mixed attention dispatch). Each subsequent commit
/// replaces one error path with a real batched body until the chunk
/// runs end-to-end.
///
/// **Stages and their status (2026-05-18):**
///   ✓ token-ids upload → pbs.tokens
///   ✓ positions upload → pbs.positions
///   ✓ batched embedding lookup → pbs.embed_batch
///   ✓ HC streams broadcast init → pbs.streams_batch
///   ✓ per-layer q_lora_batched (Phase B2)
///   ✓ per-layer kv_joint_batched (Phase B2)
///   ✓ per-layer apply_tail_rope_batched (Phase B2)
///   ☐ per-layer mhc_pre_batched
///   ☐ per-layer compressor (loop sequential per A4 deferral)
///   ☐ per-layer indexer_forward_batched
///   ☐ per-layer mixed attention (wire v4f_attn_swa_topk_batched)
///   ☐ per-layer wo projection (gemv_auto_batched, two-stage O-LoRA)
///   ☐ per-layer hc_attn_mix_batched
///   ☐ per-layer ffn_routed_batched + hc_ffn_mix_batched
///   ☐ final_norm + lm_head (last position only)
///
/// Until all stages are wired this function returns an error from the
/// first unimplemented stage; callers should keep dispatching through
/// `forward_prefill_batch`'s per-token fallback for now.
#[allow(dead_code)]
pub fn forward_prefill_batch_chunk(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    tokens: &[u32],
    start_pos: u32,
) -> Result<(), String> {
    let n = tokens.len();
    if n == 0 {
        return Err("forward_prefill_batch_chunk: empty tokens".to_string());
    }
    if n > pbs.max_batch {
        return Err(format!(
            "forward_prefill_batch_chunk: chunk size {n} > max_batch {}",
            pbs.max_batch
        ));
    }

    // Phase C: ensure we have an active stream so all the small h2d
    // uploads in this chunk forward go async-on-stream via
    // `memcpy_htod_auto`. Subsequent kernels submitted to the same
    // stream order naturally — no host blocking on each tiny upload.
    // Opt out via HIPFIRE_V4F_ASYNC_HTOD=0.
    let async_htod = std::env::var("HIPFIRE_V4F_ASYNC_HTOD")
        .map(|s| s != "0").unwrap_or(true);
    if async_htod && gpu.active_stream.is_none() {
        let new_stream = gpu.hip.stream_create()
            .map_err(|e| format!("stream_create for async htod: {e:?}"))?;
        gpu.active_stream = Some(new_stream);
    }

    // 1. Upload token ids and absolute positions for this chunk.
    let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let token_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(token_ids_host.as_ptr() as *const u8, n * 4)
    };
    gpu.memcpy_htod_auto(&pbs.tokens.buf, token_bytes)
        .map_err(|e| format!("htod tokens: {e:?}"))?;

    let positions_host: Vec<i32> = (0..n).map(|i| (start_pos as i32) + i as i32).collect();
    let positions_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4)
    };
    gpu.memcpy_htod_auto(&pbs.positions.buf, positions_bytes)
        .map_err(|e| format!("htod positions: {e:?}"))?;

    // 2. Batched embedding lookup → pbs.embed_batch [n, hidden].
    let token_embd = weights.token_embd.as_ref()
        .ok_or_else(|| "forward_prefill_batch_chunk: token_embd not uploaded".to_string())?;
    gpu.embedding_lookup_q8_batched(token_embd, &pbs.embed_batch, &pbs.tokens, n, cfg.hidden_size)
        .map_err(|e| format!("embedding_lookup_q8_batched: {e:?}"))?;

    // 3. Broadcast embed → all 4 HC residual streams [n, hc_mult, hidden].
    gpu.hc_streams_init_from_embed_batched(
        &pbs.embed_batch, &pbs.streams_batch,
        cfg.hidden_size as i32, cfg.hc_mult as i32, n as i32,
    ).map_err(|e| format!("hc_streams_init_from_embed_batched: {e:?}"))?;

    // 4. Per-layer loop. Stages that DO run:
    //   ✓ mhc_pre_batched(is_attn=true)  → pbs.{hc_pre,hc_post,hc_comb,hc_x_in}_batch
    //   ✓ q_lora_batched   (consumes hc_x_in_batch) → pbs.q_batch
    //   ✓ kv_joint_batched (consumes tmp/tmp_plain) → pbs.kv_batch
    //   ✓ apply_tail_rope_batched         (in-place on q_batch & kv_batch)
    //
    // Then we hit the attention stage which still needs per-batch SWA
    // staging + indexer top-K gather + wo_a/wo_b O-LoRA projection. Bail
    // out cleanly so callers know the integration path is partial.
    // Bisection-aid env: HIPFIRE_V4F_FORWARD_LAYER_END=N stops after N layers.
    let layer_end: usize = std::env::var("HIPFIRE_V4F_FORWARD_LAYER_END")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(cfg.num_hidden_layers);
    for layer_idx in 0..cfg.num_hidden_layers.min(layer_end) {
        // Attention-side HC pre + per-stream input mapping.
        mhc_pre_batched(cfg, weights, pbs, gpu, layer_idx, /*is_attn=*/true, n)?;

        // Q-LoRA: pbs.hc_x_in_batch → tmp/tmp_plain → q_lat → q_batch.
        q_lora_batched(cfg, weights, pbs, &pbs.hc_x_in_batch, gpu, layer_idx, n)?;

        // Joint KV projection: tmp/tmp_plain → kv_batch.
        kv_joint_batched(cfg, weights, pbs, gpu, layer_idx, n)?;

        // Tail-only RoPE on q_batch and kv_batch in-place.
        apply_tail_rope_batched(cfg, weights, pbs, gpu, layer_idx, n)?;

        // ── Attention block: pure-SWA for compress_ratio==0, mixed
        //    (SWA + indexer/identity topk) for compress_ratio>0.
        let layer = weights.resolve_layer(layer_idx);
        if layer.compress_ratio == 0 {
            attention_block_batched_swa_only(
                cfg, weights, state, pbs, gpu, layer_idx, start_pos, n,
            )?;
        } else {
            attention_block_batched_mixed(
                cfg, weights, state, pbs, gpu, layer_idx, start_pos, n,
            )?;
        }

        // hc_attn_mix: integrate attn_out_batch into streams_batch.
        hc_attn_mix_batched(cfg, pbs, gpu, n)?;

        // Bisection break (mirror decode_step's env-gated stop).
        if layer_idx + 1 == cfg.num_hidden_layers.min(layer_end)
            && std::env::var("HIPFIRE_V4F_BISECT_BREAK").ok().as_deref()
                == Some("after_attn_mix")
        {
            return Ok(());
        }

        // FFN side: mhc_pre(is_attn=false) → ffn_batched (shared + routed)
        // → hc_ffn_mix_batched.
        mhc_pre_batched(cfg, weights, pbs, gpu, layer_idx, /*is_attn=*/false, n)?;
        if layer_idx + 1 == cfg.num_hidden_layers.min(layer_end)
            && std::env::var("HIPFIRE_V4F_BISECT_BREAK").ok().as_deref()
                == Some("after_mhc_pre_ffn")
        {
            return Ok(());
        }
        ffn_batched(cfg, weights, pbs, gpu, layer_idx, n, tokens)?;
        hc_ffn_mix_batched(cfg, pbs, gpu, n)?;
    }

    Ok(())
}

/// Top-level batched-prefill driver — chunks the prompt by max_batch
/// and dispatches each chunk through `forward_prefill_batch_chunk`.
///
/// Returns logits at the LAST position only (matches the qwen35
/// contract). Falls back to per-token decode_step if any chunk fails
/// (typically because a layer's compress_ratio path isn't yet wired —
/// pure-SWA-only for now, mixed-attention layers error out).
///
/// **Phase B2 status (2026-05-18):** the chunk-forward path handles
/// pure-SWA layers (compress_ratio == 0) end-to-end including the
/// MoE FFN; mixed-attention layers (compress_ratio > 0) still bail
/// at the indexer chain. Until mixed is wired, this function falls
/// back to per-token decode_step for any chunk that contains a
/// mixed-attention layer (i.e. all V4F prompts except the trivial
/// case where all 43 layers are dense, which doesn't exist).
#[allow(dead_code)]
pub fn forward_prefill_batch_chunked(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
    pbs: &PrefillBatchScratch,
) -> Result<Vec<f32>, String> {
    if tokens.is_empty() {
        return Err("forward_prefill_batch_chunked: empty tokens".to_string());
    }

    // Try the chunked batched path. If it fails (mixed-attention layer
    // not yet wired), fall back to per-token decode_step. The fallback
    // is byte-identical to current sequential prefill semantics.
    let mut pos_cursor = start_pos as usize;
    let mut remaining = tokens;
    while !remaining.is_empty() {
        let take = remaining.len().min(pbs.max_batch);
        let chunk = &remaining[..take];
        match forward_prefill_batch_chunk(
            cfg, weights, state, gpu, pbs, chunk, pos_cursor as u32,
        ) {
            Ok(()) => {
                // If this was the last chunk, run the head on the last
                // batch position.
                if take == remaining.len() {
                    return final_norm_and_head_last_batched(
                        cfg, weights, state, pbs, gpu, take,
                    );
                }
                pos_cursor += take;
                remaining = &remaining[take..];
            }
            Err(_) => {
                // Chunk failed (mixed-attention layer not wired).
                // Fall through to per-token decode_step for this and
                // remaining chunks.
                let mut last_logits = Vec::new();
                for (i, &tok) in remaining.iter().enumerate() {
                    last_logits = decode_step(
                        cfg, weights, state, gpu, tok, (pos_cursor + i) as u32,
                    )?;
                }
                return Ok(last_logits);
            }
        }
    }
    Err("forward_prefill_batch_chunked: chunk loop completed without producing logits".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_aware_topk_picks_biased_indices() {
        // Bias steers selection. scores=[1,1,1,1,1,1], bias=[0,0,0,3,2,0]
        // → biased=[1,1,1,4,3,1] → top-2 = [3, 4].
        let scores = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let bias   = vec![0.0, 0.0, 0.0, 3.0, 2.0, 0.0];
        let (idx, wts) = bias_aware_topk_weights(&scores, &bias, 2).unwrap();
        assert_eq!(idx, vec![3, 4]);
        // Weights come from UNBIASED scores (both 1.0), normalized.
        assert!((wts[0] - 0.5).abs() < 1e-6);
        assert!((wts[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn bias_aware_topk_weights_use_unbiased_scores() {
        // scores=[5, 1, 1], bias=[0, 10, 10] → biased=[5, 11, 11].
        // Top-2 by biased = [1, 2]. Weights from unbiased = [1, 1] → [0.5, 0.5].
        let scores = vec![5.0, 1.0, 1.0];
        let bias   = vec![0.0, 10.0, 10.0];
        let (idx, wts) = bias_aware_topk_weights(&scores, &bias, 2).unwrap();
        assert!(idx == vec![1, 2] || idx == vec![2, 1]);
        assert!((wts[0] - 0.5).abs() < 1e-6);
        assert!((wts[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn bias_aware_topk_falls_back_zero_bias() {
        // No bias → pure top-K from scores.
        let scores = vec![0.1, 0.9, 0.5, 0.7];
        let bias: Vec<f32> = vec![];
        let (idx, wts) = bias_aware_topk_weights(&scores, &bias, 2).unwrap();
        assert_eq!(idx, vec![1, 3]);
        let s = 0.9 + 0.7;
        assert!((wts[0] - 0.9/s).abs() < 1e-6);
        assert!((wts[1] - 0.7/s).abs() < 1e-6);
    }

    #[test]
    fn bias_aware_topk_returns_none_on_zero_sum() {
        // All scores zero → no positive weight sum.
        let scores = vec![0.0, 0.0, 0.0];
        let bias   = vec![5.0, 0.0, 0.0];  // bias picks idx 0 but score is 0
        assert!(bias_aware_topk_weights(&scores, &bias, 1).is_none());
    }

    #[test]
    fn bias_aware_topk_handles_k_geq_n() {
        // k=4 but only n=2 scores — caller's job to set k correctly,
        // but we silently clamp rather than panic.
        let scores = vec![1.0, 2.0];
        let bias   = vec![0.0, 0.0];
        let (idx, wts) = bias_aware_topk_weights(&scores, &bias, 4).unwrap();
        assert_eq!(idx.len(), 2);
        assert!(wts.iter().sum::<f32>() > 0.99 && wts.iter().sum::<f32>() < 1.01);
    }

    #[test]
    fn gather_normalized_weights_basic() {
        let scores = vec![0.0, 2.0, 0.0, 1.0, 0.0];
        let idx    = vec![1u32, 3];
        let wts = gather_normalized_weights(&scores, &idx).unwrap();
        // scores at idx = [2, 1] → normalized [2/3, 1/3]
        assert!((wts[0] - 2.0/3.0).abs() < 1e-6);
        assert!((wts[1] - 1.0/3.0).abs() < 1e-6);
    }

    #[test]
    fn gather_normalized_weights_zero_sum_returns_none() {
        let scores = vec![0.0; 8];
        let idx    = vec![0u32, 1, 2];
        assert!(gather_normalized_weights(&scores, &idx).is_none());
    }

    #[test]
    fn gather_normalized_weights_out_of_range_idx_is_zero() {
        // Hash table can in theory point past scores; we treat OOR as 0
        // (better than panicking — tid2eid is supposed to be in range).
        let scores = vec![1.0, 2.0, 3.0];
        let idx    = vec![1u32, 999];
        let wts = gather_normalized_weights(&scores, &idx).unwrap();
        // sum = 2 + 0 = 2 → normalized [1.0, 0.0]
        assert!((wts[0] - 1.0).abs() < 1e-6);
        assert!((wts[1] - 0.0).abs() < 1e-6);
    }
}
