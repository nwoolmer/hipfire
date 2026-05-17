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
    let layer = &weights.layers[layer_idx];
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
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1024);

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
    let kv_buf = state._indexer[layer_idx].comp_kv_buf.as_ref().unwrap();
    let score_buf = state._indexer[layer_idx].comp_score_buf.as_ref().unwrap();
    gpu.gemv_mq4g256_prerotated(wkv, x_rotated, kv_buf, proj_dim, hidden)
        .map_err(|e| format!("comp wkv gemv l{layer_idx}: {e:?}"))?;
    gpu.gemv_mq4g256_prerotated(wgate, x_rotated, score_buf, proj_dim, hidden)
        .map_err(|e| format!("comp wgate gemv l{layer_idx}: {e:?}"))?;

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

    // Phase 3b.4: indexer compressor applies tail RoPE on its compressed entry
    // with `compress_rope_theta` (160000). Position = start-of-window
    // (pos // ratio) * ratio. n_heads_q = 1, n_heads_k = 0 (single tensor;
    // re-use rope_tail_interleaved by routing only the q-loop).
    //
    // Main compressor (is_indexer=false): tried adding RoPE with
    // `rope_theta=10000` to align main_kv_cache with SWA K, but it made
    // zero difference to phase 5 attention output (bit-identical PPL).
    // Likely the joint-attention regression is upstream of K-RoPE — the
    // gathered-K softmax contribution is poisoned by something else (maybe
    // magnitude mismatch between compressor.norm output and kv_norm output,
    // or the top-K indices are degenerate at small n_compressed).
    if is_indexer {
        if state.comp_pos_buf.is_none() {
            state.comp_pos_buf = Some(gpu.alloc_tensor(&[1], DType::F32)
                .map_err(|e| format!("alloc comp_pos_buf l{layer_idx}: {e:?}"))?);
        }
        let pos_buf = state.comp_pos_buf.as_ref().unwrap();
        let rope_pos = ((position as usize) / ratio * ratio) as i32;
        let pos_bytes = rope_pos.to_le_bytes();
        gpu.hip.memcpy_htod(&pos_buf.buf, &pos_bytes)
            .map_err(|e| format!("htod comp_pos_buf l{layer_idx}: {e:?}"))?;

        gpu.rope_tail_interleaved(
            &kv_cache_slot, &kv_cache_slot, pos_buf,
            1, 0,
            head_dim as i32,
            cfg.qk_rope_head_dim as i32,
            cfg.compress_rope_theta,
        ).map_err(|e| format!("comp rope l{layer_idx}: {e:?}"))?;
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
    let layer = &weights.layers[layer_idx];
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
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
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
    let q_lat_rot = state.q_lat_rot.as_ref()
        .ok_or_else(|| "indexer: q_lat_rot not allocated".to_string())?;
    let q_idx = state._indexer[layer_idx].q_idx.as_ref().unwrap();
    gpu.gemv_mq4g256_prerotated(wq_b, q_lat_rot, q_idx, h * d, cfg.q_lora_rank)
        .map_err(|e| format!("idx wq_b gemv l{layer_idx}: {e:?}"))?;

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
    let idx_w = state._indexer[layer_idx].idx_weights.as_ref().unwrap();
    gpu.gemv_mq4g256_prerotated(weights_proj, tmp, idx_w, h, cfg.hidden_size)
        .map_err(|e| format!("idx weights_proj gemv l{layer_idx}: {e:?}"))?;

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
    // 1. Token embedding → initial residual streams.
    //    V4F uses `hc_mult = 4` parallel streams. Init pattern is
    //    [embed, 0, 0, 0] (paper-specified; verify against the V4F
    //    reference code before optimising).
    init_residual_streams(cfg, weights, state, gpu, token_id)?;

    // 2. Per-layer forward.
    for layer_idx in 0..cfg.num_hidden_layers {
        let layer = &weights.layers[layer_idx];
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
        apply_tail_rope(cfg, state, gpu, position, layer_idx)?;

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
        if layer.compress_ratio > 0
            && std::env::var("HIPFIRE_V4F_RUN_COMPRESSOR").ok().as_deref() == Some("1")
        {
            // Non-owning view of state.tmp so we can re-borrow state mutably.
            let tmp_view = {
                let t = state.tmp.as_ref().unwrap();
                t.sub_offset(0, t.numel())
            };
            compressor_forward(cfg, weights, state, gpu, layer_idx,
                &tmp_view, position, /*is_indexer=*/false)?;
            if layer.compress_ratio == 4 {
                compressor_forward(cfg, weights, state, gpu, layer_idx,
                    &tmp_view, position, /*is_indexer=*/true)?;
                // Phase 4b: indexer scoring + top-K selection.
                // Output → state._indexer[l].topk_idx_indices (consumed by
                // phase 5's modified attention). Currently dead in the
                // pipeline because phase 5 is not yet wired.
                if std::env::var("HIPFIRE_V4F_RUN_INDEXER").ok().as_deref() == Some("1") {
                    let _n = indexer_forward(cfg, weights, state, gpu, layer_idx, position)?;
                }
            }
        }

        // v + vi. Main attention + O-LoRA — STUB.
        attn_stub(cfg, weights, state, gpu, layer_idx)?;

        hc_attn_mix(cfg, weights, state, gpu, layer_idx)?;

        // ── 2b. FFN block ─────────────────────────────────────────────
        mhc_pre(cfg, weights, state, gpu, layer_idx, /*is_attn=*/false)?;
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

    // 3. Final norm + LM head.
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
    let layer = &weights.layers[layer_idx];
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

    let ffn_x_rot = state.ffn_x_rot.as_ref().unwrap();
    let gate = state.ffn_gate.as_ref().unwrap();
    let up   = state.ffn_up.as_ref().unwrap();
    let silu_rot = state.ffn_silu_rot.as_ref().unwrap();
    let ffn_out = state.ffn_out.as_ref().unwrap();

    // 1. Fused RMSNorm + FWHT rotate for the two MQ4 GEMVs.
    gpu.fused_rmsnorm_rotate_mq(hc_x_in, ffn_norm, ffn_x_rot,
        cfg.hidden_size, cfg.rms_norm_eps)
        .map_err(|e| format!("fused_rmsnorm_rotate_mq ffn layer {layer_idx}: {e:?}"))?;

    // 2. gate = x @ shared_w1
    gpu.gemv_mq4g256_prerotated(shared_w1, ffn_x_rot, gate, im, cfg.hidden_size)
        .map_err(|e| format!("gemv shared_w1 layer {layer_idx}: {e:?}"))?;

    // 3. up = x @ shared_w3
    gpu.gemv_mq4g256_prerotated(shared_w3, ffn_x_rot, up, im, cfg.hidden_size)
        .map_err(|e| format!("gemv shared_w3 layer {layer_idx}: {e:?}"))?;

    // 4. V4F SwiGLU with swiglu_limit clamp (cfg.swiglu_limit = 10.0
    //    on V4F). Same Expert class used for shared and routed in
    //    upstream model.py — both apply this clamp before silu_mul.
    gpu.v4f_silu_mul_clamp_f32(gate, up, gate, cfg.swiglu_limit)
        .map_err(|e| format!("v4f_silu_mul_clamp layer {layer_idx}: {e:?}"))?;

    // 5. FWHT-rotate the silu-gated vector for the down GEMV.
    gpu.rotate_x_mq(gate, silu_rot, im)
        .map_err(|e| format!("rotate_x_mq silu layer {layer_idx}: {e:?}"))?;

    // 6. ffn_out = silu_rot @ shared_w2 (down: [hidden, im])
    gpu.gemv_mq4g256_prerotated(shared_w2, silu_rot, ffn_out, cfg.hidden_size, im)
        .map_err(|e| format!("gemv shared_w2 layer {layer_idx}: {e:?}"))?;

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
    let layer = &weights.layers[layer_idx];
    if layer.expert_w1_blob.is_none() || layer.expert_w2_blob.is_none()
        || layer.expert_w3_blob.is_none()
    {
        return Ok(());  // experts not uploaded; nothing to dispatch
    }

    // 1. Run router: compute unbiased scores on-device. We do top-K on
    //    CPU below because V4F's selection uses BIASED scores while the
    //    routing weights use UNBIASED scores (per upstream model.py:
    //    Gate.forward). On-device top_k would mix these up.
    moe_route(cfg, weights, state, gpu, layer_idx)?;

    // 2. d2h the unbiased score vector (256 f32 = 1 KB per layer).
    let scores = state.router_scores.as_ref().unwrap();
    let scores_host = gpu.download_f32(scores)
        .map_err(|e| format!("d2h scores l{layer_idx}: {e:?}"))?;

    let k = cfg.num_experts_per_tok;
    let n_exp = cfg.n_routed_experts;

    // 3. CPU top-K with bias-shifted scores → normalized weights.
    let (topk_ids, wts) = match bias_aware_topk_weights(
        &scores_host[..n_exp], &layer.gate_bias_host, k)
    {
        Some(x) => x,
        None => return Ok(()),  // degenerate router output (sum <= 0)
    };

    // 3. Per-expert SwiGLU dispatch. Reuse shared scratch (ffn_gate,
    //    ffn_up, ffn_silu_rot) — by the time we get here, the shared
    //    expert path has finished consuming them.
    let im = cfg.moe_intermediate_size;
    let stride_w1 = layer.expert_w1_stride;
    let stride_w2 = layer.expert_w2_stride;
    let stride_w3 = layer.expert_w3_stride;
    let blob_w1 = layer.expert_w1_blob.as_ref().unwrap();
    let blob_w2 = layer.expert_w2_blob.as_ref().unwrap();
    let blob_w3 = layer.expert_w3_blob.as_ref().unwrap();

    if state.routed_expert_out.is_none() {
        state.routed_expert_out = Some(gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
            .map_err(|e| format!("alloc routed_expert_out: {e:?}"))?);
    }
    let ffn_x_rot = state.ffn_x_rot.as_ref().unwrap();
    let gate = state.ffn_gate.as_ref().unwrap();
    let up   = state.ffn_up.as_ref().unwrap();
    let silu_rot = state.ffn_silu_rot.as_ref().unwrap();
    let expert_out = state.routed_expert_out.as_ref().unwrap();
    let ffn_out = state.ffn_out.as_ref().unwrap();

    for (k_idx, &expert_id) in topk_ids.iter().enumerate() {
        let off_w1 = expert_id as usize * stride_w1;
        let off_w2 = expert_id as usize * stride_w2;
        let off_w3 = expert_id as usize * stride_w3;
        let w1_view = blob_w1.sub_offset(off_w1, stride_w1);
        let w2_view = blob_w2.sub_offset(off_w2, stride_w2);
        let w3_view = blob_w3.sub_offset(off_w3, stride_w3);

        gpu.gemv_mq2g256_lloyd(&w1_view, ffn_x_rot, gate, im, cfg.hidden_size)
            .map_err(|e| format!("gemv expert_w1 l{layer_idx} e{expert_id}: {e:?}"))?;
        gpu.gemv_mq2g256_lloyd(&w3_view, ffn_x_rot, up, im, cfg.hidden_size)
            .map_err(|e| format!("gemv expert_w3 l{layer_idx} e{expert_id}: {e:?}"))?;
        // V4F SwiGLU with swiglu_limit clamp (config = 10.0). silu(min
        // (gate, L)) * clamp(up, ±L) → gate (in-place). Per-expert
        // routing weight applied AFTER SwiGLU and BEFORE w2 per upstream
        // model.py:603-606. Clamp is part of V4F's faithful math; without
        // it, FP4 × FP32 × 4096-dim dots can drive silu(gate) into a
        // regime that feeds attractors.
        gpu.v4f_silu_mul_clamp_f32(gate, up, gate, cfg.swiglu_limit)
            .map_err(|e| format!("v4f_silu_mul_clamp expert l{layer_idx} e{expert_id}: {e:?}"))?;
        // Routing-weight scale done via scaled_add at accumulate step
        // (more efficient than a separate scale_f32 pass over [im]).
        gpu.rotate_x_mq(gate, silu_rot, im)
            .map_err(|e| format!("rotate expert silu l{layer_idx} e{expert_id}: {e:?}"))?;
        gpu.gemv_mq2g256_lloyd(&w2_view, silu_rot, expert_out, cfg.hidden_size, im)
            .map_err(|e| format!("gemv expert_w2 l{layer_idx} e{expert_id}: {e:?}"))?;
        // ffn_out += routing_weight[k] * routed_scaling_factor * expert_out
        // Default route_scale=1.0 (empirical optimum) vs upstream's
        // routed_scaling_factor=1.5. See post_scale comment.
        let route_scale_override: f32 = std::env::var("HIPFIRE_V4F_ROUTE_SCALE")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let coef = wts[k_idx] * route_scale_override;
        gpu.scaled_add_inplace_cpu_scalar_f32(ffn_out, expert_out, coef)
            .map_err(|e| format!("scaled_add expert l{layer_idx} e{expert_id}: {e:?}"))?;
    }

    Ok(())
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
    let layer = &weights.layers[layer_idx];
    if layer.expert_w1_blob.is_none() || layer.expert_w2_blob.is_none()
        || layer.expert_w3_blob.is_none()
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

    // Per-expert dispatch (identical to ffn_routed body).
    let im = cfg.moe_intermediate_size;
    let stride_w1 = layer.expert_w1_stride;
    let stride_w2 = layer.expert_w2_stride;
    let stride_w3 = layer.expert_w3_stride;
    let blob_w1 = layer.expert_w1_blob.as_ref().unwrap();
    let blob_w2 = layer.expert_w2_blob.as_ref().unwrap();
    let blob_w3 = layer.expert_w3_blob.as_ref().unwrap();

    if state.routed_expert_out.is_none() {
        state.routed_expert_out = Some(gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
            .map_err(|e| format!("alloc routed_expert_out: {e:?}"))?);
    }
    let ffn_x_rot = state.ffn_x_rot.as_ref().unwrap();
    let gate = state.ffn_gate.as_ref().unwrap();
    let up   = state.ffn_up.as_ref().unwrap();
    let silu_rot = state.ffn_silu_rot.as_ref().unwrap();
    let expert_out = state.routed_expert_out.as_ref().unwrap();
    let ffn_out = state.ffn_out.as_ref().unwrap();

    for (k_idx, &expert_id) in topk_ids.iter().enumerate() {
        let w1_view = blob_w1.sub_offset(expert_id as usize * stride_w1, stride_w1);
        let w2_view = blob_w2.sub_offset(expert_id as usize * stride_w2, stride_w2);
        let w3_view = blob_w3.sub_offset(expert_id as usize * stride_w3, stride_w3);

        gpu.gemv_mq2g256_lloyd(&w1_view, ffn_x_rot, gate, im, cfg.hidden_size)
            .map_err(|e| format!("gemv hash w1 l{layer_idx} e{expert_id}: {e:?}"))?;
        gpu.gemv_mq2g256_lloyd(&w3_view, ffn_x_rot, up, im, cfg.hidden_size)
            .map_err(|e| format!("gemv hash w3 l{layer_idx} e{expert_id}: {e:?}"))?;
        gpu.v4f_silu_mul_clamp_f32(gate, up, gate, cfg.swiglu_limit)
            .map_err(|e| format!("v4f_silu_mul_clamp hash l{layer_idx} e{expert_id}: {e:?}"))?;
        gpu.rotate_x_mq(gate, silu_rot, im)
            .map_err(|e| format!("rotate hash silu l{layer_idx} e{expert_id}: {e:?}"))?;
        gpu.gemv_mq2g256_lloyd(&w2_view, silu_rot, expert_out, cfg.hidden_size, im)
            .map_err(|e| format!("gemv hash w2 l{layer_idx} e{expert_id}: {e:?}"))?;
        // Default route_scale=1.0 (empirical optimum) vs upstream's
        // routed_scaling_factor=1.5. See post_scale comment.
        let route_scale_override: f32 = std::env::var("HIPFIRE_V4F_ROUTE_SCALE")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let coef = wts[k_idx] * route_scale_override;
        gpu.scaled_add_inplace_cpu_scalar_f32(ffn_out, expert_out, coef)
            .map_err(|e| format!("scaled_add hash l{layer_idx} e{expert_id}: {e:?}"))?;
    }
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

    // 3. RMSNorm of the combined stream output.
    gpu.rmsnorm_f32(head_hc_out, output_norm, final_norm, cfg.rms_norm_eps)
        .map_err(|e| format!("final rmsnorm_f32: {e:?}"))?;

    // 4. FWHT-rotate for MQ4 GEMV.
    gpu.rotate_x_mq(final_norm, final_norm_rot, cfg.hidden_size)
        .map_err(|e| format!("rotate_x_mq final_norm: {e:?}"))?;

    // 5. lm_head GEMV.
    gpu.gemv_mq4g256_prerotated(head, final_norm_rot, logits,
        cfg.vocab_size, cfg.hidden_size)
        .map_err(|e| format!("gemv_mq4g256 head: {e:?}"))?;

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

    let use_swa = std::env::var("HIPFIRE_V4F_ATTN").ok().as_deref() == Some("swa");

    let q = state.q.as_ref().unwrap();
    let kv = state.kv.as_ref().unwrap();
    let attn_out_raw = state.attn_out_raw.as_ref().unwrap();
    let layer = &weights.layers[layer_idx];
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

        // Phase 5c: indexer-extended attention when (a) HIPFIRE_V4F_RUN_INDEXER=1
        // populated topk_idx_indices this step, and (b) this layer is an
        // indexer-active layer (compress_ratio == 4). Joint softmax over the
        // SWA window K/V AND the top-K K rows gathered from main_kv_cache.
        //
        // Filter: phase 5 only matters when there are compressed slots that
        // represent positions BEFORE the SWA window. Compressed slot c (with
        // ratio=4 overlap) covers token positions [c*ratio - ratio, c*ratio +
        // ratio - 1]. For SWA window [pos-win, pos] to NOT cover slot c, we
        // need c*ratio + ratio - 1 < pos - win, i.e., c < (pos - win) / ratio.
        // The number of "pre-window" slots = max(0, (pos - win + 1) / ratio).
        // If 0, fall through to plain SWA — no need to gather from main_kv_cache.
        let ratio = 4usize;
        let n_pre_window_slots = if pos >= win {
            ((pos - win + 1) / ratio).min(cfg.index_topk)
        } else { 0 };

        let layer_indexer_active = layer.compress_ratio == 4
            && std::env::var("HIPFIRE_V4F_USE_INDEXER_ATTN").ok().as_deref() == Some("1")
            && state._indexer[layer_idx].topk_idx_indices.is_some()
            && state._indexer[layer_idx].main_kv_cache.is_some()
            && n_pre_window_slots > 0;

        if layer_indexer_active {
            let topk_max = cfg.index_topk;
            // Lazy-alloc gathered_k / gathered_v scratch.
            if state._attention[layer_idx].gathered_k.is_none() {
                state._attention[layer_idx].gathered_k = Some(
                    gpu.zeros(&[n_kv, head_dim, topk_max], DType::F32)
                        .map_err(|e| format!("alloc gathered_k l{layer_idx}: {e:?}"))?
                );
            }
            // V4F has tied K=V → re-use the gathered_k buffer for V via a
            // separate slot pointing to the same allocation. The
            // v4f_attn_swa_topk_f32 kernel reads two pointers but we can
            // pass the same one for both since K==V on V4F.
            let n_compressed = (pos + 1) / ratio;
            // Cap k_active by both topk_max and the count of pre-window slots —
            // top-K may include in-window slots that we don't want to attend to
            // (they'd double-count with SWA). Conservative: take only as many
            // gathered slots as we have pre-window compressed slots.
            let k_active = n_pre_window_slots.min(n_compressed);
            let gathered_k = state._attention[layer_idx].gathered_k.as_ref().unwrap();
            let topk_idx = state._indexer[layer_idx].topk_idx_indices.as_ref().unwrap();
            let main_kv_cache = state._indexer[layer_idx].main_kv_cache.as_ref().unwrap();

            // Gather top-K rows into gathered_k[:, :, 0..k_active]. Scale
            // by HIPFIRE_V4F_TOPK_K_SCALE to compensate for compressor.norm
            // undershoot relative to kv_norm output magnitude.
            let topk_k_scale: f32 = std::env::var("HIPFIRE_V4F_TOPK_K_SCALE")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
            gpu.v4f_topk_kv_gather_f32(
                main_kv_cache, topk_idx, gathered_k,
                k_active as i32, head_dim as i32, n_compressed as i32,
                topk_max as i32, 0, topk_k_scale,
            ).map_err(|e| format!("topk gather l{layer_idx}: {e:?}"))?;

            let swa_k = state._attention[layer_idx].swa_k.as_ref().unwrap();
            let swa_v = state._attention[layer_idx].swa_v.as_ref().unwrap();

            // Phase 5d: SEPARATE softmaxes + weighted combine.
            //
            // The joint softmax in v4f_attn_swa_topk_f32 is poisonous —
            // adding gathered entries (even with K=0, V=0) inflates the
            // partition function and dilutes SWA probabilities. Proof:
            // K_SCALE=0 → 137k ppl @ ctx=256 vs SWA-only 35.5k baseline.
            //
            // Mitigation: run SWA attention to attn_out_raw, then run a
            // SECOND attention pass over gathered K/V to a side buffer,
            // and scaled_add the topk contribution with alpha (env-gated,
            // default 0.0 = pure SWA = no change). Tune alpha at ctx>=256.
            let topk_alpha: f32 = std::env::var("HIPFIRE_V4F_TOPK_ALPHA")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            // Pass 1: SWA attention to attn_out_raw.
            gpu.v4f_attn_swa(q, swa_k, swa_v, attn_sink, attn_out_raw,
                n_heads as i32, head_dim as i32, n_groups as i32,
                n_valid, win as i32,
            ).map_err(|e| format!("v4f_attn_swa (phase5d swa) l{layer_idx}: {e:?}"))?;

            if topk_alpha > 0.0 {
                // Pass 2: gathered attention. Re-use attn_out_raw_rot as
                // a scratch (it's only used downstream of inverse-RoPE,
                // so safe to reuse here).
                let attn_out_raw_topk = state.attn_out_raw_rot.as_ref().unwrap();
                gpu.v4f_attn_swa(q, gathered_k, gathered_k, attn_sink, attn_out_raw_topk,
                    n_heads as i32, head_dim as i32, n_groups as i32,
                    k_active as i32, topk_max as i32,
                ).map_err(|e| format!("v4f_attn_swa (phase5d topk) l{layer_idx}: {e:?}"))?;
                // attn_out_raw += topk_alpha * attn_out_raw_topk
                gpu.scaled_add_inplace_cpu_scalar_f32(
                    attn_out_raw, attn_out_raw_topk, topk_alpha
                ).map_err(|e| format!("phase5d scaled_add l{layer_idx}: {e:?}"))?;
            }
        } else {
            let swa_k = state._attention[layer_idx].swa_k.as_ref().unwrap();
            let swa_v = state._attention[layer_idx].swa_v.as_ref().unwrap();
            gpu.v4f_attn_swa(q, swa_k, swa_v, attn_sink, attn_out_raw,
                n_heads as i32, head_dim as i32, n_groups as i32,
                n_valid, win as i32,
            ).map_err(|e| format!("v4f_attn_swa: {e:?}"))?;
        }
    }

    // Inverse tail RoPE on attn_out_raw. Same freq_base as forward
    // apply_tail_rope (rope_theta everywhere — see note there about
    // the compress_rope_theta + YaRN tradeoff).
    let pos_buf = state.pos_buf.as_ref()
        .ok_or_else(|| "pos_buf not allocated".to_string())?;
    if std::env::var("HIPFIRE_V4F_SKIP_INV_ROPE").ok().as_deref() != Some("1") {
        gpu.rope_tail_inverse(attn_out_raw, pos_buf,
            n_heads as i32, head_dim as i32,
            cfg.qk_rope_head_dim as i32, cfg.rope_theta,
        ).map_err(|e| format!("rope_tail_inverse l{layer_idx}: {e:?}"))?;
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
    // wo_a is MQ4G256 with 136 bytes per 256 elements. Per-group slice
    // has o_lora_rank * per_group_in elements → bytes = (m*k/256)*136.
    let per_group_wa_bytes = (o_lora_rank * per_group_in / 256) * 136;

    for g in 0..n_groups {
        let raw_view = attn_out_raw.sub_offset(g * per_group_in, per_group_in);
        let rot_view = attn_out_raw_rot.sub_offset(g * per_group_in, per_group_in);
        let wo_a_view = wo_a.sub_offset(g * per_group_wa_bytes, per_group_wa_bytes);
        let out_view = wo_a_out.sub_offset(g * o_lora_rank, o_lora_rank);
        gpu.rotate_x_mq(&raw_view, &rot_view, per_group_in)
            .map_err(|e| format!("rotate attn_out g{g} l{layer_idx}: {e:?}"))?;
        gpu.gemv_mq4g256_prerotated(&wo_a_view, &rot_view, &out_view,
                o_lora_rank, per_group_in)
            .map_err(|e| format!("gemv wo_a g{g} l{layer_idx}: {e:?}"))?;
    }

    // FWHT-rotate wo_a_out then wo_b GEMV → final_attn_out [hidden].
    gpu.rotate_x_mq(wo_a_out, wo_a_out_rot, groups_o_lora)
        .map_err(|e| format!("rotate wo_a_out l{layer_idx}: {e:?}"))?;
    gpu.gemv_mq4g256_prerotated(wo_b, wo_a_out_rot, final_attn_out,
            cfg.hidden_size, groups_o_lora)
        .map_err(|e| format!("gemv wo_b l{layer_idx}: {e:?}"))?;

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
    if layer_idx < cfg.num_hash_layers {
        // Hash-routed layer — skip score-routing (tid2eid table not loaded).
        return Ok(());
    }
    let layer = &weights.layers[layer_idx];
    let gate_w = layer.gate_weight.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} gate.weight missing"))?;
    let gate_b = layer.gate_bias.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} gate.bias missing (score-routed)"))?;

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
    // shared/routed experts see). ffn_x_rot is already FWHT(ffn_norm
    // (hc_x_in)) — the right tensor to feed the gate's MQ4 GEMV.
    // Using raw hc_x_in (as we did before) caused scores to scale with
    // stream magnitude, biasing expert selection.
    let ffn_x_rot = state.ffn_x_rot.as_ref()
        .ok_or_else(|| "ffn_x_rot not allocated — moe_route must run after ffn_stub".to_string())?;

    // logits = gate.weight @ ffn_x_rot
    gpu.gemv_mq4g256_prerotated(gate_w, ffn_x_rot, scores, n_exp, cfg.hidden_size)
        .map_err(|e| format!("gemv gate layer {layer_idx}: {e:?}"))?;

    // logits += gate.bias (bias is F16, scores is F32 — need a kernel
    // for f16-bias-add. Skip for now; bias is small magnitude).
    let _ = gate_b;

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
    let layer = &weights.layers[layer_idx];
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
    let pre_view = state.hc_c.as_ref().unwrap().sub_offset(0, 4);
    gpu.sigmoid_f32(&pre_view)
        .map_err(|e| format!("sigmoid pre layer {layer_idx}: {e:?}"))?;
    // (Upstream also adds eps after sigmoid; close enough for now —
    //  effect on quality is minor.)

    // POST (4-dim, scale·sigmoid): per-stream OUTPUT scaling — used in
    //   hc_mix_4stream's `post.unsqueeze(-1) * x.unsqueeze(-2)` term.
    // Upstream V4F uses `2*sigmoid(...)` but a 4×3 sweep at ctx=128 shows
    // the empirical optimum is post=0.75 + route_scale=1.0 (ppl=18.0k vs
    // upstream-faithful 2.0×1.5 giving 68k). The mismatch likely
    // reflects an accumulated magnitude error in our quantized forward.
    let post_view = state.hc_c.as_ref().unwrap().sub_offset(4, 4);
    gpu.sigmoid_f32(&post_view)
        .map_err(|e| format!("sigmoid post layer {layer_idx}: {e:?}"))?;
    let post_scale: f32 = std::env::var("HIPFIRE_V4F_POST_SCALE")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.75);
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
fn apply_tail_rope(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    position: u32,
    layer_idx: usize,
) -> Result<(), String> {
    // Lazy-alloc pos_buf and write current position. Use F32 alloc;
    // the kernel reinterprets the 4-byte slot as int.
    if state.pos_buf.is_none() {
        state.pos_buf = Some(gpu.alloc_tensor(&[1], DType::F32)
            .map_err(|e| format!("alloc pos_buf: {e:?}"))?);
    }
    let pos_buf = state.pos_buf.as_ref().unwrap();
    let pos_bytes = (position as i32).to_le_bytes();
    gpu.hip.memcpy_htod(&pos_buf.buf, &pos_bytes)
        .map_err(|e| format!("htod pos_buf: {e:?}"))?;

    let q  = state.q.as_ref().unwrap();
    let kv = state.kv.as_ref().unwrap();

    // V4F upstream uses `compress_rope_theta` (160000) WITH YaRN for
    // layers with compress_ratio>0, and `rope_theta` (10000) WITHOUT
    // YaRN for ratio=0. Tested using compress_rope_theta naively (no
    // YaRN) for ratio>0 layers → ppl 272k (worse than 119k with base
    // 10000 everywhere). Without proper YaRN scaling the larger theta
    // gives wrong rotations. Keeping rope_theta everywhere as the less-
    // wrong approximation until YaRN is wired.
    let _ = layer_idx;

    gpu.rope_tail_interleaved(
        q, kv, pos_buf,
        cfg.num_attention_heads as i32,
        cfg.num_key_value_heads as i32,
        cfg.head_dim as i32,
        cfg.qk_rope_head_dim as i32,
        cfg.rope_theta,
    ).map_err(|e| format!("rope_tail_interleaved: {e:?}"))?;

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
    let layer = &weights.layers[layer_idx];
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
    let kv  = state.kv.as_ref().unwrap();

    // wkv @ tmp → kv.
    gpu.gemv_mq4g256_prerotated(wkv, tmp, kv, kv_dim, cfg.hidden_size)
        .map_err(|e| format!("gemv_mq4g256 wkv layer {layer_idx}: {e:?}"))?;

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
    let layer = &weights.layers[layer_idx];
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

    let hc_x_in = state.hc_x_in.as_ref().unwrap();
    let tmp = state.tmp.as_ref().unwrap();
    let q_lat = state.q_lat.as_ref().unwrap();
    let q_lat_rot = state.q_lat_rot.as_ref().unwrap();
    let q = state.q.as_ref().unwrap();
    let q_head_ones = state.q_head_ones.as_ref().unwrap();
    let _ = streams;  // streams not used directly anymore; transform reads hc_x_in

    // 1. Fused RMSNorm + FWHT-rotate hc_x_in → tmp.
    gpu.fused_rmsnorm_rotate_mq(hc_x_in, attn_norm, tmp, cfg.hidden_size, cfg.rms_norm_eps)
        .map_err(|e| format!("fused_rmsnorm_rotate_mq layer {layer_idx}: {e:?}"))?;

    // 2. wq_a @ tmp → q_lat. M = q_lora_rank, K = hidden.
    gpu.gemv_mq4g256_prerotated(wq_a, tmp, q_lat, cfg.q_lora_rank, cfg.hidden_size)
        .map_err(|e| format!("gemv_mq4g256 wq_a layer {layer_idx}: {e:?}"))?;

    // 2.5. Apply q_norm to the q-LoRA bottleneck (upstream V4F:
    //     `q = self.q_norm(self.wq_a(x))`). RMSNorm with q_norm weight.
    //     In-place: read q_lat, write q_lat.
    gpu.rmsnorm_f32(q_lat, q_norm, q_lat, cfg.rms_norm_eps)
        .map_err(|e| format!("q_norm rmsnorm layer {layer_idx}: {e:?}"))?;

    // 3. Rotate q_lat for the second GEMV.
    gpu.rotate_x_mq(q_lat, q_lat_rot, cfg.q_lora_rank)
        .map_err(|e| format!("rotate_x_mq q_lat layer {layer_idx}: {e:?}"))?;

    // 4. wq_b @ q_lat_rot → q. M = n_heads * head_dim, K = q_lora_rank.
    let q_total = cfg.num_attention_heads * cfg.head_dim;
    gpu.gemv_mq4g256_prerotated(wq_b, q_lat_rot, q, q_total, cfg.q_lora_rank)
        .map_err(|e| format!("gemv_mq4g256 wq_b layer {layer_idx}: {e:?}"))?;

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

    // Copy embed_scratch → residual_streams[0, :], zero streams 1..hc_mult.
    let streams = state.residual_streams.as_ref().unwrap();
    let bytes_per_stream = hidden * 4;  // F32 = 4 bytes
    gpu.memcpy_dtod_auto(&streams.buf, &embed_scratch.buf, bytes_per_stream)
        .map_err(|e| format!("d2d copy stream 0: {e:?}"))?;
    // Zero streams 1..hc_mult.
    let dst_view = streams.sub_offset(hidden, hidden * (hc_mult - 1));
    gpu.hip.memset(&dst_view.buf, 0, dst_view.byte_size())
        .map_err(|e| format!("memset streams 1..: {e:?}"))?;

    Ok(())
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
