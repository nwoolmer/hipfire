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
        // i+ii. Fused RMSNorm + Q-LoRA. The fused_rmsnorm_rotate_mq
        //       kernel inside q_lora() does the RMSNorm AND the
        //       FWHT rotation needed by MQ4 GEMV in a single call.
        //       (V4F paper: one stream feeds the transform; we
        //       conservatively use stream 0 — revisit during
        //       numerical-correctness gate.)
        q_lora(cfg, weights, state, gpu, layer_idx)?;

        // (Q-LoRA call moved above into the fused RMSNorm + GEMV step.)

        // iii. Joint KV: wkv @ tmp → kv [head_dim = 512] (tied K=V).
        //      Apply tail RoPE on the last 64 of 512 dims.
        //      SWA ring write deferred (needs swa_k state alloc per
        //      layer; lands in step 5).
        kv_joint(cfg, weights, state, gpu, layer_idx)?;

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
        if layer.compress_ratio > 0 {
            unimplemented_step("indexer score → top_k → KV gather")?;
        }

        // v. Main attention over (SWA window ∪ gathered top-k).
        //    Q[n_heads, head_dim] · concat(swa_k, k_gathered)[n_kv_heads, head_dim, m].
        //    Output: attn_out [n_heads, head_dim].
        //    Uses: existing FlashAttention kernel with `start_pos` wrap parameter
        //    (Phase 4 modification — needs adding).
        unimplemented_step("main attention over SWA + gathered")?;

        // vi. O-LoRA: attn_out @ wo_a → o_lat, o_lat @ wo_b → x_attn
        unimplemented_step("O-LoRA")?;

        // vii. Hyper-Connection mix for attention block:
        //      a. `gpu.hc_compute_control(x_flat, hc_attn_fn, hc_attn_base, c_ctrl)`
        //      b. Interpret first 16 entries of c_ctrl as a 4x4 matrix.
        //      c. `gpu.hc_sinkhorn_4x4(A, eps = hc_eps, iters = hc_sinkhorn_iters)`
        //      d. `gpu.hc_mix_4stream(x_in = streams, A, scale = hc_attn_scale, transform_out = x_attn, x_out)`
        //      e. Update residual_streams = x_out.
        unimplemented_step("HC attn mix")?;

        // ── 2b. FFN block ─────────────────────────────────────────────
        //
        // i. RMSNorm against `layer.ffn_norm` on residual stream[0].
        unimplemented_step("FFN RMSNorm")?;

        // ii. Router: x @ gate.weight + (gate.bias if not hash_routed)
        //     → top-k expert indices (= num_experts_per_tok = 6).
        //     V4F: noaux_tc routing, sqrtsoftplus scoring.
        //     First `num_hash_layers` layers (= 3 on V4F) use the
        //     `tid2eid` hash table instead of softmax routing.
        unimplemented_step("MoE router (noaux_tc or hash)")?;

        // iii. Shared expert: x @ shared_w1, x @ shared_w3 (gate/up),
        //      silu(gate) * up, then @ shared_w2 (down).
        //      Output: shared_out [hidden].
        unimplemented_step("shared expert (w1/w3/w2)")?;

        // iv. Routed experts: for each of top-k (= 6) expert indices E,
        //     run x @ expert_w1[E], x @ expert_w3[E], silu*up,
        //     @ expert_w2[E]. Sum with topk_weights[E] * expert_out.
        //     Output: routed_out [hidden].
        unimplemented_step("routed experts (top-6 of 256)")?;

        // v. x_ffn = shared_out + routed_scaling_factor * routed_out.
        //    routed_scaling_factor = 1.5 (config).
        unimplemented_step("FFN combine: shared + routed_scaling * routed")?;

        // vi. HC FFN mix:
        //      a. `gpu.hc_compute_control(x_flat, hc_ffn_fn, hc_ffn_base, c_ctrl)`
        //      b. `gpu.hc_sinkhorn_4x4(A, ...)`
        //      c. `gpu.hc_mix_4stream(streams, A, hc_ffn_scale, x_ffn, streams_out)`
        unimplemented_step("HC ffn mix")?;
    }

    // 3. Final norm + LM head.
    //    a. RMSNorm of residual_streams[0] against weights.output_norm
    //    b. logits = x @ weights.head (Q8F16 or MQ4 quantized GEMV)
    //    c. Apply head-level Hyper-Connection scale if present
    //       (V4F: hc_head_base/fn/scale exist; check if applied at output).
    unimplemented_step("final norm + lm_head")?;

    Err("V4F forward: layout-only — no executable forward yet".to_string())
}

fn unimplemented_step(name: &str) -> Result<(), String> {
    let _ = name;  // silence unused; kept for stack-trace clarity later.
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

    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    if state.kv.is_none() {
        state.kv = Some(gpu.alloc_tensor(&[kv_dim], DType::F32)
            .map_err(|e| format!("alloc kv: {e:?}"))?);
    }
    let tmp = state.tmp.as_ref().unwrap();
    let kv  = state.kv.as_ref().unwrap();

    gpu.gemv_mq4g256_prerotated(wkv, tmp, kv, kv_dim, cfg.hidden_size)
        .map_err(|e| format!("gemv_mq4g256 wkv layer {layer_idx}: {e:?}"))?;

    // Tail-only RoPE on the last qk_rope_head_dim of kv. The rope
    // kernel takes Q and K separately and rotates both; here we
    // apply to KV only (Q has its own rotation when wq_b output
    // lands — TODO once position counter and head iteration are
    // wired). For per-head loop semantics, treat kv as 1 head of
    // head_dim=512 — n_heads_q = 0 (no Q to rotate from this call).
    //
    // Skipped for now (needs pos_buf in state); rope_tail_halfsplit
    // will be called once on (q, k) when position counter is
    // available.
    let _ = (kv,);
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
        let q_total = cfg.num_attention_heads * cfg.head_dim;
        state.q = Some(gpu.alloc_tensor(&[q_total], DType::F32)
            .map_err(|e| format!("alloc q: {e:?}"))?);
    }

    let stream0 = streams.sub_offset(0, cfg.hidden_size);
    let tmp = state.tmp.as_ref().unwrap();
    let q_lat = state.q_lat.as_ref().unwrap();
    let q_lat_rot = state.q_lat_rot.as_ref().unwrap();
    let q = state.q.as_ref().unwrap();

    // 1. Fused RMSNorm + FWHT-rotate stream0 → tmp.
    gpu.fused_rmsnorm_rotate_mq(&stream0, attn_norm, tmp, cfg.hidden_size, cfg.rms_norm_eps)
        .map_err(|e| format!("fused_rmsnorm_rotate_mq layer {layer_idx}: {e:?}"))?;

    // 2. wq_a @ tmp → q_lat. M = q_lora_rank, K = hidden.
    gpu.gemv_mq4g256_prerotated(wq_a, tmp, q_lat, cfg.q_lora_rank, cfg.hidden_size)
        .map_err(|e| format!("gemv_mq4g256 wq_a layer {layer_idx}: {e:?}"))?;

    // 3. Rotate q_lat for the second GEMV.
    gpu.rotate_x_mq(q_lat, q_lat_rot, cfg.q_lora_rank)
        .map_err(|e| format!("rotate_x_mq q_lat layer {layer_idx}: {e:?}"))?;

    // 4. wq_b @ q_lat_rot → q. M = n_heads * head_dim, K = q_lora_rank.
    let q_total = cfg.num_attention_heads * cfg.head_dim;
    gpu.gemv_mq4g256_prerotated(wq_b, q_lat_rot, q, q_total, cfg.q_lora_rank)
        .map_err(|e| format!("gemv_mq4g256 wq_b layer {layer_idx}: {e:?}"))?;

    // 5. Apply tail-only RoPE on last qk_rope_head_dim of each head.
    // TODO: this needs a `pos_buf` tensor with the current position
    // and a separate K argument (rope_tail_halfsplit applies to both
    // Q and K). For Q-only forward we'd pass q as both args, but that
    // would double-apply rotation. Defer until KV step lands and we
    // call rope_tail_halfsplit once on (q, k) together.
    // For now: skip rotation here; the KV step will do it.

    Ok(())
}

/// Step 2 (attention block): RMSNorm of residual stream 0 against
/// `layer.attn_norm`. Output in `state.tmp [hidden] f32`.
fn attn_rms_norm(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let layer = &weights.layers[layer_idx];
    let attn_norm = layer.attn_norm.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} attn_norm not uploaded"))?;
    let streams = state.residual_streams.as_ref()
        .ok_or_else(|| "residual_streams not allocated".to_string())?;
    let tmp = state.tmp.as_ref()
        .ok_or_else(|| "tmp not allocated".to_string())?;

    // Stream 0 view: first `hidden` floats of the [hc_mult, hidden] tensor.
    let stream0 = streams.sub_offset(0, cfg.hidden_size);
    gpu.rmsnorm_f32(&stream0, attn_norm, tmp, cfg.rms_norm_eps)
        .map_err(|e| format!("rmsnorm_f32 layer {layer_idx}: {e:?}"))?;
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
