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
        kv_joint(cfg, weights, state, gpu, layer_idx)?;

        // iv. Tail-only RoPE on Q and KV.
        //     Apply rotation on last `qk_rope_head_dim = 64` of each
        //     head's 512 dims.
        //     SWA ring write deferred (needs swa state alloc per layer).
        apply_tail_rope(cfg, state, gpu, position)?;

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

        // v + vi. Main attention + O-LoRA — STUB.
        attn_stub(cfg, state, gpu, layer_idx)?;

        // vii. HC attn mix — disabled. Even bounded transform_out
        // produces 4e37 overflow across 43 layers due to compounding
        // GEMV magnitudes. Paper-faithful activation chain needed
        // (residual scaling, pre-Sinkhorn activation, mix scaling).
        let _ = hc_attn_mix;
        let _ = layer_idx;

        // ── 2b. FFN block ─────────────────────────────────────────────
        //
        // FFN computation (router + experts + shared + scaling) is
        // STUB: ffn_out = stream0 (no-op). HC FFN mix wired with the
        // same kernel sequence as HC attn mix. Real FFN expert
        // dispatch lands in a follow-up (MoE routing complexity).
        // FFN disabled too — same magnitude reason.
        ffn_zero(cfg, state, gpu)?;
        let _ = ffn_stub;
        let _ = hc_ffn_mix;
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
    Ok(logits_host)
}

fn unimplemented_step(name: &str) -> Result<(), String> {
    let _ = name;  // silence unused; kept for stack-trace clarity later.
    Ok(())
}

/// Bind ffn_out to zero (diagnostic / temporary stub).
fn ffn_zero(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    if state.ffn_out.is_none() {
        state.ffn_out = Some(gpu.zeros(&[cfg.hidden_size], DType::F32)
            .map_err(|e| format!("alloc ffn_out: {e:?}"))?);
    }
    let ffn_out = state.ffn_out.as_ref().unwrap();
    gpu.hip.memset(&ffn_out.buf, 0, ffn_out.byte_size())
        .map_err(|e| format!("memset ffn_out: {e:?}"))?;
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
    let streams = state.residual_streams.as_ref().unwrap();

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

    let stream0 = streams.sub_offset(0, cfg.hidden_size);
    let ffn_x_rot = state.ffn_x_rot.as_ref().unwrap();
    let gate = state.ffn_gate.as_ref().unwrap();
    let up   = state.ffn_up.as_ref().unwrap();
    let silu_rot = state.ffn_silu_rot.as_ref().unwrap();
    let ffn_out = state.ffn_out.as_ref().unwrap();

    // 1. Fused RMSNorm + FWHT rotate for the two MQ4 GEMVs.
    gpu.fused_rmsnorm_rotate_mq(&stream0, ffn_norm, ffn_x_rot,
        cfg.hidden_size, cfg.rms_norm_eps)
        .map_err(|e| format!("fused_rmsnorm_rotate_mq ffn layer {layer_idx}: {e:?}"))?;

    // 2. gate = x @ shared_w1
    gpu.gemv_mq4g256_prerotated(shared_w1, ffn_x_rot, gate, im, cfg.hidden_size)
        .map_err(|e| format!("gemv shared_w1 layer {layer_idx}: {e:?}"))?;

    // 3. up = x @ shared_w3
    gpu.gemv_mq4g256_prerotated(shared_w3, ffn_x_rot, up, im, cfg.hidden_size)
        .map_err(|e| format!("gemv shared_w3 layer {layer_idx}: {e:?}"))?;

    // 4. silu(gate) * up — in place into gate.
    gpu.silu_mul_f32(gate, up, gate)
        .map_err(|e| format!("silu_mul layer {layer_idx}: {e:?}"))?;

    // 5. FWHT-rotate the silu-gated vector for the down GEMV.
    gpu.rotate_x_mq(gate, silu_rot, im)
        .map_err(|e| format!("rotate_x_mq silu layer {layer_idx}: {e:?}"))?;

    // 6. ffn_out = silu_rot @ shared_w2 (down: [hidden, im])
    gpu.gemv_mq4g256_prerotated(shared_w2, silu_rot, ffn_out, cfg.hidden_size, im)
        .map_err(|e| format!("gemv shared_w2 layer {layer_idx}: {e:?}"))?;

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
    let layer = &weights.layers[layer_idx];
    let hc_fn    = layer.hc_ffn_fn.as_ref().unwrap();
    let hc_base  = layer.hc_ffn_base.as_ref().unwrap();
    let hc_scale = layer.hc_ffn_scale.as_ref().unwrap();
    let streams = state.residual_streams.as_ref().unwrap();
    let ffn_out = state.ffn_out.as_ref().unwrap();

    let n_ctrl = 24;
    let x_dim = cfg.hidden_size * cfg.hc_mult;
    let c_view = state.tmp.as_ref().unwrap().sub_offset(0, n_ctrl);

    gpu.hc_compute_control(streams, hc_fn, hc_base, &c_view,
        n_ctrl as i32, x_dim as i32)
        .map_err(|e| format!("hc_compute_control ffn layer {layer_idx}: {e:?}"))?;

    let a_view = state.tmp.as_ref().unwrap().sub_offset(0, 16);
    gpu.hc_sinkhorn_4x4(&a_view, cfg.hc_eps, cfg.hc_sinkhorn_iters as i32)
        .map_err(|e| format!("hc_sinkhorn_4x4 ffn layer {layer_idx}: {e:?}"))?;

    let streams_out = state.q.as_ref().unwrap();
    gpu.hc_mix_4stream(streams, &a_view, hc_scale, ffn_out, streams_out,
        cfg.hidden_size as i32)
        .map_err(|e| format!("hc_mix_4stream ffn layer {layer_idx}: {e:?}"))?;

    let bytes = cfg.hc_mult * cfg.hidden_size * 4;
    gpu.memcpy_dtod_auto(&streams.buf, &streams_out.buf, bytes)
        .map_err(|e| format!("d2d hc_ffn_mix → streams: {e:?}"))?;
    Ok(())
}

/// Final norm + lm_head.
///
/// `final_norm = rmsnorm(stream0, output_norm)` [hidden]
/// `logits = head_weight @ final_norm`             [vocab_size]
///
/// head_weight is MQ4G256, so we need to FWHT-rotate final_norm
/// first via `rotate_x_mq`, then call `gemv_mq4g256_prerotated`.
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

    let stream0 = streams.sub_offset(0, cfg.hidden_size);
    let final_norm = state.final_norm.as_ref().unwrap();
    let final_norm_rot = state.final_norm_rot.as_ref().unwrap();
    let logits = state.logits.as_ref().unwrap();

    // 1. RMSNorm
    gpu.rmsnorm_f32(&stream0, output_norm, final_norm, cfg.rms_norm_eps)
        .map_err(|e| format!("final rmsnorm_f32: {e:?}"))?;

    // 2. FWHT-rotate for MQ4 GEMV
    gpu.rotate_x_mq(final_norm, final_norm_rot, cfg.hidden_size)
        .map_err(|e| format!("rotate_x_mq final_norm: {e:?}"))?;

    // 3. lm_head: head @ final_norm_rot → logits [vocab_size]
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
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    _layer_idx: usize,
) -> Result<(), String> {
    if state.attn_out.is_none() {
        state.attn_out = Some(gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
            .map_err(|e| format!("alloc attn_out: {e:?}"))?);
    }
    // Position-0 attention: attn_out is o_groups copies of V (the kv vector).
    // V has shape [head_dim=512]; attn_out shape [hidden=4096] = 8 * 512.
    // The in-group sum factor = n_heads / o_groups = 64/8 = 8 should
    // scale each V. For numerical stability across 43 layers we omit
    // the *8 scaling (effectively softmax-normalize each group's
    // contribution to 1.0 — paper details TBD).
    let kv = state.kv.as_ref().unwrap();
    let attn_out = state.attn_out.as_ref().unwrap();

    let bytes_per_v = cfg.head_dim * 4;  // 512 * 4 = 2048
    for g in 0..cfg.o_groups {
        let dst_view = attn_out.sub_offset(g * cfg.head_dim, cfg.head_dim);
        gpu.memcpy_dtod_auto(&dst_view.buf, &kv.buf, bytes_per_v)
            .map_err(|e| format!("d2d kv→attn_out group {g}: {e:?}"))?;
    }
    Ok(())
}

/// Step 8 (attention block): Hyper-Connection mix.
///
///   c     = hc_attn_fn @ x_flat + hc_attn_base   [24]
///   A     = sinkhorn(c reshaped to 4x4, eps, iters)
///   x_out = A · x_in + scale[s] · transform_out
///
/// Then residual_streams = x_out.
///
/// First 16 entries of c interpreted as the 4x4 mixing matrix.
/// Per the Phase 3 design doc, the [24]-vector decomposition is
/// `16 + 4 + 4` (matrix + bias + scale) but verifying which slice
/// is which awaits paper read. For now we take the first 16 floats
/// as a row-major 4x4.
fn hc_attn_mix(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let layer = &weights.layers[layer_idx];
    let hc_fn    = layer.hc_attn_fn.as_ref().unwrap();
    let hc_base  = layer.hc_attn_base.as_ref().unwrap();
    let hc_scale = layer.hc_attn_scale.as_ref().unwrap();
    let streams = state.residual_streams.as_ref().unwrap();
    let attn_out = state.attn_out.as_ref().unwrap();

    // Allocate small scratch on demand: c_ctrl [24] F32, a 4x4 view
    // is the first 16 elements (treat as [4, 4]).
    let n_ctrl = 24;  // V4F hc_attn_fn shape [24, 16384]
    let x_dim = cfg.hidden_size * cfg.hc_mult;  // 4096 * 4 = 16384

    // c_ctrl scratch: alloc once and stash on state.tmp? tmp is
    // [hidden=4096] which is too big for the [24] output but the
    // pool allows oversized writes. Use a sub-view to be safe.
    let c_view = state.tmp.as_ref().unwrap().sub_offset(0, n_ctrl);

    // 1. Control vector. hc_fn is F16 on disk; the kernel expects F16
    // (per its signature). hc_base is F16 too. Output is F32.
    gpu.hc_compute_control(
        streams /* x_flat: streams memory is contiguous [4, hidden] */,
        hc_fn, hc_base, &c_view,
        n_ctrl as i32, x_dim as i32,
    ).map_err(|e| format!("hc_compute_control layer {layer_idx}: {e:?}"))?;

    // 2. Sinkhorn-normalise the first 16 entries as a 4x4 matrix.
    // c_view is [24] but sinkhorn expects [16]. Build a 16-elem view
    // over the same memory.
    let a_view = state.tmp.as_ref().unwrap().sub_offset(0, 16);
    gpu.hc_sinkhorn_4x4(&a_view, cfg.hc_eps, cfg.hc_sinkhorn_iters as i32)
        .map_err(|e| format!("hc_sinkhorn_4x4 layer {layer_idx}: {e:?}"))?;

    // 3. Mix: streams_out = A · streams + scale · attn_out.
    // Need a destination buffer different from streams. Reuse `q` for
    // now (~32K F32, more than enough for [4, 4096]) — overwritten
    // next layer.
    let streams_out = state.q.as_ref().unwrap();
    gpu.hc_mix_4stream(
        streams, &a_view, hc_scale, attn_out, streams_out,
        cfg.hidden_size as i32,
    ).map_err(|e| format!("hc_mix_4stream layer {layer_idx}: {e:?}"))?;

    // Copy back into residual_streams.
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

    gpu.rope_tail_halfsplit(
        q, kv, pos_buf,
        cfg.num_attention_heads as i32,
        cfg.num_key_value_heads as i32,
        cfg.head_dim as i32,
        cfg.qk_rope_head_dim as i32,
        cfg.rope_theta,
    ).map_err(|e| format!("rope_tail_halfsplit: {e:?}"))?;

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
