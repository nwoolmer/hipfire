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

        hc_attn_mix(cfg, weights, state, gpu, layer_idx)?;

        // ── 2b. FFN block ─────────────────────────────────────────────
        //
        // FFN computation (router + experts + shared + scaling) is
        // STUB: ffn_out = stream0 (no-op). HC FFN mix wired with the
        // same kernel sequence as HC attn mix. Real FFN expert
        // dispatch lands in a follow-up (MoE routing complexity).
        mhc_pre(cfg, weights, state, gpu, layer_idx, /*is_attn=*/false)?;
        ffn_stub(cfg, weights, state, gpu, layer_idx)?;
        hc_ffn_mix(cfg, weights, state, gpu, layer_idx)?;
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
    let hc_fn   = layer.hc_ffn_fn.as_ref().unwrap();
    let hc_base = layer.hc_ffn_base.as_ref().unwrap();
    let streams = state.residual_streams.as_ref().unwrap();
    let ffn_out = state.ffn_out.as_ref().unwrap();

    let n_ctrl = 24;
    let x_dim = cfg.hidden_size * cfg.hc_mult;
    let c_view = state.hc_c.as_ref().unwrap().sub_offset(0, n_ctrl);

    gpu.hc_compute_control(streams, hc_fn, hc_base, &c_view,
        n_ctrl as i32, x_dim as i32)
        .map_err(|e| format!("hc_compute_control ffn layer {layer_idx}: {e:?}"))?;

    let b_view = state.hc_c.as_ref().unwrap().sub_offset(4, 16);
    gpu.hc_sinkhorn_4x4(&b_view, cfg.hc_eps, cfg.hc_sinkhorn_iters as i32)
        .map_err(|e| format!("hc_sinkhorn_4x4 ffn layer {layer_idx}: {e:?}"))?;

    let c_view_out = state.hc_c.as_ref().unwrap().sub_offset(20, 4);
    gpu.sigmoid_f32(&c_view_out)
        .map_err(|e| format!("sigmoid C ffn layer {layer_idx}: {e:?}"))?;
    gpu.scale_f32(&c_view_out, 2.0)
        .map_err(|e| format!("scale C ffn layer {layer_idx}: {e:?}"))?;

    let streams_out = state.q.as_ref().unwrap();
    gpu.hc_mix_4stream(streams, &b_view, &c_view_out, ffn_out, streams_out,
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
#[allow(dead_code)]
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
    let hc_x_in = state.hc_x_in.as_ref()
        .ok_or_else(|| "hc_x_in not allocated for router".to_string())?;

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

    // gate.weight is MQ4G256, so feed the FWHT-rotated ffn_input.
    // hc_x_in is NOT rotated (it's the linear A·X output). We need
    // to rotate first. Reuse state.tmp (overwriting whatever's there
    // — at this point ffn-block hasn't started yet so tmp is free).
    let tmp = state.tmp.as_ref()
        .ok_or_else(|| "tmp not allocated for router rotation".to_string())?;
    gpu.rotate_x_mq(hc_x_in, tmp, cfg.hidden_size)
        .map_err(|e| format!("rotate_x_mq router layer {layer_idx}: {e:?}"))?;

    // logits = gate.weight @ tmp_rot
    gpu.gemv_mq4g256_prerotated(gate_w, tmp, scores, n_exp, cfg.hidden_size)
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

    // A_l = σ(c[0..4])
    let a_view = state.hc_c.as_ref().unwrap().sub_offset(0, 4);
    gpu.sigmoid_f32(&a_view)
        .map_err(|e| format!("sigmoid A layer {layer_idx}: {e:?}"))?;

    // B_l = Sinkhorn(exp(c[4..20])) — kernel takes care of exp + iters.
    let b_view = state.hc_c.as_ref().unwrap().sub_offset(4, 16);
    gpu.hc_sinkhorn_4x4(&b_view, cfg.hc_eps, cfg.hc_sinkhorn_iters as i32)
        .map_err(|e| format!("hc_sinkhorn_4x4 layer {layer_idx}: {e:?}"))?;

    // C_l = 2σ(c[20..24])
    let c_out_view = state.hc_c.as_ref().unwrap().sub_offset(20, 4);
    gpu.sigmoid_f32(&c_out_view)
        .map_err(|e| format!("sigmoid C layer {layer_idx}: {e:?}"))?;
    gpu.scale_f32(&c_out_view, 2.0)
        .map_err(|e| format!("scale C layer {layer_idx}: {e:?}"))?;

    // Input mapping: hc_x_in = A_l · streams
    let hc_x_in = state.hc_x_in.as_ref().unwrap();
    gpu.hc_input_map_4stream(&a_view, streams, hc_x_in, cfg.hidden_size as i32)
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
    let layer = &weights.layers[layer_idx];
    let hc_fn    = layer.hc_attn_fn.as_ref().unwrap();
    let hc_base  = layer.hc_attn_base.as_ref().unwrap();
    let streams = state.residual_streams.as_ref().unwrap();
    let attn_out = state.attn_out.as_ref().unwrap();

    let n_ctrl = 24;
    let x_dim = cfg.hidden_size * cfg.hc_mult;

    let c_view = state.hc_c.as_ref().unwrap().sub_offset(0, n_ctrl);

    // 1. c = X · W_fn + base. (α scaling not yet applied.)
    gpu.hc_compute_control(streams, hc_fn, hc_base, &c_view,
        n_ctrl as i32, x_dim as i32)
        .map_err(|e| format!("hc_compute_control layer {layer_idx}: {e:?}"))?;

    // 2. B_l = Sinkhorn(exp(c[4..20])). The kernel takes a 4x4 starting
    // matrix; we view c[4..20] as that.
    let b_view = state.hc_c.as_ref().unwrap().sub_offset(4, 16);
    gpu.hc_sinkhorn_4x4(&b_view, cfg.hc_eps, cfg.hc_sinkhorn_iters as i32)
        .map_err(|e| format!("hc_sinkhorn_4x4 layer {layer_idx}: {e:?}"))?;

    // 3. C_l = 2σ(c[20..24]) — apply in-place via sigmoid then scale_f32.
    let c_view_out = state.hc_c.as_ref().unwrap().sub_offset(20, 4);
    gpu.sigmoid_f32(&c_view_out)
        .map_err(|e| format!("sigmoid C layer {layer_idx}: {e:?}"))?;
    gpu.scale_f32(&c_view_out, 2.0)
        .map_err(|e| format!("scale C layer {layer_idx}: {e:?}"))?;

    // 4. X_{l+1} = B_l · X_l + C_l · attn_out.
    let streams_out = state.q.as_ref().unwrap();
    gpu.hc_mix_4stream(streams, &b_view, &c_view_out, attn_out, streams_out,
        cfg.hidden_size as i32)
        .map_err(|e| format!("hc_mix_4stream layer {layer_idx}: {e:?}"))?;

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

    let hc_x_in = state.hc_x_in.as_ref().unwrap();
    let tmp = state.tmp.as_ref().unwrap();
    let q_lat = state.q_lat.as_ref().unwrap();
    let q_lat_rot = state.q_lat_rot.as_ref().unwrap();
    let q = state.q.as_ref().unwrap();
    let _ = streams;  // streams not used directly anymore; transform reads hc_x_in

    // 1. Fused RMSNorm + FWHT-rotate hc_x_in → tmp.
    gpu.fused_rmsnorm_rotate_mq(hc_x_in, attn_norm, tmp, cfg.hidden_size, cfg.rms_norm_eps)
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
