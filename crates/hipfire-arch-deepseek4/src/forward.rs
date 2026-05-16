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
use rdna_compute::Gpu;

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
    //    likely [embed, 0, 0, 0] (paper-specified; verify against
    //    the V4F reference code before optimising).
    //    Output: residual_streams [4, hidden = 4096] fp16
    //    Uses: existing `embedding_lookup` against weights.token_embd
    //    (which is Q8_0 raw bytes — the dequant kernel reads it).
    unimplemented_step("embed → residual streams init")?;

    // 2. Per-layer forward.
    for layer_idx in 0..cfg.num_hidden_layers {
        let layer = &weights.layers[layer_idx];
        let l_state = &mut state._indexer[layer_idx];
        let l_attn  = &mut state._attention[layer_idx];

        // ── 2a. Attention block ───────────────────────────────────────
        //
        // i. RMSNorm against `layer.attn_norm` — input is residual
        //    stream[0] (paper says one stream feeds the transform;
        //    verify against reference for which index).
        unimplemented_step("attn RMSNorm")?;

        // ii. Q via Q-LoRA: x @ wq_a → q_lat, q_lat @ wq_b → q
        //     q has shape [n_heads = 64, head_dim = 512].
        //     Then apply tail-only RoPE on q[:, head_dim - 64..].
        //     Uses: `gpu.rope_tail_halfsplit` with freq_base = rope_theta = 10000.
        unimplemented_step("Q-LoRA + tail RoPE on Q")?;

        // iii. Joint KV: x @ wkv → kv, split into k + v.
        //      Apply tail RoPE on k. Append k/v to SWA ring at
        //      slot = position % 128.
        unimplemented_step("KV joint + tail RoPE on K + SWA ring write")?;

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
