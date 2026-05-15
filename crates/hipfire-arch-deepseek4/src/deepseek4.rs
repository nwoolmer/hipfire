//! Config / Weights / State types for DeepSeek V4 Flash.
//!
//! `DeepseekV4Config` mirrors the fields in the upstream
//! `config.json`. Defaults come from the released
//! `deepseek-ai/DeepSeek-V4-Flash` checkpoint.

use hipfire_runtime::hfq::HfqFile;
use serde::{Deserialize, Serialize};

/// Per-layer compression mode for the indexer / KV path.
///
/// `compress_ratios` in `config.json` is a per-layer array. The
/// observed pattern on the released V4F is `[0, 0, 4, 128, 4, 128,
/// ..., 4, 128, 4, 0]` — i.e. the first two and the last layer use
/// `0` (no compression / full attention), and the middle layers
/// alternate `4` / `128`. We carry the raw `u32` per layer rather
/// than collapsing to an enum so future fine-tunes that pick
/// different ratios still round-trip cleanly.
pub type CompressRatio = u32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepseekV4Config {
    // ── transformer-shape basics ────────────────────────────────
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,

    // ── DeepSeek-specific attention ─────────────────────────────
    /// Q-projection LoRA rank. Q goes through a `hidden × q_lora_rank`
    /// then `q_lora_rank × (n_heads · head_dim)` factorisation rather
    /// than full `hidden × (n_heads · head_dim)`.
    pub q_lora_rank: usize,
    /// O-projection LoRA rank. Same shape pattern as Q but on the
    /// output side.
    pub o_lora_rank: usize,
    /// Number of tail dimensions per head that carry RoPE. The
    /// remaining `head_dim - qk_rope_head_dim` dims are straight Q·K.
    pub qk_rope_head_dim: usize,
    /// O-projection grouping for the LoRA-bottlenecked output.
    pub o_groups: usize,

    // ── MoE ─────────────────────────────────────────────────────
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub routed_scaling_factor: f32,
    /// `noaux_tc` etc. — only `noaux_tc` is supported initially.
    pub topk_method: String,
    /// `sqrtsoftplus` — the routing-score function.
    pub scoring_func: String,
    pub norm_topk_prob: bool,
    pub swiglu_limit: f32,

    // ── Hyper-Connections ───────────────────────────────────────
    /// Number of residual streams (typically 4).
    pub hc_mult: usize,
    /// Sinkhorn iteration count for the residual gating matrix.
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,

    // ── Compressed-KV indexer ───────────────────────────────────
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    /// Per-layer compression ratio array (length =
    /// `num_hidden_layers + num_nextn_predict_layers`). `0` = no
    /// compression, otherwise the indexer stride for that layer.
    pub compress_ratios: Vec<CompressRatio>,
    /// Compressed-KV path uses its own rope_theta.
    pub compress_rope_theta: f32,

    // ── RoPE / sliding window ───────────────────────────────────
    pub rope_theta: f32,
    /// YaRN scaling factor (typically 16 for V4F's 1M context).
    pub rope_scaling_factor: f32,
    pub rope_scaling_original_max_position_embeddings: usize,
    pub rope_scaling_beta_fast: usize,
    pub rope_scaling_beta_slow: usize,
    /// SWA window length for the main attention path (V4F: 128).
    pub sliding_window: usize,

    // ── Multi-Token-Prediction (MTP) ────────────────────────────
    /// Number of next-token prediction layers appended after the
    /// main stack. V4F ships `1` (one MTP head).
    pub num_nextn_predict_layers: usize,

    // ── hash-routing (V4F-only) ─────────────────────────────────
    pub num_hash_layers: usize,
}

/// Raw upstream JSON shape — only the fields we read. Used to drive
/// `from_hfq`. We deliberately mirror the upstream key names with
/// `#[serde(rename)]` rather than renaming on the HFQ side so the
/// metadata is the byte-for-byte same JSON the converter sees.
#[derive(Debug, Deserialize)]
struct RawDeepseekV4Config {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,

    q_lora_rank: usize,
    o_lora_rank: usize,
    qk_rope_head_dim: usize,
    o_groups: usize,

    n_routed_experts: usize,
    n_shared_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    routed_scaling_factor: f32,
    topk_method: String,
    scoring_func: String,
    norm_topk_prob: bool,
    swiglu_limit: f32,

    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    hc_eps: f32,

    index_n_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    compress_ratios: Vec<u32>,
    compress_rope_theta: f32,

    rope_theta: f32,
    rope_scaling: RawYarnScaling,
    sliding_window: usize,

    num_nextn_predict_layers: usize,
    num_hash_layers: usize,
}

#[derive(Debug, Deserialize)]
struct RawYarnScaling {
    factor: f32,
    original_max_position_embeddings: usize,
    beta_fast: usize,
    beta_slow: usize,
    #[serde(rename = "type")]
    _kind: String,
}

impl DeepseekV4Config {
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        let raw: RawDeepseekV4Config = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("deepseek4: parsing metadata_json failed: {e}"))?;
        Ok(DeepseekV4Config {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            max_position_embeddings: raw.max_position_embeddings,
            rms_norm_eps: raw.rms_norm_eps,
            q_lora_rank: raw.q_lora_rank,
            o_lora_rank: raw.o_lora_rank,
            qk_rope_head_dim: raw.qk_rope_head_dim,
            o_groups: raw.o_groups,
            n_routed_experts: raw.n_routed_experts,
            n_shared_experts: raw.n_shared_experts,
            num_experts_per_tok: raw.num_experts_per_tok,
            moe_intermediate_size: raw.moe_intermediate_size,
            routed_scaling_factor: raw.routed_scaling_factor,
            topk_method: raw.topk_method,
            scoring_func: raw.scoring_func,
            norm_topk_prob: raw.norm_topk_prob,
            swiglu_limit: raw.swiglu_limit,
            hc_mult: raw.hc_mult,
            hc_sinkhorn_iters: raw.hc_sinkhorn_iters,
            hc_eps: raw.hc_eps,
            index_n_heads: raw.index_n_heads,
            index_head_dim: raw.index_head_dim,
            index_topk: raw.index_topk,
            compress_ratios: raw.compress_ratios,
            compress_rope_theta: raw.compress_rope_theta,
            rope_theta: raw.rope_theta,
            rope_scaling_factor: raw.rope_scaling.factor,
            rope_scaling_original_max_position_embeddings:
                raw.rope_scaling.original_max_position_embeddings,
            rope_scaling_beta_fast: raw.rope_scaling.beta_fast,
            rope_scaling_beta_slow: raw.rope_scaling.beta_slow,
            sliding_window: raw.sliding_window,
            num_nextn_predict_layers: raw.num_nextn_predict_layers,
            num_hash_layers: raw.num_hash_layers,
        })
    }
}

/// V4F weights — scaffold-stage placeholder. Real impl will hold per-
/// layer `WeightTensor` arrays for: Q-LoRA (`q_a`, `q_b`), KV joint
/// (`kv_a`, `kv_b`), O-LoRA (`o_a`, `o_b`), indexer (`idx_q`, `idx_k`,
/// `idx_v` over compressed positions), Hyper-Connection gating, routed
/// experts (gate / up / down), shared expert, MTP head.
pub struct DeepseekV4Weights {
    pub _scaffold: (),
}

/// V4F state — scaffold. Real impl will hold:
/// - main-path KV cache for the SWA window (`sliding_window = 128`)
/// - compressed-KV cache (per-layer, stride = `compress_ratios[l]`)
/// - 4 residual streams (Hyper-Connections)
/// - indexer top-k positions per layer
pub struct DeepseekV4State {
    pub _scaffold: (),
}

impl DeepseekV4State {
    pub fn new(_cfg: &DeepseekV4Config) -> Result<Self, String> {
        Ok(DeepseekV4State { _scaffold: () })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const V4F_CONFIG_JSON: &str = r#"{
        "vocab_size": 129280, "hidden_size": 4096, "num_hidden_layers": 43,
        "num_attention_heads": 64, "num_key_value_heads": 1, "head_dim": 512,
        "max_position_embeddings": 1048576, "rms_norm_eps": 1e-6,
        "q_lora_rank": 1024, "o_lora_rank": 1024, "qk_rope_head_dim": 64,
        "o_groups": 8,
        "n_routed_experts": 256, "n_shared_experts": 1,
        "num_experts_per_tok": 6, "moe_intermediate_size": 2048,
        "routed_scaling_factor": 1.5, "topk_method": "noaux_tc",
        "scoring_func": "sqrtsoftplus", "norm_topk_prob": true,
        "swiglu_limit": 10.0,
        "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
        "index_n_heads": 64, "index_head_dim": 128, "index_topk": 512,
        "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 0],
        "compress_rope_theta": 160000,
        "rope_theta": 10000,
        "rope_scaling": {
            "factor": 16, "original_max_position_embeddings": 65536,
            "beta_fast": 32, "beta_slow": 1, "type": "yarn"
        },
        "sliding_window": 128,
        "num_nextn_predict_layers": 1, "num_hash_layers": 3
    }"#;

    #[test]
    fn parses_v4f_config_shape() {
        let raw: RawDeepseekV4Config = serde_json::from_str(V4F_CONFIG_JSON).unwrap();
        assert_eq!(raw.num_hidden_layers, 43);
        assert_eq!(raw.head_dim, 512);
        assert_eq!(raw.qk_rope_head_dim, 64);
        assert_eq!(raw.q_lora_rank, 1024);
        assert_eq!(raw.o_lora_rank, 1024);
        assert_eq!(raw.n_routed_experts, 256);
        assert_eq!(raw.num_experts_per_tok, 6);
        assert_eq!(raw.hc_mult, 4);
        assert_eq!(raw.hc_sinkhorn_iters, 20);
        assert_eq!(raw.index_n_heads, 64);
        assert_eq!(raw.index_head_dim, 128);
        assert_eq!(raw.index_topk, 512);
        assert_eq!(raw.sliding_window, 128);
        assert_eq!(raw.compress_ratios.len(), 8);
    }
}
