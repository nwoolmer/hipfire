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
        // The quantizer wraps the V4F config inside an outer
        // `{"architecture":..., "config":{...}, "tokenizer":...,
        // "tokenizer_config":...}` envelope (matches the Qwen3.5
        // pattern; see crates/hipfire-quantize/src/main.rs around
        // line ~3805). Unwrap the inner `config` slice before parsing.
        let wrapper: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("deepseek4: metadata_json not valid JSON: {e}"))?;
        let inner = wrapper.get("config").ok_or_else(|| {
            "deepseek4: metadata_json missing `config` wrapper".to_string()
        })?;
        let raw: RawDeepseekV4Config = serde_json::from_value(inner.clone())
            .map_err(|e| format!("deepseek4: parsing inner config failed: {e}"))?;
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

/// Per-layer GPU-resident weights. Slots match V4F shipped tensor
/// inventory; each is `Option<GpuTensor>` so partial-upload paths
/// (host walk only / minimal upload / full upload) can populate
/// progressively. Forward bring-up asserts all relevant slots are
/// Some before dispatching.
pub struct DeepseekV4LayerWeights {
    pub compress_ratio: u32,  // 0 = no indexer; otherwise stride

    // Norms (F16 vectors).
    pub attn_norm: Option<rdna_compute::GpuTensor>,
    pub ffn_norm:  Option<rdna_compute::GpuTensor>,
    pub q_norm:    Option<rdna_compute::GpuTensor>,
    pub kv_norm:   Option<rdna_compute::GpuTensor>,
    pub attn_sink: Option<rdna_compute::GpuTensor>,  // [n_heads]

    // Attention LoRA + KV joint (MQ-family quantized).
    pub wq_a:   Option<rdna_compute::GpuTensor>,
    pub wq_b:   Option<rdna_compute::GpuTensor>,
    pub wkv:    Option<rdna_compute::GpuTensor>,
    pub wo_a:   Option<rdna_compute::GpuTensor>,
    pub wo_b:   Option<rdna_compute::GpuTensor>,

    // Indexer (compressor) — present only when compress_ratio > 0.
    pub compressor_wkv:   Option<rdna_compute::GpuTensor>,
    pub compressor_wgate: Option<rdna_compute::GpuTensor>,
    pub compressor_norm:  Option<rdna_compute::GpuTensor>,

    // Hyper-Connections (F16 small matrices).
    pub hc_attn_base:  Option<rdna_compute::GpuTensor>,
    pub hc_attn_fn:    Option<rdna_compute::GpuTensor>,
    pub hc_attn_scale: Option<rdna_compute::GpuTensor>,
    pub hc_ffn_base:   Option<rdna_compute::GpuTensor>,
    pub hc_ffn_fn:     Option<rdna_compute::GpuTensor>,
    pub hc_ffn_scale:  Option<rdna_compute::GpuTensor>,

    // FFN router. `gate.bias` is None for hash-routed layers (first
    // `num_hash_layers`).
    pub gate_weight: Option<rdna_compute::GpuTensor>,
    pub gate_bias:   Option<rdna_compute::GpuTensor>,

    // Shared expert (one per layer, w1/w2/w3, MQ-family quantized).
    pub shared_w1: Option<rdna_compute::GpuTensor>,
    pub shared_w2: Option<rdna_compute::GpuTensor>,
    pub shared_w3: Option<rdna_compute::GpuTensor>,

    // Routed experts. To avoid 256 × 43 × 3 = 33K separate hipMalloc
    // calls (drives load time to 3+ minutes), all 256 experts for each
    // (layer, projection) are uploaded as ONE contiguous blob. The
    // indexed MoE GEMV kernels consume a device-side pointer table.
    //
    // Layout per blob: `[n_routed_experts × bytes_per_expert]` raw bytes.
    // Pointer table: F32 GpuTensor of length `2 * n_routed_experts`
    //   (two F32 slots per u64 pointer, matching qwen35 convention).
    pub expert_w1_blob: Option<rdna_compute::GpuTensor>,
    pub expert_w2_blob: Option<rdna_compute::GpuTensor>,
    pub expert_w3_blob: Option<rdna_compute::GpuTensor>,
    pub expert_w1_ptrs: Option<rdna_compute::GpuTensor>,
    pub expert_w2_ptrs: Option<rdna_compute::GpuTensor>,
    pub expert_w3_ptrs: Option<rdna_compute::GpuTensor>,
    /// Bytes per expert (uniform across all experts in a layer). Used
    /// for sub_offset math when forward needs a per-expert view (rarely).
    pub expert_w1_stride: usize,
    pub expert_w2_stride: usize,
    pub expert_w3_stride: usize,
}

impl DeepseekV4LayerWeights {
    pub fn new_empty(compress_ratio: u32) -> Self {
        DeepseekV4LayerWeights {
            compress_ratio,
            attn_norm: None, ffn_norm: None, q_norm: None, kv_norm: None,
            attn_sink: None,
            wq_a: None, wq_b: None, wkv: None, wo_a: None, wo_b: None,
            compressor_wkv: None, compressor_wgate: None, compressor_norm: None,
            hc_attn_base: None, hc_attn_fn: None, hc_attn_scale: None,
            hc_ffn_base: None, hc_ffn_fn: None, hc_ffn_scale: None,
            gate_weight: None, gate_bias: None,
            shared_w1: None, shared_w2: None, shared_w3: None,
            expert_w1_blob: None, expert_w2_blob: None, expert_w3_blob: None,
            expert_w1_ptrs: None, expert_w2_ptrs: None, expert_w3_ptrs: None,
            expert_w1_stride: 0, expert_w2_stride: 0, expert_w3_stride: 0,
        }
    }
}

/// V4F weights — scaffold-stage placeholder with two GPU-resident
/// globals already uploaded (`token_embd`, `output_norm`). Per-layer
/// uploads (LoRAs, KV, HC, experts) land in forward bring-up; this
/// commit establishes the upload contract end-to-end with the cheapest
/// pair of tensors.
///
/// `mtp_layer` is `Some` after Phase 5 lands (when the
/// `mtp.` prefix-skip in `hipfire-quantize` is lifted and MTP
/// tensors are quantized alongside main layers).
pub struct DeepseekV4Weights {
    /// Token embedding table. Stored as raw Q8F16 bytes on GPU
    /// (matches the `embed.weight` quant_type from Phase 1 ingest).
    pub token_embd: Option<rdna_compute::GpuTensor>,
    /// Final output norm (RMSNorm scale, F32 — converted from F16 at load time).
    pub output_norm: Option<rdna_compute::GpuTensor>,
    /// LM head weight (MQ4G256, shape [vocab_size, hidden]).
    pub head: Option<rdna_compute::GpuTensor>,
    /// One bundle per `num_hidden_layers` (43 on V4F).
    pub layers: Vec<DeepseekV4LayerWeights>,
    /// MTP head — structurally identical to a main layer, plus an
    /// `input_proj` conditioning on the base model's hidden state.
    /// `None` at scaffold stage; populated when Phase 5 ships.
    pub mtp_layer: Option<DeepseekV4LayerWeights>,
    pub _scaffold: (),
}

/// Per-layer state for the compressed-KV indexer (Phase 2, Lever 3).
///
/// Active only on layers with `compress_ratios[l] > 0`. Each layer
/// holds:
/// - a sparse compressed-K cache at stride `compress_ratios[l]`
/// - scratch for the current-step top-k position indices
///
/// See `docs/plans/deepseek4-phase2-indexer.md` for the full kernel
/// design and forward sequence.
pub struct IndexerLayerState {
    /// `compress_ratios[layer]` — stride of the compressed cache.
    /// `0` means this layer doesn't use the indexer (full SWA only).
    pub compress_ratio: u32,
    /// `[n_idx_heads, idx_head_dim, n_compressed_capacity]`
    /// Stub: real impl is a GPU tensor.
    pub _k_idx_compressed: (),
    /// `[n_idx_heads, index_topk]` of i32 position indices. Filled by
    /// `indexer_top_k`; consumed by `kv_gather`.
    pub _top_k_indices: (),
}

/// Per-layer scratch for the main attention path's gathered K/V rows.
///
/// The main attention attends to `sliding_window + index_topk` total
/// positions per step: a bounded ring of the last 128 raw KV rows
/// (SWA window) plus 512 rows gathered from the indexer's top-k.
pub struct MainAttentionLayerState {
    /// SWA ring K cache `[n_kv_heads, head_dim, sliding_window]` F32.
    /// `None` until `decode_step` allocates on first call.
    pub swa_k: Option<rdna_compute::GpuTensor>,
    /// SWA ring V cache. V4F has tied K=V so this is a copy of swa_k.
    pub swa_v: Option<rdna_compute::GpuTensor>,
    /// K rows gathered from the indexer's top-k indices (Phase 2). Stub.
    pub _k_gathered: (),
    pub _v_gathered: (),
}

/// V4F per-decode state. Held on the daemon's per-session struct,
/// reused across decode steps. Allocated once via `new_state`.
pub struct DeepseekV4State {
    /// Per-layer (43 + 1 MTP = 44). Layers with `compress_ratio == 0`
    /// skip the indexer.
    pub _indexer: Vec<IndexerLayerState>,
    pub _attention: Vec<MainAttentionLayerState>,

    /// Hyper-Connections residual streams `[hc_mult = 4, hidden = 4096]`.
    /// Stored as F32 to match hipfire's standard residual convention
    /// (llama / qwen35 use f32 residuals + f32 RMSNorm). Quantized
    /// kernels handle the f32 input directly.
    /// `None` until `decode_step` allocates on first call.
    pub residual_streams: Option<rdna_compute::GpuTensor>,

    /// Single-row embedding scratch `[hidden]` for the current decode
    /// step's token lookup. F32 to match residual_streams convention.
    pub embed_scratch: Option<rdna_compute::GpuTensor>,

    /// Per-step scratch `[hidden]` F32 — used for RMSNorm output,
    /// FWHT-rotated input to first GEMV, etc. Reused across layers.
    pub tmp: Option<rdna_compute::GpuTensor>,

    /// Q-LoRA bottleneck `[q_lora_rank = 1024]` F32. Output of
    /// `wq_a @ x`, input to `wq_b`. Reused across layers.
    pub q_lat: Option<rdna_compute::GpuTensor>,

    /// Q-LoRA bottleneck rotated `[q_lora_rank]` F32. FWHT-rotated
    /// view of q_lat, input to the MQ4 GEMV against wq_b.
    pub q_lat_rot: Option<rdna_compute::GpuTensor>,

    /// Full Q `[n_heads * head_dim = 64 * 512 = 32768]` F32. Output
    /// of `wq_b @ q_lat_rot`. Tail-only RoPE applied in place.
    pub q: Option<rdna_compute::GpuTensor>,

    /// Joint KV stream `[n_kv_heads * head_dim = 1 * 512 = 512]` F32.
    /// Output of `wkv @ x`. V4F uses tied K=V via this single vector
    /// (MQA with V tied to K — see project memory for the layout
    /// open question; revisit during numerical-correctness gate).
    /// Tail-only RoPE applied to last `qk_rope_head_dim = 64` dims.
    pub kv: Option<rdna_compute::GpuTensor>,

    /// Position counter for RoPE. Stored as a 1-element F32 GpuTensor
    /// where we write the i32 position bits via memcpy_htod (the
    /// rope_tail kernel reinterprets the bytes as int via cast).
    pub pos_buf: Option<rdna_compute::GpuTensor>,

    /// Per-token attention output `[hidden]` F32, fed to HC attn mix
    /// as the `transform_out` arg. Currently a stub: holds a sliced
    /// view of `q` until real attention + O-LoRA lands.
    pub attn_out: Option<rdna_compute::GpuTensor>,

    /// Per-token FFN output `[hidden]` F32, fed to HC FFN mix as
    /// `transform_out`. Currently = shared expert output (real),
    /// routed experts pending.
    pub ffn_out: Option<rdna_compute::GpuTensor>,

    /// FFN normalised input `[hidden]` F32. RMSNorm(stream0, ffn_norm)
    /// then FWHT-rotated for the shared-expert MQ4 GEMVs.
    pub ffn_x_rot: Option<rdna_compute::GpuTensor>,

    /// Shared expert SwiGLU gate scratch `[moe_intermediate=2048]` F32.
    pub ffn_gate: Option<rdna_compute::GpuTensor>,
    /// Shared expert SwiGLU up scratch `[moe_intermediate]` F32.
    pub ffn_up:   Option<rdna_compute::GpuTensor>,
    /// FWHT-rotated silu(gate)*up for the down GEMV.
    pub ffn_silu_rot: Option<rdna_compute::GpuTensor>,

    /// Final pre-lm_head normalized residual `[hidden]` F32. Output
    /// of the global RMSNorm against `output_norm`.
    pub final_norm: Option<rdna_compute::GpuTensor>,

    /// LM head output logits `[vocab_size = 129280]` F32. Output of
    /// `head_weight @ final_norm`.
    pub logits: Option<rdna_compute::GpuTensor>,

    /// FWHT-rotated `final_norm` for the MQ4 head GEMV. Shape `[hidden]`.
    pub final_norm_rot: Option<rdna_compute::GpuTensor>,

    /// Input-mapping output: `x_in = A · X`. Fed to the transform (attn
    /// or FFN) as its [hidden] input.
    pub hc_x_in: Option<rdna_compute::GpuTensor>,

    /// mHC control vector `[24]` F32, set by `hc_compute_control` and
    /// consumed by `hc_mix_4stream`. Allocated once per session.
    /// Layout: c[0..4]=Ã, c[4..20]=B̃, c[20..24]=C̃.
    pub hc_c: Option<rdna_compute::GpuTensor>,

    /// MoE router scores `[n_routed_experts = 256]` F32, set by the
    /// router step (gate.weight @ ffn_input + bias → sqrt_softplus).
    pub router_scores: Option<rdna_compute::GpuTensor>,
    /// Top-K expert indices, allocated as F32 view but interpreted
    /// as i32. Shape `[num_experts_per_tok = 6]`.
    pub topk_indices: Option<rdna_compute::GpuTensor>,

    /// Monotonic position counter — how many tokens this session has
    /// processed. Used to compute the SWA cache slot (`pos % window`)
    /// and number of valid cached positions.
    pub n_tokens: u64,

    pub _scaffold: (),
}

impl DeepseekV4State {
    pub fn new(cfg: &DeepseekV4Config) -> Result<Self, String> {
        let n_layers_total = cfg.num_hidden_layers + cfg.num_nextn_predict_layers;
        let mut indexer = Vec::with_capacity(n_layers_total);
        let mut attention = Vec::with_capacity(n_layers_total);
        for layer in 0..n_layers_total {
            let ratio = *cfg.compress_ratios.get(layer).unwrap_or(&0);
            indexer.push(IndexerLayerState {
                compress_ratio: ratio,
                _k_idx_compressed: (),
                _top_k_indices: (),
            });
            attention.push(MainAttentionLayerState {
                swa_k: None, swa_v: None,
                _k_gathered: (), _v_gathered: (),
            });
        }
        Ok(DeepseekV4State {
            _indexer: indexer,
            _attention: attention,
            residual_streams: None,  // allocated on first `decode_step` (needs Gpu).
            embed_scratch: None,
            tmp: None,
            q_lat: None,
            q_lat_rot: None,
            q: None,
            kv: None,
            pos_buf: None,
            attn_out: None,
            ffn_out: None,
            ffn_x_rot: None,
            ffn_gate: None,
            ffn_up: None,
            ffn_silu_rot: None,
            final_norm: None,
            logits: None,
            final_norm_rot: None,
            hc_x_in: None,
            hc_c: None,
            router_scores: None,
            topk_indices: None,
            n_tokens: 0,
            _scaffold: (),
        })
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

    /// Verify the parser handles the actual released V4F config.json
    /// (snapshot 6976c7ff). Catches schema drift if the upstream
    /// model card adds or renames fields.
    #[test]
    fn parses_real_v4f_config_json() {
        let real_config_path =
            "/home/nick/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/\
             snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136/config.json";
        let raw_json = match std::fs::read_to_string(real_config_path) {
            Ok(s) => s,
            Err(_) => {
                eprintln!("skipping real-config test — V4F not locally available");
                return;
            }
        };
        // Real config has extra fields beyond what RawDeepseekV4Config
        // reads (architectures, attention_bias, etc). serde silently
        // ignores them — verify we still parse the fields we care about.
        let raw: RawDeepseekV4Config = serde_json::from_str(&raw_json)
            .expect("real V4F config.json must parse — schema drift detected");

        // Cross-check against the documented V4F constants.
        assert_eq!(raw.num_hidden_layers, 43);
        assert_eq!(raw.head_dim, 512);
        assert_eq!(raw.qk_rope_head_dim, 64);
        assert_eq!(raw.q_lora_rank, 1024);
        assert_eq!(raw.o_lora_rank, 1024);
        assert_eq!(raw.n_routed_experts, 256);
        assert_eq!(raw.num_experts_per_tok, 6);
        assert_eq!(raw.hc_mult, 4);
        assert_eq!(raw.index_n_heads, 64);
        assert_eq!(raw.sliding_window, 128);

        // The released checkpoint's compress_ratios has length
        // num_hidden_layers + num_nextn_predict_layers = 44.
        assert_eq!(
            raw.compress_ratios.len(),
            raw.num_hidden_layers + raw.num_nextn_predict_layers
        );

        // Check the alternating pattern in the middle layers.
        // V4F shipped pattern: [0, 0, 4, 128, 4, 128, ..., 4, 0].
        for (i, &r) in raw.compress_ratios.iter().enumerate() {
            if i < 2 || i == raw.compress_ratios.len() - 1 {
                assert_eq!(r, 0, "layer {i}: expected ratio=0, got {r}");
            } else {
                let expected = if i % 2 == 0 { 4 } else { 128 };
                assert_eq!(r, expected, "layer {i}: expected ratio={expected}, got {r}");
            }
        }

        // Verify DeepseekV4State::new accepts the real config.
        // (Reconstruct the full Config from raw to drive State::new.)
        // Skip — would require a fake HfqFile. The shape test above
        // is enough for the schema-drift gate.
    }
}
