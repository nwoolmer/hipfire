//! `Architecture` trait impl for DeepSeek V4 Flash (`arch_id = 7`).
//!
//! V4F diverges from the Qwen3.5 / LLaMA paths in several places —
//! Hyper-Connections, compressed-KV indexer, tail-only RoPE,
//! Q/O-LoRA, raw SWA cache, FP4 experts — but the bring-up triple
//! (`config_from_hfq` / `load_weights` / `new_state`) follows the
//! same Architecture-trait shape as the other arch crates.
//!
//! At scaffold stage (this commit) `load_weights` and forward are
//! stubbed; only `config_from_hfq` and `new_state` are wired through
//! so the workspace builds and the metadata parser is exercised by
//! the tests.

use crate::deepseek4::{
    DeepseekV4Config, DeepseekV4LayerWeights, DeepseekV4State, DeepseekV4Weights,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Type marker for DeepSeek V4 Flash. `arch_id = 7` (next free slot
/// after `6 = Qwen3.5/3.6 MoE`). The marker is zero-sized; trait
/// dispatch uses the type, not a value.
pub struct DeepseekV4;

impl DeepseekV4 {
    /// Phase 1.5 walk: verify every expected V4F tensor is present in
    /// the HFQ index. No GPU upload. Returns a populated `Weights` with
    /// `_scaffold: ()` per layer; the real `WeightTensor` handles get
    /// filled in as Phases 2-5 wire the kernels.
    ///
    /// Catches missing-tensor / naming-mismatch problems before forward
    /// triggers them. Per-layer tensor inventory derived from the V4F
    /// safetensors index (see Phase 1 commit 8ccfa42).
    pub fn load_weights_host_only_walk(
        hfq: &HfqFile,
        cfg: &DeepseekV4Config,
    ) -> Result<DeepseekV4Weights, String> {
        let n_layers = cfg.num_hidden_layers;
        let mut layers: Vec<DeepseekV4LayerWeights> = Vec::with_capacity(n_layers);

        // Global tensors.
        for name in &[
            "embed.weight",
            "head.weight",
            "norm.weight",
            "hc_head_base",
            "hc_head_fn",
            "hc_head_scale",
        ] {
            if hfq.find_tensor_info(name).is_none() {
                return Err(format!("deepseek4: missing global tensor '{name}'"));
            }
        }

        // Per-layer tensors.
        for l in 0..n_layers {
            // Attention LoRA + KV joint + norms.
            for suffix in &[
                "attn.wq_a.weight",
                "attn.wq_b.weight",
                "attn.wkv.weight",
                "attn.wo_a.weight",
                "attn.wo_b.weight",
                "attn.q_norm.weight",
                "attn.kv_norm.weight",
                "attn_norm.weight",
                "ffn_norm.weight",
                "attn.attn_sink",
            ] {
                let name = format!("layers.{l}.{suffix}");
                if hfq.find_tensor_info(&name).is_none() {
                    return Err(format!("deepseek4: layer {l} missing '{suffix}'"));
                }
            }

            // Indexer (compressor) tensors — present only when
            // compress_ratio[l] > 0. V4F config records the ratio array;
            // layers 0, 1, and 43 (MTP) have ratio = 0.
            let ratio = *cfg.compress_ratios.get(l).unwrap_or(&0);
            if ratio > 0 {
                for suffix in &[
                    "attn.compressor.wkv.weight",
                    "attn.compressor.wgate.weight",
                    "attn.compressor.norm.weight",
                ] {
                    let name = format!("layers.{l}.{suffix}");
                    if hfq.find_tensor_info(&name).is_none() {
                        return Err(format!(
                            "deepseek4: layer {l} (ratio={ratio}) missing '{suffix}'"
                        ));
                    }
                }
            }

            // Hyper-Connections per-layer.
            for suffix in &[
                "hc_attn_base",
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_ffn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
            ] {
                let name = format!("layers.{l}.{suffix}");
                if hfq.find_tensor_info(&name).is_none() {
                    return Err(format!("deepseek4: layer {l} missing HC tensor '{suffix}'"));
                }
            }

            // FFN router. The first `num_hash_layers` layers are HASH-
            // ROUTED — they have `gate.weight` but NO `gate.bias`. The
            // hash-routing table (`tid2eid`) is an I64 tensor that we
            // skip at ingest time (see commit 8ccfa42's skip-I64 path)
            // and restore as raw bytes in forward bring-up. Layers
            // beyond `num_hash_layers` use the standard `noaux_tc`
            // scoring path with `gate.weight` + `gate.bias`.
            //
            // On V4F: num_hash_layers=3 → layers 0, 1, 2 are hash;
            // layers 3..43 are score-routed.
            let is_hash_routed = l < cfg.num_hash_layers;
            let name = format!("layers.{l}.ffn.gate.weight");
            if hfq.find_tensor_info(&name).is_none() {
                return Err(format!("deepseek4: layer {l} missing 'ffn.gate.weight'"));
            }
            if !is_hash_routed {
                let name = format!("layers.{l}.ffn.gate.bias");
                if hfq.find_tensor_info(&name).is_none() {
                    return Err(format!(
                        "deepseek4: layer {l} (score-routed) missing 'ffn.gate.bias'"
                    ));
                }
            }
            // Shared expert.
            for suffix in &[
                "ffn.shared_experts.w1.weight",
                "ffn.shared_experts.w2.weight",
                "ffn.shared_experts.w3.weight",
            ] {
                let name = format!("layers.{l}.{suffix}");
                if hfq.find_tensor_info(&name).is_none() {
                    return Err(format!("deepseek4: layer {l} missing shared '{suffix}'"));
                }
            }
            // Routed experts: 256 × {w1, w2, w3}.
            for e in 0..cfg.n_routed_experts {
                for proj in &["w1", "w2", "w3"] {
                    let name = format!("layers.{l}.ffn.experts.{e}.{proj}.weight");
                    if hfq.find_tensor_info(&name).is_none() {
                        return Err(format!(
                            "deepseek4: layer {l} expert {e} missing '{proj}'"
                        ));
                    }
                }
            }

            layers.push(DeepseekV4LayerWeights {
                compress_ratio: ratio,
                _scaffold: (),
            });
        }

        Ok(DeepseekV4Weights {
            layers,
            mtp_layer: None,  // skipped by quantize per `mtp.` prefix; Phase 5 work.
            _scaffold: (),
        })
    }
}

impl Architecture for DeepseekV4 {
    type Weights = DeepseekV4Weights;
    type State = DeepseekV4State;
    type Config = DeepseekV4Config;

    fn arch_id() -> u32 {
        // 7 = DeepSeek V4 Flash. Reserve in docs/architecture-ids.md
        // when this crate's HFQ writer lands.
        7
    }

    fn name() -> &'static str {
        "deepseek4"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        DeepseekV4Config::from_hfq(hfq)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        _gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        // Phase 1.5 (host-only) walk: enumerate the V4F tensor names and
        // confirm every expected per-layer + global tensor is present in
        // the HFQ index. GPU upload, dtype conversion, and `WeightTensor`
        // construction land in forward bring-up.
        //
        // This catches missing-tensor / naming-mismatch problems before
        // the forward path triggers them. Returns a populated layer Vec
        // with all `_scaffold: ()` slots; the real WeightTensor handles
        // get filled in as Phases 2-5 wire the kernels.
        Self::load_weights_host_only_walk(hfq, cfg)
    }

    fn new_state(_gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        DeepseekV4State::new(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek4_arch_id_is_seven() {
        assert_eq!(DeepseekV4::arch_id(), 7);
        assert_eq!(DeepseekV4::name(), "deepseek4");
    }
}
