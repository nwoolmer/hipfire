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

use crate::deepseek4::{DeepseekV4Config, DeepseekV4State, DeepseekV4Weights};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Type marker for DeepSeek V4 Flash. `arch_id = 7` (next free slot
/// after `6 = Qwen3.5/3.6 MoE`). The marker is zero-sized; trait
/// dispatch uses the type, not a value.
pub struct DeepseekV4;

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
        _hfq: &mut HfqFile,
        _cfg: &Self::Config,
        _gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        Err("deepseek4: load_weights not yet implemented (scaffold-stage crate)".to_string())
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
