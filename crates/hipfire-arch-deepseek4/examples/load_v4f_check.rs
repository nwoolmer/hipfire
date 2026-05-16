//! Sanity-load the V4F HFQ file produced by Phase 1 ingest and verify
//! the runtime's HfqFile parser + DeepseekV4Config::from_hfq accept it.
//!
//! Usage:
//!   cargo run --release --example load_v4f_check -p hipfire-arch-deepseek4
//!
//! Reads the file lazily (no GPU); just confirms metadata round-trip
//! and that the parsed Config matches expected V4F shape constants.

use hipfire_arch_deepseek4::DeepseekV4;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;

fn main() {
    let path = "/home/nick/.hipfire/models/v4f.mq2lloyd-gptq-all";
    eprintln!("opening {}", path);
    let hfq = HfqFile::open(std::path::Path::new(path)).expect("HfqFile::open");
    eprintln!("  arch_id           = {}", hfq.arch_id);
    eprintln!("  metadata bytes    = {}", hfq.metadata_json.len());

    assert_eq!(hfq.arch_id, DeepseekV4::arch_id(),
        "expected arch_id {}, got {}", DeepseekV4::arch_id(), hfq.arch_id);

    // Parse the V4F config out of the metadata blob.
    // The metadata is wrapped as {"architecture": ..., "config": {...},
    // "tokenizer": ..., "tokenizer_config": ...}. The config sub-object
    // is the actual V4F config.json. Our `from_hfq` reads metadata_json
    // directly, which assumes the JSON has the V4F shape at the top
    // level. The current Phase 1 metadata wraps it as `config`, so we
    // need to either:
    //   (a) Update `DeepseekV4Config::from_hfq` to unwrap `config`, OR
    //   (b) Update the quantizer to flatten the config at the top level
    // Pick (a): unwrapping is cleaner — keeps the wrapper for
    // tokenizer + architecture tag while letting Config::from_hfq
    // navigate to its own slice. Note this for follow-up.
    //
    // For this sanity-load test, just verify the wrapper JSON is valid.
    let wrapper: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
        .expect("metadata is valid JSON");
    let arch_str = wrapper["architecture"].as_str().expect("architecture field");
    assert_eq!(arch_str, "deepseek_v4");
    let config = &wrapper["config"];
    eprintln!("  config.num_hidden_layers = {}", config["num_hidden_layers"]);
    eprintln!("  config.head_dim          = {}", config["head_dim"]);
    eprintln!("  config.hc_mult           = {}", config["hc_mult"]);
    eprintln!("  config.index_topk        = {}", config["index_topk"]);

    // Real Config parse through the Architecture trait.
    let cfg = DeepseekV4::config_from_hfq(&hfq)
        .expect("DeepseekV4::config_from_hfq");
    eprintln!("\nDeepseekV4Config parsed:");
    eprintln!("  num_hidden_layers       = {}", cfg.num_hidden_layers);
    eprintln!("  num_nextn_predict_layers= {}", cfg.num_nextn_predict_layers);
    eprintln!("  hidden_size             = {}", cfg.hidden_size);
    eprintln!("  n_routed_experts        = {}", cfg.n_routed_experts);
    eprintln!("  num_experts_per_tok     = {}", cfg.num_experts_per_tok);
    eprintln!("  hc_mult                 = {}", cfg.hc_mult);
    eprintln!("  index_topk              = {}", cfg.index_topk);
    eprintln!("  sliding_window          = {}", cfg.sliding_window);
    eprintln!("  compress_ratios.len()   = {}", cfg.compress_ratios.len());

    // And `new_state` should succeed on the parsed config (no GPU).
    // Skip — `new_state` takes &mut Gpu which we don't have here.
    // Validates only that Config parses; state allocation deferred.

    // Spot-check a few known V4F tensor names through the public API.
    for name in &[
        "embed.weight",
        "head.weight",
        "norm.weight",
        "layers.0.attn.q_norm.weight",
        "layers.0.attn.wq_a.weight",
        "layers.0.ffn.experts.0.w1.weight",
        "layers.0.ffn.experts.0.w2.weight",
        "layers.0.ffn.experts.0.w3.weight",
        "layers.0.hc_attn_fn",
        "layers.42.ffn.experts.255.w1.weight",
    ] {
        match hfq.find_tensor_info(name) {
            Some(info) => eprintln!(
                "  ok {:<40} qt={} shape={:?}",
                name, info.quant_type, info.shape
            ),
            None => eprintln!("  MISSING {}", name),
        }
    }
}
