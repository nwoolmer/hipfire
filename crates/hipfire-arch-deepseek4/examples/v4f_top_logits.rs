//! Dump top-K predicted tokens after a prompt — diagnostic for ppl issues.

use hipfire_arch_deepseek4::{forward::decode_step, DeepseekV4, DeepseekV4State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::path::Path;

fn main() -> Result<(), String> {
    let path = std::env::var("HIPFIRE_V4F_MODEL")
        .unwrap_or_else(|_| "/home/nick/.hipfire/models/v4f.mq2lloyd-fp4fix".to_string());
    let prompt = std::env::args().skip(1).next()
        .unwrap_or_else(|| "Robert Boulter is an English actor".to_string());
    let k_top: usize = std::env::var("TOPK").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(10);

    let mut hfq = HfqFile::open(Path::new(&path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .ok_or("no tokenizer")?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = DeepseekV4State::new(&cfg)?;

    let tokens = tokenizer.encode(&prompt);
    eprintln!("Prompt: {prompt:?}");
    eprintln!("Tokens: {tokens:?} ({} tokens)", tokens.len());

    // Feed prompt; capture logits at each position.
    for (i, &tok) in tokens.iter().enumerate() {
        let logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, i as u32)?;
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        // Compute top-K
        let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate()
            .map(|(j, &v)| (j, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top: Vec<String> = indexed.iter().take(k_top)
            .map(|(j, v)| {
                let s = tokenizer.decode(&[*j as u32]);
                format!("{:>6} {:?} ({:.3})", j, s, v)
            }).collect();
        eprintln!("\n[pos={i}] token={} ({:?})", tok,
            tokenizer.decode(&[tok]));
        eprintln!("  logits min={min:.3} max={max:.3}");
        eprintln!("  top-{}:", k_top);
        for t in &top { eprintln!("    {t}"); }
    }
    Ok(())
}
