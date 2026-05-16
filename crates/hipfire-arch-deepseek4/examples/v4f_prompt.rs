//! Feed V4F a multi-token "prompt" and let it continue.
//!
//! Uses SWA attention (HIPFIRE_V4F_ATTN=swa) so the cache fills with
//! the prompt's K, V before generation starts. This is the V4F-
//! intended usage pattern (multi-token prompt → single-token decode).

use hipfire_arch_deepseek4::{forward::decode_step, DeepseekV4, DeepseekV4State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

fn main() -> Result<(), String> {
    // SWA attention to enable cache-based context integration.
    std::env::set_var("HIPFIRE_V4F_ATTN", "swa");

    let path = "/home/nick/.hipfire/models/v4f.mq2lloyd-gptq-all";
    let mut hfq = HfqFile::open(std::path::Path::new(path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = DeepseekV4State::new(&cfg)?;

    // Hand-encoded prompt: [BOS, "Hello", ",", " world", "!"]
    // From V4F tokenizer vocab inspection:
    //   BOS=0, "Hello"=19923, ","=14, " world"=2058, "!"=3
    let prompt: Vec<u32> = vec![0, 19923, 14, 2058, 3];
    let n_generate = 20u32;

    eprintln!("Prompt tokens: {prompt:?}");
    let mut sequence = prompt.clone();

    // Process prompt tokens (their logits discarded; only the LAST
    // step's argmax is used to start generation).
    let t0 = std::time::Instant::now();
    let mut last_logits = vec![];
    for (i, &tok) in prompt.iter().enumerate() {
        last_logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, i as u32)?;
    }

    // Generate from the last prompt token's logits.
    let mut tok = last_logits.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
    sequence.push(tok);
    for pos in prompt.len() as u32..(prompt.len() as u32 + n_generate) {
        let logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos)?;
        let argmax = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        sequence.push(argmax.0 as u32);
        tok = argmax.0 as u32;
    }
    let elapsed = t0.elapsed();

    eprintln!("Generated sequence ({} prompt + {} gen, total {:?}, {:.1} tok/s):",
        prompt.len(), n_generate, elapsed,
        sequence.len() as f64 / elapsed.as_secs_f64());
    eprintln!("  {:?}", sequence);
    Ok(())
}
