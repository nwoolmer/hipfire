//! Run V4F decode_step in a loop simulating greedy generation.
//! Each step uses the previous step's argmax as the next input
//! (no KV cache currently, so context isn't actually carried — each
//! step is independent forward through 43 layers).

use hipfire_arch_deepseek4::{forward::decode_step, DeepseekV4, DeepseekV4State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

fn main() -> Result<(), String> {
    let path = "/home/nick/.hipfire/models/v4f.mq2lloyd-gptq-all";
    let mut hfq = HfqFile::open(std::path::Path::new(path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = DeepseekV4State::new(&cfg)?;

    let mut tok: u32 = 100;  // input '¤'
    let max_steps: u32 = 20;

    eprintln!("Starting from token {tok}, generating {max_steps} steps");
    let mut sequence: Vec<u32> = vec![tok];
    let t0 = std::time::Instant::now();
    for pos in 0..max_steps {
        let logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos)?;
        let argmax = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        sequence.push(argmax.0 as u32);
        tok = argmax.0 as u32;
    }
    let elapsed = t0.elapsed();

    eprintln!("\nGenerated sequence ({} tokens in {:?}, {:.1} tok/s):",
        sequence.len(), elapsed, sequence.len() as f64 / elapsed.as_secs_f64());
    eprintln!("  {:?}", sequence);
    Ok(())
}
