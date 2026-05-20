//! Run V4F decode with and without HIP graphs, compare argmax/top-1 per step.
//! Catches numerical drift between captured-replay and direct-dispatch paths.

use hipfire_arch_deepseek4::{
    forward::{decode_step, decode_step_with_graph},
    DeepseekV4, DeepseekV4State,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

fn run(use_graph: bool, n_steps: usize, model_path: &str) -> Result<Vec<usize>, String> {
    let mut hfq = HfqFile::open(std::path::Path::new(model_path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = DeepseekV4State::new(&cfg)?;

    let mut out = Vec::with_capacity(n_steps);
    let mut tok: u32 = 100;
    for i in 0..n_steps {
        let logits = if use_graph {
            decode_step_with_graph(&cfg, &weights, &mut state, &mut gpu, tok, i as u32)?
        } else {
            decode_step(&cfg, &weights, &mut state, &mut gpu, tok, i as u32)?
        };
        let am = argmax(&logits);
        out.push(am);
        tok = am as u32;
    }
    Ok(out)
}

fn main() -> Result<(), String> {
    let path = std::env::var("HIPFIRE_V4F_MODEL")
        .unwrap_or_else(|_| "/data/hipfire-models/v4f.mq2lloyd-q8.hfq".to_string());
    let n: usize = std::env::var("N_STEPS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    eprintln!("Comparing direct vs graph over {n} steps on {path}...");

    eprintln!("=== Direct path ===");
    let direct = run(false, n, &path)?;
    eprintln!("Done: {direct:?}");

    eprintln!("\n=== Graph path ===");
    let graphed = run(true, n, &path)?;
    eprintln!("Done: {graphed:?}");

    eprintln!("\n=== Comparison ===");
    let mut first_diff = None;
    let mut matches = 0;
    for (i, (a, b)) in direct.iter().zip(graphed.iter()).enumerate() {
        if a == b { matches += 1; }
        else if first_diff.is_none() { first_diff = Some(i); }
    }
    eprintln!("Match: {matches}/{n} = {:.1}%", 100.0 * matches as f64 / n as f64);
    if let Some(i) = first_diff {
        eprintln!("First divergence at step {i}: direct={} graph={}", direct[i], graphed[i]);
    } else {
        eprintln!("Byte-equivalent argmax across all {n} steps.");
    }
    Ok(())
}
