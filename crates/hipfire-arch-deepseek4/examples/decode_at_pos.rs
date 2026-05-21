//! Fill context to a target position, then time N decode steps at that
//! position. Used for kernel-level rocprof analysis of long-context
//! decode (where bench_decode_vs_ctx prefills the full ladder but only
//! times one decode per checkpoint).

use hipfire_arch_deepseek4::{forward::decode_step, DeepseekV4, DeepseekV4State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::time::Instant;

fn main() -> Result<(), String> {
    let target_pos: u32 = std::env::var("TARGET_POS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let n_timed: usize = std::env::var("N_TIMED")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(5);

    let path = std::env::var("HIPFIRE_V4F_MODEL")
        .unwrap_or_else(|_| "/data/hipfire-models/v4f.mq2lloyd-q8.hfq".to_string());
    eprintln!("Loading V4F from {path} (TARGET_POS={target_pos}, N_TIMED={n_timed})...");
    let mut hfq = HfqFile::open(std::path::Path::new(&path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = DeepseekV4State::new(&cfg)?;

    eprintln!("Filling to position {target_pos}...");
    for pos in 0..target_pos {
        let _ = decode_step(&cfg, &weights, &mut state, &mut gpu, 100, pos)?;
        if pos > 0 && pos % 256 == 0 {
            eprintln!("  filled to {pos}");
        }
    }

    eprintln!("Timing {n_timed} decode steps at position {target_pos}...");
    let mut times = Vec::new();
    for i in 0..n_timed {
        let pos = target_pos + i as u32;
        let t = Instant::now();
        let _ = decode_step(&cfg, &weights, &mut state, &mut gpu, 100, pos)?;
        let us = t.elapsed().as_micros();
        times.push(us);
        eprintln!("  step {i} @ pos {pos}: {us} us = {:.2} tok/s", 1_000_000.0 / us as f64);
    }
    times.sort();
    let median = times[n_timed / 2];
    eprintln!("\nMedian: {median} us = {:.2} tok/s", 1_000_000.0 / median as f64);
    Ok(())
}
