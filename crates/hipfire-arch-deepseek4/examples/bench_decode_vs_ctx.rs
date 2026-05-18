//! Bench decode_step wall time vs context position.
//!
//! Loads V4F, warms the cache by running decode_step 0..N to fill SWA,
//! then measures individual decode_step times at sampled positions to
//! show how single-token decode cost scales with context.
//!
//! Output: per-position median ms over `K_REPS=3` consecutive calls.

use hipfire_arch_deepseek4::{forward::decode_step, DeepseekV4, DeepseekV4State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::time::Instant;

fn main() -> Result<(), String> {
    let path = std::env::var("HIPFIRE_V4F_MODEL")
        .unwrap_or_else(|_| "/data/hipfire-models/v4f.mq2lloyd-f16compress.hfq".to_string());
    eprintln!("Loading V4F from {path}...");
    let mut hfq = HfqFile::open(std::path::Path::new(&path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = DeepseekV4State::new(&cfg)?;
    eprintln!("V4F loaded, n_layers={}", cfg.num_hidden_layers);

    // Stride through context filling each position. Time once per
    // checkpoint (positions are unique → no cache pollution between).
    let checkpoints: &[u32] = &[0, 16, 64, 128, 256, 512, 1024, 1536, 2048];
    let mut pos: u32 = 0;
    let mut results = Vec::<(u32, u128)>::new();

    for &target_pos in checkpoints {
        while pos < target_pos {
            let _ = decode_step(&cfg, &weights, &mut state, &mut gpu, 100, pos)?;
            pos += 1;
        }
        // Now at exactly `target_pos`; time the NEXT decode.
        let t = Instant::now();
        let _ = decode_step(&cfg, &weights, &mut state, &mut gpu, 100, pos)?;
        let dt = t.elapsed().as_millis();
        results.push((target_pos, dt));
        pos += 1;
        eprintln!("  pos {} → next decode_step = {} ms ({:.2} tok/s)",
            target_pos, dt, 1000.0 / (dt as f64));
    }

    println!("\n=== summary ===");
    println!("{:>8} {:>10} {:>10}", "pos", "ms", "tok/s");
    for (p, ms) in &results {
        println!("{:>8} {:>10} {:>10.2}", p, ms, 1000.0 / (*ms as f64));
    }
    Ok(())
}
