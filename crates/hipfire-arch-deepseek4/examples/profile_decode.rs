//! Profile V4F decode to identify the hottest kernels.
//!
//! Uses rdna_compute::profile to collect per-launch hipEvent timings,
//! aggregates by kernel name, prints top-20 by total ms.

use hipfire_arch_deepseek4::{forward::decode_step, DeepseekV4, DeepseekV4State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{profile, Gpu};
use std::collections::HashMap;

fn main() -> Result<(), String> {
    let path = std::env::var("HIPFIRE_V4F_MODEL")
        .unwrap_or_else(|_| "/data/hipfire-models/v4f.mq2lloyd-q8.hfq".to_string());
    eprintln!("Loading V4F from {path}...");
    let mut hfq = HfqFile::open(std::path::Path::new(&path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = DeepseekV4State::new(&cfg)?;
    eprintln!("V4F loaded.");

    // Warmup so kernel JIT / scratch allocs happen outside profile window.
    for i in 0..8 {
        let _ = decode_step(&cfg, &weights, &mut state, &mut gpu, 100u32, i as u32)?;
    }
    eprintln!("warmed.");

    profile::start();
    let n = 5usize;
    for i in 0..n {
        let _ = decode_step(&cfg, &weights, &mut state, &mut gpu, 100u32, (8 + i) as u32)?;
    }
    let entries = profile::stop().unwrap_or_default();
    eprintln!("collected {} profile entries over {} decodes", entries.len(), n);

    let mut by_kernel: HashMap<&str, (f64, u64)> = HashMap::new();
    let mut total_ms = 0.0;
    for e in &entries {
        let slot = by_kernel.entry(e.kernel).or_insert((0.0, 0));
        slot.0 += (e.time_us / 1000.0) as f64;
        slot.1 += 1;
        total_ms += (e.time_us / 1000.0) as f64;
    }
    eprintln!("total kernel time: {:.2} ms ({:.2} ms/decode avg)\n",
        total_ms, total_ms / n as f64);

    let mut sorted: Vec<(&&str, &(f64, u64))> = by_kernel.iter().collect();
    sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());

    println!("{:>40} {:>10} {:>10} {:>10} {:>10}",
        "kernel", "calls", "total ms", "ms/call", "% total");
    for (kernel, (ms, count)) in sorted.iter().take(25) {
        let pct = 100.0 * ms / total_ms;
        println!("{:>40} {:>10} {:>10.3} {:>10.4} {:>9.1}%",
            kernel, count, ms, ms / (*count as f64), pct);
    }
    Ok(())
}
