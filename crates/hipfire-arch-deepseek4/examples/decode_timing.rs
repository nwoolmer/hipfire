//! Quick single-token decode timing for V4F (no cache fill).
//! Goes through `decode_step_with_graph` which auto-selects graph
//! capture/replay vs direct dispatch per-arch (default on gfx11/gfx12).
//! Override with `HIPFIRE_V4F_GRAPH=0` to force direct.

use hipfire_arch_deepseek4::{forward::decode_step_with_graph, DeepseekV4, DeepseekV4State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::time::Instant;

fn main() -> Result<(), String> {
    let path = std::env::var("HIPFIRE_V4F_MODEL")
        .unwrap_or_else(|_| "/data/hipfire-models/v4f.mq2lloyd-q8.hfq".to_string());
    let graph_env = std::env::var("HIPFIRE_V4F_GRAPH").ok();
    eprintln!("Loading V4F from {path}... HIPFIRE_V4F_GRAPH={:?}", graph_env);
    let mut hfq = HfqFile::open(std::path::Path::new(&path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = DeepseekV4State::new(&cfg)?;
    eprintln!("V4F loaded. arch={}", gpu.arch);

    let step = |gpu: &mut Gpu, state: &mut DeepseekV4State, tok: u32, pos: u32| -> Result<Vec<f32>, String> {
        decode_step_with_graph(&cfg, &weights, state, gpu, tok, pos)
    };

    let warm_count = 8usize;
    for i in 0..warm_count {
        let _ = step(&mut gpu, &mut state, 100u32, i as u32)?;
    }
    eprintln!("warm to pos={warm_count}");

    let n = 20usize;
    let mut times = Vec::<u128>::with_capacity(n);
    for i in 0..n {
        let t = Instant::now();
        let _ = step(&mut gpu, &mut state, 100u32, (warm_count + i) as u32)?;
        let us = t.elapsed().as_micros();
        times.push(us);
        eprintln!("  step {}: {} us = {:.2} tok/s", i, us, 1_000_000.0 / (us as f64));
    }
    times.sort();
    let med = times[n / 2];
    eprintln!("\nMedian decode_step: {} us = {:.2} tok/s", med, 1_000_000.0 / (med as f64));
    Ok(())
}
