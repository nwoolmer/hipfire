//! Time each stage of V4F forward to understand the per-layer cost.

use hipfire_arch_deepseek4::{forward::decode_step_with_graph, DeepseekV4, DeepseekV4State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::time::Instant;

fn main() -> Result<(), String> {
    let path = "/home/nick/.hipfire/models/v4f.mq2lloyd-gptq-all";
    let mut hfq = HfqFile::open(std::path::Path::new(path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;

    let t_init = Instant::now();
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    eprintln!("GPU init: {:?}", t_init.elapsed());

    let t_load = Instant::now();
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    eprintln!("load_weights (non-expert): {:?}", t_load.elapsed());

    let t_state = Instant::now();
    let mut state = DeepseekV4State::new(&cfg)?;
    eprintln!("state init: {:?}", t_state.elapsed());

    // Warmup
    let _ = decode_step_with_graph(&cfg, &weights, &mut state, &mut gpu, 100, 0)?;

    // Time 10 decodes
    let n = 10;
    let t_dec = Instant::now();
    for i in 0..n {
        let _logits = decode_step_with_graph(&cfg, &weights, &mut state, &mut gpu, i as u32, i as u32)?;
    }
    let total = t_dec.elapsed();
    eprintln!("{n} decode_steps in {total:?} → {:.1} tok/s avg, {:?} per step",
        n as f64 / total.as_secs_f64(),
        total / n as u32);

    Ok(())
}
