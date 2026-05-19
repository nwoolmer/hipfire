//! Minimal driver for rocprofv3 — just runs forward_prefill_batch_chunked
//! once on a short prompt. Use:
//!   HIPFIRE_V4F_UPLOAD_EXPERTS=1 HIPFIRE_V4F_MOE=1 \
//!     rocprofv3 --kernel-trace -f csv -o /tmp/v4f_bat -- \
//!       ./target/release/examples/profile_v4f_batched_prefill 32 32
//! Output: /tmp/v4f_bat.kernel_trace.csv

use hipfire_arch_deepseek4::{
    forward::{forward_prefill_batch_chunked, decode_step, PrefillBatchScratch},
    DeepseekV4, DeepseekV4State,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let prompt_len: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(32);
    let max_batch: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let mode = args.get(3).cloned().unwrap_or_else(|| "batched".to_string());

    let path = std::env::var("HIPFIRE_V4F_MODEL")
        .unwrap_or_else(|_| "/data/hipfire-models/v4f.mq2lloyd-f16compress.hfq".to_string());
    eprintln!("model={path}  mode={mode}  prompt={prompt_len} max_batch={max_batch}");

    let mut hfq = HfqFile::open(std::path::Path::new(&path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let pbs = PrefillBatchScratch::new(&mut gpu, &cfg, max_batch)?;
    let mut state = DeepseekV4State::new(&cfg)?;
    let tokens: Vec<u32> = (0..prompt_len).map(|i| 1 + (i as u32 % 100)).collect();

    // Warmup (lazy alloc per-layer state).
    let _ = decode_step(&cfg, &weights, &mut state, &mut gpu, tokens[0], 0)?;
    gpu.hip.device_synchronize().map_err(|e| format!("sync warmup: {e:?}"))?;
    let mut state = DeepseekV4State::new(&cfg)?;

    eprintln!("=== PROFILED REGION START ===");
    if mode == "batched" {
        // Bypass forward_prefill_batch_chunked's silent error swallow —
        // call the chunk directly and surface any failure.
        use hipfire_arch_deepseek4::forward::forward_prefill_batch_chunk;
        let take = tokens.len().min(pbs.max_batch);
        match forward_prefill_batch_chunk(
            &cfg, &weights, &mut state, &mut gpu, &pbs, &tokens[..take], 0,
        ) {
            Ok(()) => eprintln!("forward_prefill_batch_chunk OK ({take} tokens)"),
            Err(e) => eprintln!("forward_prefill_batch_chunk ERROR: {e}"),
        }
    } else {
        for (i, &tok) in tokens.iter().enumerate() {
            let _ = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, i as u32)?;
        }
    }
    gpu.hip.device_synchronize().map_err(|e| format!("sync: {e:?}"))?;
    eprintln!("=== PROFILED REGION END ===");
    Ok(())
}
