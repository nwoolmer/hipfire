//! Two-chunk profiler — distinguishes chunk 1 vs chunk 2 kernel cost.
//!
//! Background: bench_v4f_batched_prefill shows 1.82× per-token cost
//! decay between chunk 1 (positions 0..63) and chunk 2 (positions
//! 64..127) at B=64. The flash-attention research suggested attention
//! reuse, but the math says attention is only ~0.3% of wallclock at
//! V4F's MQA shape (n_kv=1 → L2 serves cross-head K reads). This
//! profiler isolates which kernels actually grow chunk-over-chunk.
//!
//! Method: warmup, then run two chunks with explicit device sync
//! between them. Rocprofv3 picks up per-kernel timings; aggregator
//! script splits the CSV at the gap between chunks.
//!
//! Use:
//!   HIPFIRE_V4F_UPLOAD_EXPERTS=1 HIPFIRE_V4F_MOE=1 \
//!     HIPFIRE_V4F_MEM_GUARD_GB=125 \
//!     rocprofv3 --kernel-trace -f csv -o /tmp/v4f_2chunk -- \
//!       ./target/release/examples/profile_v4f_two_chunk 64 64
//!
//! Output: /tmp/v4f_2chunk.kernel_trace.csv with all kernel timestamps;
//! the printed wallclock from the bench identifies the chunk boundary.

use hipfire_arch_deepseek4::{
    forward::{decode_step, forward_prefill_batch_chunk, PrefillBatchScratch},
    DeepseekV4, DeepseekV4State,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::time::Instant;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let chunk_size: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let max_batch: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);

    let path = std::env::var("HIPFIRE_V4F_MODEL")
        .unwrap_or_else(|_| "/data/hipfire-models/v4f.mq2lloyd-q8.hfq".to_string());
    eprintln!("model={path}  chunk_size={chunk_size} max_batch={max_batch}");

    let mut hfq = HfqFile::open(std::path::Path::new(&path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let pbs = PrefillBatchScratch::new(&mut gpu, &cfg, max_batch)?;
    let mut state = DeepseekV4State::new(&cfg)?;

    // chunk-size tokens for chunk 1, chunk-size tokens for chunk 2.
    let tokens: Vec<u32> = (0..(2 * chunk_size)).map(|i| 1 + (i as u32 % 100)).collect();

    // Warmup — lazy-alloc per-layer state via one decode step.
    let _ = decode_step(&cfg, &weights, &mut state, &mut gpu, tokens[0], 0)?;
    gpu.hip.device_synchronize().map_err(|e| format!("sync warmup: {e:?}"))?;
    let mut state = DeepseekV4State::new(&cfg)?;

    // ── Chunk 1: positions 0..chunk_size ──────────────────────────────
    eprintln!("=== CHUNK 1 START (start_pos=0) ===");
    let t0 = Instant::now();
    forward_prefill_batch_chunk(
        &cfg, &weights, &mut state, &mut gpu, &pbs,
        &tokens[..chunk_size], 0,
    )?;
    gpu.hip.device_synchronize().map_err(|e| format!("sync c1: {e:?}"))?;
    let c1_us = t0.elapsed().as_micros();
    eprintln!("=== CHUNK 1 END    ({c1_us} us, {:.1} tok/s) ===",
        1_000_000.0 * chunk_size as f64 / c1_us as f64);

    // ── Chunk 2: positions chunk_size..2*chunk_size ───────────────────
    eprintln!("=== CHUNK 2 START (start_pos={chunk_size}) ===");
    let t1 = Instant::now();
    forward_prefill_batch_chunk(
        &cfg, &weights, &mut state, &mut gpu, &pbs,
        &tokens[chunk_size..], chunk_size as u32,
    )?;
    gpu.hip.device_synchronize().map_err(|e| format!("sync c2: {e:?}"))?;
    let c2_us = t1.elapsed().as_micros();
    eprintln!("=== CHUNK 2 END    ({c2_us} us, {:.1} tok/s) ===",
        1_000_000.0 * chunk_size as f64 / c2_us as f64);

    let ratio = c2_us as f64 / c1_us as f64;
    eprintln!("\nper-token cost ratio chunk2/chunk1 = {ratio:.2}×");
    Ok(())
}
