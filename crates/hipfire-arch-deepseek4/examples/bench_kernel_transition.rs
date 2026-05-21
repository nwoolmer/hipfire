//! Microbench: measure the actual GPU-side cost of running two kernels
//! sequentially vs each in isolation, using `hipEvent` timing (no rocprof,
//! no rocprof artifacts).
//!
//! The session's profile analysis attributed ~23% of V4F decode wallclock
//! to "intra-graph idle" between specific kernel transitions, with the
//! `hc_pre_post_sigmoid_scale_f32 → hc_sinkhorn_4x4` pair flagged as the
//! highest-leverage candidate for kernel fusion.
//!
//! This bench answers definitively: is there a real GPU-side gap that
//! fusion could recover, or is the gap a rocprof instrumentation artifact?
//!
//! Reads:
//!     elapsed_A   — N iterations of kernel A alone
//!     elapsed_B   — N iterations of kernel B alone
//!     elapsed_AB  — N iterations of (A then B) sequentially
//!
//! If `elapsed_AB - (elapsed_A + elapsed_B)` is positive and large per
//! iteration, the transition gap is real and fusion-recoverable.
//! If it's ≈ 0 (within noise), kernels are already pipelined and fusion
//! saves nothing.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const N_ITERS: usize = 10_000;
const WARMUP_ITERS: usize = 100;

fn time_loop<F>(gpu: &mut Gpu, label: &str, mut body: F) -> Result<f64, String>
where
    F: FnMut(&mut Gpu) -> Result<(), String>,
{
    // Warmup
    for _ in 0..WARMUP_ITERS {
        body(gpu)?;
    }
    gpu.hip.device_synchronize().map_err(|e| format!("sync warmup: {e:?}"))?;

    // hipEvents bracket the whole loop (so the elapsed includes any
    // serialization between iterations as it naturally would).
    let start = gpu.hip.event_create().map_err(|e| format!("event_create: {e:?}"))?;
    let stop  = gpu.hip.event_create().map_err(|e| format!("event_create: {e:?}"))?;

    let cpu_t0 = Instant::now();
    gpu.hip.event_record(&start, None).map_err(|e| format!("event_record start: {e:?}"))?;
    for _ in 0..N_ITERS {
        body(gpu)?;
    }
    gpu.hip.event_record(&stop, None).map_err(|e| format!("event_record stop: {e:?}"))?;
    gpu.hip.event_synchronize(&stop).map_err(|e| format!("event_synchronize: {e:?}"))?;
    let cpu_elapsed = cpu_t0.elapsed().as_secs_f64() * 1000.0; // ms
    let gpu_elapsed_ms = gpu.hip.event_elapsed_ms(&start, &stop)
        .map_err(|e| format!("event_elapsed: {e:?}"))? as f64;

    let per_iter_us = gpu_elapsed_ms * 1000.0 / N_ITERS as f64;
    eprintln!("  {label:>16}  gpu={gpu_elapsed_ms:>8.2}ms  cpu={cpu_elapsed:>8.2}ms  per-iter={per_iter_us:>7.3}us  ({N_ITERS} iters)");
    Ok(gpu_elapsed_ms)
}

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    eprintln!("GPU: {}", gpu.arch);

    // hc_c buffer: 24 floats holding [pre(4), post(4), comb(16)] interleaved
    // as `hc_pre_post_sigmoid_scale_f32` expects. Initialize with non-trivial
    // values so the kernels do real work (sigmoid + sinkhorn behave normally).
    let hc_c_init: Vec<f32> = (0..24).map(|i| 0.1_f32 + (i as f32) * 0.013).collect();
    let hc_c = gpu.upload_f32(&hc_c_init, &[24])
        .map_err(|e| format!("upload hc_c: {e:?}"))?;
    let comb_view = hc_c.sub_offset(8, 16);  // sinkhorn reads the 4x4 comb

    // V4F production values for hc_eps + sinkhorn iters + post_scale.
    let hc_eps = 1e-6_f32;
    let post_scale = 1.5_f32;
    let sinkhorn_iters: i32 = 2;

    eprintln!("\n=== hc_pre_post_sigmoid_scale + hc_sinkhorn_4x4 transition bench ===");
    eprintln!("N_ITERS={N_ITERS}, WARMUP={WARMUP_ITERS}, hc_eps={hc_eps}, sinkhorn_iters={sinkhorn_iters}\n");

    // Pull the closure-needed values once.
    let elapsed_a = time_loop(&mut gpu, "A (pre_post)", |gpu| {
        gpu.hc_pre_post_sigmoid_scale_f32(&hc_c, hc_eps, post_scale)
            .map_err(|e| format!("A: {e:?}"))
    })?;
    let elapsed_b = time_loop(&mut gpu, "B (sinkhorn)", |gpu| {
        gpu.hc_sinkhorn_4x4(&comb_view, hc_eps, sinkhorn_iters)
            .map_err(|e| format!("B: {e:?}"))
    })?;
    let elapsed_ab = time_loop(&mut gpu, "A → B (seq)", |gpu| {
        gpu.hc_pre_post_sigmoid_scale_f32(&hc_c, hc_eps, post_scale)
            .map_err(|e| format!("A in AB: {e:?}"))?;
        gpu.hc_sinkhorn_4x4(&comb_view, hc_eps, sinkhorn_iters)
            .map_err(|e| format!("B in AB: {e:?}"))
    })?;

    // Excess time when running A and B sequentially in the same iteration:
    // that's the transition cost (per iter) plus any extra dispatch
    // overhead. If 0, fusion saves nothing.
    let sum_isolated = elapsed_a + elapsed_b;
    let excess_ms = elapsed_ab - sum_isolated;
    let excess_per_iter_us = excess_ms * 1000.0 / N_ITERS as f64;
    let a_per_us = elapsed_a * 1000.0 / N_ITERS as f64;
    let b_per_us = elapsed_b * 1000.0 / N_ITERS as f64;
    let ab_per_us = elapsed_ab * 1000.0 / N_ITERS as f64;

    eprintln!("\n=== Result ===");
    eprintln!("  A alone:        {a_per_us:.3} us/iter");
    eprintln!("  B alone:        {b_per_us:.3} us/iter");
    eprintln!("  A→B together:   {ab_per_us:.3} us/iter");
    eprintln!("  A+B summed:     {:.3} us/iter", a_per_us + b_per_us);
    eprintln!("  excess (gap):   {excess_per_iter_us:.3} us/iter");
    eprintln!();

    if excess_per_iter_us > 20.0 {
        eprintln!("[VERDICT] Significant transition cost (> 20 us/iter).");
        eprintln!("          Fusion would recover this. Worth attacking.");
    } else if excess_per_iter_us > 5.0 {
        eprintln!("[VERDICT] Modest transition cost ({excess_per_iter_us:.1} us/iter).");
        eprintln!("          Fusion would help but the win is bounded.");
    } else if excess_per_iter_us > -2.0 {
        eprintln!("[VERDICT] Negligible transition cost ({excess_per_iter_us:.1} us/iter).");
        eprintln!("          Kernels are already pipelined. Fusion saves nothing.");
        eprintln!("          The 23% session gap was likely rocprof artifact.");
    } else {
        eprintln!("[VERDICT] Negative excess ({excess_per_iter_us:.1} us/iter) — measurement");
        eprintln!("          noise or hipEvent overhead dominates. Repeat with more iters.");
    }

    Ok(())
}
