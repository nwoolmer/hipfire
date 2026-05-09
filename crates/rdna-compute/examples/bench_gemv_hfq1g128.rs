//! Bench gemv_hfq1g128 single-row vs multi-row R∈{2,4,8} on a Bonsai-shaped
//! matmul (FFN up: 12288×4096). Reports effective DRAM bandwidth and the
//! decode-tok/s ceiling that bandwidth implies for an 8B model at 1.125 bpw.
//!
//! Run:  cargo run -p rdna-compute --release --example bench_gemv_hfq1g128

use hipfire_quantize::q1_0::quantize_row;

const M: usize = 12288;
const K: usize = 4096;
const WARMUP: usize = 20;
const ITERS: usize = 200;

const BONSAI_8B_BYTES: f64 = 1_158_654_496.0; // weight bytes per decode step

fn lcg(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    (*state as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn bench_one(
    gpu: &mut rdna_compute::Gpu,
    label: &str,
    d_a: &rdna_compute::GpuTensor,
    d_x: &rdna_compute::GpuTensor,
    d_y: &rdna_compute::GpuTensor,
    launch: &mut dyn FnMut(&mut rdna_compute::Gpu) -> hip_bridge::HipResult<()>,
) -> Result<(), String> {
    // Warmup
    for _ in 0..WARMUP {
        launch(gpu).map_err(|e| format!("warmup: {e:?}"))?;
    }
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip.stream_synchronize(s).map_err(|e| format!("sync: {e:?}"))?;
    }

    // Time
    let start = gpu.hip.event_create().map_err(|e| format!("ev: {e:?}"))?;
    let stop = gpu.hip.event_create().map_err(|e| format!("ev: {e:?}"))?;
    gpu.hip.event_record(&start, gpu.active_stream.as_ref())
        .map_err(|e| format!("rec: {e:?}"))?;
    for _ in 0..ITERS {
        launch(gpu).map_err(|e| format!("iter: {e:?}"))?;
    }
    gpu.hip.event_record(&stop, gpu.active_stream.as_ref())
        .map_err(|e| format!("rec: {e:?}"))?;
    gpu.hip.event_synchronize(&stop).map_err(|e| format!("sync: {e:?}"))?;
    let total_ms = gpu.hip.event_elapsed_ms(&start, &stop)
        .map_err(|e| format!("elapsed: {e:?}"))?;
    gpu.hip.event_destroy(start).ok();
    gpu.hip.event_destroy(stop).ok();

    // Suppress unused-warning for d_a/d_x/d_y; the launches use them.
    let _ = (d_a, d_x, d_y);

    let per_launch_ms = total_ms / ITERS as f32;
    let weight_bytes = (M * K) as f64 * 9.0 / 64.0; // HFQ1G128 = 9 bits/8 elements
    let bandwidth_gb_s = (weight_bytes * ITERS as f64) / (total_ms as f64 / 1000.0) / 1e9;
    let bonsai_tok_s = 1000.0 / (per_launch_ms as f64 * (BONSAI_8B_BYTES / weight_bytes));

    println!(
        "  {label:34} {per_launch_ms:>7.4} ms/launch  {bandwidth_gb_s:>6.1} GB/s  → {bonsai_tok_s:>5.1} tok/s @ Bonsai-8B"
    );
    Ok(())
}

fn main() -> Result<(), String> {
    println!("=== gemv_hfq1g128 microbench ({M}×{K}) ===");
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    println!("  GPU: {} (active stream: {})", gpu.arch, gpu.active_stream.is_some());

    // Random weights → HFQ1G128 bytes.
    let mut s = 0xDEADBEEFu32;
    let mut weights = vec![0.0f32; M * K];
    for v in weights.iter_mut() {
        *v = lcg(&mut s);
    }
    let mut weight_bytes: Vec<u8> = Vec::with_capacity(M * (K / 128) * 18);
    for row in 0..M {
        weight_bytes.extend_from_slice(&quantize_row(&weights[row * K..(row + 1) * K]));
    }

    let mut x = vec![0.0f32; K];
    for v in x.iter_mut() {
        *v = lcg(&mut s) * 0.5;
    }

    let d_a = gpu.upload_raw(&weight_bytes, &[M, K]).map_err(|e| format!("up a: {e:?}"))?;
    let d_x = gpu.upload_f32(&x, &[K]).map_err(|e| format!("up x: {e:?}"))?;
    let d_y = gpu.zeros(&[M], rdna_compute::DType::F32).map_err(|e| format!("z y: {e:?}"))?;

    let weight_bytes_total = (M * K) as f64 * 9.0 / 64.0;
    println!(
        "  Weight footprint: {:.2} MB ({:.2} bpw avg)",
        weight_bytes_total / 1e6,
        (weight_bytes_total * 8.0) / (M * K) as f64
    );
    println!(
        "  Bonsai-8B model:  {:.0} MB total weight bytes",
        BONSAI_8B_BYTES / 1e6
    );
    println!();

    // Single-row baseline.
    bench_one(&mut gpu, "single-row (Phase 1 baseline)", &d_a, &d_x, &d_y, &mut |gpu| {
        gpu.gemv_hfq1g128(&d_a, &d_x, &d_y, M, K)
    })?;

    // Multi-row variants.
    bench_one(&mut gpu, "multirow R=2", &d_a, &d_x, &d_y, &mut |gpu| {
        gpu.gemv_hfq1g128_multirow(&d_a, &d_x, &d_y, M, K, 2)
    })?;
    bench_one(&mut gpu, "multirow R=4", &d_a, &d_x, &d_y, &mut |gpu| {
        gpu.gemv_hfq1g128_multirow(&d_a, &d_x, &d_y, M, K, 4)
    })?;
    bench_one(&mut gpu, "multirow R=8", &d_a, &d_x, &d_y, &mut |gpu| {
        gpu.gemv_hfq1g128_multirow(&d_a, &d_x, &d_y, M, K, 8)
    })?;

    // Packed-2-groups variants.
    bench_one(&mut gpu, "packed-2g R=1 (single row)", &d_a, &d_x, &d_y, &mut |gpu| {
        gpu.gemv_hfq1g128_packed(&d_a, &d_x, &d_y, M, K, 1)
    })?;
    bench_one(&mut gpu, "packed-2g R=2", &d_a, &d_x, &d_y, &mut |gpu| {
        gpu.gemv_hfq1g128_packed(&d_a, &d_x, &d_y, M, K, 2)
    })?;
    bench_one(&mut gpu, "packed-2g R=4", &d_a, &d_x, &d_y, &mut |gpu| {
        gpu.gemv_hfq1g128_packed(&d_a, &d_x, &d_y, M, K, 4)
    })?;

    // Quad-group multirow variants (4 groups packed per K-step).
    bench_one(&mut gpu, "multirow-quad R=2", &d_a, &d_x, &d_y, &mut |gpu| {
        gpu.gemv_hfq1g128_multirow_quad(&d_a, &d_x, &d_y, M, K, 2)
    })?;
    bench_one(&mut gpu, "multirow-quad R=4", &d_a, &d_x, &d_y, &mut |gpu| {
        gpu.gemv_hfq1g128_multirow_quad(&d_a, &d_x, &d_y, M, K, 4)
    })?;
    bench_one(&mut gpu, "multirow-quad R=8", &d_a, &d_x, &d_y, &mut |gpu| {
        gpu.gemv_hfq1g128_multirow_quad(&d_a, &d_x, &d_y, M, K, 8)
    })?;

    gpu.free_tensor(d_a).ok();
    gpu.free_tensor(d_x).ok();
    gpu.free_tensor(d_y).ok();
    Ok(())
}
