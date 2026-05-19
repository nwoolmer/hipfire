//! Standalone microbench for the V4F hot-path kernels — no model load,
//! just enough scratch to exercise each kernel with V4F shapes at B=64.
//!
//! Usage:
//!   cargo run --release --example microbench_v4f_kernels -p rdna-compute -- <kernel> [iters]
//!
//! Kernels:
//!   gemm_f32              — gemm_f32_register_tiled at compressor shape (M=1024, K=4096, B=64)
//!   gemm_f32_w2           — 64-thread variant
//!   gemm_f32_bt32         — BATCH_TILE=32 variant
//!   moe_gateup_k4         — gemv_mq2g256_lloyd_moe_gate_up_k4 (M=4096, K=4096, B=64, K_TOP=6)
//!   moe_gateup_grouped_k4 — grouped variant
//!   gemm_hfq4g256         — attention proj shape (M=2048, K=4096, B=64)
//!   wo_per_group_hfq4     — wo_a shape (G=8, M=1024, K=4096, B=64)
//!   all                   — run all of them
//!
//! Each runs <iters> consecutive launches with hipDeviceSynchronize at
//! the end and reports total + per-iter wallclock + an estimated
//! bandwidth (kernel-perspective bytes / time).

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn parse_iters(args: &[String]) -> usize {
    args.get(2).and_then(|s| s.parse().ok()).unwrap_or(40)
}

fn bench<F: FnMut() -> Result<(), String>>(
    name: &str, iters: usize, kbytes_per_iter: f64, mut f: F,
) -> Result<(), String> {
    // Warm up.
    for _ in 0..2 {
        f()?;
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let per_iter_ms = total_ms / iters as f64;
    let bytes_per_iter = kbytes_per_iter * 1024.0;
    let bw_gbs = (bytes_per_iter * iters as f64) / (total_ms / 1000.0) / 1e9;
    println!(
        "{name:<32} iters={iters:>3} total={total_ms:>8.2} ms  per={per_iter_ms:>7.3} ms  est_bw={bw_gbs:>7.1} GB/s  (per_iter_bytes={:.1} MB)",
        bytes_per_iter / 1_048_576.0
    );
    Ok(())
}

fn run_gemm_f32_per_output(gpu: &mut Gpu, iters: usize) -> Result<(), String> {
    let m = 1024;
    let k = 4096;
    let b = 64;
    let weight = gpu.alloc_tensor(&[m, k], DType::F32)
        .map_err(|e| format!("alloc weight: {e:?}"))?;
    let x = gpu.alloc_tensor(&[b, k], DType::F32)
        .map_err(|e| format!("alloc x: {e:?}"))?;
    let y = gpu.alloc_tensor(&[b, m], DType::F32)
        .map_err(|e| format!("alloc y: {e:?}"))?;
    let kbytes = ((m * k * 4) + (b * k * 4) + (b * m * 4)) as f64 / 1024.0;
    bench("gemm_f32_per_output", iters, kbytes, || {
        gpu.gemm_f32_per_output(&weight, &x, &y, m, k, b)
            .map_err(|e| format!("gemm_f32_per_output: {e:?}"))?;
        gpu.hip.device_synchronize()
            .map_err(|e| format!("sync: {e:?}"))
    })?;
    Ok(())
}

fn run_gemm_f16_wmma(gpu: &mut Gpu, iters: usize) -> Result<(), String> {
    let m = 1024;
    let k = 4096;
    let b = 64;
    // F16 weight: M*K*2 bytes — allocate as Raw to bypass dtype size.
    let weight_bytes = m * k * 2;
    let weight = gpu.zeros(&[weight_bytes], DType::Raw)
        .map_err(|e| format!("alloc weight: {e:?}"))?;
    let x = gpu.zeros(&[b * k * 2], DType::Raw)
        .map_err(|e| format!("alloc x: {e:?}"))?;
    let y = gpu.alloc_tensor(&[b, m], DType::F32)
        .map_err(|e| format!("alloc y: {e:?}"))?;
    // Bytes-per-iter: weight F16 + x F16 + y F32.
    let kbytes = ((m * k * 2) + (b * k * 2) + (b * m * 4)) as f64 / 1024.0;
    bench("gemm_f16_x_f16_wmma", iters, kbytes, || {
        gpu.gemm_f16_x_f16_wmma(&weight, &x, &y, m, k, b)
            .map_err(|e| format!("gemm_f16_x_f16_wmma: {e:?}"))?;
        gpu.hip.device_synchronize()
            .map_err(|e| format!("sync: {e:?}"))
    })?;
    Ok(())
}

fn run_gemm_f32_per_output_v4(gpu: &mut Gpu, iters: usize) -> Result<(), String> {
    let m = 1024;
    let k = 4096;
    let b = 64;
    let weight = gpu.alloc_tensor(&[m, k], DType::F32)
        .map_err(|e| format!("alloc weight: {e:?}"))?;
    let x = gpu.alloc_tensor(&[b, k], DType::F32)
        .map_err(|e| format!("alloc x: {e:?}"))?;
    let y = gpu.alloc_tensor(&[b, m], DType::F32)
        .map_err(|e| format!("alloc y: {e:?}"))?;
    let kbytes = ((m * k * 4) + (b * k * 4) + (b * m * 4)) as f64 / 1024.0;
    bench("gemm_f32_per_output_v4", iters, kbytes, || {
        gpu.gemm_f32_per_output_v4(&weight, &x, &y, m, k, b)
            .map_err(|e| format!("gemm_f32_per_output_v4: {e:?}"))?;
        gpu.hip.device_synchronize()
            .map_err(|e| format!("sync: {e:?}"))
    })?;
    Ok(())
}

fn run_gemm_f32(gpu: &mut Gpu, iters: usize, env_flag: &str, env_val: &str, name: &str) -> Result<(), String> {
    let m = 1024;
    let k = 4096;
    let b = 64;
    let weight = gpu.alloc_tensor(&[m, k], DType::F32)
        .map_err(|e| format!("alloc weight: {e:?}"))?;
    let x = gpu.alloc_tensor(&[b, k], DType::F32)
        .map_err(|e| format!("alloc x: {e:?}"))?;
    let y = gpu.alloc_tensor(&[b, m], DType::F32)
        .map_err(|e| format!("alloc y: {e:?}"))?;
    std::env::set_var(env_flag, env_val);
    // Bytes per iter: weight read (m*k*4) + x read (b*k*4) + y write (b*m*4).
    // Kernel-perspective; cache reuse will reduce DRAM bytes below this.
    let kbytes = ((m * k * 4) + (b * k * 4) + (b * m * 4)) as f64 / 1024.0;
    bench(name, iters, kbytes, || {
        gpu.gemm_f32_register_tiled(&weight, &x, &y, m, k, b)
            .map_err(|e| format!("gemm_f32_register_tiled: {e:?}"))?;
        gpu.hip.device_synchronize()
            .map_err(|e| format!("sync: {e:?}"))
    })?;
    Ok(())
}

fn run_gemm_hfq4_wmma(gpu: &mut Gpu, iters: usize) -> Result<(), String> {
    let m = 2048;
    let k = 4096;
    let b = 64;
    let weight_bytes = m * (k / 256) * 136;
    let weight_raw = gpu.zeros(&[weight_bytes], DType::Raw)
        .map_err(|e| format!("alloc weight: {e:?}"))?;
    let x_f16 = gpu.zeros(&[b * k * 2], DType::Raw)
        .map_err(|e| format!("alloc x_f16: {e:?}"))?;
    let y = gpu.alloc_tensor(&[b, m], DType::F32)
        .map_err(|e| format!("alloc y: {e:?}"))?;
    let kbytes = (weight_bytes + (b * k * 2) + (b * m * 4)) as f64 / 1024.0;
    bench("gemm_hfq4g256_wmma", iters, kbytes, || {
        gpu.gemm_hfq4g256_wmma(&weight_raw, &x_f16, &y, m, k, b)
            .map_err(|e| format!("gemm_hfq4g256_wmma: {e:?}"))?;
        gpu.hip.device_synchronize()
            .map_err(|e| format!("sync: {e:?}"))
    })?;
    Ok(())
}

fn run_gemm_hfq4(gpu: &mut Gpu, iters: usize) -> Result<(), String> {
    let m = 2048;
    let k = 4096;
    let b = 64;
    // HFQ4G256 row stride = (k/256) * 136 bytes per row, M rows.
    let weight_bytes = m * (k / 256) * 136;
    let weight_raw = gpu.zeros(&[weight_bytes], DType::Raw)
        .map_err(|e| format!("alloc weight: {e:?}"))?;
    let x = gpu.alloc_tensor(&[b, k], DType::F32)
        .map_err(|e| format!("alloc x: {e:?}"))?;
    let y = gpu.alloc_tensor(&[b, m], DType::F32)
        .map_err(|e| format!("alloc y: {e:?}"))?;
    let kbytes = (weight_bytes + (b * k * 4) + (b * m * 4)) as f64 / 1024.0;
    bench("gemm_hfq4g256", iters, kbytes, || {
        gpu.gemm_hfq4g256(&weight_raw, &x, &y, m, k, b)
            .map_err(|e| format!("gemm_hfq4g256: {e:?}"))?;
        gpu.hip.device_synchronize()
            .map_err(|e| format!("sync: {e:?}"))
    })?;
    Ok(())
}

fn run_wo_per_group_hfq4(gpu: &mut Gpu, iters: usize) -> Result<(), String> {
    let g = 8;
    let m = 1024;
    let k = 4096;
    let b = 64;
    let weight_bytes = g * m * (k / 256) * 136;
    let weight_raw = gpu.zeros(&[weight_bytes], DType::Raw)
        .map_err(|e| format!("alloc weight: {e:?}"))?;
    let x = gpu.alloc_tensor(&[b, g, k], DType::F32)
        .map_err(|e| format!("alloc x: {e:?}"))?;
    let y = gpu.alloc_tensor(&[b, g, m], DType::F32)
        .map_err(|e| format!("alloc y: {e:?}"))?;
    let kbytes = (weight_bytes + (b * g * k * 4) + (b * g * m * 4)) as f64 / 1024.0;
    bench("wo_per_group_hfq4g256", iters, kbytes, || {
        gpu.wo_per_group_batched_hfq4g256(&weight_raw, &x, &y,
            g as i32, m as i32, k as i32, b as i32)
            .map_err(|e| format!("wo_per_group: {e:?}"))?;
        gpu.hip.device_synchronize()
            .map_err(|e| format!("sync: {e:?}"))
    })?;
    Ok(())
}

fn run_moe_gateup_k4(gpu: &mut Gpu, iters: usize, grouped: bool) -> Result<(), String> {
    let m = 4096;
    let k = 4096;
    let k_top = 6;
    let b = 64;
    let n_exp = 256;
    let mi = m / 2;
    // MQ2-Lloyd per-expert: (m * k / 256) * 72 bytes for gate_up (M includes both gate & up).
    let per_expert_bytes = m * (k / 256) * 72;
    let total_exp_bytes = per_expert_bytes * n_exp;

    let expert_blob = gpu.zeros(&[total_exp_bytes], DType::Raw)
        .map_err(|e| format!("alloc expert blob: {e:?}"))?;
    // expert_ptrs table — n_exp × u64. We need to upload device pointers.
    let mut ptrs: Vec<u64> = Vec::with_capacity(n_exp);
    let base = expert_blob.buf.as_ptr() as u64;
    for e in 0..n_exp {
        ptrs.push(base + (e * per_expert_bytes) as u64);
    }
    // expert_ptrs is u64 — 8 bytes each. Allocate as 2× F32 slots.
    let expert_ptrs = gpu.alloc_tensor(&[n_exp * 2], DType::F32)
        .map_err(|e| format!("alloc expert_ptrs: {e:?}"))?;
    let ptr_bytes = unsafe {
        std::slice::from_raw_parts(ptrs.as_ptr() as *const u8, n_exp * 8)
    };
    gpu.hip.memcpy_htod(&expert_ptrs.buf, ptr_bytes)
        .map_err(|e| format!("htod expert_ptrs: {e:?}"))?;

    // Build topk_indices: random expert ids in [0, n_exp).
    let n_routings = b * k_top;
    let mut topk_host = Vec::with_capacity(n_routings);
    for i in 0..n_routings {
        topk_host.push(((i * 31 + 7) % n_exp) as i32);
    }
    let topk_indices = gpu.alloc_tensor(&[b, k_top], DType::F32)
        .map_err(|e| format!("alloc topk: {e:?}"))?;
    let topk_bytes = unsafe {
        std::slice::from_raw_parts(topk_host.as_ptr() as *const u8, n_routings * 4)
    };
    gpu.hip.memcpy_htod(&topk_indices.buf, topk_bytes)
        .map_err(|e| format!("htod topk: {e:?}"))?;

    let x_rot = gpu.alloc_tensor(&[b, k], DType::F32)
        .map_err(|e| format!("alloc x_rot: {e:?}"))?;
    let y_gate = gpu.alloc_tensor(&[b, k_top, mi], DType::F32)
        .map_err(|e| format!("alloc y_gate: {e:?}"))?;
    let y_up = gpu.alloc_tensor(&[b, k_top, mi], DType::F32)
        .map_err(|e| format!("alloc y_up: {e:?}"))?;

    // kernel-perspective bytes: per routing we read per_expert_bytes; total over routings.
    let kbytes = (per_expert_bytes * n_routings) as f64 / 1024.0;

    if grouped {
        let sorted_b = gpu.alloc_tensor(&[n_routings], DType::F32)
            .map_err(|e| format!("alloc sorted_b: {e:?}"))?;
        let sorted_krank = gpu.alloc_tensor(&[n_routings], DType::F32)
            .map_err(|e| format!("alloc sorted_krank: {e:?}"))?;
        let sorted_expert = gpu.alloc_tensor(&[n_routings], DType::F32)
            .map_err(|e| format!("alloc sorted_expert: {e:?}"))?;
        let expert_starts = gpu.alloc_tensor(&[n_exp + 1], DType::F32)
            .map_err(|e| format!("alloc expert_starts: {e:?}"))?;
        gpu.moe_routing_sort_by_expert(
            &topk_indices, &sorted_b, &sorted_krank, &sorted_expert,
            &expert_starts, b as i32, k_top as i32, n_exp as i32,
        ).map_err(|e| format!("sort: {e:?}"))?;
        bench("moe_gateup_grouped_k4", iters, kbytes, || {
            gpu.v4f_gemv_mq2g256_lloyd_moe_gate_up_grouped_k4(
                &expert_ptrs, &sorted_b, &sorted_krank, &sorted_expert,
                &x_rot, &y_gate, &y_up,
                m, k, k_top, b,
            ).map_err(|e| format!("moe_gateup_grouped_k4: {e:?}"))?;
            gpu.hip.device_synchronize()
                .map_err(|e| format!("sync: {e:?}"))
        })?;
    } else {
        bench("moe_gateup_k4", iters, kbytes, || {
            gpu.v4f_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4(
                &expert_ptrs, &topk_indices, &x_rot, &y_gate, &y_up,
                m, k, k_top, b,
            ).map_err(|e| format!("moe_gateup_k4: {e:?}"))?;
            gpu.hip.device_synchronize()
                .map_err(|e| format!("sync: {e:?}"))
        })?;
    }
    Ok(())
}

fn run_dram_peak(gpu: &mut Gpu, iters: usize) -> Result<(), String> {
    // Stream 256 MB src→dst, no compute. This is the ceiling we can
    // ever achieve from DRAM on this device.
    let n_floats = 64 * 1024 * 1024; // 256 MB
    let src = gpu.alloc_tensor(&[n_floats], DType::F32)
        .map_err(|e| format!("alloc src: {e:?}"))?;
    let dst = gpu.alloc_tensor(&[n_floats], DType::F32)
        .map_err(|e| format!("alloc dst: {e:?}"))?;
    let n_f4 = (n_floats / 4) as i64;
    let bytes = n_floats * 4 * 2; // read + write
    let kbytes = bytes as f64 / 1024.0;
    bench("dram_peak (read+write 256MB)", iters, kbytes, || {
        gpu.microbench_dram_read_copy(&src, &dst, n_f4)
            .map_err(|e| format!("microbench: {e:?}"))?;
        gpu.hip.device_synchronize()
            .map_err(|e| format!("sync: {e:?}"))
    })?;
    Ok(())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let which = args.get(1).cloned().unwrap_or_else(|| "all".to_string());
    let iters = parse_iters(&args);

    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    eprintln!("V4F kernel microbench, B=64, iters={iters} (after 2 warmup)");

    match which.as_str() {
        "dram_peak" => run_dram_peak(&mut gpu, iters)?,
        "gemm_f32" => run_gemm_f32(&mut gpu, iters, "HIPFIRE_GEMM_F32_W2", "0", "gemm_f32 (BT=8)")?,
        "gemm_f32_w2" => run_gemm_f32(&mut gpu, iters, "HIPFIRE_GEMM_F32_W2", "1", "gemm_f32_w2")?,
        "gemm_f32_bt32" => run_gemm_f32(&mut gpu, iters, "HIPFIRE_GEMM_F32_BT32", "1", "gemm_f32_bt32")?,
        "gemm_f32_per_output" => run_gemm_f32_per_output(&mut gpu, iters)?,
        "gemm_f32_per_output_v4" => run_gemm_f32_per_output_v4(&mut gpu, iters)?,
        "gemm_f16_wmma" => run_gemm_f16_wmma(&mut gpu, iters)?,
        "gemm_hfq4" => run_gemm_hfq4(&mut gpu, iters)?,
        "gemm_hfq4_wmma" => run_gemm_hfq4_wmma(&mut gpu, iters)?,
        "wo_per_group_hfq4" => run_wo_per_group_hfq4(&mut gpu, iters)?,
        "moe_gateup_k4" => run_moe_gateup_k4(&mut gpu, iters, false)?,
        "moe_gateup_grouped_k4" => run_moe_gateup_k4(&mut gpu, iters, true)?,
        "all" => {
            run_dram_peak(&mut gpu, iters)?;
            run_gemm_f32(&mut gpu, iters, "HIPFIRE_GEMM_F32_W2", "0", "gemm_f32 (BT=8)")?;
            run_gemm_f32(&mut gpu, iters, "HIPFIRE_GEMM_F32_W2", "1", "gemm_f32_w2")?;
            std::env::set_var("HIPFIRE_GEMM_F32_W2", "0");
            run_gemm_f32(&mut gpu, iters, "HIPFIRE_GEMM_F32_BT32", "1", "gemm_f32_bt32")?;
            std::env::set_var("HIPFIRE_GEMM_F32_BT32", "0");
            run_gemm_f32_per_output(&mut gpu, iters)?;
            run_gemm_f32_per_output_v4(&mut gpu, iters)?;
            run_gemm_f16_wmma(&mut gpu, iters)?;
            run_gemm_hfq4(&mut gpu, iters)?;
            run_gemm_hfq4_wmma(&mut gpu, iters)?;
            run_wo_per_group_hfq4(&mut gpu, iters)?;
            run_moe_gateup_k4(&mut gpu, iters, false)?;
            run_moe_gateup_k4(&mut gpu, iters, true)?;
        }
        _ => return Err(format!("unknown kernel: {which}")),
    }
    Ok(())
}
