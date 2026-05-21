//! Apples-to-apples MoE kernel TFLOPs comparison: MQ2-Lloyd vs HFQ4.
//!
//! Same kernel family (`gemv_*_moe_gate_up_k8_indexed_batched(_k4)`),
//! identical shape `[M, K, K_TOP, B]`, synthetic weights + inputs. Times
//! each with hipEvents (no rocprof inflation), reports sustained TFLOPs.
//!
//! Purpose: prove whether V4F's ~0.47 TFLOPs sustained on MoE is a
//! QUANT-FORMAT cost (MQ2-Lloyd codebook lookup vs HFQ4 affine dequant)
//! or a SHAPE/KERNEL-DESIGN cost (the K4 unroll itself is inefficient).
//!
//! Test matrix: V4F shape (M=4096 K=4096 K_TOP=6 B=64) — direct match
//! to the rocprof'd MoE kernel call.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const M: usize = 4096;       // 2 * IM for gate_up
const K: usize = 4096;       // hidden
const K_TOP: usize = 6;
const B: usize = 64;
const N_EXP: usize = 256;    // realistic expert count for routing
const N_ITERS: usize = 30;
const WARMUP_ITERS: usize = 5;

fn time_loop<F>(gpu: &mut Gpu, label: &str, gflops_per_iter: f64, mut body: F) -> Result<f64, String>
where F: FnMut(&mut Gpu) -> Result<(), String>,
{
    for _ in 0..WARMUP_ITERS { body(gpu)?; }
    gpu.hip.device_synchronize().map_err(|e| format!("{label} warmup sync: {e:?}"))?;

    let t = Instant::now();
    for _ in 0..N_ITERS { body(gpu)?; }
    gpu.hip.device_synchronize().map_err(|e| format!("{label} timed sync: {e:?}"))?;
    let elapsed_s = t.elapsed().as_secs_f64();
    let us_per_iter = elapsed_s * 1e6 / N_ITERS as f64;
    let tflops = gflops_per_iter / 1000.0 / (us_per_iter / 1e6);
    eprintln!("{label:<60} {us_per_iter:>10.1} us/iter   {tflops:>6.2} TFLOPs");
    Ok(elapsed_s)
}

// Build a synthetic per-expert MQ2-Lloyd-G256 weight slab [M, K].
// Per group: 8 B (4×F16 codebook) + 64 B (256 2-bit indices) = 72 B.
fn build_mq2_expert_slab(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 72;
    let mut buf = vec![0u8; m * row_bytes];
    let mut state = seed;
    for row in 0..m {
        let row_off = row * row_bytes;
        for g in 0..groups_per_row {
            let off = row_off + g * 72;
            let cb_f16: [u16; 4] = [
                f32_to_f16_bits(-3.0), f32_to_f16_bits(-1.0),
                f32_to_f16_bits( 1.0), f32_to_f16_bits( 3.0),
            ];
            for i in 0..4 {
                buf[off + i*2] = (cb_f16[i] & 0xFF) as u8;
                buf[off + i*2 + 1] = (cb_f16[i] >> 8) as u8;
            }
            for b in 0..64 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                buf[off + 8 + b] = (state >> 32) as u8;
            }
        }
    }
    buf
}

// Build a synthetic per-expert HFQ4-G256 weight slab [M, K].
// Per group: 4 B (F32 scale) + 4 B (F32 zero) + 128 B (256 4-bit indices) = 136 B.
fn build_hfq4_expert_slab(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;
    let mut buf = vec![0u8; m * row_bytes];
    let mut state = seed;
    for row in 0..m {
        let row_off = row * row_bytes;
        for g in 0..groups_per_row {
            let off = row_off + g * 136;
            buf[off..off+4].copy_from_slice(&(0.01f32).to_le_bytes());
            buf[off+4..off+8].copy_from_slice(&(-0.075f32).to_le_bytes());
            for b in 0..128 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                buf[off + 8 + b] = (state >> 32) as u8;
            }
        }
    }
    buf
}

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 0xFF {
        let m = if mant != 0 { 0x200 } else { 0 };
        return (sign << 15) | (0x1F << 10) | m;
    }
    let new_exp = exp - 127 + 15;
    if new_exp <= 0 { return sign << 15; }
    if new_exp >= 0x1F { return (sign << 15) | (0x1F << 10); }
    (sign << 15) | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    eprintln!("GPU: {}", gpu.arch);
    eprintln!("Shape: M={M}  K={K}  K_TOP={K_TOP}  B={B}  N_EXP={N_EXP}");
    eprintln!();

    // Per-call FMA count for gate_up: M * K * K_TOP * B
    let gflops_per_iter = 2.0 * (M as f64) * (K as f64) * (K_TOP as f64) * (B as f64) / 1e9;
    eprintln!("Theoretical work per call: {:.2} GFLOPs", gflops_per_iter);
    eprintln!();

    // Build expert pointer tables. Each expert gets its own slab. To exercise
    // realistic cache pressure across experts, allocate N_EXP slabs each.
    eprintln!("Building {N_EXP} expert weight slabs (MQ2-Lloyd, HFQ4)...");
    let mut mq2_slabs: Vec<rdna_compute::GpuTensor> = Vec::with_capacity(N_EXP);
    let mut hfq4_slabs: Vec<rdna_compute::GpuTensor> = Vec::with_capacity(N_EXP);
    let mut mq2_ptrs: Vec<u64> = Vec::with_capacity(N_EXP);
    let mut hfq4_ptrs: Vec<u64> = Vec::with_capacity(N_EXP);
    for e in 0..N_EXP {
        let mq2_bytes = build_mq2_expert_slab(M, K, 0xC0FFEE_00 + e as u64);
        let mq2_w = gpu.upload_raw(&mq2_bytes, &[mq2_bytes.len()])
            .map_err(|e| format!("upload mq2 exp {e:?}"))?;
        mq2_ptrs.push(mq2_w.buf.as_ptr() as u64);
        mq2_slabs.push(mq2_w);

        let hfq4_bytes = build_hfq4_expert_slab(M, K, 0xDEADBEEF + e as u64);
        let hfq4_w = gpu.upload_raw(&hfq4_bytes, &[hfq4_bytes.len()])
            .map_err(|e| format!("upload hfq4 exp {e:?}"))?;
        hfq4_ptrs.push(hfq4_w.buf.as_ptr() as u64);
        hfq4_slabs.push(hfq4_w);
    }
    let mq2_per_expert = M * (K/256) * 72;
    let hfq4_per_expert = M * (K/256) * 136;
    eprintln!("  MQ2-Lloyd:  {} MB per expert × {} = {} MB total",
        mq2_per_expert / (1024*1024), N_EXP,
        mq2_per_expert * N_EXP / (1024*1024));
    eprintln!("  HFQ4-G256:  {} MB per expert × {} = {} MB total",
        hfq4_per_expert / (1024*1024), N_EXP,
        hfq4_per_expert * N_EXP / (1024*1024));

    // Expert pointer tables.
    let mq2_ptr_bytes: Vec<u8> = mq2_ptrs.iter().flat_map(|p| p.to_le_bytes()).collect();
    let mq2_ptr_tensor = gpu.upload_raw(&mq2_ptr_bytes, &[mq2_ptrs.len()])
        .map_err(|e| format!("upload mq2 ptr: {e:?}"))?;
    let hfq4_ptr_bytes: Vec<u8> = hfq4_ptrs.iter().flat_map(|p| p.to_le_bytes()).collect();
    let hfq4_ptr_tensor = gpu.upload_raw(&hfq4_ptr_bytes, &[hfq4_ptrs.len()])
        .map_err(|e| format!("upload hfq4 ptr: {e:?}"))?;

    // topk_indices [B, K_TOP] i32. Spread across experts to simulate real routing.
    let mut topk_idx_host = Vec::<i32>::with_capacity(B * K_TOP);
    for b in 0..B {
        for k in 0..K_TOP {
            topk_idx_host.push(((b * 17 + k * 19) % N_EXP) as i32);
        }
    }
    let topk_bytes: Vec<u8> = topk_idx_host.iter().flat_map(|i| i.to_le_bytes()).collect();
    let topk = gpu.upload_raw(&topk_bytes, &[B, K_TOP])
        .map_err(|e| format!("upload topk: {e:?}"))?;

    // Input x [B, K] f32 (post-rotation, as the kernel expects).
    let x_host: Vec<f32> = (0..B*K).map(|i| ((i as f32) * 0.001).sin() * 0.3).collect();
    let x = gpu.upload_f32(&x_host, &[B, K])
        .map_err(|e| format!("upload x: {e:?}"))?;

    // Outputs: y_gate, y_up [B, K_TOP, MI] (with MI = M/2).
    let mi = M / 2;
    let y_gate = gpu.zeros(&[B, K_TOP, mi], DType::F32)
        .map_err(|e| format!("alloc y_gate: {e:?}"))?;
    let y_up = gpu.zeros(&[B, K_TOP, mi], DType::F32)
        .map_err(|e| format!("alloc y_up: {e:?}"))?;

    eprintln!();
    eprintln!("─── kernel timing ─────────────────────────────────────────────────");

    // MQ2-Lloyd K4 (what V4F currently uses).
    time_loop(&mut gpu, "MQ2-Lloyd K4  (V4F production)", gflops_per_iter, |gpu| {
        gpu.v4f_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4(
            &mq2_ptr_tensor, &topk, &x, &y_gate, &y_up,
            M, K, K_TOP, B,
        ).map_err(|e| format!("mq2 k4: {e:?}"))
    })?;

    // HFQ4 K8-indexed-batched (Qwen35 production class).
    time_loop(&mut gpu, "HFQ4    K8 batched (Qwen35 production)", gflops_per_iter, |gpu| {
        gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
            &hfq4_ptr_tensor, &topk, &x, &y_gate, &y_up,
            M, K, K_TOP, B,
        ).map_err(|e| format!("hfq4: {e:?}"))
    })?;

    Ok(())
}
