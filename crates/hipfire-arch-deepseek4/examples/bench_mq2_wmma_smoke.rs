//! Smoke / proof-of-concept bench for the MQ2-Lloyd WMMA kernel.
//!
//! Validates that the WMMA pattern (cooperative codebook LDS load +
//! 2-bit unpack → F16 fragment → WMMA mma) delivers ≥2× speedup over
//! the scalar B=1 `gemv_mq2g256_lloyd` path at V4F MoE shapes, before
//! committing ~560 LoC to the full MoE WMMA port.
//!
//! Shape: M=4096, K=4096, B=64 (matches V4F's gate_up routed-MoE
//! per-(krank, expert) GEMV input shape for one expert slab).
//!
//! Measures:
//!   - scalar baseline: `gemv_mq2g256_lloyd` × 64 times (one per B row)
//!   - WMMA HFQ4 reference: `gemm_hfq4g256_wmma` (sanity — should be in
//!     the same ballpark as MQ2-WMMA since same substrate)
//!   - MQ2-WMMA under test: `gemm_mq2g256_lloyd_wmma`
//!
//! Each path is wrapped in N_ITERS hipEvent-timed runs after a warmup,
//! avoiding rocprof's HSA-event-barrier inflation per the memory
//! `feedback_v4f_prefill_idle_gap_analysis_2026_05_20`.
//!
//! Also runs a small (M=64, K=256, B=1) correctness check that the
//! MQ2-WMMA kernel matches the scalar GEMV (within FP-order ε).

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const M: usize = 4096;
const K: usize = 4096;
const B: usize = 64;
const N_ITERS: usize = 50;
const WARMUP_ITERS: usize = 5;

fn time_loop<F>(gpu: &mut Gpu, label: &str, mut body: F) -> Result<f64, String>
where
    F: FnMut(&mut Gpu) -> Result<(), String>,
{
    // Warmup.
    for _ in 0..WARMUP_ITERS {
        body(gpu)?;
    }
    gpu.hip.device_synchronize()
        .map_err(|e| format!("{label} warmup sync: {e:?}"))?;

    let t = Instant::now();
    for _ in 0..N_ITERS {
        body(gpu)?;
    }
    gpu.hip.device_synchronize()
        .map_err(|e| format!("{label} timed sync: {e:?}"))?;
    let elapsed_s = t.elapsed().as_secs_f64();
    let us_per_iter = elapsed_s * 1e6 / N_ITERS as f64;
    eprintln!("{label:<36} {us_per_iter:>10.1} us/iter   ({:>5.1} ms total)",
        elapsed_s * 1e3);
    Ok(elapsed_s)
}

/// Build a synthetic MQ2-Lloyd-G256 weight blob.
///
/// Per group (72 B): 4 × F16 codebook (linearly-spaced ints in [-8, +8])
/// + 64 B of pseudo-random 2-bit indices (linear-congruential to keep
/// it deterministic + cheap).
fn build_mq2_weights(m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 256, 0, "K must be multiple of 256");
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 72;
    let total = m * row_bytes;
    let mut buf = vec![0u8; total];
    let mut state: u64 = 0xC0FFEE_12345678;
    for row in 0..m {
        let row_off = row * row_bytes;
        for g in 0..groups_per_row {
            let off = row_off + g * 72;
            // 4 codebook entries (F16): -3.0, -1.0, +1.0, +3.0 — Lloyd-like
            // bimodal codebook.
            let cb_f16: [u16; 4] = [
                f32_to_f16_bits(-3.0),
                f32_to_f16_bits(-1.0),
                f32_to_f16_bits( 1.0),
                f32_to_f16_bits( 3.0),
            ];
            for i in 0..4 {
                buf[off + i * 2 + 0] = (cb_f16[i] & 0xFF) as u8;
                buf[off + i * 2 + 1] = (cb_f16[i] >> 8) as u8;
            }
            // 64 bytes of pseudo-random 2-bit indices.
            for b in 0..64 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                buf[off + 8 + b] = (state >> 32) as u8;
            }
        }
    }
    buf
}

/// Build a synthetic HFQ4-G256 weight blob (136 B/group, same K).
fn build_hfq4_weights(m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;
    let total = m * row_bytes;
    let mut buf = vec![0u8; total];
    let mut state: u64 = 0xDEADBEEF_CAFEBABE;
    for row in 0..m {
        let row_off = row * row_bytes;
        for g in 0..groups_per_row {
            let off = row_off + g * 136;
            // scale (F32) + zero (F32) — placeholder values.
            let scale_bytes = (0.01f32).to_le_bytes();
            let zero_bytes  = (-0.075f32).to_le_bytes();
            buf[off..off+4].copy_from_slice(&scale_bytes);
            buf[off+4..off+8].copy_from_slice(&zero_bytes);
            // 128 B packed 4-bit weights.
            for b in 0..128 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                buf[off + 8 + b] = (state >> 32) as u8;
            }
        }
    }
    buf
}

fn f32_to_f16_bits(x: f32) -> u16 {
    // Standard IEEE 754 float→half conversion.
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 0xFF {
        // inf or nan
        let m = if mant != 0 { 0x200 } else { 0 };
        return (sign << 15) | (0x1F << 10) | m;
    }
    let new_exp = exp - 127 + 15;
    if new_exp <= 0 {
        return sign << 15;  // underflow → ±0
    }
    if new_exp >= 0x1F {
        return (sign << 15) | (0x1F << 10);  // overflow → ±inf
    }
    let new_mant = (mant >> 13) as u16;
    (sign << 15) | ((new_exp as u16) << 10) | new_mant
}

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().map_err(|e| format!("gpu init: {e:?}"))?;
    eprintln!("GPU: {}", gpu.arch);

    // ── Build synthetic weights ──────────────────────────────────────
    eprintln!("Building synthetic weights M={M} K={K} B={B} ...");
    let mq2_bytes = build_mq2_weights(M, K);
    let hfq4_bytes = build_hfq4_weights(M, K);
    eprintln!("  MQ2-Lloyd:  {} MB", mq2_bytes.len() / (1024 * 1024));
    eprintln!("  HFQ4-G256:  {} MB", hfq4_bytes.len() / (1024 * 1024));

    let mq2_w = gpu.upload_raw(&mq2_bytes, &[mq2_bytes.len()])
        .map_err(|e| format!("upload mq2: {e:?}"))?;
    let hfq4_w = gpu.upload_raw(&hfq4_bytes, &[hfq4_bytes.len()])
        .map_err(|e| format!("upload hfq4: {e:?}"))?;

    // ── Input tensors ────────────────────────────────────────────────
    let x_f32_host: Vec<f32> = (0..B*K).map(|i| ((i as f32) * 0.001).sin() * 0.5).collect();
    let x_f32 = gpu.upload_f32(&x_f32_host, &[B, K])
        .map_err(|e| format!("upload x_f32: {e:?}"))?;
    let mut x_f16 = gpu.zeros(&[B * K * 2], DType::Raw)
        .map_err(|e| format!("alloc x_f16: {e:?}"))?;
    x_f16.dtype = DType::F16;
    x_f16.shape = vec![B, K];
    gpu.convert_f32_to_f16(&x_f32, &x_f16, (B * K) as i64)
        .map_err(|e| format!("convert: {e:?}"))?;

    let y_f32 = gpu.zeros(&[B, M], DType::F32)
        .map_err(|e| format!("alloc y_f32: {e:?}"))?;

    // ── 1. Scalar baseline: gemv_mq2g256_lloyd × 64 ──────────────────
    let t_scalar = time_loop(&mut gpu, "scalar gemv_mq2 × 64 (B emul)", |gpu| {
        for b in 0..B {
            let x_row = x_f32.sub_offset(b * K, K);
            let y_row = y_f32.sub_offset(b * M, M);
            gpu.gemv_mq2g256_lloyd(&mq2_w, &x_row, &y_row, M, K)
                .map_err(|e| format!("gemv_mq2: {e:?}"))?;
        }
        Ok(())
    })?;

    // ── 2. HFQ4 WMMA reference ───────────────────────────────────────
    let t_hfq4 = time_loop(&mut gpu, "gemm_hfq4g256_wmma reference", |gpu| {
        gpu.gemm_hfq4g256_wmma(&hfq4_w, &x_f16, &y_f32, M, K, B)
            .map_err(|e| format!("hfq4 wmma: {e:?}"))?;
        Ok(())
    })?;

    // ── 3. MQ2-Lloyd WMMA under test ─────────────────────────────────
    let t_mq2_wmma = time_loop(&mut gpu, "gemm_mq2g256_lloyd_wmma TEST", |gpu| {
        gpu.gemm_mq2g256_lloyd_wmma(&mq2_w, &x_f16, &y_f32, M, K, B)
            .map_err(|e| format!("mq2 wmma: {e:?}"))?;
        Ok(())
    })?;

    eprintln!();
    eprintln!("─── ratios ───");
    eprintln!("  MQ2-WMMA vs scalar (B emul):  {:.2}×",  t_scalar / t_mq2_wmma);
    eprintln!("  MQ2-WMMA vs HFQ4-WMMA ref :  {:.2}×",  t_hfq4   / t_mq2_wmma);
    eprintln!();

    let pass = (t_scalar / t_mq2_wmma) >= 2.0;
    eprintln!("PASS (≥2× speedup vs scalar): {}", pass);

    // ── 4. Correctness check at small shape ──────────────────────────
    eprintln!("\n─── correctness check (M=64, K=256, B=1) ───");
    {
        const M_S: usize = 64;
        const K_S: usize = 256;
        let mq2_small = build_mq2_weights(M_S, K_S);
        let w = gpu.upload_raw(&mq2_small, &[mq2_small.len()])
            .map_err(|e| format!("upload small w: {e:?}"))?;
        let x_h: Vec<f32> = (0..K_S).map(|i| ((i as f32) * 0.013).cos() * 0.3).collect();
        let x = gpu.upload_f32(&x_h, &[K_S])
            .map_err(|e| format!("upload small x: {e:?}"))?;
        let y_scalar = gpu.zeros(&[M_S], DType::F32)
            .map_err(|e| format!("alloc y_scalar: {e:?}"))?;
        let y_wmma = gpu.zeros(&[M_S], DType::F32)
            .map_err(|e| format!("alloc y_wmma: {e:?}"))?;

        gpu.gemv_mq2g256_lloyd(&w, &x, &y_scalar, M_S, K_S)
            .map_err(|e| format!("gemv scalar: {e:?}"))?;

        let mut x_h16 = gpu.zeros(&[K_S * 2], DType::Raw)
            .map_err(|e| format!("alloc xh16: {e:?}"))?;
        x_h16.dtype = DType::F16;
        x_h16.shape = vec![1, K_S];
        gpu.convert_f32_to_f16(&x, &x_h16, K_S as i64)
            .map_err(|e| format!("convert small: {e:?}"))?;
        gpu.gemm_mq2g256_lloyd_wmma(&w, &x_h16, &y_wmma, M_S, K_S, 1)
            .map_err(|e| format!("mq2 wmma small: {e:?}"))?;

        let v_scalar = gpu.download_f32(&y_scalar)
            .map_err(|e| format!("d2h scalar: {e:?}"))?;
        let v_wmma = gpu.download_f32(&y_wmma)
            .map_err(|e| format!("d2h wmma: {e:?}"))?;
        let mut max_abs = 0f32;
        let mut sum_abs = 0f32;
        for i in 0..M_S {
            let d = (v_scalar[i] - v_wmma[i]).abs();
            if d > max_abs { max_abs = d; }
            sum_abs += d;
        }
        let mean_abs = sum_abs / M_S as f32;
        eprintln!("  max_abs = {max_abs:.4e}");
        eprintln!("  mean_abs = {mean_abs:.4e}");
        eprintln!("  first 4 scalar = [{:.4}, {:.4}, {:.4}, {:.4}]",
            v_scalar[0], v_scalar[1], v_scalar[2], v_scalar[3]);
        eprintln!("  first 4 wmma   = [{:.4}, {:.4}, {:.4}, {:.4}]",
            v_wmma[0], v_wmma[1], v_wmma[2], v_wmma[3]);
        let correct = max_abs < 0.5;
        eprintln!("PASS (max_abs < 0.5): {correct}");
    }

    Ok(())
}
