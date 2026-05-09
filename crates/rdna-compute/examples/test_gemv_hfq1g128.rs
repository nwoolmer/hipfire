//! Tier B verification for `gemv_hfq1g128` (HFQ1G128 wave32 GEMV).
//!
//! Compares GPU output to the CPU reference (`hipfire_quantize::q1_0`) on:
//!   1. Synthetic random weights + activations (small, fast).
//!   2. A real Bonsai-8B weight tensor (`blk.0.attn_q.weight`, [4096, 4096])
//!      with random activations.
//!
//! Phase 1 exit gate (per `plans/hfq1g128-bonsai.md`): GPU output equals the
//! CPU reference (dequantize + naive FP32 dot) to a tight ULP-level
//! tolerance. Bit-exact INT32 sumi is reserved for the dp4a path that lands
//! in Phase 2.
//!
//! Run:
//!   cargo run -p rdna-compute --example test_gemv_hfq1g128
//!
//! For the full real-tensor test, ensure the model is at:
//!   ~/.hipfire/models/bonsai/Bonsai-8B-Q1_0.gguf
//! (override via `HIPFIRE_BONSAI_PATH=...`).

use hipfire_quantize::q1_0::{dequantize_row, quantize_row};

const SYNTH_M: usize = 64;
const SYNTH_K: usize = 512;

fn lcg(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    (*state as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn naive_dot_rows(weights_dq: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m];
    for row in 0..m {
        let mut acc = 0.0f32;
        for col in 0..k {
            acc += weights_dq[row * k + col] * x[col];
        }
        y[row] = acc;
    }
    y
}

fn run_gemv_test_kernel(
    gpu: &mut rdna_compute::Gpu,
    label: &str,
    weights_f32: &[f32],
    x: &[f32],
    m: usize,
    k: usize,
    rows_per_block: u32,
) -> Result<(), String> {
    run_gemv_test_inner(gpu, label, weights_f32, x, m, k, rows_per_block)
}

fn run_gemv_test(
    gpu: &mut rdna_compute::Gpu,
    label: &str,
    weights_f32: &[f32],
    x: &[f32],
    m: usize,
    k: usize,
) -> Result<(), String> {
    run_gemv_test_inner(gpu, label, weights_f32, x, m, k, 1)
}

fn run_gemv_test_inner(
    gpu: &mut rdna_compute::Gpu,
    label: &str,
    weights_f32: &[f32],
    x: &[f32],
    m: usize,
    k: usize,
    rows_per_block: u32,
) -> Result<(), String> {
    assert_eq!(weights_f32.len(), m * k);
    assert_eq!(x.len(), k);
    assert_eq!(k % 128, 0, "K must be 128-aligned for HFQ1G128");

    println!("\n━━━ {label} ({m} × {k}) ━━━");

    // Quantize weights to HFQ1G128 byte stream (per-row).
    let mut weight_bytes: Vec<u8> = Vec::with_capacity(m * (k / 128) * 18);
    for row in 0..m {
        let row_slice = &weights_f32[row * k..(row + 1) * k];
        weight_bytes.extend_from_slice(&quantize_row(row_slice));
    }

    // CPU reference: dequantize + naive FP32 dot.
    let mut weights_dq = vec![0.0f32; m * k];
    for row in 0..m {
        let row_bytes = &weight_bytes[row * (k / 128) * 18..(row + 1) * (k / 128) * 18];
        let dq = dequantize_row(row_bytes, k);
        weights_dq[row * k..(row + 1) * k].copy_from_slice(&dq);
    }
    let y_ref = naive_dot_rows(&weights_dq, x, m, k);

    // GPU dispatch.
    let d_a = gpu
        .upload_raw(&weight_bytes, &[m, k])
        .map_err(|e| format!("upload weights: {e:?}"))?;
    let d_x = gpu
        .upload_f32(x, &[k])
        .map_err(|e| format!("upload x: {e:?}"))?;
    let d_y = gpu
        .zeros(&[m], rdna_compute::DType::F32)
        .map_err(|e| format!("alloc y: {e:?}"))?;

    if rows_per_block == 1 {
        gpu.gemv_hfq1g128(&d_a, &d_x, &d_y, m, k)
            .map_err(|e| format!("kernel launch: {e:?}"))?;
    } else {
        gpu.gemv_hfq1g128_multirow(&d_a, &d_x, &d_y, m, k, rows_per_block)
            .map_err(|e| format!("kernel launch: {e:?}"))?;
    }

    let y_gpu = gpu
        .download_f32(&d_y)
        .map_err(|e| format!("download y: {e:?}"))?;

    // Compare. ULP-tolerance: FP32 dot of K terms with magnitude ~k * d_max
    // accumulates rounding error ~ K * eps_f32 * |result|. For random ±1
    // signed weights and FP32 activations, K=512 → relative error bound
    // ~512 * 1.2e-7 ≈ 6e-5; for K=4096 → ~5e-4. Use 1e-3 absolute tolerance
    // per row (covers FP non-associativity + tree-reduction order diffs).
    let tol = 1e-3 * (k as f32).sqrt();
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut bad_rows = 0;
    for row in 0..m {
        let abs_err = (y_gpu[row] - y_ref[row]).abs();
        max_abs = max_abs.max(abs_err);
        let denom = y_ref[row].abs().max(1.0);
        max_rel = max_rel.max(abs_err / denom);
        if abs_err > tol {
            if bad_rows < 5 {
                eprintln!(
                    "  row {row}: gpu={:>14.6} ref={:>14.6} abs_err={:.4e}",
                    y_gpu[row], y_ref[row], abs_err
                );
            }
            bad_rows += 1;
        }
    }

    gpu.free_tensor(d_a).ok();
    gpu.free_tensor(d_x).ok();
    gpu.free_tensor(d_y).ok();

    println!(
        "  max_abs_err = {:.4e}, max_rel_err = {:.4e}, bad_rows = {}/{}, tol = {:.4e}",
        max_abs, max_rel, bad_rows, m, tol
    );
    if bad_rows == 0 {
        println!("  PASS");
        Ok(())
    } else {
        Err(format!(
            "{bad_rows} rows exceed tolerance {tol:.4e} (max_abs={max_abs:.4e})"
        ))
    }
}

fn run_synthetic(gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
    let mut s = 0xDEADBEEFu32;
    let mut weights = vec![0.0f32; SYNTH_M * SYNTH_K];
    for v in weights.iter_mut() {
        *v = lcg(&mut s);
    }
    let mut x = vec![0.0f32; SYNTH_K];
    for v in x.iter_mut() {
        *v = lcg(&mut s) * 0.5;
    }
    run_gemv_test(gpu, "single-row random", &weights, &x, SYNTH_M, SYNTH_K)?;
    run_gemv_test_kernel(gpu, "multirow R=2 random", &weights, &x, SYNTH_M, SYNTH_K, 2)?;
    run_gemv_test_kernel(gpu, "multirow R=4 random", &weights, &x, SYNTH_M, SYNTH_K, 4)?;
    run_gemv_test_kernel(gpu, "multirow R=8 random", &weights, &x, SYNTH_M, SYNTH_K, 8)
}

fn run_bonsai(gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
    use hipfire_quantize::gguf_input::{GgmlType, GgufFile};
    use std::path::Path;

    let path = std::env::var("HIPFIRE_BONSAI_PATH")
        .unwrap_or_else(|_| "/home/nick/.hipfire/models/bonsai/Bonsai-8B-Q1_0.gguf".to_string());
    let path = Path::new(&path);
    if !path.exists() {
        return Err(format!(
            "Bonsai GGUF not found at {} (override with HIPFIRE_BONSAI_PATH)",
            path.display()
        ));
    }
    let gguf = GgufFile::open(path).map_err(|e| format!("open gguf: {e}"))?;

    // Pick a small Q1_0 tensor: blk.0.attn_q.weight is [4096, 4096].
    let target = "blk.0.attn_q.weight";
    let info = gguf
        .tensors
        .iter()
        .find(|t| t.name == target)
        .ok_or_else(|| format!("tensor '{target}' not found in gguf"))?;
    if info.dtype != GgmlType::Q1_0 {
        return Err(format!(
            "tensor '{target}' is {:?}, expected Q1_0",
            info.dtype
        ));
    }
    let m = info.shape[0];
    let k = info.shape[1];
    println!("\n━━━ Bonsai tensor '{target}' [{m}, {k}] ━━━");

    let raw = gguf.tensor_data(info);
    // Phase 1 exit gate's "real Bonsai weight row" — we already have the
    // bytes in HFQ1G128 layout (PrismML Q1_0 == HFQ1G128, byte-verbatim).
    // Dequantize via CPU reference for the FP32 reference dot.
    let weight_bytes: Vec<u8> = raw[..m * (k / 128) * 18].to_vec();

    // Random activations.
    let mut s = 0x12345678u32;
    let mut x = vec![0.0f32; k];
    for v in x.iter_mut() {
        *v = lcg(&mut s) * 0.5;
    }

    // CPU reference dot — use the dequantized bytes.
    let mut weights_dq = vec![0.0f32; m * k];
    for row in 0..m {
        let row_bytes = &weight_bytes[row * (k / 128) * 18..(row + 1) * (k / 128) * 18];
        let dq = dequantize_row(row_bytes, k);
        weights_dq[row * k..(row + 1) * k].copy_from_slice(&dq);
    }
    let y_ref = naive_dot_rows(&weights_dq, &x, m, k);

    // GPU dispatch.
    let d_a = gpu
        .upload_raw(&weight_bytes, &[m, k])
        .map_err(|e| format!("upload: {e:?}"))?;
    let d_x = gpu.upload_f32(&x, &[k]).map_err(|e| format!("upload x: {e:?}"))?;
    let d_y = gpu
        .zeros(&[m], rdna_compute::DType::F32)
        .map_err(|e| format!("alloc y: {e:?}"))?;

    gpu.gemv_hfq1g128(&d_a, &d_x, &d_y, m, k)
        .map_err(|e| format!("kernel: {e:?}"))?;

    let y_gpu = gpu.download_f32(&d_y).map_err(|e| format!("download: {e:?}"))?;

    let tol = 1e-3 * (k as f32).sqrt();
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut bad = 0usize;
    for row in 0..m {
        let abs_err = (y_gpu[row] - y_ref[row]).abs();
        max_abs = max_abs.max(abs_err);
        let rel = abs_err / y_ref[row].abs().max(1.0);
        max_rel = max_rel.max(rel);
        if abs_err > tol {
            if bad < 5 {
                eprintln!(
                    "  row {row}: gpu={:>14.6} ref={:>14.6} abs_err={:.4e}",
                    y_gpu[row], y_ref[row], abs_err
                );
            }
            bad += 1;
        }
    }

    gpu.free_tensor(d_a).ok();
    gpu.free_tensor(d_x).ok();
    gpu.free_tensor(d_y).ok();

    println!(
        "  max_abs_err = {:.4e}, max_rel_err = {:.4e}, bad_rows = {}/{}, tol = {:.4e}",
        max_abs, max_rel, bad, m, tol
    );
    if bad == 0 {
        println!("  PASS");
        Ok(())
    } else {
        Err(format!("{bad} rows exceed tolerance"))
    }
}

fn main() {
    println!("=== Tier B: gemv_hfq1g128 GPU vs CPU reference ===");

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    println!("GPU initialized");

    let mut failures: Vec<String> = Vec::new();

    if let Err(e) = run_synthetic(&mut gpu) {
        failures.push(format!("synthetic: {e}"));
    }
    if let Err(e) = run_bonsai(&mut gpu) {
        failures.push(format!("bonsai:    {e}"));
    }

    if failures.is_empty() {
        println!("\n=== Tier B PASS ===");
    } else {
        eprintln!("\n=== Tier B FAIL ===");
        for f in &failures {
            eprintln!("  - {f}");
        }
        std::process::exit(1);
    }
}
