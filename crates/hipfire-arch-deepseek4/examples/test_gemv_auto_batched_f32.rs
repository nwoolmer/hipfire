//! Validate `gemm_f32_batched`'s row-major output layout (Phase B2, 2026-05-18).
//!
//! Critical pre-flight check for V4F's `gemv_auto_batched` F32 dispatch arm:
//! confirms `gemm_f32_batched(X_batch, W, Y_batch, batch, k, m)` produces
//! `Y_batch[b, :]` byte-equal to `gemv_f32(W, X_batch[b, :], y_row)` for
//! every b. Validates argument ordering (input is the M_kernel dim, weight
//! is the N_kernel dim) and output row-majorness.

use rdna_compute::{DType, Gpu};

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().expect("GPU init");

    const M: usize = 64;   // output dim
    const K: usize = 128;  // input dim
    let b_sizes = [1usize, 4, 32];

    let seed = |off: usize, len: usize| -> Vec<f32> {
        (0..len).map(|i| {
            let x = (off + i) as f32 * 0.011;
            (x.sin() * 0.6) + (x * 0.31).cos() * 0.4
        }).collect()
    };

    // Single shared weight matrix [M, K], row-major. gemv_f32 and
    // gemm_f32_batched both consume the same layout.
    let w = seed(7, M * K);
    let d_w = gpu.upload_f32(&w, &[M, K]).map_err(|e| format!("w: {e:?}"))?;

    for &b in &b_sizes {
        eprintln!("--- batch_size = {b} ---");
        let mut x_all = vec![0.0f32; b * K];
        for row in 0..b {
            let x_row = seed(row * 1000 + 100, K);
            x_all[row * K..][..K].copy_from_slice(&x_row);
        }

        // Sequential baseline.
        let mut seq_out = vec![0.0f32; b * M];
        for row in 0..b {
            let x_slice = &x_all[row * K..][..K];
            let d_x = gpu.upload_f32(x_slice, &[K]).map_err(|e| format!("x: {e:?}"))?;
            let d_y = gpu.zeros(&[M], DType::F32).map_err(|e| format!("y: {e:?}"))?;
            gpu.gemv_f32(&d_w, &d_x, &d_y).map_err(|e| format!("gemv: {e:?}"))?;
            let r = gpu.download_f32(&d_y).map_err(|e| format!("d2h: {e:?}"))?;
            seq_out[row * M..][..M].copy_from_slice(&r);
            gpu.free_tensor(d_x).ok();
            gpu.free_tensor(d_y).ok();
        }

        // Batched.
        let d_x_b = gpu.upload_f32(&x_all, &[b, K]).map_err(|e| format!("x_b: {e:?}"))?;
        let d_y_b = gpu.zeros(&[b, M], DType::F32).map_err(|e| format!("y_b: {e:?}"))?;
        gpu.gemm_f32_batched(&d_x_b, &d_w, &d_y_b, b, K, M)
            .map_err(|e| format!("gemm_f32_batched: {e:?}"))?;
        let bo = gpu.download_f32(&d_y_b).map_err(|e| format!("d2h: {e:?}"))?;
        gpu.free_tensor(d_x_b).ok();
        gpu.free_tensor(d_y_b).ok();

        // Row-by-row compare.
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for i in 0..b * M {
            let d = (seq_out[i] - bo[i]).abs();
            let denom = seq_out[i].abs().max(bo[i].abs()).max(1e-8);
            let rel = d / denom;
            if d > max_abs { max_abs = d; }
            if rel > max_rel { max_rel = rel; }
        }
        eprintln!("  B={b}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
        // FMA reordering between GEMV's single-thread loop and GEMM's
        // shuffle-reduce can produce small ε differences. Accept ≤ 1e-4 abs.
        if max_abs > 1e-4 {
            return Err(format!("B={b}: row-major mismatch (max_abs={max_abs:.3e})"));
        }
    }

    gpu.free_tensor(d_w).ok();
    eprintln!("\nOK: gemm_f32_batched(X[b,K], W[M,K], Y[b,M], b, K, M) matches gemv_f32 per-row");
    Ok(())
}
