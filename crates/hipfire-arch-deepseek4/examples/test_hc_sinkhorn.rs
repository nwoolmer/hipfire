//! Exercise the V4F Phase 3 stub kernel `hc_sinkhorn_4x4` end-to-end on
//! a real GPU and verify the output is doubly-stochastic.
//!
//! Sinkhorn iterates row+column normalisation. After enough iterations
//! every row and column should sum to 1 (within `eps`).
//!
//! Usage:
//!   cargo run --release --example test_hc_sinkhorn -p hipfire-arch-deepseek4
//!
//! Run on the same GPU pool as the rest of hipfire (`gpu-lock.sh`).

use rdna_compute::{DType, Gpu};

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().expect("GPU init");

    // Start with an arbitrary positive 4x4 matrix.
    let m_in: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        2.0, 1.0, 5.0, 1.0,
        3.0, 4.0, 1.0, 2.0,
        1.0, 5.0, 2.0, 3.0,
    ];
    eprintln!("input matrix:");
    for r in 0..4 {
        eprintln!("  {:.3} {:.3} {:.3} {:.3}",
            m_in[r * 4], m_in[r * 4 + 1], m_in[r * 4 + 2], m_in[r * 4 + 3]);
    }

    let d_m = gpu.upload_f32(&m_in, &[4, 4]).map_err(|e| format!("upload: {e:?}"))?;
    let _ = DType::F32;  // sanity import use

    let eps = 1e-6f32;
    let iters = 20i32;
    gpu.hc_sinkhorn_4x4(&d_m, eps, iters).map_err(|e| format!("dispatch: {e:?}"))?;

    let m_out: Vec<f32> = gpu.download_f32(&d_m).map_err(|e| format!("download: {e:?}"))?;
    eprintln!("\nafter {iters} Sinkhorn iters (eps={eps}):");
    for r in 0..4 {
        eprintln!("  {:.5} {:.5} {:.5} {:.5}",
            m_out[r * 4], m_out[r * 4 + 1], m_out[r * 4 + 2], m_out[r * 4 + 3]);
    }

    // Verify row sums ≈ 1 and column sums ≈ 1.
    let tol = 1e-3;
    let mut row_ok = true;
    let mut col_ok = true;
    for r in 0..4 {
        let s: f32 = (0..4).map(|c| m_out[r * 4 + c]).sum();
        let dev = (s - 1.0).abs();
        eprintln!("  row {r} sum = {s:.6} (dev {dev:.6})");
        if dev > tol { row_ok = false; }
    }
    for c in 0..4 {
        let s: f32 = (0..4).map(|r| m_out[r * 4 + c]).sum();
        let dev = (s - 1.0).abs();
        eprintln!("  col {c} sum = {s:.6} (dev {dev:.6})");
        if dev > tol { col_ok = false; }
    }

    if row_ok && col_ok {
        eprintln!("\nOK: doubly-stochastic within tol={tol}");
        Ok(())
    } else {
        Err(format!("doubly-stochastic check FAILED (row_ok={row_ok}, col_ok={col_ok})"))
    }
}
