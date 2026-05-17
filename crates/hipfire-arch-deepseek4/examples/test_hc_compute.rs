//! GPU validation for Phase 3 `hc_compute_control`.
//!
//! Verifies the small GEMV that produces the HC control vector
//! from concatenated 4-stream residual.

use rdna_compute::Gpu;

fn f32_to_f16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bits = v.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let frac = (bits >> 13) & 0x3FF;
        let h: u16 = if v == 0.0 { 0 }
            else if exp <= 0 { (sign << 15) as u16 }
            else if exp >= 31 { ((sign << 15) | (0x1F << 10)) as u16 }
            else { ((sign << 15) | ((exp as u32) << 10) | frac) as u16 };
        out.push((h & 0xFF) as u8);
        out.push((h >> 8) as u8);
    }
    out
}

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().expect("GPU init");

    const N_CTRL: usize = 4;
    const X_DIM: usize = 256;

    // x_flat[d] = 0.01
    let x_flat: Vec<f32> = vec![0.01; X_DIM];
    // W_fn[c, d] = 0.001 * (c + 1)
    let mut w_fn = vec![0.0f32; N_CTRL * X_DIM];
    for c in 0..N_CTRL {
        for d in 0..X_DIM {
            w_fn[c * X_DIM + d] = 0.001 * (c + 1) as f32;
        }
    }
    // base[c] = 0.5 * (c + 1)
    let base: Vec<f32> = (0..N_CTRL).map(|c| 0.5 * (c + 1) as f32).collect();

    // x_flat is F32 (V4F residual convention); w_fn and base are F16
    // (HFQ-converted F16 storage).
    let d_x  = gpu.upload_f32(&x_flat, &[X_DIM])
        .map_err(|e| format!("up x: {e:?}"))?;
    let d_w  = gpu.upload_raw(&f32_to_f16_bytes(&w_fn), &[N_CTRL, X_DIM])
        .map_err(|e| format!("up w: {e:?}"))?;
    let d_b  = gpu.upload_raw(&f32_to_f16_bytes(&base), &[N_CTRL])
        .map_err(|e| format!("up b: {e:?}"))?;
    let d_c  = gpu.zeros(&[N_CTRL], rdna_compute::DType::F32)
        .map_err(|e| format!("z c: {e:?}"))?;

    gpu.hc_compute_control(&d_x, &d_w, &d_b, &d_c, N_CTRL as i32, X_DIM as i32)
        .map_err(|e| format!("dispatch: {e:?}"))?;

    let c_out = gpu.download_f32(&d_c).map_err(|e| format!("d2h: {e:?}"))?;

    // V4F-faithful: c = (X·W) * rsqrt(mean(X^2) + eps) + base
    //   X·W summed = X_DIM * 0.01 * 0.001 * (c+1) = 0.00256 * (c+1)
    //   mean(X^2) = 0.01^2 = 1e-4
    //   rsqrt = 1 / sqrt(1e-4 + 1e-6) ≈ 1 / 0.01005 ≈ 99.504
    //   c[c] = 0.00256*(c+1) * 99.504 + 0.5*(c+1) ≈ 0.2547*(c+1) + 0.5*(c+1)
    let mean_sq = 0.01f32 * 0.01f32;
    let rsqrt = 1.0 / (mean_sq + 1e-6f32).sqrt();
    let expected: Vec<f32> = (0..N_CTRL).map(|c| {
        let f = (c + 1) as f32;
        X_DIM as f32 * 0.01 * 0.001 * f * rsqrt + 0.5 * f
    }).collect();

    eprintln!("c_out:    {c_out:?}");
    eprintln!("expected: {expected:?}");

    let tol = 1e-2;
    let mut ok = true;
    for (i, (&g, &e)) in c_out.iter().zip(&expected).enumerate() {
        if (g - e).abs() > tol {
            eprintln!("  MISMATCH {i}: got={g:.4} exp={e:.4}");
            ok = false;
        }
    }
    if ok {
        eprintln!("\nOK: hc_compute_control matches CPU reference (tol={tol})");
        Ok(())
    } else {
        Err("compute_control mismatch".into())
    }
}
