//! GPU validation for Phase 2 `indexer_compressed_k_score`.
//!
//! Per-head dot products of Q against the compressed-K cache.

use rdna_compute::Gpu;

fn f32_to_f16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        // crude f32→f16 (bias=15, no NaN/Inf handling — fine for test).
        let bits = v.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let frac = (bits >> 13) & 0x3FF;
        let h: u16 = if v == 0.0 {
            0
        } else if exp <= 0 {
            ((sign << 15) as u16)
        } else if exp >= 31 {
            ((sign << 15) | (0x1F << 10)) as u16
        } else {
            ((sign << 15) | ((exp as u32) << 10) | frac) as u16
        };
        out.push((h & 0xFF) as u8);
        out.push((h >> 8) as u8);
    }
    out
}

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().expect("GPU init");

    const H: usize = 2;   // n_idx_heads
    const D: usize = 8;   // idx_head_dim
    const N: usize = 4;   // n_compressed

    // Q[h, d] = h + 0.1 * d
    let q: Vec<f32> = (0..H * D).map(|i| (i / D) as f32 + 0.1 * (i % D) as f32).collect();
    // K[h, d, n] = 0.01 * (h + d + n)  →  small simple values
    let mut k = vec![0.0f32; H * D * N];
    for h in 0..H {
        for d in 0..D {
            for n in 0..N {
                k[(h * D + d) * N + n] = 0.01 * ((h + d + n) as f32);
            }
        }
    }

    let d_q = gpu.upload_raw(&f32_to_f16_bytes(&q), &[H, D]).map_err(|e| format!("up q: {e:?}"))?;
    let d_k = gpu.upload_raw(&f32_to_f16_bytes(&k), &[H, D, N]).map_err(|e| format!("up k: {e:?}"))?;
    let d_s = gpu.zeros(&[H, N], rdna_compute::DType::F32).map_err(|e| format!("z s: {e:?}"))?;

    gpu.indexer_compressed_k_score(&d_q, &d_k, &d_s,
        H as i32, D as i32, N as i32)
        .map_err(|e| format!("dispatch: {e:?}"))?;

    let scores = gpu.download_f32(&d_s).map_err(|e| format!("d2h: {e:?}"))?;

    // CPU reference: scores[h, n] = sum_d (q[h,d] * k[h,d,n]).
    let mut expected = vec![0.0f32; H * N];
    for h in 0..H {
        for n in 0..N {
            let mut acc = 0.0f32;
            for d in 0..D {
                acc += q[h * D + d] * k[(h * D + d) * N + n];
            }
            expected[h * N + n] = acc;
        }
    }

    eprintln!("scores:   {scores:?}");
    eprintln!("expected: {expected:?}");
    let tol = 5e-3;  // fp16 accumulation tolerance
    let mut ok = true;
    for (i, (&g, &e)) in scores.iter().zip(&expected).enumerate() {
        let dev = (g - e).abs();
        if dev > tol {
            eprintln!("  MISMATCH {i}: got={g:.4} expected={e:.4} dev={dev:.4}");
            ok = false;
        }
    }
    if ok {
        eprintln!("\nOK: indexer_compressed_k_score matches CPU reference (tol={tol})");
        Ok(())
    } else {
        Err("score mismatch".into())
    }
}
