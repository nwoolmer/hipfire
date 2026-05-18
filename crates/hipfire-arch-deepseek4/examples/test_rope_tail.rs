//! GPU validation for Phase 4 `rope_tail_halfsplit_f32`.
//!
//! Verifies tail-only RoPE: the first `head_dim - n_rot` dims are
//! untouched; the last `n_rot` dims get the half-split rotation.

use rdna_compute::{DType, Gpu};

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().expect("GPU init");

    const HEAD_DIM: usize = 8;
    const N_HEADS: usize = 1;
    const N_ROT: usize = 4;       // qk_rope-like (tail dim)
    const POS: i32 = 7;
    const FREQ_BASE: f32 = 10000.0;

    // q[h, d] = (h + 1) * 10 + d  (distinct values per element).
    let q_in: Vec<f32> = (0..N_HEADS * HEAD_DIM)
        .map(|i| ((i / HEAD_DIM) + 1) as f32 * 10.0 + (i % HEAD_DIM) as f32)
        .collect();
    let k_in = q_in.clone();
    let mut pos = vec![POS];

    let d_q = gpu.upload_f32(&q_in, &[N_HEADS, HEAD_DIM]).map_err(|e| format!("up q: {e:?}"))?;
    let d_k = gpu.upload_f32(&k_in, &[N_HEADS, HEAD_DIM]).map_err(|e| format!("up k: {e:?}"))?;
    let pos_bytes: &[u8] = unsafe { std::slice::from_raw_parts(pos.as_ptr() as *const u8, 4) };
    let d_pos = gpu.upload_raw(pos_bytes, &[1]).map_err(|e| format!("up pos: {e:?}"))?;
    let _ = (DType::F32, &mut pos);

    gpu.rope_tail_halfsplit(&d_q, &d_k, &d_pos,
        N_HEADS as i32, N_HEADS as i32,
        HEAD_DIM as i32, N_ROT as i32, FREQ_BASE)
        .map_err(|e| format!("dispatch: {e:?}"))?;

    let q_out = gpu.download_f32(&d_q).map_err(|e| format!("d2h q: {e:?}"))?;

    eprintln!("q_in :  {q_in:?}");
    eprintln!("q_out:  {q_out:?}");

    // CPU reference: tail dims rotate, leading dims untouched.
    let tail_off = HEAD_DIM - N_ROT;
    let half = N_ROT / 2;
    let mut expected = q_in.clone();
    for i in 0..half {
        let freq = 1.0f32 / (FREQ_BASE.powf((2 * i) as f32 / N_ROT as f32));
        let ang = POS as f32 * freq;
        let c = ang.cos();
        let s = ang.sin();
        for h in 0..N_HEADS {
            let base = h * HEAD_DIM + tail_off;
            let q0 = q_in[base + i];
            let q1 = q_in[base + i + half];
            expected[base + i]        = q0 * c - q1 * s;
            expected[base + i + half] = q0 * s + q1 * c;
        }
    }
    eprintln!("expect: {expected:?}");

    let tol = 1e-4;
    let mut ok = true;
    for (i, (&g, &e)) in q_out.iter().zip(&expected).enumerate() {
        let dev = (g - e).abs();
        if dev > tol {
            eprintln!("  MISMATCH at {i}: got={g:.6} expected={e:.6} dev={dev:.6}");
            ok = false;
        }
    }
    if ok {
        eprintln!("\nOK: rope_tail_halfsplit matches CPU reference (tol={tol})");
        // Sanity: leading dims unchanged.
        for d in 0..tail_off {
            assert!((q_out[d] - q_in[d]).abs() < 1e-6,
                "leading dim {d} changed: {} vs {}", q_out[d], q_in[d]);
        }
        eprintln!("Leading dims (0..{tail_off}) preserved verbatim.");
        Ok(())
    } else {
        Err("rope_tail mismatch".into())
    }
}
