//! GPU validation for Phase 3 `hc_mix_4stream`.
//!
//! Verifies: `x_out[s, d] = sum_t(A[s, t] * x_in[t, d]) + scale[s] * transform_out[d]`.
//!
//! With identity `A` and zero `scale`, output should equal input.
//! With identity `A` and unit `scale`, output should equal input + transform_out.

use rdna_compute::Gpu;

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;
    if exp == 0xFF {
        let f16_frac = if frac == 0 { 0 } else { (frac >> 13) | 1 };
        return ((sign << 15) | (0x1F << 10) | f16_frac) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 { return ((sign << 15) | (0x1F << 10)) as u16; }
    if new_exp <= 0 {
        if new_exp < -10 { return (sign << 15) as u16; }
        let f = frac | 0x800000;
        let shift = (1 - new_exp + 13) as u32;
        return ((sign << 15) | (f >> shift)) as u16;
    }
    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 { return f32::from_bits(sign << 31); }
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 { f <<= 1; e -= 1; }
        f &= 0x3FF;
        let exp32 = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13));
    }
    if exp == 31 {
        let frac32 = if frac == 0 { 0 } else { frac << 13 | 1 };
        return f32::from_bits((sign << 31) | (0xFF << 23) | frac32);
    }
    f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
}

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().expect("GPU init");
    const HIDDEN: usize = 4096;

    // Identity A (4×4) → x_out = x_in (when scale = 0).
    // Use scale = [1, 1, 1, 1] and transform_out = small constant, check addition.
    let a: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    let scale: Vec<f32> = vec![1.0, 0.5, 2.0, 0.25];

    // x_in[s, d] = (s + 1) * 0.1 for all d (constant per stream).
    let mut x_in_f16 = vec![0u8; 4 * HIDDEN * 2];
    for s in 0..4 {
        let v = ((s + 1) as f32) * 0.1;
        let h = f32_to_f16_bits(v);
        for d in 0..HIDDEN {
            let off = (s * HIDDEN + d) * 2;
            x_in_f16[off] = (h & 0xFF) as u8;
            x_in_f16[off + 1] = (h >> 8) as u8;
        }
    }

    // transform_out[d] = 0.05 for all d.
    let mut t_out_f16 = vec![0u8; HIDDEN * 2];
    {
        let h = f32_to_f16_bits(0.05);
        for d in 0..HIDDEN {
            t_out_f16[d * 2] = (h & 0xFF) as u8;
            t_out_f16[d * 2 + 1] = (h >> 8) as u8;
        }
    }

    let d_xin = gpu.upload_raw(&x_in_f16, &[4, HIDDEN]).map_err(|e| format!("up xin: {e:?}"))?;
    let d_a   = gpu.upload_f32(&a, &[4, 4]).map_err(|e| format!("up a: {e:?}"))?;
    let d_sc  = gpu.upload_f32(&scale, &[4]).map_err(|e| format!("up scale: {e:?}"))?;
    let d_to  = gpu.upload_raw(&t_out_f16, &[HIDDEN]).map_err(|e| format!("up to: {e:?}"))?;
    let d_xout = gpu.zeros(&[4, HIDDEN], rdna_compute::DType::F16)
        .map_err(|e| format!("zeros xout: {e:?}"))?;

    gpu.hc_mix_4stream(&d_xin, &d_a, &d_sc, &d_to, &d_xout, HIDDEN as i32)
        .map_err(|e| format!("dispatch: {e:?}"))?;

    // Download as raw bytes via direct memcpy (no f16-typed helper exists).
    let mut xout_bytes = vec![0u8; 4 * HIDDEN * 2];
    gpu.hip.memcpy_dtoh(&mut xout_bytes, &d_xout.buf)
        .map_err(|e| format!("d2h: {e:?}"))?;
    let mut xout = vec![0.0f32; 4 * HIDDEN];
    for i in 0..4 * HIDDEN {
        let bits = u16::from_le_bytes([xout_bytes[i * 2], xout_bytes[i * 2 + 1]]);
        xout[i] = f16_bits_to_f32(bits);
    }

    // Expected: with identity A:
    //   x_out[s, d] = 1 * x_in[s, d] + scale[s] * transform_out[d]
    //   x_in[s, d]  = (s+1) * 0.1
    //   transform_out[d] = 0.05
    //   → x_out[s, d] = (s+1)*0.1 + scale[s]*0.05
    let expected: Vec<f32> = (0..4)
        .map(|s| (s as f32 + 1.0) * 0.1 + scale[s] * 0.05)
        .collect();

    let tol = 1e-3;
    let mut ok = true;
    for s in 0..4 {
        let v = xout[s * HIDDEN + 0];  // sample any dim, all are equal
        let dev = (v - expected[s]).abs();
        eprintln!("  stream {s}: expected={:.4} got={:.4} dev={:.6}", expected[s], v, dev);
        if dev > tol { ok = false; }
    }
    if ok {
        eprintln!("\nOK: identity-A + per-stream scale check passes within tol={tol}");
        Ok(())
    } else {
        Err(format!("hc_mix_4stream FAILED"))
    }
}
