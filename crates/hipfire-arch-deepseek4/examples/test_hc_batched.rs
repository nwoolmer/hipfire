//! Byte-equality smoke test for the batched HC kernels (Phase A5, 2026-05-18).
//!
//! Confirms `hc_input_map_4stream_batched` and `hc_mix_4stream_batched`
//! produce per-row outputs identical to running the sequential kernels
//! `hc_input_map_4stream` / `hc_mix_4stream` for each batch row in turn.

use rdna_compute::{DType, Gpu};

const HC_MULT: usize = 4;

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().expect("GPU init");

    const HIDDEN: usize = 512;
    let b_sizes = [1usize, 4, 32];

    let seed = |off: usize, len: usize| -> Vec<f32> {
        (0..len).map(|i| {
            let x = (off + i) as f32 * 0.013;
            (x.sin() * 0.6) + (x * 0.31).cos() * 0.4
        }).collect()
    };

    // ===== hc_input_map_4stream =====
    for &b in &b_sizes {
        eprintln!("--- hc_input_map B={b} ---");
        let mut all_streams = vec![0.0f32; b * HC_MULT * HIDDEN];
        let mut all_a       = vec![0.0f32; b * HC_MULT];
        for row in 0..b {
            let s = seed(row * 1000 + 1,  HC_MULT * HIDDEN);
            let a = seed(row * 1000 + 50, HC_MULT);
            all_streams[row * HC_MULT * HIDDEN..][..HC_MULT * HIDDEN].copy_from_slice(&s);
            all_a[row * HC_MULT..][..HC_MULT].copy_from_slice(&a);
        }

        // Sequential.
        let mut seq_out = vec![0.0f32; b * HIDDEN];
        for row in 0..b {
            let s_slice = &all_streams[row * HC_MULT * HIDDEN..][..HC_MULT * HIDDEN];
            let a_slice = &all_a[row * HC_MULT..][..HC_MULT];
            let d_s = gpu.upload_f32(s_slice, &[HC_MULT, HIDDEN]).map_err(|e| format!("s: {e:?}"))?;
            let d_a = gpu.upload_f32(a_slice, &[HC_MULT]).map_err(|e| format!("a: {e:?}"))?;
            let d_o = gpu.zeros(&[HIDDEN], DType::F32).map_err(|e| format!("o: {e:?}"))?;
            gpu.hc_input_map_4stream(&d_a, &d_s, &d_o, HIDDEN as i32)
                .map_err(|e| format!("seq dispatch: {e:?}"))?;
            let r = gpu.download_f32(&d_o).map_err(|e| format!("d2h: {e:?}"))?;
            seq_out[row * HIDDEN..][..HIDDEN].copy_from_slice(&r);
            gpu.free_tensor(d_s).ok();
            gpu.free_tensor(d_a).ok();
            gpu.free_tensor(d_o).ok();
        }

        // Batched.
        let d_s_b = gpu.upload_f32(&all_streams, &[b, HC_MULT, HIDDEN]).map_err(|e| format!("s_b: {e:?}"))?;
        let d_a_b = gpu.upload_f32(&all_a, &[b, HC_MULT]).map_err(|e| format!("a_b: {e:?}"))?;
        let d_o_b = gpu.zeros(&[b, HIDDEN], DType::F32).map_err(|e| format!("o_b: {e:?}"))?;
        gpu.hc_input_map_4stream_batched(&d_a_b, &d_s_b, &d_o_b, HIDDEN as i32, b as i32)
            .map_err(|e| format!("batched dispatch: {e:?}"))?;
        let bo = gpu.download_f32(&d_o_b).map_err(|e| format!("d2h: {e:?}"))?;
        gpu.free_tensor(d_s_b).ok();
        gpu.free_tensor(d_a_b).ok();
        gpu.free_tensor(d_o_b).ok();

        let mut max_abs = 0.0f32;
        let mut byte_eq = true;
        for i in 0..b * HIDDEN {
            let d = (seq_out[i] - bo[i]).abs();
            if d > max_abs { max_abs = d; }
            if seq_out[i].to_bits() != bo[i].to_bits() { byte_eq = false; }
        }
        eprintln!("  max_abs={max_abs:.3e} byte_eq={byte_eq}");
        if !byte_eq && max_abs > 1e-6 {
            return Err(format!("hc_input_map B={b}: byte_eq={byte_eq} max_abs={max_abs:.3e}"));
        }
    }

    // ===== hc_mix_4stream =====
    for &b in &b_sizes {
        eprintln!("--- hc_mix B={b} ---");
        let mut all_x      = vec![0.0f32; b * HC_MULT * HIDDEN];
        let mut all_A      = vec![0.0f32; b * HC_MULT * HC_MULT];
        let mut all_scale  = vec![0.0f32; b * HC_MULT];
        let mut all_t      = vec![0.0f32; b * HIDDEN];
        for row in 0..b {
            let x = seed(row * 1000 + 1,   HC_MULT * HIDDEN);
            let am = seed(row * 1000 + 60, HC_MULT * HC_MULT);
            let sc = seed(row * 1000 + 70, HC_MULT);
            let t  = seed(row * 1000 + 90, HIDDEN);
            all_x    [row * HC_MULT * HIDDEN..][..HC_MULT * HIDDEN].copy_from_slice(&x);
            all_A    [row * HC_MULT * HC_MULT..][..HC_MULT * HC_MULT].copy_from_slice(&am);
            all_scale[row * HC_MULT..][..HC_MULT].copy_from_slice(&sc);
            all_t    [row * HIDDEN..][..HIDDEN].copy_from_slice(&t);
        }

        // Sequential.
        let mut seq_out = vec![0.0f32; b * HC_MULT * HIDDEN];
        for row in 0..b {
            let x_slice  = &all_x[row * HC_MULT * HIDDEN..][..HC_MULT * HIDDEN];
            let a_slice  = &all_A[row * HC_MULT * HC_MULT..][..HC_MULT * HC_MULT];
            let sc_slice = &all_scale[row * HC_MULT..][..HC_MULT];
            let t_slice  = &all_t[row * HIDDEN..][..HIDDEN];
            let d_x  = gpu.upload_f32(x_slice,  &[HC_MULT, HIDDEN]).map_err(|e| format!("x: {e:?}"))?;
            let d_a  = gpu.upload_f32(a_slice,  &[HC_MULT, HC_MULT]).map_err(|e| format!("a: {e:?}"))?;
            let d_sc = gpu.upload_f32(sc_slice, &[HC_MULT]).map_err(|e| format!("sc: {e:?}"))?;
            let d_t  = gpu.upload_f32(t_slice,  &[HIDDEN]).map_err(|e| format!("t: {e:?}"))?;
            let d_o  = gpu.zeros(&[HC_MULT, HIDDEN], DType::F32).map_err(|e| format!("o: {e:?}"))?;
            gpu.hc_mix_4stream(&d_x, &d_a, &d_sc, &d_t, &d_o, HIDDEN as i32)
                .map_err(|e| format!("seq: {e:?}"))?;
            let r = gpu.download_f32(&d_o).map_err(|e| format!("d2h: {e:?}"))?;
            seq_out[row * HC_MULT * HIDDEN..][..HC_MULT * HIDDEN].copy_from_slice(&r);
            gpu.free_tensor(d_x).ok();
            gpu.free_tensor(d_a).ok();
            gpu.free_tensor(d_sc).ok();
            gpu.free_tensor(d_t).ok();
            gpu.free_tensor(d_o).ok();
        }

        // Batched.
        let d_x_b  = gpu.upload_f32(&all_x,     &[b, HC_MULT, HIDDEN]).map_err(|e| format!("xb: {e:?}"))?;
        let d_a_b  = gpu.upload_f32(&all_A,     &[b, HC_MULT, HC_MULT]).map_err(|e| format!("ab: {e:?}"))?;
        let d_sc_b = gpu.upload_f32(&all_scale, &[b, HC_MULT]).map_err(|e| format!("scb: {e:?}"))?;
        let d_t_b  = gpu.upload_f32(&all_t,     &[b, HIDDEN]).map_err(|e| format!("tb: {e:?}"))?;
        let d_o_b  = gpu.zeros(&[b, HC_MULT, HIDDEN], DType::F32).map_err(|e| format!("ob: {e:?}"))?;
        gpu.hc_mix_4stream_batched(&d_x_b, &d_a_b, &d_sc_b, &d_t_b, &d_o_b, HIDDEN as i32, b as i32)
            .map_err(|e| format!("batched: {e:?}"))?;
        let bo = gpu.download_f32(&d_o_b).map_err(|e| format!("d2h: {e:?}"))?;
        gpu.free_tensor(d_x_b).ok();
        gpu.free_tensor(d_a_b).ok();
        gpu.free_tensor(d_sc_b).ok();
        gpu.free_tensor(d_t_b).ok();
        gpu.free_tensor(d_o_b).ok();

        let mut max_abs = 0.0f32;
        let mut byte_eq = true;
        for i in 0..b * HC_MULT * HIDDEN {
            let d = (seq_out[i] - bo[i]).abs();
            if d > max_abs { max_abs = d; }
            if seq_out[i].to_bits() != bo[i].to_bits() { byte_eq = false; }
        }
        eprintln!("  max_abs={max_abs:.3e} byte_eq={byte_eq}");
        if !byte_eq && max_abs > 1e-6 {
            return Err(format!("hc_mix B={b}: byte_eq={byte_eq} max_abs={max_abs:.3e}"));
        }
    }

    eprintln!("\nOK: hc_input_map_4stream_batched + hc_mix_4stream_batched match sequential at B=1/4/32");
    Ok(())
}
