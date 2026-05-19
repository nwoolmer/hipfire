//! Byte-equality smoke test for `v4f_attn_swa_topk_batched_f32` (Phase A1).
//!
//! Generates B independent (Q, swa_k, swa_v, topk_k, topk_v) inputs, runs
//! them once each through the SEQUENTIAL kernel (`v4f_attn_swa_topk_f32`),
//! packs them into the batched [B, head_dim, window] layout, runs the
//! BATCHED kernel once, and asserts every row matches the corresponding
//! sequential output within FMA-order ε.
//!
//! At B=1 the math is byte-identical (identical buffer layout & FMA order).
//! At B>1 each block computes its row independently with the same math, so
//! row-wise byte-equality should hold modulo HIP launch-order nondeterminism
//! in floating point (none expected, since each block writes its own slot).

use rdna_compute::{DType, Gpu};

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().expect("GPU init");

    // Small but representative shapes.
    const N_HEADS: usize = 4;
    const HEAD_DIM: usize = 128;
    const SWA_WIN: usize = 64;
    const TOPK_WIN: usize = 96;
    // Per-row valid counts (each row uses a different window slice).
    // Length 32 — used at all B up to 32.
    let n_valid_swa_per_row: [i32; 32] = [
        12, 33, 64, 50, 5,  7,  20, 41, 60, 1,  64, 64, 11, 25, 8,  44,
        17, 39, 55, 22, 64, 30, 14, 9,  3,  48, 36, 19, 28, 51, 6,  57,
    ];
    let n_active_topk_per_row: [i32; 32] = [
        8,  16, 64, 96, 0,  12, 32, 80, 48, 96, 96, 1,  64, 20, 56, 24,
        88, 4,  72, 40, 96, 28, 60, 16, 84, 52, 36, 92, 8,  68, 44, 76,
    ];
    let b_sizes = [1usize, 4, 32]; // B=1 byte-eq, B=4 + B=32 full coverage

    // Build a deterministic but non-trivial test input.
    let seed_input = |off: usize, len: usize| -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = (off + i) as f32 * 0.013;
                (x.sin() * 0.7) + (x * 0.4).cos() * 0.3
            })
            .collect()
    };

    for &b in &b_sizes {
        eprintln!("--- batch_size = {b} ---");
        // Per-row inputs.
        let mut q_full        = vec![0.0f32; b * N_HEADS * HEAD_DIM];
        let mut swa_k_full    = vec![0.0f32; b * HEAD_DIM * SWA_WIN];
        let mut swa_v_full    = vec![0.0f32; b * HEAD_DIM * SWA_WIN];
        let mut topk_k_full   = vec![0.0f32; b * HEAD_DIM * TOPK_WIN];
        let mut topk_v_full   = vec![0.0f32; b * HEAD_DIM * TOPK_WIN];

        for row in 0..b {
            let q_row = seed_input(row * 1000 + 1,    N_HEADS * HEAD_DIM);
            let kk    = seed_input(row * 1000 + 100,  HEAD_DIM * SWA_WIN);
            let vv    = seed_input(row * 1000 + 200,  HEAD_DIM * SWA_WIN);
            let tkk   = seed_input(row * 1000 + 300,  HEAD_DIM * TOPK_WIN);
            let tvv   = seed_input(row * 1000 + 400,  HEAD_DIM * TOPK_WIN);
            q_full     [row * N_HEADS * HEAD_DIM..][..N_HEADS * HEAD_DIM].copy_from_slice(&q_row);
            swa_k_full [row * HEAD_DIM * SWA_WIN..][..HEAD_DIM * SWA_WIN].copy_from_slice(&kk);
            swa_v_full [row * HEAD_DIM * SWA_WIN..][..HEAD_DIM * SWA_WIN].copy_from_slice(&vv);
            topk_k_full[row * HEAD_DIM * TOPK_WIN..][..HEAD_DIM * TOPK_WIN].copy_from_slice(&tkk);
            topk_v_full[row * HEAD_DIM * TOPK_WIN..][..HEAD_DIM * TOPK_WIN].copy_from_slice(&tvv);
        }
        // attn_sink shared across batch (per-head).
        let attn_sink_data: Vec<f32> = (0..N_HEADS).map(|h| 0.1 * (h as f32 + 1.0)).collect();

        // ---- 1. SEQUENTIAL baseline: run kernel B times with row-slice inputs ----
        let mut seq_out = vec![0.0f32; b * N_HEADS * HEAD_DIM];
        let d_sink = gpu.upload_f32(&attn_sink_data, &[N_HEADS])
            .map_err(|e| format!("sink up: {e:?}"))?;
        for row in 0..b {
            let q_slice = &q_full[row * N_HEADS * HEAD_DIM..][..N_HEADS * HEAD_DIM];
            let k_slice = &swa_k_full[row * HEAD_DIM * SWA_WIN..][..HEAD_DIM * SWA_WIN];
            let v_slice = &swa_v_full[row * HEAD_DIM * SWA_WIN..][..HEAD_DIM * SWA_WIN];
            let tk_slice = &topk_k_full[row * HEAD_DIM * TOPK_WIN..][..HEAD_DIM * TOPK_WIN];
            let tv_slice = &topk_v_full[row * HEAD_DIM * TOPK_WIN..][..HEAD_DIM * TOPK_WIN];

            let d_q  = gpu.upload_f32(q_slice,  &[N_HEADS, HEAD_DIM]).map_err(|e| format!("q up: {e:?}"))?;
            let d_k  = gpu.upload_f32(k_slice,  &[1, HEAD_DIM, SWA_WIN]).map_err(|e| format!("k up: {e:?}"))?;
            let d_v  = gpu.upload_f32(v_slice,  &[1, HEAD_DIM, SWA_WIN]).map_err(|e| format!("v up: {e:?}"))?;
            let d_tk = gpu.upload_f32(tk_slice, &[1, HEAD_DIM, TOPK_WIN]).map_err(|e| format!("tk up: {e:?}"))?;
            let d_tv = gpu.upload_f32(tv_slice, &[1, HEAD_DIM, TOPK_WIN]).map_err(|e| format!("tv up: {e:?}"))?;
            let d_out = gpu.zeros(&[N_HEADS, HEAD_DIM], DType::F32).map_err(|e| format!("out zeros: {e:?}"))?;

            gpu.v4f_attn_swa_topk_f32(
                &d_q, &d_k, &d_v, &d_tk, &d_tv, &d_sink, &d_out,
                N_HEADS as i32, HEAD_DIM as i32,
                SWA_WIN as i32, TOPK_WIN as i32,
                n_valid_swa_per_row[row], n_active_topk_per_row[row],
            ).map_err(|e| format!("seq dispatch: {e:?}"))?;

            let row_out = gpu.download_f32(&d_out).map_err(|e| format!("seq d2h: {e:?}"))?;
            seq_out[row * N_HEADS * HEAD_DIM..][..N_HEADS * HEAD_DIM].copy_from_slice(&row_out);

            gpu.free_tensor(d_q).ok();
            gpu.free_tensor(d_k).ok();
            gpu.free_tensor(d_v).ok();
            gpu.free_tensor(d_tk).ok();
            gpu.free_tensor(d_tv).ok();
            gpu.free_tensor(d_out).ok();
        }

        // ---- 2. BATCHED: pack per-row inputs into [B, head_dim, window] layout ----
        let d_q_b   = gpu.upload_f32(&q_full,      &[b, N_HEADS, HEAD_DIM]).map_err(|e| format!("Q b up: {e:?}"))?;
        let d_sk_b  = gpu.upload_f32(&swa_k_full,  &[b, HEAD_DIM, SWA_WIN]).map_err(|e| format!("swa_k b up: {e:?}"))?;
        let d_sv_b  = gpu.upload_f32(&swa_v_full,  &[b, HEAD_DIM, SWA_WIN]).map_err(|e| format!("swa_v b up: {e:?}"))?;
        let d_tk_b  = gpu.upload_f32(&topk_k_full, &[b, HEAD_DIM, TOPK_WIN]).map_err(|e| format!("topk_k b up: {e:?}"))?;
        let d_tv_b  = gpu.upload_f32(&topk_v_full, &[b, HEAD_DIM, TOPK_WIN]).map_err(|e| format!("topk_v b up: {e:?}"))?;
        let d_out_b = gpu.zeros(&[b, N_HEADS, HEAD_DIM], DType::F32).map_err(|e| format!("out_b zeros: {e:?}"))?;

        // Per-row valid counts as i32 byte arrays.
        let nv_bytes: Vec<u8> = n_valid_swa_per_row[..b].iter().flat_map(|x| x.to_le_bytes()).collect();
        let na_bytes: Vec<u8> = n_active_topk_per_row[..b].iter().flat_map(|x| x.to_le_bytes()).collect();
        let d_nv = gpu.upload_raw(&nv_bytes, &[b]).map_err(|e| format!("nv up: {e:?}"))?;
        let d_na = gpu.upload_raw(&na_bytes, &[b]).map_err(|e| format!("na up: {e:?}"))?;

        gpu.v4f_attn_swa_topk_batched_f32(
            &d_q_b, &d_sk_b, &d_sv_b, &d_tk_b, &d_tv_b,
            &d_sink, &d_nv, &d_na, &d_out_b,
            N_HEADS as i32, HEAD_DIM as i32,
            SWA_WIN as i32, TOPK_WIN as i32,
            b as i32,
        ).map_err(|e| format!("batched dispatch: {e:?}"))?;

        let batched_out = gpu.download_f32(&d_out_b).map_err(|e| format!("batched d2h: {e:?}"))?;

        // ---- 3. Compare row-by-row ----
        let mut max_abs_diff = 0.0f32;
        let mut max_rel_diff = 0.0f32;
        let mut byte_eq = true;
        for row in 0..b {
            for j in 0..N_HEADS * HEAD_DIM {
                let i = row * N_HEADS * HEAD_DIM + j;
                let s = seq_out[i];
                let bv = batched_out[i];
                let d = (s - bv).abs();
                let denom = s.abs().max(bv.abs()).max(1e-8);
                let rel = d / denom;
                if d > max_abs_diff { max_abs_diff = d; }
                if rel > max_rel_diff { max_rel_diff = rel; }
                if s.to_bits() != bv.to_bits() { byte_eq = false; }
            }
        }

        eprintln!(
            "  B={b}: max_abs_diff={max_abs_diff:.3e} max_rel_diff={max_rel_diff:.3e} byte_eq={byte_eq}"
        );

        gpu.free_tensor(d_q_b).ok();
        gpu.free_tensor(d_sk_b).ok();
        gpu.free_tensor(d_sv_b).ok();
        gpu.free_tensor(d_tk_b).ok();
        gpu.free_tensor(d_tv_b).ok();
        gpu.free_tensor(d_out_b).ok();
        gpu.free_tensor(d_nv).ok();
        gpu.free_tensor(d_na).ok();

        // Acceptance: at B=1, demand byte-equality. At B>1, demand ε ≤ 1e-5.
        if b == 1 {
            if !byte_eq {
                return Err(format!(
                    "B=1 byte-equality FAILED: max_abs_diff={max_abs_diff:.3e}"
                ));
            }
        } else if max_abs_diff > 1e-5 || max_rel_diff > 1e-5 {
            return Err(format!(
                "B={b} ε exceeded: max_abs={max_abs_diff:.3e} max_rel={max_rel_diff:.3e}"
            ));
        }
        gpu.free_tensor(d_sink).ok();
    }

    eprintln!("\nOK: v4f_attn_swa_topk_batched_f32 matches sequential at B=1 (byte-eq) and B=4 (FMA-ε)");
    Ok(())
}
