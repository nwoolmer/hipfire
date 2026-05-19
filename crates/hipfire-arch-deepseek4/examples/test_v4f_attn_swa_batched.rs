//! Byte-equality smoke test for `v4f_attn_swa_batched` (Phase A2).
//! Mirrors `test_v4f_attn_swa_topk_batched` but without the top-K path.

use rdna_compute::{DType, Gpu};

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().expect("GPU init");

    const N_HEADS: usize = 4;
    const HEAD_DIM: usize = 128;
    const WIN: usize = 64;
    const O_GROUPS: i32 = 1;
    let n_valid_per_row: [i32; 32] = [
        12, 33, 64, 50, 5,  7,  20, 41, 60, 1,  64, 64, 11, 25, 8,  44,
        17, 39, 55, 22, 64, 30, 14, 9,  3,  48, 36, 19, 28, 51, 6,  57,
    ];
    let b_sizes = [1usize, 4, 32];

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
        let mut q_full     = vec![0.0f32; b * N_HEADS * HEAD_DIM];
        let mut swa_k_full = vec![0.0f32; b * HEAD_DIM * WIN];
        let mut swa_v_full = vec![0.0f32; b * HEAD_DIM * WIN];
        for row in 0..b {
            let q_row = seed_input(row * 1000 + 1,   N_HEADS * HEAD_DIM);
            let kk    = seed_input(row * 1000 + 100, HEAD_DIM * WIN);
            let vv    = seed_input(row * 1000 + 200, HEAD_DIM * WIN);
            q_full[row * N_HEADS * HEAD_DIM..][..N_HEADS * HEAD_DIM].copy_from_slice(&q_row);
            swa_k_full[row * HEAD_DIM * WIN..][..HEAD_DIM * WIN].copy_from_slice(&kk);
            swa_v_full[row * HEAD_DIM * WIN..][..HEAD_DIM * WIN].copy_from_slice(&vv);
        }
        let attn_sink_data: Vec<f32> = (0..N_HEADS).map(|h| 0.1 * (h as f32 + 1.0)).collect();
        let d_sink = gpu.upload_f32(&attn_sink_data, &[N_HEADS])
            .map_err(|e| format!("sink up: {e:?}"))?;

        // Sequential baseline.
        let mut seq_out = vec![0.0f32; b * N_HEADS * HEAD_DIM];
        for row in 0..b {
            let q_slice = &q_full[row * N_HEADS * HEAD_DIM..][..N_HEADS * HEAD_DIM];
            let k_slice = &swa_k_full[row * HEAD_DIM * WIN..][..HEAD_DIM * WIN];
            let v_slice = &swa_v_full[row * HEAD_DIM * WIN..][..HEAD_DIM * WIN];

            let d_q = gpu.upload_f32(q_slice, &[N_HEADS, HEAD_DIM]).map_err(|e| format!("q up: {e:?}"))?;
            let d_k = gpu.upload_f32(k_slice, &[1, HEAD_DIM, WIN]).map_err(|e| format!("k up: {e:?}"))?;
            let d_v = gpu.upload_f32(v_slice, &[1, HEAD_DIM, WIN]).map_err(|e| format!("v up: {e:?}"))?;
            let d_out = gpu.zeros(&[N_HEADS, HEAD_DIM], DType::F32).map_err(|e| format!("out: {e:?}"))?;

            gpu.v4f_attn_swa(
                &d_q, &d_k, &d_v, &d_sink, &d_out,
                N_HEADS as i32, HEAD_DIM as i32, O_GROUPS,
                n_valid_per_row[row], WIN as i32,
            ).map_err(|e| format!("seq: {e:?}"))?;

            let row_out = gpu.download_f32(&d_out).map_err(|e| format!("d2h: {e:?}"))?;
            seq_out[row * N_HEADS * HEAD_DIM..][..N_HEADS * HEAD_DIM].copy_from_slice(&row_out);

            gpu.free_tensor(d_q).ok();
            gpu.free_tensor(d_k).ok();
            gpu.free_tensor(d_v).ok();
            gpu.free_tensor(d_out).ok();
        }

        // Batched.
        let d_q_b  = gpu.upload_f32(&q_full,     &[b, N_HEADS, HEAD_DIM]).map_err(|e| format!("Q b: {e:?}"))?;
        let d_sk_b = gpu.upload_f32(&swa_k_full, &[b, HEAD_DIM, WIN]).map_err(|e| format!("K b: {e:?}"))?;
        let d_sv_b = gpu.upload_f32(&swa_v_full, &[b, HEAD_DIM, WIN]).map_err(|e| format!("V b: {e:?}"))?;
        let d_out_b = gpu.zeros(&[b, N_HEADS, HEAD_DIM], DType::F32).map_err(|e| format!("out b: {e:?}"))?;
        let nv_bytes: Vec<u8> = n_valid_per_row[..b].iter().flat_map(|x| x.to_le_bytes()).collect();
        let d_nv = gpu.upload_raw(&nv_bytes, &[b]).map_err(|e| format!("nv: {e:?}"))?;

        gpu.v4f_attn_swa_batched(
            &d_q_b, &d_sk_b, &d_sv_b, &d_sink, &d_nv, &d_out_b,
            N_HEADS as i32, HEAD_DIM as i32, O_GROUPS, WIN as i32, b as i32,
        ).map_err(|e| format!("batched: {e:?}"))?;

        let batched_out = gpu.download_f32(&d_out_b).map_err(|e| format!("d2h: {e:?}"))?;

        // Compare.
        let mut max_abs = 0.0f32;
        let mut byte_eq = true;
        for i in 0..b * N_HEADS * HEAD_DIM {
            let d = (seq_out[i] - batched_out[i]).abs();
            if d > max_abs { max_abs = d; }
            if seq_out[i].to_bits() != batched_out[i].to_bits() { byte_eq = false; }
        }
        eprintln!("  B={b}: max_abs={max_abs:.3e} byte_eq={byte_eq}");

        gpu.free_tensor(d_q_b).ok();
        gpu.free_tensor(d_sk_b).ok();
        gpu.free_tensor(d_sv_b).ok();
        gpu.free_tensor(d_out_b).ok();
        gpu.free_tensor(d_nv).ok();
        gpu.free_tensor(d_sink).ok();

        if b == 1 && !byte_eq {
            return Err(format!("B=1 byte-eq FAILED max_abs={max_abs:.3e}"));
        }
        if max_abs > 1e-5 {
            return Err(format!("B={b} ε exceeded: max_abs={max_abs:.3e}"));
        }
    }

    eprintln!("\nOK: v4f_attn_swa_batched matches v4f_attn_swa byte-for-byte at B=1/4/32");
    Ok(())
}
