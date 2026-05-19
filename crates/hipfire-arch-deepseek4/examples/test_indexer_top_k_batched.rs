//! GPU validation for `indexer_top_k_batched` (Phase A3, 2026-05-18).
//!
//! Confirms the batched stub produces the same per-(batch, head) top-K
//! ordering as B independent sequential calls.

use rdna_compute::{DType, Gpu};

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().expect("GPU init");

    const H: usize = 3;
    const N: usize = 24;
    const K: usize = 6;
    let b_sizes = [1usize, 4, 32];

    let seed = |off: usize, len: usize| -> Vec<f32> {
        (0..len).map(|i| {
            let x = (off + i) as f32 * 0.011;
            (x.sin() * 0.8) + (x * 0.31).cos() * 0.5
        }).collect()
    };

    for &b in &b_sizes {
        eprintln!("--- batch_size = {b} ---");
        let mut all_scores = vec![0.0f32; b * H * N];
        for row in 0..b {
            let row_scores = seed(row * 1000 + 7, H * N);
            all_scores[row * H * N..][..H * N].copy_from_slice(&row_scores);
        }

        // Sequential: call indexer_top_k once per batch row.
        let mut seq_indices = vec![0i32; b * H * K];
        for row in 0..b {
            let row_scores = &all_scores[row * H * N..][..H * N];
            let d_s = gpu.upload_f32(row_scores, &[H, N]).map_err(|e| format!("s up: {e:?}"))?;
            let d_top = gpu.zeros(&[H, K], DType::F32).map_err(|e| format!("top: {e:?}"))?;
            gpu.indexer_top_k(&d_s, &d_top, H as i32, N as i32, K as i32)
                .map_err(|e| format!("seq dispatch: {e:?}"))?;

            let mut bytes = vec![0u8; H * K * 4];
            gpu.hip.memcpy_dtoh(&mut bytes, &d_top.buf).map_err(|e| format!("d2h: {e:?}"))?;
            for i in 0..H * K {
                seq_indices[row * H * K + i] =
                    i32::from_le_bytes(bytes[i * 4..(i + 1) * 4].try_into().unwrap());
            }
            gpu.free_tensor(d_s).ok();
            gpu.free_tensor(d_top).ok();
        }

        // Batched.
        let d_s_b = gpu.upload_f32(&all_scores, &[b, H, N]).map_err(|e| format!("s_b: {e:?}"))?;
        let d_top_b = gpu.zeros(&[b, H, K], DType::F32).map_err(|e| format!("top_b: {e:?}"))?;
        gpu.indexer_top_k_batched(&d_s_b, &d_top_b, H as i32,
            N as i32, N as i32, K as i32, K as i32, b as i32)
            .map_err(|e| format!("batched dispatch: {e:?}"))?;
        let mut bytes = vec![0u8; b * H * K * 4];
        gpu.hip.memcpy_dtoh(&mut bytes, &d_top_b.buf).map_err(|e| format!("d2h: {e:?}"))?;
        let mut batched_indices = vec![0i32; b * H * K];
        for i in 0..b * H * K {
            batched_indices[i] = i32::from_le_bytes(bytes[i * 4..(i + 1) * 4].try_into().unwrap());
        }
        gpu.free_tensor(d_s_b).ok();
        gpu.free_tensor(d_top_b).ok();

        let mut mismatches = 0;
        let mut first_bad: Option<(usize, usize, i32, i32)> = None;
        for i in 0..b * H * K {
            if seq_indices[i] != batched_indices[i] {
                mismatches += 1;
                if first_bad.is_none() {
                    first_bad = Some((i / (H * K), i % (H * K), seq_indices[i], batched_indices[i]));
                }
            }
        }
        eprintln!("  B={b}: mismatches={mismatches}/{}", b * H * K);
        if mismatches > 0 {
            return Err(format!(
                "B={b} top-K mismatch (first: row={:?})",
                first_bad
            ));
        }
    }

    eprintln!("\nOK: indexer_top_k_batched matches indexer_top_k at B=1/4/32");
    Ok(())
}
