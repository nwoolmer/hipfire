//! Byte-equality smoke test for `hc_streams_init_from_embed_batched` (Phase B2).
//!
//! Confirms the broadcast kernel produces `[B, HC_MULT, hidden]` output where
//! each batch row's `HC_MULT` slots are identical copies of the input embed
//! row, matching the per-token init pattern.

use rdna_compute::{DType, Gpu};

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().expect("GPU init");

    const HIDDEN: usize = 256;
    const HC_MULT: usize = 4;
    let b_sizes = [1usize, 4, 32];

    let seed = |off: usize, len: usize| -> Vec<f32> {
        (0..len).map(|i| {
            let x = (off + i) as f32 * 0.017;
            (x.sin() * 0.5) + (x * 0.7).cos() * 0.3
        }).collect()
    };

    for &b in &b_sizes {
        eprintln!("--- batch_size = {b} ---");
        let embed_data: Vec<f32> = seed(b * 1000, b * HIDDEN);

        let d_embed = gpu.upload_f32(&embed_data, &[b, HIDDEN])
            .map_err(|e| format!("embed up: {e:?}"))?;
        let d_streams = gpu.zeros(&[b, HC_MULT, HIDDEN], DType::F32)
            .map_err(|e| format!("streams: {e:?}"))?;

        gpu.hc_streams_init_from_embed_batched(
            &d_embed, &d_streams,
            HIDDEN as i32, HC_MULT as i32, b as i32,
        ).map_err(|e| format!("dispatch: {e:?}"))?;

        let streams_out = gpu.download_f32(&d_streams)
            .map_err(|e| format!("d2h: {e:?}"))?;

        // Build expected: streams[b, h, d] == embed[b, d] for all h.
        let mut max_abs = 0.0f32;
        let mut byte_eq = true;
        for row in 0..b {
            for h in 0..HC_MULT {
                for d in 0..HIDDEN {
                    let got = streams_out[(row * HC_MULT + h) * HIDDEN + d];
                    let want = embed_data[row * HIDDEN + d];
                    let diff = (got - want).abs();
                    if diff > max_abs { max_abs = diff; }
                    if got.to_bits() != want.to_bits() { byte_eq = false; }
                }
            }
        }
        eprintln!("  B={b}: max_abs={max_abs:.3e} byte_eq={byte_eq}");
        gpu.free_tensor(d_embed).ok();
        gpu.free_tensor(d_streams).ok();
        if !byte_eq {
            return Err(format!("B={b}: stream broadcast not byte-identical (max_abs={max_abs:.3e})"));
        }
    }

    eprintln!("\nOK: hc_streams_init_from_embed_batched broadcasts embed → all 4 streams at B=1/4/32");
    Ok(())
}
