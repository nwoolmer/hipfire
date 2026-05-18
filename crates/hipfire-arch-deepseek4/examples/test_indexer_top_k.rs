//! GPU validation for Phase 2 `indexer_top_k`.
//!
//! Verifies the stub picks the K largest score indices per head.

use rdna_compute::Gpu;

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().expect("GPU init");

    const H: usize = 2;
    const N: usize = 16;
    const K: usize = 4;

    // Per-head scores: stream of N floats with known top-K positions.
    // Head 0: ascending — top-4 should be indices N-1, N-2, N-3, N-4.
    // Head 1: a known permutation — embed the truth into the values.
    let mut scores = vec![0.0f32; H * N];
    for n in 0..N {
        scores[n] = n as f32;             // head 0: 0..N-1
        scores[N + n] = (N - 1 - n) as f32; // head 1: N-1..0
    }

    let d_s = gpu.upload_f32(&scores, &[H, N]).map_err(|e| format!("up s: {e:?}"))?;
    // No DType::I32 in rdna-compute — allocate as F32 (4 bytes/element)
    // and treat values as raw i32 bits.
    let d_top = gpu.zeros(&[H, K], rdna_compute::DType::F32)
        .map_err(|e| format!("z top: {e:?}"))?;

    gpu.indexer_top_k(&d_s, &d_top, H as i32, N as i32, K as i32)
        .map_err(|e| format!("dispatch: {e:?}"))?;

    // Download as i32: reuse download_f32 buffer + transmute.
    // Need raw bytes — use direct memcpy.
    let mut top_bytes = vec![0u8; H * K * 4];
    gpu.hip.memcpy_dtoh(&mut top_bytes, &d_top.buf)
        .map_err(|e| format!("d2h: {e:?}"))?;
    let mut top = vec![0i32; H * K];
    for i in 0..H * K {
        top[i] = i32::from_le_bytes(top_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
    }
    eprintln!("head 0 top-{K}: {:?}", &top[..K]);
    eprintln!("head 1 top-{K}: {:?}", &top[K..]);

    // Head 0 expects [N-1, N-2, N-3, N-4] = [15, 14, 13, 12].
    let expected_h0 = [(N - 1) as i32, (N - 2) as i32, (N - 3) as i32, (N - 4) as i32];
    // Head 1 expects scores[N+n] = N-1-n, top values at n=0 (15), n=1 (14)...
    // → indices [0, 1, 2, 3].
    let expected_h1 = [0i32, 1, 2, 3];

    if &top[..K] == &expected_h0 && &top[K..] == &expected_h1 {
        eprintln!("\nOK: indexer_top_k picked correct top-{K} per head");
        Ok(())
    } else {
        eprintln!("expected h0={expected_h0:?}, h1={expected_h1:?}");
        Err("top-k mismatch".into())
    }
}
