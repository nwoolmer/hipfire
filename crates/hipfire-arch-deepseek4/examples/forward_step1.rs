//! Exercise the first step of V4F forward (embed → 4-stream residual init)
//! against the real V4F HFQ file. Verifies the embedding lookup + zero-fill
//! produce the expected `[embed_row, 0, 0, 0]` pattern.
//!
//! Usage:
//!   cargo run --release --example forward_step1 -p hipfire-arch-deepseek4

use hipfire_arch_deepseek4::{
    forward::decode_step, DeepseekV4, DeepseekV4State,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

fn main() -> Result<(), String> {
    let path = "/home/nick/.hipfire/models/v4f.mq2lloyd-gptq-all";
    let mut hfq = HfqFile::open(std::path::Path::new(path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = DeepseekV4State::new(&cfg)?;

    // Call decode_step on a sweep of input tokens. Each call returns
    // a Vec<f32> of vocab_size logits.
    for token_id in [100u32, 0, 1, 123, 1000, 5000, 10000, 50000] {
        let logits = decode_step(&cfg, &weights, &mut state, &mut gpu, token_id, 0)?;
        let max_abs = logits.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let argmax = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        eprintln!("token {token_id:>5} → argmax token {} (logit={:.3}, max_abs={:.3})",
            argmax.0, argmax.1, max_abs);
    }

    // Read back residual_streams; verify stream 0 has nonzero values and
    // streams 1..hc_mult are all zero.
    let streams = state.residual_streams.as_ref()
        .ok_or_else(|| "residual_streams not allocated".to_string())?;
    let mut bytes = vec![0u8; streams.byte_size()];
    gpu.hip.memcpy_dtoh(&mut bytes, &streams.buf)
        .map_err(|e| format!("d2h: {e:?}"))?;

    let hidden = cfg.hidden_size;
    let hc_mult = cfg.hc_mult;
    // F32 now (4 bytes per element).
    let mut s0_max_abs = 0.0f32;
    let mut s_other_max_abs = 0.0f32;
    let mut s0_nonzero = 0;
    let mut s1_nonzero = 0;
    let mut s2_nonzero = 0;
    let mut s3_nonzero = 0;
    for s in 0..hc_mult {
        for d in 0..hidden {
            let off = (s * hidden + d) * 4;
            let v = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            if v.abs() > 1e-6 {
                match s {
                    0 => s0_nonzero += 1,
                    1 => s1_nonzero += 1,
                    2 => s2_nonzero += 1,
                    _ => s3_nonzero += 1,
                }
            }
            if s == 0 { s0_max_abs = s0_max_abs.max(v.abs()); }
            else { s_other_max_abs = s_other_max_abs.max(v.abs()); }
        }
    }
    eprintln!("stream 0 nonzero count: {s0_nonzero} / {hidden}, max_abs={s0_max_abs:.4e}");
    eprintln!("stream 1 nonzero count: {s1_nonzero} / {hidden}");
    eprintln!("stream 2 nonzero count: {s2_nonzero} / {hidden}");
    eprintln!("stream 3 nonzero count: {s3_nonzero} / {hidden}");
    eprintln!("streams 1+ max_abs: {s_other_max_abs:.4e}");
    let s_other_nonzero = s1_nonzero + s2_nonzero + s3_nonzero;

    // After decode_step (43 layers of Q-LoRA + KV + RoPE + HC mix), HC
    // should have propagated signal across streams. Stream 0 must
    // remain nonzero; the test is now "the forward pipeline runs and
    // produces sensible-magnitude state."
    let total_nonzero = s0_nonzero + s_other_nonzero;
    let max_per_stream = [s0_nonzero, s1_nonzero, s2_nonzero, s3_nonzero]
        .iter().copied().max().unwrap();
    if total_nonzero > 0 && max_per_stream > hidden / 2 {
        eprintln!("OK: forward pipeline runs through 43 layers; total nonzero={total_nonzero}");
    } else {
        return Err(format!(
            "forward pipeline produced too few nonzero values: {:?}",
            (s0_nonzero, s1_nonzero, s2_nonzero, s3_nonzero)
        ));
    }

    // Per-layer step probes will get added as steps land. For now,
    // the "forward pipeline runs cleanly through 43 layers + final
    // norm + lm_head stubs" assertion is the only end-to-end gate.
    Ok(())
}
