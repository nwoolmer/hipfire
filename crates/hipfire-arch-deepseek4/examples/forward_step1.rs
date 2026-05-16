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

    // Call decode_step on token 100. Even though the function returns Err
    // at the end (`layout-only — no executable forward yet`), the first
    // step (init_residual_streams) should populate the state.
    let _ = decode_step(&cfg, &weights, &mut state, &mut gpu, 100, 0);

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
        }
    }
    eprintln!("stream 0 nonzero count: {s0_nonzero} / {hidden}");
    eprintln!("stream 1 nonzero count: {s1_nonzero} / {hidden}");
    eprintln!("stream 2 nonzero count: {s2_nonzero} / {hidden}");
    eprintln!("stream 3 nonzero count: {s3_nonzero} / {hidden}");
    let s_other_nonzero = s1_nonzero + s2_nonzero + s3_nonzero;

    if s0_nonzero > 0 && s_other_nonzero == 0 {
        eprintln!("OK: forward step 1 (embed → [embed, 0, 0, 0]) works on real V4F");
    } else {
        return Err(format!(
            "step 1 wrong shape: s0_nonzero={s0_nonzero} s_other_nonzero={s_other_nonzero}"
        ));
    }

    // Verify step 2 (RMSNorm) populates state.tmp with nonzero values.
    let tmp = state.tmp.as_ref()
        .ok_or_else(|| "state.tmp not allocated".to_string())?;
    let mut tmp_bytes = vec![0u8; tmp.byte_size()];
    gpu.hip.memcpy_dtoh(&mut tmp_bytes, &tmp.buf)
        .map_err(|e| format!("d2h tmp: {e:?}"))?;
    let mut tmp_f32 = vec![0.0f32; hidden];
    for i in 0..hidden {
        tmp_f32[i] = f32::from_le_bytes(tmp_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
    }
    let nonzero = tmp_f32.iter().filter(|v| v.abs() > 1e-6).count();
    let max_abs = tmp_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    eprintln!("RMSNorm output: {nonzero}/{hidden} nonzero, max_abs={max_abs:.4}");
    if nonzero == 0 {
        return Err("step 2 RMSNorm produced all zeros".into());
    }
    eprintln!("OK: forward step 2 (attn_rms_norm) populates tmp[hidden]");

    Ok(())
}
