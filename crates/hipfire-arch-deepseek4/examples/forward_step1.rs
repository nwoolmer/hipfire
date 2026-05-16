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
    let mut s0_nonzero = 0;
    let mut s_other_nonzero = 0;
    for s in 0..hc_mult {
        for d in 0..hidden {
            let off = (s * hidden + d) * 2;
            let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            if bits != 0 && bits != 0x8000 {
                if s == 0 { s0_nonzero += 1; } else { s_other_nonzero += 1; }
            }
        }
    }
    eprintln!("stream 0  nonzero count: {s0_nonzero} / {hidden}");
    eprintln!("streams 1+ nonzero count: {s_other_nonzero} / {}", hidden * (hc_mult - 1));

    if s0_nonzero > 0 && s_other_nonzero == 0 {
        eprintln!("OK: forward step 1 (embed → [embed, 0, 0, 0]) works on real V4F");
        Ok(())
    } else {
        Err(format!("step 1 wrong shape: s0_nonzero={s0_nonzero} s_other_nonzero={s_other_nonzero}"))
    }
}
