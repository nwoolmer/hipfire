//! V4F batched-prefill divergence bisector.
//!
//! At B=1, prompt=1 the chunk forward produces different logits than
//! sequential decode_step. This compares intermediate state between
//! the two paths to find where they first diverge.
//!
//! Step 2 of the bisection: compare residual_streams AFTER all layers
//! (i.e. before final_norm_and_head). If they match, the bug is in
//! final_norm_and_head_last_batched. If they don't, the bug is in
//! some per-layer batched stage.

use hipfire_arch_deepseek4::{
    forward::{decode_step, forward_prefill_batch_chunk, PrefillBatchScratch},
    DeepseekV4, DeepseekV4State,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

fn main() -> Result<(), String> {
    let path = std::env::var("HIPFIRE_V4F_MODEL")
        .unwrap_or_else(|_| "/data/hipfire-models/v4f.mq2lloyd-f16compress.hfq".to_string());
    eprintln!("model={path}");

    let mut hfq = HfqFile::open(std::path::Path::new(&path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let pbs = PrefillBatchScratch::new(&mut gpu, &cfg, /*max_batch=*/1)?;

    let token: u32 = 1;

    // ── Sequential: decode_step at pos=0 with fresh state.
    let mut seq_state = DeepseekV4State::new(&cfg)?;
    let _ = decode_step(&cfg, &weights, &mut seq_state, &mut gpu, token, 0)?;
    let seq_streams_tensor = seq_state.residual_streams.as_ref()
        .ok_or_else(|| "seq residual_streams not allocated".to_string())?;
    let seq_streams = gpu.download_f32(seq_streams_tensor)
        .map_err(|e| format!("d2h seq_streams: {e:?}"))?;

    // Also capture attn_out, ffn_out, hc_x_in for stage-level comparison.
    let seq_attn_out = if let Some(t) = seq_state.attn_out.as_ref() {
        gpu.download_f32(t).map_err(|e| format!("d2h seq_attn_out: {e:?}"))?
    } else { Vec::new() };
    let seq_ffn_out = if let Some(t) = seq_state.ffn_out.as_ref() {
        gpu.download_f32(t).map_err(|e| format!("d2h seq_ffn_out: {e:?}"))?
    } else { Vec::new() };
    let seq_hc_x_in = if let Some(t) = seq_state.hc_x_in.as_ref() {
        gpu.download_f32(t).map_err(|e| format!("d2h seq_hc_x_in: {e:?}"))?
    } else { Vec::new() };
    let seq_topk_w = if let Some(t) = seq_state.moe_topk_weights.as_ref() {
        gpu.download_f32(t).map_err(|e| format!("d2h seq_topk_w: {e:?}"))?
    } else { Vec::new() };
    let seq_topk_idx_bytes = if let Some(t) = seq_state.moe_topk_indices.as_ref() {
        let mut buf = vec![0u8; t.byte_size()];
        gpu.hip.memcpy_dtoh(&mut buf, &t.buf)
            .map_err(|e| format!("d2h seq_topk_idx: {e:?}"))?;
        buf
    } else { Vec::new() };
    let seq_router_scores = if let Some(t) = seq_state.router_scores.as_ref() {
        gpu.download_f32(t).map_err(|e| format!("d2h seq_router_scores: {e:?}"))?
    } else { Vec::new() };

    // ── Batched: forward_prefill_batch_chunk at start_pos=0, B=1.
    let mut bat_state = DeepseekV4State::new(&cfg)?;
    forward_prefill_batch_chunk(
        &cfg, &weights, &mut bat_state, &mut gpu, &pbs, &[token], 0,
    )?;
    gpu.hip.device_synchronize().map_err(|e| format!("sync: {e:?}"))?;

    // Compare per-stage outputs at layer (layer_end - 1).
    let bat_attn_out = gpu.download_f32(&pbs.attn_out_batch)
        .map_err(|e| format!("d2h bat_attn_out: {e:?}"))?;
    let bat_ffn_out = gpu.download_f32(&pbs.ffn_out_batch)
        .map_err(|e| format!("d2h bat_ffn_out: {e:?}"))?;
    let bat_hc_x_in = gpu.download_f32(&pbs.hc_x_in_batch)
        .map_err(|e| format!("d2h bat_hc_x_in: {e:?}"))?;
    let bat_topk_w = gpu.download_f32(&pbs.moe_topk_weights_batch)
        .map_err(|e| format!("d2h bat_topk_w: {e:?}"))?;
    let mut bat_topk_idx_bytes = vec![0u8; pbs.moe_topk_indices_batch.byte_size()];
    gpu.hip.memcpy_dtoh(&mut bat_topk_idx_bytes, &pbs.moe_topk_indices_batch.buf)
        .map_err(|e| format!("d2h bat_topk_idx: {e:?}"))?;
    let bat_router_scores = gpu.download_f32(&pbs.moe_scores_batch)
        .map_err(|e| format!("d2h bat_router_scores: {e:?}"))?;
    let stage_compare = |name: &str, seq: &[f32], bat: &[f32]| {
        if seq.is_empty() {
            eprintln!("  [{name}] seq tensor not allocated (stage skipped)");
            return;
        }
        let n = seq.len().min(bat.len());
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        for i in 0..n {
            let d = (seq[i] - bat[i]).abs();
            if d > max_abs { max_abs = d; }
            sum_abs += d as f64;
        }
        let mean_abs = (sum_abs / n as f64) as f32;
        eprintln!("  [{name:<12}] max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} seq[0..3]={:?} bat[0..3]={:?}",
            &seq[..3.min(n)], &bat[..3.min(n)]);
    };
    eprintln!("\n=== Per-stage comparison after layer (layer_end-1) ===");
    stage_compare("attn_out", &seq_attn_out, &bat_attn_out);
    stage_compare("ffn_out", &seq_ffn_out, &bat_ffn_out);
    stage_compare("hc_x_in", &seq_hc_x_in, &bat_hc_x_in);
    stage_compare("router_scores", &seq_router_scores, &bat_router_scores);
    stage_compare("moe_topk_w", &seq_topk_w, &bat_topk_w);
    if !seq_topk_idx_bytes.is_empty() && !bat_topk_idx_bytes.is_empty() {
        let seq_i: Vec<i32> = seq_topk_idx_bytes.chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        let bat_i: Vec<i32> = bat_topk_idx_bytes.chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        let n = seq_i.len().min(bat_i.len());
        let mismatches: Vec<usize> = (0..n).filter(|&i| seq_i[i] != bat_i[i]).collect();
        eprintln!("  [moe_topk_idx] seq={:?} bat={:?} mismatches={}",
            &seq_i[..n], &bat_i[..n], mismatches.len());
    }
    eprintln!("");

    // pbs.streams_batch shape: [max_batch=1, hc_mult, hidden]. At b=0 the
    // whole buffer IS the streams for our single position.
    let bat_streams = gpu.download_f32(&pbs.streams_batch)
        .map_err(|e| format!("d2h bat_streams: {e:?}"))?;
    // Trim to actual layer-output size (= seq_streams.len()).
    let n = seq_streams.len();
    let bat_view = &bat_streams[..n];
    eprintln!("bat residual_streams: numel={} first5={:?} last5={:?}",
        bat_view.len(),
        &bat_view[..5.min(n)],
        &bat_view[n.saturating_sub(5)..]);

    // Compare.
    assert_eq!(seq_streams.len(), bat_view.len(),
        "shape mismatch seq={} vs bat={}", seq_streams.len(), bat_view.len());
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut first_diff: Option<(usize, f32, f32)> = None;
    for (i, (&s, &b)) in seq_streams.iter().zip(bat_view.iter()).enumerate() {
        let d = (s - b).abs();
        if d > max_abs { max_abs = d; }
        sum_abs += d as f64;
        if first_diff.is_none() && d > 1e-3 {
            first_diff = Some((i, s, b));
        }
    }
    let mean_abs = (sum_abs / n as f64) as f32;

    eprintln!("\n=== residual_streams comparison ===");
    eprintln!("max_abs = {max_abs:.4e}");
    eprintln!("mean_abs = {mean_abs:.4e}");
    if let Some((i, s, b)) = first_diff {
        let hidden = cfg.hidden_size;
        let stream_idx = i / hidden;
        let dim_idx = i % hidden;
        eprintln!("first diff @ idx={i} (stream={stream_idx}, dim={dim_idx}): seq={s:.6} bat={b:.6}");
    } else {
        eprintln!("no diff > 1e-3");
    }
    Ok(())
}

fn argmax(xs: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = xs[0];
    for (i, &v) in xs.iter().enumerate() {
        if v > bv { bv = v; best = i; }
    }
    best as u32
}
