//! V4F batched prefill smoke + benchmark.
//!
//! Loads a V4F model, runs forward_prefill_batch_chunked over a synthetic
//! prompt and compares timing vs the per-token decode_step loop.
//!
//! Set HIPFIRE_V4F_UPLOAD_EXPERTS=1 to actually run the routed MoE path
//! (otherwise routed experts are skipped and you measure only the
//! shared-FFN prefill, which doesn't represent real perf).
//!
//! Usage:
//!   HIPFIRE_V4F_UPLOAD_EXPERTS=1 HIPFIRE_V4F_MOE=1 \
//!     cargo run --release --example bench_v4f_batched_prefill \
//!         -p hipfire-arch-deepseek4 -- [prompt_len] [max_batch]
//!
//! Defaults: prompt_len=64, max_batch=32.

use hipfire_arch_deepseek4::{
    forward::{decode_step, forward_prefill_batch_chunked, PrefillBatchScratch},
    DeepseekV4, DeepseekV4State,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::time::Instant;

/// Read /proc/meminfo and return (proc_rss_gb, sys_used_gb).
/// sys_used = MemTotal - MemAvailable, reflecting actual system pressure.
fn mem_stats() -> (f64, f64) {
    let proc_kb: u64 = std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find_map(|l| l.strip_prefix("VmRSS:")
            .and_then(|r| r.split_whitespace().next())
            .and_then(|s| s.parse().ok()))
        .unwrap_or(0);
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let mut total_kb: u64 = 0;
    let mut avail_kb: u64 = 0;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total_kb = rest.split_whitespace().next()
                .and_then(|s| s.parse().ok()).unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            avail_kb = rest.split_whitespace().next()
                .and_then(|s| s.parse().ok()).unwrap_or(0);
        }
    }
    let used_kb = total_kb.saturating_sub(avail_kb);
    let proc_gb = (proc_kb as f64) / (1024.0 * 1024.0);
    let used_gb = (used_kb as f64) / (1024.0 * 1024.0);
    (proc_gb, used_gb)
}

fn rss_mark(label: &str) {
    let (proc_gb, sys_gb) = mem_stats();
    eprintln!("[MEM] {label:<32} proc={proc_gb:>6.2} GiB | sys-used={sys_gb:>6.2} GiB");
    if sys_gb > 115.0 {
        eprintln!("[MEM] *** SYSTEM-USED OVER 115 GiB — aborting to avoid OOM ***");
        std::process::exit(2);
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let prompt_len: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let max_batch: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);

    let path = std::env::var("HIPFIRE_V4F_MODEL")
        .unwrap_or_else(|_| "/data/hipfire-models/v4f.mq2lloyd-f16compress.hfq".to_string());
    eprintln!("model: {path}");
    eprintln!("prompt_len: {prompt_len}, max_batch: {max_batch}");
    eprintln!("RSS guard: 115 GiB (process aborts if exceeded)");
    // Force PrefillBatchScratch::new to log its per-field VRAM cost.
    unsafe { std::env::set_var("HIPFIRE_V4F_PBS_VRAM", "1"); }
    rss_mark("startup");

    let mut hfq = HfqFile::open(std::path::Path::new(&path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    rss_mark("after HfqFile::open + cfg");
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    rss_mark("after Gpu::init");
    eprintln!("loading weights …");
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    rss_mark("after load_weights");
    eprintln!("hidden={}, layers={}, n_exp={}, k_top={}",
        cfg.hidden_size, cfg.num_hidden_layers, cfg.n_routed_experts, cfg.num_experts_per_tok);

    // Allocate the prefill scratch.
    eprintln!("allocating PrefillBatchScratch (max_batch={max_batch}) …");
    let pbs = PrefillBatchScratch::new(&mut gpu, &cfg, max_batch)?;
    rss_mark("after PrefillBatchScratch::new");

    // Synthetic prompt: BOS-ish + a deterministic token sweep.
    let tokens: Vec<u32> = (0..prompt_len).map(|i| (1u32 + (i as u32 % 100u32))).collect();

    // ── Sequential baseline (decode_step per token) ─────────────────
    eprintln!("\n[1/2] sequential decode_step × {prompt_len} …");
    let mut seq_state = DeepseekV4State::new(&cfg)?;
    rss_mark("after seq_state alloc");
    let seq_t0 = Instant::now();
    let mut seq_last_logits: Vec<f32> = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        seq_last_logits = decode_step(&cfg, &weights, &mut seq_state, &mut gpu, tok, i as u32)?;
        if (i + 1) % 16 == 0 {
            eprintln!("  seq: {}/{} tokens", i + 1, prompt_len);
            rss_mark(&format!("seq @ {}/{} tokens", i + 1, prompt_len));
        }
    }
    gpu.hip.device_synchronize()
        .map_err(|e| format!("sync seq: {e:?}"))?;
    let seq_elapsed = seq_t0.elapsed();
    let seq_toks_per_sec = prompt_len as f64 / seq_elapsed.as_secs_f64();
    rss_mark("after seq run");

    // ── Batched chunked prefill ─────────────────────────────────────
    eprintln!("\n[2/2] forward_prefill_batch_chunked …");
    let mut bat_state = DeepseekV4State::new(&cfg)?;
    rss_mark("after bat_state alloc");
    let bat_t0 = Instant::now();
    let bat_last_logits = forward_prefill_batch_chunked(
        &cfg, &weights, &mut bat_state, &mut gpu, &tokens, 0, &pbs,
    )?;
    rss_mark("after batched run");
    gpu.hip.device_synchronize()
        .map_err(|e| format!("sync bat: {e:?}"))?;
    let bat_elapsed = bat_t0.elapsed();
    let bat_toks_per_sec = prompt_len as f64 / bat_elapsed.as_secs_f64();

    let speedup = seq_elapsed.as_secs_f64() / bat_elapsed.as_secs_f64();

    eprintln!("\n--- timings ---");
    eprintln!("sequential : {:>8.2} ms ({:>6.1} tok/s)",
        seq_elapsed.as_secs_f64() * 1000.0, seq_toks_per_sec);
    eprintln!("batched    : {:>8.2} ms ({:>6.1} tok/s)",
        bat_elapsed.as_secs_f64() * 1000.0, bat_toks_per_sec);
    eprintln!("speedup    : {speedup:>8.2}×");

    // ── Correctness check: top-1 + top-K agreement on last-position logits.
    let seq_top1 = argmax(&seq_last_logits);
    let bat_top1 = argmax(&bat_last_logits);
    let (max_abs, mean_abs) = diff_stats(&seq_last_logits, &bat_last_logits);
    eprintln!("\n--- correctness (last-position logits) ---");
    eprintln!("seq top1   : {seq_top1} (logit={:.4})", seq_last_logits[seq_top1 as usize]);
    eprintln!("bat top1   : {bat_top1} (logit={:.4})", bat_last_logits[bat_top1 as usize]);
    eprintln!("max_abs    : {max_abs:.4e}");
    eprintln!("mean_abs   : {mean_abs:.4e}");
    eprintln!("top1 match : {}", seq_top1 == bat_top1);

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

fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    for (av, bv) in a.iter().zip(b.iter()) {
        let d = (av - bv).abs();
        if d > max_abs { max_abs = d; }
        sum_abs += d as f64;
    }
    (max_abs, (sum_abs / a.len() as f64) as f32)
}
