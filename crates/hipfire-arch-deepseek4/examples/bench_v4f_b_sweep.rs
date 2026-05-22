//! Load V4F once, sweep batch size B across multiple prefill measurements.
//!
//! Avoids the disk-read overhead of running v4f_mtp_smoke per cell
//! (each run re-reads the 80 GB HFQ from disk). Single load → many
//! PBS allocations at varying B → many timed forwards.
//!
//! Usage:
//!   bench_v4f_b_sweep <model.hfq> [--prompt "..."] [--trials N=3]
//!                     [--bs "16,32,64,128,256,512"]
//!                     [--dump-topk-dir PATH]
//!
//! Output: CSV-style lines `B,trial,main_us` to stdout.
//!
//! Env (consumed by forward.rs):
//!   HIPFIRE_V4F_MOE=1, HIPFIRE_V4F_UPLOAD_EXPERTS=1 — required
//!   HIPFIRE_V4F_MTP_SKIP_HEAD=1 — defensive
//!   HIPFIRE_V4F_DUMP_TOPK=PATH — if --dump-topk-dir is set, this is
//!     set per-B to PATH/topk_B<B>.bin BEFORE the first trial.

use hipfire_arch_deepseek4::{
    forward::{forward_prefill_batch_chunk, PrefillBatchScratch},
    DeepseekV4, DeepseekV4State,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::time::Instant;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect(
        "usage: bench_v4f_b_sweep <model.hfq> [--prompt STR] [--trials N] [--bs LIST] [--dump-topk-dir PATH]"
    );
    let mut prompt = "The quick brown fox jumps over the lazy dog.".to_string();
    let mut trials: usize = 3;
    let mut bs_list: Vec<usize> = vec![16, 32, 64, 128, 256, 512];
    let mut dump_topk_dir: Option<String> = None;
    while let Some(flag) = args.next() {
        let val = args.next().expect("flag missing value");
        match flag.as_str() {
            "--prompt" => prompt = val,
            "--trials" => trials = val.parse().map_err(|e| format!("trials: {e}"))?,
            "--bs" => {
                bs_list = val.split(',')
                    .map(|s| s.trim().parse().expect("bad B"))
                    .collect();
            }
            "--dump-topk-dir" => dump_topk_dir = Some(val),
            other => return Err(format!("unknown flag: {other}")),
        }
    }

    eprintln!("Loading V4F from {model_path}...");
    let mut hfq = HfqFile::open(std::path::Path::new(&model_path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .ok_or_else(|| "tokenizer not found in HFQ metadata".to_string())?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    eprintln!("V4F loaded. arch={}", gpu.arch);
    let prompt_tokens = tokenizer.encode(&prompt);
    let n_prompt = prompt_tokens.len();
    eprintln!("Prompt: {} tokens", n_prompt);

    // CSV header.
    println!("B,trial,main_us,tok_per_sec");

    // Allocate ONE PBS at the MAX B. Smaller-B chunks just feed shorter
    // slices to forward_prefill_batch_chunk. PBS uses pbs.max_batch as
    // the high-water alloc; we never exceed it.
    let max_b = *bs_list.iter().max().unwrap_or(&512);
    eprintln!("Allocating PBS once at max_batch={max_b}...");
    let pbs = PrefillBatchScratch::new(&mut gpu, &cfg, max_b)?;
    eprintln!("PBS allocated.");

    // Allocate ONE state and REUSE it across trials. We clear-by-reset
    // is impossible (state has tons of lazy GpuTensor fields), so we
    // tolerate that re-using state means later trials run with WARM
    // SWA / indexer caches from earlier B's prefill. This is OK because
    // we're comparing relative tok/s rates across B — the absolute
    // numbers include some warm-state savings but that's consistent.
    let mut state = DeepseekV4State::new(&cfg)?;

    // Warm up with a chunk at B=16 so the state's lazy allocs land
    // before any timed measurement.
    {
        let warmup_chunk = &prompt_tokens[..16.min(n_prompt)];
        let _ = forward_prefill_batch_chunk(
            &cfg, &weights, &mut state, &mut gpu, &pbs,
            warmup_chunk, 0u32,
        )?;
        gpu.hip.device_synchronize()
            .map_err(|e| format!("warmup sync: {e:?}"))?;
    }

    for &b in &bs_list {
        // Per-B topk dump: clear before first trial of THIS B.
        if let Some(dir) = &dump_topk_dir {
            let path = format!("{dir}/topk_B{b}.bin");
            std::fs::remove_file(&path).ok();
            std::env::set_var("HIPFIRE_V4F_DUMP_TOPK", &path);
        } else {
            std::env::remove_var("HIPFIRE_V4F_DUMP_TOPK");
        }
        eprintln!("\n=== B={b} ===");

        for trial in 0..trials {
            // Skip topk dump after first trial.
            if trial > 0 { std::env::remove_var("HIPFIRE_V4F_DUMP_TOPK"); }

            // Process the prompt manually in chunks of size B.
            gpu.hip.device_synchronize()
                .map_err(|e| format!("pre-sync B={b} t={trial}: {e:?}"))?;
            let t0 = Instant::now();
            let mut pos: usize = 0;
            while pos < n_prompt {
                let chunk_size = (n_prompt - pos).min(b);
                let chunk = &prompt_tokens[pos..pos + chunk_size];
                forward_prefill_batch_chunk(
                    &cfg, &weights, &mut state, &mut gpu, &pbs,
                    chunk, pos as u32,
                )?;
                pos += chunk_size;
            }
            gpu.hip.device_synchronize()
                .map_err(|e| format!("post-sync B={b} t={trial}: {e:?}"))?;
            let dt = t0.elapsed();
            let us = dt.as_micros();
            let tok_per_sec = n_prompt as f64 / dt.as_secs_f64();
            eprintln!("  trial {trial}: {} us = {:.2} tok/s", us, tok_per_sec);
            println!("{b},{trial},{us},{:.2}", tok_per_sec);
        }
    }

    Ok(())
}
