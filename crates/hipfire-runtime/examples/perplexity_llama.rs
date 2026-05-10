//! Perplexity / NLL eval for LLaMA-arch models (Bonsai HFQ1, etc.) on a
//! text corpus, single-window. Mirrors `perplexity.rs` but uses the LLaMA
//! forward pass instead of Qwen35/DeltaNet.
//!
//! Usage:
//!   perplexity_llama <model.hfq> <corpus.txt> [--ctx 2048] [--warmup 8] [--offset 0] [--kv-mode q8|fp32]
//!
//! For comparing quants: same model class, same corpus, same offset/ctx/warmup.
//! 2K tokens is enough to see sub-4-bit deltas (single decimal of ppl);
//! 8K+ if you want stable second-decimal numbers.

use hipfire_arch_llama::Llama;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{self, KvCache};
use std::path::Path;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect("usage: perplexity_llama <model> <corpus> [--ctx N] [--warmup N] [--offset N] [--kv-mode q8|fp32]");
    let corpus_path = args.next().expect("usage: perplexity_llama <model> <corpus> [--ctx N] [--warmup N] [--offset N] [--kv-mode q8|fp32]");

    let mut ctx_len: usize = 2048;
    let mut warmup: usize = 8;
    let mut offset: usize = 0;
    let mut kv_mode: String = "q8".to_string();

    while let Some(flag) = args.next() {
        let val = args.next().expect("flag missing value");
        match flag.as_str() {
            "--ctx" => ctx_len = val.parse().unwrap(),
            "--warmup" => warmup = val.parse().unwrap(),
            "--offset" => offset = val.parse().unwrap(),
            "--kv-mode" => kv_mode = val,
            _ => panic!("unknown flag: {flag}"),
        }
    }
    assert!(ctx_len > warmup + 4, "ctx must exceed warmup by enough to score");

    // Read enough of the corpus to safely cover offset + ctx tokens at ~3 chars/token
    // (8x slack for non-ASCII / markup).
    let want_bytes = (offset + ctx_len) * 8;
    let raw = std::fs::read(&corpus_path).expect("read corpus");
    let take = want_bytes.min(raw.len());
    let corpus = String::from_utf8_lossy(&raw[..take]).to_string();
    eprintln!("Corpus: {} bytes (of {}) from {corpus_path}", corpus.len(), raw.len());

    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = <Llama as Architecture>::config_from_hfq(&hfq).expect("config");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");

    eprintln!("Tokenizing...");
    let t_tok = Instant::now();
    let all_tokens: Vec<u32> = tokenizer.encode(&corpus);
    eprintln!("Tokenized: {} tokens in {:.2}s",
              all_tokens.len(), t_tok.elapsed().as_secs_f64());

    let end = (offset + ctx_len).min(all_tokens.len());
    if end <= offset + warmup + 4 {
        panic!("not enough tokens past offset={offset} for warmup={warmup} + scoring");
    }
    let window = &all_tokens[offset..end];
    eprintln!("Window: offset={offset} ctx={} (warmup {warmup}, scoring {})",
              window.len(), window.len() - warmup - 1);

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {arch}");
    eprintln!("Loading weights from {model_path}...");
    let weights = <Llama as Architecture>::load_weights(&mut hfq, &config, &mut gpu)
        .expect("load_weights");

    let kv_max = window.len() + 16;
    eprintln!("KV mode: {kv_mode}");
    let mut kv_cache = match kv_mode.as_str() {
        "q8" => KvCache::new_gpu_q8(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max
        ).unwrap(),
        "fp32" => KvCache::new_gpu(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max
        ).unwrap(),
        other => panic!("unknown --kv-mode: {other} (q8, fp32)"),
    };
    let scratch = <Llama as Architecture>::new_state(&mut gpu, &config).unwrap();

    let mut total_nll: f64 = 0.0;
    let mut scored: usize = 0;
    let t0 = Instant::now();

    for (pos, &tok) in window.iter().enumerate().take(window.len() - 1) {
        // Embed + compute one position. For HFQ1G128, forward_scratch_embed
        // writes the embedded x into scratch.x; forward_scratch_compute runs
        // the layer loop and produces logits.
        llama::forward_scratch_embed(&mut gpu, &weights, &config, tok, pos, &scratch)
            .expect("forward_scratch_embed");
        llama::forward_scratch_compute(&mut gpu, &weights, &config, pos, &mut kv_cache, &scratch)
            .expect("forward_scratch_compute");

        if pos < warmup { continue; }

        let logits = gpu.download_f32(&scratch.logits).unwrap();
        let target = window[pos + 1] as usize;
        let nll = neg_log_softmax_at(&logits, target);
        if !nll.is_finite() {
            eprintln!("  warn: non-finite NLL at pos={pos} target={target}, skipping");
            continue;
        }
        total_nll += nll as f64;
        scored += 1;

        if scored == 1 || scored % 256 == 0 {
            let avg_nll = total_nll / scored as f64;
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = scored as f64 / elapsed.max(1e-9);
            eprintln!(
                "  pos={:5} scored={:5} nll/tok={:.4} ppl={:.3} ({:.1} tok/s)",
                pos, scored, avg_nll, avg_nll.exp(), rate,
            );
        }
    }

    let avg_nll = if scored > 0 { total_nll / scored as f64 } else { 0.0 };
    let ppl = avg_nll.exp();
    let elapsed = t0.elapsed().as_secs_f64();
    println!();
    println!("Model:    {model_path}");
    println!("Corpus:   {corpus_path}");
    println!("Tokens:   offset={offset} ctx={} warmup={warmup}", window.len());
    println!("KV mode:  {kv_mode}");
    println!("Scored:   {scored}");
    println!("NLL/tok:  {:.10}", avg_nll);
    println!("PPL:      {:.4}", ppl);
    println!("Elapsed:  {:.1}s ({:.1} tok/s)", elapsed,
             scored as f64 / elapsed.max(1e-9));
}

fn neg_log_softmax_at(logits: &[f32], target: usize) -> f32 {
    if target >= logits.len() {
        return f32::NAN;
    }
    let mut max = f32::NEG_INFINITY;
    for &v in logits { if v > max { max = v; } }
    let mut sum = 0.0f64;
    for &v in logits { sum += ((v - max) as f64).exp(); }
    let log_sum = max as f64 + sum.ln();
    (log_sum - logits[target] as f64) as f32
}
