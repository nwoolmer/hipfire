//! Perplexity / NLL for V4F on a text corpus.
//!
//! Usage: v4f_perplexity <model.hfq> <corpus.txt> [--ctx N] [--warmup N] [--moe 0|1]
//!
//! For comparing fixes across iterations: keep model/corpus/ctx/warmup
//! fixed, only vary one thing per measurement.

use hipfire_arch_deepseek4::{forward::decode_step, DeepseekV4, DeepseekV4State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect(
        "usage: v4f_perplexity <model> <corpus> [--ctx N] [--warmup N] [--moe 0|1]");
    let corpus_path = args.next().expect(
        "usage: v4f_perplexity <model> <corpus> [--ctx N] [--warmup N] [--moe 0|1]");

    let mut ctx_len: usize = 256;
    let mut warmup: usize = 8;
    let mut use_moe: bool = true;
    while let Some(flag) = args.next() {
        let val = args.next().expect("flag missing value");
        match flag.as_str() {
            "--ctx" => ctx_len = val.parse().map_err(|e| format!("ctx: {e}"))?,
            "--warmup" => warmup = val.parse().map_err(|e| format!("warmup: {e}"))?,
            "--moe" => use_moe = val == "1",
            _ => return Err(format!("unknown flag: {flag}")),
        }
    }

    eprintln!("Loading V4F from {model_path}...");
    let mut hfq = HfqFile::open(Path::new(&model_path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .ok_or_else(|| "tokenizer not found in HFQ metadata".to_string())?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;

    if use_moe {
        std::env::set_var("HIPFIRE_V4F_UPLOAD_EXPERTS", "1");
        std::env::set_var("HIPFIRE_V4F_MOE", "1");
    }
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = DeepseekV4State::new(&cfg)?;

    let raw = std::fs::read(&corpus_path).map_err(|e| format!("read corpus: {e}"))?;
    let corpus = String::from_utf8_lossy(&raw).to_string();
    let tokens = tokenizer.encode(&corpus);
    eprintln!("Corpus: {} bytes → {} tokens", raw.len(), tokens.len());
    assert!(tokens.len() > ctx_len + 1, "corpus too short for ctx={}", ctx_len);

    // Use SWA attention path for the perplexity run (V4F's production mode).
    std::env::set_var("HIPFIRE_V4F_ATTN", "swa");

    let mut sum_nll = 0.0f64;
    let mut n_scored = 0usize;

    let t0 = Instant::now();
    for pos in 0..ctx_len {
        let tok = tokens[pos];
        let logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos as u32)?;
        if pos < warmup { continue; }
        // Score the next token's probability under the current logits.
        let next_tok = tokens[pos + 1] as usize;
        // log_softmax: log(exp(x_target) / sum(exp(x))) = x_target - log_sum_exp(x)
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut lse = 0.0f64;
        for &x in &logits { lse += ((x - max_logit) as f64).exp(); }
        let log_z = lse.ln() + max_logit as f64;
        let nll = log_z - logits[next_tok] as f64;
        sum_nll += nll;
        n_scored += 1;
        if (pos + 1) % 32 == 0 {
            let avg = sum_nll / n_scored as f64;
            eprintln!("  [pos={pos:>4}] nll={nll:.4} running avg={avg:.4} ppl={:.2}",
                avg.exp());
        }
    }
    let elapsed = t0.elapsed();

    let avg_nll = sum_nll / n_scored as f64;
    let ppl = avg_nll.exp();
    eprintln!("\nResults:");
    eprintln!("  ctx_len = {ctx_len}, warmup = {warmup}, scored = {n_scored}");
    eprintln!("  total time = {:?} ({:.2} tok/s)",
        elapsed, ctx_len as f64 / elapsed.as_secs_f64());
    eprintln!("  avg_nll = {avg_nll:.6}");
    eprintln!("  ppl     = {ppl:.4}");
    Ok(())
}
