//! Minimum-viable V4F chat. Reads prompts from stdin, runs them
//! through the V4F decode_step pipeline, generates N tokens per
//! turn, prints continuation to stdout.
//!
//! Usage:
//!   v4f_chat               # interactive: prompt > generate
//!   echo "Hello" | v4f_chat
//!
//! ENV:
//!   HIPFIRE_V4F_ATTN=pos0    fall back to pos-0 attention (default: SWA)
//!   HIPFIRE_V4F_GEN_TOKENS=N max tokens per turn (default 50)
//!   HIPFIRE_V4F_MODEL=PATH   V4F HFQ path

use hipfire_arch_deepseek4::{forward::decode_step, DeepseekV4, DeepseekV4State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::io::{self, BufRead, Write};

fn main() -> Result<(), String> {
    let path = std::env::var("HIPFIRE_V4F_MODEL")
        .unwrap_or_else(|_| "/home/nick/.hipfire/models/v4f.mq2lloyd-gptq-all".to_string());
    let max_gen: u32 = std::env::var("HIPFIRE_V4F_GEN_TOKENS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(50);

    eprintln!("Loading V4F from {path}...");
    let mut hfq = HfqFile::open(std::path::Path::new(&path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .ok_or_else(|| "tokenizer not found in HFQ metadata".to_string())?;

    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = DeepseekV4State::new(&cfg)?;

    eprintln!("V4F ready. Type a prompt and press enter (or pipe text). EOF to quit.");
    eprintln!("Config: layers={} hidden={} vocab={} window={}",
        cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size, cfg.sliding_window);
    eprintln!("Generation: max_tokens={} attention={}", max_gen,
        std::env::var("HIPFIRE_V4F_ATTN").unwrap_or_else(|_| "swa".to_string()));

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut pos: u32 = 0;

    for line in stdin.lock().lines() {
        let prompt = line.map_err(|e| format!("stdin: {e:?}"))?;
        if prompt.trim().is_empty() { continue; }

        let prompt_tokens = tokenizer.encode(&prompt);
        eprintln!("[prompt: {} tokens]", prompt_tokens.len());
        write!(stdout, "{}", prompt).ok();
        stdout.flush().ok();

        // Feed prompt tokens through decode_step (their logits ignored,
        // we just want their K/V in cache if SWA is enabled).
        let mut last_logits = vec![];
        for &t in &prompt_tokens {
            last_logits = decode_step(&cfg, &weights, &mut state, &mut gpu, t, pos)?;
            pos += 1;
        }

        // Greedy decode from the last prompt token's logits.
        let mut tok = last_logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        let mut generated: Vec<u32> = Vec::with_capacity(max_gen as usize);
        for _ in 0..max_gen {
            generated.push(tok);
            let logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos)?;
            pos += 1;
            let argmax = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
            tok = argmax.0 as u32;
            // Stop on BOS/EOS as a crude end marker.
            if tok == 0 || tok == 1 { break; }
        }

        let text = tokenizer.decode(&generated);
        writeln!(stdout, "{}", text).ok();
        stdout.flush().ok();
    }

    Ok(())
}
