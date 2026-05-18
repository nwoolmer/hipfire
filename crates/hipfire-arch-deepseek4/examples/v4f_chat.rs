//! V4F chat. Wraps user input in the DeepSeek chat template
//! (`<｜User｜>...<｜Assistant｜>`) and stops on `<｜end▁of▁sentence｜>`.
//! Multi-turn KV cache is preserved across turns; `/reset` starts a
//! new conversation. `HIPFIRE_V4F_CHAT_RAW=1` falls back to base
//! completion (no template, no EOS stop) for diagnostics.
//!
//! Usage:
//!   v4f_chat               # interactive: prompt > generate
//!   echo "Hello" | v4f_chat
//!
//! ENV:
//!   HIPFIRE_V4F_ATTN=pos0      fall back to pos-0 attention (default: SWA)
//!   HIPFIRE_V4F_GEN_TOKENS=N   max tokens per turn (default 200)
//!   HIPFIRE_V4F_MODEL=PATH     V4F HFQ path
//!   HIPFIRE_V4F_CHAT_RAW=1     disable chat template (base-completion mode)

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
        .ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let raw_mode = std::env::var("HIPFIRE_V4F_CHAT_RAW").ok().as_deref() == Some("1");

    eprintln!("Loading V4F from {path}...");
    let mut hfq = HfqFile::open(std::path::Path::new(&path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .ok_or_else(|| "tokenizer not found in HFQ metadata".to_string())?;

    // Look up special-token ids by encoding the literals — Tokenizer's
    // special-token table contains the DeepSeek `<｜...｜>` markers.
    let lookup_id = |s: &str| -> Option<u32> {
        let ids = tokenizer.encode(s);
        if ids.len() == 1 { Some(ids[0]) } else { None }
    };
    let bos_tok   = lookup_id("<｜begin▁of▁sentence｜>");
    let user_tok  = lookup_id("<｜User｜>");
    let asst_tok  = lookup_id("<｜Assistant｜>");
    let eos_tok   = lookup_id("<｜end▁of▁sentence｜>")
        .unwrap_or(tokenizer.eos_id);

    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = DeepseekV4State::new(&cfg)?;

    eprintln!("V4F ready. Type a prompt and press enter (or pipe text). EOF to quit. /reset to clear context.");
    eprintln!("Config: layers={} hidden={} vocab={} window={}",
        cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size, cfg.sliding_window);
    eprintln!("Generation: max_tokens={} attention={} mode={}", max_gen,
        std::env::var("HIPFIRE_V4F_ATTN").unwrap_or_else(|_| "swa".to_string()),
        if raw_mode { "raw" } else { "chat" });
    if !raw_mode {
        eprintln!("Chat tokens: bos={:?} user={:?} assistant={:?} eos={}",
            bos_tok, user_tok, asst_tok, eos_tok);
        if user_tok.is_none() || asst_tok.is_none() {
            eprintln!("WARNING: <｜User｜> or <｜Assistant｜> not found as single special token — chat template may not work. Set HIPFIRE_V4F_CHAT_RAW=1 to bypass.");
        }
    }

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut pos: u32 = 0;
    let mut first_turn = true;

    for line in stdin.lock().lines() {
        let prompt = line.map_err(|e| format!("stdin: {e:?}"))?;
        let trimmed = prompt.trim();
        if trimmed.is_empty() { continue; }
        if trimmed == "/reset" {
            state = DeepseekV4State::new(&cfg)?;
            pos = 0;
            first_turn = true;
            eprintln!("[context cleared]");
            continue;
        }

        // Build the token sequence for this turn.
        let mut prompt_tokens: Vec<u32> = Vec::new();
        if raw_mode {
            prompt_tokens.extend(tokenizer.encode(&prompt));
        } else {
            if first_turn {
                if let Some(b) = bos_tok { prompt_tokens.push(b); }
            }
            if let Some(u) = user_tok { prompt_tokens.push(u); }
            prompt_tokens.extend(tokenizer.encode(&prompt));
            if let Some(a) = asst_tok { prompt_tokens.push(a); }
        }
        eprintln!("[prompt: {} tokens (pos {} → {})]",
            prompt_tokens.len(), pos, pos + prompt_tokens.len() as u32);

        // Feed prompt tokens through decode_step; keep the last token's
        // logits to pick the first generated token.
        let mut last_logits = vec![];
        for &t in &prompt_tokens {
            last_logits = decode_step(&cfg, &weights, &mut state, &mut gpu, t, pos)?;
            pos += 1;
        }

        // Greedy decode.
        let mut tok = last_logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        let mut generated: Vec<u32> = Vec::with_capacity(max_gen as usize);
        for _ in 0..max_gen {
            if !raw_mode && tok == eos_tok { break; }
            generated.push(tok);
            let logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos)?;
            pos += 1;
            let argmax = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
            tok = argmax.0 as u32;
        }

        let text = tokenizer.decode(&generated);
        writeln!(stdout, "{}", text).ok();
        stdout.flush().ok();
        first_turn = false;
    }

    Ok(())
}
