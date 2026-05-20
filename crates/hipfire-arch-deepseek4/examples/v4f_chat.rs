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
//!   HIPFIRE_V4F_TEMP=F         sampling temperature (default 0.7; 0 = greedy argmax)
//!   HIPFIRE_V4F_TOP_K=N        top-K filter before softmax (default 40; 0 = full vocab)
//!   HIPFIRE_V4F_SEED=N         PRNG seed (default: time-based)

use hipfire_arch_deepseek4::{
    forward::{decode_step, forward_prefill_batch_chunked, PrefillBatchScratch},
    DeepseekV4, DeepseekV4State,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::io::{self, BufRead, Write};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Xorshift64* PRNG — tiny, deterministic, no deps.
struct Xorshift { s: u64 }
impl Xorshift {
    fn new(seed: u64) -> Self { Self { s: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed } } }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.s = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32)
    }
}

/// Sample next token from logits.
/// - temp == 0.0: greedy argmax
/// - top_k > 0: keep only K largest logits before softmax
/// - else: temperature-scaled softmax over (filtered) logits, multinomial draw
fn sample_token(logits: &[f32], temp: f32, top_k: usize, rng: &mut Xorshift) -> u32 {
    if temp <= 0.0 {
        return logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
    }
    let n = logits.len();
    let k = if top_k == 0 || top_k >= n { n } else { top_k };
    // Indices of top-k logits by value.
    let mut idx: Vec<usize> = (0..n).collect();
    if k < n {
        idx.select_nth_unstable_by(k - 1, |&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        idx.truncate(k);
    }
    // Softmax over selected logits with temperature.
    let max_l = idx.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<f32> = idx.iter().map(|&i| ((logits[i] - max_l) / temp).exp()).collect();
    let sum: f32 = weights.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return idx.iter().max_by(|&&a, &&b| logits[a].partial_cmp(&logits[b]).unwrap()).copied().unwrap_or(0) as u32;
    }
    for w in weights.iter_mut() { *w /= sum; }
    // Multinomial draw via inverse CDF.
    let r = rng.next_f32();
    let mut acc = 0.0;
    for (j, &w) in weights.iter().enumerate() {
        acc += w;
        if r <= acc { return idx[j] as u32; }
    }
    idx[idx.len() - 1] as u32
}

fn main() -> Result<(), String> {
    let path = std::env::var("HIPFIRE_V4F_MODEL")
        .unwrap_or_else(|_| "/home/nick/.hipfire/models/v4f.mq2lloyd-gptq-all".to_string());
    let max_gen: u32 = std::env::var("HIPFIRE_V4F_GEN_TOKENS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let raw_mode = std::env::var("HIPFIRE_V4F_CHAT_RAW").ok().as_deref() == Some("1");
    let temp: f32 = std::env::var("HIPFIRE_V4F_TEMP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.7);
    let top_k: usize = std::env::var("HIPFIRE_V4F_TOP_K")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(40);
    let seed: u64 = std::env::var("HIPFIRE_V4F_SEED")
        .ok().and_then(|s| s.parse().ok())
        .unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0xC0FFEE));
    let mut rng = Xorshift::new(seed);

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

    // Batched prefill scratch — allocated once, reused for every turn.
    // B=64 matches the bench memory ("V4F batched prefill 23.2 → 44 tok/s @ B=64").
    // PrefillBatchScratch::new prints VRAM cost when HIPFIRE_V4F_PBS_VRAM=1.
    let pbs_max_batch: usize = std::env::var("HIPFIRE_V4F_PP_BATCH")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let pbs = PrefillBatchScratch::new(&mut gpu, &cfg, pbs_max_batch)?;

    eprintln!("V4F ready. Type a prompt and press enter (or pipe text). EOF to quit. /reset to clear context.");
    eprintln!("Config: layers={} hidden={} vocab={} window={}",
        cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size, cfg.sliding_window);
    eprintln!("Generation: max_tokens={} attention={} mode={} temp={} top_k={} seed={}", max_gen,
        std::env::var("HIPFIRE_V4F_ATTN").unwrap_or_else(|_| "swa".to_string()),
        if raw_mode { "raw" } else { "chat" }, temp, top_k, seed);
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
        let prompt_token_count = prompt_tokens.len();
        eprintln!("[prompt: {} tokens (pos {} → {})]",
            prompt_token_count, pos, pos + prompt_token_count as u32);

        // PP: batched chunked forward at B=pbs.max_batch. Returns the
        // last position's logits so TG can pick the first generated
        // token. ~3-4× faster than sequential decode_step for prompts
        // longer than the chunk size; falls back to per-token decode
        // internally if a chunk's path errors.
        //
        // `decode_step` increments state.n_tokens internally; the batched
        // path uses start_pos directly and does NOT touch state.n_tokens.
        // We update it manually below so the subsequent TG decode_step
        // calls write SWA at the right ring slots.
        let pp_start = Instant::now();
        let start_pp_pos = pos;
        let last_logits = forward_prefill_batch_chunked(
            &cfg, &weights, &mut state, &mut gpu, &prompt_tokens, start_pp_pos, &pbs,
        )?;
        pos = start_pp_pos + prompt_tokens.len() as u32;
        state.n_tokens = pos as u64;
        let pp_elapsed = pp_start.elapsed();

        // TG: sample + decode loop.
        let tg_start = Instant::now();
        let mut tok = sample_token(&last_logits, temp, top_k, &mut rng);
        let mut generated: Vec<u32> = Vec::with_capacity(max_gen as usize);
        for _ in 0..max_gen {
            if !raw_mode && tok == eos_tok { break; }
            generated.push(tok);
            let logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos)?;
            pos += 1;
            tok = sample_token(&logits, temp, top_k, &mut rng);
        }
        let tg_elapsed = tg_start.elapsed();

        let text = tokenizer.decode(&generated);
        writeln!(stdout, "{}", text).ok();
        stdout.flush().ok();

        let pp_s = pp_elapsed.as_secs_f64();
        let tg_s = tg_elapsed.as_secs_f64();
        let pp_rate = if pp_s > 0.0 { prompt_token_count as f64 / pp_s } else { 0.0 };
        let tg_rate = if tg_s > 0.0 { generated.len() as f64 / tg_s } else { 0.0 };
        eprintln!("[stats] PP {} tok in {:.2}s = {:.2} tok/s | TG {} tok in {:.2}s = {:.2} tok/s",
            prompt_token_count, pp_s, pp_rate, generated.len(), tg_s, tg_rate);
        first_turn = false;
    }

    Ok(())
}
