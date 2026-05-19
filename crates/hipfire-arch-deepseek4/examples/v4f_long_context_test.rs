//! V4F long-context fact-recall regression test — hipfire equivalent of
//! antirez/ds4 `--long-context` from `tests/ds4_test.c`.
//!
//! Reads a long story prompt that embeds 16 hidden `Name = number` assignments
//! in casual prose (~35K tokens), runs the model on it, and verifies that every
//! `(Name, Value)` pair from the fact table is recovered correctly in the
//! generated output as `Name=number` lines.
//!
//! This is a HARD test — it exercises the compressed-KV indexer (since the
//! facts are spread across the prompt and only attendable via the compressor
//! at distances > sliding_window=128) AND the model's instruction-following.
//!
//! Pass condition: all 16 (Name=Value) pairs found in the generated output,
//! matching the hardcoded reference table exactly. Test prints PASS/FAIL plus
//! per-fact breakdown. Reference fact table matches antirez/ds4's
//! `tests/ds4_test.c:long_facts[]` (16 entries).
//!
//! Usage:
//!   v4f_long_context_test <model.hfq> [--ctx N] [--max-gen N] [--moe 0|1]
//!
//! Default `--ctx 65536` (the prompt is ~35K tokens; need ample room for
//! generation + indexer state). Default `--max-gen 256`.

use hipfire_arch_deepseek4::{forward::decode_step, DeepseekV4, DeepseekV4State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::path::Path;
use std::time::Instant;

const PROMPT_PATH: &str =
    "benchmarks/quality-baselines/ds4/long-context/story_prompt.txt";

/// The 16-entry fact table from antirez/ds4 `tests/ds4_test.c`.
/// MUST match upstream byte-for-byte — these are the values the prompt teaches
/// the model.
const FACTS: &[(&str, u32)] = &[
    ("Bob",   34),
    ("Alice", 52),
    ("Clara", 71),
    ("Diego", 93),
    ("Elena", 16),
    ("Felix", 88),
    ("Greta", 47),
    ("Hugo",  29),
    ("Iris",  64),
    ("Jonas", 12),
    ("Kira",  81),
    ("Leo",   39),
    ("Marta", 76),
    ("Nadia", 23),
    ("Owen",  58),
    ("Priya", 97),
];

fn argmax(logits: &[f32]) -> u32 {
    let mut best_i: u32 = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v { best_v = v; best_i = i as u32; }
    }
    best_i
}

/// Parse `Name=Value` from the generated text. Tolerates whitespace + minor
/// formatting noise. Returns the map of Name → recovered value.
fn parse_assignments(text: &str) -> std::collections::HashMap<String, u32> {
    let mut out = std::collections::HashMap::new();
    for line in text.lines() {
        let line = line.trim().trim_start_matches('*').trim_start_matches('-').trim();
        // Tolerate "Name = 42" with optional spaces.
        let eq = match line.find('=') { Some(i) => i, None => continue };
        let name = line[..eq].trim();
        if name.is_empty() { continue; }
        let val_str: String = line[eq + 1..].trim()
            .chars().take_while(|c| c.is_ascii_digit()).collect();
        if val_str.is_empty() { continue; }
        let val: u32 = match val_str.parse() { Ok(v) => v, Err(_) => continue };
        // Use first occurrence only — if the model repeats, the first answer wins.
        out.entry(name.to_string()).or_insert(val);
    }
    out
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect(
        "usage: v4f_long_context_test <model.hfq> [--ctx N] [--max-gen N] [--moe 0|1]");
    let mut ctx_size: usize = 65536;
    let mut max_gen: usize = 256;
    let mut use_moe = true;
    while let Some(flag) = args.next() {
        let val = args.next().expect("flag missing value");
        match flag.as_str() {
            "--ctx" => ctx_size = val.parse().map_err(|e| format!("ctx: {e}"))?,
            "--max-gen" => max_gen = val.parse().map_err(|e| format!("max-gen: {e}"))?,
            "--moe" => use_moe = val == "1",
            _ => return Err(format!("unknown flag: {flag}")),
        }
    }

    let prompt_path = std::env::var("HIPFIRE_LONG_CONTEXT_PROMPT")
        .unwrap_or_else(|_| PROMPT_PATH.to_string());
    let prompt_text = std::fs::read_to_string(&prompt_path)
        .map_err(|e| format!("read prompt {prompt_path}: {e}"))?;
    eprintln!("Loaded prompt: {} bytes from {prompt_path}", prompt_text.len());

    eprintln!("Loading V4F from {model_path}...");
    let mut hfq = HfqFile::open(Path::new(&model_path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .ok_or_else(|| "tokenizer not found in HFQ metadata".to_string())?;
    let eos_id = tokenizer.eos_id;

    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    if use_moe {
        std::env::set_var("HIPFIRE_V4F_UPLOAD_EXPERTS", "1");
        std::env::set_var("HIPFIRE_V4F_MOE", "1");
    }
    std::env::set_var("HIPFIRE_V4F_ATTN", "swa");
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;

    // The prompt file already has chat-template tokens baked in (BOS, system
    // text, <｜User｜>, content, <｜Assistant｜>). Encode raw — tokenizer
    // recognizes the special-token literals.
    let prompt_tokens = tokenizer.encode(&prompt_text);
    eprintln!("Encoded prompt: {} tokens (ctx_size={ctx_size})", prompt_tokens.len());
    if prompt_tokens.len() + max_gen + 16 > ctx_size {
        return Err(format!("prompt+max_gen ({}+{}) exceeds ctx={ctx_size}",
            prompt_tokens.len(), max_gen));
    }

    let mut state = DeepseekV4State::new(&cfg)?;
    let mut pos: u32 = 0;
    let mut last_logits: Vec<f32> = Vec::new();

    let t_prefill = Instant::now();
    eprintln!("Prefill: {} tokens...", prompt_tokens.len());
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        last_logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos)?;
        pos += 1;
        if (i + 1) % 1024 == 0 {
            eprintln!("  ... {i} / {}  ({:.1} tok/s)",
                prompt_tokens.len(),
                (i + 1) as f64 / t_prefill.elapsed().as_secs_f64());
        }
    }
    eprintln!("Prefill done in {:?}", t_prefill.elapsed());

    let t_gen = Instant::now();
    let mut gen_ids: Vec<u32> = Vec::with_capacity(max_gen);
    for _ in 0..max_gen {
        let next = argmax(&last_logits);
        gen_ids.push(next);
        if next == eos_id { break; }
        last_logits = decode_step(&cfg, &weights, &mut state, &mut gpu, next, pos)?;
        pos += 1;
    }
    let gen_text = tokenizer.decode(&gen_ids);
    eprintln!("Generation: {} tokens in {:?} ({:.1} tok/s)",
        gen_ids.len(), t_gen.elapsed(),
        gen_ids.len() as f64 / t_gen.elapsed().as_secs_f64());
    eprintln!("\n--- Generated output ---");
    eprintln!("{gen_text}");
    eprintln!("--- end ---\n");

    let recovered = parse_assignments(&gen_text);
    let mut correct = 0usize;
    let mut wrong = 0usize;
    let mut missing = 0usize;
    eprintln!("Per-fact check:");
    for &(name, want) in FACTS {
        match recovered.get(name) {
            Some(&got) if got == want => {
                eprintln!("  ✓ {name:<6} = {got}");
                correct += 1;
            }
            Some(&got) => {
                eprintln!("  ✗ {name:<6} = {got}    (expected {want})");
                wrong += 1;
            }
            None => {
                eprintln!("  · {name:<6} = ?        (expected {want}, not found)");
                missing += 1;
            }
        }
    }

    let total = FACTS.len();
    eprintln!("\nResult: {correct}/{total} correct, {wrong} wrong, {missing} missing");
    if correct == total {
        eprintln!("PASS");
        Ok(())
    } else {
        Err(format!("FAIL: only {correct}/{total} facts recovered"))
    }
}
