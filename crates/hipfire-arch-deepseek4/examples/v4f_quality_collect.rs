//! V4F quality continuation collector — equivalent of antirez/ds4's
//! `collect_official.py` but using a local hipfire model as the reference
//! instead of the DeepSeek API.
//!
//! For each prompt in a JSONL, generates a greedy continuation (temperature 0,
//! no thinking) of up to `--max-tokens N` (default 24, matching antirez's
//! configuration) and emits a new JSONL with the `target` field populated.
//!
//! The output is consumable by `v4f_quality_score`.
//!
//! When the DeepSeek API is available the proper reference is upstream V4F-Flash
//! greedy continuations (see antirez/ds4 `gguf-tools/quality-testing/collect_official.py`).
//! In its absence, using our best local quant as reference gives us cross-quant
//! AGREEMENT numbers — useful for relative comparison but not directly
//! comparable to antirez's published 0.173895 baseline.
//!
//! Usage:
//!   v4f_quality_collect <reference-model.hfq> <prompts.jsonl> <out-cases.jsonl> \
//!       [--ctx N] [--max-tokens N] [--moe 0|1]

use hipfire_arch_deepseek4::{forward::decode_step, DeepseekV4, DeepseekV4State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::io::{BufRead, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

fn parse_prompt_jsonl(line: &str) -> Option<(String, String)> {
    // Accepts both {"id": "...", "prompt": "..."} and
    // {"id": "...", "prompt": "...", "target": "..."} (ignoring target).
    let b = line.as_bytes();
    let mut i = 0;
    fn skip_ws(b: &[u8], i: &mut usize) {
        while *i < b.len() && matches!(b[*i], b' ' | b'\t' | b'\n' | b'\r') { *i += 1; }
    }
    fn parse_string(b: &[u8], i: &mut usize) -> Option<String> {
        skip_ws(b, i);
        if *i >= b.len() || b[*i] != b'"' { return None; }
        *i += 1;
        let mut out = String::new();
        while *i < b.len() {
            let c = b[*i];
            if c == b'"' { *i += 1; return Some(out); }
            if c == b'\\' {
                *i += 1;
                if *i >= b.len() { return None; }
                match b[*i] {
                    b'"'  => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/'  => out.push('/'),
                    b'n'  => out.push('\n'),
                    b'r'  => out.push('\r'),
                    b't'  => out.push('\t'),
                    b'b'  => out.push('\u{0008}'),
                    b'f'  => out.push('\u{000C}'),
                    b'u'  => {
                        if *i + 4 >= b.len() { return None; }
                        let hex = std::str::from_utf8(&b[*i + 1..*i + 5]).ok()?;
                        let cp = u32::from_str_radix(hex, 16).ok()?;
                        out.push(char::from_u32(cp)?);
                        *i += 4;
                    }
                    _ => return None,
                }
                *i += 1;
            } else {
                let len = if c & 0x80 == 0 { 1 }
                    else if c & 0xE0 == 0xC0 { 2 }
                    else if c & 0xF0 == 0xE0 { 3 }
                    else if c & 0xF8 == 0xF0 { 4 }
                    else { return None; };
                if *i + len > b.len() { return None; }
                out.push_str(std::str::from_utf8(&b[*i..*i + len]).ok()?);
                *i += len;
            }
        }
        None
    }
    fn expect(b: &[u8], i: &mut usize, c: u8) -> Option<()> {
        skip_ws(b, i);
        if *i >= b.len() || b[*i] != c { return None; }
        *i += 1;
        Some(())
    }

    expect(b, &mut i, b'{')?;
    let mut id = None;
    let mut prompt = None;
    loop {
        skip_ws(b, &mut i);
        if i < b.len() && b[i] == b'}' { break; }
        let key = parse_string(b, &mut i)?;
        expect(b, &mut i, b':')?;
        let val = parse_string(b, &mut i)?;
        match key.as_str() {
            "id" => id = Some(val),
            "prompt" => prompt = Some(val),
            _ => {} // ignore
        }
        skip_ws(b, &mut i);
        if i < b.len() && b[i] == b',' { i += 1; continue; }
        break;
    }
    Some((id?, prompt?))
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"'  => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_i: u32 = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v { best_v = v; best_i = i as u32; }
    }
    best_i
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect(
        "usage: v4f_quality_collect <model> <prompts.jsonl> <out-cases.jsonl> [--ctx N] [--max-tokens N] [--moe 0|1]");
    let prompts_path = args.next().expect(
        "usage: v4f_quality_collect <model> <prompts.jsonl> <out-cases.jsonl> [--ctx N] [--max-tokens N] [--moe 0|1]");
    let out_path = args.next().expect(
        "usage: v4f_quality_collect <model> <prompts.jsonl> <out-cases.jsonl> [--ctx N] [--max-tokens N] [--moe 0|1]");
    let mut ctx_size: usize = 4096;
    let mut max_tokens: usize = 24;
    let mut use_moe = true;
    while let Some(flag) = args.next() {
        let val = args.next().expect("flag missing value");
        match flag.as_str() {
            "--ctx" => ctx_size = val.parse().map_err(|e| format!("ctx: {e}"))?,
            "--max-tokens" => max_tokens = val.parse().map_err(|e| format!("max-tokens: {e}"))?,
            "--moe" => use_moe = val == "1",
            _ => return Err(format!("unknown flag: {flag}")),
        }
    }

    eprintln!("Loading V4F reference model from {model_path}...");
    let mut hfq = HfqFile::open(Path::new(&model_path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .ok_or_else(|| "tokenizer not found in HFQ metadata".to_string())?;

    let lookup_id = |s: &str| -> Option<u32> {
        let ids = tokenizer.encode(s);
        if ids.len() == 1 { Some(ids[0]) } else { None }
    };
    let bos_tok  = lookup_id("<｜begin▁of▁sentence｜>");
    let user_tok = lookup_id("<｜User｜>");
    let asst_tok = lookup_id("<｜Assistant｜>");
    let eos_tok  = lookup_id("<｜end▁of▁sentence｜>").unwrap_or(tokenizer.eos_id);
    eprintln!("Chat tokens: bos={bos_tok:?} user={user_tok:?} assistant={asst_tok:?} eos={eos_tok}");
    if user_tok.is_none() || asst_tok.is_none() {
        return Err("missing chat tokens".into());
    }

    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    if use_moe {
        std::env::set_var("HIPFIRE_V4F_UPLOAD_EXPERTS", "1");
        std::env::set_var("HIPFIRE_V4F_MOE", "1");
    }
    std::env::set_var("HIPFIRE_V4F_ATTN", "swa");
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;

    let prompts_file = std::fs::File::open(&prompts_path)
        .map_err(|e| format!("open prompts {prompts_path}: {e}"))?;
    let reader = std::io::BufReader::new(prompts_file);
    let out_file = std::fs::File::create(&out_path)
        .map_err(|e| format!("create {out_path}: {e}"))?;
    let mut out_w = BufWriter::new(out_file);

    let t_start = Instant::now();
    let mut n: usize = 0;
    for (line_no, line_res) in reader.lines().enumerate() {
        let line = line_res.map_err(|e| format!("read line {line_no}: {e}"))?;
        if line.trim().is_empty() { continue; }
        let (id, prompt) = parse_prompt_jsonl(&line)
            .ok_or_else(|| format!("parse line {line_no}: {line}"))?;

        // Build chat-template token stream, thinking disabled.
        let mut prompt_tokens: Vec<u32> = Vec::new();
        if let Some(b) = bos_tok { prompt_tokens.push(b); }
        if let Some(u) = user_tok { prompt_tokens.push(u); }
        prompt_tokens.extend(tokenizer.encode(&prompt));
        if let Some(a) = asst_tok { prompt_tokens.push(a); }

        if prompt_tokens.len() + max_tokens + 1 >= ctx_size {
            eprintln!("{id}: prompt+max_tokens exceeds ctx={ctx_size}, skipping");
            continue;
        }

        let mut state = DeepseekV4State::new(&cfg)?;
        let mut pos: u32 = 0;
        let mut last_logits: Vec<f32> = Vec::new();
        for &tok in &prompt_tokens {
            last_logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos)?;
            pos += 1;
        }
        let mut gen_ids: Vec<u32> = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            let next = argmax(&last_logits);
            if next == eos_tok { break; }
            gen_ids.push(next);
            last_logits = decode_step(&cfg, &weights, &mut state, &mut gpu, next, pos)?;
            pos += 1;
        }

        let target_text = tokenizer.decode(&gen_ids);
        writeln!(out_w, "{{\"id\":\"{}\",\"prompt\":\"{}\",\"target\":\"{}\"}}",
            json_escape(&id), json_escape(&prompt), json_escape(&target_text))
            .map_err(|e| format!("write: {e}"))?;
        out_w.flush().ok();
        n += 1;
        eprintln!("[{n:>3}] {id} prompt_tokens={} gen={} target=\"{}\"",
            prompt_tokens.len(), gen_ids.len(),
            target_text.chars().take(64).collect::<String>());
    }
    eprintln!("\nDone. {n} cases written to {out_path} in {:?}", t_start.elapsed());
    Ok(())
}
