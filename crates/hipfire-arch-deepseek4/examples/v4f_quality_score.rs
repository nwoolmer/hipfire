//! V4F quality scorer — hipfire equivalent of antirez/ds4's score_official.c.
//!
//! Loads a quant + a JSONL of (id, prompt, target) cases, teacher-forces each
//! prompt, then per target token records:
//!   - greedy argmax (for first-token-match + greedy LCP)
//!   - logprob of the target token (for NLL)
//!
//! Output is one summary line per case to stdout (TSV), plus a final summary
//! to stderr matching antirez's `score_official` output format:
//!
//!   id  prompt_tokens  target_tokens  nll  avg_nll  first_match  greedy_lcp
//!
//! Methodology matches antirez/ds4 `gguf-tools/quality-testing/score_official.c`
//! line-by-line. See docs/plans/antirez-ds4-reference.md §3b for the contract.
//!
//! Usage:
//!   v4f_quality_score <model.hfq> <cases.jsonl> [--ctx N] [--moe 0|1] \
//!       [--out PATH]
//!
//! The cases JSONL must contain one object per line with at least:
//!   {"id": "case_000", "prompt": "...", "target": "..."}
//!
//! Reference baseline (antirez published):
//!   Q4 baseline avg_nll = 0.177358
//!   Q4 imatrix  avg_nll = 0.173895 (-1.95% relative)
//! Antirez does NOT publish a Q2 reference — our runs establish that.

use hipfire_arch_deepseek4::{forward::decode_step, DeepseekV4, DeepseekV4State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::io::{BufRead, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

fn parse_jsonl_line(line: &str) -> Option<(String, String, String)> {
    // Minimal JSON object parser for `{"id": "...", "prompt": "...", "target": "..."}`.
    // Avoids pulling in serde_json as a per-example dep. Handles standard
    // backslash-escapes (\n, \t, \", \\, \uXXXX); errors out on anything weird.
    fn skip_ws(b: &[u8], i: &mut usize) {
        while *i < b.len() && matches!(b[*i], b' ' | b'\t' | b'\n' | b'\r') {
            *i += 1;
        }
    }
    fn expect_byte(b: &[u8], i: &mut usize, c: u8) -> Option<()> {
        skip_ws(b, i);
        if *i >= b.len() || b[*i] != c { return None; }
        *i += 1;
        Some(())
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
                // Find end of UTF-8 char.
                let len = utf8_char_len(c)?;
                if *i + len > b.len() { return None; }
                out.push_str(std::str::from_utf8(&b[*i..*i + len]).ok()?);
                *i += len;
            }
        }
        None
    }
    fn utf8_char_len(b: u8) -> Option<usize> {
        if b & 0x80 == 0 { Some(1) }
        else if b & 0xE0 == 0xC0 { Some(2) }
        else if b & 0xF0 == 0xE0 { Some(3) }
        else if b & 0xF8 == 0xF0 { Some(4) }
        else { None }
    }

    let b = line.as_bytes();
    let mut i = 0;
    expect_byte(b, &mut i, b'{')?;
    let mut id = None;
    let mut prompt = None;
    let mut target = None;
    loop {
        skip_ws(b, &mut i);
        if i < b.len() && b[i] == b'}' { break; }
        let key = parse_string(b, &mut i)?;
        expect_byte(b, &mut i, b':')?;
        let val = parse_string(b, &mut i)?;
        match key.as_str() {
            "id" => id = Some(val),
            "prompt" => prompt = Some(val),
            "target" => target = Some(val),
            _ => {} // ignore unknown keys
        }
        skip_ws(b, &mut i);
        if i < b.len() && b[i] == b',' { i += 1; continue; }
        break;
    }
    Some((id?, prompt?, target?))
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_i: u32 = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v { best_v = v; best_i = i as u32; }
    }
    best_i
}

/// Returns log_softmax[target] = logits[target] - log_sum_exp(logits).
fn target_logprob(logits: &[f32], target: u32) -> f64 {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut lse = 0.0f64;
    for &x in logits {
        lse += ((x as f64) - max_l).exp();
    }
    let log_z = lse.ln() + max_l;
    logits[target as usize] as f64 - log_z
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect(
        "usage: v4f_quality_score <model.hfq> <cases.jsonl> [--ctx N] [--moe 0|1] [--out PATH]");
    let cases_path = args.next().expect(
        "usage: v4f_quality_score <model.hfq> <cases.jsonl> [--ctx N] [--moe 0|1] [--out PATH]");
    let mut ctx_size: usize = 4096;
    let mut use_moe = true;
    let mut out_path: Option<String> = None;
    while let Some(flag) = args.next() {
        let val = args.next().expect("flag missing value");
        match flag.as_str() {
            "--ctx" => ctx_size = val.parse().map_err(|e| format!("ctx: {e}"))?,
            "--moe" => use_moe = val == "1",
            "--out" => out_path = Some(val),
            _ => return Err(format!("unknown flag: {flag}")),
        }
    }

    eprintln!("Loading V4F from {model_path}...");
    let mut hfq = HfqFile::open(Path::new(&model_path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .ok_or_else(|| "tokenizer not found in HFQ metadata".to_string())?;

    // Chat template ids — antirez's collect_official.py uses
    // thinking={"type":"disabled"}, which for V4F means NO <think>...</think>.
    let lookup_id = |s: &str| -> Option<u32> {
        let ids = tokenizer.encode(s);
        if ids.len() == 1 { Some(ids[0]) } else { None }
    };
    let bos_tok  = lookup_id("<｜begin▁of▁sentence｜>");
    let user_tok = lookup_id("<｜User｜>");
    let asst_tok = lookup_id("<｜Assistant｜>");
    eprintln!("Chat tokens: bos={bos_tok:?} user={user_tok:?} assistant={asst_tok:?}");
    if user_tok.is_none() || asst_tok.is_none() {
        return Err("missing chat tokens — quality scoring requires them".into());
    }

    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    if use_moe {
        std::env::set_var("HIPFIRE_V4F_UPLOAD_EXPERTS", "1");
        std::env::set_var("HIPFIRE_V4F_MOE", "1");
    }
    std::env::set_var("HIPFIRE_V4F_ATTN", "swa");
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;

    let cases_file = std::fs::File::open(&cases_path)
        .map_err(|e| format!("open cases {cases_path}: {e}"))?;
    let reader = std::io::BufReader::new(cases_file);

    let stdout = std::io::stdout();
    let mut out_handle: Box<dyn Write> = match out_path.as_deref() {
        Some(p) => Box::new(BufWriter::new(std::fs::File::create(p)
            .map_err(|e| format!("create {p}: {e}"))?)),
        None => Box::new(BufWriter::new(stdout.lock())),
    };

    writeln!(out_handle, "id\tprompt_tokens\ttarget_tokens\tnll\tavg_nll\tfirst_match\tgreedy_lcp")
        .map_err(|e| format!("write header: {e}"))?;

    let mut total_nll = 0.0f64;
    let mut total_tokens: usize = 0;
    let mut total_lcp: usize = 0;
    let mut first_matches: usize = 0;
    let mut n_cases: usize = 0;
    let t_start = Instant::now();

    for (line_no, line_res) in reader.lines().enumerate() {
        let line = line_res.map_err(|e| format!("read line {line_no}: {e}"))?;
        if line.trim().is_empty() { continue; }
        let (id, prompt, target) = parse_jsonl_line(&line)
            .ok_or_else(|| format!("parse jsonl line {line_no}: {line}"))?;

        // Build the prompt token stream with the V4F chat template, no thinking.
        let mut prompt_tokens: Vec<u32> = Vec::new();
        if let Some(b) = bos_tok { prompt_tokens.push(b); }
        if let Some(u) = user_tok { prompt_tokens.push(u); }
        prompt_tokens.extend(tokenizer.encode(&prompt));
        if let Some(a) = asst_tok { prompt_tokens.push(a); }
        let target_tokens: Vec<u32> = tokenizer.encode(&target);

        if prompt_tokens.len() + target_tokens.len() + 1 >= ctx_size {
            eprintln!("{id}: prompt+target ({}+{}) exceeds ctx={ctx_size}, skipping",
                prompt_tokens.len(), target_tokens.len());
            continue;
        }

        // Fresh state per case. This is the part where having a long-running
        // session would help — we'd rewind to a checkpoint. For 100 cases at
        // a few hundred prompt tokens each, fresh-per-case is acceptable.
        let mut state = DeepseekV4State::new(&cfg)?;
        let mut pos: u32 = 0;
        let mut last_logits: Vec<f32> = Vec::new();

        // Teacher-force the prompt. The last decode_step's logits predict the
        // FIRST target token.
        for &tok in &prompt_tokens {
            last_logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos)?;
            pos += 1;
        }

        let mut nll = 0.0f64;
        let mut lcp = 0usize;
        let mut still_matching = true;
        let mut first_match = false;

        for (i, &tgt) in target_tokens.iter().enumerate() {
            let greedy = argmax(&last_logits);
            if i == 0 { first_match = greedy == tgt; }
            if still_matching && greedy == tgt {
                lcp += 1;
            } else {
                still_matching = false;
            }
            let lp = target_logprob(&last_logits, tgt);
            nll += -lp;
            // Teacher-force advance.
            last_logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tgt, pos)?;
            pos += 1;
        }

        let avg = if !target_tokens.is_empty() {
            nll / target_tokens.len() as f64
        } else { 0.0 };
        writeln!(out_handle, "{id}\t{}\t{}\t{:.9}\t{:.9}\t{}\t{lcp}",
            prompt_tokens.len(), target_tokens.len(), nll, avg,
            if first_match { 1 } else { 0 })
            .map_err(|e| format!("write row: {e}"))?;
        out_handle.flush().ok();

        n_cases += 1;
        total_nll += nll;
        total_tokens += target_tokens.len();
        total_lcp += lcp;
        if first_match { first_matches += 1; }

        eprintln!("[{n_cases:>3}] {id} prompt={} target={} avg_nll={avg:.6} lcp={lcp} fm={}",
            prompt_tokens.len(), target_tokens.len(),
            if first_match { 1 } else { 0 });
    }

    let elapsed = t_start.elapsed();
    let overall_avg = if total_tokens > 0 {
        total_nll / total_tokens as f64
    } else { 0.0 };
    let avg_lcp = if n_cases > 0 {
        total_lcp as f64 / n_cases as f64
    } else { 0.0 };
    eprintln!();
    eprintln!("Summary:");
    eprintln!("  cases:              {n_cases}");
    eprintln!("  target tokens:      {total_tokens}");
    eprintln!("  avg_nll (overall):  {overall_avg:.9}");
    eprintln!("  first_match:        {first_matches} / {n_cases}");
    eprintln!("  avg greedy lcp:     {avg_lcp:.3}");
    eprintln!("  total time:         {elapsed:?}");
    eprintln!();
    eprintln!("antirez published Q4 reference numbers (for comparison):");
    eprintln!("  old Q4:         avg_nll = 0.177358   (cases=100)");
    eprintln!("  Q4 imatrix:     avg_nll = 0.173895   (-1.95%)");
    eprintln!("antirez does NOT publish a Q2 reference — these are first-of-kind.");

    Ok(())
}
