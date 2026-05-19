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

/// Parsed case from the JSONL input.
/// `target_bytes_per_step` is None when the JSONL uses the legacy text-only
/// `target` field — in that case the scorer falls back to tokenizer.encode(target).
struct ParsedCase {
    id: String,
    prompt: String,
    /// Set when JSONL provides `target_token_bytes`: a list of byte arrays, one
    /// per token from the upstream API's `steps[].token.bytes` field. Using
    /// these directly avoids re-tokenization boundary drift.
    target_bytes_per_step: Option<Vec<Vec<u8>>>,
    /// Set when JSONL uses the legacy `target` (assembled text) field.
    target_text: Option<String>,
}

fn parse_jsonl_line(line: &str) -> Option<ParsedCase> {
    // Minimal JSON object parser supporting:
    //   {"id": "...", "prompt": "...", "target": "..."}                  (legacy)
    //   {"id": "...", "prompt": "...", "target_token_bytes": [[65,100,97], ...]}
    //
    // For target_token_bytes, expects an array of arrays of integers 0-255.
    // Other keys ignored. Handles standard backslash-escapes in strings.
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

    fn parse_uint(b: &[u8], i: &mut usize) -> Option<u32> {
        skip_ws(b, i);
        let start = *i;
        while *i < b.len() && b[*i].is_ascii_digit() { *i += 1; }
        if *i == start { return None; }
        std::str::from_utf8(&b[start..*i]).ok()?.parse().ok()
    }
    fn parse_byte_array(b: &[u8], i: &mut usize) -> Option<Vec<u8>> {
        skip_ws(b, i);
        if *i >= b.len() || b[*i] != b'[' { return None; }
        *i += 1;
        let mut out = Vec::new();
        loop {
            skip_ws(b, i);
            if *i < b.len() && b[*i] == b']' { *i += 1; return Some(out); }
            let v = parse_uint(b, i)?;
            if v > 255 { return None; }
            out.push(v as u8);
            skip_ws(b, i);
            if *i < b.len() && b[*i] == b',' { *i += 1; continue; }
        }
    }
    fn parse_byte_array_array(b: &[u8], i: &mut usize) -> Option<Vec<Vec<u8>>> {
        skip_ws(b, i);
        if *i >= b.len() || b[*i] != b'[' { return None; }
        *i += 1;
        let mut out = Vec::new();
        loop {
            skip_ws(b, i);
            if *i < b.len() && b[*i] == b']' { *i += 1; return Some(out); }
            let arr = parse_byte_array(b, i)?;
            out.push(arr);
            skip_ws(b, i);
            if *i < b.len() && b[*i] == b',' { *i += 1; continue; }
        }
    }
    /// Skip an arbitrary JSON value when we don't care about it (numbers,
    /// strings, arrays, objects, true/false/null). Used to step over unknown
    /// keys when the value isn't one we explicitly parse.
    fn skip_value(b: &[u8], i: &mut usize) -> Option<()> {
        skip_ws(b, i);
        if *i >= b.len() { return None; }
        match b[*i] {
            b'"' => { let _ = parse_string(b, i)?; }
            b'[' | b'{' => {
                let open = b[*i]; let close = if open == b'[' { b']' } else { b'}' };
                let mut depth = 1; *i += 1;
                while *i < b.len() && depth > 0 {
                    if b[*i] == b'"' { let _ = parse_string(b, i); continue; }
                    if b[*i] == open { depth += 1; }
                    else if b[*i] == close { depth -= 1; }
                    *i += 1;
                }
            }
            _ => {
                // Number / bool / null — scan to next , or } or ]
                while *i < b.len() && b[*i] != b',' && b[*i] != b'}' && b[*i] != b']' { *i += 1; }
            }
        }
        Some(())
    }

    let b = line.as_bytes();
    let mut i = 0;
    expect_byte(b, &mut i, b'{')?;
    let mut id = None;
    let mut prompt = None;
    let mut target_text = None;
    let mut target_bytes_per_step = None;
    loop {
        skip_ws(b, &mut i);
        if i < b.len() && b[i] == b'}' { break; }
        let key = parse_string(b, &mut i)?;
        expect_byte(b, &mut i, b':')?;
        match key.as_str() {
            "id" => id = Some(parse_string(b, &mut i)?),
            "prompt" => prompt = Some(parse_string(b, &mut i)?),
            "target" => target_text = Some(parse_string(b, &mut i)?),
            "target_token_bytes" => {
                target_bytes_per_step = Some(parse_byte_array_array(b, &mut i)?);
            }
            _ => { skip_value(b, &mut i)?; }
        }
        skip_ws(b, &mut i);
        if i < b.len() && b[i] == b',' { i += 1; continue; }
        break;
    }
    Some(ParsedCase {
        id: id?,
        prompt: prompt?,
        target_bytes_per_step,
        target_text,
    })
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
    // We build the prompt as a single string per the DeepSeek-V4 reference
    // encoding (huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/encoding/encoding_dsv4.py)
    // and let the tokenizer handle the special-token recognition. This
    // matches encode_messages(messages=[{"role":"user","content":X}],
    // thinking_mode="chat") byte-for-byte:
    //
    //   <｜begin▁of▁sentence｜><｜User｜>{content}<｜Assistant｜></think>
    //
    // "chat" mode in the reference impl = API's thinking={"type":"disabled"},
    // which is what antirez's test fixtures use.
    let bos_tok  = lookup_id("<｜begin▁of▁sentence｜>");
    let user_tok = lookup_id("<｜User｜>");
    let asst_tok = lookup_id("<｜Assistant｜>");
    let think_close_tok = lookup_id("</think>");
    eprintln!("Verifying chat tokens are single-token: bos={bos_tok:?} user={user_tok:?} assistant={asst_tok:?} </think>={think_close_tok:?}");
    if user_tok.is_none() || asst_tok.is_none() {
        return Err("missing chat tokens — quality scoring requires them".into());
    }
    if think_close_tok.is_none() {
        eprintln!("warning: </think> didn't single-token-encode — DeepSeek's tokenizer may split it. Continuing with full-string tokenize which handles this correctly.");
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

    let mut multi_token_warnings: usize = 0;
    for (line_no, line_res) in reader.lines().enumerate() {
        let line = line_res.map_err(|e| format!("read line {line_no}: {e}"))?;
        if line.trim().is_empty() { continue; }
        let case = parse_jsonl_line(&line)
            .ok_or_else(|| format!("parse jsonl line {line_no}: {line}"))?;
        let ParsedCase { id, prompt, target_bytes_per_step, target_text } = case;

        // Build the full prompt STRING per the DeepSeek-V4 reference encoder
        // (`encode_messages(thinking_mode="chat")`), then tokenize in one shot.
        let full_prompt = format!(
            "<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜></think>"
        );
        let prompt_tokens: Vec<u32> = tokenizer.encode(&full_prompt);

        // Resolve target tokens. Preferred path: use the upstream API's exact
        // byte sequences per step → each is a SINGLE token in DeepSeek's vocab,
        // we look it up by re-tokenizing the byte sequence (interpreted as UTF-8)
        // and expecting one token id back. Avoids re-tokenization boundary drift.
        // Fallback: tokenize the assembled `target` string.
        let target_tokens: Vec<u32> = if let Some(byte_steps) = target_bytes_per_step {
            let mut ids: Vec<u32> = Vec::with_capacity(byte_steps.len());
            for (step_idx, b) in byte_steps.iter().enumerate() {
                // Decode bytes as UTF-8 (replace invalid sequences — these
                // happen for tokens that are partial multi-byte chars).
                let s: String = String::from_utf8_lossy(b).into_owned();
                let encoded = tokenizer.encode(&s);
                if encoded.len() != 1 {
                    multi_token_warnings += 1;
                    if multi_token_warnings <= 5 {
                        eprintln!("warn: {id} step {step_idx} bytes={b:?} \
                                   decodes to {s:?} which tokenizes to {} tokens \
                                   ({encoded:?}) — using all of them",
                                  encoded.len());
                    }
                    ids.extend(encoded);
                } else {
                    ids.push(encoded[0]);
                }
            }
            ids
        } else if let Some(text) = target_text {
            tokenizer.encode(&text)
        } else {
            return Err(format!("case {id}: needs `target` or `target_token_bytes`"));
        };

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
    if multi_token_warnings > 0 {
        eprintln!();
        eprintln!("note: {multi_token_warnings} target-byte-step(s) tokenized to >1 token.");
        eprintln!("       Likely tokenizer-vocab disagreement between hipfire and upstream.");
        eprintln!("       Numbers above include those extra tokens; comparable but inflated.");
    }

    Ok(())
}
