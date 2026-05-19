//! V4F capability smoke eval — minimal port of antirez/ds4 `ds4-eval`.
//!
//! Runs a small fixed subset of GPQA Diamond / SuperGPQA / AIME 2025 / COMPSEC
//! questions through the model, parses the answer, and grades pass/fail.
//! Intentionally NOT a full ds4-eval port — the upstream runner uses
//! `max_tokens=16000` with thinking enabled, which would take ~30 hours on
//! this hardware. Smoke uses chat mode (thinking disabled) with `max_tokens=256`
//! so it runs in ~3 min/quant and serves as a regression sanity check, not a
//! capability benchmark.
//!
//! Grading logic mirrors ds4_eval.c:answer_matches:
//!   - Multi-choice: parse first letter after "Answer" pattern (fall back to
//!     last letter in output). Compare against expected letter.
//!   - Integer (AIME): parse first integer near "Answer" (fall back to last
//!     integer in output). String-compare against expected.
//!   - COMPSEC: parse line spec "N" / "N,M" / "N-M", check the model's set
//!     is a non-empty subset of expected accepted lines.
//!
//! Usage:
//!   v4f_eval_smoke <model.hfq> [--cases PATH] [--max-tokens N] [--moe 0|1]
//!
//! Defaults: cases = benchmarks/quality-baselines/ds4/eval/cases_smoke10.json,
//! max-tokens = 256, moe = 1.

use hipfire_arch_deepseek4::{forward::decode_step, DeepseekV4, DeepseekV4State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::path::Path;
use std::time::Instant;

const DEFAULT_CASES_PATH: &str =
    "benchmarks/quality-baselines/ds4/eval/cases_smoke10.json";

fn argmax(logits: &[f32]) -> u32 {
    let mut best_i: u32 = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v { best_v = v; best_i = i as u32; }
    }
    best_i
}

#[derive(Clone, Debug)]
struct Case {
    source: String,
    id: String,
    title: String,
    question: String,
    answer: String,
    choices: Vec<String>,
}

/// Parse the smoke JSON into a Vec<Case>. Hand-rolled to avoid pulling
/// serde_json into a per-example dep. Format is a JSON array of objects.
fn parse_cases(text: &str) -> Result<Vec<Case>, String> {
    let b = text.as_bytes();
    let mut i = 0;
    fn skip_ws(b: &[u8], i: &mut usize) {
        while *i < b.len() && matches!(b[*i], b' ' | b'\t' | b'\n' | b'\r') { *i += 1; }
    }
    fn expect(b: &[u8], i: &mut usize, c: u8) -> Result<(), String> {
        skip_ws(b, i);
        if *i >= b.len() || b[*i] != c {
            return Err(format!("expected {:?} at byte {}", c as char, *i));
        }
        *i += 1;
        Ok(())
    }
    fn parse_str(b: &[u8], i: &mut usize) -> Result<String, String> {
        skip_ws(b, i);
        if *i >= b.len() || b[*i] != b'"' {
            return Err(format!("expected string at byte {}", *i));
        }
        *i += 1;
        let mut out = String::new();
        while *i < b.len() {
            let c = b[*i];
            if c == b'"' { *i += 1; return Ok(out); }
            if c == b'\\' {
                *i += 1;
                if *i >= b.len() { return Err("unterminated escape".into()); }
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
                        if *i + 4 >= b.len() { return Err("bad \\u".into()); }
                        let hex = std::str::from_utf8(&b[*i + 1..*i + 5])
                            .map_err(|e| e.to_string())?;
                        let cp = u32::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                        if let Some(c) = char::from_u32(cp) {
                            out.push(c);
                        }
                        *i += 4;
                    }
                    _ => return Err("bad escape".into()),
                }
                *i += 1;
            } else {
                let len = if c & 0x80 == 0 { 1 }
                    else if c & 0xE0 == 0xC0 { 2 }
                    else if c & 0xF0 == 0xE0 { 3 }
                    else if c & 0xF8 == 0xF0 { 4 }
                    else { return Err("bad utf-8".into()); };
                if *i + len > b.len() { return Err("truncated utf-8".into()); }
                out.push_str(std::str::from_utf8(&b[*i..*i + len])
                    .map_err(|e| e.to_string())?);
                *i += len;
            }
        }
        Err("unterminated string".into())
    }
    fn parse_array_of_strings(b: &[u8], i: &mut usize) -> Result<Vec<String>, String> {
        skip_ws(b, i);
        if *i >= b.len() || b[*i] != b'[' {
            return Err(format!("expected [ at byte {}", *i));
        }
        *i += 1;
        let mut out = Vec::new();
        loop {
            skip_ws(b, i);
            if *i < b.len() && b[*i] == b']' { *i += 1; return Ok(out); }
            out.push(parse_str(b, i)?);
            skip_ws(b, i);
            if *i < b.len() && b[*i] == b',' { *i += 1; continue; }
        }
    }
    fn skip_value(b: &[u8], i: &mut usize) -> Result<(), String> {
        skip_ws(b, i);
        if *i >= b.len() { return Err("eof in value".into()); }
        match b[*i] {
            b'"' => { let _ = parse_str(b, i)?; }
            b'[' | b'{' => {
                let open = b[*i]; let close = if open == b'[' { b']' } else { b'}' };
                let mut depth = 1; *i += 1;
                while *i < b.len() && depth > 0 {
                    if b[*i] == b'"' { let _ = parse_str(b, i)?; continue; }
                    if b[*i] == open { depth += 1; }
                    else if b[*i] == close { depth -= 1; }
                    *i += 1;
                }
            }
            _ => while *i < b.len() && !matches!(b[*i], b',' | b'}' | b']') { *i += 1; },
        }
        Ok(())
    }

    expect(b, &mut i, b'[')?;
    let mut cases: Vec<Case> = Vec::new();
    loop {
        skip_ws(b, &mut i);
        if i < b.len() && b[i] == b']' { break; }
        expect(b, &mut i, b'{')?;
        let mut source = None;
        let mut case_id = None;
        let mut title = None;
        let mut question = None;
        let mut answer = None;
        let mut choices: Vec<String> = Vec::new();
        loop {
            skip_ws(b, &mut i);
            if i < b.len() && b[i] == b'}' { i += 1; break; }
            let key = parse_str(b, &mut i)?;
            expect(b, &mut i, b':')?;
            match key.as_str() {
                "source"   => source   = Some(parse_str(b, &mut i)?),
                "id"       => case_id  = Some(parse_str(b, &mut i)?),
                "title"    => title    = Some(parse_str(b, &mut i)?),
                "question" => question = Some(parse_str(b, &mut i)?),
                "answer"   => answer   = Some(parse_str(b, &mut i)?),
                "choices"  => choices  = parse_array_of_strings(b, &mut i)?,
                _ => skip_value(b, &mut i)?,
            }
            skip_ws(b, &mut i);
            if i < b.len() && b[i] == b',' { i += 1; continue; }
        }
        cases.push(Case {
            source: source.ok_or("missing source")?,
            id: case_id.ok_or("missing id")?,
            title: title.unwrap_or_default(),
            question: question.ok_or("missing question")?,
            answer: answer.ok_or("missing answer")?,
            choices,
        });
        skip_ws(b, &mut i);
        if i < b.len() && b[i] == b',' { i += 1; continue; }
    }
    Ok(cases)
}

// ---- Grading helpers, ported from antirez/ds4 ds4_eval.c ----

fn is_letter_boundary(before: char, after: char) -> bool {
    !before.is_ascii_alphabetic() && !after.is_ascii_alphabetic()
}

fn find_answer_letter(generated: &str, nchoices: usize) -> char {
    if nchoices == 0 { return '?'; }
    let max_answer = (b'A' + (nchoices as u8) - 1) as char;
    // Strip everything up to and including </think> if present.
    let visible = match generated.find("</think>") {
        Some(p) => &generated[p + 8..],
        None => generated,
    };
    // Prefer first valid letter after "Answer".
    let lower = visible.to_ascii_lowercase();
    if let Some(answer_pos) = lower.find("answer") {
        let window_end = (answer_pos + 96).min(visible.len());
        let window = &visible[answer_pos..window_end];
        let mut prev = ' ';
        for (i, c) in window.char_indices() {
            let up = c.to_ascii_uppercase();
            if up >= 'A' && up <= max_answer {
                let after = window[i + c.len_utf8()..].chars().next().unwrap_or(' ');
                if is_letter_boundary(prev, after) { return up; }
            }
            prev = c;
        }
    }
    // Fallback: last valid letter in visible region.
    let chars: Vec<char> = visible.chars().collect();
    for j in (0..chars.len()).rev() {
        let up = chars[j].to_ascii_uppercase();
        if up >= 'A' && up <= max_answer {
            let before = if j == 0 { ' ' } else { chars[j - 1] };
            let after = chars.get(j + 1).copied().unwrap_or(' ');
            if is_letter_boundary(before, after) { return up; }
        }
    }
    '?'
}

fn normalize_integer(s: &str) -> String {
    let trimmed = s.trim_start_matches('0');
    if trimmed.is_empty() && !s.is_empty() {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

fn find_integer_answer(generated: &str) -> String {
    let visible = match generated.find("</think>") {
        Some(p) => &generated[p + 8..],
        None => generated,
    };
    let lower = visible.to_ascii_lowercase();
    if let Some(p) = lower.find("answer") {
        let window_end = (p + 160).min(visible.len());
        let window = &visible[p..window_end];
        let mut start = None;
        for (i, c) in window.char_indices() {
            if c.is_ascii_digit() {
                if start.is_none() { start = Some(i); }
            } else if let Some(s) = start {
                return normalize_integer(&window[s..i]);
            }
        }
        if let Some(s) = start {
            return normalize_integer(&window[s..]);
        }
    }
    // Last integer in visible region.
    let mut last_start = None;
    let mut last_end = None;
    let mut cur_start = None;
    for (i, c) in visible.char_indices() {
        if c.is_ascii_digit() {
            if cur_start.is_none() { cur_start = Some(i); }
            last_end = Some(i + c.len_utf8());
        } else if cur_start.is_some() {
            last_start = cur_start;
            cur_start = None;
        }
    }
    if cur_start.is_some() { last_start = cur_start; }
    if let (Some(s), Some(e)) = (last_start, last_end) {
        return normalize_integer(&visible[s..e]);
    }
    "?".to_string()
}

/// Parse a line spec like "17", "11,18", or "17-20" into a set of line ints.
fn parse_line_spec(spec: &str) -> Option<Vec<u32>> {
    let bytes = spec.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    let mut any = false;
    while i < bytes.len() {
        while i < bytes.len() && !bytes[i].is_ascii_digit() { i += 1; }
        if i >= bytes.len() { break; }
        let s = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
        let a: u32 = match std::str::from_utf8(&bytes[s..i]).unwrap().parse() {
            Ok(v) => v, Err(_) => return None,
        };
        let mut b = a;
        if i < bytes.len() && bytes[i] == b'-' {
            i += 1;
            let s2 = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
            if s2 < i {
                b = match std::str::from_utf8(&bytes[s2..i]).unwrap().parse() {
                    Ok(v) => v, Err(_) => return None,
                };
            }
        }
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        for v in lo..=hi { out.push(v); any = true; }
    }
    if any { Some(out) } else { None }
}

fn find_compsec_answer(generated: &str) -> String {
    let visible = match generated.find("</think>") {
        Some(p) => &generated[p + 8..],
        None => generated,
    };
    let lower = visible.to_ascii_lowercase();
    // Look for "answer" then take everything to end-of-line.
    if let Some(p) = lower.find("answer") {
        let after_ans = &visible[p..];
        let line_end = after_ans.find('\n').unwrap_or(after_ans.len());
        return after_ans[..line_end].to_string();
    }
    // Fallback: last "digit-something" sequence on its own line.
    let last_line = visible.lines().rev()
        .find(|l| l.chars().any(|c| c.is_ascii_digit()))
        .unwrap_or("");
    last_line.to_string()
}

fn answer_matches(case: &Case, got: &str) -> bool {
    if !case.choices.is_empty() {
        // Multi-choice
        let expected = case.answer.chars().next().unwrap_or('?');
        let got_letter = got.chars().next().unwrap_or('?');
        return got_letter == expected;
    }
    if case.source == "COMPSEC" {
        let expected = match parse_line_spec(&case.answer) { Some(v) => v, None => return false };
        let got_set = match parse_line_spec(got) { Some(v) => v, None => return false };
        if got_set.is_empty() { return false; }
        for g in &got_set {
            if !expected.contains(g) { return false; }
        }
        return true;
    }
    // Integer
    let expected_norm = normalize_integer(case.answer.trim());
    let got_norm = normalize_integer(got.trim());
    got_norm == expected_norm
}

fn extract_answer(case: &Case, generated: &str) -> String {
    if !case.choices.is_empty() {
        find_answer_letter(generated, case.choices.len()).to_string()
    } else if case.source == "COMPSEC" {
        find_compsec_answer(generated)
    } else {
        find_integer_answer(generated)
    }
}

// ---- Prompt building ----

fn build_prompt_string(case: &Case) -> String {
    // Render the question with choices for multi-choice, plain for others.
    // Chat-mode framing: <bos><User>{rendered}<Assistant></think>.
    // Smoke mode does NOT enable thinking — model goes straight to the answer.
    let mut q = String::new();
    q.push_str(&case.question);
    if !case.choices.is_empty() {
        q.push('\n');
        for (i, ch) in case.choices.iter().enumerate() {
            let letter = (b'A' + i as u8) as char;
            q.push_str(&format!("\n{}. {}", letter, ch));
        }
        q.push_str("\n\nReply with: Answer X");
    } else if case.source == "AIME2025" {
        q.push_str("\n\nReply with the integer answer only, in the form: Answer N");
    } else if case.source == "COMPSEC" {
        q.push_str("\n\nReply with the answer line spec, in the form: Answer N or Answer N-M");
    }
    format!("<｜begin▁of▁sentence｜><｜User｜>{q}<｜Assistant｜></think>")
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect(
        "usage: v4f_eval_smoke <model.hfq> [--cases PATH] [--max-tokens N] [--moe 0|1]");
    let mut cases_path = DEFAULT_CASES_PATH.to_string();
    let mut max_tokens: usize = 256;
    let mut use_moe = true;
    while let Some(flag) = args.next() {
        let val = args.next().expect("flag missing value");
        match flag.as_str() {
            "--cases" => cases_path = val,
            "--max-tokens" => max_tokens = val.parse().map_err(|e| format!("max-tokens: {e}"))?,
            "--moe" => use_moe = val == "1",
            _ => return Err(format!("unknown flag: {flag}")),
        }
    }

    eprintln!("Loading cases from {cases_path}...");
    let cases_text = std::fs::read_to_string(&cases_path)
        .map_err(|e| format!("read cases {cases_path}: {e}"))?;
    let cases = parse_cases(&cases_text)?;
    eprintln!("  {} cases loaded", cases.len());

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

    let t_start = Instant::now();
    let mut pass = 0;
    let mut fail = 0;
    let mut by_source: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
    for (idx, case) in cases.iter().enumerate() {
        let prompt = build_prompt_string(case);
        let prompt_tokens = tokenizer.encode(&prompt);
        eprintln!("\n[{}/{}] {} {} ({} prompt tokens)",
            idx + 1, cases.len(), case.source, case.id, prompt_tokens.len());
        eprintln!("  title: {}", case.title.chars().take(80).collect::<String>());

        let mut state = DeepseekV4State::new(&cfg)?;
        let mut pos: u32 = 0;
        let mut last_logits: Vec<f32> = Vec::new();
        let t_case = Instant::now();
        for &tok in &prompt_tokens {
            last_logits = decode_step(&cfg, &weights, &mut state, &mut gpu, tok, pos)?;
            pos += 1;
        }
        let mut gen_ids: Vec<u32> = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            let next = argmax(&last_logits);
            gen_ids.push(next);
            if next == eos_id { break; }
            last_logits = decode_step(&cfg, &weights, &mut state, &mut gpu, next, pos)?;
            pos += 1;
        }
        let gen_text = tokenizer.decode(&gen_ids);
        let extracted = extract_answer(case, &gen_text);
        let ok = answer_matches(case, &extracted);
        eprintln!("  generated ({}t in {:?}): {:.120}",
            gen_ids.len(), t_case.elapsed(),
            gen_text.replace('\n', " ⏎ "));
        eprintln!("  extracted={extracted:?}  expected={:?}  {}",
            case.answer, if ok { "PASS" } else { "FAIL" });
        let e = by_source.entry(case.source.clone()).or_default();
        if ok { pass += 1; e.0 += 1; } else { fail += 1; e.1 += 1; }
    }
    let elapsed = t_start.elapsed();
    eprintln!("\n=== Summary ===");
    eprintln!("  total: {} cases in {elapsed:?}", cases.len());
    for (src, (p, f)) in &by_source {
        eprintln!("  {src:<14} {p:>3} pass / {} ({}%)",
            p + f, if p + f > 0 { 100 * p / (p + f) } else { 0 });
    }
    eprintln!("  overall: {pass}/{}  ({:.0}%)",
        pass + fail,
        if pass + fail > 0 { 100.0 * pass as f64 / (pass + fail) as f64 } else { 0.0 });
    Ok(())
}
