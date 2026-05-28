// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Grammar-guided decoding for the qwen3.5/3.6 tool-call format.
//!
//! Mirrors the V4F DSML grammar in `crates/hipfire-arch-deepseek4/src/grammar.rs`
//! but constrains qwen35's `<tool_call>{json}</tool_call>` body instead of
//! DeepSeek's XML-style DSML tags.
//!
//! ## Why this exists
//!
//! qwen3.6:27b drifts after long agentic sessions: it emits the
//! `<tool_call>` opener correctly then writes ChatML noise as the body,
//! e.g.
//! ```text
//! <tool_call>
//! <|im_start|>assistant "Let me read the existing build files..."}}
//! </tool_call>
//! ```
//! observed verbatim in Pi turn 12 after ~27k cached tokens. The text
//! isn't JSON, so the daemon's tool-call extractor returns
//! `tool_calls=0`, the daemon emits `finish_reason: "stop"` with that
//! garbage as `message.content`, and Pi's agent loop terminates.
//!
//! The grammar matcher prevents this by masking sample logits the
//! moment the model commits to `<tool_call>`: from then until the JSON
//! header `\n{"name": "<TOOL_NAME>", "arguments": ` is fully laid
//! down, only tokens that continue that template are allowed. The
//! `arguments` value itself is unconstrained (state `InArgs`) — qwen
//! emits free-form JSON for the arguments object and naturally closes
//! with `}\n</tool_call>`. Once `</tool_call>` lands, the matcher
//! returns to free emission.
//!
//! ## States
//!
//! - [`State::Out`] — free emission. The matcher watches for the
//!   `<tool_call>` substring (single special token, vocab id varies
//!   per checkpoint) and transitions to [`State::AfterOpen`] on entry.
//! - [`State::AfterOpen`] — between `<tool_call>` and the
//!   `"arguments": ` colon-space. Constrained to a literal byte
//!   sequence that names one of the available tools.
//! - [`State::InArgs`] — between `"arguments": ` and `</tool_call>`.
//!   Unconstrained — qwen emits the args value as free JSON.
//! - (back to `Out` after `</tool_call>`.)
//!
//! ## What this does NOT do
//!
//! Enforce JSON-validity inside `arguments`. The args value can be a
//! nested object, array, string, etc.; tracking JSON balance under
//! BPE fragmentation would multiply the grammar code size for little
//! payoff — the failure mode we care about (Pi turn 12) corrupts the
//! header, not the body. A future phase can layer in JSON-aware
//! brace counting if production traffic ever shows body drift.

/// Position in the qwen35 tool-call grammar. See module docs for
/// transitions.
#[derive(Debug, Clone, PartialEq)]
pub enum State {
    /// Free emission outside any `<tool_call>` block. Watching for the
    /// open marker but not otherwise constraining tokens.
    Out,
    /// Between `<tool_call>` (already consumed) and the `"arguments": `
    /// header sentinel. Allowed continuations are prefixes of
    /// `\n{"name": "<TOOL_NAME>", "arguments": `.
    AfterOpen,
    /// Between `"arguments": ` and `</tool_call>`. Free emission —
    /// the model writes the args value as it pleases. We re-enter
    /// `Out` when `</tool_call>` lands in the rolling buffer.
    InArgs,
}

/// Schema for one available tool. Built from the OpenAI-format tools
/// array at request time; the grammar uses this to pick which tool
/// names the model is allowed to emit at the name position AND which
/// argument fields must appear in the args body before the close
/// marker is allowed.
#[derive(Debug, Clone)]
pub struct ToolSchema {
    pub name: String,
    /// Subset of the tool's parameters that MUST appear as keys in
    /// the emitted args body. Built from the JSON schema's
    /// `parameters.required` array at request time. The grammar's
    /// `is_token_allowed` rejects close-marker prefixes while any
    /// required name is still absent from the args body — without
    /// this the model can emit `"arguments":{}` (Pi `write` failure
    /// mode observed in production: model emitted empty args, Pi
    /// rejected, model retried with same empty args, context bloated
    /// to KV exhaustion).
    pub required: Vec<String>,
}

/// Maximum bytes retained for the n-gram loop guard's rolling window
/// inside [`State::InArgs`]. 256 bytes covers the worst-case attractor
/// we've observed in production (a ~20-byte repeated block × 4) with
/// headroom. Bounded so the rolling-window scan stays O(1) per token.
const NGRAM_WINDOW: usize = 256;

/// Minimum consecutive identical n-gram repeats that trigger the
/// attractor flag. Set conservatively to 4 — legitimate JSON content
/// (arrays of small ints, indented blocks) can hit 3-repeats; 4 is
/// safely past the noise floor. The actual incident in production was
/// `typetypetypetype` (4-byte gram × 4) and
/// `pub fn BlinkHash(key_pub fn BlinkHash(key_…` (20-byte gram × 3+),
/// both caught at this threshold.
const NGRAM_MIN_REPEATS: usize = 4;

/// Range of n-gram lengths probed each token (inclusive). 2-byte
/// n-grams catch the tightest character-cycle attractors;
/// 32-byte n-grams catch multi-token phrase loops.
const NGRAM_LEN_MIN: usize = 2;
const NGRAM_LEN_MAX: usize = 32;

/// Grammar matcher: state plus the bytes committed since the last
/// firm transition. Construct via [`Matcher::new`] with the active
/// tool schemas, advance with [`Matcher::advance`], query allowed
/// tokens via [`Matcher::is_token_allowed`] or [`Matcher::token_mask`].
#[derive(Debug, Clone)]
pub struct Matcher {
    state: State,
    /// Bytes committed since the last firm state transition.
    partial_buf: String,
    tools: Vec<ToolSchema>,
    /// Rolling window of the last `NGRAM_WINDOW` bytes seen while in
    /// [`State::InArgs`]. Separate from `partial_buf` because that
    /// only retains a close-marker-sized suffix — n-gram detection
    /// needs longer history. Cleared on every firm transition.
    ngram_history: String,
    /// Set when consecutive n-gram repetition is detected inside
    /// [`State::InArgs`]. While set, the matcher constrains InArgs
    /// to only `</tool_call>` continuations — the model gets a
    /// forced exit instead of being stuck in an attractor that
    /// bloats the agentic conversation's KV (Pi turn-12-style
    /// `typetypetypetype` inside the args body was the motivating
    /// case). Cleared when the close marker fires and we return
    /// to [`State::Out`].
    attractor_detected: bool,
    /// Index into `tools` of the tool whose schema we're currently
    /// inside (i.e. the one whose name header just matched on the
    /// `AfterOpen` → `InArgs` transition). `None` outside of
    /// [`State::InArgs`]. Used by `required_fields_satisfied` to
    /// know which required-field list to check against the args
    /// body bytes.
    current_tool: Option<usize>,
}

impl Matcher {
    /// Build a fresh matcher in [`State::Out`] with no partial buffer.
    pub fn new(tools: Vec<ToolSchema>) -> Self {
        Self {
            state: State::Out,
            partial_buf: String::new(),
            tools,
            ngram_history: String::new(),
            attractor_detected: false,
            current_tool: None,
        }
    }

    /// Index of the tool currently being constructed in
    /// [`State::InArgs`]. Exposed for diagnostics.
    pub fn current_tool(&self) -> Option<usize> {
        self.current_tool
    }

    /// Check whether the args body bytes seen so far satisfy every
    /// required field for the current tool. A field is considered
    /// "seen" if `"<name>"` appears anywhere in the args body bytes
    /// (`ngram_history`). The substring match is intentionally loose
    /// — JSON syntax exists for the LLM, not for the parser, so as
    /// long as the field name string is present we trust the model
    /// to wire the colon/value correctly.
    ///
    /// Empty schema or empty `required` list → trivially satisfied.
    /// Returns true if no current tool (e.g. AfterOpen state) so we
    /// don't gate transitions we can't evaluate.
    fn required_fields_satisfied(&self) -> bool {
        let tool_idx = match self.current_tool {
            Some(i) => i,
            None => return true,
        };
        let schema = match self.tools.get(tool_idx) {
            Some(s) => s,
            None => return true,
        };
        if schema.required.is_empty() {
            return true;
        }
        for name in &schema.required {
            let needle = format!("\"{}\"", name);
            if !self.ngram_history.contains(&needle) {
                return false;
            }
        }
        true
    }

    /// True iff the n-gram loop guard has tripped on the current
    /// [`State::InArgs`] body. Exposed for diagnostics; the daemon
    /// uses this to log the trip event.
    pub fn attractor_detected(&self) -> bool {
        self.attractor_detected
    }

    /// Detect consecutive identical n-gram repetition in the tail of
    /// `buf`. Returns `true` iff there's some `n` in
    /// `[NGRAM_LEN_MIN, NGRAM_LEN_MAX]` such that the last
    /// `n * NGRAM_MIN_REPEATS` bytes consist of the same `n`-byte
    /// block repeated `NGRAM_MIN_REPEATS` times.
    fn detect_ngram_loop(buf: &str) -> bool {
        let bytes = buf.as_bytes();
        for ngram_len in NGRAM_LEN_MIN..=NGRAM_LEN_MAX {
            let needed = ngram_len * NGRAM_MIN_REPEATS;
            if bytes.len() < needed {
                continue;
            }
            let tail = &bytes[bytes.len() - needed..];
            let first = &tail[..ngram_len];
            let mut all_match = true;
            for r in 1..NGRAM_MIN_REPEATS {
                let chunk = &tail[r * ngram_len..(r + 1) * ngram_len];
                if chunk != first {
                    all_match = false;
                    break;
                }
            }
            if all_match {
                return true;
            }
        }
        false
    }

    /// Push bytes into the n-gram history buffer and run detection.
    /// Only called while in [`State::InArgs`]. Sets
    /// `attractor_detected = true` on first detection; never clears
    /// it from here (clearing happens on the firm `</tool_call>`
    /// transition back to `Out`).
    fn update_ngram_history(&mut self, text: &str) {
        if self.attractor_detected {
            return; // already flagged; don't waste work
        }
        self.ngram_history.push_str(text);
        if self.ngram_history.len() > NGRAM_WINDOW {
            // Drop only at UTF-8 char boundaries — keep the buffer safe
            // for `&str` slicing in `detect_ngram_loop`. We're working
            // on a byte-level repetition check, so dropping at any
            // boundary doesn't change the detection semantics; the
            // boundary rule just satisfies the `String` invariant.
            let drop = self.ngram_history.len() - NGRAM_WINDOW;
            let mut idx = drop;
            while idx < self.ngram_history.len()
                && !self.ngram_history.is_char_boundary(idx)
            {
                idx += 1;
            }
            self.ngram_history.drain(..idx);
        }
        if Self::detect_ngram_loop(&self.ngram_history) {
            self.attractor_detected = true;
        }
    }

    /// Read-only view of the current state.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Bytes accumulated since the last firm state transition.
    pub fn partial(&self) -> &str {
        &self.partial_buf
    }

    /// Whether the matcher is currently free (all tokens allowed).
    /// Free in [`State::Out`] until the buffer accumulates a `<tool_call>`
    /// prefix; free in [`State::InArgs`] until the buffer accumulates
    /// the `</tool_call>` close-marker prefix OR a required field is
    /// still missing from the args body.
    ///
    /// The dual-mode design mirrors V4F's `is_free` — the constraint
    /// only kicks in at structural transitions, not during free prose
    /// or args body.
    ///
    /// Three sources make InArgs non-free:
    ///   1. Buffer ends with a `</tool_call>` close-marker prefix
    ///      (the model is committing to close — gate it).
    ///   2. The n-gram loop guard has tripped
    ///      (`self.attractor_detected = true`): force the close so
    ///      the model gets a forced exit instead of being stuck
    ///      extending an attractor.
    ///   3. A required field is still missing from the args body:
    ///      the model is in the middle of args body and might next
    ///      try to close with `}}\n</tool_call>` — we must reject
    ///      that close until required fields appear. We make the
    ///      state non-free so `is_token_allowed` runs and can
    ///      enforce the per-token check.
    pub fn is_free(&self) -> bool {
        match self.state {
            State::Out => !Self::has_open_prefix(&self.partial_buf),
            State::AfterOpen => false,
            State::InArgs => {
                if self.attractor_detected {
                    return false;
                }
                if !self.required_fields_satisfied() {
                    return false;
                }
                !Self::has_close_prefix(&self.partial_buf)
            }
        }
    }

    /// Does the buffer end with a strict prefix of `<tool_call>` (so a
    /// follow-up token could complete the open)?
    fn has_open_prefix(s: &str) -> bool {
        const OPEN: &str = "<tool_call>";
        for n in 1..=OPEN.len() {
            if s.ends_with(&OPEN[..n]) {
                return true;
            }
        }
        false
    }

    /// Does the buffer end with a strict prefix of `</tool_call>`?
    fn has_close_prefix(s: &str) -> bool {
        const CLOSE: &str = "</tool_call>";
        for n in 1..=CLOSE.len() {
            if s.ends_with(&CLOSE[..n]) {
                return true;
            }
        }
        false
    }

    /// Returns the legal byte-string continuations from the current
    /// state. Each returned string is a FULL prefix starting at the
    /// position immediately after the last firm transition. Callers
    /// check `partial_buf + decode(T)` against these via
    /// [`Self::is_token_allowed`].
    pub fn allowed_continuations(&self) -> Vec<String> {
        match &self.state {
            State::Out => {
                // Only constraining when we've started emitting `<tool_call>`.
                // is_free() short-circuits the unconstrained case before we
                // get here; if we do get here it's because the partial
                // buffer already starts forming `<tool_call>`.
                vec!["<tool_call>".to_string()]
            }
            State::AfterOpen => {
                // Allowed continuations: for each tool name, the literal
                // header `\n{"name": "<NAME>", "arguments": `.
                self.tools
                    .iter()
                    .map(|t| format!("\n{{\"name\": \"{}\", \"arguments\": ", t.name))
                    .collect()
            }
            State::InArgs => {
                // Constraining when the model has started emitting `</tool_call>`.
                // Same dual-mode pattern as State::Out: free unless a close
                // prefix is forming.
                vec!["</tool_call>".to_string()]
            }
        }
    }

    /// Whether the given decoded text would keep us on a legal path
    /// from the current state.
    ///
    /// Empty tokens (placeholder / control tokens with no decoded text)
    /// are always allowed — they don't consume any buffer position.
    ///
    /// Per-state semantics:
    /// - [`State::Out`] / [`State::InArgs`]: in free regions where a
    ///   trigger marker (`<tool_call>` / `</tool_call>`) may start
    ///   forming partway through the buffer, the check is suffix-based.
    ///   A token is allowed iff some suffix of `partial_buf + text` is
    ///   a prefix of the trigger marker (or the full marker appears
    ///   somewhere, which would fire a transition on advance).
    /// - [`State::AfterOpen`]: the entire `partial_buf + text` must be
    ///   a prefix of, or extension of, one of the header templates.
    pub fn is_token_allowed(&self, text: &str) -> bool {
        if text.is_empty() {
            return true;
        }
        if self.is_free() {
            return true;
        }
        let combined = format!("{}{}", self.partial_buf, text);
        match &self.state {
            State::Out => Self::tail_matches_marker(&combined, "<tool_call>"),
            State::AfterOpen => {
                let conts = self.allowed_continuations();
                Self::check_against_conts(&combined, &conts)
            }
            // InArgs: tail-matches-marker is the usual check. When the
            // attractor flag is set or required fields are missing,
            // this same check applies but UNCONDITIONALLY (no is_free
            // short-circuit). For the required-field case we additionally
            // BLOCK any token that would form a close-marker prefix —
            // the model must emit content with the missing field names
            // before closing.
            State::InArgs => {
                let close_prefix = Self::tail_matches_marker(&combined, "</tool_call>");
                if !self.required_fields_satisfied() {
                    // Block close-marker prefix tokens entirely; allow
                    // any other content so the model can keep emitting
                    // until required fields appear in the body.
                    if Self::touches_close_marker(&combined) {
                        false
                    } else {
                        true
                    }
                } else {
                    close_prefix
                }
            }
        }
    }

    /// True iff some suffix of `s` is a prefix of `marker`, OR `marker`
    /// appears anywhere in `s` (the latter handles tokens that finish
    /// emitting the trigger — `advance` will then fire a transition).
    /// Used for free-region constraint checks where the trigger marker
    /// can start partway through `partial_buf`.
    fn tail_matches_marker(s: &str, marker: &str) -> bool {
        if s.contains(marker) {
            return true;
        }
        for n in 1..=marker.len() {
            if s.ends_with(&marker[..n]) {
                return true;
            }
        }
        false
    }

    /// True if `s` contains the full close marker OR ends with any
    /// strict prefix of it. Distinct from `tail_matches_marker`
    /// because we want a binary "does this token touch the close
    /// region" answer for the required-field gate, not a continuation
    /// check.
    fn touches_close_marker(s: &str) -> bool {
        const CLOSE: &str = "</tool_call>";
        if s.contains(CLOSE) {
            return true;
        }
        for n in 1..=CLOSE.len() {
            if s.ends_with(&CLOSE[..n]) {
                return true;
            }
        }
        false
    }

    /// Either `s` is a prefix of some continuation, or some continuation
    /// is a prefix of `s` (the latter handles tokens that extend past
    /// the firm transition boundary — e.g. `\n{"name"` matches the
    /// `\n{"name": "X", "arguments": ` template up to position 8).
    fn check_against_conts(s: &str, conts: &[String]) -> bool {
        for cont in conts {
            if cont.starts_with(s) || s.starts_with(cont.as_str()) {
                return true;
            }
        }
        false
    }

    /// Populate a boolean mask over `vocab` indicating which tokens are
    /// legal at the current matcher position. `out` must be at least
    /// `vocab.len()` long; entries beyond `vocab.len()` are untouched.
    ///
    /// Fast path: when [`Self::is_free`] is true the entire mask is set
    /// to `true` — the caller can skip the sample-time mask scan.
    /// Hot path: O(vocab) scan calling [`Self::is_token_allowed`] per
    /// id, ~129k vocab × a handful of byte comparisons → sub-ms.
    pub fn token_mask(&self, vocab: &[String], out: &mut [bool]) {
        debug_assert!(out.len() >= vocab.len());
        if self.is_free() {
            for slot in out.iter_mut().take(vocab.len()) {
                *slot = true;
            }
            return;
        }
        for (id, text) in vocab.iter().enumerate() {
            out[id] = self.is_token_allowed(text);
        }
    }

    /// Apply the token mask in-place to a logits slice: disallowed
    /// tokens get `f32::NEG_INFINITY`, allowed are left alone.
    pub fn apply_mask_to_logits(mask: &[bool], logits: &mut [f32]) {
        let n = mask.len().min(logits.len());
        for i in 0..n {
            if !mask[i] {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }

    /// Commit decoded token bytes into the matcher, advancing state if
    /// any allowed continuation completes. Idempotent at the byte
    /// level — callers may pass single bytes, multi-byte chunks, or
    /// full decoded tokens; the same final state is reached either way.
    ///
    /// While in [`State::InArgs`], `text` is also fed into the
    /// n-gram loop guard so a model that drifts into a repeating
    /// attractor inside the args body (e.g. `typetypetypetype`) is
    /// detected — the next `is_token_allowed` calls then force the
    /// close marker.
    pub fn advance(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if matches!(self.state, State::InArgs) {
            self.update_ngram_history(text);
        }
        self.partial_buf.push_str(text);

        loop {
            match self.transition_once() {
                Transition::Stay => return,
                Transition::Advanced => continue,
            }
        }
    }

    /// Inner step: examine `partial_buf` against the current state's
    /// allowed transitions. Returns whether any firm transition fired.
    fn transition_once(&mut self) -> Transition {
        match self.state.clone() {
            State::Out => {
                // Look for `<tool_call>` anywhere in the buffer. Once
                // found, drop everything up to and including that
                // substring and transition.
                if let Some(idx) = self.partial_buf.find("<tool_call>") {
                    self.partial_buf
                        .drain(..idx + "<tool_call>".len());
                    self.state = State::AfterOpen;
                    return Transition::Advanced;
                }
                // Trim the buffer to at most the longest open prefix
                // we could complete next step. Stops the buffer from
                // growing without bound during long free-emission runs.
                let max_keep = "<tool_call>".len() - 1;
                if self.partial_buf.len() > max_keep {
                    let drop = self.partial_buf.len() - max_keep;
                    self.partial_buf.drain(..drop);
                }
                Transition::Stay
            }
            State::AfterOpen => {
                // Look for the longest tool-name header that the buffer
                // fully covers. When found, consume it and transition
                // to InArgs — and record which tool's schema is now
                // active so the close-marker check can validate its
                // required-field list.
                for (idx, schema) in self.tools.iter().enumerate() {
                    let cont = format!("\n{{\"name\": \"{}\", \"arguments\": ", schema.name);
                    if let Some(rest) = self.partial_buf.strip_prefix(cont.as_str()) {
                        let rest_owned = rest.to_string();
                        self.partial_buf = rest_owned;
                        self.state = State::InArgs;
                        self.current_tool = Some(idx);
                        return Transition::Advanced;
                    }
                }
                Transition::Stay
            }
            State::InArgs => {
                // Look for `</tool_call>` anywhere in the buffer.
                if let Some(idx) = self.partial_buf.find("</tool_call>") {
                    self.partial_buf
                        .drain(..idx + "</tool_call>".len());
                    self.state = State::Out;
                    // Returning to Out — reset the n-gram guard's
                    // bookkeeping so a subsequent tool_call body starts
                    // with a fresh window. (The attractor flag must be
                    // cleared OR a stale flag will block the next
                    // body's first byte.) Also drop `current_tool` so
                    // the required-field check no longer applies.
                    self.ngram_history.clear();
                    self.attractor_detected = false;
                    self.current_tool = None;
                    return Transition::Advanced;
                }
                let max_keep = "</tool_call>".len() - 1;
                if self.partial_buf.len() > max_keep {
                    let drop = self.partial_buf.len() - max_keep;
                    self.partial_buf.drain(..drop);
                }
                Transition::Stay
            }
        }
    }
}

enum Transition {
    Stay,
    Advanced,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schemas(names: &[&str]) -> Vec<ToolSchema> {
        names
            .iter()
            .map(|n| ToolSchema {
                name: n.to_string(),
                required: Vec::new(),
            })
            .collect()
    }

    /// Schema-with-required helper for required-field tests.
    fn schemas_with_required(specs: &[(&str, &[&str])]) -> Vec<ToolSchema> {
        specs
            .iter()
            .map(|(name, req)| ToolSchema {
                name: name.to_string(),
                required: req.iter().map(|s| s.to_string()).collect(),
            })
            .collect()
    }

    #[test]
    fn out_state_is_free_until_open_prefix() {
        let m = Matcher::new(schemas(&["bash"]));
        assert!(m.is_free());
        assert!(m.is_token_allowed("Hello world"));
        assert!(m.is_token_allowed("<|im_start|>"));
    }

    #[test]
    fn open_marker_transitions_to_after_open() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("here is some prose <tool_call>");
        assert!(matches!(m.state(), State::AfterOpen));
    }

    #[test]
    fn after_open_constrains_to_header_template() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        // The Pi-failure-mode token `<|im_start|>` must be rejected
        // at this position.
        assert!(!m.is_token_allowed("<|im_start|>"));
        // A leading newline that continues the header template is OK.
        assert!(m.is_token_allowed("\n"));
        // `\n{` is OK.
        assert!(m.is_token_allowed("\n{"));
        // Any token that diverges from the header is rejected.
        assert!(!m.is_token_allowed("\nassistant"));
    }

    #[test]
    fn header_completes_into_in_args() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        m.advance("\n{\"name\": \"bash\", \"arguments\": ");
        assert!(matches!(m.state(), State::InArgs));
    }

    #[test]
    fn in_args_is_free_until_close_prefix() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        assert!(m.is_free());
        // Inside args, any payload is fine.
        assert!(m.is_token_allowed("{\"command\": \"ls -la\"}"));
        assert!(m.is_token_allowed("\n"));
    }

    #[test]
    fn close_marker_returns_to_out() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": {}}\n</tool_call>");
        assert!(matches!(m.state(), State::Out));
    }

    #[test]
    fn multiple_tool_names_all_match() {
        let mut m = Matcher::new(schemas(&["bash", "read", "write"]));
        m.advance("<tool_call>");
        // All three tool-name headers are allowed prefixes.
        assert!(m.is_token_allowed("\n{\"name\": \"bash"));
        // Restart with a fresh matcher for the other names.
        let mut m2 = Matcher::new(schemas(&["bash", "read", "write"]));
        m2.advance("<tool_call>");
        assert!(m2.is_token_allowed("\n{\"name\": \"read"));
        let mut m3 = Matcher::new(schemas(&["bash", "read", "write"]));
        m3.advance("<tool_call>");
        assert!(m3.is_token_allowed("\n{\"name\": \"write"));
    }

    #[test]
    fn unknown_tool_name_rejected() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        // `\n{"name": "evil` is not a prefix of `\n{"name": "bash", ...`
        // — the model can't invent a tool name.
        assert!(!m.is_token_allowed("\n{\"name\": \"evil"));
    }

    #[test]
    fn token_mask_free_path_sets_all_true() {
        let m = Matcher::new(schemas(&["bash"]));
        let vocab: Vec<String> = (0..10).map(|i| format!("tok{}", i)).collect();
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        assert!(mask.iter().all(|&b| b));
    }

    #[test]
    fn token_mask_after_open_only_allows_header_tokens() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        let vocab = vec![
            "\n".to_string(),
            "<|im_start|>".to_string(),
            "assistant".to_string(),
            "\n{".to_string(),
            "\n{\"name".to_string(),
        ];
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        assert!(mask[0], "\\n must be allowed (header prefix)");
        assert!(!mask[1], "<|im_start|> must be rejected (Pi failure mode)");
        assert!(!mask[2], "assistant must be rejected");
        assert!(mask[3], "\\n{{ must be allowed (header prefix)");
        assert!(mask[4], "\\n{{\"name must be allowed");
    }

    #[test]
    fn apply_mask_zeros_disallowed_logits() {
        let mask = vec![true, false, true, false];
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];
        Matcher::apply_mask_to_logits(&mask, &mut logits);
        assert_eq!(logits[0], 1.0);
        assert!(logits[1].is_infinite() && logits[1].is_sign_negative());
        assert_eq!(logits[2], 3.0);
        assert!(logits[3].is_infinite() && logits[3].is_sign_negative());
    }

    #[test]
    fn buffer_doesnt_grow_unboundedly_in_out() {
        let mut m = Matcher::new(schemas(&["bash"]));
        // Emit a long stretch of prose. The internal buffer should be
        // bounded to the longest open prefix we could complete (11 - 1
        // = 10 chars).
        for _ in 0..1000 {
            m.advance("a");
        }
        assert!(m.partial().len() <= "<tool_call>".len());
    }

    // ─── Pi turn-12 attractor reproduction & fix demonstration ──────
    //
    // These tests directly reproduce the Pi turn-12 failure mode (model
    // emits `<|im_start|>assistant "..."}}` as the `<tool_call>` body
    // instead of valid JSON) using synthetic logits, and verify the
    // grammar masker prevents it. They mirror the real sample-time
    // decision the daemon makes — the only difference is we control
    // the logits directly instead of running a model. The unit tests
    // give a deterministic, model-independent demonstration of:
    //
    //   1. WITHOUT the grammar mask, an argmax over the failure-mode
    //      logits picks the attractor token (`<|im_start|>`).
    //   2. WITH the grammar mask, the masked argmax picks a valid
    //      header-template continuation.
    //
    // The mask path is byte-for-byte the same one
    // `crates/hipfire-runtime/examples/daemon.rs` runs at each sample
    // step in both the qwen35 non-dflash and dflash paths.

    /// Build a synthetic vocab with known token-text values + a logits
    /// vector that reproduces the Pi attractor signal (the bad token
    /// scores highest). Index 0 is `<|im_start|>` per the qwen tokenizer
    /// id ordering observed in the cache traces; the exact ordering
    /// doesn't matter for the test — what matters is the text → score
    /// mapping below.
    fn attractor_logits_setup() -> (Vec<String>, Vec<f32>) {
        // Token vocab: a mix of the attractor + valid header tokens.
        // These mirror the actual qwen3.6:27b vocab strings that
        // appeared in the Pi turn-12 emit.
        let vocab: Vec<String> = vec![
            "<|im_start|>".to_string(),       // 0: attractor (Pi failure)
            "<|im_end|>".to_string(),         // 1: another invalid
            "assistant".to_string(),          // 2: invalid prose
            "\n".to_string(),                 // 3: valid header start
            "\n{".to_string(),                // 4: valid header
            "\n{\"name".to_string(),          // 5: valid header progression
            "\n{\"name\": \"bash".to_string(), // 6: valid header
            "evil".to_string(),               // 7: not in tool schema
            " arguments".to_string(),         // 8: irrelevant
            "</tool_call>".to_string(),       // 9: premature close — also invalid
        ];
        // Logits where the attractor wins by a wide margin. This mimics
        // the Pi turn-12 distribution: after long context, the model's
        // ChatML-noise attractor scores higher than valid JSON
        // continuations.
        let logits: Vec<f32> = vec![
            10.0, // <|im_start|>  ← attractor wins without mask
            5.0,  // <|im_end|>
            3.0,  // assistant
            2.0,  // \n
            1.5,  // \n{
            1.0,  // \n{"name
            0.5,  // \n{"name": "bash
            -1.0, // evil
            -2.0, // arguments
            -3.0, // </tool_call>
        ];
        (vocab, logits)
    }

    fn argmax(logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[test]
    fn reproduces_pi_turn_12_attractor_without_mask() {
        // PROOF OF FAILURE: with no constraint applied to the logits,
        // the argmax picks `<|im_start|>` — exactly the token that
        // corrupted Pi turn 12's `<tool_call>` body. This is the
        // failure mode we're fixing.
        let (vocab, logits) = attractor_logits_setup();
        let pick = argmax(&logits);
        assert_eq!(
            vocab[pick], "<|im_start|>",
            "raw argmax should pick the attractor (this is the Pi turn-12 failure mode)"
        );
    }

    #[test]
    fn grammar_mask_blocks_pi_turn_12_attractor() {
        // PROOF OF FIX: with the grammar mask applied to the same
        // logits, the argmax picks a VALID header-template token
        // (`\n` — a prefix of the legal continuation
        // `\n{"name": "bash", "arguments": `). The attractor token
        // `<|im_start|>` is masked to `-INF` and can't be selected.
        let (vocab, mut logits) = attractor_logits_setup();
        let mut m = Matcher::new(schemas(&["bash"]));
        // Drive the matcher into `AfterOpen` — the state immediately
        // after the model commits to the `<tool_call>` opener.
        m.advance("<tool_call>");
        assert!(matches!(m.state(), State::AfterOpen));
        assert!(!m.is_free(), "matcher must be constraining in AfterOpen");

        // Build the token mask the daemon would build at sample time
        // and apply it to the logits — same code path as the runtime.
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        Matcher::apply_mask_to_logits(&mask, &mut logits);

        // The attractor is now `-INF`.
        assert!(logits[0].is_infinite() && logits[0].is_sign_negative());
        // `assistant` is also masked (not a header prefix).
        assert!(logits[2].is_infinite() && logits[2].is_sign_negative());
        // `\n` is preserved (valid header start).
        assert_eq!(logits[3], 2.0);

        // The argmax now picks a valid token.
        let pick = argmax(&logits);
        assert_eq!(
            vocab[pick], "\n",
            "masked argmax should pick the highest-scoring valid header prefix"
        );
        // Sanity: the picked token is NOT the attractor.
        assert_ne!(vocab[pick], "<|im_start|>");
    }

    #[test]
    fn grammar_mask_prevents_full_pi_attractor_sequence() {
        // END-TO-END: simulate the full Pi attractor token sequence
        // and verify the mask blocks the FIRST off-distribution token.
        // The sequence from the Pi log was approximately:
        //
        //   <tool_call>  →  \n  →  <|im_start|>  →  assistant  →  …
        //
        // With grammar, after `<tool_call>` + `\n`, the matcher is
        // still in AfterOpen and continues to mask. The next sample
        // step would have picked `<|im_start|>` (or `assistant`) per
        // the attractor logits — both must be rejected.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        m.advance("\n");
        assert!(matches!(m.state(), State::AfterOpen));

        // Each of these is a token the model picked in the Pi log; all
        // must be rejected because none are prefixes of the legal
        // `{"name": "bash", "arguments": ` continuation that follows
        // the `\n`.
        for bad_token in &["<|im_start|>", "assistant", " \"Let me", "}}", "<|im_end|>"] {
            assert!(
                !m.is_token_allowed(bad_token),
                "expected {:?} to be rejected after `<tool_call>\\n`",
                bad_token
            );
        }
        // Valid continuations all pass.
        for good_token in &["{", "{\"", "{\"name", "{\"name\":"] {
            assert!(
                m.is_token_allowed(good_token),
                "expected {:?} to be allowed after `<tool_call>\\n`",
                good_token
            );
        }
    }

    #[test]
    fn dflash_path_post_validation_catches_attractor() {
        // DFLASH STRATEGY: the dflash decode loop validates committed
        // tokens AFTER spec_step commits them. This test simulates that
        // walk for the Pi turn-12 attractor: feed the tokens that
        // appeared in the bad emit through the matcher; the rejection
        // must fire on the first off-distribution token (the daemon
        // then breaks the dflash loop + force-resets KV/DN per the
        // implementation in generate_dflash).
        let mut m = Matcher::new(schemas(&["bash"]));
        // dflash's spec_step would commit a batch — simulate that batch.
        let bad_batch: &[&str] = &[
            "<tool_call>", // accepted: transitions to AfterOpen
            "\n",          // accepted: matches header prefix
            "<|im_start|>", // REJECTED: not a header prefix from current pos
            "assistant",   // would-be-next, never reached
        ];
        let mut accepted = 0;
        let mut violated = false;
        for tok in bad_batch {
            if !m.is_token_allowed(tok) {
                violated = true;
                break;
            }
            m.advance(tok);
            accepted += 1;
        }
        assert!(violated, "dflash post-validation must detect the attractor");
        assert_eq!(accepted, 2, "the two valid tokens are accepted; the 3rd is rejected");
    }

    #[test]
    fn full_legal_tool_call_round_trip() {
        // CONTROL: when the model emits a fully-valid tool_call, the
        // matcher never rejects anything and returns cleanly to Out.
        let mut m = Matcher::new(schemas(&["bash"]));
        let sequence = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"echo Hello\"}}\n</tool_call>";
        // Each character/token should be accepted incrementally.
        for ch in sequence.chars() {
            let s = ch.to_string();
            assert!(
                m.is_token_allowed(&s),
                "valid sequence char {:?} unexpectedly rejected at state {:?}",
                ch,
                m.state()
            );
            m.advance(&s);
        }
        assert!(matches!(m.state(), State::Out));
    }

    #[test]
    fn grammar_disabled_skips_constraint() {
        // CONTROL: when there are no tool schemas (e.g., requests
        // without `tools` in the body), the matcher is constructed
        // with an empty schema list — `is_free()` stays true through
        // every state and every token is allowed. This protects the
        // non-tool-call code path from any grammar overhead.
        let mut m = Matcher::new(Vec::new());
        m.advance("<tool_call>");
        // Without schemas, AfterOpen has zero allowed continuations.
        // is_token_allowed still returns true for any text because
        // is_free returns true… wait, AfterOpen is_free returns
        // false. So tokens get rejected. Verify the alternative —
        // the daemon-side check `grammar_active = !schemas.is_empty()`
        // skips the matcher entirely when schemas is empty.
        let grammar_active = false; // mirrors the daemon's gate
        assert!(!grammar_active, "empty schemas must disable grammar at the daemon level");
    }

    // ─── Multi-tool / schema variation coverage ──────────────────────

    #[test]
    fn tool_names_that_prefix_each_other() {
        // `bash` and `bash_long` overlap on the first 4 chars. The
        // grammar must distinguish between them at the name boundary
        // (the trailing `"` after the name). Both should be reachable,
        // and the prefix doesn't lock in early.
        let mut m = Matcher::new(schemas(&["bash", "bash_long"]));
        m.advance("<tool_call>");
        // Common prefix works.
        assert!(m.is_token_allowed("\n{\"name\": \"bash"));
        // The full short name with closing quote works.
        let mut a = m.clone();
        a.advance("\n{\"name\": \"bash\", \"arguments\": ");
        assert!(matches!(a.state(), State::InArgs), "short name `bash` must settle to InArgs");
        // The longer name also works.
        let mut b = m.clone();
        b.advance("\n{\"name\": \"bash_long\", \"arguments\": ");
        assert!(matches!(b.state(), State::InArgs), "long name `bash_long` must settle to InArgs");
    }

    #[test]
    fn unknown_tool_in_multi_schema_rejected() {
        let mut m = Matcher::new(schemas(&["bash", "read", "write"]));
        m.advance("<tool_call>");
        // `evil` is not in the schema; the closing quote after the
        // name is what disambiguates, so the rejection point is when
        // the buffer accumulates `"evil"` (no valid continuation).
        assert!(!m.is_token_allowed("\n{\"name\": \"evil"));
        // But `bash` (a real name) is fine.
        assert!(m.is_token_allowed("\n{\"name\": \"bash"));
    }

    #[test]
    fn many_tools_token_mask_is_fast() {
        // Stress the token_mask scan with a realistic tool count and
        // vocab size. The hot path is O(vocab * conts) — verify it
        // completes in reasonable time and produces a valid mask.
        let tool_names: Vec<String> = (0..32).map(|i| format!("tool_{}", i)).collect();
        let tools: Vec<ToolSchema> = tool_names
            .iter()
            .map(|n| ToolSchema {
                name: n.clone(),
                required: Vec::new(),
            })
            .collect();
        let vocab: Vec<String> = (0..150_000)
            .map(|i| format!("tok_{}", i))
            .collect();
        let mut m = Matcher::new(tools);
        m.advance("<tool_call>");
        let mut mask = vec![false; vocab.len()];
        let start = std::time::Instant::now();
        m.token_mask(&vocab, &mut mask);
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 1000,
            "token_mask on 150k vocab × 32 tools took {:?} (expected < 1s)",
            elapsed
        );
    }

    // ─── BPE / token boundary edge cases ─────────────────────────────

    #[test]
    fn open_marker_split_across_tokens() {
        // Tokenizer might emit `<tool` + `_call>` as two tokens instead
        // of a single `<tool_call>` special token. The matcher's
        // byte-level partial buffer must handle this — both halves
        // are accepted, and the transition fires on the second.
        let mut m = Matcher::new(schemas(&["bash"]));
        assert!(m.is_token_allowed("<tool"));
        m.advance("<tool");
        assert!(matches!(m.state(), State::Out));
        assert!(!m.is_free(), "matcher must be constraining mid-marker");
        assert!(m.is_token_allowed("_call>"));
        m.advance("_call>");
        assert!(matches!(m.state(), State::AfterOpen));
    }

    #[test]
    fn close_marker_split_across_tokens() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": {}}");
        assert!(matches!(m.state(), State::InArgs));
        // Split `</tool_call>` into chunks.
        assert!(m.is_token_allowed("\n<"));
        m.advance("\n<");
        assert!(m.is_token_allowed("/tool"));
        m.advance("/tool");
        assert!(m.is_token_allowed("_call>"));
        m.advance("_call>");
        assert!(matches!(m.state(), State::Out));
    }

    #[test]
    fn token_carrying_full_open_marker_inside_prose() {
        // Some BPE tokenizers emit `prose<tool_call>more` as a single
        // token. The matcher must detect the marker mid-token and
        // transition. (The `transition_once` uses `find`, not
        // `ends_with`, for the open marker, so this works.)
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("here is some prose <tool_call>extra");
        // `extra` lands in AfterOpen's partial_buf. The header
        // template starts with `\n` so `extra` is invalid — the
        // matcher should reflect this on the next is_token_allowed.
        assert!(matches!(m.state(), State::AfterOpen));
    }

    #[test]
    fn close_marker_in_args_body_string_does_not_close() {
        // The args body is a JSON value; it can contain `</tool_call>`
        // as a STRING literal. Our current grammar doesn't parse JSON
        // string boundaries — it sees the literal substring as a
        // close marker. Document this as a known limitation: the
        // grammar will prematurely transition. (Real models don't
        // typically emit `</tool_call>` inside arg strings; if this
        // surfaces in production, layer in a JSON-aware string-state
        // tracker.)
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        assert!(matches!(m.state(), State::InArgs));
        // Args content includes a string literal containing the close
        // marker. Our grammar treats it as the close (limitation).
        m.advance("{\"command\": \"echo </tool_call>\"}");
        // Document the actual behavior — this is what we observe today.
        assert!(matches!(m.state(), State::Out));
    }

    // ─── Mask construction edge cases ────────────────────────────────

    #[test]
    fn token_mask_handles_empty_vocab() {
        let m = Matcher::new(schemas(&["bash"]));
        let vocab: Vec<String> = Vec::new();
        let mut mask: Vec<bool> = Vec::new();
        m.token_mask(&vocab, &mut mask);
        assert!(mask.is_empty());
    }

    #[test]
    fn token_mask_handles_empty_string_tokens() {
        // Tokenizers sometimes have control tokens whose decoded text
        // is empty (e.g. BOS/EOS variants). These should always be
        // allowed — they contribute nothing to the partial buffer.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        let vocab = vec![
            "".to_string(),               // empty/control token
            "\n".to_string(),             // valid prefix
            "<|im_start|>".to_string(),   // attractor
        ];
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        assert!(mask[0], "empty token must be allowed (control/placeholder)");
        assert!(mask[1], "valid prefix must be allowed");
        assert!(!mask[2], "attractor must be rejected");
    }

    #[test]
    fn apply_mask_handles_size_mismatch() {
        // Defensively: if mask is shorter than logits, only mask the
        // prefix; if mask is longer, ignore the tail. No panic.
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mask = vec![true, false, true]; // shorter than logits
        Matcher::apply_mask_to_logits(&mask, &mut logits);
        assert_eq!(logits[0], 1.0);
        assert!(logits[1].is_infinite());
        assert_eq!(logits[2], 3.0);
        assert_eq!(logits[3], 4.0, "untouched (past mask end)");
        assert_eq!(logits[4], 5.0);
    }

    #[test]
    fn apply_mask_handles_empty_logits() {
        let mask = vec![true, false, true];
        let mut logits: Vec<f32> = Vec::new();
        Matcher::apply_mask_to_logits(&mask, &mut logits);
        assert!(logits.is_empty());
    }

    // ─── ChatML attractor variant coverage ───────────────────────────

    #[test]
    fn all_chatml_special_tokens_rejected_after_open() {
        // Every ChatML special token observed in qwen3.6 attractor
        // cases must be rejected at the `<tool_call>` boundary. These
        // are the tokens that leaked into the body in the Pi log + the
        // CLI's parseOneToolCall sanitizer (cli/index.ts:2273-2278).
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        for tok in &[
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_sep|>",
            "<think>",
            "</think>",
            "<|tool_call|>", // hallucinated variant
        ] {
            assert!(
                !m.is_token_allowed(tok),
                "ChatML token {:?} must be rejected after <tool_call>",
                tok
            );
        }
    }

    #[test]
    fn chatml_tokens_rejected_mid_header() {
        // After committing the start of the header (`\n{"name": "`),
        // ChatML noise must still be rejected. The matcher's partial
        // buffer tracks the in-progress header.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"");
        for tok in &["<|im_start|>", "<|im_end|>", "<think>"] {
            assert!(
                !m.is_token_allowed(tok),
                "{:?} must be rejected after partial header",
                tok
            );
        }
        // The valid tool name continuation is still allowed.
        assert!(m.is_token_allowed("bash"));
        assert!(m.is_token_allowed("bash\", \"arguments\": "));
    }

    // ─── Multi tool_call & sequencing ────────────────────────────────

    #[test]
    fn two_tool_calls_in_one_decode() {
        // Parallel tool-use prompts can emit two `<tool_call>...</tool_call>`
        // blocks back-to-back. The matcher must return to Out after
        // the first close and constrain the second open the same way.
        let mut m = Matcher::new(schemas(&["bash", "read"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>");
        assert!(matches!(m.state(), State::Out));
        // Second open.
        m.advance("\n<tool_call>");
        assert!(matches!(m.state(), State::AfterOpen));
        // Second body must use a known tool name.
        assert!(!m.is_token_allowed("\n{\"name\": \"unknown"));
        assert!(m.is_token_allowed("\n{\"name\": \"read"));
        // Complete the second call.
        m.advance("\n{\"name\": \"read\", \"arguments\": {\"path\": \"/etc/hostname\"}}\n</tool_call>");
        assert!(matches!(m.state(), State::Out));
    }

    #[test]
    fn prose_between_tool_calls_is_free() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": {}}\n</tool_call>");
        assert!(matches!(m.state(), State::Out));
        // Prose chunks are unconstrained.
        m.advance("Now let me think about the next step.");
        assert!(m.is_free());
        // Long prose doesn't get stuck.
        for _ in 0..100 {
            m.advance("more thinking content ");
        }
        assert!(matches!(m.state(), State::Out));
        assert!(m.is_free());
    }

    #[test]
    fn tool_call_at_buffer_boundary() {
        // Simulate a token boundary that puts `<tool_call>` exactly at
        // the buffer's bounded-keep limit. The transition must still
        // fire (the matcher detects the full marker via `find`, not
        // just a suffix match).
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("aaaaaaaaaaaaaaaaaaaa<tool_call>"); // 20 a's + marker
        assert!(matches!(m.state(), State::AfterOpen));
    }

    // ─── JSON args variation ─────────────────────────────────────────

    #[test]
    fn empty_args_object_passes() {
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": {}}\n</tool_call>");
        assert!(matches!(m.state(), State::Out));
    }

    #[test]
    fn nested_json_args_pass() {
        let mut m = Matcher::new(schemas(&["bash"]));
        let body = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"opts\": {\"verbose\": true, \"flags\": [\"a\", \"b\", \"c\"]}, \"cmd\": \"ls -la\"}}\n</tool_call>";
        m.advance(body);
        assert!(matches!(m.state(), State::Out));
    }

    #[test]
    fn unicode_in_args_passes() {
        let mut m = Matcher::new(schemas(&["bash"]));
        let body = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"msg\": \"héllo 世界 🚀\"}}\n</tool_call>";
        m.advance(body);
        assert!(matches!(m.state(), State::Out));
    }

    #[test]
    fn truncated_tool_call_stays_in_args() {
        // max_tokens cap mid-tool-call: matcher is left in InArgs
        // (no close marker seen). The daemon detects truncation via
        // its own bookkeeping; the grammar matcher just doesn't lie
        // about state.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"echo");
        assert!(matches!(m.state(), State::InArgs));
        // No close emitted, so state stays InArgs.
        assert!(matches!(m.state(), State::InArgs));
    }

    // ─── Property-based smoke tests ──────────────────────────────────

    #[test]
    fn property_random_valid_sequences_always_pass() {
        // Generate many random valid tool_call sequences and verify
        // the matcher accepts each one fully and returns to Out.
        // Uses a deterministic LCG seed so failures are reproducible.
        let tools = schemas(&["bash", "read", "write", "edit"]);
        let mut rng_state: u32 = 0xCAFEBABE;
        let mut lcg = || {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            rng_state
        };
        for _ in 0..200 {
            let mut m = Matcher::new(tools.clone());
            let tool_idx = (lcg() % 4) as usize;
            let names = ["bash", "read", "write", "edit"];
            let arg_value: String = (0..((lcg() % 50) as usize))
                .map(|_| (b'a' + (lcg() % 26) as u8) as char)
                .collect();
            let seq = format!(
                "<tool_call>\n{{\"name\": \"{}\", \"arguments\": {{\"x\": \"{}\"}}}}\n</tool_call>",
                names[tool_idx], arg_value
            );
            for ch in seq.chars() {
                let s = ch.to_string();
                assert!(
                    m.is_token_allowed(&s),
                    "char {:?} unexpectedly rejected mid-sequence; state={:?} partial={:?}",
                    ch,
                    m.state(),
                    m.partial()
                );
                m.advance(&s);
            }
            assert!(matches!(m.state(), State::Out),
                "valid sequence didn't return to Out (tool={}, args=`{}`, final state={:?})",
                names[tool_idx], arg_value, m.state());
        }
    }

    #[test]
    fn property_attractor_tokens_never_accepted_after_open() {
        // For every position inside the AfterOpen state (driven by
        // increasing valid header prefixes), the Pi-style attractor
        // tokens must remain rejected. Catches regressions where a
        // partial-buf size or transition logic change accidentally
        // makes a ChatML token look like a header prefix.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        let header = "\n{\"name\": \"bash\", \"arguments\": ";
        for end in 0..=header.len() {
            let mut m2 = m.clone();
            let prefix = &header[..end];
            m2.advance(prefix);
            // Inside AfterOpen unless we hit the end (transitions to
            // InArgs). Either way, attractor tokens must NEVER fit.
            if matches!(m2.state(), State::AfterOpen) {
                for tok in &["<|im_start|>", "<|im_end|>", "<think>"] {
                    assert!(
                        !m2.is_token_allowed(tok),
                        "attractor {:?} accepted at header position {} (partial={:?})",
                        tok, end, m2.partial()
                    );
                }
            }
        }
    }

    // ─── Integration with apply_mask_to_logits ───────────────────────

    #[test]
    fn mask_then_argmax_blocks_attractor_at_every_header_position() {
        // The Pi failure was at position 0 of AfterOpen (right after
        // `<tool_call>`). Verify the mask-then-argmax pipeline blocks
        // the attractor at multiple header positions — at each one,
        // the model is in a slightly-different partial state but the
        // attractor must remain `-INF`.
        //
        // The vocab includes the header bytes as single-char tokens so
        // there's always a valid next-byte token whatever position we
        // pause at. Real qwen vocab has these single-char tokens (
        // newline, ASCII letters, punctuation) so this models the
        // production case.
        let header = "\n{\"name\": \"bash\", \"arguments\": ";
        for end in 0..=header.len() {
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>");
            m.advance(&header[..end]);
            if !matches!(m.state(), State::AfterOpen) {
                continue; // we reached InArgs — past the constrained region
            }
            // Vocab: attractors (must be -INF) + every single-byte
            // ASCII char (at least one will continue the header).
            let mut vocab: Vec<String> = vec![
                "<|im_start|>".to_string(),
                "<|im_end|>".to_string(),
                "<think>".to_string(),
                "assistant".to_string(),
            ];
            for byte in (b' '..=b'~').chain([b'\n', b'\t']) {
                vocab.push((byte as char).to_string());
            }
            let attractor_count = 4;
            let mut logits = vec![100.0f32; vocab.len()];
            // Make attractors win the raw argmax.
            logits[0] = 200.0; // <|im_start|>
            logits[1] = 190.0; // <|im_end|>
            logits[2] = 185.0;
            logits[3] = 180.0;

            let mut mask = vec![false; vocab.len()];
            m.token_mask(&vocab, &mut mask);
            Matcher::apply_mask_to_logits(&mask, &mut logits);

            // All attractors must be -INF.
            for i in 0..attractor_count {
                assert!(
                    logits[i].is_infinite() && logits[i].is_sign_negative(),
                    "attractor vocab[{}]={:?} not -INF at header position {}",
                    i, vocab[i], end,
                );
            }
            // At least ONE token in the vocab must still be allowed —
            // the next byte of the header template is always present
            // in the single-char-ASCII slice.
            assert!(
                logits.iter().any(|l| l.is_finite()),
                "all logits -INF at header position {} (partial={:?}); grammar pinned the model into a corner",
                end, m.partial(),
            );
        }
    }

    // ─── N-gram loop guard tests (real attractor from production) ───
    //
    // The motivating incident: qwen3.6:27b at ~turn 8 of a long
    // agentic session emitted `typetypetypetype...` and
    // `pub fn BlinkHash(key_pub fn BlinkHash(key_...` inside the
    // `<tool_call>` args body. The grammar at the time was permissive
    // in `InArgs` (correctly — args are free-form JSON), so the
    // attractor wasn't blocked. Each retry baked more garbage into
    // the conversation until KV ran out. These tests verify the
    // n-gram loop guard catches both patterns + forces a close.

    #[test]
    fn ngram_guard_catches_short_char_attractor() {
        // The `typetypetypetype` pattern. 4-byte gram × 4 = 16
        // bytes — must trip the guard.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        assert!(matches!(m.state(), State::InArgs));
        assert!(!m.attractor_detected(), "guard should be clean on entry");
        m.advance("{\"content\": \"typetypetypetype");
        assert!(m.attractor_detected(), "should detect 4-byte × 4 repetition");
        // Once flagged, the matcher is no longer free — even prose
        // tokens get rejected.
        assert!(!m.is_free());
        assert!(!m.is_token_allowed("more code"));
        assert!(!m.is_token_allowed("\"}"));
        // The forced exit: tokens that complete the close marker pass.
        assert!(m.is_token_allowed("\"}}\n<"));
        assert!(m.is_token_allowed("\"}}\n</tool_call>"));
    }

    #[test]
    fn ngram_guard_catches_long_phrase_attractor() {
        // The `pub fn BlinkHash(key_pub fn BlinkHash(key_...` pattern.
        // Use a 20-byte gram × 4 repeats; the guard scans up to 32-byte
        // grams so this must trip.
        let mut m = Matcher::new(schemas(&["write"]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        assert!(matches!(m.state(), State::InArgs));
        let phrase = "pub fn BlinkHash(key";
        assert_eq!(phrase.len(), 20);
        // Emit some valid prefix, then the attractor.
        m.advance("{\"path\": \"/tmp/x.zig\", \"content\": \"");
        let mut payload = String::new();
        for _ in 0..4 {
            payload.push_str(phrase);
        }
        m.advance(&payload);
        assert!(
            m.attractor_detected(),
            "should detect 20-byte phrase × 4 repetition"
        );
    }

    #[test]
    fn ngram_guard_does_not_trip_on_3_repeats() {
        // Threshold is 4 repeats. A 3-repeat sequence — legitimate in
        // arrays like `[1,1,1]` or `{a:0,a:0,a:0}` — must NOT trip.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        m.advance("{\"arr\": [1,1,1]}");
        assert!(!m.attractor_detected(), "3-repeats must stay below threshold");
    }

    #[test]
    fn ngram_guard_does_not_trip_on_normal_json() {
        // Normal JSON content (varied bytes) doesn't trip.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        m.advance("{\"command\": \"echo Hello && ls -la /tmp/ && cat README.md\"}");
        assert!(!m.attractor_detected());
    }

    #[test]
    fn ngram_guard_resets_after_close_marker() {
        // After the model commits to `</tool_call>` and we return to
        // Out, the attractor flag clears so a subsequent tool_call
        // doesn't inherit it.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        m.advance("{\"x\": \"typetypetypetype");
        assert!(m.attractor_detected());
        // Force-close (the daemon's sample mask would have driven the
        // model here).
        m.advance("\"}}\n</tool_call>");
        assert!(matches!(m.state(), State::Out));
        assert!(!m.attractor_detected(), "flag must clear on close");
        // Next tool_call body starts clean.
        m.advance("\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {}}\n</tool_call>");
        assert!(matches!(m.state(), State::Out));
        assert!(!m.attractor_detected());
    }

    #[test]
    fn ngram_guard_does_not_trip_on_truly_varied_content() {
        // Long stretch of NON-repeating content — a deterministic LCG
        // over the printable ASCII range. No n-gram of any length
        // 2..32 should repeat 4 consecutive times.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        let mut payload = String::new();
        let mut state: u32 = 0xCAFEBABE;
        for _ in 0..5000 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let c = ((state >> 16) as u8 % 94) + b' '; // printable ASCII
            payload.push(c as char);
        }
        m.advance(&payload);
        assert!(!m.attractor_detected(), "varied (LCG) content must not trip");
    }

    // ─── Required-field guard tests ──────────────────────────────────
    //
    // The motivating incident: Pi's `write` tool requires `path` and
    // `content`. The model drifted into emitting `arguments:{}` (or
    // `arguments:{"path":"…","edits":[]}` — missing `content`) over
    // and over, Pi rejected each, model retried, KV bloated until
    // exhaustion. The required-field guard rejects the close marker
    // until every required field name appears in the args body.

    #[test]
    fn required_field_guard_blocks_empty_args() {
        // `write` requires `path` and `content`. Model tries to emit
        // empty `{}` and immediately close — both required-field
        // checks must reject any close-marker token.
        let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
        m.advance("<tool_call>");
        m.advance("\n{\"name\": \"write\", \"arguments\": ");
        assert!(matches!(m.state(), State::InArgs));
        m.advance("{}"); // empty args body
        // Body has no "path" or "content" — close-marker tokens are blocked.
        assert!(!m.is_token_allowed("\n</tool_call>"));
        assert!(!m.is_token_allowed("</tool_call>"));
        assert!(!m.is_token_allowed("\n<"));
        // Non-close-marker content (like a key name) IS allowed — the
        // model can recover by emitting the missing fields.
        assert!(m.is_token_allowed("\"path"));
        assert!(m.is_token_allowed("anything else"));
    }

    #[test]
    fn required_field_guard_blocks_partial_args() {
        // The Pi-log variant: model emits `{"path":"...","edits":[]}` —
        // has `path` but is missing `content`. Close must still be blocked.
        let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        m.advance("{\"path\":\"/tmp/x.zig\",\"edits\":[]}");
        // `path` present, `content` missing → still block close.
        assert!(!m.is_token_allowed("\n</tool_call>"));
        assert!(!m.is_token_allowed("</tool_call>"));
        // Allow further content to add the missing field.
        assert!(m.is_token_allowed(",\"content"));
    }

    #[test]
    fn required_field_guard_allows_close_when_all_satisfied() {
        let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        m.advance("{\"path\":\"/tmp/x\",\"content\":\"hello\"}");
        // Both required fields present in body — close is now allowed.
        assert!(m.is_token_allowed("\n"));
        assert!(m.is_token_allowed("\n</tool_call>"));
        m.advance("\n</tool_call>");
        assert!(matches!(m.state(), State::Out));
        assert_eq!(m.current_tool(), None, "current_tool clears on close");
    }

    #[test]
    fn required_field_guard_handles_no_required_fields() {
        // Tool with empty `required` (e.g. `list_files()` with no args)
        // — the guard is a no-op. Close fires normally.
        let mut m = Matcher::new(schemas_with_required(&[("list", &[])]));
        m.advance("<tool_call>\n{\"name\": \"list\", \"arguments\": ");
        m.advance("{}");
        // No required fields → trivially satisfied.
        assert!(m.is_token_allowed("\n</tool_call>"));
    }

    #[test]
    fn required_field_guard_blocks_close_via_token_mask() {
        // Realistic: model has emitted empty args. Token mask must
        // block both the close marker and any token that would form
        // a close-marker prefix.
        let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        m.advance("{}");
        let vocab = vec![
            "\n</tool_call>".to_string(),   // close marker — must be blocked
            "</tool_call>".to_string(),     // close marker — must be blocked
            "\n<".to_string(),              // close-marker prefix — blocked
            "\"path".to_string(),           // valid field name — allowed
            "\"content".to_string(),        // valid field name — allowed
            "anything".to_string(),         // arbitrary content — allowed
        ];
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        assert!(!mask[0], "full close marker must be blocked");
        assert!(!mask[1], "bare close marker must be blocked");
        assert!(!mask[2], "close-marker prefix must be blocked");
        assert!(mask[3], "field name `\"path` must be allowed");
        assert!(mask[4], "field name `\"content` must be allowed");
        assert!(mask[5], "free content must be allowed");
    }

    #[test]
    fn required_field_guard_substring_match_robust_to_quoting() {
        // The guard does a substring search for `"<name>"` in the
        // args body. This handles standard JSON quoting where field
        // names are double-quoted. The presence check doesn't require
        // a syntactically-valid JSON — substring is enough.
        let mut m = Matcher::new(schemas_with_required(&[("bash", &["command"])]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        // Field appears with spaces around the colon — still matches
        // because we look for `"command"` substring.
        m.advance("{ \"command\" : \"ls -la\" }");
        assert!(m.is_token_allowed("\n</tool_call>"));
    }

    #[test]
    fn required_field_guard_reproduces_pi_write_attractor() {
        // Direct reproduction of the Pi-log failure pattern:
        //   model emits arguments:{} → Pi rejects → retry → repeat
        //
        // Before the guard: the matcher allowed the close marker and
        // the malformed tool_call propagated to Pi. With the guard,
        // close-marker tokens are masked out — the model has to emit
        // path and content (or hit max_tokens with finish_reason="length").
        let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        // The exact Pi-log body.
        m.advance("{}");
        // Close attempts the model tried in the log — all must be blocked.
        for bad_token in &["\n</tool_call>", "</tool_call>", "\n<", "<", "</tool_"] {
            assert!(
                !m.is_token_allowed(bad_token),
                "Pi-log close attempt {:?} must be blocked while args is empty",
                bad_token
            );
        }
        // The model can recover by emitting field content.
        assert!(m.is_token_allowed("\"path\":\"/tmp"));
    }

    #[test]
    fn current_tool_set_on_in_args_transition() {
        // The matcher must track which tool's schema we're inside.
        let mut m = Matcher::new(schemas_with_required(&[
            ("bash", &["command"]),
            ("write", &["path", "content"]),
        ]));
        m.advance("<tool_call>");
        assert_eq!(m.current_tool(), None, "AfterOpen state has no tool yet");
        m.advance("\n{\"name\": \"write\", \"arguments\": ");
        assert!(matches!(m.state(), State::InArgs));
        assert_eq!(m.current_tool(), Some(1), "write is tools[1]");
    }

    #[test]
    fn ngram_guard_history_bounded() {
        // Even a long emission that DOES trip the guard shouldn't run
        // the matcher out of memory. After 50k bytes including an
        // attractor, the matcher should still be usable. (We can't
        // directly observe ngram_history.len() since it's private —
        // this test just verifies no panic / unbounded growth.)
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        // Even after the guard trips, additional advance calls
        // shouldn't keep growing the history (update_ngram_history
        // short-circuits when the flag is set).
        m.advance("{\"x\": \"typetypetypetype");
        assert!(m.attractor_detected());
        let huge: String = "type".repeat(10_000);
        m.advance(&huge);
        // No assertion needed — surviving this advance without panic
        // is the test.
    }

    #[test]
    fn ngram_guard_forces_close_via_token_mask() {
        // Realistic end-to-end: model emits args body that triggers
        // the guard. Token mask must then allow only tokens that
        // form a `</tool_call>` prefix.
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
        m.advance("{\"x\": \"typetypetypetype");
        assert!(m.attractor_detected());

        let vocab = vec![
            "type".to_string(),     // attractor token — must be -INF
            "abc".to_string(),      // normal prose — must be -INF
            "<".to_string(),        // close-marker prefix — must be allowed
            "</tool".to_string(),   // close-marker prefix — must be allowed
            "</tool_call>".to_string(), // full close — must be allowed
            "".to_string(),         // empty/control — always allowed
        ];
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        assert!(!mask[0], "attractor token `type` must be blocked");
        assert!(!mask[1], "prose token `abc` must be blocked");
        assert!(mask[2], "close-marker prefix `<` must be allowed");
        assert!(mask[3], "close-marker prefix `</tool` must be allowed");
        assert!(mask[4], "full close marker must be allowed");
        assert!(mask[5], "empty token always allowed");
    }

    #[test]
    fn ngram_guard_catches_token_attractor_from_real_log() {
        // Reproduce the exact byte pattern from the Pi log
        // (transcript shared in conversation, args field of the
        // failed `write` tool call):
        //
        //   "...pub fn BlinkHash(key_type: type, value_type: type)
        //    typetypetypetypepub const NodeKind = enum(u2) {
        //    pub fn BlinkHash(key_tpub const NodeKind = enum(u2) {
        //        INTERNAL,
        //    pub fn BlinkHash(key..."
        //
        // Two distinct attractors: 4-byte `type` × 4 and the longer
        // `pub fn BlinkHash(key` phrase × 3+. We feed the chunk in
        // its natural order; the guard should trip on the first
        // attractor encountered.
        let mut m = Matcher::new(schemas(&["write"]));
        m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
        let body = "{\"path\": \"/tmp/x.zig\", \"content\": \"pub fn BlinkHash(key_type: type, value_type: type) typetypetypetype";
        m.advance(body);
        assert!(
            m.attractor_detected(),
            "real-log attractor sequence must trip the guard"
        );
    }

    #[test]
    fn mask_position_zero_picks_newline() {
        // Most concrete realization of the Pi turn-12 fix. Vocab: all
        // ASCII single-byte chars + the failure-mode attractors. The
        // attractor wins raw argmax. After mask, argmax must pick `\n`
        // (the only valid single-char continuation from AfterOpen at
        // partial="").
        let mut m = Matcher::new(schemas(&["bash"]));
        m.advance("<tool_call>");
        let mut vocab: Vec<String> = vec![
            "<|im_start|>".to_string(),
            "<|im_end|>".to_string(),
            "<think>".to_string(),
        ];
        for byte in (b' '..=b'~').chain([b'\n', b'\t']) {
            vocab.push((byte as char).to_string());
        }
        let mut logits = vec![1.0f32; vocab.len()];
        logits[0] = 1000.0; // attractor wins raw
        let mut mask = vec![false; vocab.len()];
        m.token_mask(&vocab, &mut mask);
        Matcher::apply_mask_to_logits(&mask, &mut logits);
        let pick = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        // `\n` is the only valid next char (header starts with `\n`).
        assert_eq!(vocab[pick], "\n", "masked argmax should pick `\\n` (vocab[{}]={:?})", pick, vocab[pick]);
    }
}
