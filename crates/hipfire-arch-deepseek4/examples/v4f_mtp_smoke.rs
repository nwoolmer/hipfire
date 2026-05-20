//! V4F MTP / speculative-decode smoke test.
//!
//! Loads a V4F build with MTP weights (in-band base HFQ that contains
//! mtp.0.* tensors, OR a base + sibling .mtp-addon.hfq), seeds a short
//! prompt, then drives `speculative_decode_step` for a few windows and
//! reports per-window acceptance rates.
//!
//! Pipeline per window:
//!   1. decode_step on last_token at position N — populates state.mtp_last_hidden
//!      (the h_n that the MTP block reads). On the first window this also
//!      finishes prefilling the prompt.
//!   2. speculative_decode_step(last_token, N, last_hidden=None, k=K)
//!      → returns SpecStepResult { accepted_tokens, n_accepted, n_proposed }
//!   3. update last_token to accepted_tokens.last(); advance position;
//!      decode the accepted tokens for display + state advancement.
//!
//! Usage:
//!   v4f_mtp_smoke <model.hfq> [--prompt "..."] [--k N] [--windows W]
//!                 [--moe 0|1]
//!
//! Defaults: --k 4 --windows 8 --moe 1, prompt = "Generate a fibonacci
//! function in C\n".
//!
//! ENV:
//!   HIPFIRE_V4F_LOAD_MTP=1  (default — must stay on for this binary)
//!   HIPFIRE_V4F_MTP_ADDON=PATH  optional addon HFQ override

use hipfire_arch_deepseek4::{
    forward::{decode_step, mtp_forward, PrefillBatchScratch},
    spec_decode::{speculative_decode_step_with_pbs, logits_argmax},
    DeepseekV4, DeepseekV4State,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect(
        "usage: v4f_mtp_smoke <model.hfq> [--prompt STR] [--k N] [--windows W] [--moe 0|1]");

    let mut prompt = "Generate a fibonacci function in C\n".to_string();
    let mut k: usize = 4;
    let mut windows: usize = 8;
    let mut use_moe: bool = true;
    while let Some(flag) = args.next() {
        let val = args.next().expect("flag missing value");
        match flag.as_str() {
            "--prompt"  => prompt = val,
            "--k"       => k = val.parse().map_err(|e| format!("k: {e}"))?,
            "--windows" => windows = val.parse().map_err(|e| format!("windows: {e}"))?,
            "--moe"     => use_moe = val == "1",
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

    // MUST load MTP — the whole point of this binary.
    std::env::set_var("HIPFIRE_V4F_LOAD_MTP", "1");
    if use_moe {
        std::env::set_var("HIPFIRE_V4F_UPLOAD_EXPERTS", "1");
        std::env::set_var("HIPFIRE_V4F_MOE", "1");
    }
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = DeepseekV4State::new(&cfg)?;

    if weights.mtp_layer.is_none() {
        return Err("MTP weights not loaded — HFQ missing mtp.0.* tensors. Re-quant with --format v4f-q8-mtp or supply <base>.mtp-addon.hfq alongside the base.".to_string());
    }
    eprintln!("MTP layer loaded ✓");

    // Allocate spec-decode batched scratch ONCE. The internal
    // `forward_prefill_batch_chunk` + `final_norm_and_head_all_batched`
    // both want a `PrefillBatchScratch`. Allocating it per spec call
    // (the old default) costs ~30 GpuTensor allocations per window.
    let pbs = PrefillBatchScratch::new(&mut gpu, &cfg, k.max(8))?;

    // ── Prefill (main + MTP) ──────────────────────────────────────────
    //
    // For MTP attention to have prompt context during spec-decode windows,
    // the MTP layer's SWA cache (state._attention[num_hidden_layers]) must
    // also be populated during prefill — NOT just the main 0..N-1 layers.
    //
    // At each prefill position i, after main forward produces h_i, we know
    // the NEXT prompt token T_{i+1} (since this is prefill). Run mtp_forward
    // with (hidden=h_i, next_token=T_{i+1}, position=i) — this is the
    // V3 §4 contract for MTP step k=1 — and lets the MTP attn_stub write
    // its SWA slot i.
    //
    // The LAST prefill position (i = N-1) is skipped because the "next
    // token" T_N isn't known yet (it's what main is about to predict).
    // That leaves a 1-slot gap in MTP SWA at position N-1; for short
    // prompts (N < window=128) the slot reads as zero-init and contributes
    // negligibly to attention. For long prompts, an extra MTP step using
    // the predicted T_N would close this — TODO if needed.
    let prompt_tokens = tokenizer.encode(&prompt);
    eprintln!("Prefilling {} prompt tokens (main + MTP)...", prompt_tokens.len());
    let mut last_logits = vec![];
    let pp_start = Instant::now();
    for i in 0..prompt_tokens.len() {
        let pos = i as u32;
        last_logits = decode_step(
            &cfg, &weights, &mut state, &mut gpu, prompt_tokens[i], pos,
        )?;
        // decode_step captured h_i into state.mtp_last_hidden and advanced
        // state.n_tokens to i+1. Roll back to i so the MTP forward's
        // attn_stub writes the correct SWA ring slot, then restore.
        if i + 1 < prompt_tokens.len() {
            // Decouple the borrow: hidden lives in state, mtp_forward takes
            // &mut state. SAFETY: the GEMV chain in mtp_forward reads
            // mtp_last_hidden via mtp_h_norm_scratch (written in step 3
            // before the read into h_proj at step 4), then capture_mtp_hidden
            // overwrites mtp_last_hidden in step 7 — reads complete before
            // writes on the same stream.
            let hidden_ptr: *const rdna_compute::GpuTensor =
                state.mtp_last_hidden.as_ref().unwrap();
            let hidden: &rdna_compute::GpuTensor = unsafe { &*hidden_ptr };
            let next_tok = prompt_tokens[i + 1];
            let saved_n = state.n_tokens;
            state.n_tokens = pos as u64;
            let _ = mtp_forward(
                &cfg, &weights, &mut state, &mut gpu, hidden, next_tok, pos,
            )?;
            state.n_tokens = saved_n;
        }
    }
    eprintln!("Prefill done in {:.2}s, n_tokens={}",
        pp_start.elapsed().as_secs_f64(), state.n_tokens);

    let mut last_token = logits_argmax(&last_logits) as u32;
    // last_position is the position from which `last_token` was PREDICTED
    // (i.e. the hidden at that position predicts last_token). After prefill
    // of N tokens, the last decode_step ran at position N-1; its hidden h_{N-1}
    // was captured to state.mtp_last_hidden and predicts last_token (= T_N).
    let mut last_position = prompt_tokens.len() as u32 - 1;
    let mut all_emitted: Vec<u32> = vec![last_token];

    // ── Speculative-decode windows ────────────────────────────────────
    //
    // Per V3 §4, MTP step k=1 wants:
    //   - hidden  = h_{N-1}  (the main-model hidden that predicts T_N)
    //   - next_tk = T_N      (the just-predicted token)
    // and outputs prediction of T_{N+1}.
    //
    // Critically, mtp_last_hidden must hold h_{N-1}, NOT h_N. The last
    // decode_step in prefill already populated it correctly; calling
    // decode_step again here on last_token (= T_N) would OVERWRITE it
    // with h_N, breaking the MTP input contract. We don't do that.
    //
    // After spec_decode_step, mtp_last_hidden is polluted by MTP's
    // internal K-step chain. Between windows we restore the correct
    // hidden by running ONE decode_step on the divergence-preferred /
    // last-accepted token at its emitted position — that re-runs main
    // forward there, captures the right h, and advances state.
    let mut total_proposed = 0usize;
    let mut total_accepted = 0usize;
    let mut win_tokens_emitted = 0usize;
    let total_start = Instant::now();
    for w in 0..windows {
        let win_start = Instant::now();

        // Drive K MTP drafts + a B=K verify pass. Reuses the
        // session-wide PBS to avoid per-window allocation overhead.
        let res = speculative_decode_step_with_pbs(
            &cfg, &weights, &mut state, &mut gpu, &pbs,
            last_token, last_position, /*last_hidden=*/ None, k,
        )?;

        let win_elapsed = win_start.elapsed().as_secs_f64();
        eprintln!("[win {w:>2}] n_accepted={}/{}  emitted={}  in {:.2}s",
            res.n_accepted, res.n_proposed, res.accepted_tokens.len(), win_elapsed);
        eprint!("    \"{}\"\n", tokenizer.decode(&res.accepted_tokens));

        total_proposed += res.n_proposed;
        total_accepted += res.n_accepted;
        win_tokens_emitted += res.accepted_tokens.len();
        all_emitted.extend(res.accepted_tokens.iter().copied());

        // Advance the cursor: spec_decode_step emitted accepted_tokens
        // covering positions [last_position+1 .. last_position+accepted_len].
        // mtp_last_hidden was already refreshed inside spec_decode_step from
        // the verify pass's last-emitted position, so we don't need an extra
        // decode_step call here.
        last_position = last_position + res.accepted_tokens.len() as u32;
        last_token = *res.accepted_tokens.last().unwrap();
    }
    let total_elapsed = total_start.elapsed().as_secs_f64();

    eprintln!("\n=== Summary ===");
    eprintln!("Accepted {} / {} drafts ({:.1}%)",
        total_accepted, total_proposed,
        100.0 * total_accepted as f64 / total_proposed.max(1) as f64);
    eprintln!("Emitted {} tokens over {} windows in {:.2}s = {:.2} tok/s",
        win_tokens_emitted, windows, total_elapsed,
        win_tokens_emitted as f64 / total_elapsed);
    eprintln!("Per-window: avg accepted = {:.2}, avg tokens emitted = {:.2}",
        total_accepted as f64 / windows as f64,
        win_tokens_emitted as f64 / windows as f64);
    eprintln!("\n=== Full generation ===");
    eprintln!("{}", tokenizer.decode(&all_emitted));

    Ok(())
}
