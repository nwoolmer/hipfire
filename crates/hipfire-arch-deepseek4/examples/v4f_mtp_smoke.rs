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
    forward::{
        decode_step, mtp_forward, PrefillBatchScratch,
        forward_prefill_batch_chunk, final_norm_and_head_last_batched,
        precompute_positions,
    },
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
    // K=3 is the measured sweet spot on v4f.mq2lloyd-q8 + MoE:
    // 62.5% accept, +31% effective tok/s vs plain decode. K=2 has
    // higher accept (84.4%) but lower throughput; K≥4 collapses
    // (40% accept at K=4, 28% at K=6). See memory entry
    // `project_v4f_mtp_hc_plumbing_fixed.md` for the full K-sweep table.
    let mut k: usize = 3;
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
    // Env override: HIPFIRE_V4F_PREFILL_BATCH lets us use a larger pbs
    // for batched prefill amortization at long prompts. Default 8 keeps
    // backward-compat / VRAM use bounded for spec-decode-only runs.
    let prefill_batch: usize = std::env::var("HIPFIRE_V4F_PREFILL_BATCH")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let pbs = PrefillBatchScratch::new(&mut gpu, &cfg, k.max(prefill_batch))?;

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
    eprintln!("Prefilling {} prompt tokens (batched main + per-position MTP, B={})...",
        prompt_tokens.len(), pbs.max_batch);
    let prefill_path = std::env::var("HIPFIRE_V4F_PREFILL_PATH")
        .ok().unwrap_or_else(|| "batched".to_string());
    let mut last_logits = vec![];
    let pp_start = Instant::now();
    if prefill_path == "seq" {
        // ── Legacy per-token path (kept for A/B comparison via env) ─────
        for i in 0..prompt_tokens.len() {
            let pos = i as u32;
            last_logits = decode_step(
                &cfg, &weights, &mut state, &mut gpu, prompt_tokens[i], pos,
            )?;
            if i + 1 < prompt_tokens.len() {
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
    } else {
        // ── Phase A batched-main + per-position-MTP path ────────────────
        // For each prompt chunk:
        //   1. forward_prefill_batch_chunk(chunk) — main forward at B=chunk
        //   2. For each batch position b within the chunk (except the
        //      global last position), copy pbs.streams_batch[b] into
        //      state.mtp_last_hidden and run mtp_forward(token[b+1], pos=b)
        //      to populate the MTP layer's SWA cache slot.
        //   3. After the last chunk, final_norm_and_head_last_batched(...)
        //      gives last_logits for the first generation token.
        //
        // Lever 1 measurement instrumentation (2026-05-21): tracks per-stage
        // wallclock to inform whether batched-MTP fill is worth implementing.
        let mut main_us_total: u128 = 0;
        let mut mtp_us_total: u128 = 0;
        let mut mtp_us_each: Vec<u128> = Vec::new();
        let stream_len = cfg.hc_mult * cfg.hidden_size;
        // mtp_last_hidden is lazily allocated by final_norm_and_head /
        // mtp_forward — but our loop writes into it BEFORE either runs.
        // Pre-allocate here so the d2d copy has a valid dest.
        if state.mtp_last_hidden.is_none() {
            state.mtp_last_hidden = Some(
                gpu.alloc_tensor(&[cfg.hc_mult, cfg.hidden_size], rdna_compute::DType::F32)
                    .map_err(|e| format!("alloc mtp_last_hidden (prefill): {e:?}"))?
            );
        }
        // compressor_forward_prebatched (called inside forward_prefill_batch_chunk
        // for any compressed layer) reads state.pos_array_device via pos_slot().
        // The batched main forward doesn't otherwise initialize it (it uses
        // pbs.comp_positions instead). Allocate + populate here for position 0;
        // the MTP-fill inner loop refreshes it per absolute_pos.
        precompute_positions(&cfg, &mut state, &mut gpu, 0)?;
        let mut pos_cursor = 0usize;
        while pos_cursor < prompt_tokens.len() {
            let chunk_size = (prompt_tokens.len() - pos_cursor).min(pbs.max_batch);
            let chunk = &prompt_tokens[pos_cursor..pos_cursor + chunk_size];
            let is_last_chunk = pos_cursor + chunk_size == prompt_tokens.len();

            // 1. Main forward batched on chunk — TIMED with device sync.
            let t_main = Instant::now();
            forward_prefill_batch_chunk(
                &cfg, &weights, &mut state, &mut gpu, &pbs, chunk, pos_cursor as u32,
            )?;
            gpu.hip.device_synchronize()
                .map_err(|e| format!("sync after main chunk: {e:?}"))?;
            main_us_total += t_main.elapsed().as_micros();

            // 2. MTP fill for positions [pos_cursor..pos_cursor + chunk_size)
            //    (skip the LAST position of the last chunk — its next_tok is
            //    unknown, and that's the gap mtp_forward would normally fill
            //    on the first spec-decode window's draft).
            // HIPFIRE_V4F_PREFILL_SKIP_MTP=1 disables the MTP fill entirely
            // (cold MTP cache). Diagnostic: lets us A/B "batched main +
            // MTP fill" vs "batched main + no MTP" to localise whether
            // an accept-rate drop comes from the MTP fill code itself
            // or from batched-main FP noise affecting MTP draft accuracy.
            let skip_mtp_fill = std::env::var("HIPFIRE_V4F_PREFILL_SKIP_MTP")
                .ok().as_deref() == Some("1");
            let mtp_end_b = if skip_mtp_fill {
                0
            } else if is_last_chunk {
                chunk_size.saturating_sub(1)
            } else {
                chunk_size
            };
            // MTP fill inner loop — TIMED with device sync after each call.
            // Skip lm_head + logits d2h during the fill loop: those outputs
            // are never read here, and the d2h syncs the stream per call.
            // The env is scoped to this loop and cleared before spec-decode.
            std::env::set_var("HIPFIRE_V4F_MTP_SKIP_HEAD", "1");
            let t_mtp_total = Instant::now();
            for b in 0..mtp_end_b {
                let absolute_pos = pos_cursor + b;
                // Copy h_i (= streams_batch[b]) into mtp_last_hidden so
                // mtp_forward reads it as h_n.
                let off = b * stream_len;
                let slice = pbs.streams_batch.sub_offset(off, stream_len);
                let dst = state.mtp_last_hidden.as_ref().unwrap();
                gpu.memcpy_dtod_auto(&dst.buf, &slice.buf, stream_len * 4)
                    .map_err(|e| format!("d2d streams[{b}]→mtp_last_hidden: {e:?}"))?;
                let next_tok = prompt_tokens[absolute_pos + 1];
                state.n_tokens = absolute_pos as u64;
                // mtp_forward reads state.pos_array_device via pos_slot()
                // for the MTP layer's RoPE position. Populate it here
                // (the batched main forward uses pbs.positions instead and
                // doesn't touch pos_array_device, so the per-position MTP
                // can't piggyback on its state).
                precompute_positions(&cfg, &mut state, &mut gpu, absolute_pos as u32)?;
                let hidden_ptr: *const rdna_compute::GpuTensor =
                    state.mtp_last_hidden.as_ref().unwrap();
                let hidden: &rdna_compute::GpuTensor = unsafe { &*hidden_ptr };
                let t_each = Instant::now();
                let _ = mtp_forward(
                    &cfg, &weights, &mut state, &mut gpu, hidden, next_tok,
                    absolute_pos as u32,
                )?;
                gpu.hip.device_synchronize()
                    .map_err(|e| format!("sync after mtp_forward b={b}: {e:?}"))?;
                mtp_us_each.push(t_each.elapsed().as_micros());
            }
            mtp_us_total += t_mtp_total.elapsed().as_micros();
            // Restore default so spec-decode draft steps compute logits.
            std::env::remove_var("HIPFIRE_V4F_MTP_SKIP_HEAD");

            pos_cursor += chunk_size;
            state.n_tokens = pos_cursor as u64;

            if is_last_chunk {
                // Head on the last position of the last chunk → last_logits.
                last_logits = final_norm_and_head_last_batched(
                    &cfg, &weights, &mut state, &pbs, &mut gpu, chunk_size,
                )?;
            }
        }
        // Spec decode generation calls mtp_forward which reads
        // pos_array_device + attn_state_buf via pos_slot(). The batched
        // main path doesn't touch them; if HIPFIRE_V4F_PREFILL_SKIP_MTP=1
        // also skipped the inner precompute_positions, the arrays are
        // un-init at this point. Stage them for the LAST prompt position
        // so generation can start cleanly.
        let last_pos = (prompt_tokens.len() - 1) as u32;
        precompute_positions(&cfg, &mut state, &mut gpu, last_pos)?;
        // ── Lever 1 measurement breakdown ──────────────────────────────
        eprintln!("\n--- prefill stage breakdown ---");
        eprintln!(
            "  main forward total: {:>10} us  ({:.2}s)",
            main_us_total, main_us_total as f64 / 1e6,
        );
        eprintln!(
            "  mtp fill total    : {:>10} us  ({:.2}s)  [{} calls]",
            mtp_us_total, mtp_us_total as f64 / 1e6, mtp_us_each.len(),
        );
        if !mtp_us_each.is_empty() {
            mtp_us_each.sort();
            let n = mtp_us_each.len();
            let med = mtp_us_each[n / 2];
            let min = mtp_us_each[0];
            let max = mtp_us_each[n - 1];
            let mean = mtp_us_each.iter().sum::<u128>() / n as u128;
            eprintln!(
                "  mtp per-call us   : min={min} med={med} mean={mean} max={max}"
            );
        }
        let total = main_us_total + mtp_us_total;
        if total > 0 {
            eprintln!(
                "  mtp share of stage: {:.1}%",
                100.0 * mtp_us_total as f64 / total as f64
            );
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
