# Pre-merge cleanup audit — feat/deepseek4-v4f → master

**Branch state**: 320 commits ahead of master, +149,189 / -11,074 LoC across 465 files.

**Production V4F path** (verified safe — never recommend removing anything below):
- `crates/hipfire-arch-deepseek4/examples/v4f_chat.rs` → `forward_prefill_batch_chunked` (default `HIPFIRE_V4F_PP_BATCH=16`) + `decode_step_with_graph`.
- `crates/hipfire-arch-deepseek4/examples/v4f_mtp_smoke.rs` → MTP / spec-decode test harness.
- `crates/hipfire-arch-deepseek4/examples/v4f_perplexity.rs` → PPL evaluator.
- `dev/bench/v4f_multi_turn/run.sh` → harness that invokes `target/release/examples/v4f_chat`.
- All four production paths consume only the env vars enumerated in the brief: `HIPFIRE_V4F_{MODEL,GEN_TOKENS,TEMP,TOP_K,SEED,PP_BATCH,GRAPH,SPEC_DECODE,SPEC_K,CHAT_RAW,ATTN}` (and downstream-set `MTP_SKIP_HEAD`, `LOAD_MTP`).

The audit found nothing reachable from these paths that is being recommended for removal in section A.

---

## A. Falsified opt-in experiments — RECOMMEND REMOVING

These three rows are the explicit "kept as opt-in dead code, measured slower/null on V4F" items called out in the brief. They sit on the `feat/deepseek4-v4f` branch, are gated behind dormant env vars that nothing in production sets, and the kernels backing them are only reachable through those guards.

### A.1 — `HIPFIRE_V4F_ATTN_WMMA` (WMMA Q·K^T attention variant)

- **Falsification evidence**: commit `956ae5c` ships it as `+0.8% / noise`. Comment in `forward.rs:4926-4930` records the F32 path is L2-BW-bound (91% L2 hit, 95% MemUnit busy), WMMA "increases compute per L2 byte" — no real headroom on the bound axis.
- **Source removals**:
  - `crates/hipfire-arch-deepseek4/src/forward.rs:4924-4955` — drop the `attn_wmma` branch and inline the `else` body (the existing `v4f_attn_swa_topk_batched_f32` call) into the unconditional path.
  - `crates/rdna-compute/src/dispatch.rs:21434-end-of-fn` — drop `v4f_attn_swa_topk_batched_wmma_f32`.
  - `crates/rdna-compute/src/kernels.rs:2021` — drop the `include_str!` for `v4f_attn_swa_topk_batched_wmma.hip`.
  - `kernels/src/v4f_attn_swa_topk_batched_wmma.hip` — delete.
- **Env var removal**: `HIPFIRE_V4F_ATTN_WMMA`.
- **Risk**: zero. Production never sets `HIPFIRE_V4F_ATTN_WMMA=1`, and the kernel has no other dispatcher.

### A.2 — `HIPFIRE_V4F_MQ2_WMMA` (single-col MQ2-Lloyd MoE WMMA)

- **Falsification evidence**: commit `f23cc3c` ships the dispatch wiring, comment at `forward.rs:5260-5271` records "FALSIFIED at B=64, 2026-05-21 … 3× SLOWER than K4 in production. Reason: single-col WMMA wastes 15/16 of WMMA hardware". Same fact in memory entry `feedback_v4f_prefill_optimization_dead_ends`.
- **Source removals**:
  - `crates/hipfire-arch-deepseek4/src/forward.rs:5272-5273` (the `mq2_wmma` env read) and the two `if mq2_wmma { … }` branches at `~5276-5285` (gate-up) and `~5323-5335` (down). Replace with the existing K4 path.
  - `crates/rdna-compute/src/dispatch.rs:23675-23744` — drop `gemm_mq2g256_lloyd_moe_gate_up_wmma` and `gemm_mq2g256_lloyd_moe_down_residual_scaled_wmma`.
  - `crates/rdna-compute/src/kernels.rs:1890,1895` — drop both `include_str!`s.
  - `kernels/src/gemm_mq2g256_lloyd_moe_gate_up_wmma.hip`, `kernels/src/gemm_mq2g256_lloyd_moe_down_residual_scaled_wmma.hip` — delete.
- **Env var removal**: `HIPFIRE_V4F_MQ2_WMMA`.
- **Risk**: zero. Same: production never sets it; no other caller.

### A.3 — `HIPFIRE_V4F_COMP_F16_WMMA_FUSED` (ZA-fused F16 WMMA compressor)

- **Falsification evidence**: commit `b3a0836`. Comment at `forward.rs:4582-4592` and memory entry `feedback_v4f_za4_f16_wmma_falsified` both record −1.5% at B=64; "same block count, fused pays routing-branch overhead w/o new parallelism".
- **Source removals**:
  - `crates/hipfire-arch-deepseek4/src/forward.rs:4593-4639` — drop the `za_fused` env read and both `if za_fused` arms; keep only the `else` legacy 4-call dispatch (which is unconditionally used by production today).
  - `crates/rdna-compute/src/dispatch.rs:24028-…` — drop `gemm_f16_x_f16_wmma_za4` wrapper.
  - `crates/rdna-compute/src/kernels.rs:1874` — drop the `include_str!`.
  - `kernels/src/gemm_f16_x_f16_wmma_za4.hip` — delete.
- **Env var removal**: `HIPFIRE_V4F_COMP_F16_WMMA_FUSED`.
- **Risk**: zero, by the same argument.

### A.4 — `v4f_fused_silu_mul_clamp_mq_rotate_batched` dispatcher

- **Status**: the *batched* dispatcher at `crates/rdna-compute/src/dispatch.rs:3833-3883` has **no callers** (grep across all crates returns only the dispatch.rs definitions themselves). The brief explicitly calls it "kept as opt-in dead code, falsified −1.4% earlier this session".
- **Important — keep the underlying kernel + non-batched wrapper**: the **non-batched** `v4f_fused_silu_mul_clamp_mq_rotate` (dispatch.rs:3778) **is live** at `forward.rs:2105` (used in MTP forward shared-FFN). The kernel source file `kernels/src/v4f_fused_silu_mul_clamp_mq_rotate.hip` is still required and must NOT be deleted.
- **Source removals**:
  - `crates/rdna-compute/src/dispatch.rs:3829-3883` — drop the `_batched` wrapper only.
- **Env var removal**: none (the wrapper had no env gate, just no callers).
- **Risk**: zero — orphan code.

---

## B. Falsified opt-in experiments — KEEP AS DOCUMENTED FOUNDATION

Things that are "measured worse on V4F today, but the dispatch wiring or sibling kernel is the cheapest part of the future re-test cost". Each already has the right "opt-in, measured slower" comment. Default OFF — they cost only ROM in the binary.

| Env var | Where | Why keep |
|---|---|---|
| `HIPFIRE_V4F_MOE_GROUPED` | `forward.rs:5249-5258` (+ scatter-by-expert kernels `gemv_mq2g256_lloyd_moe_*_grouped_k4.hip`) | Comment records 18% slower on Radeon 8060S BUT explicitly notes "trade flips" at larger batches/other archs. Production-default is OFF (`==Some("1")`). Sibling-kernel cost is small. |
| `HIPFIRE_V4F_F16_WMMA` | `forward.rs:198` (B=1 GEMV F16 path) | Comment at 191-196 records F16-WMMA loses ~13 mantissa bits → worse quality, but path is needed for "perf comparison" + future kernel work. Default OFF. |
| `HIPFIRE_V4F_F16XF32_R2` | `forward.rs:103-112` | Comment records +0.7% within noise on this build, "kept for future tuning". Default OFF. |
| `HIPFIRE_V4F_WO_A_WMMA` | `forward.rs:4339-4357` | Comment at 4332-4338 records 44.7 vs 45.6 tok/s on 8060S (slower, ~2%). "Keep WMMA wired as opt-in for other arches". Default OFF. |
| `HIPFIRE_V4F_GEMM_F32_BT32` / `_W2` (internal in `gemm_f32_register_tiled`) | `dispatch.rs:19410-19435` | Both gated, marginally slower today. Comment records the why; sibling kernels `gemm_f32_register_tiled_{bt32,w2}.hip` provide the future regression-investigation A/B without a rewrite. Default OFF. |
| `HIPFIRE_V4F_MTP_NO_ROUTED` | `forward.rs:114-123` | Documented as accept-rate-vs-latency knob; not falsified, trade-space. |
| `HIPFIRE_V4F_CPU_TOPK` | `forward.rs:71-74,2183` | Parity-testing fallback. Documented as such. Keep. |
| `HIPFIRE_V4F_NO_HASH`, `HIPFIRE_V4F_NO_FUSED_MOE`, `HIPFIRE_V4F_NO_COMPRESSOR`, `HIPFIRE_V4F_NO_MAIN_ROPE`, `HIPFIRE_V4F_NO_YARN`, `HIPFIRE_V4F_NO_MIXED`, `HIPFIRE_V4F_SKIP_FFN`, `HIPFIRE_V4F_SKIP_INV_ROPE`, `HIPFIRE_V4F_SKIP_QHN`, `HIPFIRE_V4F_BISECT_BREAK`, `HIPFIRE_V4F_DUMP_MAG`, `HIPFIRE_V4F_DUMP_PHASE5_K`, `HIPFIRE_V4F_F32_TRACE`, `HIPFIRE_V4F_FORWARD_LAYER_END`, `HIPFIRE_V4F_PREFILL_PATH`, `HIPFIRE_V4F_PREFILL_BATCHED`, `HIPFIRE_V4F_F16_DECODE` | `forward.rs` and `arch.rs` various | These are diagnostic / bisection / fallback knobs (49 call sites total). Per the project's "validate every claim before acting" discipline, they're the cheap-diagnostic toolkit. Each costs one env-var read. Keep. |
| `HIPFIRE_V4F_RUN_COMPRESSOR`, `HIPFIRE_V4F_COMP_BATCHED_GEMV`, `HIPFIRE_V4F_COMP_FULLY_BATCHED`, `HIPFIRE_V4F_COMP_ROPE_POS`, `HIPFIRE_V4F_WO_A_BATCHED`, `HIPFIRE_V4F_GATEUP_K4`, `HIPFIRE_V4F_DOWN_K4`, `HIPFIRE_V4F_POST_SCALE`, `HIPFIRE_V4F_ROUTE_SCALE`, `HIPFIRE_V4F_HFQ4_WMMA`, `HIPFIRE_V4F_Q8_WMMA`, `HIPFIRE_V4F_MAX_COMPRESS_POS`, `HIPFIRE_V4F_ASYNC_HTOD`, `HIPFIRE_V4F_PBS_VRAM`, `HIPFIRE_V4F_MEM_GUARD_GB` | `forward.rs` / `arch.rs` / `deepseek4.rs` | These are **default-on perf knobs with measured wins** — opt-OUT semantics, not opt-in. They live for the (necessary) ability to roll back a fast path if a regression is observed. Keep all. |

---

## C. One-off probe / bench examples — RECOMMEND REMOVING

These live in `crates/hipfire-arch-deepseek4/examples/` and are not referenced by `scripts/coherence-gate.sh`, `scripts/coherence-gate-dflash.sh`, `scripts/pp-gate.sh`, `scripts/probe_commits.sh`, `.githooks/pre-commit`, or `dev/bench/v4f_multi_turn/run.sh`. Most have their results captured in either committed `docs/plans/v4f-*.md` files or `memory/` entries. The brief calls for aggressive pruning of investigation-only probes.

| File | Original purpose (docstring) | Why redundant |
|---|---|---|
| `forward_step1.rs` | "Exercise the first step of V4F forward (embed → 4-stream residual init) against the real V4F HFQ file". | Phase-1 bring-up smoke from `deepseek4-bringup.md`. The full forward has been validated end-to-end via PPL + chat for months. |
| `load_v4f_check.rs` | "Sanity-load the V4F HFQ file produced by Phase 1 ingest". | Phase-1 ingest. Superseded by `v4f_perplexity` / `v4f_chat` doing real loads on every run. |
| `test_hc_compute.rs`, `test_hc_mix.rs`, `test_hc_sinkhorn.rs` | "GPU validation for Phase 3 `hc_compute_control` / `hc_mix_4stream` / `hc_sinkhorn_4x4`". | Phase-3 indexer/HC unit tests. The HC stack has been wired through prefill + decode and PPL-validated; these standalone smokes have no remaining role. |
| `test_indexer_score.rs`, `test_indexer_gather.rs`, `test_indexer_top_k.rs`, `test_indexer_top_k_batched.rs` | "GPU validation for Phase 2 `indexer_*`". | Phase-2 stub validators. Same as above — the indexer is in production. |
| `test_rope_tail.rs` | "GPU validation for Phase 4 `rope_tail_halfsplit_f32`". | Phase-4 stub validator. RoPE tail is in production. |
| `test_v4f_attn_swa_batched.rs`, `test_v4f_attn_swa_topk_batched.rs`, `test_hc_batched.rs`, `test_hc_streams_init_batched.rs`, `test_indexer_top_k_batched.rs` (already above), `test_gemv_auto_batched_f32.rs` | "Byte-equality smoke test for Phase A/B batched kernels". | Investigation-phase byte-equality probes mentioned in `docs/plans/v4f-batched-prefill.md` (lines 81/85/90/109). The batched-prefill path has shipped; these are one-time gates that already passed and aren't re-run. |
| `bench_v4f_forward.rs` | "Time each stage of V4F forward to understand the per-layer cost". | One-off investigation; data captured in `project_v4f_sequential_prefill_profile`. |
| `bench_decode_vs_ctx.rs` | "Bench `decode_step` wall time vs context position". | Investigation. Plans/perf-plan.md refers to it but no script automates it. Data captured in `project_v4f_prefill_idle_gap_analysis_2026_05_20`. |
| `bench_expert_upload.rs` | "Time the V4F batched expert upload". | One-off; landed result is the default-on batched uploader. |
| `bench_kernel_transition.rs` | Microbench for "intra-graph idle between specific kernel transitions". | One-off ("hc_pre_post_sigmoid_scale → hc_sinkhorn_4x4 fusion candidate" investigation, captured in idle-gap analysis memo). |
| `bench_moe_kernel_apples.rs` | "Apples-to-apples MoE kernel TFLOPs comparison: MQ2-Lloyd vs HFQ4". | One-off, results in `project_v4f_prefill_kernel_bw_audit_2026_05_19`. |
| `bench_mq2_wmma_smoke.rs` | "Smoke / proof-of-concept bench for the MQ2-Lloyd WMMA kernel". | Drives the **A.2 falsified path**. With A.2 removed, this bench targets a deleted kernel and must go too. |
| `bench_v4f_batched_prefill.rs` | "V4F batched prefill smoke + benchmark". | Phase-A/B investigation. Plans reference it but no CI does. Result data in `project_v4f_prefill_optimization_2026_05_19`. Marginal; could be promoted into `dev/bench/` if you want a permanent harness — **KEEP — needs human review** if you'd rather promote than delete. |
| `bisect_v4f_batched.rs` | "V4F batched-prefill divergence bisector". | One-off divergence-hunt. Bisection completed; B=64 vs sequential drift documented in `feedback_v4f_prefill_poor_batched_speedup`. |
| `decode_timing.rs` | "Quick single-token decode timing for V4F (no cache fill)". | Subsumed by `v4f_chat`'s built-in PP/TG line + `dev/bench/v4f_multi_turn`. |
| `decode_at_pos.rs` | "Fill context to a target position, then time N decode steps at that position". | Same — rocprof-driven one-off. |
| `profile_decode.rs` | "Profile V4F decode to identify the hottest kernels". | rocprof workflow. Result captured. |
| `profile_v4f_batched_prefill.rs` | "Minimal driver for rocprofv3 — runs `forward_prefill_batch_chunked` once". | Same. |
| `profile_v4f_two_chunk.rs` | "Two-chunk profiler — distinguishes chunk 1 vs chunk 2 kernel cost". | Same. |
| `graph_drift_check.rs` | "Run V4F decode with and without HIP graphs, compare argmax/top-1". | Phase-bringup HIP-graphs drift check. The drift gate (200/200) shipped in `4d160e0`. **KEEP — needs human review**: this is the only standalone graph-vs-direct A/B harness, and the SWA-kernarg bug memo (`feedback_v4f_hipgraphs_swa_kernarg_bug`) explicitly notes a future need to re-verify when graphs are extended. |
| `v4f_top_logits.rs` | "Dump top-K predicted tokens after a prompt — diagnostic for ppl issues". | Diagnostic one-off. PPL evaluator covers the same ground for any future investigation. |
| `v4f_generate.rs` | "Run V4F decode_step in a loop … no KV cache currently". | Comment "no KV cache currently, so context isn't actually carried" makes clear this is pre-cache scaffolding. Superseded by `v4f_chat` / `v4f_prompt`. |
| `v4f_long_context_test.rs` | "long-context fact-recall regression test — hipfire equivalent of antirez/ds4 `--long-context`". | **KEEP — needs human review**: this is a *capability* eval, not a perf probe. If you want long-context recall to stay covered by an automated artifact for the merge, leave it. |
| `v4f_eval_smoke.rs` | "V4F capability smoke eval — minimal port of antirez/ds4 `ds4-eval`". | **KEEP — needs human review**: capability smoke (GPQA / SuperGPQA / AIME / COMPSEC subset). Same argument as `v4f_long_context_test`. |
| `v4f_quality_collect.rs`, `v4f_quality_score.rs` | "antirez `collect_official.py` / `score_official.c` equivalents". | **KEEP — needs human review**: quality-eval pair (greedy continuation + teacher-forced NLL/argmax) referenced in `docs/plans/issue-113-quant-quality-eval.md`. Keep until that issue closes. |
| `v4f_prompt.rs` | "Feed V4F a multi-token prompt and let it continue (HIPFIRE_V4F_ATTN=swa)". | Superseded by `v4f_chat --raw`. Marginal; **KEEP — needs human review** if you'd rather not break a "give me a one-shot continuation" workflow. |

Net concrete C-list (high-confidence deletions): `forward_step1.rs`, `load_v4f_check.rs`, `test_hc_{compute,mix,sinkhorn,batched,streams_init_batched}.rs`, `test_indexer_{score,gather,top_k,top_k_batched}.rs`, `test_rope_tail.rs`, `test_v4f_attn_swa_{batched,topk_batched}.rs`, `test_gemv_auto_batched_f32.rs`, `bench_v4f_forward.rs`, `bench_decode_vs_ctx.rs`, `bench_expert_upload.rs`, `bench_kernel_transition.rs`, `bench_moe_kernel_apples.rs`, `bench_mq2_wmma_smoke.rs`, `bisect_v4f_batched.rs`, `decode_timing.rs`, `decode_at_pos.rs`, `profile_decode.rs`, `profile_v4f_batched_prefill.rs`, `profile_v4f_two_chunk.rs`, `v4f_top_logits.rs`, `v4f_generate.rs`.

= 23 files. Each is a `cargo` example binary; no Cargo.toml edits needed (examples are auto-discovered, the deepseek4 `Cargo.toml` declares no `[[example]]` sections).

---

## D. Probe / bench examples — KEEP

These ARE permanent V4F infrastructure (production user-facing or evaluation infrastructure with no replacement):

| File | Role |
|---|---|
| `v4f_chat.rs` | Production interactive chat binary. Wired into `dev/bench/v4f_multi_turn/run.sh`. |
| `v4f_mtp_smoke.rs` | Spec-decode test harness (per brief). |
| `v4f_perplexity.rs` | PPL evaluator (per brief). |
| `v4f_long_context_test.rs` | Long-context (35K-token) fact recall regression eval. *Confirm with human before deleting.* |
| `v4f_eval_smoke.rs` | GPQA/SuperGPQA/AIME/COMPSEC subset capability eval. *Confirm with human.* |
| `v4f_quality_collect.rs`, `v4f_quality_score.rs` | Quality-eval pair tied to `docs/plans/issue-113-quant-quality-eval.md`. |
| `graph_drift_check.rs` | Sole graph-vs-direct drift verifier; needed for any future HIP-graphs extension (see `feedback_v4f_hipgraphs_swa_kernarg_bug`). |
| `v4f_prompt.rs` | Single-prompt continuation harness. *Marginal — confirm with human.* |
| `bench_v4f_batched_prefill.rs` | Sole batched-prefill smoke harness. *Marginal — confirm with human; the alternative is promoting it into `dev/bench/`.* |

---

## E. Dead / one-off scripts in `scripts/`

Cross-reference: `.githooks/pre-commit` invokes `scripts/coherence-gate.sh`, `scripts/coherence-gate-dflash.sh`, and (conditionally) `scripts/pflash-gate.sh`. `scripts/probe_commits.sh` is invoked manually. `scripts/pp-gate.sh` is project-wide. `scripts/speed-gate.sh` is invoked by the same hook flow.

The scripts NEWLY added on this branch (per `git log --diff-filter=A`) are:

```
scripts/agentic-gate-jinja-tools.sh
scripts/analyze_quant_mse.py
scripts/astrea.py                       — see Section F
scripts/awq_alpha_sweep.sh
scripts/awq_coherence_check.sh
scripts/awq_f1_vs_f2.sh
scripts/awq_f2_alpha_sweep_wait.sh
scripts/bench_humaneval_completion.sh
scripts/bench_quant_quality.sh
scripts/compare_hidden_states.py
scripts/compare_layer_positions.py
scripts/cross_engine_check.py
scripts/dump_hf_hidden_states.py
scripts/fetch-eval-refs.sh
scripts/kernel_atlas.py                 — see Section F
scripts/mq2lloyd_coherence_harness.py
scripts/quant_cohort.sh
scripts/test_astrea.py                  — see Section F (Astrea unit tests, callable from `python3 -m unittest`)
```

None of these are referenced by `coherence-gate.sh`, `coherence-gate-dflash.sh`, `pp-gate.sh`, `probe_commits.sh`, or `pre-commit`. Categorization:

**Likely PERMANENT, KEEP** (tied to a documented workflow that the branch is shipping):
- `scripts/fetch-eval-refs.sh` — referenced by `docs/plans/issue-113-quant-quality-eval.md` (commit `dec913e`). Quality-eval infra.
- `scripts/bench_quant_quality.sh`, `scripts/analyze_quant_mse.py`, `scripts/quant_cohort.sh` — quant-quality eval workflow (commit `0d733b1` "Phase A Step 0+4 — cohort runner, imatrix collector, MSE harness"). Tied to AWQ + quant-quality work.
- `scripts/awq_*` — AWQ feature shipped via PR #273 + #284 (`fcb23c7`, `3f5e8b0`). Part of AWQ rollout.
- `scripts/agentic-gate-jinja-tools.sh` — Jinja-tools phase shipped (`e301c27` "Jinja-everywhere phase 1-3a").
- `scripts/dump_hf_hidden_states.py`, `scripts/compare_hidden_states.py`, `scripts/compare_layer_positions.py`, `scripts/cross_engine_check.py` — engine drift diagnostics (commit `5726dbe` "salvage engine drift diagnostics"). Diagnostic but referenced by the salvage commit.

**KEEP — needs human review** (no obvious consumer; could be moved or deleted):
- `scripts/mq2lloyd_coherence_harness.py` — MQ2-Lloyd-specific. The MQ2-Lloyd format is shipping (`feat/deepseek4-v4f` uses it as the V4F default routed-MoE quant), so this is plausibly the per-format coherence probe. Keep, but if MQ2-Lloyd already has coverage in `coherence-gate.sh` then this is redundant.
- `scripts/bench_humaneval_completion.sh` — HumanEval-completion driver. Not in any gate. Could be a permanent quality bench or a one-off — confirm.

I did not find scripts that are flat-out unreachable from any documented workflow; nothing in this section is a high-confidence deletion candidate.

---

## F. Big special-purpose scripts (`astrea.py`, `kernel_atlas.py`, `test_astrea.py`)

All three landed in a single commit: `f222c0a` ("Add Python Astrea and Atlas tooling"). The Atlas commit also ships `.agents/skills/astrea/SKILL.md`, which is the agent-orchestration contract:

- **`scripts/astrea.py` (3,539 LoC)**: Agent-native model calibration CLI. Per its docstring + the `SKILL.md` skill description, it is an *agent-orchestrated* tool (not a one-off). The `astrea` skill is registered under `.agents/skills/` and the CLI defines `inspect / fingerprint / plan / calibrate / eval / bundle-plan` subcommands. **STATUS: standalone tool tied to a registered agent skill. KEEP.**
- **`scripts/kernel_atlas.py` (3,139 LoC)**: "Phase-aware Kernel Atlas collector for hipfire benches." Docstring describes it as a measurement harness that turns existing AR/DFlash bench output into JSONL Atlas corpus rows. Companion to Astrea. Referenced from `docs/methodology/astrea-atlas-pareto-workflow.md` and `docs/methodology/kernel-atlas-architecture.md`. **STATUS: standalone measurement harness, agent-orchestrated. KEEP.**
- **`scripts/test_astrea.py` (1,486 LoC)**: Unit tests for `scripts/astrea.py`. Standard `python3 -m unittest` runnable. **STATUS: unit-test suite for `astrea.py`. KEEP.**

No removals recommended in this section.

---

## G. Stale docs/plans

New docs landed on this branch (`docs/plans/`):

| File | Status (KEEP / DELETE) | Reason |
|---|---|---|
| `antirez-ds4-reference.md` | KEEP | Reference source for V4F arch; cited by memory entry `reference_antirez_ds4`. |
| `deepseek4-bringup.md` | KEEP | Plan-of-record for the V4F arch landing. Historical reference. |
| `deepseek4-phase2-indexer.md` | KEEP | Indexer design doc; Phase-2 work is shipped but doc is canonical reference. |
| `v4f-batched-prefill.md` | KEEP | Active plan referenced by `forward.rs` comments; batched-prefill is live. |
| `v4f-big-levers-roadmap.md` | KEEP | Active roadmap; called out as plan-of-record in commit `9d3284e`. |
| `v4f-mtp-requant-2026-05-20.md` | KEEP | Spec for `v4f.mq2lloyd-mtp.hfq`; current default model is built per this spec. |
| `v4f-perf-plan.md` | KEEP | Historical perf plan; Phase A–F shipped/falsified noted; landing doc. |
| `v4f-prefill-perf-2026-05-20.md` | KEEP | Recent implementation plan, idle-gap-analysis-driven. |
| `v4f-quant-optimization-playbook.md` | KEEP | Active V4F quant playbook. |
| `v4f-sub-mq2-quant-research.md` | KEEP | Most recent, 2026-05-21, current research queue. |
| `v4f-to-100-tps-roadmap.md` | KEEP | Current 100-tok/s roadmap. |
| `eval_hipfire_speedup.md` | KEEP — needs human review | Not opened during this audit; confirm relevance. |
| `awq_bug_hunt_glm5.md`, `awq_fix_claude.md`, `awq_hipfire.md` | KEEP — needs human review | AWQ research files; the AWQ feature shipped — these may be historical or active. Owner should classify. |
| `dflash-vram-bloat-2026-05-15.md` | KEEP — needs human review | One-shot investigation; OK to keep as historical record but flag for the dflash owner. |
| `issue-113-quant-quality-eval.md` | KEEP | Live issue. |
| `mq6_gemm.md` | KEEP | New mq6 GEMM research doc. |
| `oversize-files-modularization.md` | KEEP — needs human review | Refactoring plan; status unknown. |
| `pflash-drafter-asym3-handoff.md` | KEEP | Handoff doc; conventionally kept. |
| `pp-gate-fix.md` | KEEP — needs human review | Implementation note; if PP gate fix has shipped, this is historical. |
| `quant-code-rev-claude.md` | KEEP — needs human review | Code-review note. |
| `qwen35-moe-coherence-investigation.md`, `qwen35-moe-precision-vllm-comparison.md`, `qwen35-mq4-quality-gap.md` | KEEP — needs human review | qwen35 investigation set; orthogonal to V4F merge. |
| `size-limit-ci-gates.md` | KEEP — needs human review | CI/size-limit plan. |
| `tbq4-kv-cache-plan.md` | KEEP — needs human review | TBQ4 plan. |

Net: I have **zero high-confidence delete recommendations** in `docs/plans/`. The plans are uniformly attached either to live code or to scheduled work. The "needs human review" set is the V4F-orthogonal plans (AWQ, qwen35, dflash, etc.) — those are out-of-scope for the V4F merge audit.

---

## H. Unused kernels / dispatchers (V4F-branch additions only)

Tracked against the falsified-opt-in env vars in Section A. The full V4F-branch kernel additions are 147 files; the vast majority are referenced by both `kernels.rs` `include_str!` and `dispatch.rs` wrappers, AND are reachable from at least one default path in `forward.rs` / `mtp_forward` / `attn_*`. The kernels that are reachable ONLY through opt-in env-gated code and not from any default path:

| Kernel file | Reached only via | Removal recommendation |
|---|---|---|
| `kernels/src/v4f_attn_swa_topk_batched_wmma.hip` | `HIPFIRE_V4F_ATTN_WMMA=1` (Section A.1) | **DELETE** (with A.1) |
| `kernels/src/gemm_mq2g256_lloyd_moe_gate_up_wmma.hip` | `HIPFIRE_V4F_MQ2_WMMA=1` (Section A.2) | **DELETE** (with A.2) |
| `kernels/src/gemm_mq2g256_lloyd_moe_down_residual_scaled_wmma.hip` | `HIPFIRE_V4F_MQ2_WMMA=1` (Section A.2) | **DELETE** (with A.2) |
| `kernels/src/gemm_f16_x_f16_wmma_za4.hip` | `HIPFIRE_V4F_COMP_F16_WMMA_FUSED=1` (Section A.3) | **DELETE** (with A.3) |
| `kernels/src/gemm_mq2g256_lloyd_wmma.hip` | Used by `bench_mq2_wmma_smoke.rs` only (see Section C). | **DELETE** if Section C removes `bench_mq2_wmma_smoke.rs`. Verify it has no other dispatch.rs caller before deletion. (My grep found only the kernels.rs include + the smoke bench's invocation.) |
| `kernels/src/bench_q8_fp16wmma.hip` | Microbench-only kernel (1 ref total: `kernels.rs` include). | **DELETE** — orphan. |
| `kernels/src/microbench_dram_peak.hip` | Microbench-only kernel (1 ref total). | **KEEP — needs human review**: it IS the DRAM-peak baseline used by the `crates/rdna-compute/examples/microbench_v4f_kernels.rs` harness referenced in `project_v4f_prefill_kernel_bw_audit_2026_05_19`. Delete only if the user is OK losing the DRAM-peak microbench. |

**Kernels reachable ONLY via opt-in but KEPT (per Section B foundation-keeps):**
- `gemv_mq2g256_lloyd_moe_{gate_up,down}_grouped_k4.hip` — reachable via `HIPFIRE_V4F_MOE_GROUPED=1`; documented as "trade flips at larger batches"; KEEP.
- `gemv_f16_xf32_multirow_r2.hip` — reachable via `HIPFIRE_V4F_F16XF32_R2=1`; KEEP.
- `wo_per_group_batched_hfq4g256_wmma.hip` — reachable via `HIPFIRE_V4F_WO_A_WMMA=1`; documented; KEEP.
- `gemm_f32_register_tiled_bt32.hip`, `gemm_f32_register_tiled_w2.hip` — reachable via `HIPFIRE_GEMM_F32_BT32` / `_W2`; KEEP.

---

## I. Summary recommendation

**Ship after a small bounded cleanup pass.** The branch is structurally sound — production paths (`v4f_chat`, `v4f_mtp_smoke`, `v4f_perplexity`, `forward_prefill_batch_chunked`, `decode_step_with_graph`) are not touched by anything in the recommended-delete list. The opt-in/foundation discipline that the project has been practicing (one env var per experiment, comment with falsification evidence) has paid off here: the falsified experiments are surgically removable.

Concrete pre-merge cleanup, in this order:

1. **A.1 / A.2 / A.3 / A.4** — drop the three falsified WMMA env-gated dispatch arms (`HIPFIRE_V4F_ATTN_WMMA`, `_MQ2_WMMA`, `_COMP_F16_WMMA_FUSED`), the orphan `v4f_fused_silu_mul_clamp_mq_rotate_batched` dispatcher, the four `.hip` files behind them, and the `kernels.rs` `include_str!` entries. ~600 LoC net removal, zero default-path risk.
2. **C (23 high-confidence files)** — delete the 23 phase-bringup / one-off investigation examples called out in Section C. Each is auto-discovered, no `Cargo.toml` edits needed. Keep the four "needs human review" examples (`v4f_long_context_test.rs`, `v4f_eval_smoke.rs`, `v4f_quality_collect.rs`, `v4f_quality_score.rs`, optionally `v4f_prompt.rs` + `bench_v4f_batched_prefill.rs` + `graph_drift_check.rs`) unless the maintainer signs off.
3. **H — orphan kernel `bench_q8_fp16wmma.hip`** — single-ref dead kernel.
4. **Leave all of B, D, E, F, G as-is** — these are either documented foundation, permanent infra, or out-of-scope for this merge.

After (1)–(3), the merge is in good shape. The remaining "needs human review" buckets (Section C marginals, Section E `mq2lloyd_coherence_harness.py` / `bench_humaneval_completion.sh`, Section G AWQ/qwen35 docs) can be deferred to a follow-up tidy commit on master without blocking the merge.

Do **not** block the merge on the "needs human review" items. The V4F arch landing is large but disciplined; the artifacts that need to ship for production V4F use are exactly the ones in Section D, and they are untouched by every recommendation above.
