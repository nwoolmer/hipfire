#!/usr/bin/env python3
"""Multi-prompt coherence harness for MQ2-Lloyd quant iteration.

Runs ~10 prompts through `coherence_probe` against the supplied model,
aggregates pass/fail/severity per detector, and emits a single summary
score so we can compare quant variants apples-to-apples.

Usage:
    mq2lloyd_coherence_harness.py <model.hfq> [--max-tokens N] [--json OUT]

The prompts deliberately cover the failure modes observed in the
2026-05-15 chat session:
  - Code-gen with iterative follow-ups (attractor risk)
  - Short / ambiguous acknowledgments (empty-think risk)
  - Math precision (precision-loss-visible to argmax)
  - SIMD optimization (long-context coherence)
  - The sheep prompt (sanity)

Output JSON shape:
    {
      "model": "...",
      "n_prompts": N,
      "per_prompt": [
         {"label": "...", "verdict": "ok|warn|fail", "fired": [...], "tok_s": ...},
         ...
      ],
      "score": {
         "n_ok": K,
         "n_warn": K,
         "n_fail": K,
         "attractor_first128_max_freq": worst across prompts,
         "attractor_last128_max_freq": worst,
         "ngram_density_worst": worst,
         "n_empty_think": count
      }
    }
"""
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path

# Prompts ordered roughly worst-case-first so failures show up quickly.
PROMPTS = [
    ("fibonacci_c",
     "Generate a fibonacci function in C. Show the complete code with comments.",
     400),
    ("simd_followup",
     "Here is a C function:\n\n```c\nuint64_t fibonacci(uint64_t n) {\n    if (n == 0) return 0;\n    uint64_t a = 0, b = 1;\n    for (uint64_t i = 2; i <= n; ++i) {\n        uint64_t temp = a + b;\n        a = b;\n        b = temp;\n    }\n    return b;\n}\n```\n\nCan this be optimised with SIMD? Explain your reasoning briefly.",
     400),
    ("ack_short_1",
     "Nicely formatted!",
     200),
    ("ack_short_2",
     "Thanks, that's great.",
     200),
    ("math_precision",
     "What is 1248 multiplied by 37? Show your work step by step.",
     250),
    ("sheep_classic",
     "A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.",
     200),
    ("lru_summary",
     "Briefly answer in two short paragraphs: Why is a doubly linked list (rather than singly linked) necessary for an LRU cache that maintains recency order in O(1)? Keep under 100 words.",
     300),
    ("python_classic",
     "Write a Python function `reverse_linked_list(head)` that reverses a singly linked list iteratively. Include a Node class definition.",
     400),
    ("explain_concept",
     "Explain in two short sentences what an MoE (Mixture-of-Experts) model is and why it can be efficient at inference time.",
     200),
    ("haiku",
     "Write three haiku about quantization. Each one must follow the 5-7-5 syllable pattern.",
     250),
]

def run_prompt(model, label, prompt, max_tokens):
    """Run one prompt through coherence_probe, return the parsed JSON report."""
    prompt_file = Path(f"/tmp/coh_harness_prompt_{os.getpid()}_{label}.txt")
    report_file = Path(f"/tmp/coh_harness_report_{os.getpid()}_{label}.json")
    prompt_file.write_text(prompt)
    cmd = [
        "/home/nick/.hipfire/src/target/release/examples/coherence_probe",
        "--model", model,
        "--prompt-file", str(prompt_file),
        "--max-tokens", str(max_tokens),
        "--temperature", "0.0",
        "--report-json", str(report_file),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    wall = time.time() - t0
    if report_file.exists():
        try:
            report = json.loads(report_file.read_text())
        except json.JSONDecodeError:
            report = {"raw": report_file.read_text()[:1000]}
        report_file.unlink()
    else:
        report = {"no_report": True, "stderr": proc.stderr[-500:]}
    prompt_file.unlink()
    report["_wall_s"] = wall
    report["_exit"] = proc.returncode
    return report

def score_report(report):
    """Distil one prompt's coherence_probe report into headline numbers.

    coherence_probe JSON schema (from PR #194):
      header.{total_tokens, tok_s, gen_tok_s, daemon_*}
      rows[].{name, status, severity, detail}
      hard_fails, soft_warns (top-level counters)
    """
    header = report.get("header", {}) or {}
    rows = report.get("rows", []) or []
    hard_fails = report.get("hard_fails", 0)
    soft_warns = report.get("soft_warns", 0)

    out = {
        "verdict": "ok",
        "fired": [],
        "details": {},
        "n_tokens": header.get("total_tokens", 0),
        "gen_tok_s": header.get("gen_tok_s", 0.0),
        "empty_think": False,
        "attractor_first128": False,
        "attractor_last128": False,
        "ngram_density_fired": False,
        "loop_guard_mirror_fired": False,
        "wall_s": report.get("_wall_s", 0.0),
        "exit": report.get("_exit", -1),
    }
    if hard_fails > 0:
        out["verdict"] = "fail"
    elif soft_warns > 0:
        out["verdict"] = "warn"

    for r in rows:
        if r.get("status") != "fired":
            continue
        name = r.get("name", "")
        sev = r.get("severity", "warn")
        detail = r.get("detail", "")
        out["fired"].append(name)
        out["details"][name] = detail
        if name == "think_empty":
            out["empty_think"] = True
        elif name == "attractor_first_128":
            out["attractor_first128"] = True
        elif name == "attractor_last_128":
            out["attractor_last128"] = True
        elif name == "ngram_density":
            out["ngram_density_fired"] = True
        elif name == "loop_guard_mirror":
            out["loop_guard_mirror_fired"] = True

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to .hfq model")
    ap.add_argument("--json", default=None, help="output aggregate JSON to this file")
    ap.add_argument("--label", default=None, help="label this run (for the JSON output)")
    ap.add_argument("--quick", action="store_true", help="only run the first 5 prompts")
    args = ap.parse_args()

    model = str(Path(args.model).expanduser().resolve())
    if not Path(model).exists():
        print(f"ERROR: model not found: {model}", file=sys.stderr)
        sys.exit(2)

    label = args.label or Path(model).stem
    prompts = PROMPTS[:5] if args.quick else PROMPTS

    print(f"=== coherence harness: {label} ===")
    print(f"model:   {model}")
    print(f"prompts: {len(prompts)}")
    print()

    results = []
    for prompt_label, prompt, max_tok in prompts:
        print(f"  [{prompt_label}] running...", flush=True)
        report = run_prompt(model, prompt_label, prompt, max_tok)
        scored = score_report(report)
        scored["label"] = prompt_label
        results.append(scored)
        marker = {"ok": " ✓ ", "warn": " ! ", "fail": " × "}.get(scored["verdict"], " ? ")
        flags = []
        if scored["attractor_first128"]: flags.append("ATTR1")
        if scored["attractor_last128"]:  flags.append("ATTR2")
        if scored["ngram_density_fired"]: flags.append("NGRAM")
        if scored["loop_guard_mirror_fired"]: flags.append("LOOP")
        if scored["empty_think"]: flags.append("ETHK")
        flag_str = ",".join(flags) if flags else "-"
        print(f"  [{prompt_label:18s}]{marker}{scored['verdict']:5s} "
              f"n_tok={scored['n_tokens']:3d} gen={scored['gen_tok_s']:5.1f}/s "
              f"flags={flag_str}")

    n_ok = sum(1 for r in results if r["verdict"] == "ok")
    n_warn = sum(1 for r in results if r["verdict"] == "warn")
    n_fail = sum(1 for r in results if r["verdict"] == "fail")
    aggregate = {
        "label": label,
        "model": model,
        "n_prompts": len(results),
        "n_ok": n_ok,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "n_attractor_first128": sum(1 for r in results if r["attractor_first128"]),
        "n_attractor_last128":  sum(1 for r in results if r["attractor_last128"]),
        "n_ngram_density":      sum(1 for r in results if r["ngram_density_fired"]),
        "n_loop_guard_mirror":  sum(1 for r in results if r["loop_guard_mirror_fired"]),
        "n_empty_think":        sum(1 for r in results if r["empty_think"]),
        "total_tokens_gen":     sum(r["n_tokens"] for r in results),
        "mean_gen_tok_s":       (sum(r["gen_tok_s"] for r in results if r["gen_tok_s"] > 0) /
                                 max(1, sum(1 for r in results if r["gen_tok_s"] > 0))),
        "per_prompt": results,
    }
    print()
    print(f"=== summary: {label} ===")
    print(f"  verdict counts:        ok={n_ok}  warn={n_warn}  fail={n_fail}  (of {len(results)})")
    print(f"  attractor_first128:    {aggregate['n_attractor_first128']}")
    print(f"  attractor_last128:     {aggregate['n_attractor_last128']}")
    print(f"  ngram_density:         {aggregate['n_ngram_density']}")
    print(f"  loop_guard_mirror:     {aggregate['n_loop_guard_mirror']}")
    print(f"  empty_think (warn):    {aggregate['n_empty_think']}")
    print(f"  total tokens generated: {aggregate['total_tokens_gen']}")
    print(f"  mean gen tok/s:         {aggregate['mean_gen_tok_s']:.1f}")
    if args.json:
        Path(args.json).write_text(json.dumps(aggregate, indent=2))
        print(f"  → JSON written to {args.json}")

if __name__ == "__main__":
    main()
