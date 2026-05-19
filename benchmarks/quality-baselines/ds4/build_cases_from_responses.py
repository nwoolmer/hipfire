#!/usr/bin/env python3
"""Build a hipfire v4f_quality_score input JSONL from upstream API responses.

Handles two source formats:

  (a) **5-case test-vectors** at `test-vectors/official/*.official.json`:
      Each file has `prompt`, `message.content`, and `steps[].token.bytes` /
      `steps[].token.text`. Iterate the manifest at `test-vectors/manifest.json`.

  (b) **100-case API capture** at `<dir>/responses/case_NNN.json`,
      `<dir>/prompts/case_NNN.txt`:
      Each response is the raw DeepSeek API JSON. Token bytes live under
      `choices[0].logprobs.content[].bytes`.

Both sources produce a JSONL with one line per case:
  {"id": "...", "prompt": "...", "target_token_bytes": [[65,100,97], ...]}

The byte arrays preserve upstream's exact token boundaries — the scorer can
then resolve each step's bytes to a single token id without re-tokenizing the
assembled message.content (which could disagree on boundaries).

Usage:
  # 5-case fixture (committed in antirez/ds4 tests/test-vectors/):
  python3 build_cases_from_responses.py \
      --source test-vectors \
      --root test-vectors/ \
      --out cases/upstream-ds4-5cases-bytes.jsonl

  # 100-case capture (produced by collect_official.py):
  python3 build_cases_from_responses.py \
      --source api-capture \
      --root collected/ \
      --out cases/upstream-ds4-100cases-bytes.jsonl
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def from_test_vectors(root: Path) -> list[dict]:
    manifest = json.load(open(root / "manifest.json"))
    out = []
    for entry in manifest["prompts"]:
        official = json.load(open(root / entry["official_file"]))
        prompt = (root / entry["prompt_file"]).read_text()
        # Each step: {"step": N, "token": {"text": "...", "bytes": [...]}, ...}
        target_bytes = [step["token"]["bytes"] for step in official["steps"]]
        out.append({
            "id": entry["id"],
            "prompt": prompt,
            "target_token_bytes": target_bytes,
        })
    return out


def from_api_capture(root: Path) -> list[dict]:
    """Read responses/case_NNN.json + prompts/case_NNN.txt from a directory
    populated by `collect_official.py`."""
    out = []
    responses_dir = root / "responses"
    prompts_dir = root / "prompts"
    case_paths = sorted(responses_dir.glob("case_*.json"))
    if not case_paths:
        raise SystemExit(
            f"no responses/case_*.json under {root} — did you run collect_official.py?")
    for resp_path in case_paths:
        case_id = resp_path.stem  # e.g. "case_042"
        prompt_path = prompts_dir / f"{case_id}.txt"
        if not prompt_path.exists():
            print(f"warn: {case_id}: prompts/{case_id}.txt missing, skipping")
            continue
        prompt = prompt_path.read_text()
        response = json.load(open(resp_path))
        # OpenAI-compatible: choices[0].logprobs.content[] with .bytes per step.
        choice = response["choices"][0]
        logprobs = choice.get("logprobs", {}) or {}
        content_steps = logprobs.get("content", []) or []
        if not content_steps:
            print(f"warn: {case_id}: no logprobs.content steps, skipping")
            continue
        target_bytes = [step["bytes"] for step in content_steps]
        out.append({
            "id": case_id,
            "prompt": prompt,
            "target_token_bytes": target_bytes,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["test-vectors", "api-capture"], required=True)
    ap.add_argument("--root", type=Path, required=True,
                    help="test-vectors dir (with manifest.json + official/) or "
                         "api-capture dir (with responses/ + prompts/)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.source == "test-vectors":
        cases = from_test_vectors(args.root)
    else:
        cases = from_api_capture(args.root)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        for c in cases:
            fp.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
