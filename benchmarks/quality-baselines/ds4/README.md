# antirez/ds4 quality baseline — hipfire mirror

Mirrors the quality-testing fixtures from https://github.com/antirez/ds4 so
hipfire can produce NLL numbers comparable to antirez's published Q4 reference.

## What's here

* `prompts.jsonl` (100 prompts) — antirez's quality-testing prompt set
  (`gguf-tools/quality-testing/prompts.jsonl`). English + Italian; algorithms,
  completion, translation, short-answer.
* `collect_official.py`, `compare_scores.py`, `score_official.c` — antirez's
  reference pipeline, for documentation. We do NOT run these directly; we
  reimplement them in hipfire as `crates/hipfire-arch-deepseek4/examples/
  v4f_quality_{collect,score}.rs`.
* `test-vectors/` — antirez's 5-prompt set with **upstream V4-Flash API
  continuations COMMITTED to the repo** (`tests/test-vectors/` in his repo).
  Captured from `https://api.deepseek.com/chat/completions`, model
  `deepseek-v4-flash`, `temperature=0`, `thinking=disabled`, `top_logprobs=20`,
  `max_tokens=4` for short prompts (1-4 tokens per case).
* `cases/upstream-ds4-5cases.jsonl` — derived from `test-vectors/` by
  extracting the API's `message.content` per prompt into our case JSONL
  format. This is what `v4f_quality_score` runs against.

## How to run

```sh
# 1. Build the scorer
cargo build --release --example v4f_quality_score

# 2. Score a model against the 5-case upstream reference
./target/release/examples/v4f_quality_score \
    /data/hipfire-models/v4f.mq2lloyd-f16compress.hfq \
    benchmarks/quality-baselines/ds4/cases/upstream-ds4-5cases.jsonl \
    --ctx 8192 --moe 1 \
    --out benchmarks/quality-baselines/ds4/results/<modelname>.tsv

# Output TSV columns: id, prompt_tokens, target_tokens, nll, avg_nll,
#                     first_match, greedy_lcp
```

## Reference numbers from antirez

The published reference (from `gguf-tools/imatrix/README.md` in antirez/ds4):

```
old Q4 baseline:    avg_nll = 0.177358    (cases=100, ctx=4096)
Q4 imatrix:         avg_nll = 0.173895   (-1.95% relative)
```

These numbers are from the **100-prompt** quality-testing battery, scored
against DeepSeek API continuations captured at `max_tokens=24`.

**Antirez does NOT publish a Q2 reference.** Our Q2 numbers establish the
first baseline.

The 5-case `test-vectors/` set is captured at `max_tokens=4, top_logprobs=20`
for the C `--logprob-vectors` regression test — it's a different (smaller)
fixture. Our 5-case NLL numbers are NOT directly comparable to antirez's
100-case 0.173895, but the comparison across our own quants on the same
5-case set IS valid.

## To run the full 100-prompt comparison

Requires `DEEPSEEK_API_KEY`. Run antirez's `collect_official.py` against
`prompts.jsonl` to capture 100 continuations, then score each of our HFQ
files against the captured manifest.

```sh
DEEPSEEK_API_KEY=... python3 collect_official.py \
    --prompts prompts.jsonl \
    --out-dir collected/ \
    --max-tokens 24
# Then convert the per-prompt JSON outputs to our JSONL format and run
# v4f_quality_score as above.
```

We haven't run this yet — pending API key access.
