#!/usr/bin/env bash
# Compare KV-quantization configurations on long-context quality.
#
# For each (prompt-length × kv-config) cell, runs a greedy decode with
# the same prompt and grades:
#   1. Needle recall — did the response contain the expected substring?
#   2. Output divergence — does the generated text match the FP32 baseline?
#   3. Coherence — no obvious gibberish / loops in the first 64 tokens?
#
# Usage:
#   ./longctx_compare.sh [4k|32k|60k|all]
#
# Default `all` runs all three lengths. Each cell takes ~5-15 min on
# gfx1151 due to prefill latency.

set -u
cd "$(dirname "$0")/.."

LENGTH="${1:-all}"
MODEL="${HIPFIRE_BONSAI_HFQ:-$HOME/.hipfire/models/bonsai/bonsai-8b.hfq}"
NEEDLE="${LONGCTX_NEEDLE:-DELTA-7-RAVEN-WINDFALL}"
MAXGEN=128
EXE="./target/release/examples/infer_qwen3"
OUT_DIR="${HIPFIRE_LONGCTX_OUT:-/tmp/longctx-$(date +%Y%m%d-%H%M%S)}"

mkdir -p "$OUT_DIR"
echo "output directory: $OUT_DIR"
echo "needle:           $NEEDLE"
echo "model:            $MODEL"
echo

if [ ! -x "$EXE" ]; then
    echo "building infer_qwen3..."
    cargo build --release -p hipfire-runtime --example infer_qwen3 >&2 || exit 2
fi

declare -A LENGTHS=(
    [4k]=4096
    [14k]=14000
    [32k]=32768
    [60k]=60000
)

declare -A KV_FLAGS=(
    [fp32]=""
    [q8]="--q8kv"
)

run_one() {
    local len_label="$1"
    local len_tokens="$2"
    local kv_label="$3"
    local kv_flag="$4"
    local prompt_file="$OUT_DIR/prompt-$len_label.txt"
    local out_file="$OUT_DIR/$len_label-$kv_label.out"

    if [ ! -f "$prompt_file" ]; then
        echo "  generating prompt at $len_tokens tokens..."
        python3 scripts/longctx_gen.py \
            --tokens "$len_tokens" --depth 0.5 \
            --needle "$NEEDLE" \
            --out "$prompt_file" >/dev/null
    fi

    echo "  → $len_label / $kv_label (kv_flag='$kv_flag')"
    local start=$(date +%s)
    if [ -z "$kv_flag" ]; then
        "$EXE" "$MODEL" --temp 0 --maxgen "$MAXGEN" --prompt-file "$prompt_file" \
            > "$out_file" 2>&1
    else
        "$EXE" "$MODEL" "$kv_flag" --temp 0 --maxgen "$MAXGEN" --prompt-file "$prompt_file" \
            > "$out_file" 2>&1
    fi
    local end=$(date +%s)
    local elapsed=$((end - start))

    # Grade: needle recall.
    local recalled="MISS"
    if grep -qF "$NEEDLE" "$out_file"; then
        recalled="HIT"
    fi

    # Grade: prefill + decode tok/s.
    local prefill_tps=$(grep -oE 'Prompt: [0-9]+ms \([0-9]+ tokens, [0-9]+ tok/s\)' "$out_file" | grep -oE '[0-9]+ tok/s' | head -1)
    local decode_tps=$(grep -oE 'Done: [0-9]+ tokens in [0-9]+ms \([0-9.]+ tok/s\)' "$out_file" | grep -oE '\([0-9.]+ tok/s\)' | tr -d '()')

    # Coherence smoke: count distinct n-grams in last 200 chars.
    local last_200=$(grep -A 1000 'Generating' "$out_file" | tail -c 800 | head -c 800)
    local repeat_density=0
    # Heuristic: count repeated 3-grams. Simple shell version: count words.
    local word_count=$(echo "$last_200" | wc -w)
    local unique_words=$(echo "$last_200" | tr -s '[:space:]' '\n' | sort -u | wc -l)
    if [ "$word_count" -gt 0 ]; then
        repeat_density=$(awk "BEGIN { printf \"%.2f\", 1 - $unique_words/$word_count }")
    fi

    printf "    needle: %-4s  prefill: %-10s  decode: %-10s  rep_density: %s  elapsed: %ds\n" \
        "$recalled" "$prefill_tps" "$decode_tps" "$repeat_density" "$elapsed"
}

run_length() {
    local len_label="$1"
    local len_tokens="${LENGTHS[$len_label]}"
    echo "── length: $len_label (~$len_tokens tokens) ──"
    for kv_label in fp32 q8; do
        run_one "$len_label" "$len_tokens" "$kv_label" "${KV_FLAGS[$kv_label]}"
    done
    echo
}

if [ "$LENGTH" = "all" ]; then
    for label in 4k 32k 60k; do
        run_length "$label"
    done
else
    run_length "$LENGTH"
fi

echo "── divergence summary ──"
for len_label in 4k 32k 60k; do
    if [ -f "$OUT_DIR/$len_label-fp32.out" ] && [ -f "$OUT_DIR/$len_label-q8.out" ]; then
        # Extract just the generated text (between "Generating" and "Done") and diff.
        for label in fp32 q8; do
            sed -n '/Generating/,/=== Done/p' "$OUT_DIR/$len_label-$label.out" \
                | sed '/=== Done/d;/^Generating/d' \
                > "$OUT_DIR/$len_label-$label.gen"
        done
        diff_lines=$(diff -q "$OUT_DIR/$len_label-fp32.gen" "$OUT_DIR/$len_label-q8.gen" 2>/dev/null | wc -l)
        if [ "$diff_lines" -eq 0 ]; then
            echo "  $len_label: outputs IDENTICAL"
        else
            echo "  $len_label: outputs DIFFER  (see $OUT_DIR/$len_label-{fp32,q8}.gen)"
        fi
    fi
done

echo
echo "full outputs in: $OUT_DIR"
