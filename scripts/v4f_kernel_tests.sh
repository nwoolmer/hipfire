#!/usr/bin/env bash
# Run all V4F GPU validation tests in order. Acquires the gpu lock
# once for the whole batch; emits per-test status.
#
# Usage:
#   ./scripts/v4f_kernel_tests.sh
#
# Each test exits non-zero on validation failure; this script
# accumulates failures and exits non-zero if any failed.

set -u

cd "$(dirname "$0")/.."

# Build all V4F examples once (cargo will no-op recompile).
cargo build --release -p hipfire-arch-deepseek4 \
    --example test_hc_compute \
    --example test_hc_sinkhorn \
    --example test_hc_mix \
    --example test_rope_tail \
    --example test_indexer_score \
    --example test_indexer_top_k \
    --example test_indexer_gather \
    --example load_v4f_check \
    2>&1 | tail -2

# shellcheck disable=SC1091
source "$(dirname "$0")/gpu-lock.sh"
gpu_acquire "v4f-kernel-tests"
trap 'gpu_release' EXIT

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/target/release/examples"

declare -i ok=0 fail=0
for test in test_hc_compute test_hc_sinkhorn test_hc_mix \
            test_rope_tail test_indexer_score test_indexer_top_k \
            test_indexer_gather; do
    printf "  %-25s ... " "$test"
    out=$("${BIN}/${test}" 2>&1)
    if echo "$out" | grep -q "^OK:"; then
        echo "ok"
        ok=$((ok + 1))
    else
        echo "FAIL"
        echo "$out" | tail -10 | sed 's/^/    /'
        fail=$((fail + 1))
    fi
done

# load_v4f_check requires the V4F HFQ file at the conventional path.
if [ -f "${HOME}/.hipfire/models/v4f.mq2lloyd-gptq-all" ]; then
    printf "  %-25s ... " "load_v4f_check"
    out=$("${BIN}/load_v4f_check" 2>&1)
    if echo "$out" | grep -q "load_weights walk OK"; then
        echo "ok"
        ok=$((ok + 1))
    else
        echo "FAIL"
        echo "$out" | tail -10 | sed 's/^/    /'
        fail=$((fail + 1))
    fi
else
    printf "  %-25s ... skipped (V4F HFQ not at ~/.hipfire/models/v4f.mq2lloyd-gptq-all)\n" "load_v4f_check"
fi

echo
echo "V4F kernel tests: ${ok} ok, ${fail} fail"
exit ${fail}
