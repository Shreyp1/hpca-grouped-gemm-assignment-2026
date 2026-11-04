#!/bin/sh
set -eu

BIN=${BIN:-./grouped_gemm}
for impl in baseline v1 v2 v3 v4 v5; do
    for case_name in toy boundary small skewed empty; do
        echo "=== impl=${impl} case=${case_name} ==="
        "$BIN" "--impl=${impl}" "--case=${case_name}" --verify --iters=3 --warmup=1
    done
done
