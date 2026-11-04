#!/bin/sh
set -eu

BIN=${BIN:-./grouped_gemm}
mode=${1:-routine}
case "$mode" in
    routine|stress) ;;
    *) echo "Usage: $0 [routine|stress]" >&2; exit 2 ;;
esac

run_case() {
    echo "=== impl=${impl} seed=${seed} $* ==="
    "$BIN" "--impl=${impl}" "--seed=${seed}" --verify --iters=1 --warmup=0 "$@"
}

for impl in ${IMPLS:-baseline v1 v2 v3 v4 v5}; do
    for seed in 2026 42; do
        for case_name in toy tiny boundary small balanced skewed wide many-tiny empty; do
            run_case "--case=${case_name}"
        done
        run_case --case=custom --m-list=0,1,31,2,7 --k=17 --n=19
        output=$(run_case --case=custom --groups=16 --k=3 --n=5 \
            --max-m=4096 --pattern=uniform)
        printf '%s\n' "$output"
        printf '%s\n' "$output" | grep -F 'M_range=[4096,4096]' >/dev/null
    done
    if [ "$mode" = stress ]; then
        seed=2026
        for case_name in medium large large-skewed; do
            run_case "--case=${case_name}"
        done
        run_case --case=custom --m-list=0,1,4095,4096 --k=4096 --n=7
        run_case --case=custom --m-list=0,1,4095,4096 --k=7 --n=4096
    fi
done
