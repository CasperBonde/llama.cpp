#!/bin/bash
# Test whether demoting non-expert layer weights to host helps.
#
# Options (sizes from the GGUF tensor inventory):
#   1. GDN/linear weights, all in blk.N.ssm*        -> ~727 MiB total
#   2. shared expert, all in blk.N.ffn_*shexp*      -> ~216 MiB (+32 MiB routers)
#   3a. output.weight, one global tensor            -> ~160 MiB, NOT per-layer
#
# Usage: ./run_ot_tests.sh [quick|full] [b]
#   b is the batch/ubatch size, default 512 (kept equal, see note below)
#   quick: prefill only
#   full:  adds prefill-streaming variants
#
# NOTE: llama-bench CROSS-PRODUCTS -b and -ub, so they are always passed equal here.
#       -b matters independently because n_batch clamps the sequence length that the
#       FA mask is sized from (llama-context.cpp cparams.n_batch).
#
# Run ONE llama-bench at a time. Never run this in parallel with another benchmark.

set -u
cd "$(dirname "$0")"

MODE="${1:-quick}"
B="${2:-512}"
M=models/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
OUT=results/prefill_streaming/ot_tests_${MODE}_b${B}.txt

# every run uses the same -p/-n, otherwise the blocks are not comparable
BASE="-m $M -ngl 99 -ncmoe 48 -lm none -b $B -ub $B -p 2048 -n 0 -r 1 -ctk q4_0 -ctv q4_0 -fa 1"

OPT_SSM="blk\\.[0-9]+\\.ssm.*=CPU"
OPT_SHEXP="blk\\.[0-9]+\\.ffn_.*shexp.*=CPU"
OPT_OUTPUT="output\\.weight=CPU"

: > "$OUT"

run() {
    local label="$1"; shift
    {
        echo "########## ${label} ##########"
        echo "# args: $*"
    } >> "$OUT"
    timeout 1800 ./build/bin/llama-bench "$@" -v 2>&1 | grep -E \
        "CUDA0 model buffer size|CUDA_Host model buffer size|CUDA0 compute buffer size|CUDA_Host compute buffer size|CUDA0 KV buffer size|CUDA0 RS buffer size|pp2048|error" \
        >> "$OUT"
    echo "--- ${label} ---"
    tail -8 "$OUT"
}

run "BASELINE"            $BASE
run "OPT1_ssm"            $BASE -ot "$OPT_SSM"
run "OPT2_shexp"          $BASE -ot "$OPT_SHEXP"
run "OPT3_output"         $BASE -ot "$OPT_OUTPUT"
run "OPT1+2_ssm_shexp"    $BASE -ot "$OPT_SSM" -ot "$OPT_SHEXP"

if [ "$MODE" = "full" ]; then
    run "OPT1+2_streaming"    $BASE -ot "$OPT_SSM" -ot "$OPT_SHEXP" -ps 1
    run "BASELINE_streaming"  $BASE -ps 1
fi

echo "results: $OUT"
