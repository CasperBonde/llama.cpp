#!/bin/bash
# micro-batch (ub) sweep for the qwen3next 80B target model: prefill streaming vs baseline
M=models/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
BIN=./build/bin/llama-bench
# note: -ngl 1 only. at -ngl 99 the two 1022 MiB slots do not fit next to the CUDA0
# weights (1446 MiB) and the compute buffer (516 MiB) on a 3716 MiB card.
# prefill streaming falls back to the standard path in that case.
COMMON="-ngl 1 -ncmoe 99 -lm none -n 0 -r 1"

for B in 512 1024 2048; do
    echo "## pp$B (b=$B)"
    echo ""
    $BIN -m $M $COMMON -p $B -b $B -ub 64,128,256,512,1024 -ps 0,1 -o md
    echo ""
done
