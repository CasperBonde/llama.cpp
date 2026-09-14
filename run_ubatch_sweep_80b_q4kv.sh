#!/bin/bash
# micro-batch (ub) sweep for the qwen3next 80B target model with Q4_0 KV cache
# (matches start-qwen.sh: --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn on)
M=models/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
BIN=./build/bin/llama-bench
# -ngl 1 only: at -ngl 99 the two 1022 MiB slots do not fit next to the CUDA0 weights
# (1446 MiB) and the compute buffer (516 MiB) on a 3716 MiB card, so prefill streaming
# silently falls back to the standard path.
COMMON="-ngl 1 -ncmoe 99 -lm none -n 0 -r 1 -ctk q4_0 -ctv q4_0 -fa on"

for B in 512 1024 2048; do
    echo "## pp$B (b=$B) - Q4_0 KV, flash-attn on"
    echo ""
    $BIN -m $M $COMMON -p $B -b $B -ub 64,128,256,512,1024 -ps 0,1 -o md
    echo ""
done
