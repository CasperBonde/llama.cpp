#!/bin/bash
# micro-batch (ub) sweep: prefill streaming vs baseline
M=models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
BIN=./build/bin/llama-bench
COMMON="-ngl 99 -ncmoe 99 -lm none -n 0 -r 1"

echo "# Prefill streaming vs baseline - micro-batch (ub) sweep"
echo "# date    : $(date -Iseconds)"
echo "# model   : $M"
echo "# flags   : $COMMON -o md"
echo "# note    : baseline (pfs=0) vs streaming (pfs=1), b == p in every row"
echo ""

echo "## pp512 (b=512)"
echo ""
$BIN -m $M $COMMON -p 512  -b 512  -ub 16,32,64,128,256,512 -ps 0,1 -o md
echo ""
echo "## pp4096 (b=4096)"
echo ""
$BIN -m $M $COMMON -p 4096 -b 4096 -ub 16,32,64,128,256,512 -ps 0,1 -o md
