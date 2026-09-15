#!/usr/bin/env bash
# Is the async slot fill actually overlapping the compute?
#
# Compare the default async fill against LLAMA_PREFILL_STREAMING_SYNC=1 (blocking fill,
# so the timed region covers real completion) at the SAME single ubatch.
#
# If async == blocking, the copy is not overlapped at all: the async path is enqueueing
# on a stream that the compute then waits for, so the copy cost is fully exposed.
#
# Single-ubatch sizes only (b == ub == p) so there is exactly one ubatch and the
# per-ubatch time is the whole prefill.
set -u

M=models/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
OUT=results/prefill_streaming_v3/async_vs_blocking.txt

: > "$OUT"
{
    echo "# async fill vs blocking fill, single ubatch (b == ub == p)"
    echo "# model: $M"
    echo "# date: $(date -Iseconds)"
    echo "# build: $(git rev-parse --short HEAD)"
    echo "#"
    echo "# fill mode  p    t/s      seconds per prefill"
} >> "$OUT"

for p in 512 1024; do
    for mode in async blocking; do
        if [ "$mode" = "blocking" ]; then
            export LLAMA_PREFILL_STREAMING_SYNC=1
        else
            unset LLAMA_PREFILL_STREAMING_SYNC
        fi

        ts=$(LLAMA_PREFILL_STREAMING_DEBUG=1 timeout 1800 \
            ./build/bin/llama-bench -v -m "$M" -ngl 1 -ncmoe 99 -lm none \
                -p "$p" -b "$p" -ub "$p" -ps 1 -r 1 2>&1 |
            grep -oE "pp[0-9]+ \| *[0-9.]+" | tail -1 | grep -oE "[0-9.]+$")

        secs=$(awk -v p="$p" -v t="$ts" 'BEGIN{ if (t>0) printf "%.2f", p/t; else print "n/a" }')
        printf "%-10s %-4s %-8s %s\n" "$mode" "$p" "$ts" "$secs" >> "$OUT"
        echo "  done $mode p=$p -> $ts t/s"
    done
done

echo ""
echo "===== RESULT ====="
cat "$OUT"
