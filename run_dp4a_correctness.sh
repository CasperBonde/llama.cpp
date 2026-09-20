#!/usr/bin/env bash
# Does the Turing dp4a MMQ build compute the same thing as the tensor core build?
#
# The dp4a path lays its shared memory tiles out differently and uses a different
# vec-dot, so a layout bug would show up as garbage or NaNs, not as a crash. Greedy
# decoding at temp 0 makes any numeric difference visible in the text.
#
# Three builds are compared on identical input:
#   cpu    -ngl 0, the reference, no MMQ at all
#   base   build/           ampere tile table + IMMA
#   dp4a   build-turingdp4a/ pascal-dp4a tile table + IDP
set -u

M=models/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
OUT=results/prefill_streaming_v3/dp4a_correctness.txt
PROMPT="The capital of France is"

run_one() {
    local label="$1" bin="$2" ub="$3"
    local log text
    log=$(mktemp)

    # llama-cli writes the transcript to the tty, so run it under a pty and strip ANSI.
    script -qec "timeout 2400 $bin -m $M -ngl 1 -ncmoe 99 -lm none \
        -b $ub -ub $ub -c 512 --temp 0 --seed 42 -n 32 -st --no-display-prompt \
        -p '$PROMPT'" "$log" < /dev/null > /dev/null 2>&1

    text=$(sed 's/\x1b\[[0-9;]*[a-zA-Z]//g; s/\r//g' "$log" | grep -F "The capital of France" | tail -1)
    rm -f "$log"

    [ -z "$text" ] && text="*** NO OUTPUT (BROKEN) ***"
    printf "%s|%s|%s\n" "$label" "$ub" "$text"
}

{
    echo "# dp4a vs tensor core vs CPU, greedy temp 0, ub 512 and 1024"
    echo "# model: $M"
    echo "# date: $(date -Iseconds)"
    echo "# build: $(git rev-parse --short HEAD)"
} > "$OUT"

for ub in 512 1024; do
    for spec in "cpu:./build/bin/llama-cli" "base:./build/bin/llama-cli" "dp4a:./build-turingdp4a/bin/llama-cli"; do
        label=${spec%%:*}
        bin=${spec#*:}
        if [ "$label" = "cpu" ]; then
            # a CPU reference ignores the GPU flags, but they are harmless
            bin="./build/bin/llama-cli"
        fi
        printf "  ub=%s %-5s " "$ub" "$label"
        run_one "$label" "$bin" "$ub" | tee -a "$OUT"
    done
done

echo ""
echo "=== verdict ==="
awk -F'|' '!/^#/ && NF>=3 {gsub(/^ +| +$/, "", $3); if ($3 != "") { seen[$3]=1; n++ } }
           END { if (length(seen) == 1 && n > 0) print "PASS: all outputs identical"; else print "DIFFER: " length(seen) " distinct outputs" }' "$OUT"
