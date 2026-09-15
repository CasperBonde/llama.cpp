#!/usr/bin/env bash
# Measure the blocking (true) slot fill cost as a function of ubatch size.
#
# Rationale: the fill uses ggml_backend_tensor_set_async, which dispatches to
# ggml_backend_cuda_set_tensor_async -> cudaMemcpyAsync on the compute context's own
# stream. The fill is triggered from the layer-boundary eval callback, i.e. from inside
# ggml_backend_graph_compute_async, so the copy is enqueued on the very stream the
# compute uses and can never overlap it.
#
# If the copy is scalar-bound (it grows when the compute is scaled up), the only way to
# hide it is to replace the per-tensor cudaMemcpyAsync loop with a real parallel copy.
#
# LLAMA_PREFILL_STREAMING_SYNC=1 forces the blocking fill, so fill_us is real completion
# time instead of enqueue time.
set -u

M=models/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
OUT=results/prefill_streaming_v3/copy_cost_vs_ubatch.txt

: > "$OUT"
echo "# blocking fill cost vs ubatch size (LLAMA_PREFILL_STREAMING_SYNC=1)" >> "$OUT"
echo "# model: $M" >> "$OUT"
echo "# date: $(date -Iseconds)" >> "$OUT"
echo "# build: $(git rev-parse --short HEAD)" >> "$OUT"

for p in 1024 2048; do
    for ub in 1024 512; do
        {
            echo ""
            echo "##### pp=$p ub=$ub #####"
        } >> "$OUT"

        LLAMA_PREFILL_STREAMING_SYNC=1 LLAMA_PREFILL_STREAMING_DEBUG=1 \
            timeout 1800 ./build/bin/llama-bench -v -m "$M" \
                -ngl 1 -ncmoe 99 -lm none \
                -p "$p" -b "$ub" -ub "$ub" -ps 1 -r 1 \
                2>&1 | grep -E "compute [0-9]|pp[0-9]{4} \|" >> "$OUT"

        echo "  done pp=$p ub=$ub"
    done
done

echo ""
echo "===== RESULT ====="
cat "$OUT"
