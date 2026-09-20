# Finding: the per-split synchronize is the cause of the pp decay

Date: measured on build 8a1540095 with a temporary, gated experiment.

## What was tested

`ggml_backend_synchronize(split_backend)` at `ggml/src/ggml-backend.cpp:1833` runs once per
split, i.e. once per layer boundary, on the eval-callback path. The hypothesis was that this
call drains the compute stream, leaving nothing in flight for the slot copy to overlap with.

The experiment gated the call behind `LLAMA_SCHED_NO_SPLIT_SYNC`. It is NOT a proposed fix:
the change is kept only as a measurement, and it was reverted after the result below.

## Result: fast, but incorrect

Run: `-ngl 1 -ncmoe 99 -lm none -b == -ub == 1024 -ps 1 -r 1`.

| p | split sync (normal) | no split sync | streaming off |
|---:|---:|---:|---:|
| 1024 | 87.03 | **115.59** | 53.71 |
| 2048 | 50.42 | **111.49** | 52.98 |
| 3072 | 50.38 | **107.06** | - |

The pp decay disappears completely. Per-ubatch compute also changes:

| state | per-ubatch compute |
|---|---|
| normal | bimodal, 11.8 s and 20.0 s |
| no split sync | tight, 8.8 s to 9.8 s, every ubatch |

So the barrier accounts for both the pp decay and the 11.8/20.0 s bimodality.

But the model output is wrong with the barrier removed:

```
normal:     The capital of France is Paris.
no sync:    / far of UBL中心主任age.sinuj足Y
```

The second case also raises a peg-native format error. The split synchronize is therefore a
correctness barrier, not a scheduling artifact. Without it the layer-boundary callback races
the compute stream: the callback is told layer `il` is done while it is still running, and it
refills the slot that the running layer may still read.

`ggml_backend_cuda_synchronize` is `cudaStreamSynchronize(cuda_ctx->stream())`, i.e.
stream-scoped, not device-wide. So the barrier does not stall the copy stream directly. It
empties the compute stream at every boundary, so nothing is in flight for the copy to overlap
with, and the copy runs mostly alone. That is consistent with the measured 11.5% overlap
(0.446 s hidden of 3.893 s).

## Consequence

The 48 `cudaStreamSynchronize` calls are the price of driving the refill from a
layer-boundary callback. Removing the barrier is not an option. A real fix must give the
callback a finer-grained completion signal, for example a per-layer CUDA event that the copy
stream waits on, instead of a full stream drain. That touches the shared scheduler contract
and needs sign-off before any code is written.

## Reproduce

```
LLAMA_SCHED_NO_SPLIT_SYNC=1 ./build/bin/llama-bench -m <model> -ngl 1 -ncmoe 99 -lm none \
    -p 1024,2048,3072 -b 1024 -ub 1024 -ps 1 -r 1
```

`llama-bench` installs a null log callback, so per-ubatch `LLAMA_PREFILL_STREAMING_DEBUG`
lines only appear with `-v`.
