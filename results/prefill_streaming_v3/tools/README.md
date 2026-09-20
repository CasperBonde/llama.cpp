# Tools for this investigation

| file | what it is |
|---|---|
| `imma_bench.cu` | the TU117 tensor-pipe microbenchmark. Measures the exact `mma.sync.aligned.m8n8k16...s8.s8` that `mma.cuh` emits on Turing, against `__dp4a` and FFMA, with NVML power. Runs in seconds, no model load. |
| `imma_bench_results.txt` | its output on the Quadro T1000. FFMA reaches 95.9% of the published FP32 peak, which validates the harness. |

## Build and run

```sh
nvcc -O3 -arch=sm_75 --cudart shared imma_bench.cu -o imma_bench -lnvidia-ml -lpthread
./imma_bench
```

## Why this exists

The J=128 `mul_mat_q` kernel in this build contains 1024 `IMMA.8816.S8.S8` and 0 `HMMA`, while
`ncu` reports `sm__pipe_tensor_op_hmma_cycles_active = 39.45%`. That counter counts something other
than `HMMA`, so it cannot answer whether the part has a usable tensor pipe. This measures the
instruction directly.

## Result

| path | GMAC/s | MAC/SM/cycle | power |
|---|---:|---:|---:|
| `IMMA.8816.S8.S8` | 500.9 | 24.6 | 16.0 W |
| `__dp4a` (IDP) | 5015.2 | 246.2 | 25.4 W |
| `FFMA` | 1252.5 | 61.4 | 24.1 W |
| spin, no memory traffic | - | - | 23.4 W |

`__dp4a` is **10.0x** faster than `IMMA` for the same arithmetic, and `IMMA` draws less power than a
loop that moves no bytes. See section 7 of `../NCU_MOE_KERNEL.md`.

## Two traps this file already hit, worth remembering

1. **A `__global__` function called without `<<<>>>` is a no-op, and it fails silently in a timing
   loop.** The first version did exactly that and reported clean, plausible, wrong numbers
   (13.5 W, i.e. idle). `cudaGetLastError()` immediately after the launch is what exposed it. Always
   check the launch, not just the synchronize.
2. **If every operand is loop-invariant the compiler collapses the loop.** The kernel still existed
   in SASS but was never executed for most iterations. The scaling check - print time against
   iteration count and look for the slope - is what caught it, and it is cheap enough to always
   include. A single averaged number would have hidden it completely.
