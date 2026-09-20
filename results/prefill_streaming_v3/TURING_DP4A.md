# Turing without tensor cores: use the dp4a path (1.77x on pp1024)

## Summary

The Quadro T1000 is TU117: compute capability 7.5 with **no tensor cores**. It was running the
tensor core MMQ kernel anyway, which on this part costs **10x the equivalent dp4a throughput** and
draws less power than an idle loop. Compiling the Turing family onto the dp4a path instead:

| test | baseline | dp4a build | change |
|---|---:|---:|---:|
| pp1024 | 84.49 ± 0.08 t/s | **149.92 ± 0.21 t/s** | **1.77x** |
| pp512 | 63.39 ± 0.20 t/s | **96.79 ± 0.32 t/s** | **1.53x** |
| GPU power, busy samples | 18.2 W mean, 22.3 W max | 22.3 W mean, 30.2 W max | +22.5% mean |

Correctness is unaffected: greedy output is byte-identical to the tensor core build and to the CPU
at both ub 512 and ub 1024 (see below).

## Why the old path was wrong, twice over

Two independent mistakes, both because cc 75 covers two different dies. TU117 has no tensor cores;
TU106 and TU116 do. Nothing about `compute capability 7.5` distinguishes them.

1. **The tile table.** `ggml_cuda_mmq_get_config` routed everything `>= VOLTA (700)` to the
   **ampere** table, so Turing got Ampere's tiles. That table has no `I` below 128, on a part with
   14 SMs, 48 KiB of shared memory per block and a 255 register limit.
2. **The data layout.** `use_mma_data_layout()` returned true, which `#else`-es the dp4a code out of
   the kernel entirely. The J=128 kernel contained **1024 `IMMA` and 0 `IDP`** - no dp4a instruction
   was present to run.

There was already a `turing_devices_without_mma` list in `ggml-cuda.cu`, but it only ever *logged a
warning*, and the Quadro T1000 was not in it (it listed MX450, MX550 and GTX 16xx).

## The measurement that identified the cause

`tools/imma_bench.cu` microbenchmarks the exact instruction `mma.cuh` emits on Turing, with operands
in registers so the only variable is the instruction:

| path | GMAC/s | MAC/SM/cycle | power |
|---|---:|---:|---:|
| `IMMA.8816.S8.S8` | 500.9 | 24.6 | **16.0 W** |
| `__dp4a` (IDP) | 5015.2 | 246.2 | 25.4 W |
| `FFMA` | 1252.5 | 61.4 | 24.1 W |
| spin, no memory traffic | - | - | 23.4 W |

`FFMA` reaches 95.9% of the T1000's published FP32 peak (1305 GMAC/s), which validates the harness.
`dp4a` is **10.0x** faster than `IMMA` at the same arithmetic, and `IMMA` draws **less power than a
loop that moves no bytes** - the signature of an emulated instruction on a part without the unit.

## What was changed

The choice cannot be made at runtime. `mmq-load-tiles.cuh` and `mmq-vec-dot.cuh` pick the shared
memory tile format with the preprocessor, so a runtime table switch would desynchronise the host
from the device and corrupt memory. It has to be a build choice.

- `ggml/CMakeLists.txt`: new option `GGML_CUDA_TURING_NO_MMA`, off by default.
- `common.cuh`: new `MMQ_MMA_TILE` macro, set when `>= TURING` and **not**
  (`GGML_CUDA_TURING_NO_MMA` and `< AMPERE`). Deliberately separate from `TURING_MMA_AVAILABLE`,
  which flash attention needs - see the mistakes section.
- `mmq.cuh`: the tile loaders, vec-dot, and both `use_mma_data_layout()` overloads key off
  `MMQ_MMA_TILE`; `ggml_cuda_mmq_get_config` sends Turing to the pascal-dp4a table.
- `mmq.cu`: `should_use_mmq` no longer rejects MMQ above batch 64 - that test assumes an fp16 MMA
  fallback exists, and on this part it does not.
- `ggml-cuda.cu`: per-device `tensor_cores` flag, Quadro T1000 added to the name list, and the
  warning text corrected (it told the user to rebuild with an outright broken cmake flag, a
  misspelled `DGGML_CUDA_FORCE_MMQ`, and an arch list `61-virtual;80-virtual` that cannot serve
  cc 75 at all).

Build used for the measurements:

```sh
cmake -S . -B build-turingdp4a -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_TURING_NO_MMA=ON -DGGML_NATIVE=ON
```

Result: `CMAKE_CUDA_ARCHITECTURES=75-real`, 0 compile errors.

## Instruction level proof

SASS census of `mul_mat_q<Q4_K, J=64, fallback=false>` in the new build:

| instruction | baseline build | dp4a build |
|---|---:|---:|
| `IDP` (dp4a) | 0 | **512** |
| `IMMA` | 1024 | **0** |
| `FFMA` | 2048 | 160 |

## Correctness

`run_dp4a_correctness.sh`, greedy `--temp 0`, three configurations on identical input:

```
  ub=512  cpu     The capital of France is Paris.
  ub=512  base    The capital of France is Paris.
  ub=512  dp4a    The capital of France is Paris.
  ub=1024 cpu     The capital of France is Paris.
  ub=1024 base    The capital of France is Paris.
  ub=1024 dp4a    The capital of France is Paris.
=== verdict ===
PASS: all outputs identical
```

Two repeats of the benchmark agree to within 0.4% (149.57 and 149.92 t/s), so this is not noise.

## What is established and what is not

**Established:** the dp4a path selects, the kernel contains `IDP` instead of `IMMA`, output is
identical, and throughput rises 1.53x at pp512 and 1.77x at pp1024 while power rises 22.5%.

**Not established:** how much of the gain is the change of *instruction* rather than the change of
*tile shape*. The dp4a table also uses `I=64, J<=64` and `stream_k=false`, where the ampere table
used `I=128, J<=128` and `stream_k=true`. All three differ at once. Isolating them needs further
builds. The microbenchmark shows the instruction is 10x slower in isolation, but that is not the
same as attributing the end-to-end gain to it.

**Also not measured:** whether `-ngl 99` benefits as much. It was slower than `-ngl 1` on the
baseline (74.95 vs 81.45 t/s), which the IMMA explanation would predict, but that has not been
re-run.

**Relevance to the power question.** The standing question was why power only reaches ~25% of the
50 W cap. On the baseline, GPU power during this workload sits at 18.2 W. With the workload now
almost twice as fast it still only reaches 22.3 W mean and 30.2 W peak. So the dp4a change explains
why the part was *slow*, and it does raise power per unit time as predicted, but it does not by
itself explain the ceiling: even at 1.77x throughput the part is not near its limit. The remaining
gap is a separate question.

## Upstream status

Prototype only. The `turing_devices_without_mma` list in `ggml-cuda.cu` shows the maintainers
already intended to handle these parts; the list is incomplete and only warns rather than acting.
Any upstream change needs an issue and maintainer agreement first, per `AGENTS.md`. The build option
added here is a local experiment, not a proposed patch.
