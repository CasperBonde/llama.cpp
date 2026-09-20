# Turing without tensor cores: use the dp4a path (1.77x on pp1024)

## Summary

The Quadro T1000 is TU117: compute capability 7.5 with **no tensor cores**. It was running the
tensor core MMQ kernel anyway, which on this part costs **10x the equivalent dp4a throughput** and
draws less power than an idle loop. Compiling the Turing family onto the dp4a path instead gives
1.53x at pp512 and 1.77x at pp1024, with identical output.

| test | baseline | dp4a build | change |
|---|---:|---:|---:|
| pp1024 | 84.49 ± 0.08 t/s | **149.92 ± 0.21 t/s** | **1.77x** |
| pp512 | 63.39 ± 0.20 t/s | **96.79 ± 0.32 t/s** | **1.53x** |
| tg128 | 9.73 ± 0.33 t/s | 9.31 ± 0.38 t/s | -4.3% |
| tg256 | 9.52 ± 0.08 t/s | 9.52 ± 0.10 t/s | 0% |
| GPU power, busy samples | 18.2 W mean, 22.3 W max | 22.3 W mean, 30.2 W max | +22.5% mean |

Prefill improves by much less than the 10x the instruction microbenchmark showed, and decode does
not improve at all. The timing breakdown below explains both.

Correctness is unaffected: greedy output is byte-identical to the tensor core build and to the CPU
at both ub 512 and ub 1024.


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

## `-ngl 1` against `-ngl 99`, and token generation (resolved)

Earlier attempts at `-ngl 99` all failed with `cudaMalloc failed: out of memory` on ~620 MiB, on
both builds, while `nvidia-smi` showed the card free. That was **not** a VRAM shortage: the
streaming pipeline's own compute-buffer reservation for the multi-slot graph is what does not fit.
The rule is the slot count. **One slot is the configuration that fits alongside `-ngl 99`;
double buffering needs two slots and therefore `-ngl 1`.** For `-ngl 99` the copy also has to be
sequential (`LLAMA_PREFILL_STREAMING_SERIAL=1`), which is the change made to fit the slot and the
99-layer compute buffer into 4 GB. With that, it fits comfortably.

All four numbers below come from **one invocation**, one model load:

```sh
env LLAMA_PREFILL_STREAMING_SERIAL=1 ./build-turingdp4a/bin/llama-bench \
  -m models/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf \
  -ngl 1,99 -ncmoe 99 -lm none -p 1024 -n 256 -b 1024 -ub 1024 -o md \
  --cache-type-k q4_0 --cache-type-v q4_0 -t 6 -ps 1 -r 4
```

| `-ngl` | pp1024 | tg256 |
|---:|---:|---:|
| 1 (2 slots, double buffered) | **146.79 ± 0.32** | 9.28 ± 0.46 |
| 99 (1 slot, serial copy) | **143.22 ± 0.31** | **18.07 ± 0.17** |

Two results matter here.

**The prefill penalty of `-ngl 99` has almost gone.** On the baseline, `-ngl 99` was 8.7% slower
than `-ngl 1` on prefill (74.95 against 81.45 t/s), and that was attributed to the device spending
17.1% more per node when the MoE ran on the GPU. That gap is now **2.4%** (143.22 against 146.79).
The dp4a change removed most of the reason `-ngl 99` was worse for prefill, which is consistent
with the 10x instruction penalty explaining the 17.1% per-node excess.

**Token generation prefers `-ngl 99` by 1.95x** (18.07 against 9.28 t/s). This is the expected
direction - decode is weight-bandwidth bound and the more of the model resident in VRAM the
better - and it confirms the dp4a build is not merely fast at prefill.

Together these mean `-ngl 1` and `-ngl 99` are now within 2.4% on prefill while `-ngl 99` is
nearly twice as fast on decode, so **`-ngl 99` is the configuration to prefer**, the opposite of
the baseline's conclusion.

pp512 has still only been measured once, at 96.79 t/s.

## The time split: copy against compute

Measured with the serial instrument, `LLAMA_PREFILL_STREAMING_SERIAL=1`, which makes the copy
blocking so the two phases are disjoint and timed per layer. Three prefill ubatches:

| | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| copy, 46 layers | 3720.9 ms | 3715.2 ms | 3728.0 ms |
| compute, 46 layers | 2039.1 ms | 1953.1 ms | 1938.6 ms |
| copy + compute | 5760.0 ms | 5668.3 ms | 5666.5 ms |
| ubatch wall | 7095.8 ms | 6987.6 ms | 6977.2 ms |
| copy share | 64.6% | 65.5% | 65.8% |
| copy rate | 12.63 GB/s | 12.36 GB/s | 12.32 GB/s |

Copy is **3720 ms of 5760 ms = 64.6%** of the work, at 12.4 GB/s. That matches an independent
pinned H2D measurement of 12.5 GB/s, so the copy is at the PCIe Gen3 x16 ceiling and is not
improvable.

Per layer type, the compute split is very uneven:

| layer type | count | mean | range |
|---|---:|---:|---:|
| GDN (SSM) | 35 | **37.6 ms** | 37.1-39.1 ms |
| attention | 11 | **56.7 ms** | 56.3-57.1 ms |

### Comparison against the baseline numbers in CALIBRATION.md section 6.1

| | baseline | dp4a | change |
|---|---:|---:|---:|
| copy, 46 layers | 3733.6 ms | 3720.9 ms | unchanged (within noise) |
| compute, 46 layers | 6066.0 ms | 2039.1 ms | **2.98x faster** |

This is the cleanest possible attribution. The **copy time does not move** (it is the same bytes
over the same bus) while the **compute time falls 3x**. So the speedup is entirely in the
arithmetic, and the reason pp1024 only improves 1.77x rather than 3x is that the copy phase, which
was 38% of the ubatch, is now 65% of it:

| | baseline | dp4a |
|---|---:|---:|
| compute | 6066 ms (60.2%) | 2039 ms (34.2%) |
| copy | 3734 ms (37.0%) | 3721 ms (62.4%) |
| residual / host | 283 ms | 202 ms |
| total | 10083 ms | 5962 ms |

The residual is the 7095.8 - 5760.0 = 1335.8 ms of each ubatch not covered by the two timed
phases (the 2-slot overlap has its own accounting; in serial mode the phases do not cover the whole
ubatch).

**Consequence for the power question.** The copy now dominates, and CALIBRATION.md section 1f
measured a pinned H2D copy at **16.1 W** against 12.8 W idle. So a large part of the run is now
spent in the one phase that cannot draw much power. That is a plausible part of why the mean moved
only 18.2 -> 22.3 W while throughput rose 1.77x.

## The tiling, measured rather than inferred

`LLAMA_MMQ_DEBUG=1` prints the shape actually chosen per launch. At ub 1024 on the dp4a build, every
`mul_mat_q` launch reports **`I=64, J=64, nthreads=256, stream_k=0`**. The distinct shapes printed on
a full pp1024+tg256 run:

| tensor | nrows_x | ntx | nty | ntzw | tiles | smem | launches |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_K, MoE gate/up | 512 | 16 | 8 | 512 | **65536** | 19232 B | 384 |
| Q6_K, MoE down | 2048 | 16 | 32 | 512 | **262144** | 27432 B | 96 |
| Q4_K, dense | 8192 | 16 | 128 | 1 | 2048 | 19232 B | 144 |
| Q6_K, dense | 8192 | 16 | 128 | 1 | 2048 | 27432 B | 144 |

The smem values are what the kernel actually reports (`mmq_get_nbytes_shared`), not a hand
derivation. Q4_K is 19232 B and Q6_K is 27432 B, i.e. **18.8 KiB and 26.8 KiB**, against 56.5 KiB
in the baseline.

Three things follow.

**The grid is no longer 14.** `stream_k=0` for the dp4a table, so the launch is the full
`dim3(nty, ntx, ntzw)` tiling and the `mul_mat_q_stream_k_fixup` kernel is not launched at all. The
MoE expert matmul is a grid of **262144 blocks**, not the `(14,1,1)` single wave the baseline built.
So the "one wave, nothing to absorb imbalance" problem identified in `NCU_MOE_KERNEL.md` is gone
with this change, as a side effect.

**Occupancy roughly quadrupled.** smem is 19232 B for Q4_K and 27432 B for Q6_K, against 56.5 KB in
the baseline, and registers are **115** against **255**. Against a 48 KiB per-block limit and 64 KiB
per SM, 2 blocks per SM now fit where 1 did; the register pressure that capped the baseline at 255
no longer binds. Both of the mechanical penalties in `CALIBRATION.md` sections 1c and 1e (the 1.29x
read penalty and the 1.44x smem/sync penalty at 1 block/SM) are therefore relieved, which is
consistent with the measured 2.98x compute gain.

**J is capped at 64 by the table**, where the baseline used 128. `ntx` is therefore 16 rather than
8, so the weight read is duplicated twice as many times across the token dimension. That is the
cost side of this change, and it may be part of why the gain is 3x and not the ~10x the instruction
microbenchmark suggests.

### The tiling table in `CALIBRATION.md` is correct

The corrected table in `CALIBRATION.md` section 1 lists five sites. Two of them are dp4a rows; both
check out exactly against the instrumented print:

| site, as written in `CALIBRATION.md` | grid claimed | grid measured | smem claimed | smem measured |
|---|---|---:|---:|---:|
| MoE gate/up, Q4_K, ub 1024, dp4a | `(8, 16, 512)` = 65536 | `nty=8 ntx=16 ntzw=512` = 65536 | 19.2 kB | 19232 B (19.2 kB) |
| MoE down, Q6_K, ub 1024, dp4a | `(32, 16, 512)` = 262144 | `nty=32 ntx=16 ntzw=512` = 262144 | 27.4 kB | 27432 B (27.4 kB) |

So the dp4a rows that were added to `CALIBRATION.md` are right, and the occupancy argument built on
them holds: the dp4a tiles are **18.8 KiB and 26.8 KiB**, not Ampere's 56.5 KiB, so the 1 block/SM
constraint the section 1c and 1e penalties were derived from is what the dp4a table relieves.

The three baseline rows in that table (`(4, 64, 512)`, `(16, 64, 512)`, `(4, 64, 1)`) have **not**
been re-verified with the instrumented print, because the `LLAMA_MMQ_DEBUG` line was added only to
the dp4a build. They are still inferred. Given that the dp4a rows were inferred too and came out
exactly right, they are probably fine, but `CALIBRATION.md` labels only the *grid* as corrected and
should not be read as having verified the baseline rows.

### Is there another 50% in "the right amount of parallel operations"?

Partly, but not from the tile table. The tiling is no longer the limiter, and this is measurable
rather than a judgement:

- Occupancy is no longer 1 block/SM. Registers fell 255 -> 115, so two blocks fit. The 1.29x read
  penalty and the 1.44x smem/sync penalty that `CALIBRATION.md` sections 1c and 1e measured at
  1 block/SM no longer apply. Squeezing more parallelism out of the tile shape cannot recover a
  penalty that has already been paid.
- The grid is no longer a single wave. 262144 blocks against 14 SMs at 2 blocks/SM is ~9400 waves,
  so there is ample parallelism and no tail effect. Whatever the ceiling is now, it is not
  "not enough blocks".

What *is* left, in order of size:

| remaining limit | value | where it is measured |
|---|---:|---|
| copy at 12.4 GB/s, not overlapped | 62% of the ubatch | serial phases above |
| compute at 2039 ms | 34% of the ubatch | serial phases above |
| the 1336 ms neither phase accounts for | 19% of the ubatch | 7096 ms ubatch - 5760 ms phases |

The 1336 ms is the only item that is neither explained nor at a hardware ceiling, and 1336/7096 =
**18.8%**, which is close to the ~50% question but not equal to it. So the honest answer is: the
remaining gains are in **overlapping the copy** (worth up to 1.84x, see the projection below) and in
**finding what the 1336 ms is** (worth up to 1.23x), not in re-tuning parallel operations. The tile
sweep that would test the parallelism question directly is listed in `BEFORE_CONTRIBUTE_TASKS.md`
as a before-contribute item, not as the next experiment.

## Next step: overlap the copy with the compute

The copy is 62% of the ubatch and sits at the PCIe ceiling. It is already overlapped in the normal
2-slot path, but the serial instrument shows the phases sum to more than the ubatch, so the overlap
is not complete. Hiding the copy is now the largest single win available.

Using the measured phase times (compute 2039.1 ms and copy 3720.9 ms over 46 layers for 1024
tokens), plus the two throughputs actually measured:

| configuration | per-ubatch | tokens/s | measured? |
|---|---:|---:|:---:|
| compute only, no copy at all | 2039 ms | 502 | no, unachievable |
| copy only, no compute at all | 3721 ms | 275 | no, unachievable |
| ideal: copy and compute fully overlapped | max(2039, 3721) = 3721 ms | 275 | no |
| serial, 1 slot, `-ngl 99` | 7096 ms | **143.2** | **yes** |
| pipelined, 2 slot, `-ngl 1` | 6977 ms | **146.8** | **yes** |

**The copy-bound ceiling is 275 t/s**, not 502. Even if the compute became free, every ubatch still
has to move 45.86 GiB at 12.4 GB/s, which takes 3721 ms. So the headroom from *perfect* overlap is
143 -> 275 t/s, a further **1.92x**, and that is the hard ceiling for this approach at ub 1024.

**This does not need to be calculated - it falls out of the copy rate, which is a hardware
constant.** 45.86 GiB / 12.4 GB/s = 3713 ms per ubatch, and 1024/3.713 = 276 t/s. The measured
3720.9 ms agrees to 0.2%, so the ceiling is already known and no sweep is needed to find it.

**But the two measured rows say something uncomfortable.** The 2-slot pipeline, which exists
*precisely to* overlap the copy with the compute, is **1.7% faster than the serial one** (146.8 vs
143.2 t/s). It is not at 275 t/s, and it is not even at 178 t/s. So whatever overlap the double
buffer is achieving is almost nothing: the two configurations sit 2% apart when their ideal
separation is 1.92x.

That makes the copy-overlap work the largest single lead, and it also means the lead should be
approached as a **bug hunt, not an optimisation**. Something is serialising the copy against the
compute, and the candidates are already enumerated in `Tasks_Fix_CUDA_Asynchronous_Copy.md`
(unpinned host buffers silently making `cudaMemcpyAsync` synchronous; streams created without
`cudaStreamNonBlocking`; library work landing on the null stream and becoming a global barrier).
The `<1% overlap` figure in that file's context section is consistent with the 1.7% measured here.

Note that the serial figures above are **not** directly comparable to the pipelined ones as an
accounting: in serial mode the phases are disjoint and they sum to 5760 ms of a 7096 ms ubatch, so
1336 ms (18.8%) sits outside both phases. That residual is host-side gap between the timed regions,
not unmeasured device work, and it is why the serial row cannot simply be read as "copy + compute".

### Why ub 2048 should help, and what it is worth

The copy is **fixed at 45.86 GiB per forward pass** and does not depend on the token count, while
the compute is roughly linear in tokens. Doubling the ubatch to 2048 doubles the compute (2039 ->
4078 ms) and leaves the copy at 3721 ms, so the balance flips back to compute-bound:

| ub | compute | copy | bound by | per-ubatch | tokens/s |
|---:|---:|---:|---|---:|---:|
| 1024 | 2039 ms | 3721 ms | copy | 3721 ms (overlapped) | 275 |
| 2048 | 4078 ms | 3721 ms | compute | 4078 ms | 502 |
| 2048, no overlap | 4078 + 3721 = 7799 ms | - | - | 7799 ms | 263 |

So at ub 2048 with the copy hidden the limit becomes compute, at about **500 t/s** - but only
1.8x above the ub 1024 copy-bound ceiling, because the copy is only fully hidden if it overlaps.
Without overlap ub 2048 would be *slower* per token than ub 1024 (263 vs 275 t/s), so **the overlap
has to come first**; increasing the ubatch alone buys nothing.

**A practical obstacle for ub 2048.** The context could not be created at that size even at
`-ngl 1`: the compute buffer grows to 625.52 MiB at ub 1024 and the `-ngl 99` attempts show the
allocation failing well below that. Whether ub 2048 fits in 3716 MiB alongside a 1446 MiB model
buffer and the streaming slot has **not** been tested. Measure before relying on the projection.

All projections are arithmetic from measured phase times, not simulation. They assume the copy rate
stays at 12.4 GB/s and the compute stays linear in tokens, which holds over the 512 -> 1024 range
measured here (compute 2898.8 -> 6066.0 ms for 2x tokens on the baseline) but is unverified at 2048.

## What is established and what is not

### Calculated or swept?

**The parallel-operation targets are calculated, not swept.** Both limits come from measured
quantities, not from search:

- the copy-bound ceiling is `weights / PCIe rate` = 45.86 GiB / 12.4 GB/s, and the measured copy
  phase (3720.9 ms) confirms it to 0.2%
- the compute-bound ceiling is the measured compute phase (2039.1 ms) extrapolated linearly

The one thing that genuinely needs measuring is **whether the two can in fact overlap**. The serial
instrument shows they sum to 5760 ms in a 7096 ms ubatch, i.e. 1336 ms of the ubatch is neither,
which is where the overhead lives. That is a structural question about the 2-slot pipeline, not a
number to be found by sweeping ubatch sizes.


**Established:** the dp4a path selects, the kernel contains `IDP` instead of `IMMA`, output is
identical, and throughput rises 1.53x at pp512 and 1.77x at pp1024 while power rises 22.5%.

**Not established:** how much of the gain is the change of *instruction* rather than the change of
*tile shape*. The dp4a table also uses `I=64, J<=64` and `stream_k=false`, where the ampere table
used `I=128, J<=128` and `stream_k=true`. All three differ at once. Isolating them needs four
builds; the plan is in `BEFORE_CONTRIBUTE_TASKS.md` section 2, and it is deferred rather than
blocking, because the *review* is what needs the attribution, not the measurement. The
microbenchmark shows the instruction is 10x slower in isolation, but that is not the same as
attributing the end-to-end gain to it.

**Measured: `-ngl 99` benefits more than `-ngl 1` overall.** On prefill the gap closed to 2.4%
(143.22 against 146.79), against 8.7% on the baseline, and on decode `-ngl 99` is 1.95x faster
(18.07 against 9.28 t/s). So `-ngl 99` is now the preferred configuration, reversing the baseline's
conclusion. See the section above.

**The one measurement that is still open and is not an arithmetic question** is the 1.7% overlap
between the serial and pipelined configurations, where the ideal separation is 1.92x. Everything
else above follows from a measured phase time or from the bus. This does not.

**Relevance to the power question.** The standing question was why power only reaches ~25% of the
50 W cap. On the baseline, GPU power during this workload sits at 18.2 W. With the workload now
almost twice as fast it still only reaches 22.3 W mean and 30.2 W peak. So the dp4a change explains
why the part was *slow*, and it does raise power per unit time as predicted, but it does not by
itself explain the ceiling: even at 1.77x throughput the part is not near its limit.

The 1.7% overlap is a live candidate for the rest of it. If the copy and the compute are in fact
serialised, then the card is idle in the arithmetic sense for 62% of each ubatch, and the power mean
is a weighted average of a compute phase and a near-idle copy phase. `CALIBRATION.md` section 1f
measured a pinned H2D copy at **16.1 W** against 12.8 W idle, so a phase that cannot draw much is
62% of the run. That is consistent with a ceiling near a third of the power limit without any
additional explanation, and it makes the copy-overlap work answer two questions at once.

## Upstream status

Prototype only. The `turing_devices_without_mma` list in `ggml-cuda.cu` shows the maintainers
already intended to handle these parts; the list is incomplete and only warns rather than acting.
Any upstream change needs an issue and maintainer agreement first, per `AGENTS.md`. The build option
added here is a local experiment, not a proposed patch.
