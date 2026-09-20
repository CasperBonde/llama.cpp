# Calibration: the GPU is healthy, the kernel is not

Date: 2026-09-20. Bench: `/tmp/prof/synth.cu` and `/tmp/prof/occ.cu` (in the repo under `tmp/prof/`).
Run time: seconds, not hours. This is the fast synthetic path in `MEASUREMENT_STRATEGY.md` section 3.

**Power section added 2026-09-20** (`tmp/prof/powmeas.cu`). Power is the one instrument that
cannot be faked by a launched-but-stalled grid, so every synthetic extreme here now carries its
wattage and the real run is compared against them on that instrument alone. See section 1f.

---

## 0. Hardware facts, measured directly

| item | value |
|---|---:|
| SMs | 14 |
| threads/SM (hard limit) | 1024 |
| shared memory per block | **48 KB** |
| shared memory per SM | 64 KB |
| registers per SM | 65536 |
| L2 | 1024 KB |
| sm clock / mem clock | 1455 / 4001 MHz |
| theoretical DRAM | 128.0 GB/s |
| FP32 peak | 2.61 TFLOP/s |
| **idle power draw** | **12.6-13.2 W** |
| **power limit (enforced)** | **50.00 W** |

Idle power is a real floor, not zero: the card draws ~13 W doing nothing. Every figure below is
an absolute draw, so subtract that floor to get the cost of the work.

## 1. The synthetic kernels

All with **grid 8192, block 256, 996 MiB** - identical to the llama.cpp grid-8192 expert
kernel's launch shape.

| test | GB/s | notes |
|---|---:|---|
| `read_chunk` sequential uint4 read | **118.1** | 4 blocks/SM |
| `read_strided` strided uint4 read | **118.1** | same, access pattern irrelevant |
| `mixed` read + 8x FMA passes | **118.1** | compute does not slow the read |
| `busy_spin` `clock64()` poll, no memory | - | burns SM cycles, moves zero bytes |
| `read_chunk` with 33-48 KB smem, **1 block/SM** | **112.2** | **== expert kernel occupancy** |
| **llama.cpp grid-8192 expert kernel** | **11.7** | measured earlier, 996 MiB in ~85 ms |

### The headline

**Same bytes, same grid, same block size, same occupancy: 112.2 GB/s synthetic vs
11.7 GB/s in llama.cpp. A 9.6x gap.**

The synthetic read reaches **88% of the 128 GB/s theoretical peak**. The hardware is working
as specified.

## 1b. Correction: the first `mixed` test was not a real compute proxy

`synth.cu`'s `mixed` kernel passed at 118.2 GB/s and was reported as "compute does not slow
the read". That result is **too weak to carry the conclusion**. It did 8 FMAs per 16-byte load,
i.e. **0.125 flop/byte**, while the real expert kernel is **30-35 flop/byte** at ub512. Also,
its FMA count was tunable but not truly measured. So it showed only that a *little* compute is
free, not that the *real* compute load is free.

`intense.cu` redid this properly, sweeping FMAs per 16-byte load and reporting achieved
GFLOP/s next to GB/s:

| flop/byte | achieved GB/s | achieved GFLOP/s | % of FP32 peak | **power W** | **% of 50 W** |
|---:|---:|---:|---:|---:|---:|
| 2.0 | 118.1 | 236 | 9.1% | 28.8 | 57.6% |
| 8.0 | 118.0 | 945 | 36.2% | 31.3 | 62.7% |
| 16.0 | 118.0 | 1888 | 72.4% | 34.1 | 68.1% |
| 18.0 | 118.0 | 2123 | 81.4% | ~34.5 | ~69% |
| **20.0** | **116.8** | **2337** | **89.6%** | **35.4** | **70.8%** |
| 22.0 | 108.5 | 2387 | 91.5% | ~35 | ~70% |
| 24.0 | 100.5 | 2411 | 92.5% | 34.4 | 68.9% |
| 28.0 | 86.9 | 2434 | 93.3% | ~33.5 | ~67% |
| **30.0** | **81.2** | **2437** | **93.5%** | **32.9** | **65.8%** |
| 32.0 | 76.3 | 2441 | 93.6% | 32.4 | 64.8% |

(By-8 flop/byte rows are direct measurements from `powmeas.cu`; the interleaved rows are
interpolated because the two harnesses sweep slightly different grids. Re-measured throughput
reproduces `intense.cu` to within 0.2 GB/s.)

The **crossover is at 18-20 flop/byte**, which is the (empirical) ridge point: 2.61 TFLOP/s /
118 GB/s = 22. Read bandwidth survives up to ~18 flop/byte and then the FMA pipe takes over.

So a kernel at the real 30-35 flop/byte should run at **~76 GB/s and ~2.44 TFLOP/s**. The real
kernel does **12.3 GB/s and 0.36 TFLOP/s**. It is **6x slower than its own arithmetic intensity
allows**, and 6-7x below both peaks in both pipes simultaneously.

**Power peaks exactly at that same ridge (20 flop/byte, 35.4 W) and falls away on both sides.**
Below the ridge the memory system is the limit and the SMs idle in the memory shadow; above it
the FMA pipe is the limit and the memory system idles. Peak board draw for the hardest workload
the GPU can be given is **35.4 W of a 50 W budget** - the card cannot draw its own limit, and it
never throttles (section 1f).

(Note: `intense.cu` had to be fixed twice. `n = 0` let the compiler delete the load entirely -
it briefly reported 4283 GB/s, which is above the hardware, and also 24621 GB/s at the top of
the table, both impossible and both discarded. And `acc[k & 7]` has a dynamic index, which
spills the array to local memory; it was replaced with 8 static accumulators. Any number above
118.1 GB/s in this file is a dead-code bug, never a result.)

## 1c. Occupancy does throttle throughput, but only 1.5x

`occ2.cu` swept the dynamic smem request while doing the read+FMA mix, to check whether the
1 block/SM that MMQ is stuck with throttles the pipes:

| smem | blocks/SM | threads/SM | read+FMA GB/s at 8 flop/byte |
|---:|---:|---:|---:|
| 1 KB | 4 | 1024 | 118.1 |
| 16 KB | 3 | 768 | 118.1 |
| 24 KB | 2 | 512 | 117.9 |
| **40 KB** | **1** | **256** | **91.8** |
| 48 KB | 1 | 256 | 91.8 |

Pure FMA, no memory, at the same occupancies:

| smem | blocks/SM | GFLOP/s | % FP32 peak |
|---:|---:|---:|---:|
| 1 KB | 4 | 2482 | 95.2% |
| 40 KB | **1** | 2461 | **94.4%** |

Two conclusions:

1. **The FP pipe saturates fine at 1 block/SM** (94.4% of peak, 8 warps, 8 independent chains).
   So low occupancy does **not** explain 0.36 TFLOP/s - the pipe itself is capable.
2. **The read does throttle at 1 block/SM**: 118 -> 91.8 GB/s, a 1.29x penalty. Real, but it
   accounts for only a fraction of the 9.6x.

## 1d. dp4a is fast, so the integer path is not the limit

MMQ is an **integer** kernel (dp4a = 4 int8 MACs per instruction), and every synthetic test up
to here measured FP32 FMA. `mmqshape.cu` measured dp4a directly:

| smem | blocks/SM | dp4a/s | int8 GOPS | % lane rate |
|---:|---:|---:|---:|---:|
| 0 | 4 | 1.242e12 | 9940 | 95.3% |
| 40 KB | **1** | 1.187e12 | 9493 | **91.0%** |

dp4a runs at **91-95% of the full lane rate**. It is not a slow path on TU117 and it does not
degrade at 1 block/SM. So the integer arithmetic is not why the expert kernel crawls.

-> the arithmetic can be executed at ~16x the rate the kernel achieves. The kernel is not
compute bound in the ordinary sense; it is not issuing.

## 1e. The smem + `__syncthreads` structure *does* reproduce the slowdown

`mmqshape.cu` also built the MMQ *structure* without the MMQ math: per K-tile, cooperatively
load a tile into smem, `__syncthreads()`, read it back and accumulate, `__syncthreads()`.
Sweeping the tile size sweeps occupancy (64 KB smem per SM):

| kernel | smem | blocks/SM | GB/s |
|---|---:|---:|---:|
| `direct_read` (no smem) | 0 | 4 | **118.1** |
| `smem_nosync` | 8 KB | 4 | 114.9 |
| `smem_serial` | 8 KB | 4 | 114.9 |
| `smem_nosync` | 16 KB | 4 | 114.9 |
| `smem_serial` | 16 KB | 4 | 114.9 |
| `smem_nosync` | 32 KB | 2 | 114.8 |
| `smem_serial` | 32 KB | 2 | 114.7 |
| **`smem_nosync`** | **40 KB** | **1** | **82.2** |
| **`smem_serial`** | **40 KB** | **1** | **79.6** |
| `smem_double` | 16 KB | 4 | 96.7 |
| `smem_double` | 32 KB | 2 | 104.4 |

Reading this carefully, because it is easy to overclaim:

- the syncs themselves are **nearly free**: 114.9 vs 114.9 GB/s at 8 and 16 KB, and 82.2 vs
  79.6 at 40 KB. The barriers cost 3%.
- the **1 block/SM is most of the damage**: 114.7 (2 blocks) -> 79.6 (1 block). A 1.44x penalty.
- double buffering did **not** help here, because without `cp.async` the "prefetch" is still a
  synchronous load that blocks the same warp.

So the smem+sync structure reproduces **1.44x**, not 9.6x. It is a real contributor and it is
the largest single mechanical factor found so far, but **it is not the whole gap**. The search
continues: something accounts for the remaining ~6x.

> **Correction (2026-09-20, later).** `mmqstruct.cu` passed the tile size as a **runtime**
> argument, so its loop had no compile-time trip count and could not unroll - each thread had
> one 4-byte load in flight. `tmp/prof/unroll.cu` shows the same structure with a **compile-time**
> trip count reaching **118.1 GB/s at 1 block/SM**. So the 79.6-82.2 GB/s measured here is partly
> a property of the harness, not of the structure, and the 1.44x is an upper bound. The real
> kernel was then disassembled: its `#pragma unroll` is fully honoured (7944 straight-line
> instructions, 18 independent loads back to back, median load-use distance 145). **The load
> width and the unrolling were never the problem.** The numbers above are kept because they are
> correct measurements of what they measured; the interpretation is withdrawn.

---

## 1f. Power: the instrument that cannot be faked, and what it says

`utilization.gpu` is satisfied by any launched grid, so it cannot tell work from waiting. Power
can: a grid that is resident and stalled burns little. `powmeas.cu` measures power with NVML
directly (not `nvidia-smi`) over every synthetic extreme above, with the same grid 8192 /
block 256 / 996 MiB as the real expert kernel.

| workload | throughput | power | % of 50 W | x idle |
|---|---:|---:|---:|---:|
| idle | - | **12.8** | 25.7% | 1.00 |
| spin, no memory at all | 0 bytes | **25.9** | 51.7% | 2.02 |
| H2D copy, pinned | 12.5 GB/s | **16.1** | 32.2% | 1.26 |
| pure VRAM read | 118.1 GB/s | **28.0** | 56.0% | 2.19 |
| pure VRAM read, 56.5 KB smem, 1 block/SM | 112.8 GB/s | **28.5** | 56.9% | 2.22 |
| read + dp4a, real kernel's intensity class | 117.9 GB/s | **35.4** | 70.8% | 2.76 |
| read + FMA, at the throughput ridge | 116.8 GB/s | **35.4** | 70.8% | 2.76 |

The real run, reference config, from a ~175 Hz NVML trace (`tmp/prof/pow_real.sh`):

| | |
|---|---:|
| throughput | 84.37 t/s (reproduced) |
| compute window | 69.6 s over 4 reps = 17.39 s/rep |
| mean draw over the whole window | **18.71 W** |
| energy per rep | 325 J |
| less the copy (3.83 s x 16.1 W = 62 J) | 13.56 s, 264 J |
| **the compute half alone** | **19.5 W** |
| min / max in the window | 15.9 / 22.6 W |
| sm clock | **1395 MHz for 99.1% of the window** (max 1455) |

### What it says

**19.5 W for doing the arithmetic is less than a spin loop that moves zero bytes (25.9 W), and
less than a plain VRAM read (28.0 W).** A full-bandwidth read of the same 996 MiB in the same
launch shape costs 28.0 W. The kernel draws less power than a workload that merely *reads* the
bytes it is supposed to compute on.

This is an independent confirmation of the corrected idle finding, reached by a different route.
The GPU is not idle - a grid is resident - but it is doing almost no work: the draw is 6-9 W
below the cost of either a pure read or a pure compute spin, and 14-16 W below the draw of a
workload that genuinely uses both pipes at once (35.4 W). It does not throttle: the clock sits at
1395 MHz for 99.1% of the window, so the low draw is not a power cap holding the clock down.

### The measurement limit

NVML `nvmlDeviceGetPowerUsage` updates on a **~95 ms internal window**, which is coarser than the
per-layer round trip (~90-130 ms). The trace therefore shows flat plateaus (18.12, 18.67, 18.90,
19.01, 21.94, 22.60 W...) that are **averages, not instantaneous values**, and the copy phase is
never resolved as a separate level. So:

- the copy/compute split (the 62 J subtraction above) is a subtraction, not a direct observation
- no per-layer periodicity is recoverable at this rate; measured autocorrelation is ~0 at lag 46
  samples (263 ms), and 1.2 W of variation from a 1100 J compute phase is under the noise floor

What power *can* tell us at this rate is the thing that matters: the total is far too low for the
work being claimed.

### Resolution: separate the phases by *folding*, not by polling faster

CUPTI is not needed after all. The serial mode (`LLAMA_PREFILL_STREAMING_SERIAL=1`) makes the
per-layer structure deterministic: one blocking copy (measured mean **80.9 ms**) followed by that
layer's compute (measured mean **131.9 ms**), so the trace has a known period of **212.8 ms** and a
known copy duty cycle of 38.0%. Folding the 173 Hz power samples modulo that period and scanning the
phase offset recovers the two levels *without* any per-sample resolution, because the averaging
window smears uniformly over a *periodic* signal and therefore does not bias the two halves.

Sensitivity check (the part that makes this result usable): fold at decoy periods with the same
duty cycle and see what contrast they produce by chance.

| period used | contrast | verdict |
|---|---:|---|
| **TRUE 212.8 ms (80.9 + 131.9)** | **-0.11 W** | real (i.e. indistinguishable from noise) |
| decoy 150.0 ms | 0.03 W | noise floor |
| decoy 240.5 ms | -0.03 W | noise floor |
| decoy 100.0 ms | 0.01 W | noise floor |
| decoy 300.0 ms | 0.15 W | noise floor |

The method resolves differences down to about **0.15 W**. The true-period contrast is **0.11 W**,
which is *inside* the noise floor, so:

| phase | power | samples |
|---|---:|---:|
| copy bins | **18.87 W** | 3328 |
| compute bins | **18.76 W** | 5440 |
| difference | **-0.11 W (-0.6%)** | |

**The copy phase and the compute phase draw the same power.** Streaming 1022.41 MiB at 12.35 GB/s
costs the same as the compute it competes with, and neither is distinguishable from a constant
18.8 W draw for the whole run.

This kills a specific and attractive explanation: that the copy is cheap on the power rail while
the compute is duty-cycled, and that the low average is the two averaging out. The low average is
**not** a duty cycle. Both phases are low. `-ngl 99` serial (compute 213.0 ms/layer, copy 78.7 ms)
folds to the same flat result: 19.07 W copy, 19.00 W compute, contrast -0.07 W.

What this leaves: the SMs are resident and not issuing for most of *both* phases, and the rate at
which that happens is not sensitive to whether a DMA engine is streaming through the same card.

## 2. Occupancy is ruled out as the cause

`occ.cu` sweeps the dynamic shared memory request, which is the only thing limiting occupancy
here (threads: 1024/SM / 256 = 4 blocks; smem: 64 KB/SM / 48 KB = 1 block):

| smem | blocks/SM | threads/SM | GB/s |
|---:|---:|---:|---:|
| 1 KB | 4 | 1024 | 118.1 |
| 9 KB | 4 | 1024 | 118.1 |
| 17 KB | 3 | 768 | 118.1 |
| 25 KB | 2 | 512 | 118.1 |
| 33 KB | **1** | **256** | **112.2** |
| 41 KB | **1** | **256** | **112.3** |
| 48 KB | **1** | **256** | **112.2** |

Falling from 1024 to 256 threads per SM costs only **5%**. So:

- **low occupancy is not the problem** (256 threads/SM still streams at 112 GB/s)
- **wave quantization is not the problem** (1274 blocks over 56 concurrent slots = 23 waves,
  still 112 GB/s)
- **only 1-2 SMs running is not the problem** (the synthetic uses all 14 and so, by geometry,
  does llama.cpp)

If the expert kernel were simply "a read with low occupancy", it would run at ~112 GB/s, i.e.
8.9 ms, not 85 ms.

## 3. What this means

The gap is **inside the kernel**, not in the hardware, the launch shape, the memory system or
the scheduling. Three candidates remain, in order of plausibility:

1. **Load/compute serialisation with no co-resident block to hide it.** The MMQ kernel
   cooperatively loads a tile into shared memory, `__syncthreads()`, then does the vec_dot
   compute, then syncs again. At 1 block/SM there is no second block on the SM to use the
   memory pipe while block 0 is computing, or the ALUs while block 0 is loading. Both pipes
   then idle in alternation. This predicts exactly what is observed: 13.7% of compute peak,
   9.6% of memory peak, 18.6 W of a 50 W budget.
2. **`stream_k` fixup barrier.** The `mul_mat_q_stream_k_fixup` pass and the `while(true)`
   loop in `mmq.cuh:1275` synchronise partial sums across blocks. With 1 block/SM and 23
   waves, blocks in later waves wait on earlier ones.
3. **dp4a dependency chains** serialising behind the loads at only 8 warps/SM.

Note the deliberate omission: `cp.async` is **not** a candidate. It is gated to
`__CUDA_ARCH__ >= 800` by design and this is an sm_75 part, so the hardware cannot do
asynchronous copies and there is nothing to fix there.

## 4. The busy-poll hypothesis: source-checked and not found

Checked directly, since it predicts a precise signature (SM busy, no traffic, low power):

- no `nanosleep`, no spin on a shared flag, no `__threadfence` polling in `mmq.cuh`
- the only `while(true)` is the fixup loop in `mul_mat_q_stream_k_fixup`, which reads real
  partial sums, so it moves data when it runs
- the one `cudaLaunchHostFunc` in `ggml-cuda.cu:4493` is inside `#if 0`, i.e. dead code
- host-side waits are `cudaStreamSynchronize` / `cudaStreamWaitEvent`, which block in the
  driver, not spin in userspace

So there is no explicit busy-wait. But the *symptom* is still consistent with your idea in a
different form: the SMs hold 1 block each and that block is stalled between phases. That looks
like 100% occupancy to `utilization.gpu`, moves little data, and burns little power - without
anyone writing a poll loop.

---

## 5. The test that would confirm the mechanism (fast, ~1 hour to write)

Build a kernel that copies the MMQ *structure* rather than the MMQ *math*:

```
for each K-tile:
    cooperative load of the weight tile into shared memory
    __syncthreads()
    dummy compute over the smem tile
    __syncthreads()
```

Sweep: (a) with/without the syncs, (b) with the tile loaded into smem vs read directly,
(c) 1 vs 4 blocks/SM. If the smem+syncs+1-block variant drops to ~10-15 GB/s while the direct
1-block read stays at 112 GB/s, the mechanism is confirmed and it is a *software pipelining*
problem, fixable by restructuring the kernel. If it stays fast, the cost is in the dp4a math
or the fixup barrier instead.

This is cheap enough to iterate on, and it validates against llama.cpp afterwards by
predicting the 85 ms launch time.

---

## 6. Method: the serial measurement window, and the LD_PRELOAD sync audit

Two instruments added this session. Both are reusable, and neither needs ncu or CUPTI.

### 6.1 The serial measurement window

`LLAMA_PREFILL_STREAMING_SERIAL=1` uses **one** slot instead of two and forces the next layer's
copy to be **blocking** inside the eval callback, before that layer is submitted. Copy and compute
are then disjoint intervals and the code times them per layer:

```
one callback invocation per layer:
    refill(layer n+2)   <- blocking copy, timed around the call
    compute(layer n+1)  <- timed from the ask to the next callback
```

Properties that make it a usable instrument:

- the result reproduces: **81.36 +- 0.06 t/s** at pp1024 `-ngl 1`, against 82.49 +- 0.05 for the
  overlapped pipeline,
- the copy rate it reports (12.34-12.35 GB/s) matches an independent pinned H2D benchmark
  (12.4-12.5 GB/s) and the measured per-layer transfer time (81.7 ms with no compute at all),
- the copy it reports is **token-count independent**: the same 48 layer transfers take 3733.6 ms
  at p512 and 3720.1 ms at p1024, 0.4% apart, which is what a fixed 45.86 GiB of weights must do,
- the compute it reports scales correctly: 2898.8 ms at p512 and 6066 ms at p1024 over the same 46
  layers, a ratio of 2.09 for 2x the tokens,
- it turns the power trace into a known-period signal, which is what section 1f folds on.

It also answers the design question directly: serial 81.36 against overlapped 82.49 means the
overlap buys **1.4%**, so the copy already fits inside the compute time.

### 6.2 LD_PRELOAD sync audit

`tmp/prof/syncspy.c` interposes `cudaStreamSynchronize`, `cudaDeviceSynchronize`,
`cudaEventSynchronize`, `cudaStreamWaitEvent`, `cudaMemcpyAsync`, `cudaMemcpy`, `cudaStreamQuery`
and `cudaLaunchKernel`. It works because `libggml-cuda.so.0` links `libcudart.so.12` dynamically,
so `LD_PRELOAD` interposition applies. Run:

```sh
SYNCSPY_OUT=tmp/prof/spy.log LD_PRELOAD=tmp/prof/syncspy.so <llama-bench ...>
```

Result on `-p 128 -ub 128 -r 1` (4058 events, 5415 launches):

| call | n | total ms | max us | mean us |
|---|---:|---:|---:|---:|
| `cudaMemcpyAsync` | 2283 | 6.4 | 53 | 2.8 |
| **`cudaStreamSynchronize`** | **1432** | **15062.8** | **271014** | **10518.7** |
| `cudaEventSynchronize` | 249 | 0.3 | 28 | 1.1 |
| `cudaStreamWaitEvent` | 92 | 0.1 | 3 | 1.0 |

Two populations in the sync count:

- **1002** syncs on the compute stream, about **15 ms each**. These are the real per-split waits and
  they line up with the ~15 ms of kernel per layer, so that part is the GPU genuinely busy.
- **430** syncs on `cudaStreamPerThread` (`0x2`), paired with 428 `cudaMemcpyAsync` on the same
  stream. These are `ggml_backend_cuda_buffer_set_tensor`, which does `cudaMemcpyAsync` then
  `cudaStreamSynchronize` **per tensor**.

What it establishes: llama.cpp is not blocking the device. Total `cudaStreamSynchronize` time is
15.06 s in a ~32 s compute phase, and the excess over 15.4 s of actual kernel is the per-tensor
`set_tensor` population, not a device-wide stall. The audit rules out "the host is waiting" as the
mechanism; it does not by itself explain the low power, which is what the folded trace in 1f does.
