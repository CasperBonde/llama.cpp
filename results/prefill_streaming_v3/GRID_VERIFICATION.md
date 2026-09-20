# Verifying the 84.75 ms figure, and what I found while doing it

Date: 2026-09-20. Source of the number: `tmp/prof/inventory.txt` (raw ncu CSV, 900 rows).

---

## 1. The figure is confirmed from raw data

The CSV has **15 columns**, not 14: `...,"Block Size"(8),"Grid Size"(9),...`. This differs from
the reduced-metric files, where Block Size is column 7 and Grid Size column 8. Every earlier
mistake of mine in this investigation came from assuming a fixed column order, so the parse was
redone from the header:

```
kernel                    grid          block         n    total_ms  ms_each
mul_mat_q<ggml_type 12,   (8192, 1, 1)  (32, 8, 1)   32    2711.9    84.75
  (int)128, (bool)0>
mul_mat_q<...>            (32768, 1, 1) (32, 8, 1)    9     853.1    94.78
mul_mat_q<...>            (32768, 1, 1) (32, 8, 1)    7     521.7    74.53
mul_mat_q<...>            (256, 1, 1)   (32, 8, 1)   13     233.5    17.96
mul_mat_q<...>            (64, 1, 1)    (32, 8, 1)   16     150.1     9.38
mul_mat_q<...>            (128, 1, 1)   (32, 8, 1)   12     113.2     9.43
gated_delta_net_cuda      (32, 1, 32)   (32, 4, 1)   13      79.1     6.08
volta_h884gemm_256x128    (1, 4, 16)    (256, 1, 1)   4      32.5     8.12
```
Total kernel time across the 900-launch window: **4942.6 ms**. The grid-8192 `mul_mat_q` is
**2711.9 ms = 54.9%** of it, at 84.75 ms per launch. The number in `PHASE5_RESULTS.md` is
correct. Note the metric values carry thousands separators (`"76,928"`), which silently broke a
first parse.

## 2. The templated J is 128, and that resolves the whole launch shape

The kernel is named `mul_mat_q<ggml_type 12, (int)128, (bool)0>`, i.e. `type = Q4_K`,
**`J = 128`**, `fallback = false`. Two things follow.

**2a. The kernel used is not the one I expected.** Source dispatch in `mmq.cuh:229`:

```cpp
if (blackwell_mma_available(cc))                                  -> blackwell
if (ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)    -> ampere   // sm_75 hits this
if (ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_DP4A)     -> pascal_dp4a
```

The dispatch tests the **highest architecture the binary was compiled for**, not the running
device. If the build includes sm_80+, then on this sm_75 device `highest_compiled_arch >= VOLTA`
and the **ampere** config table is selected. The pascal-dp4a table (which gives I=64, J=8..64,
`stream_k = false`) is *not* what runs.

**2b. Grid and block follow from I=128, J=128.** From `mmq.cu`'s MoE branch, `nte10 (ncols)` =
512 tokens, `nrows_x` = 512 output rows, `nchannels_y` = 512 experts:

```
nty  = (512   + 128 - 1) / 128 = 4
ntx  = (512   + 128 - 1) / 128 = 4
ntzw = 512 experts
ntiles_dst = 4 * 4 * 512 = 8192
```

And because Q4_K has `stream_k = true` in the ampere table, `launch_mul_mat_q` chooses the grid
by efficiency:

```
tiles_nwaves = (8192 + 14 - 1) / 14 = 586
efficiency   = 100 * 8192 / (14 * 586) = 99%   >= 90%
-> block_nums_stream_k = (ntiles_dst, 1, 1) = (8192, 1, 1)
```

**Grid (8192, 1, 1) reproduced exactly**, and `block = (warp_size, nthreads/warp_size, 1) =
(32, 8, 1)` with `nthreads = 256`. Both fields in the inventory are now explained from source
rather than assumed.

Note also `use_mma_data_layout()` returns true for `turing_mma_available(cc)`, so this path
allocates the MMA-layout smem (`I * sram_stride * 4 = 128 * 136` = **68 KB**), not the dp4a
layout (`(qs+dm+sc)*4 = 256*4 = 1 KB`). See section 4.

## 3. The new finding: token tiling duplicates the weight read 4x

Read the tile assignment in `mmq.cuh`:

```
nty = (nrows_x   + I - 1) / I      I = 128  (output rows per tile)
ntx = (ncols_max + J - 1) / J      J = 128  (TOKENS per tile)
grid = (ntx, nty, ntzw)
```

Block `(tx, ty, e)` reads weight rows `[ty*I, (ty+1)*I)` of expert `e`, across the **full** K
dimension. Blocks with the same `ty` and `e` but different `tx` therefore load **the same weight
bytes**. The total weight read *demand* is

```
demand = ntx x (weight bytes)
```

At ub = 512 and q4_K, I = J = 128, so `ntx = 4`: **every expert weight byte is requested 4
times.** `tmp/prof/ntx.cu` measures whether L2 absorbs this. It builds grid `(ntx, 4, 512)` where
every block reads its own 128-row slice, so the `tx` blocks genuinely duplicate, and sweeps ntx:

| ntx | blocks | demand | time ms | time vs ntx=1 | DRAM traffic (inferred) | L2 absorbed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2048 | 512 MiB | 4.556 | 1.00x | 512 MiB | - |
| 2 | 4096 | 1024 MiB | 4.624 | 1.02x | 521 MiB | 49% |
| **4** | **8192** | **2048 MiB** | **7.311** | **1.60x** | **823 MiB** | **60%** |
| 8 | 16384 | 4096 MiB | 14.211 | 3.12x | 1601 MiB | 61% |

Reading this:

- at ntx = 1 the kernel runs at **117.8 GB/s**, i.e. exactly the DRAM ceiling
- **L2 absorbs about 60%** of the duplicated read, as expected of a 1 MB cache
- the remaining 40% leaks to DRAM, so at ntx = 4 DRAM traffic is **1.60x** what it needs to be
  (823 MiB instead of 512 MiB), and the time grows by exactly that factor
- beyond ntx = 4 the leak saturates at ~61% and stops growing linearly, because the demand
  outruns what L2 can hold

**Consequence for this investigation:** the real kernel at ub = 512 pays a **1.6x DRAM
amplification** on every expert weight read, purely from choosing a 128-token tile. The bytes the
kernel *needs* are 512 MiB per projection; the bytes DRAM actually serves are ~820 MiB. This is
the first mechanism found that is (a) real, (b) mechanical, (c) directly attributable to a
source-level design choice, and (d) **fixable by making the token tile larger** - the opposite of
the token-tiling idea, which would make it worse.

It also explains an old bench observation that was never accounted for: throughput *falls* as
`-b/-ub` gets smaller. Smaller ub -> smaller `ncols_max` -> for `ncols <= 128` ntx stays 1 and
this cost vanishes, but for large ub it scales with ntx.

## 4. What this does *not* explain

Applying the 1.6x correction to the real kernel:

| quantity | before | after 1.6x correction |
|---|---:|---:|
| distinct weight bytes per launch | 512 MiB | 512 MiB |
| DRAM bytes | (assumed 512 MiB) | ~820 MiB |
| measured time | 84.75 ms | 84.75 ms |
| **implied DRAM rate** | 6.3 GB/s | **10.1 GB/s** |
| share of 118 GB/s ceiling | 5.3% | **8.6%** |

So the duplication takes the kernel from 5.3% to 8.6% of the ceiling. It is real, and it is
the largest single inefficiency identified so far, but it is **not** the 9.6x. Meanwhile the
synthetic at the **identical grid (8192 blocks), identical block size and identical bytes**
finishes in **7.3 ms** against the real kernel's **84.75 ms**. That is **11.6x** unexplained.

The mechanisms measured and ruled out so far:

| mechanism | measured cost | how |
|---|---:|---|
| occupancy 1 block/SM | 1.29x | `occ2.cu` smem sweep, read+FMA |
| smem tile + `__syncthreads` | 1.44x | `mmqshape.cu` tile sweep |
| dp4a integer arithmetic | none (91-95% lane rate) | `mmqshape.cu` A |
| FP32 pipe saturation at 1 block/SM | none (94.4% peak) | `occ2.cu` A |
| token-tile weight duplication | 1.60x DRAM | `ntx.cu` |

Those multiply to roughly 1.3-1.9x. Against 11.6x, **~6x is still unattributed.**

## 5. An arithmetic inconsistency worth resolving before trusting more timings

`pp512` streaming is 64.5 t/s, so one prefill takes **7.94 s**. The model has **48 MoE layers**
(144 expert tensors / 3 per layer), and each projection launches grid 8192. That is
**144 grid-8192 launches per prefill**. At the measured 84.75 ms each that would be
**12.2 s** - more than the entire prefill takes.

So one of these is false:
1. 144 launches per prefill,
2. 84.75 ms per launch, or
3. the 7.94 s prefill time.

The inventory window captured only 32 grid-8192 launches, i.e. **22% of a prefill's worth**
(32/144). So the ncu window covers roughly a fifth of one ubatch. That is a small enough sample
that per-launch inflation from application replay is a real possibility, and it means the 84.75
ms has **not** been cross-checked against an independent timing.

**Cheap resolution, no profiler:** run one `llama-bench` at `-b 512 -ub 512` with
`LLAMA_PREFILL_STREAMING_DEBUG=1` and `-v`, and read the per-layer timings the streaming code
already logs. That costs one model load and gives host-side, unprofiled numbers for the same
work. If the summed layer time is far below 12.2 s, the 84.75 ms is profiler-inflated and every
derived bandwidth number in `PHASE5_RESULTS.md` needs re-deriving from the new figure.

This is the cheapest and most consequential remaining measurement, and it should happen before
any 3-hour stall-counter batch, because the stall counters describe a kernel whose duration is
not yet trustworthy.
