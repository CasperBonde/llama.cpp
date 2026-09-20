# RETRACTION: the "smoking gun" was a bug in my own test

Date: 2026-09-20. Supersedes `SMOKING_GUN.md` sections 2-4 and 6.

---

## What I claimed

I built `tmp/prof/mmqstruct.cu` to reproduce the MMQ load/compute structure and measured
**34.2 GB/s** against the real kernel's 12.3 GB/s, and concluded:

- the weight loader's 32-bit `get_int_b4` loads plus 1 block/SM were the cause
- **vectorising** the load to 128 bits was worth **2.50x** and was the fix to pursue

## Why it was wrong

`mmqstruct.cu` passed the tile size as a **runtime** argument, so its load loop was:

```cpp
for (int i = threadIdx.x; i < tile_ints; i += blockDim.x) { tile[i] = src[i]; }
```

`tile_ints` is not a compile-time constant, so the compiler **cannot unroll** it. Each thread
therefore had exactly **one load in flight**, and the loop overhead was paid on every 4 bytes.

The real kernel is not in that state. In `ggml_cuda_mmq_load_tiles_q4_K`:

```cpp
constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_K);   // constexpr
constexpr int nrows = warp_size / threads_per_row;          // constexpr
#pragma unroll
for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {              // constexpr trip count
    const int qs0 = get_int_b4(bxi->qs, txi);
```

I = 128, threads_per_row = 8 (Q4_K), warp 32, so nrows = 4, nwarps = 8, and the loop takes
**4 compile-time iterations, already unrolled**. That is **4 outstanding 4-byte loads per
thread = 16 bytes in flight** - exactly what vectorising to one `uint4` would have given.

**So my "fix" was already implemented by nvcc.** I measured my own bug and attributed it to
llama.cpp.

## The corrected measurement

`tmp/prof/unroll.cu`, same structure, same 38 KB smem, same 1 block/SM, only the trip count
changed:

| variant | bytes in flight/thread | GB/s |
|---|---:|---:|
| runtime trip count, 32-bit | 4 (1 load) | **34.3** |
| static trip count, 32-bit, unroll 1 | 16 (4 loads) | **118.1** |
| static trip count, 32-bit, unroll 2 | 32 | **118.1** |

**The moment the trip count becomes compile-time and the loop unrolls, the load reaches the full
118.1 GB/s ceiling at 1 block/SM.** There is no vectorisation gain left, and there is no
load-width problem. The user's read was correct: vectorising here is just unrolling by 4, and
the compiler already does it.

Note the sharpness of the result: 34.3 -> 118.1 is a **3.4x** swing from a
`constexpr` keyword alone. That is exactly the magnitude I had misattributed.

## What this means for the other numbers

The synthetic result that *does* still stand from the earlier work is the pure read: **118.1
GB/s** at grid 8192 / block 256, at any occupancy from 4 blocks/SM down to 1. The hardware is
healthy. What has now been eliminated as the cause, each by measurement:

| candidate | verdict |
|---|---|
| DRAM bandwidth / access pattern | 118.1 GB/s achieved, not the limit |
| occupancy (1 block/SM) | 112.2 -> 118.1 GB/s, not the limit |
| smem + `__syncthreads` structure | ~3% once unrolled |
| load width (`get_int_b4`) | unrolled already; 118.1 GB/s |
| dp4a instruction throughput | see below |
| stream-k fixup kernel | **0.3 ms total, 0.0% of kernel time** |

## dp4a is not saturated either

Measured dp4a peak (`mmqshape.cu` A): **1.239e12 dp4a/s**. Per grid-8192 launch, only the tiles
with `jt*J < col_diff` run the K loop, i.e. 2048 of the 8192 blocks:

| quantity | value |
|---|---:|
| dp4a issued by the working blocks | 1.718e10 |
| time at dp4a peak | 13.9 ms |
| measured | 84.75 ms |
| **dp4a pipe utilisation** | **16%** |

So the kernel is **not** compute bound on dp4a.

There is a subtlety worth stating, because I got it wrong first. If *all* 8192 blocks ran the K
loop the issued work would be 55 ms at peak, which is **65%** of the 84.75 ms - close enough to
the measured time to look like a compute-bound story. But the MoE branch returns early for the
`ntx` tiles with `jt*J >= col_diff`, and with ~10 tokens per expert only the `jt = 0` tile has
work. So the honest figure is 16%, not 65%. The check that mattered was the early-return, not
the peak rate. The accounting does surface a genuine inefficiency though:

| quantity | value |
|---|---:|
| output elements computed (512 experts x 512 x 512) | 1.34e8 |
| output elements that are *useful* (512 tokens x top-10) | 5.12e3 |
| **fraction of issued dp4a that is padding** | **98%** |

Because the routing averages **10 tokens per expert** and the tile width is **J = 128**, each
tile is ~92% empty and `ntx = 4` tiles the output four times over. So the kernel does **50x more
arithmetic than the useful result requires**. That is real waste, but since the pipe sits at 16%
it is not what is setting the clock.

## Where this leaves the investigation

The throughput hypotheses are eliminated **as descriptions of the slow thing**, each by
measurement:

| candidate | verdict |
|---|---|
| DRAM bandwidth / access pattern | 118.1 GB/s achieved in isolation, not the limit |
| occupancy (1 block/SM) | 112.2 -> 118.1 GB/s, not the limit |
| smem + `__syncthreads` structure | ~3% once the loop unrolls |
| load width (`get_int_b4`) | already 4-way unrolled; 118.1 GB/s |
| dp4a instruction throughput | 16% of its measured peak |
| stream-k fixup kernel | **0.3 ms total, 0.0% of kernel time** |

But the reason they were all wrong is not that one of them was *nearly* right. It is that they
were all describing **11.5% of the run**. The kernel sum is 4.94 s inside a 42.9 s wall time, so
the GPU has nothing running for 88.5% of the run. That is the finding, and it is written up in
`THE_WAIT_FOUND.md`. Every number in the table above is a property of the 4.94 s.

This also retroactively explains why the investigation would not converge: I kept profiling the
part that was working.

## Lesson

A synthetic that shares the *shape* of a kernel but not its *compile-time constants* is not a
reproduction. Two of my three "smoking guns" this session came from a constant that the compiler
could fold in the real code but not in my harness.

The second lesson is more expensive and is the real one: **before optimising anything, check what
fraction of wall time it occupies.** One division of two numbers already in the log would have
redirected the entire investigation and saved several long profiler runs. The ratio of kernel
time to wall time should have been the first measurement, not the last.
