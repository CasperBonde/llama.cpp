# Unroll audit: does llama.cpp have my harness's runtime-loop-bound flaw?

Short answer: **no.** The real kernel compiles the weight-load loop to straight-line
code with deep memory-level parallelism. The 34.2 GB/s number in `mmqstruct.cu` was
my harness bug, not a property of llama.cpp. No code change is warranted.

This closes T1.13/T1.14.

## Why this was even a question

`tmp/prof/mmqstruct.cu` reproduced the MMQ tile structure and measured 34.2 GB/s at
1 block/SM, versus 118.1 GB/s for a plain copy. That looked like a smoking gun. But
`unroll.cu` then showed that the *only* thing that matters is whether the trip count
is compile-time:

| variant (38 KB smem, 1 block/SM) | GB/s |
|---|---:|
| A: runtime trip count, 32-bit, 1 load in flight | 34.3 |
| B: `constexpr` trip count, 32-bit, unroll 1 | 118.1 |
| B: `constexpr` trip count, 32-bit, unroll 2 | 118.1 |

In `mmqstruct.cu` the bound was a function argument:

```cpp
for (int i = threadIdx.x; i < tile_ints; i += blockDim.x) { tile[i] = src_it[i]; }
```

`tile_ints` is a runtime value, so nvcc cannot unroll, and each thread has exactly
one outstanding load. The real kernel's equivalent loop
(`ggml_cuda_mmq_load_tiles_q4_K`, `mmq-load-tiles.cuh:732`) is:

```cpp
#pragma unroll
for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
```

with `I`, `nrows`, `nwarps` all `constexpr`. So the question was whether the compiler
really did fold and unroll it. Source reading says yes; SASS settles it.

## Source audit

Every loop in `ggml/src/ggml-cuda/mmq-load-tiles.cuh` has a `constexpr` bound and a
`#pragma unroll`:

| variable | value for Q4_K, J=128, fallback=false | origin |
|---|---|---|
| `I` | 128 | `ggml_cuda_mmq_get_I()` constexpr |
| `nwarps` | 8 | `nthreads/warp_size`, both constexpr |
| `threads_per_row` | 32 | `MMQ_ITER_K/(4*QR4_K)` = `256/(4*2)` |
| `nrows` | **1** | `warp_size/threads_per_row` = `32/32` |
| `nthreads` | 256 | config table |
| `sram_stride` | 76 | constexpr |

So the qs loop trips `I/(nrows*nwarps)` = `128/8` = **16 times**, one 32-bit
`get_int_b4` per trip, 64 bytes per thread.

The y-tile loader in `mmq.cuh:908` is separate:
`for (l0 = 0; l0 < J*MMQ_TILE_Y_K; l0 += nwarps*warp_size)` with
`J*MMQ_TILE_Y_K` = `128*36` = 4608, so `4608/256` = **18 trips**.

`git status --porcelain ggml/src/ggml-cuda/` is clean, so this is unmodified upstream
code.

Variant resolution for the observed kernel (`mmq.cu:180`+, MoE path):
`ne01` is 512 for `ffn_gate_exps`/`ffn_up_exps` and 2048 for `ffn_down_exps`. Both are
divisible by 128, so `fallback = false` and `J = 128`. The executed symbol is
`_Z9mul_mat_qIL9ggml_type12ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf5uint3iiiiiS8_S8_iiiS8_S8_iiiS8_`.

## SASS evidence

Extracted with `cuobjdump -xelf all build/bin/libggml-cuda.so.0`, kernel disassembled
with `nvdisasm -c` (188 cubins; this kernel is in cubin 117). Raw text in
`tmp/prof/q4k_j128_body.sass`.

### There is no load loop in the SASS

The kernel body is 7944 instructions with 17 branches and only **one** backward
branch other than an `@!P0` re-entry guard and a trap loop. The weight-load region
has no loop at all. The `#pragma unroll` was honoured, fully.

For a loop of `I/(nrows*nwarps)` static trips the compiler emitted straight-line
code. That is exactly variant B in the table above.

### Loads are long runs of independent 32-bit accesses

119 `LDG`, all of them `LDG.E.CONSTANT.SYS`, i.e. one 32-bit word each. No
`LDG.E.64` and no `LDG.E.128`. Consecutive-load run lengths:

| run length | count | attributed to |
|---:|---:|---|
| 18 | 1 | **y-tile loader** (`J*MMQ_TILE_Y_K/nthreads` = 4608/256) |
| 11 + 7 | 1 each | a second y-tile load, split by the scheduler |
| 5 | 1 | third y-tile load |
| 2 | 13 | qs loads, paired by the scheduler |
| 1 | 48 | qs loads and metadata |

All four long runs share the signature `[Rbase+0x0]`, `[Rbase+0x400]`, ... striding
by exactly `nthreads*4` = 1024 bytes. That is the y-tile loop above, fully unrolled:
**18 independent loads issued back to back, all compiled before any consumer runs.**

The 18-trip y load is the dominant source of memory-level parallelism in this kernel.
The qs loop contributes 16 x 4 bytes = 64 bytes per thread, which is real but secondary.

### The loads have very long use-distances

Distance in instructions from each `LDG` to the first instruction that reads its
destination register:

| statistic | value |
|---|---:|
| median | **145** |
| p25 / p75 | 100 / 170 |
| mean | 133.4 |
| loads with distance >= 50 | **114 of 119** |

That is 114 loads in flight at 4 bytes each, about **456 bytes of outstanding load
per thread**. Most of that is the unrolled y-tile loop. On a memory system that
needs roughly 1 us of latency covered at 128 GB/s this is far more MLP than is
required. Compare variant A, which had 4 bytes of MLP per thread.

This is also the key contrast with the harness. `mmqstruct.cu` reproduced the *tile
loading and syncthreads structure* but not the y-loader, and passed the trip count
as a runtime argument. So it had one 4-byte load in flight where the real kernel
has 18 in one loop plus 16 in another.

### Register count is the consequence, not a separate problem

```
Function _Z9mul_mat_qIL9ggml_type12ELi128ELb0EEv...:
  REG:255 STACK:48 SHARED:0 LOCAL:0 CONSTANT[0]:524
```

Two corrections to earlier notes in this series:

1. **REG:255**, not ~128. 255 x 256 threads = 65280 of the 65536 registers per SM.
   Two blocks would need 130560, so the register file is *also* a hard 1-block/SM
   cap, not just the 56.5 KB of shared memory.
2. `SHARED:0` here because this kernel takes its shared memory dynamically
   (`extern __shared__`), so the static figure is expected to be zero. The 56.5 KB
   derived earlier stands.

The 255 registers are not an independent defect. They are what you get when the
compiler exposes 18 outstanding loads: each one needs its own destination register.
The compiler traded occupancy for MLP deliberately, and at 1 block/SM the MLP is
still ample.

### `fallback` really is false

Predicated `LDG` count: **0**. The `if (fallback) i = min(i, i_max)` in the source
compiled away entirely, consistent with `ne01 % 128 == 0`. Only 2 `IMNMX` remain in
the whole body. So there is no masked-out-lane overhead either.

## What this means

The load-width / unrolling line of investigation is closed:

| claim | verdict |
|---|---|
| llama.cpp has a runtime loop bound in the weight load | **false** - all bounds `constexpr`, no loop in SASS |
| llama.cpp has only 1 load in flight per thread | **false** - 18 independent loads in one unrolled loop, median use-distance 145 |
| 32-bit loads are the reason for 12.3 GB/s | **false** - 32-bit loads reach 118.1 GB/s once unrolled |
| narrowing to `LDG.E.128` would help | **no** - 118.1 GB/s already achieved with 32-bit loads |

Also worth recording explicitly: the user's objection is what found this.
"Isn't that just unrolling x4?" was correct, and the answer was already yes in the
shipped code - in fact 18x in the y loader.

## What is NOT established

The audit only says the load side is not the problem. It does **not** explain why
this kernel still achieves only 12.32 GB/s of DRAM while a pure-read kernel reaches
118 GB/s on the same device.

The shape of the kernel body is suggestive here, so it is worth stating plainly:
only 4 backward branches exist, of which one is a trap loop and one a re-entry
guard. The main K loop (`mmq.cuh:899`) is `kb0` from `0` to
`ggml_cuda_mmq_get_K_vram(...)` in steps of `blocks_per_iter`. For Q4_K, `qk` is
256 and `ITER_K` is chosen per config, so this is a short loop of a handful of
iterations, each containing 4 `__syncthreads()` and only 18+16 loads of work.

That is a *waiting* problem: too little work per barrier to amortise the pipeline
fill on 8 warps at 1 block/SM. It belongs to the T2.0 workstream, not here.

One unverified observation, recorded so it is not lost: the loads are emitted as
`LDG.E.CONSTANT.SYS`. The `.SYS` scope is broader than `.GPU`. Whether this costs
anything has not been measured and is not claimed.

## Reproduction

```sh
cd /home/cbonde/dev/llama.cpp
cuobjdump -xelf all build/bin/libggml-cuda.so.0     # -> 188 cubins
cuobjdump -symbols <cubin> | grep _Z9mul_mat_qIL9ggml_type12ELi128ELb0E
nvdisasm -c <cubin> > tmp/prof/q4k_j128_full.sass
```

The kernel is in cubin `tmpxft_0001311d_00000000-117.sm_75.cubin`, section starting
at line 23579 of the full dump. `tmp/prof/q4k_j128_body.sass` is the extracted
function body.
