# ncu on the MoE kernels: what the GPU does in there

Measured 2026-09-20 with `ncu` 2022.4.1 (which works here, unlike `nsys`, whose importer is
missing). Model `Qwen3-Next-80B-A3B-Instruct-Q4_K_M`, `-ngl 1 -ncmoe 99 -lm none -p 128 -b 128
-ub 128 -r 1 -t 6 -ps 1`.

## 0. A claim I made and must retract

I told you: "at 1024 tokens the quantized MoE takes the fallback path that requires stream
synchronization and disables CUDA graphs." **That is wrong.** Here is the code path, traced:

```cpp
// ggml/src/ggml-cuda/mmq.cu:259
bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
    ...
    if (turing_mma_available(cc)) {
        return true;                      // <-- taken, before any ne11 check
    }
    ...
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }                                     // <-- never reached
```

`turing_mma_available(cc)` is `ggml_cuda_highest_compiled_arch(cc) >= 750`. This build contains a
single cubin, `sm_75` (`flags.make` has `arch=compute_75`), so `highest_compiled_arch(750) = 750`
and the test is true. So:

- `ggml_cuda_should_use_mmq(Q4_K, 750, 1024, n_experts=512)` returns **true**
- `ggml_cuda_mul_mat_id_needs_sync()` therefore returns **false**
- **no stream sync, and CUDA graphs stay enabled**

My "verification" was invalid. I checked for the `disabling CUDA graphs` log line, found none, and
treated that as confirmation. That log is inside `#ifndef NDEBUG`, and this is a release build
(`-O3 -DNDEBUG`), so the line can never print. The absence proved nothing. The correct check is
that the code returns `false`, which it does.

Confirmation the other way: the run performed **104** `CUDA graph warmup complete`, so graphs were
live. The false-claim text was only ever in chat, not in a document, so there is nothing on disk to
correct.

## 1. Where the ncu results are

| path | what it is |
|---|---|
| `tmp/prof/inventory.txt` | the raw 900-launch ncu CSV from the earlier session. This is the source of the 84.75 ms figure, and of the grid table in `GRID_VERIFICATION.md`. The richest file. |
| `results/prefill_streaming_v3/GRID_VERIFICATION.md` | the parse and interpretation of `inventory.txt`: kernel, grid, block, launch count, ms each. Read section 2 for the launch shape. |
| `tmp/prof/ncu/j128.sass` | **new.** full SASS disassembly (17585 lines) of the J=128 Q4_K MMA kernel currently in use. |
| `tmp/prof/ncu/mmq_ngl1.csv` | **new.** a CSV attempt that got truncated by the `timeout`; the metrics below came from the same command run to completion, not from this file. |
| `tmp/archive/prof_logs_pre_retraction/` | older dmon/clock traces. Superseded for timing, still fine for clocks. |

## 2. The kernel geometry, measured

`ncu --replay-mode application --launch-skip 3 --launch-count 1 --kernel-name regex:"mul_mat_q"`

| metric | value |
|---|---|
| kernel | `mul_mat_q<ggml_type 12 (Q4_K), J=128, fallback=false>` |
| grid | **(14, 1, 1)** = one block per SM, 14 SMs |
| block | (32, 8, 1) = 256 threads |
| registers/thread | **255** (the hardware maximum) |
| waves per SM | **1** |
| occupancy | `sm__warps_active` = **25.00%** |
| warp issue rate | `smsp__issue_active` = **4.17%** |
| warp latency | 47.95 cycles per instruction issued |
| dominant stall | `long_scoreboard` = **78.86%** of active warps |

Read together: the kernel launches **one wave** of 14 blocks, each block is register-limited to
**25% occupancy** (8 warps of a possible 32), and those warps spend **79% of their time stalled on
memory** while issuing an instruction only **4% of cycles**. There is not enough occupancy to hide
the latency. This is a latency-bound kernel, not a throughput-bound one.

That matches everything measured before it: `dmon mem` 8-15% (DRAM not saturated), power below a
zero-traffic spin loop, and a per-node GPU time 17% higher at `-ngl 99`.

## 3. Why the grid is 14, and this is the lever you asked about

`ggml/src/ggml-cuda/mmq.cuh:1437`:

```cpp
const int ntiles_dst = ntx * nty * ntzw;
const int tiles_nwaves = (ntiles_dst + nsm - 1) / nsm;
const int tiles_efficiency_percent = 100 * ntiles_dst / (nsm * tiles_nwaves);
const dim3 block_nums_stream_k(GGML_CUDA_CC_IS_NVIDIA(cc) && tiles_efficiency_percent >= 90
                               ? ntiles_dst    // many blocks, no fixup kernel
                               : nsm, 1, 1);   // ONE WAVE, plus a fixup kernel
```

When the tile count does not divide evenly enough into 14 SMs (efficiency < 90%), the launch
collapses to `nsm` = 14 blocks and adds a second `mul_mat_q_stream_k_fixup` kernel to combine the
partial results. One block per SM, one wave, and an extra kernel.

The tile count is `ntx * nty * ntzw`, with `ntx = ceil(ncols/J)`, `nty = ceil(nrows/I)`, and `ntzw`
the number of experts. **This is exactly the "split it earlier" idea, and the `J` in the config
table is the knob that does it.**

## 4. The build picks the tile table for hardware this card may not have

`ggml/src/ggml-cuda/mmq.cuh:246`:

```cpp
if (ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) {
    return ggml_cuda_mmq_get_config_ampere(type, J, fallback);   // <-- taken
}
if (ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_DP4A) {
    return ggml_cuda_mmq_get_config_pascal_dp4a(type, J, fallback);
}
```

The test is against the **highest architecture in the binary**, not the running device. A
`compute_75`-only build makes `highest_compiled_arch(750) = 750 >= VOLTA(700)`, so the **ampere**
tile table is used on a Quadro T1000. The two tables differ exactly where it matters:

| table | Q4_K config | stream_k |
|---|---|---|
| ampere (used) | `nthreads 256, I=128, J=8..128` | **true** |
| pascal-dp4a | `nthreads 256, I=64, J=8..64` | **false** |

The ampere table reflects VOLTA-class tensor hardware and chooses the stream-k schedule (one wave
plus a fixup kernel). The pascal-dp4a table is a plain tiled schedule.

**And the SASS says the compute is not what the table assumes.** `tmp/prof/ncu/j128.sass`:

| instruction | count |
|---|---|
| `HMMA.*` (fp16 tensor core) | **0** |
| `IMMA.8816.S8.S8` (int8) | **1024** |
| `FFMA.FTZ` | 2048 |

The kernel is built from INT8 `IMMA`, which is what `turing_mma_available` promises. ncu reports
`sm__pipe_tensor_op_hmma_cycles_active` = **39.45%**, so *some* tensor-class pipe is active - but
the kernel contains no `HMMA` at all, only `IMMA` plus `FFMA` dequant work.

**Open, and it matters:** whether TU117 (Quadro T1000) has a real tensor pipe for `IMMA`, or
whether the driver runs the `IMMA` on the integer/FP32 datapath. If it is the latter, the build is
selecting a tensor-core schedule and an `I=128, J=128, nthreads 256, 255-register` tile shape for
hardware that cannot run it at rate. That would explain 255 registers, 25% occupancy and a kernel
that is 1.42x slower than the same work split across CPU cores. **Answered in section 7: there is
no tensor pipe, and the penalty is 10x.**

## 5. On `-ub 64`

Agreed, and the numbers make it trivial: 45.86 GiB of weights per ubatch against a 12.4 GB/s
link is ~4 s of copy no matter how the compute is split. A 64-token ubatch divides the compute by
16 while leaving the copy alone, so it cannot win. The tiling idea is **different and compatible**:
keep `-ub 1024`, so one 1024-token act set per layer, and make the *matmul inside that layer* use
smaller `J` tiles so the grid is large instead of collapsing to one wave. Same copy, same 1024
tokens, more blocks and lower register pressure.

## 6. What to measure next, in order

1. **Which launch is the expert matmul.** The profiled launch had ~100 tiles (`grid = nsm` plus a
   fixup kernel). Print the actual `ntx, nty, ntzw, I, J` per MoE launch so the tiles are known
   rather than inferred, and so the `-ngl 1` against `-ngl 99` per-node difference can be tied to a
   specific launch shape.
2. **Does TU117 have a usable IMMA tensor pipe.** DONE, see section 7.
3. **`GGML_CUDA_FORCE_MMQ` / forcing the pascal-dp4a table** to compare `I=64, stream_k=false`
   against the ampere table on identical work. This needs a build flag, so it is a build A/B, not a
   runtime switch.

## 7. TU117 has no tensor pipe, and the MoE uses it anyway - 10x penalty

`tmp/prof/imma.cu` + `tmp/prof/ncu/imma_results.txt`. Runs in seconds, no model load. Operands in
registers, zero memory traffic, so the only thing changing between rows is the instruction. The
IMMA kernel issues the exact asm `mma.cuh` emits on Turing, 4 independent accumulator chains so the
measurement is not latency-bound on the dependency chain.

| path | GMAC/s | MAC/SM/cycle | power |
|---|---:|---:|---:|
| `IMMA.8816.S8.S8` (the m8n8k16 that Turing gets) | **500.9** | **24.6** | **16.0 W** |
| `__dp4a` (IDP, the int8 doing the same math) | 5015.2 | 246.2 | 25.4 W |
| `FFMA` (FP32 pipe reference) | 1252.5 | 61.4 | 24.1 W |
| spin loop, no memory traffic | - | - | 23.4 W |

**The harness is validated, which is what makes the rest credible:** FFMA reaches
**1252.5 GMAC/s = 95.9%** of the T1000's published FP32 peak (2.61 TFLOP/s = 1305 GMAC/s). A test
rig that hits 96% of a known peak is measuring what it claims to. The scaling check prints time
against iteration count and gives a clean 4x slope per 4x iterations for all three rows, so none of
the loops were eliminated.

The reading:

- **`__dp4a` is 10.0x faster than `IMMA` at the same arithmetic.** dp4a does 4 MAC per instruction
  and issues at ~61 instr/SM/cycle. `IMMA.8816` claims 1024 MAC per warp instruction (32 per
  thread) yet lands at 24.6 MAC/SM/cycle, i.e. about **one instruction per 42 cycles per SM**
  against dp4a's 4.2 cycles for the same 1024 MACs. If TU117 had a tensor pipe of even modest rate
  it would beat dp4a outright, since the instruction nominally carries 8x the work. It does not;
  it is 10x slower.
- **`IMMA` is also 2.5x slower than plain FFMA**, which needs ~17 cycles to do 1024 MACs at the
  measured FFMA rate. So `IMMA` is not merely riding the FP32 pipe - it is worse than just
  dequantizing and using FFMA. This is the signature of an emulated instruction on a part without
  the unit.
- **Power confirms it independently.** `IMMA` draws **16.0 W**, *below* a zero-byte spin loop at
  23.4 W and far below dp4a at 25.4 W. The card is barely working while it issues `IMMA`. This is
  the same signature as the unexplained 18.8 W of the real run.

**Why it is selected.** `mma.cuh` and the MMQ config both gate on
`turing_mma_available(cc)`, which is `GGML_CUDA_CC_IS_NVIDIA(cc) && highest_compiled_arch(cc) >= 750`.
The real device reports `cc = 750`, so the gate opens. But **sm_75 spans two different dies**: TU117
(the Quadro T1000) has no tensor cores at all, while TU106/TU116 do. The gate tests the compute
capability, which is identical, and cannot tell them apart. Every MoE matmul in this build is
therefore issued as `IMMA` on a part where `IMMA` is 10x slower than the `__dp4a` it replaced.

This is a candidate single explanation for all three open anomalies at once:

- **The 18.8 W of a 50 W budget** - `IMMA` at 16.0 W is measured to draw less than a spin loop.
- **`-ngl 99` being slower than `-ngl 1`** - the per-node GPU time is 17% higher when the same
  weights are resident, because the resident path runs more `IMMA` per unit work.
- **The `-ngl 1` against CPU verdict** - the user's observation that CPU cores beat the GPU per
  millisecond is exactly what a 10x `IMMA` penalty predicts.

Not yet established: what fraction of the real kernel's time is `IMMA`. The J=128 kernel contains
1024 `IMMA` and 2048 `FFMA`, so `IMMA` is on the critical path, but the split has not been measured.
Also not established: whether an upstream fix already exists for TU117 detection. Per the project
rules I am not writing that change; it is reported here for the maintainer to judge.
