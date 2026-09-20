# Task list: from "the GPU looks 100% busy" to "here is the pipe that is empty"

Date: 2026-09-20. Status: **THE WAIT IS FOUND - see `THE_WAIT_FOUND.md`.** The GPU is idle
**88.5%** of the wall time (4.94 s of kernels inside a 42.9 s run). Everything below about
kernel internals describes the other 11.5%.

---

## RESOLVED: the wait exists, and it is 88.5%, and it is outside the kernels

Two numbers from **one** run in `tmp/prof/inventory.txt`, so directly comparable:

| quantity | value |
|---|---:|
| pp512 throughput | 11.94 t/s |
| wall time for one 512-token prefill | **42.9 s** |
| sum of all 900 kernels | **4.94 s** |
| **GPU idle** | **38.0 s = 88.5%** |

The standing concern - "there may be a simple wait somewhere consuming 70-80% of the compute
time" - is **confirmed**. It is not inside `mul_mat_q`; it is time when no kernel runs at all.

This also reconciles the earlier `utilization.gpu` = 100% reading: that metric is satisfied by a
launched-but-stalled grid, so it cannot distinguish work from waiting. **The kernel-sum-over-wall
ratio cannot be fooled that way and should have been the first measurement.**

Caveat: this run used `-ngl 1 -ncmoe 99`, which is **not** the configuration behind the 64.5 t/s
figures. Re-measure on a config with known throughput before generalising.

---

## T2.0 - find what the GPU waits for during the 88.5% (the live work)
- [ ] T2.0.1 **measure the copy cost per pass.** `LLAMA_PREFILL_STREAMING_TIMECOPY=1` is already
      implemented. Arithmetic to check against: 48 layers x 1022 MiB = 48 GiB per ubatch, which
      is **4.07 s at 11.8 GB/s**. Kernels are 4.94 s. If copy and kernels were fully serialized
      that is 9.0 s of 42.9 s; the remaining 34 s is the real question.
- [ ] T2.0.2 **is the copy actually running at 11.8 GB/s here?** If it degrades to ~1.5 GB/s in
      this configuration, 48 GiB becomes 32 s and the whole 42.9 s is explained. One measurement,
      already gated by env var.
- [ ] T2.0.3 **count host-side syncs per pass** (`ggml_backend_event_wait`,
      `cudaStreamWaitEvent`, `cudaStreamSynchronize`). Free grep + count.
- [ ] T2.0.4 **re-run the inventory on a known-throughput config** so the 88.5% can be trusted
      as a general property rather than a property of `-ngl 1 -ncmoe 99`.
- [ ] T2.0.5 decide whether the copy can overlap the compute at all. The 3x interference law
      says concurrent copy costs 3x, so the answer may be that serialization is correct and the
      copy simply has to be made faster - see `FINDINGS_the_cliff.md`.

---

## The parameters worth extracting, and why each one

Grouped by the question each answers. "Cheap" = seconds (synthetic) or one metric pass;
"expensive" = one full model load (~20 min per metric on this box).

### A. Is the kernel actually busy, or just resident?

| parameter | what it tells us | cost |
|---|---|---|
| `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` | fraction of cycles the SM is doing anything at all | cheap |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | **issue slots used** - the honest "is it issuing" number | cheap |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | which of the SM sub-pipes is the busiest | cheap |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | warps resident and not finished | cheap |
| `smsp__warps_eligible.avg.per_cycle_active` | **warps ready to issue** - if this is ~0 while warps_active is high, the SMs are parked | cheap |

This group separates "the GPU is working" from "the GPU is holding warps that cannot proceed".
`utilization.gpu` = 100% is compatible with both; `issue_active` and `warps_eligible` are not.

### B. How much runs in parallel (the question about 1-2 SMs)

| parameter | what it tells us | cost |
|---|---|---|
| `launch__grid_size`, `launch__block_size` | the shape actually launched | free |
| `launch__waves_per_multiprocessor` | how many sequential waves of blocks there are | cheap |
| `launch__occupancy_limit_shared_mem` | blocks/SM allowed by smem | cheap |
| `launch__occupancy_limit_registers` | ................. by registers | cheap |
| `launch__occupancy_limit_warps` | ................. by the 1024 threads/SM cap | cheap |
| `launch__occupancy_limit_blocks` | ................. by the per-SM block cap | cheap |
| `sm__maximum_warps_per_active_cycle_pct` | the theoretical ceiling on this device | cheap |
| `sm__ctas_active.avg` | **blocks actually resident per SM over time** | cheap |

`sm__ctas_active.avg` is the direct answer to "are only 1-2 SMs running?" - it is measured, not
inferred from grid size. The occupancy-limit set identifies *what* caps it, with the arithmetic
already cross-checked: 48 KB smem/block and 64 KB/SM gives 1 block, 8 warps, 256 threads.

### C. Where the time goes: the stall breakdown

This is the money group. One pass, all of them, because each costs a model load.

| parameter | what it tells us | cost |
|---|---|---|
| `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` | waiting on **memory** (global/local) | expensive |
| `smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct` | waiting on **shared memory / MIO** | expensive |
| `smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct` | the FMA/ALU pipe is full | expensive |
| `smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct` | the MIO queue is full | expensive |
| `smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct` | the LSU queue is full | expensive |
| `smsp__warp_issue_stalled_barrier_per_warp_active.pct` | **`__syncthreads()`** | expensive |
| `smsp__warp_issue_stalled_membar_per_warp_active.pct` | memory fences | expensive |
| `smsp__warp_issue_stalled_wait_per_warp_active.pct` | fixed-latency dependency | expensive |
| `smsp__warp_issue_stalled_not_selected_per_warp_active.pct` | runnable but not picked (fine-grained contention) | expensive |
| `smsp__warp_issue_stalled_drain_per_warp_active.pct` | runnable, waiting to drain before exit | expensive |
| `smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct` | branch overhead | expensive |
| `smsp__warp_issue_stalled_no_instruction_per_warp_active.pct` | instruction fetch | expensive |
| `smsp__warp_issue_stalled_selected_per_warp_active.pct` | issuing (the good one) | expensive |

**Pre-registered reading** (decided in advance so the result is conclusive, not debatable):

| dominant stall | conclusion | fix direction |
|---|---|---|
| `barrier` | the load/sync/compute phases serialise, and at 1 block/SM nothing fills the gaps | software pipeline: double-buffer the smem tile (load tile K+1 while computing tile K) |
| `long_scoreboard` | global load latency is exposed | raise occupancy, or prefetch, or smaller K step |
| `short_scoreboard` / `mio_throttle` | shared memory path bound | reduce smem traffic, or change the tile layout |
| `math_pipe_throttle` | actually compute bound and 15% is the real answer | stop looking for a memory fix |
| `not_selected` / `wait` | fine-grained dependency chains | more ILP per thread |

### D. Traffic and instruction accounting (validates the counters)

| parameter | what it tells us | cost |
|---|---|---|
| `dram__bytes_read.sum` | must be >= 996 MB; if it is ~220 MB as before, the counter is broken | cheap |
| `dram__bytes_read.sum.per_second` | measured bandwidth, to cross-check the 11.7 GB/s | cheap |
| `lts__t_sector_hit_rate.pct` | L2 hit rate | cheap |
| `sm__inst_executed.sum` | total instructions issued | cheap |
| `smsp__inst_executed.sum`, `sm__sass_thread_inst_executed_op_*` | instruction mix: how much is dp4a vs load vs address math | cheap |
| `sm__sass_thread_inst_executed_op_ffma_pred_on.sum` | actual FP work done | cheap |

This group exists to make the earlier contradiction impossible to repeat: any counter that
disagrees with a known size (996 MiB) gets caught here rather than being written up as fact.

---

## Task list

Statuses: `[ ]` todo, `[x]` done, `[~]` in progress.

### T0 - discussion (blocking)
- [~] Agree the parameter set above and the pre-registered stall readings
- [ ] Decide whether to invest in the structural kernel (T3) before or after T1

### T1 - fast synthetic tests (seconds each, iterate freely)
- [x] T1.1 measure pure read bandwidth at the expert kernel's grid/block (118.1 GB/s)
- [x] T1.2 measure whether access pattern matters (no: strided == chunked)
- [x] T1.3 sweep occupancy via smem from 4 blocks/SM down to 1 (112.2 GB/s pure read, -5%)
- [x] T1.4 check source for spin/poll constructs (none; the one `cudaLaunchHostFunc` is `#if 0`)
- [x] T1.5 arithmetic intensity sweep (`intense.cu`): bandwidth holds to 18-20 flop/byte, the
      ridge. At the real 30 flop/byte the hardware still gives 81 GB/s + 2.44 TFLOP/s. The real
      kernel gets 12.3 GB/s + 0.36 TFLOP/s -> **6x below what its own arithmetic allows**.
- [x] T1.6 dp4a throughput (`mmqshape.cu` A): 91-95% of lane rate, at 1 block/SM too. The
      integer path is fast. **Ruled out.**
- [x] T1.7 read+FMA occupancy sweep (`occ2.cu`): 1 block/SM costs 1.29x on the read
      (118 -> 91.8 GB/s). Pure FMA still reaches 94.4% of peak at 1 block/SM.
- [x] T1.8 smem + `__syncthreads` structure (`mmqshape.cu` B): the structure costs **1.44x**
      (114.7 at 2 blocks/SM -> 79.6 at 1 block/SM). The syncs themselves cost only 3%.
- [ ] T1.9 **the remaining ~6x.** T1.5-T1.8 together account for ~1.5x of the ~9.6x. Something
      else is dominant and has not been reproduced synthetically. Next candidates below.

### T1.11 - RESOLVED: the launch shape is now derived from source, not assumed
- [x] T1.11.1 re-parsed `tmp/prof/inventory.txt` from the header (15 columns, not 14; metric
      values carry thousands separators). **84.75 ms confirmed**, 32 launches, 2711.9 ms total.
- [x] T1.11.2 kernel is `mul_mat_q<Q4_K, J=128, fallback=false>`. Dispatch at `mmq.cuh:229`
      keys off `ggml_cuda_highest_compiled_arch`, so sm_75 uses the **ampere** config table
      (I=128, J=128, nthreads=256, **stream_k=true**), not the pascal-dp4a table.
- [x] T1.11.3 grid reproduced from source: nty 4 x ntx 4 x ntzw 512 = 8192 tiles, efficiency
      99% >= 90% so `block_nums_stream_k = ntiles_dst` = **(8192, 1, 1)**, block (32,8,1).
- [x] T1.11.4 `use_mma_data_layout()` is true via `turing_mma_available(7.5)`, so smem is the MMA
      layout. **CORRECTED:** `sram_stride` is **76**, not 136. `QI8_1 = QK8_1/(4*QR8_1) = 32/4 = 8`,
      so `sram_stride = 2*32 + 2*32/8 + 4 = 76`. Full size is
      `mmq_get_nbytes_shared` = `J*4 (ids) + I*76*4 (x) + PAD(J*36*4, nthreads*4) (y)`
      = 512 + 38912 + 18432 = **57856 B = 56.5 KB**, above the 48 KB static cap - which is why a
      dynamic allocation is needed at all. 2 blocks/SM would need <= 32 KB, so this is a hard
      1-block/SM cap. Registers cap it too: **REG:255** x 256 threads = 65280 of 65536 per SM.

### T1.12 - NEW FINDING: token tiling duplicates the weight read (fixable)
- [x] T1.12.1 `ntx.cu`: grid `(ntx, 4, 512)`, each block reads its own 128-row slice, so the tx
      blocks genuinely duplicate. L2 absorbs **~60%**; the rest leaks to DRAM.
- [x] T1.12.2 at ntx = 1 the kernel hits **117.8 GB/s** = the DRAM ceiling. At ntx = 4, DRAM
      traffic is **1.60x** what it needs to be (823 MiB vs 512 MiB) and the time grows by the
      same 1.60x. At ntx = 8 the leak saturates at ~61%.
- [x] T1.12.3 so every expert weight byte is requested **4x** at ub=512 (I = J = 128), for a
      **1.6x DRAM amplification**. Fixable by a *larger* token tile - the opposite of the
      token-tiling idea, which would worsen it.
- [x] T1.12.4 corrected implied DRAM rate for the real kernel: 6.3 GB/s -> **10.1 GB/s** with the
      amplification accounted for. Still only 8.6% of the 118 GB/s ceiling, and the synthetic at
      the *identical grid and identical bytes* runs in **7.3 ms vs 84.75 ms = 11.6x**.

### T0.2 - NEW BLOCKING ITEM: the 84.75 ms is not yet independently corroborated
- [ ] T0.2.1 arithmetic check: 144 grid-8192 launches per prefill (48 MoE layers x 3 projections)
      at 84.75 ms = **12.2 s**, but pp512 streaming takes **7.94 s** total. These cannot both be
      right. The ncu window held only 32 of the 144 launches (22%), so replay inflation is live.
- [ ] T0.2.2 resolve with **no profiler**: one `llama-bench -b 512 -ub 512` with
      `LLAMA_PREFILL_STREAMING_DEBUG=1 -v`, read the per-layer host timings the streaming code
      already logs. One model load. **Do this before T1.10** - the stall counters describe a
      kernel whose duration is not yet trustworthy.

### T2.1 - the "wait inside the kernel" branch (REPROMOTED - the idle theory was wrong)
- [x] T2.1.1 **W1 barrier *concept* checked.** The stream-k fixup kernel - the one cross-block
      barrier - is **0.3 ms total, 0.0%** of kernel time. Not the wait.
- [x] T2.1.2 **W2 idle fraction. RETRACTED.** `THE_WAIT_FOUND.md` claimed 88.5% idle from a run
      whose own table says 11.94 t/s - a config that normally gives 63 t/s. See
      `CORRECTION_gpu_idle.md`. Measured at the operating point: `utilization.gpu` **99-100%**,
      `dmon sm` **100%**, DRAM **8-15%**, power **18.4 W** of 50 W. The GPU is occupied, not idle.
- [x] T2.1.3 the four ncu logs used to reinforce 88.5% all gave different answers (88.5, 82.8,
      67.6, 98.1%) because none of them had a clean compute-only window. `layer_cycle.txt`
      reported 98.1% idle at **102.28 t/s** - the fastest and the most idle at once. Disregard.
- [x] T2.1.4 **the concern is still open, and now better supported than before:** 100% occupancy
      at 18.4 W with DRAM at 8-15% is the signature of warps resident and stalled on memory
      inside a launched grid. That is what the original "70-80% wait" intuition describes.
- [ ] T2.1.5 **the stall batch (group C) is now the primary instrument again.** With the idle
      theory removed, per-kernel stall reasons are the direct way to test T2.1.4.

### T2.4 - NEW: reference configuration and comparability
- [x] T2.4.1 pin the reference config. Every log in this investigation used different arguments,
      which is why four logs gave four different "idle" figures. Reference:
      `-ngl 1 -ncmoe 99 -lm none -n 0 -r 4 -p 1024 -b 1024 -ub 1024 --cache-type-k q4_0
      --cache-type-v q4_0 -t 6 -ps 1` -> **84.53 t/s**, reproduces the user's 84.62.
- [x] T2.4.2 measured flag sensitivity, one flag at a time:
      `-p512 -b512 -ub512` **63.25** | with q4_0 KV **63.23** | `-p1024 -ub512` **32.30** |
      `-p1024 -b1024 -ub1024` **82.34** | with q4_0 KV **84.53**.
      So the KV quantisation is **not** the variable (< 2%); the **ubatch size** is, and a
      ubatch smaller than the prefill chunk is pathological (2.6x loss).
- [ ] T2.4.3 do not change more than one flag between comparable runs. `--cache-type-*` had
      started being dropped from commands; restore it for comparability.

### T2.5 - RESOLVED: the 48 vs 94 refill "discrepancy" was my misreading
- [x] T2.5.1 a refill is **per layer**, not per tensor: one call moves the whole 1022 MiB layer.
      One pass is **48 refills / 48 splits** for 48 layers. Nothing is missing.
- [x] T2.5.2 the earlier "94 refills / 96 layer splits" was a **log bug**: the counters are reset
      and the slots re-seeded only on the new-graph path, so a reused-graph line printed the
      previous ubatch's accumulated counts. Never trust a refill count until the reset covers
      every ubatch.
- [ ] T2.5.3 SEPARATE, REAL BUG (correctness, not performance): because the reused path skips the
      reset, slots were never re-seeded, so layers 0/1 could read whatever the previous ubatch
      left behind. Was already noted; keep it on the list.

### T2.2 - the real control for the copy
- [ ] T2.2.1 patch `prefill_streaming_fill_slot` to refill from a VRAM-resident copy
      (host -> VRAM once, then VRAM -> VRAM). This is the only valid control; NOCOPY is not
      (it feeds zeros and changes what the kernel does). Needed for W4 and for Phase 3.

### T2.3 - the padding waste (newly quantified, separate from the slowdown)
- [x] T2.3.1 measured: the kernel issues **50x more dp4a than the useful result needs** - 98% of
      the arithmetic is padding, because routing averages **10 tokens per expert** while the
      tile width is **J = 128**. Real waste, and it explains the low power/compute utilisation at
      the *work* level even though the pipe itself is only 16% busy.
- [ ] T2.3.2 worth a separate look once the wait is found: J is chosen for throughput, not for
      routing sparsity. Not the current bottleneck (pipe at 16%), but it caps any future gain.

### T1.13 / T1.14 - CLOSED. No code change needed.
`SMOKING_GUN.md` sections 2-4 and 6 are withdrawn. `mmqstruct.cu` passed the tile size as a
**runtime** argument, so its load loop could not unroll and each thread had one load in flight.
That is why it measured 34.2 GB/s. `tmp/prof/unroll.cu` shows the same structure with a
**compile-time** trip count reaches **118.1 GB/s at 1 block/SM** - the full ceiling.
The real kernel's loop is already `#pragma unroll` over a `constexpr` trip count, so
**vectorising the load is worthless** and was never the fix. Retained only as a record.

- [x] T1.13.5 DISPROVED: vectorising `get_int_b4` gains nothing; it is already 4-way unrolled.
- [x] T1.13.6 the residual was never 2.8x; the baseline it was measured against was invalid.
- [x] T1.14.1 **SASS audit confirms the real code has no such flaw.** Disassembled
      `mul_mat_q<GGML_TYPE_Q4_K, 128, false>` from `build/bin/libggml-cuda.so.0`. The
      weight-load loop is **gone**: no backward branch, 7944 straight-line instructions.
      The `#pragma unroll` was fully honoured.
- [x] T1.14.2 **18 independent 32-bit loads issued back to back**, median load-use distance
      **145 instructions** (114 of 119 loads have distance >= 50). That is ~456 bytes of
      outstanding load per thread. Compare the harness, which had 4 bytes. The 18-load
      groups are the **y-tile loader** (`J*MMQ_TILE_Y_K/nthreads` = 4608/256 = 18 trips);
      the qs loop contributes a further 16 x 4 bytes. Both fully unrolled.
- [x] T1.14.3 **Register count is 255**, not ~128: 255 x 256 = 65280 of 65536 registers/SM,
      so the register file is *also* a 1-block/SM cap alongside the 56.5 KB of shared
      memory. It is a consequence of exposing 18 outstanding loads, not a separate defect.
- [x] T1.14.4 **Zero predicated loads**: `fallback` really is false, so the `min(i, i_max)`
      compiled away. No masked-lane overhead.
- [x] T1.14.5 no benchmark run. Nothing actionable was found, and the user's own caveat
      applies: inside an 88.5% idle run, 64 vs 86 t/s changes nothing.
- [x] T1.14.6 written up in `UNROLL_AUDIT.md`. Side note recorded there but **not claimed**:
      loads are emitted as `LDG.E.CONSTANT.SYS`; `.SYS` scope is wider than `.GPU`. Unmeasured.

### The remaining gap - candidate list in priority order

Now that occupancy (1.29x), the smem+sync structure (1.44x), dp4a (ruled out) and the FP pipe
(ruled out) are all measured, the ~6x remainder needs a mechanism that is **not** any of these.
Ranked by what would produce 100% `utilization.gpu`, 10% `utilization.memory`, 18.6 W and no
traffic, while issuing almost nothing:

- [ ] T1.9.1 **`stream_k` split-K fixup.** grid.x = 1274 = 7 x 182 and the kernel runs a
      `while(true)` loop reading partial sums and a `mul_mat_q_stream_k_fixup` pass. If that
      path spins or serialises across 23 waves at 1 block/SM, it would cost exactly this shape.
      Testable: build a synthetic with an equivalent cross-block reduction barrier.
- [ ] T1.9.2 **dependency stalls in the real inner loop.** 8 warps/SM with 4-cycle dp4a latency
      needs 32+ independent operations in flight per warp. If MMQ's inner loop has a short
      chain, issue collapses. Testable: a dp4a chain with a *short* dependency graph, rather
      than the 8 independent accumulators T1.6 used.
- [ ] T1.9.3 **the launch itself is not what `PHASE5_RESULTS.md` says.** Given how many numbers
      in this investigation have turned out to be parse artefacts, re-derive the 84.75 ms from
      the raw ncu CSV before building any more kernels on top of it. **Cheapest check, do first.**
- [ ] T1.9.4 **the weights are read more than once per launch.** If the kernel re-reads the
      tile rather than caching it, the effective bytes are a multiple of 996 MiB and the rate
      per launch is fine. Check `dram__bytes_read.sum` against 996 MiB (this was previously
      contradicted by a 220 MB reading, which was catalogued as a broken counter - re-verify).
- [ ] T1.9.5 **L2/TLB behaviour at the real access pattern.** The synthetic walks a dense
      996 MiB with perfect locality; MMQ's expert grid may stride far more.

### T1.10 - direct measurement instead of more inference (decide with the user)
- [ ] T1.10 one `ncu` batch on the grid-8192 kernel with the group C stall counters. This
      ends the guessing: it says which pipe or barrier the warps are parked on. ~3 h of wall
      time, one model load, cannot be iterated. Feeble cost/benefit versus T1.9.3, which is
      minutes, so **do T1.9.3 first**.

### T2 - llama.cpp profiling (one model load per metric batch, so batch aggressively)
- [ ] T2.1 group C stall breakdown on `mul_mat_q` grid 8192, single invocation (~3 h, 13 metrics)
- [ ] T2.2 group A + B in the same invocation where the pass budget allows
- [ ] T2.3 group D counter validation against the known 996 MiB
- [ ] T2.4 repeat on the dense Q5_K grid-256 kernel as the healthy control (76% throughput)

### T3 - decide and act
- [ ] T3.1 pick the fix direction from the pre-registered table in C
- [ ] T3.2 if it is the barrier phase: prototype smem double buffering in the structural kernel
      (T1.5) first, then port to `mmq.cuh`
- [ ] T3.3 validate against llama.cpp: predicted vs actual launch time

### T4 - housekeeping
- [x] T4.1 record the calibration numbers (`CALIBRATION.md`)
- [x] T4.2 retract the invalid token-tiling experiment (`TOKEN_TILING.md` section 0)
- [ ] T4.3 keep `MEASUREMENT_STRATEGY.md` section 3 in sync with what was actually run

---

## Why T1.5 before T2 (the cost argument)

T2 costs ~3 hours of wall time for one batch and cannot be iterated: changing anything means
another model load. T1.5 costs ~1 hour of writing and runs in seconds, so it can be swept over
dozens of variants. If T1.5 reproduces the 11.7 GB/s, the mechanism is identified without a
single profiler run, and T2 becomes a confirmation with a known answer rather than an
exploration.

If T1.5 does *not* reproduce it, that is also decisive: it moves the suspect from the
kernel structure to the dp4a math or the stream_k fixup, and T2 then knows exactly which two
stalls to look for.
