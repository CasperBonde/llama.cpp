# Measurement strategy: is the GPU saturated, or is something else the wall?

Date: 2026-09-19. Model Qwen3-Next-80B-A3B-Instruct-Q4_K_M, `-ngl 1 -ncmoe 99 -lm none`,
GPU Quadro T1000 (TU117): 14 SMs, 896 FP32 cores @ 1.455 GHz, no tensor cores,
L2 = 1024 KB, **BAR1 = 256 MiB**, VRAM 3716 MiB.

---

## 0. What this investigation already established (do not re-derive)

Read from the actual gguf tensor table and the runtime logs, all trustworthy:

| fact | value | source |
|---|---:|---|
| expert tensors per MoE layer | 3 (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) | gguf |
| expert tensor sizes per layer | 288 + 288 + 420 = **996 MiB** | gguf |
| total expert bytes | **43.59 GiB** over 48 layers | gguf |
| slot size, p512 config | **1022.41 MiB** | `prefill_streaming_init` |
| slot tensors, p512 config | **840** (17.5 per layer) | `prefill_streaming_init` |
| slot = experts + the rest of the layer | 996 + 26.4 = 1022.4 MiB | matches GDN layers |
| graph operands rewritten | **840** | `prefill_streaming_rewrite_graph` |
| placement check | **840 ops read slot tensors, 0 on CPU, slot device CUDA0** | `check_placement` |
| ops still reading original host tensors | **0 (0 MiB)** | `check_placement` |
| build with CUDA graphs | **not used** - ncu cannot see kernels through the graphs | `build-nograph` |
| slots zero-filled at init | yes, `ggml_backend_buffer_clear(slot.buffer, 0)` | `llama-context.cpp:152` |
| copy rate measured | 2044.82 MiB in 173.54 ms = **11.8 GB/s** | seeding log |

### Answer to "are you 100% sure the graph reads the weights from VRAM and not elsewhere?"

**At p512 the code is doing the right thing, and it is verified at runtime.** The slot holds
1022.41 MiB per layer, which is exactly the 996 MiB of experts plus the 26.4 MiB of remaining
layer weights. All 840 operands are rewritten. The placement check confirms 840 ops read slot
tensors, zero read the original host tensors, and zero are scheduled on the CPU.

Caveat worth keeping: older logs on disk show `2 slots of 373.50 MiB, 144 tensors`. That is a
different, older tensor set (144 = 48 x 3, experts only) and was NOT the configuration used for
the p512 streaming numbers. It should not be compared against the p512 results.

So the "weights are secretly read over PCIe directly" hypothesis is **disproved for the steady
state**: the weights land in VRAM and the ops that read them are placed on CUDA0.

---

## 1. What is still not trustworthy, and why that changes the plan

The counters produced numbers that are physically impossible, so they cannot be used as the
foundation of a conclusion.

**Contradiction 1: the expert kernel does not read the bytes it must.**

From the streaming-path sweep, the grid-8192 kernel:

- weights it must consume: **996 MiB**
- `dram__bytes_read.sum`: **220 MB** (80 samples, ~69 ms each)
- summed over all 400 sampled matmuls in a pass: **10.24 GB total**

10.24 GB cannot cover 43.59 GiB of experts, let alone the copy write. Whatever these counters
are counting, it is not the tensor traffic.

**Contradiction 2: NOCOPY is not a control.**

Under `LLAMA_PREFILL_STREAMING_NOCOPY=1` the same kernel runs 11.3 ms and reads 55.7 MB, versus
69 ms and 220 MB under streaming. The slots are zero-filled and, with NOCOPY, never refilled,
so the kernel reads zeros. A 6x difference from reading zeros instead of real quantized data
means NOCOPY changes what the kernel does, not just how fast it runs. Every conclusion that
leans on NOCOPY as a compute baseline is therefore unsound, including the "compute 4163 ms"
column in `FINDINGS_the_cliff.md`.

**Contradiction 3: an earlier guess about the kernel was wrong.**

An earlier draft of this document suspected the expert math might run under `mul_mat_vec_q_moe`
instead of `mul_mat_q`. Checked: that kernel exists **only in the SYCL backend**
(`ggml/src/ggml-sycl/mmvq.cpp:2676`). The CUDA backend has no such kernel, so for CUDA the MoE
path is `mul_mat_q` with an expert-sized grid. The 900-launch inventory below confirms this by
measurement.

**Consequence for planning.** Do not build the argument on counters until they agree with a
quantity that is already known independently. The known quantities are: 996 MiB per layer,
43.59 GiB per pass, 11.8 GB/s of copy, and the kernel grid/block geometry. Use those as the
invariant.

---

## 2. The two hypotheses, stated so they are distinguishable

**H1 - the GPU is saturated and the 15% figure is an artifact.**
The 2.61 TFLOP/s peak is FP32 FMA on the CUDA cores. If the quantized matmul actually runs on a
lower-throughput path (int8 dp4a, or the fp16 path without tensor cores), then 15% of FP32 peak
could be 60-80% of the real achievable peak. Under H1 the kernel inventory will show the SMs
busy, stalls dominated by `math_pipe_throttle` / `mio_throttle` rather than memory, and the
achieved FLOP/s will sit near the rate of the path actually taken.

**H2 - something else is the wall.**
The layer time is set by something other than SM work: the PCIe copy, cache/memory latency, or
scheduling. Under H2 the kernel inventory will show large idle gaps between kernels, or kernels
dominated by `long_scoreboard` stalls with `sm__throughput` low.

These have opposite remedies, so the measurement must separate them rather than accumulate more
numbers in the ambiguous middle.

---

## 3. Phase 1 - the measurement that matters most: a complete kernel inventory

> DONE in part, see `PHASE5_RESULTS.md` and `CALIBRATION.md`. The inventory found `mul_mat_q`
> grid 8192 at 54.9% of kernel time, and the calibration then showed the GPU does **112.2 GB/s
> on the identical launch geometry** (grid 8192 x 256, 1 block/SM) where that kernel achieves
> 11.7 GB/s. The live task list and the parameter set for the stall counters are in
> `TASKLIST.md`.

Everything else is secondary. One run, one replay pass, all kernels, time only.

```sh
ncu --replay-mode application --target-processes all \
    --launch-count 0 --metrics gpu__time_duration.sum --csv \
    ./build-nograph/bin/llama-bench -m models/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf \
    -ngl 1 -ncmoe 99 -lm none -p 512 -b 512 -ub 512 -ps 1 -r 1 -n 0 --no-warmup
```

Deliverable: every kernel, its launch count, its total time, and its share of the pass, with
grid and block geometry. This answers:

- Which kernel actually consumes the ~8-15 s? (Not `mul_mat_q` by assumption - by measurement.)
- Is the time inside kernels, or in the gaps between them?
- Does an expert-shaped kernel (`mul_mat_vec_q_moe`, or a `mul_mat_q` with an expert-sized
  launch) appear at all, and what fraction is it?
- Sum of kernel time versus wall time gives the **idle fraction** directly, which is the
  cleanest possible test of H2 and needs no counter trust.

Expected total FLOP for the pass, for the utilization arithmetic afterwards:
`2 x 3 x 2048 x 512 x 10 x 48 x 512 tokens` = 1.545 TFLOP (MoE), plus attention/GDN.

Control run: same command with `-ps 0` (baseline, no streaming) for comparison.

## 4. Phase 2 - convert the inventory into achieved FLOP/s (no counters needed)

From Phase 1, take the top kernels by total time. For each, compute achieved FLOP/s from its
own geometry and its measured time, and compare against:

| candidate peak | value | when it applies |
|---|---:|---|
| FP32 FMA on CUDA cores | 2.61 TFLOP/s | fp32 math |
| fp16x2 (no tensor cores) | 2.61 TFLOP/s | packed fp16 |
| int8 dp4a | ~5.2 TOPS | dp4a path |

This yields the honest utilization number. It is the direct answer to the 15% question and it
does not depend on any profiler counter being correct.

## 5. Phase 3 - a real control for the copy (this does not exist yet)

NOCOPY cannot serve as a control (see 1). The missing control is a slot filled from a
**VRAM-resident** source instead of host memory:

- add a mode that copies the layer's weights host -> VRAM once, then refills the slot from that
  VRAM copy (VRAM -> VRAM, no PCIe in the loop);
- then the same kernels run over the same real bytes at the same cadence, with the PCIe copy
  removed from the critical path.

| variant | PCIe in path | real weights | what it isolates |
|---|---|---|---|
| streaming (today) | yes | yes | baseline |
| **VRAM-sourced refill (new)** | **no** | **yes** | pure compute + kernel cost |
| NOCOPY (today) | no | **no (zeros)** | invalid, do not use |

If the pass time drops to roughly the Phase 1 kernel-sum under the VRAM-sourced variant, the
copy was the wall and the SMs were starved by it. If it stays high, the SM-side cost is real.

Cost: one small patch in `prefill_streaming_fill_slot`, gated by an env var like the others.

## 6. Phase 4 - the byte-ramp test: the true weight-read rate

The cleanest direct measurement of "how fast can this GPU consume quantized expert weights",
and it validates or invalidates the counters at the same time.

Patch the filler to repeat a **known** byte count `piece_size` (e.g. 1, 2, 4, 8, 16, 32, 64,
128, 256 MiB) sampled from the real tensor, keeping the graph, the slots and the 48 splits
unchanged. Measure kernel time versus `piece_size`.

- slope of time vs `piece_size` = **bytes per second the SMs can consume**, with no PCIe and no
  expert-routing effects in the way
- if that slope is ~88 GB/s (69% of VRAM peak) the GPU is healthy and the wall is elsewhere
- if it plateaus near 9-12 GB/s, the SMs genuinely cannot consume the weights faster than that

Cross-check: run the same with `--metrics dram__bytes_read.sum` and require the counter to
report approximately `piece_size`. If it does not, the counter is wrong and Phases 2 and 5 must
not rely on it. This is the validation step the earlier work skipped.

## 7. Phase 5 - utilization over a long enough window

`dmon` needs a window long enough to sample. Use a long run so the samples land:

```sh
nvidia-smi dmon -s mu -d 1 -c 40 &     # 1 Hz for 40 s
./build/bin/llama-bench ... -p 2048 -b 1024 -ub 1024 -ps 1 -r 3 -n 0 --no-warmup
```

Report `sm%`, `mem%` and the SM clock. Two things to watch, both decisive in their own right:

- **`mem%` near zero while `sm%` is also low** means the memory system is idle and the problem
  is latency or scheduling.
- **SM clock during the run.** `dmon` showed sm clock 300 MHz at idle. If the laptop GPU does
  not sustain its 1455 MHz boost under sustained load, the 2.61 TFLOP/s peak is wrong and part
  of the 15% is explained by the clock alone. Sample `clocks.sm` throughout; this is a
  first-order effect that no counter above would reveal.

Note: do NOT sample during an ncu run. Replay dilutes and delays the work, so the samples are
meaningless (already observed).

## 8. Phase 6 - the MoE path specifically

Once Phase 1 says which kernels matter, profile those exact kernels (not a name guess) with:

```
gpu__time_duration.sum
dram__bytes_read.sum
lts__t_sector_hit_rate.pct
l1tex__t_sector_hit_rate.pct
sm__throughput.avg.pct_of_peak_sustained_elapsed
sm__warps_active.avg.pct_of_peak_sustained_active
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
launch__occupancy_limit_blocks / _registers / _shared_mem
```

Request them in ONE invocation. Each metric costs a full model load (~20 min here), so nine
metrics is roughly three hours; batch them and do not iterate casually.

Pre-registered reading, so the result is decisive rather than discussed:

| signature | verdict |
|---|---|
| `sm__throughput` high, stalls dominated by `math_pipe_throttle` / `mio_throttle` | H1, GPU saturated |
| `sm__throughput` low, stalls dominated by `long_scoreboard`, `sm__warps_active` low | H2, latency-bound |
| occupancy capped by `launch__occupancy_limit_*` | H2 variant, fixable by tiling/launch shape |

## 9. Ordering, cost and stopping rule

1. ~~**Phase 5 first**~~ DONE, see `PHASE5_RESULTS.md`. It reframed the problem: the GPU is not
   power or clock limited, yet both its compute and memory pipes run at ~10%.
2. ~~**Phase 1**~~ DONE for a 900-launch window. `mul_mat_q` grid 8192 holds 54.9% of kernel
   time, at 84.75 ms per launch and 14.6% of compute peak.
3. **Phase 2** (arithmetic on Phase 1, no new runs). Gives achieved FLOP/s.
4. **Phase 6** on only the kernels Phase 1 identifies (~40 min for one batch).
5. **Phase 3 and Phase 4** only if 1-3 leave the question open. These need a small code change
   and are the strongest controls, but they are wasted if Phase 1 already answers it.

Stopping rule: if Phase 1 shows the kernel sum is close to the wall time, the GPU is busy and
the question becomes which pipe is saturated - Phase 2 and 6 answer it. If the kernel sum is far
below the wall time, the GPU is idle and the answer is in the gaps - Phase 3 is then mandatory.

## 10. Reporting rule for the results

For every number, state which of the two it is:

- **counted** - a profiler counter,
- **derived** - arithmetic on known quantities (sizes from the gguf, times, grids).

Counters must be validated against derived quantities before they are used to support a
conclusion. The earlier round mixed the two, which is how "the weights are not passing through
DRAM at all" got written down: that sentence recorded a counter disagreement, not a fact about
the hardware. No conclusion should rest on a counter that contradicts a known size.
