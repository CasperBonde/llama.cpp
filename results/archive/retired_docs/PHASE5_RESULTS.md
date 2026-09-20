# Phase 5 + Phase 1 results: the GPU is neither power limited nor memory limited

Date: 2026-09-19. Model Qwen3-Next-80B-A3B-Instruct-Q4_K_M, `-ngl 1 -ncmoe 99 -lm none`,
GPU Quadro T1000 (TU117): 14 SMs, 896 FP32 cores @ 1.455 GHz, L2 1 MiB, VRAM 128 GB/s peak,
**BAR1 256 MiB**, power limit 50 W.

---

## 1. Phase 5: clocks, power, utilization

`nvidia-smi --query-gpu=clocks.sm,... -l 1` alongside a plain
`-p 2048 -b 1024 -ub 1024 -ps 1 -r 2` streaming run. 150 samples, 85 in the busy region
(85 consecutive seconds at `utilization.gpu >= 99%`).

| metric | idle | busy (85 s) |
|---|---:|---:|
| SM clock | 300 MHz | **1395 MHz** (constant, never drops) |
| memory clock | 405 MHz | 4001 MHz |
| `utilization.gpu` | 0% | **100%** (99-100% for the whole window) |
| `utilization.memory` | 0% | **10%** (range 7-12%) |
| power draw | 12.4 W | **18.6 W** (max 21.95 W) |
| temperature | 45 C | 63 C |

Power and throttle state at the time:

| item | value |
|---|---|
| power limit (default = max) | **50.00 W**, 3.18 A at 15.7 V |
| SW Power Cap | Not Active |
| HW Slowdown / HW Thermal / HW Brake | Not Active |
| SW Thermal Slowdown | Not Active |
| pstate | P0 |

### Reading

- **Not power limited.** 18.6 W of a 50 W budget is 37%. No capping flag is active during the
  run, and the SM clock holds 1395 MHz for the entire 85 s with zero variation. The clock
  limit is 1455 MHz, so we are at 96% of the achievable clock. ~~Clock throttling~~
  and ~~power capping~~ are both eliminated.
- **The GPU is "100% busy" and almost completely idle at the same time.** `utilization.gpu`
  (SM occupancy) is pegged at 100%, while `utilization.memory` (memory controller throughput)
  sits at **10% of a 128 GB/s bus = roughly 13 GB/s**. Nothing is queueing on the memory side,
  and the SMs, though nominally occupied, are consuming 18.6 W instead of the 40-50 W you would
  expect if they were issuing FMAs continuously.

That combination is diagnostic: **the SMs are resident but not executing.** They are parked on
a dependency. This is independent confirmation of the ~9-10 GB/s weight-read ceiling from the
earlier arithmetic, and it is measured with a tool that cannot be confused by replay: a
sustained 85 s window at 1 Hz.

It also rules out the remaining bandwidth hypotheses: the DRAM bus is at 10%, so neither
weight re-reads nor copy contention can be the limit. The limiter is not a throughput of
anything. It is latency.

---

## 2. Phase 1: kernel inventory, 900 consecutive launches

`ncu --replay-mode application --launch-count 900 --metrics gpu__time_duration.sum`.

| total ms | launches | us each | kernel |
|---:|---:|---:|---|
| **4728.2** | **215** | 21992 | `mul_mat_q` |
| 79.1 | 13 | 6085 | `gated_delta_net_cuda` |
| 32.5 | 4 | 8122 | `volta_h884gemm_256x128_ldg8_tn` |
| 13.9 | 16 | 869 | `volta_sgemm_128x32_tn` |
| 12.8 | 163 | 79 | `quantize_mmq_q8_1` |
| 9.2 | 46 | 200 | `rms_norm_f32` |
| 8.6 | 103 | 83 | `k_bin_bcast` |
| 8.5 | 48 | 177 | `unary_gated_op_kernel` |
| 7.9 | 13 | 611 | `concat_non_cont` |
| 6.9 | 13 | 529 | `ssm_conv_long_token_f32` |
| 6.9 | 48 | 143 | `mm_ids_helper` |
| 6.6 | 4 | 1662 | `volta_sgemm_32x128_tn` |
| 6.5 | 16 | 409 | `moew_weighted_reduction_f32` |

**`mul_mat_q` is 95.6% of all kernel time.** Everything else is under 2% each. There is no
mystery kernel and no missing work: the matmuls are the wall.

### The matmuls by grid size

| grid | quant type | launches | total ms | ms each |
|---|---:|---:|---:|---:|
| **(8192, 1, 1)** | **12 (Q4_K)** | **32** | **2711.9** | **84.75** |
| (32768, 1, 1) | 14 | 9 | 853.1 | 94.78 |
| (32768, 1, 1) | 12 | 7 | 521.7 | 74.53 |
| (256, 1, 1) | 13 | 13 | 233.5 | 17.96 |
| (64, 1, 1) | 12 | 16 | 150.1 | 9.38 |
| (128, 1, 1) | 12 | 12 | 113.2 | 9.43 |

Grid 8192 = 512 experts x 16 blocks, launched 32 times in the window = 2 launches per MoE layer
x 16 layers. **This is the expert GEMM**, at **84.75 ms per launch**, and it alone is **54.9% of
all kernel time**.

## 3. The central number: where does the 2712 ms go?

The grid-8192 launch must consume the layer's 996 MiB of expert weights. It takes 84.75 ms.

| quantity | value |
|---|---:|
| weight bytes per launch | 996 MiB |
| measured time | 84.75 ms |
| **achieved read rate** | **12.3 GB/s** |
| VRAM peak | 128 GB/s |
| **share of VRAM peak** | **9.6%** |

And the compute it performs:

| quantity | value |
|---|---:|
| useful FLOPs per launch | 30.4 GFLOP (measured requirement, see below) |
| measured time | 84.75 ms |
| **achieved compute rate** | **0.36 TFLOP/s** |
| FP32 peak | 2.61 TFLOP/s |
| **share of compute peak** | **13.7%** |

The two independent numbers agree with each other and with everything measured before:
**~10% of memory peak and ~14% of compute peak, simultaneously.** Neither pipe is the
constraint; both are running at about a tenth of capacity.

Your ~15% hunch is confirmed a third time, now by a direct throughput measurement rather than
by dividing total time by peak.

### The FLOPs-per-byte check pins it to the weight path

Per expert the geometry is `[M x 2048] @ [2048 x 512]` for gate and up, and
`[M x 512] @ [512 x 2048]` for down, where M = 512 tokens / 512 experts = 1.0 under
`-ncmoe 99`, rising toward 10 as you approach full residency.

- ideal FLOP per expert at M=10: 3 x 2 x 10 x 2048 x 512 = **62.9 MFLOP**
- weight bytes per expert: 1.77 MB
- **intensity: 35.5 FLOP/byte**
- ridge point (2.61 TFLOP/s / 128 GB/s): **20.4 FLOP/byte**

35.5 is above the ridge, so by arithmetic this kernel should be **compute bound**. It measures
13.7% of compute peak and 9.6% of memory peak. When a kernel is above the ridge and neither
pipe is saturated, the only remaining explanation is that the SMs are waiting on something that
is neither throughput: memory latency they cannot hide.

The 1-token control from the earlier analysis proves the same thing from the other end. At one
token there is no possible cache pressure (8 KB working set in a 1 MiB L2) and the rate is the
same 9.1 GB/s.

## 4. Consequences

### 4.1 the cp.async hypothesis is dead

An earlier draft of this strategy proposed that `cp.async` might be disabled on sm_75 and that
async copies were falling back to synchronous loads, and proposed adding them as a fix. The code
does the opposite:

```
ggml/src/ggml-cuda/common.cuh:
  #define GGML_USE_CP_ASYNC                       // sm_80+
  #define CP_ASYNC_AVAILABLE (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
```

`cp.async` is **enabled only for sm_80 and above**. On this sm_75 part `CP_ASYNC_AVAILABLE` is
false by design: the asynchronous copy path is deliberately not used, and the synchronous load
path is what runs. So `cp.async` is not missing on sm_75 - it is intentionally gated off, and
enabling it is not a fix that exists. Two candidate fixes are therefore eliminated in one step:
the hardware cannot do it, and there is nothing broken to repair.

### 4.2 what the remaining explanation has to account for

Any correct explanation must simultaneously satisfy:

1. grid-8192 runs at 9.6% of memory peak and 13.7% of compute peak, while being **above the
   arithmetic ridge** and therefore nominally compute bound;
2. the rate is **identical at M=1 and M=10**, with no knee;
3. `utilization.gpu` is 100% while `utilization.memory` is 10% and power is 37% of budget;
4. at ub=64 the same kernel reads the same 996 MiB at **88 GB/s, 69% of VRAM peak** - so the
   kernel *can* stream fast when checked at small launch counts;
5. the copy is at the PCIe link limit (11.8-12.4 GB/s) and needs no fixing.

Point 4 is the sharpest remaining clue and is not yet explained: the same bytes, read by the
same kernel, go 7x faster in one measurement than the other. That gap - not any counter - is
now the thing to chase, and it is a measurement artefact question before it is a hardware
question. Reconciling 4 against 1-3 is the next step.

## 5. Method notes

- `ncu --launch-count 0` is rejected by this build ("Please specify non-zero launch count").
  Use a large explicit bound.
- `utilization.gpu` is SM occupancy, not issue rate. It reads 100% for a kernel that is fully
  latency-parked, which is why it must be read next to `utilization.memory` and power.
- Sample clocks/util with `nvidia-smi -l 1` during a plain run. During an `ncu` run the replay
  dilutes the work and the samples are meaningless.
- `mul_mat_vec_q_moe` exists only in `ggml-sycl`. The CUDA MoE path is `mul_mat_q` with an
  expert-sized grid; verify by grid geometry, not by symbol name.
