# Cache-miss analysis of the prefill streaming path

Date: measured on `27543a7ae`. Model Qwen3-Next-80B-A3B-Instruct-Q4_K_M,
`-ngl 1 -ncmoe 99 -lm none`, GPU Quadro T1000 (TU117): 14 SMs, 896 FP32 cores @ 1.455 GHz,
**2.61 TFLOP/s**, **L2 = 1024 KB**, **VRAM peak 128 GB/s** (measured ~110).

Tool: `ncu` 2022.4.1, non-root (profiling permission unlocked by
`NVreg_RestrictProfilingToAdminUsers=0`). Kernel replay does not work on this box
(3.7 GB device backup), so all runs use `--replay-mode application`. CUDA graphs hide
kernels from ncu, so the profiled runs use `build-nograph`.

The question: is the ~15% SM utilization caused by L2 thrashing, where CUDA-core
intermediates evict expert weights so the weights must be re-fetched from VRAM?

---

## 1. Verdict: no. Two independent measurements contradict it.

### 1.1 The expert read rate does not move when the batch changes by 512x

All from data already on disk (`copy_cost_vs_ubatch.txt`, `ub512_single.txt`).

| case | experts read per layer | time per layer | effective rate |
|---|---:|---:|---:|
| **1 token** | 10 x 1.77 MB = **17.7 MB** | **1.94 ms** | **9.1 GB/s** |
| 512 tokens | 512 x 1.77 MB = **864 MiB** | **88 ms** | **9.9 GB/s** |

The 1-token case is the control, and it is decisive. At one token the activation working
set is `2048 x 4 B = 8 KB`. L2 is 1 MB. The working set fits **128 times over**. There is
no eviction pressure of any kind. Yet the expert weights are still read at 9.1 GB/s, the
same rate as at 512 tokens where the working set is 4 MB (4x L2).

If thrashing set the ceiling, the 1-token case would be fast. It is not.

### 1.2 The MoE expert GEMM has a ~76% L2 hit rate and reads almost nothing from VRAM

`ncu`, streaming path (`-ps 1`), `mul_mat_q<(ggml_type)12, (int)128, (bool)0>`, grid
`(8192, 1, 1)`, 80 samples:

| metric | value |
|---|---:|
| `gpu__time_duration.sum` | 69.0 ms per launch |
| `dram__bytes_read.sum` | **220 MB** per launch |
| weights the launch must consume | 512 experts x 1.77 MB = **906 MB** |

906 MB of weights, 220 MB from VRAM: **~76% is served by L2.** Hit rates of 76% against a
1 MB cache are the opposite of a thrashing signature. A thrashing kernel would read the
906 MB from VRAM more than once, not 0.24x of it once.

Summed over all 400 sampled matmul launches in one pass: **10.24 GB of DRAM read total**.
The model streams 45.86 GiB of weights. The weights are not passing through DRAM at all -
they are written to VRAM by the copy engine and read from L2 by the SMs.

Also: no `lts__t_sectors_aperture_sysmem_op_read` traffic (0.00 MB on every kernel), so the
SMs are not reaching out over PCIe for weights either.

### 1.3 Control: a healthy matmul on this GPU looks like this

`mul_mat_q<(ggml_type)13, (int)128, (bool)0>` (Q5_K, grid (256,1,1)):

| metric | value |
|---|---:|
| `dram__bytes_read.sum` | 38.35 MB |
| `gpu__time_duration.sum` | 18.21 ms |
| `lts__t_sector_hit_rate.pct` | **78.05 %** |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | **76.18 %** |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 25.02 % |

76% SM throughput with 78% L2 hit and 2.1 GB/s of DRAM read. This is what the hardware can
do when the kernel is healthy. Note the L2 hit rate is essentially the same as the MoE
kernel's - so hit rate is not what separates the good kernel from the bad one.

---

## 2. What the evidence does say

The rate that needs explaining is the flat **~9.5 GB/s of quantized expert weight read,
independent of M**. Facts that constrain it:

- **It is below every hardware limit simultaneously.** 9.5 GB/s against 128 GB/s of VRAM
  (7%), against a much larger L2 bandwidth, and 15% of the 2.61 TFLOP/s compute peak. No
  single resource is saturated, so the limit is not throughput of any one unit.
- **It is linear, with no knee.** From `ub512_single.txt`, same 94 refills, reused path:
  p512 = 7947 ms, p768 = 9900 ms. Marginal cost = **7.63 ms/token**, fixed = **4042 ms**.
  A straight line from 1 token upward. A cache-capacity effect would show a knee where the
  working set crosses 1 MB (about 250 tokens). There is no knee.
- **The fixed term is the copy.** 4042 ms fixed is the 3.7 s PCIe transfer. The GPU-side
  marginal is 7.6 ms/token against an ideal 1.16 ms/token = **15% of peak**, matching the
  independently derived figure from FLOP arithmetic.
- **Streaming is already 1.7x better than the baseline.** Baseline pp512 = 38.15 t/s =
  13.4 s; streaming = 7.95 s. Both move the same 45.86 GiB over PCIe. Having the copy engine
  place the weights in VRAM and letting the SMs read from there beats having the GPU pull
  them over PCIe on demand. So whatever the SM-side limit is, VRAM-resident weights already
  relieve it substantially.

The consistent reading: the SMs cannot consume expert weights faster than ~9.5 GB/s because
of how the work is shaped, not because the data is far away or being repeatedly re-fetched.
At M~=10 per expert, each expert is a `[10 x 2048] @ [2048 x 512]` matmul - a weight load
that feeds only 10 output rows. The kernel is spending its time on the weight path rather
than on FMAs. That is a **latency/occupancy** shape, which is what CUDA-core intermediates
competing with weights would be expected to cause only if they actually evicted the weights -
and sections 1.1 and 1.2 show they do not.

---

## 3. Anomaly that is NOT yet resolved

Under `LLAMA_PREFILL_STREAMING_NOCOPY=1` the same grid-8192 kernel runs **11.3 ms** and
reads **55.7 MB**, versus 69.0 ms and 220 MB under streaming. A 6x kernel-time difference
with no matching traffic difference is not explained.

`NOCOPY` is therefore **not** a clean compute reference for this kernel: if the slot is not
filled, the kernel is not reading the real weights, so it is not doing the real work. The
`NOCOPY` total of 4.17 s must not be read as "this is how long the compute takes". Removing
the copy changes what the kernel does. Any conclusion that relies on `NOCOPY` as a compute
baseline - including the "compute 4163 ms" column in `FINDINGS_the_cliff.md` - needs to be
re-derived by another means.

This is flagged rather than papered over. It does not affect sections 1.1-1.3, which use
the streaming path only.

---

## 4. Measurements that would settle the remaining question

Cheap, in order of value:

1. `sm__throughput`, `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct`,
   `sm__warps_active` on the grid-8192 kernel via application replay. High long-scoreboard
   with low SM throughput confirms latency-bound; high DRAM/L2 throughput would not.
   Cost: one metric per ~20 min replay on this box, so request them in a single invocation.
2. A sweep of the same kernel at M=10 vs M=20 vs M=40 (ub 512/1024/2048) with
   `gpu__time_duration.sum` only. If the per-weight-byte cost is flat across M, the limit is
   structural in the kernel; if it improves, it is latency hiding.
3. Attribution inside the kernel: split the 3 expert GEMMs (w1/w3, w2) and time them
   separately, to see whether the cost is the matmul proper or the surrounding routing
   (`mm_ids_helper`, `quantize_mmq_q8_1`, `mul_mat_q_stream_k_fixup`).

---

## 5. Method notes, so this is reproducible

```sh
# streaming path, application replay, no CUDA graphs
ncu --replay-mode application --target-processes all \
    -k "regex:mul_mat_q$" --launch-skip 0 --launch-count 400 --csv \
    --metrics dram__bytes_read.sum,gpu__time_duration.sum \
    ./build-nograph/bin/llama-bench -m models/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf \
    -ngl 1 -ncmoe 99 -lm none -p 512 -b 512 -ub 512 -ps 1 -r 1 -n 0 --no-warmup
```

Pitfalls hit while producing this:

- `dram__bytes_read.sum` reports unit `byte` (the "Mbyte" label appeared only with a
  reduced metric set). Values in the sweep files are bytes.
- CSV columns: index 8 = grid, 12 = metric name, 14 = value.
- Each requested metric costs a full application-replay pass, and a pass is a full model
  load (45 GiB). Four metrics is roughly 40 minutes of wall time. Request the minimum.
- Kernel replay fails with "Backing up device memory in system memory" (3.7 GB VRAM).
  `--replay-mode application` is mandatory here.
- `build/` uses `GGML_CUDA_GRAPHS=ON` and ncu cannot see the kernels. Use
  `build-nograph/`.
- `nvidia-smi dmon` during a profiled run is not representative: the replay dilutes and
  delays the work being measured.
