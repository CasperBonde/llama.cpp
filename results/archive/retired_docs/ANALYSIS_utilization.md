# Utilization analysis: is prefill streaming bandwidth-bound or latency-bound?

Model Qwen3-Next-80B-A3B-Instruct-Q4_K_M: 48 layers, n_embd 2048, n_expert 512,
n_expert_used 10, moe_intermediate 512.
GPU Quadro T1000 (TU117): 14 SMs, 896 FP32 cores, 1.455 GHz, **no tensor cores**,
L2 = **1024 KB**, peak DRAM 128 GB/s (measured achievable ~110).

## 1. Your 15-20% is confirmed by independent arithmetic

MoE FLOPs per token = 10 experts x 3 mats x 2 x 2048 x 512 = 62.9 MFLOP
per layer, x 48 layers = **3.02 GFLOP/token**.

Peak FP32 (no tensor cores, so quantized matmul runs on the INT8/dp4a path at
roughly FP32 rate) = 896 x 2 x 1.455e9 = **2.61 TFLOP/s**.

| throughput | achieved | % of peak |
|---|---:|---:|
| 64.5 t/s (pp512) | 195 GFLOP/s | **7.5%** |
| 100 t/s | 302 GFLOP/s | **11.6%** |

Attention + GDN add ~1% more FLOPs, so the real figure is ~12-15%. Your 15-20% is right.

## 2. The decisive test: how much DRAM traffic per ubatch?

One pp512 ubatch streams 45.86 GiB through the slots. Predicted GPU-side DRAM traffic:

| source | traffic |
|---|---|
| H2D copy landing in VRAM | 45.86 GiB **write** |
| compute reading the weights | 45.86 GiB **read** |
| activations (small) | ~1-2 GiB |
| **total if every weight is read exactly once** | **~92 GiB** |

Note: that is **2x the model size**, which is exactly the bound you proposed. So:

| measured total DRAM traffic | verdict |
|---|---|
| ~92 GiB | clean: each weight read once, NOT thrashing |
| ~184 GiB (2x) | rereading once, mild |
| **~810 GiB (9x)** | thrashing, and DRAM bandwidth is the binding constraint |

The 9x figure is the sharp one. Wall time for pp512 is 7.94 s. To be DRAM-bandwidth-bound,
traffic must be 7.94 s x 110 GB/s = **813 GiB**. Below that, DRAM cannot be what limits us:

| traffic | DRAM time at 110 GB/s | % of the 7.94 s wall |
|---|---:|---:|
| 92 GiB (clean) | 0.84 s | 11% |
| 200 GiB | 1.83 s | 23% |
| 813 GiB | 7.94 s | 100% |

**So the test is binary and falsifiable**: if `dram__bytes_read.sum + dram__bytes_write.sum`
comes back near 92 GiB, the cache-miss theory is dead and the 12% is a latency/occupancy problem.
If it comes back near 800 GiB, you are right and DRAM is the wall.

## 3. Second discriminator: which failure mode is it?

Same run, these tell bandwidth from latency apart:

| hypothesis | signature |
|---|---|
| bandwidth-bound | `dram__throughput` high, `sm__throughput` low, hit rates low, traffic ~9x |
| **latency-bound** | `sm__throughput` low, **`dram__throughput` LOW**, high `long_scoreboard` stalls, low `sm__warps_active` |
| occupancy/tile-bound | `launch__waves_per_multiprocessor` < 1, low occupancy not explained by stalls |

My prior is latency/tile, not bandwidth, for this reason: at ub=512, 512 tokens x 10 experts =
5120 token-expert pairs spread over 512 experts, so **each expert GEMM has M ~= 10** (M ~= 30 at
ub=1536). A GEMM of shape [10 x 2048] @ [2048 x 512] is extremely skinny. Each expert reads
~590 KB of weights to do ~63 MFLOP, and the M dimension cannot fill a tile. 512 tiny GEMMs with
M~=10 will not cover DRAM latency, however much bandwidth is spare.

## 4. Correction to the stated mechanism

"Cache miss because of the large micro-batch size" does not hold for the dominant traffic.
**A larger batch gives MORE weight reuse, not less**: M per expert rises from ~10 (ub512) to
~30 (ub1536), so each weight load feeds more tokens. Shrinking the batch would make weight reuse
WORSE. The activation working set does grow with batch (ub1536 x 2048 x 4 B = 12.6 MB, 12x L2),
but activations are ~1-2% of traffic.

The real batch-size effect is a different one: at small ub only some experts are routed to
(ub=64 touches ~365 of 512 experts), at large ub all 512 are touched. That changes WHICH weights
are read, not how many times each is read.

## 4a. The decisive number: the expert read rate does NOT change with batch size

Two measurements from data already on disk, same model, same 48 layers, same 94 refills,
`-ngl 1 -ncmoe 99 -lm none`. Both are the streaming path.

**(a) one token** (`copy_cost_vs_ubatch.txt`, 1-token ubatches, 0 refills):

| | value |
|---|---:|
| total per layer | 93 ms / 48 = **1.94 ms** |
| experts read | 10 (n_expert_used) x 1.77 MB = **17.7 MB** |
| rate | **9.1 GB/s** |

**(b) 512 tokens** (`ub512_single.txt`, reused/serialized path):

| | value |
|---|---:|
| total per layer | 7947 ms / 48 = 165 ms |
| minus copy | 165 - 78 = **88 ms** |
| experts read | 512 x 1.77 MB = **864 MiB** |
| rate | **9.9 GB/s** |

Both cases read the expert weights at **~9-10 GB/s, against a 128 GB/s VRAM bus (7-8%)**.
The batch size changes by 512x and the rate does not move.

This is the falsification of the L2-thrashing mechanism, and it is airtight because of the
1-token case: at 1 token the activation working set is 2048 x 4 B = **8 KB**, which fits in the
1 MB L2 by a factor of 128. There is no eviction pressure, no thrashing, nothing to miss.
Yet the rate is the same 9.1 GB/s. So thrashing cannot be what sets the ceiling.

**(c) the marginal cost is linear in tokens** (`ub512_single.txt`, 94 refills, reused path):

| p | wall | tokens |
|---|---:|---:|
| 512 | 7947 ms | 512 |
| 768 | 9900 ms | 768 |

marginal = (9900 - 7947) / 256 = **7.63 ms/token**; fixed = 7947 - 512 x 7.63 = **4042 ms**.

Subtracting the 3.7 s copy: GPU-side = 0.35 s fixed + 7.6 ms/token. Ideal compute at
2.61 TFLOP/s is 1.16 ms/token, so the marginal runs at **15% of peak** - exactly the figure
derived in section 1, and exactly the user's 15-20%.

There is no threshold anywhere: the per-token cost is constant from 1 token upward. A cache
effect would have a knee at the point the working set crosses 1 MB (about 250 tokens); the
data shows a straight line instead.

**What the signature is instead:** a fixed ~9 GB/s cap on quantized weight reads, independent
of M, that also applies at M=1. That is the shape of a **per-load latency ceiling**, not a
bandwidth or capacity one. The kernel issues dependent loads and cannot cover the latency with
the limited warps it has resident (the control kernel in 4b shows what a healthy one looks like:
76% SM throughput, 78% L2 hit, 2.1 GB/s). A correct next measurement would be
`smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` on the MoE kernel, which is
being collected.

## 4b. First profiler data point: the dense Q5_K matmul is healthy

`ncu` on `mul_mat_q<(ggml_type)13, (int)128, (bool)0>` (Q5_K, grid (32,8,1) = 256 blocks,
the dense/shared-expert path, 18.2 ms per launch):

| metric | value |
|---|---:|
| `dram__bytes_read.sum` | 38.35 MB |
| `gpu__time_duration.sum` | 18.21 ms |
| `lts__t_sector_hit_rate.pct` | 78.05 % |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 76.18 % |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 25.02 % |

38.35 MB / 18.21 ms = **2.1 GB/s** of DRAM read, against a 128 GB/s bus. 78% L2 hit rate,
76% SM throughput. This kernel is **compute-bound and healthy**: it reads each weight about
once, L2 absorbs most of the reuse, and the SMs are the limit.

This is the control. It proves the measurement works and that a well-behaved matmul on this
GPU reaches 76% SM throughput. Whatever the MoE path is doing, it is not what this kernel does.

## 5. Commands (run as root until the module is reloaded)

Gauge the cost first with a small launch count:

```bash
sudo ncu --target-processes all -k regex:mul_mat --launch-count 8 \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sector_hit_rate.pct \
  ./build/bin/llama-bench -v -m models/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf \
  -ngl 1 -ncmoe 99 -lm none -p 512 -b 512 -ub 512 -ps 1 -r 1 --no-warmup
```

Then, for the actual totals, drop `-k` and `--launch-count` so every launch is counted
(slow, run once):

```bash
sudo ncu --target-processes all --launch-count 0 \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum \
  <same llama-bench args>
```

Control run: add `LLAMA_PREFILL_STREAMING_NOCOPY=1` to remove the H2D write, so the read
traffic is isolated from the copy.

## 6. Limitation

ncu stops and REPLAYS kernels to read counters (6 passes for a single-metric test kernel).
It therefore cannot measure overlap or the 3x copy interference. Use it for this question only.
