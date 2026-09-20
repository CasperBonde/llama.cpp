# Prefill streaming: current state of knowledge

Date: 2026-09-20. Build `27543a7ae` (10843).
This is the single working document. Older per-topic notes were folded in here and then removed.
`CALIBRATION.md` (hardware facts + synthetic benchmarks, now with power) and `PARAMS.md`
(model/GPU reference tables) are the two supporting documents kept live alongside it.
Everything else is under `../archive/`: `retired_docs/` (the superseded per-topic notes, including
the retracted write-ups - kept only as a record of the errors), `raw_logs/`,
`prefill_streaming_v1/` + `prefill_streaming_v1_logs_2026-09-13/` (the old
`prefill_double_buffering2` branch runs), and `synthetic_bench/` (the compiled CUDA probes).

---

## 1. What this is

Qwen3-Next-80B-A3B (45.17 GiB, Q4_K_M) on a 3716 MiB Quadro T1000 (`-ngl 1 -ncmoe 99`).
The MoE weights do not fit in VRAM, so the prefill path streams them layer by layer:

- two 1022.41 MiB slots in VRAM, 840 tensors (17.5 per layer)
- one slot holds layer `il` for the GPU to compute, the other is being filled from host RAM
- a per-layer `sync` keeps the fill and the consume from overlapping; the pipeline is serial
  round trips of (copy 1022 MiB) -> (compute 1022 MiB), 48 layers per ubatch

Without streaming: 52.32 t/s at pp1024. With streaming: 84.53 t/s. The win comes from the
slot mechanism, not from weight residency (attn weights in VRAM change prefill by <2%).

---

## 2. The reference configuration (pinned; do not vary it silently)

```sh
./build/bin/llama-bench -m models/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf \
  -ngl 1 -ncmoe 99 -lm none -n 0 -r 4 -p 1024 -b 1024 -ub 1024 -o md \
  --cache-type-k q4_0 --cache-type-v q4_0 -t 6 -ps 1
```

Measured **84.53 +/- 0.08 t/s**. The user's own reference sweep (`results/Manual_rerun_og_sweep.txt`,
build 10833) gives pp512 **64.59**, pp1024 **86.92**, pp1536 **97.80**.

Rules learned the hard way:

- `-lm none` is required. It is worth about 10 t/s at pp1024 (86 vs 76). It is not KV quantisation. It pins CPU buffer pages.
- KV quantisation (q4_0 vs f16) is worth <2%. It was never the variable. Do not drop the flag to
  "simplify", and do not credit it with a win it did not produce.
- llama-bench cross-products `-b` and `-ub`; it does not zip them. To sweep one, pass one value.
- `-ps` in llama-bench takes a value (0 or 1, default 0); `-ps 0,1` runs both in one process. This enables the prefill feature we are working on.
- `-v` is required for the `LLAMA_LOG_*` diagnostic output to appear.
- ONE llama process at a time. Check `pgrep -af "llama-bench|llama-cli"` first.
- llama-cli stdout cannot be captured by plain redirection; use `script -qec`.

---

## 3. Flag sensitivity (one flag at a time, from the pinned config)

| `-p` / `-b` / `-ub` | cache | t/s |
|---|---|---:|
| 512 / 512 / 512 | - | 63.25 |
| 512 / 512 / 512 | q4_0 | 63.23 |
| 1024 / 1024 / 512 | - | **32.30** |
| 1024 / 1024 / 1024 | - | 82.34 |
| 1024 / 1024 / 1024 | q4_0 | 84.53 |

Two conclusions, both solid:

1. **KV quantisation is not the variable.** 63.25 vs 63.23, 82.34 vs 84.53. Under 2% either way.
2. **The ubatch size is the variable.** `-ub` smaller than the prefill chunk costs 2.6x
   (82.34 -> 32.30). This is the unresolved item in section 7.

### 3.1 The ubatch sweep at pp512, both flags (direct evidence)

`-p 512 -b 512`, so `-ub` below 512 means the prefill is split into `512/ub` ubatches.

| `-ub` | ubatches | streaming off | streaming on |
|---:|---:|---:|---:|
| 64 | 8 | 18.87 | 14.44 |
| 128 | 4 | 17.62 | 12.89 |
| 256 | 2 | 24.93 | 21.21 |
| 512 | 1 | 37.68 | **64.67** |
| 1024 | 1 | 37.77 | **64.67** |

(from `sweep_80b.txt`, f16 KV, build `08b667fab`. `sweep_80b_q4kv.txt` is the identical sweep with
q4_0 KV and differs by under 0.3%, which is the KV-is-irrelevant result again.)

Two things this shows:

- **Streaming is worth +1.72x only when the ubatch is not split** (37.68 -> 64.67 at `-ub 512`), and
  the moment the ubatch splits it becomes a **loss** (24.93 -> 21.21 at `-ub 256`). The mechanism
  that produces the win inverts into a cost. This is the same effect as section 7, seen from the
  other side.
- **Splitting costs throughput even with streaming off** (37.68 -> 24.93 for 2 ubatches) - so
  splitting is not free in general. It is specifically the *streaming* path that turns the split
  from a moderate cost into a halving.

The 30B model shows the same shape (`../archive/root_notes_2026-09/ubatch_sweep_results.txt`):
at `-p 512 -b 512`, streaming off, `-ub 512` = 58.83 vs `-ub 256` = 39.19 vs `-ub 128` = 24.86.

---

## 4. Measured truth at the operating point

Earlier in this investigation the GPU was claimed to be idle 88.5% of the wall time. **That claim
is retracted.** The run behind it achieved 11.94 t/s where the same flags normally give 63.25 - a
run 5x below known-good. `kernel_sum / wall` is only meaningful over a compute-only window; with
`-r 1` the denominator includes CUDA init, model load (fb 304 -> 3112 MiB), warmup and replay
overhead.

Direct measurement, reference config, `-r 4`, `nvidia-smi` sampled at 0.5 s:

| quantity | value |
|---|---:|
| `utilization.gpu` | **99-100%** for the whole compute phase |
| `dmon sm` | **100%** |
| `dmon mem` | 8-15% |
| power | 12.6 W idle -> **18.4 W avg / 21.7 W max** during compute (50 W limit) |
| sm clock | **1395 MHz for 99.1%** of the compute window (max 1455) - no throttling |

**There is no idle window. The GPU is occupied but stalled** - warps are resident and waiting,
inside a launched grid. That is consistent with an SM-attached occupancy of 100% and a DRAM
utilization of 8-15%: the memory system is not saturated either, so the wait is dominated by
latency, not bandwidth (see section 6).

### 4.1 Power: the independent confirmation

`utilization.gpu` cannot tell work from waiting, because it is satisfied by any launched grid.
Power can. Measured with NVML directly, same grid/block/bytes as the real expert kernel
(`tmp/prof/powmeas.cu`, full table in `CALIBRATION.md` section 1f):

| workload | power |
|---|---:|
| idle | 12.8 W |
| **spin loop, moves zero bytes** | **25.9 W** |
| H2D copy, pinned | 16.1 W |
| pure VRAM read, 118 GB/s | 28.0 W |
| read at the expert kernel's 1 block/SM | 28.5 W |
| read + dp4a at the real intensity | 35.4 W |
| **real run, compute half (by subtraction)** | **19.5 W** |

**The computation draws less power than a spin loop that moves zero bytes, and less than a plain
read of the same bytes.** A read-only kernel of the same shape costs 28.0 W. This is the strongest
available evidence that the SMs are resident but doing almost nothing, and it does not depend on
any timing assumption.

Clock is not the reason: the SM clock is **1395 MHz for 99.1% of the compute window** (max 1455),
so there is no power or thermal throttling. The card cannot even draw its own 50 W limit; the
hardest sustained workload it can be given peaks at 35.4 W.

Limit of the instrument: NVML power updates on a **~95 ms window**, coarser than one layer round
trip (~130 ms). The trace shows flat plateaus that are averages, and never resolves the copy phase
as a separate level, so the copy/compute split above is a subtraction. *This was true when written;
section 4.2 now splits the two phases directly, without CUPTI and without a subtraction.*

Method (reusable): `tmp/archive/prof_logs_pre_retraction/ss_test.sh` +
`ss_analyze.py` for the coarse (0.5 s) power/utilization trace, and
`tmp/prof/pow_real.sh` + `tmp/prof/powlog` for the high-rate (175 Hz) NVML trace.
The compute window is the set of samples where power reports `utilization.gpu >= 90` (coarse)
or power > 16 W (high-rate), which excludes load and teardown.

Rule learned: a derived ratio needs a clean measurement window. Before using a ratio in any
conclusion, cross-check it against a direct instrument (power, dmon).

### 4.2 The copy/compute split, measured directly (serial mode)

The subtraction in 4.1 is no longer needed. `LLAMA_PREFILL_STREAMING_SERIAL=1` makes the loop
serial **on purpose**: one slot instead of two, and inside the eval callback the next layer's
weights are copied with a blocking `set_tensor` *before* that layer is submitted. The copy and the
compute are therefore disjoint intervals and the code times them per layer, which is exactly the
`copy_layer / compute_layer` flow the user asked for:

```
one callback invocation per layer:
    refill(layer n+2)   <- blocking copy, timed around the call
    compute(layer n+1)  <- timed from the ask to the next callback
```

`process_ubatch: serial phases over 46 layers, 1 slot: refill ... compute ...` prints the totals.
pp1024, `-ngl 1 -ncmoe 99 -lm none -r 3`, build `27543a7ae` + instrumentation:

| quantity | value |
|---|---:|
| refill (copy) total | 3720 - 3722 ms |
| refill mean / max | **80.9 ms / 87.3 ms** |
| copy throughput | **12.34 - 12.35 GB/s** |
| compute total | 6064 - 6068 ms |
| compute mean / min-max | **131.9 ms / 124.7 - 151.7 ms** |
| copy share of the ubatch | **38.0%** |
| compute share | **62.0%** |
| copy + compute | 9784 - 9790 ms of a 12575 - 12582 ms ubatch |
| result | **81.36 +- 0.06 t/s** |

The copy moves 1022.41 MiB per layer in 80.9 ms, i.e. **12.35 GB/s, which is 99% of the
12.4-12.5 GB/s a pinned H2D copy reaches on this link.** The same transfer with no compute at all
takes 81.7 ms (`tmp/prof/powmeas.cu`, 996 MiB). The copy phase is at the hardware limit and nothing
the GPU is doing delays it.

Two consequences:

- **The overlapped pipeline is already doing its job.** Serial 81.36 against overlapped 82.49
t/s is a 1.4% difference, so the overlap buys almost nothing: the copy fits inside the compute
time already. The ceiling is therefore not the copy, and the 3x interference law (6.2) is not
what limits the current config.
- **The copy is token-count independent and the compute scales.** The same 48 layer transfers take
  3733.6 ms at p512 and 3720.1 ms at p1024 (0.4% apart), while the compute goes 2898.8 ms ->
  6066 ms (2.09x for 2x the tokens).
trace can be folded modulo that period instead of subtracted. 173 Hz sampling, 51 s measurement
window, best of a full scan over the phase offset:

| phase | power | samples |
|---|---:|---:|
| copy bins | 18.87 W | 3328 |
| compute bins | 18.76 W | 5440 |
| **contrast** | **-0.11 W** | |

**The copy phase and the compute phase draw the same power, to within a tenth of a watt.** This
does not show what the SMs wait on, but it removes a family of explanations: the compute is not
resting between bursts, the copy is not leaving the card idle, and the two are not alternating on
the power rail in a way that would expose a duty cycle. Whatever the SMs wait on during compute
costs the same energy as streaming 12.35 GB/s in front of them.

That **both** phases sit at 18.8 W while a *spin loop* sits at 25.9 W remains the central anomaly
(4.1). The serial split rules the copy out as its cause.

---

## 5. Hardware and model reference

All values from `cudaGetDeviceProperties` / `nvidia-smi` / GGUF metadata on this box.

### GPU: Quadro T1000 (TU117, sm_75)

| param | value |
|---|---:|
| SMs | **14** |
| FP32 cores | 896 (64/SM) |
| threads / SM (hard limit) | 1024 |
| shared mem / block | 48 KB (static cap; 64 KB per SM) |
| registers / SM | 65536 |
| L2 | **1024 KB** |
| sm / mem clock | 1455 / 4001 MHz |
| bus width | 128 bit |
| DRAM peak | 128 GB/s (measured achievable ~110) |
| FP32 peak | 2.61 TFLOP/s |
| PCIe | Gen3 x16, practical **12.4 GB/s** |
| BAR1 | **256 MiB** |
| VRAM usable | 3716 MiB |
| power limit | **50 W** |
| tensor cores | **none** (TU117) |
| copy engines | 3 |
| concurrent kernels | 1 |

### Model: Qwen3-Next-80B-A3B-Instruct-Q4_K_M

| param | value |
|---|---:|
| size | 45.17 GiB |
| params | 79.67 B |
| n_layer | 48 |
| n_embd | 2048 |
| n_ff (dense) | 5120 |
| n_expert / n_expert_used | 512 / 10 |
| moe_intermediate_size | 512 |
| attention layers (full) | 12 |
| GDN / linear layers | 36 |
| expert tensors | 3 per layer, 144 total, 44 640 MiB (930 MiB per layer) |
| per-layer non-expert | 26.41 MiB (GDN) / 21.03 MiB (attention) |

Derived:

| quantity | value |
|---|---:|
| MoE FLOPs per token | 10 x 3 x 2 x 2048 x 512 x 48 = **3.02 GFLOP** |
| streamed bytes per ubatch | 45.86 GiB (whole model, every ubatch) |
| ridge point (FLOP/byte) | 2.61e12 / 128e9 = **20.4** |
| MoE intensity at ub512 | 3.02e12 x 512 / 45.86e9 = **33.7** |
| MoE intensity at ub2048 | 3.02e12 x 2048 / 45.86e9 = **135** |
| tokens per expert at ub512 | 5120 pairs / 512 experts = **10** |
| experts touched at ub512 | ~511 of 512 |

Consequence: the MoE layer is compute-bound by arithmetic intensity at prefill batch sizes
(33.7 > 20.4). A compute-bound kernel should sit near 100% utilization, which it does - but at
18.4 W, which is the puzzle the stall investigation must explain.

For reference, the other model used to exercise the same code path is Qwen3-Coder-30B-A3B
(48 layers, 128 experts, 8 used, 373.50 MiB of experts per layer, ~385 MiB per layer,
2 slots = 747 MiB). It fits comfortably; the 80B does not, which is why it is the hard case.

Note on "-ot": it is not free for generation. Moving tensors off CUDA costs generation throughput
because at bs 1 every weight is read once per token. See `../archive/prefill_streaming_v1/VRAM_BUDGET.md`
for the measured `-ot` tables and the separator-syntax trap.

---

## 6. Surviving technical findings

### 6.1 Copy is at the PCIe link limit

Measured 12.4 GB/s, which is **99% of the 12.5 GB/s 128b/130b Gen3 x16 ceiling**. There is no
headroom left in the copy itself. 48 layers x 1022 MiB = 48 GiB per ubatch = 4.07 s at 12.4 GB/s,
against kernels at roughly 4.9 s. The only lever left is hiding the copy, not speeding it up.

### 6.2 The 3x interference law

Serialized copy costs 1x. Concurrent copy costs 3x: 1x for the transfer plus 2x in damage to the
concurrent compute. The extra does not scale with token count, it scales with the copy. This is
why "just overlap them" does not work here.

### 6.3 The split sync is a correctness barrier, not a scheduling artifact

`ggml_backend_synchronize(split_backend)` at `ggml/src/ggml-backend.cpp:1833` looks like a
scheduling wart. Gating it behind `LLAMA_SCHED_NO_SPLIT_SYNC` gives **115.59 t/s** at pp1024 and
removes the pp decay entirely - but **produces garbage output**
(`/ far of UBL中心主任age.sinuj足Y`). The callback is told layer `il` is done while it is still
running, and refills the slot the running layer may still read.
`ggml_backend_cuda_synchronize` is `cudaStreamSynchronize(cuda_ctx->stream())` - stream-scoped,
not device-wide.

Consequence: 115.59 is the speed of a broken pipeline, not a target. Any scheme that removes the
barrier must first prove the slot is not being read.

### 6.4 L2 is not thrashing

The expert read rate is 9.1 GB/s at 1 token (working set 8 KB, fits L2 128x over) and 9.9 GB/s at
512 tokens - effectively identical. So eviction is not the limiter. `dmon mem` at 8-15% agrees:
DRAM is not saturated.

### 6.5 The kernel already tiles tokens into J

`M = ntoken * n_expert_used`. At ub512 that is 5120 tokens over 512 experts = 10 per expert, and
the MMQ tile is `J = 128`. So the kernel already tiles tokens; a "token tiling" improvement is
redundant. `ntx = (ncols + J - 1) / J`; at ub512, ntx = 4, which duplicates the weight read
1.60x. A **larger** token tile would fix that; token tiling would worsen it.

### 6.6 Padding waste

Routing averages **10 tokens per expert** while the tile width is **J = 128**. The kernel issues
about **50x more dp4a than the useful result needs** - 98% of the arithmetic is padding. That caps
power and compute utilization at the *work* level. Not the current bottleneck (the pipe itself is
only ~16% busy), but it bounds any future gain.

### 6.7 Launch shape (verified against the real kernel)

`mul_mat_q<ggml_type 12, J=128, fallback=false>`: grid (8192,1,1), block (32,8,1),
`nthreads=256`, occupancy 1, `I=128`, `J=128`, `stream_k=true`, ampere config table
(`ggml_cuda_highest_compiled_arch >= VOLTA`).

smem = 512 + 38912 + 18432 = **57856 B = 56.5 KB** -> 1 block/SM.
registers **REG:255** x 256 = 65280 of 65536 -> registers are a *second*, independent 1 block/SM
cap. 119 `LDG.E.CONSTANT.SYS` loads, all 32-bit. The qs loop is `I/(nrows*nwarps)` with
`nrows = 1` -> 16 trips.

### 6.8 The graph is free

New vs reused graph is not the cause. `LLAMA_PREFILL_STREAMING_NOCOPY=1` keeps all 48 splits and
barriers: p512 reused = 7947 ms actual vs 7890 ms predicted (extra 0.01x); p512 new = 15382 ms
(extra 2.02x). The pp2048 cliff is the new-graph path and it is streaming-only: reuse ON 84.80 vs
OFF 49.48 at p1024; the baseline is unaffected (38.18 vs 38.14). After ubatch 1 the causal mask
shape changes, so `can_reuse()` returns false every ubatch.

---

## 7. PENDING ISSUE: pp1024 as 2x `-ub 512` in one run is half the speed of two pp512 runs

Not solved. Recorded here so it is not lost.

### The observation

Measured this session, build `27543a7ae`, `-lm none`, `-r 2` or `-r 4`:

| run | config | t/s |
|---|---|---:|
| **no streaming** | `-ps 0 -p 512 -b 512 -ub 512` | **36.52** |
| **no streaming** | `-ps 0 -p 1024 -b 1024 -ub 512` | **36.70** |
| streaming | `-ps 1 -p 512 -b 512 -ub 512` | 63.25 |
| streaming | `-ps 1 -p 1024 -b 1024 -ub 512` | **32.46** |
| streaming | `-ps 1 -p 1024 -b 1024 -ub 1024` | 82.34 |

`-ps 0` is flat: 1024 tokens as 2x512 (36.70) runs at exactly the same rate as 512 tokens
(36.52). Splitting costs nothing without streaming.

`-ps 1` is not flat. 1024 tokens as 2x512 (32.46) runs at roughly **half** the rate of
1024 tokens as 1x1024 (82.34) - and also at half the rate of a *separate* 512-token run
(63.25). Same total tokens, same ubatch size, same cache. This is the exact form the user
reported.

Note the inversion it produces: with streaming, **one run of pp1024 as 2x512 (32.46) is
slower than doing nothing special at all (36.70)**. Streaming is a pessimisation in that
configuration, not an optimisation.

### Why it matters, and what NOCOPY is for

This is the reason `LLAMA_PREFILL_STREAMING_NOCOPY=1` exists. It **literally skips copying any
weights from CPU RAM to VRAM**, so the run measures only the compute side. It was written to
isolate whether the loss lives in the copy path or in the compute path for the 2x512 case.

**The NOCOPY results serve no other purpose.** They are a debugging probe for this specific
question. They are **not** a valid compute baseline for any other conclusion, because with NOCOPY
the slots are zero-filled and never refilled, so the kernel does different work (11.3 ms reading
55.7 MB, versus 69 ms reading 220 MB in the streaming case). Removing the copy changes what the
kernel does, not just how fast it runs. This was already argued in the folded-in notes; it is
restated here because it is easy to reach for NOCOPY as a "pure compute" number and it is not one.

### What is already on record about it

- The `-ub` smaller than `-b` case is pathological, not gradual: 82.34 -> 32.30 is a 2.6x cliff.
- The pp2048 cliff is the new-graph path and is streaming-only (section 6.8). The 2x512 case has
  the same flavour - a second ubatch in the same run behaves differently from the first - so the
  graph-reuse path is the first place to look.
- `ngl 1` versus `ngl 99` makes no difference to prefill, **but that was a claim taken from runs
  that never completed.** Both `-ngl 99` runs here died in `llama_init_from_model`; see below.

### `-ngl 99` is not a control run, it is a VRAM-infeasible run

At pp1024 the 2-slot configuration at `-ngl 99` **cannot be constructed at all** on this card:

```
done_getting_tensors: tensor 'blk.0.ffn_down_exps.weight' (q6_K) (and 143 others)
    cannot be used with preferred buffer type CUDA0, using CUDA_Host instead
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 619.52 MiB on device 0: cudaMalloc failed: out of memory
graph_reserve: failed to allocate compute buffers
llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers
```

The MoE tensors cannot live in VRAM even at `-ngl 99` (3716 MiB card, 45.17 GiB model), so they
land on `CUDA_Host`, and the streaming init is guarded on the streamed tensors *not* being host
resident - so the 2-slot path still allocates 2x996 MiB of slots, on top of a 619.52 MiB compute
buffer, and fails. This is a **structural** limit, not a bug: streaming and `-ngl 99` are mutually
exclusive on this hardware. `-ngl 99` only serves as a control if the slots are disabled.

The 1-slot serial mode fits, and it is the useful comparison: it moves the **whole layer including
attention** (there is no host-resident attention left to keep), 78.7 ms per refill at 12.35 GB/s.

### The control the user actually wants: compute without any CPU weight traffic

Serial `-ngl 99` gives exactly that, and it should be read as a **hypothetical** result - it is what
the reference config would look like if all weights were in VRAM:

| run | serial phases (per ubatch) | copy / compute |
|---|---|---|
| `-ngl 1 -ps 1` serial | refill 3720 ms (80.9 ms/layer) + compute 6064 ms (131.9 ms/layer) | 38% / 62% |
| `-ngl 99 -ps 1` serial | refill 3620 ms (78.7 ms/layer) + compute 9798 ms (213.0 ms/layer) | 27% / 73% |
| overlapped `-ngl 1 -ps 1` | refill+compute inside 82.49 t/s | - |

| run | result |
|---|---:|
| `-ngl 1 -ps 1` overlapped (reference) | 82.49 +- 0.05 t/s |
| `-ngl 1 -ps 1` serial | 81.36 +- 0.06 t/s |
| `-ngl 99 -ps 1` serial | 74.52 +- 0.24 t/s |
| `-ngl 99 -ps 1` overlapped | **does not fit** (OOM at context creation) |

**Processing the whole layer on the GPU (213.0 ms/layer) is slower than processing it with the
attention kept on the CPU while the weights are posted in (131.9 ms/layer).** The compute is 61%
slower, and the result is 9.6% slower end to end despite removing the attention host traffic (78.7
vs 80.9 ms of copy). So the PCIe link is not the constraint the design assumes; the **GPU work per
layer is**. Note also that `-ngl 99` moves *less* per refill (996.00 vs 1022.41 MiB, the attention
is already resident) yet computes 213 ms/layer against 131.9 ms/layer - the extra GPU-resident
attention is itself the expensive part.

### Status

**Open. To be taken up again later.** Do not spend time on it until the stall investigation
(section 8) has produced a result, because both may share a root cause in the graph-reuse path.

---

## 8. Open questions, in priority order

1. **Why is the GPU at 100% occupancy but drawing less power than a spin loop that moves zero
   bytes?** Both the idle theory and any throttling theory are dead (section 4, 4.1). Section 4.2
   now splits copy from compute directly and shows **both phases draw 18.8 W**, so the copy is
   ruled out as the cause and the question is entirely about what the SMs wait on during compute.
   The right instrument is `smsp__warp_issue_stalled_*` (ncu group C) on the real kernel, on a
   known-throughput config.
2. **Re-check T2.5.3** (slots not re-seeded on the reused-graph path). It is now believed to be a
   **false alarm** (section 9) - it was inferred from NOCOPY runs, which zero-fill the slots and
   produce wrong output by design. No correctness failure has ever been observed without a debug
   flag that is documented as breaking the output. It needs a clean re-test or deletion, not a fix.
3. **Build the valid copy control** (T2.2.1): refill the slot from a VRAM-resident copy
   (host -> VRAM once, then VRAM -> VRAM). NOCOPY is not it.
4. **The 2x512 single-run loss** (section 7). Blocked on 3.
5. **Why does GPU-resident attention cost 81 ms more per layer than host-resident attention**
   (section 7, 213.0 vs 131.9 ms/layer)? This is a new question the `-ngl 99` serial run raised
   and it is not explained by the copy.

---

## 9. Retracted / withdrawn (do not cite these)

- **"The GPU is idle 88.5% of the wall time"** - retracted. See section 4. Source run was 11.94 t/s.
- **"32-bit loads plus 1 block/SM are the smoking gun; vectorising the load is worth 2.50x"** -
  withdrawn. `tmp/prof/mmqstruct.cu` passed the tile size as a **runtime** argument, so its load
  loop could not unroll and each thread had one 4-byte load in flight. `tmp/prof/unroll.cu` shows
  the same structure with a **compile-time** trip count reaches 118.1 GB/s at 1 block/SM. SASS
  disassembly of the real kernel proves the `#pragma unroll` is fully honoured: 7944 straight-line
  instructions, only 4 backward branches (2 are guards), no load loop at all, 18 independent
  32-bit loads back to back, median load-use distance 145 instructions, REG:255, zero predicated
  loads. **Vectorising is worthless - it is already 4-way unrolled.**
- **"48 refills vs 94 tensors per layer is a 2x discrepancy"** - wrong. A refill is per **layer**,
  not per tensor: one call moves the whole 1022 MiB layer. One pass = 48 refills / 48 splits for
  48 layers. The earlier "94 refills" was a log bug: counters are reset and slots re-seeded only
  on the new-graph path, so a reused-graph line printed the *previous* ubatch's accumulated counts.
- **"T2.5.3: the slots are never re-seeded on the reused-graph path, so layers 0/1 can read stale
  data"** - **believed to be a false alarm, and no correctness failure has been observed without a
  debug flag.** It was inferred from the NOCOPY runs. `LLAMA_PREFILL_STREAMING_NOCOPY=1` skips the
  copy by design and the slot keeps whatever it held, so wrong output there is the documented
  expected behaviour of the flag, not evidence of a bug. The supporting correctness runs were
  never actually captured either: `results/archive/raw_logs/correctness_multibatch.txt` and
  `results/archive/prefill_streaming_v1/correctness_80b.txt` both contain **empty** baseline and
  streaming sections, while the run that did produce text
  (`results/archive/raw_logs/correctness_repeat.txt`) **passes** - baseline and both streaming
  responses are `The capital of France is Paris.` Do not cite T2.5.3 as a real bug without a
  clean re-test.
- **"`ngl 1` versus `ngl 99` makes no difference to prefill (52.32 vs 53.12)"** - cannot be used.
  The `-ngl 99` configuration does not fit on this card (section 7) and those runs died at
  context creation. The comparison has to be made with the serial 1-slot mode, where it does
  fit, and there it shows a 9.6% difference in favour of `-ngl 1` (81.36 vs 74.52).

---

## 10. Origin and design intent (harvested from the original notes)

Goal: improve prompt processing for large MoE models that do not fit in VRAM, by keeping only two
complete transformer-layer weight sets in VRAM - layer N being computed, layer N+1 being
transferred. The unit of transfer is the **whole layer including all its experts**, not individual
experts, because at prefill batch sizes routing selects nearly every expert anyway.

The implementation rests on three observations from the original note:

- With a low `-ngl`, the graph is already built up front, so the weight pointers for each layer can
  be repointed at whichever slot holds that layer.
- MoE already needs a per-layer sync point to route experts, which is where the slot handoff
  (free / filled) can be signalled.
- Setting ubatch == prompt size (up to ~4096) forces every layer through the most efficient shape.

The original hypothesis for the low utilization, worth restating because section 8.1 is still
testing it: "we are only utilizing the GPU cuda cores around 15-20% when we get ~100 t/s. I have a
suspicion that the cause is severe cache miss - because of the large micro-batch size ... this
severe cache miss could also explain why the copy stalls over PCI as soon as we start the
calculations."

Section 6.4 measured this and found the L2 read rate is unchanged between a 1-token and a 512-token
working set, so eviction is **not** the limiter - the DRAM utilization of 8-15% agrees. The
hypothesis is not supported, but the underlying question (what are the warps waiting on) is the
same one in section 8.1.

Build commands used throughout (from the same note):

```sh
cmake -B build -DLLAMA_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build build --config Release -j$(nproc)
```

---

## 11. Lessons that generalise

1. Any benchmark number must reproduce at the same config in the same session before it is used in
   a derivation.
2. A derived ratio needs a clean measurement window (`-r 4` plus a power trace, not `-r 1`).
3. Cross-check a derived quantity against a direct instrument (power, dmon).
4. Pin ONE reference configuration; change one flag at a time; do not silently drop flags.
5. A synthetic that shares a kernel's SHAPE but not its COMPILE-TIME CONSTANTS is not a
   reproduction. (This single mistake cost the load-width detour.)
6. Before optimising anything, measure what fraction of wall time it occupies.
7. **Read the feature's `git diff` against the base branch before touching the code.** The whole
   surface of this feature is 15 files and 821 added lines, and the diff took seconds to read
   against ~1 minute of model loading per trial-and-error iteration. The touch points are
   `include/llama.h`, `common/arg.cpp`, `common/common.{h,cpp}`, `src/llama-cparams.h`,
   `src/llama-graph.{h,cpp}`, `src/llama-model.{h,cpp}`, `src/models/{models.h,qwen3moe.cpp,
   qwen3next.cpp}`, `src/llama-context.{h,cpp}` and `tools/llama-bench/llama-bench.cpp`. Notably
   **the feature does not touch the arch loaders or `ggml-cuda` at all** - it is entirely a
   context-level redirection plus a `llama_layer` threaded into the qwen3 graph builders.
