# Findings: why prefill streaming does not overlap

Date: measured on `f4c71c721` plus instrumentation. Model Qwen3-Next-80B-A3B-Instruct-Q4_K_M,
`-ngl 1 -ncmoe 99 -lm none -b == -ub`, GPU Quadro T1000 (3716 MiB, sm_75, 3 copy engines),
PCIe Gen3 x16.

This supersedes the reasoning in `DESIGN_v5.md` section 2.3, which guessed that pageable
source memory was the cause. It is not: the source is already pinned.

---

## 1. The copy is at the link limit, so it cannot be made faster

Measured with `copy_bench.cu` (2 GiB transfers, H2D):

| source | 1 block | 64 MiB | 16 MiB | 6 MiB | 1 MiB |
|---|---:|---:|---:|---:|---:|
| pageable | 9.58 | 9.73 | 9.56 | 9.16 | 6.65 |
| **pinned** | **12.42** | **12.47** | **12.38** | **12.23** | 11.33 |

PCIe Gen3 x16 theoretical is 15.75 GB/s, and 128b/130b encoding puts the practical ceiling
near 12.5 GB/s. **The pinned rate of 12.4 GB/s is 99% of that.**

The model's streamed weights already come from pinned memory: with `-ncmoe` the expert
tensors are overridden to buffer type `CUDA_Host`, which is allocated with `cudaMallocHost`
(`ggml/src/ggml-cuda/ggml-cuda.cu:1279`). Chunk size barely matters; even 1 MiB chunks hold
11.33 GB/s.

So there is **no headroom in the copy**. 45.86 GiB / 12.4 GB/s = 3.7 s, and the measured
3.9 s is that number. Any design that tries to speed the copy up is dead on arrival. The
only lever is hiding it.

The reported enqueue rate of ~274 GB/s in the debug log is a scheduling artefact and must
not be quoted as a rate.

---

## 2. The GPU can overlap a long kernel with an H2D copy

Measured with `overlap_bench.cu`: a spin kernel on stream A, 512 MiB across 8 chunks on
stream B.

| | |
|---|---:|
| kernel alone | 37.63 ms |
| copy alone | 43.08 ms (12.46 GB/s) |
| **both, overlapped** | **43.37 ms** |
| sum if serialized | 80.72 ms |

Overlapped is `max(kernel, copy)`, so the copy is fully hidden, and the copy still runs at
12.38 GB/s while the kernel runs.

That kernel is ALU only, which is not representative of a prefill GEMM, so the test was
repeated with `overlap_mem_bench.cu` against a kernel that saturates device memory
bandwidth. The copy engine and the SMs contend for the same memory subsystem there, which
is the honest comparison for a real GEMM.

| | |
|---|---:|
| kernel alone (256 MiB x 48 reps, 94.6 GiB/s device) | 126.82 ms |
| copy alone (512 MiB H2D) | 43.43 ms (12.36 GB/s) |
| **both, overlapped** | **115.08 ms** (kernel 115.08, copy 43.36) |
| sum if serialized | 170.25 ms |

The copy is still fully hidden, and it still holds 12.38 GB/s while the device memory bus is
saturated. **The hardware and driver overlap correctly, including under memory contention.
Bandwidth contention is not the cause.**

---

## 3. The split and barrier structure is almost free

With `LLAMA_PREFILL_STREAMING_NOCOPY=1` (copy skipped, 48 splits and 48 barriers kept):

| configuration | time | t/s |
|---|---:|---:|
| no copy at all, 48 splits, 48 barriers | **8511 ms** | **120.28** |
| barrier removed (earlier, incorrect output) | 8977 ms | 115.59 |
| with copy, async fill (current) | 12068 ms | 87.03 |
| with copy, blocking fill | 12494 ms | 81.95 |

The layer-boundary splits and the per-split synchronize cost about nothing. Splitting every
layer into its own range is 48 splits for 48 layers, exactly one range per layer, and that
is the correct granularity. **The per-layer split is not the problem and does not need to be
removed.**

---

## 4. The copy is 100% exposed

| | ms |
|---|---:|
| with copy | 12068 |
| without copy | 8511 |
| **difference** | **3557** |
| measured copy cost | 3908 |

3557 of 3908 ms appears directly in the wall clock. Worse, comparing the two fill modes at
the same ubatch:

| p | async | blocking | hidden by async |
|---:|---:|---:|---:|
| 512 | 7.98 s | 8.19 s | 0.21 s (2.6%) |
| 1024 | 12.13 s | 12.57 s | 0.44 s (3.6%) |

The async path hides only 3.6% of its copy. **The async copy is effectively synchronous, as
the user said.**

---

## 5. Yet the copy does complete during the ubatch

`LLAMA_PREFILL_STREAMING_DEBUG=1` now also reports how much copy work is still outstanding
when the ubatch ends, by timing a synchronize of the copy backend at that point:

```
async:    48 refills, 48 layer splits, copy drain at end 0.004 ms
blocking: 48 refills, 48 layer splits, copy drain at end 0.012 ms
```

Both are ~0. If the copies were queued and then run as one block at the end, this would be
seconds. So the copies are being consumed steadily across the ubatch, and the copy stream is
empty by the time the ubatch finishes.

So the copies are interleaved with the compute in time, and they still add their full
duration to the wall clock. That is only possible if the copy engine and the compute are not
actually running at the same time, even though section 2 shows that they can.

---

## 6. What this rules out

| hypothesis | verdict |
|---|---|
| Copy is slow because the source is pageable | **ruled out.** Source is `CUDA_Host`, `cudaMallocHost`, 12.4 GB/s. |
| Copy is slow because it is done in small chunks | **ruled out.** 1 MiB chunks still reach 11.33 GB/s. |
| Copy is slow because the slot layout is many small tensors | **weakened.** Per-tensor calls cost about 2% (12.4 vs 12.2 GB/s between 1 block and 6 MiB chunks). Not the 3.9 s. |
| The GPU cannot overlap compute and copy | **ruled out.** Section 2: 80.72 ms serial becomes 43.37 ms overlapped, and 170.25 ms serial becomes 115.08 ms against a memory bound kernel. |
| The layer splits / barriers are the cost | **ruled out.** Section 3: 8511 ms with all 48 splits and barriers. |
| CUDA graphs | **ruled out earlier.** Same result with `GGML_CUDA_GRAPHS=OFF`. |
| Slot duplication on later ubatches | **ruled out.** Copy bytes identical per ubatch. |
| The copy stream is the same stream as the compute | **ruled out.** `ggml_backend_cuda_context` owns `streams[]` and `ggml_backend_cuda_init` allocates a fresh context, so the copy backend has its own non-blocking stream. |

---

## 7. What is still open

The copy runs at link speed, the hardware overlaps, the splits are free, the streams are
separate, and the copy still lands fully on the critical path. Something couples the copy
and the compute so they do not execute at the same time.

The most likely place to look next, in order:

1. **The event direction.** `prefill_streaming_wait_slot` does
   `cudaStreamWaitEvent(compute_stream, copy_event)`. The copy event is recorded once per
   slot fill, after all tensors of that layer. If the compute stream must wait at every
   layer boundary and the copy for the next layer has not been issued far enough ahead, the
   GPU ends up alternating: copy a layer, compute a layer, copy a layer. That produces
   exactly the observed `compute + copy` total with a ~0 drain at the end. Instrument with
   per-layer `cudaEventElapsedTime` on both streams to confirm.

2. **Look-ahead depth.** The current design prefetches one layer ahead. With compute at
   ~177 ms per layer and copy at ~81 ms per layer there is already enough headroom, so depth
   may not help. Confirm with 1 before changing it.

3. **The split between the slot tensors and the graph.** The slot tensors are inputs to the
   first op of each layer. If the scheduler puts a copy of the slot tensor into the graph as
   an input, the scheduler's own input-copy path would run a second copy, and that one is
   definitely on the compute stream. Check `rewrote %d graph operands` against the actual
   number of ops per layer, and check whether the slot tensors appear in any split's
   `inputs[]`.

---

## 8. Reproduce

```
results/prefill_streaming_v3/copy_bench.cu         # link limit, pageable vs pinned
results/prefill_streaming_v3/overlap_bench.cu      # kernel vs copy overlap
results/prefill_streaming_v3/overlap_mem_bench.cu  # memory bound kernel vs copy overlap

LLAMA_PREFILL_STREAMING_DEBUG=1  ./build/bin/llama-bench -v -m <model> -ngl 1 -ncmoe 99 \
    -lm none -p 1024 -b 1024 -ub 1024 -ps 1 -r 1
LLAMA_PREFILL_STREAMING_NOCOPY=1 ...            # the copy is skipped, output is wrong
```

llama-bench installs a null log callback, so `-v` is required for the debug lines.
