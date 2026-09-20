# Findings: overlapping the slot copy costs 3x, it does not hide it

Date: measured on `27543a7ae`. Model Qwen3-Next-80B-A3B-Instruct-Q4_K_M,
`-ngl 1 -ncmoe 99 -lm none`, GPU Quadro T1000 (3716 MiB, sm_75), PCIe Gen3 x16.

This supersedes the "new graph vs reused graph" framing, and CORRECTS the earlier claim in
section 3 below that the per-range `ggml_backend_synchronize` is the cost. The barrier is
where the host waits; it is not the cause.

---

## 0. The law

Copy per ubatch = 45.86 GiB at 12.4 GB/s = 3.7 s. Compute is measured directly by
`LLAMA_PREFILL_STREAMING_NOCOPY=1`, which skips the copy and keeps every split and barrier.

| case | compute | copy | compute+copy | actual | extra | extra/copy |
|---|---:|---:|---:|---:|---:|---:|
| p512 reused graph | 4163 ms | 3727 ms | 7890 ms | **7947 ms** | 57 ms | **0.01** |
| p512 new graph | 4170 ms | 3717 ms | 7887 ms | **15382 ms** | 7495 ms | **2.02** |
| p1024 new graph | 8516 ms | 3894 ms | 12410 ms | **20333 ms** | 7923 ms | **2.03** |
| copy-thread build | 8515 ms | 3900 ms | 12415 ms | **20485 ms** | 8070 ms | **2.07** |

- Serialized copy costs **1x**. Concurrent copy costs **3x** (1x of real transfer + 2x of damage).
- The extra is a fixed multiple of the copy and does NOT scale with the token count, so it is
  not the compute carrying a fixed overhead: the interference scales with the copy.
- The copy itself is NOT duplicated and NOT slower on the new path: same bytes (45 843 vs
  43 798 MiB) and the same blocking wall time (3716.8 vs 3716.9 ms). The 48-vs-46 refill
  difference is exactly the two synchronously seeded layers (2 x 1022 MiB).
- So: same work, same copy duration, same compute, 7.4 s more. The copy must be slowing the
  compute while it runs.

This also means the earlier DRAM saturation test (`dram_saturate.cu`, 108.7 GB/s kernel next
to a copy) is a FALSE NEGATIVE and must not be cited. It measured the COPY's rate while a
synthetic kernel ran. The damage is in the other direction: the kernels slow down. A pure
bandwidth kernel does not reproduce the contention behaviour of the real MoE kernels.

## 1. The graph is free, so new vs reused is not the cause

`LLAMA_PREFILL_STREAMING_NOCOPY=1` skips the copy but keeps all 48 splits and all 48
per-range barriers. p512, `-b 512 -ub 512`, one process:

| p512 | new graph | reused graph |
|---|---:|---:|
| with copy | **15 382 ms** | 7 947 ms |
| `NOCOPY`  | 4 172 ms | 4 161 ms |

Without the copy the two are identical. So the graph build/reuse is not the cost.

## 2. The host never blocks, so the wait is not the cause

`LLAMA_PREFILL_STREAMING_NOWAIT=1` (skip the slot event wait) changes nothing:
15 362 ms vs 15 357 ms. The callback host time is 0.6 ms total across the ubatch, and
`wait_slot` is ~0.1 ms. The stall is GPU-side.

NOTE: `NOWAIT` was documented in an earlier session but never actually implemented.
It was implemented for this measurement. The earlier "NOWAIT: 50.15 vs 50.19" result was
a no-op, as was `LLAMA_PREFILL_STREAMING_NO_THREAD` (also never implemented).

## 3. CORRECTED: the per-range synchronize is where the host WAITS, not the cause

Scheduler breakdown, `GGML_SCHED_SPLIT_TIME=1`, p512, 48 layer ranges:

| p512 | submit | per-range sync | bytes copied |
|---|---:|---:|---:|
| new graph | 594 ms | 11 336 ms (236 ms/layer) | 45 843 MiB |
| reused graph | 45 ms | 4 346 ms (91 ms/layer) | 43 798 MiB |
| `NOCOPY` | 182 ms | 3 914 ms | 0 |

"The time is inside the synchronize" is a CEILING, not a cause. The synchronize blocks until the
compute stream is empty, and the compute stream is running slower because the copy is in flight
(section 0). So this line measures where the host waits for the interference, not the interference
itself. It also cannot be the whole story on its own: the aggregate is 11 336 ms, which is not
2x the copy.

(Caveat on this number: it first read 0 because of a static-shadowing bug in my instrumentation.
After the fix it is the correct aggregate, but it is still a consequence.)

The barrier is `ggml_backend_cuda_synchronize` -> `cudaStreamSynchronize(cuda_ctx->stream())`,
i.e. stream-scoped on the compute stream. It does not drain the copy stream by itself.

`GGML_SCHED_NOSYNC=1` (skip that barrier) collapses new/reused to 5 309 / 5 183 ms. That shows
the serialization is carried by the barrier, but the barrier is unsafe to remove: it is the
ordering that stops the prefetch for layer il+2 from overwriting the slot layer il is still
reading (`(il+2)%2 == il%2`). Removing it was tested and produced garbage output.

## 4. The reused path is serialized (1x) and is the optimum

Per-layer budget, p512 reused path:

```
compute 4163 ms / 48 =  87 ms per layer
copy                    78 ms per layer
sum                    165 ms
observed               166 ms  (7947 / 48)
```

Serialized: each layer costs compute + copy, and the copy is not hidden, it is merely paid once.
This is the same result as `FINDINGS_no_overlap.md`. On the NEW path the layer budget is
~320 ms, i.e. 2x the copy on top of the serialized cost. Overlapping is not hiding the copy,
it is tripling it.

Consequence: the premise "hide the copy behind compute" is BACKWARDS on this box. Overlap is
worse than serialization. A two-ubatch p2048 pass would be ~15.8 s (129 t/s) if it could hold
the serialized state, instead of the measured 41.9 s.

## 5. Why this becomes the pp2048 cliff

- p512 `-b 512 -ub 512`: one ubatch per pass. Consecutive passes have the same mask shape, so
  `can_reuse()` succeeds and the pass runs SERIALIZED (1x). This is the 64.5 t/s case.
- p2048 `-b 1024 -ub 1024`: two ubatches per pass. The mask shape changes within the pass
  (1024 positions, then 2048) and also differs from the previous pass, so `can_reuse()` is
  false for EVERY ubatch. Every ubatch runs CONCURRENT (3x). This is the 48.9 t/s case.

`LLAMA_GRAPH_REUSE_DISABLE=1` reproduces it directly at p512 in one process:

| p512 | reuse ON | reuse OFF |
|---|---:|---:|
| baseline  | 38.18 | 38.14 |
| streaming | **64.42** | **33.49** |

The baseline is unaffected. Only the streaming path pays, because only it issues the copy.

A NEW GRAPH IS NOT THE CAUSE - it is the TRIGGER. Graph reuse is just what keeps the multi-ubatch
pass on the serialized path. So the defect is: the streaming path leaks into a state where the
copy is concurrent with compute, and any multi-ubatch pass lands in that state.

This is also the old "11.8 s vs 20.0 s bimodality" (measured again as 12.07 s vs 20.55 s,
ratio 1.70). One phenomenon, two symptoms: 12 s = serialized, 20 s = concurrent.

## 6. The one clean discriminating test (not yet run)

`LLAMA_PREFILL_STREAMING_SYNC=1` makes the fill BLOCKING. It removes the concurrency, so if the
3x law is right a new-graph ubatch should fall from ~15.4 s toward the serialized ~8 s. This uses
existing instrumentation only, no new code. p2048 should also recover toward ~15.8 s.

## 7. Why the barrier is hard to remove

The prefetch for layer `il + 2` writes slot `(il + 2) % 2`, which is **the same slot that
layer `il` just read** (`il % 2`). So the copy must not start until layer `il` has finished
on the GPU. The barrier is that ordering, done on the host: the callback runs after the
compute stream has drained, so by the time it issues the copy the hazard is gone.

Removing the barrier was already tried and produces garbage output, because the callback
then observes layer `il` before it has finished and refills the wrong slot. The barrier is
a correctness barrier.

Note that the hazard and the concurrency are the same mechanism: the copy can only be issued
once the compute stream is drained, and the drain is what the barrier does. So the two cannot
be fixed independently by adding slots alone - extra slots remove the ALIASING but the callback
still fires immediately after submit, so without the barrier it would refill a slot that is
still in flight. That needs both the refill target and the protocol changed.

## 8. Consequence for the design

- The copy cannot be made faster: it is pinned memory at 12.4 GB/s, 99% of the PCIe Gen3
  x16 practical ceiling.
- OVERLAPPING IT IS NOT THE GOAL. Serialized costs 1x, overlapped costs 3x. The reused path at
  7.95 s is already the optimum.
- The lever is therefore to keep multi-ubatch passes on the serialized path, i.e. prevent the
  streaming path from entering the concurrent state, or serialize the copy explicitly.
- A concurrent host copy thread is not the fix: it measured 50.20 t/s against 84.84 for the
  committed code, with the same 2.07x-copy signature. It forced the concurrent state.
- The honest position until this is fixed: the design wins exactly when there is one ubatch per
  batch and loses otherwise, and it breaks the documented "hide the copy" premise.

## 9. Status of the earlier claim in this file

The previous version of this file said the per-range synchronize "holds the time" (11 336 ms vs
4 346 ms) and that the fix is to move the ordering onto the device so the copy can be hidden.
That is WRONG on both counts and is corrected above. The synchronize is where the host waits for
the interference, not the interference itself; and moving the ordering onto the device would
INCREASE concurrency, which is the thing that costs 3x.

## 10. Measurement notes

- All A/B pairs above are `-ps 0,1` in ONE process. Single-run comparisons are not valid
  here; run-to-run spread reaches 2x.
- `nsys` cannot be used on this box. `nsys profile --trace=cuda,nvtx` runs but then fails
  to import: "Importer error status: The importer binary and its dependencies were not
  found." It leaves a `.qdstrm` with no stats and no timeline. `--trace=cudart` is
  rejected. In-code timers were used instead.
- llama-bench needs `-v`, otherwise the log callback is null and all `LLAMA_LOG_*` output
  is discarded.
- The per-ubatch counters used to leak across ubatches (the reset ran only on the new-graph
  path), which produced the bogus "94 refills, 96 layer splits" lines. The reset now runs
  on both paths.
