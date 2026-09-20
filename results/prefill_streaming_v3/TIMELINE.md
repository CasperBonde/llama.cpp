# Full time-line of the prefill ubatch

Measured 2026-09-20. Model `Qwen3-Next-80B-A3B-Instruct-Q4_K_M`, serial measurement window
(`LLAMA_PREFILL_STREAMING_SERIAL=1`), `-ncmoe 99 -lm none -p 1024 -b 1024 -ub 1024 -r 1 -t 6
-ps 1`. The numbers are the **second** pass of the ubatch (the reused-graph pass, which is the one
llama-bench reports).

Every row carries its t/s, because a time-line that does not sum to a plausible t/s is worthless.

## 0. The correction that made this document necessary

The earlier finding was written as "the per-split synchronize is 66-73% of the ubatch". **That
label was wrong, and it was wrong in a way that inverted the conclusion.** The 8690.9 ms I called
"sync time" is not wait time. It is GPU execution time that a CPU-side timer was wrongly credited
with.

The proof is two measurement points in the same step of the same loop, on the same stream:

| build | CPU-side `ggml_backend_synchronize` | GPU-side event on the same range |
|---|---:|---:|
| before the event probe | 8690.894 ms | (not measured) |
| after the event probe | **0.547 ms** | **8690.926 ms** |

Same 290 ranges, same 679 calls, same work. The old timer sat directly on the stream sync, so it
absorbed the GPU execution and reported it as host wait. Once an event records on the stream, the
event wait absorbs the GPU execution (8690.9 ms) and the stream sync right behind it finds the
stream already drained (0.547 ms, max 0.007 ms).

So the answer to the question that started this is: **the CPU is waiting for the GPU to finish
calculations it has already started.** It is not the GPU waiting for the CPU, and it is not host
bookkeeping that got mislabelled. The host is essentially never idle: see item 2 in every row
below, which is the entire non-GPU, non-copy remainder and never exceeds 1.5%.

## 1. Legend

| item | what it is | source |
|---|---|---|
| 1. GPU | the range's own execution time on the device | `ggml_backend_event_record` + `ggml_backend_event_synchronize` around each submitted range (`LLAMA_SCHED_SYNC_AUDIT=1`) |
| 2. CPU | host work outside the copy: range scan, `callback_eval`, graph submit | callback counters, `LLAMA_PREFILL_STREAMING_DEBUG=1` |
| 3. wait | time the host blocks and the device is NOT executing | ubatch minus items 1, 2 and 4 |
| 4. copy | the blocking slot refill | `t_fill` bracket in `prefill_streaming_fill_slot` |
| 5. t/s | llama-bench `pp1024` | the table line |

The window is the same in every row: `t_compute_us`, wall time of the ubatch from the first
callback to the end of the graph compute.

## 2. `-ngl 1`, barrier on - serial (reliable)

t/s **81.45**, ubatch 12551.6 ms.

| item | ms | share |
|---|---:|---:|
| 1. GPU range execution | 8645.0 | 68.9% |
| 2. CPU (range scan 149.9 + submit 43.7) | 193.6 | 1.5% |
| 3. wait | ~0.0 | 0.0% |
| 4. blocking slot copy (46 x 80.7 ms) | 3713.0 | 29.6% |
| **total** | **12551.6** | **100%** |

Item 1 is 68.9% of the ubatch and it is genuine device work. Item 3 is zero because the event
wait, which runs first, already drained the stream before the barrier sync measured 0.547 ms.

## 3. `-ngl 99`, barrier on - serial (reliable)

t/s **74.95**, ubatch 13655.9 ms.

| item | ms | share |
|---|---:|---:|
| 1. GPU range execution | 10004.2 | 73.3% |
| 2. CPU | 20.3 | 0.1% |
| 3. wait | ~0.0 | 0.0% |
| 4. blocking slot copy (46 x 78.9 ms) | 3631.5 | 26.6% |
| **total** | **13655.9** | **100%** |

## 4. The red flag, quantified

Same graph, same ~5000 nodes, same tokens. The only change is where the weights live.

| | `-ngl 1` | `-ngl 99` | ratio |
|---|---:|---:|---:|
| 1. GPU range execution | 8645.0 ms | 10004.2 ms | **1.157** |
| ranges submitted | 290 | 49 | 0.17 |
| nodes submitted | 5059 | 4999 | 0.99 |
| GPU ms per node | 1.709 | **2.001** | **1.171** |
| 4. copy | 3713.0 ms | 3631.5 ms | 0.978 |
| ubatch | 12551.6 ms | 13655.9 ms | 1.088 |
| t/s | 81.45 | 74.95 | 1.087 |

The t/s ratio (1.087) matches the ubatch ratio (1.088) to 0.1%, so the ubatch accounting is sound
and the `-ngl 99` slowdown is real. It sits **entirely in item 1**: the same node count costs
17.1% more device time per node when the weights are GPU resident. Items 2, 3 and 4 are flat or
slightly better at `-ngl 99`.

That is the four-reason contradiction, now reduced to one number: **per-node GPU time is 17.1%
higher at `-ngl 99`.** Everything else about the `-ngl 99` run is better.

## 5. `-ngl 1`, barrier off - serial (timing only, output wrong)

t/s **115.46**, ubatch 8713.2 ms.

| item | ms | share |
|---|---:|---:|
| 1. GPU range execution | **not measured** (probe not in that build) | - |
| 2. CPU + 3. wait + any GPU tail | 5005.7 | 57.4% |
| 4. blocking slot copy | 3707.5 | 42.6% |
| **total** | **8713.2** | **100%** |

Usable: t/s 115.46 implies 8.869 s per ubatch against 8.713 s measured, 1.8% agreement. The gain
is real and reproducible (115.59 in an earlier session, by a different route).

## 6. `-ngl 99`, barrier off - serial (NOT usable)

| | pass 1 | pass 2 |
|---|---:|---:|
| ubatch | 6708.1 ms | 3647.8 ms |
| t/s implies | - | 8747 ms |

The two passes disagree by 1.84x, and the reported t/s implies 8747 ms against a measured 3648 ms
(2.4x). This run contradicts itself, so no time-line can be drawn from it. With the barrier gone
the blocking copy overwrites slot storage that a still-running range reads, and at `-ngl 99` the
ranges are long enough to make that race change the timing run to run. **Discard this row.**

## 7. What the time-line says

- Items 2 and 3 are ~0 everywhere. The host is not the bottleneck, and it is not stalling.
- Item 1, the GPU, is the whole ubatch: 68.9% to 73.3%.
- Item 4, the copy, is ~28% and runs strictly serial with item 1, because the serial window blocks
  on it. At `-ngl 1` removing the barrier gives back 3838 ms of an 12551.6 ms ubatch, which is
  103% of the copy time: in this window the barrier's entire cost is that it forbids the copy from
  overlapping the compute. In the normal two-slot window the copy is already asynchronous, which
  is why overlap alone bought only 1.4% there.
- The `-ngl 99` problem is not the barrier and not the copy. It is that the device spends 17.1%
  more time per node.

## 8. Missing information

1. **Item 1 for the barrier-off runs.** The probe was added after those runs, so the barrier-off
   time-line has no GPU number and items 2 and 3 cannot be separated. One re-run with
   `LLAMA_SCHED_SYNC_AUDIT=1` on the current build fills this.
2. **Item 2 split out.** The callback counters (`callback`, `wait_slot`, `scheduler host`,
   `submit+sync`, `blocking copy`) exist but were not printed in the event-probe run, so item 2 is
   currently a remainder. Adding `LLAMA_PREFILL_STREAMING_DEBUG=1` costs nothing and splits it.
3. **What is inside the 10004 ms.** The GPU time is known to be 17.1% higher per node at `-ngl 99`
   but nothing yet says which ops or which nodes. Needs either a per-range node list with timings,
   or an external trace of the launch gaps. `nsys` cannot run here (its importer is missing), so
   this needs the in-tree equivalent.
4. **The `-ngl 99` barrier-off row.** Needs a reliable configuration, or it stays discarded.

## 9. Method note

The event probe had two faults before it worked. `ggml_backend_event_new` returns `NULL` for a
device with no `event_new` iface, and `ggml_backend_event_record` asserts when the *backend being
recorded* has no `event_record` iface. The probe must therefore pick a backend that can create
events and must skip the ranges that run on CPU or `CUDA_Host`. Both are handled in
`ggml/src/ggml-backend.cpp`.
