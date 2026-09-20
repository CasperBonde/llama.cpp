# The per-split synchronize is NOT a stall - retracted

Measured 2026-09-20, build `27543a7ae` + instrumentation. Model
Qwen3-Next-80B-A3B-Instruct-Q4_K_M, `-ncmoe 99 -lm none -p 1024 -b 1024 -ub 1024 -r 1 -t 6`,
`-ps 1`, serial measurement window (`LLAMA_PREFILL_STREAMING_SERIAL=1`).

> **RETRACTED by `TIMELINE.md`.** The heading and the central claim of this document were wrong.
> The 8690.9 ms counted here as "sync time" is GPU execution time, not host wait. An event placed
> on the same stream measures the same 8690.9 ms and drops the stream sync to 0.547 ms. The
> number was real, the label inverted the conclusion. The measurements in sections 1 to 3 below
> are still correct; the interpretation is not. See `TIMELINE.md` for the corrected time-line.

This was the answer I thought closed the question of why the card is resident and busy yet draws
18.8 W of a 50 W budget. It does not. That question is still open.

## 0. The claim, as it was made (now known to be wrong)

`ggml_backend_sched_compute_splits` calls `ggml_backend_synchronize(split_backend)` after
**every** graph range it submits, and I read the host as blocking there for 66-73% of the ubatch.
The time is real; the host is not blocking. It is waiting for the device to finish work it has
already submitted, which is exactly what a synchronize is for.

## 1. What the numbers in the table actually are

Instrumented with `LLAMA_SCHED_SYNC_AUDIT=1` (counts and times every synchronize the scheduler
performs at the split level).

| `-ngl` | sync calls | measured at the sync | ranges | callbacks | ubatch | share |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 679 | **8690.9 ms** | 290 | 290 | 13211.6 ms | 65.8% |
| 1 | 679 | 8646.3 ms | 290 | 290 | 12568.7 ms | 68.8% |
| 99 | 61 | **10065.6 ms** | 50 | 50 | 13764.6 ms | 73.1% |
| 99 | 61 | 10006.9 ms | 50 | 50 | 13660.3 ms | 73.3% |

These four numbers are correct. Both later measurements confirm them: the event on the same
stream reports 8690.9 / 8645.0 / 10059.5 / 10004.2 ms on the same ranges. The error was only in
calling them host wait. They are GPU execution time that the CPU-side timer absorbed.

Two things to note, both still true:

- The measured interval scales with the work in the range, not with any host cost.
- `-ngl 99` makes *fewer* but *much longer* intervals: 61 of **165 ms** average, against 679 of
  **12.8 ms** average at `-ngl 1`. That is what first drew attention here, and it is the red flag
  that `TIMELINE.md` section 4 quantifies: 17.1% more GPU time per node.

## 2. Removing the barrier

`LLAMA_SCHED_NO_SPLIT_SYNC=1` skips it. The output is wrong by design - the eval callback refills
slot storage that the running range may still read - so only the timing is meaningful.

| `-ngl` | barrier on | barrier off | delta |
|---:|---:|---:|---:|
| 1 | 81.45 t/s | **115.46 t/s** | **+41.8%** |
| 99 | 74.95 t/s | **117.08 t/s** | **+56.2%** |

The `-ngl 1` barrier-off number reproduces the 115.59 t/s recorded in the earlier session, from a
different measurement route. That is a useful cross-check: the effect is real and repeatable.

| `-ngl` | barrier | ubatch | copy | contract |
|---:|---|---:|---:|---|
| 1 | on | 12568.7 ms | 3726.8 ms | sync 8646.3 ms |
| 1 | off | 8713.2 ms | 3707.5 ms | sync 0 |
| 99 | on | 13660.3 ms | 3634.4 ms | sync 10006.9 ms |
| 99 | off | 3647.8 ms | 3624.4 ms | sync 0 |

At `-ngl 1` the copy is unchanged (3726.8 vs 3707.5 ms, 0.5%), so the barrier is not slowing the
copy - it is adding wall time around it. The ubatch drops by 3855.5 ms when the barrier goes.

## 3. Where the `-ngl 99` per-layer spike came from

Per-layer compute, in order, barrier **on**, last pass (`LLAMA_PREFILL_STREAMING_SERIAL_LIST=1`).
This was the observation that started this chase.

| layers | `-ngl 1` | `-ngl 99` |
|---|---:|---:|
| 0 | 125.5 | **463.4** |
| 1 | 125.5 | **463.9** |
| 2 | 125.3 | **493.0** |
| 3 (attn) | 151.7 | **446.6** |
| 4 | 125.5 | **391.2** |
| 5 | 125.5 | **394.6** |
| 6-45 | 124.7-125.8 (gdn), 151.0-151.8 (attn) | **170.4-177.6**, uniform |
| GDN mean | 125.27 | 214.76 |
| attn mean | 151.38 | 195.82 |

At `-ngl 1` **every** layer is reproducible to +-0.6 ms and there is no spike at all. At `-ngl 99`
layers 0-5 cost 391-493 ms and layers 6-45 settle to a flat 170-177 ms. The spike appears in both
passes, so it is not warmup.

The earlier conclusion drawn from the grouped means - "processing the whole layer on the GPU costs
61% more per layer, so GPU-resident attention is expensive" - was **wrong**. The grouped means were
an artifact of a 2.6x spike in the first six layers. The retraction was right; the reason given for
it here ("the spike is a barrier stall, not compute") was also wrong. The spike IS device work.

With the barrier off the `-ngl 99` ubatch is 3647.8 ms and the measured compute is ~0, because with
nothing forcing the host to wait the callbacks no longer bracket the GPU work. That is the tell:
**the per-split synchronize is what makes the callback interval a usable compute measurement.** All
per-layer compute numbers in this investigation are conditional on the barrier being present, and
each one therefore contains an unknown amount of host-block time. The event probe in `TIMELINE.md`
removes that condition.

## 4. Why I thought this was the power answer (retracted)

> **RETRACTED.** The mechanism below is wrong. The probe shows the device is executing during the
> measured interval, and the host-side remainder is under 1.5% of the ubatch, so the host is not
> withholding work and the GPU is not idling in that window. The t/s gains in section 2 are real;
> the explanation here is not. Where the energy actually goes is still open.

The SMs are not short of work; I claimed the host will not feed them. Every range is followed by a
blocking `cudaStreamSynchronize`, so the next range is submitted only after the previous one has
fully retired, plus whatever the eval callback does in between - and I claimed the GPU sits idle in
that window. **The probe refutes the last step.** The event records on the same stream and the
event wait absorbs the same interval, so the device is executing throughout it. The host is inside
the synchronize because the synchronize is doing its job.

The observations below are unaffected by the retraction - they are inputs, not conclusions:

- `utilization.gpu` 99-100%: a range is always resident while it executes.
- `dmon mem` 8-15%: nothing is bandwidth-bound.
- power flat at 18.8 W in both the copy and the compute phase (CALIBRATION 1f).
- the SM clock held at 1395 MHz: not throttling.

What they now have to be explained by is different, and harder: the device is busy executing, and
drawing 18.8 W of 50 W while it does so. That is the open question, not a solved one.

It also explains why the overlap question kept coming out flat. Serial 81.36 against overlapped
82.49 t/s (section 4.2) held the barrier constant, so both were starved equally. Overlapping the
copy cannot help while a 3855 ms barrier stall dominates a 12.5 s ubatch.

## 5. What this does and does not say

- It is **not** a request to delete the barrier. The barrier is a correctness requirement: the
  callback refills storage the running range may still read, and without the ordering the model
  emits garbage (`far of UBL中心主任age.sinuj足Y`, recorded in section 6.3).
- 115-117 t/s is therefore an **upper bound**: it is the speed of a broken pipeline. The real
  target is to reach it while keeping the ordering.
- The measured interval is **device execution**, so the barrier is not converting host block time
  into GPU idle time. The 3838 ms it costs at `-ngl 1` is the blocking copy's serialization: in
  the serial window the copy cannot overlap the compute while the barrier is obeyed, and 3838 ms
  is 103% of the 3713 ms copy. The two-slot window already runs the copy asynchronously, which is
  why overlap alone bought only 1.4% there.

## 6. Method notes

- `LLAMA_SCHED_SYNC_AUDIT=1` - per-split synchronize count and time, per graph compute.
- `LLAMA_SCHED_NO_SPLIT_SYNC=1` - skips the barrier. Measurement only; output is wrong.
- `LLAMA_PREFILL_STREAMING_SERIAL_LIST=1` - every layer in order, with type, so a spike or a ramp
  is visible. A grouped mean hides both.
- `-ngl 1,99` is a cross-product vector, so both `-ngl` values run inside one process after a
  single model load. Two configurations for one load.
- Scripts: `tmp/prof/split_sync_ab.sh`, `tmp/prof/ngl_layer_split.sh`.
- Logs: `tmp/prof/syncab/barrier_{on,off}.out`, `tmp/prof/serial/ngl_layer_split.out`.
