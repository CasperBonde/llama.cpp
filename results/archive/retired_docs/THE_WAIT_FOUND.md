# The wait, found: the GPU is idle 88.5% of the wall time

> ## RETRACTED. Read `CORRECTION_gpu_idle.md` instead.
>
> Sections 1-3 and 6 are withdrawn. At the operating point (84.53 t/s) a direct
> measurement gives `utilization.gpu` = 99-100%, `dmon sm` = 100%, at 18.4 W. There is
> no idle window. The 88.5% came from a run whose own table says **11.94 t/s**, a config
> that normally yields 63 t/s - so the ratio described a broken run, not the
> performance under investigation.
>
> Retained as a record of the error.

Date: 2026-09-20. Source: `tmp/prof/inventory.txt`, an ncu application-replay run of
`llama-bench -p 512 -b 512 -ub 512 -ps 1 -r 1 -n 0 --no-warmup`, `-ngl 1 -ncmoe 99 -lm none`.

---

## 1. The measurement

Both numbers come from the same run, so they are directly comparable. No counter, no inference.

| quantity | value | source |
|---|---:|---|
| pp512 throughput | **11.94 t/s** | llama-bench table in the same log |
| wall time for one 512-token prefill | **42.9 s** | 512 / 11.94 |
| sum of all GPU kernel time | **4.94 s** | 900 rows of `gpu__time_duration.sum` |
| **GPU busy fraction** | **11.5%** | 4.94 / 42.9 |
| **GPU idle fraction** | **88.5%** | the rest |

**There is kernel activity on the GPU for 11.5% of the run. For 88.5% of the run, no kernel is
running at all.**

This is the wait. It has been the answer the whole time, and it was sitting in data already on
disk, unread, because I kept looking for a throughput explanation inside the kernels.

## 2. This settles the standing concern

The concern was stated as "there may be a simple wait somewhere that consumes 70-80% of the
compute time". It is **80-90%**, and it is not inside the expert kernel - it is time when the GPU
has nothing running at all. Everything I measured about `mul_mat_q` was a description of the
11.5%, not of the bottleneck.

This also explains, retroactively, why every throughput investigation failed to converge:

- the expert kernel is not slow per byte; it is a small part of a mostly idle run
- nothing was found in `mmq.cuh` because nothing is wrong with `mmq.cuh`
- dp4a at 16%, loads at 118 GB/s, and 1 block/SM were all red herrings: they describe a kernel
  that gets to run 11.5% of the time

## 3. Reconciles the `utilization.gpu` = 100% reading

`PHASE5_RESULTS.md` recorded `utilization.gpu` at a steady **100%** during streaming. That
contradicted "the GPU is idle", and I could not reconcile them. The resolution is that
`utilization.gpu` is not an issue-rate or even an SM-execution measure - it is satisfied when the
GPU has a context with work queued. A launched-but-stalled grid still reads as 100%.

The kernel-sum-over-wall ratio cannot be fooled that way: if 4.94 s of kernel time happens inside
a 42.9 s window, the remaining 38 s has no kernel executing. **Prefer the ratio.**

This is the same class of error as the ones in `MEASUREMENT_STRATEGY.md` section 1: a single
instrument that disagrees with a derived quantity, believed over the derivation. Here the derived
quantity is simple division of two numbers from the same log.

## 4. What to measure next, and it is cheap

The open question is now narrow and specific: **what is the GPU waiting for during the 88.5%?**
Candidates, in the order I would test them:

| # | candidate | why it is plausible | how to confirm |
|---|---|---|---|
| 1 | **per-layer host copy is serialized with the kernels**, so the GPU idles while 1022 MiB crosses PCIe at 11.8 GB/s | 48 layers x 1022 MiB = 48 GiB per ubatch = **4.07 s at 11.8 GB/s**, and the kernels are 4.94 s. Together 9.0 s of the 42.9 s. Does not explain all of it, but it is the right order and it is the designed mechanism | `LLAMA_PREFILL_STREAMING_TIMECOPY=1`, already implemented |
| 2 | **a host-side sync per layer** that waits more conservatively than needed | the streaming path calls `ggml_backend_event_wait` per layer | W3 in `TASKLIST.md`: count the syncs per pass |
| 3 | **the copy is much slower than 11.8 GB/s in this configuration** | if the copy runs at ~1.5 GB/s rather than 11.8, 48 GiB becomes 32 s and the whole 42.9 s is explained | measure the copy directly, `TIMECOPY` |
| 4 | **`-ngl 1` puts only one layer on the GPU**, so 47 of 48 layers run somewhere much slower | `-ngl 1` is in this run's configuration. This may make the 11.94 t/s figure not representative | re-run the inventory with a config whose throughput is already known |

Note candidate 4: this run used `-ngl 1 -ncmoe 99`, which is *not* the same configuration as the
64.5 t/s streaming numbers. The 88.5% idle figure is real for this config; **it should be
re-measured on a config with a known throughput before it is generalised.**

## 5. What is retracted

Two earlier "smoking guns" are withdrawn, both because my harness was slower for a reason of its
own rather than because llama.cpp was slow:

1. **The load-width / 1-block-SM claim** (34.2 GB/s reproduction). `mmqstruct.cu` passed the tile
   size as a **runtime** argument, so its loop could not unroll. With a compile-time trip count
   the same structure reaches **118.1 GB/s at 1 block/SM** - the full ceiling. The real kernel's
   loop is already `#pragma unroll` over a `constexpr` trip count. **Vectorising is worthless.**
2. **The config-table claim** (switch sm_75 to pascal-dp4a). Even taking the dispatch at face
   value, `I=64, J=128` still gives 1 block/SM, and every reconfiguration multiplies traffic.

Both are recorded in `RETRACTION_load_width.md`. [The answer was not in the kernel.]

## 6. The corrected picture

| layer | time | share |
|---|---:|---:|
| GPU executing kernels | 4.94 s | 11.5% |
| **GPU with nothing running** | **38.0 s** | **88.5%** |

Within the 11.5% that runs, the accounting already done still stands: `mul_mat_q` is 95.7% of
kernel time, the stream-k fixup is 0.0%, and 98% of the issued dp4a is padding
(`RETRACTION_load_width.md` section on padding). Some of that 4.94 s is worth reclaiming later,
but it is a 4.94 s problem inside a 42.9 s run.

**The bottleneck is the 38 s.**
