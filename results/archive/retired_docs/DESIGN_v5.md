# Prefill weight streaming: revised design (v5)

Supersedes the design notes in `DoubleBuffer_Prefill_Design*.md` and the implementation on
`prefill_double_buffering2`. Written after measuring the current implementation, not before.

All numbers below were measured on this checkout.

| | |
|---|---|
| GPU | NVIDIA Quadro T1000, 3716 MiB usable, sm_75, 1 copy engine per direction |
| Model | Qwen3-Next-80B-A3B-Instruct-Q4_K_M, 48 layers, 45.17 GiB, `-ngl 1 -ncmoe 99` |
| Host | 12 cores, Xeon E-2276M, 62 GiB RAM, pageable, no NVLink |
| Build | `f4c71c721` |

---

## 1. Facts established by measurement

### 1.1 The reported async fill rate is not a rate

`LLAMA_PREFILL_STREAMING_DEBUG=1` prints the fill as `48 refills, 45842.96 MiB, 175.699 ms
total (273.59 GB/s)`. 273 GB/s over PCIe is impossible on this machine. That number is an
enqueue time, not a completion time. It must never be used as a copy rate.

### 1.2 The true cost of the fill

`LLAMA_PREFILL_STREAMING_SYNC=1` forces a blocking fill, so the timed region covers real
completion.

| ubatch | refills | bytes | blocking fill | implied rate |
|---:|---:|---:|---:|---:|
| 1024 | 48 | 45842.96 MiB | 3909.2 ms | 11.97 GB/s |
| 1024 | 48 | (repeat) | 3893.7 ms | 12.35 GB/s |
| 1024 | 48 | (repeat) | 3909.2 ms | 11.97 GB/s |

Two independent runs agree. **The fill costs about 3.9 s per ubatch, i.e. about 81 ms per
layer.** That is the number the design has to hide.

### 1.3 Compute alone

With the correctness barrier removed the same ubatch runs in 8.8 s to 9.8 s and reaches
115.6 t/s at pp1024 and 111.5 t/s at pp2048. The output is garbage, so this is not a
configuration anyone can ship, but it is a valid upper bound: **the GPU can do the whole
layer in about 8.8 s if it never waits.** 45.86 GiB of weights per 8.8 s is 5.6 GB/s, which
is the PCIe bandwidth this machine actually has.

### 1.4 The three numbers that matter

| quantity | value |
|---|---|
| compute only (upper bound, incorrect) | 8.8 s / ubatch |
| copy only (true) | 3.9 s / ubatch |
| correct current implementation, single ubatch | 11.8 s / ubatch |
| correct current implementation, multi ubatch | 20.0 s / ubatch |

Perfect overlap would give `max(8.8, 3.9) = 8.8 s`. The single-ubatch case is at 11.8 s,
which is 3.0 s worse. The multi-ubatch case is at 20.0 s, which is 11.2 s worse.

---

## 2. Why the current implementation cannot overlap

### 2.1 The fill is issued from inside the compute submission, on a stream the compute then waits on

The eval callback is invoked from inside `ggml_backend_sched_compute_splits`
(`ggml/src/ggml-backend.cpp:1802`), which is called from `ggml_backend_graph_compute_async
(sched->backends[0])`, which is the same backend whose `stream()` was just used to enqueue
the layer. So the whole callback chain runs with the compute stream still being filled.

The dependency the design intends is:

```
copy(layer il)  ->  compute(layer il)
```

That dependency is implemented as `ggml_backend_event_wait(prefill_streaming_backend, ev)`
= `cudaStreamWaitEvent(compute_stream, ev)`. This is correct and necessary. The problem is
not the dependency, it is that everything else is designed around it as if the copy were free.

### 2.2 The copy is up to 288 separate blocking driver calls per layer pair

`prefill_streaming_fill_slot` loops over the slot mapping and issues one
`ggml_backend_tensor_set_async` per tensor. For the 80B model one layer holds about 144
tensors, so one refill is 144 calls, and a ubatch is 48 refills, so about 2300 calls. Each
one goes:

```
ggml_backend_tensor_set_async
  -> backend->iface.set_tensor_async
     -> ggml_backend_cuda_set_tensor_async      (ggml-cuda.cu:2440)
        -> cudaMemcpyAsync(...)
```

Each call is small, and each call has a driver cost that is not amortised. This is the most
likely reason the achieved rate is 12 GB/s and not the 20+ GB/s a single large contiguous
transfer would give on this link.

The slot is already one contiguous device buffer with a precomputed offset table
(`prefill_streaming_maps`), so the fix is available and cheap: copy one layer as a single
range. The blocker is that the tensor offsets are not necessarily contiguous, so this needs
a per-layer contiguity check plus a contiguous fast path, falling back to the per-tensor
loop when the layout is fragmented.

### 2.3 The source memory is pageable

The host weights live in the CPU backend buffer, allocated with `ggml_aligned_malloc(size)`
(`ggml/src/ggml-backend.cpp:2437`), which is `posix_memalign` plus an alignment header. That
is pageable memory.

`cudaMemcpyAsync` from pageable memory is not the fast or the predictable path. The driver
must stage through an internal pinned bounce buffer, and for pageable H2D the transfer is
not guaranteed to make progress independently of kernel execution. The whole point of a
prefetch is that it runs ahead; if the copy engine does not make progress while kernels run,
the prefetch turns into a serialised transfer that simply blocks the next layer, which is
exactly the observed 20.0 s.

This is testable in isolation and is the cheapest experiment in this document. See 6.1.

### 2.4 The barrier is required for correctness, and it is what makes the bug in 2.5 fatal

The layer-boundary callback relies on `ggml_backend_synchronize(split_backend)` after every
split (`ggml-backend.cpp:1833`). Removing it raises throughput to 115 t/s and produces
garbage output, because:

- slot index is `il % 2`, so layer `il` and layer `il + 2` use the same slot
- `il + 2` is copied into that slot from the `ask == false` callback
- that callback runs immediately after the layer's range is *enqueued*, not completed
- so without the barrier the copy overwrites the weights of the layer that is still running

The barrier is therefore load bearing. It must stay.

### 2.5 There is no host-side agent, so there is no ping-pong

This is the structural reason the intended schedule does not happen.

The intended schedule is host side work against device side work, the same shape the MoE
path already uses:

| | device (GPU) | host (CPU) |
|---|---|---|
| step 1 | idle | copy layer 1 into slot 0 |
| step 2 | compute layer 1 from slot 0 | copy layer 2 into slot 1 |
| step 3 | compute layer 2 from slot 1 | copy layer 3 into slot 0 |
| ... | | |

What the code actually does is a re-entrant callback inside the submission of one layer,
executing on the calling thread, which has nowhere to do work except issue another copy
call. There is no thread that owns the copy, so there is no thread that can be told to stop
or to run ahead, and no thread to wait on. The copy is not parallel; it is callable.

The consequence is visible in 1.4. A single ubatch gets 11.8 s, which is close to the
11.8 s the current code happens to reach, but the multi ubatch case degrades to 20.0 s
instead of staying flat. That degradation, and the bimodal 11.8 s / 20.0 s split inside a
single process, are consistent with a copy that sometimes manages to stay in flight and
sometimes does not, with nothing in the code that decides which.

---

## 3. Requirements for the revised design

1. The compute stream must never wait for a copy that has not started.
2. Every layer must be complete in a slot before its range is submitted.
3. The copy must run on a thread that is not the thread submitting compute, so it can be
   blocked and sequenced explicitly.
4. The copy must be a small number of large transfers, not thousands of small ones.
5. The first layer must be resident at ubatch start, and chunked early, because a single
   contiguous full layer costs 71 MiB and the first copy is on the critical path.
6. No change to model code and no change to the scheduler split loop. The slots are already
   reached by rewriting graph operands, which the current code does and which works.
7. The current barrier stays. Any redesign that needs it removed is rejected.
8. If any of this cannot be made to work, streaming must fail cleanly and fall back to the
   normal path rather than degrade it.

---

## 4. Proposed design

### 4.1 One contiguous resident slot per layer slot

Keep the current two-slot device allocation. Add to `prefill_streaming_maps` a per-layer
description of the largest contiguous run:

```
struct prefill_streaming_span {
    size_t dst_off;   // offset of the first tensor in the slot
    size_t src_off;   // offset of the matching tensor in the host layer buffer
    size_t size;      // total bytes when the layer is contiguous in both
};
```

If a layer is contiguous in the source and the slot offsets increase by exactly `ggml_nbytes`
for the sorted tensor order, one `cudaMemcpyAsync` replaces 144 calls. If not, fall back per
layer to the existing loop. No correctness change, no model change.

### 4.2 Two host threads and two semaphores

This is the ping-pong the design is named after. It is small: about 150 lines, all inside
`llama-context.cpp`, using `std::thread`, `std::mutex` and `std::condition_variable`.

```
copy thread                          compute thread (the caller)
-----------                          ---------------------------
loop:
  wait(slot_ready[slot])             submit layer il (compute stream)
  copy layer il+2 into slot          wait(slot_ready[il%2])            <- event wait on compute stream
  record event done[slot]            ...
  signal(slot_free[slot])            at boundary: signal(slot_ready[...])
```

The important part is the direction of the waits:

- the compute thread waits on a **device event** recorded by the copy thread, so the wait
  costs the host nothing and is enforced on the GPU
- the copy thread waits on a **host semaphore** signalled by the compute thread at the layer
  boundary, after the barrier

That is exactly the user's proposal: GPU signals "layer done" (as a host-side signal after
the barrier), CPU copies and signals "copy done" (as a recorded event), GPU starts.

The copy thread deliberately does not use `cudaMemcpyAsync` from the caller's context. It
calls `ggml_backend_tensor_set` on the copy backend, so the copy is a blocking call on a
dedicated thread rather than an async call whose completion nothing tracks. Blocking on a
thread is what makes the sequencing explicit and reproducible.

### 4.3 Why two threads and not more

The copy is bounded by PCIe, not by CPU throughput. One thread issuing large transfers
saturates the link. A second copy thread would contend for the same link and add a second
staging buffer. Start with one.

### 4.4 Chunked first fill

At ubatch start, layer 0 and layer 1 must be resident before any compute is submitted. A
full layer is about 955 MiB, which is 71 MiB per expert chunk. Copying all 955 MiB before
the first layer starts puts 71 ms on the critical path of every ubatch.

Instead, copy layer 0 in chunks in expert order and let the graph start after the chunks that
the graph actually needs for its first matmul. The `init_tensor` hook on the slot buffer is
the natural place: it is already called by `ggml_backend_sched_alloc_graph` once per ubatch
and it already receives the graph, so it can copy exactly the tensors that are inputs of the
first split, then let the callback thread take over.

This is deliberately last. It only matters once 4.2 works.

### 4.5 Fallback

`prefill_streaming_failed` already exists and already falls back. A failed precondition
(no accelerator, no event support, copy thread failed to start, slot allocation failed) must
disable streaming for the whole context and log once. It must never leave a slot half
filled and readable.

---

## 5. Rejected alternatives

| alternative | why rejected |
|---|---|
| Remove the split synchronize | produces wrong output. Measured. The barrier keeps layer `il` alive while layer `il+2` is copied into the same slot. |
| `-ot` to shrink the model and free VRAM for slots | zero sum with streaming, and costs 18 to 32 percent of generation speed. Measured. |
| Rely on `ggml_backend_tensor_set_async` as if it were a parallel copy | measurement shows the reported rate is an enqueue rate. See 1.1. |
| Skip only the first copy of each ubatch | the copy is per ubatch, fixed at 45.86 GiB, independent of ubatch size. Skipping it leaves stale weights in the slot. |
| More slots to avoid the `il % 2` reuse | slot count does not help. The problem is not slot reuse, it is that the copy does not run in parallel. |
| Change `ggml-backend.cpp` to give the callback finer-grained completion | touches the shared scheduler contract, which serves every backend. Only consider it if 4.2 cannot reach the target. |

---

## 6. Test plan, and the experiment that decides this

### 6.1 Experiment 1: is pageable memory the reason the copy is slow? (cheapest)

Pin the host weights once with `cudaHostRegister` and re-measure the blocking fill from 1.2.
If the fill drops well below 3.9 s per ubatch, pageable memory is a first-order cause and
pinning belongs in the design. If it does not move, focus on 4.1 and 4.2.

This needs about 20 lines in the CUDA backend and no design change. It must be tried before
any restructuring, because if pinning alone gets the fill under about 1.5 s, the overlap
problem in 1.4 largely disappears on its own.

Caveat: 45 GiB of pinned memory on a 62 GiB machine is tight. The experiment should pin only
the tensors that the streaming path actually reads, which is the same set that goes into a
slot.

### 6.2 Experiment 2: does one large transfer beat 144 small ones?

Same harness, but with the contiguity fast path from 4.1 forced on. Compare blocking fill
time at ub=1024. Expected to move; if it does not, the per-call overhead theory in 2.2 is
wrong.

### 6.3 Experiment 3: does the thread ping-pong hold flat?

Once 4.2 is in, run the sweep that exposed the problem:

```
-ngl 1 -ncmoe 99 -lm none -b 1024 -ub 1024 -ps 1 -r 1 -p 1024,1536,2048,3072
```

Target: flat, and at or below `max(compute, copy)` plus one copy. With the current numbers
that is 8.8 s plus roughly one layer copy, so about 9 s per ubatch, about 110 t/s.
The correctness check is the same one that caught the barrier removal: greedy output on
"The capital of France is" must be "The capital of France is Paris."

Every experiment must be run with `-lm none`, `-b` equal to `-ub`, one benchmark at a time,
and `llama-cli` always with `< /dev/null`. llama-bench with `-v` is required for the debug
lines, because llama-bench otherwise installs a null log callback and silently discards them.

### 6.4 What must be true before this is proposed upstream

- correct greedy output across ubatch sizes 1 to 4096, not just 512
- flat throughput across pp, including the multi ubatch case
- no regression when streaming is off
- debug env vars and the diagnostic counters removed
- a single commit that a human can review

---

## 7. Open questions

1. **Why is the single ubatch case 11.8 s and not 8.8 s?** 3.0 s is unexplained. Experiment
   6.1 or 6.2 should account for it.
2. **Why is the multi ubatch case 20.0 s and not 11.8 s?** The copy is identical per ubatch
   (45842.96 MiB each). Something about the second and later ubatches prevents the overlap
   that the first one partly has. The graph reuse branch does not reseed the slots, which is
   correct, but it also does not reset the diagnostic counters, so the per ubatch attribution
   of the copy is currently unreliable. Fix the counters before drawing conclusions here.
3. **Is the double buffer even needed if the copy runs on its own thread?** With an explicit
   copy thread the ordering can be expressed directly and the second slot may become
   unnecessary, which would halve slot VRAM and remove the `il % 2` aliasing that makes the
   barrier load bearing. Worth evaluating only after 4.2 works.
4. **Does the MoE expert path and the streaming path fight over PCIe?** Both move large
   amounts of data across the same link in the same direction. Not measured.
