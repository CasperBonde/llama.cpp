# Tasks: Diagnose & Resolve CUDA Layer Streaming Pipeline Idle Stalls

## Context
The GPU (T1000) exhibits a low duty cycle (~18W power draw out of 50W TGP) during layer-by-layer prefill, despite async weight pre-fetching for Layer N+1 on a separate DMA stream. The objective is to identify and eliminate host launch latency, pageable memory fallbacks, or implicit stream serialization in the codebase.

---

## Phase 1: Memory Allocation & Pinned Host Buffer Audit
- [ ] **1.1 Trace Weight Buffer Allocations**
  - Locate where host-side weight buffers used for layer double-buffering are allocated.
  - Verify that host memory is strictly allocated via `cudaHostAlloc()` or `cudaMallocHost()` with flags `cudaHostAllocDefault` or `cudaHostAllocWriteCombined`.
  - Ensure standard `malloc()`, `posix_memalign()`, or `std::vector` host buffers are **not** being passed directly to `cudaMemcpyAsync()`.

- [ ] **1.2 Verify Asynchronous Transfer Compliance**
  - Audit all `cudaMemcpyAsync` calls transferring weights from host RAM to VRAM.
  - If memory is allocated outside `cudaHostAlloc`, verify `cudaHostRegister()` is explicitly called on the address range before transfer.

---

## Phase 2: Stream Creation & Default Stream (Stream 0) Isolation
- [ ] **2.1 Audit Stream Initialization Flags**
  - Locate stream creation routines for both `stream_compute` and `stream_dma`.
  - Confirm both streams are explicitly instantiated using:
    ```cpp
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    ```
  - Ensure `cudaStreamLegacy` or default Stream `0` is never implicitly targeted.

- [ ] **2.2 Search for Implicit Default Stream Operations**
  - Grep the codebase for un-streamed CUDA API calls: `cudaMemcpy`, `cudaMemset`, or kernel launches missing stream parameters (`<<<grid, block, shared, stream>>>`).
  - Ensure intermediate kernels (e.g., RoPE, quant/dequant, gating reductions, vector math) explicitly pass `stream_compute` and do not fall back to Stream `0`.

---

## Phase 3: Stream Synchronization & Event Dependency Audit
- [ ] **3.1 Audit Event-Based Synchronization Logic**
  - Locate the synchronization barriers between `stream_dma` and `stream_compute`.
  - Verify synchronization uses non-blocking GPU-side events:
    ```cpp
    // Correct non-blocking pipeline pattern:
    cudaEventRecord(event_dma_ready, stream_dma);
    cudaStreamWaitEvent(stream_compute, event_dma_ready, 0); // Non-blocking GPU wait
    ```
  - Confirm `cudaEventSynchronize()` or `cudaStreamSynchronize()` are **not** being called on the CPU host thread inside the layer loop.

- [ ] **3.2 Eliminate Host-Side Thread Blocking**
  - Inspect the layer loop for CPU conditional waits (`std::mutex`, `std::condition_variable`, atomic spin-locks) that force the host thread to wait for `cudaMemcpyAsync` or prior kernel completions.

---

## Phase 4: Profiling & Automated Verification
- [ ] **4.1 Profile Execution with Nsight Systems (`nsys`)**
  - Run the prefill target under `nsys`:
    ```bash
    nsys profile --trace=cuda,nvtx,osrt --output=layer_stream_trace ./bin/main [args]
    ```
  - Inspect the timeline trace:
    - **Host Launch Overhead:** Check duration between `cudaLaunchKernel` calls vs actual GPU execution duration.
    - **Stream Overlap:** Verify `cudaMemcpyAsync` blocks on `stream_dma` run concurrently with kernel execution on `stream_compute`.

- [ ] **4.2 Measure Post-Fix Metrics**
  - Confirm GPU power consumption rises toward 35W–50W during prefill execution.
  - Confirm inter-layer kernel launch gaps on `stream_compute` drop below 10 microseconds.
  