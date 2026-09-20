# DESIGN.md: Asynchronous Layer-Streaming Prefill & EMA Hot-Expert VRAM Cache for llama.cpp

## 1. Executive Summary

This design document specifies two decoupled, complementary architectural extensions for `llama.cpp` (`ggml`) targeting low-VRAM edge devices (e.g., workstations with 4GB–12GB GPUs and 32GB–64GB System RAM).

* **Feature A (Prefill Phase):** Asynchronous Pipeline Layer-Streaming (Double-Buffered PCIe Transfers) for high-throughput prompt processing ($N_{\text{tokens}} \ge 512$).
* **Feature B (Generation Phase):** Dynamic EMA-driven Hot-Expert VRAM Cache with asynchronous background swapping for Mixture-of-Experts (MoE) models.

Together, these features guarantee a **minimum generation speed floor of ~20 t/s** (pure CPU fallback) and an **expected generation throughput of 44–80 t/s** (when MoE cache hits reach 80%+), while unlocking **100+ t/s prefill speeds** over mobile/desktop PCIe buses.

---

## 2. System Architecture & Memory Model

+-----------------------------------------------------------------------------------+
|                                 SYSTEM RAM (DDR4/5)                               |
|  +-----------------------------------------------------------------------------+  |
|  | Full MoE Model Weights (Q4_K_M) ~ 17.28 GB                                 |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
|                                                                     ^
| PCIe Async DMA Transfer                                             | Hit / Swap
v                                                                     v
+-----------------------------------------------------------------------------------+
|                                 GPU VRAM (e.g., 4GB T1000)                        |
|  +--------------------------+  +-----------------------------------------------+  |
|  | Layer Stream Buffers     |  | Hot-Expert EMA Cache                          |  |
|  | [Buf A: N]  [Buf B: N+1] |  | (Shared Attn + Hot MoE Experts ~ 2.5 - 3.2GB) |  |
|  +--------------------------+  +-----------------------------------------------+  |
+-----------------------------------------------------------------------------------+

---

## 3. Feature A: Asynchronous Layer-Streaming Prefill

### 3.1 Problem Statement
During prefill ($b \ge 512$), computing layer-by-layer on the CPU causes thrashing of CPU L2/L3 caches and limits processing to system RAM bandwidth (~35–40 GB/s). Offloading statically to GPU is impossible when the full model exceeds VRAM capacity.

### 3.2 Solution: Double-Buffered CUDA Pipeline
We utilize two CUDA streams and a double-buffered allocation in VRAM corresponding to the size of **2 Transformer Layers** ($\approx 2 \times 375 \text{ MB} = 750 \text{ MB}$).

1. **Pinned Host Allocations:** Model layers in System RAM are registered via `cudaHostRegister()` to allow Pinned DMA transfers.
2. **Double Buffering Execution Loop:**
   * **Stream 0 (Compute Engine):** Executes GEMM / Matrix Multiplications for Layer $N$ using GPU Tensor Cores.
   * **Stream 1 (DMA Transfer Engine):** Asynchronously copies Layer $N+1$ weights from Pinned CPU RAM to VRAM Buffer B via `cudaMemcpyAsync()`.
   * **Synchronization:** `cudaStreamWaitEvent()` ensures Layer $N+1$ transfer completes before Stream 0 swaps buffers and executes Layer $N+1$.

### 3.3 Overlap Condition & Mathematical Proof
Let $T_{\text{transfer}} = \frac{\text{Layer Size (MB)}}{\text{PCIe Bandwidth (MB/s)}}$ and $T_{\text{compute}} = \frac{\text{FLOPs}(N_{\text{tokens}})}{\text{GPU FLOPS}}$.

For $N_{\text{tokens}} \ge 1024$:
$$T_{\text{compute}} \ge T_{\text{transfer}}$$
The PCIe transfer overhead is **100% hidden behind GPU computation**, yielding near-native GPU execution speed during prefill.

---

## 4. Feature B: EMA-Driven Dynamic Hot-Expert VRAM Cache

### 4.1 Concept & 80/20 Locality Principle
In domain-specific tasks (e.g., coding, system administration), expert utilization in MoE models is non-uniform. Approximately 20–30% of experts handle 80% of routing requests.

Feature B maintains a **Dynamic Expert Cache** in GPU VRAM, backed by an Exponential Moving Average (EMA) routing metric.

### 4.2 Cache Data Structures
```cpp
struct MoEExpertMeta {
    int expert_id;
    int layer_id;
    float ema_score;         // Exponential Moving Average of usage
    uint64_t last_used_tick; // Timestamp for eviction tie-breaking
    bool is_in_vram;
    void* vram_ptr;          // Pointer to VRAM allocation block
};

struct MoECacheManager {
    size_t total_vram_budget_bytes;
    std::unordered_map<uint32_t, MoEExpertMeta> cache_map; // Key: (layer << 16) | expert_id
    float alpha; // Decay factor for EMA (e.g., 0.05)
};

