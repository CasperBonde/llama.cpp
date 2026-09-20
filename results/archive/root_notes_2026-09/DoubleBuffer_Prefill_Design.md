# Layer-Streaming Prefill for Low-VRAM MoE Models

## 1. Goal

Improve prompt-processing speed for large MoE models when the complete model cannot fit in GPU VRAM.

During prefill, llama.cpp processes a prompt batch through the transformer one layer at a time. This makes it possible to keep only two complete transformer-layer weight sets in VRAM:

```text
Host RAM:  full model weights

VRAM:      layer N     -> compute on prompt batch
           layer N + 1 -> transfer asynchronously
```

The design targets prompt batches large enough that GPU computation for layer `N` overlaps most or all of the transfer time for layer `N + 1`.

This design does **not** stream individual MoE experts. For a prefill batch containing thousands of tokens, routing can select most or all experts in a layer. The transfer unit is therefore the complete layer, including all of its MoE expert weights.

Generation-time hot-expert caching is outside the scope of this document.

## 2. Constraints and Assumptions

- The initial implementation targets the CUDA backend.
- The feature is opt-in and must have a normal scheduler-based fallback.
- The model weights remain resident in host memory, preferably through the existing mmap and host-buffer mechanisms.
- A layer includes every tensor needed by that layer: attention weights, norms, biases, scales, shared FFN weights, and all MoE expert tensors.
- The prompt batch is processed through the complete layer before the activation is passed to the next layer.
- The first and last layers may have different tensor layouts and must not be assumed to have the same size as repeating layers. So the size of the dual buffers shall be large enough to hold the largest layer.
- Generation should continue using the existing graph execution path.

The feature should not depend on a fixed assumption such as `1024` tokens.

## 3. Existing Execution Model

The relevant current path is:

```text
llama_decode()
  -> llama_context::decode()
  -> llama_context::process_ubatch()
  -> llama_model::build_graph()
  -> ggml_backend_sched_alloc_graph()
  -> llama_context::graph_compute()
  -> ggml_backend_sched graph splits and backend execution
```
TODO: Locate the code that handles layer compleation (like Expert selection) as that is whete the GPU/CPY sync point needs to be located.

Important current ownership points:

- `llama_context::decode` divides a logical batch into `llama_ubatch` instances. 
- `llama_context::process_ubatch` builds or reuses one graph and executes it.
- `llama_model::build_graph` constructs the complete transformer graph. (This is where every second layer needs to point to the same odd or even buffer)
- `llama_model_base::load_tensors` assigns model tensors to backend buffer types and loads them.
- `ggml_backend_sched_split_graph` assigns graph operations to backends and inserts copies between backend splits.
- CUDA operations execute through a backend-owned CUDA stream.

The existing scheduler is designed to execute a complete graph whose weight tensors already have backend buffers. It is not currently a layer-streaming scheduler. Adding independent `cudaMemcpyAsync` calls around `llama_decode` would be unsafe because scheduler copies, graph allocation, backend execution, and buffer reuse would not share the required dependencies.

## 4. Proposed Execution Model

For an eligible prefill batch, replace one full-model graph execution with a sequence of layer executions:

```text
prepare input and KV state

for each transformer layer il:
    wait until layer il weights are resident in compute slot
    execute layer il for the complete prompt batch
    record layer il compute completion
    allow the slot to be reused after the completion event

finish output layer and sampling/output handling
```

With two VRAM slots:

```text
Transfer stream: slot A <- layer 0
Compute stream:  slot A -> compute layer 0
Transfer stream: slot B <- layer 1      (overlaps layer 0 compute)
Compute stream:  slot B -> compute layer 1
Transfer stream: slot A <- layer 2      (overlaps layer 1 compute)
```

The transfer stream must wait for the previous use of a slot before overwriting it. The compute stream must wait for the transfer-complete event before using the slot.

The implementation must preserve the normal dependency chain:

```text
activation after layer N -> activation input to layer N + 1
KV writes from layer N     -> visible before the next relevant operation
```

The input and output activations should be allocated from normal ggml backend compute buffers. Only model weight residency is double-buffered.

## 5. Layer Weight Representation

### 5.1 Required metadata

Add internal metadata for every streamable layer containing:

- Layer index.
- Device and backend buffer type used for the staging slot.
- All tensors belonging to the layer.
- Host source address or model-file offset for each tensor.
- Tensor byte size and alignment.
- Destination offset in the layer slot.
- Total slot size. Mind the data alignment requirements - I believe it is 256bytes

The metadata must account for architecture-specific tensors rather than enumerating only the common Llama attention and FFN weights.

### 5.2 Existing files and functions

Primary files:

- `src/llama-model.h`
- `src/llama-model.cpp`
- `src/llama-model-loader.h`
- `src/llama-model-loader.cpp`

Important functions:

- `llama_model_base::load_tensors`
- `llama_model_base::create_tensor`
- `llama_model::select_buft`
- `llama_model::dev_layer`

`llama_model_base::load_tensors` currently creates backend buffers and loads model data. The streaming implementation should extend this ownership model rather than adding a second unrelated model loader.
The idea is to have everything in CPU RAM - and then depending on mode(Prefill vs output) use VRAM either for dual buffer prefill - or model layers. To realize this the loader needs to load everything into CPU RAM - then the run-time needs to handle the swap.

Potential approaches, in increasing order of invasiveness:

1. Keep host-backed model tensors as the canonical source and create two CUDA layer buffers.
2. Add a layer pack/unpack helper that copies all tensors for a layer into a contiguous device slot.
3. Represent graph weight tensors as views into the active layer slot.

The third option is likely the cleanest for execution, but it requires careful handling of tensor strides, quantized layouts, scales, and `view_src` ownership. A layer slot must remain alive until all graph operations using it have completed.

### 5.3 Host memory

Do not assume that every mmap region can be passed directly to `cudaHostRegister`.

The implementation should first check whether the existing CUDA host buffer type supports the required asynchronous transfer path. If it does not, add a backend-managed pinned staging source or make pinned-memory registration explicit and failure-aware.

Required behavior:

- Registration failure must fall back to synchronous or normal scheduler execution.
- Model unloading must unregister host regions after all transfers complete.
- Host memory use must be bounded and reported.
- Multiple model files and discontiguous mmap ranges must be supported.

## 6. Layer-Scoped Graph Construction

### 6.1 Required change

The current graph builder constructs the full transformer graph. Streaming requires a graph or graph fragment that computes one layer for an activation batch and exposes the resulting activation for the next iteration.

Relevant files:

- `src/llama-graph.h`
- `src/llama-graph.cpp`
- `src/llama-model.cpp`
- `src/llama-context.h`
- `src/llama-context.cpp`

Important functions:

- `llama_model::build_graph`
- Architecture-specific graph-building functions in `src/llama-graph.cpp`
- `llm_graph_context::build_moe_ffn`
- `llama_context::process_ubatch`
- `llama_context::sched_reserve`

The preferred design is to factor the repeated transformer-layer construction into a layer-scoped builder that accepts:

- Layer index.
- Current activation tensor.
- Prompt positions and sequence metadata.
- KV-cache inputs and outputs.
- Active layer weight bindings.
- Final-layer/output flags.

It should return the activation needed by the next layer and any outputs required by the memory module or final output stage.

### 6.2 KV-cache considerations

The KV cache is not simply an intermediate activation. Attention in layer `N` writes keys and values associated with that layer, and later generation reads those layer-specific entries.

The layer-scoped graph must therefore:

- Use the correct layer-specific KV-cache tensors.
- Preserve the existing cache layout and sequence positions.
- Commit KV writes before the layer’s compute-complete event is considered safe.
- Work with causal and non-causal attention modes supported by the model.
- Preserve context shifting and cache optimization behavior.

Relevant files:

- `src/llama-kv-cache.cpp`
- `src/llama-kv-cache.h`
- `src/llama-memory*.cpp`
- `src/llama-memory*.h`
- `src/llama-context.cpp`

The first implementation should support the common causal decoder path before extending to recurrent, encoder, or hybrid architectures.

## 7. CUDA Backend and Synchronization

Relevant files:

- `ggml/src/ggml-cuda/ggml-cuda.cu`
- CUDA backend context and implementation files under `ggml/src/ggml-cuda/`
- `ggml/src/ggml-backend.cpp`
- Backend declarations under `ggml/include/`

The CUDA backend already owns execution streams and launches kernels through its backend context. Add streaming resources there:

- One transfer stream.
- The existing compute stream, or a backend-managed compute stream for the streaming path.
- Two layer-slot allocations.
- Transfer-complete events for each slot.
- Compute-complete events for each slot.

Required operations are conceptually:

```text
begin_layer_transfer(layer, slot)
wait_layer_transfer(slot)
begin_layer_compute(layer, slot)
record_layer_compute_complete(slot)
release_layer_slot(slot)
```

These should be exposed through backend-neutral operations only if they are needed by the layer-streaming executor. `cudaStream_t`, `cudaEvent_t`, and CUDA allocation details must remain private to the CUDA backend.

The backend must verify support for:

- Asynchronous host-to-device copies.
- Events.
- Concurrent copy and compute.
- The required buffer type and tensor layout.

If any capability is absent, the executor must disable streaming and use the existing scheduler path.

## 8. Scheduler Integration

Relevant files and functions:

- `ggml/src/ggml-backend.cpp`
  - `ggml_backend_sched_split_graph`
  - `ggml_backend_sched_alloc_graph`
  - `ggml_backend_sched_synchronize`
  - scheduler copy handling and event handling
- `src/llama-context.cpp`
  - `llama_context::sched_reserve`
  - `llama_context::process_ubatch`
  - `llama_context::synchronize`
  - `llama_context::graph_compute`

The streaming executor should not fight the scheduler by mutating buffers behind its back. Choose one explicit boundary:

### Option A: Backend-level staged graph execution

Extend ggml backend interfaces so the scheduler can submit layer graphs and backend-managed transfer dependencies. This is the most architecturally consistent option, but it has the largest API and implementation scope.

### Option B: A llama.cpp streaming executor using stable backend buffers

Keep the first prototype in `llama_context`, but require the CUDA backend to provide an internal layer-slot API. The executor submits one layer graph at a time and calls scheduler allocation only for activation graphs whose weight tensors are already bound to the active slot.

This is smaller, but it creates a llama.cpp-specific path that must be carefully isolated and documented.

For a first contribution, Option B is a reasonable prototype if it does not modify normal scheduler behavior. If the prototype is successful, the backend-level abstractions can be generalized in a later change.

## 9. MoE Layer Handling

The layer transfer includes all expert tensors for that layer:

- `ffn_gate_exps`
- `ffn_up_exps`
- `ffn_down_exps`
- Their optional biases and scale tensors
- Any architecture-specific shared expert tensors

Relevant files and functions:

- `src/llama-model.h`
  - `struct llama_layer`
- `src/llama-model.cpp`
  - model tensor creation and loading
- `src/llama-graph.cpp`
  - `llm_graph_context::build_moe_ffn`
- CUDA MoE implementation files:
  - `ggml/src/ggml-cuda/mmid.cu`
  - `ggml/src/ggml-cuda/mmid.cuh`
  - `ggml/src/ggml-cuda/topk-moe.cu`
  - `ggml/src/ggml-cuda/moe-weighted-reduction.cu`

The graph must continue to perform normal routing and top-k selection. No routing approximation is needed for this design. All experts remain available in the active layer slot, so numerical behavior should remain unchanged.

The main MoE requirement is ensuring that the active layer slot contains the complete expert tensor layout expected by `GGML_OP_MUL_MAT_ID` and the CUDA implementation of that operation.

## 10. Public Configuration

Possible public fields belong in:

- `include/llama.h`
- `src/llama-cparams.h`
- `common/arg.cpp`

Suggested configuration:

- `prefill_streaming`: disabled by default.
- `prefill_streaming_slots`: initially restricted to `2`.
- `prefill_streaming_max_memory`: maximum VRAM reserved for layer slots.
- Optional diagnostic logging.

The implementation should reject or disable streaming when:

- The model has unsupported architecture features.
- The layer does not fit in the configured VRAM budget.
- Required CUDA async capabilities are unavailable.
- The request is generation-sized rather than a large prefill batch.
- The graph requires unsupported cross-layer state.

## 11. Implementation Phases

### Phase 1: Instrumentation

Add diagnostics without changing execution:

- Per-layer weight sizes.
- Per-layer compute duration.
- Host-to-device copy duration.
- CUDA device free memory before and after model loading.
- Scheduler split and copy counts.

Use this to confirm whether layer compute time is sufficient to hide the next layer transfer.

### Phase 2: Synchronous layer execution

Implement a correctness-first path that:

- Loads one complete layer into a device slot.
- Builds and executes a layer-scoped graph.
- Carries the activation into the next layer.
- Uses synchronous transfers.
- Falls back to the existing full graph for unsupported models.

This phase validates graph decomposition, tensor binding, KV-cache behavior, and numerical equivalence.

### Phase 3: Two-slot asynchronous execution

Add:

- Two layer slots.
- Transfer and compute streams.
- Events for transfer completion and slot reuse.
- Pinned host source memory where supported.
- Stall and overlap diagnostics.

The synchronous path should remain available as a debugging mode.

### Phase 4: Broader model support

Extend carefully to:

- More MoE architectures.
- Multi-GPU layer assignment.
- Hybrid and recurrent architectures.
- Encoder and non-causal paths.
- Adapter and control-vector combinations.

Each extension should be validated independently because these paths may not have the same layer-local state model.

## 12. Correctness Tests

Compare streaming against normal execution for the same model, prompt, and settings:

- Logits for every prompt output requested.
- KV-cache contents where an internal comparison is practical.
- First generated token and subsequent generated tokens.
- Different batch sizes.
- Prompt lengths below and above the threshold.
- Multiple sequences in one batch.
- Context reuse and context shifting.
- Quantized MoE models with per-tensor and per-expert scales.
- Aborted transfers and context destruction.

The expected output should match within the normal backend numerical tolerance. The layer-streaming path must not alter routing decisions or sampling behavior.

## 13. Performance Measurements

Report at least:

- End-to-end prefill tokens per second.
- Per-layer compute time.
- Per-layer transfer time.
- Transfer/compute overlap percentage.
- Number of staging stalls.
- Peak VRAM used by layer slots and activations.
- Generation speed after streamed prefill.
- CPU utilization and host memory bandwidth.

Benchmark prompt sizes such as 128, 512, 1024, 2048, and 4096 tokens. Include cases where a layer is larger than the available staging budget; those must cleanly fall back.

## 14. Risks and Design Decisions

### Layer graph decomposition

This is the main engineering risk. The current graph builder and memory module are designed around a complete model graph. The implementation must avoid duplicating architecture-specific logic or silently changing cache semantics.

### Tensor binding and views

Rebinding model tensors to a new slot must preserve quantized tensor metadata, strides, scales, and views. A contiguous packed layer buffer is attractive, but it must not force unsupported tensor layouts.

### Transfer source memory

Asynchronous transfer requires suitable host memory. Mmap-backed model data may need backend-specific registration or an intermediate pinned buffer.

### VRAM budgeting

The two layer slots compete with KV cache, activation buffers, output buffers, and CUDA workspace allocations. Reserve the slots only after accounting for the existing scheduler requirements, and fail gracefully when the budget is insufficient.

### Stream ordering

The transfer stream and compute stream must be synchronized with events, not host blocking. Slot reuse before compute completion can corrupt weights while kernels are still running.

### Portability

The initial implementation can be CUDA-specific, but the llama.cpp execution path must retain normal behavior for CPU, Metal, Vulkan, HIP, and other backends.

## 15. Recommended Starting Point

Start with a single CUDA device, one supported decoder-only MoE architecture, causal attention, and a fixed two-slot configuration.

The first useful prototype should prove this sequence:

```text
load layer 0 -> execute layer 0
load layer 1 -> execute layer 1
...
```

Then add overlap:

```text
compute layer N while transferring layer N + 1
```

Do not begin with EMA expert caching, multi-GPU support, or every model architecture. The central contribution is a correct layer-scoped prefill executor that can reuse the existing model weights and CUDA kernels while reducing the VRAM residency requirement to approximately two layers plus runtime state.
