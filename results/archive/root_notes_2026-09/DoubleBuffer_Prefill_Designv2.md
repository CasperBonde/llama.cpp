# Layer-Streaming Prefill: Reduced Validation Design

## 1. Purpose

This document replaces the first design as the implementation starting point. Its purpose is to validate one narrow question:

> Can llama.cpp execute a large prefill one transformer layer at a time, with the layer weights copied into a small CUDA buffer, while preserving logits and KV-cache behavior?

The first implementation should optimize for an understandable correctness experiment. It should not attempt to turn the normal scheduler into a general streaming scheduler.

The target workload is a decoder-only MoE model such as the Qwen model used for testing, with:

- CUDA as the only accelerated backend.
- One CUDA device.
- A large prefill ubatch, normally the complete input and capped at 4096 tokens.
- A single reusable device layer slot in the first implementation.
- All model weights retained in host memory.
- Normal existing execution for generation and unsupported requests.

If this version does not produce a useful result, adding asynchronous streams and a second slot would only make the failure harder to diagnose.

## 2. Main Design Decision

Do not try to execute the existing full graph while changing its weight pointers between layers.

The current graph is constructed as a dependency graph and submitted to the scheduler as one unit. MoE routing is also represented by graph nodes. There is no host-side completion point after expert selection at which it is safe to overwrite a layer's weights. The graph can contain work from many layers in flight, and scheduler allocation may already have captured the tensor buffers.

Therefore the smallest correct execution boundary is a complete transformer layer:

```text
build layer 0 graph -> copy layer 0 weights -> execute layer 0
build layer 1 graph -> copy layer 1 weights -> execute layer 1
...
```

The first version deliberately waits for each copy and graph execution to finish. This gives a direct correctness baseline and proves the required ownership model.

## 3. What Is In Scope

### 3.1 Prefill

Enable the path only when all of the following are true:

- The request is decoder-only causal prefill.
- The ubatch contains more than one token.
- The ubatch is large enough to justify layer streaming.
- The model architecture is explicitly supported by the prototype.
- All repeating layers use one CUDA device.
- Every supported layer fits in the configured slot size.
- No unsupported adapter, control-vector, recurrent, encoder, hybrid, or cross-attention feature is active.

The initial threshold should be configurable internally and can start at 1024 tokens. The preferred benchmark point is 4096 tokens, subject to context and memory limits. The implementation must not require exactly 4096 tokens.

### 3.2 Model residency

Model weights have two different roles:

```text
Host RAM:  canonical copy of all model weights
VRAM:      one temporary layer slot plus normal activations, KV cache, and workspaces
```

The first version should make the canonical host copy explicit. It should not depend on preserving a full GPU copy of the model and then trying to reclaim it after graph construction.

The existing loader already has host buffer and mmap concepts. Reuse them where possible. If the selected mmap memory cannot be copied asynchronously later, that is acceptable in the synchronous first stage.

### 3.3 MoE

For prefill, copy the complete MoE layer into the slot, including all experts and any associated biases, scales, shared experts, and routing tensors required by the architecture.

Do not stream individual experts in this project. Do not add a CPU expert-routing implementation. The GPU executes the normal MoE graph with all experts for the active layer available.

For generation, use the existing execution path. It is acceptable that the prototype leaves generation with experts on CPU or otherwise falls back to the normal scheduler. The prototype is successful if streamed prefill completes correctly and the following generation remains correct, even if generation performance is unchanged.

## 4. Execution Model

The initial executor owns one layer slot and processes one ubatch as follows:

```text
prepare the normal input and memory context

for il in [0, n_layer):
    bind the model tensors for layer il to the layer slot
    copy the host weights for layer il into the slot
    build a graph for exactly layer il
    allocate and execute that graph
    retain the resulting activation as input to il + 1

run the final norm/output stage
return logits and updated memory state
```

The slot is not overwritten until the graph for the current layer has completed. This removes the need for a new CUDA transfer stream, CUDA events, or backend synchronization API in stage 1.

The activation between layers must remain in the normal backend compute buffers. Only model weights are replaced between iterations.

### 4.1 What a layer graph includes

A layer graph includes all operations that consume the current activation and produce the next activation, including:

- Attention projections and attention computation.
- The layer's KV-cache writes.
- Dense FFN or complete MoE FFN computation.
- Residual connections and layer norms.
- Any architecture-specific operation belonging to that transformer layer.

The graph must not include unrelated layers. The final output norm, output projection, pooling, and sampling are separate final-stage work as appropriate for the selected architecture.

### 4.2 KV-cache ordering

The layer graph must finish all writes to that layer's KV cache before the next layer begins. The synchronous graph completion provides this ordering in stage 1.

The executor must pass through the existing memory context rather than creating a second KV-cache implementation. It must preserve:

- Sequence positions.
- Causal masking.
- Multiple sequences in one ubatch where supported.
- Existing cache addressing and context limits.

Context shifting and hybrid/recurrent memory are out of scope for the first prototype. They must select the normal path instead.

## 5. Layer Weight Binding

### 5.1 Avoid a packed universal layer format initially

A universal packed layer blob would require reproducing every tensor layout, quantization layout, stride, scale, and view relationship. That is unnecessary for the first correctness experiment.

Instead, build a layer binding object for the selected architecture. It contains the layer's existing tensor metadata and a slot-backed replacement for each tensor consumed by the layer graph.

Conceptually:

```text
model.layers[il].ffn_gate_exps  -> slot view for layer il
model.layers[il].ffn_up_exps    -> slot view for layer il
model.layers[il].ffn_down_exps  -> slot view for layer il
model.layers[il].attn_norm      -> slot view for layer il
...
```

The binding must preserve the original ggml tensor type, dimensions, strides, quantization block format, and optional scale/bias tensors. A view is valid only when the destination slot has the same backend buffer type and compatible alignment.

Do not mutate the model's canonical tensor pointers globally. Pass an explicit layer binding to the layer graph builder, or make a temporary layer descriptor that contains the bound tensors. This avoids races and makes fallback behavior straightforward.

### 5.2 Scope of architecture changes

The current architecture graph implementations access `llama_layer` tensors directly in their layer loops. A practical first refactor is:

1. Add a small internal `llama_layer_view` or equivalent binding type.
2. Change only the selected architecture's layer-building function to accept that binding for the current layer.
3. Keep the existing full-graph implementation unchanged for all other architectures and normal execution.

Do not refactor every model file before the experiment has results. The binding interface should be shaped by one working architecture.

### 5.3 Host-to-slot copy

Stage 1 may use synchronous backend copies through existing ggml backend copy functions. The copy operation must be explicit and complete before graph allocation or execution uses the slot.

The layer manifest should record:

- Layer index.
- Tensor identity.
- Host source tensor.
- Slot offset.
- Tensor byte size.
- Alignment requirement.
- Destination tensor type and shape.
- Total slot size.

The slot size is the maximum required layer size, rounded according to the selected backend buffer type's alignment requirements. Do not hard-code 256 bytes without checking the backend contract. The allocation and view offsets must satisfy the actual buffer and tensor alignment requirements.

If a tensor cannot be represented as a slot view without changing its layout, the prototype must reject the architecture and use the normal path. It must not silently repack quantized tensors in a way that changes the kernel contract.

## 6. Graph Construction Strategy

A complete general layer builder is a large change because model-specific files currently construct their own layer loops. Stage 1 should therefore use a narrow internal graph-building entry point for one architecture.

The entry point needs these inputs:

- Architecture and model hyperparameters.
- Layer index.
- Current activation tensor.
- Existing ubatch and position inputs.
- Existing memory context.
- The bound tensors for the active layer.
- A flag indicating whether this is the final layer.
- The existing graph result/context used for input and output registration.

It returns:

- The activation consumed by the next layer.
- The graph containing only the selected layer.
- Any final-stage tensors needed by the normal output path.

The implementation should share the existing operations and helper functions such as attention, norm, FFN, and MoE builders. It should not duplicate CUDA kernels or implement a second scheduler.

The graph must be built after the current layer is bound and before that layer is allocated. This avoids allocating a full graph whose model tensors point at the wrong buffer.

## 7. Scheduler Boundary

Stage 1 uses the existing scheduler for each layer graph:

```text
reset graph result and scheduler state
build one layer graph
allocate one layer graph
set graph inputs
compute graph synchronously
```

This is intentionally less efficient than the final design. It is useful because it keeps scheduler ownership clear and makes the first failure local:

- Incorrect layer binding.
- Incorrect activation handoff.
- Incorrect KV-cache update.
- Incorrect graph allocation.
- Numerical mismatch.

Do not add scheduler APIs in stage 1. Do not call CUDA APIs directly from `llama_context`.

The normal full graph remains the fallback whenever the prototype cannot establish all required bindings.

## 8. Public and Internal Configuration

Stage 1 should preferably be enabled by an internal development option or a narrowly scoped context flag. Avoid adding several public parameters before the behavior is validated.

The minimum controls are:

- Streaming prefill enabled or disabled.
- Maximum streamed prefill tokens, default 4096.
- Optional diagnostic logging.

A fixed two-slot public setting is premature because stage 1 uses one slot. Add the second slot only after measurements prove that transfer/compute overlap is worth the additional lifetime and synchronization complexity.

The mode must automatically disable itself when:

- The request is generation-sized.
- The model is unsupported.
- A layer is too large for the slot budget.
- A required tensor has an incompatible layout or backend buffer type.
- Memory or graph allocation fails.
- Any unsupported memory path is active.

Disablement must leave the existing execution path usable.

## 9. Implementation Stages

### Stage 0: Measure the current path

Before changing execution, collect:

- Per-layer weight byte sizes.
- Total host and device model buffer sizes.
- Per-layer compute time, if layer boundaries can be instrumented without changing behavior.
- Prompt prefill time for 1024, 2048, and 4096 tokens.
- VRAM use before and after model load.

This confirms whether the target model has enough layer compute to amortize copies and identifies the largest layer.

### Stage 1: One-slot synchronous correctness path

Implement only:

- Host-resident canonical model weights.
- One selected CUDA architecture.
- One slot sized for the largest supported layer.
- One-layer graph construction.
- Synchronous host-to-device copies.
- Existing scheduler allocation and synchronous compute.
- Existing full-graph fallback.

The required result is numerical equivalence, not speed.

### Stage 2: Two-slot asynchronous prototype

Only after Stage 1 is correct, add:

- Slot 0 and slot 1.
- A transfer stream.
- Completion events for transfer and compute.
- A scheduler/backend-owned way to order slot reuse.
- Copy of layer `il + 1` while layer `il` computes.

The execution should become:

```text
copy layer 0 -> slot 0
compute layer 0 on slot 0
copy layer 1 -> slot 1 while layer 0 computes
compute layer 1 on slot 1
copy layer 2 -> slot 0 after layer 0 is complete
```

The second slot is a performance optimization, not part of the correctness proof.

### Stage 3: Generalization

Only if the target benchmark improves, consider:

- More MoE architectures.
- Dense architectures.
- Pinned host memory and asynchronous mmap handling.
- Layer manifests shared across architectures.
- Backend-neutral streaming abstractions.
- Context reuse and context shifting.
- Multi-GPU execution.
- Generation-time GPU residency or expert caching.

## 10. Correctness Plan

For the same model, prompt, context parameters, and random seed, compare normal and streamed prefill:

- Logits for all requested prompt outputs.
- First generated token.
- Several subsequent generated tokens.
- KV-cache behavior by continuing generation from the streamed prefill.
- Prompt sizes below and above the streaming threshold.
- A prompt exactly at the 4096 cap if supported by the context.
- Quantized weights and MoE per-expert scales.
- Multiple sequences only after the single-sequence case works.

The first test target should be one supported Qwen MoE configuration. Keep the comparison harness outside the normal test suite until the design is stable; use existing benchmark or example tooling where practical.

A mismatch should first be classified as one of:

1. Layer binding or tensor view error.
2. Activation handoff error.
3. KV-cache ordering or address error.
4. Final output-stage error.
5. Existing numerical nondeterminism.

## 11. Performance Plan

Stage 1 is expected to be slower than the existing path. Measure it anyway to validate the cost model:

```text
streamed time = sum(layer copy time + layer compute time + graph overhead)
```

For Stage 2, report:

- Prefill tokens per second.
- Layer copy time.
- Layer compute time.
- Copy/compute overlap.
- Slot reuse stalls.
- Peak VRAM.
- Host memory bandwidth.
- Generation speed after streamed prefill.

The idea is promising only if the two-slot version reduces end-to-end prefill time while keeping VRAM below the intended limit. A faster layer copy alone is not sufficient if graph rebuild and scheduler allocation dominate.

## 12. Risks and Explicit Non-Goals

### Main risks

- Refactoring one architecture's layer loop without changing its exact memory semantics.
- Creating views that are technically valid ggml tensors but incompatible with a quantized CUDA kernel.
- Reallocating compute buffers for every layer and losing the expected VRAM savings.
- Rebuilding scheduler state per layer at a cost larger than the transfer benefit.
- Assuming MoE expert selection creates a host synchronization point. It does not in the current graph model.

### Non-goals for the validation version

- Rebinding pointers in a full graph while it is executing.
- Streaming individual experts.
- A general backend API.
- A general packed tensor format.
- HIP, Metal, Vulkan, CPU, or multi-GPU support.
- Recurrent or hybrid memory.
- Generation optimization.
- Public configuration for every tuning parameter.

## 13. Acceptance Criteria

The validation is complete when all of these are true:

1. The selected model can run a streamed prefill with one device layer slot.
2. The streamed logits match normal execution within the expected backend tolerance.
3. Generation after streamed prefill remains correct.
4. Unsupported requests automatically use the existing path.
5. Peak VRAM is measured and reduced as expected.
6. The two-slot asynchronous design has enough measured benefit to justify its added synchronization complexity.

The likely contribution boundary is therefore:

```text
first contribution: one-slot synchronous layer-streamed prefill for one architecture
follow-up:         two-slot asynchronous overlap
later:             broader architecture and backend support
```

This keeps the first change small enough to review while directly testing the central idea.
