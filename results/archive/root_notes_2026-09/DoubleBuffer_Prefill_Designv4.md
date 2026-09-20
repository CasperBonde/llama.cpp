# Design Notes: Prefill Double Buffering in llama.cpp (v4, corrected)

Supersedes v3. v3 mixed the intended design together with a post-mortem of a failed
implementation, and it never stated the central idea (two execution modes). This
version separates the two and replaces every assumed number with a measured one.

All measurements below were taken on this checkout with:

- GPU: NVIDIA Quadro T1000, 3716 MiB total
- Model: Qwen3-Coder-30B-A3B-Instruct-Q4_K_M (qwen3moe, 48 layers, 30.53 B params)
- Build: 9714eff49

---

## 1. Core Concept: Two Execution Modes

llama.cpp today uses a single configuration for two very different workloads. That is
the inefficiency this work targets.

| | Prefill | Output generation |
|---|---|---|
| Batch shape | large parallel ubatch, 1024 - 4096 tokens | 1 token |
| Time per layer | hundreds of ms | well under 1 ms |
| Weights touched per step | the whole model | the whole model |
| Streaming viable | yes, wide compute window | no, ratio inverts by ~1000x |

The design therefore splits execution into two modes:

- **Prefill mode.** The whole forward pass runs on the GPU. Weight residency is only
  two complete layers. Everything else streams from host RAM. `-ngl` and `-ncmoe` are
  ignored in this mode. It is never used for output generation.
- **Output mode.** Exactly the current path. `-ngl` and `-ncmoe` mean what they mean
  today. It takes nothing from prefill mode.

The modes must not share ownership of model tensors. Prefill mode must never leave the
normal model path pointing at a temporary slot.

The two modes cannot use the same settings because the optimum is opposite in each
case:

- Prefill wants all compute on the GPU, and can afford to stream weights because the
  compute window is large.
- Output generation is latency bound per token and cannot stream, so it should keep
  using the existing partial offload, whatever that is.

## 2. Why Streaming Prefill Is Feasible

This is the arithmetic that earlier drafts got wrong. The streaming window is the
compute time of a whole layer **over the entire batch**, not the compute time of one
token. Confusing the two makes the transfer look 1000x too expensive.

Take 2048 tokens at 100 t/s. That prefill takes about 20 s. All 48 layers have to arrive
somewhere in that window:

| Quantity | Value |
|---|---|
| Prefill duration | 20.5 s |
| Time per layer | 20.5 / 48 = **427 ms** |
| Bytes per layer | 384.90 MiB |
| Required bandwidth | 18.05 GiB / 20.5 s = **0.88 GiB/s** |

Using the measured configuration from section 3.4, `-ngl 99 -ncmoe 99 -lm none` at
85.4 t/s for 1024 tokens:

| Quantity | Value |
|---|---|
| Prefill duration | 12.0 s |
| Time per layer | 12.0 / 48 = **250 ms** |
| Bytes per layer | 384.90 MiB |
| Measured transfer per layer | ~38 ms |
| Ratio compute : transfer | **about 6.6 : 1** |
| Required bandwidth | 1.44 GiB/s |

The conclusion is the opposite of "transfer bound". PCIe delivers roughly 10 GB/s in
practice, the transfer occupies about 15 percent of the compute window, and a single
link has several times the bandwidth required. This is a VRAM capacity play where the
compute window is wide enough to hide the transfer completely.

The corollary matters for implementation: because the window is wide, the design does
not need exotic overlap machinery to succeed. It needs correctness. Per-layer
synchronization is acceptable at this ratio, and the design should prefer the simplest
ordering that is provably right. A 38 ms transfer hidden behind a 250 ms compute step is
already invisible.

## 3. Measured Facts

### 3.1 Layer contents (blk.0, a representative MoE layer)

| Tensor | Type | Size |
|---|---|---|
| attn_norm.weight | f32 | 0.008 MiB |
| attn_q.weight | q4_K | 4.500 MiB |
| attn_k.weight | q4_K | 0.562 MiB |
| attn_v.weight | q6_K | 0.820 MiB |
| attn_output.weight | q4_K | 4.500 MiB |
| attn_q_norm.weight | f32 | ~0 MiB |
| attn_k_norm.weight | f32 | ~0 MiB |
| ffn_norm.weight | f32 | 0.008 MiB |
| ffn_gate_inp.weight | f32 | 1.000 MiB |
| ffn_gate_exps.weight | q4_K | 108.000 MiB |
| ffn_down_exps.weight | q6_K | 157.500 MiB |
| ffn_up_exps.weight | q4_K | 108.000 MiB |
| **Total** | | **384.90 MiB** |

The split matters:

| Part | Bytes | Share |
|---|---|---|
| Expert tensors (`*_exps`) | 373.50 MiB | 97.0% |
| Everything else | 11.40 MiB | 3.0% |

A layer is 97% expert weights. Attention, norms and the router are 11.4 MiB, which is
small enough to keep resident permanently at negligible cost.

### 3.2 Layer size distribution

Only two layer sizes exist in this model:

| Size | Count |
|---|---|
| 403596288 B (384.90 MiB) | 24 |
| 351421440 B (335.14 MiB) | 24 |

Total layer bytes: 17280.98 MiB. Non-layer bytes (token embedding, output norm, output
projection): 410.36 MiB.

Slot size must be the maximum, 384.90 MiB, not the mean. Two slots are therefore
**~770 MiB**.

### 3.3 VRAM budget

Configuration `-ngl 99 -ncmoe 99 -lm none`, which places all non-expert weights on the
GPU and all expert weights in pinned host memory:

| Item | Size |
|---|---|
| CUDA0 model buffer (non-expert weights) | 784.42 MiB |
| CUDA0 KV buffer (default ctx) | 96.00 MiB |
| CUDA0 compute buffer | 601.50 MiB |
| CUDA_Host compute buffer | 18.03 MiB |
| Free VRAM observed | ~2141 MiB |

Two slots at 770 MiB fit comfortably in the observed free VRAM, leaving roughly 1.3 GiB
margin.

Note: `-ngl 99` alone (all weights resident) fails to load on this machine. The attempt
needs 17524.43 MiB of device buffer and `cudaMalloc` returns out of memory. The model
does not fit, which is exactly the problem this design exists to solve.

### 3.4 Reference performance

| Configuration | Prefill t/s (pp1024) |
|---|---|
| `-ngl 0 -ncmoe 48` | 82.0 |
| `-ngl 99 -ncmoe 99` (mmap) | 83.1 |
| `-ngl 99 -ncmoe 99 -lm none` (pinned) | 85.4 |
| `-ngl 99` (all resident) | does not fit, OOM |

The suggested prototype configuration is `-ngl 99 -ncmoe 99`. It gives the highest
measured baseline, places attention, norms, router and KV on the GPU, and leaves only
the expert weights in host RAM. That is precisely the split a slot needs to supply:
373.50 MiB of experts per layer, while the 11.40 MiB of non-expert weights stay
resident.

`-lm none` is worth using because it gives a CUDA_Host pinned copy of the weights, which
is the correct source for a device upload. It also emits this warning today:

```
load_all_data: buffer type CUDA_Host is not the default buffer type for device CUDA0 for async uploads
```

That warning must be resolved or accounted for before any asynchronous upload path is
claimed to work.

### 3.5 Graph splits

`-ngl 99 -ncmoe 99` reports 146 graph splits at bs=1024. The graph alternates between
CUDA and CPU for the expert operations. This is the current behavior, and it is the
reason the layer boundary is not currently an execution boundary.

## 4. Execution Model

### 4.1 What happens today

One `ggml_cgraph` is built for the whole model, then submitted once through
`ggml_backend_sched_graph_compute_async`. The scheduler splits it by backend and submits
each split in turn. On CUDA, compatible graphs are captured as CUDA graphs and replayed.

The graph is built with fixed tensor pointers. Nothing in the normal path re-points a
weight tensor during execution.

### 4.2 The existing pause point

The scheduler already supports pausing between node ranges. With an evaluation callback
installed through `ggml_backend_sched_set_eval_callback`, the compute loop changes from
"submit the whole split" to:

```cpp
for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
    // ask the callback about node j0
    bool need = callback(t, true, user_data);

    // extend the range until a node the callback wants
    while (!need && j1 < n_nodes - 1) { ... }

    // compute only [j0, j1]
    ggml_backend_graph_compute_async(backend, &range_view);

    // hard synchronize the backend
    ggml_backend_synchronize(backend);

    // second callback, ask == false, after the range is complete
    if (need && !callback(t, false, user_data)) { break; }

    j0 = j1;
}
```

This is exactly the sync point the design needs, and it already exists in the codebase.

If the callback returns `need == true` on the last node of each layer, then for every
layer the scheduler will:

1. Compute that layer's nodes on the GPU.
2. Synchronize, so all reads of the current slot are complete.
3. Call back, which is the safe moment to refill the other slot.

**No graph restructuring is required.** The single full graph is kept. The layer
boundary becomes an execution boundary through the callback. This is a small, local
change, not a new subsystem.

### 4.3 Validated: the callback sees layer boundaries

This was tested end to end on the target model, not assumed. Using the existing
`llama-eval-callback` example with `-ngl 99 -ncmoe 99 -lm none`:

- The callback is invoked once per graph node, 3030 times for a 2-token prompt.
- Every layer emits a distinct terminal node named `l_out-<il>`.
- All 48 markers appear exactly once each, in strict order 0..47.
- Each `l_out-il` is immediately followed by the next layer's work (`norm-1`,
  `attn_norm-1`, `ffn_inp-1`, ... for layer 1).

So the key for the callback is exact and cheap to match:

```cpp
if (ask) {
    // return true only for the node that closes a layer
    return name is "l_out-<il>";
}
```

Return `false` for every other node. The scheduler then computes each layer as one
range and synchronizes once per layer, which is 48 synchronizations per ubatch instead
of 3030.

### 4.4 Measured cost of the callback path

The debug callback used in the test returns `true` for every node, which is the worst
possible case and synchronizes 3030 times. On a 900-token prompt that path took 29.2 s,
against roughly 12.5 s for the normal path at the same settings. That is about 2.3x
slower, and it is the upper bound of the overhead.

The design returns `true` only for the 48 `l_out-*` nodes, so it takes roughly 1/63 of
those synchronizations. This is a strong argument for keeping the design's per-layer
granularity rather than any finer-grained pause point.

The remaining overhead to measure is the per-range `ggml_backend_synchronize` plus the
change from one whole-split submission to 48 separate submissions, including what that
does to CUDA graph capture. Measure this before Stage 2.

### 4.5 Constraint: one callback slot

`cparams.cb_eval` is public API, and `llama_context` passes it straight through to the
scheduler. It is currently used by `common/debug.cpp` and by
`examples/eval-callback/eval-callback.cpp`, which is the supported way for callers to
observe graph nodes. Any internal prefill-streaming callback must therefore compose with
a user-supplied callback rather than silently replace it. This is a design constraint,
not a detail.

### 4.6 Slot binding

Weight tensors are rebound once at graph build time to views into the two slots:

```
layer 0 weights -> slot 0
layer 1 weights -> slot 1
layer 2 weights -> slot 0
layer 3 weights -> slot 1
...
```

The tensor objects and their addresses are then stable for the lifetime of the
submitted graph, which satisfies the requirement that backend allocation does not move
under running work. Correctness depends on the refill happening between layer 0 and
layer 2, which the callback provides.

The failure mode to avoid, described in the v3 post-mortem, is refilling eagerly before
submission. Then layers 0, 2, 4 all see slot 0 containing the last layer written to it.
Every layer must be refilled lazily at its own sync point.

### 4.7 Known limitations of this approach

The callback path synchronizes after every layer, so it serializes transfer and compute.
Given the 6.8 : 1 ratio in section 2, that still leaves the design far ahead of the
alternative, and it is the right way to establish correctness first. True overlap is a
later optimization and should not be attempted before the ordering is proven.

Enabling the callback also changes the compute call from one whole-split submission to
one submission per range, which interacts with CUDA graph capture. This must be measured
rather than assumed.

## 5. Post-Mortem: What Was Wrong Before

The v3 implementation must not be reused. Recording why, so it is not repeated.

1. **Eager fill instead of lazy refill.** Slot 0 was filled with layers 0, 2, 4, ... and
   slot 1 with 1, 3, 5, ... before one graph was submitted. All those layers alias the
   same storage, so earlier layers read overwritten weights. Any throughput reported
   from that configuration is meaningless.

2. **Binding was re-enabled after being deliberately disabled.** Commits `0400acba5` and
   `e6af8865f` disabled full-graph slot binding and skipped the slot path, precisely
   because layer chunking did not exist. A later uncommitted change re-added the binding
   lambda, which reverts that safety fix. It did not compile either, because `gparams`
   was declared `const auto` and cannot be assigned. The `const` has since been removed,
   so the tree now builds, but the re-enabled binding is still the v3 failure mode.

3. **The binding was unreachable.** `dbprefill_slots_filled` was never assigned, so
   `bind_prefill_streaming_tensor` always returned its input unchanged. The feature was
   a no-op even after the compile error was fixed.

4. **The custom node measured the wrong thing.** `ggml_map_custom1` is not supported by
   CUDA, so the node is split out to the CPU and forces a device-to-host copy plus a
   host synchronization. It was appended at the end of the graph, so it reported
   `boundary=graph_end` rather than a layer boundary. It validated nothing that
   section 4.2 does not validate better.

5. **The probe was appended after graph build.** The scheduler is sized from
   `graph_max_nodes(n_tokens)` and asserts `hash_set.size >= n_nodes + n_leafs`.
   Appending a node afterwards can trip that assert on architectures whose node budget
   is already exhausted. A probe node belongs inside graph construction.

6. **Buffer checks were missing.** The two slot allocations are never validated against
   the quantized layout. `bind_prefill_streaming_tensor` builds an `I8` 1D view and then
   overwrites `type`, `ne` and `nb` afterwards. That is a silent repack, which the
   design explicitly forbids for tensors that cannot be represented as valid views.

7. **The benchmark harness could not detect corruption.** With the aliased binding
   enabled, `llama-bench` reported `pp1024 = 31.24 t/s`, a clean exit code and no error.
   Two runs that differ only by the experimental flag produce `-o csv` output that is
   byte-identical except for the timestamp. Correctness must be established outside
   `llama-bench`.

## 6. Implementation Plan

### Stage 0: Instrument

Already largely done and useful. Keep:

- Per-layer byte sizes and the largest-layer requirement.
- Device capability dump (`async`, `events`, free and total memory).
- Timing split for graph build, graph alloc, input set, graph compute.

Fix:

- Run the layer-size diagnostics once per context, not on every ubatch. It currently
  scans all tensors per layer and logs per token during generation.
- Gate the benchmark `fprintf` output on the experimental flag.
- Add `prefill_streaming` to the reported fields in `llama-bench`, so results are
  attributable.

### Stage 1: Slot binding and lazy refill, synchronous

Target `-ngl 99 -ncmoe 99 -lm none` on Qwen3-MoE only.

1. Keep the single full graph.
2. Bind every `blk.il.` weight tensor to a view into `slot[il % 2]`, once, at build time.
3. Allocate two slots of `max_layer_bytes`, aligned to the backend buffer alignment
   queried from the buffer type, not assumed.
4. Install an eval callback. Return `true` only for the `l_out-il` node, which section
   4.3 confirms is a unique, correctly ordered per-layer marker. Return `false` for
   every other node, so the scheduler still batches the rest of the layer.
5. In the `ask == false` callback for layer `il`, refill `slot[(il + 1) % 2]` with layer
   `il + 1`. The backend has been synchronized, so this is safe.
6. Do not refill during the same call that follows the final layer.

Refill source: the pinned host buffer that `-lm none` creates. Do not copy through a
temporary `std::vector` per tensor; that was the v3 mistake.

Validation, in order:

1. Assert that the bound bytes for a layer equal the layer's total bytes. This catches
   an incomplete binding list immediately.
2. Compare logits against the normal path for the same prompt and seed.
3. Confirm the slot for layer `il + 1` is written exactly once per ubatch.

### Stage 2: Asynchronous refill

Once Stage 1 is correct, replace the synchronous refill with an async upload plus events,
so transfer overlaps with the following layer's compute. Per the ratio in section 2, this
should be a modest gain on this hardware, and the synchronous version should already be
close to the baseline. Measure before optimizing further.

### Stage 3: Generation

Nothing. Output mode is unchanged by this work. It keeps using `-ngl` and `-ncmoe` as
today, including shared GPU and CPU expert execution.

## 7. Mode Selection

A ubatch is served by prefill mode when it has more than one token. Single-token ubatches
are output mode. A threshold of 256 tokens is acceptable if it is a single condition;
otherwise use `n_tokens > 1` and keep it simple.

Whatever the rule, it must be part of the graph reuse key. `llm_graph_params::allow_reuse`
does not currently compare the bind state, so a graph built without binding could be
reused for a request that expects binding.

## 8. Fallback Rules

Use the normal full-graph path when any of these hold:

- The architecture is not the one explicitly supported.
- A layer does not fit in the slot budget.
- A tensor cannot be represented as a valid view of its quantized layout.
- Retrieving the layer bytes back for a logits comparison. The correctness probe needs
  a path back to the original weights that does not depend on the slots.
- The backend does not expose the required buffer alignment or event support.
- Slot allocation fails.

Fallback must occur before any change to model tensor ownership. A failed attempt must
not leave the normal path pointing at a slot.

## 9. Open Questions

1. The eval callback is currently a public hook, used by `common/debug.cpp` and by
   `examples/eval-callback`. If prefill streaming uses it internally, it must compose
   with a user callback. Is that acceptable, or does this need a second internal
   callback channel in the scheduler?
2. With the callback installed, does CUDA graph capture still apply per range, and what
   does that do to per-layer launch overhead? Measure before Stage 2.
3. `-lm none` produces the pinned CUDA_Host buffer but warns that it is not the default
   buffer type for async uploads. Resolve this before claiming overlap.
4. Where exactly should the mode switch live so that a ubatch is served by prefill mode
   while the KV cache from that prefill is handed to output mode unchanged?
5. Per AGENTS.md, a change of this size should be discussed in an issue before a pull
   request is opened. The prototype is fine to build first, but the PR needs an issue and
   a maintainer conversation.

## 10. Status

| Item | State |
|---|---|
| Two-mode requirement | agreed |
| Streaming arithmetic | corrected and measured, section 2 |
| Layer contents and slot size | measured, section 3.1 and 3.2 |
| VRAM budget | measured, section 3.3, two slots fit |
| Prototype configuration | `-ngl 99 -ncmoe 99 -lm none`, section 3.4 |
| Per-layer sync point | validated in code and at runtime, section 4.2 and 4.3 |
| Slot binding and lazy refill | designed, not implemented |
| v3 implementation | must be reverted, section 5 |

Current uncommitted state of `src/llama-context.cpp`: the `const` was removed from
`gparams`, and the v3 `tensor_bind` lambda is still present. The tree builds, but the
lambda restores the aliasing bug from section 5.1 and should be removed before any
further work.

