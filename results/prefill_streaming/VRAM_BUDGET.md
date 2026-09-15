# VRAM / RAM budget and identified improvements

Model: `Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf`
Server config: `-ngl 99 -ncmoe 48 -lm none -c 80000 -np 1 -b 4096 -ub 2048 -ctk q4_0 -ctv q4_0 -fa on`
Device: Quadro T1000, 3716 MiB usable VRAM. Host: 62 GB RAM.

## 1. VRAM budget at the server config

| component | size (MiB) | % of VRAM | note |
|---|---:|---:|---|
| CUDA0 model weights | 1446.39 | 38.9% | non-expert weights only, see section 2 |
| CUDA0 KV | 528.19 | 14.2% | q4_0, 12 full-attention layers, 44.02 each |
| CUDA0 RS (SSM state) | 75.38 | 2.0% | hybrid arch recurrent state |
| CUDA0 compute | 1345.09 | 36.2% | scales with -c and -ub, see section 4 |
| unaccounted (CUDA ctx, cudnn) | ~173 | 4.7% | fixed driver/library overhead |
| **subtotal on device** | **3567.05** | **96.0%** | |
| CUDA_Host compute | 345.31 | - | host-side staging scratch, see section 4 |
| CUDA_Host model weights | 44806.92 | - | pageable RAM |
| **free VRAM** | **147** | **4.0%** | |

## 2. Weight decomposition (exact, from GGUF metadata)

| bucket | size (MiB) | tensors | where |
|---|---:|---:|---|
| experts, all 48 layers | 44640.00 | 144 | CPU (all offloaded) |
| GDN / linear layers (36) | 727.06 | 288 | GPU |
| shared expert + router | 297.56 | 240 | GPU |
| output + final norm | 243.44 | 2 | GPU |
| full attention layers (12) | 177.96 | 120 | GPU |
| token_embd | 166.92 | 1 | CPU |
| norms / other | 0.38 | 48 | GPU |
| **total** | **46253.31** | | |

Consistency check: non-expert total 1613.31 minus token_embd 166.92 = 1446.39, which equals
the measured CUDA0 model buffer. So with `-ncmoe 48` every layer's experts are already on CPU.
`-ncmoe` cannot be raised further (48 = all layers). There is no expert VRAM left to reclaim.

Per-layer sizes: experts 930.00, shared expert + router 6.20, GDN layer 20.20, attention layer 14.83.

## 3. Identified improvements

| # | improvement | saves (MiB) | cost / risk | status |
|---|---|---:|---|---|
| 1 | prefill streaming: remove host-weight staging scratch | `CUDA_Host compute` 345.31 -> ~64 | none, indirect win | implemented |
| 2 | prefill streaming: shrink `CUDA0 compute` | up to 356 (see section 4) | needs slots to fit | blocked |
| 3 | KV layer-by-layer streaming (2 of 12 attn layers) | 88.04 | code complexity | arithmetic only, small |
| 4 | evict `token_embd` (layer 0 only) + output (last layer) | 410.36 | these are CPU-side already, so it saves RAM not VRAM | not useful for VRAM |
| 5 | reduce `-ub` 2048 -> 512 | 378.82 | prefill throughput, see sweep | measured |
| 6 | `-ctk q8_0 -ctv q8_0` | none, it is worse | see section 4a | rejected |
| 7 | raise `-ncmoe` to keep experts resident for generation | - | impossible, already 48/48 | n/a |
| 8 | demote GDN/linear weights via `-ot` | **169.06** | +0.16% prefill, **-18.4% tg** | measured, NOT free |
| 9 | demote shared expert via `-ot` | **105.56** | see section 9 | measured |
| 10 | demote 8+9 together via `-ot` | **274.62** | +0.34% prefill, **-18.4% tg** | measured, NOT free |
| 11 | demote `output.weight` via `-ot` | 217.25 net | +80.18 compute | measured, see below |
| 12 | exclude experts from the streamed slot | **1150.14** | makes streaming start at all | see section 11 |

### Correction: `-ot` is not free for generation

Earlier versions of this table said the `-ot` saves were "free". That was measured on prefill
only and is wrong for a server. Measured `-ngl 99 -ncmoe 48 -lm none -p 512 -n 128 -r 1`:

| `-ot` | freed MiB | pp512 | tg128 | tg change |
|---|---:|---:|---:|---:|
| none | - | 38.70 | **18.51** | - |
| `ssm` + `shexp` | 274.62 | 38.58 | **15.11** | **-18.4%** |
| `(ssm\|attn_qkv\|attn_gate)` + `shexp` | 727.06 | 38.51 | **12.52** | **-32.4%** |

Prefill really is unchanged, generation is not. The demoted weights are read every decode step
instead of living on the GPU. So the correct recommendation for an interactive server is to
leave `-ot` off unless the VRAM is needed to make the context fit at all.

### The VRAM ceiling for this route

`-ncmoe 48` is already all 48 layers, so all experts are on the CPU host buffer
(`CUDA_Host model` 44806.92 MiB) and cannot move to VRAM. The only weights `-ot` can demote are
the 1446.39 MiB of non-experts. So the ceiling for this whole approach is about 1.4 GB, not the
9 GB one might hope for from "move the experts".

## 9. Measured -ot results

Command base: `-ngl 99 -ncmoe 48 -lm none -b 512 -ub 512 -p 2048 -n 0 -r 1 -ctk q4_0 -ctv q4_0 -fa 1`

| run | CUDA0 model | CUDA0 compute | CUDA_Host compute | pp2048 | freed |
|---|---:|---:|---:|---:|---:|
| baseline | 1446.39 | 512.02 | 10.28 | 38.33 | - |
| `ssm` | 1277.33 | 512.02 | 10.28 | 38.27 | 169.06 |
| `shexp` | 1340.83 | 514.30 | 10.28 | 38.21 | 105.56 |
| `ssm` + `shexp` | **1171.77** | 514.30 | 10.28 | 38.20 | **274.62** |
| `output.weight` | 1148.96 | 592.20 | 10.28 | 38.30 | 297.43 gross |

Patterns used:

```
ssm    = blk\.[0-9]+\.ssm.*
shexp  = blk\.[0-9]+\.ffn_.*shexp.*
linear = blk\.[0-9]+\.(ssm|attn_qkv|attn_gate).*     (this is the full 727.06 MiB bucket)
output = output\.weight
```

### Findings

1. **The saving is exactly the tensor size.** 169.06 and 105.56 are the exact GGUF byte counts,
   so `-ot` moves the weights and nothing else.

2. **The compute buffer does not shrink.** 512.02 stays 512.02 for `ssm`. So at `-ub 512` the
   compute buffer is NOT staging of these weights. It is transient activations. Streaming these
   weights therefore cannot reclaim compute-buffer space, because there is nothing to reclaim.
   This is the decisive negative result for the streaming path at this configuration.

3. **`output.weight` is different.** Demoting it raises the compute buffer by 80.18 MiB
   (512.02 -> 592.20), because when host-resident it must be staged as a graph input. Net saving
   is 217.25, not 297.43.

4. **Speed cost is negligible.** 274.62 MiB for 0.34% prefill, inside run-to-run noise.

### -ot separator syntax (a real trap)

`-ot` is parsed differently by llama-bench than by the CLI and server:

| tool | separator | effect |
|---|---|---|
| `common/arg.cpp:252` (CLI, server) | `,` | comma joins patterns into ONE override list |
| `tools/llama-bench/llama-bench.cpp:992` | `,` between groups, `;` inside a group | comma CROSS-PRODUCTS into separate runs |

So for llama-bench the combined form is `-ot 'a=CPU;b=CPU'`, while for the server it is
`-ot 'a=CPU,b=CPU'`. Using commas in llama-bench does not error, it silently runs each pattern
as its own benchmark, which is easy to misread as a combined result.

## 10. Consequences for the streaming design

The slots need 1992 MiB and the compute buffer cannot be reduced by moving weights, so the
streaming path has no VRAM to win at `-c 80000`. That number is wrong for the fix in section 11,
where the slots hold only the non-expert weights and cost 52.82 MiB.

The flag path, by contrast, frees 274.62 MiB for 0.34%.

Recommended server setting:

```
-ot 'blk\.[0-9]+\.ssm.*=CPU,blk\.[0-9]+\.ffn_.*shexp.*=CPU'
```

Deliberately excluding `output.weight`, because it raises the compute buffer more than it is
worth in a tight budget.

## 11. Why streaming fails at the server config, and the fix

Measured with the server's own placement (`-ngl 99 -ncmoe 48`), `-p 512 -b 512 -ub 512 -ps 1`:

```
prefill_streaming_init: prefill streaming enabled, 2 slots of 996.00 MiB, 48 layers, 144 tensors
model 1446.39 + KV 3.38 + RS 75.38 + slots 1992.00 = 3517.15   (free 198.85)
graph_reserve needs 309.53 MiB for the compute buffer ->
ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory
graph_reserve: failed to allocate compute buffers
```

At the server's `-c 80000` the KV buffer alone is 528.19 MiB, so the budget left for the compute
buffer goes negative and the context cannot be created at all. Note this fails in
`graph_reserve`, before the slots are even used, so it is a pure capacity problem.

### Root cause: the slot holds the wrong thing

`prefill_streaming_init` keeps every `blk.<il>.*` tensor that is host-resident. With
`-ncmoe 48` the experts are the *only* host-resident layer tensors, so the slot becomes the
expert block. Those weights do not leave `CUDA_Host` (44806.92 MiB), so the slot **duplicates**
them at a cost of 1992 MiB. Streaming is a net VRAM loss here, by construction.

| placement | slot | 2 slots |
|---|---:|---:|
| `-ngl 1 -ncmoe 99` (all layer tensors host-side) | 1022.41 | 2044.82 |
| `-ngl 99 -ncmoe 48` (experts host-side) | 996.00 | 1992.00 |

The difference, 26.41 MiB, is exactly the non-expert part of a layer (GDN 20.20 + shared
expert and router 6.20). Everything else in both slots is the expert block.

### The fix

Exclude the expert tensors from the slot. Then the slot is bounded by the largest
non-expert layer instead of by the expert block:

| | slot | 2 slots | freed |
|---|---:|---:|---:|
| today | 1022.41 | 2044.82 | - |
| experts excluded | 26.41 | **52.82** | **1992.00** |

Budget at the server config with experts on CPU, non-expert demoted to CPU, and small slots:

```
model            0.00          (everything was demoted)
KV             528.19
RS              75.38
slots           52.82
unaccounted    173.00
free_for_compute = 3716 - 829.39 = 2886.61   vs compute 1345.09  -> fits
```

This also removes the pp2048 failure at `-ngl 1 -ncmoe 99`, where the 2044.82 MiB slot is what
drops headroom to 176.72 MiB and trips `cudaMalloc`. With 52.82 MiB of slots, headroom becomes
about 2.1 GiB.

### The unresolved part

Streaming only fires when `n_tokens > 1`, so it is prefill-only. During generation the
demoted weights must be read per token, and a "restore before generation" step needs about
1203 MiB of free VRAM at a moment when the compute buffer is already reserved at its prefill
maximum. That restore path is the real open problem, not the slot size.

## 12. Streaming gain depends on the ubatch count

Measured, 80B Q4_K_M, `-ngl 1 -ncmoe 99 -lm none -b == -ub == p -r 1`:

| p | b=ub | ubatches | baseline | streaming | delta |
|---:|---:|---:|---:|---:|---:|
| 512 | 512 | 1 | 38.09 | 64.55 | **+69.5%** |
| 1024 | 1024 | 1 | 54.31 | 87.01 | **+60.2%** |
| 1536 | 1536 | 1 | 64.85 | **97.80** | **+50.8%** |
| 2048 | 2048 | 1 | 71.83 | FAIL | VRAM |
| 1536 | 1024 | 1.5 | 47.28 | 43.18 | -8.7% |
| 1024 | 512 | 2 | 37.95 | 33.50 | -11.7% |
| 1536 | 512 | 3 | 37.97 | 33.53 | -11.7% |
| 2048 | 1024 | 2 | 53.80 | 50.51 | -6.1% |
| 2048 | 1536 | 1.33 | 54.95 | 51.27 | -6.7% |
| 2048 | 512 | 4 | 53.85 | 50.37 | -6.5% |

Exactly one ubatch per mmap batch: large win. More than one: small consistent loss. The slots
are re-seeded synchronously at every ubatch start (`llama-context.cpp`, `process_ubatch`), so
the refill cost is paid per ubatch and only amortizes when there is one. The baseline
scheduler instead overlaps expert staging across ubatches with an async `cpy_tensor`.

So the claim is `+50..70% for single-ubatch prefill`, which requires `-b >= p`. It is not a
claim about prompt length. This supersedes the accuracy of the earlier pp2048 row (`-8.3%`),
which was a two-ubatch config, and it confirms the pp512 and pp1024 rows.

## 8. Layer-by-layer granularity of the three non-expert options

For streaming, what matters is not the total bytes but whether each piece already sits inside a
per-layer region. Weights pinned at the graph front ("global") do not.

| option | size (MiB) | tensors | all inside blk.N.* regions? | verdict |
|---|---:|---:|---|---|
| 1a. `ssm` weights (36 layers) | 169.06 | 216 | **yes**, all `blk.N.ssm*` | already per-layer, 4.70 MiB/layer |
| 1b. `attn_qkv` + `attn_gate` (36 layers) | 558.00 | 72 | **yes** | already per-layer, 15.50 MiB/layer |
| 1 (full GDN/linear bucket) = 1a+1b | 727.06 | 288 | **yes** | already per-layer |
| 2. shared expert | 105.56 | 192 | **yes**, all `blk.N.ffn_*_shexp*` | already per-layer, ~2.2 MiB/layer |
| 2b. routers `ffn_gate_inp*` | 192.19 | 96 | **yes** | already per-layer, 4.00 MiB/layer |
| 3. output + final norm | 243.56 | **2** | **no**, `output.weight` + `output_norm.weight` | global, not per-layer |

The split in 1 matters because `ssm` alone is only 169.06 MiB while the linear weights
(`attn_qkv` 396.00 + `attn_gate` 162.00 = 558.00) are the larger part of the same bucket. In this
model the 36 GDN layers carry the linear attention weights, and the 12 full-attention layers are
named differently, so a pattern must cover both groups to get the whole 727.06.

Option 3 is the awkward one: it is two tensors, not a layer. `output.weight` is used once, in the
final matmul, so it can never be streamed layer-by-layer. It could only be streamed as a one-shot
transfer before the final node, which is a different mechanism from the layer slots.

Note on #4: `token_embd` and `output` are already on CUDA_Host in the measured config, so
freeing them does not add VRAM. They only matter for RAM. The 410.36 MiB figure is their
weight size, not a VRAM saving.

## 4. Compute buffer scales with both -c and -ub

| -c | -ub | CUDA0 compute | CUDA_Host compute | KV |
|---:|---:|---:|---:|---:|
| 80000 | 2048 | 1345.09 | 345.31 | 528.19 |
| 80000 | 512 | 966.27 | 86.53 | 528.19 |
| 16000 | 512 | 529.75 | 24.03 | 106.31 |
| 8000 | 2048 | 796.00 | 64.31 | 54.00 |
| 8000 | 1024 | 608.00 | 32.29 | 54.00 |
| 8000 | 512 | 514.00 | 16.28 | 54.00 |
| 8000 | 384 | 490.50 | 12.27 | 54.00 |
| 8000 | 256 | 467.00 | 8.27 | 54.00 |

`CUDA_Host compute` is the scratch that exists only because host-resident weights must be
staged into the graph. It falls from 345.31 MiB to 8.27 MiB as the working set shrinks.
This is the quantity streaming removes, and it is roughly 4x larger than the KV streaming prize.

## 4a. The CUDA0 compute buffer has a hidden context-dependent term

At `-ub 512` the CUDA0 compute buffer is 514.00 MiB at `-c 8000` but 966.27 MiB at `-c 80000`.
That 452.27 MiB difference is not explained by `-ub`, and no `-ub` sweep can remove it.

Cause: f16 KV staging for quantized KV in flash attention.

`ggml/src/ggml-cuda/fattn.cu:669` `ggml_cuda_flash_attn_ext_get_alloc_size()` extends the
compute-buffer allocation of every `FLASH_ATTN_EXT` node with `ggml_cuda_flash_attn_ext_get_f16_extra_data()`
(`fattn-common.cuh:53`). When the selected kernel needs f16 K or V and the cache type is not f16:

```
data.end += ggml_nelements(K)*ggml_type_size(GGML_TYPE_F16)
```

`K` is the whole cache tensor, so the size scales with `n_ctx`, not with `n_kv` and not with
`n_ubatch`. With `-ctk q4_0 -ctv q4_0` the staging is always needed, because the kernel
selector only skips the f16 copy for the VEC quantized kernel, which is gated on `Q->ne[1] == 1`
(one query row, generation) on this device. Prefill uses MMA_F16, which needs f16 and therefore
copies.

Numbers at `-c 80000`: `nelements(K) = 512 x 80128 = 41,025,536` -> 78.3 MiB f16 for K, the same
for V -> 156.6 MiB per attention layer. Two or three nodes are live at once, giving 313 to 470 MiB,
which brackets the observed 452.27 MiB. At `-c 8000` the same term is only about 31 MiB, which is
why an `-ub` sweep alone could not explain the delta.

| KV type | KV buffer | f16 staging | total | note |
|---|---:|---:|---:|---|
| f16 | ~1877 | 0 | ~1877 | staging gone, but KV itself is huge |
| q8_0 | ~1056 | ~452 | ~1508 | WORSE than q4_0 |
| q4_0 (current) | 528 | ~452 | ~980 | best of the three |

The staging is one attention layer worth of f16 K+V regardless of cache type, so it does not
disappear when the cache gets bigger. q8_0 therefore ADDS about 528 MiB of KV for no staging
benefit. The current `-ctk q4_0 -ctv q4_0` is already the right choice. An earlier note claiming
q8_0 removes the staging was wrong.

Cross-check at `-c 8000 -ub 512`: measured compute 514.00 = valid activations ~429 + KV 54 +
staging ~31. The same decomposition holds, so the ~429 MiB is the ub-driven activation term.

## 5. Can the two streaming slots fit?

Slots are one full layer each: 996.00 MiB, so 2 slots need 1992.00 MiB.

| config | free VRAM (MiB) | shortfall (MiB) |
|---|---:|---:|
| c80000 ub2048 (server) | 147 | -1845 |
| c80000 ub512 | 547 | -1445 |
| c8000 ub2048 | 1171 | -821 |
| c8000 ub512 | 1453 | -539 |

They never fit. Fixed floor at `-c 8000 -ub 512` is model 1446.39 + KV 54.00 + RS 75.38 +
unaccounted 173 = 1748.77 MiB, leaving 1967.23 MiB for slots plus compute. A single slot is
996.00, so two slots plus any compute buffer exceeds that.

## 6. Why the design doc's overlap condition does not hold here

CORRECTION: an earlier version of this section claimed the activated expert fraction is about
2 percent and that the transfer therefore cannot be hidden. That was WRONG, see section 6a.

`DoubleBuffer_Perfill_opt.md` assumes dense layer weights of about 375 MB and argues
`T_compute >= T_transfer` for `N_tokens >= 1024`, hiding the PCIe transfer entirely.

## 6a. The activated expert fraction is NOT 2 percent

A single token selects 10 of 512 experts, which is where the 2 percent figure came from. But an
ubatch of 512 tokens corresponds to 5120 expert selections, and 5120 selections over 512 experts
means every expert is expected to be touched. So the fraction of loaded expert bytes that is
actually used is close to 100 percent for any practical ubatch.

The union lower bound is $n_{\text{used}}/n_{\text{experts}}$ from the union bound
$1 - (1 - k/n)^T$ evaluated in the other direction; at $T = 512$ and $k/n = 10/512$ roughly
63 percent of experts are individually guaranteed to be touched, and the expected distinct count
is $512 \cdot (1 - (502/512)^{512}) \approx 512 \cdot (1 - e^{-10}) \approx 511$. So essentially
all of them.

Consequence: there is nothing to save by being selective about which experts to transfer. The
whole 930.00 MiB per layer is needed, so the transfer volume is not compressible, and the MoE
structure is not a weakness to exploit here.

Also, the copy runs on a background stream, so transfer time does not add to the critical path.
It affects throughput only through bandwidth contention, and RAM usage. So the useful question is
not "can the transfer be hidden" but "is it already overlapped by the scheduler".

## 7. What the 539 MiB shortfall actually includes

This section answers the question of which improvements are already counted in the shortfall
table, because the table is easy to misread.

The shortfall is computed as

```
    shortfall = 2 x slot_size - free
    free      = 3716 - model - KV - RS - unaccounted - compute
```

Note the `- compute` term. The compute buffer is reserved by `graph_reserve()` before
`prefill_streaming_init()` tries to allocate the slots, so the slots must fit in whatever is left
after compute. Worked example for the 539 row:

```
    3716 - 1446.39 - 54.00 - 75.38 - 173 - 514.00 = 1453.23  ->  shortfall 1992 - 1453 = 539
```

and the same formula reproduces the server row:

```
    3716 - 1446.39 - 528.19 - 75.38 - 173 - 1345.09 = 147.95 ->  shortfall 1992 - 148 = 1844
```

So which improvements are in the 539? Only the ones baked into that row's column values:

| improvement | effect on `free` | already in the 539 row? |
|---|---|---|
| `-c` 80000 -> 8000 | KV 528.19 -> 54.00, compute 966 -> 514 | **yes**, the row is `-c 8000` |
| `-ub` 2048 -> 512 | compute 1345.09 -> 514.00 | **yes**, the row is `-ub 512` |
| `-ot` non-expert weights | model 1446.39 -> 1171.77 | **no** |
| KV layer-by-layer streaming | KV smaller by ~88 | **no** |
| unaccounted 173 MiB | not addressable | yes, as a fixed cost |

Adding the measured `-ot` saving on top of the 539 row:

```
    free      = 1453.23 + 274.62 = 1727.85
    shortfall = 1992 - 1727.85   = 264.15
```

So the 539 MiB is the shortfall with only `-c` and `-ub` already reduced. Applying the `-ot`
win takes it to about 264 MiB. KV streaming would take off a further ~88, giving about 176 MiB.

None of them close it. The structural reason is that a slot is 996.00 MiB while the entire
non-expert, non-token_embd weight set is 1446.39 MiB. Two slots are more than the whole rest of
the model, so no amount of marginal trimming reaches 1992 MiB.

## 13. The slot scales with the host weight set, not with the streamed part

This is the finding that decides the design. Demoting the GDN weights via `-ot` at the server
placement makes the slot slightly BIGGER, not smaller:

| config | `-ot` | tensors in slot | slot (MiB) | CUDA0 model |
|---|---|---:|---:|---:|
| `-ngl 99 -ncmoe 48` | none | 144 | 996.00 | 1446.39 |
| `-ngl 99 -ncmoe 48` | `ssm` | 360 | **1000.70** | 1277.33 |

The slot grew by 4.70 MiB, exactly the `ssm` bytes, because `get_prefill_streaming_layer_tensors`
collects every host-resident `blk.<il>.*` tensor. Demoting a weight into the slot-ready set
therefore charges the slot for it as well as charging `CUDA_Host` for it. Both copies are live,
so `-ot` and streaming are **zero-sum**:

```
-with -ot ssm:      model 1277.33 + KV 3.38 + RS 75.38 + slots 2001.40 = 3357.49
-without -ot:       model 1446.39 + KV 3.38 + RS 75.38 + slots 1992.00 = 3517.15
```

Net VRAM saved: **none**. And in both cases `graph_reserve` then needs 309.53 MiB for the
compute buffer and `cudaMalloc` fails.

So the real defect is not the slot size. It is that the slot holds the same weights as
`CUDA_Host`. That is the expert case too, where the duplication is 1992 MiB.

### The duplication-free design

A slot only pays off if the weights it holds are live ONLY in the slot. That means:

1. `prefill_streaming_init` must exclude the experts from the slot, otherwise the slot
   duplicates the 996.00 MiB expert block.
2. The non-expert weights must be assigned exclusively to the slot buffer instead of ending up
   in both `CUDA0` and the slot. Streaming the same tensor from two buffers is not a saving.

Neither is true today, which is why every measured configuration is worse than the baseline.

### What I got wrong

I earlier computed a 1150.14 MiB saving for "exclude experts from the slot". That figure is
still correct for the slot itself, but it comes with a second change I did not check: the
non-expert weights have to move out of `CUDA0` at the same time. `-ot` alone does not do that,
it makes the duplication worse. The 1150.14 is therefore a conditional number, not a win that
`-ot` delivers.

### One remaining unknown

The `cudaMalloc` failure happens in `graph_reserve`, after `prefill_streaming_init` has already
allocated the 2 slots. That ordering is what makes the failure unconditional: at the server
config the model plus two expert-sized slots can never leave room for compute. With
expert-excluded slots (52.82 MiB) this resolves, because then the model, KV, RS and slots
together come to well under half of VRAM. That arithmetic has not been measured yet, because it
needs the code change.

## 8. Open questions

1. Does `-ub 256/384/512` beat `-ub 2048` on prefill throughput? Sweep interrupted, not concluded.
2. Is the slot worth reworking to sub-layer granularity? Design change, needs agreement.
3. Does excluding experts from the slot, and giving the non-expert weights a single home,
   make the feature fit? Needs the code change to measure.
4. `-ps` is a bare toggle in the CLI and server but takes `<0|1>` in llama-bench. Pre-existing
   inconsistency, not yet aligned.
