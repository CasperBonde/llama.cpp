# Useful parameters - model, GPU, and derived limits

All values from `--help`/GGUF metadata/`cudaGetDeviceProperties`/`nvidia-smi` on this box.

## Model: Qwen3-Next-80B-A3B-Instruct-Q4_K_M

| param | value |
|---|---|
| file | `models/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf` |
| size | 45.17 GiB |
| params | 79.67 B |
| n_layer | 48 |
| n_embd | 2048 |
| n_ff (dense) | 5120 |
| n_expert | 512 |
| n_expert_used | 10 |
| moe_intermediate_size | 512 |
| n_embd_head_k | 256 |
| n_embd_head_v | 256 (n_embd_k_gqa/v_gqa = 512) |
| attention layers (full) | 12 |
| GDN / linear layers | 36 |
| expert tensors | 3 per layer (gate/up/down), 144 total, 44 640 MiB |
| per-layer expert bytes | 930 MiB |
| per-layer non-expert | 26.41 MiB (GDN) / 21.03 MiB (attention) |

Derived per token (prefill):
- MoE FLOPs = 10 x 3 x 2 x 2048 x 512 x 48 = **3.02 GFLOP/token**
- streamed weight bytes per ubatch = 45.86 GiB (whole model, every ubatch)

## GPU: Quadro T1000 (TU117, sm_75)

| param | value |
|---|---|
| SMs | **14** |
| FP32 cores | 896 (64/SM) |
| core clock | 1455 MHz |
| **FP32 peak** | **2.61 TFLOP/s** |
| **L2 cache** | **1024 KB** |
| shared mem / block | 48 KB |
| registers / block | 65536 |
| memory clock | 4001 MHz |
| bus width | 128 bit |
| **DRAM peak** | **128 GB/s** |
| achievable DRAM | ~110 GB/s (85% of peak, measured) |
| PCIe | Gen3 x16, practical ~12.4 GB/s |
| copy engines | 3 |
| tensor cores | **none** (Turing TU117 has no TC) |
| concurrent kernels | 1 |

## Derived limits

| quantity | value |
|---|---|
| ridge point (FLOP/byte) | 2.61e12 / 128e9 = **20.4** |
| MoE intensity at ub=512 | 3.02e12 x 512 / 45.86e9 = **33.7 FLOP/byte** |
| MoE intensity at ub=2048 | 3.02e12 x 2048 / 45.86e9 = **135 FLOP/byte** |
| M per expert at ub=512 | 5120 pairs / 512 experts = **10** |
| M per expert at ub=1536 | 15360 / 512 = **30** |
| experts touched at ub=512 | ~511 of 512 (512 x (1-(502/512)^512)) |
| experts touched at ub=64 | ~365 of 512 |

Consequence: on this GPU the MoE layer is **compute-bound** at prefill batch sizes, because
33.7 FLOP/byte at ub512 already exceeds the 20.4 ridge. That is why the measured 12% SM
utilization is a puzzle: a compute-bound kernel should be near 100%.

## Simulator: the other model (for reference)

Qwen3-Coder-30B-A3B, 48 layers, expert_count 128, expert_used_count 8, per-layer experts
373.50 MiB, layer total ~385 MiB, 2 slots = 747 MiB.

## Measured baseline numbers to compare against

| config | t/s |
|---|---:|
| pp512, b=ub=512, streaming | 64.5 |
| pp512, b=ub=512, baseline | 38.2 |
| pp1024, b=ub=1024, streaming | 84.8 |
| pp2048, b=1024 ub=1024, streaming | 48.9 |
| pp512 NOCOPY (no copy at all) | ~123 |
