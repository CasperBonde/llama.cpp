# Prefill streaming - measured findings

Date: 2026-09-13
Branch: prefill_double_buffering2 (HEAD 2eb542a4d)
Model: Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf (45.17 GiB, qwen3next, 48 layers)
GPU: Quadro T1000, 3716 MiB. 12 CPU cores. Storage is NVMe (~1.2 GB/s at 1M blocks).

All numbers below come from logs in this directory. No estimates.

## 1. The 1.5B numbers in the old logs are NOT the 80B

/tmp/an512.log and /tmp/an1024.log say "2 slots of 373.50 MiB, 144 tensors".
That is a 1.5B model, not the 80B. The 80B logs are /tmp/e512.log and
/tmp/e1024.log: "2 slots of 1022.41 MiB, 840 tensors".

Do not mix these up. The 373.50 / 1022.41 gap is model size, not a filter
difference.

## 2. The host filter already selects experts only at -ngl 99

At -ngl 99 -ncmoe 99 the only host-resident tensors are the MoE experts.
The slot is sized from those, so the failure message gives the exact size:

    prefill_streaming_init: failed to allocate slot 1 (1044381696 bytes)
    1044381696 / 1048576 = 996.00 MiB

996.0 MiB is exactly the sum of the three expert tensors of one 80B layer
(ffn_down_exps 420 + ffn_gate_exps 288 + ffn_up_exps 288).

So `get_prefill_streaming_layer_tensors` + the `is_host` check already does the
right thing. No narrowing of the tensor list is needed for -ngl 99.

## 3. The compute buffer does NOT shrink, so there is no JIT staging duplicate

30B, -ngl 99 -ncmoe 99, c=2048 -b 512 -ub 512:

    streaming off:  CUDA0 compute buffer 219.51 MiB
    streaming on :  CUDA0 compute buffer 219.51 MiB
    streaming on :  prefill streaming enabled, 2 slots of 373.50 MiB, 144 tensors

Identical. Hypothesis that ~420 MiB of the 80B compute buffer is a duplicate of
the expert weights is WRONG. There is no second copy to reclaim.

## 4. The gating constraint is the compute buffer, not staging

Why: CUDA_Host is last in the buft priority list, so `ggml_backend_sched_backend_from_buffer`
returns the CUDA backend for an op whose weights are in CUDA_Host. The op is then
scheduled on CUDA0 and the host tensor is staged into the compute buffer at eval
time (see the MUL_MAT_ID path around ggml-backend.cpp:1706).

Our `tensor_bind` swaps in a slot-backed tensor with the exact same shape, so the
scheduler makes the same choice and asks for the same staging block. The two paths
coincide exactly => no net change in compute buffer. The win is that the disk/PCIe
copy happens earlier (prefetch) instead of inline.

## 5. 80B VRAM budget

    VRAM total                                    3716.00
      CUDA0 model (attn/norm/emb/output)          1446.39
      KV (c=2048) + RS                             123.38
      CUDA0 compute buffer                          604.04
      USED                                        2173.81
      free for slots                              1542.19

    need 2 x 996.00 = 1992.00   -> SHORT by 449.81 MiB

This is why the 80B fails at -ngl 99. It is a hard VRAM limit, not a bug.

## 6. Slot fit matrix (80B)

    2 full-layer slots (ngl 1 path)   2044.82   no
    2 expert-only slots (ngl 99 path) 1992.00   no
    free after 2 slots at ngl 1       ~1227     (what the working config uses)

At -ngl 1 the slots are 1022.41 MiB each and it fits because the model leaves
only 243.44 MiB in VRAM (output.weight) instead of 1446.39.

## 7. Placement was always 100% GPU in our prototype

    /tmp/e1024.log: 840 graph ops read slot tensors, 0 of them scheduled on the CPU
    slot device: CUDA0

The 70% CPU figure belongs to the PR 28414 branch at -ngl 99 -ncmoe 99, where
`-ncmoe` deliberately runs the expert matmul on the CPU. It was never a property
of our implementation. Do not cite it for our branch.

## 8. Refill traffic

    prefill streaming ubatch of 512 tokens: 2 refills, 2044.82 MiB, 172.388 ms (12.44 GB/s)
    prefill streaming ubatch of   2 tokens: 2 refills, 2044.82 MiB, 174.448 ms (12.29 GB/s)

Refill is 2 slots = 2/48 of the model regardless of ubatch size, and it costs
~174 ms. With -b 1024 -ub 1024 there is one ubatch per 1024 tokens, so the cost
is ~174 ms per 1024 tokens. That is the maximum attainable pp rate:

    1024 tokens / 0.174 s = 5885 t/s   <- traffic ceiling, not compute

So at large ub the copy is NOT the limit; at small ub it is, because the same
174 ms is paid per tiny ubatch.

## 9. Why the 30B is much slower with streaming at tiny ub

    30B, ub 2 tokens: 6.1 t/s

Two refills of 747 MiB for 2 tokens. Pure overhead. Confirms that ubatch size is
the controlling parameter and that streaming must be paired with a large ub.

## 10. Consequences for the design

- Two slots of a full layer is the minimum that can overlap, and it does not fit
  the 80B at -ngl 99.
- Shrinking the slot to experts only does not help the 80B: 1992 vs 2044 is a
  52 MiB saving against a 450 MiB deficit.
- Freeing hidden-byte that we wrongly believed existed (the "staging duplicate")
  is not available.
- Real options, in order of promise:
  a) narrower slot contents (only what prefill needs), see next section
  b) KV streaming (KV is 96 MiB at c=2048, ~1070 MiB at c=80000 with q4_0)
  c) a GPU with more VRAM

## 11. The per-tensor staging block is the real worst case

The scheduler sizes the compute buffer as, roughly:

    max_block * GGML_SCHED_MAX_COPIES + max(compute_metadata_for_graph)

At -ngl 99 the largest host tensor is ffn_down_exps = 420 MiB, and that single
block dominates the 604 MiB. If the largest streamed tensor is bounded, the
compute buffer shrinks. This is the lever the PR 28414 design was reaching for,
and it is available to us as long as we stream from the SLOT rather than from a
CUDA_Host buffer.

Not yet measured: whether the compute buffer at -ngl 99 with slot-backed experts
(no CUDA_Host experts at all) drops below 604. That requires the placement to
succeed first, which the 80B cannot do at -ngl 99 on this GPU.

Actionable version: keep `-ncmoe` so the experts are never counted as VRAM, and
have the slot be the only CUDA-visible copy. Then the compute block for the
expert op is the slot itself (a WEIGHTS buffer, already resident) and no staging
block should be needed. The measured "unchanged 219.51 MiB" above says the current
code does not yet achieve this, because the scheduler still stages.

## Log index

    base_ngl1.log            80B ngl 1, no streaming
    base_ngl99_ncmoe99.log   80B ngl 99 ncmoe 99, no streaming
    ngl99_stream.log         80B ngl 99 ncmoe 99 + streaming -> slot 996.0 MiB, does not fit
    30b_off.log              30B ngl 99 ncmoe 99, no streaming
    30b_on.log               30B ngl 99 ncmoe 99 + streaming -> fits, compute unchanged
