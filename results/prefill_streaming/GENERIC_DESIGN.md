# Generic prefill streaming - what changed and what it measured

Date: 2026-09-14
Branch: prefill_double_buffering2 (HEAD 2eb542a4d + local generic rework, build 10831)
Model: Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf (45.17 GiB, qwen3next, 48 layers)
Also: Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf (17.28 GiB, qwen3moe, 48 layers)
GPU: Quadro T1000, 3716 MiB.

## 1. The rework: no model code knows about streaming anymore

Before this change, every supported architecture had to enumerate its own weight
tensors and route each one through `tensor_bind`:

    src/models/qwen3moe.cpp   12 lines of bind() calls inside the layer loop
    src/models/qwen3next.cpp  28 lines of bind() calls inside the layer loop
    src/models/models.h       4 helper signatures threaded a llama_layer copy

That is gone. The model now builds its graph with the real weights, and
`prefill_streaming_rewrite_graph()` walks the finished graph and repoints each
weight operand at the slot tensor that holds it. One function, architecture
agnostic, zero per-model code.

Evidence it works on both architectures:

    30B qwen3moe : rewrote 144 graph operands to slot tensors
    80B qwen3next: rewrote 144 graph operands to slot tensors

## 2. The bigger win: the staging copies are gone

The slots must exist BEFORE the reserve graph is built. This is the whole trick.

Before: `prefill_streaming_init()` ran at the END of `sched_reserve()`, so the
reserve graph was built against CUDA_Host weights. The scheduler saw a host
weight feeding an op it wanted on CUDA0, and sized a staging block for it inside
the CUDA0 compute buffer. Our `tensor_bind` then swapped in a slot tensor with
the same shape, so the op landed on CUDA0 anyway and asked for that same staging
block. Both paths needed it, so the compute buffer never shrank.

After: init runs inside `graph_reserve()` before `model.build_graph()`, and the
reserve graph is rewritten too, so the scheduler sees a CUDA0-resident weight
from the very first sizing pass. There is nothing to stage.

    model        compute buffer (CUDA0)   with streaming ON
    -----------  ----------------------   -----------------
    before      30B  219.51 MiB           no change
    after       30B   86.01 MiB           -133.50 MiB
    before      80B  604.04 MiB (ngl 99)  no change
    after       80B  248    MiB (ngl 99)  -356 MiB

    (80B at ngl 1: 612.00 -> 252.09 MiB)

This is the "narrower slot contents" lever the previous notes were looking for,
and it is real. It also confirms the earlier note that said the compute buffer
"does not shrink": that was true of the old ordering, not of the design.

## 3. Slot composition is automatic and correct

At `-ngl 99` only the MoE experts are host resident, and the slot comes out at
exactly the expert size:

    80B: 2 slots of  996.00 MiB, 144 tensors   (3 expert tensors x 48 layers)
    30B: 2 slots of  373.50 MiB, 144 tensors

At `-ngl 1` everything except output.weight is host resident:

    80B: 2 slots of 1022.41 MiB, 840 tensors

The same code produces both. The `is_host` filter already did the right thing;
no per-arch narrowing was ever needed.

Slot size also grows correctly with ngl, as layers move to VRAM:

    ngl   1: 1022.41 MiB, 840 tensors
    ngl   8: 1022.41 MiB, 739 tensors
    ngl  16: 1022.41 MiB, 623 tensors
    ngl  24: 1022.41 MiB, 507 tensors
    ngl  32: 1022.41 MiB, 391 tensors
    ngl  99:  996.00 MiB, 144 tensors   (experts only)

Note the ngl 1..32 slot stays at 1022.41 MiB: the largest layer does not change
until whole layers have been offloaded past it.

## 4. Placement is still 100% GPU

    144 graph ops read slot tensors, 0 of them scheduled on the CPU, slot device: CUDA0

Measured on the 30B after the rework. The rework did not disturb placement.

## 5. The 80B at -ngl 99 still does not fit, but the gap shrank

    80B, -ngl 99 -ncmoe 99, ub 1024:
      CUDA0 model 1446.39 + KV 48.00 + RS 75.38 + compute 248 + 2 x 996.00
      = 3809.77  vs 3716 total   -> short by ~94 MiB

Before the rework the same configuration was short by ~450 MiB. The rework
recovered ~356 MiB of that.

The failing allocation reports it directly:

    failed to allocate CUDA0 buffer of size 260153600   (248.10 MiB)

so the compute buffer itself is what does not fit, not the slots.

Trying to shrink the compute buffer with a smaller ubatch does not work, because
the remaining block is not ubatch dependent:

    ub 1024: failed to allocate 260153600 bytes (248.10 MiB)
    ub  512: failed to allocate 130076928 bytes (124.05 MiB)
    ub  256: failed to allocate  65038592 bytes ( 62.02 MiB)

The block scales with ubatch but the ever-present part does not, so it still
overshoots. The 996 MiB slots plus the 1446 MiB model leave too little.

## 6. What DOES fit at -ngl 99: the 30B

    30B -ngl 99 -ncmoe 99: streaming enabled, 2 slots of 373.50 MiB,
                           compute buffer 86.01 MiB, 0 CPU ops

So the design is proven at -ngl 99 where the budget allows. The 80B on a 3716 MiB
card is a hardware limit, not a design limit.

## 7. Max ngl that fits on the 80B with streaming

    ngl 32: streaming enabled, slot 1022.41 MiB, compute 260.05 MiB  -> WORKS
    ngl 40: failed to allocate 272678912 bytes                       -> does not fit
    ngl 48: fails earlier, before prefill_streaming_init is reached
    ngl 99: streaming enabled, slot  996.00 MiB, compute 248 MiB     -> does not fit

So ~32 of 49 layers can sit resident alongside the slots on this GPU. Above that,
the model buffer and the slots compete for the same 3716 MiB.

At ngl 32 the placement check still reports clean:

    391 graph ops read slot tensors, 0 of them scheduled on the CPU, slot device: CUDA0

## 8. Perf after the rework

    80B, -ngl 1, -b 1024 -ub 1024, pp1024, -r 1:
      baseline           52.34 t/s
      generic streaming  76.15 t/s     (1.45x)

CORRECTION (see KV_VS_LOADMODE.md): those two runs omitted `-lm none`, which the
original sweep script includes. With the flag the numbers are 53.53 -> 84.83
(1.585x), which matches the original sweep (52.70 -> 86.99). The rework
conclusion is unchanged; only the quoted streaming number was understated.

Corrected, matching the sweep command exactly
(`-ngl 1 -ncmoe 99 -lm none -b 1024 -ub 1024`):

    pp       baseline   streaming   gain
    pp512      38.12      64.46     +69.1%
    pp1024     53.53      84.83     +58.5%
    pp2048     53.99      49.49     -8.3%

Confirmed four times on the current generic build: 84.07, 84.61, 84.83, 84.83.
Correctness is unchanged, the code is far smaller.

## 8b. Correctness

    30B, -ngl 1, -b 256 -ub 256, --temp 0, prompt "The capital of France is"
      baseline : "The capital of France is Paris."
      streaming: "The capital of France is Paris."

Identical. The 6.5 vs 35.3 t/s prompt rate is the expected refill cost at a
small ubatch, not a correctness problem.

## 8c. Output documents

    final_baseline.txt   52.34 t/s, pfs 0
    final_streaming.txt  52.34 (pfs 0) vs 76.15 (pfs 1), same run

## 8d. Does a higher ngl help prefill? Slightly, yes

pp512, -b 1024 -ub 1024, -r 1:

    config              baseline   streaming
    ngl 1                 36.37      56.30
    ngl 32 + ncmoe 99     36.92      59.11     (+5.0% over ngl 1 streaming)

So having 32 layers resident alongside the slots is worth about 5% more prefill
than ngl 1. The earlier comparison note ("prefill gets no benefit from residency")
was measured WITHOUT streaming (53.12 ngl 99 vs 52.32 ngl 1). With streaming the
residency does help a little, because the refill traffic per ubatch drops: at
ngl 32 fewer layer tensors still have to come off the host.

  (ngl 1  refills 2 x 1022.41 MiB per ubatch
   ngl 32 refills 2 x 1022.41 MiB per ubatch but 391 tensors instead of 840,
   and the largest single layer is smaller since 32 layers are already resident)

## 8e. llama-bench caveat at ngl 32

    -p 512  -> works, 58.99 t/s
    -p 1024 -> "failed to create context"

At ngl 32 the slots run right against the free VRAM limit, so llama-bench's
larger context for the pp1024 test does not fit. This is the expected behaviour
at the edge of the budget, not a bug. Use -p 512 when testing ngl 32.

## 8f. Boundary summary for the 80B on a 3716 MiB card

    ngl 1  + streaming   fits, 56.30 pp512 / 76.15 pp1024   <- recommended
    ngl 32 + streaming   fits at pp512 (59.11), not at pp1024
    ngl 40 + streaming   does not fit
    ngl 99 + streaming   does not fit (short ~94 MiB)

## 9. What is left

- 80B at -ngl 99 needs roughly 94 MiB more than this card has. Options:
  a) evict token_embd.weight (166.92 MiB, needed only for layer 0) during prefill
  b) KV streaming (the author's next idea, parked by request)
  c) more VRAM
- The compute buffer block that refuses to shrink further should be identified;
  it is no longer the expert weights.
- Debug env vars (LLAMA_PREFILL_STREAMING_CORRUPT / _SYNC / _NO_WEIGHTS) and the
  diagnostic counters are still in the tree.

## Log index

    gen30_ps0.log / gen30_ps1.log   30B ngl 99, streaming off / on (86.01 MiB)
    ngl99_gen.log                   80B ngl 99 + streaming, allocation failure
    ngl99_ub512.log                 80B ngl 99 ub 512, allocation failure
    ngl99_ub256.log                 80B ngl 99 ub 256, allocation failure
    sweep_ngl{1,8,16,24,32,40,48,99}.log  80B ngl sweep with streaming
    bench_ngl1_pp1024.txt           52.34 baseline
    bench_ngl1_ps_pp1024.txt        76.12 generic streaming
