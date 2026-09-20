# Token tiling: what was tested, what was wrong with the test, and what the kernel already does

Date: 2026-09-19. `-ngl 1 -ncmoe 99 -lm none`, one process, `-ps 0,1`.

---

## 0. CORRECTION: the experiment in section 1 tested the wrong thing

Section 1 below ran with `-b == -ub == 64`. That makes llama.cpp issue eight separate
**ubatches**, and the streaming path refills the slots once per ubatch. So that run measured
"re-stream every layer over PCIe eight times", which is exactly the thing that must not be
done. It says nothing about tiling tokens **inside** the GPU on a 1024-token ubatch.

The proposal under test is: keep `-ub 1024`, and have the GPU process tokens in tiles of 64,
one layer at a time. The test below does not measure that.

---

## 1. The measurement that was run (and why it is not the proposal)

Fixed prompt 512, varying ubatch size. `-b == -ub == tile`, one process, `-ps 0,1`, `-r 3`.
This changes how many ubatches are issued, i.e. how many times the slots are refilled.

| tile | baseline t/s | streaming t/s | vs ub=512 |
|---:|---:|---:|---:|
| 64 | 18.24 | 13.90 | 0.26x |
| 128 | 16.78 | 21.34 | 0.40x |
| 256 | 24.77 | 20.26 | 0.38x |
| 512 | 36.30 | 52.88 | 1.00x |
| 1024 | 36.34 | 52.82 | 1.00x |

This is a valid measurement of "smaller ubatch is worse", which is unsurprising: it multiplies
the PCIe copy (48 layer visits at ub=512 becomes 384 at ub=64, i.e. 46.7 GiB -> 373.5 GiB).
It is NOT a measurement of GPU-side token tiling.

## 2. The important finding: the kernel ALREADY tiles tokens

`ggml/src/ggml-cuda/mmq.cuh`:

```cpp
struct ggml_cuda_mmq_config {
    int I;  // SRAM tile width in src0->ne[1]/dst->ne[0] direction   (output rows)
    int J;  // SRAM tile width in src1->ne[1]/dst->ne[1] direction   (tokens)
```

and the selector that runs at every launch (`mul_mat_q_switch_J`, mmq.cuh:1473):

```cpp
for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_config(type, J, fallback, cc);
    if (mmq_get_nbytes_shared(config, cc) > smpbo) { continue; }
    const int ntiles_x = (args.ncols_max + config.J - 1) / config.J;   // ncols_max = tokens
    if (ntiles_x < ntiles_J_best) { J_best = J; ntiles_J_best = ntiles_x; }
}
```

**The token dimension is already tiled, in units of J, and J is chosen to be the smallest
value that covers the token count.** So for a 1024-token ubatch the kernel does not run one
1664-wide block; it splits the token dimension into tiles. For the dense matmuls (ncols =
1024) the loop reaches J = 128 and produces `ntx = 8` token tiles. For the MoE matmuls
(M ~= 20 tokens per expert) it picks J = 24 and produces one token tile per expert.

So "process 64 tokens at a time" is not an unexploited idea: a tiling of the token dimension
already exists, and its size is selected automatically per launch, subject to the shared
memory budget (`mmq_get_nbytes_shared(config, cc) > smpbo` rejects larger J).

## 3. Why forcing 64 would be worse, not better

For the MoE matmuls, the tiling interacts with expert selection. With `n_expert 512`,
`n_expert_used 10`, a tile of `t` tokens touches `512 * (1 - (1 - 10/512)^t)` distinct
experts, and each selected expert's weights must be read:

| token tile | experts touched | tokens per expert (M) | weight bytes per layer |
|---:|---:|---:|---:|
| 1024 (today, J=24) | 512 | 20.0 | 996 MiB |
| 256 | 509 | 5.0 | 1.8 GiB |
| 128 | 471 | 2.5 | 3.3 GiB |
| 64 | 367 | 1.25 | 5.1 GiB |

Forcing 64-token tiles multiplies MoE weight traffic by ~5x, and the weights come from VRAM
(they are resident there - see `PHASE5_RESULTS.md`). That is a real 5x on the read side, not
a saving. It only becomes a saving if the prefetch is made expert-aware (copy only the experts
a tile selects), which is a different mechanism.

## 4. What is worth doing instead

The kernel's own tile selector is the natural place to look, and it is already tuned. What is
NOT tuned is the interaction between the tile size and **shared memory**: larger J is rejected
when `mmq_get_nbytes_shared` exceeds 48 KB on this sm_75 part. That is the one input into the
existing tiling decision that this GPU constrains hard.

Concretely, the open questions are now:

1. Which J does the MoE kernel actually select on this device, and is it rejected from a
   larger one by shared memory? If so, tiling is being forced smaller by smem, not chosen.
2. Does the expert kernel's time scale with tiles or with bytes?
3. Is the 48 KB smem limit (this sm_75 part) what caps occupancy at 1 block/SM?

All three are answerable from the occupancy/stall counters in one batch, which is the next
measurement regardless.

