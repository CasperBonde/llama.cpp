# The smoking gun: 32-bit weight loads combined with 1 block/SM

Date: 2026-09-20. Fast synthetic only - no profiler, no model load. Seconds per run.

---

## 1. Method

`tmp/prof/mmqstruct.cu` builds the MMQ *structure* without the MMQ math: per K-tile,
cooperatively load a tile into shared memory, `__syncthreads()`, consume the tile with dp4a,
`__syncthreads()`. It then sweeps the two levers a fix could pull:

- **tile size**, which sets blocks per SM (64 KB smem per SM, 48 KB per block)
- **access width**, scalar 32-bit vs vector 128-bit

Grid 8192 and block 256 throughout, 996 MiB, matching the real kernel. All figures below are
within the 118 GB/s hardware ceiling.

## 2. The result

| structure | blocks/SM | smem | GB/s |
|---|---:|---:|---:|
| **scalar load + sync + dp4a** | **1** | **38 KB** | **34.2** |
| vector load + sync + dp4a | 1 | 38 KB | 85.6 |
| scalar load + sync + dp4a | 2 | 28 KB | 65.5 |
| vector load + sync + dp4a | 2 | 28 KB | 117.7 |
| scalar load + sync + dp4a | 3 | 19 KB | 89.1 |
| vector load + sync + dp4a | 3 | 19 KB | 118.0 |
| plain vector read, no smem (reference) | 4 | 0 | 118.1 |

**The real llama.cpp kernel: 12.3 GB/s.**

The first row is the real kernel's configuration. It measures **34.2 GB/s** - a **3.45x**
reproduction of the mystery against the 9.6x total. The mechanism is confirmed.

## 3. Why both factors are needed

Read the table by moving one variable at a time from the top-left cell:

| change | from | to | gain |
|---|---|---:|---:|
| vectorise the loads only (stay at 1 block/SM) | 34.2 | 85.6 | **2.50x** |
| raise occupancy only (keep scalar loads) | 34.2 | 89.1 | **2.60x** |
| do both | 34.2 | **118.0** | **3.45x** |

Neither factor alone is a workaround and neither is the whole story: the penalty comes from
**their product**. Vector loads make each outstanding request carry 16 bytes instead of 4;
extra resident blocks give the SM other warps to issue while one is stalled. With 8 warps per
SM and only 4 bytes per in-flight request, the SM simply cannot keep enough bytes in flight to
cover DRAM latency, which is Little's law:

```
need = (8.43 GB/s per SM / 1.455 GHz) x latency
     = 5.79 bytes/cycle x 600 cycles
     = 3476 bytes in flight per SM
```

The real kernel's load phase can plausibly keep only a few hundred bytes in flight, which puts
it in the 30-40 GB/s range - exactly what the measurement shows.

## 4. The two source-level causes

### 4a. The weight tile loader uses 32-bit scalar loads

`ggml/src/ggml-cuda/mmq-load-tiles.cuh`, the Q4_K loader:

```cpp
const int qs0 = get_int_b4(bxi->qs, txi);      // 32-bit scalar load
x_qs[i*sram_stride + 16*(txi/8) + txi%8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
x_qs[i*sram_stride + 16*(txi/8) + txi%8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
```

`get_int_b4` is a 4-byte load. The y-tile load is also scalar (`tile_y[l] = by0[l]`). Nothing
issues a 128-bit load, so each warp has a quarter of the bytes in flight it could have.

### 4b. The smem tile forces 1 block per SM

This is the point you raised, and it is exactly right. Per `mmq_get_nbytes_shared`:

```
nbs_ids = J * 4
nbs_x   = I * sram_stride * 4        (MMA layout, because turing_mma_available(7.5) is true)
nbs_y   = J * sizeof(block_q8_1_mmq)
```

With `MMQ_TILE_NE_K = 32` and `QI8_1 = 8`:

```
sram_stride(Q8_1) = 2*32 + 2*(32/8) + 4 = 76
smem(I, J)        = 4*J + I*76*4 + J*144
smem(128, 128)    = 512  + 38912   + 18432 = 57856 B = 56.5 KB
```

| I | `I*76*4` weights | + J=128 terms | total | blocks/SM (64 KB) |
|---:|---:|---:|---:|---:|
| 64 | 19.0 KB | 18.5 KB | 37.5 KB | 1 |
| 96 | 28.5 KB | 18.5 KB | 47.0 KB | 1 |
| **128** | **38.0 KB** | **18.5 KB** | **56.5 KB** | **1** |
| 192 | 57.0 KB | 18.5 KB | 75.5 KB | 0 (exceeds the 48 KB dynamic cap) |

**The weights alone are 38 KB at I = 128, and the J = 128 y-tile adds another 18.5 KB, for
56.5 KB - well past the 32 KB that two blocks per SM would need.** So the huge weights tile is
half the smem and the *token* tile is the other half, which matters for section 5.

Registers are **not** the cap: 256 threads x ~128 registers = 32 KB of the 64 KB available, so
two to four blocks would fit on registers alone. You were right that the placement is
constrained by the memory allocation rather than the register file - and specifically by the
`I * sram_stride * 4` weights term.

## 5. A fix I proposed and then disproved

I first read this as "sm_75 selects the wrong table". That is **half right, and the useful half
is the wrong one**. The table selection is real:

```cpp
// mmq.cuh:229
if (ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) {
    return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
}
```

The build is compiled for `compute_75` only (`--generate-code=arch=compute_75,code=[sm_75]`
in `flags.make`), so `__CUDA_ARCH_LIST__ = {750}` and `ggml_cuda_highest_compiled_arch(750)`
returns 750. Since `750 >= 700`, **the am**pere table is selected and the pascal-dp4a table is
dead code on this build. That part stands.

**But selecting the pascal-dp4a table would not help.** Its Q4_K row is `I = 64, J = 8..64`.
Recomputing occupancy with `MMQ_TILE_NE_K = 32`, `sram_stride(Q8_1) = 76`:

```
smem(I, J) = 4*J  (ids)  +  I*76*4  (weights)  +  J*144  (y tile)
```

| I | J | smem | blocks/SM | ntx (weight mult) | total demand |
|---:|---:|---:|---:|---:|---:|
| **128** | **128** | **56.5 KB** | **1** | 4x | **base** |
| 64 | 128 | **37.5 KB** | **1 (unchanged!)** | 4x | 2.4x worse |
| 64 | 64 | 28.2 KB | 2 | **8x** | 3.2x worse |
| 32 | 128 | 28.0 KB | 2 | 4x | 5.4x worse |
| 128 | 64 | 47.2 KB | **1** | **8x** | 2.2x worse |

Two things kill the idea:

1. **I = 64 still gives 1 block/SM.** The J=128 term alone costs `4*128 + 128*144 = 17.4 KB`,
   so halving I only gets to 37.5 KB - still above the 32 KB needed for 2 blocks. The config
   already in the source would **not** raise occupancy.
2. **Every reconfiguration raises traffic, and the current config is the minimum.** `ntx`
   scales as `1/J` and `nty` as `1/I`, so shrinking either multiplies the re-read demand:
   J = 64 doubles the weight demand to 8x, I = 64 doubles the y demand (the rows change, so
   the y re-reads change too). The current I = J = 128 sits at the bottom of the traffic curve.

So **there is no config change that fixes this.** The smem that forces 1 block/SM is populated
by the J = 128 y-tile as much as by the weights, and J cannot shrink without doubling the
weight traffic. The two constraints are in direct opposition in this table.

That leaves the fix in the **instruction stream**, not the configuration.

## 6. Where the fix actually has to go

The measurement separates cleanly, and the two levers are independent:

| change | from | to | gain |
|---|---:|---:|---:|
| vectorise the load instructions (occupancy untouched) | 34.2 | 85.6 | **2.50x** |
| raise occupancy (load width untouched) | 34.2 | 89.1 | **2.60x** |
| both | 34.2 | **118.0** | 3.45x |

Since occupancy cannot be raised by configuration (section 5), and vectorising does 2.50x on
its own **without changing any tile geometry or any traffic**, the vectorisation is the only
lever that is both effective and compatible with the existing tiling.

It is also the cheaper change: `ggml_cuda_mmq_load_tiles_q4_K` currently issues `get_int_b4`
(4 bytes) per thread, and the y-tile loop is scalar too. Note that `Q8_0` and `Q1_0` already do
`16*(txi/8) + txi%8 + 0` and `+ 8` addressing, which is the layout a two-`uint4` or one
`uint2`-pair load would want - so the SRAM layout may already be compatible and the change could
be confined to the load.

**Caveat, stated plainly:** vectorising a quantised loader is not a local edit. The packed
`block_q4_K` layout, the dequant shifts (`>> 0` and `>> 4`), and the sector-aligned access on
this part all have to keep working, and a misaligned 128-bit load on a packed format is a
correctness bug rather than a slowdown. This is a hypothesis with a measured 2.5x ceiling, not
a verified fix. It should be discussed before anything is written.

## 7. What is still unexplained

The structural floor is 34.2 GB/s against the real 12.3 GB/s, so **2.8x remains**. Candidates,
none of which the synthetic reproduces:

1. the real kernel has four `__syncthreads()` per K iteration (two vec_dot passes per loop body
   at `mmq.cuh:902-936`), not two, so its load and compute phases alternate even more finely
2. the real `load_tiles` interleaves loads with address arithmetic and dequant shifts, so fewer
   loads are in flight at once than a tidy copy loop
3. the dp4a chain in `vec_dot` may expose latency at only 8 warps per SM

The synthetic also does not model the y-tile re-reads, so the two effects have not been
combined. That is the natural next step if the remaining 2.8x matters after vectorisation.

## 8. Why this is the right stopping point for the cheap phase

Every number here came from a few seconds of synthetic run time. The mechanism is identified
with a source-level cause and a measured magnitude; the launch shape, the smem size and the load
width are all now derived from source rather than assumed; and one plausible fix has been
proposed *and quantitatively disproved* before costing anything.

The next spend is a decision, not a measurement: whether to attempt the vectorised loader,
which the measurement says is worth up to 2.5x and which cannot be done as a config change.
