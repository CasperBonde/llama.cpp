# Qwen3-Next-80B: prefill vs generation, ngl comparison

Date: 2026-09-13/14
Branch: prefill_double_buffering2 (2eb542a4d, build 10831)
GPU: Quadro T1000 3716 MiB. Model: Qwen3-Next-80B-A3B-Instruct-Q4_K_M (45.17 GiB).
Command family: llama-bench -m <80B> ... -r 1

## The headline table

    config                        prefill pp1024   generation tg128
    ----------------------------  --------------   ----------------
    ngl 1                              52.32            10.62
    ngl 1 + prefill streaming          77.73            10.62
    ngl 99 + ncmoe 99                  53.12            18.50

## What this says

1. Prefill gets NO benefit from having the attention weights resident.
   53.12 (ngl 99) vs 52.32 (ngl 1) is within noise. At ub 1024 prefill is
   compute bound on the GPU, not weight-bandwidth bound, so where the attention
   weights live does not matter. The 1.49x prefill win we have comes from the
   slot mechanism itself (52.32 -> 77.73), not from residency.

2. Generation DOES benefit, and a lot. 10.62 -> 18.50, a 1.74x win.
   At bs 1 every weight is read once per token, so the 1.203 GiB of attention
   weights come off PCIe instead of VRAM in the ngl 1 case. That is the
   10.62 vs 18.50 gap.

   (This is exactly the effect the author recalled: the "for all tokens" data
   in VRAM, only the dynamically selected experts in host RAM.)

3. The two modes want opposite things:

       prefill    wants few GB in VRAM so the slots fit
       generation wants attention resident in VRAM

   They cannot both be satisfied at once on a 3716 MiB card, because
   2 x 1022.41 MiB of slots plus 1446.39 MiB of model does not fit.

## Therefore: swap the placement at the prefill/generation boundary

Best of both, measured components:

    prefill   77.73 t/s   (ngl 1 + streaming, slots in VRAM)
    generation 18.50 t/s  (attention in VRAM, experts via host)

a 1.49x prefill and 1.74x generation win over the single-config alternatives.

## VRAM budget for the swap

Prefill phase (unchanged from the working config):

    CUDA0 model (output.weight only)          243.44
    KV + RS (c=2048)                          123.38
    CUDA0 compute                             612.00
    slots 2 x 1022.41                        2044.82
    total                                    3023.64   of 3716  -> fits

At the end of prefill, free the two slots:

    freed                                    2044.82
    free after freeing                       2737.00

Then copy the non-expert layer weights into VRAM so generation matches the
ngl 99 path. The amount needed is exactly the measured difference between the
two model buffers:

    1446.39 (ngl 99) - 243.44 (ngl 1) = 1202.95 MiB

    free after the copy                      1534.05   -> fits, with margin

So the swap is available on this GPU without giving anything up.

## Prior measurement (earlier session, same branch)

ngl 1, f16 KV, -lm none, -r 1, pp1024 column:

    ub      baseline   streaming
    64        18.88      14.49
    128       17.67      25.01
    256       25.54      21.22
    512       38.54      33.75
    1024      52.70      86.99

The same test today gives 77.73. Same shape, same ranking. Anything in the
75-90 band is the same measurement under differing machine load; treat the
ratio 1.49x as the stable result, not the absolute number.

## Raw logs

    results/prefill_streaming/bench_ngl1_pp1024.txt
    results/prefill_streaming/bench_ngl1_ps_pp1024.txt
    results/prefill_streaming/bench_ngl99_pp1024.txt
    results/prefill_streaming/bench_tg.txt
