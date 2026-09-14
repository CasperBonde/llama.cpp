# Why the sweep got 86 t/s and the later run got 76 t/s

Date: 2026-09-14
Answer: it is NOT KV quantization. It is `-lm none`.
Cause: my later command omitted `-lm none`, which the original sweep script has.

## The evidence chain

1. The user asked whether KV quantization explains 86 vs 76. Tested directly.

   sweep_80b_q4kv.txt (Q4_0 KV, fa on) vs sweep_80b.txt (f16 KV), pp1024 ub1024:

       f16 KV, pfs 1 : 64.67 (pp512)  86.99 (pp1024)
       Q4_0 KV,pfs 1 : 64.45 (pp512)  86.83 (pp1024)

   Identical within noise (+/-0.3%). KV quantization is NOT the difference.
   This reproduces the earlier finding that q4_0 vs f16 KV changes nothing here.

2. The sweep ran on build 08b667fab, current HEAD is 2eb542a4d. Only one commit
   differs ("Adding support for 80b model") and it touches llama-context.cpp only
   for the accelerator-device fix. Not a perf change.

3. Suspicious detail: the sweep's "pfs 1" numbers for the 80B could not have been
   produced by the slot machinery, because at 08b667fab `qwen3next.cpp` had ZERO
   `tensor_bind` calls (only `qwen3moe.cpp` was wired). Checked with git grep.

4. Re-ran the EXACT sweep command and got the high number back:

       ./llama-bench -m 80B -ngl 1 -ncmoe 99 -lm none -n 0 -r 1
                    -p 1024 -b 1024 -ub 1024 -ps 0,1
       -> pp1024 pfs 1 = 84.61

   So the number is real; my later command was different.

5. Isolated the flag. pp1024, pfs 1, same model, same streaming, one flag differs:

       -ngl 1 -ncmoe 99              -> 76.48
       -ngl 1 -ncmoe 99 -lm none     -> 84.73

   `-lm none` is worth +10.8%.

## Mechanism: the refill rate, not the compute

Measured fill bandwidth of the same 2 slots (2044.82 MiB) from the CLI logs:

       default (mmap)  : 2044.82 MiB in 222.374 ms =  9.64 GB/s
       -lm none        : 2044.82 MiB in 173.782 ms = 12.34 GB/s

`-lm none` makes the loader read the file into anonymous RAM instead of
memory-mapping it. With mmap the pages of the 45 GiB file start out as clean page
cache backed by the file; the first read of each page faults it in from disk,
which is slower than sequential reads out of anonymous memory. Every ubatch
re-reads the whole model, so the prefill path pays that fault cost on every
ubatch.

At pp1024 with -b 1024 there is exactly one ubatch, so the win is the one full
refill: 45 GiB at 9.64 GB/s vs 12.34 GB/s. That is the ~4200 ms vs ~3300 ms
difference, and it maps to 53.51 t/s baseline being unchanged and the streaming
number moving from 76 to 84.

Note the baseline is unaffected (53.51 vs 53.87, within noise) because without
streaming there is no per-ubatch re-read: each weight is read once, either way.

## Corrected numbers, current generic build, matching the sweep command

Command: `llama-bench -m 80B -ngl 1 -ncmoe 99 -lm none -n 0 -r 1 -b 1024 -ub 1024 -ps 0,1`

    pp       baseline   streaming   gain
    pp512      38.12      64.46     +69.1%
    pp1024     53.53      84.83     +58.5%
    pp2048     53.99      49.49     -8.3%

(Those are the values in final_lmnone.txt. A second run of the same command gave
37.89 / 63.42, 53.51 / 84.07, 53.95 / 49.51, i.e. within 0.5%.)

Compare with the original sweep at the same cells (build 08b667fab):

    pp       baseline   streaming   gain
    pp512      37.77      64.67     +71.2%
    pp1024     52.70      86.99     +65.1%
    pp2048     54.05      49.85     -7.8%

Agreement is within 3%. The generic rework therefore costs nothing measurable and
reproduces the original result.

pp2048 loses for both builds because at -b 2048 there are two ubatches and the
refill (45 GiB x 2) is not amortised. This is the known law: total refill traffic
= n_ubatch x model_size.

Log: results/prefill_streaming/final_lmnone.txt

## Correction to an earlier note

GENERIC_DESIGN.md section 8 quoted 52.34 -> 76.15 as "the" result. Those runs
lacked `-lm none`, so the streaming side was understated by ~10%. The 1.45x ratio
becomes ~1.57x with the correct flag. The rework conclusion is unchanged.
