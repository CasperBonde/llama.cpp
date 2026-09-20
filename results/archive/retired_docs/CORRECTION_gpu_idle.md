# Correction: the GPU is not idle. `utilization.gpu` is 99-100%, power is 18.5 W.

Date: 2026-09-20. Supersedes `THE_WAIT_FOUND.md` sections 1-3 and 6.

---

## 1. What was wrong

`THE_WAIT_FOUND.md` reported "GPU idle 88.5% of the wall time" and I presented that as
a result for the streaming configuration that reaches 64.5 t/s. **That was wrong on
both counts.**

The 88.5% came from `tmp/prof/inventory.txt`, whose bench table reads:

```
| qwen3next 80B.A3B Q4_K - Medium |  45.17 GiB |  79.67 B | CUDA | 1 | 99 | 512 | none | 1 | pp512 | 11.94 t/s |
```

**11.94 t/s.** For comparison, the same `-ngl 1 -ncmoe 99 -lm none -p 512 -b 512 -ub 512`
flags, run normally on this machine today, give **63.25 t/s**. So the run the ratio was
computed from was not representative of anything, and the ratio `4.94 s / 42.9 s` did not
describe the performance we are trying to improve. It described a broken run.

The tell was visible in my own document and I did not act on it: the table said 11.94 t/s
while the surrounding discussion talked about 64.5 t/s. I noticed the config difference
(`-ngl 1`) and wrote a caveat, but the real problem was the *magnitude* - a result 5x below
the known-good number should have been rejected outright, not caveated.

## 2. Second error: every other profile log had the same problem

I then reinforced the claim by computing kernel-sum/wall ratios from four other ncu logs.
All four are contaminated and none of them measures what I said:

| log | reported t/s | kernel sum | derived "idle" | why it is not valid |
|---|---:|---:|---:|---|
| `inventory.txt` | 11.94 | 4.94 s | 88.5% | config yields 63 t/s normally |
| `sweep.txt` | 29.51 | 2.99 s | 82.8% | same |
| `sweep_stream.txt` | 16.77 | 9.88 s | 67.6% | same |
| `layer_cycle.txt` | 102.28 | 0.095 s | 98.1% | only 12 kernel rows: a single layer, not a pass |

The spread is 88.5%, 82.8%, 67.6%, 98.1%. Four numbers that disagree with each other by
30 points cannot be a measurement of one physical quantity. They are an artefact of
dividing kernel time by a wall time that includes CUDA init, model load (`fb` grows
304 -> 3112 MiB), warmup, and ncu replay overhead.

**The `layer_cycle.txt` row is the one that should have killed the hypothesis on the spot:**
98.1% "idle" at 102.28 t/s. A run cannot be the *fastest* and the *most idle* at the same
time. I did not look at it.

## 3. The measured truth at the operating point

Your config, run four times (`-r 4`), with `nvidia-smi` sampling power and utilisation at
500 ms while `dmon` sampled SM occupancy at 1 s:

```
./build/bin/llama-bench -m models/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf \
  -ngl 1 -ncmoe 99 -lm none -n 0 -r 4 -p 1024 -b 1024 -ub 1024 -o md \
  --cache-type-k q4_0 --cache-type-v q4_0 -t 6 -ps 1
```

| measurement | value |
|---|---:|
| throughput | **84.53 ± 0.08 t/s** |
| your reference | 84.62 t/s - reproduced |
| compute per 1024-token prefill | **12.1 s** |
| `utilization.gpu` during compute | **99-100%** |
| `dmon sm` (SM occupancy) during compute | **100%** |
| `dmon mem` (DRAM bandwidth) during compute | **8-15%** |
| `power.draw` during compute | **18.4 W avg, 21.7 W max** |
| idle (pre-init / load) power for reference | 12.6 W |

Steady-state power trace, 10 s buckets, at the good config:

```
  t(s)      W    util%
    50   13.28     4.0     <- model load, not compute
    60   18.57    99.0     <- compute starts
    70   19.04   100.0
    80   18.21    99.9
    90   18.05    99.8
   100   18.16    99.9
   110   18.62    99.9
   120   17.69    89.5     <- compute ends
```

**There is no idle window during compute.** `utilization.gpu` is 99-100% for the whole
compute phase. The 88.5% figure is retracted.

## 4. What is actually true, and it is a weaker statement

The GPU is **occupied but under-exercised**: 100% occupancy at 18.4 W against a 50 W
limit, with DRAM at 8-15% of its 128 GB/s peak. That is consistent with a *stall inside a
launched grid* - warps resident and counted as occupied while they wait on memory - not
with an empty GPU.

`utilization.gpu` cannot distinguish these two. It is satisfied by a grid that is launched
and stalled, exactly as `THE_WAIT_FOUND.md` section 3 argued. But the conclusion I drew
from that - that the ratio of kernel time to wall time proves an empty GPU - was applied
to a broken run and is withdrawn.

So the standing concern ("there may be a simple wait somewhere that consumes 70-80% of the
compute time") is **still open**. It is supported by the 18.4 W and the 8-15% DRAM, and it
is *not* supported by any gap in the timeline.

## 5. Why the config matters so much - a real finding worth keeping

The benchmark arguments change throughput by 2.7x, and this is not noise:

| `-p` / `-b` / `-ub` | cache | t/s | notes |
|---|---|---:|---|
| 512 / 512 / 512 | - | 63.25 | |
| 512 / 512 / 512 | q4_0 | 63.23 | **cache type makes no difference** |
| 1024 / 1024 / 512 | - | 32.30 | **ub=512 with p=1024 is pathological** |
| 1024 / 1024 / 1024 | - | 82.34 | |
| 1024 / 1024 / 1024 | q4_0 | **84.53** | matches your 84.62 |

Two things follow:

1. **The KV cache quantisation is not the variable.** q4_0 KV vs f16 KV: 63.25 vs 63.23,
   and 82.34 vs 84.53. Within 2%. The win is the *ubatch size*, which was already
   established in `KV_VS_LOADMODE.md`. I had started dropping `--cache-type-*` from
   commands, which was wrong for comparability even though it is not the cause.
2. **`-ub` must equal 1024.** At `-ub 512` with `-p 1024` throughput collapses to 32.30.
   The streaming path has 2 slots, so with 1024-token ubatches the next layer is prefetched
   while the current one computes; at 512 the refill rate doubles and the pipeline drains.

## 6. One loose end found while checking

I initially read the instrumentation line "48 refills per 1024-token ubatch" as
contradicting the ~94 weight tensors per layer. **That was my error and it is already
resolved:** the per-ubatch count is 48 refills / 48 splits because a refill is *per layer*,
not per tensor - one call moves the whole 1022 MiB layer. An earlier log that showed
"94 refills / 96 layer splits" was a **log bug**: the counters are reset and the slots
re-seeded only on the new-graph path, so a reused-graph line printed the previous ubatch's
accumulated counts. One pass is 48 refills / 48 splits.

Worth recording from that same finding: on the reused path the slots were never re-seeded,
so layers 0 and 1 could read whatever the previous ubatch left behind. That is a real latent
correctness bug in the instrumentation, not a performance one.

## 7. What is retracted

| claim | status |
|---|---|
| "GPU idle 88.5% of wall time" | **retracted** - measured `utilization.gpu` = 99-100% |
| "the ratio cannot be fooled, prefer it over `utilization.gpu`" | **retracted** - the ratio was contaminated by load/init/replay time; both need a clean measurement window |
| "kernel sum 4.94 s in a 42.9 s run" | **invalid** - 11.94 t/s is not a runnable configuration's throughput |
| "88.5% confirms the 70-80% wait concern" | **invalid** - the concern is still open, but this did not confirm it |

What survives: `utilization.gpu` is still a poor instrument for detecting stalls in a
launched grid, and it should not be believed over a well-formed derived quantity. The error
was not in that reasoning - it was in never checking whether the run I applied it to was
reproducible.

Also retracted from this session: my claim that "48 refills vs 94 tensors per layer" was a
2x discrepancy. A refill is per layer, not per tensor. See section 6.

## 8. Method, so this cannot recur

- **Any benchmark number must be reproducible at the same config, in the same session,
  before it is used in a derivation.** 11.94 t/s next to known-good 63 t/s should have
  stopped the analysis.
- **A derived ratio must have a clean measurement window.** Kernel-sum/wall only means
  something if the wall time is compute and nothing else. `-r 4` with a power trace
  separates init, load, warmup and compute; a single `-r 1` run does not.
- **Cross-check a derived quantity against a direct instrument.** Power and `dmon` are
  direct. If a derivation says 88.5% idle and `dmon` says `sm` is 100%, the derivation is
  wrong.
- **Records**: every benchmark command in this investigation had different arguments. That
  is why four logs disagreed. Pin one configuration as the reference and change one flag at
  a time.

## Reproduce

```sh
cd /home/cbonde/dev/llama.cpp
sh tmp/prof/ss_test.sh                 # r 4, power + dmon, both configs
python3 tmp/prof/ss_analyze.py         # per-window power / util / sm
```

Raw data: `tmp/prof/ss_GOOD_p1024_ub1024_q4kv.txt`, `tmp/prof/ss_D_p512_ub512_nocache.txt`,
`tmp/prof/ss_power.txt`, `tmp/prof/ss_dmon.txt`, `tmp/prof/ss_marks.txt`.
A/B across five configs: `tmp/prof/ab_test.sh`, `tmp/prof/ab_*.txt`.
