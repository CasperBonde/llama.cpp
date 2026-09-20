# STOP AND READ THIS BEFORE RUNNING ANY BENCHMARK

## The one rule

**Never re-run a benchmark invocation to get a number you could have collected in the same run.**

The model is 45.17 GiB and is read from SSD on every invocation. One `llama-bench` run costs
**4-8 minutes of pure loading** before the first token moves. A benchmark that produces one number
wastes everything except that number. A benchmark that produces four costs the same 4-8 minutes.

Two specific failures to never repeat:

1. **Do not re-run to collect a second configuration.** Pass both configurations to one invocation:
   `-ngl 1,99 -p 1024 -n 256` collects four rows from one load. I ran this four separate times and
   got two rows each time, because half the configurations were failing - and I did not read the
   output to notice. That is hours of SSD reads for data I could have had in one pass.
2. **Do not treat a failure as "needs a retry".** When the output shows fewer rows than requested,
   the correct response is to read the error at the top of the output, not to run it again. The
   failure was in the log the very first time:
   `llama_bench: error: failed to create context`.

**Before running anything, ask: is every number I need coming out of this single invocation?**
If not, add the missing axis to the command line (`-ngl 1,99`, `-p 512,1024`, `-n 128,256`) rather
than planning a second run.

Corollary: a quick command that takes 1-2 ms is fine to try 10-20 times. This is not that. When a
command costs 5 minutes, stop and get it right on paper first, and **show the command to the user
before running it** when there is any doubt.

---

## The hardware, and what it forces

**NVIDIA Quadro T1000, 4 GB VRAM (3716 MiB usable), TU117, sm_75.**

This is the constraint that explains almost every configuration oddity:

| fact | consequence |
|---|---|
| VRAM is 3716 MiB usable | a 45.17 GiB model cannot be resident |
| 45.17 GiB model | **only** the Q4_K_M quant fits the intent; the experts must stay on CPU |
| experts are 45 GiB of the 45.17 | **`-ncmoe 99` is mandatory**, or the load tries to place 45 GiB on the GPU and dies with `cudaMalloc failed` |
| KV cache at ub 1024 | **`--cache-type-k q4_0 --cache-type-v q4_0` is mandatory**; f16 KV does not fit at large context |
| 3716 MiB total | the compute buffer at `-b 1024 -ub 1024` is ~620 MiB, which is why ub and context are tightly budgeted |
| BAR1 256 MiB | streaming weights must be copied per layer; large simultaneous mappings are not available |
| 14 SMs, 64 KiB smem/SM | tile choices and occupancy are unusually sensitive here |
| no tensor cores (TU117) | the whole subject of this branch of work |

---

## The prefill streaming feature

**This is a feature of this branch, not of upstream llama.cpp.** It is a prototype and is not rock
solid. Treat its failure modes as expected and readable, not as mysteries requiring a retry.

`-lm none` is required: the streaming path bypasses the normal model allocation.

### The slot rule - this is what made `-ngl 99` "impossible"

The streaming feature keeps a weight **slot** in VRAM and refills it per layer. The number of slots
is the number of buffers it double-buffers with:

| slots | how to select | `-ngl` that works | why |
|---|---|---|---|
| **1** | `-ps 1` | **`-ngl 99`** | one slot plus the compute buffer fits; no double buffering |
| **2** | `-ps 1` default / `-ps 0,1` | **`-ngl 1`** | two slots plus compute do not fit; weights must stream from host |

`-ps` is the parallel-sequences (slot) count. **One slot means no double buffering, and that is the
configuration that fits alongside `-ngl 99`.** Two slots need `-ngl 1`.

So, the two working configurations are:

```sh
# -ngl 99, one slot, no double buffering
./build-turingdp4a/bin/llama-bench -m models/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf \
  -ngl 99 -ncmoe 99 -lm none -p 1024 -n 256 -b 1024 -ub 1024 -o md \
  --cache-type-k q4_0 --cache-type-v q4_0 -t 6 -ps 1 -r 4

# -ngl 1, two slots, double buffered
./build-turingdp4a/bin/llama-bench -m models/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf \
  -ngl 1 -ncmoe 99 -lm none -p 1024 -n 256 -b 1024 -ub 1024 -o md \
  --cache-type-k q4_0 --cache-type-v q4_0 -t 6 -ps 1 -r 4
```

**Both in one invocation** - this is the form that should always be used:

```sh
env LLAMA_PREFILL_STREAMING_SERIAL=1 ./build-turingdp4a/bin/llama-bench \
  -m models/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf \
  -ngl 1,99 -ncmoe 99 -lm none -p 1024 -n 256 -b 1024 -ub 1024 -o md \
  --cache-type-k q4_0 --cache-type-v q4_0 -t 6 -ps 1 -r 4
```

`LLAMA_PREFILL_STREAMING_SERIAL=1` is required for the `-ngl 99` case to allocate: the non-serial
path reserves a compute buffer sized from the multi-slot graph and fails with
`cudaMalloc failed: out of memory` on ~620 MiB, **even though `nvidia-smi` shows the card is free**.
This is not a VRAM shortage. It is the streaming pipeline's own reservation. Serial mode makes the
copy blocking, which is why the two phases can be timed separately.

---

## Other traps hit while collecting this data, worth not repeating

- **`-ngl 99` without `-ncmoe 99`** tries to put 45.17 GiB on a 4 GB card and fails at load with
  `allocating 46086.39 MiB ... cudaMalloc failed`.
- **`-t` must be set** (6 here) so the CPU-side expert work is not starved by thread oversubscription.
- **Always run one llama process at a time.** Two concurrent runs exhaust RAM (62 GB total) and the
  machine stalls in SSD swap. Check with `pgrep -af "llama-bench|llama-cli"` first.
- **Direct output to a file with `nohup`.** Do not pipe a long build or benchmark through `grep`;
  it has caused the tool to hang on an already-finished command.
- **Read the whole output, not the summary line.** Half the configurations were silently absent from
  four separate runs and it went unnoticed because only the last rows were inspected.
- **Never hand-derive a value the code can print.** The tile sizes in `TURING_DP4A.md` were first
  written from a hand calculation that used the wrong smem formula. `LLAMA_MMQ_DEBUG=1` prints the
  real `nty/ntx/ntzw/I/J/smem` per launch; use it and quote it.
