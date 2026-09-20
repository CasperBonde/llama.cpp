# Tasks to finish before any upstream contribution

Prototype work: none of this blocks local measurement. Recorded here so it is not lost and not
done now.

## 1. Test on the GTX 1650 - likely unnecessary, but not zero value

The GTX 1650 is **TU117 as well**, the same die as the Quadro T1000, so it is not independent
evidence about the die, and the user's own view is that there is little point. Agreed for the
instruction-level question: IMMA is emulated on the die, and the die is the same. If the 1650 also
reports cc 7.5 and also lacks tensor cores, the result should be identical and the test proves
nothing new.

The one thing it *would* test is the **string match**, which is the fragile part of the fix:

- `ggml_cuda_init()` matches the device name as a string. The T1000 reports `Quadro T1000` with
  **no `NVIDIA ` prefix** (my first attempt matched `NVIDIA Quadro T1000` and silently never
  fired). A 1650 reports whatever its own driver string is, which may or may not have the prefix.
- the 1650 is a distinct PCI device id (`0x1F82`) from the T1000 (`0x1FB9`).
- the 1650's SM count is also 14, so the tile choice should not change, but its clock is higher
  (1665 vs 1455 MHz), which shifts the memory/compute balance.

Expected outcome: the same result. Worth running once **only if** the contribution is going to
keep name-based detection; if the detection is replaced by a device-id table or a capability probe
(task 4), the 1650 adds nothing and should be skipped.

## 2. Separate the three variables in the tile table change

The measured 1.77x comes from three changes applied at once. The pascal-dp4a table differs from
the ampere table in:

| variable | ampere | pascal-dp4a |
|---|---|---|
| `I` (src0 tile) | 128 | 64 |
| `J` (src1 tile) | 8..128 | 8..64 |
| `stream_k` | true | false |

and separately, the vec-dot changes from the MMA path to dp4a. Four differences total.

To attribute the gain, build four variants and measure each. This is the single most important
before-contribute task, because the *reason* given to a reviewer must match the measurement.

## 3. Re-check `-ngl 99` - DONE

Measured. On the baseline, `-ngl 99` was 8.7% slower than `-ngl 1` on prefill (74.95 vs 81.45
t/s). On the dp4a build the gap is **2.4%** (143.22 vs 146.79), and decode is **1.95x faster** at
`-ngl 99` (18.07 vs 9.28 t/s). So the IMMA explanation is supported and `-ngl 99` is now the
preferred configuration. Both numbers came from a single invocation; see `TURING_DP4A.md`.

Note for reproducing: `-ngl 99` needs **one slot** and `LLAMA_PREFILL_STREAMING_SERIAL=1`, and it
will not allocate otherwise. Two slots (double buffering) is the `-ngl 1` configuration.

## 4. Decide whether the detection belongs in ggml at all

The name-based list already exists in `ggml-cuda.cu` and is acknowledged by the maintainers, so
extending it is in keeping with the existing design. But the deeper issue is that the detection
is a **string match on a marketing name**, which cannot be complete. Options to raise in an issue:
- a device-id table instead of names (stable, but needs a PCI id list)
- a runtime probe (run one small IMMA kernel on a test buffer and check it does not trap)
- drop detection and instead choose the tile table by SM count and shared memory, which is what
  actually constrains the choice: the ampere table needs 56.5 KB, the dp4a table 24 KB, and the
  device reports 48 KB usable

None of these should be implemented before the maintainers agree on an approach.

## 5. Check the flash attention interaction

`TURING_MMA_AVAILABLE` is needed by flash attention, and FA dispatches to its MMA kernels by cc.
On a device with no tensor cores FA will still select those kernels. If IMMA is emulated there (it
appears to be), FA is also running at a penalty, on the same wrong assumption, but it cannot be
fixed by the same macro. Not investigated.

## 6. Numeric comparison over a longer generation

The correctness check is 32 greedy tokens on one prompt. Before contributing it should be a longer
generation on several prompts, comparing logits or a longer transcript, to be confident the tile
layout change is numerically harmless. `tests/test-backend-ops` would be the natural vehicle, but
it must not be added to `tests/` without maintainer approval.
