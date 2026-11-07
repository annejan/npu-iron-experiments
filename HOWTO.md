# How to add an experiment here (for humans and agents)

Read `README.md` first: hardware, measured ceilings, and every trap we hit. This file is the
recipe.

## Running

```
cd ~/Projects/npu/iron                     # always from here
./env.sh python fun/<script>.py ...        # relative path! (env.sh quirk, see README)
./env.sh python -c "..."                   # ad-hoc
```
`realpath: ... No such file` printed by env.sh is harmless. Never `python -I`. The NPU is one
device: run one NPU job at a time (a second process waits or breaks timings). No pygame windows
in unattended runs: write PNG (cv2.imwrite) or WAV (soundfile) outputs instead.

## Templates (copy the closest one)

| pattern | file | shape |
|---|---|---|
| per-row frame output, params in | `probe/plasma.py`, `fun/sdf.py` | 32 cores x 34 rows of a 1920x1088 RGBA frame; memtile join of 4 rows; one `In` of parameters broadcast per column |
| on-chip stencil world, N steps per dispatch | `fun/slime.py`, `fun/water.py`, `fun/life_chip.py` | world in L1 (~48 KB per core usable), halo rows through shared memory in a snake of 32 cores, ping-pong buffers |
| DDR world streamed through the cores every step | `fun/lbm.py` | 3-D shim DMA access pattern gathers each core's band plus halo rows; `acquire(3)` sliding window |
| batch of independent jobs per core | `fun/reverb.py` (FIR), `fun/fft.py` (design only, kernel compiled out) | memtile split in / join out; `sliding_mac<32,8>` FIR |
| per-element over big tensors | `fun/mandel.py` | memtile split/join of TILE-sized objects, 32 cores |

## Checklist before you call it done

1. **Kernel keyed by variant.** Any `@iron.jit` design whose C++ comes from a Python string must
   take a `variant: CompileTime[str]` (or other CompileTime arg) that changes whenever the source
   changes, else the cached binary runs. Derive it from the parameters (`"lenia-" + str(PARAMS)`).
2. **Rounding.** `aie::set_rounding(aie::rounding_mode::conv_even)` at the top of every kernel
   that converts accumulators (`to_vector<bf16>()`, `to_vector<int16>(shift)`).
3. **Static check first.** `remarks.analyze(ext_fn, "aie2p", Path(dir))` (see `probe/fir_check.py`):
   no `libcalls` in hot loops, note the inner-loop `ii`.
4. **Verify against numpy** on real data (not zeros): print max/rms error in LSB; bit-exact for
   integer kernels. Poison outputs (`iron.zeros` is not poison; fill with a sentinel if the kernel
   might skip writes).
5. **Time after a warm-up call.** The first call loads or compiles (0.4-1.7 s). Report min and
   median over >= 10 calls, and say what is DMA, fixed cost (~160 us) and compute.
6. **Budgets.** Per core 64 KB L1 (fifo depths x object sizes + static tables + stack); per column
   one memtile split plus one join (two splits + join does not place); shim 2 DMA channels each
   way per column. Pack all per-cell fields into one stream.
7. **Alignment.** `aie::load_v`/`store_v` need 64-byte alignment and silently round a misaligned
   address down. Use `load_unaligned_v`/`store_unaligned_v` for any offset that is not a multiple
   of 32 elements (16-bit) / 16 (32-bit). Static kernel variables persist across dispatches: reset.
8. **Not native, avoid in hot loops:** float vector multiply, scalar float mul/div, int divide,
   64-bit multiply, `aie::neg` on bf16 (multiply by -1), `aie::shift_bytes` (use
   `shuffle_down_fill`), cint16/complex vectors, per-lane gather.
9. **bf16 integrates badly** (8 mantissa bits), and so does int16 for second-order updates: the
   ripple tank needed int16 Q12, the 3-D wave equation needed int32 Q28 (int16 Q14 lost its energy
   in 100 ms; a numpy emulation with exact rounding proved it was the format). Counters, height
   fields, wave equations: int32 fixed point. Store deviations from a rest value when you must use
   bf16 (LBM).
10. **Deliver:** the script in `fun/` with a docstring (what, how, how to run), a row in the
   README measured table, a short README section with the lessons, output files (`*.png`,
   `*.wav`, gitignored), and a commit (`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`
   trailer, no session links).
