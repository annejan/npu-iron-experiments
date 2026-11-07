# IRON on the Krackan NPU

mlir-aie 1.4.5 (IRON) with XRT 2.26 unpacked in `../ryzenai`. `./setup.sh` once, then
`./env.sh CMD...` for everything.

```
./env.sh python fun/mandel.py          # scripts: relative path, from this directory
./env.sh python probe/plasma.py --window
```

## Hardware (xrt-smi: "NPU Krackan 1", aie2p, 6x8)

| | |
|---|---|
| Compute cores | 32 (8 columns x rows 2..5), 1.76 GHz |
| Per core | 64 KB data memory (4 banks of 16 KB), 512-bit vector unit, 2 DMA in + 2 out |
| Mem tiles | 8 (row 1), 512 KB each |
| Shim tiles | 8 (row 0), 2 DMA channels each way, host DDR access |
| Vector lanes | 64 x int8, 32 x int16, 16 x int32, 32 x bf16 |
| Native math | int add/mul/mac/shift/cmp/select/shuffle, bf16 mul, `aie::inv`, `aie::invsqrt`, `aie::sqrt`, `aie::exp2`, `aie::tanh` (bf16) |
| Not native | float vector multiply (emulated via bf16, ~50 bundles/vector), scalar float mul/div, int32 divide, 64-bit multiply, sin/cos (use tables / polynomials), per-lane gather |

## Measured ceilings (2026-11-07, `probe/`)

| What | Number |
|---|---|
| Dispatch floor, host wall clock | ~155-180 us per call |
| Host <-> NPU DMA, all 8 columns, big transfers | ~26 GB/s each way (`probe/bw.py`) |
| 1920x1088 RGBA frame out | ~0.31 ms |
| One core, int16 vector mul loop at II 1 | ~52 G elements/s (ideal 56) (`probe/burn.py`) |
| All 32 cores, same loop, wall clock incl. floor | ~850 G int16 ops/s |
| 1080p plasma, 32 cores, packed RGBA, bit-exact | 0.34 ms/frame (`probe/plasma.py`) |
| On-chip Game of Life, 32 cores | 27.8 Gcell/s vs numpy 1.5 (`fun/life_chip.py`) |
| Rotozoomer / tunnel 1080p, scalar texture gathers, 32 cores | ~1.1 ms/frame (`fun/rotozoom.py`) |
| SDF sphere tracer 1080p, bf16 MAC + fp32 accumulators, 32 cores | 48 steps 7.1 ms (141 fps), 64 steps 8.3 ms (120 fps), 96 steps 11.4 ms (88 fps) (`fun/sdf.py --steps N`) |
| Blobs 1080p, 4 spheres + floor smooth-min, 48 steps, 5 distance evals per step | 38 ms/frame, 26 fps (`fun/blob.py`) |
| Slime chemotaxis 320x256, bf16, world on chip, 15 steps/dispatch | 41k steps/s, 3.4 Gcell-steps/s (`fun/slime.py`) |
| Ripple tank 320x384, int16 Q12, world on chip, 5 steps/dispatch | 17k steps/s, 2.1 Gcell-steps/s (`fun/water.py`) |
| Convolution reverb, 65536 taps, 2048-sample blocks, stereo | 0.43 ms per block per channel, 308 GMAC/s, 41x real time, bit-exact vs FFT (`fun/reverb.py`) |
| LBM D2Q9 384x256, bf16 deviations, 1 step per dispatch (DMA-bound) | 2700 steps/s, 0.27 Gcell-steps/s (`fun/lbm.py`) |
| Mandelbrot 1080p, 400 iterations, Q24 int32, 16 lanes, 32 cores | 26.9 ms (numpy 5.4 s; old scalar int64 kernel on 8 cores 34x slower) (`fun/mandel.py --bench`) |

Rule of thumb for effects: a dispatch costs ~160 us before any work, a full-HD frame
costs ~0.3 ms of DMA, and compute is almost free unless the kernel is scalar.
Batch work per dispatch; keep one call per frame.

## Demos (`fun/`)

| | |
|---|---|
| `mandel.py` | zoom in the terminal; `--bench`, `--png out.png`, `--scalar` for the old kernel |
| `life.py`, `life_chip.py`, `life_window.py` | Game of Life, host-halo and on-chip (snake of 32 cores) variants, pygame window |
| `rotozoom.py` | rotozoomer + tunnel, `--window` (TAB switches) |
| `sdf.py` | sphere-traced signed distance field (repeated spheres + floor), `--window`, `--check` for the static loop report |
| `blob.py` | four spheres + floor joined with smooth-min (the "bloep"), numeric normals, specular; reuses sdf.py's plumbing |
| `slime.py` | slime mould / ant trails as grid chemotaxis (density + pheromone, conservative upwind flux), on-chip world, `--window` (click = food) |
| `water.py` | ripple tank, int16 Q12 height field, on-chip world, `--window` (drag = drops, R = rain) |
| `reverb.py` | convolution reverb: 65536-tap IR (1.4 s), 2048 taps per core, int16 Q15, `sliding_mac` FIR; `in.wav` offline with numpy check, `--play` live through the speakers (sounddevice) |
| `lbm.py` | Lattice-Boltzmann D2Q9 fluid, flow past a cylinder; world in DDR, streamed through all 32 cores every step, shim DMA gathers the halo rows with a 3-D pattern, cores use a 3-row sliding window (`acquire(3)`); `--window` (drag = obstacles, C = clear) |
| `../probe/plasma.py` | 3-sine plasma, `--window` |

Shim DMA budget: each column's shim has 2 channels in and 2 out, so a design that gives every
core its own host stream cannot go past 1 worker per column (`transform_parallel_binary` with
`num_channels=4` fails placement). Pack inputs into one stream and fan out through the memtile
(`split` on the way in, `join` on the way out); the memtile has 6 DMA channels each way, so two
splits plus a join is already too many.

## What fits this chip, demo-wise

Good: anything branchless and MAC-shaped where every lane does the same work and data stays
on chip. Stencils (Life, reaction-diffusion, fluids), convolution filters, vertex transforms,
FFT (`aie::fft_dit`), tiny per-pixel neural nets, and SDF ray marching: the whole march is
bf16 multiplies into fp32 accumulators (native), `invsqrt`/`inv`/`min`/`max`/`select` in bf16
(native), fp32 adds for the ray position (native). Keep fp32 values in `aie::accum<accfloat>`
and convert to bf16 only for the multiplies; round-to-grid with the magic-number trick in
fp32 adds (`p + 1.5*2^25 - 1.5*2^25` rounds to a multiple of 4), never through bf16.

Poor: per-pixel gathers (texture lookups, palettes: scalar unit, one per cycle), anything needing
a big table or framebuffer in L1 (64 KB), fp32 multiplies, divides, trig, branchy per-pixel code.
Trivial effects (plasma) are DMA-bound: writing the frame costs more than computing it.

A ray-march step for 32 pixels is ~80 bundles with two rays interleaved (one ray alone: ~157;
the chain is latency-bound, `ns` 1). AIE2P has only five accumulator registers, so more rays
in flight spill to the stack.

Precision lessons from `sdf.py`, each one cost a wrong picture:
* a bf16 ray direction is too coarse (8 mantissa bits: neighbouring rows share a direction and
  silhouettes terrace). Split it into bf16 hi + lo parts and march with two `mac`s;
* no hit test = grazing rays that ran out of steps get shaded as the surface. Test the final
  distance against an eps that grows with t, else sky;
* clamp the step. Unbounded steps push p past 2^24, the magic-number rounding breaks and the
  lane turns NaN, which `to_fixed` makes black;
* numeric normals from bf16 distances come out as moire rings. Subtract the fp32 accumulators
  (eps 0.08), and keep the squared gradient away from 0: a sky ray's flat gradient gives
  `invsqrt(0) * 0` = NaN, which `to_fixed` made white.

On-chip stencil sims (`slime.py`, `water.py`, after `life_chip.py`): the world lives in the 32
cores' L1 (about 48 KB each after edge buffers: 1.5 MB in all), every step hands one row up and
one down through shared memory, and one dispatch runs many steps. Lessons:
* `accum::to_vector<bf16>()` and `to_vector<int16>(shift)` round with the core's rounding-mode
  register, which defaults to **truncation**. A conservative scheme then loses mass every step
  (0.3% per step in the slime sim). Call `aie::set_rounding(aie::rounding_mode::conv_even)`
  at the top of the kernel;
* bf16 cannot integrate `h += v` when v is far smaller than h (8 mantissa bits): ripples
  freeze. Height fields want int16 fixed point, the Amiga way;
* a ping-pong world buffer only keeps what the kernel writes: a constant field (food) that
  rides along in the band must be copied by the kernel each step;
* the memtile takes one split plus one join per column; a second input stream per core does
  not fit, so pack all per-cell fields into one band.

LBM lessons (`lbm.py`): store distributions as deviations from rest (`f_i - w_i`) or bf16 has no
precision left; a body force small enough to be stable vanishes in bf16, so drive the flow with an
inlet boundary the host rewrites between steps; start from rest and ramp the inlet (an impulsive
start shocks rho by 20% and bf16 noise then blows up); `aie::neg` on a bf16 vector is not
legalised by Peano (multiply by -1 instead). Nine fields per cell do not fit L1 with a useful world,
so the world lives in DDR: one step per dispatch, ~0.26 ms, all of it DMA and fixed cost.

## Beyond demos: audio

`fun/reverb.py` is the first non-demo use and the best fit so far: a long FIR is nothing but
multiply-accumulates on int16, no gathers, no branches. 65536 taps at 48 kHz is 3.1 GMAC/s per
channel; the chip does 308 GMAC/s in the dispatch (one `sliding_mac<32, 8>` per 8 taps for 32
outputs, 87 bundles per 32 taps per core; the naive unaligned-load version reaches 207), so a
stereo 1.4 s reverb costs 2% of real time in 43 ms blocks. The same shape covers cabinet
simulation, HRTF/3D audio (dozens of FIRs), filter banks and channelizers; `aie::fft_dit` is there
for spectral effects, and int8 matmul for neural audio (denoise, amp models). Dispatch cost
(~160 us) is 8 samples at 48 kHz, so 256-512 sample blocks are fine for live use.

Benchmark trap: the first call of a design loads (or compiles) it and costs 0.4-1.7 s. Time
after a warm-up call, or the throughput number is garbage (the first version of this bench
reported 16 GMAC/s for that reason).

## FFT: not in aie_api on this toolchain

`aie::fft_dit_r4_stage` and friends compile only for AIE1; AIE2P has the class form, but
`detail/aie2p/fft_dit.hpp` sits behind `__AIE_API_COMPLEX_VECTOR_SUPPORT__`, which
`detail/config.hpp` sets to 0 whenever `__AIECC__` (Peano) is defined. No `cint16` vectors, no
FFT (`fun/fft.py` is the attempt; it fails at compile time). Options: a hand-written radix-2/4
on separate int16 re/im vectors, or a DFT as an int16 matmul (W is 2048x2048 for 1024 complex
points; batches of frames make it a wide GEMM, which this chip likes, but the acc32 sum of 2048
Q15 products needs the output shift the stock kernel does not expose).

## Dense matmul: how much of the 50 TOPS the stock example gets

`mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/whole_array.py`, 8 columns,
int8 in and out, NPU time from the example's own benchmark (2026-11-07):

| shape | tile m,k,n | TOPS | of 50 |
|---|---|---|---|
| 1024^3 | 64,64,32 (default) | 3.5 | 7% |
| 2048^3 | 64,64,32 | 3.9 | 8% |
| 2048^3 | 64,64,64 | 8.8 | 18% |
| 2048^3 | 128,64,64 | 15.8 | 32% |
| 2048^3 | 128,32,128 | 16.9 | 34% |
| 4096^3 | 128,64,64 | 14.6 | 29% |
| 2048^3 int16 | 128,32,64 | 6.1 | 12% (int16 is 128 MAC/cycle, a quarter of int8) |

The default tiles are DMA-bound; bigger output tiles (more MACs per byte moved) give 4x. Larger
still does not fit: A, B and C tiles double-buffered must stay under 64 KB per core
(`'aie.buffer' op could not be placed`). The ghost-intuit report reached 60% on a conv net with
hand-written kernels and all 32 tiles per layer, so ~34% is the stock-example ceiling, not the
chip's.

```
./env.sh python mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/whole_array.py \
    -M 2048 -K 2048 -N 2048 -m 128 -k 32 -n 128 --n-aie-cols 8 --dtype_in i8 --dtype_out i8
```

## From ghost-intuit/xdna2-npu-experiments (Strix Point, Windows, same silicon family)

Read 2026-11-07, https://github.com/ghost-intuit/xdna2-npu-experiments (CC BY 4.0):
* **NPU power mode.** `xrt-smi configure --pmode performance` raised the core clock from 1.40 GHz
  (default) to 1.81 GHz on their part. On this Linux box the ioctl needs root
  (`DRM_IOCTL_AMDXDNA_SET_STATE ... Permission denied`); try
  `sudo ../ryzenai/xrt/bin/xrt-smi configure --pmode performance` before compute-bound demos,
  and `--pmode default` after. Turbo did not help them; disabling CPU idle states made everything slower.
  Measured here (2026-11-07): **no clear effect on this Linux/Krackan box.** A proper A/B with
  `probe/sustain.py` (same session, back to back): long 5 ms `burn` calls 5.35-5.64 ms in default
  vs 5.14 ms in performance, short 0.5 ms calls 512-659 vs 511 us: 4-9%, inside the run-to-run
  noise. The ray marchers and Mandelbrot did not move at all, and the SDF slope per march step
  (164k cycles per core in ~0.09 ms) implies ~1.8 GHz in both modes. An earlier "default" burn
  figure of 630 us was a cold first run, not the mode. The paper's 1.40 GHz default was measured on
  Windows; the Linux amdxdna driver appears to run near the top clock under load already. Setting
  the mode needs root (`./pmode.sh performance|default` wraps the sudo and library path).
* **Fixed cost per call ~100 us** (89-95 us with no array activity), measured by trace; matches
  the ~160 us we see from Python. One call per frame; batch steps.
* **Heavy CPU load slows the NPU's data movement by up to 59%** (shared power budget, lower
  fabric clock). A pygame loop that burns CPU on drawing slows the NPU it is timing. Pin the host
  process and keep the CPU quiet when measuring.
* **int8 matmul is the chip's real peak**: 8x8x8 `mmul` = 512 MAC/cycle/tile, sustained 338-382;
  bfp16 (block floating point, shared 8-bit exponent) 261-274; **native bf16 matmul only 20-30
  MAC/cycle/tile**, about ten times slower. For heavy MAC work use int8 or bfp16, not bf16.
  (Our bf16 use is elementwise, 32 lanes/cycle, which is fine.)
* On a 32-layer conv net the wins were spreading every layer over all 32 tiles and cutting
  calls per inference; cutting DDR traffic by 61% gained 3%. Vitis AI reached 3-12% of the
  rating, hand-written IRON 24-60%.

Honest verdict: the iGPU (Radeon 860M, ~3 TFLOPS fp32, native rsqrt, per-pixel early exit)
renders these SDF scenes several times faster and in full precision. The NPU wins on dense
int8/int16 matrix work (~50 TOPS vs ~12 for the iGPU), on power, and because it is a third
engine running alongside CPU and GPU. Its demo role is neural: upscale, denoise, style the
GPU's frame; generate audio; run a model (FastFlowLM in `../flm-portable`) next to the demo.

## Gotchas

### Kernel source edits can run stale on the NPU

`@iron.jit` caches a compiled design by the jit function's **name, bytecode, constants,
closure and `CompileTime[...]` kwargs**. The C++ text handed to
`ExternalFunction(source_string=...)` is **not** in the key when it lives in a
module-level string (`KERNEL`, `CELL`, `STEP`, `KSRC` ...), because only the global's
name is in the bytecode, not its value. Editing that string and rerunning in the same
process (or against the on-disk cache in `~/.npu/cache`, override with `NPU_CACHE_HOME`)
runs the **old binary** without any warning. Three different kernels measured identical
timings this way before a `variant: CompileTime[str]` argument was added and the real
numbers appeared (2.3 vs 26 G ops/s per core).

When a kernel edit "does nothing": bump or add a compile-time argument, or clear
`~/.npu/cache`, before debugging the kernel.

### env.sh and script paths

`env.sh` sources `mlir-aie/utils/env_setup.sh` with the command line as its arguments.
An absolute script path is then taken as a Peano install directory and fails with
`Invalid Peano install directory: /...`. Run scripts with a relative path from this
directory. The `realpath: invalid option` line it prints is harmless.

### No `python -I`

`-I` drops `PYTHONPATH`, so `pyxrt` (shipped with XRT, not pip) disappears:
`Unsupported device: npu (CPUOnlyTensor is in use: pyxrt is not importable)`.

### Static schedule check before hardware

```python
from pathlib import Path
from aie.utils.compile import remarks            # needs: ironenv/bin/pip install pyyaml
rep, why = remarks.analyze(ext_fn, "aie2p", Path("/tmp/rem"))
for key, loop in rep.loops.items():
    print(key, loop.ii, loop.ns, loop.zol, loop.trips)
print(rep.libcalls)                              # __muldi3, __divsi3, __mulsf3 ... = slow scalar paths
```

An II of 1 with `ns` 14 is the best case (one 512-bit load, one op, one store per cycle).
`fun/mandel.py --scalar` is an example of the other end: its int64 multiplies become
`__muldi3` library calls; the 16-lane int32 version with `acc64` is 34x faster per core.

See `mlir-aie/programming_guide/section-4/` and `mlir-aie/skills/aie-kernel-opt/SKILL.md`
for the full optimisation playbook.
