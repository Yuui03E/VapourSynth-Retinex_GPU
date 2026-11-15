# Implementation

## The original-compatible pipeline

The exact plugin follows the audited kagefunc implementation:

1. Extract luma and apply MSRCP with scales `[50, 200, 350]` and percentile tails `.001/.005`.
2. Apply Gaussian blur sigma 1 and Prewitt gradient magnitude, equivalent to TCanny mode 1.
3. Take the minimum over the center and four diagonal neighbors.
4. Add the Kirsch response of the original luma and clamp to the output peak.

The original mask uses four absolute Kirsch responses despite an eight-direction
description. The integer Retinex intermediate uses limited range. The recursive
Gaussian's endpoint behavior, double precision and calculation order are part of
the compatibility target; an idealized Gaussian is not automatically equivalent.

`src/plugin.cpp` validates the shared API and selects the engine/backend.
`src/exact/cpu.cpp` provides native exact CPU processing, including all four stages.
`src/exact/gpu.cu` implements the math and CUDA graphs. Each worker owns a CUDA
stream and preallocated buffers. Intermediate images stay on the GPU between
one input upload and one output download. The default is two workers.

## Why fast mode is faster

Fast mode averages small luma cells, normally reducing illumination estimation
by a factor of 8 in each dimension. It computes the large recursive blurs on that
smaller image and reconstructs illumination for the original-resolution pixels.
It does not enlarge a small finished mask.

The shared code uses float full-frame scoring, percentile balancing, a positive-log
approximation and an illumination floor. It retains the original mask arrangement.
This reduces work and buffers, with measurable approximation differences.

`src/fast/common.hpp` holds shared math; `cpu.cpp` and `simd.hpp` implement CPU
processing; `gpu.cu` implements the CUDA path. `src/plugin.cpp` manages the API,
workspace budgets, backend selection and fallback. Fast has one workspace per
filter instance. Fastest uses multiple workspaces scheduled by VapourSynth;
there is no separate internal CPU thread pool. Input requests can overlap.

All three modes support CPU and CUDA through one DLL. The CPU-only variant
exposes the same functions and omits CUDA. Exact CPU uses double recursive
Gaussian operations and float FIR/edge stages matching the CUDA arithmetic order.
Auto mode checks GPU availability (and fast/fastest memory) and can fall back to CPU. A reported
GPU frame failure retries that frame on CPU and keeps subsequent work on CPU.
Budgets cover this instance's accounted buffers, not total process memory or
other filters. Low stream priority is not a GPU utilization controller.

## Fastest local illumination

Fastest combines the small blurred illumination planes with a harmonic mean.
It evaluates one full-resolution `log(1 + normalized_luma / illumination)`
instead of three separate log ratios. Percentile balancing uses actual luma
samples at the centers of the 8×8 grid cells, reducing histogram work by about
64 times on dimensions divisible by eight. Input min/max still uses the whole
frame; edges and source Kirsch remain full resolution.

CPU uses SIMD scoring and independent frame workspaces, defaulting to core
threads capped at 16. CUDA uses two workspaces and keeps intermediate data on
device. Frame properties record the actual workspace count. Higher throughput
therefore depends on concurrent frame requests, not just the new approximation.
Sampled percentiles can miss isolated extreme pixels or fluctuate on moving small
features; the consecutive-frame residual comparison records that practical limit.

## How Atom and VSJet differ

Atom normal uses the original CPU MSRCP followed by TCanny and its own Kirsch
function. Atom draft replaces MSRCP with square-root brightening. Its Kirsch code
rotates a flattened coefficient list into four nonstandard kernels. For a gentle
left-to-right brightness ramp its strongest absolute response is 8 times the slope;
for a top-to-bottom ramp it is 24. Our original-compatible kernels reach 24 in
either case. This weakens Atom's Kirsch response to vertical contours; TCanny still
contributes to the final mask.

VSJet has a different Retinex implementation. Fast mode substitutes the frame
average for the largest blur. CPU fast also downsamples the other blurs; native
CUDA/OpenCL fast keeps them at full resolution. GPU Gaussian nodes are combined
with CPU expression/statistics stages. Its DRE mask also uses different edge stages,
including eight Kirsch directions. We tested its soft mask with `brz=0`; the
usual default binarizes at `brz=0.122`. More selected texture can be useful or
unwanted depending on the next filter.

## Compatibility limits

Exact mode is an algorithmic compatibility target, not a universal bit-exact
promise. Float support extends an original plugin that rejects float. Collapsed
percentile intervals and nearly flat inputs have documented numerical exceptions.
Uniform input now remains constant, and extremely small FIR blur values remain finite. These correctness fixes intentionally differ from unstable upstream behavior.

The detailed equations and pinned upstream revisions are in `docs/reference.md`.
API/resource parameters are in `docs/usage.md` and `docs/fast.md`; contracts and
raw numerical evidence are under `tests/` and `docs/results/`.
