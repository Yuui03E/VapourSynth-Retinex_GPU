# Using the unified plugin

Choose `dist/retinex_gpu.dll` for CPU + NVIDIA CUDA or
`dist/cpu-only/retinex_gpu.dll` for CPU only. Install one variant; they have the
same filename, plugin identity, namespace and four functions. Neither requires
another Retinex/TCanny DLL. Exact CPU processing is included in both builds.
CUDA is NVIDIA GPU processing; there is no OpenCL backend in this plugin.

```python
import vapoursynth as vs
core = vs.core
core.std.LoadPlugin("dist/retinex_gpu.dll")
mask = core.retinex_gpu.RetinexEdgeMask(src, mode="fast", backend="auto")
ret = core.retinex_gpu.MSRCP(src, mode="exact", backend="cpu")
kirsch = core.retinex_gpu.Kirsch(src)
gradient = core.retinex_gpu.TCanny(src, blur_sigma=1, op=1)
```

Input must be constant-format GRAY or YUV, integer 8/10/12/16-bit or float32.
Only luma is processed. Output is GRAY at the source size/depth. Nonfinite floats
become zero; samples are clamped to their representable range. Integer MSRCP
maps to limited range unless sigma is all zero, which preserves sanitized samples
and the source range tag. Masks and edge responses use full range.

## Parameters

All parameters belong to the DLL; Python wrappers are optional.

| Parameter | Default | Meaning |
|---|---|---|
| `mode` | `"exact"` | `"exact"`, `"fast"`, or `"fastest"`; affects MSRCP and RetinexEdgeMask. Kirsch/TCanny retain exact edge stages. |
| `backend` | `"auto"` | `"auto"`, `"gpu"`, `"cpu"`. Auto prefers CUDA in the all-in-one build; CPU-only auto uses CPU. |
| `fallback` | `True` | GPU creation/processing failures switch to CPU and retry the current frame. Set False to require successful GPU processing. |
| `sigma` | `[50,200,350]` | 1–32 finite scales in [0,1000]. All zero bypasses Retinex. |
| `lower_thr`, `upper_thr` | `.001`, `.005` | Nonnegative percentile fractions; sum must be less than 1. |
| `blur_sigma` | `1` | FIR blur before gradient, [0,100]; zero disables blur. |
| `op` | `1` | Prewitt (1) or Sobel (2). |
| `device` | `0` | Nonnegative CUDA device index. |
| `workers` | exact: `2`; fast: `1`; fastest CPU: core threads capped at `16`; fastest GPU: `2` | Independent frame workspaces, 1..16. Fast requires 1. CPU-only auto uses the CPU default; all-in-one auto uses the GPU default. |
| `fast_pyramid` | `False` | Legacy exact-mode 4× illumination approximation; disabled for fidelity. Fast mode uses `downscale` instead. |
| `downscale` | `8` | Fast MSRCP/mask only: 2,4,8,16,32,64. Fastest fixes the illumination grid at 8; omit this parameter. |
| `max_gpu_mb` | `128` | Fast/fastest mode's accounted device-buffer budget in MiB. |
| `reserve_gpu_mb` | `512` | Fast/fastest mode leaves this much currently free device memory at allocation time. |
| `max_cpu_mb` | `128` | Fast/fastest mode's CPU/staging budget in MiB. |

Memory options for fast/fastest are rejected in exact mode and standalone edge functions,
so unused settings cannot silently affect your expectations. Exact mode has no
workspace budget; use fewer workers to reduce memory. Fast/fastest budgets apply to each worker's buffers, excluding driver overhead, VapourSynth frames and other filters.

Every output frame records `RetinexMode`, `RetinexBackend`, `RetinexFallback`
and `RetinexWorkers`.
GPU fallback remains in effect for that workspace; concurrent exact workers can
report different backends after a failure. Simulated failure tests do not prove
recovery from every physical driver/device-loss condition.

## Optional Python wrappers

`python/retinex_gpu.py` contains `retinex_edgemask_gpu`, `retinex_fast` and
`retinex_edgemask_fast`, plus `retinex_fastest` and `retinex_edgemask_fastest`. All parameters are registered in the DLL; wrappers are
optional. Both builds use `core.retinex_gpu`; load one variant only.

Uniform input remains constant through exact Retinex. Collapsed percentile
intervals use normalized input. These deliberate correctness choices differ
from unstable or undefined original behavior. Near-flat nonuniform inputs can
still show numerical sensitivity; exact is not a universal bit-exact promise.

[Fast algorithms and resources](fast.md) · [Examples](../examples/mask.vpy) ·
[VSView comparisons](../examples/compare_vsview.vpy).

## Fastest preset

```python
mask = core.retinex_gpu.RetinexEdgeMask(src, mode="fastest", backend="cpu")
mask = core.retinex_gpu.RetinexEdgeMask(src, mode="fastest", backend="gpu", fallback=False)
```

Fastest keeps local illumination, source detail and the same edge arrangement.
It combines the small Gaussian illumination planes into one harmonic estimate,
uses one full-resolution log ratio and estimates percentiles from grid-center
samples. CPU workspaces process independent frames in parallel; reduce `workers`
for lower CPU use. High throughput needs concurrent VapourSynth frame requests.
One sequential request cannot use the full CPU pool. On all-in-one auto, a CUDA
failure retains the GPU-default worker count when falling back; choose CPU
explicitly for its larger default pool. [Measured limits](wiki/Benchmark-results.md).
