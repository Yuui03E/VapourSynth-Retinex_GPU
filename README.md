# VapourSynth Retinex GPU

Retinex and full-resolution edge masks for dark lines and details. One plugin,
two builds; both expose `core.retinex_gpu` and include all parameters in the DLL.

| Install one DLL | Includes |
|---|---|
| `dist/retinex_gpu.dll` | CPU + NVIDIA CUDA, with CPU fallback |
| `dist/cpu-only/retinex_gpu.dll` | CPU only; no NVIDIA software needed |

[Download CPU + CUDA](https://github.com/Yuui03E/VapourSynth-Retinex_GPU/releases/download/v2.1/retinex-2.1-all-in-one-windows-x64.zip) ·
[Download CPU only](https://github.com/Yuui03E/VapourSynth-Retinex_GPU/releases/download/v2.1/retinex-2.1-cpu-only-windows-x64.zip).

Current releases target **Windows x64** and Intel/AMD x64 CPUs. CUDA requires
a compatible NVIDIA GPU and driver; physical testing covers Ryzen 9950X / RTX 5070.
Requires VapourSynth API 4. GRAY/YUV luma → GRAY, at the source size/depth:
integer 8/10/12/16-bit or float32. Kirsch and TCanny are also included.

```python
core.std.LoadPlugin("dist/retinex_gpu.dll")
mask = core.retinex_gpu.RetinexEdgeMask(src, mode="fastest", backend="cpu")
# Use backend="auto" to prefer CUDA, or "gpu", fallback=False to require it.
```

`exact` (default) follows the original algorithm; `fast` approximates illumination;
`fastest` combines local illumination and sampled percentiles for higher speed.
[Parameters](docs/usage.md) · [Build](docs/building.md).

## Tested results

RTX 5070 + Ryzen 9 9950X; 1080p 16-bit luma, 60-frame repeated batches,
32 requests, three-trial medians. Transfers included; decoding excluded.

| Native implementation | Retinex FPS | Mask FPS |
|---|---:|---:|
| Our fastest CUDA (2 workers) | 1421 | 1222 |
| Our fastest CPU (16 workers) | 848 | 507 |
| Our exact CUDA (2 workers) | 301 | 288 |
| Our fast CUDA (downscale 8) | 870 | 792 |
| Our exact CPU (2 workers) | 24 | 14 |
| Our fast CPU (downscale 8) | 40 | 28 |
| Atom normal CPU | 57 | 62 |
| Atom draft CPU (square root) | 2608 | 368 |
| VSJet fast CPU | 283 | 227 |

Fastest CPU masks reached **2.23× VSJet fast CPU** on this system, using
16 independent frame workspaces; speed needs concurrent requests and available
cores. Fastest masks averaged **0.052%** difference from exact
over 29 frames and retained **98.8%** of exact's
selected dark-region pixels at threshold .122. Standalone fastest Retinex differs
more (**2.43%** average).
These measure agreement, not objective quality. **28 regression tests pass.**

VSJet 3.0.0 also has genuine CUDA/OpenCL blur paths with CPU graph stages.
[All native results](docs/wiki/Benchmark-results.md) ·
[Five-scene image comparisons](docs/wiki/Image-comparisons.md) ·
[Reproduce tests](docs/wiki/Reproduce-the-tests.md).

[GitHub Wiki](https://github.com/Yuui03E/VapourSynth-Retinex_GPU/wiki) contains
the same detailed method, benchmark and comparison pages.

GPL-3.0-or-later · [License](LICENSE) · [Credits](NOTICE.md).
