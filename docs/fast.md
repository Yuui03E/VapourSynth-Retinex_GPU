# Fast and fastest modes

Fast mode is included in **both** DLL variants. Select it with
`core.retinex_gpu.RetinexEdgeMask(src, mode="fast", backend="auto")` or
`core.retinex_gpu.MSRCP(src, mode="fast")`. There is no separate fast plugin.

The engine averages illumination cells, normally 8×8, and runs the recursive
Gaussian blurs on that smaller plane. It reconstructs illumination at the original
pixel positions and retains full-resolution luma detail and edge extraction.
Float scoring, a bounded log approximation and an illumination floor reduce work
and memory. This is an approximation to exact Retinex, especially for the standalone
MSRCP image; compatibility is not an objective quality score.

Use `downscale=4` for a closer illumination estimate or 8 for the routine default.
Available scales are 2,4,8,16,32,64. Zero scales bypass MSRCP; sigma lists have at most
32 entries. One workspace processes ready frames serially while input requests
can overlap. CPU uses a score cache when its budget allows and streaming otherwise.

Defaults: 128 MiB device budget, 128 MiB CPU/staging budget, 512 MiB GPU reserve.
These cover accounted buffers for this instance, excluding driver overhead, frame
caches and competing filters. Insufficient GPU/staging memory falls back to CPU;
an insufficient essential CPU budget is an error. Reserve is checked at allocation,
not a guarantee against later memory pressure. Low CUDA stream priority does not
limit GPU utilization.

`backend="cpu"` keeps the GPU available for other filters. `backend="gpu",
fallback=False` is appropriate for GPU benchmarks. A reported frame-processing
failure releases that workspace's GPU resources, retries the frame on CPU and
keeps later processing on CPU. Default CPU-only auto uses CPU without a warning.

Diagnostics retain `RetinexFastDownscale`, `RetinexFastGPUBytes`,
`RetinexFastCPUBytes` and `RetinexFastScoreCache`, alongside the unified backend
and fallback properties. Use [the parameter reference](usage.md) for limits.

[Measured results](wiki/Benchmark-results.md) ·
[Fast/CPU/exact example](../examples/compare_retinex_fast.vpy) ·
[Implementation](wiki/Implementation.md).

## Fastest

`mode="fastest"` uses an 8×8 illumination grid and the same small recursive
Gaussian blurs. Their harmonic mean supplies one illumination plane, replacing
three full-resolution log/interpolation evaluations with one. A 4096-bin
percentile histogram uses one original-luma sample at each grid-cell center.
The finished mask stays at full resolution and retains Gaussian/Prewitt, corner
minimum and the original source Kirsch stage.

There is no cached full-frame score image on CPU. Fastest uses multiple independent
frame workspaces: the CPU default equals core threads capped at 16; CUDA uses 2.
Budgets apply per workspace. Low request counts or fewer cores reduce the speed
benefit. `sigma`, percentile tails, blur and operator still apply; `downscale`
is fixed and is rejected when explicitly supplied.

Approximate illumination and sampled percentiles can differ on isolated extremes,
tiny frames and moving small features. Uniform or collapsed sampled ranges fall
back to normalized input. Use exact for algorithm fidelity or fast for the closer
three-ratio approximation. Fastest is measured as a mask/detail approximation,
not a universal perceptual-quality winner.
