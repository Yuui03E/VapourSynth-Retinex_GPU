# Retinex GPU 2.1

Adds `mode="fastest"` to both all-in-one CPU/CUDA and CPU-only builds.

```python
mask = core.retinex_gpu.RetinexEdgeMask(src, mode="fastest", backend="cpu")
mask = core.retinex_gpu.RetinexEdgeMask(src, mode="fastest", backend="gpu")
```

Fastest keeps local illumination on an 8x8 grid, combines the blurred illumination
planes with their harmonic mean, scores one full-resolution log ratio, and
estimates percentile bounds from grid-center luma samples. The edge arrangement
remains full resolution. CPU uses concurrent frame workspaces, defaulting to
core threads capped at 16; CUDA uses 2. Memory budgets apply per workspace.
`sigma`, percentile tails, blur and operator still apply. Omit `downscale`, which
is fixed in this preset. Exact remains the default mode.

All 28 regression tests pass, including both builds, CPU/CUDA parity, format and
border cases, concurrent requests, uniform/tiny-blur fixes and simulated fallback.
A separate synthetic test checks faint dark lines beside a bright region.
[Current benchmarks](wiki/Benchmark-results.md) compare all native VSJet 3.0.0
CPU/CUDA/OpenCL modes and Atom functions. High CPU speed needs concurrent frame
requests and available cores; it is not a single-core speed claim. Approximation
differences and sampled-percentile limits are documented in the image comparisons.

Release archives each contain one `dist/retinex_gpu.dll`, corresponding source,
licenses and documentation. CUDA targets 75/86/89/120 plus compute_75 PTX. Physical
validation covers Windows x64 / RTX 5070 / Ryzen 9 9950X. Checksums and source
metadata accompany the archives.

[Parameters](usage.md) | [Images](wiki/Image-comparisons.md) | [License](../LICENSE).
