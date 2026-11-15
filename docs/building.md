# Build and verify

Requires VapourSynth API 4 headers/runtime, CMake 3.25+, and a C++17 compiler.
Windows scripts locate MSVC Build Tools and Ninja. The all-in-one build also
needs CUDA Toolkit 12.8+. Using its CUDA backend needs an NVIDIA driver; CPU-only
needs no NVIDIA software. Linux support has not been verified on hardware.

```powershell
./scripts/build.ps1                    # build/all-in-one/retinex_gpu.dll
./scripts/build.ps1 -CpuOnly           # build/cpu-only/retinex_gpu.dll
```

CPU-only sets `RETINEX_CUDA=OFF`. All-in-one targets SM 75/86/89/120 plus
compute_75 PTX. Override with `-Architectures 120`. CUDA is found from PATH or
local `.tools/cuda/`. Only Windows x64 / RTX 5070 was physically tested.

Current release DLLs target Windows x64 and Intel/AMD x64 CPUs. CPU SIMD uses
baseline SSE2, with no AVX2 requirement. The CPU hardware tested was Ryzen 9950X;
performance on other CPUs is not established. CUDA binaries target compute
capability 7.5 and newer through the listed native targets and PTX, subject to
driver compatibility. Older CUDA GPUs, AMD/Intel GPUs, ARM and 32-bit Windows
are outside the current release target.

## Correctness tests

Install Python, NumPy and VapourSynth; fetch the pinned optional reference sources:

```powershell
python scripts/fetch_references.py
./scripts/build.ps1 -References
./scripts/build.ps1 -CpuOnly
./scripts/build.ps1 -Testing -Architectures 120
python tests/test_unified.py
python tests/test_fast.py
python tests/test_fastest.py
python tests/test_contract.py
python tests/verify.py
```

These checks need a CUDA GPU. The failure-injection build must never be shipped.
The original CPU plugins are verification dependencies only. Uniform correctness
is tested separately from upstream fidelity because the original algorithm can
create false contrast on constant input.

## Native comparisons and packaging

[Reproduce the comparisons](wiki/Reproduce-the-tests.md) covers pinned VSJet,
native plugins, the source video and Atom functions. Reports go to ignored
`build/reports/`; reviewed current evidence is under `docs/results/`.

```powershell
python scripts/check_repository.py
python scripts/package_release.py
```

Packaging requires passing contract reports and curated comparison/audit reports
whose checksums identify the current binaries. It creates two ZIPs in
`dist/releases/`, each containing one `dist/retinex_gpu.dll`, corresponding
source, licenses and documentation. Binaries and downloaded tools stay out of Git.
