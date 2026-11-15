# Reproduce the tests

Use your own source video and the supplied `atomchtools.py`; neither is bundled.
Source/Atom SHA256, plugin hashes and frame selections are recorded in
[comparison.json](../results/comparison.json). The tested episode has 34,608
frames at 24000/1001 FPS and native 1920×1080 resolution.

## Build and pin dependencies

Follow [building](../building.md). Install VapourSynth, NumPy, Pillow and
BestSource. Pin the comparison package without changing a global VSJet install:

```powershell
python -m pip install --no-deps --target .tools/vsjet vsjetpack==3.0.0
python -m pip install vapoursynth-vszip==24.0.0 vapoursynth-vszipcu==1.2.0 vapoursynth-vszipcl==1.2.0
python scripts/fetch_references.py
./scripts/build.ps1 -References
./scripts/build.ps1 -CpuOnly
cmake --install build/all-in-one --prefix dist
cmake --install build/cpu-only --prefix dist/cpu-only
```

The no-deps isolated install assumes the VSJet requirements are already installed.
Plugin loading must select the DLL hashes recorded by the audit. The installed
CPU vszip build is architecture-specific. Akarin, resizers and std filters are
also used. VSJet is a Python graph over native filters, not a replacement DLL
created by this project.

## Run the native audit and comparison

```powershell
python tests/audit_backends.py "D:/video.mkv" --atom "D:/atomchtools.py"
python tests/compare_implementations.py "D:/video.mkv" --atom "D:/atomchtools.py" --output build/reports/current-comparison-2.1 --frames 1440 7200 10080 18720 30240 --starts 1440 10080 30240 --per-sequence 20 --requests 32 --repeats 3 --warmup 8 --minimum-seconds 1
```

For other videos choose valid scenes and starts. The recorded run evaluates
29 unique quality frames: five selected scenes plus 24 distributed samples.
Three 24-frame mask sequences add 69 transitions. Timings use 60 unique frames
per batch, repeating entire batches to at least one second. Warmup is separate.

`make()` explicitly chooses each native backend. All three of our CUDA modes require
GPU success. Atom's two supplied function definitions execute unchanged;
unrelated imports are omitted. The original Retinex and TCanny CPU adapters
compile upstream numerical code with API 4 entry points. All graph caches are
disabled for timing, including intermediate native nodes.

The graph audit fetches a final output for each path, verifies Gaussian counts
and resolution, matches Python source to pinned upstream, and checks installed
native DLL bytes against official PyPI wheels. GPU/CPU code uses different
Gaussian algorithms; no claim of mathematically identical work is made.

## Metrics and display

MSRCP integer range 4096–60160 maps to 0–1 for ours and Atom normal; VSJet and
Atom draft map 0–65535 to 0–1. Soft masks map 0–65535 to 0–1. PNGs use fixed
8-bit rounding with no per-image contrast stretch. Numerical metrics precede
PNG conversion. Mean absolute error measures similarity to exact; RMSE pools
squared per-frame errors. `mean_frame_p99` is the mean of frame percentiles,
not a global percentile. Maximum is the largest sampled pixel difference.
Uniform frames are excluded from compatibility averages and checked separately
by correctness tests. The primary 29-frame sample has no uniform frames.

Binary agreement uses threshold .122. Dark-region agreement also uses .03 and
.122, with source luma below .15; a second region requires more than .01 of the
exact mask to come from its Retinex branch. Recall and precision are aggregated
from pixel counts relative to exact, not labels of real edges. Temporal metrics measure change in the
candidate-minus-exact residual, without motion compensation. Neither is a
perceptual ground truth or a downstream denoising/restoration quality score.

## VSView and slow.pics

```powershell
vsview --arg source=D:/video.mkv --arg atom=D:/atomchtools.py --arg stage=mask examples/compare_vsview.vpy
python scripts/upload_comparisons.py build/reports/current-comparison-2.1/images.json --upload --name "Retinex GPU 2.1 vs Atom and VSJet 3.0.0 - 1080p"
```

Install VSView and its vsview-comp extension. Review PNGs before uploading.
Each collection has five scenes with nine lossless images, below the service's
50-image limit. The uploader respects the service retry timer and resumes already
completed collections. Private upload records stay in ignored build folders.
After upload, verify all decoded pixels against the local originals:

```powershell
python scripts/verify_comparisons.py build/reports/current-comparison-2.1/images.json
```

PNG encoding bytes can differ after lossless optimization; both file and decoded
pixel hashes are recorded. Grayscale palette conversion is checked without
accepting colored or changed pixels.

## Fastest validation and report generation

```powershell
./scripts/build.ps1 -Testing -Architectures 120
python tests/test_fastest.py
python tests/benchmark_fastest.py "D:/video.mkv" --skip-timing --output build/reports/fastest-video-parity.json
```

Omit `--skip-timing` to run the optional CPU/CUDA worker-count sweep. The recorded
parity run compares 16 CPU workspaces with two CUDA workspaces on 29 frames.
The synthetic test uses different mask thresholds, exact .03 and fastest .025;
the native video comparison uses equal thresholds .03 and .122.

Copy passing current reports into `docs/results/`. Then regenerate the README,
Wiki tables and charts directly from those reports (requires matplotlib):

```powershell
python scripts/render_comparison_report.py
python scripts/check_repository.py
python scripts/package_release.py
```

The renderer documents this measured system and frame selection. When measuring
another system or source, update the hardware and scene descriptions too.

The commands above build, test, upload comparison images and package releases.
They do not push source or publish a GitHub release. Wiki sources are retained
in `docs/wiki/` alongside the public evidence for reproduction.
