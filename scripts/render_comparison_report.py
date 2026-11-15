"""Regenerate the current README, Wiki tables and charts from curated evidence.

Run after compare_implementations.py and verify_comparisons.py, then copy their
reports to docs/results/. Charts require matplotlib; no network calls occur.
"""
from pathlib import Path
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    'our_fastest': 'Our fastest CUDA (2 workers)',
    'our_fastest_cpu': 'Our fastest CPU (16 workers)',
    'our_exact': 'Our exact CUDA (2 workers)',
    'our_fast8': 'Our fast CUDA (downscale 8)',
    'our_exact_cpu': 'Our exact CPU (2 workers)',
    'our_fast_cpu': 'Our fast CPU (downscale 8)',
    'atom_normal': 'Atom normal CPU', 'atom_draft': 'Atom draft CPU (square root)',
    'vsjet_cpu_fast': 'VSJet fast CPU', 'vsjet_cpu_full': 'VSJet full CPU',
    'vsjet_cuda_fast': 'VSJet fast CUDA blurs + CPU',
    'vsjet_cuda_full': 'VSJet full CUDA blurs + CPU',
    'vsjet_opencl_fast': 'VSJet fast OpenCL blurs + CPU',
    'vsjet_opencl_full': 'VSJet full OpenCL blurs + CPU',
}


def read(name):
    return json.loads((ROOT / 'docs/results' / (name + '.json')).read_text(encoding='utf-8'))


def write(name, content):
    (ROOT / name).write_text(content.strip() + '\n', encoding='utf-8')


def main():
    data, images, audit, parity = (read(n) for n in (
        'comparison', 'comparison-images', 'backend-audit', 'fastest-video-parity'))
    assert all(r['passed'] for r in (data, images, audit, parity))
    assert audit['project_binary_sha256'] == data['sha256']['retinex_gpu.dll']
    assert parity['candidate_sha256'] == parity['anchor_sha256'] == data['sha256']['retinex_gpu.dll']
    contracts = [read(n) for n in (
        'unified-contracts', 'fast-contracts', 'contract-tests', 'fastest-contracts')]
    assert all(r['passed'] for r in contracts)
    tests = sum(r['tests'] for r in contracts)
    assert len(LABELS) * 2 == len(data['timing'])
    assert images['frames'] == data['visual_frames']
    timing = {(r['candidate'], r['stage']): r['fps'] for r in data['timing']}

    def timing_table(names, digits=1):
        return '\n'.join(['| Native implementation | Retinex FPS | Mask FPS |',
            '|---|---:|---:|'] + [f"| {LABELS[n]} | {timing[n, 'retinex']:.{digits}f} | {timing[n, 'mask']:.{digits}f} |" for n in names])

    fastest = data['summary']['mask']['our_fastest']
    dark = data['dark_summary']['our_fastest']['dark']
    ratio = timing['our_fastest_cpu', 'mask'] / timing['vsjet_cpu_fast', 'mask']
    write('README.md', f'''
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

{timing_table(['our_fastest', 'our_fastest_cpu', 'our_exact', 'our_fast8', 'our_exact_cpu', 'our_fast_cpu', 'atom_normal', 'atom_draft', 'vsjet_cpu_fast'], 0)}

Fastest CPU masks reached **{ratio:.2f}× VSJet fast CPU** on this system, using
16 independent frame workspaces; speed needs concurrent requests and available
cores. Fastest masks averaged **{fastest['mae'] * 100:.3f}%** difference from exact
over 29 frames and retained **{dark['0.122']['recall'] * 100:.1f}%** of exact's
selected dark-region pixels at threshold .122. Standalone fastest Retinex differs
more (**{data['summary']['retinex']['our_fastest']['mae'] * 100:.2f}%** average).
These measure agreement, not objective quality. **{tests} regression tests pass.**

VSJet 3.0.0 also has genuine CUDA/OpenCL blur paths with CPU graph stages.
[All native results](docs/wiki/Benchmark-results.md) ·
[Five-scene image comparisons](docs/wiki/Image-comparisons.md) ·
[Reproduce tests](docs/wiki/Reproduce-the-tests.md).

[GitHub Wiki](https://github.com/Yuui03E/VapourSynth-Retinex_GPU/wiki) contains
the same detailed method, benchmark and comparison pages.

GPL-3.0-or-later · [License](LICENSE) · [Credits](NOTICE.md).
''')

    write('docs/wiki/Benchmark-results.md', f'''
# Benchmark results

Current 2.1 DLLs, VSJet **3.0.0**, official vszip **24.0.0**, vszipcu/vszipcl
**1.2.0**. RTX 5070 and Ryzen 9 9950X, Windows x64, VapourSynth R81,
Python 3.14.10, BestSource 23. This is one 1920×1080 episode on one system.

{timing_table(LABELS)}

Fastest CPU masks reached **{ratio:.2f}× VSJet fast CPU**, while fastest CUDA
was the fastest tested complete mask. The CPU result uses 16 frame workspaces
and many concurrent requests. It does not establish the same advantage on a
smaller CPU or during sequential frame requests. CPU-only output agrees with
the CPU engine in the all-in-one build in regression tests; this throughput
table times the all-in-one CPU engine.

Standalone Retinex and complete masks are separate workloads. Atom draft's
square-root prefilter is faster than our fastest standalone Retinex, while its
complete mask is slower here. Native masks have different edge pipelines.
Similarity to exact does not establish a perceptual-quality winner.

![Complete-mask throughput](images/mask-throughput.png)
![Standalone-prefilter throughput](images/retinex-throughput.png)

## Timing method

Each trial receives the same **60 unique predecoded frames**: 20 from each
sequence starting at 1440, 10080 and 30240. Eight separate warmup frames repeat
for at least 0.5 seconds. An entire timed batch repeats until at least one second
has elapsed; slow paths may complete just one batch. Actual frames, cycles,
duration, process CPU use and project worker counts are recorded per trial.
The table is the median of three trials, with 32 requests and 32 VapourSynth
core threads: 84 trials covering 28 candidate/stage combinations.

Fresh graphs and all native intermediate caches are disabled. Trial order rotates.
Input/output transfers and allocation are included. Decoding, luma/depth conversion,
graph construction and warmup are excluded. This measures warmed filter throughput
on repeated fixed input, not whole-episode decoding/encoding speed. Unrelated
system activity was not controlled by the harness.

Our exact uses two workspaces; fast uses one with downscale 8. Fastest uses 16
CPU workspaces or two CUDA workspaces. GPU fallback is disabled for all project
CUDA paths. Options: sigma 50/200/350, percentile tails .001/.005, blur sigma 1,
Prewitt op 1. VSJet DRE uses `brz=0` to compare soft masks rather than its
default thresholded mask. Memory limits apply per fastest workspace.

## Native backend verification

VSJet CUDA and OpenCL Retinex combine full-resolution native Gaussian filters
with CPU expressions/statistics and other CPU stages. Fast has two Gaussian
nodes, full has three; masks add a sigma-1 blur. CPU fast also downsamples its
remaining Gaussian work; GPU fast does not. Our CUDA intermediates remain on
device between upload and download. Native GPU Gaussian uses finite float FIR
taps, while our illumination uses recursive Gaussian. These are different
native implementations, not equal-work microbenchmarks.

All {len(audit['executions'])} candidate/stage graphs were executed. Installed
VSJet Python files match pinned 3.0.0 source, and native comparison DLLs match
official wheels. See [raw comparison](../results/comparison.json),
[backend audit](../results/backend-audit.json),
[image differences](Image-comparisons.md) and [reproduction](Reproduce-the-tests.md).
''')

    def similarity(stage):
        return '\n'.join(['| Candidate | Mean absolute difference | Largest sampled pixel difference |',
            '|---|---:|---:|'] + [f"| {LABELS[n]} | {data['summary'][stage][n]['mae'] * 100:.5f}% | {data['summary'][stage][n]['maximum'] * 100:.3f}% |" for n in LABELS if n != 'our_exact'])

    descriptions = ['Dark interior / hair contours', 'Bright face / fine lines',
        'Buildings / fine structure', 'Shop shelves / weak texture', 'Dark foliage / contours']
    scenes = ['| Frame | Timestamp | Inspect | Mask | Retinex |', '|---|---|---|---|---|']
    for index, frame in enumerate(images['frames']):
        seconds = frame / images['fps']
        stamp = f'{int(seconds // 60):02d}:{seconds % 60:06.3f}'
        scenes.append(f"| {frame} | {stamp} | {descriptions[index]} | [Compare]({images['stages']['mask']['scenes'][index]['url']}) | [Compare]({images['stages']['retinex']['scenes'][index]['url']}) |")
    dark_table = ['| Candidate | Exact-selected pixels retained (recall) | Candidate pixels also selected by exact (precision) |', '|---|---:|---:|']
    for n in ('our_fastest', 'our_fast8', 'atom_normal', 'atom_draft', 'vsjet_cpu_fast'):
        row = data['dark_summary'][n]['dark']['0.122']
        dark_table.append(f"| {LABELS[n]} | {row['recall'] * 100:.3f}% | {row['precision'] * 100:.3f}% |")
    temporal = ['| Candidate | Mean change in candidate-minus-exact mask residual |', '|---|---:|']
    for n in ('our_fastest', 'our_fast8', 'atom_normal', 'atom_draft', 'vsjet_cpu_fast'):
        value = statistics.mean(r['residual_change_mae'] for r in data['temporal'] if r['candidate'] == n)
        temporal.append(f'| {LABELS[n]} | {value * 100:.5f}% |')
    synthetic = read('fastest-dark-line')
    contribution = data['dark_summary']['our_fastest']['dark_retinex_contribution']['0.122']
    write('docs/wiki/Image-comparisons.md', f'''
# Image comparisons

Five scenes, two collections, nine switchable images per scene: source luma,
our exact, fast and fastest CUDA, Atom normal/draft, VSJet CPU fast, CUDA fast
and CUDA full. OpenCL and project CPU modes are retained in numerical comparisons.
Fastest CPU and CUDA matched exactly on all 29 sampled frames in both stages;
[parity evidence](../results/fastest-video-parity.json). Regression tests also
cover the CPU-only build. This is sampled parity, not a universal bit-exact promise.

- [Switch between complete masks]({images['stages']['mask']['url']})
- [Switch between Retinex / substitute prefilters]({images['stages']['retinex']['url']})

Select a scene and click the image or use the selector to switch at the same
position. Zoom to inspect thin contours. Captures are lossless 1920×1080
grayscale PNGs with fixed range mapping and no per-image contrast stretch.
**All {images['verified_images']} uploaded images have identical decoded pixels
to their local originals.** PNG encoding or grayscale palettes can change during
lossless upload optimization. File and pixel hashes are recorded separately in
[image evidence](../results/comparison-images.json).

## Scenes

{chr(10).join(scenes)}

Five selected scenes plus 24 distributed frames yield 29 unique numerical samples.
This is one anime episode; no downstream denoised/encoded output, labelled edge
map or perceptual ground truth was tested.

## Output similarity

Exact CUDA is an original-algorithm compatibility anchor. Errors below are
percentages of the mapped 0–1 range, measured before 8-bit PNG rounding.
Lower values mean closer to this chosen algorithm, not necessarily better-looking.

### Complete native soft masks

{similarity('mask')}

### Standalone Retinex / substitute prefilters

{similarity('retinex')}

Fastest stays close to exact masks on these samples, but its standalone Retinex
changes brightness and contrast more substantially. A small whole-frame average
does not rule out larger differences in isolated pixels, tiny frames or small
moving highlights. Grid-center percentile sampling can miss isolated extremes.
Use exact when the original algorithm's behavior matters.

Atom normal Retinex matches exact on these nonuniform samples, but its native
mask uses different Kirsch kernels. VSJet DRE has its own edge arrangement and
is compared with `brz=0`. Selecting more weak texture can be useful or unwanted
depending on the next filter; additional selected pixels are not labelled noise.

## Dark-detail agreement

For source luma below .15 after range alignment, threshold both soft masks at
**.122**. Recall counts exact-selected pixels retained; precision counts candidate
pixels also selected by exact. Both use pooled pixel counts, not a per-frame mean.

{chr(10).join(dark_table)}

At the lower threshold .03, fastest recall is {dark['0.03']['recall'] * 100:.3f}%
and precision is {dark['0.03']['precision'] * 100:.3f}%. Restricting the .122 test
to dark pixels where exact's Retinex branch contributes more than .01 gives
fastest recall {contribution['recall'] * 100:.3f}% and precision
{contribution['precision'] * 100:.3f}%. This helps separate Retinex agreement from
the shared source Kirsch response. It still measures agreement with exact,
not labelled real edges or an objective ability to reject noise.

A separate synthetic scene places faint lines (contrast .0006) beside a bright
patch (level .6). Fastest retains {synthetic['recall'] * 100:.3f}% of the
{synthetic['reference_pixels']:,} exact-selected pixels near the line, with
**different thresholds: exact .03, fastest .025**. This checks line preservation
with modest threshold adjustment; it is not an equal-threshold benchmark.
[Synthetic test evidence](../results/fastest-dark-line.json).

## Correctness and limits

Uniform black, mid-level and white inputs produce constant Retinex and empty
masks in the tested CPU/CUDA formats. Tiny positive FIR blur values stay finite
and behave as identity blur. Collapsed percentile intervals use normalized input;
fastest can also fall back when its sampled score range collapses. Near-flat
nonuniform inputs can remain sensitive to precision. These cases are covered
separately by regression tests; the video samples are nonuniform.

## Consecutive-frame residuals

Three 24-frame sequences produce 69 transitions:

{chr(10).join(temporal)}

This is not motion compensated and is not an absolute flicker score. The upload
uses VSView's installed slow.pics extension through the reproducible uploader.
Collections are link-only with no scheduled expiry.

[Native timings](Benchmark-results.md) · [Test reproduction](Reproduce-the-tests.md).
''')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for stage, title in [('mask', 'Complete native soft masks'), ('retinex', 'Standalone Retinex / substitute prefilters')]:
        names = sorted(LABELS, key=lambda n: timing[n, stage], reverse=True)
        values = [timing[n, stage] for n in names]
        colors = ['#086f77' if n.startswith('our_fastest') else '#668cbb' if n.startswith('our_') else '#8b909b' for n in names]
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.barh([LABELS[n] for n in names], values, color=colors)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.15)
        for i, value in enumerate(values):
            ax.text(value + max(values) * .012, i, f'{value:,.1f}', va='center', fontsize=10)
        ax.set_xlabel('Frames per second — median of three trials')
        ax.set_title(title + '\nRTX 5070 + Ryzen 9 9950X | 1080p Gray16 | 32 requests', fontsize=13)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
        fig.text(.02, .015, 'Transfers included; decode excluded. Fastest CPU: 16 workspaces. Native algorithms differ.', fontsize=9)
        fig.tight_layout(rect=(0, .045, 1, 1))
        fig.savefig(ROOT / f'docs/wiki/images/{stage}-throughput.png', dpi=160)
        plt.close(fig)
    print('Regenerated README, Wiki tables and two throughput charts from verified evidence.')


if __name__ == '__main__':
    main()
