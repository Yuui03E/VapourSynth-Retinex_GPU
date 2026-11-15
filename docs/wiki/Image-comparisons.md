# Image comparisons

Five scenes, two collections, nine switchable images per scene: source luma,
our exact, fast and fastest CUDA, Atom normal/draft, VSJet CPU fast, CUDA fast
and CUDA full. OpenCL and project CPU modes are retained in numerical comparisons.
Fastest CPU and CUDA matched exactly on all 29 sampled frames in both stages;
[parity evidence](../results/fastest-video-parity.json). Regression tests also
cover the CPU-only build. This is sampled parity, not a universal bit-exact promise.

- [Switch between complete masks](https://slow.pics/c/B5GxGitq)
- [Switch between Retinex / substitute prefilters](https://slow.pics/c/Xg9o1BUH)

Select a scene and click the image or use the selector to switch at the same
position. Zoom to inspect thin contours. Captures are lossless 1920×1080
grayscale PNGs with fixed range mapping and no per-image contrast stretch.
**All 90 uploaded images have identical decoded pixels
to their local originals.** PNG encoding or grayscale palettes can change during
lossless upload optimization. File and pixel hashes are recorded separately in
[image evidence](../results/comparison-images.json).

## Scenes

| Frame | Timestamp | Inspect | Mask | Retinex |
|---|---|---|---|---|
| 1440 | 01:00.060 | Dark interior / hair contours | [Compare](https://slow.pics/c/eatGXQJj) | [Compare](https://slow.pics/c/sCXMvBYu) |
| 7200 | 05:00.300 | Bright face / fine lines | [Compare](https://slow.pics/c/65VeEjKL) | [Compare](https://slow.pics/c/qdk9RLwN) |
| 10080 | 07:00.420 | Buildings / fine structure | [Compare](https://slow.pics/c/lD68Ou7l) | [Compare](https://slow.pics/c/PeDphsxJ) |
| 18720 | 13:00.780 | Shop shelves / weak texture | [Compare](https://slow.pics/c/ASgWiS0X) | [Compare](https://slow.pics/c/WJQFnpyO) |
| 30240 | 21:01.260 | Dark foliage / contours | [Compare](https://slow.pics/c/zrQLM8up) | [Compare](https://slow.pics/c/3uEc0DWU) |

Five selected scenes plus 24 distributed frames yield 29 unique numerical samples.
This is one anime episode; no downstream denoised/encoded output, labelled edge
map or perceptual ground truth was tested.

## Output similarity

Exact CUDA is an original-algorithm compatibility anchor. Errors below are
percentages of the mapped 0–1 range, measured before 8-bit PNG rounding.
Lower values mean closer to this chosen algorithm, not necessarily better-looking.

### Complete native soft masks

| Candidate | Mean absolute difference | Largest sampled pixel difference |
|---|---:|---:|
| Our fastest CUDA (2 workers) | 0.05184% | 7.903% |
| Our fastest CPU (16 workers) | 0.05184% | 7.903% |
| Our fast CUDA (downscale 8) | 0.04732% | 7.930% |
| Our exact CPU (2 workers) | 0.00000% | 0.000% |
| Our fast CPU (downscale 8) | 0.04732% | 7.930% |
| Atom normal CPU | 4.15682% | 75.782% |
| Atom draft CPU (square root) | 4.64081% | 76.443% |
| VSJet fast CPU | 2.56041% | 83.674% |
| VSJet full CPU | 2.57812% | 83.674% |
| VSJet fast CUDA blurs + CPU | 2.56940% | 83.674% |
| VSJet full CUDA blurs + CPU | 2.57735% | 83.674% |
| VSJet fast OpenCL blurs + CPU | 2.56940% | 83.674% |
| VSJet full OpenCL blurs + CPU | 2.57735% | 83.674% |

### Standalone Retinex / substitute prefilters

| Candidate | Mean absolute difference | Largest sampled pixel difference |
|---|---:|---:|
| Our fastest CUDA (2 workers) | 2.42542% | 42.141% |
| Our fastest CPU (16 workers) | 2.42542% | 42.141% |
| Our fast CUDA (downscale 8) | 2.39318% | 42.302% |
| Our exact CPU (2 workers) | 0.00000% | 0.000% |
| Our fast CPU (downscale 8) | 2.39318% | 42.302% |
| Atom normal CPU | 0.00000% | 0.000% |
| Atom draft CPU (square root) | 15.12813% | 64.512% |
| VSJet fast CPU | 5.26582% | 54.926% |
| VSJet full CPU | 3.32741% | 44.803% |
| VSJet fast CUDA blurs + CPU | 5.08052% | 54.909% |
| VSJet full CUDA blurs + CPU | 3.32697% | 44.985% |
| VSJet fast OpenCL blurs + CPU | 5.08052% | 54.909% |
| VSJet full OpenCL blurs + CPU | 3.32697% | 44.985% |

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

| Candidate | Exact-selected pixels retained (recall) | Candidate pixels also selected by exact (precision) |
|---|---:|---:|
| Our fastest CUDA (2 workers) | 98.804% | 99.831% |
| Our fast CUDA (downscale 8) | 99.009% | 99.768% |
| Atom normal CPU | 71.885% | 98.510% |
| Atom draft CPU (square root) | 58.078% | 99.046% |
| VSJet fast CPU | 99.389% | 74.194% |

At the lower threshold .03, fastest recall is 99.613%
and precision is 99.856%. Restricting the .122 test
to dark pixels where exact's Retinex branch contributes more than .01 gives
fastest recall 98.744% and precision
99.861%. This helps separate Retinex agreement from
the shared source Kirsch response. It still measures agreement with exact,
not labelled real edges or an objective ability to reject noise.

A separate synthetic scene places faint lines (contrast .0006) beside a bright
patch (level .6). Fastest retains 97.876% of the
2,354 exact-selected pixels near the line, with
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

| Candidate | Mean change in candidate-minus-exact mask residual |
|---|---:|
| Our fastest CUDA (2 workers) | 0.01352% |
| Our fast CUDA (downscale 8) | 0.01133% |
| Atom normal CPU | 1.15549% |
| Atom draft CPU (square root) | 1.22163% |
| VSJet fast CPU | 0.80595% |

This is not motion compensated and is not an absolute flicker score. The upload
uses VSView's installed slow.pics extension through the reproducible uploader.
Collections are link-only with no scheduled expiry.

[Native timings](Benchmark-results.md) · [Test reproduction](Reproduce-the-tests.md).
