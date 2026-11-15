# Reference and native backend audit

Current comparison: Python 3.14.10, VapourSynth R81/API 4.3, VSJet 3.0.0.
Pinned verification sources (downloaded by `scripts/fetch_references.py`):

* kagefunc `6b84049e295f35d3f1fe3197d460c7bdb99c0014`
* VapourSynth-Retinex `6bfbdd429159c85075dc4b08e0ac4e706470916b`
* VapourSynth-TCanny `14ac2ceeb59afc7089974d0ae233fe8d0ea183c8`
* VapourSynth R81 `dd11a9da6f8e2bb24ab4bb084d44bf01fb93612a`
* VSJet 3.0.0 `de0ba4a1ab2c1aa1bdf8024e62d551402e2a2cf6`
* vszip 24.0.0 `f6a4c86ee086ad51897796da6e92e6c154e0539c`
* vszipcu 1.2.0 `ca15eff265b81b8f851ca74fed09f1ef22ce7990`
* vszipcl 1.2.0 `6c64f91055eb1925cfe203de7286b69c04cea75f`

[Executed graphs and official binary provenance](results/backend-audit.json).

## Compatibility target

`kagefunc.py:232-253` is authoritative where the prose plan conflicts:

```
Y = get_y(src)
R = retinex.MSRCP(Y, sigma=[50,200,350], upper_thr=.005)
T = tcanny.TCanny(R, mode=1, sigma=1)
M = std.Minimum(T, coordinates=[1,0,1,0,0,1,0,1])
out = min(kirsch(Y) + M, peak)
```

Minimum includes the center and four diagonal neighbors, not all nine pixels.
The original Kirsch implementation has FOUR absolute-response kernels (despite
its eight-direction docstring); it is not the maximum of eight distinct Kirsch
kernels. `std.Convolution(saturate=False)` uses divisor 1 for zero-sum weights:

```
 5  5  5     -3  5  5     -3 -3  5     -3 -3 -3
-3  0 -3     -3  0  5     -3  0  5     -3  0  5
-3 -3 -3     -3 -3 -3     -3 -3  5     -3  5  5
```

3x3 std filters replicate borders. Integer convolution clips each absolute
response to the bit-depth peak; float convolution does not clip. The final mask
is clamped to [0,peak], peak = 2^bits-1 or 1 for float.

## MSRCP

`include/MSR.h`, `include/MSRCP.h`, `source/MSR.cpp`, `source/Gaussian.cpp`:

* `lower_thr=.001`, `upper_thr=.005` (kagefunc override), 4096 histogram bins.
* All calculations use **double**. Gray defaults to `fulls=False, fulld=False`,
  independently of frame range metadata. Normalize the observed frame min/max
  to [0,1]. A constant input uses floor 0 and ceil `peak` instead.
* Each sigma uses the Young/van Vliet third-order recursive Gaussian. Horizontal
  forward and backward passes seed three states with the endpoint; the first
  and last samples are not recomputed. Vertical passes DO recompute endpoints,
  reading clamped previously computed rows. This asymmetry must be preserved.
* q = 3.97156 - 4.14554 sqrt(1-.26891 sigma) for sigma < 2.5;
  otherwise q = .98711 sigma - .96330.
  b0 = 1.57825 + 2.44413 q + 1.4281 q² + .422205 q³;
  b1 = 2.44413 q + 2.85619 q² + 1.26661 q³;
  b2 = -(1.4281 q² + 1.26661 q³); b3 = .422205 q³.
  B = 1-(b1+b2+b3)/b0, B1=b1/b0, B2=b2/b0, B3=b3/b0.
  Recurrence: B*x + B1*y1 + B2*y2 + B3*y3, in that order.
* Initialize product to 1. Multiply `(I/G + 1)` at each positive sigma, or 1
  where G<=0. A zero sigma multiplies by 2. Then log(product)/number_of_scales.
  An entirely zero sigma list bypasses MSRCP.
* Find min/max. If max<=min, use normalized input. Otherwise histogram index
  is trunc(value*gain+offset), gain=4095/(max-min), offset=-min*gain.
  Lower/upper rejected counts are int(N*threshold+.5). Scan each tail until
  cumulative count is STRICTLY greater than rejected count. Threshold values
  are bin_index*(max-min)/4095+min. Rescale and clip to [0,1].
* Convert to integer with trunc(value*(219 << (bits-8)) + (16 << (bits-8))+.5).
  This intermediate limited range is essential for matching the final mask.
* The upstream plugin rejects float. There is no original kagefunc float result
  to match. This implementation extends the same double algorithm to float,
  normalizing observed min/max and returning [0,1] without integer quantization.
  This extension is checked against a separate CPU implementation.
* A collapsed percentile interval is undefined upstream (division by zero).
  The implementation falls back to normalized input to guarantee finite output.

## TCanny mode=1

`TCanny.cpp:gaussianWeights, gaussianBlur, detectEdge, discretizeGM`:

* Default operator is **Prewitt op=1**, not Sobel. kagefunc passes sigma=1,
  overriding TCanny's own sigma=1.5 default. Scale=1.
* Gaussian radius=max(int(sigma*3+.5),1); sigma=0 bypasses blur.
  float weights exp(-k²/(2 sigma²)), normalized using a float sum.
  Vertical then horizontal convolution, mirror-without-repeating-endpoint
  borders. No quantization between blur and gradient.
* Gx=(c3+c6+c9-c1-c4-c7)/2; Gy=(c1+c2+c3-c7-c8-c9)/2;
  magnitude=sqrt(Gx²+Gy²). No suppression or hysteresis.
* Integer input stays in native code values; output is trunc(magnitude+.5)
  capped at the actual bit-depth peak. Float input stays in its native [0,1]
  units and mode=1 output is clamped to [0,1]. **No factor of 255** belongs in
  mode=1; 255 scaling applies only to hysteresis thresholds for mode=0.
* Optional op=2 exposes the plan's Sobel request explicitly, without changing
  the compatible default (weights [1,2,1] perpendicular to [-1,0,1]).

## VSJet 3.0.0 native paths

`vsmasktools/spat_funcs.py:104` builds Retinex using normalized floats,
expressions/logs, Gaussian filters and `vszip.PlaneMinMax`. It does not call the
original Retinex DLL. `vsrgtools/blur.py:251` selects CPU, CUDA (`vszipcu.GaussBlur`)
or OpenCL (`vszipcl.GaussBlur`). Both GPU plugins are version 1.2.0; installed
DLLs match their official PyPI wheels. Final-frame execution proves the selected
nodes actually run.

Fast replaces the largest scale with the frame mean. CPU fast downsamples the
remaining blurs; CUDA/OpenCL fast retain full-resolution blur nodes. For scales
50/200/350, fast GPU has two native blurs and full GPU has three. DRE masks add
one sigma-1 blur. Both native GPU implementations use separable finite Gaussian
taps in float, unlike our double recursive Gaussian. Expression/statistics and
other graph stages still run on CPU, so these are hybrid GPU Retinex pipelines.

DRE soft masks (`brz=0`) use different edge stages, including eight Kirsch
responses. They are useful native comparisons, not interchangeable oracles for
original kagefunc compatibility. `tcanny_retinex` is also a different component.

## Deliberate numerical exceptions

Uniform input bypasses percentile balancing and returns its constant normalized
level, avoiding false contrast from recursive roundoff. Collapsed percentile
intervals use normalized input. Extremely small positive FIR blur values retain
a center weight of one, avoiding underflow-induced 0/0. Tests check these
correctness cases separately from fidelity to upstream. Near-flat nonuniform
inputs and collapsed histogram boundaries remain sensitive to roundoff.

## API and architecture

Read installed VapourSynth4.h and VSHelper4.h. Use VapourSynthPluginInit2,
VSPLUGINAPI::configPlugin/registerFunction, createVideoFilter, VSVideoInfo with
embedded format, fmParallel and rpStrictSpatial. No API 3 types in the GPU DLL.
Preallocated bounded worker pool, each with its own CUDA stream, pinned staging,
device workspace and device-only statistics/histogram. One host-to-device input
and one device-to-host output transfer per frame. GPU intermediates never leave
VRAM. Full resolution uses the reference IIR, not an idealized FFT Gaussian.
Optional pyramid mode averages 4x4 cells, uses sigma/4 at low resolution and
bilinearly upsamples; it is explicitly approximate.
