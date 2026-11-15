// SPDX-License-Identifier: GPL-3.0-or-later
// Retinex equations/boundary behavior adapted from mawen1250's GPLv3 Retinex.
#include "gpu.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace retinex {
namespace {
void check(cudaError_t e) {
    if (e != cudaSuccess) {
        // A handled creation failure (e.g. an invalid device) must not poison a
        // later workspace's launch checks on this same host thread.
        cudaGetLastError();
        throw std::runtime_error(cudaGetErrorString(e));
    }
}
struct Coeff {
    double b, b1, b2, b3;
};
Coeff coefficients(double s) {
    double q = s < 2.5 ? 3.97156 - 4.14554 * std::sqrt(1 - .26891 * s) : .98711 * s - .96330;
    double b0 = 1.57825 + 2.44413 * q + 1.4281 * q * q + .422205 * q * q * q;
    double b1 = 2.44413 * q + 2.85619 * q * q + 1.26661 * q * q * q;
    double b2 = -(1.4281 * q * q + 1.26661 * q * q * q), b3 = .422205 * q * q * q;
    return {1 - (b1 + b2 + b3) / b0, b1 / b0, b2 / b0, b3 / b0};
}
template <class T> struct Device {
    T *p = nullptr;
    T *storage = nullptr;
    void allocate(size_t n) {
        check(cudaMalloc(reinterpret_cast<void **>(&storage), n * sizeof(T)));
        p = storage;
    }
    void alias(T *view) {
        // Non-owning view. The single-stream workspace graph orders all uses
        // of aliased buffers, and the owning Device retains its allocation.
        p = view;
    }
    ~Device() {
        if (storage)
            cudaFree(storage);
    }
};
struct Pinned {
    void *p = nullptr;
    void allocate(size_t bytes) {
        check(cudaMallocHost(&p, bytes));
    }
    ~Pinned() {
        if (p)
            cudaFreeHost(p);
    }
};
struct Stream {
    cudaStream_t s = nullptr;
    void create() {
        check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    }
    ~Stream() {
        if (s) {
            cudaStreamSynchronize(s);
            cudaStreamDestroy(s);
        }
    }
};
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~Graph() {
        if (executable)
            cudaGraphExecDestroy(executable);
        if (graph)
            cudaGraphDestroy(graph);
    }
};
template <class T> __global__ void unpack(const T *in, float *out, int n, float peak) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = static_cast<float>(in[i]);
        out[i] = isfinite(v) ? fminf(peak, fmaxf(0.f, v)) : 0.f;
    }
}
template <class T> __global__ void pack(const float *in, T *out, int n, float peak) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = static_cast<T>(fminf(peak, fmaxf(0.f, in[i])));
}
template <class T> __global__ void unpackRange(const T *in, float *out, double2 *partial, int n, float peak) {
    __shared__ double lo[256], hi[256];
    double a = DBL_MAX, b = -DBL_MAX;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float value = static_cast<float>(in[i]);
        const float v = isfinite(value) ? fminf(peak, fmaxf(0.f, value)) : 0.f;
        out[i] = v;
        a = fmin(a, double(v));
        b = fmax(b, double(v));
    }
    const int t = threadIdx.x;
    lo[t] = a;
    hi[t] = b;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (t < s) {
            lo[t] = fmin(lo[t], lo[t + s]);
            hi[t] = fmax(hi[t], hi[t + s]);
        }
        __syncthreads();
    }
    if (!t)
        partial[blockIdx.x] = make_double2(lo[0], hi[0]);
}
template <class T> __global__ void reduceRange(const T *in, double2 *partial, int n) {
    __shared__ double lo[256], hi[256];
    double a = DBL_MAX, b = -DBL_MAX;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        double v = in[i];
        a = fmin(a, v);
        b = fmax(b, v);
    }
    int t = threadIdx.x;
    lo[t] = a;
    hi[t] = b;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (t < s) {
            lo[t] = fmin(lo[t], lo[t + s]);
            hi[t] = fmax(hi[t], hi[t + s]);
        }
        __syncthreads();
    }
    if (!t)
        partial[blockIdx.x] = make_double2(lo[0], hi[0]);
}
__global__ void finishRange(const double2 *in, double *stats, int n, double peak, bool inputRange) {
    // Stats layout: observed range [0,1], percentile bounds [2,3], balance
    // gain/offset [4,5], histogram gain/offset [6,7], normalization low/gain [8,9],
    // uniform-input flag [10]. Flat input must not amplify recurrence roundoff.
    __shared__ double lo[256], hi[256];
    double a = DBL_MAX, b = -DBL_MAX;
    int t = threadIdx.x;
    for (int i = t; i < n; i += 256) {
        a = fmin(a, in[i].x);
        b = fmax(b, in[i].y);
    }
    lo[t] = a;
    hi[t] = b;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (t < s) {
            lo[t] = fmin(lo[t], lo[t + s]);
            hi[t] = fmax(hi[t], hi[t + s]);
        }
        __syncthreads();
    }
    if (!t) {
        stats[0] = lo[0];
        stats[1] = hi[0];
        // These expressions used to run independently at every pixel. Keep the
        // same double operations, but evaluate frame-wide constants just once.
        const double gain = hi[0] > lo[0] ? 1.0 / (hi[0] - lo[0]) : 0;
        stats[4] = gain;
        stats[5] = -lo[0] * gain;
        const double histGain = hi[0] > lo[0] ? 4095.0 / (hi[0] - lo[0]) : 0;
        stats[6] = histGain;
        stats[7] = -lo[0] * histGain;
        // Retain the original input normalization through the product's range
        // reduction: later kernels reconstruct normalized pixels on demand.
        if (inputRange) {
            const double normLo = hi[0] <= lo[0] ? 0 : lo[0];
            const double normHi = hi[0] <= lo[0] ? peak : hi[0];
            stats[8] = normLo;
            stats[9] = 1.0 / (normHi - normLo);
            stats[10] = hi[0] <= lo[0] ? 1.0 : 0.0;
        }
    }
}
__device__ double normalized(float value, const double *stats) {
    return (double(value) - stats[8]) * stats[9];
}
template <bool Normalize = false, class T>
__global__ void transpose(const T *in, double *out, int w, int h, const Coeff *coeff = nullptr, int first = 0,
                          const double *stats = nullptr) {
    const int scale = blockIdx.z;
    if (coeff && coeff[first + scale].b == 0)
        return;
    in += size_t(scale) * w * h;
    out += size_t(scale) * w * h;
    __shared__ double tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8)
        if (x < w && y + j < h) {
            if constexpr (Normalize)
                tile[threadIdx.y + j][threadIdx.x] = normalized(in[(y + j) * w + x], stats);
            else
                tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * w + x];
        }
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8)
        if (x < h && y + j < w)
            out[(y + j) * h + x] = tile[threadIdx.x][threadIdx.y + j];
}
__device__ double recur(double x, double a, double b, double c, Coeff k) {
    return k.b * x + k.b1 * a + k.b2 * b + k.b3 * c;
}
// One independent line per thread. Transposition makes both axes coalesced.
__global__ void recursive(const double *input, double *v, int lines, int length, const Coeff *coeff, int first,
                          bool vertical, bool commonInput) {
    const int scale = blockIdx.y;
    const Coeff k = coeff[first + scale];
    if (k.b == 0)
        return;
    const size_t offset = size_t(scale) * lines * length;
    if (!commonInput)
        input += offset;
    v += offset;
    int line = blockIdx.x * blockDim.x + threadIdx.x;
    if (line >= lines)
        return;
    double a = input[line];
    if (vertical) {
        a = recur(a, a, a, a, k);
    }
    v[line] = a;
    double b = a, c = a;
    for (int j = 1; j < length; ++j) {
        int i = j * lines + line;
        double z = recur(input[i], a, b, c, k);
        c = b;
        b = a;
        a = z;
        v[i] = z;
    }
    int end = (length - 1) * lines + line;
    a = v[end];
    if (vertical) {
        a = recur(a, a, a, a, k);
        v[end] = a;
    }
    b = c = a;
    for (int j = length - 2; j >= 0; --j) {
        int i = j * lines + line;
        double z = recur(v[i], a, b, c, k);
        c = b;
        b = a;
        a = z;
        v[i] = z;
    }
}
// One warp owns 32 horizontal lines. Retain the identical forward/backward
// recurrence, but transpose each completed 32x32 reverse-pass tile on-chip.
// The backward result no longer needs to be written, read, and transposed in
// a separate full-frame pass. Launch with exactly 32 threads per block.
__global__ void recursiveHorizontalTranspose(const double *input, double *forward, double *out, int lines,
                                             int length, const Coeff *coeff, int first) {
    const Coeff k = coeff[first + blockIdx.y];
    if (k.b == 0)
        return;
    const size_t offset = size_t(blockIdx.y) * lines * length;
    forward += offset;
    out += offset;
    __shared__ double tile[32][33];
    const int lane = threadIdx.x, lineBase = blockIdx.x * 32, line = lineBase + lane;
    double a = 0, b = 0, c = 0;
    if (line < lines) {
        a = input[line];
        forward[line] = a;
        b = c = a;
        for (int j = 1; j < length; ++j) {
            const int i = j * lines + line;
            const double z = recur(input[i], a, b, c, k);
            c = b;
            b = a;
            a = z;
            forward[i] = z;
        }
        b = c = a;
    }
    for (int base = ((length - 1) / 32) * 32; base >= 0; base -= 32) {
        if (line < lines) {
            for (int j = min(length - 1, base + 31); j >= base; --j) {
                if (j != length - 1) {
                    const double z = recur(forward[j * lines + line], a, b, c, k);
                    c = b;
                    b = a;
                    a = z;
                }
                tile[lane][j - base] = a;
            }
        }
        __syncwarp();
        const int x = base + lane;
        if (x < length)
            for (int row = 0; row < min(32, lines - lineBase); ++row)
                out[(lineBase + row) * length + x] = tile[row][lane];
        __syncwarp();
    }
}
__global__ void logarithm(double *v, int n, int scales) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        v[i] = log(v[i]) / scales;
}
__global__ void histogram(const double *v, const double *range, unsigned *hist, int n) {
    __shared__ unsigned local[4096];
    for (int i = threadIdx.x; i < 4096; i += 256)
        local[i] = 0;
    __syncthreads();
    double gain = range[6], offset = range[7];
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += 256 * gridDim.x) {
        int bin = max(0, min(4095, int(v[i] * gain + offset)));
        atomicAdd(local + bin, 1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 4096; i += 256)
        if (local[i])
            atomicAdd(hist + i, local[i]);
}
__global__ void percentiles(unsigned *hist, double *range, int n, int lowCount, int highCount) {
    __shared__ unsigned warpTotals[8];
    __shared__ unsigned lower, upper;
    int t = threadIdx.x;
    unsigned counts[16];
    for (int j = 0; j < 16; ++j) {
        counts[j] = hist[16 * t + j];
        // Each bin has one owner; leave this workspace's histogram ready for
        // its next graph replay after retaining the current counts in registers.
        hist[16 * t + j] = 0;
    }
    if (!t) {
        lower = 4095;
        upper = 4095;
    }
    __syncthreads();
    // Integer prefix sums are exact regardless of the scan tree. Each thread
    // owns 16 consecutive bins and contributes at most two threshold atomics.
    unsigned threadTotal = 0;
    for (unsigned count : counts)
        threadTotal += count;
    unsigned inclusive = threadTotal;
    const int lane = t % 32, warp = t / 32;
    for (int step = 1; step < 32; step *= 2) {
        const unsigned previous = __shfl_up_sync(0xffffffff, inclusive, step);
        if (lane >= step)
            inclusive += previous;
    }
    if (lane == 31)
        warpTotals[warp] = inclusive;
    __syncthreads();
    unsigned prefix = inclusive - threadTotal;
    for (int i = 0; i < warp; ++i)
        prefix += warpTotals[i];
    unsigned firstLower = 4095, firstUpper = 4095;
    for (int j = 0; j < 16; ++j) {
        prefix += counts[j];
        if (prefix > unsigned(lowCount))
            firstLower = min(firstLower, unsigned(16 * t + j));
        if (prefix >= unsigned(n - highCount))
            firstUpper = min(firstUpper, unsigned(16 * t + j));
    }
    atomicMin(&lower, firstLower);
    atomicMin(&upper, firstUpper);
    __syncthreads();
    if (!t) {
        double gain = (range[1] - range[0]) / 4095, offset = range[0];
        range[2] = lower * gain + offset;
        range[3] = upper * gain + offset;
        const double balanceGain = range[3] > range[2] ? 1.0 / (range[3] - range[2]) : 0;
        range[4] = balanceGain;
        range[5] = -range[2] * balanceGain;
    }
}
template <class T>
__global__ void balance(const double *product, const float *original, const double *range, T *out, int n, int bits,
                        bool floating, bool thresholds, float peak) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double lo = range[thresholds ? 2 : 0], hi = range[thresholds ? 3 : 1];
        double gain = range[4], offset = range[5];
        double v = range[10] == 0 && hi > lo ? fmin(1.0, fmax(0.0, product[i] * gain + offset))
                                            : normalized(original[i], range);
        const float pixel = floating ? float(v) : float(int(v * (219 << (bits - 8)) + (16 << (bits - 8)) + .5));
        out[i] = static_cast<T>(fminf(peak, fmaxf(0.f, pixel)));
    }
}
__device__ __constant__ int kirschWeights[36] = {5,  5,  5, -3, 0,  -3, -3, -3, -3, -3, 5, 5,
                                                 -3, 0,  5, -3, -3, -3, -3, -3, 5,  -3, 0, 5,
                                                 -3, -3, 5, -3, -3, -3, -3, 0,  5,  -3, 5, 5};
__device__ float kirschValue(const float *in, int i, int w, int h, float peak) {
    int x = i % w, y = i / w;
    float best = 0;
    for (int k = 0; k < 4; ++k) {
        float sum = 0;
        for (int j = 0; j < 9; ++j) {
            int xx = max(0, min(w - 1, x + j % 3 - 1)), yy = max(0, min(h - 1, y + j / 3 - 1));
            sum += in[yy * w + xx] * kirschWeights[k * 9 + j];
        }
        best = fmaxf(best, fabsf(sum));
    }
    return fminf(peak, best);
}
template <class T> __global__ void kirschEdges(const float *in, T *out, int w, int h, float peak) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w * h)
        out[i] = static_cast<T>(fminf(peak, fmaxf(0.f, kirschValue(in, i, w, h, peak))));
}
__device__ int mirror(int x, int n) {
    if (n <= 1)
        return 0;
    int period = 2 * (n - 1);
    x %= period;
    if (x < 0)
        x += period;
    return x < n ? x : period - x;
}
__global__ void gaussianFIR(const float *in, float *out, const float *weights, int radius, int w, int h,
                            bool vertical) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * h)
        return;
    int x = i % w, y = i / w;
    float sum = 0;
    for (int j = -radius; j <= radius; ++j) {
        int at = vertical ? mirror(y + j, h) * w + x : y * w + mirror(x + j, w);
        sum += in[at] * weights[j + radius];
    }
    out[i] = sum;
}
template <class T>
__global__ void tcannyGradient(const float *in, T *out, int w, int h, int op, float peak, bool floating) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * h)
        return;
    int x = i % w, y = i / w, l = mirror(x - 1, w), r = mirror(x + 1, w), u = mirror(y - 1, h) * w,
        d = mirror(y + 1, h) * w;
    float c1 = in[u + l], c2 = in[u + x], c3 = in[u + r], c4 = in[y * w + l], c6 = in[y * w + r], c7 = in[d + l],
          c8 = in[d + x], c9 = in[d + r];
    float gx, gy;
    if (op == 1) {
        gx = (c3 + c6 + c9 - c1 - c4 - c7) / 2.f;
        gy = (c1 + c2 + c3 - c7 - c8 - c9) / 2.f;
    } else {
        gx = c3 + 2.f * c6 + c9 - c1 - 2.f * c4 - c7;
        gy = c1 + 2.f * c2 + c3 - c7 - 2.f * c8 - c9;
    }
    float v = sqrtf(gx * gx + gy * gy);
    const float pixel = fminf(peak, floating ? v : floorf(v + .5f));
    out[i] = static_cast<T>(fminf(peak, fmaxf(0.f, pixel)));
}
template <class T>
__global__ void finishMask(const float *original, const float *gradient, T *out, int w, int h, float peak) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * h)
        return;
    int x = i % w, y = i / w;
    float v = gradient[i];
    for (int dy = -1; dy <= 1; dy += 2)
        for (int dx = -1; dx <= 1; dx += 2)
            v = fminf(v, gradient[max(0, min(h - 1, y + dy)) * w + max(0, min(w - 1, x + dx))]);
    // Preserve the Kirsch clamp, combination clamp, and final packing clamp.
    // No intermediate image is needed for these independent per-pixel steps.
    const float combined = fminf(peak, kirschValue(original, i, w, h, peak) + v);
    out[i] = static_cast<T>(fminf(peak, fmaxf(0.f, combined)));
}
__global__ void downsample(const float *in, double *out, int w, int h, int sw, int sh, const double *stats) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sw * sh)
        return;
    int x = (i % sw) * 4, y = (i / sw) * 4, count = 0;
    double sum = 0;
    for (int dy = 0; dy < 4 && y + dy < h; ++dy)
        for (int dx = 0; dx < 4 && x + dx < w; ++dx) {
            sum += normalized(in[(y + dy) * w + x + dx], stats);
            ++count;
        }
    out[i] = sum / count;
}
__device__ double samplePyramid(const double *blur, int i, int w, int sw, int sh) {
    double x = fmax(0., fmin(double(sw - 1), ((i % w) + .5) / 4 - .5));
    double y = fmax(0., fmin(double(sh - 1), ((i / w) + .5) / 4 - .5));
    int x0 = int(x), y0 = int(y), x1 = min(sw - 1, x0 + 1), y1 = min(sh - 1, y0 + 1);
    x -= x0;
    y -= y0;
    double a = blur[y0 * sw + x0] * (1 - x) + blur[y0 * sw + x1] * x;
    double b = blur[y1 * sw + x0] * (1 - x) + blur[y1 * sw + x1] * x;
    return a * (1 - y) + b * y;
}
__global__ void multiplyScales(const float *original, const double *blur, double *product, int w, int h, int sw,
                               int sh, const Coeff *coeff, int first, int count, bool pyramid,
                               const double *stats) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * h)
        return;
    const double norm = normalized(original[i], stats);
    double value = first == 0 ? 1.0 : product[i];
    // Scales run concurrently, but their product is accumulated in the original
    // caller-specified order with the same rounding at every multiplication.
    for (int scale = 0; scale < count; ++scale) {
        if (coeff[first + scale].b == 0) {
            value *= 2.0;
        } else {
            const double *plane = blur + size_t(scale) * sw * sh;
            const double g = pyramid ? samplePyramid(plane, i, w, sw, sh) : plane[i];
            value *= g <= 0 ? 1.0 : norm / g + 1.0;
        }
    }
    product[i] = value;
}
} // namespace
struct Workspace::Impl {
    Parameters p;
    int n, blocks, reduceBlocks, bytes, radius = 0;
    float peak;
    Stream stream;
    Pinned hostIn, hostOut;
    Device<unsigned char> raw;
    Device<float> original, result, temp, blur, gradient, weights;
    Device<double> product, gauss, transposed, horizontal, range, small;
    Device<double2> partial;
    Device<unsigned> hist;
    std::vector<Coeff> coeff;
    Device<Coeff> deviceCoeff;
    int batchSize = 1;
    Graph graph;
    explicit Impl(const Parameters &params)
        : p(params), n(p.width * p.height), blocks((n + 255) / 256), reduceBlocks(std::min(blocks, 1024)),
          bytes(p.floating    ? 4
                : p.bits == 8 ? 1
                              : 2),
          peak(p.floating ? 1.f : float((1 << p.bits) - 1)) {
        check(cudaSetDevice(p.device));
        stream.create();
        hostIn.allocate(size_t(n) * bytes);
        hostOut.allocate(size_t(n) * bytes);
        raw.allocate(size_t(n) * bytes);
        original.allocate(n);
        if (p.stage == Stage::MSRCP || p.stage == Stage::Mask) {
            product.allocate(n);
            batchSize = std::min(3, int(p.sigma.size()));
            const size_t blurPixels = p.fast ? size_t((p.width + 3) / 4) * ((p.height + 3) / 4) : size_t(n);
            gauss.allocate(blurPixels * batchSize);
            horizontal.allocate(blurPixels * batchSize);
            // A single scale batch finishes consuming the transposed input
            // before writing product. Later batches need an independent copy.
            if (p.sigma.size() <= 3)
                transposed.alias(product.p);
            else
                transposed.allocate(blurPixels);
            // Horizontal scratch is dead before balance produces the result.
            // Full-resolution scratch holds at least two float-sized images.
            if (p.stage == Stage::Mask && !p.fast)
                result.alias(reinterpret_cast<float *>(horizontal.p));
            else if (p.stage == Stage::Mask)
                result.allocate(n);
            range.allocate(11);
            partial.allocate(reduceBlocks);
            hist.allocate(4096);
            check(cudaMemsetAsync(hist.p, 0, 4096 * sizeof(unsigned), stream.s));
            for (double s : p.sigma)
                // Only a caller-supplied zero is inactive. Preserve the original
                // coefficient calculation even if a tiny positive sigma/4 underflows.
                coeff.push_back(s == 0 ? Coeff{} : coefficients(p.fast ? s / 4 : s));
            deviceCoeff.allocate(coeff.size());
            check(cudaMemcpy(deviceCoeff.p, coeff.data(), coeff.size() * sizeof(Coeff), cudaMemcpyHostToDevice));
            if (p.fast)
                small.allocate(blurPixels);
        }
        if (p.stage == Stage::Mask) {
            // Gaussian scale outputs are dead before edge extraction begins.
            if (!p.fast)
                gradient.alias(reinterpret_cast<float *>(gauss.p));
            else
                gradient.allocate(n);
        }
        if (p.blur > 0 && (p.stage == Stage::TCanny || p.stage == Stage::Mask)) {
            if (p.stage == Stage::Mask) {
                // Product is dead after balance. Its two float-sized halves
                // hold the vertical and horizontal FIR outputs without overlap.
                temp.alias(reinterpret_cast<float *>(product.p));
                blur.alias(reinterpret_cast<float *>(product.p) + n);
            } else {
                temp.allocate(n);
                blur.allocate(n);
            }
            radius = std::max(int(p.blur * 3.f + .5f), 1);
            std::vector<float> taps(2 * radius + 1);
            float sum = 0;
            for (int k = -radius; k <= radius; ++k) {
                // The center is exactly one, including when blur^2 underflows.
                float v = k == 0 ? 1.f : std::exp(-(k * k) / (2.f * p.blur * p.blur));
                taps[k + radius] = v;
                sum += v;
            }
            for (float &v : taps)
                v /= sum;
            weights.allocate(taps.size());
            check(cudaMemcpy(weights.p, taps.data(), taps.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
        // Stable per-workspace buffers allow the entire unchanged operation sequence,
        // including transfers, to be captured once instead of resubmitted per frame.
        check(cudaStreamBeginCapture(stream.s, cudaStreamCaptureModeThreadLocal));
        try {
            enqueue();
            check(cudaStreamEndCapture(stream.s, &graph.graph));
        } catch (...) {
            cudaGraph_t aborted = nullptr;
            cudaStreamEndCapture(stream.s, &aborted);
            if (aborted)
                cudaGraphDestroy(aborted);
            throw;
        }
        check(cudaGraphInstantiateWithFlags(&graph.executable, graph.graph, 0));
        check(cudaGraphUpload(graph.executable, stream.s));
        check(cudaStreamSynchronize(stream.s));
    }
    template <class T> void rangeOf(const T *v, bool inputRange = false) {
        reduceRange<<<reduceBlocks, 256, 0, stream.s>>>(v, partial.p, n);
        finishRange<<<1, 256, 0, stream.s>>>(partial.p, range.p, reduceBlocks, peak, inputRange);
    }
    template <class T> void msr(T *output) {
        if (std::all_of(p.sigma.begin(), p.sigma.end(), [](double s) { return s == 0; })) {
            pack<<<blocks, 256, 0, stream.s>>>(original.p, output, n, peak);
            return;
        }
        int w = p.fast ? (p.width + 3) / 4 : p.width, h = p.fast ? (p.height + 3) / 4 : p.height;
        if (p.fast) {
            downsample<<<(w * h + 255) / 256, 256, 0, stream.s>>>(original.p, small.p, p.width, p.height, w, h,
                                                                  range.p);
            transpose<<<dim3((w + 31) / 32, (h + 31) / 32), dim3(32, 8), 0, stream.s>>>(small.p, transposed.p, w,
                                                                                        h);
        } else {
            transpose<true><<<dim3((w + 31) / 32, (h + 31) / 32), dim3(32, 8), 0, stream.s>>>(
                original.p, transposed.p, w, h, nullptr, 0, range.p);
        }
        for (int first = 0; first < int(p.sigma.size()); first += batchSize) {
            int count = std::min(batchSize, int(p.sigma.size()) - first);
            // The on-chip transpose pays off for large planes; at SD/HD the
            // simpler recurrence and separate transpose measured faster.
            if (size_t(w) * h >= size_t(3840) * 2160) {
                recursiveHorizontalTranspose<<<dim3((h + 31) / 32, count), 32, 0, stream.s>>>(
                    transposed.p, horizontal.p, gauss.p, h, w, deviceCoeff.p, first);
            } else {
                recursive<<<dim3((h + 31) / 32, count), 32, 0, stream.s>>>(transposed.p, horizontal.p, h, w,
                                                                           deviceCoeff.p, first, false, true);
                transpose<<<dim3((h + 31) / 32, (w + 31) / 32, count), dim3(32, 8), 0, stream.s>>>(
                    horizontal.p, gauss.p, h, w, deviceCoeff.p, first);
            }
            recursive<<<dim3((w + 31) / 32, count), 32, 0, stream.s>>>(gauss.p, gauss.p, w, h, deviceCoeff.p,
                                                                       first, true, false);
            multiplyScales<<<blocks, 256, 0, stream.s>>>(original.p, gauss.p, product.p, p.width, p.height, w, h,
                                                         deviceCoeff.p, first, count, p.fast, range.p);
        }
        logarithm<<<blocks, 256, 0, stream.s>>>(product.p, n, int(p.sigma.size()));
        rangeOf(product.p);
        bool thresholds = p.lower > 0 || p.upper > 0;
        if (thresholds) {
            histogram<<<std::min(blocks, 128), 256, 0, stream.s>>>(product.p, range.p, hist.p, n);
            percentiles<<<1, 256, 0, stream.s>>>(hist.p, range.p, n, int(n * p.lower + .5), int(n * p.upper + .5));
        }
        balance<<<blocks, 256, 0, stream.s>>>(product.p, original.p, range.p, output, n, p.bits, p.floating,
                                              thresholds, peak);
    }
    template <class T> void tcanny(const float *input, T *output) {
        if (radius) {
            gaussianFIR<<<blocks, 256, 0, stream.s>>>(input, temp.p, weights.p, radius, p.width, p.height, true);
            gaussianFIR<<<blocks, 256, 0, stream.s>>>(temp.p, blur.p, weights.p, radius, p.width, p.height, false);
            input = blur.p;
        }
        tcannyGradient<<<blocks, 256, 0, stream.s>>>(input, output, p.width, p.height, p.op, peak, p.floating);
    }
    template <class T> void unpackInput() {
        const T *input = reinterpret_cast<const T *>(raw.p);
        if ((p.stage == Stage::MSRCP || p.stage == Stage::Mask) &&
            std::any_of(p.sigma.begin(), p.sigma.end(), [](double s) { return s != 0; })) {
            unpackRange<<<reduceBlocks, 256, 0, stream.s>>>(input, original.p, partial.p, n, peak);
            finishRange<<<1, 256, 0, stream.s>>>(partial.p, range.p, reduceBlocks, peak, true);
        } else {
            unpack<<<blocks, 256, 0, stream.s>>>(input, original.p, n, peak);
        }
    }
    template <class T> void render() {
        T *output = reinterpret_cast<T *>(raw.p);
        if (p.stage == Stage::MSRCP)
            msr(output);
        else if (p.stage == Stage::Kirsch)
            kirschEdges<<<blocks, 256, 0, stream.s>>>(original.p, output, p.width, p.height, peak);
        else if (p.stage == Stage::TCanny)
            tcanny(original.p, output);
        else {
            msr(result.p);
            tcanny(result.p, gradient.p);
            finishMask<<<blocks, 256, 0, stream.s>>>(original.p, gradient.p, output, p.width, p.height, peak);
        }
    }
    void enqueue() {
        check(cudaMemcpyAsync(raw.p, hostIn.p, size_t(n) * bytes, cudaMemcpyHostToDevice, stream.s));
        if (p.floating) {
            unpackInput<float>();
            render<float>();
        } else if (bytes == 1) {
            unpackInput<uint8_t>();
            render<uint8_t>();
        } else {
            unpackInput<uint16_t>();
            render<uint16_t>();
        }
        check(cudaGetLastError());
        check(cudaMemcpyAsync(hostOut.p, raw.p, size_t(n) * bytes, cudaMemcpyDeviceToHost, stream.s));
    }
    void process(const void *src, ptrdiff_t srcStride, void *dst, ptrdiff_t dstStride) {
        check(cudaSetDevice(p.device));
        const size_t rowBytes = size_t(p.width) * bytes;
        if (srcStride == ptrdiff_t(rowBytes))
            std::memcpy(hostIn.p, src, size_t(n) * bytes);
        else
            for (int y = 0; y < p.height; ++y)
                std::memcpy(static_cast<char *>(hostIn.p) + size_t(y) * rowBytes,
                            static_cast<const char *>(src) + y * srcStride, rowBytes);
        check(cudaGraphLaunch(graph.executable, stream.s));
        check(cudaStreamSynchronize(stream.s));
        if (dstStride == ptrdiff_t(rowBytes))
            std::memcpy(dst, hostOut.p, size_t(n) * bytes);
        else
            for (int y = 0; y < p.height; ++y)
                std::memcpy(static_cast<char *>(dst) + y * dstStride,
                            static_cast<const char *>(hostOut.p) + size_t(y) * rowBytes, rowBytes);
    }
    ~Impl() {
        cudaSetDevice(p.device);
        if (stream.s)
            cudaStreamSynchronize(stream.s);
    }
};
Workspace::Workspace(const Parameters &p) : impl(std::make_unique<Impl>(p)) {}
Workspace::~Workspace() = default;
void Workspace::process(const void *src, ptrdiff_t ss, void *dst, ptrdiff_t ds) {
    try {
        impl->process(src, ss, dst, ds);
    } catch (...) {
        cudaStreamSynchronize(impl->stream.s);
        throw;
    }
}
} // namespace retinex
