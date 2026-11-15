// SPDX-License-Identifier: GPL-3.0-or-later
#include <cuda_runtime.h>
#include "engine.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
namespace rf {
namespace {
void check(cudaError_t e) {
    if (e != cudaSuccess) {
        cudaGetLastError();
        throw std::runtime_error(cudaGetErrorString(e));
    }
}
struct DeviceScope {
    int old = -1;
    explicit DeviceScope(int device) {
        if (cudaGetDevice(&old) != cudaSuccess) {
            old = -1;
            cudaGetLastError();
        }
        check(cudaSetDevice(device));
    }
    ~DeviceScope() {
        if (old >= 0)
            cudaSetDevice(old);
    }
};
template <class T> __global__ void rangeInput(const T *in, float2 *part, const Config *cfg) {
    const Config &c = *cfg;
    __shared__ float lo[256], hi[256];
    int t = threadIdx.x;
    float a = c.peak, b = 0;
    for (int i = blockIdx.x * 256 + t; i < c.w * c.h; i += 256 * gridDim.x) {
        float v = clean(float(in[i]), c.peak);
        a = fminf(a, v);
        b = fmaxf(b, v);
    }
    lo[t] = a;
    hi[t] = b;
    __syncthreads();
    for (int k = 128; k; k /= 2) {
        if (t < k) {
            lo[t] = fminf(lo[t], lo[t + k]);
            hi[t] = fmaxf(hi[t], hi[t + k]);
        }
        __syncthreads();
    }
    if (!t)
        part[blockIdx.x] = make_float2(lo[0], hi[0]);
}
__global__ void combineIllumination(double *planes, const Config *cfg) {
    const Config &c = *cfg;
    int i = blockIdx.x * 256 + threadIdx.x;
    size_t count = size_t(c.sw) * c.sh;
    if (i < count)
        planes[i] = effectiveIllumination(planes, i, count, c);
}
template <class T>
__global__ void quickScoreRange(const T *src, const double *planes, float *scores, float2 *part, const Config *cfg,
                                const Stats *s) {
    const Config &c = *cfg;
    __shared__ float lo[256], hi[256];
    int t = threadIdx.x;
    float a = INFINITY, b = 0;
    for (int i = blockIdx.x * 256 + t; i < c.sw * c.sh; i += 256 * gridDim.x) {
        int x = clampi((i % c.sw) * c.scale + c.scale / 2, c.w),
            y = clampi((i / c.sw) * c.scale + c.scale / 2, c.h);
        float value = quickScore(read(src, c.w, x, y, c), x, y, planes, c, *s);
        scores[i] = value;
        a = fminf(a, value);
        b = fmaxf(b, value);
    }
    lo[t] = a;
    hi[t] = b;
    __syncthreads();
    for (int k = 128; k; k /= 2) {
        if (t < k) {
            lo[t] = fminf(lo[t], lo[t + k]);
            hi[t] = fmaxf(hi[t], hi[t + k]);
        }
        __syncthreads();
    }
    if (!t)
        part[blockIdx.x] = make_float2(lo[0], hi[0]);
}
template <class T>
__global__ void balanceQuick(const T *in, T *out, const double *planes, const Config *cfg, const Stats *s) {
    const Config &c = *cfg;
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < c.w * c.h) {
        float value = clean(float(in[i]), c.peak);
        out[i] =
            T(balanced(value, s->balanceGain > 0 ? quickScore(value, i % c.w, i / c.w, planes, c, *s) : 0, c, *s));
    }
}
__global__ void finishRange(const float2 *part, int count, Stats *stats, const Config *cfg, bool input) {
    if (threadIdx.x)
        return;
    float lo = INFINITY, hi = -INFINITY;
    for (int i = 0; i < count; ++i) {
        lo = fminf(lo, part[i].x);
        hi = fmaxf(hi, part[i].y);
    }
    if (input) {
        stats->uniform = hi <= lo;
        stats->inputLo = hi > lo ? lo : 0;
        stats->inputSpan = hi > lo ? hi - lo : cfg->peak;
        stats->inputGain = 1.f / stats->inputSpan;
        if (!isfinite(stats->inputGain))
            stats->inputGain = 0;
    } else {
        stats->lo = lo;
        stats->hi = hi;
    }
}
template <class T> __global__ void downsample(const T *src, double *low, const Config *cfg, const Stats *s) {
    const Config &c = *cfg;
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= c.sw * c.sh)
        return;
    int x = i % c.sw, y = i / c.sw, count = 0;
    double sum = 0;
    for (int dy = 0; dy < c.scale && y * c.scale + dy < c.h; ++dy)
        for (int dx = 0; dx < c.scale && x * c.scale + dx < c.w; ++dx) {
            sum += normalized(read(src, c.w, x * c.scale + dx, y * c.scale + dy, c), *s);
            ++count;
        }
    low[i] = sum / count;
}
__global__ void horizontal(const double *low, double *temp, const Config *cfg) {
    const Config &c = *cfg;
    int y = blockIdx.x * 64 + threadIdx.x, k = blockIdx.y;
    if (y >= c.sh || c.coeff[k].b == 0)
        return;
    line(low + y * c.sw, temp + size_t(k) * c.sw * c.sh + y * c.sw, c.sw, 1, c.coeff[k], false);
}
__global__ void vertical(const double *temp, double *planes, const Config *cfg) {
    const Config &c = *cfg;
    int x = blockIdx.x * 64 + threadIdx.x, k = blockIdx.y;
    if (x >= c.sw || c.coeff[k].b == 0)
        return;
    size_t base = size_t(k) * c.sw * c.sh + x;
    line(temp + base, planes + base, c.sh, c.sw, c.coeff[k], true);
}
template <class T>
__global__ void scoreRange(const T *src, const double *planes, float *scores, float2 *part, const Config *cfg,
                           const Stats *s) {
    const Config &c = *cfg;
    __shared__ float lo[256], hi[256];
    int t = threadIdx.x;
    float a = INFINITY, b = 0;
    for (int i = blockIdx.x * 256 + t; i < c.w * c.h; i += 256 * gridDim.x) {
        float v = score(clean(float(src[i]), c.peak), i % c.w, i / c.w, planes, c, *s);
        scores[i] = v;
        a = fminf(a, v);
        b = fmaxf(b, v);
    }
    lo[t] = a;
    hi[t] = b;
    __syncthreads();
    for (int k = 128; k; k /= 2) {
        if (t < k) {
            lo[t] = fminf(lo[t], lo[t + k]);
            hi[t] = fmaxf(hi[t], hi[t + k]);
        }
        __syncthreads();
    }
    if (!t)
        part[blockIdx.x] = make_float2(lo[0], hi[0]);
}
__global__ void histogram(const float *scores, unsigned *hist, int n, const Stats *s) {
    __shared__ unsigned local[4096];
    int t = threadIdx.x;
    for (int i = t; i < 4096; i += 256)
        local[i] = 0;
    __syncthreads();
    for (int i = blockIdx.x * 256 + t; i < n; i += 256 * gridDim.x)
        atomicAdd(local + bin(scores[i], *s), 1);
    __syncthreads();
    for (int i = t; i < 4096; i += 256)
        if (local[i])
            atomicAdd(hist + i, local[i]);
}
__global__ void bounds(const unsigned *hist, Stats *s, const Config *c) {
    if (!threadIdx.x)
        percentile(hist, *s, *c);
}
template <class T>
__global__ void balance(const T *src, T *out, const float *scores, const Config *cfg, const Stats *s) {
    const Config &c = *cfg;
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < c.w * c.h)
        out[i] = T(balanced(clean(float(src[i]), c.peak), c.bypass ? 0 : scores[i], c, *s));
}
template <class T> __global__ void verticalMask(const T *in, float *out, const Config *cfg) {
    const Config &c = *cfg;
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= c.w * c.h)
        return;
    int x = i % c.w, y = i / c.w;
    float sum = 0;
    for (int j = -c.radius; j <= c.radius; ++j)
        sum += float(in[mirror(y + j, c.h) * c.w + x]) * c.weights[j + c.radius];
    out[i] = sum;
}
template <class T> __global__ void finishMask(const float *input, const T *original, T *out, const Config *cfg) {
    const Config &c = *cfg;
    constexpr int TW = 32, TH = 8, BW = TW + 4, BH = TH + 4, GW = TW + 2, GH = TH + 2;
    __shared__ float blurred[BW * BH], grad[GW * GH];
    int t = threadIdx.y * TW + threadIdx.x, bx = blockIdx.x * TW, by = blockIdx.y * TH;
    for (int i = t; i < BW * BH; i += TW * TH) {
        int x = mirror(bx + i % BW - 2, c.w), y = mirror(by + i / BW - 2, c.h);
        float sum = 0;
        for (int j = -c.radius; j <= c.radius; ++j)
            sum += input[y * c.w + mirror(x + j, c.w)] * c.weights[j + c.radius];
        blurred[i] = sum;
    }
    __syncthreads();
    for (int i = t; i < GW * GH; i += TW * TH) {
        int x = clampi(bx + i % GW - 1, c.w), y = clampi(by + i / GW - 1, c.h),
            at = (y - by + 2) * BW + x - bx + 2;
        grad[i] = gradient(blurred[at - BW - 1], blurred[at - BW], blurred[at - BW + 1], blurred[at - 1],
                           blurred[at + 1], blurred[at + BW - 1], blurred[at + BW], blurred[at + BW + 1], c);
    }
    __syncthreads();
    int x = bx + threadIdx.x, y = by + threadIdx.y;
    if (x < c.w && y < c.h) {
        int at = (threadIdx.y + 1) * GW + threadIdx.x + 1;
        float v = grad[at];
        for (int dy = -1; dy <= 1; dy += 2)
            for (int dx = -1; dx <= 1; dx += 2)
                v = fminf(v, grad[at + dy * GW + dx]);
        out[y * c.w + x] = T(clip(v + kirsch(original, c.w, x, y, c), c.peak));
    }
}
struct CudaGpu final : Gpu {
    Config c;
    size_t rawBytes = 0, allocated = 0, pinned = 0;
    int blocks = 0, reductions = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t done = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    void *memory = nullptr, *host = nullptr;
    unsigned char *rawIn = nullptr, *rawOut = nullptr;
    Config *cfg = nullptr;
    Stats *stats = nullptr;
    float2 *partial = nullptr;
    unsigned *hist = nullptr;
    double *low = nullptr, *temp = nullptr, *planes = nullptr;
    float *scores = nullptr;
    explicit CudaGpu(const Config &v) : c(v) {}
    template <class T> size_t reserve(T *&p, size_t n) {
        size_t offset = allocated;
        allocated += (n * sizeof(T) + 255) & ~size_t(255);
        p = reinterpret_cast<T *>(offset);
        return offset;
    }
    template <class T> void relocate(T *&p) {
        p = reinterpret_cast<T *>(static_cast<char *>(memory) + reinterpret_cast<size_t>(p));
    }
    void init(size_t budget, size_t headroom, size_t hostBudget) {
        size_t n = size_t(c.w) * c.h, l = c.bypass ? 0 : size_t(c.sw) * c.sh;
        rawBytes = n * c.bytes;
        blocks = int((n + 255) / 256);
        reductions = std::min(blocks, 128);
        reserve(rawIn, rawBytes);
        reserve(rawOut, rawBytes);
        reserve(cfg, 1);
        reserve(stats, 1);
        reserve(partial, reductions);
        reserve(hist, 4096);
        reserve(scores, !c.quick || c.mask ? n : l);
        reserve(low, l);
        reserve(temp, l * c.count);
        reserve(planes, l * c.count);
        if (allocated > budget)
            throw std::runtime_error("GPU workspace exceeds max_gpu_mb");
        pinned = 2 * rawBytes;
        if (pinned > hostBudget)
            throw std::runtime_error("GPU staging exceeds max_cpu_mb");
        DeviceScope device(c.device);
        size_t free = 0, total = 0;
        check(cudaMemGetInfo(&free, &total));
        if (headroom > free || allocated > free - headroom)
            throw std::runtime_error("insufficient free GPU memory after reserve_gpu_mb");
        int least = 0, greatest = 0;
        check(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        check(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, least));
        check(cudaEventCreateWithFlags(&done, cudaEventBlockingSync | cudaEventDisableTiming));
        check(cudaMalloc(&memory, allocated));
        check(cudaMallocHost(&host, pinned));
        relocate(rawIn);
        relocate(rawOut);
        relocate(cfg);
        relocate(stats);
        relocate(partial);
        relocate(hist);
        relocate(scores);
        relocate(low);
        relocate(temp);
        relocate(planes);
        check(cudaMemcpyAsync(cfg, &c, sizeof(c), cudaMemcpyHostToDevice, stream));
        check(cudaStreamSynchronize(stream));
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        try {
            if (c.floating)
                enqueue<float>();
            else if (c.bytes == 1)
                enqueue<uint8_t>();
            else
                enqueue<uint16_t>();
            check(cudaGetLastError());
        } catch (...) {
            cudaGraph_t abandoned = nullptr;
            cudaStreamEndCapture(stream, &abandoned);
            if (abandoned)
                cudaGraphDestroy(abandoned);
            throw;
        }
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    }
    template <class T> void enqueue() {
        check(cudaMemcpyAsync(rawIn, host, rawBytes, cudaMemcpyHostToDevice, stream));
        const T *in = reinterpret_cast<const T *>(rawIn);
        T *out = reinterpret_cast<T *>(rawOut);
        if (!c.bypass) {
            rangeInput<<<reductions, 256, 0, stream>>>(in, partial, cfg);
            finishRange<<<1, 1, 0, stream>>>(partial, reductions, stats, cfg, true);
            downsample<<<(c.sw * c.sh + 255) / 256, 256, 0, stream>>>(in, low, cfg, stats);
            horizontal<<<dim3((c.sh + 63) / 64, c.count), 64, 0, stream>>>(low, temp, cfg);
            vertical<<<dim3((c.sw + 63) / 64, c.count), 64, 0, stream>>>(temp, planes, cfg);
            if (c.quick) {
                combineIllumination<<<(c.sw * c.sh + 255) / 256, 256, 0, stream>>>(planes, cfg);
                quickScoreRange<<<reductions, 256, 0, stream>>>(in, planes, scores, partial, cfg, stats);
            } else
                scoreRange<<<reductions, 256, 0, stream>>>(in, planes, scores, partial, cfg, stats);
            finishRange<<<1, 1, 0, stream>>>(partial, reductions, stats, cfg, false);
            check(cudaMemsetAsync(hist, 0, 4096 * sizeof(unsigned), stream));
            histogram<<<reductions, 256, 0, stream>>>(scores, hist, c.quick ? c.sw * c.sh : c.w * c.h, stats);
            bounds<<<1, 1, 0, stream>>>(hist, stats, cfg);
        }
        if (c.quick && !c.bypass)
            balanceQuick<<<blocks, 256, 0, stream>>>(in, out, planes, cfg, stats);
        else
            balance<<<blocks, 256, 0, stream>>>(in, out, scores, cfg, stats);
        if (c.mask) {
            verticalMask<<<blocks, 256, 0, stream>>>(out, scores, cfg);
            finishMask<<<dim3((c.w + 31) / 32, (c.h + 7) / 8), dim3(32, 8), 0, stream>>>(scores, in, out, cfg);
        }
        check(cudaMemcpyAsync(static_cast<char *>(host) + rawBytes, rawOut, rawBytes, cudaMemcpyDeviceToHost,
                              stream));
    }
    void process(const void *src, ptrdiff_t ss, void *dst, ptrdiff_t ds) override {
        DeviceScope device(c.device);
        size_t row = size_t(c.w) * c.bytes;
        for (int y = 0; y < c.h; ++y)
            std::memcpy(static_cast<char *>(host) + size_t(y) * row, static_cast<const char *>(src) + y * ss, row);
        check(cudaGraphLaunch(executable, stream));
        check(cudaEventRecord(done, stream));
        check(cudaEventSynchronize(done));
        for (int y = 0; y < c.h; ++y)
            std::memcpy(static_cast<char *>(dst) + y * ds,
                        static_cast<const char *>(host) + rawBytes + size_t(y) * row, row);
    }
    size_t deviceBytes() const override {
        return allocated;
    }
    size_t hostBytes() const override {
        return pinned;
    }
    ~CudaGpu() override {
        if (!stream && !memory && !host && !done && !graph && !executable)
            return;
        int old = -1;
        if (cudaGetDevice(&old) != cudaSuccess)
            cudaGetLastError();
        cudaSetDevice(c.device);
        if (stream)
            cudaStreamSynchronize(stream);
        if (executable)
            cudaGraphExecDestroy(executable);
        if (graph)
            cudaGraphDestroy(graph);
        if (memory)
            cudaFree(memory);
        if (host)
            cudaFreeHost(host);
        if (done)
            cudaEventDestroy(done);
        if (stream)
            cudaStreamDestroy(stream);
        cudaGetLastError();
        if (old >= 0)
            cudaSetDevice(old);
    }
};
} // namespace
std::unique_ptr<Gpu> createGpu(const Config &c, size_t budget, size_t reserve, size_t hostBudget) {
    auto p = std::make_unique<CudaGpu>(c);
    p->init(budget, reserve, hostBudget);
    return p;
}
} // namespace rf
