// SPDX-License-Identifier: GPL-3.0-or-later
// API 4 test adapter. Includes UNCHANGED upstream Gaussian and MSR function bodies.
#include <VapourSynth4.h>
#include <VSHelper4.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>
using FLType = double;
constexpr double FLType_MAX = DBL_MAX;
constexpr size_t Alignment = 32;
template <class T> T Min(T a, T b) {
    return b < a ? b : a;
}
template <class T> T Max(T a, T b) {
    return b > a ? b : a;
}
template <class T> T Clip(T x, T a, T b) {
    return x <= a ? a : x >= b ? b : x;
}
template <class T> T *vs_aligned_malloc(size_t bytes, size_t alignment) {
    return vsh::vsh_aligned_malloc<T>(bytes, alignment);
}
void vs_aligned_free(void *p) {
    vsh::vsh_aligned_free(p);
}
struct Data {
    VSNode *node = nullptr;
    VSVideoInfo vi{};
    std::vector<double> sigma{50, 200, 350};
    double lower_thr = .001, upper_thr = .005;
    int HistBins = 4096;
};
struct MSRProcess {
    const Data &d;
    int width, height, stride, pcount;
    int MSRKernel(double *, const double *) const;
    int SimplestColorBalance(double *, const double *) const;
};
#include "Gaussian.inc"
#include "MSR.inc"
template <class T> void process(const VSFrame *src, VSFrame *dst, const Data &d, const VSAPI *api) {
    int w = d.vi.width, h = d.vi.height, bits = d.vi.format.bitsPerSample;
    const T *in = reinterpret_cast<const T *>(api->getReadPtr(src, 0));
    T *out = reinterpret_cast<T *>(api->getWritePtr(dst, 0));
    ptrdiff_t ss = api->getStride(src, 0) / sizeof(T), ds = api->getStride(dst, 0) / sizeof(T);
    if (std::all_of(d.sigma.begin(), d.sigma.end(), [](double s) { return s == 0; })) {
        for (int y = 0; y < h; ++y)
            std::copy_n(in + y * ss, w, out + y * ds);
        return;
    }
    double lo = (1 << bits) - 1, hi = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            lo = std::min(lo, double(in[y * ss + x]));
            hi = std::max(hi, double(in[y * ss + x]));
        }
    if (hi <= lo) {
        lo = 0;
        hi = (1 << bits) - 1;
    }
    double gain = 1 / (hi - lo);
    std::vector<double> norm(size_t(w) * h), value(size_t(w) * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            norm[y * w + x] = (in[y * ss + x] - lo) * gain;
    MSRProcess p{d, w, h, w, w * h};
    p.MSRKernel(value.data(), norm.data());
    p.SimplestColorBalance(value.data(), norm.data());
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            out[y * ds + x] = static_cast<T>(value[y * w + x] * (219 << (bits - 8)) + (16 << (bits - 8)) + .5);
}
const VSFrame *VS_CC get(int n, int reason, void *data, void **, VSFrameContext *ctx, VSCore *core,
                         const VSAPI *api) {
    auto &d = *static_cast<Data *>(data);
    if (reason == arInitial)
        api->requestFrameFilter(n, d.node, ctx);
    else if (reason == arAllFramesReady) {
        auto src = api->getFrameFilter(n, d.node, ctx);
        auto dst = api->newVideoFrame(&d.vi.format, d.vi.width, d.vi.height, src, core);
        if (d.vi.format.bitsPerSample == 8)
            process<uint8_t>(src, dst, d, api);
        else
            process<uint16_t>(src, dst, d, api);
        api->freeFrame(src);
        return dst;
    }
    return nullptr;
}
void VS_CC freeData(void *data, VSCore *, const VSAPI *api) {
    auto d = static_cast<Data *>(data);
    api->freeNode(d->node);
    delete d;
}
void VS_CC create(const VSMap *in, VSMap *out, void *, VSCore *core, const VSAPI *api) {
    auto d = std::make_unique<Data>();
    d->node = api->mapGetNode(in, "clip", 0, nullptr);
    d->vi = *api->getVideoInfo(d->node);
    if (d->vi.format.colorFamily != cfGray || d->vi.format.sampleType != stInteger) {
        api->mapSetError(out, "Reference adapter accepts integer GRAY only");
        api->freeNode(d->node);
        return;
    }
    int count = api->mapNumElements(in, "sigma"), err = 0;
    if (count > 0) {
        d->sigma.clear();
        for (int i = 0; i < count; ++i)
            d->sigma.push_back(api->mapGetFloat(in, "sigma", i, nullptr));
    }
    double lower = api->mapGetFloat(in, "lower_thr", 0, &err);
    if (!err)
        d->lower_thr = lower;
    double upper = api->mapGetFloat(in, "upper_thr", 0, &err);
    if (!err)
        d->upper_thr = upper;
    VSFilterDependency dep{d->node, rpStrictSpatial};
    api->createVideoFilter(out, "MSRCPReference", &d->vi, get, freeData, fmParallel, &dep, 1, d.get(), core);
    d.release();
}
VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *p, const VSPLUGINAPI *api) {
    api->configPlugin("org.retinex-gpu.cpu-reference", "retinex", "Upstream MSRCP test adapter",
                      VS_MAKE_VERSION(4, 0), VAPOURSYNTH_API_VERSION, 0, p);
    api->registerFunction("MSRCP", "clip:vnode;sigma:float[]:opt;lower_thr:float:opt;upper_thr:float:opt;",
                          "clip:vnode;", create, nullptr, p);
}
