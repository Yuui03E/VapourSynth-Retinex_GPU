// SPDX-License-Identifier: GPL-3.0-or-later
#include <VapourSynth4.h>
#include <VSHelper4.h>
#include "exact/cpu.hpp"
#include "fast/engine.hpp"
#include <algorithm>
#include <climits>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
namespace {
struct Worker {
    retinex::Parameters p;
    rf::Config c;
    bool fast = false, quick = false, fallback = true;
    std::string why;
    std::unique_ptr<retinex::CpuWorkspace> exactCpu;
    std::unique_ptr<retinex::Workspace> exactGpu;
    std::unique_ptr<rf::Cpu> fastCpu;
    std::unique_ptr<rf::Gpu> fastGpu;
#ifdef RETINEX_FAST_TESTING
    int failAfter = -1, processed = 0;
#endif
    bool gpu() const {
        return bool(exactGpu) || bool(fastGpu);
    }
    void cpu(const void *src, ptrdiff_t ss, void *dst, ptrdiff_t ds) {
        if (fast)
            fastCpu->process(src, ss, dst, ds);
        else {
            if (!exactCpu)
                exactCpu = std::make_unique<retinex::CpuWorkspace>(p);
            exactCpu->process(src, ss, dst, ds);
        }
    }
    void process(const void *src, ptrdiff_t ss, void *dst, ptrdiff_t ds, const VSAPI *api, VSCore *core) {
        if (!gpu()) {
            cpu(src, ss, dst, ds);
            return;
        }
        try {
#ifdef RETINEX_FAST_TESTING
            if (failAfter >= 0 && processed++ >= failAfter)
                throw std::runtime_error("simulated GPU processing failure (test build)");
#endif
            if (fast)
                fastGpu->process(src, ss, dst, ds);
            else
                exactGpu->process(src, ss, dst, ds);
        } catch (const std::exception &e) {
            if (!fallback)
                throw;
            why = e.what();
            exactGpu.reset();
            fastGpu.reset();
            api->logMessage(mtWarning, ("retinex_gpu: using CPU (" + why + ")").c_str(), core);
            cpu(src, ss, dst, ds);
        }
    }
};
struct Filter {
    const VSAPI *api = nullptr;
    VSNode *node = nullptr;
    VSVideoInfo vi{};
    std::vector<std::unique_ptr<Worker>> workers;
    std::vector<bool> busy;
    std::mutex mutex;
    std::condition_variable available;
    ~Filter() {
        if (node)
            api->freeNode(node);
    }
    size_t acquire() {
        std::unique_lock<std::mutex> lock(mutex);
        available.wait(lock, [&] { return std::find(busy.begin(), busy.end(), false) != busy.end(); });
        size_t i = std::find(busy.begin(), busy.end(), false) - busy.begin();
        busy[i] = true;
        return i;
    }
    void release(size_t i) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            busy[i] = false;
        }
        available.notify_one();
    }
};
struct Lease {
    Filter &f;
    size_t i;
    explicit Lease(Filter &filter) : f(filter), i(f.acquire()) {}
    ~Lease() {
        f.release(i);
    }
};
const VSFrame *VS_CC frame(int n, int reason, void *instance, void **, VSFrameContext *ctx, VSCore *core,
                           const VSAPI *api) {
    auto &f = *static_cast<Filter *>(instance);
    if (reason == arInitial)
        api->requestFrameFilter(n, f.node, ctx);
    else if (reason == arAllFramesReady) {
        const VSFrame *src = api->getFrameFilter(n, f.node, ctx);
        VSFrame *dst = nullptr;
        try {
            Lease lease(f);
            auto &w = *f.workers[lease.i];
            dst = api->newVideoFrame(&f.vi.format, f.vi.width, f.vi.height, src, core);
            if (!dst)
                throw std::runtime_error("cannot allocate output frame");
            w.process(api->getReadPtr(src, 0), api->getStride(src, 0), api->getWritePtr(dst, 0),
                      api->getStride(dst, 0), api, core);
            VSMap *props = api->getFramePropertiesRW(dst);
            bool bypass = w.p.stage == retinex::Stage::MSRCP &&
                          std::all_of(w.p.sigma.begin(), w.p.sigma.end(), [](double s) { return s == 0; });
            if (!bypass)
                api->mapSetInt(props, "_ColorRange", w.p.stage == retinex::Stage::MSRCP && !w.p.floating ? 1 : 0,
                               maReplace);
            api->mapDeleteKey(props, "_ChromaLocation");
            api->mapSetData(props, "RetinexMode",
                            w.quick  ? "fastest"
                            : w.fast ? "fast"
                                     : "exact",
                            -1, dtUtf8, maReplace);
            api->mapSetInt(props, "RetinexWorkers", int64_t(f.workers.size()), maReplace);
            api->mapSetData(props, "RetinexBackend", w.gpu() ? "gpu" : "cpu", -1, dtUtf8, maReplace);
            api->mapSetData(props, "RetinexFallback", w.why.c_str(), -1, dtUtf8, maReplace);
            if (w.fast) {
                // Retained diagnostic properties for scripts using the fast engine.
                api->mapSetData(props, "RetinexFastBackend", w.gpu() ? "gpu" : "cpu", -1, dtUtf8, maReplace);
                api->mapSetData(props, "RetinexFastFallback", w.why.c_str(), -1, dtUtf8, maReplace);
                api->mapSetInt(props, "RetinexFastDownscale", w.c.scale, maReplace);
                if (w.quick)
                    api->mapSetData(props, "RetinexFastestTone", "local-ratio", -1, dtUtf8, maReplace);
                api->mapSetInt(props, "RetinexFastGPUBytes", w.fastGpu ? int64_t(w.fastGpu->deviceBytes()) : 0,
                               maReplace);
                api->mapSetInt(props, "RetinexFastCPUBytes",
                               int64_t(w.fastCpu->bytes() + (w.fastGpu ? w.fastGpu->hostBytes() : 0)), maReplace);
                api->mapSetInt(props, "RetinexFastScoreCache", w.fastCpu->cached(), maReplace);
            }
            api->freeFrame(src);
            return dst;
        } catch (const std::exception &e) {
            if (dst)
                api->freeFrame(dst);
            api->freeFrame(src);
            api->setFilterError(("retinex_gpu: " + std::string(e.what())).c_str(), ctx);
        }
    }
    return nullptr;
}
void VS_CC freeFilter(void *p, VSCore *, const VSAPI *) {
    delete static_cast<Filter *>(p);
}
int integer(const VSMap *in, const char *key, int def, const VSAPI *api) {
    int error = 0;
    int64_t v = api->mapGetInt(in, key, 0, &error);
    if (error)
        return def;
    if (v < INT_MIN || v > INT_MAX)
        throw std::invalid_argument(std::string(key) + " is out of range");
    return int(v);
}
double number(const VSMap *in, const char *key, double def, const VSAPI *api) {
    int error = 0;
    double v = api->mapGetFloat(in, key, 0, &error);
    return error ? def : v;
}
std::string string(const VSMap *in, const char *key, const char *def, const VSAPI *api) {
    int error = 0;
    const char *v = api->mapGetData(in, key, 0, &error);
    return error ? def : std::string(v, api->mapGetDataSize(in, key, 0, nullptr));
}
void VS_CC create(const VSMap *in, VSMap *out, void *user, VSCore *core, const VSAPI *api) {
    auto f = std::make_unique<Filter>();
    f->api = api;
    try {
        f->node = api->mapGetNode(in, "clip", 0, nullptr);
        f->vi = *api->getVideoInfo(f->node);
        auto fmt = f->vi.format;
        if (!vsh::isConstantVideoFormat(&f->vi) || (fmt.colorFamily != cfGray && fmt.colorFamily != cfYUV))
            throw std::invalid_argument("input must be constant-format GRAY or YUV");
        if (!((fmt.sampleType == stFloat && fmt.bitsPerSample == 32) ||
              (fmt.sampleType == stInteger && (fmt.bitsPerSample == 8 || fmt.bitsPerSample == 10 ||
                                               fmt.bitsPerSample == 12 || fmt.bitsPerSample == 16))))
            throw std::invalid_argument("supported depths: integer 8/10/12/16 and float32");
        if (f->vi.width > INT_MAX / 2 - 1024 || f->vi.height > INT_MAX / 2 - 1024 ||
            int64_t(f->vi.width) * f->vi.height > INT_MAX - 4096)
            throw std::invalid_argument("frame is too large");
        std::string mode = string(in, "mode", "exact", api), backend = string(in, "backend", "auto", api);
        if (mode != "exact" && mode != "fast" && mode != "fastest")
            throw std::invalid_argument("mode must be exact, fast, or fastest");
        if (backend != "auto" && backend != "cpu" && backend != "gpu")
            throw std::invalid_argument("backend must be auto, cpu, or gpu");
        retinex::Parameters p;
        p.width = f->vi.width;
        p.height = f->vi.height;
        p.bits = fmt.bitsPerSample;
        p.floating = fmt.sampleType == stFloat;
        p.stage = static_cast<retinex::Stage>(reinterpret_cast<intptr_t>(user));
        bool quick = mode == "fastest" && (p.stage == retinex::Stage::MSRCP || p.stage == retinex::Stage::Mask);
        bool fast =
            (mode == "fast" || quick) && (p.stage == retinex::Stage::MSRCP || p.stage == retinex::Stage::Mask);
        if (quick)
            for (const char *key : {"downscale"})
                if (api->mapNumElements(in, key) >= 0)
                    throw std::invalid_argument(std::string(key) + " is fixed at 8 in fastest mode");
        int count = api->mapNumElements(in, "sigma");
        if (count == 0 || count > 32)
            throw std::invalid_argument("sigma must contain 1 to 32 values");
        if (count > 0) {
            p.sigma.clear();
            for (int i = 0; i < count; ++i)
                p.sigma.push_back(api->mapGetFloat(in, "sigma", i, nullptr));
        }
        for (double s : p.sigma)
            if (!std::isfinite(s) || s < 0 || s > 1000)
                throw std::invalid_argument("sigma must be finite and in [0,1000]");
        p.lower = number(in, "lower_thr", .001, api);
        p.upper = number(in, "upper_thr", .005, api);
        if (!std::isfinite(p.lower) || !std::isfinite(p.upper) || p.lower < 0 || p.upper < 0 ||
            p.lower + p.upper >= 1)
            throw std::invalid_argument("thresholds must be finite, nonnegative, and sum to less than 1");
        double blur = number(in, "blur_sigma", 1, api);
        if (!std::isfinite(blur) || blur < 0 || blur > 100)
            throw std::invalid_argument("blur_sigma must be finite and in [0,100]");
        p.blur = float(blur);
        p.op = integer(in, "op", 1, api);
        p.device = integer(in, "device", 0, api);
        if (p.op != 1 && p.op != 2)
            throw std::invalid_argument("op must be 1 (Prewitt) or 2 (Sobel)");
        if (p.device < 0)
            throw std::invalid_argument("device must be nonnegative");
        int pyramid = integer(in, "fast_pyramid", 0, api), fallback = integer(in, "fallback", 1, api);
        if (pyramid != 0 && pyramid != 1)
            throw std::invalid_argument("fast_pyramid must be 0 or 1");
        if (fallback != 0 && fallback != 1)
            throw std::invalid_argument("fallback must be 0 or 1");
        if (fast && pyramid)
            throw std::invalid_argument("fast_pyramid is for exact mode; fast mode uses downscale");
        p.fast = pyramid != 0;
        const char *fastKeys[] = {"downscale", "max_gpu_mb", "max_cpu_mb", "reserve_gpu_mb"};
        if (!fast)
            for (const char *key : fastKeys)
                if (api->mapNumElements(in, key) >= 0)
                    throw std::invalid_argument(std::string(key) + " is only used by fast MSRCP/RetinexEdgeMask");
        VSCoreInfo coreInfo{};
        api->getCoreInfo(core, &coreInfo);
        int cpuWorkers = std::clamp(coreInfo.numThreads, 1, 16);
        int defaultWorkers = quick ? (backend == "cpu" ? cpuWorkers : 2) : fast ? 1 : 2;
#ifndef RETINEX_FAST_CUDA
        if (quick && backend == "auto")
            defaultWorkers = cpuWorkers;
#endif
        int workers = integer(in, "workers", defaultWorkers, api);
        if (workers < 1 || workers > 16)
            throw std::invalid_argument("workers must be in [1,16]");
        if (fast && !quick && workers != 1)
            throw std::invalid_argument("fast mode uses one workspace; workers must be 1");
        int scale = integer(in, "downscale", 8, api), gpuMB = integer(in, "max_gpu_mb", 128, api),
            cpuMB = integer(in, "max_cpu_mb", 128, api), reserveMB = integer(in, "reserve_gpu_mb", 512, api);
        if (fast && scale != 2 && scale != 4 && scale != 8 && scale != 16 && scale != 32 && scale != 64)
            throw std::invalid_argument("downscale must be 2, 4, 8, 16, 32, or 64");
        if (gpuMB < 0 || gpuMB > 1048576 || cpuMB < 1 || cpuMB > 1048576 || reserveMB < 0 || reserveMB > 1048576)
            throw std::invalid_argument("memory budgets must be in [0,1048576] MiB (CPU minimum 1)");
        for (int i = 0; i < workers; ++i) {
            auto w = std::make_unique<Worker>();
            w->p = p;
            w->fast = fast;
            w->quick = quick;
            w->fallback = fallback != 0;
#ifdef RETINEX_FAST_TESTING
            w->failAfter = integer(in, "test_fail_after", -1, api);
#endif
            size_t cpuBudget = size_t(cpuMB) * rf::MiB, essential = 0;
            if (fast) {
                w->c = rf::configure(p.width, p.height, p.bits, p.floating, p.stage == retinex::Stage::Mask, scale,
                                     p.sigma, blur, quick);
                w->c.lower = p.lower;
                w->c.upper = p.upper;
                w->c.op = p.op;
                w->c.device = p.device;
                essential = rf::cpuEssential(w->c);
                if (essential > cpuBudget)
                    throw std::runtime_error(
                        "minimum CPU workspace exceeds max_cpu_mb; increase budget or downscale");
                w->fastCpu = std::make_unique<rf::Cpu>(w->c);
            }
            if (backend != "cpu") {
                try {
#ifdef RETINEX_FAST_CUDA
                    if (fast)
                        w->fastGpu = rf::createGpu(w->c, size_t(gpuMB) * rf::MiB, size_t(reserveMB) * rf::MiB,
                                                   cpuBudget - essential);
                    else
                        w->exactGpu = std::make_unique<retinex::Workspace>(p);
#else
                    if (backend == "gpu")
                        throw std::runtime_error("this is the CPU-only build");
#endif
                } catch (const std::exception &e) {
                    if (!fallback)
                        throw;
                    w->why = e.what();
                    api->logMessage(mtWarning, ("retinex_gpu: using CPU (" + w->why + ")").c_str(), core);
                }
            }
            if (fast)
                w->fastCpu->enableCache(cpuBudget - (w->fastGpu ? w->fastGpu->hostBytes() : 0));
            else if (!w->exactGpu)
                w->exactCpu = std::make_unique<retinex::CpuWorkspace>(p);
            f->workers.push_back(std::move(w));
        }
        f->busy.resize(workers, false);
        if (!api->queryVideoFormat(&f->vi.format, cfGray, fmt.sampleType, fmt.bitsPerSample, 0, 0, core))
            throw std::runtime_error("cannot create GRAY output");
        VSFilterDependency dep{f->node, rpStrictSpatial};
        api->createVideoFilter(out, "Retinex", &f->vi, frame, freeFilter,
                               fast && !quick ? fmParallelRequests : fmParallel, &dep, 1, f.get(), core);
        f.release();
    } catch (const std::exception &e) {
        api->mapSetError(out, ("retinex_gpu: " + std::string(e.what())).c_str());
    }
}
} // namespace
VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *api) {
    api->configPlugin("org.retinex-gpu", "retinex_gpu", "Retinex CPU/CUDA", VS_MAKE_VERSION(2, 1),
                      VAPOURSYNTH_API_VERSION, 0, plugin);
    const char *args =
        "clip:vnode;mode:data:opt;backend:data:opt;sigma:float[]:opt;lower_thr:float:opt;upper_thr:float:opt;"
        "blur_sigma:float:opt;op:int:opt;device:int:opt;fallback:int:opt;workers:int:opt;fast_pyramid:int:opt;"
        "downscale:int:opt;max_gpu_mb:int:opt;reserve_gpu_mb:int:opt;max_cpu_mb:int:opt;"
#ifdef RETINEX_FAST_TESTING
        "test_fail_after:int:opt;"
#endif
        ;
    const char *names[] = {"MSRCP", "Kirsch", "TCanny", "RetinexEdgeMask"};
    for (intptr_t i = 0; i < 4; ++i)
        api->registerFunction(names[i], args, "clip:vnode;", create, reinterpret_cast<void *>(i), plugin);
}
