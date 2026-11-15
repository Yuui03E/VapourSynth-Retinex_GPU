// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine.hpp"
#include "simd.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <type_traits>
namespace rf {
Config configure(int w, int h, int bits, bool floating, bool mask, int scale, const std::vector<double> &sigma,
                 double blur, bool quick) {
    Config c;
    c.w = w;
    c.h = h;
    c.bits = bits;
    c.floating = floating;
    c.mask = mask;
    c.quick = quick;
    c.scale = scale;
    c.sw = (w + scale - 1) / scale;
    c.sh = (h + scale - 1) / scale;
    c.bytes = floating ? 4 : bits == 8 ? 1 : 2;
    c.peak = floating ? 1.f : float((1 << bits) - 1);
    c.count = int(sigma.size());
    c.bypass = true;
    for (int i = 0; i < c.count; ++i)
        if (sigma[i] > 0) {
            c.bypass = false;
            double s = sigma[i] / scale;
            double q = s < 2.5 ? 3.97156 - 4.14554 * std::sqrt(1 - .26891 * s) : .98711 * s - .96330;
            double b0 = 1.57825 + 2.44413 * q + 1.4281 * q * q + .422205 * q * q * q;
            double b1 = 2.44413 * q + 2.85619 * q * q + 1.26661 * q * q * q,
                   b2 = -(1.4281 * q * q + 1.26661 * q * q * q), b3 = .422205 * q * q * q;
            c.coeff[i] = {1 - (b1 + b2 + b3) / b0, b1 / b0, b2 / b0, b3 / b0};
        }
    c.radius = blur == 0 ? 0 : std::max(1, int(blur * 3 + .5));
    float sum = 0;
    for (int j = -c.radius; j <= c.radius; ++j) {
        float v = j == 0 ? 1.f : float(std::exp(-double(j) * j / (2 * blur * blur)));
        c.weights[j + c.radius] = v;
        sum += v;
    }
    for (int j = 0; j <= 2 * c.radius; ++j)
        c.weights[j] /= sum;
    return c;
}
size_t cpuEssential(const Config &c) {
    return (c.bypass ? 0 : size_t(c.sw) * c.sh * (c.count + 2) * sizeof(double)) +
           (c.quick && !c.bypass ? size_t(c.sw) * c.sh * sizeof(float) : 0) +
           (c.quick && !c.mask ? size_t(c.w) * sizeof(float) : 0) +
           (c.mask ? size_t(c.w) * (2 * c.radius + 10) * sizeof(float) + size_t(2 * c.radius + 9) * sizeof(int)
                   : 0) +
           4096 * sizeof(unsigned) + sizeof(Config) + sizeof(Stats);
}
struct Rows {
    int w, n;
    std::vector<float> data;
    std::vector<int> tags;
    Rows(int width, int count) : w(width), n(count), data(size_t(width) * count), tags(count, -1) {}
    void reset() {
        std::fill(tags.begin(), tags.end(), -1);
    }
    float *at(int y) {
        return data.data() + size_t(y % n) * w;
    }
    bool contains(int y) const {
        return tags[y % n] == y;
    }
    void mark(int y) {
        tags[y % n] = y;
    }
};
struct Cpu::Impl {
    Config c;
    Stats stats;
    std::vector<double> low, tmp, planes;
    std::vector<float> cache, vertical, quickScores;
    Rows ret, blur, grad;
    std::array<unsigned, 4096> hist{};
    explicit Impl(const Config &v)
        : c(v), low(c.bypass ? 0 : size_t(c.sw) * c.sh), tmp(low.size()), planes(low.size() * c.count),
          vertical(c.mask ? c.w : 0), ret(c.mask || c.quick ? c.w : 0, c.mask ? 2 * c.radius + 3 : 1),
          blur(c.mask ? c.w : 0, 3), grad(c.mask ? c.w : 0, 3) {
        if (c.quick && !c.bypass)
            quickScores.resize(low.size());
    }
    template <class T> void quickRow(const T *src, ptrdiff_t ss, int y, float *out) const {
        if (!c.bypass && stats.balanceGain > 0) {
            Config one = c;
            one.count = 1;
            one.coeff[0].b = 1;
#ifdef RF_SSE2
            scoresRow(src, ss, y, out, planes.data(), one, stats);
#else
            for (int x = 0; x < c.w; ++x)
                out[x] = quickScore(read(src, ss, x, y, c), x, y, planes.data(), c, stats);
#endif
        }
        int x = 0;
#ifdef RF_SSE2
        for (; x + 3 < c.w; x += 4) {
            __m128 v;
            if (c.bypass)
                v = load4(src + y * ss + x, c);
            else {
                if (stats.balanceGain > 0)
                    v = _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(out + x), _mm_set1_ps(stats.balanceLo)),
                                   _mm_set1_ps(stats.balanceGain));
                else {
                    v = _mm_sub_ps(load4(src + y * ss + x, c), _mm_set1_ps(stats.inputLo));
                    v = stats.inputGain > 0 ? _mm_mul_ps(v, _mm_set1_ps(stats.inputGain))
                                            : _mm_div_ps(v, _mm_set1_ps(stats.inputSpan));
                }
                v = _mm_min_ps(_mm_set1_ps(1), _mm_max_ps(_mm_setzero_ps(), v));
                if (!c.floating)
                    v = _mm_cvtepi32_ps(
                        _mm_cvttps_epi32(_mm_add_ps(_mm_set1_ps(float(16 << (c.bits - 8)) + .5f),
                                                    _mm_mul_ps(v, _mm_set1_ps(float(219 << (c.bits - 8)))))));
            }
            _mm_storeu_ps(out + x, v);
        }
#endif
        for (; x < c.w; ++x)
            out[x] = balanced(read(src, ss, x, y, c), c.bypass || stats.balanceGain <= 0 ? 0 : out[x], c, stats);
    }
    template <class T> void inputRangeQuick(const T *src, ptrdiff_t ss) {
        float lo = c.peak, hi = 0;
#ifdef RF_SSE2
        __m128 lowValues = _mm_set1_ps(c.peak), highValues = _mm_setzero_ps();
#endif
        for (int y = 0; y < c.h; ++y) {
            int x = 0;
#ifdef RF_SSE2
            for (; x + 3 < c.w; x += 4) {
                __m128 value = load4(src + y * ss + x, c);
                lowValues = _mm_min_ps(lowValues, value);
                highValues = _mm_max_ps(highValues, value);
            }
#endif
            for (; x < c.w; ++x) {
                float v = read(src, ss, x, y, c);
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
        }
#ifdef RF_SSE2
        alignas(16) float lows[4], highs[4];
        _mm_store_ps(lows, lowValues);
        _mm_store_ps(highs, highValues);
        for (int j = 0; j < 4; ++j) {
            lo = std::min(lo, lows[j]);
            hi = std::max(hi, highs[j]);
        }
#endif
        stats.uniform = hi <= lo;
        stats.inputLo = hi > lo ? lo : 0;
        stats.inputSpan = hi > lo ? hi - lo : c.peak;
        stats.inputGain = 1.f / stats.inputSpan;
        if (!std::isfinite(stats.inputGain))
            stats.inputGain = 0;
    }
    template <class T> float scoreAt(const T *src, ptrdiff_t ss, int x, int y) {
        return cache.empty() ? score(read(src, ss, x, y, c), x, y, planes.data(), c, stats)
                             : cache[size_t(y) * c.w + x];
    }
    template <class T> float *retRow(const T *src, ptrdiff_t ss, int y) {
        float *row = ret.at(y);
        if (ret.contains(y))
            return row;
        if (c.quick) {
            quickRow(src, ss, y, row);
        } else
            for (int x = 0; x < c.w; ++x)
                row[x] = balanced(read(src, ss, x, y, c), c.bypass ? 0 : scoreAt(src, ss, x, y), c, stats);
        ret.mark(y);
        return row;
    }
    template <class T> float *blurRow(const T *src, ptrdiff_t ss, int y) {
        float *row = blur.at(y);
        if (blur.contains(y))
            return row;
        std::fill(vertical.begin(), vertical.end(), 0.f);
        for (int j = -c.radius; j <= c.radius; ++j) {
            const float *r = retRow(src, ss, mirror(y + j, c.h));
            float k = c.weights[j + c.radius];
            for (int x = 0; x < c.w; ++x)
                vertical[x] += r[x] * k;
        }
        int x = 0;
        for (; x < c.w; ++x) {
#ifdef RF_SSE2
            if (x >= c.radius && x + 3 + c.radius < c.w) {
                __m128 sum = _mm_setzero_ps();
                for (int j = -c.radius; j <= c.radius; ++j)
                    sum = _mm_add_ps(sum, _mm_mul_ps(_mm_loadu_ps(vertical.data() + x + j),
                                                     _mm_set1_ps(c.weights[j + c.radius])));
                _mm_storeu_ps(row + x, sum);
                x += 3;
                continue;
            }
#endif
            float sum = 0;
            for (int j = -c.radius; j <= c.radius; ++j)
                sum += vertical[mirror(x + j, c.w)] * c.weights[j + c.radius];
            row[x] = sum;
        }
        blur.mark(y);
        return row;
    }
    template <class T> float *gradRow(const T *src, ptrdiff_t ss, int y) {
        float *row = grad.at(y);
        if (grad.contains(y))
            return row;
        const float *a = blurRow(src, ss, mirror(y - 1, c.h)), *b = blurRow(src, ss, y),
                    *d = blurRow(src, ss, mirror(y + 1, c.h));
        for (int x = 0; x < c.w; ++x) {
#ifdef RF_SSE2
            if (x >= 1 && x + 4 < c.w) {
                _mm_storeu_ps(row + x, gradient4(a, b, d, x, c));
                x += 3;
                continue;
            }
#endif
            int l = mirror(x - 1, c.w), r = mirror(x + 1, c.w);
            row[x] = gradient(a[l], a[x], a[r], b[l], b[r], d[l], d[x], d[r], c);
        }
        grad.mark(y);
        return row;
    }
    template <class T> void run(const T *src, ptrdiff_t ss, T *dst, ptrdiff_t ds) {
        ss /= sizeof(T);
        ds /= sizeof(T);
        stats = {};
        ret.reset();
        if (!c.bypass) {
            if (c.quick)
                inputRangeQuick(src, ss);
            else {
                float lo = c.peak, hi = 0;
                for (int y = 0; y < c.h; ++y)
                    for (int x = 0; x < c.w; ++x) {
                        float v = read(src, ss, x, y, c);
                        lo = std::min(lo, v);
                        hi = std::max(hi, v);
                    }
                stats.inputLo = hi > lo ? lo : 0;
                stats.inputSpan = hi > lo ? hi - lo : c.peak;
                stats.inputGain = 1.f / stats.inputSpan;
                if (!std::isfinite(stats.inputGain))
                    stats.inputGain = 0;
            }
            for (int y = 0; y < c.sh; ++y)
                for (int x = 0; x < c.sw; ++x) {
                    double sum = 0;
                    int count = 0;
                    for (int dy = 0; dy < c.scale && y * c.scale + dy < c.h; ++dy)
                        for (int dx = 0; dx < c.scale && x * c.scale + dx < c.w; ++dx) {
                            sum += normalized(read(src, ss, x * c.scale + dx, y * c.scale + dy, c), stats);
                            ++count;
                        }
                    low[size_t(y) * c.sw + x] = sum / count;
                }
            for (int k = 0; k < c.count; ++k)
                if (c.coeff[k].b != 0) {
                    for (int y = 0; y < c.sh; ++y)
                        line(low.data() + y * c.sw, tmp.data() + y * c.sw, c.sw, 1, c.coeff[k], false);
                    double *p = planes.data() + size_t(k) * low.size();
                    for (int x = 0; x < c.sw; ++x)
                        line(tmp.data() + x, p + x, c.sh, c.sw, c.coeff[k], true);
                }
            if (c.quick) {
                for (size_t i = 0; i < low.size(); ++i)
                    planes[i] = effectiveIllumination(planes.data(), i, low.size(), c);
                stats.lo = std::numeric_limits<float>::max();
                stats.hi = 0;
                for (int y = 0; y < c.sh; ++y)
                    for (int x = 0; x < c.sw; ++x) {
                        int px = clampi(x * c.scale + c.scale / 2, c.w),
                            py = clampi(y * c.scale + c.scale / 2, c.h);
                        float value = quickScore(read(src, ss, px, py, c), px, py, planes.data(), c, stats);
                        quickScores[size_t(y) * c.sw + x] = value;
                        stats.lo = std::min(stats.lo, value);
                        stats.hi = std::max(stats.hi, value);
                    }
                hist.fill(0);
                for (float value : quickScores)
                    ++hist[bin(value, stats)];
                percentile(hist.data(), stats, c);
            } else {
                stats.lo = std::numeric_limits<float>::max();
                stats.hi = 0;
                for (int y = 0; y < c.h; ++y) {
#ifdef RF_SSE2
                    if (!cache.empty())
                        scoresRow(src, ss, y, cache.data() + size_t(y) * c.w, planes.data(), c, stats);
#else
                    if (!cache.empty())
                        for (int x = 0; x < c.w; ++x)
                            cache[size_t(y) * c.w + x] =
                                score(read(src, ss, x, y, c), x, y, planes.data(), c, stats);
#endif
                    for (int x = 0; x < c.w; ++x) {
                        float v = scoreAt(src, ss, x, y);
                        stats.lo = std::min(stats.lo, v);
                        stats.hi = std::max(stats.hi, v);
                    }
                }
                hist.fill(0);
                for (int y = 0; y < c.h; ++y)
                    for (int x = 0; x < c.w; ++x)
                        ++hist[bin(scoreAt(src, ss, x, y), stats)];
                percentile(hist.data(), stats, c);
            }
        }
        if (!c.mask) {
            for (int y = 0; y < c.h; ++y) {
                if (c.quick) {
                    float *row = retRow(src, ss, y);
                    for (int x = 0; x < c.w; ++x)
                        dst[y * ds + x] = T(row[x]);
                } else
                    for (int x = 0; x < c.w; ++x)
                        dst[y * ds + x] =
                            T(balanced(read(src, ss, x, y, c), c.bypass ? 0 : scoreAt(src, ss, x, y), c, stats));
            }
            return;
        }
        ret.reset();
        blur.reset();
        grad.reset();
        for (int y = 0; y < c.h; ++y) {
            const float *a = gradRow(src, ss, clampi(y - 1, c.h)), *b = gradRow(src, ss, y),
                        *d = gradRow(src, ss, clampi(y + 1, c.h));
            for (int x = 0; x < c.w; ++x) {
#ifdef RF_SSE2
                if (x >= 1 && x + 4 < c.w) {
                    __m128 v =
                        _mm_min_ps(_mm_loadu_ps(b + x),
                                   _mm_min_ps(_mm_min_ps(_mm_loadu_ps(a + x - 1), _mm_loadu_ps(a + x + 1)),
                                              _mm_min_ps(_mm_loadu_ps(d + x - 1), _mm_loadu_ps(d + x + 1))));
                    v = _mm_min_ps(_mm_set1_ps(c.peak), _mm_add_ps(v, kirsch4(src, ss, x, y, c)));
                    alignas(16) float values[4];
                    _mm_store_ps(values, v);
                    for (int j = 0; j < 4; ++j)
                        dst[y * ds + x + j] = T(values[j]);
                    x += 3;
                    continue;
                }
#endif
                int l = clampi(x - 1, c.w), r = clampi(x + 1, c.w);
                float v = std::min(b[x], std::min(std::min(a[l], a[r]), std::min(d[l], d[r])));
                dst[y * ds + x] = T(clip(v + kirsch(src, ss, x, y, c), c.peak));
            }
        }
    }
};
Cpu::Cpu(const Config &c) : impl(std::make_unique<Impl>(c)) {}
Cpu::~Cpu() = default;
size_t Cpu::bytes() const {
    return cpuEssential(impl->c) + impl->cache.size() * sizeof(float);
}
bool Cpu::cached() const {
    return !impl->cache.empty();
}
void Cpu::enableCache(size_t budget) {
    size_t n = size_t(impl->c.w) * impl->c.h;
    if (!impl->c.quick && !impl->c.bypass && bytes() <= budget && n * sizeof(float) <= budget - bytes()) {
        try {
            impl->cache.resize(n);
        } catch (const std::bad_alloc &) { /* Streaming remains usable. */
        }
    }
}
void Cpu::process(const void *src, ptrdiff_t ss, void *dst, ptrdiff_t ds) {
    if (impl->c.floating)
        impl->run(static_cast<const float *>(src), ss, static_cast<float *>(dst), ds);
    else if (impl->c.bytes == 1)
        impl->run(static_cast<const uint8_t *>(src), ss, static_cast<uint8_t *>(dst), ds);
    else
        impl->run(static_cast<const uint16_t *>(src), ss, static_cast<uint16_t *>(dst), ds);
}
#ifndef RETINEX_FAST_CUDA
std::unique_ptr<Gpu> createGpu(const Config &, size_t, size_t, size_t) {
    throw std::runtime_error("this is the CPU-only build");
}
#endif
} // namespace rf
