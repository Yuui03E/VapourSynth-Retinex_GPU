// SPDX-License-Identifier: GPL-3.0-or-later
// Retinex equations/boundaries adapted from mawen1250's GPLv3 Retinex.
#include "cpu.hpp"
#include "../fast/common.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
namespace retinex {
struct CpuWorkspace::Impl {
    Parameters p;
    int n, w, h, radius = 0;
    float peak;
    std::vector<float> original, result, temp, blur, gradient, taps;
    std::vector<double> normalized, small, horizontal, gauss, product;
    std::vector<rf::Coeff> coeff;
    explicit Impl(const Parameters &params)
        : p(params), n(p.width * p.height), w(p.fast ? (p.width + 3) / 4 : p.width),
          h(p.fast ? (p.height + 3) / 4 : p.height), peak(p.floating ? 1.f : float((1 << p.bits) - 1)),
          original(n), result(n) {
        if (p.stage == Stage::MSRCP || p.stage == Stage::Mask) {
            normalized.resize(n);
            product.resize(n);
            small.resize(size_t(w) * h);
            horizontal.resize(size_t(w) * h);
            gauss.resize(size_t(w) * h);
            for (double s : p.sigma) {
                if (s == 0) {
                    coeff.push_back({});
                    continue;
                }
                s = p.fast ? s / 4 : s;
                double q = s < 2.5 ? 3.97156 - 4.14554 * std::sqrt(1 - .26891 * s) : .98711 * s - .96330;
                double b0 = 1.57825 + 2.44413 * q + 1.4281 * q * q + .422205 * q * q * q;
                double b1 = 2.44413 * q + 2.85619 * q * q + 1.26661 * q * q * q;
                double b2 = -(1.4281 * q * q + 1.26661 * q * q * q), b3 = .422205 * q * q * q;
                coeff.push_back({1 - (b1 + b2 + b3) / b0, b1 / b0, b2 / b0, b3 / b0});
            }
        }
        if (p.stage == Stage::Mask)
            gradient.resize(n);
        if (p.blur > 0 && (p.stage == Stage::TCanny || p.stage == Stage::Mask)) {
            radius = std::max(int(p.blur * 3.f + .5f), 1);
            taps.resize(2 * radius + 1);
            temp.resize(n);
            blur.resize(n);
            float sum = 0;
            for (int k = -radius; k <= radius; ++k) {
                float v = k == 0 ? 1.f : std::exp(-(k * k) / (2.f * p.blur * p.blur));
                taps[k + radius] = v;
                sum += v;
            }
            for (float &v : taps)
                v /= sum;
        }
    }
    double sample(int i) const {
        double x = std::clamp(((i % p.width) + .5) / 4 - .5, 0., double(w - 1));
        double y = std::clamp(((i / p.width) + .5) / 4 - .5, 0., double(h - 1));
        int x0 = int(x), y0 = int(y), x1 = std::min(w - 1, x0 + 1), y1 = std::min(h - 1, y0 + 1);
        x -= x0;
        y -= y0;
        double a = gauss[y0 * w + x0] * (1 - x) + gauss[y0 * w + x1] * x;
        double b = gauss[y1 * w + x0] * (1 - x) + gauss[y1 * w + x1] * x;
        return a * (1 - y) + b * y;
    }
    void msr() {
        if (std::all_of(p.sigma.begin(), p.sigma.end(), [](double s) { return s == 0; })) {
            result = original;
            return;
        }
        auto range = std::minmax_element(original.begin(), original.end());
        double lo = *range.first, hi = *range.second;
        if (hi <= lo) {
            // A constant image has no contrast to enhance. Preserve its normalized
            // level instead of stretching recursive-Gaussian roundoff into edges.
            for (int i = 0; i < n; ++i) {
                double v = double(original[i]) / peak;
                result[i] = p.floating ? float(v) : float(int(v * (219 << (p.bits - 8)) +
                    (16 << (p.bits - 8)) + .5));
            }
            return;
        }
        double gain = 1 / (hi - lo);
        for (int i = 0; i < n; ++i)
            normalized[i] = (double(original[i]) - lo) * gain;
        if (p.fast) {
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    double sum = 0;
                    int count = 0;
                    for (int dy = 0; dy < 4 && y * 4 + dy < p.height; ++dy)
                        for (int dx = 0; dx < 4 && x * 4 + dx < p.width; ++dx) {
                            sum += normalized[(y * 4 + dy) * p.width + x * 4 + dx];
                            ++count;
                        }
                    small[y * w + x] = sum / count;
                }
        } else
            std::copy(normalized.begin(), normalized.end(), small.begin());
        std::fill(product.begin(), product.end(), 1.);
        for (auto k : coeff) {
            if (k.b == 0) {
                for (double &v : product)
                    v *= 2.;
                continue;
            }
            for (int y = 0; y < h; ++y)
                rf::line(small.data() + y * w, horizontal.data() + y * w, w, 1, k, false);
            for (int x = 0; x < w; ++x)
                rf::line(horizontal.data() + x, gauss.data() + x, h, w, k, true);
            for (int i = 0; i < n; ++i) {
                double g = p.fast ? sample(i) : gauss[i];
                product[i] *= g <= 0 ? 1. : normalized[i] / g + 1.;
            }
        }
        for (double &v : product)
            v = std::log(v) / int(p.sigma.size());
        auto pr = std::minmax_element(product.begin(), product.end());
        lo = *pr.first;
        hi = *pr.second;
        if (p.lower > 0 || p.upper > 0) {
            std::array<unsigned, 4096> hist{};
            double hg = hi > lo ? 4095 / (hi - lo) : 0, offset = -lo * hg;
            for (double v : product)
                ++hist[std::clamp(int(v * hg + offset), 0, 4095)];
            unsigned low = unsigned(n * p.lower + .5), high = unsigned(n * p.upper + .5), sum = 0;
            int a = 4095, b = 4095;
            for (int j = 0; j < 4096; ++j) {
                sum += hist[j];
                if (sum > low) {
                    a = j;
                    break;
                }
            }
            sum = 0;
            for (int j = 0; j < 4096; ++j) {
                sum += hist[j];
                if (sum >= unsigned(n) - high) {
                    b = j;
                    break;
                }
            }
            double step = (hi - lo) / 4095, base = lo;
            lo = a * step + base;
            hi = b * step + base;
        }
        gain = hi > lo ? 1 / (hi - lo) : 0;
        double offset = -lo * gain;
        for (int i = 0; i < n; ++i) {
            double v = hi > lo ? std::clamp(product[i] * gain + offset, 0., 1.) : normalized[i];
            result[i] = p.floating ? float(v) : float(int(v * (219 << (p.bits - 8)) + (16 << (p.bits - 8)) + .5));
        }
    }
    float kirsch(int x, int y) const {
        static constexpr int weights[36] = {5,  5,  5, -3, 0, -3, -3, -3, -3, -3, 5,  5,  -3, 0, 5, -3, -3, -3,
                                            -3, -3, 5, -3, 0, 5,  -3, -3, 5,  -3, -3, -3, -3, 0, 5, -3, 5,  5};
        float best = 0;
        for (int k = 0; k < 4; ++k) {
            float sum = 0;
            for (int j = 0; j < 9; ++j) {
                int xx = std::clamp(x + j % 3 - 1, 0, p.width - 1),
                    yy = std::clamp(y + j / 3 - 1, 0, p.height - 1);
                sum += original[yy * p.width + xx] * weights[k * 9 + j];
            }
            best = std::max(best, std::fabs(sum));
        }
        return std::min(peak, best);
    }
    void tcanny(const float *input, float *output) {
        if (radius) {
            for (int y = 0; y < p.height; ++y)
                for (int x = 0; x < p.width; ++x) {
                    float sum = 0;
                    for (int j = -radius; j <= radius; ++j)
                        sum += input[rf::mirror(y + j, p.height) * p.width + x] * taps[j + radius];
                    temp[y * p.width + x] = sum;
                }
            for (int y = 0; y < p.height; ++y)
                for (int x = 0; x < p.width; ++x) {
                    float sum = 0;
                    for (int j = -radius; j <= radius; ++j)
                        sum += temp[y * p.width + rf::mirror(x + j, p.width)] * taps[j + radius];
                    blur[y * p.width + x] = sum;
                }
            input = blur.data();
        }
        rf::Config c;
        c.op = p.op;
        c.floating = p.floating;
        c.peak = peak;
        for (int y = 0; y < p.height; ++y)
            for (int x = 0; x < p.width; ++x) {
                int l = rf::mirror(x - 1, p.width), r = rf::mirror(x + 1, p.width),
                    u = rf::mirror(y - 1, p.height) * p.width, d = rf::mirror(y + 1, p.height) * p.width,
                    row = y * p.width;
                output[row + x] = rf::gradient(input[u + l], input[u + x], input[u + r], input[row + l],
                                               input[row + r], input[d + l], input[d + x], input[d + r], c);
            }
    }
    template <class T> void render(const void *src, ptrdiff_t ss, void *dst, ptrdiff_t ds) {
        for (int y = 0; y < p.height; ++y) {
            const T *row = reinterpret_cast<const T *>(static_cast<const char *>(src) + y * ss);
            for (int x = 0; x < p.width; ++x)
                original[y * p.width + x] = rf::clean(float(row[x]), peak);
        }
        if (p.stage == Stage::MSRCP)
            msr();
        else if (p.stage == Stage::Kirsch)
            for (int y = 0; y < p.height; ++y)
                for (int x = 0; x < p.width; ++x)
                    result[y * p.width + x] = kirsch(x, y);
        else if (p.stage == Stage::TCanny)
            tcanny(original.data(), result.data());
        else {
            msr();
            tcanny(result.data(), gradient.data());
            for (int y = 0; y < p.height; ++y)
                for (int x = 0; x < p.width; ++x) {
                    float v = gradient[y * p.width + x];
                    for (int dy = -1; dy <= 1; dy += 2)
                        for (int dx = -1; dx <= 1; dx += 2)
                            v = std::min(v, gradient[std::clamp(y + dy, 0, p.height - 1) * p.width +
                                                     std::clamp(x + dx, 0, p.width - 1)]);
                    result[y * p.width + x] = std::min(peak, kirsch(x, y) + v);
                }
        }
        for (int y = 0; y < p.height; ++y) {
            T *row = reinterpret_cast<T *>(static_cast<char *>(dst) + y * ds);
            for (int x = 0; x < p.width; ++x)
                row[x] = static_cast<T>(std::clamp(result[y * p.width + x], 0.f, peak));
        }
    }
};
CpuWorkspace::CpuWorkspace(const Parameters &p) : impl(std::make_unique<Impl>(p)) {}
CpuWorkspace::~CpuWorkspace() = default;
void CpuWorkspace::process(const void *src, ptrdiff_t ss, void *dst, ptrdiff_t ds) {
    if (impl->p.floating)
        impl->render<float>(src, ss, dst, ds);
    else if (impl->p.bits == 8)
        impl->render<uint8_t>(src, ss, dst, ds);
    else
        impl->render<uint16_t>(src, ss, dst, ds);
}
#ifndef RETINEX_FAST_CUDA
struct Workspace::Impl {};
Workspace::Workspace(const Parameters &) {
    throw std::runtime_error("this is the CPU-only build");
}
Workspace::~Workspace() = default;
void Workspace::process(const void *, ptrdiff_t, void *, ptrdiff_t) {
    throw std::runtime_error("this is the CPU-only build");
}
#endif
} // namespace retinex
