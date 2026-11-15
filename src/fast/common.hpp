// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#ifdef __CUDACC__
#define RF_HD __host__ __device__
#else
#define RF_HD
#endif

namespace rf {
struct Coeff {
    double b = 0, b1 = 0, b2 = 0, b3 = 0;
};
struct Config {
    int w = 0, h = 0, sw = 0, sh = 0, scale = 8, count = 3, bits = 8, bytes = 1, device = 0, op = 1, radius = 3;
    bool floating = false, mask = false, bypass = false, quick = false;
    float peak = 255, weights[601]{};
    double lower = .001, upper = .005;
    Coeff coeff[32]{};
};
struct Stats {
    float inputLo = 0, inputGain = 1, inputSpan = 1, lo = 0, hi = 0, balanceLo = 0, balanceGain = 0;
    bool uniform = false;
};
RF_HD inline int clampi(int x, int n) {
    return x < 0 ? 0 : x >= n ? n - 1 : x;
}
RF_HD inline int mirror(int x, int n) {
    if (x >= 0 && x < n)
        return x;
    if (n <= 1)
        return 0;
    int p = 2 * (n - 1);
    x %= p;
    if (x < 0)
        x += p;
    return x < n ? x : p - x;
}
RF_HD inline float clean(float x, float peak) {
    return (x >= 0 && x <= peak) ? x : (x > peak && x < INFINITY ? peak : 0.f);
}
RF_HD inline float clip(float x, float peak = 1) {
    return x < 0 ? 0 : x > peak ? peak : x;
}
template <class T> RF_HD inline float read(const T *p, ptrdiff_t stride, int x, int y, const Config &c) {
    return clean(float(p[y * stride + x]), c.peak);
}
RF_HD inline float normalized(float v, const Stats &s) {
    // Dividing directly avoids an overflowing reciprocal for subnormal float
    // source ranges. Ordinary video takes the faster multiply path.
    return clip(s.inputGain > 0 ? (v - s.inputLo) * s.inputGain : (v - s.inputLo) / s.inputSpan);
}
// Positive-normal log approximation. The input is >=1; five atanh terms
// bound series truncation error below 1.2e-6 before FP32 rounding, and avoid
// platform libm differences.
RF_HD inline float fastLog(float x) {
    uint32_t bits;
#ifdef __CUDA_ARCH__
    bits = __float_as_uint(x);
#else
    std::memcpy(&bits, &x, sizeof(bits));
#endif
    int exponent = int(bits >> 23) - 127;
    bits = (bits & 0x7fffffu) | 0x3f800000u;
    float m;
#ifdef __CUDA_ARCH__
    m = __uint_as_float(bits);
#else
    std::memcpy(&m, &bits, sizeof(m));
#endif
    float z = (m - 1.f) / (m + 1.f), z2 = z * z;
    float series = 1.f + z2 * (1.f / 3.f + z2 * (1.f / 5.f + z2 * (1.f / 7.f + z2 / 9.f)));
    return float(exponent) * 0.6931471805599453f + 2.f * z * series;
}
RF_HD inline double recur(double x, double a, double b, double d, Coeff k) {
    return k.b * x + k.b1 * a + k.b2 * b + k.b3 * d;
}
// Same endpoint convention as the original Young/van Vliet implementation.
RF_HD inline void line(const double *in, double *out, int length, int step, Coeff k, bool vertical) {
    double a = in[0];
    if (vertical)
        a = recur(a, a, a, a, k);
    out[0] = a;
    double b = a, d = a;
    for (int j = 1; j < length; ++j) {
        double z = recur(in[j * step], a, b, d, k);
        d = b;
        b = a;
        a = z;
        out[j * step] = z;
    }
    int end = (length - 1) * step;
    a = out[end];
    if (vertical) {
        a = recur(a, a, a, a, k);
        out[end] = a;
    }
    b = d = a;
    for (int j = length - 2; j >= 0; --j) {
        double z = recur(out[j * step], a, b, d, k);
        d = b;
        b = a;
        a = z;
        out[j * step] = z;
    }
}
RF_HD inline float sample(const double *p, int x, int y, const Config &c) {
    float fx = clip((float(x) + .5f) / float(c.scale) - .5f, float(c.sw - 1));
    float fy = clip((float(y) + .5f) / float(c.scale) - .5f, float(c.sh - 1));
    int ix = int(fx), iy = int(fy), jx = clampi(ix + 1, c.sw), jy = clampi(iy + 1, c.sh);
    fx -= ix;
    fy -= iy;
    float a = float(p[iy * c.sw + ix]) * (1 - fx) + float(p[iy * c.sw + jx]) * fx;
    float b = float(p[jy * c.sw + ix]) * (1 - fx) + float(p[jy * c.sw + jx]) * fx;
    return a * (1 - fy) + b * fy;
}
RF_HD inline float score(float value, int x, int y, const double *planes, const Config &c, const Stats &s) {
    float v = normalized(value, s), sum = 0;
    for (int k = 0; k < c.count; ++k) {
        if (c.coeff[k].b == 0)
            sum += 0.6931471805599453f;
        else {
            float g = sample(planes + size_t(k) * c.sw * c.sh, x, y, c);
            // Bound near-zero illumination so dark/impulse frames stay finite.
            if (g > 0)
                sum += fastLog(1.f + v / (g < 1e-12f ? 1e-12f : g));
        }
    }
    return sum / float(c.count);
}
// Approximate the multi-scale illumination with its harmonic mean on the
// small grid. Full-resolution scoring then needs one ratio/log rather than three.
RF_HD inline float effectiveIllumination(const double *planes, size_t i, size_t count, const Config &c) {
    float inverse = 0;
    int positive = 0;
    for (int k = 0; k < c.count; ++k)
        if (c.coeff[k].b != 0) {
            float g = float(planes[size_t(k) * count + i]);
            inverse += 1.f / (g > 1e-12f ? g : 1e-12f);
            ++positive;
        }
    return positive ? float(positive) / inverse : 1.f;
}
RF_HD inline float quickScore(float value, int x, int y, const double *planes, const Config &c, const Stats &s) {
    float g = sample(planes, x, y, c), v = normalized(value, s);
    return fastLog(1.f + v / (g > 1e-12f ? g : 1e-12f));
}
RF_HD inline int bin(float v, const Stats &s) {
    float t = s.hi > s.lo ? (v - s.lo) * (4095.f / (s.hi - s.lo)) : 0;
    return clampi(int(clip(t, 4095.f)), 4096);
}
RF_HD inline void percentile(const unsigned *hist, Stats &s, const Config &c) {
    double count = c.quick ? double(c.sw) * c.sh : double(c.w) * c.h;
    unsigned low = unsigned(count * c.lower + .5), high = unsigned(count * c.upper + .5), sum = 0;
    int a = 0, b = 4095;
    for (; a < 4095; ++a) {
        sum += hist[a];
        if (sum > low)
            break;
    }
    sum = 0;
    for (; b > 0; --b) {
        sum += hist[b];
        if (sum > high)
            break;
    }
    float step = (s.hi - s.lo) / 4095.f;
    s.balanceLo = s.lo + a * step;
    float hi = s.lo + b * step;
    s.balanceGain = hi > s.balanceLo && !(c.quick && s.uniform) ? 1.f / (hi - s.balanceLo) : 0;
}
RF_HD inline float balanced(float value, float sc, const Config &c, const Stats &s) {
    if (c.bypass)
        return value;
    float v = s.balanceGain > 0 ? clip((sc - s.balanceLo) * s.balanceGain) : normalized(value, s);
    return c.floating ? v : floorf(v * float(219 << (c.bits - 8)) + float(16 << (c.bits - 8)) + .5f);
}
RF_HD inline float gradient(float a, float b, float d, float e, float f, float g, float h, float i,
                            const Config &c) {
    float gx, gy;
    if (c.op == 1) {
        gx = (d + f + i - a - e - g) * .5f;
        gy = (a + b + d - g - h - i) * .5f;
    } else {
        gx = d + 2.f * f + i - a - 2.f * e - g;
        gy = a + 2.f * b + d - g - 2.f * h - i;
    }
    float v = sqrtf(gx * gx + gy * gy);
    return clip(c.floating ? v : floorf(v + .5f), c.peak);
}
template <class T> RF_HD inline float kirsch(const T *in, ptrdiff_t stride, int x, int y, const Config &c) {
    int l = clampi(x - 1, c.w), r = clampi(x + 1, c.w), u = clampi(y - 1, c.h), d = clampi(y + 1, c.h);
    float a = read(in, stride, l, u, c), b = read(in, stride, x, u, c), cc = read(in, stride, r, u, c),
          e = read(in, stride, l, y, c), f = read(in, stride, r, y, c), g = read(in, stride, l, d, c),
          h = read(in, stride, x, d, c), i = read(in, stride, r, d, c);
    float v1 = fabsf(5 * a + 5 * b + 5 * cc - 3 * e - 3 * f - 3 * g - 3 * h - 3 * i);
    float v2 = fabsf(-3 * a + 5 * b + 5 * cc - 3 * e + 5 * f - 3 * g - 3 * h - 3 * i);
    float v3 = fabsf(-3 * a - 3 * b + 5 * cc - 3 * e + 5 * f - 3 * g - 3 * h + 5 * i);
    float v4 = fabsf(-3 * a - 3 * b - 3 * cc - 3 * e + 5 * f - 3 * g + 5 * h + 5 * i);
    return clip(fmaxf(fmaxf(v1, v2), fmaxf(v3, v4)), c.peak);
}
} // namespace rf
