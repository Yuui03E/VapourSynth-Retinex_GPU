// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "common.hpp"
#if defined(_M_X64) || defined(__x86_64__) || defined(__SSE2__)
#define RF_SSE2 1
#include <emmintrin.h>
#include <type_traits>
namespace rf {
template <class T> inline __m128 load4(const T *p, const Config &c) {
    __m128 v;
    if constexpr (std::is_same_v<T, float>)
        v = _mm_loadu_ps(p);
    else if constexpr (sizeof(T) == 2)
        v = _mm_cvtepi32_ps(
            _mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(p)), _mm_setzero_si128()));
    else {
        int data;
        std::memcpy(&data, p, 4);
        __m128i a = _mm_unpacklo_epi8(_mm_cvtsi32_si128(data), _mm_setzero_si128());
        v = _mm_cvtepi32_ps(_mm_unpacklo_epi16(a, _mm_setzero_si128()));
    }
    return _mm_and_ps(_mm_and_ps(_mm_cmpge_ps(v, _mm_setzero_ps()), _mm_cmplt_ps(v, _mm_set1_ps(INFINITY))),
                      _mm_min_ps(v, _mm_set1_ps(c.peak)));
}
inline __m128 gradient4(const float *a, const float *b, const float *d, int x, const Config &c) {
    __m128 al = _mm_loadu_ps(a + x - 1), am = _mm_loadu_ps(a + x), ar = _mm_loadu_ps(a + x + 1),
           bl = _mm_loadu_ps(b + x - 1), br = _mm_loadu_ps(b + x + 1), dl = _mm_loadu_ps(d + x - 1),
           dm = _mm_loadu_ps(d + x), dr = _mm_loadu_ps(d + x + 1);
    __m128 gx, gy;
    if (c.op == 1) {
        gx = _mm_mul_ps(_mm_sub_ps(_mm_sub_ps(_mm_sub_ps(_mm_add_ps(_mm_add_ps(ar, br), dr), al), bl), dl),
                        _mm_set1_ps(.5f));
        gy = _mm_mul_ps(_mm_sub_ps(_mm_sub_ps(_mm_sub_ps(_mm_add_ps(_mm_add_ps(al, am), ar), dl), dm), dr),
                        _mm_set1_ps(.5f));
    } else {
        __m128 two = _mm_set1_ps(2);
        gx = _mm_sub_ps(
            _mm_sub_ps(_mm_sub_ps(_mm_add_ps(_mm_add_ps(ar, _mm_mul_ps(two, br)), dr), al), _mm_mul_ps(two, bl)),
            dl);
        gy = _mm_sub_ps(
            _mm_sub_ps(_mm_sub_ps(_mm_add_ps(_mm_add_ps(al, _mm_mul_ps(two, am)), ar), dl), _mm_mul_ps(two, dm)),
            dr);
    }
    __m128 v = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(gx, gx), _mm_mul_ps(gy, gy)));
    if (!c.floating)
        v = _mm_cvtepi32_ps(_mm_cvttps_epi32(_mm_add_ps(v, _mm_set1_ps(.5f))));
    return _mm_min_ps(v, _mm_set1_ps(c.peak));
}
template <class T> inline __m128 kirsch4(const T *src, ptrdiff_t ss, int x, int y, const Config &c) {
    const T *a = src + clampi(y - 1, c.h) * ss + x, *b = src + y * ss + x, *d = src + clampi(y + 1, c.h) * ss + x;
    __m128 al = load4(a - 1, c), am = load4(a, c), ar = load4(a + 1, c), bl = load4(b - 1, c),
           br = load4(b + 1, c), dl = load4(d - 1, c), dm = load4(d, c), dr = load4(d + 1, c);
    __m128 five = _mm_set1_ps(5), three = _mm_set1_ps(3), sign = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    __m128 v1 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(five, al), _mm_mul_ps(five, am)), _mm_mul_ps(five, ar));
    v1 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(_mm_sub_ps(_mm_sub_ps(v1, _mm_mul_ps(three, bl)), _mm_mul_ps(three, br)),
                                          _mm_mul_ps(three, dl)),
                               _mm_mul_ps(three, dm)),
                    _mm_mul_ps(three, dr));
    __m128 v2 =
        _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(-3), al), _mm_mul_ps(five, am)), _mm_mul_ps(five, ar));
    v2 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(_mm_add_ps(_mm_sub_ps(v2, _mm_mul_ps(three, bl)), _mm_mul_ps(five, br)),
                                          _mm_mul_ps(three, dl)),
                               _mm_mul_ps(three, dm)),
                    _mm_mul_ps(three, dr));
    __m128 v3 =
        _mm_add_ps(_mm_sub_ps(_mm_mul_ps(_mm_set1_ps(-3), al), _mm_mul_ps(three, am)), _mm_mul_ps(five, ar));
    v3 = _mm_add_ps(_mm_sub_ps(_mm_sub_ps(_mm_add_ps(_mm_sub_ps(v3, _mm_mul_ps(three, bl)), _mm_mul_ps(five, br)),
                                          _mm_mul_ps(three, dl)),
                               _mm_mul_ps(three, dm)),
                    _mm_mul_ps(five, dr));
    __m128 v4 =
        _mm_sub_ps(_mm_sub_ps(_mm_mul_ps(_mm_set1_ps(-3), al), _mm_mul_ps(three, am)), _mm_mul_ps(three, ar));
    v4 = _mm_add_ps(_mm_add_ps(_mm_sub_ps(_mm_add_ps(_mm_sub_ps(v4, _mm_mul_ps(three, bl)), _mm_mul_ps(five, br)),
                                          _mm_mul_ps(three, dl)),
                               _mm_mul_ps(five, dm)),
                    _mm_mul_ps(five, dr));
    return _mm_min_ps(_mm_set1_ps(c.peak), _mm_max_ps(_mm_max_ps(_mm_and_ps(v1, sign), _mm_and_ps(v2, sign)),
                                                      _mm_max_ps(_mm_and_ps(v3, sign), _mm_and_ps(v4, sign))));
}
inline __m128 log4(__m128 v) {
    __m128i bits = _mm_castps_si128(v), exponent = _mm_sub_epi32(_mm_srli_epi32(bits, 23), _mm_set1_epi32(127));
    __m128 m =
        _mm_castsi128_ps(_mm_or_si128(_mm_and_si128(bits, _mm_set1_epi32(0x7fffff)), _mm_set1_epi32(0x3f800000)));
    __m128 one = _mm_set1_ps(1), z = _mm_div_ps(_mm_sub_ps(m, one), _mm_add_ps(m, one)), z2 = _mm_mul_ps(z, z);
    __m128 series = _mm_add_ps(_mm_set1_ps(1.f / 7.f), _mm_div_ps(z2, _mm_set1_ps(9)));
    series = _mm_add_ps(_mm_set1_ps(1.f / 5.f), _mm_mul_ps(z2, series));
    series = _mm_add_ps(_mm_set1_ps(1.f / 3.f), _mm_mul_ps(z2, series));
    series = _mm_add_ps(one, _mm_mul_ps(z2, series));
    return _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(exponent), _mm_set1_ps(.6931471805599453f)),
                      _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(2), z), series));
}
template <class T>
inline void scoresRow(const T *src, ptrdiff_t ss, int y, float *out, const double *planes, const Config &c,
                      const Stats &s) {
    int x = 0;
    __m128 zero = _mm_setzero_ps(), one = _mm_set1_ps(1);
    float fy = clip((float(y) + .5f) / float(c.scale) - .5f, float(c.sh - 1));
    int iy = int(fy), jy = clampi(iy + 1, c.sh);
    fy -= iy;
    for (; x + 3 < c.w; x += 4) {
        __m128 v = load4(src + y * ss + x, c);
        v = _mm_sub_ps(v, _mm_set1_ps(s.inputLo));
        v = s.inputGain > 0 ? _mm_mul_ps(v, _mm_set1_ps(s.inputGain)) : _mm_div_ps(v, _mm_set1_ps(s.inputSpan));
        v = _mm_min_ps(one, _mm_max_ps(zero, v));
        __m128 fx = _mm_setr_ps(float(x) + .5f, float(x) + 1.5f, float(x) + 2.5f, float(x) + 3.5f);
        fx = _mm_min_ps(
            _mm_set1_ps(float(c.sw - 1)),
            _mm_max_ps(zero, _mm_sub_ps(_mm_div_ps(fx, _mm_set1_ps(float(c.scale))), _mm_set1_ps(.5f))));
        __m128i ix = _mm_cvttps_epi32(fx);
        fx = _mm_sub_ps(fx, _mm_cvtepi32_ps(ix));
        alignas(16) int indices[4];
        _mm_store_si128(reinterpret_cast<__m128i *>(indices), ix);
        int jx[4];
        for (int j = 0; j < 4; ++j)
            jx[j] = clampi(indices[j] + 1, c.sw);
        __m128 sum = zero;
        for (int k = 0; k < c.count; ++k) {
            if (c.coeff[k].b == 0) {
                sum = _mm_add_ps(sum, _mm_set1_ps(.6931471805599453f));
                continue;
            }
            const double *a = planes + size_t(k) * c.sw * c.sh + iy * c.sw,
                         *b = planes + size_t(k) * c.sw * c.sh + jy * c.sw;
            __m128 al = _mm_setr_ps(float(a[indices[0]]), float(a[indices[1]]), float(a[indices[2]]),
                                    float(a[indices[3]]));
            __m128 ar = _mm_setr_ps(float(a[jx[0]]), float(a[jx[1]]), float(a[jx[2]]), float(a[jx[3]]));
            __m128 bl = _mm_setr_ps(float(b[indices[0]]), float(b[indices[1]]), float(b[indices[2]]),
                                    float(b[indices[3]]));
            __m128 br = _mm_setr_ps(float(b[jx[0]]), float(b[jx[1]]), float(b[jx[2]]), float(b[jx[3]]));
            __m128 aa = _mm_add_ps(_mm_mul_ps(al, _mm_sub_ps(one, fx)), _mm_mul_ps(ar, fx));
            __m128 bb = _mm_add_ps(_mm_mul_ps(bl, _mm_sub_ps(one, fx)), _mm_mul_ps(br, fx));
            __m128 g = _mm_add_ps(_mm_mul_ps(aa, _mm_set1_ps(1 - fy)), _mm_mul_ps(bb, _mm_set1_ps(fy)));
            __m128 term = log4(_mm_add_ps(one, _mm_div_ps(v, _mm_max_ps(g, _mm_set1_ps(1e-12f)))));
            sum = _mm_add_ps(sum, _mm_and_ps(_mm_cmpgt_ps(g, zero), term));
        }
        _mm_storeu_ps(out + x, _mm_div_ps(sum, _mm_set1_ps(float(c.count))));
    }
    for (; x < c.w; ++x)
        out[x] = score(read(src, ss, x, y, c), x, y, planes, c, s);
}
} // namespace rf
#endif
