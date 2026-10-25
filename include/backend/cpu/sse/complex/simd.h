#pragma once

#include "backend/simd.h"

#include "backend/cpu/detail/complex_simd.h"

namespace scalars::detail_sse_complex
{
template <class T> struct ops;

template <> struct ops<float>
{
    using packet = detail_complex_simd::complex_packet<float>;
    using packed = detail_complex_simd::packed_registers<float>;

    static packet deinterleave(packed a)
    {
        return {_mm_shuffle_ps(a[0], a[1], _MM_SHUFFLE(2, 0, 2, 0)),
            _mm_shuffle_ps(a[0], a[1], _MM_SHUFFLE(3, 1, 3, 1))};
    }
    static packed interleave(packet a)
    {
        return {_mm_unpacklo_ps(a.real, a.imag), _mm_unpackhi_ps(a.real, a.imag)};
    }
    static bool division_is_safe(packet a, packet b)
    {
        return _mm_movemask_ps(detail_complex_simd::division_mask(a, b)) == 0xf;
    }
    static __m128 reciprocal(__m128 a) { return _mm_div_ps(_mm_set1_ps(1), a); }
};

template <> struct ops<double>
{
    using packet = detail_complex_simd::complex_packet<double>;
    using packed = detail_complex_simd::packed_registers<double>;

    static packet deinterleave(packed a)
    {
        return {_mm_unpacklo_pd(a[0], a[1]), _mm_unpackhi_pd(a[0], a[1])};
    }
    static packed interleave(packet a)
    {
        return {_mm_unpacklo_pd(a.real, a.imag), _mm_unpackhi_pd(a.real, a.imag)};
    }
    static bool division_is_safe(packet a, packet b)
    {
        return _mm_movemask_pd(detail_complex_simd::division_mask(a, b)) == 0x3;
    }
    static __m128d reciprocal(__m128d a) { return _mm_div_pd(_mm_set1_pd(1), a); }
};
}  // namespace scalars::detail_sse_complex

template <>
struct simd<scalars::complex<float>>
    : scalars::detail_complex_simd::complex_simd<float, scalars::detail_sse_complex::ops<float>>
{
};
template <>
struct simd<scalars::complex<double>>
    : scalars::detail_complex_simd::complex_simd<double, scalars::detail_sse_complex::ops<double>>
{
};
