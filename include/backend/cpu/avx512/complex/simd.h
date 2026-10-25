#pragma once

#include "backend/simd.h"

#include "backend/cpu/detail/complex_simd.h"

namespace scalars::detail_avx512_complex
{
template <class T> struct ops;

template <> struct ops<float>
{
    using packet = detail_complex_simd::complex_packet<float>;
    using packed = detail_complex_simd::packed_registers<float>;

    static packet deinterleave(packed a)
    {
        const auto real =
            _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        const auto imag =
            _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        return {_mm512_permutex2var_ps(a[0], real, a[1]), _mm512_permutex2var_ps(a[0], imag, a[1])};
    }
    static packed interleave(packet a)
    {
        const auto lo = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
        const auto hi =
            _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
        return {
            _mm512_permutex2var_ps(a.real, lo, a.imag), _mm512_permutex2var_ps(a.real, hi, a.imag)};
    }
    static bool division_is_safe(packet a, packet b)
    {
        return detail_complex_simd::division_mask(a, b) == __mmask16(0xffff);
    }
    static __m512 reciprocal(__m512 a) { return _mm512_div_ps(_mm512_set1_ps(1), a); }
};

template <> struct ops<double>
{
    using packet = detail_complex_simd::complex_packet<double>;
    using packed = detail_complex_simd::packed_registers<double>;

    static packet deinterleave(packed a)
    {
        const auto real = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
        const auto imag = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        return {_mm512_permutex2var_pd(a[0], real, a[1]), _mm512_permutex2var_pd(a[0], imag, a[1])};
    }
    static packed interleave(packet a)
    {
        const auto lo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
        const auto hi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
        return {
            _mm512_permutex2var_pd(a.real, lo, a.imag), _mm512_permutex2var_pd(a.real, hi, a.imag)};
    }
    static bool division_is_safe(packet a, packet b)
    {
        return detail_complex_simd::division_mask(a, b) == __mmask8(0xff);
    }
    static __m512d reciprocal(__m512d a) { return _mm512_div_pd(_mm512_set1_pd(1), a); }
};
}  // namespace scalars::detail_avx512_complex

template <>
struct simd<scalars::complex<float>>
    : scalars::detail_complex_simd::complex_simd<float, scalars::detail_avx512_complex::ops<float>>
{
};
template <>
struct simd<scalars::complex<double>> : scalars::detail_complex_simd::complex_simd<double,
                                            scalars::detail_avx512_complex::ops<double>>
{
};
