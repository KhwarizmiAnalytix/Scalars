#pragma once

#include "backend/simd.h"

#include "backend/cpu/detail/complex_simd.h"

namespace scalars::detail_avx_complex
{
template <class T> struct ops;

// Use AVX instructions only: AVX2 selects this same folder, like real packets.
template <> struct ops<float>
{
    using packet = detail_complex_simd::complex_packet<float>;
    using packed = detail_complex_simd::packed_registers<float>;

    static packet deinterleave(packed a)
    {
        const auto r0 = _mm_shuffle_ps(
            _mm256_castps256_ps128(a[0]), _mm256_extractf128_ps(a[0], 1), _MM_SHUFFLE(2, 0, 2, 0));
        const auto i0 = _mm_shuffle_ps(
            _mm256_castps256_ps128(a[0]), _mm256_extractf128_ps(a[0], 1), _MM_SHUFFLE(3, 1, 3, 1));
        const auto r1 = _mm_shuffle_ps(
            _mm256_castps256_ps128(a[1]), _mm256_extractf128_ps(a[1], 1), _MM_SHUFFLE(2, 0, 2, 0));
        const auto i1 = _mm_shuffle_ps(
            _mm256_castps256_ps128(a[1]), _mm256_extractf128_ps(a[1], 1), _MM_SHUFFLE(3, 1, 3, 1));
        return {_mm256_insertf128_ps(_mm256_castps128_ps256(r0), r1, 1),
            _mm256_insertf128_ps(_mm256_castps128_ps256(i0), i1, 1)};
    }
    static packed interleave(packet a)
    {
        const auto lo = _mm256_unpacklo_ps(a.real, a.imag);
        const auto hi = _mm256_unpackhi_ps(a.real, a.imag);
        return {_mm256_permute2f128_ps(lo, hi, 0x20), _mm256_permute2f128_ps(lo, hi, 0x31)};
    }
    static bool division_is_safe(packet a, packet b)
    {
        return _mm256_movemask_ps(detail_complex_simd::division_mask(a, b)) == 0xff;
    }
    static __m256 reciprocal(__m256 a) { return _mm256_div_ps(_mm256_set1_ps(1), a); }
};

template <> struct ops<double>
{
    using packet = detail_complex_simd::complex_packet<double>;
    using packed = detail_complex_simd::packed_registers<double>;

    static packet deinterleave(packed a)
    {
        const auto r0 =
            _mm_unpacklo_pd(_mm256_castpd256_pd128(a[0]), _mm256_extractf128_pd(a[0], 1));
        const auto i0 =
            _mm_unpackhi_pd(_mm256_castpd256_pd128(a[0]), _mm256_extractf128_pd(a[0], 1));
        const auto r1 =
            _mm_unpacklo_pd(_mm256_castpd256_pd128(a[1]), _mm256_extractf128_pd(a[1], 1));
        const auto i1 =
            _mm_unpackhi_pd(_mm256_castpd256_pd128(a[1]), _mm256_extractf128_pd(a[1], 1));
        return {_mm256_insertf128_pd(_mm256_castpd128_pd256(r0), r1, 1),
            _mm256_insertf128_pd(_mm256_castpd128_pd256(i0), i1, 1)};
    }
    static packed interleave(packet a)
    {
        const auto lo = _mm256_unpacklo_pd(a.real, a.imag);
        const auto hi = _mm256_unpackhi_pd(a.real, a.imag);
        return {_mm256_permute2f128_pd(lo, hi, 0x20), _mm256_permute2f128_pd(lo, hi, 0x31)};
    }
    static bool division_is_safe(packet a, packet b)
    {
        return _mm256_movemask_pd(detail_complex_simd::division_mask(a, b)) == 0xf;
    }
    static __m256d reciprocal(__m256d a) { return _mm256_div_pd(_mm256_set1_pd(1), a); }
};
}  // namespace scalars::detail_avx_complex

template <>
struct simd<scalars::complex<float>>
    : scalars::detail_complex_simd::complex_simd<float, scalars::detail_avx_complex::ops<float>>
{
};
template <>
struct simd<scalars::complex<double>>
    : scalars::detail_complex_simd::complex_simd<double, scalars::detail_avx_complex::ops<double>>
{
};
