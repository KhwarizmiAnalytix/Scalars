#pragma once

// Packed 16-bit storage is widened to the selected float packet before math.
// Architectural reference: ATen/cpu/vec/{vec256,vec512,vec128} reduced floats.
// Independent implementation; no dependency on ATen or its vector classes.

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "types/bfloat16.h"
#include "types/half.h"

namespace scalars::detail_reduced
{
using float_backend = ::simd<float>;
using float_packet  = float_backend::simd_t;

#if SCALARS_HAS_SSE || SCALARS_HAS_AVX || SCALARS_HAS_AVX2 || SCALARS_HAS_AVX512
// MSVC's _mm*_cvtps_ph immediate check rejects _MM_FROUND_NO_EXC (8) as C4556.
#if defined(_MSC_VER)
#define SCALARS_F16C_ROUND _MM_FROUND_TO_NEAREST_INT
#else
#define SCALARS_F16C_ROUND (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
#endif

SCALARS_FORCE_INLINE __m128i select_bits(__m128i mask, __m128i yes, __m128i no)
{
    return _mm_or_si128(_mm_and_si128(mask, yes), _mm_andnot_si128(mask, no));
}

// SSE2 has signed saturation, but no unsigned 32-to-16 pack. Shift the range
// before packing so every 16-bit encoding (including negative values) survives.
SCALARS_FORCE_INLINE __m128i pack_u16(__m128i x)
{
    x = _mm_sub_epi32(x, _mm_set1_epi32(0x8000));
    return _mm_xor_si128(_mm_packs_epi32(x, x), _mm_set1_epi16(short(-32768)));
}

template <class T> SCALARS_FORCE_INLINE __m128 widen4(const uint16_t* from)
{
    __m128i packed = _mm_setzero_si128();
    std::memcpy(&packed, from, 4 * sizeof(uint16_t));
    if constexpr (std::is_same_v<T, half>)
    {
#if defined(__F16C__)
        return _mm_cvtph_ps(packed);
#else
        const auto bits    = _mm_unpacklo_epi16(packed, _mm_setzero_si128());
        const auto mag     = _mm_and_si128(bits, _mm_set1_epi32(0x7fff));
        const auto sign    = _mm_slli_epi32(_mm_and_si128(bits, _mm_set1_epi32(0x8000)), 16);
        auto       normal  = _mm_add_epi32(_mm_slli_epi32(mag, 13), _mm_set1_epi32(0x38000000));
        const auto special = _mm_cmpgt_epi32(mag, _mm_set1_epi32(0x7bff));
        normal = _mm_add_epi32(normal, _mm_and_si128(special, _mm_set1_epi32(0x38000000)));
        const auto subnormal =
            _mm_castps_si128(_mm_mul_ps(_mm_cvtepi32_ps(mag), _mm_set1_ps(0x1p-24f)));
        const auto small = _mm_cmpgt_epi32(_mm_set1_epi32(0x400), mag);
        return _mm_castsi128_ps(_mm_or_si128(sign, select_bits(small, subnormal, normal)));
#endif
    }
    else
    {
        return _mm_castsi128_ps(
            _mm_slli_epi32(_mm_unpacklo_epi16(packed, _mm_setzero_si128()), 16));
    }
}

template <class T> SCALARS_FORCE_INLINE void narrow4(__m128 from, uint16_t* to)
{
    __m128i packed;
    if constexpr (std::is_same_v<T, half>)
    {
#if defined(__F16C__)
        packed = _mm_cvtps_ph(from, SCALARS_F16C_ROUND);
#else
        // Scale into a float range where its rounding also rounds the half
        // mantissa, including half subnormals. This mirrors the scalar format
        // conversion's arithmetic and does not require SSE4 or variable shifts.
        const auto bits = _mm_castps_si128(from);
        const auto mag  = _mm_and_si128(bits, _mm_set1_epi32(0x7fffffff));
        const auto sign = _mm_srli_epi32(_mm_and_si128(bits, _mm_set1_epi32(int(0x80000000u))), 16);
        auto       base = _mm_mul_ps(_mm_castsi128_ps(mag), _mm_set1_ps(0x1p112f));
        base            = _mm_mul_ps(base, _mm_set1_ps(0x1p-110f));
        auto       bias = _mm_and_si128(_mm_slli_epi32(bits, 1), _mm_set1_epi32(int(0xff000000u)));
        const auto smaller = _mm_cmpgt_epi32(_mm_set1_epi32(int(0xf1000000u)),
            _mm_xor_si128(bias, _mm_set1_epi32(int(0x80000000u))));
        bias               = select_bits(smaller, _mm_set1_epi32(0x71000000), bias);
        base               = _mm_add_ps(base,
            _mm_castsi128_ps(_mm_add_epi32(_mm_srli_epi32(bias, 1), _mm_set1_epi32(0x07800000))));
        const auto rounded = _mm_castps_si128(base);
        auto       result =
            _mm_add_epi32(_mm_and_si128(_mm_srli_epi32(rounded, 13), _mm_set1_epi32(0x7c00)),
                _mm_and_si128(rounded, _mm_set1_epi32(0xfff)));
        result = select_bits(
            _mm_cmpgt_epi32(mag, _mm_set1_epi32(0x7f800000)), _mm_set1_epi32(0x7e00), result);
        packed = pack_u16(_mm_or_si128(sign, result));
#endif
    }
    else
    {
        const auto bits = _mm_castps_si128(from);
        const auto lsb  = _mm_and_si128(_mm_srli_epi32(bits, 16), _mm_set1_epi32(1));
        const auto rounded =
            _mm_srli_epi32(_mm_add_epi32(bits, _mm_add_epi32(_mm_set1_epi32(0x7fff), lsb)), 16);
        const auto nan = _mm_castps_si128(_mm_cmpunord_ps(from, from));
        // Match bfloat16::bfloat16(float), which canonicalizes every NaN to 0x7fc0.
        packed = pack_u16(select_bits(nan, _mm_set1_epi32(0x7fc0), rounded));
    }
    std::memcpy(to, &packed, 4 * sizeof(uint16_t));
}
#endif

template <class T> SCALARS_FORCE_INLINE float_packet widen(const uint16_t* from)
{
#if SCALARS_HAS_AVX512
    __m256i packed;
    std::memcpy(&packed, from, sizeof(packed));
    if constexpr (std::is_same_v<T, half>)
        return _mm512_cvtph_ps(packed);
    else
        return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
#elif SCALARS_HAS_AVX || SCALARS_HAS_AVX2
    if constexpr (std::is_same_v<T, half>)
    {
#if defined(__F16C__)
        __m128i packed;
        std::memcpy(&packed, from, sizeof(packed));
        return _mm256_cvtph_ps(packed);
#else
        return _mm256_insertf128_ps(
            _mm256_castps128_ps256(widen4<T>(from)), widen4<T>(from + 4), 1);
#endif
    }
    else
    {
#if SCALARS_HAS_AVX2
        __m128i packed;
        std::memcpy(&packed, from, sizeof(packed));
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
#else
        return _mm256_insertf128_ps(
            _mm256_castps128_ps256(widen4<T>(from)), widen4<T>(from + 4), 1);
#endif
    }
#elif SCALARS_HAS_SSE
    return widen4<T>(from);
#elif SCALARS_HAS_NEON || SCALARS_HAS_SVE
    const auto packed = vld1_u16(from);
    if constexpr (std::is_same_v<T, half>)
    {
        return vcvt_f32_f16(vreinterpret_f16_u16(packed));
    }
    else
    {
        return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(packed), 16));
    }
#endif
}

template <class T> SCALARS_FORCE_INLINE void narrow(float_packet from, uint16_t* to)
{
#if SCALARS_HAS_AVX512
    __m256i packed;
    if constexpr (std::is_same_v<T, half>)
    {
        packed = _mm512_cvtps_ph(from, SCALARS_F16C_ROUND);
    }
    else
    {
        const auto bits    = _mm512_castps_si512(from);
        const auto lsb     = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
        auto       rounded = _mm512_srli_epi32(
            _mm512_add_epi32(bits, _mm512_add_epi32(_mm512_set1_epi32(0x7fff), lsb)), 16);
        rounded = _mm512_mask_mov_epi32(
            rounded, _mm512_cmp_ps_mask(from, from, _CMP_UNORD_Q), _mm512_set1_epi32(0x7fc0));
        packed = _mm512_cvtepi32_epi16(rounded);
    }
    std::memcpy(to, &packed, sizeof(packed));
#elif SCALARS_HAS_AVX || SCALARS_HAS_AVX2
    if constexpr (std::is_same_v<T, half>)
    {
#if defined(__F16C__)
        const auto packed = _mm256_cvtps_ph(from, SCALARS_F16C_ROUND);
        std::memcpy(to, &packed, sizeof(packed));
#else
        narrow4<T>(_mm256_castps256_ps128(from), to);
        narrow4<T>(_mm256_extractf128_ps(from, 1), to + 4);
#endif
    }
    else
    {
#if SCALARS_HAS_AVX2
        const auto bits    = _mm256_castps_si256(from);
        const auto lsb     = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
        auto       rounded = _mm256_srli_epi32(
            _mm256_add_epi32(bits, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), lsb)), 16);
        rounded           = _mm256_blendv_epi8(rounded,
            _mm256_set1_epi32(0x7fc0),
            _mm256_castps_si256(_mm256_cmp_ps(from, from, _CMP_UNORD_Q)));
        const auto packed = _mm256_castsi256_si128(
            _mm256_permute4x64_epi64(_mm256_packus_epi32(rounded, rounded), 0xd8));
        std::memcpy(to, &packed, sizeof(packed));
#else
        narrow4<T>(_mm256_castps256_ps128(from), to);
        narrow4<T>(_mm256_extractf128_ps(from, 1), to + 4);
#endif
    }
#elif SCALARS_HAS_SSE
    narrow4<T>(from, to);
#elif SCALARS_HAS_NEON || SCALARS_HAS_SVE
    uint16x4_t packed;
    if constexpr (std::is_same_v<T, half>)
    {
        packed = vreinterpret_u16_f16(vcvt_f16_f32(from));
    }
    else
    {
#if defined(__ARM_FEATURE_BF16)
        // Use the architectural conversion on BF16 targets, as PyTorch does.
        packed = vreinterpret_u16_bf16(vcvt_bf16_f32(from));
#else
        const auto bits    = vreinterpretq_u32_f32(from);
        const auto lsb     = vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1));
        const auto rounded = vshrq_n_u32(vaddq_u32(bits, vaddq_u32(vdupq_n_u32(0x7fff), lsb)), 16);
        packed = vmovn_u32(vbslq_u32(vceqq_f32(from, from), rounded, vdupq_n_u32(0x7fc0)));
#endif
    }
    vst1_u16(to, packed);
#endif
}
#if SCALARS_HAS_SSE || SCALARS_HAS_AVX || SCALARS_HAS_AVX2 || SCALARS_HAS_AVX512
#undef SCALARS_F16C_ROUND
#endif
}  // namespace scalars::detail_reduced
