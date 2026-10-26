#pragma once

#include "backend/simd.h"

#include <bit>
#include <cstdint>

#include "backend/cpu/detail/complex_simd.h"

namespace scalars::detail_neon_complex
{
template <class T> struct ops
{
    using r      = ::simd<T>;
    using packet = detail_complex_simd::complex_packet<T>;
    using packed = detail_complex_simd::packed_registers<T>;

    static packet deinterleave(packed a)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            const auto split = vuzpq_f32(a[0], a[1]);
            return {split.val[0], split.val[1]};
        }
        else
        {
            return {vuzp1q_f64(a[0], a[1]), vuzp2q_f64(a[0], a[1])};
        }
    }
    static packed interleave(packet a)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            const auto packed = vzipq_f32(a.real, a.imag);
            return {packed.val[0], packed.val[1]};
        }
        else
        {
            return {vzip1q_f64(a.real, a.imag), vzip2q_f64(a.real, a.imag)};
        }
    }

    static SCALARS_FORCE_INLINE bool division_is_safe(packet a, packet b)
    {
        bool all_safe;
        if constexpr (std::is_same_v<T, float>)
        {
            const auto bounded = [](float32x4_t v)
            {
                const auto         bits  = vreinterpretq_u32_f32(vabsq_f32(v));
                constexpr uint32_t lower = (127u - 30u) << 23;
                constexpr uint32_t upper = ((127u + 31u) << 23) - 1;
                return vcleq_u32(vsubq_u32(bits, vdupq_n_u32(lower)), vdupq_n_u32(upper - lower));
            };
            const auto ar = bounded(a.real), ai = bounded(a.imag);
            const auto br = bounded(b.real), bi = bounded(b.imag);
            auto       safe = vandq_u32(vandq_u32(ar, ai), vandq_u32(br, bi));
            all_safe        = vminvq_u32(safe) != 0;
            if (!all_safe)
            {
                safe = vandq_u32(
                    vandq_u32(vorrq_u32(ar, vceqzq_f32(a.real)), vorrq_u32(ai, vceqzq_f32(a.imag))),
                    vandq_u32(
                        vorrq_u32(br, vceqzq_f32(b.real)), vorrq_u32(bi, vceqzq_f32(b.imag))));
                safe     = vbicq_u32(safe, vandq_u32(vceqzq_f32(b.real), vceqzq_f32(b.imag)));
                all_safe = vminvq_u32(safe) != 0;
            }
        }
        else
        {
            // The high words contain the complete exponent. Checking all
            // four double components in one uint32 packet halves the common
            // guard's work. Exact-zero handling is only needed on a miss.
            all_safe = false;
            if constexpr (std::endian::native == std::endian::little)
            {
                constexpr uint32_t lower_word      = (1023u - 250u) << 20;
                constexpr uint32_t upper_word      = ((1023u + 251u) << 20) - 1;
                const auto         bounded_nonzero = [](packet v)
                {
                    const auto words = vandq_u32(
                        vuzp2q_u32(vreinterpretq_u32_f64(v.real), vreinterpretq_u32_f64(v.imag)),
                        vdupq_n_u32(0x7fffffffu));
                    return vcleq_u32(vsubq_u32(words, vdupq_n_u32(lower_word)),
                        vdupq_n_u32(upper_word - lower_word));
                };
                all_safe = vminvq_u32(vandq_u32(bounded_nonzero(a), bounded_nonzero(b))) != 0;
            }
            if (!all_safe)
            {
                const auto bounded = [](float64x2_t v)
                {
                    const auto         bits  = vreinterpretq_u64_f64(vabsq_f64(v));
                    constexpr uint64_t lower = (1023ull - 250ull) << 52;
                    constexpr uint64_t upper = ((1023ull + 251ull) << 52) - 1;
                    return vorrq_u64(vceqzq_u64(bits),
                        vcleq_u64(vsubq_u64(bits, vdupq_n_u64(lower)), vdupq_n_u64(upper - lower)));
                };
                auto       safe    = vandq_u64(vandq_u64(bounded(a.real), bounded(a.imag)),
                    vandq_u64(bounded(b.real), bounded(b.imag)));
                const auto nonzero = vorrq_u64(vreinterpretq_u64_f64(vabsq_f64(b.real)),
                    vreinterpretq_u64_f64(vabsq_f64(b.imag)));
                safe               = vbicq_u64(safe, vceqzq_u64(nonzero));
                all_safe           = vminvq_u32(vreinterpretq_u32_u64(safe)) != 0;
            }
        }
        return all_safe;
    }
    static bool all(typename r::mask_t mask)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return vminvq_u32(mask) != 0;
        }
        else
        {
            return vminvq_u32(vreinterpretq_u32_u64(mask)) != 0;
        }
    }
    static typename r::simd_t reciprocal(typename r::simd_t denominator)
    {
        // Eight-bit estimates need two float / three double Newton steps.
        // This preserves component precision without promising correct rounding.
        if constexpr (std::is_same_v<T, float>)
        {
            auto inverse = vrecpeq_f32(denominator);
            inverse      = vmulq_f32(inverse, vrecpsq_f32(denominator, inverse));
            return vmulq_f32(inverse, vrecpsq_f32(denominator, inverse));
        }
        else
        {
            auto inverse = vrecpeq_f64(denominator);
            inverse      = vmulq_f64(inverse, vrecpsq_f64(denominator, inverse));
            inverse      = vmulq_f64(inverse, vrecpsq_f64(denominator, inverse));
            return vmulq_f64(inverse, vrecpsq_f64(denominator, inverse));
        }
    }
};
}  // namespace scalars::detail_neon_complex

template <>
struct simd<scalars::complex<float>>
    : scalars::detail_complex_simd::complex_simd<float, scalars::detail_neon_complex::ops<float>>
{
};
template <>
struct simd<scalars::complex<double>>
    : scalars::detail_complex_simd::complex_simd<double, scalars::detail_neon_complex::ops<double>>
{
};
