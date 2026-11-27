#pragma once

// Operations structs for backend/cpu/detail/vector_math.h on AArch64 NEON,
// which the fixed 128-bit SVE tier shares. Masks are float vectors whose lanes
// are all ones; NEON min/max return NaN when either operand is NaN.

#include <cstdint>

#include "common/intrinsics.h"
#include "scalars_macros.h"

#if defined(__FAST_MATH__) && (defined(__GNUC__) || defined(__clang__))
#define SCALARS_VMATH_NEON_BARRIER(v) __asm__("" : "+w"(v))
#else
#define SCALARS_VMATH_NEON_BARRIER(v) ((void)(v))
#endif

namespace scalars::detail::vmath
{
struct ops_neon_f32
{
    using vec_t    = float32x4_t;
    using scalar_t = float;

    static constexpr int  size    = 4;
    static constexpr bool has_fma = true;

    SCALARS_FORCE_INLINE static uint32x4_t u(vec_t v) { return vreinterpretq_u32_f32(v); }
    SCALARS_FORCE_INLINE static vec_t      f(uint32x4_t v) { return vreinterpretq_f32_u32(v); }

    SCALARS_FORCE_INLINE static vec_t set(float v) { return vdupq_n_f32(v); }
    SCALARS_FORCE_INLINE static vec_t set_bits(std::uint32_t b) { return f(vdupq_n_u32(b)); }
    SCALARS_FORCE_INLINE static vec_t zero() { return vdupq_n_f32(0.0f); }
    SCALARS_FORCE_INLINE static vec_t loadu(const float* p) { return vld1q_f32(p); }
    SCALARS_FORCE_INLINE static void  storeu(float* p, vec_t v) { vst1q_f32(p, v); }

    SCALARS_FORCE_INLINE static vec_t add(vec_t a, vec_t b) { return vaddq_f32(a, b); }
    SCALARS_FORCE_INLINE static vec_t sub(vec_t a, vec_t b) { return vsubq_f32(a, b); }
    SCALARS_FORCE_INLINE static vec_t mul(vec_t a, vec_t b) { return vmulq_f32(a, b); }
    SCALARS_FORCE_INLINE static vec_t div(vec_t a, vec_t b) { return vdivq_f32(a, b); }
    SCALARS_FORCE_INLINE static vec_t sqrt(vec_t a) { return vsqrtq_f32(a); }
    SCALARS_FORCE_INLINE static vec_t fma(vec_t a, vec_t b, vec_t c) { return vfmaq_f32(c, a, b); }
    SCALARS_FORCE_INLINE static vec_t min(vec_t a, vec_t b) { return vminq_f32(a, b); }
    SCALARS_FORCE_INLINE static vec_t max(vec_t a, vec_t b) { return vmaxq_f32(a, b); }
    SCALARS_FORCE_INLINE static vec_t rint(vec_t x) { return vrndnq_f32(x); }
    SCALARS_FORCE_INLINE static vec_t floor(vec_t x) { return vrndmq_f32(x); }
    SCALARS_FORCE_INLINE static vec_t ceil(vec_t x) { return vrndpq_f32(x); }
    SCALARS_FORCE_INLINE static vec_t trunc(vec_t x) { return vrndq_f32(x); }

    SCALARS_FORCE_INLINE static vec_t lt(vec_t a, vec_t b) { return f(vcltq_f32(a, b)); }
    SCALARS_FORCE_INLINE static vec_t le(vec_t a, vec_t b) { return f(vcleq_f32(a, b)); }
    SCALARS_FORCE_INLINE static vec_t eq(vec_t a, vec_t b) { return f(vceqq_f32(a, b)); }
    SCALARS_FORCE_INLINE static vec_t nge(vec_t a, vec_t b)
    {
        return f(vmvnq_u32(vcgeq_f32(a, b)));
    }
    SCALARS_FORCE_INLINE static vec_t isnan(vec_t a) { return f(vmvnq_u32(vceqq_f32(a, a))); }

    SCALARS_FORCE_INLINE static vec_t band(vec_t a, vec_t b) { return f(vandq_u32(u(a), u(b))); }
    SCALARS_FORCE_INLINE static vec_t bor(vec_t a, vec_t b) { return f(vorrq_u32(u(a), u(b))); }
    SCALARS_FORCE_INLINE static vec_t bxor(vec_t a, vec_t b) { return f(veorq_u32(u(a), u(b))); }
    SCALARS_FORCE_INLINE static vec_t bandnot(vec_t a, vec_t b) { return f(vbicq_u32(u(a), u(b))); }
    SCALARS_FORCE_INLINE static vec_t select(vec_t m, vec_t a, vec_t b)
    {
        return vbslq_f32(u(m), a, b);
    }

    SCALARS_FORCE_INLINE static vec_t iadd(vec_t a, vec_t b) { return f(vaddq_u32(u(a), u(b))); }
    SCALARS_FORCE_INLINE static vec_t isub(vec_t a, vec_t b) { return f(vsubq_u32(u(a), u(b))); }
    template <int N> SCALARS_FORCE_INLINE static vec_t shl(vec_t a)
    {
        return f(vshlq_n_u32(u(a), N));
    }
    template <int N> SCALARS_FORCE_INLINE static vec_t shr(vec_t a)
    {
        return f(vshrq_n_u32(u(a), N));
    }

    SCALARS_FORCE_INLINE static bool any(vec_t m) { return vmaxvq_u32(u(m)) != 0; }
    SCALARS_FORCE_INLINE static void barrier(vec_t& v) { SCALARS_VMATH_NEON_BARRIER(v); }
};

struct ops_neon_f64
{
    using vec_t    = float64x2_t;
    using scalar_t = double;

    static constexpr int  size    = 2;
    static constexpr bool has_fma = true;

    SCALARS_FORCE_INLINE static uint64x2_t u(vec_t v) { return vreinterpretq_u64_f64(v); }
    SCALARS_FORCE_INLINE static vec_t      f(uint64x2_t v) { return vreinterpretq_f64_u64(v); }

    SCALARS_FORCE_INLINE static vec_t set(double v) { return vdupq_n_f64(v); }
    SCALARS_FORCE_INLINE static vec_t set_bits(std::uint64_t b) { return f(vdupq_n_u64(b)); }
    SCALARS_FORCE_INLINE static vec_t zero() { return vdupq_n_f64(0.0); }
    SCALARS_FORCE_INLINE static vec_t loadu(const double* p) { return vld1q_f64(p); }
    SCALARS_FORCE_INLINE static void  storeu(double* p, vec_t v) { vst1q_f64(p, v); }

    SCALARS_FORCE_INLINE static vec_t add(vec_t a, vec_t b) { return vaddq_f64(a, b); }
    SCALARS_FORCE_INLINE static vec_t sub(vec_t a, vec_t b) { return vsubq_f64(a, b); }
    SCALARS_FORCE_INLINE static vec_t mul(vec_t a, vec_t b) { return vmulq_f64(a, b); }
    SCALARS_FORCE_INLINE static vec_t div(vec_t a, vec_t b) { return vdivq_f64(a, b); }
    SCALARS_FORCE_INLINE static vec_t sqrt(vec_t a) { return vsqrtq_f64(a); }
    SCALARS_FORCE_INLINE static vec_t fma(vec_t a, vec_t b, vec_t c) { return vfmaq_f64(c, a, b); }
    SCALARS_FORCE_INLINE static vec_t min(vec_t a, vec_t b) { return vminq_f64(a, b); }
    SCALARS_FORCE_INLINE static vec_t max(vec_t a, vec_t b) { return vmaxq_f64(a, b); }
    SCALARS_FORCE_INLINE static vec_t rint(vec_t x) { return vrndnq_f64(x); }
    SCALARS_FORCE_INLINE static vec_t floor(vec_t x) { return vrndmq_f64(x); }
    SCALARS_FORCE_INLINE static vec_t ceil(vec_t x) { return vrndpq_f64(x); }
    SCALARS_FORCE_INLINE static vec_t trunc(vec_t x) { return vrndq_f64(x); }

    SCALARS_FORCE_INLINE static vec_t lt(vec_t a, vec_t b) { return f(vcltq_f64(a, b)); }
    SCALARS_FORCE_INLINE static vec_t le(vec_t a, vec_t b) { return f(vcleq_f64(a, b)); }
    SCALARS_FORCE_INLINE static vec_t eq(vec_t a, vec_t b) { return f(vceqq_f64(a, b)); }
    SCALARS_FORCE_INLINE static vec_t nge(vec_t a, vec_t b)
    {
        return f(veorq_u64(vcgeq_f64(a, b), vdupq_n_u64(~std::uint64_t(0))));
    }
    SCALARS_FORCE_INLINE static vec_t isnan(vec_t a)
    {
        return f(veorq_u64(vceqq_f64(a, a), vdupq_n_u64(~std::uint64_t(0))));
    }

    SCALARS_FORCE_INLINE static vec_t band(vec_t a, vec_t b) { return f(vandq_u64(u(a), u(b))); }
    SCALARS_FORCE_INLINE static vec_t bor(vec_t a, vec_t b) { return f(vorrq_u64(u(a), u(b))); }
    SCALARS_FORCE_INLINE static vec_t bxor(vec_t a, vec_t b) { return f(veorq_u64(u(a), u(b))); }
    SCALARS_FORCE_INLINE static vec_t bandnot(vec_t a, vec_t b) { return f(vbicq_u64(u(a), u(b))); }
    SCALARS_FORCE_INLINE static vec_t select(vec_t m, vec_t a, vec_t b)
    {
        return vbslq_f64(u(m), a, b);
    }

    SCALARS_FORCE_INLINE static vec_t iadd(vec_t a, vec_t b) { return f(vaddq_u64(u(a), u(b))); }
    SCALARS_FORCE_INLINE static vec_t isub(vec_t a, vec_t b) { return f(vsubq_u64(u(a), u(b))); }
    template <int N> SCALARS_FORCE_INLINE static vec_t shl(vec_t a)
    {
        return f(vshlq_n_u64(u(a), N));
    }
    template <int N> SCALARS_FORCE_INLINE static vec_t shr(vec_t a)
    {
        return f(vshrq_n_u64(u(a), N));
    }

    SCALARS_FORCE_INLINE static bool any(vec_t m)
    {
        return vmaxvq_u32(vreinterpretq_u32_u64(u(m))) != 0;
    }
    SCALARS_FORCE_INLINE static void barrier(vec_t& v) { SCALARS_VMATH_NEON_BARRIER(v); }
};

using ops_f32 = ops_neon_f32;
using ops_f64 = ops_neon_f64;
}  // namespace scalars::detail::vmath
