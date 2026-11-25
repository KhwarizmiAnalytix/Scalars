#pragma once

// Transcendental functions of the x86 packets, as overloads on the vector type
// in scalars::x86_math (exp(__m256), exp(__m256d), ...).
//
// They forward to an external vector math provider when one is selected:
// Intel SVML or SLEEF (backend/cpu/<tier>/svml.h), or the SVML intrinsics of
// the compiler (MSVC, Intel). Otherwise, or when SCALARS_HAS_NATIVE_MATH is
// set, they use the vectorized port of Eigen's packet math in
// backend/cpu/detail/vector_math.h.

#include "common/intrinsics.h"
#include "scalars_macros.h"

#if !defined(SCALARS_HAS_NATIVE_MATH)
#define SCALARS_HAS_NATIVE_MATH 0
#endif

#if SCALARS_HAS_NATIVE_MATH ||                                                                     \
    (!SCALARS_HAS_SVML && !SCALARS_HAS_SLEEF && !SCALARS_COMPILER_SUPPORTS_SVML_EXTENSIONS &&      \
     !defined(__INTEL_COMPILER) && !defined(__INTEL_LLVM_COMPILER))
#define SCALARS_X86_NATIVE_MATH 1
#else
#define SCALARS_X86_NATIVE_MATH 0
#endif

#if SCALARS_X86_NATIVE_MATH
#include "backend/cpu/detail/vector_math.h"
#include "backend/cpu/x86/vector_math_ops.h"
#elif SCALARS_HAS_SVML || SCALARS_HAS_SLEEF
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-calling-convention"
#endif
#if SCALARS_HAS_AVX512
#include "backend/cpu/avx512/svml.h"
#elif SCALARS_HAS_AVX || SCALARS_HAS_AVX2
#include "backend/cpu/avx/svml.h"
#elif SCALARS_HAS_SSE
#include "backend/cpu/sse/svml.h"
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#endif

#if SCALARS_HAS_AVX512
#define SCALARS_X86_PREFIX _mm512
#define SCALARS_X86_PS __m512
#define SCALARS_X86_PD __m512d
#elif SCALARS_HAS_AVX || SCALARS_HAS_AVX2
#define SCALARS_X86_PREFIX _mm256
#define SCALARS_X86_PS __m256
#define SCALARS_X86_PD __m256d
#elif SCALARS_HAS_SSE
#define SCALARS_X86_PREFIX _mm
#define SCALARS_X86_PS __m128
#define SCALARS_X86_PD __m128d
#endif

#ifdef SCALARS_X86_PREFIX

#define SCALARS_X86_CAT3_(a, b, c) a##b##c
#define SCALARS_X86_CAT3(a, b, c) SCALARS_X86_CAT3_(a, b, c)

#if SCALARS_X86_NATIVE_MATH
#define SCALARS_X86_UNARY(name, impl)                                                              \
    SCALARS_FORCE_INLINE SCALARS_X86_PS name(SCALARS_X86_PS x)                                     \
    {                                                                                              \
        return detail::vmath::packet_math<detail::vmath::ops_f32>::impl(x);                        \
    }                                                                                              \
    SCALARS_FORCE_INLINE SCALARS_X86_PD name(SCALARS_X86_PD x)                                     \
    {                                                                                              \
        return detail::vmath::packet_math<detail::vmath::ops_f64>::impl(x);                        \
    }
#define SCALARS_X86_BINARY(name, impl)                                                             \
    SCALARS_FORCE_INLINE SCALARS_X86_PS name(SCALARS_X86_PS x, SCALARS_X86_PS y)                   \
    {                                                                                              \
        return detail::vmath::packet_math<detail::vmath::ops_f32>::impl(x, y);                     \
    }                                                                                              \
    SCALARS_FORCE_INLINE SCALARS_X86_PD name(SCALARS_X86_PD x, SCALARS_X86_PD y)                   \
    {                                                                                              \
        return detail::vmath::packet_math<detail::vmath::ops_f64>::impl(x, y);                     \
    }
#else
#define SCALARS_X86_UNARY(name, impl)                                                              \
    SCALARS_FORCE_INLINE SCALARS_X86_PS name(SCALARS_X86_PS x)                                     \
    {                                                                                              \
        return SCALARS_X86_CAT3(SCALARS_X86_PREFIX, _##name, _ps)(x);                              \
    }                                                                                              \
    SCALARS_FORCE_INLINE SCALARS_X86_PD name(SCALARS_X86_PD x)                                     \
    {                                                                                              \
        return SCALARS_X86_CAT3(SCALARS_X86_PREFIX, _##name, _pd)(x);                              \
    }
#define SCALARS_X86_BINARY(name, impl)                                                             \
    SCALARS_FORCE_INLINE SCALARS_X86_PS name(SCALARS_X86_PS x, SCALARS_X86_PS y)                   \
    {                                                                                              \
        return SCALARS_X86_CAT3(SCALARS_X86_PREFIX, _##name, _ps)(x, y);                           \
    }                                                                                              \
    SCALARS_FORCE_INLINE SCALARS_X86_PD name(SCALARS_X86_PD x, SCALARS_X86_PD y)                   \
    {                                                                                              \
        return SCALARS_X86_CAT3(SCALARS_X86_PREFIX, _##name, _pd)(x, y);                           \
    }
#endif

namespace scalars::x86_math
{
// clang-format off
SCALARS_X86_UNARY(exp,        exp)
SCALARS_X86_UNARY(expm1,      expm1)
SCALARS_X86_UNARY(exp2,       exp2)
SCALARS_X86_UNARY(exp10,      exp10)
SCALARS_X86_UNARY(log,        log)
SCALARS_X86_UNARY(log1p,      log1p)
SCALARS_X86_UNARY(log2,       log2)
SCALARS_X86_UNARY(log10,      log10)
SCALARS_X86_UNARY(sin,        sin)
SCALARS_X86_UNARY(cos,        cos)
SCALARS_X86_UNARY(tan,        tan)
SCALARS_X86_UNARY(asin,       asin)
SCALARS_X86_UNARY(acos,       acos)
SCALARS_X86_UNARY(atan,       atan)
SCALARS_X86_UNARY(sinh,       sinh)
SCALARS_X86_UNARY(cosh,       cosh)
SCALARS_X86_UNARY(tanh,       tanh)
SCALARS_X86_UNARY(asinh,      asinh)
SCALARS_X86_UNARY(acosh,      acosh)
SCALARS_X86_UNARY(atanh,      atanh)
SCALARS_X86_UNARY(cbrt,       cbrt)
SCALARS_X86_UNARY(cdfnorm,    normal_cdf)
SCALARS_X86_UNARY(cdfnorminv, inv_normal_cdf)
SCALARS_X86_UNARY(invsqrt,    rsqrt)
SCALARS_X86_BINARY(pow,       pow)
SCALARS_X86_BINARY(hypot,     hypot)
// clang-format on

#if SCALARS_X86_NATIVE_MATH
SCALARS_FORCE_INLINE SCALARS_X86_PS trunc(SCALARS_X86_PS x)
{
    return detail::vmath::ops_f32::trunc(x);
}
SCALARS_FORCE_INLINE SCALARS_X86_PD trunc(SCALARS_X86_PD x)
{
    return detail::vmath::ops_f64::trunc(x);
}
#else
SCALARS_X86_UNARY(trunc, trunc)
#endif
}  // namespace scalars::x86_math

#undef SCALARS_X86_BINARY
#undef SCALARS_X86_UNARY
#undef SCALARS_X86_CAT3
#undef SCALARS_X86_CAT3_
#undef SCALARS_X86_PD
#undef SCALARS_X86_PS
#undef SCALARS_X86_PREFIX

#endif  // SCALARS_X86_PREFIX
