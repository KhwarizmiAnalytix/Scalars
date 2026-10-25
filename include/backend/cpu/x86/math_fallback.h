#pragma once

// The x86 packet API uses SVML-style names for transcendental functions.
// Supply scalar mappings when no external vector math provider is selected,
// just as the NEON backend does. Arithmetic and format conversions remain SIMD.
#if !SCALARS_HAS_SVML && !SCALARS_HAS_SLEEF && !SCALARS_COMPILER_SUPPORTS_SVML_EXTENSIONS &&     \
    !defined(__INTEL_COMPILER) && !defined(__INTEL_LLVM_COMPILER)

#include <cmath>
#include <cstring>

#include "math/normal_cdf.h"

#define SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, name, expression)                    \
    SCALARS_FORCE_INLINE vec prefix##_##name##_##suffix(vec from)                                  \
    {                                                                                              \
        scalar lanes[count];                                                                       \
        std::memcpy(lanes, &from, sizeof(from));                                                   \
        for (auto& x : lanes)                                                                      \
            x = static_cast<scalar>(expression);                                                   \
        std::memcpy(&from, lanes, sizeof(from));                                                   \
        return from;                                                                               \
    }
#define SCALARS_X86_BINARY(prefix, suffix, vec, scalar, count, name)                               \
    SCALARS_FORCE_INLINE vec prefix##_##name##_##suffix(vec from, vec other)                       \
    {                                                                                              \
        scalar lanes[count], rhs[count];                                                           \
        std::memcpy(lanes, &from, sizeof(from));                                                   \
        std::memcpy(rhs, &other, sizeof(other));                                                   \
        for (int i = 0; i < count; ++i)                                                            \
            lanes[i] = std::name(lanes[i], rhs[i]);                                                \
        std::memcpy(&from, lanes, sizeof(from));                                                   \
        return from;                                                                               \
    }
#define SCALARS_X86_MATH(prefix, suffix, vec, scalar, count)                                       \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, exp, std::exp(x))                        \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, expm1, std::expm1(x))                    \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, exp2, std::exp2(x))                      \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, exp10, std::pow(scalar(10), x))          \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, log, std::log(x))                        \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, log1p, std::log1p(x))                    \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, log2, std::log2(x))                      \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, log10, std::log10(x))                    \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, sin, std::sin(x))                        \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, cos, std::cos(x))                        \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, tan, std::tan(x))                        \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, asin, std::asin(x))                      \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, acos, std::acos(x))                      \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, atan, std::atan(x))                      \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, sinh, std::sinh(x))                      \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, cosh, std::cosh(x))                      \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, tanh, std::tanh(x))                      \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, asinh, std::asinh(x))                    \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, acosh, std::acosh(x))                    \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, atanh, std::atanh(x))                    \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, cbrt, std::cbrt(x))                      \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, trunc, std::trunc(x))                    \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, invsqrt, scalar(1) / std::sqrt(x))       \
    SCALARS_X86_UNARY(prefix, suffix, vec, scalar, count, cdfnorm, scalars::normalcdf(double(x)))  \
    SCALARS_X86_UNARY(                                                                             \
        prefix, suffix, vec, scalar, count, cdfnorminv, scalars::inv_normalcdf(double(x)))         \
    SCALARS_X86_BINARY(prefix, suffix, vec, scalar, count, pow)                                    \
    SCALARS_X86_BINARY(prefix, suffix, vec, scalar, count, hypot)

#if SCALARS_HAS_AVX512
SCALARS_X86_MATH(_mm512, ps, __m512, float, 16)
SCALARS_X86_MATH(_mm512, pd, __m512d, double, 8)
#elif SCALARS_HAS_AVX || SCALARS_HAS_AVX2
SCALARS_X86_MATH(_mm256, ps, __m256, float, 8)
SCALARS_X86_MATH(_mm256, pd, __m256d, double, 4)
#elif SCALARS_HAS_SSE
SCALARS_X86_MATH(_mm, ps, __m128, float, 4)
SCALARS_X86_MATH(_mm, pd, __m128d, double, 2)
#endif

#undef SCALARS_X86_MATH
#undef SCALARS_X86_BINARY
#undef SCALARS_X86_UNARY
#endif
