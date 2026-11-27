#pragma once

#if SCALARS_HAS_SVML && SCALARS_HAS_AVX512

// Call the SVML variant for the ISA this backend is compiled for instead of the
// generic entry points (e.g. __svml_sinf16), which route through SVML's runtime
// CPU dispatcher. The build already requires AVX-512 F/CD/BW/DQ/VL, which is
// SVML's _z0 (Skylake-AVX512) variant, so the dispatch is redundant.
#define SCALARS_SVML_ISA _z0
#define SCALARS_SVML_ISA_STR "_z0"
#define SCALARS_SVML_CAT_(a, b) a##b
#define SCALARS_SVML_CAT(a, b) SCALARS_SVML_CAT_(a, b)
#define svml_ps(op) SCALARS_SVML_CAT(__svml_##op##f16, SCALARS_SVML_ISA)
#define svml_pd(op) SCALARS_SVML_CAT(__svml_##op##8, SCALARS_SVML_ISA)
#define svml_ps_ha(op) SCALARS_SVML_CAT(__svml_##op##f16_ha, SCALARS_SVML_ISA)
#define svml_pd_ha(op) SCALARS_SVML_CAT(__svml_##op##8_ha, SCALARS_SVML_ISA)
#define svml_ps_mask(op) SCALARS_SVML_CAT(__svml_##op##f16_mask, SCALARS_SVML_ISA)
#define svml_pd_mask(op) SCALARS_SVML_CAT(__svml_##op##8_mask, SCALARS_SVML_ISA)

// Under clang on Windows the entry points are redeclared below as __vectorcall
// under the same symbol names; a default-convention declaration of the same
// symbol would take precedence and reintroduce the by-pointer call.
#if !(defined(_MSC_VER) && defined(__clang__))
extern "C"
{
    __m512  svml_ps(exp)(__m512);
    __m512d svml_pd(exp)(__m512d);
    __m512  svml_ps(expm1)(__m512);
    __m512d svml_pd(expm1)(__m512d);
    __m512  svml_ps(exp2)(__m512);
    __m512d svml_pd(exp2)(__m512d);
    __m512  svml_ps(exp10)(__m512);
    __m512d svml_pd(exp10)(__m512d);
    __m512  svml_ps(log)(__m512);
    __m512d svml_pd(log)(__m512d);
    __m512  svml_ps(log1p)(__m512);
    __m512d svml_pd(log1p)(__m512d);
    __m512  svml_ps(log2)(__m512);
    __m512d svml_pd(log2)(__m512d);
    __m512  svml_ps(log10)(__m512);
    __m512d svml_pd(log10)(__m512d);
    __m512  svml_ps(sin)(__m512);
    __m512d svml_pd(sin)(__m512d);
    __m512  svml_ps(cos)(__m512);
    __m512d svml_pd(cos)(__m512d);
    __m512  svml_ps(tan)(__m512);
    __m512d svml_pd(tan)(__m512d);
    __m512  svml_ps(asin)(__m512);
    __m512d svml_pd(asin)(__m512d);
    __m512  svml_ps(acos)(__m512);
    __m512d svml_pd(acos)(__m512d);
    __m512  svml_ps(atan)(__m512);
    __m512d svml_pd(atan)(__m512d);
    __m512  svml_ps(sinh)(__m512);
    __m512d svml_pd(sinh)(__m512d);
    __m512  svml_ps(cosh)(__m512);
    __m512d svml_pd(cosh)(__m512d);
    __m512  svml_ps(tanh)(__m512);
    __m512d svml_pd(tanh)(__m512d);
    __m512  svml_ps(asinh)(__m512);
    __m512d svml_pd(asinh)(__m512d);
    __m512  svml_ps(acosh)(__m512);
    __m512d svml_pd(acosh)(__m512d);
    __m512  svml_ps(atanh)(__m512);
    __m512d svml_pd(atanh)(__m512d);
    __m512  svml_ps(cbrt)(__m512);
    __m512d svml_pd(cbrt)(__m512d);
    __m512  svml_ps(cdfnorm)(__m512);
    __m512d svml_pd(cdfnorm)(__m512d);
    __m512  svml_ps(cdfnorminv)(__m512);
    __m512d svml_pd(cdfnorminv)(__m512d);
    __m512  svml_ps(trunc)(__m512);
    __m512d svml_pd(trunc)(__m512d);
    __m512  svml_ps(invsqrt)(__m512);
    __m512d svml_pd(invsqrt)(__m512d);
    __m512  svml_ps(pow)(__m512, __m512);
    __m512d svml_pd(pow)(__m512d, __m512d);
    __m512  svml_ps_ha(pow)(__m512, __m512);
    __m512d svml_pd_ha(pow)(__m512d, __m512d);
    __m512  svml_ps(hypot)(__m512, __m512);
    __m512d svml_pd(hypot)(__m512d, __m512d);
    __m512  svml_ps_ha(hypot)(__m512, __m512);
    __m512d svml_pd_ha(hypot)(__m512d, __m512d);
#if defined(__SCALARS_COMPILER_MSVC__) && _MSC_VER <= 1920
    __m512  svml_ps(ceil)(__m512);
    __m512d svml_pd(ceil)(__m512d);
    __m512  svml_ps(floor)(__m512);
    __m512d svml_pd(floor)(__m512d);
#endif
}
#endif  // !(defined(_MSC_VER) && defined(__clang__))

// MSVC provides these names as intrinsics once the compiler probe succeeds.
// Defining them is error C2169. simd.h then calls the compiler entry points.
#if !defined(_MSC_VER) || !SCALARS_COMPILER_SUPPORTS_SVML_EXTENSIONS
#if defined(_MSC_VER) && defined(__clang__)

// Intel SVML takes and returns vectors in registers (__vectorcall). Under
// clang SCALARS_VECTORCALL is empty, so the reinterpret_cast wrapper in the
// MSVC branch below would call SVML with the default x64 convention, which
// passes the vector by pointer in rcx; SVML then computes on whatever value
// is left in its argument register. Declare each entry point __vectorcall
// instead; the \01 prefix keeps the undecorated DLL symbol (no @@N suffix).
#define SVML_FUNCTION_ONE_ARG(op)                                                                  \
    extern "C" __m512 __vectorcall scalars_svml_##op##f16(__m512) __asm__(                         \
        "\01"                                                                                      \
        "__svml_" #op "f16" SCALARS_SVML_ISA_STR);                                                 \
    extern "C" __m512d __vectorcall scalars_svml_##op##8(__m512d) __asm__(                         \
        "\01"                                                                                      \
        "__svml_" #op "8" SCALARS_SVML_ISA_STR);                                                   \
    SCALARS_FORCE_INLINE __m512 __vectorcall _mm512_##op##_ps(__m512 x)                            \
    {                                                                                              \
        return scalars_svml_##op##f16(x);                                                          \
    }                                                                                              \
    SCALARS_FORCE_INLINE __m512d __vectorcall _mm512_##op##_pd(__m512d x)                          \
    {                                                                                              \
        return scalars_svml_##op##8(x);                                                            \
    }

#define SVML_FUNCTION_TWO_ARGS(op)                                                                 \
    extern "C" __m512 __vectorcall scalars_svml_##op##f16(__m512, __m512) __asm__(                 \
        "\01"                                                                                      \
        "__svml_" #op "f16" SCALARS_SVML_ISA_STR);                                                 \
    extern "C" __m512d __vectorcall scalars_svml_##op##8(__m512d, __m512d) __asm__(                \
        "\01"                                                                                      \
        "__svml_" #op "8" SCALARS_SVML_ISA_STR);                                                   \
    SCALARS_FORCE_INLINE __m512 __vectorcall _mm512_##op##_ps(__m512 x, __m512 y)                  \
    {                                                                                              \
        return scalars_svml_##op##f16(x, y);                                                       \
    }                                                                                              \
    SCALARS_FORCE_INLINE __m512d __vectorcall _mm512_##op##_pd(__m512d x, __m512d y)               \
    {                                                                                              \
        return scalars_svml_##op##8(x, y);                                                         \
    }

#elif defined(_MSC_VER)

#define SVML_FUNCTION_ONE_ARG(op)                                                                  \
    SCALARS_FORCE_INLINE __m512 SCALARS_VECTORCALL _mm512_##op##_ps(__m512 x)                      \
    {                                                                                              \
        return reinterpret_cast<__m512(SCALARS_VECTORCALL*)(__m512)>(svml_ps(op))(x);              \
    }                                                                                              \
                                                                                                   \
    SCALARS_FORCE_INLINE __m512d SCALARS_VECTORCALL _mm512_##op##_pd(__m512d x)                    \
    {                                                                                              \
        return reinterpret_cast<__m512d(SCALARS_VECTORCALL*)(__m512d)>(svml_pd(op))(x);            \
    }

#define SVML_FUNCTION_TWO_ARGS(op)                                                                 \
    SCALARS_FORCE_INLINE __m512 SCALARS_VECTORCALL _mm512_##op##_ps(__m512 x, __m512 y)            \
    {                                                                                              \
        return reinterpret_cast<__m512(SCALARS_VECTORCALL*)(__m512, __m512)>(svml_ps(op))(x, y);   \
    }                                                                                              \
                                                                                                   \
    SCALARS_FORCE_INLINE __m512d SCALARS_VECTORCALL _mm512_##op##_pd(__m512d x, __m512d y)         \
    {                                                                                              \
        return reinterpret_cast<__m512d(SCALARS_VECTORCALL*)(__m512d, __m512d)>(svml_pd(op))(      \
            x, y);                                                                                 \
    }

#else  // !_MSC_VER

#define SVML_FUNCTION_ONE_ARG(op)                                                                  \
    SCALARS_FORCE_INLINE __m512 SCALARS_VECTORCALL _mm512_##op##_ps(__m512 x)                      \
    {                                                                                              \
        return svml_ps(op)(x);                                                                     \
    }                                                                                              \
                                                                                                   \
    SCALARS_FORCE_INLINE __m512d SCALARS_VECTORCALL _mm512_##op##_pd(__m512d x)                    \
    {                                                                                              \
        return svml_pd(op)(x);                                                                     \
    }

#define SVML_FUNCTION_TWO_ARGS(op)                                                                 \
    SCALARS_FORCE_INLINE __m512 SCALARS_VECTORCALL _mm512_##op##_ps(__m512 x, __m512 y)            \
    {                                                                                              \
        return svml_ps(op)(x, y);                                                                  \
    }                                                                                              \
                                                                                                   \
    SCALARS_FORCE_INLINE __m512d SCALARS_VECTORCALL _mm512_##op##_pd(__m512d x, __m512d y)         \
    {                                                                                              \
        return svml_pd(op)(x, y);                                                                  \
    }

#endif  // defined(_MSC_VER)

SVML_FUNCTION_ONE_ARG(exp)
SVML_FUNCTION_ONE_ARG(expm1)
SVML_FUNCTION_ONE_ARG(exp2)
SVML_FUNCTION_ONE_ARG(exp10)
SVML_FUNCTION_ONE_ARG(log)
SVML_FUNCTION_ONE_ARG(log1p)
SVML_FUNCTION_ONE_ARG(log2)
SVML_FUNCTION_ONE_ARG(log10)
SVML_FUNCTION_ONE_ARG(sin)
SVML_FUNCTION_ONE_ARG(cos)
SVML_FUNCTION_ONE_ARG(tan)
SVML_FUNCTION_ONE_ARG(asin)
SVML_FUNCTION_ONE_ARG(acos)
SVML_FUNCTION_ONE_ARG(atan)
SVML_FUNCTION_ONE_ARG(sinh)
SVML_FUNCTION_ONE_ARG(cosh)
SVML_FUNCTION_ONE_ARG(tanh)
SVML_FUNCTION_ONE_ARG(asinh)
SVML_FUNCTION_ONE_ARG(acosh)
SVML_FUNCTION_ONE_ARG(atanh)
SVML_FUNCTION_ONE_ARG(cbrt)
SVML_FUNCTION_ONE_ARG(cdfnorm)
SVML_FUNCTION_ONE_ARG(cdfnorminv)
SVML_FUNCTION_ONE_ARG(trunc)
SVML_FUNCTION_ONE_ARG(invsqrt)
SVML_FUNCTION_TWO_ARGS(pow)
SVML_FUNCTION_TWO_ARGS(hypot)

#if defined(__SCALARS_COMPILER_MSVC__) && _MSC_VER <= 1920
SVML_FUNCTION_ONE_ARG(ceil)
SVML_FUNCTION_ONE_ARG(floor)
#endif

#undef SVML_FUNCTION_TWO_ARGS_HA
#undef SVML_FUNCTION_TWO_ARGS
#undef SVML_FUNCTION_ONE_ARG
#undef svml_ps
#undef svml_pd
#undef svml_ps_ha
#undef svml_pd_ha
#undef svml_ps_mask
#undef svml_pd_mask
#undef SCALARS_SVML_CAT
#undef SCALARS_SVML_CAT_
#undef SCALARS_SVML_ISA
#undef SCALARS_SVML_ISA_STR

#endif  // !MSVC or the compiler does not already provide the SVML intrinsics
#elif SCALARS_HAS_SLEEF && SCALARS_HAS_AVX512

#include <sleef.h>

#include "math/normal_cdf.h"  // scalars::inv_normalcdf for cdfnorminv below

// clang-format off

// AVX-512 SLEEF wrappers: float=16 lanes (f16), double=8 lanes (d8), accuracy=1 ULP (u10).
// SLEEF uses standard C calling conventions — no MSVC/GCC split required.

#define SVML_FUNCTION_ONE_ARG(op)                                    \
    SCALARS_FORCE_INLINE __m512 _mm512_##op##_ps(__m512 x)     \
    { return Sleef_##op##f16_u10(x); }                               \
                                                                     \
    SCALARS_FORCE_INLINE __m512d _mm512_##op##_pd(__m512d x)   \
    { return Sleef_##op##d8_u10(x); }

#define SVML_FUNCTION_TWO_ARGS(op)                                             \
    SCALARS_FORCE_INLINE __m512 _mm512_##op##_ps(__m512 x, __m512 y)    \
    { return Sleef_##op##f16_u10(x, y); }                                      \
                                                                               \
    SCALARS_FORCE_INLINE __m512d _mm512_##op##_pd(__m512d x, __m512d y) \
    { return Sleef_##op##d8_u10(x, y); }

SVML_FUNCTION_ONE_ARG(exp)
SVML_FUNCTION_ONE_ARG(expm1)
SVML_FUNCTION_ONE_ARG(exp2)
SVML_FUNCTION_ONE_ARG(exp10)
SVML_FUNCTION_ONE_ARG(log)
SVML_FUNCTION_ONE_ARG(log1p)
SVML_FUNCTION_ONE_ARG(log2)
SVML_FUNCTION_ONE_ARG(log10)
SVML_FUNCTION_ONE_ARG(sin)
SVML_FUNCTION_ONE_ARG(cos)
SVML_FUNCTION_ONE_ARG(tan)
SVML_FUNCTION_ONE_ARG(asin)
SVML_FUNCTION_ONE_ARG(acos)
SVML_FUNCTION_ONE_ARG(atan)
SVML_FUNCTION_ONE_ARG(sinh)
SVML_FUNCTION_ONE_ARG(cosh)
SVML_FUNCTION_ONE_ARG(tanh)
SVML_FUNCTION_ONE_ARG(asinh)
SVML_FUNCTION_ONE_ARG(acosh)
SVML_FUNCTION_ONE_ARG(atanh)
SVML_FUNCTION_ONE_ARG(cbrt)
SVML_FUNCTION_TWO_ARGS(pow)

// hypot uses 0.5 ULP accuracy in SLEEF (no u10 variant exists).
SCALARS_FORCE_INLINE __m512  _mm512_hypot_ps(__m512  x, __m512  y) { return Sleef_hypotf16_u05(x, y); }
SCALARS_FORCE_INLINE __m512d _mm512_hypot_pd(__m512d x, __m512d y) { return Sleef_hypotd8_u05(x, y); }

// trunc is mathematically exact — SLEEF uses no ULP suffix.
SCALARS_FORCE_INLINE __m512  _mm512_trunc_ps(__m512  x) { return Sleef_truncf16(x); }
SCALARS_FORCE_INLINE __m512d _mm512_trunc_pd(__m512d x) { return Sleef_truncd8(x); }

// invsqrt: no direct SLEEF equivalent — derived from high-precision sqrt (0.5 ULP).
SCALARS_FORCE_INLINE __m512  _mm512_invsqrt_ps(__m512  x) { return _mm512_div_ps(_mm512_set1_ps(1.0f), Sleef_sqrtf16_u05(x)); }
SCALARS_FORCE_INLINE __m512d _mm512_invsqrt_pd(__m512d x) { return _mm512_div_pd(_mm512_set1_pd(1.0),  Sleef_sqrtd8_u05(x)); }

// cdfnorm: Φ(x) = 0.5 * erfc(-x / sqrt(2)) — derived from SLEEF erfc (1.5 ULP).
SCALARS_FORCE_INLINE __m512  _mm512_cdfnorm_ps(__m512  x) { return _mm512_mul_ps(_mm512_set1_ps(0.5f), Sleef_erfcf16_u15(_mm512_mul_ps(_mm512_set1_ps(-0.7071067811865475f), x))); }
SCALARS_FORCE_INLINE __m512d _mm512_cdfnorm_pd(__m512d x) { return _mm512_mul_pd(_mm512_set1_pd(0.5),  Sleef_erfcd8_u15(_mm512_mul_pd(_mm512_set1_pd(-0.7071067811865475244), x))); }

// cdfnorminv: no SLEEF equivalent — scalar loop via scalars::inv_normalcdf.
SCALARS_FORCE_INLINE __m512 _mm512_cdfnorminv_ps(__m512 x)
{
    alignas(64) float tmp[16];
    _mm512_store_ps(tmp, x);
    for (int i = 0; i < 16; ++i) tmp[i] = static_cast<float>(scalars::inv_normalcdf(static_cast<double>(tmp[i])));
    return _mm512_load_ps(tmp);
}
SCALARS_FORCE_INLINE __m512d _mm512_cdfnorminv_pd(__m512d x)
{
    alignas(64) double tmp[8];
    _mm512_store_pd(tmp, x);
    for (int i = 0; i < 8; ++i) tmp[i] = scalars::inv_normalcdf(tmp[i]);
    return _mm512_load_pd(tmp);
}

#undef SVML_FUNCTION_TWO_ARGS
#undef SVML_FUNCTION_ONE_ARG

// clang-format on

#endif  // SCALARS_HAS_SVML || SCALARS_HAS_SLEEF && SCALARS_HAS_AVX512
