#pragma once

#if SCALARS_HAS_SVML && (SCALARS_HAS_AVX2 || SCALARS_HAS_AVX)

#define svml_ps(op) __svml_##op##f8
#define svml_pd(op) __svml_##op##4
#define svml_ps_ha(op) __svml_##op##f8_ha
#define svml_pd_ha(op) __svml_##op##4_ha
#define svml_ps_mask(op) __svml_##op##f8_mask
#define svml_pd_mask(op) __svml_##op##4_mask
extern "C"
{
    __m256  __svml_expf8(__m256);
    __m256d __svml_exp4(__m256d);
    __m256  __svml_expm1f8(__m256);
    __m256d __svml_expm14(__m256d);
    __m256  __svml_exp2f8(__m256);
    __m256d __svml_exp24(__m256d);
    __m256  __svml_exp10f8(__m256);
    __m256d __svml_exp104(__m256d);
    __m256  __svml_logf8(__m256);
    __m256d __svml_log4(__m256d);
    __m256  __svml_log1pf8(__m256);
    __m256d __svml_log1p4(__m256d);
    __m256  __svml_log2f8(__m256);
    __m256d __svml_log24(__m256d);
    __m256  __svml_log10f8(__m256);
    __m256d __svml_log104(__m256d);
    __m256  __svml_sinf8(__m256);
    __m256d __svml_sin4(__m256d);
    __m256  __svml_cosf8(__m256);
    __m256d __svml_cos4(__m256d);
    __m256  __svml_tanf8(__m256);
    __m256d __svml_tan4(__m256d);
    __m256  __svml_asinf8(__m256);
    __m256d __svml_asin4(__m256d);
    __m256  __svml_acosf8(__m256);
    __m256d __svml_acos4(__m256d);
    __m256  __svml_atanf8(__m256);
    __m256d __svml_atan4(__m256d);
    __m256  __svml_sinhf8(__m256);
    __m256d __svml_sinh4(__m256d);
    __m256  __svml_coshf8(__m256);
    __m256d __svml_cosh4(__m256d);
    __m256  __svml_tanhf8(__m256);
    __m256d __svml_tanh4(__m256d);
    __m256  __svml_asinhf8(__m256);
    __m256d __svml_asinh4(__m256d);
    __m256  __svml_acoshf8(__m256);
    __m256d __svml_acosh4(__m256d);
    __m256  __svml_atanhf8(__m256);
    __m256d __svml_atanh4(__m256d);
    __m256  __svml_cbrtf8(__m256);
    __m256d __svml_cbrt4(__m256d);
    __m256  __svml_cdfnormf8(__m256);
    __m256d __svml_cdfnorm4(__m256d);
    __m256  __svml_cdfnorminvf8(__m256);
    __m256d __svml_cdfnorminv4(__m256d);
    __m256  __svml_truncf8(__m256);
    __m256d __svml_trunc4(__m256d);
    __m256  __svml_invsqrtf8(__m256);
    __m256d __svml_invsqrt4(__m256d);
    __m256  __svml_powf8(__m256, __m256);
    __m256d __svml_pow4(__m256d, __m256d);
    __m256  __svml_powf8_ha(__m256, __m256);
    __m256d __svml_pow4_ha(__m256d, __m256d);
    __m256  __svml_hypotf8(__m256, __m256);
    __m256d __svml_hypot4(__m256d, __m256d);
    __m256  __svml_hypotf8_ha(__m256, __m256);
    __m256d __svml_hypot4_ha(__m256d, __m256d);
}

// MSVC provides these names as intrinsics once the compiler probe succeeds.
// Defining them is error C2169. simd.h then calls the compiler entry points,
// which is how Vectorization uses the ISA when it does not wrap SVML.
#if !defined(_MSC_VER) || !SCALARS_COMPILER_SUPPORTS_SVML_EXTENSIONS
#if defined(_MSC_VER)

#define SVML_FUNCTION_ONE_ARG(op)                                                             \
    SCALARS_FORCE_INLINE __m256 SCALARS_VECTORCALL _mm256_##op##_ps(__m256 x)     \
    {                                                                                         \
        return reinterpret_cast<__m256(SCALARS_VECTORCALL*)(__m256)>(svml_ps(op))(x);   \
    }                                                                                         \
                                                                                              \
    SCALARS_FORCE_INLINE __m256d SCALARS_VECTORCALL _mm256_##op##_pd(__m256d x)   \
    {                                                                                         \
        return reinterpret_cast<__m256d(SCALARS_VECTORCALL*)(__m256d)>(svml_pd(op))(x); \
    }

#define SVML_FUNCTION_TWO_ARGS(op)                                                               \
    SCALARS_FORCE_INLINE __m256 SCALARS_VECTORCALL _mm256_##op##_ps(                 \
        __m256 x, __m256 y)                                                                      \
    {                                                                                            \
        return reinterpret_cast<__m256(SCALARS_VECTORCALL*)(__m256, __m256)>(svml_ps(op))( \
            x, y);                                                                               \
    }                                                                                            \
                                                                                                 \
    SCALARS_FORCE_INLINE __m256d SCALARS_VECTORCALL _mm256_##op##_pd(                \
        __m256d x, __m256d y)                                                                    \
    {                                                                                            \
        return reinterpret_cast<__m256d(SCALARS_VECTORCALL*)(__m256d, __m256d)>(           \
            svml_pd(op))(x, y);                                                                  \
    }

#else  // !_MSC_VER

#define SVML_FUNCTION_ONE_ARG(op)                                  \
    SCALARS_FORCE_INLINE __m256 _mm256_##op##_ps(__m256 x)   \
    {                                                              \
        return svml_ps(op)(x);                                     \
    }                                                              \
                                                                   \
    SCALARS_FORCE_INLINE __m256d _mm256_##op##_pd(__m256d x) \
    {                                                              \
        return svml_pd(op)(x);                                     \
    }

#define SVML_FUNCTION_TWO_ARGS(op)                                            \
    SCALARS_FORCE_INLINE __m256 _mm256_##op##_ps(__m256 x, __m256 y)    \
    {                                                                         \
        return svml_ps(op)(x, y);                                             \
    }                                                                         \
                                                                              \
    SCALARS_FORCE_INLINE __m256d _mm256_##op##_pd(__m256d x, __m256d y) \
    {                                                                         \
        return svml_pd(op)(x, y);                                             \
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
#if defined(_MSC_VER) && !defined(NDEBUG)
// clang-cl Debug builds mis-call Intel SVML two-arg entry points through the
// reinterpret_cast wrapper above; scalar lane fallbacks match std::pow/hypot.
#include <cmath>

SCALARS_FORCE_INLINE __m256 SCALARS_VECTORCALL _mm256_pow_ps(__m256 x, __m256 y)
{
    alignas(32) float xs[8];
    alignas(32) float ys[8];
    alignas(32) float rs[8];
    _mm256_store_ps(xs, x);
    _mm256_store_ps(ys, y);
    for (int i = 0; i < 8; ++i)
        rs[i] = std::pow(xs[i], ys[i]);
    return _mm256_load_ps(rs);
}

SCALARS_FORCE_INLINE __m256d SCALARS_VECTORCALL _mm256_pow_pd(__m256d x, __m256d y)
{
    alignas(32) double xs[4];
    alignas(32) double ys[4];
    alignas(32) double rs[4];
    _mm256_store_pd(xs, x);
    _mm256_store_pd(ys, y);
    for (int i = 0; i < 4; ++i)
        rs[i] = std::pow(xs[i], ys[i]);
    return _mm256_load_pd(rs);
}

SCALARS_FORCE_INLINE __m256 SCALARS_VECTORCALL _mm256_hypot_ps(__m256 x, __m256 y)
{
    alignas(32) float xs[8];
    alignas(32) float ys[8];
    alignas(32) float rs[8];
    _mm256_store_ps(xs, x);
    _mm256_store_ps(ys, y);
    for (int i = 0; i < 8; ++i)
        rs[i] = std::hypot(xs[i], ys[i]);
    return _mm256_load_ps(rs);
}

SCALARS_FORCE_INLINE __m256d SCALARS_VECTORCALL _mm256_hypot_pd(__m256d x, __m256d y)
{
    alignas(32) double xs[4];
    alignas(32) double ys[4];
    alignas(32) double rs[4];
    _mm256_store_pd(xs, x);
    _mm256_store_pd(ys, y);
    for (int i = 0; i < 4; ++i)
        rs[i] = std::hypot(xs[i], ys[i]);
    return _mm256_load_pd(rs);
}
#else
SVML_FUNCTION_TWO_ARGS(pow)
SVML_FUNCTION_TWO_ARGS(hypot)
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

#endif  // !MSVC or the compiler does not already provide the SVML intrinsics
#elif SCALARS_HAS_SLEEF && (SCALARS_HAS_AVX2 || SCALARS_HAS_AVX)

#include <sleef.h>

// clang-format off

// AVX2 SLEEF wrappers: float=8 lanes (f8), double=4 lanes (d4), accuracy=1 ULP (u10).
// SLEEF uses architecture-specific suffixes for 256-bit functions — no generic aliases exist.

#define SVML_FUNCTION_ONE_ARG(op)                                    \
    SCALARS_FORCE_INLINE __m256 _mm256_##op##_ps(__m256 x)     \
    { return Sleef_##op##f8_u10avx2(x); }                            \
                                                                     \
    SCALARS_FORCE_INLINE __m256d _mm256_##op##_pd(__m256d x)   \
    { return Sleef_##op##d4_u10avx2(x); }

#define SVML_FUNCTION_TWO_ARGS(op)                                             \
    SCALARS_FORCE_INLINE __m256 _mm256_##op##_ps(__m256 x, __m256 y)    \
    { return Sleef_##op##f8_u10avx2(x, y); }                                  \
                                                                               \
    SCALARS_FORCE_INLINE __m256d _mm256_##op##_pd(__m256d x, __m256d y) \
    { return Sleef_##op##d4_u10avx2(x, y); }

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

// hypot uses 0.5 ULP accuracy in SLEEF (no u10 variant exists for 256-bit).
SCALARS_FORCE_INLINE __m256  _mm256_hypot_ps(__m256  x, __m256  y) { return Sleef_hypotf8_u05avx2(x, y); }
SCALARS_FORCE_INLINE __m256d _mm256_hypot_pd(__m256d x, __m256d y) { return Sleef_hypotd4_u05avx2(x, y); }

// trunc is mathematically exact — SLEEF uses no ULP suffix.
SCALARS_FORCE_INLINE __m256  _mm256_trunc_ps(__m256  x) { return Sleef_truncf8_avx2(x); }
SCALARS_FORCE_INLINE __m256d _mm256_trunc_pd(__m256d x) { return Sleef_truncd4_avx2(x); }

// invsqrt: no direct SLEEF equivalent — derived from high-precision sqrt (0.5 ULP).
SCALARS_FORCE_INLINE __m256  _mm256_invsqrt_ps(__m256  x) { return _mm256_div_ps(_mm256_set1_ps(1.0f), Sleef_sqrtf8_u05avx2(x)); }
SCALARS_FORCE_INLINE __m256d _mm256_invsqrt_pd(__m256d x) { return _mm256_div_pd(_mm256_set1_pd(1.0),  Sleef_sqrtd4_u05avx2(x)); }

// cdfnorm: Φ(x) = 0.5 * erfc(-x / sqrt(2)) — derived from SLEEF erfc (1.5 ULP).
SCALARS_FORCE_INLINE __m256  _mm256_cdfnorm_ps(__m256  x) { return _mm256_mul_ps(_mm256_set1_ps(0.5f), Sleef_erfcf8_u15avx2(_mm256_mul_ps(_mm256_set1_ps(-0.7071067811865475f), x))); }
SCALARS_FORCE_INLINE __m256d _mm256_cdfnorm_pd(__m256d x) { return _mm256_mul_pd(_mm256_set1_pd(0.5),  Sleef_erfcd4_u15avx2(_mm256_mul_pd(_mm256_set1_pd(-0.7071067811865475244), x))); }

// cdfnorminv: no SLEEF equivalent — scalar loop via scalars::inv_normalcdf.
SCALARS_FORCE_INLINE __m256 _mm256_cdfnorminv_ps(__m256 x)
{
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, x);
    for (int i = 0; i < 8; ++i) tmp[i] = static_cast<float>(scalars::inv_normalcdf(static_cast<double>(tmp[i])));
    return _mm256_load_ps(tmp);
}
SCALARS_FORCE_INLINE __m256d _mm256_cdfnorminv_pd(__m256d x)
{
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, x);
    for (int i = 0; i < 4; ++i) tmp[i] = scalars::inv_normalcdf(tmp[i]);
    return _mm256_load_pd(tmp);
}

#undef SVML_FUNCTION_TWO_ARGS
#undef SVML_FUNCTION_ONE_ARG

// clang-format on

#endif  // SCALARS_HAS_SVML || SCALARS_HAS_SLEEF && (SCALARS_HAS_AVX2 || SCALARS_HAS_AVX)