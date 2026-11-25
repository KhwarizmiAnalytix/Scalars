#pragma once

// Operations structs for backend/cpu/detail/vector_math.h on the selected x86
// tier: SSE2 (SSE4.1 when available), AVX/AVX2 and AVX-512F. Masks are vectors
// of the same type with all-ones lanes. min/max put the first argument second
// in the instruction, which returns its second operand when either is NaN, so
// a NaN first argument propagates.

#include <cstdint>

#include "common/intrinsics.h"
#include "scalars_macros.h"

// Keeps -ffast-math from reassociating the error-free transformations; IEEE
// builds do not reorder floating-point operations, so the barrier is empty.
#if defined(__FAST_MATH__) && (defined(__GNUC__) || defined(__clang__))
#if SCALARS_HAS_AVX512
#define SCALARS_VMATH_X86_BARRIER(v) __asm__("" : "+v"(v))
#else
#define SCALARS_VMATH_X86_BARRIER(v) __asm__("" : "+x"(v))
#endif
#else
#define SCALARS_VMATH_X86_BARRIER(v) ((void)(v))
#endif

#ifdef __FMA__
#define SCALARS_VMATH_X86_HAS_FMA true
#else
#define SCALARS_VMATH_X86_HAS_FMA false
#endif

namespace scalars::detail::vmath
{
#if SCALARS_HAS_SSE && !SCALARS_HAS_AVX && !SCALARS_HAS_AVX2 && !SCALARS_HAS_AVX512

// SSE2 without SSE4.1 rounds with the 1.5 * 2^(digits - 1) trick: below it the
// sum has an ulp of one, and above it every value is already integral.
#define SCALARS_VMATH_SSE_ROUNDING(sfx, limit)                                                     \
    SCALARS_FORCE_INLINE static vec_t rint(vec_t x)                                                \
    {                                                                                              \
        const vec_t sign = band(x, set(scalar_t(-0.0)));                                           \
        const vec_t a    = bandnot(x, set(scalar_t(-0.0)));                                        \
        const vec_t m    = set(scalar_t(limit));                                                   \
        const vec_t r    = bor(sub(add(a, m), m), sign);                                           \
        return select(lt(a, m), r, x);                                                             \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t floor(vec_t x)                                               \
    {                                                                                              \
        const vec_t r = rint(x);                                                                   \
        return sub(r, band(_mm_cmpgt_##sfx(r, x), set(scalar_t(1))));                              \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t ceil(vec_t x)                                                \
    {                                                                                              \
        const vec_t r = rint(x);                                                                   \
        const vec_t c = add(r, band(lt(r, x), set(scalar_t(1))));                                  \
        return bor(c, band(x, set(scalar_t(-0.0))));                                               \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t trunc(vec_t x)                                               \
    {                                                                                              \
        return select(lt(x, zero()), ceil(x), floor(x));                                           \
    }

struct ops_sse_f32
{
    using vec_t    = __m128;
    using scalar_t = float;

    static constexpr int  size    = 4;
    static constexpr bool has_fma = SCALARS_VMATH_X86_HAS_FMA;

    SCALARS_FORCE_INLINE static vec_t set(float v) { return _mm_set1_ps(v); }
    SCALARS_FORCE_INLINE static vec_t set_bits(std::uint32_t b)
    {
        return _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(b)));
    }
    SCALARS_FORCE_INLINE static vec_t zero() { return _mm_setzero_ps(); }
    SCALARS_FORCE_INLINE static vec_t loadu(const float* p) { return _mm_loadu_ps(p); }
    SCALARS_FORCE_INLINE static void  storeu(float* p, vec_t v) { _mm_storeu_ps(p, v); }

    SCALARS_FORCE_INLINE static vec_t add(vec_t a, vec_t b) { return _mm_add_ps(a, b); }
    SCALARS_FORCE_INLINE static vec_t sub(vec_t a, vec_t b) { return _mm_sub_ps(a, b); }
    SCALARS_FORCE_INLINE static vec_t mul(vec_t a, vec_t b) { return _mm_mul_ps(a, b); }
    SCALARS_FORCE_INLINE static vec_t div(vec_t a, vec_t b) { return _mm_div_ps(a, b); }
    SCALARS_FORCE_INLINE static vec_t sqrt(vec_t a) { return _mm_sqrt_ps(a); }
    SCALARS_FORCE_INLINE static vec_t fma(vec_t a, vec_t b, vec_t c)
    {
#ifdef __FMA__
        return _mm_fmadd_ps(a, b, c);
#else
        return _mm_add_ps(_mm_mul_ps(a, b), c);
#endif
    }
    SCALARS_FORCE_INLINE static vec_t min(vec_t a, vec_t b) { return _mm_min_ps(b, a); }
    SCALARS_FORCE_INLINE static vec_t max(vec_t a, vec_t b) { return _mm_max_ps(b, a); }

    SCALARS_FORCE_INLINE static vec_t lt(vec_t a, vec_t b) { return _mm_cmplt_ps(a, b); }
    SCALARS_FORCE_INLINE static vec_t le(vec_t a, vec_t b) { return _mm_cmple_ps(a, b); }
    SCALARS_FORCE_INLINE static vec_t eq(vec_t a, vec_t b) { return _mm_cmpeq_ps(a, b); }
    SCALARS_FORCE_INLINE static vec_t nge(vec_t a, vec_t b) { return _mm_cmpnge_ps(a, b); }
    SCALARS_FORCE_INLINE static vec_t isnan(vec_t a) { return _mm_cmpunord_ps(a, a); }

    SCALARS_FORCE_INLINE static vec_t band(vec_t a, vec_t b) { return _mm_and_ps(a, b); }
    SCALARS_FORCE_INLINE static vec_t bor(vec_t a, vec_t b) { return _mm_or_ps(a, b); }
    SCALARS_FORCE_INLINE static vec_t bxor(vec_t a, vec_t b) { return _mm_xor_ps(a, b); }
    SCALARS_FORCE_INLINE static vec_t bandnot(vec_t a, vec_t b) { return _mm_andnot_ps(b, a); }
    SCALARS_FORCE_INLINE static vec_t select(vec_t m, vec_t a, vec_t b)
    {
#ifdef __SSE4_1__
        return _mm_blendv_ps(b, a, m);
#else
        return _mm_or_ps(_mm_and_ps(m, a), _mm_andnot_ps(m, b));
#endif
    }

    SCALARS_FORCE_INLINE static vec_t iadd(vec_t a, vec_t b)
    {
        return _mm_castsi128_ps(_mm_add_epi32(_mm_castps_si128(a), _mm_castps_si128(b)));
    }
    SCALARS_FORCE_INLINE static vec_t isub(vec_t a, vec_t b)
    {
        return _mm_castsi128_ps(_mm_sub_epi32(_mm_castps_si128(a), _mm_castps_si128(b)));
    }
    template <int N> SCALARS_FORCE_INLINE static vec_t shl(vec_t a)
    {
        return _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(a), N));
    }
    template <int N> SCALARS_FORCE_INLINE static vec_t shr(vec_t a)
    {
        return _mm_castsi128_ps(_mm_srli_epi32(_mm_castps_si128(a), N));
    }

    SCALARS_FORCE_INLINE static bool any(vec_t m) { return _mm_movemask_ps(m) != 0; }
    SCALARS_FORCE_INLINE static void barrier(vec_t& v) { SCALARS_VMATH_X86_BARRIER(v); }

#ifdef __SSE4_1__
    SCALARS_FORCE_INLINE static vec_t rint(vec_t x)
    {
        return _mm_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
    SCALARS_FORCE_INLINE static vec_t floor(vec_t x) { return _mm_floor_ps(x); }
    SCALARS_FORCE_INLINE static vec_t ceil(vec_t x) { return _mm_ceil_ps(x); }
    SCALARS_FORCE_INLINE static vec_t trunc(vec_t x)
    {
        return _mm_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
#else
    SCALARS_VMATH_SSE_ROUNDING(ps, 8388608.0f)
#endif
};

struct ops_sse_f64
{
    using vec_t    = __m128d;
    using scalar_t = double;

    static constexpr int  size    = 2;
    static constexpr bool has_fma = SCALARS_VMATH_X86_HAS_FMA;

    SCALARS_FORCE_INLINE static vec_t set(double v) { return _mm_set1_pd(v); }
    SCALARS_FORCE_INLINE static vec_t set_bits(std::uint64_t b)
    {
        return _mm_castsi128_pd(_mm_set1_epi64x(static_cast<long long>(b)));
    }
    SCALARS_FORCE_INLINE static vec_t zero() { return _mm_setzero_pd(); }
    SCALARS_FORCE_INLINE static vec_t loadu(const double* p) { return _mm_loadu_pd(p); }
    SCALARS_FORCE_INLINE static void  storeu(double* p, vec_t v) { _mm_storeu_pd(p, v); }

    SCALARS_FORCE_INLINE static vec_t add(vec_t a, vec_t b) { return _mm_add_pd(a, b); }
    SCALARS_FORCE_INLINE static vec_t sub(vec_t a, vec_t b) { return _mm_sub_pd(a, b); }
    SCALARS_FORCE_INLINE static vec_t mul(vec_t a, vec_t b) { return _mm_mul_pd(a, b); }
    SCALARS_FORCE_INLINE static vec_t div(vec_t a, vec_t b) { return _mm_div_pd(a, b); }
    SCALARS_FORCE_INLINE static vec_t sqrt(vec_t a) { return _mm_sqrt_pd(a); }
    SCALARS_FORCE_INLINE static vec_t fma(vec_t a, vec_t b, vec_t c)
    {
#ifdef __FMA__
        return _mm_fmadd_pd(a, b, c);
#else
        return _mm_add_pd(_mm_mul_pd(a, b), c);
#endif
    }
    SCALARS_FORCE_INLINE static vec_t min(vec_t a, vec_t b) { return _mm_min_pd(b, a); }
    SCALARS_FORCE_INLINE static vec_t max(vec_t a, vec_t b) { return _mm_max_pd(b, a); }

    SCALARS_FORCE_INLINE static vec_t lt(vec_t a, vec_t b) { return _mm_cmplt_pd(a, b); }
    SCALARS_FORCE_INLINE static vec_t le(vec_t a, vec_t b) { return _mm_cmple_pd(a, b); }
    SCALARS_FORCE_INLINE static vec_t eq(vec_t a, vec_t b) { return _mm_cmpeq_pd(a, b); }
    SCALARS_FORCE_INLINE static vec_t nge(vec_t a, vec_t b) { return _mm_cmpnge_pd(a, b); }
    SCALARS_FORCE_INLINE static vec_t isnan(vec_t a) { return _mm_cmpunord_pd(a, a); }

    SCALARS_FORCE_INLINE static vec_t band(vec_t a, vec_t b) { return _mm_and_pd(a, b); }
    SCALARS_FORCE_INLINE static vec_t bor(vec_t a, vec_t b) { return _mm_or_pd(a, b); }
    SCALARS_FORCE_INLINE static vec_t bxor(vec_t a, vec_t b) { return _mm_xor_pd(a, b); }
    SCALARS_FORCE_INLINE static vec_t bandnot(vec_t a, vec_t b) { return _mm_andnot_pd(b, a); }
    SCALARS_FORCE_INLINE static vec_t select(vec_t m, vec_t a, vec_t b)
    {
#ifdef __SSE4_1__
        return _mm_blendv_pd(b, a, m);
#else
        return _mm_or_pd(_mm_and_pd(m, a), _mm_andnot_pd(m, b));
#endif
    }

    SCALARS_FORCE_INLINE static vec_t iadd(vec_t a, vec_t b)
    {
        return _mm_castsi128_pd(_mm_add_epi64(_mm_castpd_si128(a), _mm_castpd_si128(b)));
    }
    SCALARS_FORCE_INLINE static vec_t isub(vec_t a, vec_t b)
    {
        return _mm_castsi128_pd(_mm_sub_epi64(_mm_castpd_si128(a), _mm_castpd_si128(b)));
    }
    template <int N> SCALARS_FORCE_INLINE static vec_t shl(vec_t a)
    {
        return _mm_castsi128_pd(_mm_slli_epi64(_mm_castpd_si128(a), N));
    }
    template <int N> SCALARS_FORCE_INLINE static vec_t shr(vec_t a)
    {
        return _mm_castsi128_pd(_mm_srli_epi64(_mm_castpd_si128(a), N));
    }

    SCALARS_FORCE_INLINE static bool any(vec_t m) { return _mm_movemask_pd(m) != 0; }
    SCALARS_FORCE_INLINE static void barrier(vec_t& v) { SCALARS_VMATH_X86_BARRIER(v); }

#ifdef __SSE4_1__
    SCALARS_FORCE_INLINE static vec_t rint(vec_t x)
    {
        return _mm_round_pd(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
    SCALARS_FORCE_INLINE static vec_t floor(vec_t x) { return _mm_floor_pd(x); }
    SCALARS_FORCE_INLINE static vec_t ceil(vec_t x) { return _mm_ceil_pd(x); }
    SCALARS_FORCE_INLINE static vec_t trunc(vec_t x)
    {
        return _mm_round_pd(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
#else
    SCALARS_VMATH_SSE_ROUNDING(pd, 4503599627370496.0)
#endif
};

#undef SCALARS_VMATH_SSE_ROUNDING

using ops_f32 = ops_sse_f32;
using ops_f64 = ops_sse_f64;

#elif (SCALARS_HAS_AVX || SCALARS_HAS_AVX2) && !SCALARS_HAS_AVX512

// AVX without AVX2 has no 256-bit integer instructions; apply the 128-bit ones
// to each half.
#ifdef __AVX2__
#define SCALARS_VMATH_AVX_INT(op256, op128, a, b) op256(a, b)
#else
#define SCALARS_VMATH_AVX_INT(op256, op128, a, b)                                                  \
    _mm256_insertf128_si256(                                                                       \
        _mm256_castsi128_si256(op128(_mm256_castsi256_si128(a), _mm256_castsi256_si128(b))),       \
        op128(_mm256_extractf128_si256(a, 1), _mm256_extractf128_si256(b, 1)),                     \
        1)
#endif

#define SCALARS_VMATH_AVX_COMMON(sfx, castv, casti)                                                \
    SCALARS_FORCE_INLINE static vec_t zero() { return _mm256_setzero_##sfx(); }                    \
    SCALARS_FORCE_INLINE static vec_t loadu(const scalar_t* p) { return _mm256_loadu_##sfx(p); }   \
    SCALARS_FORCE_INLINE static void  storeu(scalar_t* p, vec_t v) { _mm256_storeu_##sfx(p, v); }  \
    SCALARS_FORCE_INLINE static vec_t add(vec_t a, vec_t b) { return _mm256_add_##sfx(a, b); }     \
    SCALARS_FORCE_INLINE static vec_t sub(vec_t a, vec_t b) { return _mm256_sub_##sfx(a, b); }     \
    SCALARS_FORCE_INLINE static vec_t mul(vec_t a, vec_t b) { return _mm256_mul_##sfx(a, b); }     \
    SCALARS_FORCE_INLINE static vec_t div(vec_t a, vec_t b) { return _mm256_div_##sfx(a, b); }     \
    SCALARS_FORCE_INLINE static vec_t sqrt(vec_t a) { return _mm256_sqrt_##sfx(a); }               \
    SCALARS_FORCE_INLINE static vec_t min(vec_t a, vec_t b) { return _mm256_min_##sfx(b, a); }     \
    SCALARS_FORCE_INLINE static vec_t max(vec_t a, vec_t b) { return _mm256_max_##sfx(b, a); }     \
    SCALARS_FORCE_INLINE static vec_t lt(vec_t a, vec_t b)                                         \
    {                                                                                              \
        return _mm256_cmp_##sfx(a, b, _CMP_LT_OQ);                                                 \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t le(vec_t a, vec_t b)                                         \
    {                                                                                              \
        return _mm256_cmp_##sfx(a, b, _CMP_LE_OQ);                                                 \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t eq(vec_t a, vec_t b)                                         \
    {                                                                                              \
        return _mm256_cmp_##sfx(a, b, _CMP_EQ_OQ);                                                 \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t nge(vec_t a, vec_t b)                                        \
    {                                                                                              \
        return _mm256_cmp_##sfx(a, b, _CMP_NGE_UQ);                                                \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t isnan(vec_t a) { return _mm256_cmp_##sfx(a, a, _CMP_UNORD_Q); } \
    SCALARS_FORCE_INLINE static vec_t band(vec_t a, vec_t b) { return _mm256_and_##sfx(a, b); }    \
    SCALARS_FORCE_INLINE static vec_t bor(vec_t a, vec_t b) { return _mm256_or_##sfx(a, b); }      \
    SCALARS_FORCE_INLINE static vec_t bxor(vec_t a, vec_t b) { return _mm256_xor_##sfx(a, b); }    \
    SCALARS_FORCE_INLINE static vec_t bandnot(vec_t a, vec_t b)                                    \
    {                                                                                              \
        return _mm256_andnot_##sfx(b, a);                                                          \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t select(vec_t m, vec_t a, vec_t b)                            \
    {                                                                                              \
        return _mm256_blendv_##sfx(b, a, m);                                                       \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t rint(vec_t x)                                                \
    {                                                                                              \
        return _mm256_round_##sfx(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);               \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t floor(vec_t x) { return _mm256_floor_##sfx(x); }             \
    SCALARS_FORCE_INLINE static vec_t ceil(vec_t x) { return _mm256_ceil_##sfx(x); }               \
    SCALARS_FORCE_INLINE static vec_t trunc(vec_t x)                                               \
    {                                                                                              \
        return _mm256_round_##sfx(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);                      \
    }                                                                                              \
    SCALARS_FORCE_INLINE static bool any(vec_t m) { return _mm256_movemask_##sfx(m) != 0; }        \
    SCALARS_FORCE_INLINE static void barrier(vec_t& v) { SCALARS_VMATH_X86_BARRIER(v); }

struct ops_avx_f32
{
    using vec_t    = __m256;
    using scalar_t = float;

    static constexpr int  size    = 8;
    static constexpr bool has_fma = SCALARS_VMATH_X86_HAS_FMA;

    SCALARS_FORCE_INLINE static vec_t set(float v) { return _mm256_set1_ps(v); }
    SCALARS_FORCE_INLINE static vec_t set_bits(std::uint32_t b)
    {
        return _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(b)));
    }
    SCALARS_FORCE_INLINE static vec_t fma(vec_t a, vec_t b, vec_t c)
    {
#ifdef __FMA__
        return _mm256_fmadd_ps(a, b, c);
#else
        return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
    }
    SCALARS_VMATH_AVX_COMMON(ps, _mm256_castsi256_ps, _mm256_castps_si256)

    SCALARS_FORCE_INLINE static vec_t iadd(vec_t a, vec_t b)
    {
        const __m256i ai = _mm256_castps_si256(a);
        const __m256i bi = _mm256_castps_si256(b);
        return _mm256_castsi256_ps(SCALARS_VMATH_AVX_INT(_mm256_add_epi32, _mm_add_epi32, ai, bi));
    }
    SCALARS_FORCE_INLINE static vec_t isub(vec_t a, vec_t b)
    {
        const __m256i ai = _mm256_castps_si256(a);
        const __m256i bi = _mm256_castps_si256(b);
        return _mm256_castsi256_ps(SCALARS_VMATH_AVX_INT(_mm256_sub_epi32, _mm_sub_epi32, ai, bi));
    }
    template <int N> SCALARS_FORCE_INLINE static vec_t shl(vec_t a)
    {
#ifdef __AVX2__
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(a), N));
#else
        const __m256i ai = _mm256_castps_si256(a);
        return _mm256_castsi256_ps(_mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_slli_epi32(_mm256_castsi256_si128(ai), N)),
            _mm_slli_epi32(_mm256_extractf128_si256(ai, 1), N),
            1));
#endif
    }
    template <int N> SCALARS_FORCE_INLINE static vec_t shr(vec_t a)
    {
#ifdef __AVX2__
        return _mm256_castsi256_ps(_mm256_srli_epi32(_mm256_castps_si256(a), N));
#else
        const __m256i ai = _mm256_castps_si256(a);
        return _mm256_castsi256_ps(_mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_srli_epi32(_mm256_castsi256_si128(ai), N)),
            _mm_srli_epi32(_mm256_extractf128_si256(ai, 1), N),
            1));
#endif
    }
};

struct ops_avx_f64
{
    using vec_t    = __m256d;
    using scalar_t = double;

    static constexpr int  size    = 4;
    static constexpr bool has_fma = SCALARS_VMATH_X86_HAS_FMA;

    SCALARS_FORCE_INLINE static vec_t set(double v) { return _mm256_set1_pd(v); }
    SCALARS_FORCE_INLINE static vec_t set_bits(std::uint64_t b)
    {
        return _mm256_castsi256_pd(_mm256_set1_epi64x(static_cast<long long>(b)));
    }
    SCALARS_FORCE_INLINE static vec_t fma(vec_t a, vec_t b, vec_t c)
    {
#ifdef __FMA__
        return _mm256_fmadd_pd(a, b, c);
#else
        return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#endif
    }
    SCALARS_VMATH_AVX_COMMON(pd, _mm256_castsi256_pd, _mm256_castpd_si256)

    SCALARS_FORCE_INLINE static vec_t iadd(vec_t a, vec_t b)
    {
        const __m256i ai = _mm256_castpd_si256(a);
        const __m256i bi = _mm256_castpd_si256(b);
        return _mm256_castsi256_pd(SCALARS_VMATH_AVX_INT(_mm256_add_epi64, _mm_add_epi64, ai, bi));
    }
    SCALARS_FORCE_INLINE static vec_t isub(vec_t a, vec_t b)
    {
        const __m256i ai = _mm256_castpd_si256(a);
        const __m256i bi = _mm256_castpd_si256(b);
        return _mm256_castsi256_pd(SCALARS_VMATH_AVX_INT(_mm256_sub_epi64, _mm_sub_epi64, ai, bi));
    }
    template <int N> SCALARS_FORCE_INLINE static vec_t shl(vec_t a)
    {
#ifdef __AVX2__
        return _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_castpd_si256(a), N));
#else
        const __m256i ai = _mm256_castpd_si256(a);
        return _mm256_castsi256_pd(_mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_slli_epi64(_mm256_castsi256_si128(ai), N)),
            _mm_slli_epi64(_mm256_extractf128_si256(ai, 1), N),
            1));
#endif
    }
    template <int N> SCALARS_FORCE_INLINE static vec_t shr(vec_t a)
    {
#ifdef __AVX2__
        return _mm256_castsi256_pd(_mm256_srli_epi64(_mm256_castpd_si256(a), N));
#else
        const __m256i ai = _mm256_castpd_si256(a);
        return _mm256_castsi256_pd(_mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_srli_epi64(_mm256_castsi256_si128(ai), N)),
            _mm_srli_epi64(_mm256_extractf128_si256(ai, 1), N),
            1));
#endif
    }
};

#undef SCALARS_VMATH_AVX_COMMON
#undef SCALARS_VMATH_AVX_INT

using ops_f32 = ops_avx_f32;
using ops_f64 = ops_avx_f64;

#elif SCALARS_HAS_AVX512

// AVX-512F only: compares produce k-masks, widened here to all-ones lanes, and
// the bitwise operations go through the integer domain (the _ps/_pd forms are
// AVX-512DQ).
#define SCALARS_VMATH_AVX512_COMMON(sfx, w, castv, casti)                                          \
    SCALARS_FORCE_INLINE static vec_t from_k(kmask_t k)                                            \
    {                                                                                              \
        return castv(_mm512_maskz_set1_epi##w(k, -1));                                             \
    }                                                                                              \
    SCALARS_FORCE_INLINE static kmask_t to_k(vec_t m)                                              \
    {                                                                                              \
        return _mm512_test_epi##w##_mask(casti(m), casti(m));                                      \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t zero() { return _mm512_setzero_##sfx(); }                    \
    SCALARS_FORCE_INLINE static vec_t loadu(const scalar_t* p) { return _mm512_loadu_##sfx(p); }   \
    SCALARS_FORCE_INLINE static void  storeu(scalar_t* p, vec_t v) { _mm512_storeu_##sfx(p, v); }  \
    SCALARS_FORCE_INLINE static vec_t add(vec_t a, vec_t b) { return _mm512_add_##sfx(a, b); }     \
    SCALARS_FORCE_INLINE static vec_t sub(vec_t a, vec_t b) { return _mm512_sub_##sfx(a, b); }     \
    SCALARS_FORCE_INLINE static vec_t mul(vec_t a, vec_t b) { return _mm512_mul_##sfx(a, b); }     \
    SCALARS_FORCE_INLINE static vec_t div(vec_t a, vec_t b) { return _mm512_div_##sfx(a, b); }     \
    SCALARS_FORCE_INLINE static vec_t sqrt(vec_t a) { return _mm512_sqrt_##sfx(a); }               \
    SCALARS_FORCE_INLINE static vec_t fma(vec_t a, vec_t b, vec_t c)                               \
    {                                                                                              \
        return _mm512_fmadd_##sfx(a, b, c);                                                        \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t min(vec_t a, vec_t b) { return _mm512_min_##sfx(b, a); }     \
    SCALARS_FORCE_INLINE static vec_t max(vec_t a, vec_t b) { return _mm512_max_##sfx(b, a); }     \
    SCALARS_FORCE_INLINE static vec_t lt(vec_t a, vec_t b)                                         \
    {                                                                                              \
        return from_k(_mm512_cmp_##sfx##_mask(a, b, _CMP_LT_OQ));                                  \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t le(vec_t a, vec_t b)                                         \
    {                                                                                              \
        return from_k(_mm512_cmp_##sfx##_mask(a, b, _CMP_LE_OQ));                                  \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t eq(vec_t a, vec_t b)                                         \
    {                                                                                              \
        return from_k(_mm512_cmp_##sfx##_mask(a, b, _CMP_EQ_OQ));                                  \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t nge(vec_t a, vec_t b)                                        \
    {                                                                                              \
        return from_k(_mm512_cmp_##sfx##_mask(a, b, _CMP_NGE_UQ));                                 \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t isnan(vec_t a)                                               \
    {                                                                                              \
        return from_k(_mm512_cmp_##sfx##_mask(a, a, _CMP_UNORD_Q));                                \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t band(vec_t a, vec_t b)                                       \
    {                                                                                              \
        return castv(_mm512_and_si512(casti(a), casti(b)));                                        \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t bor(vec_t a, vec_t b)                                        \
    {                                                                                              \
        return castv(_mm512_or_si512(casti(a), casti(b)));                                         \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t bxor(vec_t a, vec_t b)                                       \
    {                                                                                              \
        return castv(_mm512_xor_si512(casti(a), casti(b)));                                        \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t bandnot(vec_t a, vec_t b)                                    \
    {                                                                                              \
        return castv(_mm512_andnot_si512(casti(b), casti(a)));                                     \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t select(vec_t m, vec_t a, vec_t b)                            \
    {                                                                                              \
        return _mm512_mask_blend_##sfx(to_k(m), b, a);                                             \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t iadd(vec_t a, vec_t b)                                       \
    {                                                                                              \
        return castv(_mm512_add_epi##w(casti(a), casti(b)));                                       \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t isub(vec_t a, vec_t b)                                       \
    {                                                                                              \
        return castv(_mm512_sub_epi##w(casti(a), casti(b)));                                       \
    }                                                                                              \
    template <int N> SCALARS_FORCE_INLINE static vec_t shl(vec_t a)                                \
    {                                                                                              \
        return castv(_mm512_slli_epi##w(casti(a), N));                                             \
    }                                                                                              \
    template <int N> SCALARS_FORCE_INLINE static vec_t shr(vec_t a)                                \
    {                                                                                              \
        return castv(_mm512_srli_epi##w(casti(a), N));                                             \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t rint(vec_t x)                                                \
    {                                                                                              \
        return _mm512_roundscale_##sfx(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);          \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t floor(vec_t x)                                               \
    {                                                                                              \
        return _mm512_roundscale_##sfx(x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);              \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t ceil(vec_t x)                                                \
    {                                                                                              \
        return _mm512_roundscale_##sfx(x, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);              \
    }                                                                                              \
    SCALARS_FORCE_INLINE static vec_t trunc(vec_t x)                                               \
    {                                                                                              \
        return _mm512_roundscale_##sfx(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);                 \
    }                                                                                              \
    SCALARS_FORCE_INLINE static bool any(vec_t m) { return to_k(m) != 0; }                         \
    SCALARS_FORCE_INLINE static void barrier(vec_t& v) { SCALARS_VMATH_X86_BARRIER(v); }

struct ops_avx512_f32
{
    using vec_t    = __m512;
    using scalar_t = float;
    using kmask_t  = __mmask16;

    static constexpr int  size    = 16;
    static constexpr bool has_fma = true;

    SCALARS_FORCE_INLINE static vec_t set(float v) { return _mm512_set1_ps(v); }
    SCALARS_FORCE_INLINE static vec_t set_bits(std::uint32_t b)
    {
        return _mm512_castsi512_ps(_mm512_set1_epi32(static_cast<int>(b)));
    }
    SCALARS_VMATH_AVX512_COMMON(ps, 32, _mm512_castsi512_ps, _mm512_castps_si512)
};

struct ops_avx512_f64
{
    using vec_t    = __m512d;
    using scalar_t = double;
    using kmask_t  = __mmask8;

    static constexpr int  size    = 8;
    static constexpr bool has_fma = true;

    SCALARS_FORCE_INLINE static vec_t set(double v) { return _mm512_set1_pd(v); }
    SCALARS_FORCE_INLINE static vec_t set_bits(std::uint64_t b)
    {
        return _mm512_castsi512_pd(_mm512_set1_epi64(static_cast<long long>(b)));
    }
    SCALARS_VMATH_AVX512_COMMON(pd, 64, _mm512_castsi512_pd, _mm512_castpd_si512)
};

#undef SCALARS_VMATH_AVX512_COMMON

using ops_f32 = ops_avx512_f32;
using ops_f64 = ops_avx512_f64;

#endif
}  // namespace scalars::detail::vmath

#undef SCALARS_VMATH_X86_HAS_FMA
