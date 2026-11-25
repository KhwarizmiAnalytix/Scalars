#pragma once

// backend/cpu/sve/{float,double}/simd.h reuse the NEON specializations below
// unconditionally (same lane layout at -msve-vector-bits=128), so these
// wrappers must also be emitted when only SCALARS_HAS_SVE is set.
#if SCALARS_HAS_NEON || SCALARS_HAS_SVE

#include "common/intrinsics.h"

#include <cmath>

#include "backend/cpu/detail/vector_math.h"
#include "backend/cpu/neon/vector_math_ops.h"
#include "scalars_macros.h"

#if !defined(SCALARS_HAS_NATIVE_MATH)
#define SCALARS_HAS_NATIVE_MATH 0
#endif

// The vectorized port of Eigen's packet math (backend/cpu/detail/vector_math.h)
// serves every function when SCALARS_HAS_NATIVE_MATH is set, and otherwise the
// functions that neither Accelerate nor SLEEF provides.
#if SCALARS_HAS_NATIVE_MATH
#define SCALARS_NEON_USE_ACCELERATE 0
#define SCALARS_NEON_USE_SLEEF 0
#else
#define SCALARS_NEON_USE_ACCELERATE SCALARS_HAS_ACCELERATE
#define SCALARS_NEON_USE_SLEEF (SCALARS_HAS_SLEEF && !SCALARS_HAS_ACCELERATE)
#endif
#define SCALARS_NEON_NATIVE_MATH (!SCALARS_NEON_USE_ACCELERATE && !SCALARS_NEON_USE_SLEEF)

#if SCALARS_NEON_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif
#if SCALARS_HAS_SLEEF
#include <sleef.h>
#endif

namespace scalars::detail_neon
{
using pmf32 = scalars::detail::vmath::packet_math<scalars::detail::vmath::ops_neon_f32>;
using pmf64 = scalars::detail::vmath::packet_math<scalars::detail::vmath::ops_neon_f64>;
}  // namespace scalars::detail_neon

// clang-format off

SCALARS_FORCE_INLINE float32x4_t vatan2q_f32(float32x4_t y, float32x4_t x)
{
#if SCALARS_HAS_SLEEF && !SCALARS_NEON_USE_ACCELERATE && !SCALARS_HAS_NATIVE_MATH
    return Sleef_atan2f4_u10(y, x);
#elif SCALARS_NEON_USE_ACCELERATE
    alignas(16) float by[4], bx[4], out[4];
    vst1q_f32(by, y);
    vst1q_f32(bx, x);
    const int n = 4;
    vvatan2f(out, by, bx, &n);
    return vld1q_f32(out);
#else
    return scalars::detail_neon::pmf32::atan2(y, x);
#endif
}

SCALARS_FORCE_INLINE float64x2_t vatan2q_f64(float64x2_t y, float64x2_t x)
{
#if SCALARS_HAS_SLEEF && !SCALARS_NEON_USE_ACCELERATE && !SCALARS_HAS_NATIVE_MATH
    return Sleef_atan2d2_u10(y, x);
#elif SCALARS_NEON_USE_ACCELERATE
    alignas(16) double by[2], bx[2], out[2];
    vst1q_f64(by, y);
    vst1q_f64(bx, x);
    const int n = 2;
    vvatan2(out, by, bx, &n);
    return vld1q_f64(out);
#else
    return scalars::detail_neon::pmf64::atan2(y, x);
#endif
}

// Share range reduction and avoid a second vForce call in complex kernels.
SCALARS_FORCE_INLINE void vsincosq_f32(float32x4_t x, float32x4_t& sine, float32x4_t& cosine)
{
#if SCALARS_NEON_USE_ACCELERATE
    alignas(16) float in[4], s[4], c[4];
    const int n = 4;
    vst1q_f32(in, x);
    vvsincosf(s, c, in, &n);
    sine = vld1q_f32(s);
    cosine = vld1q_f32(c);
#elif SCALARS_NEON_USE_SLEEF
    const auto pair = Sleef_sincosf4_u10(x);
    sine = pair.x;
    cosine = pair.y;
#else
    scalars::detail_neon::pmf32::sincos(x, sine, cosine);
#endif
}

SCALARS_FORCE_INLINE void vsincosq_f64(float64x2_t x, float64x2_t& sine, float64x2_t& cosine)
{
#if SCALARS_NEON_USE_ACCELERATE
    alignas(16) double in[2], s[2], c[2];
    const int n = 2;
    vst1q_f64(in, x);
    vvsincos(s, c, in, &n);
    sine = vld1q_f64(s);
    cosine = vld1q_f64(c);
#elif SCALARS_NEON_USE_SLEEF
    const auto pair = Sleef_sincosd2_u10(x);
    sine = pair.x;
    cosine = pair.y;
#else
    scalars::detail_neon::pmf64::sincos(x, sine, cosine);
#endif
}

// ---------------------------------------------------------------------------
// Internal dispatch macros — generate vXXXq_f32 / vXXXq_f64 wrappers.
// Priority: Accelerate vForce  >  SLEEF  >  vectorized Eigen port; the port
// alone when SCALARS_HAS_NATIVE_MATH is set.
//
// NEON_MATHF(fn, vForce_fn, sleef_fn, port_fn)  — float32x4_t unary
// NEON_MATHF2(fn, vForce_fn, sleef_fn, port_fn) — float32x4_t binary (pow)
// NEON_MATHD(fn, vForce_fn, sleef_fn, port_fn)  — float64x2_t unary
// NEON_MATHD2(fn, vForce_fn, sleef_fn, port_fn) — float64x2_t binary (pow)
//
// Accelerate vvpow* argument order:
//   vvpowf(out, exponent, base, &n)  →  out[i] = base[i]^exponent[i]
// ---------------------------------------------------------------------------

#if SCALARS_NEON_USE_ACCELERATE
#  define NEON_MATHF(fn, vv, sl, pm)                                           \
     SCALARS_FORCE_INLINE float32x4_t fn(float32x4_t x) {                      \
         alignas(16) float b[4]; const int n = 4;                              \
         vst1q_f32(b, x); vv(b, b, &n); return vld1q_f32(b); }
#  define NEON_MATHF2(fn, vv, sl, pm)                                          \
     SCALARS_FORCE_INLINE float32x4_t fn(float32x4_t x, float32x4_t y) {       \
         alignas(16) float bx[4], by[4], br[4]; const int n = 4;               \
         vst1q_f32(bx, x); vst1q_f32(by, y);                                  \
         vv(br, by, bx, &n); return vld1q_f32(br); }
#  define NEON_MATHD(fn, vv, sl, pm)                                           \
     SCALARS_FORCE_INLINE float64x2_t fn(float64x2_t x) {                      \
         alignas(16) double b[2]; const int n = 2;                             \
         vst1q_f64(b, x); vv(b, b, &n); return vld1q_f64(b); }
#  define NEON_MATHD2(fn, vv, sl, pm)                                          \
     SCALARS_FORCE_INLINE float64x2_t fn(float64x2_t x, float64x2_t y) {       \
         alignas(16) double bx[2], by[2], br[2]; const int n = 2;              \
         vst1q_f64(bx, x); vst1q_f64(by, y);                                  \
         vv(br, by, bx, &n); return vld1q_f64(br); }
#elif SCALARS_NEON_USE_SLEEF
#  define NEON_MATHF(fn, vv, sl, pm)                                           \
     SCALARS_FORCE_INLINE float32x4_t fn(float32x4_t x) { return sl(x); }
#  define NEON_MATHF2(fn, vv, sl, pm)                                          \
     SCALARS_FORCE_INLINE float32x4_t fn(float32x4_t x, float32x4_t y) { return sl(x, y); }
#  define NEON_MATHD(fn, vv, sl, pm)                                           \
     SCALARS_FORCE_INLINE float64x2_t fn(float64x2_t x) { return sl(x); }
#  define NEON_MATHD2(fn, vv, sl, pm)                                          \
     SCALARS_FORCE_INLINE float64x2_t fn(float64x2_t x, float64x2_t y) { return sl(x, y); }
#else
#  define NEON_MATHF(fn, vv, sl, pm)                                           \
     SCALARS_FORCE_INLINE float32x4_t fn(float32x4_t x) {                      \
         return scalars::detail_neon::pmf32::pm(x); }
#  define NEON_MATHF2(fn, vv, sl, pm)                                          \
     SCALARS_FORCE_INLINE float32x4_t fn(float32x4_t x, float32x4_t y) {       \
         return scalars::detail_neon::pmf32::pm(x, y); }
#  define NEON_MATHD(fn, vv, sl, pm)                                           \
     SCALARS_FORCE_INLINE float64x2_t fn(float64x2_t x) {                      \
         return scalars::detail_neon::pmf64::pm(x); }
#  define NEON_MATHD2(fn, vv, sl, pm)                                          \
     SCALARS_FORCE_INLINE float64x2_t fn(float64x2_t x, float64x2_t y) {       \
         return scalars::detail_neon::pmf64::pm(x, y); }
#endif

// ============================================================================
// float32x4_t — vXXXq_f32
//                       fn               vForce          SLEEF              port
// ============================================================================
NEON_MATHF(  vexpq_f32,    vvexpf,   Sleef_expf4_u10,   exp  )
NEON_MATHF(  vexpm1q_f32,  vvexpm1f, Sleef_expm1f4_u10, expm1)
NEON_MATHF(  vexp2q_f32,   vvexp2f,  Sleef_exp2f4_u10,  exp2 )
NEON_MATHF(  vlogq_f32,    vvlogf,   Sleef_logf4_u10,   log  )
NEON_MATHF(  vlog1pq_f32,  vvlog1pf, Sleef_log1pf4_u10, log1p)
NEON_MATHF(  vlog2q_f32,   vvlog2f,  Sleef_log2f4_u10,  log2 )
NEON_MATHF(  vlog10q_f32,  vvlog10f, Sleef_log10f4_u10, log10)
NEON_MATHF(  vsinq_f32,    vvsinf,   Sleef_sinf4_u10,   sin  )
NEON_MATHF(  vcosq_f32,    vvcosf,   Sleef_cosf4_u10,   cos  )
NEON_MATHF(  vtanq_f32,    vvtanf,   Sleef_tanf4_u10,   tan  )
NEON_MATHF(  vasinq_f32,   vvasinf,  Sleef_asinf4_u10,  asin )
NEON_MATHF(  vacosq_f32,   vvacosf,  Sleef_acosf4_u10,  acos )
NEON_MATHF(  vatanq_f32,   vvatanf,  Sleef_atanf4_u10,  atan )
NEON_MATHF(  vsinhq_f32,   vvsinhf,  Sleef_sinhf4_u10,  sinh )
NEON_MATHF(  vcoshq_f32,   vvcoshf,  Sleef_coshf4_u10,  cosh )
NEON_MATHF(  vtanhq_f32,   vvtanhf,  Sleef_tanhf4_u10,  tanh )
NEON_MATHF(  vasinhq_f32,  vvasinhf, Sleef_asinhf4_u10, asinh)
NEON_MATHF(  vacoshq_f32,  vvacoshf, Sleef_acoshf4_u10, acosh)
NEON_MATHF(  vatanhq_f32,  vvatanhf, Sleef_atanhf4_u10, atanh)
NEON_MATHF(  vcbrtq_f32,   vvcbrtf,  Sleef_cbrtf4_u10,  cbrt )
NEON_MATHF2( vpowq_f32,    vvpowf,   Sleef_powf4_u10,   pow  )

// exp10: no native Accelerate equivalent — SLEEF or the port
SCALARS_FORCE_INLINE float32x4_t vexp10q_f32(float32x4_t x)
{
#if SCALARS_HAS_SLEEF && !SCALARS_HAS_NATIVE_MATH
    return Sleef_exp10f4_u10(x);
#else
    return scalars::detail_neon::pmf32::exp10(x);
#endif
}

// ============================================================================
// float64x2_t — vXXXq_f64
//                       fn               vForce         SLEEF              port
// ============================================================================
NEON_MATHD(  vexpq_f64,    vvexp,    Sleef_expd2_u10,   exp  )
NEON_MATHD(  vexpm1q_f64,  vvexpm1,  Sleef_expm1d2_u10, expm1)
NEON_MATHD(  vexp2q_f64,   vvexp2,   Sleef_exp2d2_u10,  exp2 )
NEON_MATHD(  vlogq_f64,    vvlog,    Sleef_logd2_u10,   log  )
NEON_MATHD(  vlog1pq_f64,  vvlog1p,  Sleef_log1pd2_u10, log1p)
NEON_MATHD(  vlog2q_f64,   vvlog2,   Sleef_log2d2_u10,  log2 )
NEON_MATHD(  vlog10q_f64,  vvlog10,  Sleef_log10d2_u10, log10)
NEON_MATHD(  vsinq_f64,    vvsin,    Sleef_sind2_u10,   sin  )
NEON_MATHD(  vcosq_f64,    vvcos,    Sleef_cosd2_u10,   cos  )
NEON_MATHD(  vtanq_f64,    vvtan,    Sleef_tand2_u10,   tan  )
NEON_MATHD(  vasinq_f64,   vvasin,   Sleef_asind2_u10,  asin )
NEON_MATHD(  vacosq_f64,   vvacos,   Sleef_acosd2_u10,  acos )
NEON_MATHD(  vatanq_f64,   vvatan,   Sleef_atand2_u10,  atan )
NEON_MATHD(  vsinhq_f64,   vvsinh,   Sleef_sinhd2_u10,  sinh )
NEON_MATHD(  vcoshq_f64,   vvcosh,   Sleef_coshd2_u10,  cosh )
NEON_MATHD(  vtanhq_f64,   vvtanh,   Sleef_tanhd2_u10,  tanh )
NEON_MATHD(  vasinhq_f64,  vvasinh,  Sleef_asinhd2_u10, asinh)
NEON_MATHD(  vacoshq_f64,  vvacosh,  Sleef_acoshd2_u10, acosh)
NEON_MATHD(  vatanhq_f64,  vvatanh,  Sleef_atanhd2_u10, atanh)
NEON_MATHD(  vcbrtq_f64,   vvcbrt,   Sleef_cbrtd2_u10,  cbrt )
NEON_MATHD2( vpowq_f64,    vvpow,    Sleef_powd2_u10,   pow  )

// exp10: no native Accelerate equivalent — SLEEF or the port
SCALARS_FORCE_INLINE float64x2_t vexp10q_f64(float64x2_t x)
{
#if SCALARS_HAS_SLEEF && !SCALARS_HAS_NATIVE_MATH
    return Sleef_exp10d2_u10(x);
#else
    return scalars::detail_neon::pmf64::exp10(x);
#endif
}

#undef NEON_MATHF
#undef NEON_MATHF2
#undef NEON_MATHD
#undef NEON_MATHD2

// clang-format on

#endif  // SCALARS_HAS_NEON || SCALARS_HAS_SVE
