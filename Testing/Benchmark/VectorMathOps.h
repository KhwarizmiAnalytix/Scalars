#pragma once

// The vectorized elementary functions measured by BenchmarkVectorMath (speed) and
// AccuracyVectorMath (error): for each one, the simd<T> packet function, Eigen's
// array expression, the scalar std:: (libm) call and a long double reference,
// with the input range the speed benchmark samples. A false has_eigen or
// has_libm means there is no equivalent; the expression given then is a
// placeholder that is never called.

#include <cmath>
#include <cstdint>
#include <limits>

#include "backend/simd.h"
#include "math/normal_cdf.h"

#if SCALARS_BENCHMARK_HAS_EIGEN
#include <Eigen/Core>
#include <unsupported/Eigen/SpecialFunctions>
#endif

namespace scalars_vm
{
using scalars::simd;
using ref_t = long double;

inline const char* math_library()
{
#if SCALARS_HAS_SVML
    return "svml";
#elif SCALARS_HAS_SLEEF
    return "sleef";
#else
    return "native";
#endif
}

template <class T> const char* type_name()
{
    return sizeof(T) == 4 ? "f32" : "f64";
}

constexpr ref_t kSqrtHalf = 0.707106781186547524400844362104849039L;

inline ref_t normal_cdf_reference(ref_t x)
{
    return 0.5L * std::erfc(-x * kSqrtHalf);
}

// Inverse of the normal distribution function: scalars::inv_normalcdf refined
// by Newton steps on the lower tail, where erfc keeps its relative accuracy;
// p > 1/2 uses the symmetry with 1 - p, which is exact for p >= 1/2.
inline ref_t inv_normal_cdf_reference(ref_t p)
{
    if (std::isnan(p) || p < 0 || p > 1)
    {
        return std::numeric_limits<ref_t>::quiet_NaN();
    }
    if (p == 0 || p == 1)
    {
        return p == 0 ? -std::numeric_limits<ref_t>::infinity()
                      : std::numeric_limits<ref_t>::infinity();
    }
    const bool  upper = p > 0.5L;
    const ref_t q     = upper ? 1 - p : p;
    ref_t       x     = scalars::inv_normalcdf(static_cast<double>(q));
    for (int i = 0; i < 4; ++i)
    {
        const ref_t density = std::exp(-0.5L * x * x) * 0.398942280401432677939946059934381868L;
        if (!(density > 0))
        {
            break;
        }
        x -= (normal_cdf_reference(x) - q) / density;
    }
    return upper ? -x : x;
}

#define SCALARS_VM_UNPAREN(...) __VA_ARGS__

#if SCALARS_BENCHMARK_HAS_EIGEN
#define SCALARS_VM_EIGEN_EXPR(EXPR)                                                                \
    template <class A> static auto eigen(const A& x, const A& y)                                   \
    {                                                                                              \
        (void)y;                                                                                   \
        using T = typename A::Scalar;                                                              \
        (void)sizeof(T);                                                                           \
        return SCALARS_VM_UNPAREN EXPR;                                                            \
    }
#else
#define SCALARS_VM_EIGEN_EXPR(EXPR)
#endif

#define SCALARS_VM_OP(NAME, ARITY, LO, HI, LO2, HI2, HAS_EIGEN, EIGEN, HAS_LIBM, LIBM, REF)        \
    struct op_##NAME                                                                               \
    {                                                                                              \
        static constexpr const char* name      = #NAME;                                            \
        static constexpr bool        binary    = ARITY == 2;                                       \
        static constexpr bool        has_eigen = HAS_EIGEN;                                        \
        static constexpr bool        has_libm  = HAS_LIBM;                                         \
        static constexpr double      lo = LO, hi = HI, lo2 = LO2, hi2 = HI2;                       \
        template <class T>                                                                         \
        static typename simd<T>::simd_t packet(                                                    \
            typename simd<T>::simd_t x, typename simd<T>::simd_t y)                                \
        {                                                                                          \
            if constexpr (ARITY == 2)                                                              \
            {                                                                                      \
                return simd<T>::NAME(x, y);                                                        \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                (void)y;                                                                           \
                return simd<T>::NAME(x);                                                           \
            }                                                                                      \
        }                                                                                          \
        template <class T> static T libm(T x, T y)                                                 \
        {                                                                                          \
            (void)y;                                                                               \
            return static_cast<T>(SCALARS_VM_UNPAREN LIBM);                                        \
        }                                                                                          \
        static ref_t reference(ref_t x, ref_t y)                                                   \
        {                                                                                          \
            (void)y;                                                                               \
            return SCALARS_VM_UNPAREN REF;                                                         \
        }                                                                                          \
        SCALARS_VM_EIGEN_EXPR(EIGEN)                                                               \
    };

// Input ranges keep every result finite in float. Eigen has no normal
// distribution function; its row for cdf is the erfc-based form.
// clang-format off
//            name     arity lo      hi           lo2    hi2   eigen                                         libm                                              reference
SCALARS_VM_OP(exp,     1,    -80.0,  80.0,        0,     0,    true,  (x.exp()),                             true,  (std::exp(x)),                                    (std::exp(x)))
SCALARS_VM_OP(expm1,   1,    -80.0,  80.0,        0,     0,    true,  (x.expm1()),                           true,  (std::expm1(x)),                                  (std::expm1(x)))
SCALARS_VM_OP(exp2,    1,    -120.0, 120.0,       0,     0,    true,  (x.exp2()),                            true,  (std::exp2(x)),                                   (std::exp2(x)))
SCALARS_VM_OP(exp10,   1,    -35.0,  35.0,        0,     0,    false, (x),                                   false, (x),                                              (std::pow(ref_t(10), x)))
SCALARS_VM_OP(log,     1,    1e-30,  1e30,        0,     0,    true,  (x.log()),                             true,  (std::log(x)),                                    (std::log(x)))
SCALARS_VM_OP(log1p,   1,    -0.99,  1e6,         0,     0,    true,  (x.log1p()),                           true,  (std::log1p(x)),                                  (std::log1p(x)))
SCALARS_VM_OP(log2,    1,    1e-30,  1e30,        0,     0,    true,  (x.log2()),                            true,  (std::log2(x)),                                   (std::log2(x)))
SCALARS_VM_OP(log10,   1,    1e-30,  1e30,        0,     0,    true,  (x.log10()),                           true,  (std::log10(x)),                                  (std::log10(x)))
SCALARS_VM_OP(sin,     1,    -100.0, 100.0,       0,     0,    true,  (x.sin()),                             true,  (std::sin(x)),                                    (std::sin(x)))
SCALARS_VM_OP(cos,     1,    -100.0, 100.0,       0,     0,    true,  (x.cos()),                             true,  (std::cos(x)),                                    (std::cos(x)))
SCALARS_VM_OP(tan,     1,    -100.0, 100.0,       0,     0,    true,  (x.tan()),                             true,  (std::tan(x)),                                    (std::tan(x)))
SCALARS_VM_OP(asin,    1,    -1.0,   1.0,         0,     0,    true,  (x.asin()),                            true,  (std::asin(x)),                                   (std::asin(x)))
SCALARS_VM_OP(acos,    1,    -1.0,   1.0,         0,     0,    true,  (x.acos()),                            true,  (std::acos(x)),                                   (std::acos(x)))
SCALARS_VM_OP(atan,    1,    -100.0, 100.0,       0,     0,    true,  (x.atan()),                            true,  (std::atan(x)),                                   (std::atan(x)))
SCALARS_VM_OP(sinh,    1,    -80.0,  80.0,        0,     0,    true,  (x.sinh()),                            true,  (std::sinh(x)),                                   (std::sinh(x)))
SCALARS_VM_OP(cosh,    1,    -80.0,  80.0,        0,     0,    true,  (x.cosh()),                            true,  (std::cosh(x)),                                   (std::cosh(x)))
SCALARS_VM_OP(tanh,    1,    -10.0,  10.0,        0,     0,    true,  (x.tanh()),                            true,  (std::tanh(x)),                                   (std::tanh(x)))
SCALARS_VM_OP(asinh,   1,    -100.0, 100.0,       0,     0,    true,  (x.asinh()),                           true,  (std::asinh(x)),                                  (std::asinh(x)))
SCALARS_VM_OP(acosh,   1,    1.0,    100.0,       0,     0,    true,  (x.acosh()),                           true,  (std::acosh(x)),                                  (std::acosh(x)))
SCALARS_VM_OP(atanh,   1,    -0.99,  0.99,        0,     0,    true,  (x.atanh()),                           true,  (std::atanh(x)),                                  (std::atanh(x)))
SCALARS_VM_OP(cbrt,    1,    -1e6,   1e6,         0,     0,    true,  (x.cbrt()),                            true,  (std::cbrt(x)),                                   (std::cbrt(x)))
SCALARS_VM_OP(sqrt,    1,    0.0,    1e6,         0,     0,    true,  (x.sqrt()),                            true,  (std::sqrt(x)),                                   (std::sqrt(x)))
SCALARS_VM_OP(invsqrt, 1,    1e-6,   1e6,         0,     0,    true,  (x.rsqrt()),                           true,  (T(1) / std::sqrt(x)),                            (1 / std::sqrt(x)))
SCALARS_VM_OP(cdf,     1,    -8.0,   8.0,         0,     0,    true,  (T(0.5) * (x * T(-kSqrtHalf)).erfc()), true,  (T(0.5) * std::erfc(x * T(-kSqrtHalf))),         (normal_cdf_reference(x)))
SCALARS_VM_OP(inv_cdf, 1,    1e-6,   1.0 - 1e-6,  0,     0,    true,  (x.ndtri()),                           false, (x),                                              (inv_normal_cdf_reference(x)))
SCALARS_VM_OP(pow,     2,    0.01,   10.0,        -30.0, 30.0, true,  (x.pow(y)),                            true,  (std::pow(x, y)),                                 (std::pow(x, y)))
SCALARS_VM_OP(hypot,   2,    -1e6,   1e6,         -1e6,  1e6,  false, (x),                                   true,  (std::hypot(x, y)),                               (std::hypot(x, y)))
// clang-format on

#undef SCALARS_VM_OP
#undef SCALARS_VM_EIGEN_EXPR

template <class F> void for_each_op(F&& f)
{
    f(op_exp{}), f(op_expm1{}), f(op_exp2{}), f(op_exp10{}), f(op_log{}), f(op_log1p{});
    f(op_log2{}), f(op_log10{}), f(op_sin{}), f(op_cos{}), f(op_tan{}), f(op_asin{});
    f(op_acos{}), f(op_atan{}), f(op_sinh{}), f(op_cosh{}), f(op_tanh{}), f(op_asinh{});
    f(op_acosh{}), f(op_atanh{}), f(op_cbrt{}), f(op_sqrt{}), f(op_invsqrt{}), f(op_cdf{});
    f(op_inv_cdf{}), f(op_pow{}), f(op_hypot{});
}
}  // namespace scalars_vm
