#if !defined(SCALARS_INTERNAL_INCLUDE_COMPLEX_REMAINING_H)
#error                                                                                             \
    "math/complex_math.h is not meant to be individually included. Include types/complex.h instead."
#endif

#include "common/scalars_export.h"

namespace scalars::complex_math
{

// Exponential functions

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> exp(const scalars::complex<T>& x)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::exp(static_cast<thrust::complex<T>>(x)));
#else
    return static_cast<scalars::complex<T>>(std::exp(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> log(const scalars::complex<T>& x)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::log(static_cast<thrust::complex<T>>(x)));
#else
    return static_cast<scalars::complex<T>>(std::log(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> log10(const scalars::complex<T>& x)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::log10(static_cast<thrust::complex<T>>(x)));
#else
    return static_cast<scalars::complex<T>>(std::log10(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> log2(const scalars::complex<T>& x)
{
    const scalars::complex<T> log2 = scalars::complex<T>(::log(2.0), 0.0);
    return scalars::complex_math::log(x) / log2;
}

// Power functions
//
#if defined(_LIBCPP_VERSION) || (defined(__GLIBCXX__) && !defined(_GLIBCXX11_USE_C99_COMPLEX))
namespace _detail
{
SCALARS_API scalars::complex<float> sqrt(const scalars::complex<float>& in);
SCALARS_API scalars::complex<double> sqrt(const scalars::complex<double>& in);
SCALARS_API scalars::complex<float> acos(const scalars::complex<float>& in);
SCALARS_API scalars::complex<double> acos(const scalars::complex<double>& in);
}  // namespace _detail
#endif

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> sqrt(const scalars::complex<T>& x)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::sqrt(static_cast<thrust::complex<T>>(x)));
#elif !(defined(_LIBCPP_VERSION) || (defined(__GLIBCXX__) && !defined(_GLIBCXX11_USE_C99_COMPLEX)))
    return static_cast<scalars::complex<T>>(std::sqrt(static_cast<std::complex<T>>(x)));
#else
    return _detail::sqrt(x);
#endif
}

template <typename T>
SCALARS_HOST_DEVICE inline scalars::complex<T> pow(
    const scalars::complex<T>& x, const scalars::complex<T>& y)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(
        thrust::pow(static_cast<thrust::complex<T>>(x), static_cast<thrust::complex<T>>(y)));
#else
    return static_cast<scalars::complex<T>>(
        std::pow(static_cast<std::complex<T>>(x), static_cast<std::complex<T>>(y)));
#endif
}

// Regression in ROCm 7.2. See https://github.com/ROCm/rocm-libraries/pull/3836.
// Specialized version for complex<float> on AMD GPUs to use FMA-based
// multiplication
#if defined(__HIPCC__)
namespace detail
{
// FMA-aware complex multiplication for float precision on AMD GPUs.
// This prevents SLP vectorizer from breaking FMA formation, which causes
// numerical precision loss in complex arithmetic.
// The issue occurs when vectorizer packs scalar multiplies before backend
// can form FMA instructions, resulting in double rounding instead of single.
SCALARS_HOST_DEVICE inline thrust::complex<float> complex_mul_fma(
    thrust::complex<float> a, thrust::complex<float> b)
{
    // Complex multiplication: (a.r + a.i*i) * (b.r + b.i*i)
    // = (a.r*b.r - a.i*b.i) + (a.r*b.i + a.i*b.r)*i
    // Using __builtin_fmaf ensures FMA at source level:
    // real: a.r*b.r + (-(a.i*b.i)) = FMA(a.r, b.r, -(a.i*b.i))
    // imag: a.i*b.r + a.r*b.i = FMA(a.r, b.i, a.i*b.r)
    float real_part = __builtin_fmaf(a.real(), b.real(), -(a.imag() * b.imag()));
    float imag_part = __builtin_fmaf(a.real(), b.imag(), a.imag() * b.real());
    return thrust::complex<float>(real_part, imag_part);
}
}  // namespace detail

template <>
SCALARS_HOST_DEVICE inline scalars::complex<float> pow(
    const scalars::complex<float>& x, const scalars::complex<float>& y)
{
    auto log_x   = thrust::log(static_cast<thrust::complex<float>>(x));
    auto y_log_x = detail::complex_mul_fma(static_cast<thrust::complex<float>>(y), log_x);
    return static_cast<scalars::complex<float>>(thrust::exp(y_log_x));
}
#endif

template <typename T>
SCALARS_HOST_DEVICE inline scalars::complex<T> pow(const scalars::complex<T>& x, const T& y)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::pow(static_cast<thrust::complex<T>>(x), y));
#else
    return static_cast<scalars::complex<T>>(std::pow(static_cast<std::complex<T>>(x), y));
#endif
}

template <typename T>
SCALARS_HOST_DEVICE inline scalars::complex<T> pow(const T& x, const scalars::complex<T>& y)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::pow(x, static_cast<thrust::complex<T>>(y)));
#else
    return static_cast<scalars::complex<T>>(std::pow(x, static_cast<std::complex<T>>(y)));
#endif
}

template <typename T, typename U>
SCALARS_HOST_DEVICE inline scalars::complex<decltype(T() * U())> pow(
    const scalars::complex<T>& x, const scalars::complex<U>& y)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(
        thrust::pow(static_cast<thrust::complex<T>>(x), static_cast<thrust::complex<T>>(y)));
#else
    return static_cast<scalars::complex<T>>(
        std::pow(static_cast<std::complex<T>>(x), static_cast<std::complex<T>>(y)));
#endif
}

template <typename T, typename U>
SCALARS_HOST_DEVICE inline scalars::complex<decltype(T() * U())> pow(
    const scalars::complex<T>& x, const U& y)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::pow(static_cast<thrust::complex<T>>(x), y));
#else
    return static_cast<scalars::complex<T>>(std::pow(static_cast<std::complex<T>>(x), y));
#endif
}

template <typename T, typename U>
SCALARS_HOST_DEVICE inline scalars::complex<decltype(T() * U())> pow(
    const T& x, const scalars::complex<U>& y)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::pow(x, static_cast<thrust::complex<T>>(y)));
#else
    return static_cast<scalars::complex<T>>(std::pow(x, static_cast<std::complex<T>>(y)));
#endif
}

// Trigonometric functions

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> sin(const scalars::complex<T>& x)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::sin(static_cast<thrust::complex<T>>(x)));
#else
    return static_cast<scalars::complex<T>>(std::sin(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> cos(const scalars::complex<T>& x)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::cos(static_cast<thrust::complex<T>>(x)));
#else
    return static_cast<scalars::complex<T>>(std::cos(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> tan(const scalars::complex<T>& x)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::tan(static_cast<thrust::complex<T>>(x)));
#else
    return static_cast<scalars::complex<T>>(std::tan(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> asin(const scalars::complex<T>& x)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::asin(static_cast<thrust::complex<T>>(x)));
#else
    return static_cast<scalars::complex<T>>(std::asin(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> acos(const scalars::complex<T>& x)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::acos(static_cast<thrust::complex<T>>(x)));
#elif !defined(_LIBCPP_VERSION)
    return static_cast<scalars::complex<T>>(std::acos(static_cast<std::complex<T>>(x)));
#else
    return _detail::acos(x);
#endif
}

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> atan(const scalars::complex<T>& x)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::atan(static_cast<thrust::complex<T>>(x)));
#else
    return static_cast<scalars::complex<T>>(std::atan(static_cast<std::complex<T>>(x)));
#endif
}

// Hyperbolic functions

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> sinh(const scalars::complex<T>& x)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::sinh(static_cast<thrust::complex<T>>(x)));
#else
    return static_cast<scalars::complex<T>>(std::sinh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> cosh(const scalars::complex<T>& x)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::cosh(static_cast<thrust::complex<T>>(x)));
#else
    return static_cast<scalars::complex<T>>(std::cosh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> tanh(const scalars::complex<T>& x)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::tanh(static_cast<thrust::complex<T>>(x)));
#else
    return static_cast<scalars::complex<T>>(std::tanh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> asinh(const scalars::complex<T>& x)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::asinh(static_cast<thrust::complex<T>>(x)));
#else
    return static_cast<scalars::complex<T>>(std::asinh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> acosh(const scalars::complex<T>& x)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::acosh(static_cast<thrust::complex<T>>(x)));
#else
    return static_cast<scalars::complex<T>>(std::acosh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> atanh(const scalars::complex<T>& x)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<scalars::complex<T>>(thrust::atanh(static_cast<thrust::complex<T>>(x)));
#else
    return static_cast<scalars::complex<T>>(std::atanh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> log1p(const scalars::complex<T>& z)
{
#if defined(__APPLE__) || defined(__MACOSX) || defined(__CUDACC__) || defined(__HIPCC__) ||        \
    defined(__SYCL_DEVICE_ONLY__)
    // For Mac, the new implementation yielded a high relative error. Falling back
    // to the old version for now.
    // See https://github.com/numpy/numpy/pull/22611#issuecomment-1667945354
    // For CUDA we also use this one, as thrust::log(thrust::complex) takes
    // *forever* to compile

    // log1p(z) = log(1 + z)
    // Let's define 1 + z = r * e ^ (i * a), then we have
    // log(r * e ^ (i * a)) = log(r) + i * a
    // With z = x + iy, the term r can be written as
    // r = ((1 + x) ^ 2 + y ^ 2) ^ 0.5
    //   = (1 + x ^ 2 + 2 * x + y ^ 2) ^ 0.5
    // So, log(r) is
    // log(r) = 0.5 * log(1 + x ^ 2 + 2 * x + y ^ 2)
    //        = 0.5 * log1p(x * (x + 2) + y ^ 2)
    // we need to use the expression only on certain condition to avoid overflow
    // and underflow from `(x * (x + 2) + y ^ 2)`
    T x     = z.real();
    T y     = z.imag();
    T zabs  = std::abs(z);
    T theta = std::atan2(y, x + T(1));
    if (zabs < 0.5)
    {
        T r = x * (T(2) + x) + y * y;
        if (r == 0)
        {  // handle underflow
            return {x, theta};
        }
        return {T(0.5) * std::log1p(r), theta};
    }
    else
    {
        T z0 = std::hypot(x + 1, y);
        return {std::log(z0), theta};
    }
#else
    // CPU path
    // Based on https://github.com/numpy/numpy/pull/22611#issuecomment-1667945354
    scalars::complex<T> u = z + T(1);
    if (u == T(1))
    {
        return z;
    }
    else
    {
        auto log_u = log(u);
        if (u - T(1) == z)
        {
            return log_u;
        }
        return log_u * (z / (u - T(1)));
    }
#endif
}

template <typename T> SCALARS_HOST_DEVICE inline scalars::complex<T> expm1(const scalars::complex<T>& z)
{
    // expm1(z) = exp(z) - 1
    // Define z = x + i * y
    // f = e ^ (x + i * y) - 1
    //   = e ^ x * e ^ (i * y) - 1
    //   = (e ^ x * cos(y) - 1) + i * (e ^ x * sin(y))
    //   = (e ^ x - 1) * cos(y) - (1 - cos(y)) + i * e ^ x * sin(y)
    //   = expm1(x) * cos(y) - 2 * sin(y / 2) ^ 2 + i * e ^ x * sin(y)
    T x  = z.real();
    T y  = z.imag();
    T a  = std::sin(y / 2);
    T er = std::expm1(x) * std::cos(y) - T(2) * a * a;
    T ei = std::exp(x) * std::sin(y);
    return {er, ei};
}

}  // namespace scalars::complex_math

using scalars::complex_math::acos;
using scalars::complex_math::acosh;
using scalars::complex_math::asin;
using scalars::complex_math::asinh;
using scalars::complex_math::atan;
using scalars::complex_math::atanh;
using scalars::complex_math::cos;
using scalars::complex_math::cosh;
using scalars::complex_math::exp;
using scalars::complex_math::expm1;
using scalars::complex_math::log;
using scalars::complex_math::log10;
using scalars::complex_math::log1p;
using scalars::complex_math::log2;
using scalars::complex_math::pow;
using scalars::complex_math::sin;
using scalars::complex_math::sinh;
using scalars::complex_math::sqrt;
using scalars::complex_math::tan;
using scalars::complex_math::tanh;

namespace std
{

using scalars::complex_math::acos;
using scalars::complex_math::acosh;
using scalars::complex_math::asin;
using scalars::complex_math::asinh;
using scalars::complex_math::atan;
using scalars::complex_math::atanh;
using scalars::complex_math::cos;
using scalars::complex_math::cosh;
using scalars::complex_math::exp;
using scalars::complex_math::expm1;
using scalars::complex_math::log;
using scalars::complex_math::log10;
using scalars::complex_math::log1p;
using scalars::complex_math::log2;
using scalars::complex_math::pow;
using scalars::complex_math::sin;
using scalars::complex_math::sinh;
using scalars::complex_math::sqrt;
using scalars::complex_math::tan;
using scalars::complex_math::tanh;

}  // namespace std
