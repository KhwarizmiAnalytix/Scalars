#pragma once

// GPU scalar backend for complex<float> / complex<double> — shared by CUDA
// (nvcc) and HIP (hipcc), mirroring backend/gpu/float/simd.h and
// backend/gpu/double/simd.h: one thread per element (size = 1), so there is
// no register-level lane packing the way the CPU complex backends have.
//
// Unlike the real float/double GPU backends (which wrap distinct f-suffixed
// vs. unsuffixed libm entry points per type), math/complex_math.h and
// types/complex.h already template their scalar complex functions over T,
// so a single generic implementation serves both complex<float> and
// complex<double> without duplicating a math wrapper per type; only the two
// explicit simd<complex<T>> specializations below are type-specific, and
// those are thin.
//
// Real-only ordering (gt/lt/min/max/hmin/hmax) and hypot/cbrt/signcopy/cdf/
// inv_cdf are deleted rather than omitted, matching
// backend/cpu/detail/complex_simd.h's policy: complex numbers have no total
// order, and those other functions have no agreed complex contract.

#include <cmath>
#include <cstdint>
#include <numbers>
#include <type_traits>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#include "common/accumulate_type.h"
#include "scalars_macros.h"
#include "types/complex.h"

// Primary template forward declaration required before the explicit specialization.
template <typename T> struct simd;

namespace scalars::detail_gpu_complex
{
template <class T> struct complex_gpu_simd
{
    using value_t              = scalars::complex<T>;
    using simd_t               = value_t;
    using real_packet_t        = T;
    using mask_t               = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
    using int_t                = mask_t;
    using simd_int_t           = mask_t;
    using accumulation_t       = scalars::acc_type<value_t, true>;
    static constexpr int  size = 1;
    static constexpr bool supports_ordering = false;

    SCALARS_SIMD_RETURN_TYPE prefetch(const value_t*) {}

    // -------------------------------------------------------------------------
    // Load / store / set
    // -------------------------------------------------------------------------

    SCALARS_SIMD_METHOD simd_t load(const value_t* addr) { return *addr; }
    SCALARS_SIMD_METHOD simd_t loadu(const value_t* addr) { return *addr; }
    SCALARS_SIMD_RETURN_TYPE   store(simd_t from, value_t* to) { *to = from; }
    SCALARS_SIMD_RETURN_TYPE   storeu(simd_t from, value_t* to) { *to = from; }

    SCALARS_SIMD_METHOD simd_t set(value_t alpha) { return alpha; }
    template <typename U, typename std::enable_if<std::is_arithmetic<U>::value, bool>::type = true>
    SCALARS_SIMD_METHOD simd_t set(U alpha)
    {
        return value_t(static_cast<T>(alpha), T(0));
    }
    SCALARS_SIMD_METHOD simd_t setzero() { return value_t(T(0), T(0)); }

    // -------------------------------------------------------------------------
    // Arithmetic
    // -------------------------------------------------------------------------

    SCALARS_SIMD_METHOD simd_t add(simd_t x, simd_t y) { return x + y; }
    SCALARS_SIMD_METHOD simd_t sub(simd_t x, simd_t y) { return x - y; }
    SCALARS_SIMD_METHOD simd_t mul(simd_t x, simd_t y) { return x * y; }
    SCALARS_SIMD_METHOD simd_t div(simd_t x, simd_t y) { return x / y; }
    SCALARS_SIMD_METHOD simd_t fma(simd_t x, simd_t y, simd_t z) { return x * y + z; }
    SCALARS_SIMD_METHOD simd_t sqr(simd_t x) { return x * x; }
    SCALARS_SIMD_METHOD simd_t neg(simd_t x) { return -x; }
    SCALARS_SIMD_METHOD simd_t conj(simd_t x) { return value_t(x.real(), -x.imag()); }

    // -------------------------------------------------------------------------
    // Magnitude and components
    // -------------------------------------------------------------------------

    SCALARS_SIMD_METHOD real_packet_t real(simd_t x) { return x.real(); }
    SCALARS_SIMD_METHOD real_packet_t imag(simd_t x) { return x.imag(); }
    SCALARS_SIMD_METHOD real_packet_t abs2(simd_t x)
    {
        return x.real() * x.real() + x.imag() * x.imag();
    }
    SCALARS_SIMD_METHOD real_packet_t norm(simd_t x) { return abs2(x); }
    SCALARS_SIMD_METHOD real_packet_t abs(simd_t x) { return scalars::abs(x); }
    SCALARS_SIMD_METHOD real_packet_t fabs(simd_t x) { return abs(x); }
    SCALARS_SIMD_METHOD real_packet_t arg(simd_t x) { return scalars::arg(x); }

    // -------------------------------------------------------------------------
    // Roots, powers, and transcendentals — the scalar complex implementation
    // already in math/complex_math.h / types/complex.h, already SCALARS_HOST_DEVICE.
    // -------------------------------------------------------------------------

    SCALARS_SIMD_METHOD simd_t sqrt(simd_t x) { return scalars::sqrt(x); }
    SCALARS_SIMD_METHOD simd_t reciprocal(simd_t x) { return value_t(T(1), T(0)) / x; }
    SCALARS_SIMD_METHOD simd_t invsqrt(simd_t x) { return reciprocal(sqrt(x)); }
    SCALARS_SIMD_METHOD simd_t pow(simd_t x, simd_t y) { return scalars::pow(x, y); }
    SCALARS_SIMD_METHOD simd_t exp(simd_t x) { return scalars::exp(x); }
    SCALARS_SIMD_METHOD simd_t expm1(simd_t x) { return scalars::expm1(x); }
    // No scalar complex exp2/exp10 in math/complex_math.h; compose from exp,
    // matching backend/cpu/detail/complex_simd.h's exp2/exp10.
    SCALARS_SIMD_METHOD simd_t exp2(simd_t x) { return scalars::exp(x * std::numbers::ln2_v<T>); }
    SCALARS_SIMD_METHOD simd_t exp10(simd_t x) { return scalars::exp(x * std::numbers::ln10_v<T>); }
    SCALARS_SIMD_METHOD simd_t log(simd_t x) { return scalars::log(x); }
    SCALARS_SIMD_METHOD simd_t log1p(simd_t x) { return scalars::log1p(x); }
    SCALARS_SIMD_METHOD simd_t log2(simd_t x) { return scalars::log(x) / std::numbers::ln2_v<T>; }
    SCALARS_SIMD_METHOD simd_t log10(simd_t x) { return scalars::log10(x); }

    SCALARS_SIMD_METHOD simd_t sin(simd_t x) { return scalars::sin(x); }
    SCALARS_SIMD_METHOD simd_t cos(simd_t x) { return scalars::cos(x); }
    SCALARS_SIMD_METHOD simd_t tan(simd_t x) { return scalars::tan(x); }
    SCALARS_SIMD_METHOD simd_t asin(simd_t x) { return scalars::asin(x); }
    SCALARS_SIMD_METHOD simd_t acos(simd_t x) { return scalars::acos(x); }
    SCALARS_SIMD_METHOD simd_t atan(simd_t x) { return scalars::atan(x); }
    SCALARS_SIMD_METHOD simd_t sinh(simd_t x) { return scalars::sinh(x); }
    SCALARS_SIMD_METHOD simd_t cosh(simd_t x) { return scalars::cosh(x); }
    SCALARS_SIMD_METHOD simd_t tanh(simd_t x) { return scalars::tanh(x); }
    SCALARS_SIMD_METHOD simd_t asinh(simd_t x) { return scalars::asinh(x); }
    SCALARS_SIMD_METHOD simd_t acosh(simd_t x) { return scalars::acosh(x); }
    SCALARS_SIMD_METHOD simd_t atanh(simd_t x) { return scalars::atanh(x); }

    SCALARS_SIMD_METHOD simd_t ceil(simd_t x)
    {
        return value_t(std::ceil(x.real()), std::ceil(x.imag()));
    }
    SCALARS_SIMD_METHOD simd_t floor(simd_t x)
    {
        return value_t(std::floor(x.real()), std::floor(x.imag()));
    }
    SCALARS_SIMD_METHOD simd_t trunc(simd_t x)
    {
        return value_t(std::trunc(x.real()), std::trunc(x.imag()));
    }

    // -------------------------------------------------------------------------
    // Gather / scatter
    // -------------------------------------------------------------------------

    SCALARS_SIMD_METHOD simd_t gather(const value_t* base, const int* offsets)
    {
        return base[offsets[0]];
    }
    // On GPU, stride is the pre-computed absolute element index for this thread.
    SCALARS_SIMD_METHOD simd_t gather(const value_t* base, int stride) { return base[stride]; }
    SCALARS_SIMD_RETURN_TYPE   scatter(simd_t val, const int* offsets, value_t* to)
    {
        to[offsets[0]] = val;
    }
    SCALARS_SIMD_RETURN_TYPE scatter(simd_t val, int stride, value_t* to) { to[stride] = val; }

    // -------------------------------------------------------------------------
    // Reductions — trivial identity for size=1
    // -------------------------------------------------------------------------

    SCALARS_SIMD_METHOD accumulation_t accumulate(simd_t x) { return accumulation_t(x); }

    // -------------------------------------------------------------------------
    // Comparisons, masks, classification
    // -------------------------------------------------------------------------

    SCALARS_SIMD_METHOD mask_t eq(simd_t x, simd_t y)
    {
        return (x.real() == y.real() && x.imag() == y.imag()) ? ~mask_t(0) : mask_t(0);
    }
    SCALARS_SIMD_METHOD mask_t neq(simd_t x, simd_t y) { return ~eq(x, y); }
    SCALARS_SIMD_METHOD mask_t and_mask(mask_t x, mask_t y) { return x & y; }
    SCALARS_SIMD_METHOD mask_t or_mask(mask_t x, mask_t y) { return x | y; }
    SCALARS_SIMD_METHOD mask_t xor_mask(mask_t x, mask_t y) { return x ^ y; }
    SCALARS_SIMD_METHOD mask_t not_mask(mask_t x) { return ~x; }
    SCALARS_SIMD_METHOD simd_t if_else(mask_t m, simd_t t, simd_t f) { return m ? t : f; }

    SCALARS_SIMD_METHOD mask_t isnan(simd_t x)
    {
        return (std::isnan(x.real()) || std::isnan(x.imag())) ? ~mask_t(0) : mask_t(0);
    }
    SCALARS_SIMD_METHOD mask_t isinf(simd_t x)
    {
        return (std::isinf(x.real()) || std::isinf(x.imag())) ? ~mask_t(0) : mask_t(0);
    }
    SCALARS_SIMD_METHOD mask_t isfinite(simd_t x)
    {
        return (std::isfinite(x.real()) && std::isfinite(x.imag())) ? ~mask_t(0) : mask_t(0);
    }

    // Complex values have no ordering. Real-only analytic contracts are also
    // deliberately excluded instead of silently applying them componentwise,
    // matching detail_complex_simd::complex_simd<T, Ops>.
    SCALARS_SIMD_METHOD mask_t  gt(simd_t, simd_t)       = delete;
    SCALARS_SIMD_METHOD mask_t  ge(simd_t, simd_t)       = delete;
    SCALARS_SIMD_METHOD mask_t  lt(simd_t, simd_t)       = delete;
    SCALARS_SIMD_METHOD mask_t  le(simd_t, simd_t)       = delete;
    SCALARS_SIMD_METHOD simd_t  min(simd_t, simd_t)      = delete;
    SCALARS_SIMD_METHOD simd_t  max(simd_t, simd_t)      = delete;
    SCALARS_SIMD_METHOD value_t hmin(simd_t)             = delete;
    SCALARS_SIMD_METHOD value_t hmax(simd_t)             = delete;
    SCALARS_SIMD_METHOD simd_t  signcopy(simd_t, simd_t) = delete;
    SCALARS_SIMD_METHOD simd_t  hypot(simd_t, simd_t)    = delete;
    SCALARS_SIMD_METHOD simd_t  cbrt(simd_t)             = delete;
    SCALARS_SIMD_METHOD simd_t  cdf(simd_t)              = delete;
    SCALARS_SIMD_METHOD simd_t  inv_cdf(simd_t)          = delete;
};
}  // namespace scalars::detail_gpu_complex

template <>
struct simd<scalars::complex<float>> : scalars::detail_gpu_complex::complex_gpu_simd<float>
{
};
template <>
struct simd<scalars::complex<double>> : scalars::detail_gpu_complex::complex_gpu_simd<double>
{
};
