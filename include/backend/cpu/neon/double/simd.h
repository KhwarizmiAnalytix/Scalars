#pragma once

#include "common/intrinsics.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "backend/cpu/neon/svml.h"
#include "backend/simd.h"
#include "scalars_macros.h"

namespace scalars
{
namespace detail_neon
{
SCALARS_FORCE_INLINE float64x2_t map1_f64(float64x2_t v, double (*fn)(double))
{
    alignas(16) double b[2];
    vst1q_f64(b, v);
    b[0] = fn(b[0]);
    b[1] = fn(b[1]);
    return vld1q_f64(b);
}

// See hypot_f32 in the float NEON header for the rationale: scale by the
// larger magnitude to avoid premature overflow/underflow, and resolve
// inf/NaN explicitly to match std::hypot (inf beats NaN; otherwise NaN
// propagates).
SCALARS_FORCE_INLINE float64x2_t hypot_f64(float64x2_t x, float64x2_t y)
{
    const uint64x2_t abs_mask = vdupq_n_u64(0x7FFFFFFFFFFFFFFFULL);
    const uint64x2_t inf_bits = vdupq_n_u64(0x7FF0000000000000ULL);
    const float64x2_t zero    = vdupq_n_f64(0.);
    const float64x2_t one     = vdupq_n_f64(1.);

    const uint64x2_t ax_bits = vandq_u64(vreinterpretq_u64_f64(x), abs_mask);
    const uint64x2_t ay_bits = vandq_u64(vreinterpretq_u64_f64(y), abs_mask);
    const float64x2_t ax     = vreinterpretq_f64_u64(ax_bits);
    const float64x2_t ay     = vreinterpretq_f64_u64(ay_bits);

    const float64x2_t m = vmaxq_f64(ax, ay);
    const float64x2_t n = vminq_f64(ax, ay);

    const uint64x2_t m_is_zero = vceqq_f64(m, zero);
    const float64x2_t safe_m   = vbslq_f64(m_is_zero, one, m);
    const float64x2_t t        = vdivq_f64(n, safe_m);
    const float64x2_t scaled   = vmulq_f64(m, vsqrtq_f64(vfmaq_f64(one, t, t)));
    float64x2_t result         = vbslq_f64(m_is_zero, zero, scaled);

    const uint64x2_t any_inf =
        vorrq_u64(vceqq_u64(ax_bits, inf_bits), vceqq_u64(ay_bits, inf_bits));
    const uint64x2_t not_mask = vdupq_n_u64(~0ULL);
    const uint64x2_t any_nan  = vorrq_u64(
        veorq_u64(vceqq_f64(x, x), not_mask), veorq_u64(vceqq_f64(y, y), not_mask));
    const float64x2_t nan = vdupq_n_f64(std::numeric_limits<double>::quiet_NaN());
    const float64x2_t inf = vdupq_n_f64(std::numeric_limits<double>::infinity());

    result = vbslq_f64(vbicq_u64(any_nan, any_inf), nan, result);
    result = vbslq_f64(any_inf, inf, result);
    return result;
}
}  // namespace detail_neon

template <> struct simd<double>
{
    using simd_t      = float64x2_t;
    using mask_t      = uint64x2_t;
    using simd_half_t = float64x2_t;
    using simd_int_t  = uint32x2_t;
    using value_t     = double;
    using int_t       = uint32_t;

    static constexpr int size      = 2;
    static constexpr int half_size = 1;

    // NOLINTNEXTLINE(bugprone-throwing-static-initialization): NEON intrinsic, cannot throw.
    inline static uint64x2_t const sign_mask = vdupq_n_u64(0x8000000000000000ULL);

    SCALARS_SIMD_RETURN_TYPE prefetch(const value_t* addr) { __builtin_prefetch(addr, 0, 3); }

    SCALARS_SIMD_METHOD simd_t load(const value_t* addr) { return vld1q_f64(addr); }

    SCALARS_SIMD_METHOD simd_t loadu(const value_t* addr) { return vld1q_f64(addr); }

    SCALARS_SIMD_RETURN_TYPE store(simd_t from, value_t* to) { vst1q_f64(to, from); }

    SCALARS_SIMD_RETURN_TYPE storeu(simd_t from, value_t* to) { vst1q_f64(to, from); }

    template <typename scalar_t,
        typename std::enable_if<std::is_fundamental<scalar_t>::value, bool>::type = true>
    SCALARS_SIMD_METHOD simd_t set(const scalar_t alpha)
    {
        return vdupq_n_f64(static_cast<value_t>(alpha));
    }

    // Same fix as backend/cpu/detail/complex_simd.h::set_mask: a distinct
    // name instead of overloading set(), since the non-template set(int_t)
    // won ties against the set(scalar_t) template for an exact int_t
    // argument, silently returning a mask instead of a broadcast packet.
    SCALARS_SIMD_METHOD mask_t set_mask(int_t const from)
    {
        return vdupq_n_u64(from != 0 ? ~0ULL : 0ULL);
    }

    SCALARS_SIMD_METHOD simd_t setzero() { return vdupq_n_f64(0.); }

    SCALARS_SIMD_METHOD simd_t add(simd_t x, simd_t y) { return vaddq_f64(x, y); }

    SCALARS_SIMD_METHOD simd_t sub(simd_t x, simd_t y) { return vsubq_f64(x, y); }

    SCALARS_SIMD_METHOD simd_t mul(simd_t x, simd_t y) { return vmulq_f64(x, y); }

    SCALARS_SIMD_METHOD simd_t div(simd_t x, simd_t y) { return vdivq_f64(x, y); }

    SCALARS_SIMD_METHOD simd_t fma(simd_t x, simd_t y, simd_t z) { return vfmaq_f64(z, x, y); }

    SCALARS_SIMD_METHOD simd_t pow(simd_t x, simd_t y) { return vpowq_f64(x, y); }

    SCALARS_SIMD_METHOD simd_t hypot(simd_t x, simd_t y)
    {
        return scalars::detail_neon::hypot_f64(x, y);
    }

    SCALARS_SIMD_METHOD simd_t min(simd_t x, simd_t y) { return vminq_f64(x, y); }

    SCALARS_SIMD_METHOD simd_t max(simd_t x, simd_t y) { return vmaxq_f64(x, y); }

    SCALARS_SIMD_METHOD simd_t signcopy(simd_t x, simd_t sign)
    {
        uint64x2_t xs = vreinterpretq_u64_f64(x);
        uint64x2_t ss = vreinterpretq_u64_f64(sign);
        return vreinterpretq_f64_u64(vorrq_u64(vandq_u64(sign_mask, ss), vbicq_u64(xs, sign_mask)));
    }

    SCALARS_SIMD_METHOD simd_t sqrt(simd_t x) { return vsqrtq_f64(x); }

    SCALARS_SIMD_METHOD simd_t sqr(simd_t x) { return vmulq_f64(x, x); }

    SCALARS_SIMD_METHOD simd_t ceil(simd_t x) { return vrndpq_f64(x); }

    SCALARS_SIMD_METHOD simd_t floor(simd_t x) { return vrndmq_f64(x); }

    SCALARS_SIMD_METHOD simd_t exp(simd_t x) { return vexpq_f64(x); }
    SCALARS_SIMD_METHOD simd_t expm1(simd_t x) { return vexpm1q_f64(x); }
    SCALARS_SIMD_METHOD simd_t exp2(simd_t x) { return vexp2q_f64(x); }
    SCALARS_SIMD_METHOD simd_t exp10(simd_t x) { return vexp10q_f64(x); }
    SCALARS_SIMD_METHOD simd_t log(simd_t x) { return vlogq_f64(x); }
    SCALARS_SIMD_METHOD simd_t log1p(simd_t x) { return vlog1pq_f64(x); }
    SCALARS_SIMD_METHOD simd_t log2(simd_t x) { return vlog2q_f64(x); }
    SCALARS_SIMD_METHOD simd_t log10(simd_t x) { return vlog10q_f64(x); }
    SCALARS_SIMD_METHOD simd_t sin(simd_t x) { return vsinq_f64(x); }
    SCALARS_SIMD_METHOD simd_t cos(simd_t x) { return vcosq_f64(x); }
    SCALARS_SIMD_METHOD simd_t atan2(simd_t y, simd_t x) { return vatan2q_f64(y, x); }
    SCALARS_SIMD_RETURN_TYPE   sincos(simd_t x, simd_t& sine, simd_t& cosine)
    {
        vsincosq_f64(x, sine, cosine);
    }
    SCALARS_SIMD_METHOD simd_t tan(simd_t x) { return vtanq_f64(x); }
    SCALARS_SIMD_METHOD simd_t asin(simd_t x) { return vasinq_f64(x); }
    SCALARS_SIMD_METHOD simd_t acos(simd_t x) { return vacosq_f64(x); }
    SCALARS_SIMD_METHOD simd_t atan(simd_t x) { return vatanq_f64(x); }
    SCALARS_SIMD_METHOD simd_t sinh(simd_t x) { return vsinhq_f64(x); }
    SCALARS_SIMD_METHOD simd_t cosh(simd_t x) { return vcoshq_f64(x); }
    SCALARS_SIMD_METHOD simd_t tanh(simd_t x) { return vtanhq_f64(x); }
    SCALARS_SIMD_METHOD simd_t asinh(simd_t x) { return vasinhq_f64(x); }
    SCALARS_SIMD_METHOD simd_t acosh(simd_t x) { return vacoshq_f64(x); }
    SCALARS_SIMD_METHOD simd_t atanh(simd_t x) { return vatanhq_f64(x); }
    SCALARS_SIMD_METHOD simd_t cbrt(simd_t x) { return vcbrtq_f64(x); }

    // Neither Accelerate nor SLEEF has a normal CDF, so the vectorized port
    // serves cdf/inv_cdf whatever SCALARS_MATH_BACKEND selects.
    SCALARS_SIMD_METHOD simd_t cdf(simd_t x)
    {
        return detail_neon::pmf64::normal_cdf(x);
    }

    SCALARS_SIMD_METHOD simd_t inv_cdf(simd_t x)
    {
        return detail_neon::pmf64::inv_normal_cdf(x);
    }

    SCALARS_SIMD_METHOD simd_t trunc(simd_t x) { return vrndq_f64(x); }

    SCALARS_SIMD_METHOD simd_t invsqrt(simd_t x)
    {
        // vrsqrte estimate refined by two vrsqrts Newton steps.
        const auto refine = [](simd_t v, simd_t r)
        {
            r = vmulq_f64(vrsqrtsq_f64(vmulq_f64(v, r), r), r);
            return vmulq_f64(vrsqrtsq_f64(vmulq_f64(v, r), r), r);
        };
        const simd_t inf = set(std::numeric_limits<double>::infinity());
        const simd_t y   = refine(x, vrsqrteq_f64(x));
        // Positive normal inputs give finite results. Packets with any other lane, which are rare,
        // take a second pass: the steps are 0 * inf = NaN where the estimate is already exact
        // (+-0 -> +-inf, +inf -> 0), so those lanes keep it, and subnormals are scaled by 2^54
        // into the normal range and the result by 2^27.
        if (vminvq_u32(vreinterpretq_u32_u64(vcltq_f64(fabs(y), inf))) != 0)
        {
            return y;
        }
        const mask_t tiny  = lt(fabs(x), set(std::numeric_limits<double>::min()));
        const simd_t xs    = if_else(tiny, mul(x, set(0x1p54)), x);
        const simd_t r     = vrsqrteq_f64(xs);
        const mask_t exact = or_mask(eq(xs, setzero()), eq(fabs(xs), inf));
        const simd_t z     = if_else(exact, r, refine(xs, r));
        return if_else(tiny, mul(z, set(0x1p27)), z);
    }

    SCALARS_SIMD_METHOD simd_t fabs(simd_t x)
    {
        return vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(x), sign_mask));
    }

    SCALARS_SIMD_METHOD simd_t neg(simd_t x) { return vnegq_f64(x); }

    SCALARS_FORCE_INLINE static double accumulate(simd_t x)
    {
        return vgetq_lane_f64(x, 0) + vgetq_lane_f64(x, 1);
    }

    SCALARS_FORCE_INLINE static value_t hmax(simd_t x)
    {
        return (std::max)(vgetq_lane_f64(x, 0), vgetq_lane_f64(x, 1));
    }

    SCALARS_FORCE_INLINE static value_t hmin(simd_t x)
    {
        return (std::min)(vgetq_lane_f64(x, 0), vgetq_lane_f64(x, 1));
    }

    SCALARS_SIMD_METHOD simd_t gather(value_t const* from, int stride)
    {
        alignas(16) value_t tmp[2] = {from[0], from[stride]};
        return vld1q_f64(tmp);
    }

    SCALARS_SIMD_RETURN_TYPE scatter(simd_t from, int stride, value_t* to)
    {
        alignas(16) value_t b[2];
        vst1q_f64(b, from);
        to[0]      = b[0];
        to[stride] = b[1];
    }

    SCALARS_SIMD_METHOD simd_t gather(value_t const* from, const int* strides)
    {
        alignas(16) value_t tmp[2] = {from[strides[0]], from[strides[1]]};
        return vld1q_f64(tmp);
    }

    SCALARS_SIMD_RETURN_TYPE scatter(simd_t from, const int* stride, value_t* to)
    {
        alignas(16) value_t b[2];
        vst1q_f64(b, from);
        to[stride[0]] = b[0];
        to[stride[1]] = b[1];
    }

    SCALARS_SIMD_METHOD simd_t if_else(const mask_t& x, simd_t y, simd_t z)
    {
        return vbslq_f64(x, y, z);
    }

    SCALARS_SIMD_METHOD mask_t eq(simd_t x, simd_t y) { return vceqq_f64(x, y); }

    SCALARS_SIMD_METHOD mask_t neq(simd_t x, simd_t y)
    {
        return veorq_u64(vceqq_f64(x, y), vdupq_n_u64(~0ULL));
    }

    SCALARS_SIMD_METHOD mask_t gt(simd_t x, simd_t y) { return vcgtq_f64(x, y); }

    SCALARS_SIMD_METHOD mask_t lt(simd_t x, simd_t y) { return vcltq_f64(x, y); }

    SCALARS_SIMD_METHOD mask_t ge(simd_t x, simd_t y) { return vcgeq_f64(x, y); }

    SCALARS_SIMD_METHOD mask_t le(simd_t x, simd_t y) { return vcleq_f64(x, y); }

    SCALARS_FORCE_INLINE static uint64x2_t u32_mask_to_u64(uint32_t a, uint32_t b)
    {
        uint64_t   la = a ? ~0ULL : 0ULL;
        uint64_t   lb = b ? ~0ULL : 0ULL;
        uint64x2_t r  = vdupq_n_u64(0);
        r             = vsetq_lane_u64(la, r, 0);
        r             = vsetq_lane_u64(lb, r, 1);
        return r;
    }

    SCALARS_SIMD_METHOD mask_t loadu(int_t const* from)
    {
        return u32_mask_to_u64(from[0], from[1]);
    }

    SCALARS_SIMD_METHOD mask_t load(int_t const* from) { return u32_mask_to_u64(from[0], from[1]); }

    SCALARS_SIMD_RETURN_TYPE storeu(const mask_t& from, int_t* to)
    {
        to[0] = vgetq_lane_u64(from, 0) ? 1u : 0u;
        to[1] = vgetq_lane_u64(from, 1) ? 1u : 0u;
    }

    SCALARS_SIMD_RETURN_TYPE store(const mask_t& from, int_t* to) { storeu(from, to); }

    SCALARS_SIMD_METHOD mask_t not_mask(const mask_t& x)
    {
        return veorq_u64(x, vdupq_n_u64(~0ULL));
    }

    SCALARS_SIMD_METHOD mask_t and_mask(const mask_t& x, const mask_t& y)
    {
        return vandq_u64(x, y);
    }

    SCALARS_SIMD_METHOD mask_t or_mask(const mask_t& x, const mask_t& y) { return vorrq_u64(x, y); }

    SCALARS_SIMD_METHOD mask_t xor_mask(const mask_t& x, const mask_t& y)
    {
        return veorq_u64(x, y);
    }

    SCALARS_SIMD_RETURN_TYPE broadcast(
        const double* from, simd_t& a0, simd_t& a1, simd_t& a2, simd_t& a3)
    {
        a0 = vdupq_n_f64(from[0]);
        a1 = vdupq_n_f64(from[1]);
        a2 = vdupq_n_f64(from[2]);
        a3 = vdupq_n_f64(from[3]);
    }

    template <int N> SCALARS_SIMD_RETURN_TYPE ptranspose(simd_t simd[N])  // NOLINT
    {
        constexpr int L = size;
        if constexpr (N == 2 && L == 2)
        {
            float64x2_t t0 = vzip1q_f64(simd[0], simd[1]);
            float64x2_t t1 = vzip2q_f64(simd[0], simd[1]);
            simd[0]        = t0;
            simd[1]        = t1;
        }
        else if constexpr (N == L)
        {
            alignas(16) double buf[N * L];
            for (int i = 0; i < N; ++i)
            {
                vst1q_f64(buf + i * L, simd[i]);
            }
            alignas(16) double out[N * L];
            for (int i = 0; i < N; ++i)
            {
                for (int j = 0; j < L; ++j)
                {
                    out[i * L + j] = buf[j * L + i];
                }
            }
            for (int i = 0; i < N; ++i)
            {
                simd[i] = vld1q_f64(out + i * L);
            }
        }
        else if constexpr (N == 8 && L == 2)
        {
            alignas(16) double buf[8][2];
            for (int i = 0; i < 8; ++i)
            {
                vst1q_f64(buf[i], simd[i]);
            }
            alignas(16) double out[2][8];
            for (int i = 0; i < 8; ++i)
            {
                for (int j = 0; j < 2; ++j)
                {
                    out[j][i] = buf[i][j];
                }
            }
            for (int r = 0; r < 2; ++r)
            {
                for (int q = 0; q < 4; ++q)
                {
                    simd[4 * r + q] = vld1q_f64(&out[r][static_cast<std::ptrdiff_t>(2 * q)]);
                }
            }
        }
        else if constexpr (N == 4 && L == 2)
        {
            alignas(16) double buf[4][2];
            for (int i = 0; i < 4; ++i)
            {
                vst1q_f64(buf[i], simd[i]);
            }
            alignas(16) double out[2][4];
            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 2; ++j)
                {
                    out[j][i] = buf[i][j];
                }
            }
            for (int r = 0; r < 2; ++r)
            {
                for (int q = 0; q < 2; ++q)
                {
                    simd[2 * r + q] = vld1q_f64(&out[r][static_cast<std::ptrdiff_t>(2 * q)]);
                }
            }
        }
        else
        {
            static_assert(sizeof(simd_t) == 0 && N != 0, "ptranspose: unsupported N for NEON double");
        }
    }

    SCALARS_SIMD_RETURN_TYPE copy(const value_t* from, value_t* to)
    {
        vst1q_f64(to, vld1q_f64(from));
    }

    SCALARS_FORCE_INLINE static simd_t ploadquad(const double* from)
    {
        return vdupq_n_f64(from[0]);
    }
};

}  // namespace scalars
