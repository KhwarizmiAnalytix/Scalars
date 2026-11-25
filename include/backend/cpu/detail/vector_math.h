#pragma once

// SPDX-License-Identifier: MPL-2.0
//
// Vectorized elementary functions for the CPU SIMD backends, ported from
// Eigen's generic packet math:
//   Eigen/src/Core/arch/Default/GenericPacketMathFunctions.h
//   Eigen/src/Core/arch/Default/GenericPacketMathTrig.h
//   Eigen/src/Core/arch/Default/GenericPacketMathPow.h
//   Eigen/src/Core/arch/Default/GenericPacketMathFrexpLdexp.h
//   Eigen/src/Core/arch/Default/GenericPacketMathDoubleWord.h
//   unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h (Eigen 5.0)
// The exp and log functions there initially come from Julien Pommier's sse
// math library (http://gruntthepeon.free.fr/ssemath/); several kernels and
// ndtri come from the Cephes Math Library (Stephen L. Moshier).
//
// Copyright (C) 2007 Julien Pommier
// Copyright (C) 2014 Pedro Gonnet (pedro.gonnet@gmail.com)
// Copyright (C) 2009-2019 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2018-2025 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/. The rest of Scalars is MIT
// licensed; see NOTICE.
//
// Functions not present in Eigen (exp10, atan2, hypot, double asin/acos, and
// the expm1-based double tanh) are built here from the ported kernels, as are
// float acos (from Eigen's asin kernel) and the double-word argument of the
// normal distribution function.
//
// The algorithms are written once against a per-ISA operations struct `O`
// (see backend/cpu/x86/vector_math_ops.h and backend/cpu/neon/
// vector_math_ops.h). O provides, for vec_t lanes of scalar_t:
//
//   vec_t, scalar_t, size, has_fma
//   set(s), set_bits(u), zero(), loadu(p), storeu(p, v)
//   add sub mul div sqrt; fma(a, b, c) = a * b + c, rounded once iff has_fma
//   min(a, b), max(a, b): with a NaN operand the result is NaN or a, so a NaN
//       in the first argument always propagates
//   rint (ties to even), floor, ceil, trunc
//   lt le eq: ordered compares; nge(a, b) = !(a >= b); isnan(a)
//   select(m, a, b) = m ? a : b; band bor bxor; bandnot(a, b) = a & ~b
//   iadd isub: lane-wise integer arithmetic on the bit patterns
//   shl<N> shr<N>: logical shifts of the bit patterns
//   any(m): whether any lane of m is set
//   barrier(v&): keeps -ffast-math from reassociating across v
//
// Masks are vec_t values whose lanes are all ones or all zeros, as in Eigen.
// Accuracy is that of the Eigen kernels: within 1-3 ulp for float and double
// over their documented ranges.
//
// The public functions are SCALARS_NOINLINE, as calls into a vector math
// library would be: inlining their kernels into every caller multiplies code
// size for no gain, and the kernels stay inlined within each function.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "scalars_macros.h"

namespace scalars::detail::vmath
{
template <class T> struct ieee_traits;

template <> struct ieee_traits<float>
{
    using uint_t                           = std::uint32_t;
    static constexpr int   mantissa_bits   = 23;
    static constexpr int   exponent_bias   = 127;
    static constexpr float int_magic       = 12582912.0f;  // 1.5 * 2^23
    static constexpr float int_magic_limit = 4194304.0f;   // 2^22
};

template <> struct ieee_traits<double>
{
    using uint_t                            = std::uint64_t;
    static constexpr int    mantissa_bits   = 52;
    static constexpr int    exponent_bias   = 1023;
    static constexpr double int_magic       = 6755399441055744.0;  // 1.5 * 2^52
    static constexpr double int_magic_limit = 2251799813685248.0;  // 2^51
};

// Payne-Hanek reduction of a finite float |xf| >= 2^14 to [-pi/4, pi/4],
// with the quadrant in *quadrant. Eigen's trig_reduce_huge.
inline float trig_reduce_huge(float xf, std::int32_t* quadrant)
{
    const double        pio2_62       = 3.4061215800865545e-19;  // pi/2 * 2^-62
    const std::uint64_t zero_dot_five = std::uint64_t(1) << 61;  // 0.5 in 2.62 fixed point

    // 192 bits of 2/pi, replicated in packets of 8 for aligned reads.
    static const std::uint32_t two_over_pi[] = {
        0x00000028, 0x000028be, 0x0028be60, 0x28be60db, 0xbe60db93, 0x60db9391, 0xdb939105,
        0x9391054a, 0x91054a7f, 0x054a7f09, 0x4a7f09d5, 0x7f09d5f4, 0x09d5f47d, 0xd5f47d4d,
        0xf47d4d37, 0x7d4d3770, 0x4d377036, 0x377036d8, 0x7036d8a5, 0x36d8a566, 0xd8a5664f,
        0xa5664f10, 0x664f10e4, 0x4f10e410, 0x10e41000, 0xe4100000};

    std::uint32_t xi = 0;
    std::memcpy(&xi, &xf, sizeof(xi));
    // -118 = -126 + 8: the exponent, plus 8 to align the bits of 2/pi on 8 bits.
    const std::uint32_t e = (xi >> 23) - 118;
    xi                    = ((xi & 0x007fffffu) | 0x00800000u) << (e & 0x7);

    const std::uint32_t i        = e >> 3;
    const std::uint32_t twoopi_1 = two_over_pi[i - 1];
    const std::uint32_t twoopi_2 = two_over_pi[i + 3];
    const std::uint32_t twoopi_3 = two_over_pi[i + 7];

    // x * 2/pi in 2.62 fixed point.
    std::uint64_t p = std::uint64_t(xi) * twoopi_3;
    p               = std::uint64_t(xi) * twoopi_2 + (p >> 32);
    p               = (std::uint64_t(xi * twoopi_1) << 32) + p;

    const std::uint64_t q = (p + zero_dot_five) >> 62;
    *quadrant             = static_cast<std::int32_t>(q);
    p -= q << 62;
    return static_cast<float>(static_cast<double>(static_cast<std::int64_t>(p)) * pio2_62);
}

template <class O> struct packet_math
{
    using V = typename O::vec_t;
    using T = typename O::scalar_t;
    using U = typename ieee_traits<T>::uint_t;

    static constexpr bool kFloat   = std::is_same_v<T, float>;
    static constexpr int  kMant    = ieee_traits<T>::mantissa_bits;
    static constexpr int  kBias    = ieee_traits<T>::exponent_bias;
    static constexpr int  kBits    = int(sizeof(T) * 8);
    static constexpr int  kDigits  = std::numeric_limits<T>::digits;
    static constexpr int  kMaxExp  = std::numeric_limits<T>::max_exponent;
    static constexpr int  kSize    = O::size;
    static constexpr T    kLn2     = T(0.693147180559945309417232121458176568);
    static constexpr T    kLog2e   = T(1.442695040888963407359924681001892137);
    static constexpr T    kPi      = T(3.141592653589793238462643383279502884);
    static constexpr T    kPiOver2 = T(1.570796326794896619231321691639751442);
    static constexpr T    kEps     = std::numeric_limits<T>::epsilon();

    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

    //==========================================================================
    // Constants, bit manipulation and small helpers
    //==========================================================================
    SCALARS_FORCE_INLINE static V cst(T v) { return O::set(v); }
    SCALARS_FORCE_INLINE static V bits(U b) { return O::set_bits(b); }
    SCALARS_FORCE_INLINE static V zero() { return O::zero(); }
    SCALARS_FORCE_INLINE static V one() { return O::set(T(1)); }
    SCALARS_FORCE_INLINE static V inf() { return O::set(std::numeric_limits<T>::infinity()); }
    SCALARS_FORCE_INLINE static V sign_mask() { return bits(U(1) << (kBits - 1)); }
    SCALARS_FORCE_INLINE static V abs(V x) { return O::bandnot(x, sign_mask()); }
    SCALARS_FORCE_INLINE static V negate(V x) { return O::bxor(x, sign_mask()); }
    SCALARS_FORCE_INLINE static V sign_of(V x) { return O::band(x, sign_mask()); }
    SCALARS_FORCE_INLINE static V isinf(V x) { return O::eq(abs(x), inf()); }

    template <int N> SCALARS_FORCE_INLINE static V shl(V x) { return O::template shl<N>(x); }
    template <int N> SCALARS_FORCE_INLINE static V shr(V x) { return O::template shr<N>(x); }

    // All ones in the lanes whose sign bit is set, NaN included (Eigen psignbit).
    SCALARS_FORCE_INLINE static V signbit(V x) { return O::isub(zero(), shr<kBits - 1>(x)); }

    // Mask of the lanes whose integer bit pattern n is even.
    SCALARS_FORCE_INLINE static V int_is_even(V n)
    {
        // Bit 0 moves to the lowest exponent bit, so the lane is 0 or the smallest normal and
        // the compare holds under flush-to-zero as well.
        return O::eq(O::band(shl<kMant>(n), bits(U(1) << kMant)), zero());
    }

    SCALARS_FORCE_INLINE static V madd(V a, V b, V c) { return O::fma(a, b, c); }
    SCALARS_FORCE_INLINE static V nmadd(V a, V b, V c) { return O::fma(negate(a), b, c); }
    SCALARS_FORCE_INLINE static V msub(V a, V b, V c) { return O::fma(a, b, negate(c)); }
    SCALARS_FORCE_INLINE static V reciprocal(V x) { return O::div(one(), x); }

    // Integer lanes held in the bit pattern, for |i| < 2^22 (float) or 2^51 (double), to and
    // from integral floating-point values. 1.5 * 2^(digits - 1) has an ulp of one, so adding it
    // places the integer in the low mantissa bits.
    SCALARS_FORCE_INLINE static V int_to_float(V i)
    {
        const V magic = cst(ieee_traits<T>::int_magic);
        return O::sub(O::iadd(i, magic), magic);
    }

    SCALARS_FORCE_INLINE static V float_to_int(V n)
    {
        const V magic = cst(ieee_traits<T>::int_magic);
        return O::isub(O::add(n, magic), magic);
    }

    // Evaluates c[0] x^(N-1) + ... + c[N-1] by Horner's rule (Eigen ppolevl).
    template <std::size_t N> SCALARS_FORCE_INLINE static V polevl(V x, const T (&c)[N])
    {
        V r = cst(c[0]);
        for (std::size_t i = 1; i < N; ++i)
        {
            r = madd(r, x, cst(c[i]));
        }
        return r;
    }

    //==========================================================================
    // frexp / ldexp (GenericPacketMathFrexpLdexp.h)
    //==========================================================================

    // a = m * 2^exponent with |m| in [0.5, 1); handles subnormals. Zero, inf and NaN return a
    // with a zero exponent.
    static V frexp(V a, V& exponent)
    {
        constexpr int kExpBits       = kBits - kMant - 1;
        constexpr U   kSignMantMask  = ~(((U(1) << kExpBits) - U(1)) << kMant);
        const V       normal_min     = cst(std::numeric_limits<T>::min());
        const V       is_denormal    = O::lt(abs(a), normal_min);
        const V       normalized_a   = O::select(is_denormal, O::mul(a, cst(T(U(1) << (kMant + 1)))), a);
        const V       normal_offset  = cst(-T(kBias - 1));
        const V       exponent_offset = O::select(
            is_denormal, O::add(normal_offset, cst(-T(kMant + 1))), normal_offset);

        exponent = int_to_float(shr<kMant>(abs(normalized_a)));
        const V is_zero_or_not_finite =
            O::bor(O::eq(a, zero()), O::eq(exponent, cst(T((1 << kExpBits) - 1))));
        const V m = O::select(is_zero_or_not_finite,
                              a,
                              O::bor(O::band(normalized_a, bits(kSignMantMask)), cst(T(0.5))));
        exponent  = O::select(is_zero_or_not_finite, zero(), O::add(exponent, exponent_offset));
        return m;
    }

    // a * 2^exponent for integral exponent; the result may over- or underflow, and every
    // partial product keeps a, so no intermediate overflows where the result does not.
    static V ldexp(V a, V exponent)
    {
        constexpr int kExpBits = kBits - kMant - 1;
        const V       max_exp  = cst(T((1 << kExpBits) + (kMant - 1)));  // 278 / 2099
        const V       last_max = cst(T(kBias - 1));                       // 126 / 1022
        const V       e = float_to_int(O::min(O::max(exponent, negate(max_exp)), max_exp));
        const V       b = float_to_int(O::min(O::max(exponent, negate(last_max)), last_max));
        const V       t = O::bandnot(O::isub(e, b), bits(U(1)));  // even
        const V       c1 = shl<kMant - 1>(O::iadd(t, bits(U(2 * kBias))));       // 2^(t/2)
        const V       c2 = shl<kMant>(O::iadd(O::isub(e, t), bits(U(kBias))));  // 2^(e-t)
        V             out = O::mul(a, c1);
        O::barrier(out);
        out = O::mul(out, c1);
        O::barrier(out);
        return O::mul(out, c2);
    }

    // a * 2^exponent with the exponent clamped to the normal range: faster than ldexp, but
    // over- and underflows early where 2^exponent itself is not a normal number.
    SCALARS_FORCE_INLINE static V ldexp_fast(V a, V exponent)
    {
        const V limit = cst(T(2 * kBias + 1));
        const V e     = float_to_int(O::min(O::max(O::add(exponent, cst(T(kBias))), zero()), limit));
        return O::mul(a, shl<kMant>(e));
    }

    //==========================================================================
    // Double-word arithmetic (GenericPacketMathDoubleWord.h)
    //==========================================================================

    // Veltkamp splitting: x = x_hi + x_lo exactly, with half the significand in x_hi.
    SCALARS_FORCE_INLINE static void veltkamp(V x, V& x_hi, V& x_lo)
    {
        constexpr int shift = (kDigits + 1) / 2;
        V gamma = O::mul(cst(T(U(1) << shift) + T(1)), x);
        O::barrier(gamma);
        V rho = O::sub(x, gamma);
        O::barrier(rho);
        x_hi = O::add(rho, gamma);
        O::barrier(x_hi);
        x_lo = O::sub(x, x_hi);
    }

    // x * y = p_hi + p_lo exactly, p_hi = fl(x * y).
    SCALARS_FORCE_INLINE static void twoprod(V x, V y, V& p_hi, V& p_lo)
    {
        p_hi = O::mul(x, y);
        if constexpr (O::has_fma)
        {
            p_lo = msub(x, y, p_hi);
        }
        else
        {
            O::barrier(p_hi);
            V x_hi, x_lo, y_hi, y_lo;
            veltkamp(x, x_hi, x_lo);
            veltkamp(y, y_hi, y_lo);
            V lo = O::sub(O::mul(x_hi, y_hi), p_hi);
            O::barrier(lo);
            lo = O::add(O::mul(x_hi, y_lo), lo);
            O::barrier(lo);
            lo = O::add(O::mul(x_lo, y_hi), lo);
            O::barrier(lo);
            p_lo = O::add(O::mul(x_lo, y_lo), lo);
        }
    }

    // The low part of x * y given xy = fl(x * y).
    SCALARS_FORCE_INLINE static V twoprod_low(V x, V y, V xy)
    {
        if constexpr (O::has_fma)
        {
            return msub(x, y, xy);
        }
        else
        {
            V hi, lo;
            twoprod(x, y, hi, lo);
            return O::add(O::sub(hi, xy), lo);
        }
    }

    // Knuth two-sum: x + y = s_hi + s_lo exactly, for any magnitudes.
    SCALARS_FORCE_INLINE static void twosum(V x, V y, V& s_hi, V& s_lo)
    {
        s_hi = O::add(x, y);
        O::barrier(s_hi);
        V y_part = O::sub(s_hi, x);
        O::barrier(y_part);
        V x_part = O::sub(s_hi, y_part);
        O::barrier(x_part);
        V x_err = O::sub(x, x_part);
        O::barrier(x_err);
        V y_err = O::sub(y, y_part);
        O::barrier(y_err);
        s_lo = O::add(x_err, y_err);
    }

    // x + y = s_hi + s_lo exactly when |x| >= |y|.
    SCALARS_FORCE_INLINE static void fast_twosum(V x, V y, V& s_hi, V& s_lo)
    {
        s_hi    = O::add(x, y);
        const V t = O::sub(s_hi, x);
        s_lo    = O::sub(y, t);
    }

    // {x_hi, x_lo} + {y_hi, y_lo}, assuming |x_hi| >= |y_hi|.
    SCALARS_FORCE_INLINE static void fast_twosum(V x_hi, V x_lo, V y_hi, V y_lo, V& s_hi, V& s_lo)
    {
        V r_hi, r_lo;
        fast_twosum(x_hi, y_hi, r_hi, r_lo);
        const V s = O::add(O::add(y_lo, r_lo), x_lo);
        fast_twosum(r_hi, s, s_hi, s_lo);
    }

    // x + {y_hi, y_lo}, assuming |x| >= |y_hi|.
    SCALARS_FORCE_INLINE static void fast_twosum(V x, V y_hi, V y_lo, V& s_hi, V& s_lo)
    {
        V r_hi, r_lo;
        fast_twosum(x, y_hi, r_hi, r_lo);
        const V s = O::add(y_lo, r_lo);
        fast_twosum(r_hi, s, s_hi, s_lo);
    }

    // {x_hi, x_lo} * y (Muller, Elementary Functions, Algorithm 7).
    SCALARS_FORCE_INLINE static void twoprod(V x_hi, V x_lo, V y, V& p_hi, V& p_lo)
    {
        V c_hi, c_lo1;
        twoprod(x_hi, y, c_hi, c_lo1);
        const V c_lo2 = O::mul(x_lo, y);
        V       t_hi, t_lo1;
        fast_twosum(c_hi, c_lo2, t_hi, t_lo1);
        const V t_lo2 = O::add(t_lo1, c_lo1);
        fast_twosum(t_hi, t_lo2, p_hi, p_lo);
    }

    // {x_hi, x_lo} * {y_hi, y_lo}.
    SCALARS_FORCE_INLINE static void twoprod(V x_hi, V x_lo, V y_hi, V y_lo, V& p_hi, V& p_lo)
    {
        V p_hi_hi, p_hi_lo;
        twoprod(x_hi, x_lo, y_hi, p_hi_hi, p_hi_lo);
        V p_lo_hi, p_lo_lo;
        twoprod(x_hi, x_lo, y_lo, p_lo_hi, p_lo_lo);
        fast_twosum(p_hi_hi, p_hi_lo, p_lo_hi, p_lo_lo, p_hi, p_lo);
    }

    // {x_hi, x_lo} / y (Joldes, Muller and Popescu 2017, Algorithm 15).
    SCALARS_FORCE_INLINE static void doubleword_div_fp(V x_hi, V x_lo, V y, V& z_hi, V& z_lo)
    {
        const V t_hi = O::div(x_hi, y);
        V       pi_hi, pi_lo;
        twoprod(t_hi, y, pi_hi, pi_lo);
        const V delta_hi = O::sub(x_hi, pi_hi);
        const V delta_t  = O::sub(delta_hi, pi_lo);
        const V delta    = O::add(delta_t, x_lo);
        const V t_lo     = O::div(delta, y);
        fast_twosum(t_hi, t_lo, z_hi, z_lo);
    }

    //==========================================================================
    // Logarithms
    //==========================================================================

    // Decomposes v > 0 as 2^e (1 + f) with f in [sqrt(1/2) - 1, sqrt(2) - 1] by integer
    // arithmetic on the bit pattern (musl-inspired). Returns whether any lane is not a positive,
    // normal, finite value; only then are subnormals normalized and special values handled.
    SCALARS_FORCE_INLINE static bool log_reduce(V v, V& f, V& e)
    {
        constexpr U kSqrtHalfBits = kFloat ? U(0x3f3504f3u) : U(0x3FE6A09E667F3BCDull);
        constexpr U kOneBits      = kFloat ? U(0x3f800000u) : U(0x3FF0000000000000ull);
        constexpr U kMantMask     = (U(1) << kMant) - U(1);

        // Below the smallest normal also catches zero, negative values and NaN.
        const V    is_denormal = O::nge(v, cst(std::numeric_limits<T>::min()));
        const bool any_special =
            O::any(O::bor(is_denormal, O::lt(cst(std::numeric_limits<T>::max()), v)));
        V vi = v;
        if (any_special)
        {
            vi = O::select(is_denormal, O::mul(v, cst(T(U(1) << kMant))), vi);
        }

        // Biasing the bit pattern puts the [sqrt(1/2), sqrt(2)) half-octave boundary on an
        // exponent boundary, so one shift extracts e.
        const V biased = O::iadd(vi, bits(kOneBits - kSqrtHalfBits));
        V       e_int  = O::isub(shr<kMant>(biased), bits(U(kBias)));
        if (any_special)
        {
            e_int = O::isub(e_int, O::band(is_denormal, bits(U(kMant))));
        }
        e = int_to_float(e_int);
        f = O::sub(O::iadd(O::band(biased, bits(kMantMask)), bits(kSqrtHalfBits)), one());
        return any_special;
    }

    // log(1 + f) for f in [sqrt(1/2) - 1, sqrt(2) - 1].
    SCALARS_FORCE_INLINE static V log_mantissa(V f)
    {
        if constexpr (kFloat)
        {
            // log(1 + f) = f + f^2 P(f), Sollya fpminimax, error < 2.04e-8.
            constexpr float c[] = {8.8758550584316254e-02f,
                                   -1.4199858903884888e-01f,
                                   1.4824025332927704e-01f,
                                   -1.6583317518234253e-01f,
                                   1.9972395896911621e-01f,
                                   -2.5001299381256104e-01f,
                                   3.3333668112754822e-01f,
                                   -4.9999997019767761e-01f};
            return madd(polevl(f, c), O::mul(f, f), f);
        }
        else
        {
            // Cephes: log(1 + x) = x - x^2 / 2 + x^3 P(x) / Q(x).
            const V x  = f;
            const V x2 = O::mul(x, x);
            const V x3 = O::mul(x2, x);
            V       y  = madd(cst(1.01875663804580931796E-4), x, cst(4.97494994976747001425E-1));
            V       y1 = madd(cst(1.44989225341610930846E1), x, cst(1.79368678507819816313E1));
            y          = madd(y, x, cst(4.70579119878881725854E0));
            y1         = madd(y1, x, cst(7.70838733755885391666E0));
            V num      = madd(y, x3, y1);

            y   = O::add(x, cst(1.12873587189167450590E1));
            y1  = madd(cst(8.29875266912776603211E1), x, cst(7.11544750618563894466E1));
            y   = madd(y, x, cst(4.52279145837532221105E1));
            y1  = madd(y1, x, cst(2.31251620126765340583E1));
            V q = madd(y, x3, y1);

            num = O::mul(num, x3);
            y   = O::div(num, q);
            y   = nmadd(cst(0.5), x2, y);
            return O::add(x, y);
        }
    }

    // log(x) and log2(x) at special lanes: x < 0 or NaN -> NaN, 0 -> -inf, +inf -> +inf.
    SCALARS_FORCE_INLINE static V log_special(V x, V r)
    {
        const V invalid = O::nge(x, zero());
        const V is_zero = O::eq(x, zero());
        const V pos_inf = O::eq(x, inf());
        return O::select(is_zero, negate(inf()), O::bor(O::select(pos_inf, inf(), r), invalid));
    }

    template <bool Base2> static V log_impl(V x)
    {
        V          f, e;
        const bool special = log_reduce(x, f, e);
        const V    lm      = log_mantissa(f);
        const V    r       = Base2 ? madd(lm, cst(kLog2e), e) : madd(e, cst(kLn2), lm);
        return special ? log_special(x, r) : r;
    }

    SCALARS_NOINLINE static V log(V x) { return log_impl<false>(x); }
    SCALARS_NOINLINE static V log2(V x) { return log_impl<true>(x); }

    SCALARS_NOINLINE static V log10(V x)
    {
        if constexpr (kFloat)
        {
            // log10(e) as hi + lo, so the product with log(x) does not round log10(e) first.
            const V log_x      = log(x);
            const V finite     = O::lt(abs(log_x), inf());
            const V finite_log = O::select(finite, log_x, zero());
            const V split = madd(finite_log, cst(0.4342944920063018f),
                                 O::mul(finite_log, cst(-1.0103049952192578e-08f)));
            return O::select(finite, split, log_x);
        }
        else
        {
            return O::mul(log(x), cst(0.43429448190325182));
        }
    }

    // log(1 + x): the rounding error of u = fl(1 + x) is recovered as dx = x - (u - 1) and folded
    // in as dx / u.
    SCALARS_NOINLINE static V log1p(V x)
    {
        V u = O::add(one(), x);
        O::barrier(u);
        V u_minus_one = O::sub(u, one());
        O::barrier(u_minus_one);
        const V dx    = O::sub(x, u_minus_one);
        const V small = O::eq(u, one());

        V          f, e;
        const bool special = log_reduce(u, f, e);
        const V    log_u   = log_mantissa(f);
        V          result  = madd(e, cst(kLn2), O::add(log_u, O::div(dx, u)));
        // On the kernel's own interval, log(1 + x) comes from x itself, without the rounding of
        // u and its first-order correction: 1 ulp instead of 1.6 for float near 0.
        const V in_kernel = O::band(O::le(cst(T(-0.29289321881345247560)), x),
                                    O::le(x, cst(T(0.41421356237309504880))));
        result = O::select(in_kernel, log_mantissa(x), result);
        result = O::select(small, x, result);
        if (!special)
        {
            return result;
        }
        result = O::select(O::eq(u, inf()), inf(), result);
        result = O::select(O::eq(x, cst(T(-1))), negate(inf()), result);
        // x < -1 or NaN.
        return O::bor(O::nge(u, zero()), result);
    }

    //==========================================================================
    // Exponentials
    //==========================================================================

    // exp(r) for |r| <= ln(2) / 2.
    SCALARS_FORCE_INLINE static V exp_reduced(V r)
    {
        if constexpr (kFloat)
        {
            const V r2     = O::mul(r, r);
            V       p_even = madd(r2, cst(1.37449637986719608306884765625e-3f),
                                  cst(4.166965186595916748046875e-2f));
            const V p_odd  = madd(r2, cst(8.36894474923610687255859375e-3f),
                                  cst(0.16666518151760101318359375f));
            p_even         = madd(r2, p_even, cst(0.49999988079071044921875f));
            const V p_low  = O::add(r, one());
            const V y      = madd(r, p_odd, p_even);
            return madd(r2, y, p_low);
        }
        else
        {
            // Cephes rational interpolant exp(g) = 1 + 2 g P(g^2) / (Q(g^2) - g P(g^2)).
            const V x2 = O::mul(r, r);
            V       px = cst(1.26177193074810590878e-4);
            px         = madd(px, x2, cst(3.02994407707441961300e-2));
            px         = madd(px, x2, cst(9.99999999999999999910e-1));
            px         = O::mul(px, r);
            V qx       = cst(3.00198505138664455042e-6);
            qx         = madd(qx, x2, cst(2.52448340349684104192e-3));
            qx         = madd(qx, x2, cst(2.27265548208155028766e-1));
            qx         = madd(qx, x2, cst(2.00000000000000000009e0));
            return madd(cst(2.0), O::div(px, O::sub(qx, px)), one());
        }
    }

    // expm1(r) for |r| <= ln(2) / 2, as r + r^2 P(r), fitted for the relative error of expm1.
    SCALARS_FORCE_INLINE static V expm1_reduced(V r)
    {
        const V r2 = O::mul(r, r);
        if constexpr (kFloat)
        {
            V p_even = madd(r2, cst(1.393371261656284332275390625e-3f),
                            cst(4.16664779186248779296875e-2f));
            V p_odd  = madd(r2, cst(1.97434448637068271636962890625e-4f),
                            cst(8.333566598594188690185546875e-3f));
            p_even   = madd(r2, p_even, cst(0.5f));
            p_odd    = madd(r2, p_odd, cst(0.1666666567325592041015625f));
            const V y = madd(r, p_odd, p_even);
            return madd(r2, y, r);
        }
        else
        {
            V p_even  = madd(r2, cst(2.0889530999828774e-09), cst(2.7557352849435147e-07));
            V p_odd   = madd(r2, cst(2.510920369520403e-08), cst(2.7557243734579987e-06));
            p_even    = madd(r2, p_even, cst(2.480158723753082e-05));
            p_odd     = madd(r2, p_odd, cst(0.00019841269886601229));
            p_even    = madd(r2, p_even, cst(0.0013888888888923816));
            p_odd     = madd(r2, p_odd, cst(0.008333333333321436));
            p_even    = madd(r2, p_even, cst(0.04166666666666661));
            p_odd     = madd(r2, p_odd, cst(0.16666666666666677));
            p_even    = madd(r2, p_even, cst(0.5));
            const V y = madd(r, p_odd, p_even);
            return madd(r2, y, r);
        }
    }

    // The two parts of ln(2) of the Cody-Waite reduction of exp.
    static constexpr T kLn2Hi = kFloat ? T(0.693359375f) : T(0.693145751953125);
    static constexpr T kLn2Lo = kFloat ? T(-2.12194440e-4f) : T(1.42860682030941723212e-6);

    SCALARS_NOINLINE static V exp(V x_in)
    {
        if constexpr (kFloat)
        {
            const V x = O::min(O::max(x_in, cst(-104.f)), cst(88.723f));
            const V m = O::rint(O::mul(x, cst(1.44269504088896341f)));
            V       r = nmadd(m, cst(kLn2Hi), x);
            O::barrier(r);  // -ffast-math would otherwise fuse the two products into m ln(2)
            r = nmadd(m, cst(kLn2Lo), r);
            V y = exp_reduced(r);

            // 2^m = 2^(m/2) 2^(m - m/2): each factor is a normal float for |m| <= 150, so the
            // last product rounds correctly into the subnormal range.
            const V sum = O::iadd(float_to_int(m), bits(U(254)));
            const V hi  = shr<1>(sum);
            const V lo  = O::isub(sum, hi);
            y           = O::mul(y, shl<23>(hi));
            O::barrier(y);
            y = O::mul(y, shl<23>(lo));
            // A NaN input propagates through the clamp; exp(x) > x otherwise.
            return O::max(y, x_in);
        }
        else
        {
            const V zero_mask = O::lt(x_in, cst(-745.519));
            V       x         = O::min(x_in, cst(709.784));
            const V fx        = O::rint(O::mul(x, cst(1.4426950408889634073599)));
            x                 = nmadd(fx, cst(kLn2Hi), x);
            O::barrier(x);
            x = nmadd(fx, cst(kLn2Lo), x);
            x = exp_reduced(x);
            if (!O::any(O::lt(cst(708.0), abs(x_in))))
            {
                return O::max(ldexp_fast(x, fx), x_in);
            }
            return O::bandnot(O::max(ldexp(x, fx), x_in), zero_mask);
        }
    }

    // expm1(x) = 2^n (expm1(r) + (1 - 2^-n)) with n = rint(x / ln(2)), r = x - n ln(2).
    SCALARS_NOINLINE static V expm1(V x_in)
    {
        const T max_cap = T(kMaxExp + 1) * kLn2;
        const T min_cap = -T(kDigits + 2) * kLn2;
        const V x       = O::max(O::min(x_in, cst(max_cap)), cst(min_cap));
        const V n       = O::rint(O::mul(x, cst(kLog2e)));
        V       r       = nmadd(n, cst(kLn2Hi), x);
        O::barrier(r);
        r            = nmadd(n, cst(kLn2Lo), r);
        const V em1  = expm1_reduced(r);
        V       c    = O::sub(one(), ldexp_fast(one(), negate(O::min(n, cst(T(kDigits + 2))))));
        O::barrier(c);
        V y = ldexp_fast(O::add(em1, c), O::sub(n, one()));
        O::barrier(y);
        y = O::mul(y, cst(T(2)));
        // Below eps / 4, x is the result to a quarter of an ulp; it keeps -0 and subnormals.
        y = O::select(O::lt(abs(x_in), cst(T(0.25) * kEps)), x_in, y);
        return O::max(y, x_in);
    }

    // exp(x) / 2 for x >= 0, finite up to ln(2 max): the term sinh and cosh share.
    static V half_exp(V x_in)
    {
        const V x = O::min(x_in, cst(T(kMaxExp + 2) * kLn2));
        const V n = O::rint(O::mul(x, cst(kLog2e)));
        V       r = nmadd(n, cst(kLn2Hi), x);
        O::barrier(r);
        r   = nmadd(n, cst(kLn2Lo), r);
        V y = ldexp_fast(exp_reduced(r), O::sub(n, cst(T(2))));
        O::barrier(y);
        y = O::mul(y, cst(T(2)));
        return O::max(y, x_in);
    }

    // 2^f for |f| <= 1/2.
    SCALARS_FORCE_INLINE static V exp2_reduced(V f)
    {
        const T ln2_hi = kLn2;
        const T ln2_lo = T((0.693147180559945309417232121458176568 - double(ln2_hi)) +
                           2.3190468138462996e-17);
        V       r_hi, r_lo;
        twoprod(cst(ln2_hi), f, r_hi, r_lo);
        r_lo      = madd(cst(ln2_lo), f, r_lo);
        const V e = exp_reduced(r_hi);
        return madd(e, r_lo, e);
    }

    // y * 2^n for y in [1/sqrt(2), sqrt(2)] and integral n up to max_exponent + 1 in magnitude.
    SCALARS_FORCE_INLINE static V exp2_scale(V y, V n)
    {
        if (!O::any(O::lt(cst(T(kMaxExp - 2)), abs(n))))
        {
            return ldexp_fast(y, n);
        }
        const V h   = O::rint(O::mul(n, cst(T(0.5))));
        V       out = ldexp_fast(y, h);
        O::barrier(out);
        return ldexp_fast(out, O::sub(n, h));
    }

    SCALARS_NOINLINE static V exp2(V x_in)
    {
        const V x = O::max(O::min(x_in, cst(T(kMaxExp + 1))), cst(-T(kMaxExp + kDigits - 1)));
        const V n = O::rint(x);
        return exp2_scale(exp2_reduced(O::sub(x, n)), n);
    }

    // 10^x = 2^(x log2(10)), with the product formed as a double word so that its rounding
    // error does not grow with |x|.
    SCALARS_NOINLINE static V exp10(V x_in)
    {
        constexpr double kLog2Of10 = 3.321928094887362347870319429489390175864831393;
        constexpr T      kL_hi     = kFloat ? T(3.321928024291992) : T(3.321928094887362);
        constexpr T      kL_lo     = kFloat ? T(7.059536955011936e-08) : T(1.661617516973592e-16);
        const V          x         = O::max(O::min(x_in, cst(T((kMaxExp + 1) / kLog2Of10))),
                           cst(T(-(kMaxExp + kDigits - 1) / kLog2Of10)));
        V                t_hi, t_lo;
        twoprod(x, cst(kL_hi), t_hi, t_lo);
        t_lo      = madd(x, cst(kL_lo), t_lo);
        const V n = O::rint(t_hi);
        return exp2_scale(exp2_reduced(O::add(O::sub(t_hi, n), t_lo)), n);
    }

    //==========================================================================
    // Trigonometric functions
    //==========================================================================

    // sin and cos of x from one argument reduction.
    SCALARS_NOINLINE static void sincos(V x_in, V& sin_out, V& cos_out)
    {
        if constexpr (kFloat)
        {
            sincos_float(x_in, sin_out, cos_out);
        }
        else
        {
            sincos_double(x_in, sin_out, cos_out);
        }
    }

    SCALARS_FORCE_INLINE static V sin(V x)
    {
        V s, c;
        sincos(x, s, c);
        return s;
    }

    SCALARS_FORCE_INLINE static V cos(V x)
    {
        V s, c;
        sincos(x, s, c);
        return c;
    }

    SCALARS_FORCE_INLINE static V tan(V x)
    {
        V s, c;
        sincos(x, s, c);
        return O::div(s, c);
    }

    // Selects the sine and cosine polynomials by quadrant and applies their signs. q holds the
    // quadrant as an integer bit pattern; only its two lowest bits are used.
    SCALARS_FORCE_INLINE static void trig_finish(
        V x_in, V q, V poly_sin, V poly_cos, V& sin_out, V& cos_out)
    {
        const V even     = int_is_even(q);
        const V sign_sin = sign_of(O::bxor(x_in, shl<kBits - 2>(q)));
        const V sign_cos = sign_of(shl<kBits - 2>(O::iadd(q, bits(U(1)))));
        sin_out          = O::bxor(O::select(even, poly_sin, poly_cos), sign_sin);
        cos_out          = O::bxor(O::select(even, poly_cos, poly_sin), sign_cos);
    }

    static void sincos_float(V x_in, V& sin_out, V& cos_out)
    {
        const V x_abs = abs(x_in);
        V       x     = x_abs;
        // Nearest integer to x * 2/pi by the rounding trick; its low bits hold the quadrant.
        const V magic   = cst(12582912.f);
        V       y_round = O::add(O::mul(x, cst(0.636619746685028076171875f)), magic);
        O::barrier(y_round);
        V       q = y_round;
        const V y = O::sub(y_round, magic);

        // Cody-Waite reduction to [-pi/4, pi/4]: 1 ulp below huge_th.
        float huge_th = 0;
        if constexpr (O::has_fma)
        {
            huge_th = 71476.0625f;
            x       = madd(y, cst(-1.57079601287841796875f), x);
            x       = madd(y, cst(-3.1391647326017846353352069854736328125e-07f), x);
            x       = madd(y, cst(-5.390302529957764765544681040410068817436695098876953125e-15f), x);
        }
        else
        {
            huge_th = 18838.f;
            x       = madd(y, cst(-1.5703125f), x);
            O::barrier(x);
            x = madd(y, cst(-0.000483989715576171875f), x);
            O::barrier(x);
            x = madd(y, cst(1.62865035235881805419921875e-07f), x);
            O::barrier(x);
            x = madd(y, cst(5.5644315544167710640977020375430583953857421875e-11f), x);
        }

        if (O::any(O::le(cst(huge_th), x_abs)))
        {
            // Payne-Hanek for the large lanes, one at a time.
            float        vals[kSize];
            float        xs[kSize];
            float        q_lanes[kSize];
            std::int32_t quadrants[kSize];
            O::storeu(vals, x_abs);
            O::storeu(xs, x);
            O::storeu(q_lanes, q);
            std::memcpy(quadrants, q_lanes, sizeof(quadrants));
            for (int k = 0; k < kSize; ++k)
            {
                if (vals[k] >= huge_th && std::isfinite(vals[k]))
                {
                    xs[k] = trig_reduce_huge(vals[k], &quadrants[k]);
                }
            }
            std::memcpy(q_lanes, quadrants, sizeof(quadrants));
            x = O::loadu(xs);
            q = O::loadu(q_lanes);
        }

        const V x2 = O::mul(x, x);
        // cos(x) on [-pi/4, pi/4].
        V pc = cst(2.4372266125283204019069671630859375e-05f);
        pc   = madd(pc, x2, cst(-0.00138865201734006404876708984375f));
        pc   = madd(pc, x2, cst(0.041666619479656219482421875f));
        pc   = madd(pc, x2, cst(-0.5f));
        pc   = madd(pc, x2, one());
        // sin(x) on [-pi/4, pi/4].
        V ps = cst(-0.0001959234114083702898469196984621021329076029360294342041015625f);
        ps   = madd(ps, x2, cst(0.0083326873655616851693794799871284340042620897293090820312500000f));
        ps   = madd(ps, x2, cst(-0.1666666203982298255503735617821803316473960876464843750000000000f));
        ps   = O::mul(ps, x2);
        ps   = madd(ps, x, x);

        trig_finish(x_in, q, ps, pc, sin_out, cos_out);
    }

    // x - q pi/2 for |q pi/2 - x| <= pi/4, with pi/2 split into three doubles.
    SCALARS_FORCE_INLINE static V trig_reduce_small_double(V x, V q)
    {
        const V pio2_1 = cst(-1.5707963267948965579989817342720925807952880859375);
        const V pio2_2 = cst(-6.12323399573676603586882014729198302312846062338790e-17);
        const V pio2_3 = cst(1.4973849048591698329435081771059920083527504761695190e-33);
        if constexpr (O::has_fma)
        {
            V t = madd(pio2_1, q, x);
            t   = madd(pio2_2, q, t);
            return madd(pio2_3, q, t);
        }
        else
        {
            // Without FMA the products must be formed exactly before they cancel against x.
            V qp_hi, qp_lo;
            twoprod(pio2_1, q, qp_hi, qp_lo);
            V t_hi, t_lo;
            twosum(x, qp_hi, t_hi, t_lo);
            V qp2_hi, qp2_lo;
            twoprod(pio2_2, q, qp2_hi, qp2_lo);
            V a_hi, a_lo, b_lo;
            twosum(t_hi, qp_lo, a_hi, a_lo);
            twosum(a_hi, qp2_hi, t_hi, b_lo);
            t_lo = O::add(t_lo, O::add(a_lo, b_lo));
            t_lo = O::add(t_lo, qp2_lo);
            t_lo = madd(pio2_3, q, t_lo);
            return O::add(t_hi, t_lo);
        }
    }

    static void sincos_double(V x_in, V& sin_out, V& cos_out)
    {
        const double small_th = 15;
        const double huge_th  = 1e14;  // beyond it, the scalar libm
        const V      x_abs    = abs(x_in);
        V            q_int;
        V            s;

        if (O::any(O::le(cst(small_th), x_abs)))
        {
            // Double-word x * 2/pi for the quadrant.
            V prod_hi, prod_lo;
            twoprod(x_abs, cst(0.63661977236758138243288840385503135621547698974609375), prod_hi, prod_lo);
            prod_lo = madd(x_abs, cst(-3.9357353350364971763790381828183628368294820823718866e-17), prod_lo);
            V       q    = O::rint(prod_hi);
            const V frac = O::add(O::sub(prod_hi, q), prod_lo);
            q            = O::add(q, O::band(O::lt(cst(0.5), frac), one()));
            q            = O::add(q, O::band(O::lt(frac, cst(-0.5)), cst(-1.0)));
            q_int        = float_to_int(q);
            s            = trig_reduce_small_double(x_abs, q);
        }
        else
        {
            const V bias     = cst(6755399441055744.0);  // 1.5 * 2^52
            V       biased_q = O::add(O::mul(x_abs, cst(0.63661977236758138243288840385503135621547698974609375)), bias);
            O::barrier(biased_q);
            q_int     = biased_q;
            const V q = O::sub(biased_q, bias);
            if constexpr (O::has_fma)
            {
                s = trig_reduce_small_double(x_abs, q);
            }
            else
            {
                // |x| < 15 implies q <= 10; the first two 49-bit parts give exact products.
                s = madd(q, cst(-1.5707963267948948), x_abs);
                O::barrier(s);
                s = madd(q, cst(-1.837589179357618e-15), s);
                O::barrier(s);
                s = madd(q, cst(-8.4784276603688996e-32), s);
            }
        }

        const V ss = O::mul(s, s);
        // cos(x) = 1 + u P(u), u = x^2, on [-pi/4, pi/4].
        V scos = cst(-1.1368926065317776472832699312119132152576472805094454088248312473297119140625e-11);
        scos   = madd(scos, ss, cst(2.0875905481768720039634091158002593413556269297259859740734100341796875e-09));
        scos   = madd(scos, ss, cst(-2.7557315712466412785356544880299711763882442028261721134185791015625e-07));
        scos   = madd(scos, ss, cst(2.480158729424286522739599714082459058772656135261058807373046875e-05));
        scos   = madd(scos, ss, cst(-1.388888888888178789471350427220386336557567119598388671875e-03));
        scos   = madd(scos, ss, cst(4.166666666666664353702032030923874117434024810791015625e-02));
        scos   = madd(scos, ss, cst(-0.5));
        scos   = madd(scos, ss, one());
        // sin(x) = x (1 + u R(u)) on [-pi/4, pi/4].
        V ssin = cst(1.59193066075142890698150587293845624470289834562208852730691432952880859375e-10);
        ssin   = madd(ssin, ss, cst(-2.50511517945670206974594627392927126408039839589037001132965087890625e-08));
        ssin   = madd(ssin, ss, cst(2.755731622544328228235042954619160582296899519860744476318359375e-06));
        ssin   = madd(ssin, ss, cst(-1.9841269837089632013978068858506276228581555187702178955078125e-04));
        ssin   = madd(ssin, ss, cst(8.333333333331312264835588621281203813850879669189453125e-03));
        ssin   = madd(ssin, ss, cst(-0.1666666666666666574148081281236954964697360992431640625));
        ssin   = O::mul(ssin, ss);
        ssin   = madd(ssin, s, s);

        trig_finish(x_in, q_int, ssin, scos, sin_out, cos_out);

        if (O::any(O::le(cst(huge_th), x_abs)))
        {
            double xs[kSize];
            double sv[kSize];
            double cv[kSize];
            O::storeu(xs, x_in);
            O::storeu(sv, sin_out);
            O::storeu(cv, cos_out);
            for (int k = 0; k < kSize; ++k)
            {
                if (std::fabs(xs[k]) >= huge_th && std::isfinite(xs[k]))
                {
                    sv[k] = std::sin(xs[k]);
                    cv[k] = std::cos(xs[k]);
                }
            }
            sin_out = O::loadu(sv);
            cos_out = O::loadu(cv);
        }
    }

    //==========================================================================
    // Inverse trigonometric functions
    //==========================================================================

    // atan(x) for x in [0, 1]. The double version also subtracts the low word of pi/2 in the
    // lanes of large_mask, which the caller subtracts from pi/2.
    SCALARS_FORCE_INLINE static V atan_reduced(V x, V large_mask)
    {
        const V x2 = O::mul(x, x);
        if constexpr (kFloat)
        {
            (void)large_mask;
            constexpr float alpha[] = {1.12026982009410858154296875e-01f,
                                       7.296695709228515625e-01f,
                                       8.109951019287109375e-01f};
            constexpr float beta[]  = {1.00917108356952667236328125e-02f,
                                       2.8318560123443603515625e-01f,
                                       1.0f,
                                       8.109951019287109375e-01f};
            return O::mul(x, O::div(polevl(x2, alpha), polevl(x2, beta)));
        }
        else
        {
            // atan(x) = x + x u N(u) / Q(u), u = x^2 (Rminimax).
            constexpr double kPiOverTwoLo = 6.123233995736766e-17;
            constexpr double alpha[]      = {-2.4644487075790443e-04,
                                             -7.8073996177164305e-03,
                                             -6.2915029683194929e-02,
                                             -1.9307140255021149e-01,
                                             -2.4634297465012978e-01,
                                             -1.1001453763093239e-01};
            constexpr double beta[]       = {2.7311202462436667e-04,
                                             1.0899150928962708e-02,
                                             1.1548932646420353e-01,
                                             4.9716458728465573e-01,
                                             1.0,
                                             9.3705509168587852e-01,
                                             3.3004361289279920e-01};
            const V          p            = polevl(x2, alpha);
            const V          q            = polevl(x2, beta);
            const V          x0           = O::sub(x, O::band(large_mask, cst(kPiOverTwoLo)));
            return madd(O::mul(x, x2), O::div(p, q), x0);
        }
    }

    SCALARS_NOINLINE static V atan(V x_in)
    {
        const V abs_x  = abs(x_in);
        const V large  = O::lt(one(), abs_x);
        const V x      = O::select(large, reciprocal(abs_x), abs_x);
        const V p      = atan_reduced(x, large);
        const V result = O::select(large, O::sub(cst(kPiOver2), p), p);
        return O::bxor(result, sign_of(x_in));
    }

    SCALARS_NOINLINE static V atan2(V y, V x)
    {
        const V ay   = abs(y);
        const V ax   = abs(x);
        const V swap = O::lt(ax, ay);
        const V num  = O::select(swap, ax, ay);
        const V den  = O::select(swap, ay, ax);
        V       t    = O::div(num, den);
        t            = O::select(O::eq(den, zero()), zero(), t);
        t            = O::select(O::band(isinf(x), isinf(y)), one(), t);
        V r          = atan_reduced(t, swap);
        r            = O::select(swap, O::sub(cst(kPiOver2), r), r);
        const T pi_lo = kFloat ? T(-8.742278e-08f) : T(1.2246467991473532e-16);
        r = O::select(signbit(x), O::add(O::sub(cst(kPi), r), cst(pi_lo)), r);
        r = O::bxor(r, sign_of(y));
        return O::bor(O::bor(O::isnan(x), O::isnan(y)), r);
    }

    // asin(z) = z + z u P(u) for u = z^2 <= 1/4, to a relative error of 2^-27.6 (Sollya).
    SCALARS_FORCE_INLINE static V asin_kernel_float(V z)
    {
        const V u      = O::mul(z, z);
        const V u2     = O::mul(u, u);
        const V p_low  = madd(cst(7.4953101575374603271484375e-2f), u, cst(0.166667520999908447265625f));
        const V p_high = madd(cst(4.21570129692554473876953125e-2f), u2,
                              madd(cst(2.418550290167331695556640625e-2f), u,
                                   cst(4.54690195620059967041015625e-2f)));
        return madd(O::mul(z, u), madd(p_high, u2, p_low), z);
    }

    SCALARS_NOINLINE static V asin(V x_in)
    {
        if constexpr (kFloat)
        {
            const V half  = cst(0.5f);
            const V abs_x = abs(x_in);
            const V large = O::lt(half, abs_x);
            //   |x| <= 1/2: asin(|x|) = asin(z), z = |x|
            //   |x| >  1/2: asin(|x|) = pi/2 - 2 asin(z), z = sqrt((1 - |x|) / 2), NaN for |x| > 1
            const V z = O::select(large, O::sqrt(nmadd(half, abs_x, half)), abs_x);
            V       p = asin_kernel_float(z);
            p         = O::select(large, nmadd(cst(2.0f), p, cst(kPiOver2)), p);
            return O::bxor(p, sign_of(x_in));
        }
        else
        {
            // asin(x) = atan(x / sqrt(1 - x^2)), with 1 - x^2 = (1 - |x|)(1 + |x|) exact in its
            // first factor where cancellation matters. NaN for |x| > 1.
            const V abs_x = abs(x_in);
            const V w     = O::sqrt(O::mul(O::sub(one(), abs_x), O::add(one(), abs_x)));
            return O::bxor(atan(O::div(abs_x, w)), sign_of(x_in));
        }
    }

    SCALARS_NOINLINE static V acos(V x_in)
    {
        if constexpr (kFloat)
        {
            // From the asin kernel above, whose polynomial is more accurate than Eigen's
            // pacos_float (4 ulp near 0.6): with z = sqrt((1 - |x|) / 2),
            //   |x| <= 1/2: acos(x) = pi/2 - asin(x)
            //   x   >  1/2: acos(x) = 2 asin(z)
            //   x   < -1/2: acos(x) = pi - 2 asin(z)
            const V half  = cst(0.5f);
            const V abs_x = abs(x_in);
            const V large = O::lt(half, abs_x);
            const V z     = O::select(large, O::sqrt(nmadd(half, abs_x, half)), abs_x);
            const V p     = O::bxor(asin_kernel_float(z), O::bandnot(sign_of(x_in), large));
            const V small_result = O::sub(cst(kPiOver2), p);
            const V twice        = O::add(p, p);
            const V large_result = O::select(signbit(x_in), O::sub(cst(kPi), twice), twice);
            // Outside [-1, 1] z or the kernel is NaN.
            return O::select(large, large_result, small_result);
        }
        else
        {
            // acos(x) = 2 atan(sqrt((1 - x) / (1 + x))): 0 at 1, pi at -1, NaN for |x| > 1.
            const V r    = O::sqrt(O::div(O::sub(one(), x_in), O::add(one(), x_in)));
            const V half = atan(r);
            return O::add(half, half);
        }
    }

    //==========================================================================
    // Hyperbolic functions
    //==========================================================================

    SCALARS_NOINLINE static V sinh(V x)
    {
        const V abs_x  = abs(x);
        const V x_sign = sign_of(x);
        const V x2     = O::mul(x, x);
        V       p_small;
        if constexpr (kFloat)
        {
            constexpr float alpha[] = {2.7557314045e-06f, 1.9841270114e-04f, 8.3333335817e-03f, 1.6666666716e-01f};
            p_small                 = polevl(x2, alpha);
        }
        else
        {
            constexpr double alpha[] = {8.2206352466243297e-18,
                                        2.8114572543455206e-15,
                                        7.6471637318198164e-13,
                                        1.6059043836821613e-10,
                                        2.5052108385441718e-08,
                                        2.7557319223985893e-06,
                                        1.9841269841269841e-04,
                                        8.3333333333333332e-03,
                                        1.6666666666666666e-01};
            p_small                  = polevl(x2, alpha);
        }
        p_small = madd(O::mul(x2, x), p_small, x);

        // e = exp(|x|) / 2; sinh = e - 1 / (4 e) up to |x| = 20, e beyond.
        const V e        = half_exp(abs_x);
        const V p_medium = O::sub(e, O::div(cst(T(0.25)), e));
        const V large    = O::lt(cst(T(20)), abs_x);
        const V p_large  = O::bor(x_sign, O::select(large, e, p_medium));
        return O::select(O::lt(abs_x, one()), p_small, p_large);
    }

    SCALARS_NOINLINE static V cosh(V x)
    {
        const V abs_x    = abs(x);
        const V e        = half_exp(abs_x);
        const V p_medium = O::add(e, O::div(cst(T(0.25)), e));
        return O::select(O::lt(cst(T(20)), abs_x), e, p_medium);
    }

    SCALARS_NOINLINE static V tanh(V x)
    {
        const V x_sign = sign_of(x);
        const V abs_x  = abs(x);
        if constexpr (kFloat)
        {
            // tanh(x) ~= x^3 P(x^2) / Q(x^2) + x on [-1.25, 1.25] (Rminimax), and
            // 1 - 2 / (1 + exp(2 |x|)) beyond. Within 1 ulp.
            constexpr float alpha[] = {-1.46725140511989593505859375e-02f, -3.333333432674407958984375e-01f};
            constexpr float beta[]  = {1.570280082523822784423828125e-02f, 4.4401752948760986328125e-01f, 1.0f};
            const V         large   = O::lt(cst(1.25f), abs_x);
            // Large lanes enter the polynomial as 0, so that x^2 cannot overflow.
            const V x_small   = O::bandnot(x, large);
            const V x2        = O::mul(x_small, x_small);
            const V x3        = O::mul(x2, x_small);
            const V small_tanh =
                O::bor(x_sign, madd(x3, O::div(polevl(x2, alpha), polevl(x2, beta)), x_small));
            if (!O::any(large))
            {
                return small_tanh;
            }
            // |x| clamped to 16, where tanh rounds to 1. NaN lanes enter exp as 0.
            const V s = exp(O::mul(cst(2.0f), O::min(O::select(large, abs_x, zero()), cst(16.0f))));
            const V large_tanh = O::bor(x_sign, O::sub(one(), O::div(cst(2.0f), O::add(s, one()))));
            return O::select(large, large_tanh, small_tanh);
        }
        else
        {
            // tanh(|x|) = e / (e + 2) with e = expm1(2 |x|): expm1 keeps the relative accuracy for
            // small |x|, and |x| clamped to 20 (tanh rounds to 1 past 19.1) keeps e finite.
            // Eigen's double rational approximation reaches 8-10 ulp near |x| = 15.
            const V e = expm1(O::mul(cst(2.0), O::min(abs_x, cst(20.0))));
            return O::bor(x_sign, O::div(e, O::add(e, cst(2.0))));
        }
    }

    SCALARS_NOINLINE static V atanh(V x)
    {
        const V x2 = O::mul(x, x);
        const V x3 = O::mul(x, x2);
        V       y_small;
        if constexpr (kFloat)
        {
            constexpr float alpha[] = {0.1819281280040740966796875f,
                                       8.2311116158962249755859375e-2f,
                                       0.14672131836414337158203125f,
                                       0.1997792422771453857421875f,
                                       0.3333373963832855224609375f};
            y_small                 = madd(x3, polevl(x2, alpha), x);
        }
        else
        {
            constexpr double alpha[] = {3.3071338469301391e-03,
                                        -4.7129526768798737e-02,
                                        1.8185306179826699e-01,
                                        -2.5949536095445679e-01,
                                        1.2306328729812676e-01};
            constexpr double beta[]  = {-3.8679974580640881e-03,
                                        7.6391885763341910e-02,
                                        -4.2828141436397615e-01,
                                        9.8733495886883648e-01,
                                        -1.0000000000000000e+00,
                                        3.6918986189438030e-01};
            y_small                  = madd(x3, O::div(polevl(x2, alpha), polevl(x2, beta)), x);
        }

        const V half       = cst(T(0.5));
        const V x_gt_half  = O::le(half, abs(x));
        if (!O::any(x_gt_half))
        {
            return y_small;
        }
        // atanh(x) = log((1 + x) / (1 - x)) / 2 for |x| in (0.5, 1].
        const V y_large  = O::mul(half, log(O::div(O::add(one(), x), O::sub(one(), x))));
        const V x_eq_one = O::eq(one(), abs(x));
        const V x_gt_one = O::lt(one(), abs(x));
        return O::bor(x_gt_one,
                      O::select(x_eq_one, O::bor(sign_of(x), inf()), O::select(x_gt_half, y_large, y_small)));
    }

    SCALARS_NOINLINE static V asinh(V x)
    {
        const V abs_x  = abs(x);
        const V x_sign = sign_of(x);
        // log(2 |x|) = log1p(|x| - 1) + ln(2) where x^2 would overflow.
        const V large      = O::lt(cst(kFloat ? T(1e10f) : T(1e150)), abs_x);
        const V x2         = O::mul(abs_x, O::select(large, zero(), abs_x));
        const V normal_arg = O::add(abs_x, O::div(x2, O::add(one(), O::sqrt(O::add(one(), x2)))));
        const V large_arg  = O::sub(abs_x, one());
        V       result     = log1p(O::select(large, large_arg, normal_arg));
        result             = O::select(large, O::add(result, cst(kLn2)), result);
        // asinh(x) = x - x^3 / 6 + ...: the result rounds to x below this.
        const T tiny = kFloat ? T(2.44140625e-4f) : T(1.4901161193847656e-8);
        result       = O::select(O::lt(abs_x, cst(tiny)), abs_x, result);
        return O::bor(x_sign, result);
    }

    SCALARS_NOINLINE static V acosh(V x)
    {
        const V t     = O::sub(x, one());
        const V huge  = O::lt(cst(kFloat ? T(1e10f) : T(1e150)), x);
        const V t_tp2 = O::mul(O::select(huge, zero(), t), O::add(t, cst(T(2))));
        V       result = log1p(O::select(huge, t, O::add(t, O::sqrt(t_tp2))));
        result         = O::select(huge, O::add(result, cst(kLn2)), result);
        return O::bor(O::lt(x, one()), result);
    }

    //==========================================================================
    // Roots and powers
    //==========================================================================

    // One step of Halley's iteration for x = y^(1/3).
    SCALARS_FORCE_INLINE static V cbrt_halley(V x_k, V y)
    {
        const V x_k_cb = O::mul(x_k, O::mul(x_k, x_k));
        const V denom  = madd(cst(T(2)), x_k_cb, y);
        const V num    = O::sub(x_k_cb, y);
        return nmadd(x_k, O::div(num, denom), x_k);
    }

    // The last Halley step, which computes x_k^3 - y exactly where fma is fused.
    SCALARS_FORCE_INLINE static V cbrt_last_halley(V x_k, V y)
    {
        if constexpr (O::has_fma)
        {
            const V x_k_sq    = O::mul(x_k, x_k);
            const V x_k_sq_lo = msub(x_k, x_k, x_k_sq);
            const V x_k_cb    = O::mul(x_k_sq, x_k);
            const V denom     = madd(cst(T(2)), x_k_cb, y);
            const V num       = madd(x_k_sq_lo, x_k, msub(x_k_sq, x_k, y));
            return nmadd(x_k, O::div(num, denom), x_k);
        }
        else
        {
            return cbrt_halley(x_k, y);
        }
    }

    SCALARS_NOINLINE static V cbrt(V x)
    {
        // x^(1/3) = 2^e_div3 (s 2^e_mod3)^(1/3), x = s 2^e, e_div3 = ceil(e / 3).
        V       e;
        const V s      = frexp(abs(x), e);
        const V e_div3 = O::ceil(O::mul(e, cst(T(1) / T(3))));
        const V e_mod3 = nmadd(cst(T(3)), e_div3, e);
        const V y      = ldexp_fast(s, e_mod3);  // in [0.125, 1]

        V r;
        if constexpr (kFloat)
        {
            constexpr float alpha[] = {-0.884942114353179931640625f,
                                       2.4534566402435302734375f,
                                       -2.6808016300201416015625f,
                                       1.7967746257781982421875f,
                                       0.31364738941192626953125f};
            r                       = cbrt_last_halley(polevl(y, alpha), y);
        }
        else
        {
            constexpr double alpha[] = {0.59218535776880332210936330739059485495090484619140625,
                                        -1.3859252584188996859637654779362492263317108154296875,
                                        1.4581131468387165117661652402603067457675933837890625,
                                        0.340841899750747245878557123432983644306659698486328125};
            r                        = cbrt_last_halley(cbrt_halley(polevl(y, alpha), y), y);
        }
        r = ldexp_fast(r, e_div3);

        // Sign, and zero, inf and NaN pass through.
        const V root  = O::bor(sign_of(x), r);
        const V use_x = O::bor(O::bor(isinf(x), O::isnan(x)), O::eq(zero(), x));
        return O::select(use_x, x, root);
    }

    // log2(x) for x in [1/sqrt(2), sqrt(2)] as a double word, with the extra accuracy that
    // pow needs against the error magnification of y log2(x).
    SCALARS_FORCE_INLINE static void accurate_log2(V z, V& log2_x_hi, V& log2_x_lo)
    {
        if constexpr (kFloat)
        {
            constexpr double kC0    = 1.442695041742110273474963832995854318141937255859375e+00;
            constexpr float  kC0_hi = static_cast<float>(kC0);
            constexpr float  kC0_lo = static_cast<float>(kC0 - static_cast<double>(kC0_hi));
            constexpr double kC1    = -7.2134751588268664068692714863573201000690460205078125e-01;
            constexpr float  kC1_hi = static_cast<float>(kC1);
            constexpr float  kC1_lo = static_cast<float>(kC1 - static_cast<double>(kC1_hi));
            constexpr float  c[]    = {9.7010828554630279541015625e-02f,
                                       -1.6896486282348632812500000e-01f,
                                       1.7200836539268493652343750e-01f,
                                       -1.7892272770404815673828125e-01f,
                                       2.0505344867706298828125000e-01f,
                                       -2.4046677350997924804687500e-01f,
                                       2.8857553005218505859375000e-01f,
                                       -3.6067414283752441406250000e-01f,
                                       4.8089790344238281250000000e-01f};
            const V          x      = O::sub(z, one());
            const V          p      = polevl(x, c);
            V                p_hi, p_lo;
            twoprod(x, p, p_hi, p_lo);
            fast_twosum(cst(kC1_hi), cst(kC1_lo), p_hi, p_lo, p_hi, p_lo);
            twoprod(p_hi, p_lo, x, p_hi, p_lo);
            fast_twosum(cst(kC0_hi), cst(kC0_lo), p_hi, p_lo, p_hi, p_lo);
            twoprod(p_hi, p_lo, x, log2_x_hi, log2_x_lo);
        }
        else
        {
            // log2(x) = f(r), r = c (x - 1) / (x + 1), f odd: ((Q(r^2) r^2 + C) r^2 + 1) r.
            const V q12  = cst(2.87074255468000586e-9);
            const V q10  = cst(2.38957980901884082e-8);
            const V q8   = cst(2.31032094540014656e-7);
            const V q6   = cst(2.27279857398537278e-6);
            const V q4   = cst(2.31271023278625638e-5);
            const V q2   = cst(2.47556738444535513e-4);
            const V q0   = cst(2.88543873228900172e-3);
            const V C_hi = cst(0.0400377511598501157);
            const V C_lo = cst(-4.77726582251425391e-19);
            const V x    = z;

            V t_hi, t_lo;
            twoprod(cst(2.88539008177792677), cst(4.07660016854549667e-17), O::sub(x, one()), t_hi, t_lo);
            V s_hi, s_lo;
            fast_twosum(one(), x, s_hi, s_lo);
            V q_hi, q_lo;
            doubleword_div_fp(t_hi, t_lo, s_hi, q_hi, q_lo);
            V r_hi, r_lo;
            fast_twosum(q_hi, nmadd(q_hi, O::div(s_lo, s_hi), q_lo), r_hi, r_lo);

            V r2_hi, r2_lo;
            twoprod(r_hi, r_lo, r_hi, r_lo, r2_hi, r2_lo);
            V r4_hi, r4_lo;
            twoprod(r2_hi, r2_lo, r2_hi, r2_lo, r4_hi, r4_lo);

            V       q_even = madd(q12, r4_hi, q8);
            V       q_odd  = madd(q10, r4_hi, q6);
            q_even         = madd(q_even, r4_hi, q4);
            q_odd          = madd(q_odd, r4_hi, q2);
            q_even         = madd(q_even, r4_hi, q0);
            const V q      = madd(q_odd, r2_hi, q_even);

            V p_hi, p_lo;
            twoprod(r2_hi, r2_lo, q, p_hi, p_lo);
            V p1_hi, p1_lo;
            fast_twosum(C_hi, C_lo, p_hi, p_lo, p1_hi, p1_lo);
            V p2_hi, p2_lo;
            twoprod(r2_hi, r2_lo, p1_hi, p1_lo, p2_hi, p2_lo);
            V p3_hi, p3_lo;
            fast_twosum(one(), p2_hi, p2_lo, p3_hi, p3_lo);
            twoprod(p3_hi, p3_lo, r_hi, r_lo, log2_x_hi, log2_x_lo);
        }
    }

    // pow(x, y) for x > 0: exp2(y log2(x)) with y log2(x) in double-word arithmetic.
    static V pow_impl(V x, V y)
    {
        V       e_x;
        V       m_x   = frexp(x, e_x);
        const V scale = O::lt(m_x, cst(T(0.70710678118654752440)));
        m_x           = O::select(scale, O::mul(cst(T(2)), m_x), m_x);
        e_x           = O::select(scale, O::sub(e_x, one()), e_x);

        V rx_hi, rx_lo;
        accurate_log2(m_x, rx_hi, rx_lo);

        V f1_hi, f1_lo, f2_hi, f2_lo;
        twoprod(e_x, y, f1_hi, f1_lo);
        twoprod(rx_hi, rx_lo, y, f2_hi, f2_lo);
        // |e_x| > |log2(m_x)| unless e_x = 0, where the sum is exact anyway.
        V f_hi, f_lo;
        fast_twosum(f1_hi, f1_lo, f2_hi, f2_lo, f_hi, f_lo);

        V       n_z = O::rint(f_hi);
        V       r_z = O::add(O::sub(f_hi, n_z), f_lo);
        const V n_r = O::rint(r_z);
        r_z         = O::sub(r_z, n_r);
        n_z         = O::add(n_z, n_r);

        const V e_r = exp2_reduced(r_z);
        if (O::any(O::lt(cst(T(kMaxExp - 2)), abs(n_z))))
        {
            return ldexp(e_r, n_z);
        }
        return ldexp_fast(e_r, n_z);
    }

    // pow(x, y) with the special cases of std::pow.
    SCALARS_NOINLINE static V pow(V x, V y)
    {
        const V x_abs  = abs(x);
        V       result = pow_impl(x_abs, y);

        const V x_is_negative = O::lt(x, zero());
        const V x_is_zero     = O::eq(x, zero());
        const V x_is_one      = O::eq(x, one());
        const V x_has_signbit = signbit(x);
        const V x_abs_gt_one  = O::lt(one(), x_abs);
        const V x_abs_is_inf  = O::eq(x_abs, inf());

        const V y_abs        = abs(y);
        const V y_abs_is_inf = O::eq(y_abs, inf());
        const V y_is_negative = O::lt(y, zero());
        const V y_is_zero    = O::eq(y, zero());
        const V y_is_one     = O::eq(y, one());
        const V y_is_int     = O::bandnot(O::eq(O::floor(y), y), y_abs_is_inf);
        const V y_div_2      = O::mul(y, cst(T(0.5)));
        const V y_is_even    = O::eq(O::rint(y_div_2), y_div_2);
        const V y_is_odd_int = O::bandnot(y_is_int, y_is_even);
        // From this |y| on, x^y is 0 or inf for every |x| != 1.
        constexpr T huge_exponent =
            T(2 * (std::numeric_limits<T>::digits - std::numeric_limits<T>::min_exponent + 1)) * kLn2 / kEps;
        const V y_abs_is_huge = O::le(cst(huge_exponent), y_abs);

        // Negative finite base with a non-integer exponent: NaN.
        result = O::bor(O::bandnot(x_is_negative, y_is_int), result);
        // pow(+-0, y): +inf for negative y, +0 otherwise; the sign follows below.
        result = O::select(x_is_zero, O::band(y_is_negative, inf()), result);
        // A base with the sign bit set and an odd integer exponent negates the result.
        result = O::select(O::band(x_has_signbit, y_is_odd_int), negate(result), result);
        // |y| huge or infinite: 0 or inf by |x| < 1 or > 1, and pow(-1, +-inf) = 1.
        V inf_y_val = O::band(O::bxor(y_is_negative, x_abs_gt_one), inf());
        inf_y_val   = O::select(O::eq(x, cst(T(-1))), one(), inf_y_val);
        result      = O::select(y_abs_is_huge, inf_y_val, result);
        // pow(+-inf, y).
        const V x_pos_inf_value = O::bandnot(inf(), y_is_negative);
        const V x_neg_inf_value = O::select(y_is_odd_int, negate(x_pos_inf_value), x_pos_inf_value);
        result = O::select(x_abs_is_inf, O::select(x_is_negative, x_neg_inf_value, x_pos_inf_value), result);
        // NaN inputs give NaN, except for the two cases below.
        result = O::bor(O::bor(O::isnan(x), O::isnan(y)), result);
        // pow(x, 1) = x; pow(x, +-0) = 1 and pow(1, y) = 1, even for NaN.
        return O::select(y_is_one, x, O::select(O::bor(x_is_one, y_is_zero), one(), result));
    }

    // sqrt(x^2 + y^2) without intermediate over- or underflow: p sqrt(1 + (q / p)^2) for
    // p = max(|x|, |y|), q = min(|x|, |y|). Infinity wins over NaN, as in std::hypot.
    SCALARS_NOINLINE static V hypot(V x, V y)
    {
        const V a = abs(x);
        const V b = abs(y);
        const V p = O::max(a, b);
        const V q = O::min(a, b);
        const V r = O::div(q, p);
        V       h = O::mul(p, O::sqrt(madd(r, r, one())));
        h         = O::select(O::eq(p, zero()), zero(), h);
        h         = O::bor(O::bor(O::isnan(x), O::isnan(y)), h);
        return O::select(O::bor(isinf(x), isinf(y)), inf(), h);
    }

    SCALARS_FORCE_INLINE static V rsqrt(V x) { return O::div(one(), O::sqrt(x)); }

    //==========================================================================
    // Error function and the normal distribution (SpecialFunctionsImpl.h)
    //==========================================================================

    SCALARS_NOINLINE static V erfc(V x) { return erfc_dw(x, zero()); }

    // erfc(x_in + x_lo) for a double-word argument; x_lo enters through exp(-x^2) only, which
    // is where its effect is amplified (by 2 x^2 in relative terms).
    SCALARS_FORCE_INLINE static V erfc_dw(V x_in, V x_lo)
    {
        if constexpr (kFloat)
        {
            const V x  = O::min(O::max(x_in, cst(-11.0f)), cst(11.0f));
            x_lo       = O::select(O::lt(abs(x_in), cst(11.0f)), x_lo, zero());
            // erfc(x) = 1 + x S(x^2) for |x| <= 1.
            constexpr float alpha[] = {5.61802298761904239654541015625e-04f,
                                       -4.91381669417023658752441406250e-03f,
                                       2.67075151205062866210937500000e-02f,
                                       -1.12800106406211853027343750000e-01f,
                                       3.76122951507568359375000000000e-01f,
                                       -1.12837910652160644531250000000e+00f};
            const V         x2         = O::mul(x, x);
            const V         erfc_small = madd(x, polevl(x2, alpha), one());
            const V         gt_one     = O::lt(one(), x2);
            if (!O::any(gt_one))
            {
                return erfc_small;
            }
            // erfc(x) = exp(-x^2) P(1/x^2) / (x Q(1/x^2)) for 1 < |x| < 9.
            constexpr float gamma[] = {1.0208116471767425537109375e-01f,
                                       4.2920666933059692382812500e-01f,
                                       3.2379078865051269531250000e-01f,
                                       5.3971976041793823242187500e-02f};
            constexpr float delta[] = {1.7251677811145782470703125e-02f,
                                       3.9137163758277893066406250e-01f,
                                       1.0000000000000000000000000e+00f,
                                       6.2173241376876831054687500e-01f,
                                       9.5662862062454223632812500e-02f};
            return O::select(gt_one, erfc_large(x, x2, x_lo, gamma, delta), erfc_small);
        }
        else
        {
            const V x  = O::min(O::max(x_in, cst(-28.0)), cst(28.0));
            x_lo       = O::select(O::lt(abs(x_in), cst(28.0)), x_lo, zero());
            const V x2 = O::mul(x, x);
            // erfc(x) = 1 - x S(x^2) / T(x^2) for |x| <= 1.
            constexpr double alpha[] = {1.9493725660006057018823477644531294572516344487667083740234375e-04,
                                        1.8272566210022942682217328425053892715368419885635375976562500e-03,
                                        4.5303363351690106863856044583371840417385101318359375000000000e-02,
                                        1.4215015503619179981775744181504705920815467834472656250000000e-01,
                                        1.1283791670955125585606992899556644260883331298828125000000000e+00};
            constexpr double beta[]  = {2.0294484101083099089526257108317963684385176748037338256835938e-05,
                                        6.8117805899186819641732970609382391558028757572174072265625000e-04,
                                        1.0582026056098614921752165685120417037978768348693847656250000e-02,
                                        9.3252603143757495374188692949246615171432495117187500000000000e-02,
                                        4.5931062818368939559832142549566924571990966796875000000000000e-01,
                                        1.0};
            const V erfc_small = nmadd(x, O::div(polevl(x2, alpha), polevl(x2, beta)), one());
            const V gt_one     = O::lt(one(), x2);
            if (!O::any(gt_one))
            {
                return erfc_small;
            }
            // erfc(x) = exp(-x^2) P(1/x^2) / (x Q(1/x^2)) for 1 < |x| < 28.
            constexpr double gamma[] = {1.5252844933226974316088642158462107545346952974796295166015625e-04,
                                        1.0909912393738931124520519233556115068495273590087890625000000e-02,
                                        1.0628604636755033252537572252549580298364162445068359375000000e-01,
                                        3.3492472973137982217295416376146022230386734008789062500000000e-01,
                                        4.5065776215933289750026347064704168587923049926757812500000000e-01,
                                        2.9433039130294824659017649537418037652969360351562500000000000e-01,
                                        9.8792676360600226170838311645638896152377128601074218750000000e-02,
                                        1.7095935395503719655962981960328761488199234008789062500000000e-02,
                                        1.4249109729504577659398023570247460156679153442382812500000000e-03,
                                        4.4567378313647954771875570045835956989321857690811157226562500e-05};
            constexpr double delta[] = {2.041985103115789845773520028160419315099716186523437500000000e-03,
                                        5.316030659946043707142493417450168635696172714233398437500000000e-02,
                                        3.426242193784684864077405563875799998641014099121093750000000e-01,
                                        8.565637124308049799026321124983951449394226074218750000000000e-01,
                                        1.000000000000000000000000000000000000000000000000000000000000e+00,
                                        5.968805280570776972126623149961233139038085937500000000000000e-01,
                                        1.890922854723317836356244470152887515723705291748046875000000e-01,
                                        3.152505418656005586885981983868987299501895904541015625000000e-02,
                                        2.565085751861882583380047861965067568235099315643310546875000e-03,
                                        7.899362131678837697403017248376499992446042597293853759765625e-05};
            return O::select(gt_one, erfc_large(x, x2, x_lo, gamma, delta), erfc_small);
        }
    }

    template <std::size_t NP, std::size_t NQ>
    SCALARS_FORCE_INLINE static V erfc_large(
        V x, V x2, V x_lo, const T (&gamma)[NP], const T (&delta)[NQ])
    {
        // exp(-x^2) ~= exp(-x2) (1 - x2_lo) with x2 + x2_lo = (x + x_lo)^2 to first order.
        const V x2_lo   = madd(O::add(x, x), x_lo, twoprod_low(x, x, x2));
        const V exp2_hi = exp(negate(x2));
        const V z       = nmadd(exp2_hi, x2_lo, exp2_hi);
        const V q2      = reciprocal(x2);
        const V r       = O::div(polevl(q2, gamma), O::mul(x, polevl(q2, delta)));
        const V maybe_two = O::select(O::lt(x, zero()), cst(T(2)), zero());
        return madd(z, r, maybe_two);
    }

    // The standard normal distribution function, 0.5 erfc(-x / sqrt(2)). The argument is formed
    // as a double word: rounding it would cost about x^2 ulp of relative error in the lower tail.
    SCALARS_NOINLINE static V normal_cdf(V x)
    {
        constexpr T kHi = kFloat ? T(0.7071067690849304) : T(0.7071067811865476);
        constexpr T kLo = kFloat ? T(1.2101617485882343e-08) : T(-4.833646656726457e-17);
        V           z_hi, z_lo;
        twoprod(x, cst(-kHi), z_hi, z_lo);
        z_lo = madd(x, cst(-kLo), z_lo);
        return O::mul(cst(T(0.5)), erfc_dw(z_hi, z_lo));
    }

    // Its inverse, Cephes ndtri: NaN outside [0, 1], -inf at 0, +inf at 1.
    SCALARS_NOINLINE static V inv_normal_cdf(V a)
    {
        const V exp_neg_two     = cst(T(0.13533528323661269189));
        const V should_flipsign = O::le(a, O::sub(one(), exp_neg_two));
        const V b               = O::select(should_flipsign, a, O::sub(one(), a));
        const V ndtri           = O::select(
            O::lt(exp_neg_two, b), ndtri_gt_exp_neg_two(b), ndtri_lt_exp_neg_two(b, should_flipsign));
        return O::select(O::eq(a, zero()), negate(inf()), O::select(O::eq(one(), a), inf(), ndtri));
    }

    SCALARS_FORCE_INLINE static V ndtri_gt_exp_neg_two(V b)
    {
        constexpr T p0[] = {T(-5.99633501014107895267e1),
                            T(9.80010754185999661536e1),
                            T(-5.66762857469070293439e1),
                            T(1.39312609387279679503e1),
                            T(-1.23916583867381258016e0)};
        constexpr T q0[] = {T(1.0),
                            T(1.95448858338141759834e0),
                            T(4.67627912898881538453e0),
                            T(8.63602421390890590575e1),
                            T(-2.25462687854119370527e2),
                            T(2.00260212380060660359e2),
                            T(-8.20372256168333339912e1),
                            T(1.59056225126211695515e1),
                            T(-1.18331621121330003142e0)};
        const V     c    = O::sub(b, cst(T(0.5)));
        const V     c2   = O::mul(c, c);
        const V     r    = madd(c, O::mul(c2, O::div(polevl(c2, p0), polevl(c2, q0))), c);
        return O::mul(r, cst(T(2.50662827463100050242e0)));
    }

    SCALARS_FORCE_INLINE static V ndtri_lt_exp_neg_two(V b, V should_flipsign)
    {
        // z = sqrt(-2 log b) in [2, 8]: b between exp(-2) and exp(-32).
        constexpr T p1[] = {T(4.05544892305962419923e0),
                            T(3.15251094599893866154e1),
                            T(5.71628192246421288162e1),
                            T(4.40805073893200834700e1),
                            T(1.46849561928858024014e1),
                            T(2.18663306850790267539e0),
                            T(-1.40256079171354495875e-1),
                            T(-3.50424626827848203418e-2),
                            T(-8.57456785154685413611e-4)};
        constexpr T q1[] = {T(1.0),
                            T(1.57799883256466749731e1),
                            T(4.53907635128879210584e1),
                            T(4.13172038254672030440e1),
                            T(1.50425385692907503408e1),
                            T(2.50464946208309415979e0),
                            T(-1.42182922854787788574e-1),
                            T(-3.80806407691578277194e-2),
                            T(-9.33259480895457427372e-4)};
        // z in [8, 64]: b between exp(-32) and exp(-2048).
        constexpr T p2[] = {T(3.23774891776946035970e0),
                            T(6.91522889068984211695e0),
                            T(3.93881025292474443415e0),
                            T(1.33303460815807542389e0),
                            T(2.01485389549179081538e-1),
                            T(1.23716634817820021358e-2),
                            T(3.01581553508235416007e-4),
                            T(2.65806974686737550832e-6),
                            T(6.23974539184983293730e-9)};
        constexpr T q2[] = {T(1.0),
                            T(6.02427039364742014255e0),
                            T(3.67983563856160859403e0),
                            T(1.37702099489081330271e0),
                            T(2.16236993594496635890e-1),
                            T(1.34204006088543189037e-2),
                            T(3.28014464682127739104e-4),
                            T(2.89247864745380683936e-6),
                            T(6.79019408009981274425e-9)};
        const V     x    = O::sqrt(O::mul(cst(T(-2)), log(b)));
        const V     x0   = O::sub(x, O::div(log(x), x));
        const V     z    = reciprocal(x);
        const V     x1   = O::mul(z,
                            O::select(O::lt(x, cst(T(8))),
                                      O::div(polevl(z, p1), polevl(z, q1)),
                                      O::div(polevl(z, p2), polevl(z, q2))));
        return O::bxor(O::band(should_flipsign, sign_mask()), O::sub(x0, x1));
    }
};
}  // namespace scalars::detail::vmath
