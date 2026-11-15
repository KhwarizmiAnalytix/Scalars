// SIMD tests: the simd<complex<T>> surface of every complex scalar type
// (complex<float>, complex<double>, complex<half>, complex<bfloat16>).
//
// The typed `simd_complex` suite checks every one-, two-, and three-argument
// math function lane by lane against a std::complex reference evaluated in a
// wider type. Each function carries the smallest normwise ULP budget it meets
// (see `tol`), so a regression of one ULP is caught instead of hiding under a
// shared absolute tolerance. The `simd_complex_native` suite (float and double
// packets) keeps the contracts that are not accuracy budgets: the packet API,
// masks, memory layout, and special values (signed zeros, branch cuts,
// overflow-safe division and square root, fast-path range boundaries), using
// the same error measure wherever it compares values.
#include "backend/simd.h"

#if SCALARS_VECTORIZED

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numbers>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "scalars_test.h"
#include "types/complex.h"

using scalars::simd;

namespace
{
inline constexpr int kRandomTrials = 256;

// Per-type parameters. `reference_t` is the type the std::complex reference
// is evaluated in: double for the float-or-narrower types, and long double
// for double (80-bit on x86, 128-bit on Linux AArch64; it falls back to double
// where long double is double, as on Apple and MSVC targets). `eps` is the
// type's spacing at 1.0. `mag_lo`/`mag_hi` bound the log-uniform magnitude
// sweeps; `exp_limit` bounds the real part of the wide exp sweep.
template <class T> struct complex_traits;

template <> struct complex_traits<float>
{
    using reference_t                      = double;
    static constexpr const char* name      = "float";
    static constexpr double      eps       = 0x1p-23;
    static constexpr double      mag_lo    = 1e-30;
    static constexpr double      mag_hi    = 1e30;
    static constexpr double      exp_limit = 85;
};

template <> struct complex_traits<double>
{
    using reference_t                      = long double;
    static constexpr const char* name      = "double";
    static constexpr double      eps       = 0x1p-52;
    static constexpr double      mag_lo    = 1e-250;
    static constexpr double      mag_hi    = 1e250;
    static constexpr double      exp_limit = 710;
};

template <> struct complex_traits<scalars::half>
{
    using reference_t                      = double;
    static constexpr const char* name      = "half";
    static constexpr double      eps       = 0x1p-10;
    static constexpr double      mag_lo    = 1e-4;
    static constexpr double      mag_hi    = 1e4;
    static constexpr double      exp_limit = 11;
};

template <> struct complex_traits<scalars::bfloat16>
{
    using reference_t                      = double;
    static constexpr const char* name      = "bfloat16";
    static constexpr double      eps       = 0x1p-7;
    static constexpr double      mag_lo    = 1e-30;
    static constexpr double      mag_hi    = 1e30;
    static constexpr double      exp_limit = 85;
};

template <class T> using value_t   = scalars::complex<T>;
template <class T> using packet    = simd<value_t<T>>;
template <class T> using ref_t     = typename complex_traits<T>::reference_t;
template <class T> using reference = std::complex<ref_t<T>>;
template <class T> using lanes = std::array<value_t<T>, static_cast<std::size_t>(packet<T>::size)>;

template <class T, class U> T narrow(U v)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        return static_cast<T>(v);
    }
    else
    {
        // half also converts to float16_t on Arm; go through float explicitly.
        return T(static_cast<float>(v));
    }
}

template <class T> ref_t<T> widen_component(T v)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        return static_cast<ref_t<T>>(v);
    }
    else
    {
        return static_cast<ref_t<T>>(static_cast<float>(v));
    }
}

template <class T> reference<T> widen(value_t<T> z)
{
    return {widen_component(z.real()), widen_component(z.imag())};
}

template <class T> value_t<T> make(double re, double im)
{
    return value_t<T>(narrow<T>(re), narrow<T>(im));
}

template <class T> typename packet<T>::simd_t load(const lanes<T>& v)
{
    return packet<T>::loadu(v.data());
}

template <class T> lanes<T> store(typename packet<T>::simd_t p)
{
    lanes<T> out{};
    packet<T>::storeu(p, out.data());
    return out;
}

template <class U> long double printable(U v)
{
    if constexpr (std::is_floating_point_v<U>)
    {
        return static_cast<long double>(v);
    }
    else
    {
        return static_cast<long double>(static_cast<float>(v));
    }
}

template <class U> void show(std::ostream& os, const U& v)
{
    os << printable(v);
}

template <class U> void show(std::ostream& os, const scalars::complex<U>& z)
{
    os << '(' << printable(z.real()) << ", " << printable(z.imag()) << ')';
}

template <class U> void show(std::ostream& os, const std::complex<U>& z)
{
    os << '(' << printable(z.real()) << ", " << printable(z.imag()) << ')';
}

// mt19937 is specified bit for bit but the standard distributions are not, so
// samples are drawn with genrand_res53 and are identical on every standard
// library. That keeps the measured worst case, and with it each budget,
// portable.
double unit(std::mt19937& gen)
{
    const double hi = static_cast<double>(gen() >> 5);
    const double lo = static_cast<double>(gen() >> 6);
    return (hi * 67108864.0 + lo) * 0x1p-53;
}

// One component's input range. Uniform domains sample [lo, hi]; log-uniform
// domains sample magnitudes in [lo, hi] with a random sign. Samples are
// rounded to T and resampled if rounding leaves the domain.
struct domain
{
    double lo;
    double hi;
    bool   log_uniform = false;
};

struct region
{
    domain re;
    domain im;
};

constexpr region box(double a)
{
    return {{-a, a}, {-a, a}};
}

// Magnitudes across the type's range, in every quadrant.
template <class T> region sweep()
{
    const domain d{complex_traits<T>::mag_lo, complex_traits<T>::mag_hi, true};
    return {d, d};
}

// The square root of sweep(): products of two samples stay finite and normal.
template <class T> region root_sweep()
{
    const domain d{
        std::sqrt(complex_traits<T>::mag_lo), std::sqrt(complex_traits<T>::mag_hi), true};
    return {d, d};
}

// From just above the smallest subnormal to a quarter of the largest value.
template <class T> region full_sweep()
{
    const domain d{static_cast<double>(widen_component(std::numeric_limits<T>::denorm_min())) * 2,
        static_cast<double>(widen_component(std::numeric_limits<T>::max())) / 4,
        true};
    return {d, d};
}

// From the smallest normal number to a quarter of the largest value.
template <class T> region normal_sweep()
{
    const domain d{static_cast<double>(widen_component(std::numeric_limits<T>::min())),
        static_cast<double>(widen_component(std::numeric_limits<T>::max())) / 4,
        true};
    return {d, d};
}

template <class T> T sample(std::mt19937& gen, const domain& d)
{
    for (;;)
    {
        double v = 0;
        if (d.log_uniform)
        {
            const double sign   = (gen() & 1U) != 0 ? -1.0 : 1.0;
            const double log_lo = std::log(d.lo), log_hi = std::log(d.hi);
            v = sign * std::exp(log_lo + unit(gen) * (log_hi - log_lo));
        }
        else
        {
            v = d.lo + unit(gen) * (d.hi - d.lo);
        }
        const T      t = narrow<T>(v);
        const double r = static_cast<double>(widen_component(t));
        const double m = d.log_uniform ? std::fabs(r) : r;
        if (m >= d.lo && m <= d.hi)
        {
            return t;
        }
    }
}

template <class T> value_t<T> sample(std::mt19937& gen, const region& r)
{
    const T re = sample<T>(gen, r.re);
    const T im = sample<T>(gen, r.im);
    return value_t<T>(re, im);
}

// A per-type ULP budget, like the per-dtype tolerance overrides of PyTorch's
// OpInfo tables. Errors are normwise: the largest component error divided by
// eps * max(|re|, |im|) of the reference, so a ~1e-16 imaginary part next to
// a real part near 1 is not over-weighted, and a relative-accuracy loss in a
// small result is not hidden by an absolute floor. `floor` is the magnitude
// below which the error is measured absolutely; it defaults to the type's
// smallest normal number. `zero_sign` also requires an exact zero, with its
// sign, wherever the reference component is zero.
//
// The float and double budgets are the worst error measured over the fixed
// samples, rounded up to two significant digits and never below 0.5 (a
// correctly rounded result may be off by that much). They were measured on
// NEON with Accelerate and with SLEEF (macOS, libc++), NEON with SLEEF (Linux,
// libstdc++, 128-bit long double reference), and AVX2 with SVML and SSE
// (Linux, libstdc++). Apart from sqrt, acos, and the packet fast paths, the
// complex math forwards to std::complex<T>, so the budgets also bound the
// standard library: libc++'s float asin, acosh, asinh, and tanh are near
// 18 ULPs where libstdc++'s are near 1. The x86 runs were emulated (see
// ref_log1p_real). A failure prints the measured error and the inputs that
// produced it.
//
// complex<half> and complex<bfloat16> compute through complex<float> and
// round once, so every function is held to half a ULP plus the float
// intermediate's own error, which is far below 0.01 ULP of either type.
struct tol
{
    double f;                 // complex<float>
    double d;                 // complex<double>
    double reduced   = 0.51;  // complex<half> and complex<bfloat16>
    double floor     = 0;
    bool   zero_sign = false;
};

// Bit-exact operations: zero error and matching signed zeros.
constexpr tol kExact{0, 0, 0, 0, true};

constexpr tol signed_zeros(tol t)
{
    t.zero_sign = true;
    return t;
}

// Whether double references are evaluated in a wider type. Where long double
// is double (Apple, MSVC), a double reference is the standard library's own
// double result, and its error is part of what is measured.
constexpr bool kWideReference =
    std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits;

template <class T> double ulps_for(const tol& t)
{
    if constexpr (std::is_same_v<T, float>)
    {
        return t.f;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return t.d;
    }
    else
    {
        return t.reduced;
    }
}

// Tracks the worst lane of one check; report() asserts it once, with the
// inputs that produced it.
template <class T> class accuracy_check
{
public:
    using R = ref_t<T>;  // NOLINT(readability-identifier-naming)

    accuracy_check(std::string name, const tol& t)
        : name_(std::move(name)), ulps_(ulps_for<T>(t)),
          floor_(std::max(static_cast<R>(t.floor), widen_component(std::numeric_limits<T>::min()))),
          zero_sign_(t.zero_sign)
    {
    }

    void report() const
    {
        EXPECT_LE(worst_, ulps_) << "complex<" << complex_traits<T>::name << ">::" << name_
                                 << detail_;
    }

    template <class... Inputs>
    void check(value_t<T> actual, reference<T> expected, const Inputs&... inputs)
    {
        record(error(actual, expected, R(-1)), actual, expected, inputs...);
    }

    // Measures the error against `scale` instead of the size of the result.
    template <class... Inputs>
    void check_scaled(value_t<T> actual, reference<T> expected, R scale, const Inputs&... inputs)
    {
        record(error(actual, expected, scale), actual, expected, inputs...);
    }

private:
    double error(value_t<T> actual, reference<T> expected, R scale) const
    {
        constexpr double       inf = std::numeric_limits<double>::infinity();
        const std::array<R, 2> got{widen_component(actual.real()), widen_component(actual.imag())};
        const std::array<R, 2> want{expected.real(), expected.imag()};
        R                      err       = 0;
        R                      magnitude = 0;
        for (std::size_t i = 0; i < 2; ++i)
        {
            // Compare exceptional components against the reference rounded
            // to T, so overflow in the narrower type is expected behaviour.
            const R rounded = widen_component(narrow<T>(want[i]));
            if (std::isnan(want[i]))
            {
                if (!std::isnan(got[i]))
                {
                    return inf;
                }
                continue;
            }
            if (std::isinf(rounded))
            {
                if (got[i] != rounded)
                {
                    return inf;
                }
                continue;
            }
            if (!std::isfinite(got[i]))
            {
                return inf;
            }
            if (zero_sign_ && want[i] == 0 &&
                (got[i] != 0 || std::signbit(got[i]) != std::signbit(want[i])))
            {
                return inf;
            }
            err       = std::max(err, std::fabs(got[i] - want[i]));
            magnitude = std::max(magnitude, std::fabs(want[i]));
        }
        if (scale >= 0)
        {
            magnitude = scale;
        }
        const R unit = static_cast<R>(complex_traits<T>::eps) * std::max(magnitude, floor_);
        return static_cast<double>(err / unit);
    }

    template <class... Inputs>
    void record(double ulps, value_t<T> actual, reference<T> expected, const Inputs&... inputs)
    {
        if (!(ulps > worst_))
        {
            return;
        }
        worst_ = ulps;
        std::ostringstream os;
        os << std::setprecision(21) << "\n  inputs   =";
        ((os << ' ', show(os, inputs)), ...);
        os << "\n  simd     = ";
        show(os, actual);
        os << "\n  expected = ";
        show(os, expected);
        os << "\n  error    = " << ulps << " ulps (budget " << ulps_ << ')';
        detail_ = os.str();
    }

    std::string name_;
    double      ulps_;
    R           floor_;
    bool        zero_sign_;
    double      worst_ = 0;
    std::string detail_;
};

// One-shot comparison for the special-value tables.
template <class T, class... Inputs>
void expect_ulps(const char* what,
    value_t<T>               actual,
    reference<T>             expected,
    const tol&               t,
    const Inputs&... inputs)
{
    accuracy_check<T> check(what, t);
    check.check(actual, expected, inputs...);
    check.report();
}

// Real-valued results are compared as complex numbers with a zero imaginary
// part.
template <class T, class... Inputs>
void expect_ulps(
    const char* what, T actual, ref_t<T> expected, const tol& t, const Inputs&... inputs)
{
    expect_ulps<T>(what,
        value_t<T>(actual, narrow<T>(0.0)),
        reference<T>(expected, ref_t<T>(0)),
        t,
        inputs...);
}

// -- References. Evaluated in ref_t<T>; formulas are chosen so the reference
//    does not lose the accuracy it is meant to check. --

// Kahan's difference of products: re and im of x * y, each within about one
// ulp of R even under cancellation.
template <class R> std::complex<R> ref_mul(std::complex<R> x, std::complex<R> y)
{
    const auto diff = [](R a, R b, R c, R d)
    {
        const R cd = c * d;
        return std::fma(a, b, -cd) - std::fma(c, d, -cd);
    };
    const auto sum = [](R a, R b, R c, R d)
    {
        const R cd = c * d;
        return std::fma(a, b, cd) + std::fma(c, d, -cd);
    };
    return {
        diff(x.real(), y.real(), x.imag(), y.imag()), sum(x.real(), y.imag(), x.imag(), y.real())};
}

template <class R> std::complex<R> ref_expm1(std::complex<R> z)
{
    // expm1(x)cos(y) + (cos(y) - 1), with cos(y) - 1 = -2 sin^2(y / 2).
    const R s = std::sin(z.imag() / 2);
    return {std::expm1(z.real()) * std::cos(z.imag()) - 2 * s * s,
        std::exp(z.real()) * std::sin(z.imag())};
}

// Real log1p for the references. Emulated x87 (Rosetta, Docker on Apple
// silicon) evaluates log1pl only to about 2^-64 absolute, which is far from
// relative accuracy when log1p(u) is tiny, so a long double reference sums
// log1p(u) = 2 atanh(s), s = u / (2 + u), for small arguments instead.
template <class R> R ref_log1p_real(R u)
{
    if (std::numeric_limits<R>::digits <= std::numeric_limits<double>::digits ||
        !(std::fabs(u) <= R(0x1p-6)))
    {
        return std::log1p(u);
    }
    const R two_s  = u / (1 + u / 2);  // 2s, exact for subnormal u
    const R s2     = two_s * two_s / 4;
    R       series = 0;
    for (int k = 12; k >= 0; --k)
    {
        series = series * s2 + R(1) / R(2 * k + 1);
    }
    return two_s * series;
}

template <class R> std::complex<R> ref_log1p(std::complex<R> z)
{
    // log|1 + z| = log1p(2x + x^2 + y^2) / 2 avoids forming 1 + z.
    const R x = z.real(), y = z.imag();
    return {ref_log1p_real(std::fma(x, 2 + x, y * y)) / 2, std::atan2(y, 1 + x)};
}

// std::log rounds |z| before taking its logarithm, which loses all relative
// accuracy in log|z| near |z| = 1. There |z|^2 - 1 is formed from error-free
// products and a two-sum (not Fast2Sum, so the reference does not share the
// implementation's ordering assumption) instead.
template <class R> std::complex<R> ref_log(std::complex<R> z)
{
    const R hi = std::max(std::fabs(z.real()), std::fabs(z.imag()));
    const R lo = std::min(std::fabs(z.real()), std::fabs(z.imag()));
    if (!(hi >= R(0.5) && hi <= R(2)))
    {
        return std::log(z);
    }
    const R hi2 = hi * hi, hi2_error = std::fma(hi, hi, -hi2);
    const R lo2 = lo * lo, lo2_error = std::fma(lo, lo, -lo2);
    const R sum = hi2 + lo2, rounded_hi2 = sum - lo2;
    const R sum_error = (hi2 - rounded_hi2) + (lo2 - (sum - rounded_hi2));
    return {ref_log1p_real((sum - 1) + (sum_error + hi2_error + lo2_error)) / 2,
        std::atan2(z.imag(), z.real())};
}

// Known limitation: scalars::log, the fallback for lanes outside the packet
// fast path, delegates to std::log(std::complex<T>), and libc++ forms |z| in
// T before taking its logarithm. For finite z with |z| > max, log|z| is +inf
// (the real part of log(FLT_MAX (1 + i)) is 89.07), and for |z| below the
// smallest normal number, |z| is first rounded to the subnormal grid
// (log|denorm_min (1 - i)| comes out as -103.28 instead of -102.93). Such
// lanes are held to parity with the standard library; every other lane is
// held to the exact value.
template <class T> reference<T> log_reference(value_t<T> z)
{
    const std::complex<T> narrow_z(z.real(), z.imag());
    const T               magnitude = std::abs(narrow_z);
    if (scalars::isfinite(z) &&
        (std::isinf(magnitude) || (magnitude != T(0) && magnitude < std::numeric_limits<T>::min())))
    {
        return widen(value_t<T>(std::log(narrow_z)));
    }
    return ref_log(widen(z));
}

template <class R> R ref_abs2(std::complex<R> z)
{
    return z.real() * z.real() + z.imag() * z.imag();
}

// libc++ computes std::sqrt(z) as polar(sqrt(abs(z)), arg(z) / 2), and abs(z)
// rounds to the subnormal grid first: |denorm_min * (1 - i)| becomes
// denorm_min instead of sqrt(2) * denorm_min, a 2^(1/4) error in the result.
// Scaling by an even power of two, 2^(2k), is exact here and divides out as
// 2^k after the square root.
template <class R> std::complex<R> ref_sqrt(std::complex<R> z)
{
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag()) ||
        std::fabs(z.real()) >= std::numeric_limits<R>::min() ||
        std::fabs(z.imag()) >= std::numeric_limits<R>::min() ||
        (z.real() == R(0) && z.imag() == R(0)))
    {
        return std::sqrt(z);
    }
    constexpr int k = std::numeric_limits<R>::digits;
    const auto    r =
        std::sqrt(std::complex<R>(std::ldexp(z.real(), 2 * k), std::ldexp(z.imag(), 2 * k)));
    return {std::ldexp(r.real(), -k), std::ldexp(r.imag(), -k)};
}

template <class R, class F> std::complex<R> componentwise(std::complex<R> z, F f)
{
    return {f(z.real()), f(z.imag())};
}

// -- Lane-by-lane checkers for one-, two-, and three-argument functions. --

template <class T, class SimdOp, class RefOp>
void check_unary(const char* name, const region& d, const tol& t, SimdOp simd_op, RefOp ref_op)
{
    accuracy_check<T> check(name, t);
    // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility
    std::mt19937 gen(5489u);
    lanes<T>     x{};
    for (int trial = 0; trial < kRandomTrials; ++trial)
    {
        for (auto& v : x)
        {
            v = sample<T>(gen, d);
        }
        const lanes<T> out = store<T>(simd_op(load<T>(x)));
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            check.check(out[i], ref_op(widen(x[i])), x[i]);
        }
    }
    check.report();
}

// Complex-to-real functions (abs, arg, ...), stored through simd<T>.
template <class T, class SimdOp, class RefOp>
void check_real_valued(
    const char* name, const region& d, const tol& t, SimdOp simd_op, RefOp ref_op)
{
    accuracy_check<T> check(name, t);
    // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility
    std::mt19937                                             gen(5489u);
    lanes<T>                                                 x{};
    std::array<T, static_cast<std::size_t>(packet<T>::size)> out{};
    for (int trial = 0; trial < kRandomTrials; ++trial)
    {
        for (auto& v : x)
        {
            v = sample<T>(gen, d);
        }
        simd<T>::storeu(simd_op(load<T>(x)), out.data());
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            check.check(value_t<T>(out[i], narrow<T>(0.0)),
                reference<T>(ref_op(widen(x[i])), ref_t<T>(0)),
                x[i]);
        }
    }
    check.report();
}

template <class T, class SimdOp, class RefOp>
void check_binary(const char* name,
    const region&             dx,
    const region&             dy,
    const tol&                t,
    SimdOp                    simd_op,
    RefOp                     ref_op)
{
    accuracy_check<T> check(name, t);
    // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility
    std::mt19937 gen(5489u);
    lanes<T>     x{};
    lanes<T>     y{};
    for (int trial = 0; trial < kRandomTrials; ++trial)
    {
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            x[i] = sample<T>(gen, dx);
            y[i] = sample<T>(gen, dy);
        }
        const lanes<T> out = store<T>(simd_op(load<T>(x), load<T>(y)));
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            check.check(out[i], ref_op(widen(x[i]), widen(y[i])), x[i], y[i]);
        }
    }
    check.report();
}

// fma(a, b, c) = a * b + c. Its two real products round separately even when
// each component is fused, so under cancellation the error is bounded by the
// size of the terms, |a| |b| + |c|, not by the result.
template <class T> void check_fma(const char* name, const region& d, const tol& t, int cancel_every)
{
    using S = packet<T>;  // NOLINT(readability-identifier-naming)
    using R = ref_t<T>;   // NOLINT(readability-identifier-naming)
    accuracy_check<T> check(name, t);
    // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility
    std::mt19937 gen(5489u);
    lanes<T>     x{}, y{}, z{};
    for (int trial = 0; trial < kRandomTrials; ++trial)
    {
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            x[i] = sample<T>(gen, d);
            y[i] = sample<T>(gen, d);
            z[i] = sample<T>(gen, d);
            // Some lanes cancel the product exactly or nearly so.
            if (cancel_every > 0 && (trial + static_cast<int>(i)) % cancel_every == 0)
            {
                const auto p = ref_mul(widen(x[i]), widen(y[i]));
                z[i]         = value_t<T>(narrow<T>(-p.real()), narrow<T>(-p.imag()));
            }
        }
        const lanes<T> out = store<T>(S::fma(load<T>(x), load<T>(y), load<T>(z)));
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            const auto a = widen(x[i]), b = widen(y[i]), c = widen(z[i]);
            const R    scale = std::abs(a) * std::abs(b) + std::abs(c);
            check.check_scaled(out[i], ref_mul(a, b) + c, scale, x[i], y[i], z[i]);
        }
    }
    check.report();
}

template <class T> class simd_complex : public ::testing::Test
{
};

struct complex_type_names
{
    template <class T> static std::string GetName(int /*index*/)  // NOLINT
    {
        return complex_traits<T>::name;
    }
};

using complex_types = ::testing::Types<float, double, scalars::half, scalars::bfloat16>;
TYPED_TEST_SUITE(simd_complex, complex_types, complex_type_names);

TYPED_TEST(simd_complex, UnaryMath)
{
    using T           = TypeParam;  // NOLINT(readability-identifier-naming)
    using S           = packet<T>;  // NOLINT(readability-identifier-naming)
    using R           = ref_t<T>;   // NOLINT(readability-identifier-naming)
    using simd_t      = typename S::simd_t;
    using ref         = reference<T>;
    const region wide = sweep<T>();
    const double e    = complex_traits<T>::exp_limit;
    const ref    one(1);

#define CHECK_UNARY(fn, dom, tolerance, expr)                                                      \
    check_unary<T>(                                                                                \
        #fn " on " #dom,                                                                           \
        dom,                                                                                       \
        tolerance,                                                                                 \
        [](simd_t a) { return S::fn(a); },                                                         \
        [&](ref z) { return expr; })

    CHECK_UNARY(sqrt, wide, (tol{0.92, 1.3}), std::sqrt(z));
    CHECK_UNARY(sqrt, box(10), (tol{1.1, 1.3}), std::sqrt(z));
    CHECK_UNARY(invsqrt, wide, (tol{1.7, 2.3}), one / std::sqrt(z));
    CHECK_UNARY(invsqrt, box(10), (tol{1.7, 2.2}), one / std::sqrt(z));
    CHECK_UNARY(reciprocal, wide, (tol{0.95, 0.98}), one / z);
    CHECK_UNARY(reciprocal, box(10), (tol{1.3, 2}), one / z);
    CHECK_UNARY(sqr, box(100), (tol{0.84, 0.98}), ref_mul(z, z));
    CHECK_UNARY(neg, wide, kExact, -z);
    CHECK_UNARY(conj, wide, kExact, std::conj(z));
    CHECK_UNARY(ceil, box(75), kExact, componentwise(z, [](R v) { return std::ceil(v); }));
    CHECK_UNARY(floor, box(75), kExact, componentwise(z, [](R v) { return std::floor(v); }));
    CHECK_UNARY(trunc, box(75), kExact, componentwise(z, [](R v) { return std::trunc(v); }));

    CHECK_UNARY(exp, (region{{-8, 8}, {-25, 25}}), (tol{2.2, 2.5}), std::exp(z));
    CHECK_UNARY(exp, (region{{-e, e}, {-512, 512}}), (tol{2.2, 2.8}), std::exp(z));
    CHECK_UNARY(expm1, (region{{-8, 8}, {-25, 25}}), (tol{2, 1.8}), ref_expm1(z));
    CHECK_UNARY(expm1, box(1e-3), (tol{1.3, 0.96}), ref_expm1(z));
    CHECK_UNARY(
        exp2, (region{{-12, 12}, {-25, 25}}), (tol{8.8, 12}), std::exp(z * std::numbers::ln2_v<R>));
    CHECK_UNARY(
        exp10, (region{{-3, 3}, {-10, 10}}), (tol{12, 19}), std::exp(z * std::numbers::ln10_v<R>));
    CHECK_UNARY(log, wide, (tol{0.95, 0.98}), ref_log(z));
    // Subnormal components are excluded: libc++'s std::log, which the scalar
    // fallback uses, rounds |z| to the subnormal grid first (see log_reference).
    CHECK_UNARY(log, normal_sweep<T>(), (tol{1.1, 0.98}), ref_log(z));
    CHECK_UNARY(log, (region{{0.999, 1.001}, {-1e-3, 1e-3}}), (tol{1.3, 1.7}), ref_log(z));
    CHECK_UNARY(log1p, (region{{-0.9, 50}, {-50, 50}}), (tol{1.6, 1.3}), ref_log1p(z));
    CHECK_UNARY(log1p, box(1e-3), (tol{2.3, 2.1}), ref_log1p(z));
    CHECK_UNARY(log2, wide, (tol{0.8, 0.92}), ref_log(z) / std::numbers::ln2_v<R>);
    CHECK_UNARY(log10, wide, (tol{0.94, 1.1}), ref_log(z) / std::numbers::ln10_v<R>);

    CHECK_UNARY(sin, (region{{-25, 25}, {-8, 8}}), (tol{1.3, 1.2}), std::sin(z));
    CHECK_UNARY(cos, (region{{-25, 25}, {-8, 8}}), (tol{1.3, 1.3}), std::cos(z));
    CHECK_UNARY(tan, (region{{-25, 25}, {-8, 8}}), (tol{3.2, 2.1}), std::tan(z));
    CHECK_UNARY(asin, box(4), (tol{18, 1.1}), std::asin(z));
    CHECK_UNARY(acos, box(4), (tol{1.1, kWideReference ? 1.2 : 19}), std::acos(z));
    CHECK_UNARY(atan, box(4), (tol{6.3, 1.2}), std::atan(z));
    CHECK_UNARY(sinh, (region{{-8, 8}, {-25, 25}}), (tol{1.4, 1.5}), std::sinh(z));
    CHECK_UNARY(cosh, (region{{-8, 8}, {-25, 25}}), (tol{1.3, 1.5}), std::cosh(z));
    CHECK_UNARY(tanh, (region{{-8, 8}, {-25, 25}}), (tol{18, 2.4}), std::tanh(z));
    CHECK_UNARY(asinh, box(4), (tol{15, 1.1}), std::asinh(z));
    CHECK_UNARY(acosh, box(4), (tol{17, 0.97}), std::acosh(z));
    CHECK_UNARY(atanh, box(4), (tol{4.6, 0.69}), std::atanh(z));
#undef CHECK_UNARY
}

TYPED_TEST(simd_complex, RealValuedFunctions)
{
    using T           = TypeParam;  // NOLINT(readability-identifier-naming)
    using S           = packet<T>;  // NOLINT(readability-identifier-naming)
    using simd_t      = typename S::simd_t;
    using ref         = reference<T>;
    const region wide = sweep<T>();

#define CHECK_REAL(fn, dom, tolerance, expr)                                                       \
    check_real_valued<T>(                                                                          \
        #fn " on " #dom,                                                                           \
        dom,                                                                                       \
        tolerance,                                                                                 \
        [](simd_t a) { return S::fn(a); },                                                         \
        [](ref z) { return expr; })

    CHECK_REAL(abs, wide, (tol{1.1, 1.5}), std::hypot(z.real(), z.imag()));
    CHECK_REAL(abs, full_sweep<T>(), (tol{0.97, 0.88}), std::hypot(z.real(), z.imag()));
    CHECK_REAL(fabs, wide, (tol{1.1, 1.5}), std::hypot(z.real(), z.imag()));
    CHECK_REAL(abs2, box(100), (tol{0.91, 1}), ref_abs2(z));
    CHECK_REAL(abs2, root_sweep<T>(), (tol{0.92, 0.98}), ref_abs2(z));
    CHECK_REAL(norm, box(100), (tol{0.91, 1}), ref_abs2(z));
    CHECK_REAL(arg, wide, (tol{0.86, 0.64}), std::atan2(z.imag(), z.real()));
    CHECK_REAL(arg, box(10), (tol{1.1, 1}), std::atan2(z.imag(), z.real()));
    CHECK_REAL(real, wide, kExact, z.real());
    CHECK_REAL(imag, wide, kExact, z.imag());
#undef CHECK_REAL
}

TYPED_TEST(simd_complex, BinaryMath)
{
    using T           = TypeParam;  // NOLINT(readability-identifier-naming)
    using S           = packet<T>;  // NOLINT(readability-identifier-naming)
    using simd_t      = typename S::simd_t;
    using ref         = reference<T>;
    const region wide = sweep<T>();
    const region root = root_sweep<T>();
    const region full = full_sweep<T>();

#define CHECK_BINARY(fn, dx, dy, tolerance, expr)                                                  \
    check_binary<T>(                                                                               \
        #fn " on " #dx " x " #dy,                                                                  \
        dx,                                                                                        \
        dy,                                                                                        \
        tolerance,                                                                                 \
        [](simd_t a, simd_t b) { return S::fn(a, b); },                                            \
        [](ref x, ref y) { return expr; })

    CHECK_BINARY(add, box(40), box(40), (tol{0.5, 0.5}), x + y);
    CHECK_BINARY(add, wide, wide, (tol{0.5, 0.5}), x + y);
    CHECK_BINARY(sub, box(40), box(40), (tol{0.5, 0.5}), x - y);
    CHECK_BINARY(sub, wide, wide, (tol{0.5, 0.5}), x - y);
    CHECK_BINARY(mul, box(25), box(25), (tol{0.85, 1}), ref_mul(x, y));
    CHECK_BINARY(mul, root, root, (tol{0.91, 0.57}), ref_mul(x, y));
    CHECK_BINARY(div, box(50), box(50), (tol{1.7, 2}), x / y);
    CHECK_BINARY(div, wide, wide, (tol{1.3, 1.1}), x / y);
    // Quotients of the whole range reach overflow, underflow, and the scaled
    // fallback for operands outside the fast path's bounds.
    CHECK_BINARY(div, full, full, (tol{1.4, 1.2}), x / y);
    CHECK_BINARY(pow,
        (region{{0.05, 12}, {-3, 3}}),
        (region{{-3, 3}, {-1, 1}}),
        (tol{6.2, 6.7}),
        std::pow(x, y));
#undef CHECK_BINARY
}

TYPED_TEST(simd_complex, FusedMultiplyAdd)
{
    using T = TypeParam;  // NOLINT(readability-identifier-naming)
    check_fma<T>("fma", box(25), tol{0.7, 0.97}, 0);
    check_fma<T>("fma(cancelling)", box(25), tol{0.7, 0.91}, 4);
}

// set_mask accepts bool and plain integer literals alike (nonzero selects
// all); the mask is observed through if_else.
TYPED_TEST(simd_complex, SetMaskAndSelect)
{
    using T = TypeParam;  // NOLINT(readability-identifier-naming)
    using S = packet<T>;  // NOLINT(readability-identifier-naming)
    lanes<T> a{}, b{};
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        a[i] = make<T>(double(i + 1), -double(i));
        b[i] = make<T>(-2, 3);
    }
    const auto pa = load<T>(a), pb = load<T>(b);
    const auto expect_selects = [&](typename S::mask_t m, const lanes<T>& want, const char* what)
    {
        const auto out = store<T>(S::if_else(m, pa, pb));
        for (std::size_t i = 0; i < out.size(); ++i)
        {
            EXPECT_EQ(widen(out[i]), widen(want[i])) << what << " lane " << i;
        }
    };
    expect_selects(S::set_mask(1), a, "set_mask(1)");
    expect_selects(S::set_mask(0), b, "set_mask(0)");
    expect_selects(S::set_mask(true), a, "set_mask(true)");
    expect_selects(S::set_mask(false), b, "set_mask(false)");
    expect_selects(S::set_mask(std::int64_t(-3)), a, "set_mask(int64_t(-3))");
    expect_selects(S::set_mask(typename S::int_t(0)), b, "set_mask(int_t(0))");
}

// ---------------------------------------------------------------------------
// Packet API, layout, and special values of the float and double packets.
// ---------------------------------------------------------------------------

template <class S>
concept complex_packet_api = requires(typename S::simd_t a,
    typename S::mask_t                                   m,
    typename S::simd_half_t                              h,
    typename S::value_t*                                 out,
    const typename S::value_t*                           in,
    typename S::int_t*                                   flags,
    const int*                                           indices) {
    typename S::simd_int_t;
    typename S::real_packet_t;
    { S::add(a, a) } -> std::same_as<typename S::simd_t>;
    { S::sub(a, a) } -> std::same_as<typename S::simd_t>;
    { S::mul(a, a) } -> std::same_as<typename S::simd_t>;
    { S::div(a, a) } -> std::same_as<typename S::simd_t>;
    { S::sqr(a) } -> std::same_as<typename S::simd_t>;
    { S::neg(a) } -> std::same_as<typename S::simd_t>;
    { S::fma(a, a, a) } -> std::same_as<typename S::simd_t>;
    { S::conj(a) } -> std::same_as<typename S::simd_t>;
    { S::sqrt(a) } -> std::same_as<typename S::simd_t>;
    { S::invsqrt(a) } -> std::same_as<typename S::simd_t>;
    { S::reciprocal(a) } -> std::same_as<typename S::simd_t>;
    { S::pow(a, a) } -> std::same_as<typename S::simd_t>;
    { S::exp(a) } -> std::same_as<typename S::simd_t>;
    { S::expm1(a) } -> std::same_as<typename S::simd_t>;
    { S::exp2(a) } -> std::same_as<typename S::simd_t>;
    { S::exp10(a) } -> std::same_as<typename S::simd_t>;
    { S::log(a) } -> std::same_as<typename S::simd_t>;
    { S::log1p(a) } -> std::same_as<typename S::simd_t>;
    { S::log2(a) } -> std::same_as<typename S::simd_t>;
    { S::log10(a) } -> std::same_as<typename S::simd_t>;
    { S::sin(a) } -> std::same_as<typename S::simd_t>;
    { S::cos(a) } -> std::same_as<typename S::simd_t>;
    { S::tan(a) } -> std::same_as<typename S::simd_t>;
    { S::asin(a) } -> std::same_as<typename S::simd_t>;
    { S::acos(a) } -> std::same_as<typename S::simd_t>;
    { S::atan(a) } -> std::same_as<typename S::simd_t>;
    { S::sinh(a) } -> std::same_as<typename S::simd_t>;
    { S::cosh(a) } -> std::same_as<typename S::simd_t>;
    { S::tanh(a) } -> std::same_as<typename S::simd_t>;
    { S::asinh(a) } -> std::same_as<typename S::simd_t>;
    { S::acosh(a) } -> std::same_as<typename S::simd_t>;
    { S::atanh(a) } -> std::same_as<typename S::simd_t>;
    { S::ceil(a) } -> std::same_as<typename S::simd_t>;
    { S::floor(a) } -> std::same_as<typename S::simd_t>;
    { S::trunc(a) } -> std::same_as<typename S::simd_t>;
    S::eq(a, a);
    S::neq(a, a);
    S::and_mask(m, m);
    S::or_mask(m, m);
    S::xor_mask(m, m);
    S::not_mask(m);
    { S::if_else(m, a, a) } -> std::same_as<typename S::simd_t>;
    S::load(flags);
    S::loadu(flags);
    S::store(m, flags);
    S::storeu(m, flags);
    S::set_mask(true);
    S::set_mask(typename S::int_t(1));
    S::prefetch(in);
    S::copy(in, out);
    S::broadcast(in, a, a, a, a);
    S::gather(in, indices);
    S::scatter(a, indices, out);
    S::ploadquad(in);
    S::template ptranspose<S::size>(&a);
    S::loadu_half(in);
    S::gather_half(in, 1);
    S::gather_half(in, indices);
    S::set(in);
    S::add(h, h);
    S::sub(h, h);
    S::mul(h, h);
    S::fma(h, h, h);
    S::storeu(h, out);
    S::scatter(h, 1, out);
    S::scatter(h, indices, out);
    S::storeu(S::abs(a), static_cast<typename S::r::value_t*>(nullptr));
    S::abs2(a);
    S::fabs(a);
    S::arg(a);
    S::real(a);
    S::imag(a);
    { S::accumulate(a) } -> std::same_as<typename S::accumulation_t>;
};

// Ordering, min/max, and the real-only functions do not exist for complex
// packets of any component type.
template <class S> constexpr bool excludes_real_only()
{
    using v = typename S::simd_t;
    return !requires(v a) { S::gt(a, a); } && !requires(v a) { S::ge(a, a); } &&
           !requires(v a) { S::lt(a, a); } && !requires(v a) { S::le(a, a); } &&
           !requires(v a) { S::min(a, a); } && !requires(v a) { S::max(a, a); } &&
           !requires(v a) { S::hmin(a); } && !requires(v a) { S::hmax(a); } &&
           !requires(v a) { S::signcopy(a, a); } && !requires(v a) { S::hypot(a, a); } &&
           !requires(v a) { S::cbrt(a); } && !requires(v a) { S::cdf(a); } &&
           !requires(v a) { S::inv_cdf(a); };
}

static_assert(complex_packet_api<simd<scalars::complex<float>>>);
static_assert(complex_packet_api<simd<scalars::complex<double>>>);
static_assert(excludes_real_only<simd<scalars::complex<float>>>());
static_assert(excludes_real_only<simd<scalars::complex<double>>>());
static_assert(excludes_real_only<simd<scalars::complex<scalars::half>>>());
static_assert(excludes_real_only<simd<scalars::complex<scalars::bfloat16>>>());

template <class T> void expect_bits(scalars::complex<T> actual, scalars::complex<T> expected)
{
    using bits = typename simd<scalars::complex<T>>::int_t;
    EXPECT_EQ(std::bit_cast<bits>(actual.real()), std::bit_cast<bits>(expected.real()));
    EXPECT_EQ(std::bit_cast<bits>(actual.imag()), std::bit_cast<bits>(expected.imag()));
}

template <class T> class simd_complex_native : public ::testing::Test
{
};

using native_types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(simd_complex_native, native_types, complex_type_names);

TYPED_TEST(simd_complex_native, MasksAndSelection)
{
    using T    = TypeParam;  // NOLINT(readability-identifier-naming)
    using c    = scalars::complex<T>;
    using s    = simd<c>;
    using bits = typename s::int_t;
    static_assert(sizeof(bits) == sizeof(T));
    std::array<c, s::size>    x, y, out;
    std::array<bits, s::size> flags;
    for (int trial = 0; trial < 8; ++trial)
    {
        for (int i = 0; i < s::size; ++i)
        {
            x[i] = c(T(i), T(-i));
            y[i] = x[i];
            if ((i + trial) % 3 == 0)
            {
                y[i].imag(T(i + 1));
            }
            if ((i + trial) % 5 == 0)
            {
                x[i] = c(T(0), T(-0.0));
                y[i] = c(T(-0.0), T(0));
            }
            if ((i + trial) % 7 == 0)
            {
                x[i].real(std::numeric_limits<T>::quiet_NaN());
                y[i] = x[i];
            }
        }
        const auto a = s::loadu(x.data()), b = s::loadu(y.data());
        const auto equal = s::eq(a, b), different = s::neq(a, b);
        s::storeu(equal, flags.data());
        for (int i = 0; i < s::size; ++i)
        {
            EXPECT_EQ(flags[i], x[i] == y[i] ? ~bits(0) : bits(0));
        }
        s::storeu(different, flags.data());
        for (int i = 0; i < s::size; ++i)
        {
            EXPECT_EQ(flags[i], x[i] != y[i] ? ~bits(0) : bits(0));
        }
        s::storeu(s::if_else(equal, a, b), out.data());
        for (int i = 0; i < s::size; ++i)
        {
            expect_bits(out[i], x[i] == y[i] ? x[i] : y[i]);
        }
        s::storeu(s::and_mask(equal, different), flags.data());
        for (const auto flag : flags)
        {
            EXPECT_EQ(flag, bits(0));
        }
        s::storeu(s::or_mask(equal, different), flags.data());
        for (const auto flag : flags)
        {
            EXPECT_EQ(flag, ~bits(0));
        }
        s::storeu(s::xor_mask(equal, different), flags.data());
        for (const auto flag : flags)
        {
            EXPECT_EQ(flag, ~bits(0));
        }
        s::storeu(s::not_mask(equal), flags.data());
        for (int i = 0; i < s::size; ++i)
        {
            EXPECT_EQ(flags[i], x[i] != y[i] ? ~bits(0) : bits(0));
        }
    }
    for (int i = 0; i < s::size; ++i)
    {
        flags[i] = i % 2 ? bits(7) : bits(0);
    }
    const auto memory_mask = s::loadu(flags.data());
    s::storeu(memory_mask, flags.data());
    for (int i = 0; i < s::size; ++i)
    {
        EXPECT_EQ(flags[i], i % 2 ? ~bits(0) : bits(0));
    }
    for (int count = 0; count <= s::size; ++count)
    {
        std::array<bits, s::size + 2> partial;
        partial.fill(bits(42));
        s::storeu(s::loadu(flags.data(), count), partial.data() + 1, count);
        EXPECT_EQ(partial.front(), bits(42));
        for (int i = 0; i < count; ++i)
        {
            EXPECT_EQ(partial[i + 1], flags[i]);
        }
        for (int i = count + 1; i < s::size + 2; ++i)
        {
            EXPECT_EQ(partial[i], bits(42));
        }
    }
    s::storeu(s::loadu(static_cast<const bits*>(nullptr), 0), static_cast<bits*>(nullptr), 0);
    // set_mask(int_t), plain integer literals (once ambiguous between the bool
    // and int_t overloads), and other integer types follow the
    // nonzero-selects-all rule.
    const auto expect_all = [&](typename s::mask_t m, bool set)
    {
        s::storeu(m, flags.data());
        for (const auto flag : flags)
        {
            EXPECT_EQ(flag, set ? ~bits(0) : bits(0));
        }
    };
    expect_all(s::set_mask(bits(1)), true);
    expect_all(s::set_mask(false), false);
    expect_all(s::set_mask(1), true);
    expect_all(s::set_mask(0), false);
    expect_all(s::set_mask(std::int64_t(7)), true);
    expect_all(s::set_mask(0u), false);
    s::storeu(s::set(2), out.data());
    for (const auto value : out)
    {
        EXPECT_EQ(value, c(T(2), T(0)));
    }
}

template <class T, int N> void check_transpose()
{
    using c = scalars::complex<T>;
    using s = simd<c>;
    std::array<typename s::simd_t, N>                    rows;
    std::array<c, static_cast<std::size_t>(N) * s::size> input, output;
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = c(T(i + 1), -T(i + 3));
    }
    for (int row = 0; row < N; ++row)
    {
        rows[row] = s::loadu(input.data() + row * s::size);
    }
    s::template ptranspose<N>(rows.data());
    for (int row = 0; row < N; ++row)
    {
        s::storeu(rows[row], output.data() + row * s::size);
    }
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        EXPECT_EQ(output[i], input[(i % N) * s::size + i / N]);
    }
}

// Inputs are small dyadic rationals, so every packet result below is exact.
TYPED_TEST(simd_complex_native, MemoryPackingAndHalfWidth)
{
    using T                    = TypeParam;  // NOLINT(readability-identifier-naming)
    using c                    = scalars::complex<T>;
    using s                    = simd<c>;
    constexpr int            w = s::size;
    const c                  sentinel(T(97), T(-63));
    std::array<c, 2 * w + 8> input, output, expected;
    for (int i = 0; i < 2 * w + 8; ++i)
    {
        input[i] = c(T(i + 1) / 4, T(2 - i) / 8);
    }
    s::prefetch(input.data());
    output.fill(sentinel);
    s::copy(input.data() + 1, output.data() + 1);
    EXPECT_EQ(output.front(), sentinel);
    for (int i = 1; i <= w; ++i)
    {
        EXPECT_EQ(output[i], input[i]);
    }
    EXPECT_EQ(output[w + 1], sentinel);
    typename s::simd_t a, b, c0, d;
    s::broadcast(input.data(), a, b, c0, d);
    const std::array<typename s::simd_t, 4> broadcasts{a, b, c0, d};
    for (int i = 0; i < 4; ++i)
    {
        s::storeu(broadcasts[i], output.data());
        for (int j = 0; j < w; ++j)
        {
            EXPECT_EQ(output[j], input[i]);
        }
    }
    constexpr int repeats = std::is_same_v<T, float> && w == 4 ? 2 : 4;
    s::storeu(s::ploadquad(input.data()), output.data());
    for (int i = 0; i < w; ++i)
    {
        EXPECT_EQ(output[i], input[i / repeats]);
    }
    std::array<int, w> indices;
    for (int i = 0; i < w; ++i)
    {
        indices[i] = w - 1 - i - w / 2;
    }
    a = s::gather(input.data() + w, indices.data());
    s::storeu(a, output.data());
    for (int i = 0; i < w; ++i)
    {
        EXPECT_EQ(output[i], input[w + indices[i]]);
    }
    for (int i = 0; i < w; ++i)
    {
        indices[i] = i % 2;
    }
    output.fill(sentinel);
    expected = output;
    for (int i = 0; i < w; ++i)
    {
        expected[indices[i]] = input[i];
    }
    s::scatter(s::loadu(input.data()), indices.data(), output.data());
    EXPECT_EQ(output, expected);

    // Constant-stride gather/scatter round trip.
    output.fill(sentinel);
    s::scatter(s::gather(input.data() + w, -1), -1, output.data() + w);
    for (int i = 1; i <= w; ++i)
    {
        EXPECT_EQ(output[i], input[i]);
    }

    const auto h = s::loadu_half(input.data());
    const auto k = s::gather_half(input.data() + w, -1);
    output.fill(sentinel);
    s::storeu(s::add(h, h), output.data());
    for (int i = 0; i < s::half_size; ++i)
    {
        EXPECT_EQ(output[i], input[i] + input[i]);
    }
    for (int i = s::half_size; i < 2 * w + 8; ++i)
    {
        EXPECT_EQ(output[i], sentinel);
    }
    output.fill(sentinel);
    s::scatter(s::fma(h, k, h), 2, output.data());
    for (int i = 0; i < s::half_size; ++i)
    {
        const auto x = widen(input[i]), y = widen(input[w - i]);
        expect_ulps<T>("half-width fma", output[2 * i], ref_mul(x, y) + x, kExact, input[i]);
        EXPECT_EQ(output[2 * i + 1], sentinel);
    }
    output.fill(sentinel);
    s::storeu(s::set(input.data()).value, output.data());
    for (int i = 0; i < s::half_size; ++i)
    {
        EXPECT_EQ(output[i], input[0]);
    }
    for (int i = s::half_size; i < w; ++i)
    {
        EXPECT_EQ(output[i], c{});
    }
    for (int i = 0; i < w; ++i)
    {
        indices[i] = 2 * i;
    }
    output.fill(sentinel);
    s::scatter(s::gather_half(input.data(), indices.data()), indices.data(), output.data());
    for (int i = 0; i < s::half_size; ++i)
    {
        EXPECT_EQ(output[indices[i]], input[indices[i]]);
    }
    if constexpr (w >= 4)
    {
        s::storeu(s::predux_downto4(s::loadu(input.data())), output.data());
        for (int i = 0; i < s::half_size; ++i)
        {
            EXPECT_EQ(output[i], input[i] + input[i + s::half_size]);
        }
    }
    check_transpose<T, 1>();
    check_transpose<T, 2>();
    check_transpose<T, 4>();
    check_transpose<T, s::size>();
}

TYPED_TEST(simd_complex_native, PartialLoadsStoresAndAccumulate)
{
    using T                = TypeParam;  // NOLINT(readability-identifier-naming)
    using c                = scalars::complex<T>;
    using s                = simd<c>;
    constexpr int        w = s::size;
    const c              sentinel(T(-9), T(7));
    std::array<c, w + 2> input, output;
    for (int i = 0; i < w + 2; ++i)
    {
        input[i] = c(T(i + 1), T(2 - i));
    }
    for (int count = 0; count <= w; ++count)
    {
        output.fill(sentinel);
        const auto partial = s::loadu(input.data() + 1, count);
        s::storeu(partial, output.data() + 1, count);
        EXPECT_EQ(output[0], sentinel);
        for (int i = 0; i < count; ++i)
        {
            EXPECT_EQ(output[i + 1], input[i + 1]);
        }
        for (int i = count + 1; i < w + 2; ++i)
        {
            EXPECT_EQ(output[i], sentinel);
        }
        // Lanes past `count` load as zero.
        s::storeu(partial, output.data());
        for (int i = count; i < w; ++i)
        {
            EXPECT_EQ(output[i], c{});
        }
    }
    s::storeu(s::loadu(nullptr, 0), nullptr, 0);

    // accumulate widens float to double; these integer sums are exact.
    static_assert(std::is_same_v<typename s::accumulation_t, scalars::complex<double>>);
    double re = 0, im = 0;
    for (int i = 1; i <= w; ++i)
    {
        re += double(input[i].real());
        im += double(input[i].imag());
    }
    const auto sum = s::accumulate(s::loadu(input.data() + 1));
    EXPECT_EQ(sum.real(), re);
    EXPECT_EQ(sum.imag(), im);
    if constexpr (std::is_same_v<T, float> && w >= 3)
    {
        // Summing in float would lose the unit terms to the 1e8 terms.
        std::array<c, w> spread{};
        spread[0]          = c(T(1e8), T(-1e8));
        spread[1]          = c(T(-1e8), T(1e8));
        spread[2]          = c(T(1), T(-1));
        const auto widened = s::accumulate(s::loadu(spread.data()));
        EXPECT_EQ(widened.real(), 1.0);
        EXPECT_EQ(widened.imag(), -1.0);
    }

    // conj flips the sign bit of every imaginary part, including zero and NaN
    // lanes' signed zeros.
    s::storeu(s::conj(s::set(c(T(1), T(0)))), output.data());
    EXPECT_TRUE(std::signbit(output[0].imag()));
    const T nan = std::numeric_limits<T>::quiet_NaN();
    s::storeu(s::conj(s::set(c(nan, T(-0.0)))), output.data());
    EXPECT_TRUE(std::isnan(output[0].real()));
    EXPECT_FALSE(std::signbit(output[0].imag()));
}

// A column-major matrix product accumulated with packet fma over tails. The
// entries are multiples of 1/16 below 100, so every product and sum is exact.
TYPED_TEST(simd_complex_native, MatrixProduct)
{
    using T            = TypeParam;  // NOLINT(readability-identifier-naming)
    using c            = scalars::complex<T>;
    using s            = simd<c>;
    constexpr int rows = 2 * s::size + 1, depth = 7, cols = 5;
    std::array<c, static_cast<std::size_t>(rows) * depth> a;
    std::array<c, static_cast<std::size_t>(depth) * cols> b;
    std::array<c, static_cast<std::size_t>(rows) * cols>  result;
    for (int i = 0; i < rows * depth; ++i)
    {
        a[i] = c(T(i % 5 - 2), T(i % 3) / T(4));
    }
    for (int i = 0; i < depth * cols; ++i)
    {
        b[i] = c(T(i % 7 - 3), T(i % 4) / T(4));
    }
    const auto at = [&](int r, int k) { return a[static_cast<std::size_t>(k) * rows + r]; };
    const auto bt = [&](int k, int col) { return b[static_cast<std::size_t>(col) * depth + k]; };
    for (int col = 0; col < cols; ++col)
    {
        for (int r = 0; r < rows; r += s::size)
        {
            const int count = std::min(s::size, rows - r);
            auto      sum   = s::setzero();
            for (int k = 0; k < depth; ++k)
            {
                sum = s::fma(s::loadu(a.data() + static_cast<std::size_t>(k) * rows + r, count),
                    s::set(bt(k, col)),
                    sum);
            }
            s::storeu(sum, result.data() + static_cast<std::size_t>(col) * rows + r, count);
        }
    }
    for (int col = 0; col < cols; ++col)
    {
        for (int r = 0; r < rows; ++r)
        {
            reference<T> expected{};
            for (int k = 0; k < depth; ++k)
            {
                expected += ref_mul(widen(at(r, k)), widen(bt(k, col)));
            }
            expect_ulps<T>("matrix product",
                result[static_cast<std::size_t>(col) * rows + r],
                expected,
                tol{0, 0, 0});
        }
    }
}

// Edge operands at every lane position, around ordinary lanes. The packet
// quotient must match the scalar library's quotient in class (NaN, infinity,
// finite) and, where finite, to the budget.
TYPED_TEST(simd_complex_native, DivisionExceptionalLanes)
{
    using T                         = TypeParam;  // NOLINT(readability-identifier-naming)
    using c                         = scalars::complex<T>;
    using s                         = simd<c>;
    const T                 maximum = std::numeric_limits<T>::max();
    const T                 tiny    = std::numeric_limits<T>::denorm_min();
    const T                 inf     = std::numeric_limits<T>::infinity();
    const T                 nan     = std::numeric_limits<T>::quiet_NaN();
    const T                 lower   = std::ldexp(T(1), std::is_same_v<T, float> ? -30 : -250);
    const T                 upper   = std::ldexp(T(1), std::is_same_v<T, float> ? 31 : 251);
    const std::array<c, 15> numerators{c(maximum, maximum),
        c(maximum, tiny),
        c(maximum, 0),
        c(tiny, tiny),
        c(1, 2),
        c(0, 0),
        c(inf, 1),
        c(nan, 1),
        c(1, -2),
        c(1, 2),
        c(lower, -lower),
        c(upper, upper),
        c(std::nextafter(lower, T(0)), lower),
        c(std::nextafter(upper, T(0)), upper),
        c(T(-0.0), T(0))};
    const std::array<c, 15> denominators{c(maximum, maximum),
        c(0, tiny),
        c(tiny, 1),
        c(tiny, tiny),
        c(0, 0),
        c(0, 0),
        c(2, 3),
        c(2, 3),
        c(inf, 0),
        c(nan, 1),
        c(upper, upper),
        c(lower, -lower),
        c(lower, lower),
        c(upper, upper),
        c(1, T(-0.0))};
    std::array<c, s::size>  numerator, denominator, output;
    for (std::size_t edge = 0; edge < numerators.size(); ++edge)
    {
        for (int position = 0; position < s::size; ++position)
        {
            numerator.fill(c(1, -2));
            denominator.fill(c(3, 4));
            numerator[position]   = numerators[edge];
            denominator[position] = denominators[edge];
            s::storeu(
                s::div(s::loadu(numerator.data()), s::loadu(denominator.data())), output.data());
            for (int i = 0; i < s::size; ++i)
            {
                expect_ulps<T>("div edge",
                    output[i],
                    widen(numerator[i] / denominator[i]),
                    tol{0.63, 0.5},
                    numerator[i],
                    denominator[i]);
            }
        }
    }
}

// Square roots at and around the bounds of the packet fast path, against the
// scalar path, which normalizes first. Components are compared relative to
// themselves, so small branch-cut parts and the signs of zeros are checked.
TYPED_TEST(simd_complex_native, SquareRootRangeBoundaries)
{
    using T                      = TypeParam;  // NOLINT(readability-identifier-naming)
    using c                      = scalars::complex<T>;
    using s                      = simd<c>;
    const T                lower = scalars::detail_complex_simd::division_lower<T>();
    const T                upper = scalars::detail_complex_simd::division_upper<T>();
    const std::array<T, 9> values{T(0),
        T(-0.0),
        lower,
        std::nextafter(lower, T(0)),
        std::nextafter(lower, upper),
        T(1),
        std::nextafter(upper, T(0)),
        upper,
        std::nextafter(upper, std::numeric_limits<T>::infinity())};
    std::array<c, s::size> input, output;
    // Packet and scalar roots may land on neighbouring values across a binade.
    const tol budget = signed_zeros(tol{0.51, 0.71});
    for (T real : values)
    {
        for (T imag : values)
        {
            for (T sign : {T(1), T(-1)})
            {
                for (int lane = 0; lane < s::size; ++lane)
                {
                    input[lane] = c(sign * real, lane % 2 ? -imag : imag);
                }
                s::storeu(s::sqrt(s::loadu(input.data())), output.data());
                for (int lane = 0; lane < s::size; ++lane)
                {
                    const auto expected = scalars::sqrt(input[lane]);
                    expect_ulps<T>("sqrt bound real",
                        output[lane].real(),
                        widen_component(expected.real()),
                        budget,
                        input[lane]);
                    expect_ulps<T>("sqrt bound imag",
                        output[lane].imag(),
                        widen_component(expected.imag()),
                        budget,
                        input[lane]);
                }
            }
        }
    }
}

TYPED_TEST(simd_complex_native, MagnitudeAndSpecialValues)
{
    using T          = TypeParam;  // NOLINT(readability-identifier-naming)
    using c          = scalars::complex<T>;
    using s          = simd<c>;
    using R          = ref_t<T>;  // NOLINT(readability-identifier-naming)
    using bits       = typename s::int_t;
    const T infinity = std::numeric_limits<T>::infinity(),
            nan      = std::numeric_limits<T>::quiet_NaN();
    const T maximum = std::numeric_limits<T>::max(), tiny = std::numeric_limits<T>::denorm_min();
    const std::array<c, 16>   edges{c(3, 4),
        c(maximum, maximum),
        c(tiny, -tiny),
        c(0, T(-0.0)),
        c(T(-0.0), T(-0.0)),
        c(infinity, 0),
        c(-infinity, T(-0.0)),
        c(1, infinity),
        c(nan, 1),
        c(infinity, nan),
        c(T(-4), T(-0.0)),
        c(T(-4), T(0)),
        c(-maximum, -maximum),
        c(T(0x1p31) + T(.5), -T(0x1p31) - T(.5)),
        c(T(0x1p23) - T(.5), -T(0x1p23) + T(.5)),
        c(T(.75), T(-.75))};
    const tol                 magnitude{0.5, 0.95};
    const tol                 angle     = signed_zeros(tol{0.5, 0.5});
    const tol                 logarithm = signed_zeros(tol{0.5, 0.5});
    const tol                 root      = signed_zeros(tol{0.5, 0.92});
    const tol                 identity{1, 3};
    std::array<c, s::size>    input, output;
    std::array<T, s::size>    real;
    std::array<bits, s::size> flags;
    for (const auto edge : edges)
    {
        const auto z = widen(edge);
        input.fill(edge);
        const auto a = s::loadu(input.data());
        s::storeu(s::abs(a), real.data());
        for (const T value : real)
        {
            expect_ulps<T>("special abs", value, std::hypot(z.real(), z.imag()), magnitude, edge);
        }
        s::storeu(s::fabs(a), real.data());
        for (const T value : real)
        {
            expect_ulps<T>("special fabs", value, std::hypot(z.real(), z.imag()), magnitude, edge);
        }
        s::storeu(s::arg(a), real.data());
        for (const T value : real)
        {
            expect_ulps<T>("special arg", value, std::atan2(z.imag(), z.real()), angle, edge);
        }
        s::storeu(s::abs2(a), real.data());
        for (const T value : real)
        {
            // Overflow to infinity in T is part of the contract.
            expect_ulps<T>("special abs2", value, ref_abs2(z), magnitude, edge);
        }
        s::storeu(s::real(a), real.data());
        for (const T value : real)
        {
            EXPECT_EQ(std::bit_cast<bits>(value), std::bit_cast<bits>(edge.real()));
        }
        s::storeu(s::imag(a), real.data());
        for (const T value : real)
        {
            EXPECT_EQ(std::bit_cast<bits>(value), std::bit_cast<bits>(edge.imag()));
        }
        s::storeu(s::isfinite(a), flags.data());
        for (const auto flag : flags)
        {
            EXPECT_EQ(flag, scalars::isfinite(edge) ? ~bits(0) : bits(0));
        }
        s::storeu(s::isnan(a), flags.data());
        for (const auto flag : flags)
        {
            EXPECT_EQ(
                flag, std::isnan(edge.real()) || std::isnan(edge.imag()) ? ~bits(0) : bits(0));
        }
        s::storeu(s::isinf(a), flags.data());
        for (const auto flag : flags)
        {
            EXPECT_EQ(
                flag, std::isinf(edge.real()) || std::isinf(edge.imag()) ? ~bits(0) : bits(0));
        }
        s::storeu(s::sqrt(a), output.data());
        for (const auto value : output)
        {
            if (std::isfinite(edge.real()) && std::isfinite(edge.imag()) &&
                (std::fabs(edge.real()) > maximum / T(4) ||
                    std::fabs(edge.imag()) > maximum / T(4)))
            {
                // Verify the square-root identity after scaling. Some standard
                // libraries overflow their own reference sqrt(max + max*i).
                EXPECT_TRUE(std::isfinite(value.real()));
                EXPECT_TRUE(std::isfinite(value.imag()));
                const T               scale = std::sqrt(maximum);
                const std::complex<T> normalized(value.real() / scale, value.imag() / scale);
                expect_ulps<T>("special sqrt identity",
                    c(normalized * normalized),
                    reference<T>(R(edge.real()) / R(maximum), R(edge.imag()) / R(maximum)),
                    identity,
                    edge);
                EXPECT_GE(value.real(), T(0));
                EXPECT_EQ(std::signbit(value.imag()), std::signbit(edge.imag()));
            }
            else
            {
                expect_ulps<T>("special sqrt", value, ref_sqrt(z), root, edge);
            }
        }
        s::storeu(s::log(a), output.data());
        for (const auto value : output)
        {
            expect_ulps<T>("special log", value, log_reference<T>(edge), logarithm, edge);
        }
        const std::array<typename s::simd_t, 3> rounded{s::ceil(a), s::floor(a), s::trunc(a)};
        const std::array<reference<T>, 3>       references{
            componentwise(z, [](R v) { return std::ceil(v); }),
            componentwise(z, [](R v) { return std::floor(v); }),
            componentwise(z, [](R v) { return std::trunc(v); })};
        for (std::size_t operation = 0; operation < rounded.size(); ++operation)
        {
            s::storeu(rounded[operation], output.data());
            for (const auto value : output)
            {
                expect_ulps<T>(
                    "special ceil/floor/trunc", value, references[operation], kExact, edge);
            }
        }
        s::storeu(s::neg(a), output.data());
        for (const auto value : output)
        {
            expect_bits(value, c(-edge.real(), -edge.imag()));
        }
    }
    // Real-axis expm1 and log1p keep the signed-zero imaginary part.
    for (T x : {T(0), T(-0.0), T(1e-8), T(-1e-8), tiny, infinity, -infinity})
    {
        const auto a = s::set(c(x, T(-0.0)));
        s::storeu(s::expm1(a), output.data());
        for (const auto value : output)
        {
            expect_ulps<T>("special expm1(real)",
                value,
                reference<T>(std::expm1(R(x)), R(-0.0)),
                signed_zeros(tol{0.5, 0.5}),
                x);
        }
        s::storeu(s::log1p(a), output.data());
        if (x >= T(-1))
        {
            for (const auto value : output)
            {
                expect_ulps<T>("special log1p(real)",
                    value,
                    reference<T>(ref_log1p_real(R(x)), R(-0.0)),
                    signed_zeros(tol{0.5, 0.75}),
                    x);
            }
        }
    }
}

// Each edge value in every lane position, next to ordinary lanes, so a lane
// that leaves the vectorized fast path cannot disturb its neighbours.
TYPED_TEST(simd_complex_native, MixedLaneEdgesAndBranchCuts)
{
    using T                         = TypeParam;  // NOLINT(readability-identifier-naming)
    using c                         = scalars::complex<T>;
    using s                         = simd<c>;
    const T                 inf     = std::numeric_limits<T>::infinity();
    const T                 nan     = std::numeric_limits<T>::quiet_NaN();
    const T                 tiny    = std::numeric_limits<T>::denorm_min();
    const T                 maximum = std::numeric_limits<T>::max();
    const T                 limit   = std::is_same_v<T, float> ? T(80) : T(700);
    const std::array<c, 19> edges{c(0, -0.0),
        c(-4, -0.0),
        c(-4, 0),
        c(tiny, -tiny),
        c(maximum, maximum),
        c(-maximum, T(-0.0)),
        c(inf, 0),
        c(-inf, -0.0),
        c(nan, 1),
        c(inf, nan),
        c(1, inf),
        c(T(.5), tiny),
        c(limit, T(.25)),
        c(-limit, T(-.25)),
        c(std::nextafter(limit, T(0)), T(-.5)),
        c(T(-0.0), T(-0.0)),
        c(T(1), T(0)),
        c(T(0), T(1)),
        c(T(-1), tiny)};
    const tol               magnitude   = signed_zeros(tol{0.5, 0.5});
    const tol               exponential = signed_zeros(tol{1, 1.4});
    const tol               logarithm   = signed_zeros(tol{0.5, 0.5});
    std::array<c, s::size>  input, output;
    std::array<T, s::size>  real;
    for (const auto edge : edges)
    {
        for (int lane = 0; lane < s::size; ++lane)
        {
            input.fill(c(T(.25), T(-.5)));
            input[lane]       = edge;
            const auto packet = s::loadu(input.data());
            s::storeu(s::abs(packet), real.data());
            for (int i = 0; i < s::size; ++i)
            {
                const auto z = widen(input[i]);
                expect_ulps<T>(
                    "mixed-lane abs", real[i], std::hypot(z.real(), z.imag()), magnitude, input[i]);
            }
            s::storeu(s::exp(packet), output.data());
            for (int i = 0; i < s::size; ++i)
            {
                expect_ulps<T>(
                    "mixed-lane exp", output[i], std::exp(widen(input[i])), exponential, input[i]);
            }
            s::storeu(s::log(packet), output.data());
            for (int i = 0; i < s::size; ++i)
            {
                expect_ulps<T>(
                    "mixed-lane log", output[i], log_reference<T>(input[i]), logarithm, input[i]);
            }
        }
    }
}

#if SCALARS_HAS_SVML || SCALARS_HAS_MKL || SCALARS_HAS_SLEEF || SCALARS_HAS_ACCELERATE
// Re(log z) must be accurate relative to itself, not just to |log z|: near
// |z| = 1 it is tiny while Im(log z) = arg z is not, so the normwise budget
// cannot see errors in it (plan items 4.1 and 6.2).
TYPED_TEST(simd_complex_native, LogRealPartNearUnitCircle)
{
    using T = TypeParam;  // NOLINT(readability-identifier-naming)
    using c = scalars::complex<T>;
    using s = simd<c>;
    std::vector<c> inputs;
    // The plan's 4.1 example: |z|^2 - 1 is about 4e-8.
    inputs.emplace_back(T(0.9323273301), T(0.3616154194));
    const int deepest = std::is_same_v<T, float> ? 22 : 50;
    for (int k = 3; k <= deepest; ++k)
    {
        for (const double sign : {-1.0, 1.0})
        {
            const double radius = 1.0 + sign * std::ldexp(1.0, -k);
            for (const double angle : {0.05, 0.4, 0.785, 1.2, 2.5, -2.0})
            {
                inputs.emplace_back(T(radius * std::cos(angle)), T(radius * std::sin(angle)));
            }
        }
    }
    while (inputs.size() % s::size != 0)
    {
        inputs.push_back(inputs.front());
    }
    std::vector<c> outputs(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); i += s::size)
    {
        s::storeu(s::log(s::loadu(inputs.data() + i)), outputs.data() + i);
    }
    accuracy_check<T> check("log.real near |z| = 1", tol{1.2, 1.1});
    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        const auto expected = ref_log(widen(inputs[i])).real();
        check.check(c(outputs[i].real(), T(0)), reference<T>(expected, 0), inputs[i]);
    }
    check.report();
}
#endif
}  // namespace

#endif  // SCALARS_VECTORIZED
