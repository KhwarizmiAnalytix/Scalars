// SIMD tests: the vectorized port of Eigen's packet math
// (backend/cpu/detail/vector_math.h) on the selected CPU tier, called directly,
// so that it is covered whichever vector math library (SVML, SLEEF,
// Accelerate) the simd<T> packets forward to.
//
// Accuracy sweeps record the worst error, in units in the last place of the
// result, against libm evaluated in double; results below the smallest normal
// number are measured in units of the subnormal spacing. Budgets are the
// measured maxima on SSE2 (no FMA) and AVX2 (FMA) rounded up, plus one ulp for
// the double reference's own error where the reference is double libm.
// Special-value tables check the IEEE results of std:: (NaN, infinities,
// signed zeros, domain edges), mixed with ordinary values in each packet so
// that an exceptional lane cannot disturb its neighbours.
#include "backend/simd.h"

#if SCALARS_VECTORIZED &&                                                                          \
    (SCALARS_HAS_SSE || SCALARS_HAS_AVX || SCALARS_HAS_AVX2 || SCALARS_HAS_AVX512 ||               \
     SCALARS_HAS_NEON || SCALARS_HAS_SVE)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "backend/cpu/detail/vector_math.h"
#if SCALARS_HAS_NEON || SCALARS_HAS_SVE
#include "backend/cpu/neon/vector_math_ops.h"
#else
#include "backend/cpu/x86/vector_math_ops.h"
#endif
#include "math/normal_cdf.h"
#include "scalars_test.h"

namespace
{
namespace vmath = scalars::detail::vmath;

template <class T> struct ops_for;
template <> struct ops_for<float>
{
    using type = vmath::ops_f32;
};
template <> struct ops_for<double>
{
    using type = vmath::ops_f64;
};

template <class T> using ops_t = typename ops_for<T>::type;
template <class T> using pm    = vmath::packet_math<ops_t<T>>;
template <class T> using vec_t = typename ops_t<T>::vec_t;
template <class T> constexpr int kLanes = ops_t<T>::size;

constexpr int    kSamples   = 1 << 14;
constexpr double kInf       = std::numeric_limits<double>::infinity();
constexpr double kSqrtHalf  = 0.70710678118654752440;

// std::cbrt refined by one Newton step whose residual r^3 - x is formed exactly with fma:
// glibc's cbrt is off by up to 3 ulp, more than the port.
double accurate_cbrt(double x)
{
    const double r = std::cbrt(x);
    // Below 1e-280, r^3 would lose bits to underflow.
    if (!std::isfinite(r) || std::fabs(x) < 1e-280)
    {
        return r;
    }
    const double r2    = r * r;
    const double r2_lo = std::fma(r, r, -r2);
    const double r3    = r2 * r;
    const double r3_lo = std::fma(r2, r, -r3) + r2_lo * r;
    return r - ((r3 - x) + r3_lo) / (3 * r2);
}

template <class T> const char* type_name() { return std::is_same_v<T, float> ? "float" : "double"; }

// Error of actual in ulps of T at expected, measured relative to max(|expected|, floor).
template <class T> double ulp_error(T actual, double expected, double floor)
{
    if (std::isnan(expected))
    {
        return std::isnan(actual) ? 0 : kInf;
    }
    const double rounded = static_cast<double>(static_cast<T>(expected));
    if (std::isinf(rounded))
    {
        return static_cast<double>(actual) == rounded ? 0 : kInf;
    }
    if (!std::isfinite(actual))
    {
        return kInf;
    }
    const double scale = std::max(std::fabs(expected), floor);
    const int    e     = scale == 0 ? std::numeric_limits<T>::min_exponent - 1
                                    : std::max(std::ilogb(scale), std::numeric_limits<T>::min_exponent - 1);
    const double ulp   = std::ldexp(1.0, e - (std::numeric_limits<T>::digits - 1));
    return std::fabs(static_cast<double>(actual) - expected) / ulp;
}

// Applies a packet function to a flat array, one packet at a time.
template <class T, class F> std::vector<T> apply(const std::vector<T>& x, F f)
{
    using O = ops_t<T>;
    std::vector<T> out(x.size());
    for (std::size_t i = 0; i + kLanes<T> <= x.size(); i += kLanes<T>)
    {
        O::storeu(out.data() + i, f(O::loadu(x.data() + i)));
    }
    return out;
}

template <class T, class F>
std::vector<T> apply(const std::vector<T>& x, const std::vector<T>& y, F f)
{
    using O = ops_t<T>;
    std::vector<T> out(x.size());
    for (std::size_t i = 0; i + kLanes<T> <= x.size(); i += kLanes<T>)
    {
        O::storeu(out.data() + i, f(O::loadu(x.data() + i), O::loadu(y.data() + i)));
    }
    return out;
}

// Samples in [lo, hi]; log-uniform domains sample magnitudes in [lo, hi], negated with
// probability one half when `symmetric`.
struct domain
{
    double lo;
    double hi;
    bool   log_uniform = false;
    bool   symmetric   = false;
};

template <class T> std::vector<T> sample(const domain& d, unsigned seed)
{
    std::mt19937                           gen(seed);
    std::uniform_real_distribution<double> uni(d.lo, d.hi);
    // Linear domains may have lo <= 0; only take logs for log-uniform ones.
    std::uniform_real_distribution<double> mag(
        d.log_uniform ? std::log(d.lo) : 0.0, d.log_uniform ? std::log(d.hi) : 1.0);
    std::bernoulli_distribution            neg(0.5);
    std::vector<T>                         x(kSamples);
    for (auto& v : x)
    {
        double s = d.log_uniform ? std::exp(mag(gen)) : uni(gen);
        if (d.symmetric && neg(gen))
        {
            s = -s;
        }
        v = static_cast<T>(s);
    }
    return x;
}

// Tracks the worst lane of one sweep and reports it with its inputs.
class worst_case
{
public:
    worst_case(std::string name, double budget) : name_(std::move(name)), budget_(budget) {}

    template <class... Args> void record(double ulps, double actual, double expected, Args... args)
    {
        if (!(ulps > worst_) && !(std::isinf(ulps) && !std::isinf(worst_)))
        {
            return;
        }
        worst_ = ulps;
        std::ostringstream os;
        os << std::setprecision(17) << "\n  inputs   =";
        ((os << ' ' << args), ...);
        os << "\n  actual   = " << actual << "\n  expected = " << expected;
        detail_ = os.str();
    }

    void report() const
    {
        EXPECT_LE(worst_, budget_) << name_ << ": worst error " << worst_ << " ulp (budget "
                                   << budget_ << ")" << detail_;
        // SCALARS_VMATH_REPORT=1 prints every measured maximum, to recalibrate the budgets.
        if (std::getenv("SCALARS_VMATH_REPORT") != nullptr)
        {
            std::printf("%-24s %8.3f ulp (budget %g)\n", name_.c_str(), worst_, budget_);
        }
    }

private:
    std::string name_;
    double      budget_;
    double      worst_ = 0;
    std::string detail_;
};

template <class T, class F, class R>
void sweep(const char* name, const domain& d, double budget, F f, R ref, double floor = 0)
{
    worst_case     w(std::string(type_name<T>()) + "::" + name, budget);
    const auto     x   = sample<T>(d, 12345u);
    const auto     out = apply<T>(x, f);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const double xi = static_cast<double>(x[i]);
        const double e  = ref(xi);
        w.record(ulp_error<T>(out[i], e, floor), out[i], e, xi);
    }
    w.report();
}

template <class T, class F, class R>
void sweep2(const char* name, const domain& dx, const domain& dy, double budget, F f, R ref)
{
    worst_case     w(std::string(type_name<T>()) + "::" + name, budget);
    const auto     x   = sample<T>(dx, 12345u);
    const auto     y   = sample<T>(dy, 54321u);
    const auto     out = apply<T>(x, y, f);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const double xi = static_cast<double>(x[i]);
        const double yi = static_cast<double>(y[i]);
        const double e  = ref(xi, yi);
        w.record(ulp_error<T>(out[i], e, 0), out[i], e, xi, yi);
    }
    w.report();
}

// Per-type budgets in ulps: {float, double}.
struct budgets
{
    double f;
    double d;
};

template <class T> double pick(budgets b) { return std::is_same_v<T, float> ? b.f : b.d; }

#define VMATH_SWEEP(fn, dom, bud, ref)                                                             \
    sweep<T>(#fn, dom, pick<T>(bud), [](vec_t<T> v) { return P::fn(v); }, [](double x) { return ref; })

template <class T> void run_simd_vector_math_ExponentialsAndLogarithms()
{
    using P = pm<T>;  // NOLINT(readability-identifier-naming)
    constexpr bool kF = std::is_same_v<T, float>;
    const double   max_log = kF ? 88.7 : 709.7;
    // Down to where exp and exp2 are subnormal, and below where they round to zero.
    const domain exp_dom{kF ? -104.0 : -745.0, max_log};
    const domain exp2_dom{kF ? -150.0 : -1075.0, kF ? 128.0 : 1024.0};
    const domain exp10_dom{kF ? -45.0 : -323.0, kF ? 38.5 : 308.2};
    const domain all_pos{static_cast<double>(std::numeric_limits<T>::denorm_min()),
                         static_cast<double>(std::numeric_limits<T>::max()),
                         true};

    VMATH_SWEEP(exp, exp_dom, (budgets{2, 2}), std::exp(x));
    VMATH_SWEEP(exp, (domain{-1, 1}), (budgets{2, 2}), std::exp(x));
    VMATH_SWEEP(expm1, (domain{-30, max_log}), (budgets{2, 2}), std::expm1(x));
    VMATH_SWEEP(expm1, (domain{1e-30, 0.35, true, true}), (budgets{2, 2}), std::expm1(x));
    VMATH_SWEEP(exp2, exp2_dom, (budgets{2, 2}), std::exp2(x));
    VMATH_SWEEP(exp10, exp10_dom, (budgets{2, 2}), std::pow(10.0, x));
    VMATH_SWEEP(log, all_pos, (budgets{2, 2}), std::log(x));
    VMATH_SWEEP(log, (domain{0.5, 2}), (budgets{2, 2}), std::log(x));
    VMATH_SWEEP(log2, all_pos, (budgets{2, 2}), std::log2(x));
    VMATH_SWEEP(log2, (domain{0.5, 2}), (budgets{3, 2}), std::log2(x));
    VMATH_SWEEP(log10, all_pos, (budgets{3, 2}), std::log10(x));
    VMATH_SWEEP(log1p, (domain{-0.999, 1e30}), (budgets{2, 2}), std::log1p(x));
    VMATH_SWEEP(log1p, (domain{1e-30, 0.5, true, true}), (budgets{2, 2}), std::log1p(x));
}

template <class T> void run_simd_vector_math_Trigonometric()
{
    using P = pm<T>;  // NOLINT(readability-identifier-naming)
    constexpr bool kF = std::is_same_v<T, float>;
    const domain   small{-25, 25};
    const domain   medium{1, kF ? 1e4 : 1e13, true, true};
    // Payne-Hanek (float) and the libm fallback (double) beyond the Cody-Waite range.
    const domain huge{kF ? 1e5 : 1e14, 1e30, true, true};

    VMATH_SWEEP(sin, small, (budgets{2, 2}), std::sin(x));
    VMATH_SWEEP(cos, small, (budgets{2, 2}), std::cos(x));
    VMATH_SWEEP(tan, small, (budgets{4, 3}), std::tan(x));
    VMATH_SWEEP(sin, medium, (budgets{2, 2}), std::sin(x));
    VMATH_SWEEP(cos, medium, (budgets{2, 2}), std::cos(x));
    VMATH_SWEEP(sin, huge, (budgets{2, 1}), std::sin(x));
    VMATH_SWEEP(cos, huge, (budgets{2, 1}), std::cos(x));
    VMATH_SWEEP(asin, (domain{-1, 1}), (budgets{3, 3}), std::asin(x));
    VMATH_SWEEP(acos, (domain{-1, 1}), (budgets{2, 3}), std::acos(x));
    VMATH_SWEEP(atan, (domain{1e-30, 1e30, true, true}), (budgets{4, 2}), std::atan(x));
    VMATH_SWEEP(atan, (domain{-2, 2}), (budgets{4, 2}), std::atan(x));

    // sincos agrees with sin and cos bit for bit.
    const auto x = sample<T>(medium, 777u);
    const auto s = apply<T>(x, [](vec_t<T> v) { return P::sin(v); });
    const auto c = apply<T>(x, [](vec_t<T> v) { return P::cos(v); });
    for (std::size_t i = 0; i + kLanes<T> <= x.size(); i += kLanes<T>)
    {
        vec_t<T> vs, vc;
        P::sincos(ops_t<T>::loadu(x.data() + i), vs, vc);
        T bs[kLanes<T>];
        T bc[kLanes<T>];
        ops_t<T>::storeu(bs, vs);
        ops_t<T>::storeu(bc, vc);
        for (int k = 0; k < kLanes<T>; ++k)
        {
            ASSERT_EQ(bs[k], s[i + k]) << x[i + k];
            ASSERT_EQ(bc[k], c[i + k]) << x[i + k];
        }
    }

    const auto atan2_ref = [](double y, double xx) { return std::atan2(y, xx); };
    const auto atan2_fn  = [](vec_t<T> y, vec_t<T> xx) { return P::atan2(y, xx); };
    sweep2<T>("atan2", domain{-10, 10}, domain{-10, 10}, pick<T>({4, 3}), atan2_fn, atan2_ref);
    sweep2<T>("atan2", domain{1e-20, 1e20, true, true}, domain{1e-20, 1e20, true, true},
              pick<T>({4, 3}), atan2_fn, atan2_ref);
}

template <class T> void run_simd_vector_math_Hyperbolic()
{
    using P = pm<T>;  // NOLINT(readability-identifier-naming)
    constexpr bool kF = std::is_same_v<T, float>;
    const double   max_arg = kF ? 89.4 : 710.4;
    const double   big     = static_cast<double>(std::numeric_limits<T>::max());

    VMATH_SWEEP(sinh, (domain{-max_arg, max_arg}), (budgets{3, 3}), std::sinh(x));
    VMATH_SWEEP(sinh, (domain{1e-30, 1, true, true}), (budgets{2, 2}), std::sinh(x));
    VMATH_SWEEP(cosh, (domain{-max_arg, max_arg}), (budgets{3, 3}), std::cosh(x));
    VMATH_SWEEP(tanh, (domain{-20, 20}), (budgets{2, 3}), std::tanh(x));
    VMATH_SWEEP(tanh, (domain{1e-30, 1, true, true}), (budgets{2, 3}), std::tanh(x));
    VMATH_SWEEP(asinh, (domain{1e-30, big, true, true}), (budgets{2, 2}), std::asinh(x));
    VMATH_SWEEP(acosh, (domain{1, big, true}), (budgets{3, 3}), std::acosh(x));
    VMATH_SWEEP(acosh, (domain{1, 2}), (budgets{3, 3}), std::acosh(x));
    VMATH_SWEEP(atanh, (domain{-1, 1}), (budgets{2, 2}), std::atanh(x));
    VMATH_SWEEP(atanh, (domain{1e-30, 0.5, true, true}), (budgets{2, 2}), std::atanh(x));
}

template <class T> void run_simd_vector_math_PowersAndRoots()
{
    using P = pm<T>;  // NOLINT(readability-identifier-naming)
    constexpr bool kF = std::is_same_v<T, float>;
    const domain   all{static_cast<double>(std::numeric_limits<T>::denorm_min()),
                     static_cast<double>(std::numeric_limits<T>::max()),
                     true,
                     true};

    VMATH_SWEEP(cbrt, all, (budgets{1, 3}), accurate_cbrt(x));
    VMATH_SWEEP(rsqrt, (domain{all.lo, all.hi, true}), (budgets{2, 2}), 1 / std::sqrt(x));

    const auto pow_fn  = [](vec_t<T> x, vec_t<T> y) { return P::pow(x, y); };
    const auto pow_ref = [](double x, double y) { return std::pow(x, y); };
    sweep2<T>("pow", domain{0.05, 20}, domain{-20, 20}, pick<T>({2, 2}), pow_fn, pow_ref);
    // Results that overflow, underflow into the subnormals, or stay near 1.
    sweep2<T>("pow", domain{1e-3, 1e3, true}, domain{kF ? -40.0 : -330.0, kF ? 40.0 : 330.0},
              pick<T>({2, 2}), pow_fn, pow_ref);
    sweep2<T>("pow", domain{0.999, 1.001}, domain{-1e6, 1e6}, pick<T>({2, 2}), pow_fn, pow_ref);

    const auto hypot_fn  = [](vec_t<T> x, vec_t<T> y) { return P::hypot(x, y); };
    const auto hypot_ref = [](double x, double y) { return std::hypot(x, y); };
    sweep2<T>("hypot", all, all, pick<T>({2, 2}), hypot_fn, hypot_ref);
    sweep2<T>("hypot", domain{-10, 10}, domain{-10, 10}, pick<T>({2, 2}), hypot_fn, hypot_ref);
}

template <class T> void run_simd_vector_math_NormalDistribution()
{
    using P = pm<T>;  // NOLINT(readability-identifier-naming)
    constexpr bool kF = std::is_same_v<T, float>;
    // erfc(-x / sqrt(2)) / 2 in double is a float-accurate reference. libm has no correctly
    // rounded double cdf, so double results are held to an absolute error near 1.
    if constexpr (kF)
    {
        VMATH_SWEEP(erfc, (domain{-11, 11}), (budgets{7, 0}), std::erfc(x));
        VMATH_SWEEP(normal_cdf, (domain{-14, 6}), (budgets{6, 0}), 0.5 * std::erfc(-x * kSqrtHalf));
        VMATH_SWEEP(inv_normal_cdf, (domain{1e-30, 1, true}), (budgets{4, 0}), scalars::inv_normalcdf(x));
    }
    else
    {
        VMATH_SWEEP(erfc, (domain{-6, 27}), (budgets{0, 8}), std::erfc(x));
        sweep<T>("normal_cdf", domain{-38, 9}, 4, [](vec_t<T> v) { return P::normal_cdf(v); },
                 [](double x) { return 0.5 * std::erfc(-x * kSqrtHalf); }, 1.0);
        sweep<T>("inv_normal_cdf", domain{1e-300, 1, true}, 8,
                 [](vec_t<T> v) { return P::inv_normal_cdf(v); },
                 [](double x) { return scalars::inv_normalcdf(x); }, 1.0);
    }
}

#undef VMATH_SWEEP

// Values every unary function is checked at, mixed with ordinary ones.
template <class T> std::vector<T> special_inputs()
{
    using L = std::numeric_limits<T>;
    const T v[] = {L::quiet_NaN(), -L::quiet_NaN(), L::infinity(), -L::infinity(), T(0), -T(0),
                   L::denorm_min(), -L::denorm_min(), L::min(), -L::min(), L::max(), -L::max(),
                   T(1), T(-1), T(0.5), T(-0.5), T(2), T(-2), L::epsilon(), T(0.75), T(-3), T(10)};
    std::vector<T> x(std::begin(v), std::end(v));
    while (x.size() % kLanes<T> != 0)
    {
        x.push_back(T(0.25));
    }
    return x;
}

// NaN where std:: returns NaN; infinities and zeros exactly, including the sign of zero;
// finite results within `budget` ulps.
template <class T>
void expect_ieee(const char* name, T actual, T expected, double budget, double x, double y = 0)
{
    std::ostringstream where;
    where << std::setprecision(17) << type_name<T>() << "::" << name << '(' << x;
    if (y != 0 || std::signbit(y))
    {
        where << ", " << y;
    }
    where << ")";
    if (std::isnan(expected))
    {
        EXPECT_TRUE(std::isnan(actual)) << where.str() << " = " << actual << ", expected NaN";
    }
    else if (std::isinf(expected) || expected == 0)
    {
        EXPECT_TRUE(actual == expected && std::signbit(actual) == std::signbit(expected))
            << where.str() << " = " << actual << ", expected " << expected;
    }
    else
    {
        EXPECT_LE(ulp_error<T>(actual, static_cast<double>(expected), 0), budget)
            << where.str() << " = " << actual << ", expected " << expected;
    }
}

template <class T, class F, class R> void special_unary(const char* name, F f, R ref)
{
    const auto x   = special_inputs<T>();
    const auto out = apply<T>(x, f);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        expect_ieee<T>(name, out[i], static_cast<T>(ref(x[i])), 4, static_cast<double>(x[i]));
    }
}

template <class T, class F, class R> void special_binary(const char* name, F f, R ref)
{
    const auto     s = special_inputs<T>();
    std::vector<T> x;
    std::vector<T> y;
    for (const T a : s)
    {
        for (const T b : s)
        {
            x.push_back(a);
            y.push_back(b);
        }
    }
    const auto out = apply<T>(x, y, f);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        expect_ieee<T>(name,
                       out[i],
                       static_cast<T>(ref(x[i], y[i])),
                       4,
                       static_cast<double>(x[i]),
                       static_cast<double>(y[i]));
    }
}

template <class T> void run_simd_vector_math_SpecialValues()
{
    using P = pm<T>;  // NOLINT(readability-identifier-naming)

#define VMATH_SPECIAL(fn, ref)                                                                     \
    special_unary<T>(#fn, [](vec_t<T> v) { return P::fn(v); }, [](T x) { return ref; })

    VMATH_SPECIAL(exp, std::exp(x));
    VMATH_SPECIAL(expm1, std::expm1(x));
    VMATH_SPECIAL(exp2, std::exp2(x));
    VMATH_SPECIAL(exp10, std::pow(T(10), x));
    VMATH_SPECIAL(log, std::log(x));
    VMATH_SPECIAL(log1p, std::log1p(x));
    VMATH_SPECIAL(log2, std::log2(x));
    VMATH_SPECIAL(log10, std::log10(x));
    VMATH_SPECIAL(sin, std::sin(x));
    VMATH_SPECIAL(cos, std::cos(x));
    VMATH_SPECIAL(tan, std::tan(x));
    VMATH_SPECIAL(asin, std::asin(x));
    VMATH_SPECIAL(acos, std::acos(x));
    VMATH_SPECIAL(atan, std::atan(x));
    VMATH_SPECIAL(sinh, std::sinh(x));
    VMATH_SPECIAL(cosh, std::cosh(x));
    VMATH_SPECIAL(tanh, std::tanh(x));
    VMATH_SPECIAL(asinh, std::asinh(x));
    VMATH_SPECIAL(acosh, std::acosh(x));
    VMATH_SPECIAL(atanh, std::atanh(x));
    VMATH_SPECIAL(cbrt, std::cbrt(x));
    VMATH_SPECIAL(erfc, std::erfc(x));
#undef VMATH_SPECIAL

    special_binary<T>("pow", [](vec_t<T> x, vec_t<T> y) { return P::pow(x, y); },
                      [](T x, T y) { return std::pow(x, y); });
    special_binary<T>("hypot", [](vec_t<T> x, vec_t<T> y) { return P::hypot(x, y); },
                      [](T x, T y) { return std::hypot(x, y); });
    special_binary<T>("atan2", [](vec_t<T> y, vec_t<T> x) { return P::atan2(y, x); },
                      [](T y, T x) { return std::atan2(y, x); });

    // The normal distribution at its limits.
    using L = std::numeric_limits<T>;
    const std::vector<T> cdf_x = [] {
        std::vector<T> v = {L::quiet_NaN(), L::infinity(), -L::infinity(), T(0), -T(0), L::max(), -L::max()};
        while (v.size() % kLanes<T> != 0)
        {
            v.push_back(T(1));
        }
        return v;
    }();
    const auto cdf = apply<T>(cdf_x, [](vec_t<T> v) { return P::normal_cdf(v); });
    EXPECT_TRUE(std::isnan(cdf[0]));
    EXPECT_EQ(cdf[1], T(1));
    EXPECT_EQ(cdf[2], T(0));
    EXPECT_EQ(cdf[3], T(0.5));
    EXPECT_EQ(cdf[4], T(0.5));
    EXPECT_EQ(cdf[5], T(1));
    EXPECT_EQ(cdf[6], T(0));

    const std::vector<T> inv_x = [] {
        std::vector<T> v = {L::quiet_NaN(), T(0), T(1), T(-0.5), T(1.5), T(0.5), -T(0), L::infinity()};
        while (v.size() % kLanes<T> != 0)
        {
            v.push_back(T(0.25));
        }
        return v;
    }();
    const auto inv = apply<T>(inv_x, [](vec_t<T> v) { return P::inv_normal_cdf(v); });
    EXPECT_TRUE(std::isnan(inv[0]));
    EXPECT_EQ(inv[1], -L::infinity());
    EXPECT_EQ(inv[2], L::infinity());
    EXPECT_TRUE(std::isnan(inv[3]));
    EXPECT_TRUE(std::isnan(inv[4]));
    EXPECT_EQ(inv[5], T(0));
    EXPECT_EQ(inv[6], -L::infinity());
    EXPECT_TRUE(std::isnan(inv[7]));
}

// clang-format off
TEST(simd_vector_math, ExponentialsAndLogarithms_float)  { run_simd_vector_math_ExponentialsAndLogarithms<float>(); }
TEST(simd_vector_math, ExponentialsAndLogarithms_double) { run_simd_vector_math_ExponentialsAndLogarithms<double>(); }
TEST(simd_vector_math, Trigonometric_float)              { run_simd_vector_math_Trigonometric<float>(); }
TEST(simd_vector_math, Trigonometric_double)             { run_simd_vector_math_Trigonometric<double>(); }
TEST(simd_vector_math, Hyperbolic_float)                 { run_simd_vector_math_Hyperbolic<float>(); }
TEST(simd_vector_math, Hyperbolic_double)                { run_simd_vector_math_Hyperbolic<double>(); }
TEST(simd_vector_math, PowersAndRoots_float)             { run_simd_vector_math_PowersAndRoots<float>(); }
TEST(simd_vector_math, PowersAndRoots_double)            { run_simd_vector_math_PowersAndRoots<double>(); }
TEST(simd_vector_math, NormalDistribution_float)         { run_simd_vector_math_NormalDistribution<float>(); }
TEST(simd_vector_math, NormalDistribution_double)        { run_simd_vector_math_NormalDistribution<double>(); }
TEST(simd_vector_math, SpecialValues_float)              { run_simd_vector_math_SpecialValues<float>(); }
TEST(simd_vector_math, SpecialValues_double)             { run_simd_vector_math_SpecialValues<double>(); }
// clang-format on
}  // namespace

#endif
