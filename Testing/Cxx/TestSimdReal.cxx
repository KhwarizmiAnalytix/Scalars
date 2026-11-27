// SIMD tests: the element-wise simd<T> surface shared by every real scalar
// type (float, double, half, bfloat16), as one typed suite.
//
// Covers every unary and binary math function, fma (and fused multiply-
// subtract as fma with a negated addend), the six comparisons, if_else, and
// the mask logic operators. Each lane is checked against a double-precision
// scalar reference with a ULP budget expressed in the type's own epsilon, so
// one table of cases serves all four types. Bit-level reduced-precision
// behaviour (encodings, narrowing, NaN payloads, memory/tail ops) stays in
// TestSimdReduced.cxx; horizontal reductions and packing utilities stay in
// TestSimdReduced.cxx and TestSimdUtility.cxx.
#include "backend/simd.h"

#if SCALARS_VECTORIZED

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>

#include "math/normal_cdf.h"
#include "scalars_test.h"

using scalars::simd;

namespace
{
inline constexpr int kRandomTrials = 256;

// Per-type test parameters. `eps` is the spacing of the type at 1.0, so a
// tolerance of `ulps * eps * |expected|` is between `ulps` and `2 * ulps`
// units in the last place. `math_ulps` is the default budget for library math
// (SLEEF u10, SVML, Accelerate, libm). Reduced types compute in float and
// round once, so a single ULP of the reduced type covers them. `mag_lo` and
// `mag_hi` bound the log-uniform magnitude sweeps to the type's range.
template <class T> struct real_traits;

template <> struct real_traits<float>
{
    static constexpr const char* name      = "float";
    static constexpr double      eps       = 0x1p-23;
    static constexpr double      math_ulps = 4;
    static constexpr double      mag_lo    = 1e-30;
    static constexpr double      mag_hi    = 1e30;
};

template <> struct real_traits<double>
{
    static constexpr const char* name      = "double";
    static constexpr double      eps       = 0x1p-52;
    static constexpr double      math_ulps = 4;
    static constexpr double      mag_lo    = 1e-250;
    static constexpr double      mag_hi    = 1e250;
};

template <> struct real_traits<scalars::half>
{
    static constexpr const char* name      = "half";
    static constexpr double      eps       = 0x1p-10;
    static constexpr double      math_ulps = 1;
    static constexpr double      mag_lo    = 1e-4;
    static constexpr double      mag_hi    = 1e4;
};

template <> struct real_traits<scalars::bfloat16>
{
    static constexpr const char* name      = "bfloat16";
    static constexpr double      eps       = 0x1p-7;
    static constexpr double      math_ulps = 1;
    static constexpr double      mag_lo    = 1e-30;
    static constexpr double      mag_hi    = 1e30;
};

template <class T> double to_double(T v)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        return static_cast<double>(v);
    }
    else
    {
        // half also converts to float16_t on Arm; go through float explicitly.
        return static_cast<double>(static_cast<float>(v));
    }
}

template <class T> T from_double(double v)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        return static_cast<T>(v);
    }
    else
    {
        return T(static_cast<float>(v));
    }
}

template <class T> std::uint64_t bits_of(const T& v)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(T));
    return bits;
}

template <class T> T from_bits(std::uint64_t bits)
{
    T v;
    std::memcpy(&v, &bits, sizeof(T));
    return v;
}

template <class T> using lanes = std::array<T, static_cast<std::size_t>(simd<T>::size)>;

template <class T> typename simd<T>::simd_t load(const lanes<T>& v)
{
    return simd<T>::loadu(v.data());
}

template <class T> lanes<T> store(typename simd<T>::simd_t p)
{
    lanes<T> out{};
    simd<T>::storeu(p, out.data());
    return out;
}

// Input domain. Uniform domains sample [lo, hi]; log-uniform domains sample
// positive magnitudes across [mag_lo, mag_hi] of the type. Samples are rounded
// to T before use and resampled if rounding leaves the domain, so domain
// edges (acos at 1, inv_cdf near 0 and 1) stay valid for coarse types.
struct domain
{
    double lo;
    double hi;
    bool   log_uniform = false;
};

template <class T> domain log_uniform()
{
    return {real_traits<T>::mag_lo, real_traits<T>::mag_hi, true};
}

template <class T> T sample(std::mt19937& gen, const domain& d)
{
    if (d.log_uniform)
    {
        std::uniform_real_distribution<double> dist(std::log(d.lo), std::log(d.hi));
        return from_double<T>(std::exp(dist(gen)));
    }
    std::uniform_real_distribution<double> dist(d.lo, d.hi);
    for (;;)
    {
        const T      v = from_double<T>(dist(gen));
        const double r = to_double(v);
        if (r >= d.lo && r <= d.hi)
        {
            return v;
        }
    }
}

// Accuracy budget for one function: `ulps` scales the type epsilon, and
// `floor` is the magnitude below which the error is measured absolutely
// (ulps * eps * floor). The default floor is the smallest normal number, so
// subnormal and flushed results are held to the subnormal spacing. A floor of
// 1 suits functions whose implementations lose relative accuracy near a zero
// of the result (trig argument reduction, log near 1, erfc-based cdf tails).
struct budget
{
    double ulps;
    double floor = 0;
};

// Tracks the worst lane of one function; report() asserts it once, with the
// inputs that produced it.
template <class T> class accuracy_check
{
public:
    accuracy_check(const char* name, budget b)
        : name_(name), ulps_(b.ulps),
          floor_(std::max(b.floor, to_double(std::numeric_limits<T>::min())))
    {
    }

    void report() const
    {
        EXPECT_LE(worst_ratio_, 1.0) << real_traits<T>::name << "::" << name_ << worst_;
    }

    template <class... Inputs> void check(T actual_t, double expected, Inputs... inputs)
    {
        const double actual = to_double(actual_t);
        // Compare exceptional results against the reference rounded to T so
        // overflow to infinity in the narrower type is expected behaviour.
        const double rounded = to_double(from_double<T>(expected));
        double       ratio   = 0;
        if (std::isnan(expected) || std::isinf(rounded))
        {
            const bool same = std::isnan(expected) ? std::isnan(actual) : actual == rounded;
            ratio           = same ? 0 : std::numeric_limits<double>::infinity();
        }
        else if (!std::isfinite(actual))
        {
            ratio = std::numeric_limits<double>::infinity();
        }
        else
        {
            const double tol = ulps_ * real_traits<T>::eps * std::max(std::fabs(expected), floor_);
            const double err = std::fabs(actual - expected);
            ratio = tol > 0 ? err / tol : (err == 0 ? 0 : std::numeric_limits<double>::infinity());
        }
        if (ratio > worst_ratio_ || (std::isinf(ratio) && !std::isinf(worst_ratio_)))
        {
            worst_ratio_ = ratio;
            std::ostringstream os;
            os << std::setprecision(17) << "\n  inputs   =";
            ((os << ' ' << inputs), ...);
            os << "\n  simd     = " << actual << "\n  expected = " << expected
               << "\n  error/tolerance = " << ratio;
            worst_ = os.str();
        }
    }

private:
    const char* name_;
    double      ulps_;
    double      floor_;
    double      worst_ratio_ = 0;
    std::string worst_;
};

template <class T, class SimdOp, class RefOp>
void check_unary(const char* name, domain d, budget b, SimdOp simd_op, RefOp ref_op)
{
    accuracy_check<T> check(name, b);
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
            check.check(out[i], ref_op(to_double(x[i])), to_double(x[i]));
        }
    }
    check.report();
}

template <class T, class SimdOp, class RefOp>
void check_binary(const char* name, domain dx, domain dy, budget b, SimdOp simd_op, RefOp ref_op)
{
    accuracy_check<T> check(name, b);
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
            const double u = to_double(x[i]);
            const double v = to_double(y[i]);
            check.check(out[i], ref_op(u, v), u, v);
        }
    }
    check.report();
}

// NEON/SVE invsqrt is a vrsqrte estimate refined by Newton steps: one step
// for float (about 16 correct bits, ~120 ULP measured) and two for double
// (about 32 bits, ~1e6 ULP measured). x86 float refines rsqrt by one step to
// within a few ULP, x86 double divides by sqrt, and the reduced types divide
// in float, so they are held to the normal budget.
template <class T> constexpr double invsqrt_ulps()
{
#if SCALARS_HAS_NEON || SCALARS_HAS_SVE
    if constexpr (std::is_same_v<T, float>)
    {
        return 0x1p9;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return 0x1p22;
    }
#endif
    return std::is_floating_point_v<T> ? real_traits<T>::math_ulps : 1;
}

template <class T> void run_simd_real_UnaryMath()
{
    using S           = simd<T>;  // NOLINT(readability-identifier-naming)
    using simd_t      = typename S::simd_t;
    const double math = real_traits<T>::math_ulps;
    const budget exact{0};
    const budget rounded{1};
    const budget lib{math};
    const budget lib_abs{math, 1};
    const domain pos = log_uniform<T>();

#define CHECK_UNARY(fn, dom, bud, ref)                                                             \
    check_unary<T>(#fn, dom, bud, [](simd_t a) { return S::fn(a); }, [](double x) { return ref; })

    CHECK_UNARY(sqrt, pos, rounded, std::sqrt(x));
    CHECK_UNARY(invsqrt, pos, budget{invsqrt_ulps<T>()}, 1 / std::sqrt(x));
    CHECK_UNARY(sqr, (domain{-25, 25}), rounded, x * x);
    CHECK_UNARY(ceil, (domain{-75, 75}), exact, std::ceil(x));
    CHECK_UNARY(floor, (domain{-75, 75}), exact, std::floor(x));
    CHECK_UNARY(trunc, (domain{-75, 75}), exact, std::trunc(x));
    CHECK_UNARY(fabs, (domain{-120, 120}), exact, std::fabs(x));
    CHECK_UNARY(neg, (domain{-120, 120}), exact, -x);

    CHECK_UNARY(exp, (domain{-8, 8}), lib, std::exp(x));
    CHECK_UNARY(expm1, (domain{-8, 8}), lib, std::expm1(x));
    CHECK_UNARY(exp2, (domain{-12, 12}), lib, std::exp2(x));
    CHECK_UNARY(exp10, (domain{-3, 3}), lib, std::pow(10.0, x));
    CHECK_UNARY(log, pos, lib_abs, std::log(x));
    CHECK_UNARY(log1p, (domain{-0.95, 500}), lib, std::log1p(x));
    CHECK_UNARY(log2, pos, lib_abs, std::log2(x));
    CHECK_UNARY(log10, pos, lib_abs, std::log10(x));

    CHECK_UNARY(sin, (domain{-25, 25}), lib_abs, std::sin(x));
    CHECK_UNARY(cos, (domain{-25, 25}), lib_abs, std::cos(x));
    CHECK_UNARY(tan, (domain{-1.2, 1.2}), lib_abs, std::tan(x));
    CHECK_UNARY(asin, (domain{-1, 1}), lib, std::asin(x));
    CHECK_UNARY(acos, (domain{-1, 1}), lib_abs, std::acos(x));
    CHECK_UNARY(atan, (domain{-25, 25}), lib, std::atan(x));

    CHECK_UNARY(sinh, (domain{-8, 8}), lib, std::sinh(x));
    CHECK_UNARY(cosh, (domain{-8, 8}), lib, std::cosh(x));
    CHECK_UNARY(tanh, (domain{-8, 8}), lib, std::tanh(x));
    CHECK_UNARY(asinh, (domain{-80, 80}), lib, std::asinh(x));
    CHECK_UNARY(acosh, (domain{1, 80}), lib_abs, std::acosh(x));
    CHECK_UNARY(atanh, (domain{-0.99, 0.99}), lib, std::atanh(x));
    CHECK_UNARY(cbrt, (domain{-125, 125}), lib, std::cbrt(x));

    CHECK_UNARY(cdf, (domain{-8, 8}), lib_abs, scalars::normalcdf(x));
    CHECK_UNARY(inv_cdf, (domain{1e-6, 1 - 1e-6}), lib_abs, scalars::inv_normalcdf(x));
#undef CHECK_UNARY
}

TEST(simd_real, UnaryMath_float)
{
    run_simd_real_UnaryMath<float>();
}
TEST(simd_real, UnaryMath_double)
{
    run_simd_real_UnaryMath<double>();
}
TEST(simd_real, UnaryMath_half)
{
    run_simd_real_UnaryMath<scalars::half>();
}
TEST(simd_real, UnaryMath_bfloat16)
{
    run_simd_real_UnaryMath<scalars::bfloat16>();
}

template <class T> void run_simd_real_BinaryMath()
{
    using S      = simd<T>;  // NOLINT(readability-identifier-naming)
    using simd_t = typename S::simd_t;
    const budget exact{0};
    const budget rounded{1};
    const budget lib{real_traits<T>::math_ulps};
    const domain pos = log_uniform<T>();

#define CHECK_BINARY(fn, dx, dy, bud, ref)                                                         \
    check_binary<T>(                                                                               \
        #fn,                                                                                       \
        dx,                                                                                        \
        dy,                                                                                        \
        bud,                                                                                       \
        [](simd_t a, simd_t b) { return S::fn(a, b); },                                            \
        [](double x, double y) { return ref; })

    const domain mid{-40, 40};
    CHECK_BINARY(add, mid, mid, rounded, x + y);
    CHECK_BINARY(sub, mid, mid, rounded, x - y);
    CHECK_BINARY(mul, (domain{-25, 25}), (domain{-25, 25}), rounded, x * y);
    CHECK_BINARY(div, (domain{-50, 50}), (domain{0.001, 50}), rounded, x / y);
    // Magnitude sweeps reach overflow to infinity and underflow to subnormals.
    CHECK_BINARY(mul, pos, pos, rounded, x * y);
    CHECK_BINARY(div, pos, pos, rounded, x / y);
    CHECK_BINARY(min, mid, mid, exact, std::fmin(x, y));
    CHECK_BINARY(max, mid, mid, exact, std::fmax(x, y));
    CHECK_BINARY(signcopy, (domain{-80, 80}), (domain{-80, 80}), exact, std::copysign(x, y));
    CHECK_BINARY(hypot, (domain{-80, 80}), (domain{-80, 80}), lib, std::hypot(x, y));
    CHECK_BINARY(hypot, pos, pos, lib, std::hypot(x, y));
    CHECK_BINARY(pow, (domain{0.05, 12}), (domain{-3, 3}), lib, std::pow(x, y));
#undef CHECK_BINARY
}

TEST(simd_real, BinaryMath_float)
{
    run_simd_real_BinaryMath<float>();
}
TEST(simd_real, BinaryMath_double)
{
    run_simd_real_BinaryMath<double>();
}
TEST(simd_real, BinaryMath_half)
{
    run_simd_real_BinaryMath<scalars::half>();
}
TEST(simd_real, BinaryMath_bfloat16)
{
    run_simd_real_BinaryMath<scalars::bfloat16>();
}

// A runtime probe of whether simd<T>::fma rounds once. x86 SSE/AVX builds
// without -mfma fall back to mul+add; reduced types widen to float, where the
// product of two reduced values is exact, so they always behave as fused.
template <class T> bool fma_is_fused()
{
    if constexpr (!std::is_floating_point_v<T>)
    {
        return true;
    }
    else
    {
        const int  digits = std::numeric_limits<T>::digits;
        const T    d      = std::ldexp(T(1), -(digits / 2 + 1));
        const auto r      = store<T>(
            simd<T>::fma(simd<T>::set(T(1) + d), simd<T>::set(T(1) - d), simd<T>::set(T(-1))));
        return r[0] != T(0);
    }
}

// Reference for a*b + c. Fused: exact in double for float and reduced types,
// and std::fma for double. Unfused: round the product to T first.
template <class T> double fma_reference(bool fused, double a, double b, double c)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        if (!fused)
        {
            volatile T product = static_cast<T>(a * b);
            return static_cast<double>(static_cast<T>(product + static_cast<T>(c)));
        }
    }
    return std::fma(a, b, c);
}

template <class T> void run_simd_real_FusedMultiplyAdd()
{
    using S          = simd<T>;  // NOLINT(readability-identifier-naming)
    const bool fused = fma_is_fused<T>();
    SCOPED_TRACE(fused ? "fused" : "unfused mul+add");
    const domain d{-25, 25};

    for (const bool subtract : {false, true})
    {
        accuracy_check<T> check(subtract ? "fms" : "fma", budget{1});
        // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility
        std::mt19937 gen(5489u);
        lanes<T>     x{}, y{}, z{};
        for (int trial = 0; trial < kRandomTrials; ++trial)
        {
            for (std::size_t i = 0; i < x.size(); ++i)
            {
                x[i] = sample<T>(gen, d);
                y[i] = sample<T>(gen, d);
                // Every eighth lane cancels the product exactly or nearly so,
                // where a fused result differs most from mul+add.
                z[i] = (i % 8 == 0)
                           ? from_double<T>((subtract ? 1 : -1) * to_double(x[i]) * to_double(y[i]))
                           : sample<T>(gen, d);
            }
            const auto c   = subtract ? S::neg(load<T>(z)) : load<T>(z);
            const auto out = store<T>(S::fma(load<T>(x), load<T>(y), c));
            for (std::size_t i = 0; i < x.size(); ++i)
            {
                const double a = to_double(x[i]), b = to_double(y[i]), e = to_double(z[i]);
                check.check(out[i], fma_reference<T>(fused, a, b, subtract ? -e : e), a, b, e);
            }
        }
        check.report();
    }
}

TEST(simd_real, FusedMultiplyAdd_float)
{
    run_simd_real_FusedMultiplyAdd<float>();
}
TEST(simd_real, FusedMultiplyAdd_double)
{
    run_simd_real_FusedMultiplyAdd<double>();
}
TEST(simd_real, FusedMultiplyAdd_half)
{
    run_simd_real_FusedMultiplyAdd<scalars::half>();
}
TEST(simd_real, FusedMultiplyAdd_bfloat16)
{
    run_simd_real_FusedMultiplyAdd<scalars::bfloat16>();
}

// Special values every comparison must order like the C++ scalar operators:
// signed zeros compare equal, NaN is unordered (only neq is true).
template <class T> std::array<T, 9> comparison_values()
{
    const double inf = std::numeric_limits<double>::infinity();
    return {from_double<T>(-inf),
        from_double<T>(-2),
        from_double<T>(-1),
        from_double<T>(-0.0),
        from_double<T>(0),
        from_double<T>(1),
        from_double<T>(2),
        from_double<T>(inf),
        std::numeric_limits<T>::quiet_NaN()};
}

// Reads a mask through if_else, the only lane-observable mask use common to
// every backend (mask storage layouts differ: AVX-512 masks are bitfields).
template <class T>
std::array<bool, static_cast<std::size_t>(simd<T>::size)> mask_lanes(typename simd<T>::mask_t m)
{
    const auto sel = store<T>(
        simd<T>::if_else(m, simd<T>::set(from_double<T>(1)), simd<T>::set(from_double<T>(0))));
    std::array<bool, static_cast<std::size_t>(simd<T>::size)> out{};
    for (std::size_t i = 0; i < sel.size(); ++i)
    {
        out[i] = to_double(sel[i]) == 1;
    }
    return out;
}

template <class T> void run_simd_real_Comparisons()
{
    using S                  = simd<T>;  // NOLINT(readability-identifier-naming)
    const auto        values = comparison_values<T>();
    const std::size_t n      = values.size();
    lanes<T>          x{}, y{};
    // Walk every ordered pair of special values, one packet at a time.
    for (std::size_t first = 0; first < n * n; first += x.size())
    {
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            const std::size_t pair = (first + i) % (n * n);
            x[i]                   = values[pair / n];
            y[i]                   = values[pair % n];
        }
        const auto a = load<T>(x), b = load<T>(y);
// NOLINTBEGIN(bugprone-macro-parentheses): `op` is an operator token.
#define CHECK_COMPARE(fn, op)                                                                      \
    {                                                                                              \
        const auto got = mask_lanes<T>(S::fn(a, b));                                               \
        for (std::size_t i = 0; i < x.size(); ++i)                                                 \
        {                                                                                          \
            const double u = to_double(x[i]), v = to_double(y[i]);                                 \
            EXPECT_EQ(got[i], u op v) << #fn << '(' << u << ", " << v << ')';                      \
        }                                                                                          \
    }
        // NOLINTEND(bugprone-macro-parentheses)
        CHECK_COMPARE(eq, ==)
        CHECK_COMPARE(neq, !=)
        CHECK_COMPARE(gt, >)
        CHECK_COMPARE(ge, >=)
        CHECK_COMPARE(lt, <)
        CHECK_COMPARE(le, <=)
#undef CHECK_COMPARE
    }
}

TEST(simd_real, Comparisons_float)
{
    run_simd_real_Comparisons<float>();
}
TEST(simd_real, Comparisons_double)
{
    run_simd_real_Comparisons<double>();
}
TEST(simd_real, Comparisons_half)
{
    run_simd_real_Comparisons<scalars::half>();
}
TEST(simd_real, Comparisons_bfloat16)
{
    run_simd_real_Comparisons<scalars::bfloat16>();
}

template <class T> void run_simd_real_IfElseAndMaskLogic()
{
    using S = simd<T>;  // NOLINT(readability-identifier-naming)
    // Selected values must come through bit-for-bit, including NaN payloads
    // and the sign of zero, so select operands are built from raw bits.
    const std::uint64_t sign = std::uint64_t{1} << (8 * sizeof(T) - 1);
    const std::uint64_t qnan = bits_of(std::numeric_limits<T>::quiet_NaN());
    lanes<T>            p{}, q{}, x{}, y{};
    for (std::size_t i = 0; i < p.size(); ++i)
    {
        p[i] = i % 3 == 0 ? from_bits<T>(qnan | 1) : from_double<T>(1.5 + double(i));
        q[i] = i % 3 == 1 ? from_bits<T>(sign) : from_double<T>(-0.25 - double(i));
        // Lane pattern exercising lt/eq true, false, and NaN-unordered lanes.
        x[i] = i % 4 == 3 ? std::numeric_limits<T>::quiet_NaN() : from_double<T>(double(i % 4));
        y[i] = from_double<T>(1);
    }
    const auto a = load<T>(x), b = load<T>(y), vp = load<T>(p), vq = load<T>(q);
    const auto lt = S::lt(a, b), eq = S::eq(a, b);

    auto check_select = [&](typename S::mask_t m, auto&& expected, const char* what)
    {
        const auto out = store<T>(S::if_else(m, vp, vq));
        for (std::size_t i = 0; i < p.size(); ++i)
        {
            const bool take = expected(to_double(x[i]) < 1, to_double(x[i]) == 1);
            EXPECT_EQ(bits_of(out[i]), bits_of(take ? p[i] : q[i])) << what << " lane " << i;
        }
    };
    check_select(lt, [](bool l, bool) { return l; }, "lt");
    check_select(eq, [](bool, bool e) { return e; }, "eq");
    check_select(S::not_mask(lt), [](bool l, bool) { return !l; }, "not(lt)");
    check_select(S::and_mask(lt, eq), [](bool l, bool e) { return l && e; }, "and(lt,eq)");
    check_select(S::or_mask(lt, eq), [](bool l, bool e) { return l || e; }, "or(lt,eq)");
    check_select(
        S::xor_mask(lt, S::not_mask(eq)),
        [](bool l, bool e) { return l != !e; },
        "xor(lt,not(eq))");
}

TEST(simd_real, IfElseAndMaskLogic_float)
{
    run_simd_real_IfElseAndMaskLogic<float>();
}
TEST(simd_real, IfElseAndMaskLogic_double)
{
    run_simd_real_IfElseAndMaskLogic<double>();
}
TEST(simd_real, IfElseAndMaskLogic_half)
{
    run_simd_real_IfElseAndMaskLogic<scalars::half>();
}
TEST(simd_real, IfElseAndMaskLogic_bfloat16)
{
    run_simd_real_IfElseAndMaskLogic<scalars::bfloat16>();
}

// Confirmed bug: the NEON hypot implementation computed sqrt(x*x + y*y)
// directly, which overflows for huge components (hypot(1e20, 1e20) == inf)
// and underflows to zero for tiny/subnormal components before the square
// root even runs. Sweep huge, tiny, subnormal, inf, and NaN inputs against
// std::hypot across every CPU backend and real type.
template <class T> void run_simd_real_HypotSpecialValues()
{
    using S      = simd<T>;  // NOLINT(readability-identifier-naming)
    using limits = std::numeric_limits<T>;
    const double                                huge = to_double(limits::max()) / 4;
    const double                                tiny = to_double(limits::denorm_min()) * 4;
    const double                                inf  = std::numeric_limits<double>::infinity();
    const double                                nan  = std::numeric_limits<double>::quiet_NaN();
    const std::array<std::array<double, 2>, 11> cases{{{huge, huge},
        {tiny, tiny * 2},
        {0.0, 0.0},
        {-0.0, -0.0},
        {inf, 1},
        {1, inf},
        {inf, nan},
        {nan, inf},
        {nan, 1},
        {3, 4},
        {-3, 4}}};
    for (const auto& c : cases)
    {
        accuracy_check<T> check("hypot", budget{real_traits<T>::math_ulps});
        const T           x   = from_double<T>(c[0]);
        const T           y   = from_double<T>(c[1]);
        const auto        out = store<T>(S::hypot(S::set(x), S::set(y)));
        for (const T lane : out)
        {
            check.check(lane, std::hypot(to_double(x), to_double(y)), c[0], c[1]);
        }
        check.report();
    }
}

TEST(simd_real, HypotSpecialValues_float)
{
    run_simd_real_HypotSpecialValues<float>();
}
TEST(simd_real, HypotSpecialValues_double)
{
    run_simd_real_HypotSpecialValues<double>();
}
TEST(simd_real, HypotSpecialValues_half)
{
    run_simd_real_HypotSpecialValues<scalars::half>();
}
TEST(simd_real, HypotSpecialValues_bfloat16)
{
    run_simd_real_HypotSpecialValues<scalars::bfloat16>();
}

// Confirmed bug: the built-in float invsqrt (SSE, AVX, AVX-512, NEON) and the
// NEON double one refined the hardware estimate with a Newton step whose
// x * r^2 is 0 * inf for x = +-0 and +inf, returning NaN instead of +-inf and
// 0; the estimate also treats subnormals as zero, which the step turned into
// -inf. Sweep zeros, infinities, NaN, negatives and subnormals against 1/sqrt.
template <class T> void run_simd_real_InvsqrtSpecialValues()
{
    using S                          = simd<T>;  // NOLINT(readability-identifier-naming)
    using limits                     = std::numeric_limits<T>;
    const double                 inf = std::numeric_limits<double>::infinity();
    const double                 nan = std::numeric_limits<double>::quiet_NaN();
    const double                 sub = to_double(limits::denorm_min());
    const double                 min = to_double(limits::min());
    const std::array<double, 15> cases{0.0,
        -0.0,
        inf,
        -inf,
        nan,
        -1,
        -sub,
        sub,
        sub * 3,
        min / 2,
        min,
        to_double(limits::max()),
        4,
        0.25,
        2};
    for (const double c : cases)
    {
        accuracy_check<T> check("invsqrt", budget{invsqrt_ulps<T>()});
        const T           x   = from_double<T>(c);
        const auto        out = store<T>(S::invsqrt(S::set(x)));
        for (const T lane : out)
        {
            check.check(lane, 1 / std::sqrt(to_double(x)), c);
        }
        check.report();
    }
}

TEST(simd_real, InvsqrtSpecialValues_float)
{
    run_simd_real_InvsqrtSpecialValues<float>();
}
TEST(simd_real, InvsqrtSpecialValues_double)
{
    run_simd_real_InvsqrtSpecialValues<double>();
}
TEST(simd_real, InvsqrtSpecialValues_half)
{
    run_simd_real_InvsqrtSpecialValues<scalars::half>();
}
TEST(simd_real, InvsqrtSpecialValues_bfloat16)
{
    run_simd_real_InvsqrtSpecialValues<scalars::bfloat16>();
}

template <class T> void run_simd_real_NaNPropagatesThroughArithmetic()
{
    using S               = simd<T>;  // NOLINT(readability-identifier-naming)
    const auto nan        = S::set(std::numeric_limits<T>::quiet_NaN());
    const auto two        = S::set(from_double<T>(2));
    auto       expect_nan = [](typename S::simd_t v, const char* what)
    {
        for (const T lane : store<T>(v))
        {
            EXPECT_TRUE(std::isnan(to_double(lane))) << what;
        }
    };
    expect_nan(S::add(nan, two), "add");
    expect_nan(S::sub(two, nan), "sub");
    expect_nan(S::mul(nan, two), "mul");
    expect_nan(S::div(two, nan), "div");
    expect_nan(S::fma(nan, two, two), "fma");
    expect_nan(S::sqrt(nan), "sqrt");
    expect_nan(S::sqrt(S::set(from_double<T>(-1))), "sqrt(-1)");
    expect_nan(S::fabs(nan), "fabs");
    expect_nan(S::neg(nan), "neg");
}

TEST(simd_real, NaNPropagatesThroughArithmetic_float)
{
    run_simd_real_NaNPropagatesThroughArithmetic<float>();
}
TEST(simd_real, NaNPropagatesThroughArithmetic_double)
{
    run_simd_real_NaNPropagatesThroughArithmetic<double>();
}
TEST(simd_real, NaNPropagatesThroughArithmetic_half)
{
    run_simd_real_NaNPropagatesThroughArithmetic<scalars::half>();
}
TEST(simd_real, NaNPropagatesThroughArithmetic_bfloat16)
{
    run_simd_real_NaNPropagatesThroughArithmetic<scalars::bfloat16>();
}

}  // namespace

#endif  // SCALARS_VECTORIZED
