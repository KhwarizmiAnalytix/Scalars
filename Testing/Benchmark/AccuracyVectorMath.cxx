// Errors of the functions BenchmarkVectorMath times (VectorMathOps.h): simd<T> on
// the selected vector math library, the scalar std:: (libm) call and, when
// SCALARS_BENCHMARK_HAS_EIGEN is set, Eigen, against a long double reference.
//
// For each function, type and implementation it prints one JSON object per line:
//   core    -- the benchmark's input range, uniformly sampled;
//   wide    -- the function's whole IEEE domain, mostly log-uniform in magnitude,
//              including subnormal inputs, huge trig arguments and the
//              neighbourhoods of 1 where log-type results vanish;
//   special -- NaN, infinities, signed zeros and overflow/underflow edges, which
//              pass when the class of the result (NaN, infinity, signed zero)
//              matches the reference and a finite result is within 8 ulp.
// Errors are in units in the last place of the correctly rounded result,
// relative to the result itself; results below the smallest normal number are
// measured in units of the subnormal spacing. Double needs a reference wider than
// double: where long double is double (MSVC ABI), only float is measured.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "VectorMathOps.h"

#if SCALARS_VECTORIZED

namespace
{
using namespace scalars_vm;

constexpr std::size_t kSamples = std::size_t(1) << 20;
constexpr std::size_t kPad     = 16;  // a multiple of every packet width
constexpr bool        kExtendedReference =
    std::numeric_limits<ref_t>::digits > std::numeric_limits<double>::digits;

template <class T> struct lim
{
    static constexpr double dmin = std::numeric_limits<T>::denorm_min();
    static constexpr double nmin = std::numeric_limits<T>::min();
    static constexpr double max  = std::numeric_limits<T>::max();
    static constexpr double eps  = std::numeric_limits<T>::epsilon();
};

// x = sign * (offset + dir * m), with m uniform or log-uniform in [lo, hi] and a
// random sign when sym.
struct segment
{
    double lo, hi;
    bool   log    = false;
    bool   sym    = false;
    double offset = 0;
    double dir    = 1;
};

segment U(double lo, double hi)
{
    return {lo, hi};
}
segment L(double lo, double hi, bool sym = false, double offset = 0, double dir = 1)
{
    return {lo, hi, true, sym, offset, dir};
}

double draw(const segment& s, std::mt19937_64& g)
{
    double m;
    if (s.log)
    {
        std::uniform_real_distribution<double> d(std::log(s.lo), std::log(s.hi));
        m = std::clamp(std::exp(d(g)), s.lo, s.hi);
    }
    else
    {
        std::uniform_real_distribution<double> d(s.lo, s.hi);
        m = d(g);
    }
    double x = s.offset + s.dir * m;
    if (s.sym && std::bernoulli_distribution(0.5)(g))
    {
        x = -x;
    }
    return x;
}

template <class T> std::vector<segment> wide_domain(const std::string& n)
{
    using l              = lim<T>;
    const double cdf_low = sizeof(T) == 4 ? -13.0 : -37.5;
    if (n == "exp")
    {
        return {U(std::log(l::nmin), std::log(l::max)), U(std::log(l::dmin), std::log(l::nmin))};
    }
    if (n == "expm1")
    {
        return {L(l::nmin, std::log(l::max)), L(l::nmin, -std::log(l::dmin), false, 0, -1)};
    }
    if (n == "exp2")
    {
        return {
            U(std::log2(l::nmin), std::log2(l::max)), U(std::log2(l::dmin), std::log2(l::nmin))};
    }
    if (n == "exp10")
    {
        return {U(std::log10(l::nmin), std::log10(l::max)),
            U(std::log10(l::dmin), std::log10(l::nmin))};
    }
    if (n == "log" || n == "log2" || n == "log10")
    {
        return {L(l::dmin, l::max), L(l::eps, 0.5, false, 1, -1), L(l::eps, 1, false, 1, 1)};
    }
    if (n == "log1p")
    {
        return {L(l::nmin, l::max), L(l::nmin, 0.5, false, 0, -1), L(l::eps, 0.5, false, -1, 1)};
    }
    if (n == "sin" || n == "cos" || n == "tan")
    {
        return {L(l::dmin, l::max, true), L(1, 1e5, true)};
    }
    if (n == "asin" || n == "acos" || n == "atanh")
    {
        return {L(l::dmin, 1, true), L(l::eps / 2, 1, true, 1, -1)};
    }
    if (n == "atan" || n == "asinh" || n == "cbrt")
    {
        return {L(l::dmin, l::max, true)};
    }
    if (n == "sinh" || n == "cosh")
    {
        return {segment{0, std::log(l::max) + 0.6, false, true}, L(l::dmin, 1, true)};
    }
    if (n == "tanh")
    {
        return {L(l::dmin, 50, true)};
    }
    if (n == "acosh")
    {
        return {L(l::eps / 4, l::max, false, 1, 1)};
    }
    if (n == "sqrt" || n == "invsqrt")
    {
        return {L(l::dmin, l::max)};
    }
    if (n == "cdf")
    {
        return {U(cdf_low, 9), L(l::dmin, 1, true)};
    }
    if (n == "inv_cdf")
    {
        return {L(l::nmin, 0.5), L(l::eps / 2, 0.5, false, 1, -1)};
    }
    return {};
}

template <class T> struct samples
{
    std::vector<T>     x, y;
    std::vector<ref_t> want;
};

// pow samples keep the result a normal number, so that every sample measures
// the function rather than overflow or underflow.
template <class Op, class T> bool keep(T x, T y, ref_t want)
{
    if constexpr (std::string_view(Op::name) == "pow")
    {
        (void)x, (void)y;
        return std::isfinite(want) && std::fabs(want) >= std::numeric_limits<T>::min();
    }
    else
    {
        return true;
    }
}

template <class Op, class T> samples<T> make_samples(bool wide, unsigned seed)
{
    using l = lim<T>;
    std::mt19937_64 g(seed);
    samples<T>      s;
    s.x.reserve(kSamples), s.y.reserve(kSamples), s.want.reserve(kSamples);
    const auto  segs = wide_domain<T>(Op::name);
    std::size_t k    = 0;
    while (s.x.size() < kSamples)
    {
        double x = 0, y = 0;
        ++k;
        if (!wide)
        {
            x = std::uniform_real_distribution<double>(Op::lo, Op::hi)(g);
            y = Op::binary ? std::uniform_real_distribution<double>(Op::lo2, Op::hi2)(g) : 0;
        }
        else if constexpr (std::string_view(Op::name) == "pow")
        {
            if (k % 5 != 0)
            {
                // A positive base and an exponent that lands the result in the normal range.
                x                = draw(L(l::nmin, l::max), g);
                const double lg  = std::log2(x);
                const double top = sizeof(T) == 4 ? 125.0 : 1020.0;
                if (std::fabs(lg) < 1e-3)
                {
                    continue;
                }
                y = std::uniform_real_distribution<double>(-top, top)(g) / lg;
            }
            else
            {
                // A negative base and an integer exponent.
                x             = -draw(L(1e-3, 1e3), g);
                const int top = sizeof(T) == 4 ? 12 : 100;
                y             = std::uniform_int_distribution<int>(-top, top)(g);
            }
        }
        else if constexpr (std::string_view(Op::name) == "hypot")
        {
            x = draw(L(l::nmin, l::max / 4, true), g);
            y = k % 2 == 0 ? draw(L(l::nmin, l::max / 4, true), g)
                           : x * draw(U(0.5, 2), g) * (k % 4 == 1 ? -1 : 1);
        }
        else
        {
            x = draw(segs[k % segs.size()], g);
        }
        const T     xt   = static_cast<T>(x);
        const T     yt   = static_cast<T>(y);
        const ref_t want = Op::reference(xt, yt);
        if (!keep<Op, T>(xt, yt, want))
        {
            continue;
        }
        s.x.push_back(xt), s.y.push_back(yt), s.want.push_back(want);
    }
    return s;
}

template <class T> std::vector<T> special_values()
{
    using l                 = lim<T>;
    const double        nan = std::numeric_limits<double>::quiet_NaN();
    const double        inf = std::numeric_limits<double>::infinity();
    const double        pi  = 3.14159265358979323846;
    std::vector<double> v   = {nan,
          inf,
          -inf,
          0.0,
          -0.0,
          l::dmin,
          -l::dmin,
          l::nmin,
          -l::nmin,
          l::max,
          -l::max,
          1,
          -1,
          0.5,
          -0.5,
          2,
          -2,
          1e-10,
          -1e-10,
          10,
          -10,
          100,
          -100,
          1e10,
          -1e10,
          pi / 2,
          pi,
          -pi,
          1 + l::eps,
          1 - l::eps / 2,
          -1 - l::eps,
          -1 + l::eps / 2,
          -13,
          -38,
          9,
          40};
    for (const double e : {std::log(l::max),
             std::log(l::nmin),
             std::log(l::dmin),
             std::log2(l::max),
             std::log2(l::nmin),
             std::log2(l::dmin),
             std::log10(l::max),
             std::log10(l::nmin),
             std::log10(l::dmin)})
    {
        for (const double f : {0.999, 1.0, 1.001})
        {
            v.push_back(e * f);
        }
    }
    std::vector<T> out(v.begin(), v.end());
    return out;
}

template <class T> std::vector<T> special_pair_values()
{
    using l                 = lim<T>;
    const double        nan = std::numeric_limits<double>::quiet_NaN();
    const double        inf = std::numeric_limits<double>::infinity();
    std::vector<double> v   = {
        nan, inf, -inf, 0.0, -0.0, 1, -1, 2, -2, 0.5, -0.5, 3, -3, l::max, l::dmin};
    return {v.begin(), v.end()};
}

template <class Op, class T> samples<T> make_specials()
{
    samples<T> s;
    if constexpr (Op::binary)
    {
        const auto v = special_pair_values<T>();
        for (const T a : v)
        {
            for (const T b : v)
            {
                s.x.push_back(a), s.y.push_back(b);
            }
        }
    }
    else
    {
        s.x = special_values<T>();
        s.y.assign(s.x.size(), T(0));
    }
    for (std::size_t i = 0; i < s.x.size(); ++i)
    {
        s.want.push_back(Op::reference(s.x[i], s.y[i]));
    }
    return s;
}

// Error of actual in ulps of T at expected (see the file comment).
template <class T> ref_t ulp_error(T actual, ref_t expected)
{
    constexpr ref_t inf = std::numeric_limits<ref_t>::infinity();
    if (std::isnan(expected))
    {
        return std::isnan(actual) ? 0 : inf;
    }
    const T rounded = static_cast<T>(expected);
    if (std::isinf(rounded))
    {
        return actual == rounded ? 0 : inf;
    }
    if (!std::isfinite(actual))
    {
        return inf;
    }
    const ref_t scale = std::fabs(expected);
    const int   emin  = std::numeric_limits<T>::min_exponent - 1;
    const int   e     = scale == 0 ? emin : std::max(std::ilogb(scale), emin);
    const ref_t ulp   = std::ldexp(ref_t(1), e - (std::numeric_limits<T>::digits - 1));
    return std::fabs(static_cast<ref_t>(actual) - expected) / ulp;
}

template <class T> bool special_ok(T got, ref_t want)
{
    if (std::isnan(want))
    {
        return std::isnan(got);
    }
    if (std::isinf(static_cast<T>(want)))
    {
        return got == static_cast<T>(want);
    }
    if (want == 0)
    {
        return got == 0 && std::signbit(got) == std::signbit(want);
    }
    return ulp_error<T>(got, want) <= 8;
}

// Pads the inputs to whole packets; the padding is evaluated but not scored.
template <class T> std::size_t padded(samples<T>& s)
{
    const std::size_t n = s.x.size();
    s.x.resize((n + kPad - 1) / kPad * kPad, T(0.5));
    s.y.resize(s.x.size(), T(0.5));
    return n;
}

template <class Op, class T> std::vector<T> run_scalars(const samples<T>& s)
{
    using p = simd<T>;
    std::vector<T> out(s.x.size());
    for (std::size_t i = 0; i < s.x.size(); i += p::size)
    {
        p::storeu(Op::template packet<T>(p::loadu(s.x.data() + i), p::loadu(s.y.data() + i)),
            out.data() + i);
    }
    return out;
}

template <class Op, class T> std::vector<T> run_libm(const samples<T>& s)
{
    std::vector<T> out(s.x.size());
    for (std::size_t i = 0; i < s.x.size(); ++i)
    {
        out[i] = Op::template libm<T>(s.x[i], s.y[i]);
    }
    return out;
}

#if SCALARS_BENCHMARK_HAS_EIGEN
template <class T> using const_map = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;

template <class Op, class T> std::vector<T> run_eigen(const samples<T>& s)
{
    std::vector<T> out(s.x.size());
    const auto     n = static_cast<Eigen::Index>(s.x.size());
    const_map<T>   xm(s.x.data(), n), ym(s.y.data(), n);
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>(out.data(), n) = Op::eigen(xm, ym);
    return out;
}

template <class Op, class T> bool eigen_vectorized()
{
    using expr_t = std::decay_t<decltype(Op::eigen(
        std::declval<const_map<T>>(), std::declval<const_map<T>>()))>;
    return (Eigen::internal::evaluator<expr_t>::Flags & Eigen::PacketAccessBit) != 0;
}
#endif

std::string num(double v)
{
    if (std::isnan(v))
    {
        return "NaN";
    }
    if (std::isinf(v))
    {
        return v > 0 ? "Infinity" : "-Infinity";
    }
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.9g", v);
    return buf;
}

std::string text(double v)
{
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.9g", v);
    return buf;
}

const char* platform()
{
#ifdef _WIN32
    return "windows";
#else
    return "linux";
#endif
}

template <class T>
void print_head(const char* fn, const char* impl, const char* kind, const char* label)
{
    std::printf(
        "{\"platform\":\"%s\",\"backend\":\"%s\",\"type\":\"%s\",\"fn\":\"%s\",\"impl\":\"%s\","
        "\"label\":\"%s\",\"kind\":\"%s\"",
        platform(),
        math_library(),
        type_name<T>(),
        fn,
        impl,
        label,
        kind);
}

template <class T>
void score(const char*    fn,
    const char*           impl,
    const char*           label,
    const char*           kind,
    const samples<T>&     s,
    std::size_t           n,
    const std::vector<T>& out)
{
    ref_t       worst = -1, sum = 0;
    std::size_t worst_i = 0, finite = 0, nonfinite = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        const ref_t e = ulp_error<T>(out[i], s.want[i]);
        if (std::isinf(e))
        {
            ++nonfinite;
        }
        else
        {
            sum += e;
            ++finite;
        }
        if (e > worst)
        {
            worst   = e;
            worst_i = i;
        }
    }
    print_head<T>(fn, impl, kind, label);
    std::printf(",\"n\":%zu,\"max_ulp\":%s,\"mean_ulp\":%s,\"nonfinite\":%zu,\"worst_x\":\"%s\","
                "\"worst_y\":\"%s\",\"got\":\"%s\",\"want\":\"%s\"}\n",
        n,
        num(static_cast<double>(worst)).c_str(),
        num(finite ? static_cast<double>(sum / finite) : 0.0).c_str(),
        nonfinite,
        text(s.x[worst_i]).c_str(),
        text(s.y[worst_i]).c_str(),
        text(out[worst_i]).c_str(),
        text(static_cast<double>(s.want[worst_i])).c_str());
}

template <class Op, class T>
void score_special(const char* impl,
    const char*                label,
    const samples<T>&          s,
    std::size_t                n,
    const std::vector<T>&      out)
{
    std::size_t fails = 0;
    std::string examples;
    for (std::size_t i = 0; i < n; ++i)
    {
        if (special_ok<T>(out[i], s.want[i]))
        {
            continue;
        }
        if (++fails <= 6)
        {
            examples += std::string(examples.empty() ? "" : "; ") + Op::name + "(" + text(s.x[i]) +
                        (Op::binary ? ", " + text(s.y[i]) : std::string()) + ")=" + text(out[i]) +
                        " want " + text(static_cast<double>(s.want[i]));
        }
    }
    print_head<T>(Op::name, impl, "special", label);
    std::printf(",\"n\":%zu,\"fails\":%zu,\"examples\":\"%s\"}\n", n, fails, examples.c_str());
}

template <class Op, class T> void measure()
{
    auto              core = make_samples<Op, T>(false, 11);
    auto              wide = make_samples<Op, T>(true, 12);
    auto              spec = make_specials<Op, T>();
    const std::size_t nc = padded(core), nw = padded(wide), ns = padded(spec);

    auto run_all = [&](const char* impl, const char* label, auto run)
    {
        score<T>(Op::name, impl, label, "core", core, nc, run(core));
        if (!wide_domain<T>(Op::name).empty() || Op::binary)
        {
            score<T>(Op::name, impl, label, "wide", wide, nw, run(wide));
        }
        score_special<Op, T>(impl, label, spec, ns, run(spec));
    };
    run_all("scalars", math_library(), [](const samples<T>& s) { return run_scalars<Op, T>(s); });
    if constexpr (Op::has_libm)
    {
        run_all("libm", "libm", [](const samples<T>& s) { return run_libm<Op, T>(s); });
    }
#if SCALARS_BENCHMARK_HAS_EIGEN
    if constexpr (Op::has_eigen)
    {
        run_all("eigen",
            eigen_vectorized<Op, T>() ? "eigen" : "eigen-scalar",
            [](const samples<T>& s) { return run_eigen<Op, T>(s); });
    }
#endif
    std::fflush(stdout);
}
}  // namespace

int main()
{
    // Sample generation and scoring allocate (std::vector, std::string), so
    // report a failure instead of letting std::bad_alloc escape main.
    try
    {
        for_each_op([](auto op) { measure<decltype(op), float>(); });
        if constexpr (kExtendedReference)
        {
            for_each_op([](auto op) { measure<decltype(op), double>(); });
        }
        else
        {
            std::fprintf(stderr, "long double is double here: double accuracy is not measured\n");
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "accuracy_scalars_vectormath: %s\n", e.what());
        return 1;
    }
    return 0;
}

#else
int main()
{
    return 0;
}
#endif  // SCALARS_VECTORIZED
