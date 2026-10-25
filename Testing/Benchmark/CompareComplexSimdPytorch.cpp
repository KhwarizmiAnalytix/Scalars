// Standalone packet comparison; built by Scripts/benchmark_complex_simd.py.
#include "backend/simd.h"
#include <cstdlib>

// These compatibility aliases belong to different libraries.
#undef HIDDEN_NAMESPACE_BEGIN
#undef HIDDEN_NAMESPACE_END
#include <ATen/cpu/vec/vec.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <string_view>
#include <system_error>
#include <vector>

namespace
{
template <class T> using complex_t       = scalars::complex<T>;
template <class T> using torch_complex_t = c10::complex<T>;
template <class T> using packet          = simd<complex_t<T>>;
template <class T> using torch_packet    = at::vec::Vectorized<torch_complex_t<T>>;

enum class operation : std::uint8_t
{
    add,
    sub,
    mul,
    div,
    fma,
    sqr,
    abs,
    arg,
    sqrt,
    invsqrt,
    reciprocal,
    exp,
    expm1,
    exp2,
    exp10,
    log,
    log1p,
    log2,
    log10,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    tanh,
    asinh,
    acosh,
    atanh,
    pow,
    ceil,
    floor,
    trunc,
    neg,
    conj,
    real,
    imag,
    eq,
    neq
};

template <operation Op, class T>
SCALARS_FORCE_INLINE typename packet<T>::simd_t scalars_evaluate(
    typename packet<T>::simd_t a, typename packet<T>::simd_t b, typename packet<T>::simd_t c)
{
    using p = packet<T>;
    using r = simd<T>;
    if constexpr (Op == operation::add)
    {
        return p::add(a, b);
    }
    if constexpr (Op == operation::sub)
    {
        return p::sub(a, b);
    }
    if constexpr (Op == operation::mul)
    {
        return p::mul(a, b);
    }
    if constexpr (Op == operation::div)
    {
        return p::div(a, b);
    }
    if constexpr (Op == operation::fma)
    {
        return p::fma(a, b, c);
    }
    if constexpr (Op == operation::abs)
    {
        return {p::abs(a), r::setzero()};
    }
    if constexpr (Op == operation::arg)
    {
        return {p::arg(a), r::setzero()};
    }
    if constexpr (Op == operation::real)
    {
        return {p::real(a), r::setzero()};
    }
    if constexpr (Op == operation::imag)
    {
        return {p::imag(a), r::setzero()};
    }
    if constexpr (Op == operation::eq)
    {
        return p::if_else(p::eq(a, b), p::set(T(1)), p::setzero());
    }
    if constexpr (Op == operation::neq)
    {
        return p::if_else(p::neq(a, b), p::set(T(1)), p::setzero());
    }
    if constexpr (Op == operation::pow)
    {
        return p::pow(a, b);
    }
#define SCALARS_COMPARE_UNARY(name)                                                                \
    if constexpr (Op == operation::name)                                                           \
    {                                                                                              \
        return p::name(a);                                                                         \
    }
    SCALARS_COMPARE_UNARY(sqr);
    SCALARS_COMPARE_UNARY(sqrt);
    SCALARS_COMPARE_UNARY(invsqrt);
    SCALARS_COMPARE_UNARY(reciprocal);
    SCALARS_COMPARE_UNARY(exp);
    SCALARS_COMPARE_UNARY(expm1);
    SCALARS_COMPARE_UNARY(exp2);
    SCALARS_COMPARE_UNARY(exp10);
    SCALARS_COMPARE_UNARY(log);
    SCALARS_COMPARE_UNARY(log1p);
    SCALARS_COMPARE_UNARY(log2);
    SCALARS_COMPARE_UNARY(log10);
    SCALARS_COMPARE_UNARY(sin);
    SCALARS_COMPARE_UNARY(cos);
    SCALARS_COMPARE_UNARY(tan);
    SCALARS_COMPARE_UNARY(asin);
    SCALARS_COMPARE_UNARY(acos);
    SCALARS_COMPARE_UNARY(atan);
    SCALARS_COMPARE_UNARY(sinh);
    SCALARS_COMPARE_UNARY(cosh);
    SCALARS_COMPARE_UNARY(tanh);
    SCALARS_COMPARE_UNARY(asinh);
    SCALARS_COMPARE_UNARY(acosh);
    SCALARS_COMPARE_UNARY(atanh);
    SCALARS_COMPARE_UNARY(ceil);
    SCALARS_COMPARE_UNARY(floor);
    SCALARS_COMPARE_UNARY(trunc);
    SCALARS_COMPARE_UNARY(neg);
    SCALARS_COMPARE_UNARY(conj);
#undef SCALARS_COMPARE_UNARY
    std::abort();
}

template <operation Op, class T>
SCALARS_FORCE_INLINE torch_packet<T> torch_evaluate(
    torch_packet<T> a, torch_packet<T> b, torch_packet<T> c)
{
    if constexpr (Op == operation::add)
    {
        return a + b;
    }
    if constexpr (Op == operation::sub)
    {
        return a - b;
    }
    if constexpr (Op == operation::mul)
    {
        return a * b;
    }
    if constexpr (Op == operation::div)
    {
        return a / b;
    }
    if constexpr (Op == operation::fma)
    {
        return at::vec::fmadd(a, b, c);
    }
    if constexpr (Op == operation::sqr)
    {
        return a * a;
    }
    if constexpr (Op == operation::arg)
    {
        return a.angle();
    }
    if constexpr (Op == operation::invsqrt)
    {
        return a.rsqrt();
    }
    if constexpr (Op == operation::pow)
    {
        return a.pow(b);
    }
    if constexpr (Op == operation::eq)
    {
        return a.eq(b);
    }
    if constexpr (Op == operation::neq)
    {
        return a.ne(b);
    }
    // ATen has no exp10 packet method. Compose its scalar map with the same
    // component scaling used by Scalars, without adding complex cross-products.
    if constexpr (Op == operation::exp10)
    {
        return a.map(
            [](const torch_complex_t<T>& z) { return std::exp(z * std::numbers::ln10_v<T>); });
    }
#define SCALARS_COMPARE_UNARY(name)                                                                \
    if constexpr (Op == operation::name)                                                           \
    {                                                                                              \
        return a.name();                                                                           \
    }
    SCALARS_COMPARE_UNARY(abs);
    SCALARS_COMPARE_UNARY(sqrt);
    SCALARS_COMPARE_UNARY(reciprocal);
    SCALARS_COMPARE_UNARY(exp);
    SCALARS_COMPARE_UNARY(expm1);
    SCALARS_COMPARE_UNARY(exp2);
    SCALARS_COMPARE_UNARY(log);
    SCALARS_COMPARE_UNARY(log1p);
    SCALARS_COMPARE_UNARY(log2);
    SCALARS_COMPARE_UNARY(log10);
    SCALARS_COMPARE_UNARY(sin);
    SCALARS_COMPARE_UNARY(cos);
    SCALARS_COMPARE_UNARY(tan);
    SCALARS_COMPARE_UNARY(asin);
    SCALARS_COMPARE_UNARY(acos);
    SCALARS_COMPARE_UNARY(atan);
    SCALARS_COMPARE_UNARY(sinh);
    SCALARS_COMPARE_UNARY(cosh);
    SCALARS_COMPARE_UNARY(tanh);
    SCALARS_COMPARE_UNARY(asinh);
    SCALARS_COMPARE_UNARY(acosh);
    SCALARS_COMPARE_UNARY(atanh);
    SCALARS_COMPARE_UNARY(ceil);
    SCALARS_COMPARE_UNARY(floor);
    SCALARS_COMPARE_UNARY(trunc);
    SCALARS_COMPARE_UNARY(neg);
    SCALARS_COMPARE_UNARY(conj);
    SCALARS_COMPARE_UNARY(real);
    SCALARS_COMPARE_UNARY(imag);
#undef SCALARS_COMPARE_UNARY
    std::abort();
}

template <operation Op, class T>
SCALARS_NOINLINE void scalars_kernel(const complex_t<T>* a,
    const complex_t<T>*                                  b,
    const complex_t<T>*                                  c,
    complex_t<T>*                                        out,
    int                                                  count)
{
    using p = packet<T>;
    for (int i = 0; i < count; i += p::size)
    {
        const auto x = p::loadu(a + i), y = p::loadu(b + i);
        p::storeu(scalars_evaluate<Op, T>(x, y, p::loadu(c + i)), out + i);
    }
}

template <operation Op, class T>
SCALARS_NOINLINE void torch_kernel(const torch_complex_t<T>* a,
    const torch_complex_t<T>*                                b,
    const torch_complex_t<T>*                                c,
    torch_complex_t<T>*                                      out,
    int                                                      count)
{
    using p = torch_packet<T>;
    for (int i = 0; i < count; i += p::size())
    {
        const auto x = p::loadu(a + i), y = p::loadu(b + i);
        torch_evaluate<Op, T>(x, y, p::loadu(c + i)).store(out + i);
    }
}

template <operation Op, class T> void compare(const char* name, int count, int iterations)
{
    std::vector<complex_t<T>>       a(count), b(count), c(count), out(count);
    std::vector<torch_complex_t<T>> ta(count), tb(count), tc(count), tout(count);
    // NOLINTNEXTLINE(bugprone-random-generator-seed): reproducible benchmark inputs.
    std::mt19937                      random(123);
    std::uniform_real_distribution<T> sample(-2, 2);
    for (int i = 0; i < count; ++i)
    {
        a[i] = {sample(random), sample(random)};
        b[i] = {sample(random), sample(random)};
        c[i] = {sample(random), sample(random)};
        if constexpr (Op == operation::eq || Op == operation::neq)
        {
            if (i % 2 == 0)
            {
                b[i] = a[i];
            }
        }
        ta[i] = {a[i].real(), a[i].imag()};
        tb[i] = {b[i].real(), b[i].imag()};
        tc[i] = {c[i].real(), c[i].imag()};
    }
    const auto run_scalars = [&]
    { scalars_kernel<Op>(a.data(), b.data(), c.data(), out.data(), count); };
    const auto run_torch = [&]
    { torch_kernel<Op>(ta.data(), tb.data(), tc.data(), tout.data(), count); };
    run_scalars();
    run_torch();
    double error = 0;
    for (int i = 0; i < count; ++i)
    {
        const double scale =
            std::max(1., std::hypot(double(tout[i].real()), double(tout[i].imag())));
        const double difference = std::hypot(double(out[i].real()) - tout[i].real(),
                                      double(out[i].imag()) - tout[i].imag()) /
                                  scale;
        if (!std::isfinite(difference))
        {
            std::exit(2);
        }
        error = std::max(error, difference);
    }
    if (error > (std::is_same_v<T, float> ? 2e-6 : 2e-14))
    {
        std::cerr << "Numerical mismatch for " << name << ": " << error << '\n';
        std::exit(2);
    }

    const auto time = [&](auto run, const auto* output)
    {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            run();
            asm volatile("" : : "r"(output) : "memory");
        }
        return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
                   .count() /
               (double(iterations) * count);
    };
    for (int i = 0; i < 20; ++i)
    {
        run_scalars();
        run_torch();
    }
    std::array<double, 9> scalars_times, torch_times;
    for (int i = 0; i < 9; ++i)
    {
        // Alternate order to avoid always timing one implementation first.
        if (i % 2 == 0)
        {
            scalars_times[i] = time(run_scalars, out.data());
            torch_times[i]   = time(run_torch, tout.data());
        }
        else
        {
            torch_times[i]   = time(run_torch, tout.data());
            scalars_times[i] = time(run_scalars, out.data());
        }
    }
    std::sort(scalars_times.begin(), scalars_times.end());
    std::sort(torch_times.begin(), torch_times.end());
    std::cout << (std::is_same_v<T, float> ? "complex64" : "complex128") << ',' << name << ','
              << count << ',' << scalars_times[4] << ',' << torch_times[4] << ','
              << torch_times[4] / scalars_times[4] << ',' << error << '\n';
    // Stream completed rows while long comparisons are still running.
    std::cout.flush();
}

bool selected(std::string_view selection, std::string_view name)
{
    if (selection == "all")
    {
        return true;
    }
    while (!selection.empty())
    {
        const auto end = selection.find(',');
        if (selection.substr(0, end) == name)
        {
            return true;
        }
        if (end == std::string_view::npos)
        {
            return false;
        }
        selection.remove_prefix(end + 1);
    }
    return false;
}

template <class T> bool run(int count, int iterations, std::string_view selection)
{
    if (count % packet<T>::size != 0 || count % torch_packet<T>::size() != 0)
    {
        return false;
    }
    bool matched = false;
#define SCALARS_COMPARE_RUN(name)                                                                  \
    if (selected(selection, #name))                                                                \
    {                                                                                              \
        matched = true;                                                                            \
        compare<operation::name, T>(#name, count, iterations);                                     \
    }
    SCALARS_COMPARE_RUN(add);
    SCALARS_COMPARE_RUN(sub);
    SCALARS_COMPARE_RUN(mul);
    SCALARS_COMPARE_RUN(div);
    SCALARS_COMPARE_RUN(fma);
    SCALARS_COMPARE_RUN(sqr);
    SCALARS_COMPARE_RUN(abs);
    SCALARS_COMPARE_RUN(arg);
    SCALARS_COMPARE_RUN(sqrt);
    SCALARS_COMPARE_RUN(invsqrt);
    SCALARS_COMPARE_RUN(reciprocal);
    SCALARS_COMPARE_RUN(exp);
    SCALARS_COMPARE_RUN(expm1);
    SCALARS_COMPARE_RUN(exp2);
    SCALARS_COMPARE_RUN(exp10);
    SCALARS_COMPARE_RUN(log);
    SCALARS_COMPARE_RUN(log1p);
    SCALARS_COMPARE_RUN(log2);
    SCALARS_COMPARE_RUN(log10);
    SCALARS_COMPARE_RUN(sin);
    SCALARS_COMPARE_RUN(cos);
    SCALARS_COMPARE_RUN(tan);
    SCALARS_COMPARE_RUN(asin);
    SCALARS_COMPARE_RUN(acos);
    SCALARS_COMPARE_RUN(atan);
    SCALARS_COMPARE_RUN(sinh);
    SCALARS_COMPARE_RUN(cosh);
    SCALARS_COMPARE_RUN(tanh);
    SCALARS_COMPARE_RUN(asinh);
    SCALARS_COMPARE_RUN(acosh);
    SCALARS_COMPARE_RUN(atanh);
    SCALARS_COMPARE_RUN(pow);
    SCALARS_COMPARE_RUN(ceil);
    SCALARS_COMPARE_RUN(floor);
    SCALARS_COMPARE_RUN(trunc);
    SCALARS_COMPARE_RUN(neg);
    SCALARS_COMPARE_RUN(conj);
    SCALARS_COMPARE_RUN(real);
    SCALARS_COMPARE_RUN(imag);
    SCALARS_COMPARE_RUN(eq);
    SCALARS_COMPARE_RUN(neq);
#undef SCALARS_COMPARE_RUN
    return matched;
}

bool parse_positive(std::string_view text, int& value)
{
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size() && value > 0;
}
}  // namespace

int main(int argc, char** argv)
try
{
    int count = 4096, iterations = 1000;
    if ((argc > 1 && !parse_positive(argv[1], count)) ||
        (argc > 2 && !parse_positive(argv[2], iterations)))
    {
        return 1;
    }
    std::cout << std::setprecision(8)
              << "dtype,operation,elements,scalars_ns,aten_ns,speedup,max_normalized_error\n";
    const std::string_view selection = argc > 3 ? argv[3] : "add,mul,div,fma";
    return run<float>(count, iterations, selection) && run<double>(count, iterations, selection)
               ? 0
               : 1;
}
catch (const std::exception& error)
{
    std::fputs(error.what(), stderr);
    std::fputc('\n', stderr);
    return 1;
}
catch (...)
{
    std::fputs("Unexpected benchmark failure\n", stderr);
    return 1;
}
