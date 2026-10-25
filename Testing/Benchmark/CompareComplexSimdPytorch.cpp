// Standalone packet comparison; built by Scripts/benchmark_complex_simd.py.
#include "backend/simd.h"
#include <cstdlib>

// These compatibility aliases belong to different libraries.
#undef HIDDEN_NAMESPACE_BEGIN
#undef HIDDEN_NAMESPACE_END
#include <ATen/cpu/vec/vec.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace
{
template <class T> using complex_t       = scalars::complex<T>;
template <class T> using torch_complex_t = c10::complex<T>;
template <class T> using packet          = simd<complex_t<T>>;
template <class T> using torch_packet    = at::vec::Vectorized<torch_complex_t<T>>;

template <int Op, class T>
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
        if constexpr (Op == 0)
            p::storeu(p::add(x, y), out + i);
        if constexpr (Op == 1)
            p::storeu(p::mul(x, y), out + i);
        if constexpr (Op == 2)
            p::storeu(p::div(x, y), out + i);
        if constexpr (Op == 3)
            p::storeu(p::fma(x, y, p::loadu(c + i)), out + i);
    }
}

template <int Op, class T>
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
        if constexpr (Op == 0)
            (x + y).store(out + i);
        if constexpr (Op == 1)
            (x * y).store(out + i);
        if constexpr (Op == 2)
            (x / y).store(out + i);
        if constexpr (Op == 3)
            at::vec::fmadd(x, y, p::loadu(c + i)).store(out + i);
    }
}

template <int Op, class T> void compare(const char* name, int count, int iterations)
{
    std::vector<complex_t<T>>         a(count), b(count), c(count), out(count);
    std::vector<torch_complex_t<T>>   ta(count), tb(count), tc(count), tout(count);
    std::mt19937                      random(123);
    std::uniform_real_distribution<T> sample(-2, 2);
    for (int i = 0; i < count; ++i)
    {
        a[i]  = {sample(random), sample(random)};
        b[i]  = {sample(random), sample(random)};
        c[i]  = {sample(random), sample(random)};
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
            std::exit(2);
        error = std::max(error, difference);
    }
    if (error > (std::is_same_v<T, float> ? 2e-6 : 2e-14))
        std::exit(2);

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
}

template <class T> void run(int count, int iterations)
{
    if (count % packet<T>::size != 0 || count % torch_packet<T>::size() != 0)
        std::exit(1);
    compare<0, T>("add", count, iterations);
    compare<1, T>("mul", count, iterations);
    compare<2, T>("div", count, iterations);
    compare<3, T>("fma", count, iterations);
}
}  // namespace

int main(int argc, char** argv)
{
    const int count      = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 1000;
    if (count <= 0 || iterations <= 0)
        return 1;
    std::cout << std::setprecision(8)
              << "dtype,operation,elements,scalars_ns,aten_ns,speedup,max_normalized_error\n";
    run<float>(count, iterations);
    run<double>(count, iterations);
}
