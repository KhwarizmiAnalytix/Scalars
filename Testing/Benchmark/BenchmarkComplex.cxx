#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "backend/simd.h"
#include "math/complex_vector.h"
#include "types/complex.h"

namespace
{
enum class complex_operation : std::uint8_t
{
    add,
    multiply,
    divide,
    add_multiply,
    fma,
    magnitude,
    exponential,
    logarithm,
    root
};

template <complex_operation Op, class T>
scalars::complex<T> evaluate(scalars::complex<T> a, scalars::complex<T> b)
{
    if constexpr (Op == complex_operation::add)
    {
        return a + b;
    }
    else if constexpr (Op == complex_operation::multiply)
    {
        return a * b;
    }
    else if constexpr (Op == complex_operation::divide)
    {
        return a / b;
    }
    else if constexpr (Op == complex_operation::add_multiply)
    {
        return (a + b) * b;
    }
    else if constexpr (Op == complex_operation::fma)
    {
        return a * b + b;
    }
    else if constexpr (Op == complex_operation::magnitude)
    {
        return {scalars::abs(a), T(0)};
    }
    else if constexpr (Op == complex_operation::exponential)
    {
        return scalars::exp(a);
    }
    else if constexpr (Op == complex_operation::logarithm)
    {
        return scalars::log(a);
    }
    else
    {
        return scalars::sqrt(a);
    }
}

#if SCALARS_VECTORIZED
template <complex_operation Op, class P>
typename P::simd_t evaluate_packet(typename P::simd_t a, typename P::simd_t b)
{
    if constexpr (Op == complex_operation::add)
    {
        return P::add(a, b);
    }
    else if constexpr (Op == complex_operation::multiply)
    {
        return P::mul(a, b);
    }
    else if constexpr (Op == complex_operation::divide)
    {
        return P::div(a, b);
    }
    else if constexpr (Op == complex_operation::add_multiply)
    {
        return P::mul(P::add(a, b), b);
    }
    else if constexpr (Op == complex_operation::fma)
    {
        return P::fma(a, b, b);
    }
    else if constexpr (Op == complex_operation::magnitude)
    {
        return {P::abs(a), P::r::setzero()};
    }
    else if constexpr (Op == complex_operation::exponential)
    {
        return P::exp(a);
    }
    else if constexpr (Op == complex_operation::logarithm)
    {
        return P::log(a);
    }
    else
    {
        return P::sqrt(a);
    }
}
#endif

template <class T, complex_operation Op, bool Vectorized>
void benchmark_complex(benchmark::State& state)
{
    using c              = scalars::complex<T>;
    const auto     count = static_cast<std::size_t>(state.range(0));
    std::vector<c> left(count), right(count), output(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        left[i]  = c(T(.25) + T(i % 31) / T(32), T(-.75) + T(i % 23) / T(16));
        right[i] = c(T(1.25), T(-.5));
    }
    if constexpr (Op == complex_operation::divide)
    {
        // Ordinary packets, one tiny component per packet, or one zero divisor
        // per packet. The latter two expose exceptional-lane repair costs.
#if SCALARS_VECTORIZED
        constexpr int width = simd<c>::size;
#else
        constexpr int width = 4;
#endif
        if (state.range(1) != 0)
        {
            for (std::size_t i = 0; i < count; i += width)
            {
                right[i] = state.range(1) == 1 ? c(std::numeric_limits<T>::min(), T(1)) : c{};
            }
        }
    }
    for (auto _ : state)
    {
        std::size_t i = 0;
#if SCALARS_VECTORIZED
        if constexpr (Vectorized)
        {
            using p = simd<c>;
            for (; count - i >= p::size; i += p::size)
            {
                p::storeu(
                    evaluate_packet<Op, p>(p::loadu(left.data() + i), p::loadu(right.data() + i)),
                    output.data() + i);
            }
        }
#endif
        for (; i < count; ++i)
        {
            output[i] = evaluate<Op>(left[i], right[i]);
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
}

template <class T> void benchmark_complex_dot(benchmark::State& state)
{
    using c               = scalars::complex<T>;
    const auto     count  = static_cast<std::ptrdiff_t>(state.range(0));
    const auto     stride = static_cast<std::ptrdiff_t>(state.range(1));
    std::vector<c> left(static_cast<std::size_t>(count * stride)), right(left.size());
    for (std::size_t i = 0; i < left.size(); ++i)
    {
        left[i]  = c(T(.5) + T(i % 7) / T(8), T(-.25));
        right[i] = c(T(.75), T(i % 11) / T(16));
    }
    for (auto _ : state)
    {
        auto result = scalars::complex_dot(
            left.data(), stride, right.data(), stride, count, state.range(2) != 0);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * count);
}
}  // namespace

void BM_ComplexMultiply(benchmark::State& state)
{
    benchmark_complex<double, complex_operation::multiply, false>(state);
}

void BM_ComplexSqrt(benchmark::State& state)
{
    benchmark_complex<double, complex_operation::root, false>(state);
}

// NOLINTBEGIN(bugprone-throwing-static-initialization): Google Benchmark registration.
BENCHMARK(BM_ComplexMultiply)->Arg(257)->Arg(4096)->Arg(262144);
BENCHMARK(BM_ComplexSqrt)->Arg(257)->Arg(4096);
BENCHMARK_TEMPLATE(benchmark_complex, float, complex_operation::divide, false)
    ->Args({4096, 0})
    ->Args({4096, 1});
BENCHMARK_TEMPLATE(benchmark_complex_dot, float)
    ->Args({257, 1, 0})
    ->Args({4096, 1, 0})
    ->Args({4096, 2, 1})
    ->Args({262144, 1, 0});
BENCHMARK_TEMPLATE(benchmark_complex_dot, double)
    ->Args({257, 1, 0})
    ->Args({4096, 1, 0})
    ->Args({4096, 2, 1})
    ->Args({262144, 1, 0});
#if SCALARS_VECTORIZED
#define SCALARS_COMPLEX_BENCHMARK(op)                                                              \
    BENCHMARK_TEMPLATE(benchmark_complex, float, complex_operation::op, true)                      \
        ->Arg(257)                                                                                 \
        ->Arg(4096)                                                                                \
        ->Arg(262144);                                                                             \
    BENCHMARK_TEMPLATE(benchmark_complex, double, complex_operation::op, true)                     \
        ->Arg(257)                                                                                 \
        ->Arg(4096)                                                                                \
        ->Arg(262144)
SCALARS_COMPLEX_BENCHMARK(add);
SCALARS_COMPLEX_BENCHMARK(multiply);
SCALARS_COMPLEX_BENCHMARK(add_multiply);
SCALARS_COMPLEX_BENCHMARK(fma);
SCALARS_COMPLEX_BENCHMARK(magnitude);
SCALARS_COMPLEX_BENCHMARK(exponential);
SCALARS_COMPLEX_BENCHMARK(logarithm);
SCALARS_COMPLEX_BENCHMARK(root);
#undef SCALARS_COMPLEX_BENCHMARK
BENCHMARK_TEMPLATE(benchmark_complex, float, complex_operation::divide, true)
    ->Args({257, 0})
    ->Args({4096, 0})
    ->Args({4096, 1})
    ->Args({4096, 2});
BENCHMARK_TEMPLATE(benchmark_complex, double, complex_operation::divide, true)
    ->Args({257, 0})
    ->Args({4096, 0})
    ->Args({4096, 1})
    ->Args({4096, 2});
#endif
// NOLINTEND(bugprone-throwing-static-initialization)
