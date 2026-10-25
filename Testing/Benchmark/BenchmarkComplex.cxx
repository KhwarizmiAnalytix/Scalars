#include <benchmark/benchmark.h>

#include "types/complex.h"

void BM_ComplexMultiply(benchmark::State& state)
{
    scalars::complex<double> left{1.25, -0.5};
    scalars::complex<double> right{0.75, 1.5};
    for (auto _ : state)
    {
        left *= right;
        left += scalars::complex<double>{0.01, -0.01};
        benchmark::DoNotOptimize(left.real());
        benchmark::DoNotOptimize(left.imag());
    }
}
BENCHMARK(BM_ComplexMultiply);  // NOLINT(bugprone-throwing-static-initialization)

void BM_ComplexSqrt(benchmark::State& state)
{
    scalars::complex<double> value{-2.0, 3.5};
    for (auto _ : state)
    {
        auto result = scalars::sqrt(value);
        benchmark::DoNotOptimize(result.real());
        benchmark::DoNotOptimize(result.imag());
    }
}
BENCHMARK(BM_ComplexSqrt);  // NOLINT(bugprone-throwing-static-initialization)
