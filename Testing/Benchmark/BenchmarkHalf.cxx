#include <benchmark/benchmark.h>

#include "types/half.h"

void BM_HalfAddition(benchmark::State& state)
{
    scalars::half left(1.25f);
    scalars::half right(2.5f);
    for (auto _ : state)
    {
        left += right;
        benchmark::DoNotOptimize(left.x);
    }
}
BENCHMARK(BM_HalfAddition);

void BM_HalfMultiply(benchmark::State& state)
{
    scalars::half left(1.25f);
    scalars::half right(0.5f);
    for (auto _ : state)
    {
        left *= right;
        left += scalars::half(1.f);
        benchmark::DoNotOptimize(left.x);
    }
}
BENCHMARK(BM_HalfMultiply);
