#include <benchmark/benchmark.h>

#include "types/bfloat16.h"

void BM_BFloat16Addition(benchmark::State& state)
{
    scalars::bfloat16 left(1.25f);
    scalars::bfloat16 right(2.5f);
    for (auto _ : state)
    {
        left += right;
        benchmark::DoNotOptimize(left.x);
    }
}
BENCHMARK(BM_BFloat16Addition);

void BM_BFloat16Multiply(benchmark::State& state)
{
    scalars::bfloat16 left(1.25f);
    scalars::bfloat16 right(0.5f);
    for (auto _ : state)
    {
        left *= right;
        left += scalars::bfloat16(1.f);
        benchmark::DoNotOptimize(left.x);
    }
}
BENCHMARK(BM_BFloat16Multiply);
