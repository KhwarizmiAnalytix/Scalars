#pragma once

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "backend/simd.h"

// Bounded inputs and a separate output avoid saturation and fixed points.
template <class T, bool Multiply, bool Vectorized>
void benchmark_reduced_arithmetic(benchmark::State& state)
{
    const auto     count = static_cast<std::size_t>(state.range(0));
    std::vector<T> left(count), right(count), output(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        left[i]  = T(.5f + float(i % 31) / 32.f);
        right[i] = T(-.75f + float(i % 23) / 16.f);
    }
    for (auto _ : state)
    {
        std::size_t i = 0;
#if SCALARS_VECTORIZED
        if constexpr (Vectorized)
        {
            using p = simd<T>;
            for (; count - i >= p::size; i += p::size)
            {
                const auto a = p::loadu(left.data() + i), b = p::loadu(right.data() + i);
                p::storeu(Multiply ? p::mul(a, b) : p::add(a, b), output.data() + i);
            }
        }
#endif
        for (; i < count; ++i)
        {
            output[i] = Multiply ? left[i] * right[i] : left[i] + right[i];
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(3 * count * sizeof(T)));
}
