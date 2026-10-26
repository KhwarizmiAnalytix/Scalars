#include <benchmark/benchmark.h>

#include "scalars_macros.h"

#if SCALARS_HAS_METAL
#include "backend/gpu/metal/metal_dispatch.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace
{
namespace mb       = scalars::metal_backend;
using metal_buffer = std::unique_ptr<void, decltype(&mb::deallocate)>;

void benchmark_metal_launches(benchmark::State& state)
{
    if (!mb::device_available())
    {
        state.SkipWithMessage("No Metal device");
        return;
    }
    const auto   count = static_cast<std::size_t>(state.range(0));
    metal_buffer a(mb::allocate(count * sizeof(float)), mb::deallocate);
    metal_buffer b(mb::allocate(count * sizeof(float)), mb::deallocate);
    mb::dispatch_fill(a.get(), 1.f, count);
    const void* warmup[] = {a.get()};
    mb::dispatch("neg", warmup, 1, b.get(), count);
    mb::execution_context context;
    for (auto _ : state)
    {
        for (int i = 0; i < 8; ++i)
        {
            const void* input[] = {i % 2 == 0 ? a.get() : b.get()};
            void*       output  = i % 2 == 0 ? b.get() : a.get();
            if (state.range(1) != 0)
            {
                context.dispatch("neg", input, 1, output, count);
            }
            else
            {
                mb::dispatch("neg", input, 1, output, count);
            }
        }
        context.wait();
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(8 * count));
}

void benchmark_metal_reduction(benchmark::State& state)
{
    if (!mb::device_available())
    {
        state.SkipWithMessage("No Metal device");
        return;
    }
    const auto   count = static_cast<std::size_t>(state.range(0));
    metal_buffer input(mb::allocate(count * sizeof(float)), mb::deallocate);
    mb::dispatch_fill(input.get(), 1.f, count);
    auto sum = mb::reduce_sum(input.get(), count);
    benchmark::DoNotOptimize(sum);
    for (auto _ : state)
    {
        sum = mb::reduce_sum(input.get(), count);
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
}
}  // namespace

// NOLINTBEGIN(bugprone-throwing-static-initialization): Google Benchmark registration.
BENCHMARK(benchmark_metal_launches)
    ->Args({4096, 0})
    ->Args({4096, 1})
    ->Args({262144, 0})
    ->Args({262144, 1})
    ->UseRealTime();
BENCHMARK(benchmark_metal_reduction)->Arg(257)->Arg(65537)->Arg(1048576)->UseRealTime();
// NOLINTEND(bugprone-throwing-static-initialization)
#endif
