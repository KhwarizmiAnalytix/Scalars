#include "BenchmarkReduced.h"

#include "types/half.h"

void BM_HalfAddition(benchmark::State& state)
{
    benchmark_reduced_arithmetic<scalars::half, false, false>(state);
}
// NOLINTBEGIN(bugprone-throwing-static-initialization): Google Benchmark registration.
BENCHMARK(BM_HalfAddition)->Arg(257)->Arg(4096)->Arg(262144);

void BM_HalfMultiply(benchmark::State& state)
{
    benchmark_reduced_arithmetic<scalars::half, true, false>(state);
}
BENCHMARK(BM_HalfMultiply)->Arg(257)->Arg(4096)->Arg(262144);

#if SCALARS_VECTORIZED
BENCHMARK_TEMPLATE(benchmark_reduced_arithmetic, scalars::half, false, true)
    ->Arg(257)
    ->Arg(4096)
    ->Arg(262144);
BENCHMARK_TEMPLATE(benchmark_reduced_arithmetic, scalars::half, true, true)
    ->Arg(257)
    ->Arg(4096)
    ->Arg(262144);
#endif
// NOLINTEND(bugprone-throwing-static-initialization)
