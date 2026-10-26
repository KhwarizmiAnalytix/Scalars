#include "BenchmarkReduced.h"

#include "types/bfloat16.h"

void BM_BFloat16Addition(benchmark::State& state)
{
    benchmark_reduced_arithmetic<scalars::bfloat16, false, false>(state);
}
// NOLINTBEGIN(bugprone-throwing-static-initialization): Google Benchmark registration.
BENCHMARK(BM_BFloat16Addition)->Arg(257)->Arg(4096)->Arg(262144);

void BM_BFloat16Multiply(benchmark::State& state)
{
    benchmark_reduced_arithmetic<scalars::bfloat16, true, false>(state);
}
BENCHMARK(BM_BFloat16Multiply)->Arg(257)->Arg(4096)->Arg(262144);

#if SCALARS_VECTORIZED
BENCHMARK_TEMPLATE(benchmark_reduced_arithmetic, scalars::bfloat16, false, true)
    ->Arg(257)
    ->Arg(4096)
    ->Arg(262144);
BENCHMARK_TEMPLATE(benchmark_reduced_arithmetic, scalars::bfloat16, true, true)
    ->Arg(257)
    ->Arg(4096)
    ->Arg(262144);
#endif
// NOLINTEND(bugprone-throwing-static-initialization)
