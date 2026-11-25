// Vectorized elementary functions: simd<T> on the selected CPU tier and vector
// math library (built-in, SVML or SLEEF; SCALARS_MATH_BACKEND), against the
// scalar std:: (libm) loop and, when SCALARS_BENCHMARK_HAS_EIGEN is set,
// Eigen's packet math on the same arrays. The functions and input ranges are
// in VectorMathOps.h; AccuracyVectorMath measures the errors of the same calls.
//
// Each benchmark maps kCount inputs to a separate output array; items/s is
// elements per second of wall time. Names are <function>/<type>/<implementation>.
// An Eigen benchmark labelled "eigen-scalar" is one Eigen does not vectorize for
// that type and evaluates one element at a time through std::.
#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <random>
#include <string>
#include <type_traits>

#include "VectorMathOps.h"

#if SCALARS_VECTORIZED

namespace
{
using namespace scalars_vm;

constexpr std::size_t kCount     = 4096;
constexpr std::size_t kAlignment = 64;

template <class T> struct aligned_buffer
{
    explicit aligned_buffer(std::size_t n)
        : data(static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{kAlignment})))
    {
    }
    ~aligned_buffer() { ::operator delete(data, std::align_val_t{kAlignment}); }
    aligned_buffer(const aligned_buffer&)            = delete;
    aligned_buffer& operator=(const aligned_buffer&) = delete;

    T* data;
};

template <class T> void fill_uniform(T* out, double lo, double hi, unsigned seed)
{
    std::mt19937_64                        engine(seed);
    std::uniform_real_distribution<double> dist(lo, hi);
    for (std::size_t i = 0; i < kCount; ++i)
    {
        out[i] = static_cast<T>(dist(engine));
    }
}

template <class Op, class T> struct inputs
{
    inputs() : x(kCount), y(kCount), out(kCount)
    {
        fill_uniform(x.data, Op::lo, Op::hi, 1);
        fill_uniform(y.data, Op::lo2, Op::hi2, 2);
    }
    aligned_buffer<T> x, y, out;
};

template <class Op, class T> void bm_scalars(benchmark::State& state)
{
    using p = simd<T>;
    inputs<Op, T> in;
    for (auto _ : state)
    {
        for (std::size_t i = 0; i < kCount; i += p::size)
        {
            const auto b = Op::binary ? p::load(in.y.data + i) : p::setzero();
            p::store(Op::template packet<T>(p::load(in.x.data + i), b), in.out.data + i);
        }
        benchmark::DoNotOptimize(in.out.data);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * kCount);
    state.SetLabel(math_library());
}

template <class Op, class T> void bm_libm(benchmark::State& state)
{
    inputs<Op, T> in;
    for (auto _ : state)
    {
        for (std::size_t i = 0; i < kCount; ++i)
        {
            in.out.data[i] = Op::template libm<T>(in.x.data[i], in.y.data[i]);
        }
        benchmark::DoNotOptimize(in.out.data);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * kCount);
    state.SetLabel("libm");
}

#if SCALARS_BENCHMARK_HAS_EIGEN
template <class T>
using eigen_map = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>, Eigen::AlignedMax>;

template <class Op, class T> void bm_eigen(benchmark::State& state)
{
    inputs<Op, T>      in;
    const eigen_map<T> xm(in.x.data, kCount), ym(in.y.data, kCount);
    eigen_map<T>       om(in.out.data, kCount);
    for (auto _ : state)
    {
        om = Op::eigen(xm, ym);
        benchmark::DoNotOptimize(in.out.data);
        benchmark::ClobberMemory();
    }
    using expr_t = std::decay_t<decltype(Op::eigen(xm, ym))>;
    constexpr bool vectorized =
        (Eigen::internal::evaluator<expr_t>::Flags & Eigen::PacketAccessBit) != 0;
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * kCount);
    state.SetLabel(vectorized ? "eigen" : "eigen-scalar");
}
#endif

template <class Op, class T> void register_op()
{
    const std::string base = std::string(Op::name) + "/" + type_name<T>() + "/";
    benchmark::RegisterBenchmark((base + "scalars").c_str(), &bm_scalars<Op, T>)->UseRealTime();
    if constexpr (Op::has_libm)
    {
        benchmark::RegisterBenchmark((base + "libm").c_str(), &bm_libm<Op, T>)->UseRealTime();
    }
#if SCALARS_BENCHMARK_HAS_EIGEN
    if constexpr (Op::has_eigen)
    {
        benchmark::RegisterBenchmark((base + "eigen").c_str(), &bm_eigen<Op, T>)->UseRealTime();
    }
#endif
}

[[maybe_unused]] const bool registered = []
{
    for_each_op([](auto op) { register_op<decltype(op), float>(); });
    for_each_op([](auto op) { register_op<decltype(op), double>(); });
    return true;
}();
}  // namespace

#endif  // SCALARS_VECTORIZED
