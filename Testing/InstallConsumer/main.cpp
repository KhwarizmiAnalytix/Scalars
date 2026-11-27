#include "backend/simd.h"
#include "math/complex_vector.h"
#include "scalars.h"

#include <array>
#include <cmath>

#if SCALARS_VECTORIZED
template <class T> bool check_reduced_simd()
{
    using s = scalars::simd<T>;
    std::array<T, s::size> output;
    s::storeu(s::add(s::set(T(1.25f)), s::set(T(2.5f))), output.data());
    for (auto value : output)
    {
        if (static_cast<float>(value) != 3.75f)
        {
            return false;
        }
    }
    return true;
}

// Exercises the SVML/SLEEF/Accelerate-backed exp/log/sin entry points from a
// real external consumer of the installed package, so a missing or
// BUILD_INTERFACE-only math-library link (SVML, MKL, SLEEF) shows up here as
// a link failure instead of only inside this repository's own build.
template <class T> bool check_simd_math()
{
    using s = scalars::simd<T>;
    std::array<T, s::size> input;
    for (int i = 0; i < s::size; ++i)
    {
        input[static_cast<std::size_t>(i)] = T(0.5) + T(i) * T(0.1);
    }
    const auto             packet = s::loadu(input.data());
    std::array<T, s::size> exp_out;
    std::array<T, s::size> log_out;
    std::array<T, s::size> sin_out;
    s::storeu(s::exp(packet), exp_out.data());
    s::storeu(s::log(packet), log_out.data());
    s::storeu(s::sin(packet), sin_out.data());
    for (int i = 0; i < s::size; ++i)
    {
        const double x         = static_cast<double>(input[static_cast<std::size_t>(i)]);
        const double tolerance = 1e-3;
        if (std::fabs(static_cast<double>(exp_out[static_cast<std::size_t>(i)]) - std::exp(x)) >
            tolerance)
        {
            return false;
        }
        if (std::fabs(static_cast<double>(log_out[static_cast<std::size_t>(i)]) - std::log(x)) >
            tolerance)
        {
            return false;
        }
        if (std::fabs(static_cast<double>(sin_out[static_cast<std::size_t>(i)]) - std::sin(x)) >
            tolerance)
        {
            return false;
        }
    }
    return true;
}
#endif

int main()
{
    using namespace scalars;
    const half value{1.0f};
    using c             = scalars::complex<double>;
    const c    values[] = {c(1, 2), c(3, -4), c(2, 1)};
    const auto dot      = scalars::complex_dot(values, 1, values, 1, 3, true);
    if (dot.real() != 35 || dot.imag() != 0)
    {
        return 1;
    }
#if SCALARS_VECTORIZED
    if (!check_reduced_simd<half>() || !check_reduced_simd<bfloat16>())
    {
        return 1;
    }
    if (!check_simd_math<float>() || !check_simd_math<double>())
    {
        return 1;
    }
#endif
    return float(value) == 1.f && scalar_type_size(scalar_type::complex_double) == 16 ? 0 : 1;
}
