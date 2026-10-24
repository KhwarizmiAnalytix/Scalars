#include "backend/simd.h"
#include "math/complex_vector.h"
#include "scalars.h"

#include <array>

#if SCALARS_VECTORIZED
template <class T> bool check_reduced_simd()
{
    using S = simd<T>;
    std::array<T, S::size> output;
    S::storeu(S::add(S::set(T(1.25f)), S::set(T(2.5f))), output.data());
    for (auto value : output)
        if (static_cast<float>(value) != 3.75f)
            return false;
    return true;
}
#endif

int main()
{
    using namespace scalars;
    const half value{1.0f};
    using C             = scalars::complex<double>;
    const C    values[] = {C(1, 2), C(3, -4), C(2, 1)};
    const auto dot      = scalars::complex_dot(values, 1, values, 1, 3, true);
    if (dot.real() != 35 || dot.imag() != 0)
        return 1;
#if SCALARS_VECTORIZED
    if (!check_reduced_simd<half>() || !check_reduced_simd<bfloat16>())
        return 1;
#endif
    return float(value) == 1.f && scalar_type_size(scalar_type::complex_double) == 16 ? 0 : 1;
}
