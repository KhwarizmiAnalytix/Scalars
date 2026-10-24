#include <core/type_promotion.h>
#include <type_traits>

#include "scalars_test.h"

using namespace kh::scalars;

TEST(type_promotion, reduced_precision)
{
    static_assert(std::is_same_v<promote_t<half, bfloat16>, float>);
}

TEST(type_promotion, complex_and_real)
{
    static_assert(std::is_same_v<promote_t<complex<half>, double>, complex<double>>);
}
