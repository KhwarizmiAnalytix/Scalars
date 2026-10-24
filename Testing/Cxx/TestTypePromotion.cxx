#include "core/type_promotion.h"

#include <type_traits>

#include "scalars_test.h"

using namespace scalars;

TEST(TypePromotion, RealAndComplex)
{
    static_assert(std::is_same_v<promote_t<half, half>, half>);
    static_assert(std::is_same_v<promote_t<half, bfloat16>, float>);
    static_assert(std::is_same_v<promote_t<half, float>, float>);
    static_assert(std::is_same_v<promote_t<float, double>, double>);
    static_assert(std::is_same_v<promote_t<float, float>, float>);
    static_assert(std::is_same_v<promote_t<complex<half>, complex<bfloat16>>, complex<float>>);
    static_assert(std::is_same_v<promote_t<complex<half>, double>, complex<double>>);
    static_assert(std::is_same_v<promote_t<float, complex<double>>, complex<double>>);
    static_assert(std::is_same_v<promote_t<complex<float>, complex<float>>, complex<float>>);
    EXPECT_TRUE((std::is_same_v<promote_t<half, bfloat16>, float>));
}
