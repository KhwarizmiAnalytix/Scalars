#include <cmath>
#include <math/complex_math.h>

#include "scalars_test.h"

using namespace kh::scalars;

TEST(complex_math, exponential)
{
    const auto value = kh::scalars::exp(complex_double{0., 1.});
    EXPECT_NEAR(value.real(), std::cos(1.), 1e-14);
    EXPECT_NEAR(value.imag(), std::sin(1.), 1e-14);
}

TEST(complex_math, square_root)
{
    EXPECT_NEAR(kh::scalars::sqrt(complex_double{-1., 0.}).imag(), 1., 1e-14);
}
