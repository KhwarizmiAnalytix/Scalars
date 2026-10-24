#include <conversion/type_cast.h>

#include "scalars_test.h"

using namespace kh::scalars;

TEST(type_cast, complex_precision)
{
    const auto value = scalar_cast<complex_double>(complex_float{1, 2});
    EXPECT_EQ(value.real(), 1.);
    EXPECT_EQ(value.imag(), 2.);
}
