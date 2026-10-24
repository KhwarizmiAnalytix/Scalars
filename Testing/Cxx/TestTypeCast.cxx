#include "common/type_cast.h"

#include "scalars_test.h"

using namespace scalars;

TEST(TypeCast, RealAndComplex)
{
    EXPECT_DOUBLE_EQ(scalar_cast<double>(1.5f), 1.5);
    const auto promoted = scalar_cast<complex<double>>(1.5f);
    EXPECT_DOUBLE_EQ(promoted.real(), 1.5);
    EXPECT_DOUBLE_EQ(promoted.imag(), 0.0);
    EXPECT_FLOAT_EQ(scalar_cast<float>(complex<double>{1.5, 2.5}), 1.5f);
    const auto widened = scalar_cast<complex<double>>(complex<float>{1.0f, 2.0f});
    EXPECT_DOUBLE_EQ(widened.real(), 1.0);
    EXPECT_DOUBLE_EQ(widened.imag(), 2.0);
}
