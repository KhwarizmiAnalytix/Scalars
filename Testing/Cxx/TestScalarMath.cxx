#include "math/scalar_math.h"

#include <cmath>

#include "scalars_test.h"

using namespace scalars;

namespace
{
template <class T> void expect_rounded(T actual, float reference)
{
    EXPECT_FLOAT_EQ(static_cast<float>(actual), static_cast<float>(T(reference)));
}

template <class T> void exercise(T value)
{
    const float input = static_cast<float>(value);
    expect_rounded(exp(value), std::exp(input));
    expect_rounded(log(value), std::log(input));
    expect_rounded(sqrt(value), std::sqrt(input));
    expect_rounded(sin(value), std::sin(input));
    expect_rounded(cos(value), std::cos(input));
    expect_rounded(tan(value), std::tan(input));
    expect_rounded(sinh(value), std::sinh(input));
    expect_rounded(cosh(value), std::cosh(input));
    expect_rounded(tanh(value), std::tanh(input));
    expect_rounded(floor(value), std::floor(input));
    expect_rounded(ceil(value), std::ceil(input));
    expect_rounded(round(value), std::round(input));
    expect_rounded(trunc(value), std::trunc(input));
}
}  // namespace

TEST(ScalarMath, RealHalfAndBFloat16)
{
    exercise(half(1.5f));
    exercise(bfloat16(1.5f));
}
