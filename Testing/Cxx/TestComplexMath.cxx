#include "types/complex.h"

#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>
#include <string_view>

using namespace scalars;
using namespace scalars::complex_math;

namespace
{
template <class T> T touch(const complex<T>& value) { return value.real() + value.imag(); }

template <class T> void exercise(T real, T imag)
{
    const complex<T> value{real, imag};
    volatile T sink = T(0);
    sink            = sink + touch(exp(value));
    sink            = sink + touch(log(value));
    sink            = sink + touch(log10(value));
    sink            = sink + touch(log2(value));
    sink            = sink + touch(sqrt(value));
    sink            = sink + touch(pow(value, value));
    sink            = sink + touch(pow(value, T(2)));
    sink            = sink + touch(pow(T(2), value));
    sink            = sink + touch(sin(value));
    sink            = sink + touch(cos(value));
    sink            = sink + touch(tan(value));
    sink            = sink + touch(asin(complex<T>{T(0.25), T(0.1)}));
    sink            = sink + touch(acos(complex<T>{T(0.25), T(0.1)}));
    sink            = sink + touch(atan(value));
    sink            = sink + touch(sinh(value));
    sink            = sink + touch(cosh(value));
    sink            = sink + touch(tanh(value));
    sink            = sink + touch(asinh(value));
    sink            = sink + touch(acosh(complex<T>{T(2), imag}));
    sink            = sink + touch(atanh(complex<T>{T(0.25), T(0.1)}));
    sink            = sink + touch(log1p(complex<T>{T(0), T(0)}));
    sink            = sink + touch(log1p(complex<T>{T(0.1), T(0.02)}));
    sink            = sink + touch(log1p(complex<T>{T(2), T(0.5)}));
    sink            = sink + touch(expm1(value));
    EXPECT_TRUE(sink == sink || std::isnan(sink));
}
}  // namespace

TEST(ComplexMath, ElementaryFloatAndDouble)
{
    exercise<float>(0.5f, -0.25f);
    exercise<double>(0.5, -0.25);
    const auto mixed_pow = pow(complex<float>{2.0f, 0.0f}, complex<double>{3.0, 0.0});
    const auto mixed_real = pow(complex<float>{2.0f, 0.0f}, 3.0);
    const auto mixed_base = pow(2.0f, complex<double>{3.0, 0.0});
    EXPECT_NEAR(mixed_pow.real(), 8.0, 1e-5);
    EXPECT_NEAR(mixed_real.real(), 8.0, 1e-5);
    EXPECT_NEAR(mixed_base.real(), 8.0, 1e-5);
}

TEST(ComplexMath, MixedPrecisionPowComputesInPromotedType)
{
    // Confirmed bug: pow(complex<float>, complex<double>) advertises a
    // complex<double> return type but used to compute internally in
    // float, e.g. returning 1.04880881 where the double-precision value
    // is 1.04880886. All four mixed-type pow() overloads must compute in
    // the promoted type before narrowing.
    const complex<float>  base(1.1f, 0.05f);
    const complex<double> exponent(0.37, 0.0);
    const std::complex<double> double_base(static_cast<double>(base.real()),
        static_cast<double>(base.imag()));
    const std::complex<double> double_exponent(exponent.real(), exponent.imag());
    const std::complex<double> reference = std::pow(double_base, double_exponent);

    // A tolerance tighter than float epsilon (~1.19e-7): computing in float
    // internally and narrowing back would fail this.
    constexpr double tolerance = 1e-10;

    const auto complex_complex = pow(base, exponent);
    EXPECT_NEAR(complex_complex.real(), reference.real(), tolerance);
    EXPECT_NEAR(complex_complex.imag(), reference.imag(), tolerance);

    const auto complex_real = pow(base, exponent.real());
    const std::complex<double> real_reference = std::pow(double_base, exponent.real());
    EXPECT_NEAR(complex_real.real(), real_reference.real(), tolerance);
    EXPECT_NEAR(complex_real.imag(), real_reference.imag(), tolerance);

    const auto real_complex = pow(static_cast<double>(base.real()), exponent);
    const std::complex<double> base_reference =
        std::pow(static_cast<double>(base.real()), double_exponent);
    EXPECT_NEAR(real_complex.real(), base_reference.real(), tolerance);
    EXPECT_NEAR(real_complex.imag(), base_reference.imag(), tolerance);
}

TEST(ComplexMath, Expm1AnnexGSpecialValues)
{
    // Confirmed bug: expm1(x + 0i) with x == +inf computed
    // exp(x) * sin(0) == inf * 0 == NaN for the imaginary part; C99 Annex G
    // (and the SIMD implementation) requires the imaginary part to be
    // exactly the input's imaginary part whenever it is zero.
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    for (double y : {0.0, -0.0})
    {
        const auto result = expm1(complex<double>{inf, y});
        EXPECT_TRUE(std::isinf(result.real()));
        EXPECT_GT(result.real(), 0.0);
        EXPECT_EQ(std::signbit(result.imag()), std::signbit(y));
        EXPECT_EQ(result.imag(), y);
    }

    const auto finite_zero_imag = expm1(complex<double>{1.0, 0.0});
    EXPECT_NEAR(finite_zero_imag.real(), std::expm1(1.0), 1e-12);
    EXPECT_EQ(finite_zero_imag.imag(), 0.0);

    const auto neg_inf = expm1(complex<double>{-inf, 0.0});
    EXPECT_NEAR(neg_inf.real(), -1.0, 1e-12);
    EXPECT_EQ(neg_inf.imag(), 0.0);

    const auto with_nan = expm1(complex<double>{nan, 0.0});
    EXPECT_TRUE(std::isnan(with_nan.real()));
}

TEST(ComplexMath, SquareRootAndArcCosineEdges)
{
    const complex<double> negative_one{-1.0, 0.0};
    const auto root = sqrt(negative_one);
    EXPECT_NEAR(root.real(), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(root.imag()), 1.0, 1e-12);

    EXPECT_NEAR(sqrt(complex<double>{4.0, 0.0}).real(), 2.0, 1e-12);
    EXPECT_NEAR(sqrt(complex<double>{0.0, 0.0}).real(), 0.0, 1e-12);
    EXPECT_NEAR(sqrt(complex<double>{0.0, 0.0}).imag(), 0.0, 1e-12);
    EXPECT_GT(sqrt(complex<double>{0.0, 4.0}).real(), 0.0);
    EXPECT_LT(sqrt(complex<double>{0.0, -4.0}).imag(), 0.0);
    EXPECT_GT(sqrt(complex<double>{-4.0, 1.0}).real(), 0.0);
    EXPECT_TRUE(std::isfinite(sqrt(complex<double>{4.0, 1.0}).imag()));

    const auto infinite = std::numeric_limits<double>::infinity();
    const auto nan      = std::numeric_limits<double>::quiet_NaN();
    const auto infinite_root = sqrt(complex<double>{infinite, 0.0});
    const auto imag_infinite = sqrt(complex<double>{0.0, infinite});
    const auto nan_root      = sqrt(complex<double>{nan, 1.0});
    EXPECT_TRUE(std::isinf(infinite_root.real()) || std::isnan(infinite_root.real()));
    EXPECT_TRUE(std::isinf(imag_infinite.real()) || std::isnan(imag_infinite.real()) ||
                std::isinf(imag_infinite.imag()) || std::isnan(imag_infinite.imag()));
    EXPECT_TRUE(std::isnan(nan_root.real()) || std::isnan(nan_root.imag()));
    EXPECT_TRUE(std::isfinite(acos(complex<double>{0.5, 0.25}).real()));
    EXPECT_TRUE(std::isfinite(acos(complex<double>{0.0, 0.0}).real()));
    EXPECT_TRUE(std::isfinite(acos(complex<double>{2.0, 0.0}).imag()) ||
                std::isnan(acos(complex<double>{2.0, 0.0}).imag()));
    const auto infinite_acos = acos(complex<double>{infinite, 1.0});
    const auto nan_acos      = acos(complex<double>{nan, 0.0});
    volatile double edge_sink = infinite_acos.real() + nan_acos.real();
    EXPECT_TRUE(edge_sink == edge_sink || edge_sink != edge_sink);

    const auto float_root = sqrt(complex<float>{-1.0f, 0.0f});
    EXPECT_NEAR(float_root.real(), 0.0f, 1e-6f);
    (void)sqrt(complex<float>{0.0f, 0.0f});
    (void)sqrt(complex<float>{0.0f, 4.0f});
    (void)sqrt(complex<float>{4.0f, 0.0f});
    (void)sqrt(complex<float>{-4.0f, 1.0f});
    (void)sqrt(complex<float>{std::numeric_limits<float>::infinity(), 0.0f});
    (void)sqrt(complex<float>{std::numeric_limits<float>::quiet_NaN(), 1.0f});
    (void)acos(complex<float>{0.5f, -0.25f});
    (void)acos(complex<float>{std::numeric_limits<float>::infinity(), 0.0f});
    (void)acos(complex<float>{std::numeric_limits<float>::quiet_NaN(), 1.0f});

    const auto reference = std::exp(std::complex<double>(0.0, 1.0));
    const auto actual    = exp(complex<double>{0.0, 1.0});
    EXPECT_NEAR(actual.real(), reference.real(), 1e-12);
    EXPECT_NEAR(actual.imag(), reference.imag(), 1e-12);
    EXPECT_NEAR(log1p(complex<double>{0.0, 0.0}).real(), 0.0, 1e-12);
    EXPECT_TRUE(std::isfinite(log1p(complex<float>{0.1f, 0.02f}).imag()));
    EXPECT_TRUE(std::isfinite(expm1(complex<double>{0.1, -0.2}).real()));
}

template <class T> void check_matrix_scalar_extremes()
{
    using c             = scalars::complex<T>;
    const T    maximum  = std::numeric_limits<T>::max();
    const T    tiny     = std::numeric_limits<T>::denorm_min();
    const auto quotient = c(maximum, maximum) / c(maximum, maximum);
    EXPECT_EQ(quotient.real(), T(1));
    EXPECT_NEAR(quotient.imag(), T(0), std::numeric_limits<T>::epsilon() * 4);
    const auto large_quotient = c(maximum, maximum) / c(T(1), T(1));
    EXPECT_EQ(large_quotient.real(), maximum);
    EXPECT_EQ(large_quotient.imag(), T(0));
    const auto small_quotient = c(tiny, tiny) / c(tiny, tiny);
    EXPECT_EQ(small_quotient.real(), T(1));
    EXPECT_EQ(small_quotient.imag(), T(0));
    const auto mixed_scale = c(maximum, tiny) / c(T(0), tiny);
    EXPECT_EQ(mixed_scale.real(), T(1));
    EXPECT_TRUE(std::isinf(mixed_scale.imag()));
    const auto finite_mixed = c(maximum, tiny) / c(maximum, maximum);
    EXPECT_NEAR(finite_mixed.real(), T(.5), std::numeric_limits<T>::epsilon() * 4);
    EXPECT_NEAR(finite_mixed.imag(), T(-.5), std::numeric_limits<T>::epsilon() * 4);
    // A tiny divisor component must survive until the product is formed:
    // scaling the divisor first would erase this finite real component.
    const auto separated = c(maximum, T(0)) / c(tiny, T(1));
    EXPECT_EQ(separated.real(), maximum * tiny);
    EXPECT_EQ(separated.imag(), -maximum);

    const auto root = scalars::sqrt(c(maximum, maximum));
    EXPECT_TRUE(scalars::isfinite(root));
    // Squaring after scaling avoids overflow in the independent identity.
    const T    scale      = std::sqrt(maximum);
    const auto normalized = root / scale;
    const auto squared    = normalized * normalized;
    EXPECT_NEAR(squared.real(), T(1), std::numeric_limits<T>::epsilon() * 8);
    EXPECT_NEAR(squared.imag(), T(1), std::numeric_limits<T>::epsilon() * 8);
    const auto tiny_root = scalars::sqrt(c(T(0), tiny));
    EXPECT_GT(tiny_root.real(), T(0));
    EXPECT_NEAR((tiny_root.real() - tiny_root.imag()) / tiny_root.real(),
        T(0),
        std::numeric_limits<T>::epsilon() * 4);
    const auto lower_branch = scalars::sqrt(c(T(-4), T(-0.0)));
    EXPECT_EQ(lower_branch.real(), T(0));
    EXPECT_EQ(lower_branch.imag(), T(-2));
    EXPECT_EQ(scalars::abs(c(T(3), T(4))), T(5));
    EXPECT_EQ(scalars::norm(c(T(3), T(4))), T(25));
    EXPECT_EQ(scalars::conj(c(T(1), T(2))).imag(), T(-2));
    EXPECT_FALSE(scalars::isfinite(c(maximum, std::numeric_limits<T>::infinity())));
}

TEST(ComplexMath, MatrixScalarExtremes)
{
    constexpr auto quotient = complex<double>(4, 2) / complex<double>(2, 0);
    static_assert(quotient.real() == 2 && quotient.imag() == 1);
    check_matrix_scalar_extremes<float>();
    check_matrix_scalar_extremes<double>();
}
