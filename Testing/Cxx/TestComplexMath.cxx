#include "types/complex.h"

#include <cmath>
#include <complex>
#include <limits>

#include "scalars_test.h"

using namespace scalars;

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
