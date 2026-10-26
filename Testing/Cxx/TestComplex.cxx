#include "types/bfloat16.h"
#include "types/complex.h"
#include "types/half.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>

using namespace scalars;

namespace
{
template <class T> T real_of(const complex<T>& value) { return value.real(); }
}  // namespace

TEST(Complex, StorageAndProduct)
{
    static_assert(sizeof(complex<float>) == 8);
    static_assert(sizeof(complex<double>) == 16);
    const complex<float> product = complex<float>{1.0f, 2.0f} * complex<float>{3.0f, 4.0f};
    EXPECT_FLOAT_EQ(product.real(), -5.0f);
    EXPECT_FLOAT_EQ(product.imag(), 10.0f);
}

TEST(Complex, ConstructorsAssignmentAndPredicates)
{
    complex<double> value;
    EXPECT_DOUBLE_EQ(value.real(), 0.0);
    EXPECT_DOUBLE_EQ(value.imag(), 0.0);
    EXPECT_FALSE(static_cast<bool>(value));

    value = 4.0;
    value.real(1.5);
    value.imag(-2.5);
    EXPECT_TRUE(static_cast<bool>(value));
    EXPECT_DOUBLE_EQ(std::real(value), 1.5);
    EXPECT_DOUBLE_EQ(std::imag(value), -2.5);

    const complex<float> narrowed(value);
    const complex<double> widened = complex<float>{1.0f, 2.0f};
    EXPECT_FLOAT_EQ(narrowed.real(), 1.5f);
    EXPECT_DOUBLE_EQ(widened.imag(), 2.0);

    const std::complex<double> standard{3.0, 4.0};
    complex<double> from_std(standard);
    from_std = standard;
    EXPECT_DOUBLE_EQ(from_std.real(), 3.0);
    EXPECT_DOUBLE_EQ(from_std.imag(), 4.0);
    const std::complex<double> back = static_cast<std::complex<double>>(from_std);
    EXPECT_DOUBLE_EQ(back.real(), 3.0);

    value += 1.0;
    value -= 0.5;
    value *= 2.0;
    value /= 2.0;
    EXPECT_NEAR(value.real(), 2.0, 1e-12);

    value = complex<float>{1.0f, 2.0f};
    value += complex<float>{1.0f, -1.0f};
    value -= complex<float>{0.5f, 0.5f};
    value *= complex<float>{2.0f, 0.0f};
    EXPECT_NEAR(value.real(), 3.0, 1e-6);
    EXPECT_NEAR(value.imag(), 1.0, 1e-6);
}

TEST(Complex, ArithmeticDivisionAndIntegers)
{
    const complex<double> left{1.0, -2.0};
    EXPECT_DOUBLE_EQ((+left).real(), 1.0);
    EXPECT_DOUBLE_EQ((-left).imag(), 2.0);

    const auto sum = left + complex<double>{3.0, 4.0};
    const auto difference = left - complex<double>{0.5, -0.5};
    const auto product = left * complex<double>{2.0, 1.0};
    EXPECT_DOUBLE_EQ(sum.real(), 4.0);
    EXPECT_DOUBLE_EQ(difference.imag(), -1.5);
    EXPECT_DOUBLE_EQ(product.real(), 4.0);

    const auto same_sign = left / complex<double>{2.0, -0.5};
    const std::complex<double> same_expected =
        std::complex<double>(1.0, -2.0) / std::complex<double>(2.0, -0.5);
    EXPECT_NEAR(same_sign.real(), same_expected.real(), 1e-12);
    EXPECT_NEAR(same_sign.imag(), same_expected.imag(), 1e-12);

    complex<double> negative_divisor = left;
    negative_divisor /= complex<double>{-4.0, -0.5};
    complex<double> steep = left;
    steep /= complex<double>{0.25, 3.0};
    steep /= complex<double>{-0.25, -4.0};
    EXPECT_TRUE(std::isfinite(negative_divisor.real()));
    EXPECT_TRUE(std::isfinite(steep.imag()));

    const auto by_real = left / 2.0;
    const auto into    = 2.0 / left;
    const auto into_reference =
        std::complex<double>(2.0) / std::complex<double>(left.real(), left.imag());
    EXPECT_DOUBLE_EQ(by_real.real(), 0.5);
    EXPECT_NEAR(into.real(), into_reference.real(), 1e-12);
    EXPECT_NEAR(into.imag(), into_reference.imag(), 1e-12);

    const auto real_sum = left + 1.5;
    const auto real_from_left = 1.5 + left;
    const auto real_diff = left - 0.5;
    const auto real_from_right = 4.0 - left;
    const auto real_product = left * 2.0;
    const auto real_from_product = 2.0 * left;
    EXPECT_DOUBLE_EQ(real_sum.real(), 2.5);
    EXPECT_DOUBLE_EQ(real_from_left.imag(), -2.0);
    EXPECT_DOUBLE_EQ(real_diff.real(), 0.5);
    EXPECT_DOUBLE_EQ(real_from_right.real(), 3.0);
    EXPECT_DOUBLE_EQ(real_product.imag(), -4.0);
    EXPECT_DOUBLE_EQ(real_from_product.real(), 2.0);

    EXPECT_DOUBLE_EQ((left + 2).real(), 3.0);
    EXPECT_DOUBLE_EQ((2 + left).imag(), -2.0);
    EXPECT_DOUBLE_EQ((left - 3).real(), -2.0);
    EXPECT_DOUBLE_EQ((3 - left).real(), 2.0);
    EXPECT_DOUBLE_EQ((left * 4).imag(), -8.0);
    EXPECT_DOUBLE_EQ((4 * left).real(), 4.0);
    EXPECT_DOUBLE_EQ((left / 2).real(), 0.5);
    EXPECT_TRUE(std::isfinite((2 / left).imag()));
    EXPECT_DOUBLE_EQ((left + std::int64_t{2}).real(), 3.0);
    EXPECT_DOUBLE_EQ((std::int64_t{5} - left).real(), 4.0);
    EXPECT_DOUBLE_EQ((left * std::int64_t{3}).imag(), -6.0);
    EXPECT_DOUBLE_EQ((std::int64_t{8} / left).real(), (8.0 / left).real());

    complex<double> zero_result{1.0, -2.0};
    zero_result /= complex<double>{0.0, 0.0};
    EXPECT_TRUE(!std::isfinite(zero_result.real()) || !std::isfinite(zero_result.imag()));
}

TEST(Complex, ComparisonStreamLiteralsAndStd)
{
    const complex<double> value{1.0, 2.0};
    const complex<double> other{1.0, 0.0};
    const complex<double> real_two{2.0, 0.0};
    const complex<double> imag_two{0.0, 2.0};
    EXPECT_TRUE(value == value);
    EXPECT_FALSE(value == other);
    EXPECT_TRUE(real_two == 2.0);
    EXPECT_TRUE(2.0 == real_two);
    EXPECT_TRUE(value != imag_two);
    EXPECT_TRUE(value != 2.0);
    EXPECT_TRUE(2.0 != value);
    EXPECT_FALSE(real_two != 2.0);

    std::ostringstream written;
    written << value;
    EXPECT_FALSE(written.str().empty());
    std::stringstream parsed_stream("(3,4)");
    complex<double> parsed;
    parsed_stream >> parsed;
    EXPECT_DOUBLE_EQ(parsed.real(), 3.0);
    EXPECT_DOUBLE_EQ(parsed.imag(), 4.0);

    using namespace scalars::complex_literals;
    const auto float_literal = 2.5_if;
    const auto double_literal = 2.5_id;
    const auto float_integer = 2_if;
    const auto double_integer = 3_id;
    EXPECT_FLOAT_EQ(float_literal.imag(), 2.5f);
    EXPECT_DOUBLE_EQ(double_literal.imag(), 2.5);
    EXPECT_FLOAT_EQ(float_integer.imag(), 2.0f);
    EXPECT_DOUBLE_EQ(double_integer.real(), 0.0);

    const auto polar_value = polar(2.0, 0.0);
    EXPECT_NEAR(polar_value.real(), 2.0, 1e-12);
    EXPECT_NEAR(polar(1.0).imag(), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(value), std::abs(std::complex<double>(1.0, 2.0)), 1e-12);
    EXPECT_NEAR(std::arg(value), std::atan2(2.0, 1.0), 1e-12);
    EXPECT_DOUBLE_EQ(std::norm(value), 5.0);
    EXPECT_DOUBLE_EQ(std::conj(value).imag(), -2.0);
    EXPECT_FALSE(std::isnan(value));
    EXPECT_TRUE(std::isnan(complex<double>{std::numeric_limits<double>::quiet_NaN(), 1.0}));
    EXPECT_TRUE(std::isnan(complex<double>{1.0, std::numeric_limits<double>::quiet_NaN()}));
    EXPECT_GT(std::numeric_limits<complex<double>>::max(), 0.0);
    EXPECT_TRUE(std::numeric_limits<complex<float>>::is_specialized);
    EXPECT_NE(real_of(value), 0.0);
}

TEST(Complex, HalfAndBFloat16)
{
    complex<half> value(half(1.5f), half(-2.0f));
    value += complex<half>(half(0.5f), half(1.0f));
    value -= complex<half>(half(0.5f), half(0.0f));
    value *= complex<half>(half(2.0f), half(0.0f));
    const complex<float> widened = static_cast<complex<float>>(value);
    const complex<half> narrowed(widened);
    EXPECT_NEAR(static_cast<float>(narrowed.real()), 3.0f, 1e-3f);
    EXPECT_NEAR(static_cast<float>(narrowed.imag()), -2.0f, 1e-3f);
    EXPECT_NEAR(widened.real(), 3.0f, 1e-3f);

    complex<bfloat16> brain(bfloat16(1.0f), bfloat16(2.0f));
    brain += complex<bfloat16>(bfloat16(3.0f), bfloat16(4.0f));
    brain -= complex<bfloat16>(bfloat16(1.0f), bfloat16(1.0f));
    brain *= complex<bfloat16>(bfloat16(2.0f), bfloat16(0.0f));
    EXPECT_NEAR(static_cast<float>(brain.real()), 6.0f, 0.05f);
    EXPECT_NEAR(static_cast<float>(brain.imag()), 10.0f, 0.05f);
}

namespace
{
template <class T> void check_scalar_division_bounds()
{
    using c                       = scalars::complex<T>;
    const T                 lower = std::is_same_v<T, float> ? T(0x1p-30) : T(0x1p-250);
    const T                 upper = std::is_same_v<T, float> ? T(0x1p31) : T(0x1p251);
    const T                 tiny  = std::numeric_limits<T>::denorm_min();
    const std::array<c, 12> values{c(1, -2),
        c(-3, 4),
        c(0, -0.0),
        c(lower, -lower),
        c(std::nextafter(lower, T(0)), T(1)),
        c(lower, upper),
        c(std::nextafter(upper, T(0)), -lower),
        c(upper, -upper),
        c(tiny, T(1)),
        c(T(1), tiny),
        c(-1, -0.0),
        c(0, 1)};
    for (const auto numerator : values)
    {
        for (const auto denominator : values)
        {
            if (denominator == c{})
            {
                continue;
            }
            const auto result = numerator / denominator;
            const auto reference =
                std::complex<long double>(numerator.real(), numerator.imag()) /
                std::complex<long double>(denominator.real(), denominator.imag());
            const long double scale = std::abs(reference);
            const long double error =
                std::hypot(static_cast<long double>(result.real()) - reference.real(),
                    static_cast<long double>(result.imag()) - reference.imag());
            EXPECT_LE(error, 16 * std::numeric_limits<T>::epsilon() * scale + 4 * tiny);
        }
    }
}
}  // namespace

TEST(Complex, ScalarDivisionFastPathBoundaries)
{
    check_scalar_division_bounds<float>();
    check_scalar_division_bounds<double>();
}
