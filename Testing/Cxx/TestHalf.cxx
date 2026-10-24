#include "types/half.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>

#include "scalars_test.h"

using namespace scalars;

TEST(half, Representation)
{
    static_assert(sizeof(half) == 2);
    EXPECT_EQ(half(1.0f).x, 0x3c00);
    EXPECT_EQ(half(65504.0f).x, 0x7bff);
    EXPECT_EQ(half(0.000000059604644775390625f).x, 0x0001);
    EXPECT_EQ(half(0x8000, half::from_bits()).x, 0x8000);
    EXPECT_EQ((-half(1.0f)).x, half(-1.0f).x);
}

TEST(half, Arithmetic)
{
    half value(1.0f);
    value += half(2.0f);
    value -= half(1.0f);
    value *= half(4.0f);
    value /= half(2.0f);
    EXPECT_EQ(value.x, half(4.0f).x);
    EXPECT_EQ((half(1.0f) + half(2.0f)).x, half(3.0f).x);
    EXPECT_EQ((half(5.0f) - half(2.0f)).x, half(3.0f).x);
    EXPECT_EQ((half(2.0f) * half(3.0f)).x, half(6.0f).x);
    EXPECT_EQ((half(6.0f) / half(2.0f)).x, half(3.0f).x);

    EXPECT_FLOAT_EQ(half(1.5f) + 2.0f, 3.5f);
    EXPECT_FLOAT_EQ(2.0f + half(1.5f), 3.5f);
    EXPECT_FLOAT_EQ(half(3.5f) - 1.0f, 2.5f);
    EXPECT_FLOAT_EQ(3.5f - half(1.0f), 2.5f);
    EXPECT_FLOAT_EQ(half(1.5f) * 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(2.0f * half(1.5f), 3.0f);
    EXPECT_FLOAT_EQ(half(3.0f) / 2.0f, 1.5f);
    EXPECT_FLOAT_EQ(3.0f / half(2.0f), 1.5f);
    float slot = 1.0f;
    slot += half(2.0f);
    slot -= half(1.0f);
    slot *= half(2.0f);
    slot /= half(2.0f);
    EXPECT_FLOAT_EQ(slot, 2.0f);

    EXPECT_DOUBLE_EQ(half(1.5f) + 2.0, 3.5);
    EXPECT_DOUBLE_EQ(2.0 + half(1.5f), 3.5);
    EXPECT_DOUBLE_EQ(half(3.5f) - 1.0, 2.5);
    EXPECT_DOUBLE_EQ(3.5 - half(1.0f), 2.5);
    EXPECT_DOUBLE_EQ(half(1.5f) * 2.0, 3.0);
    EXPECT_DOUBLE_EQ(2.0 * half(1.5f), 3.0);
    EXPECT_DOUBLE_EQ(half(3.0f) / 2.0, 1.5);
    EXPECT_DOUBLE_EQ(3.0 / half(2.0f), 1.5);

    EXPECT_EQ((half(1.0f) + 2).x, half(3.0f).x);
    EXPECT_EQ((2 + half(1.0f)).x, half(3.0f).x);
    EXPECT_EQ((half(5.0f) - 2).x, half(3.0f).x);
    EXPECT_EQ((5 - half(2.0f)).x, half(3.0f).x);
    EXPECT_EQ((half(2.0f) * 3).x, half(6.0f).x);
    EXPECT_EQ((3 * half(2.0f)).x, half(6.0f).x);
    EXPECT_EQ((half(6.0f) / 2).x, half(3.0f).x);
    EXPECT_EQ((6 / half(2.0f)).x, half(3.0f).x);

    const std::int64_t wide = 2;
    EXPECT_EQ((half(1.0f) + wide).x, half(3.0f).x);
    EXPECT_EQ((wide + half(1.0f)).x, half(3.0f).x);
    EXPECT_EQ((half(5.0f) - wide).x, half(3.0f).x);
    EXPECT_EQ((std::int64_t{5} - half(2.0f)).x, half(3.0f).x);
    EXPECT_EQ((half(2.0f) * std::int64_t{3}).x, half(6.0f).x);
    EXPECT_EQ((std::int64_t{3} * half(2.0f)).x, half(6.0f).x);
    EXPECT_EQ((half(6.0f) / wide).x, half(3.0f).x);
    EXPECT_EQ((std::int64_t{6} / half(2.0f)).x, half(3.0f).x);
}

TEST(half, NumericLimitsAndStream)
{
    using limits = std::numeric_limits<half>;
    EXPECT_TRUE(limits::is_specialized);
    EXPECT_TRUE(limits::is_signed);
    EXPECT_FALSE(limits::is_integer);
    EXPECT_TRUE(limits::has_infinity);
    EXPECT_TRUE(limits::has_quiet_NaN);
    EXPECT_GT(static_cast<float>(limits::min()), 0.0f);
    EXPECT_LT(static_cast<float>(limits::lowest()), 0.0f);
    EXPECT_GT(static_cast<float>(limits::max()), 0.0f);
    EXPECT_GT(static_cast<float>(limits::epsilon()), 0.0f);
    EXPECT_GT(static_cast<float>(limits::round_error()), 0.0f);
    EXPECT_TRUE(std::isinf(static_cast<float>(limits::infinity())));
    EXPECT_TRUE(std::isnan(static_cast<float>(limits::quiet_NaN())));
    EXPECT_TRUE(std::isnan(static_cast<float>(limits::signaling_NaN())));
    EXPECT_GT(static_cast<float>(limits::denorm_min()), 0.0f);

    std::ostringstream stream;
    stream << half(1.5f);
    EXPECT_NE(stream.str().find('1'), std::string::npos);
}

TEST(half, SoftwareIeeeConversion)
{
    std::uint32_t mix = 0;
    for (std::uint32_t bits = 0; bits < 65536; ++bits)
    {
        const auto encoding = static_cast<std::uint16_t>(bits);
        const float value   = detail::fp16_ieee_to_fp32_value(encoding);
        const auto  back    = detail::fp16_ieee_from_fp32_value(value);
        mix ^= detail::fp16_ieee_to_fp32_bits(encoding);
        if ((encoding & 0x7c00u) != 0x7c00u)
        {
            EXPECT_EQ(back, encoding) << "encoding: " << bits;
        }
        else if ((encoding & 0x03ffu) != 0)
        {
            EXPECT_TRUE(std::isnan(value)) << "encoding: " << bits;
        }
        else
        {
            EXPECT_TRUE(std::isinf(value)) << "encoding: " << bits;
        }
    }
    const float specials[] = {0.0f, -0.0f, HUGE_VALF, -HUGE_VALF,
        std::numeric_limits<float>::quiet_NaN(), 1.0e-8f, 1.0e20f, 1.0f, -2.5f};
    for (float value : specials)
        mix ^= detail::fp16_ieee_from_fp32_value(value);
    EXPECT_NE(mix, 0u);
}

TEST(half, ExhaustiveEncodingRoundTrip)
{
    for (std::uint32_t bits = 0; bits < 65536; ++bits)
    {
        const auto value  = half(static_cast<unsigned short>(bits), half::from_bits());
        const auto result = half(static_cast<float>(value));
        if (((bits & 0x7c00u) == 0x7c00u) && (bits & 0x03ffu))
        {
            EXPECT_TRUE(std::isnan(static_cast<float>(result))) << "encoding: " << bits;
        }
        else
        {
            EXPECT_EQ(result.x, bits) << "encoding: " << bits;
        }
    }
}

#if defined(__aarch64__) && !defined(__CUDACC__)
TEST(half, NativeBitCast)
{
    EXPECT_FLOAT_EQ(detail::native_fp16_to_fp32_value(0x3c00), 1.0f);
    EXPECT_EQ(detail::native_fp16_from_fp32_value(1.0f), 0x3c00);
    const auto native = detail::fp16_from_bits(0x3c00);
    EXPECT_EQ(detail::fp16_to_bits(native), 0x3c00);
}
#endif
