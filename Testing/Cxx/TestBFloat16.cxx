#include "types/bfloat16.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>

#include "scalars_test.h"

using namespace scalars;

TEST(bfloat16, Representation)
{
    static_assert(sizeof(bfloat16) == 2);
    EXPECT_EQ(bfloat16(1.0f).x, 0x3f80);
    const bfloat16 from_bits(0x3f80, bfloat16::from_bits());
    EXPECT_FLOAT_EQ(static_cast<float>(from_bits), 1.0f);
    EXPECT_EQ((-bfloat16(1.0f)).x, bfloat16(-1.0f).x);
    EXPECT_TRUE(std::isnan(static_cast<float>(bfloat16(std::numeric_limits<float>::quiet_NaN()))));
    EXPECT_TRUE(std::isinf(static_cast<float>(bfloat16(std::numeric_limits<float>::infinity()))));
    EXPECT_EQ(detail::round_to_nearest_even(std::numeric_limits<float>::quiet_NaN()), 0x7fc0);
    EXPECT_EQ(detail::bits_from_f32(1.0f), 0x3f80);
    EXPECT_FLOAT_EQ(detail::f32_from_bits(0x3f80), 1.0f);
    EXPECT_EQ(detail::round_to_nearest_even(1.0f), 0x3f80);
    EXPECT_NE(detail::round_to_nearest_even(1.1f), detail::round_to_nearest_even(-1.1f));
}

TEST(bfloat16, Arithmetic)
{
    bfloat16 value(1.0f);
    value += bfloat16(2.0f);
    value -= bfloat16(1.0f);
    value *= bfloat16(4.0f);
    value /= bfloat16(2.0f);
    EXPECT_EQ(value.x, bfloat16(4.0f).x);
    EXPECT_NEAR(static_cast<float>(bfloat16(1.5f) + bfloat16(2.5f)), 4.0f, 0.05f);
    EXPECT_NEAR(static_cast<float>(bfloat16(5.0f) - bfloat16(2.0f)), 3.0f, 0.05f);
    EXPECT_NEAR(static_cast<float>(bfloat16(1.5f) * bfloat16(2.0f)), 3.0f, 0.05f);
    EXPECT_NEAR(static_cast<float>(bfloat16(3.0f) / bfloat16(2.0f)), 1.5f, 0.05f);

    EXPECT_FLOAT_EQ(bfloat16(1.0f) + 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(2.0f + bfloat16(1.0f), 3.0f);
    EXPECT_FLOAT_EQ(bfloat16(3.0f) - 1.0f, 2.0f);
    EXPECT_FLOAT_EQ(3.0f - bfloat16(1.0f), 2.0f);
    EXPECT_FLOAT_EQ(bfloat16(1.5f) * 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(2.0f * bfloat16(1.5f), 3.0f);
    EXPECT_FLOAT_EQ(bfloat16(3.0f) / 2.0f, 1.5f);
    EXPECT_FLOAT_EQ(3.0f / bfloat16(2.0f), 1.5f);
    float slot = 1.0f;
    slot += bfloat16(2.0f);
    slot -= bfloat16(1.0f);
    slot *= bfloat16(2.0f);
    slot /= bfloat16(2.0f);
    EXPECT_FLOAT_EQ(slot, 2.0f);

    EXPECT_DOUBLE_EQ(bfloat16(1.0f) + 2.0, 3.0);
    EXPECT_DOUBLE_EQ(2.0 + bfloat16(1.0f), 3.0);
    EXPECT_DOUBLE_EQ(bfloat16(3.0f) - 1.0, 2.0);
    EXPECT_DOUBLE_EQ(3.0 - bfloat16(1.0f), 2.0);
    EXPECT_DOUBLE_EQ(bfloat16(1.5f) * 2.0, 3.0);
    EXPECT_DOUBLE_EQ(2.0 * bfloat16(1.5f), 3.0);
    EXPECT_DOUBLE_EQ(bfloat16(3.0f) / 2.0, 1.5);
    EXPECT_DOUBLE_EQ(3.0 / bfloat16(2.0f), 1.5);

    EXPECT_EQ((bfloat16(1.0f) + 2).x, bfloat16(3.0f).x);
    EXPECT_EQ((2 + bfloat16(1.0f)).x, bfloat16(3.0f).x);
    EXPECT_EQ((bfloat16(5.0f) - 2).x, bfloat16(3.0f).x);
    EXPECT_EQ((5 - bfloat16(2.0f)).x, bfloat16(3.0f).x);
    EXPECT_EQ((bfloat16(2.0f) * 3).x, bfloat16(6.0f).x);
    EXPECT_EQ((3 * bfloat16(2.0f)).x, bfloat16(6.0f).x);
    EXPECT_EQ((bfloat16(6.0f) / 2).x, bfloat16(3.0f).x);
    EXPECT_EQ((6 / bfloat16(2.0f)).x, bfloat16(3.0f).x);

    EXPECT_EQ((bfloat16(1.0f) + std::int64_t{2}).x, bfloat16(3.0f).x);
    EXPECT_EQ((std::int64_t{2} + bfloat16(1.0f)).x, bfloat16(3.0f).x);
    EXPECT_EQ((bfloat16(5.0f) - std::int64_t{2}).x, bfloat16(3.0f).x);
    EXPECT_EQ((std::int64_t{5} - bfloat16(2.0f)).x, bfloat16(3.0f).x);
    EXPECT_EQ((bfloat16(2.0f) * std::int64_t{3}).x, bfloat16(6.0f).x);
    EXPECT_EQ((std::int64_t{3} * bfloat16(2.0f)).x, bfloat16(6.0f).x);
    EXPECT_EQ((bfloat16(6.0f) / std::int64_t{2}).x, bfloat16(3.0f).x);
    EXPECT_EQ((std::int64_t{6} / bfloat16(2.0f)).x, bfloat16(3.0f).x);
}

TEST(bfloat16, BitwiseAndComparison)
{
    bfloat16 left(1.0f);
    bfloat16 right(2.0f);
    EXPECT_TRUE(left < right);
    EXPECT_TRUE(right > left);
    EXPECT_FALSE(right < left);
    EXPECT_FALSE(left > right);

    bfloat16 bits(1.0f);
    bits = (bits | right);
    bits = (bits ^ right);
    bits = (bits & left);
    EXPECT_EQ(bits.x, static_cast<std::uint16_t>(bfloat16(1.0f).x & left.x));
}

TEST(bfloat16, NumericLimitsStreamAndRoundTrip)
{
    using limits = std::numeric_limits<bfloat16>;
    EXPECT_TRUE(limits::is_specialized);
    EXPECT_FALSE(limits::is_iec559);
    EXPECT_GT(static_cast<float>(limits::min()), 0.0f);
    EXPECT_LT(static_cast<float>(limits::lowest()), 0.0f);
    EXPECT_GT(static_cast<float>(limits::max()), 0.0f);
    EXPECT_GT(static_cast<float>(limits::epsilon()), 0.0f);
    EXPECT_GT(static_cast<float>(limits::round_error()), 0.0f);
    EXPECT_TRUE(std::isinf(static_cast<float>(limits::infinity())));
    EXPECT_TRUE(std::isnan(static_cast<float>(limits::quiet_NaN())));
    EXPECT_TRUE(std::isnan(static_cast<float>(limits::signaling_NaN())) ||
                std::isinf(static_cast<float>(limits::signaling_NaN())));
    EXPECT_GT(static_cast<float>(limits::denorm_min()), 0.0f);

    std::ostringstream stream;
    stream << bfloat16(1.5f);
    EXPECT_FALSE(stream.str().empty());

    for (int step = -200; step <= 200; ++step)
    {
        const float    sample = static_cast<float>(step) * 0.25f;
        const bfloat16 encoded(sample);
        const float    back = static_cast<float>(encoded);
        if (sample == 0.0f)
            EXPECT_EQ(back, 0.0f);
        else
            EXPECT_NEAR(back, sample, std::abs(sample) * 0.02f + 1.0e-3f);
    }
}
