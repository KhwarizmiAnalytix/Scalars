#include <cmath>
#include <cstdint>
#include <types/half.h>

#include "scalars_test.h"

using namespace kh::scalars;

TEST(half, representation)
{
    static_assert(sizeof(half) == 2);
    EXPECT_EQ(half(1.0f).to_bits(), 0x3c00);
    EXPECT_EQ(half(65504.0f).to_bits(), 0x7bff);
    EXPECT_EQ(half(0.000000059604644775390625f).to_bits(), 0x0001);
    EXPECT_EQ(half::from_bits(0x8000).to_bits(), 0x8000);
}

TEST(half, arithmetic)
{
    EXPECT_EQ(half(1.0f) + half(2.0f), half(3.0f));
}

TEST(half, exhaustive_encoding_round_trip)
{
    for (std::uint32_t bits = 0; bits < 65536; ++bits)
    {
        const auto value  = half::from_bits(static_cast<std::uint16_t>(bits));
        const auto result = half(static_cast<float>(value));
        if (((bits & 0x7c00u) == 0x7c00u) && (bits & 0x03ffu))
        {
            EXPECT_TRUE(std::isnan(float(result))) << "encoding: " << bits;
        }
        else
        {
            EXPECT_EQ(result.to_bits(), bits) << "encoding: " << bits;
        }
    }
}
