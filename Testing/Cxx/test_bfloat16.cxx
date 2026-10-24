#include <types/bfloat16.h>

#include "scalars_test.h"

using namespace kh::scalars;

TEST(bfloat16, representation)
{
    static_assert(sizeof(bfloat16) == 2);
    EXPECT_EQ(bfloat16(1.0f).to_bits(), 0x3f80);
    EXPECT_EQ(static_cast<float>(bfloat16::from_bits(0x3f80)), 1.0f);
}

TEST(bfloat16, arithmetic)
{
    EXPECT_EQ(bfloat16(2.0f) * bfloat16(3.0f), bfloat16(6.0f));
}
