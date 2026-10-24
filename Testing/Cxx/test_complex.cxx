#include <types/complex.h>

#include "scalars_test.h"

using namespace kh::scalars;

TEST(complex, representation)
{
    static_assert(sizeof(complex_float) == 8 && sizeof(complex_double) == 16);
}

TEST(complex, multiplication)
{
    EXPECT_EQ((complex_float{1, 2} * complex_float{3, 4}), (complex_float{-5, 10}));
}
