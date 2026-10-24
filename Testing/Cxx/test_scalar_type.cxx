#include <core/scalar_type.h>

#include "scalars_test.h"

using namespace kh::scalars;

TEST(scalar_type, metadata)
{
    EXPECT_EQ(scalar_type_size(scalar_type::complex_double), 16);
    EXPECT_EQ(scalar_type_name(scalar_type::bfloat16), "bfloat16");
}
