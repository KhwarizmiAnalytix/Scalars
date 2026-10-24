#include "common/scalar_type.h"

#include <cstdint>

#include "scalars_test.h"

using scalars::scalar_type;

TEST(ScalarType, NamesAndSizes)
{
    EXPECT_EQ(scalars::scalar_type_name(scalar_type::float16), "float16");
    EXPECT_EQ(scalars::scalar_type_name(scalar_type::bfloat16), "bfloat16");
    EXPECT_EQ(scalars::scalar_type_name(scalar_type::float32), "float32");
    EXPECT_EQ(scalars::scalar_type_name(scalar_type::float64), "float64");
    EXPECT_EQ(scalars::scalar_type_name(scalar_type::complex_half), "complex_half");
    EXPECT_EQ(scalars::scalar_type_name(scalar_type::complex_bf16), "complex_bfloat16");
    EXPECT_EQ(scalars::scalar_type_name(scalar_type::complex_float), "complex64");
    EXPECT_EQ(scalars::scalar_type_name(scalar_type::complex_double), "complex128");

    EXPECT_EQ(scalars::scalar_type_size(scalar_type::float16), 2u);
    EXPECT_EQ(scalars::scalar_type_size(scalar_type::bfloat16), 2u);
    EXPECT_EQ(scalars::scalar_type_size(scalar_type::float32), 4u);
    EXPECT_EQ(scalars::scalar_type_size(scalar_type::complex_half), 4u);
    EXPECT_EQ(scalars::scalar_type_size(scalar_type::complex_bf16), 4u);
    EXPECT_EQ(scalars::scalar_type_size(scalar_type::float64), 8u);
    EXPECT_EQ(scalars::scalar_type_size(scalar_type::complex_float), 8u);
    EXPECT_EQ(scalars::scalar_type_size(scalar_type::complex_double), 16u);

    const auto invalid = static_cast<scalar_type>(255);
    EXPECT_EQ(scalars::scalar_type_name(invalid), "unknown");
    EXPECT_EQ(scalars::scalar_type_size(invalid), 0u);
}
