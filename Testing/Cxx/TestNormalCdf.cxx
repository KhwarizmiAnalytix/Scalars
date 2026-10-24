#include "math/normal_cdf.h"

#include "scalars_test.h"

using namespace scalars;

TEST(NormalCdf, MidpointIsOneHalf)
{
    EXPECT_NEAR(normalcdf(0.0), 0.5, 1e-12);
    EXPECT_NEAR(inv_normalcdf(0.5), 0.0, 1e-12);
}

TEST(NormalCdf, RoundTripsThroughInverse)
{
    for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99})
    {
        EXPECT_NEAR(normalcdf(inv_normalcdf(p)), p, 1e-9) << "p=" << p;
    }
}

TEST(NormalCdf, Monotonic)
{
    EXPECT_LT(normalcdf(-1.0), normalcdf(0.0));
    EXPECT_LT(normalcdf(0.0), normalcdf(1.0));
}
