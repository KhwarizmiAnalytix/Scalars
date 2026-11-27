#include "math/normal_cdf.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>

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

TEST(NormalCdf, LowerTailMatchesErfcToDoublePrecision)
{
    // Confirmed bug: the asymptotic series previously used for z < -7.071
    // was truncated too early, with ~1.3e-7 relative error near z = -7.07.
    for (double z : {-7.071067811865475, -8.0, -10.0, -15.0, -20.0, -30.0, -38.0})
    {
        const double expected = 0.5 * std::erfc(-z / std::sqrt(2.0));
        const double actual   = normalcdf(z);
        ASSERT_GT(expected, 0.0) << "z=" << z;
        // Comparing two different formulations (erfc(-z/sqrt(2)) here vs.
        // erfc(-z * (1/sqrt(2))) in normalcdf) differs by a few ULP; this
        // tolerance is still ~1e4 tighter than the confirmed 1.3e-7 bug.
        EXPECT_NEAR(actual / expected, 1.0, 1e-11) << "z=" << z;
    }
}

TEST(NormalCdf, InverseEndpointsAreSignedInfinity)
{
    EXPECT_EQ(inv_normalcdf(0.0), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(inv_normalcdf(1.0), std::numeric_limits<double>::infinity());
}

TEST(NormalCdf, InverseOutOfDomainIsNaN)
{
    EXPECT_TRUE(std::isnan(inv_normalcdf(-0.1)));
    EXPECT_TRUE(std::isnan(inv_normalcdf(1.1)));
}

TEST(NormalCdf, RoundTripNearEndpoints)
{
    for (double p : {1e-300, 1e-100, 1e-10, 1.0 - 1e-10})
    {
        EXPECT_NEAR(normalcdf(inv_normalcdf(p)) / p, 1.0, 1e-6) << "p=" << p;
    }
}
