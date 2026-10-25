// Smoke test for the ported backend/ SIMD dispatcher (backend/simd.h). Exercises
// whichever CPU tier SCALARS_CPU_BACKEND selected through the common simd<T>
// interface, without depending on which concrete backend got chosen. Only
// compiled when a CPU SIMD tier is actually active (SCALARS_VECTORIZED=1) --
// simd<T> has no specialization at all on the host when
// SCALARS_CPU_BACKEND=no (see backend/simd.h).
#include "backend/simd.h"

#if SCALARS_VECTORIZED

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

TEST(BackendSimd, FloatAddMatchesScalar)
{
    using simd_f              = simd<float>;
    constexpr int      n      = simd_f::size;
    std::vector<float> a(n), b(n), out(n);
    for (int i = 0; i < n; ++i)
    {
        a[i] = static_cast<float>(i + 1);
        b[i] = static_cast<float>(2 * i);
    }

    auto va = simd_f::loadu(a.data());
    auto vb = simd_f::loadu(b.data());
    auto vc = simd_f::add(va, vb);
    simd_f::storeu(vc, out.data());

    for (int i = 0; i < n; ++i)
    {
        EXPECT_FLOAT_EQ(out[i], a[i] + b[i]);
    }
}

TEST(BackendSimd, DoubleSetBroadcastsScalar)
{
    using simd_d               = simd<double>;
    constexpr int       n      = simd_d::size;
    std::vector<double> out(n);

    auto v = simd_d::set(3.5);
    simd_d::storeu(v, out.data());

    for (int i = 0; i < n; ++i)
    {
        EXPECT_DOUBLE_EQ(out[i], 3.5);
    }
}

#endif  // SCALARS_VECTORIZED
