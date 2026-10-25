#include "backend/simd.h"
#include "math/complex_vector.h"
#include "types/complex.h"

#include "common/intrinsics.h"
#include <complex>
#include <cstddef>
#include <gtest/gtest.h>
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
template <class T> std::complex<T> standard(scalars::complex<T> x)
{
    return {x.real(), x.imag()};
}
template <class T> void expect_complex(scalars::complex<T> actual, std::complex<T> expected)
{
    const double tolerance = std::is_same_v<T, float> ? 3e-4 : 1e-11;
    EXPECT_NEAR(actual.real(), expected.real(), tolerance);
    EXPECT_NEAR(actual.imag(), expected.imag(), tolerance);
}

template <class T> void check_vectors()
{
    using c = scalars::complex<T>;
#if SCALARS_VECTORIZED
    constexpr int width = simd<c>::size;
#else
    constexpr int width = 1;
#endif
    for (int count = 0; count <= 2 * width + 3; ++count)
    {
        for (const int sx : {1, 2, -3})
        {
            for (const int sy : {1, 3, -2})
            {
                for (const bool conjugate : {false, true})
                {
                    // Non-elements contain sentinels, checked after every operation.
                    const int      length = 3 * count + 4;
                    std::vector<c> x(length, c(T(99), T(-99))), y = x;
                    c* xp = x.data() + 1 + (sx < 0 && count > 0 ? (count - 1) * -sx : 0);
                    c* yp = y.data() + 1 + (sy < 0 && count > 0 ? (count - 1) * -sy : 0);
                    for (int i = 0; i < count; ++i)
                    {
                        xp[static_cast<std::ptrdiff_t>(i) * sx] = c(T(i + 1) / T(4), T(1 - i) / T(8));
                        yp[static_cast<std::ptrdiff_t>(i) * sy] = c(T(2 - i) / T(4), T(i + 2) / T(8));
                    }
                    const auto      original_x = x, original_y = y;
                    std::complex<T> expected_dot{};
                    for (int i = 0; i < count; ++i)
                    {
                        const auto a = standard(xp[static_cast<std::ptrdiff_t>(i) * sx]);
                        expected_dot += (conjugate ? std::conj(a) : a) * standard(yp[static_cast<std::ptrdiff_t>(i) * sy]);
                    }
                    expect_complex(
                        scalars::complex_dot(xp, sx, yp, sy, count, conjugate), expected_dot);

                    const c alpha(T(.5), T(-.25));
                    auto    expected_y = original_y;
                    auto*   ep         = expected_y.data() + (yp - y.data());
                    for (int i = 0; i < count; ++i)
                    {
                        const auto a = standard(xp[static_cast<std::ptrdiff_t>(i) * sx]);
                        ep[static_cast<std::ptrdiff_t>(i) * sy]   = c(standard(ep[static_cast<std::ptrdiff_t>(i) * sy]) +
                                         standard(alpha) * (conjugate ? std::conj(a) : a));
                    }
                    scalars::complex_axpy(yp, sy, xp, sx, count, alpha, conjugate);
                    for (int i = 0; i < length; ++i)
                    {
                        expect_complex(y[i], standard(expected_y[i]));
                    }
                    EXPECT_EQ(x, original_x);

                    auto expected_x = original_x;
                    ep              = expected_x.data() + (xp - x.data());
                    for (int i = 0; i < count; ++i)
                    {
                        ep[static_cast<std::ptrdiff_t>(i) * sx] = c(standard(ep[static_cast<std::ptrdiff_t>(i) * sx]) * T(-2));
                    }
                    scalars::complex_scale(xp, sx, count, T(-2));
                    for (int i = 0; i < length; ++i)
                    {
                        expect_complex(x[i], standard(expected_x[i]));
                    }

                    const T cosine = T(.6);
                    const c sine(T(0), T(.8));
                    expected_x = x;
                    expected_y = y;
                    auto* ex   = expected_x.data() + (xp - x.data());
                    auto* ey   = expected_y.data() + (yp - y.data());
                    for (int i = 0; i < count; ++i)
                    {
                        const auto a = standard(xp[static_cast<std::ptrdiff_t>(i) * sx]), b = standard(yp[static_cast<std::ptrdiff_t>(i) * sy]);
                        ex[static_cast<std::ptrdiff_t>(i) * sx] = c(cosine * a + standard(sine) * b);
                        ey[static_cast<std::ptrdiff_t>(i) * sy] = c(cosine * b - std::conj(standard(sine)) * a);
                    }
                    scalars::complex_rotate(xp, sx, yp, sy, count, cosine, sine);
                    for (int i = 0; i < length; ++i)
                    {
                        expect_complex(x[i], standard(expected_x[i]));
                        expect_complex(y[i], standard(expected_y[i]));
                    }
                }
            }
        }
    }
}
}  // namespace

TEST(ComplexVector, FloatAndDoubleStridesAndTails)
{
    check_vectors<float>();
    check_vectors<double>();
}

TEST(ComplexVector, EmptyAndInPlace)
{
    using c = scalars::complex<double>;
    expect_complex(scalars::complex_dot<double>(nullptr, 1, nullptr, 1, 0), std::complex<double>{});
    scalars::complex_axpy<double>(nullptr, 1, nullptr, 1, 0, c(2, 1));
    scalars::complex_scale<double>(nullptr, 1, 0, 2);
    scalars::complex_rotate<double>(nullptr, 1, nullptr, 1, 0, 1, c(0, 1));
    std::vector<c> x(35, c(1, 2));
    scalars::complex_axpy(x.data(), 1, x.data(), 1, x.size(), c(2, -1), true);
    for (auto a : x)
    {
        expect_complex(a,
            std::complex<double>(1, 2) + std::complex<double>(2, -1) * std::complex<double>(1, -2));
    }
}
