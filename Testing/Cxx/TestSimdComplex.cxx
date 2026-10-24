#include "backend/simd.h"

#include <algorithm>
#include <array>
#include <complex>
#include <limits>

#include "scalars_test.h"

#if SCALARS_VECTORIZED
namespace
{
template <class T> void expect_complex(scalars::complex<T> actual, std::complex<T> expected)
{
    const double tolerance = std::is_same_v<T, float> ? 1e-4 : 1e-11;
    EXPECT_NEAR(actual.real(), expected.real(), tolerance);
    EXPECT_NEAR(actual.imag(), expected.imag(), tolerance);
}

template <class T> void check_complex_simd()
{
    using c                = scalars::complex<T>;
    using s                = simd<c>;
    constexpr int        W = s::size;
    std::array<c, W + 2> input, output;
    for (int i = 0; i < W + 2; ++i)
    {
        input[i] = c(T(i + 1), T(2 - i));
    }
    const c               factor(T(2), T(-3));
    const auto            a = s::loadu(input.data() + 1), b = s::set(factor);
    const std::complex<T> f(T(2), T(-3));
    const auto            check = [&](typename s::simd_t result, auto reference)
    {
        s::storeu(result, output.data());
        for (int i = 0; i < W; ++i)
        {
            expect_complex(
                output[i], reference(std::complex<T>(input[i + 1].real(), input[i + 1].imag())));
        }
    };
    check(s::add(a, b), [&](auto x) { return x + f; });
    check(s::sub(a, b), [&](auto x) { return x - f; });
    check(s::mul(a, b), [&](auto x) { return x * f; });
    check(s::div(a, b), [&](auto x) { return x / f; });
    check(s::fma(a, b, b), [&](auto x) { return x * f + f; });
    check(s::neg(a), [&](auto x) { return -x; });
    check(s::conj(a), [&](auto x) { return std::conj(x); });

    std::complex<T> sum{};
    for (int i = 1; i <= W; ++i)
    {
        sum += std::complex<T>(input[i].real(), input[i].imag());
    }
    expect_complex(s::accumulate(a), sum);
    s::scatter(s::gather(input.data() + W, -1), -1, output.data() + W);
    for (int i = 1; i <= W; ++i)
    {
        EXPECT_EQ(output[i], input[i]);
    }

    for (int count = 0; count <= W; ++count)
    {
        output.fill(c(T(-9), T(7)));
        const auto partial = s::loadu(input.data() + 1, count);
        s::storeu(partial, output.data() + 1, count);
        EXPECT_EQ(output[0], c(T(-9), T(7)));
        for (int i = 0; i < count; ++i)
        {
            EXPECT_EQ(output[i + 1], input[i + 1]);
        }
        for (int i = count + 1; i < W + 2; ++i)
        {
            EXPECT_EQ(output[i], c(T(-9), T(7)));
        }
        s::storeu(partial, output.data());
        for (int i = count; i < W; ++i)
        {
            EXPECT_EQ(output[i], c{});
        }
    }
    s::storeu(s::loadu(nullptr, 0), nullptr, 0);
    s::storeu(s::conj(s::set(c(T(1), T(0)))), output.data());
    EXPECT_TRUE(std::signbit(output[0].imag()));
    const T nan = std::numeric_limits<T>::quiet_NaN();
    s::storeu(s::conj(s::set(c(nan, T(-0.0)))), output.data());
    EXPECT_TRUE(std::isnan(output[0].real()));
    EXPECT_FALSE(std::signbit(output[0].imag()));
}

template <class T> void check_matrix_product()
{
    using c                          = scalars::complex<T>;
    using s                          = simd<c>;
    constexpr int               rows = 2 * s::size + 1, depth = 7, cols = 5;
    std::array<c, static_cast<std::size_t>(rows) * depth> a;
    std::array<c, static_cast<std::size_t>(depth) * cols> b;
    std::array<c, static_cast<std::size_t>(rows) * cols>  result;
    for (int i = 0; i < rows * depth; ++i)
    {
        a[i] = c(T(i % 5 - 2), T(i % 3) / T(4));
    }
    for (int i = 0; i < depth * cols; ++i)
    {
        b[i] = c(T(i % 7 - 3), T(i % 4) / T(4));
    }
    for (int c = 0; c < cols; ++c)
    {
        for (int r = 0; r < rows; r += s::size)
        {
            const int count = std::min(s::size, rows - r);
            auto      sum   = s::setzero();
            for (int k = 0; k < depth; ++k)
            {
                sum =
                    s::fma(s::loadu(a.data() + static_cast<std::size_t>(k) * rows + r, count), s::set(b[static_cast<std::size_t>(k) + static_cast<std::size_t>(c) * depth]), sum);
            }
            s::storeu(sum, result.data() + static_cast<std::size_t>(c) * rows + r, count);
        }
    }
    for (int c = 0; c < cols; ++c)
    {
        for (int r = 0; r < rows; ++r)
        {
            std::complex<T> expected{};
            for (int k = 0; k < depth; ++k)
            {
                expected += std::complex<T>(a[static_cast<std::size_t>(k) * rows + r].real(), a[static_cast<std::size_t>(k) * rows + r].imag()) *
                            std::complex<T>(b[static_cast<std::size_t>(k) + static_cast<std::size_t>(c) * depth].real(), b[static_cast<std::size_t>(k) + static_cast<std::size_t>(c) * depth].imag());
            }
            expect_complex(result[static_cast<std::size_t>(c) * rows + r], expected);
        }
    }
}

template <class T> void check_matrix_packing()
{
    using s            = simd<T>;
    constexpr int    W = s::size;
    std::array<T, W> output;
    if constexpr (requires(const T* p) { s::ploadquad(p); })
    {
        // PRETORIAN's four-wide float load duplicates pairs; wider loads
        // repeat each input four times. AVX-512 float has no ploadquad.
        constexpr int repeats = std::is_same_v<T, float> && W == 4 ? 2 : 4;
        std::array<T, (W + repeats - 1) / repeats> scalars;
        for (int i = 0; i < static_cast<int>(scalars.size()); ++i)
        {
            scalars[i] = T(i + 1);
        }
        s::storeu(s::ploadquad(scalars.data()), output.data());
        for (int i = 0; i < W; ++i)
        {
            EXPECT_EQ(output[i], scalars[i / repeats]);
        }
    }
    if constexpr (W >= 4)
    {
        typename s::simd_t rows[4];
        for (int r = 0; r < 4; ++r)
        {
            for (int c = 0; c < W; ++c)
            {
                output[c] = T(100 * r + c);
            }
            rows[r] = s::loadu(output.data());
        }
        s::template ptranspose<4>(rows);
        // PRETORIAN transposes the complete 4 x W rectangle and repacks
        // its contiguous output into four registers, ready for RHS packing.
        std::array<T, 4 * static_cast<std::size_t>(W)> packed;
        for (int r = 0; r < 4; ++r)
        {
            s::storeu(rows[r], packed.data() + r * W);
        }
        for (int k = 0; k < W; ++k)
        {
            for (int c = 0; c < 4; ++c)
            {
                EXPECT_EQ(packed[4 * k + c], T(100 * c + k));
            }
        }
    }
}
}  // namespace

TEST(SimdComplex, ArithmeticMemoryAndReduction)
{
    check_complex_simd<float>();
    check_complex_simd<double>();
}

TEST(SimdComplex, MatrixProduct)
{
    check_matrix_product<float>();
    check_matrix_product<double>();
}

TEST(Simd, MatrixPacking)
{
    check_matrix_packing<float>();
    check_matrix_packing<double>();
}
#endif
