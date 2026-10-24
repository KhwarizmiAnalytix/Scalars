// SIMD utility tests: fma, if_else, prefetch/copy/broadcast/set, ptranspose,
// half-width fma/add/gather, predux_downto4, full-width gather/scatter, masks.
#include "backend/simd.h"

#if SCALARS_VECTORIZED

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "math/scalar_helper_functions.h"
#include "scalars_test.h"

namespace
{

inline constexpr int kRandomTrials = 256;

template <typename value_t, std::size_t N>
void fill_uniform(std::array<value_t, N>& xs, std::mt19937& gen, value_t lo, value_t hi)
{
    std::uniform_real_distribution<value_t> dist(lo, hi);
    for (auto& x : xs)
    {
        x = dist(gen);
    }
}

namespace detail
{
// These detection idioms key off hardware vector types (__m256/__m128/etc.)
// via decltype/void_t; GCC/Clang ignore the vector_size attribute for
// template-argument matching purposes and warn about it, which is expected.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

template <typename V, typename = void> struct simd_has_half_fma : std::false_type
{
};

template <typename V>
struct simd_has_half_fma<V,
    std::void_t<decltype(simd<V>::fma(std::declval<typename simd<V>::simd_half_t const&>(),
        std::declval<typename simd<V>::simd_half_t const&>(),
        std::declval<typename simd<V>::simd_half_t const&>()))>> : std::true_type
{
};

template <typename V, typename = void> struct simd_has_loadu_half : std::false_type
{
};

template <typename V>
struct simd_has_loadu_half<V, std::void_t<decltype(simd<V>::loadu_half(std::declval<V const*>()))>>
    : std::true_type
{
};

template <typename V, typename = void> struct simd_has_predux_downto4 : std::false_type
{
};

template <typename V>
struct simd_has_predux_downto4<V,
    std::void_t<decltype(simd<V>::predux_downto4(std::declval<typename simd<V>::simd_t>()))>>
    : std::true_type
{
};

template <typename V, typename = void> struct simd_has_ploadquad : std::false_type
{
};

template <typename V>
struct simd_has_ploadquad<V, std::void_t<decltype(simd<V>::ploadquad(std::declval<V const*>()))>>
    : std::true_type
{
};

template <typename V, typename = void> struct simd_has_half_gather : std::false_type
{
};

template <typename V>
struct simd_has_half_gather<V,
    std::void_t<decltype(simd<V>::gather_half(std::declval<V const*>(), int{}))>> : std::true_type
{
};

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
}  // namespace detail

template <typename value_t, typename FillXYZ, typename SimdFma, typename RefFma>
void ternary_random_vs_std(
    value_t tolerance, FillXYZ&& fill_xyz, SimdFma&& simd_fma, RefFma&& ref_fma)
{
    constexpr std::size_t n = simd<value_t>::size;
    using simd_t            = typename simd<value_t>::simd_t;

    std::array<value_t, n> xs{};
    std::array<value_t, n> ys{};
    std::array<value_t, n> zs{};
    std::array<value_t, n> out{};
    // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility
    std::mt19937 gen(5489u);

    for (int trial = 0; trial < kRandomTrials; ++trial)
    {
        fill_xyz(xs, ys, zs, gen);
        simd_t a = simd<value_t>::loadu(xs.data());
        simd_t b = simd<value_t>::loadu(ys.data());
        simd_t c = simd<value_t>::loadu(zs.data());
        simd_t r;
        simd_fma(a, b, c, r);
        simd<value_t>::storeu(r, out.data());
        for (std::size_t i = 0; i < n; ++i)
        {
            value_t const ref = ref_fma(xs[i], ys[i], zs[i]);
            EXPECT_LE(std::fabs(out[i] - ref), tolerance);
        }
    }
}

template <typename value_t> void test_fma(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    ternary_random_vs_std<value_t>(
        tolerance,
        [](std::array<value_t, simd<value_t>::size>&  xs,
            std::array<value_t, simd<value_t>::size>& ys,
            std::array<value_t, simd<value_t>::size>& zs,
            std::mt19937&                             gen)
        {
            fill_uniform(xs, gen, static_cast<value_t>(-4), static_cast<value_t>(4));
            fill_uniform(ys, gen, static_cast<value_t>(-4), static_cast<value_t>(4));
            fill_uniform(zs, gen, static_cast<value_t>(-4), static_cast<value_t>(4));
        },
        [](simd_t a, simd_t b, simd_t c, simd_t& r) { r = simd<value_t>::fma(a, b, c); },
        [](value_t x, value_t y, value_t z) { return std::fma(x, y, z); });
}

template <typename value_t> void test_if_else(value_t tolerance)
{
    constexpr std::size_t n = simd<value_t>::size;
    using simd_t            = typename simd<value_t>::simd_t;
    using mask_t            = typename simd<value_t>::mask_t;

    std::array<value_t, n> xs{};
    std::array<value_t, n> ys{};
    std::array<value_t, n> zs{};
    std::array<value_t, n> ws{};
    std::array<value_t, n> out{};
    // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility
    std::mt19937 gen(31415u);

    for (int trial = 0; trial < kRandomTrials; ++trial)
    {
        fill_uniform(xs, gen, static_cast<value_t>(-12), static_cast<value_t>(12));
        fill_uniform(ys, gen, static_cast<value_t>(-12), static_cast<value_t>(12));
        fill_uniform(zs, gen, static_cast<value_t>(-30), static_cast<value_t>(30));
        fill_uniform(ws, gen, static_cast<value_t>(-30), static_cast<value_t>(30));

        simd_t vx = simd<value_t>::loadu(xs.data());
        simd_t vy = simd<value_t>::loadu(ys.data());
        simd_t vz = simd<value_t>::loadu(zs.data());
        simd_t vw = simd<value_t>::loadu(ws.data());

        mask_t m = simd<value_t>::gt(vx, vy);
        simd_t r = simd<value_t>::if_else(m, vz, vw);
        simd<value_t>::storeu(r, out.data());

        for (std::size_t i = 0; i < n; ++i)
        {
            value_t const ref = (xs[i] > ys[i]) ? zs[i] : ws[i];
            EXPECT_LE(std::fabs(out[i] - ref), tolerance);
        }
    }
}

template <typename value_t> void test_prefetch_copy_ploadquad_broadcast(value_t /* tolerance */)
{
    constexpr std::size_t n = simd<value_t>::size;
    using simd_t            = typename simd<value_t>::simd_t;
    SCALARS_ALIGN(SCALARS_ALIGNMENT) value_t block[64];
    SCALARS_ALIGN(SCALARS_ALIGNMENT) value_t dest[64];

    // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility
    std::mt19937 gen(271828u);
    for (std::size_t i = 0; i < n + 8; ++i)
    {
        block[i] = static_cast<value_t>(i * 3 - 17);
    }

    simd<value_t>::prefetch(block);

    simd<value_t>::copy(block, dest);
    for (std::size_t i = 0; i < n; ++i)
    {
        EXPECT_EQ(block[i], dest[i]);
    }

    if constexpr (detail::simd_has_ploadquad<value_t>::value)
    {
        simd_t                 q = simd<value_t>::ploadquad(block);
        std::array<value_t, n> qout{};
        simd<value_t>::storeu(q, qout.data());
        if constexpr (sizeof(value_t) == sizeof(float) && n == 8)
        {
            for (std::size_t i = 0; i < n / 2; ++i)
            {
                EXPECT_EQ(qout[i], block[0]);
            }
            for (std::size_t i = n / 2; i < n; ++i)
            {
                EXPECT_EQ(qout[i], block[1]);
            }
        }
        else if constexpr (sizeof(value_t) == sizeof(double) && n == 4)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                EXPECT_EQ(qout[i], block[0]);
            }
        }
        else if constexpr (sizeof(value_t) == sizeof(float) && n == 4)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                EXPECT_EQ(qout[i], block[i / 2]);
            }
        }
        else if constexpr (sizeof(value_t) == sizeof(double) && n == 8)
        {
            for (std::size_t i = 0; i < 4; ++i)
            {
                EXPECT_EQ(qout[i], block[0]);
            }
            for (std::size_t i = 4; i < 8; ++i)
            {
                EXPECT_EQ(qout[i], block[1]);
            }
        }
    }

    std::array<value_t, 4> four{};
    fill_uniform(four, gen, static_cast<value_t>(-3), static_cast<value_t>(3));
    simd_t b0;
    simd_t b1;
    simd_t b2;
    simd_t b3;
    simd<value_t>::broadcast(four.data(), b0, b1, b2, b3);
    std::array<value_t, n> lane0{};
    std::array<value_t, n> lane1{};
    std::array<value_t, n> lane2{};
    std::array<value_t, n> lane3{};
    simd<value_t>::storeu(b0, lane0.data());
    simd<value_t>::storeu(b1, lane1.data());
    simd<value_t>::storeu(b2, lane2.data());
    simd<value_t>::storeu(b3, lane3.data());
    for (std::size_t i = 0; i < n; ++i)
    {
        EXPECT_EQ(lane0[i], four[0]);
        EXPECT_EQ(lane1[i], four[1]);
        EXPECT_EQ(lane2[i], four[2]);
        EXPECT_EQ(lane3[i], four[3]);
    }

    simd_t                 s = simd<value_t>::set(static_cast<value_t>(1.375));
    std::array<value_t, n> sbuf{};
    simd<value_t>::storeu(s, sbuf.data());
    for (std::size_t i = 0; i < n; ++i)
    {
        EXPECT_EQ(sbuf[i], static_cast<value_t>(1.375));
    }
}

template <typename value_t> void test_ptranspose_square(value_t tolerance)
{
    constexpr int N = static_cast<int>(simd<value_t>::size);
    using simd_t    = typename simd<value_t>::simd_t;

    simd_t  rows[N];
    value_t before[N][N];
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            before[i][j] = static_cast<value_t>(i * 19 - j * 7 + 3);
        }
        rows[i] = simd<value_t>::loadu(before[i]);
    }

    simd<value_t>::template ptranspose<N>(rows);

    for (int i = 0; i < N; ++i)
    {
        std::array<value_t, simd<value_t>::size> out{};
        simd<value_t>::storeu(rows[i], out.data());
        for (int j = 0; j < N; ++j)
        {
            EXPECT_LE(std::fabs(out[j] - before[j][i]), tolerance);
        }
    }
}

template <typename value_t> void test_simd_half_fma_and_add(value_t tolerance)
{
    if constexpr (detail::simd_has_half_fma<value_t>::value &&
                  detail::simd_has_loadu_half<value_t>::value)
    {
        int const               nh = simd<value_t>::half_size;
        std::vector<value_t>    buf(64, static_cast<value_t>(0));
        std::array<value_t, 16> xs{};
        std::array<value_t, 16> ys{};
        std::array<value_t, 16> zs{};
        // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility
        std::mt19937 gen(60287u);

        for (int trial = 0; trial < kRandomTrials; ++trial)
        {
            fill_uniform(xs, gen, static_cast<value_t>(-2), static_cast<value_t>(2));
            fill_uniform(ys, gen, static_cast<value_t>(-2), static_cast<value_t>(2));
            fill_uniform(zs, gen, static_cast<value_t>(-2), static_cast<value_t>(2));

            auto vx  = simd<value_t>::loadu_half(xs.data());
            auto vy  = simd<value_t>::loadu_half(ys.data());
            auto acc = simd<value_t>::loadu_half(zs.data());
            acc      = simd<value_t>::fma(vx, vy, acc);

            std::fill(buf.begin(), buf.end(), static_cast<value_t>(0));
            simd<value_t>::scatter(acc, 1, buf.data());
            for (int i = 0; i < nh; ++i)
            {
                value_t const ref = std::fma(xs[i], ys[i], zs[i]);
                EXPECT_LE(std::fabs(buf[i] - ref), tolerance);
            }

            auto                                sx  = simd<value_t>::loadu_half(xs.data());
            auto                                sy  = simd<value_t>::loadu_half(ys.data());
            typename simd<value_t>::simd_half_t sum = simd<value_t>::add(sx, sy);
            std::fill(buf.begin(), buf.end(), static_cast<value_t>(0));
            simd<value_t>::scatter(sum, 1, buf.data());
            for (int i = 0; i < nh; ++i)
            {
                value_t const ref = xs[i] + ys[i];
                EXPECT_LE(std::fabs(buf[i] - ref), tolerance);
            }
        }
    }
}

template <typename value_t> void test_predux_downto4_maybe(value_t tolerance)
{
    if constexpr (detail::simd_has_predux_downto4<value_t>::value)
    {
        constexpr std::size_t n = simd<value_t>::size;
        using simd_t            = typename simd<value_t>::simd_t;

        std::array<value_t, n> xs{};
        // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility
        std::mt19937 gen(1618u);
        for (int trial = 0; trial < kRandomTrials; ++trial)
        {
            fill_uniform(xs, gen, static_cast<value_t>(-5), static_cast<value_t>(5));
            simd_t a = simd<value_t>::loadu(xs.data());
            auto   h = simd<value_t>::predux_downto4(a);

            std::vector<value_t> buf(32, static_cast<value_t>(0));
            simd<value_t>::scatter(h, 1, buf.data());

            if constexpr (std::is_same_v<value_t, float> && n == 4)
            {
                value_t const e0 = xs[0] + xs[2];
                value_t const e1 = xs[1] + xs[3];
                EXPECT_LE(std::fabs(buf[0] - e0), tolerance);
                EXPECT_LE(std::fabs(buf[1] - e1), tolerance);
            }
            else if constexpr (std::is_same_v<value_t, double> && n == 8)
            {
                for (int i = 0; i < 4; ++i)
                {
                    value_t const ref = xs[i] + xs[i + 4];
                    EXPECT_LE(std::fabs(buf[i] - ref), tolerance);
                }
            }
            else
            {
                for (std::size_t i = 0; i < simd<value_t>::half_size; ++i)
                {
                    EXPECT_TRUE(std::isfinite(buf[i]));
                }
            }
        }
    }
}

template <typename value_t> void test_gather_half_stride(value_t /* tolerance */)
{
    if constexpr (detail::simd_has_half_gather<value_t>::value)
    {
        int const            nh = simd<value_t>::half_size;
        std::vector<value_t> mem(32);
        std::vector<value_t> buf(32, static_cast<value_t>(0));
        for (std::size_t i = 0; i < mem.size(); ++i)
        {
            mem[i] = static_cast<value_t>(static_cast<int>(i) - 10);
        }

        for (int stride = 1; stride <= 3; ++stride)
        {
            typename simd<value_t>::simd_half_t h = simd<value_t>::gather_half(mem.data(), stride);
            std::fill(buf.begin(), buf.end(), static_cast<value_t>(0));
            simd<value_t>::scatter(h, 1, buf.data());
            for (int i = 0; i < nh; ++i)
            {
                EXPECT_TRUE(std::isfinite(buf[i]));
            }
        }
    }
}

template <typename value_t> void test_extended_simd_ops(value_t tolerance)
{
    test_fma(tolerance);
    test_if_else(tolerance);
    test_prefetch_copy_ploadquad_broadcast(tolerance);
#if !(defined(SCALARS_HAS_SVE) && SCALARS_HAS_SVE)
    test_ptranspose_square(tolerance);
#endif
    test_simd_half_fma_and_add(tolerance);
    test_predux_downto4_maybe(tolerance);
    test_gather_half_stride(tolerance);
}

// Vectorization's upstream test uses an aligned-allocator helper (allocator<T>)
// around this gather/scatter check; Scalars has no such allocator abstraction,
// so a fixed-size SCALARS_ALIGN'd stack buffer stands in for it instead.
template <typename value_t> void test_gather_scatter_and_masks(value_t)
{
    using simd_t    = typename simd<value_t>::simd_t;
    using mask_t    = typename simd<value_t>::mask_t;
    using vec_int_t = typename simd<value_t>::int_t;

    constexpr std::size_t n         = 515;
    constexpr std::size_t simd_size = simd<value_t>::size;

    SCALARS_ALIGN(SCALARS_ALIGNMENT) value_t x[n];
    SCALARS_ALIGN(SCALARS_ALIGNMENT) value_t y[n] = {};

    auto const dt = value_t(10) / static_cast<value_t>(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        x[i] = static_cast<value_t>(-5) + static_cast<value_t>(i) * dt;
    }

    for (std::size_t i = 0; i + simd_size <= n; i += simd_size)
    {
        simd_t tmp_load = simd<value_t>::load(&x[i]);
        simd<value_t>::store(tmp_load, &x[i]);
    }

    int const offset = static_cast<int>(n / simd_size);

    simd_t ret = simd<value_t>::gather(x, offset);
    simd<value_t>::scatter(ret, offset, y);
    for (std::size_t i = 0; i < simd_size; ++i)
    {
        EXPECT_EQ(x[i * static_cast<std::size_t>(offset)], y[i * static_cast<std::size_t>(offset)]);
    }

    int indices[simd_size];
    for (std::size_t i = 0; i < simd_size; ++i)
    {
        indices[i] = static_cast<int>(i * (n / simd_size));
    }

    ret = simd<value_t>::gather(x, indices);
    simd<value_t>::scatter(ret, indices, y);

    for (std::size_t i = 0; i < simd_size; ++i)
    {
        std::size_t const k = i * (n / simd_size);
        EXPECT_EQ(x[k], y[k]);
    }

    simd_t a = simd<value_t>::set(static_cast<value_t>(0.1));
    simd_t b = simd<value_t>::set(static_cast<value_t>(0.1));

    SCALARS_ALIGN(SCALARS_ALIGNMENT) vec_int_t tmp[simd_size];

    mask_t m = simd<value_t>::eq(a, b);
    simd<value_t>::storeu(m, &tmp[0]);
    m = simd<value_t>::loadu(&tmp[0]);

    m = simd<value_t>::load(&tmp[0]);
    simd<value_t>::store(m, &tmp[0]);

    m = simd<value_t>::neq(a, b);
    simd<value_t>::storeu(m, &tmp[0]);

    m = simd<value_t>::gt(a, b);
    simd<value_t>::storeu(m, &tmp[0]);

    m = simd<value_t>::lt(a, b);
    simd<value_t>::storeu(m, &tmp[0]);

    m = simd<value_t>::ge(a, b);
    simd<value_t>::storeu(m, &tmp[0]);

    mask_t l = simd<value_t>::le(a, b);
    simd<value_t>::storeu(l, &tmp[0]);

    mask_t o = simd<value_t>::and_mask(m, l);
    simd<value_t>::storeu(o, &tmp[0]);

    o = simd<value_t>::or_mask(m, l);
    simd<value_t>::storeu(o, &tmp[0]);

    o = simd<value_t>::xor_mask(m, l);
    simd<value_t>::storeu(o, &tmp[0]);

    o = simd<value_t>::not_mask(m);
    simd<value_t>::storeu(o, &tmp[0]);
}

template <typename value_t> void test_simd_misc_all(value_t tolerance)
{
    test_extended_simd_ops(tolerance);
    test_gather_scatter_and_masks(tolerance);
}

template <typename> struct simd_tolerance
{
};

template <> struct simd_tolerance<float>
{
    static constexpr float tolerance = 0.0007f;
};

template <> struct simd_tolerance<double>
{
    static constexpr double tolerance = 5.e-12;
};

}  // namespace

TEST(Simd, Utility)
{
    test_simd_misc_all<float>(simd_tolerance<float>::tolerance);
    test_simd_misc_all<double>(simd_tolerance<double>::tolerance);
}

#endif  // SCALARS_VECTORIZED
