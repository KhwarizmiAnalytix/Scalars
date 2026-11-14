#include "backend/simd.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#if SCALARS_VECTORIZED

using scalars::simd;

namespace
{
template <class S>
concept complex_packet_api = requires(typename S::simd_t a,
    typename S::mask_t                                   m,
    typename S::simd_half_t                              h,
    typename S::value_t*                                 out,
    const typename S::value_t*                           in,
    typename S::int_t*                                   flags,
    const int*                                           indices) {
    typename S::simd_int_t;
    typename S::real_packet_t;
    { S::add(a, a) } -> std::same_as<typename S::simd_t>;
    { S::sub(a, a) } -> std::same_as<typename S::simd_t>;
    { S::mul(a, a) } -> std::same_as<typename S::simd_t>;
    { S::div(a, a) } -> std::same_as<typename S::simd_t>;
    { S::sqr(a) } -> std::same_as<typename S::simd_t>;
    { S::neg(a) } -> std::same_as<typename S::simd_t>;
    { S::fma(a, a, a) } -> std::same_as<typename S::simd_t>;
    { S::conj(a) } -> std::same_as<typename S::simd_t>;
    { S::sqrt(a) } -> std::same_as<typename S::simd_t>;
    { S::invsqrt(a) } -> std::same_as<typename S::simd_t>;
    { S::pow(a, a) } -> std::same_as<typename S::simd_t>;
    { S::exp(a) } -> std::same_as<typename S::simd_t>;
    { S::expm1(a) } -> std::same_as<typename S::simd_t>;
    { S::exp2(a) } -> std::same_as<typename S::simd_t>;
    { S::exp10(a) } -> std::same_as<typename S::simd_t>;
    { S::log(a) } -> std::same_as<typename S::simd_t>;
    { S::log1p(a) } -> std::same_as<typename S::simd_t>;
    { S::log2(a) } -> std::same_as<typename S::simd_t>;
    { S::log10(a) } -> std::same_as<typename S::simd_t>;
    { S::sin(a) } -> std::same_as<typename S::simd_t>;
    { S::cos(a) } -> std::same_as<typename S::simd_t>;
    { S::tan(a) } -> std::same_as<typename S::simd_t>;
    { S::asin(a) } -> std::same_as<typename S::simd_t>;
    { S::acos(a) } -> std::same_as<typename S::simd_t>;
    { S::atan(a) } -> std::same_as<typename S::simd_t>;
    { S::sinh(a) } -> std::same_as<typename S::simd_t>;
    { S::cosh(a) } -> std::same_as<typename S::simd_t>;
    { S::tanh(a) } -> std::same_as<typename S::simd_t>;
    { S::asinh(a) } -> std::same_as<typename S::simd_t>;
    { S::acosh(a) } -> std::same_as<typename S::simd_t>;
    { S::atanh(a) } -> std::same_as<typename S::simd_t>;
    { S::ceil(a) } -> std::same_as<typename S::simd_t>;
    { S::floor(a) } -> std::same_as<typename S::simd_t>;
    { S::trunc(a) } -> std::same_as<typename S::simd_t>;
    S::eq(a, a);
    S::neq(a, a);
    S::and_mask(m, m);
    S::or_mask(m, m);
    S::xor_mask(m, m);
    S::not_mask(m);
    { S::if_else(m, a, a) } -> std::same_as<typename S::simd_t>;
    S::load(flags);
    S::loadu(flags);
    S::store(m, flags);
    S::storeu(m, flags);
    S::set_mask(true);
    S::set_mask(typename S::int_t(1));
    S::prefetch(in);
    S::copy(in, out);
    S::broadcast(in, a, a, a, a);
    S::gather(in, indices);
    S::scatter(a, indices, out);
    S::ploadquad(in);
    S::template ptranspose<S::size>(&a);
    S::loadu_half(in);
    S::gather_half(in, 1);
    S::gather_half(in, indices);
    S::set(in);
    S::add(h, h);
    S::sub(h, h);
    S::mul(h, h);
    S::fma(h, h, h);
    S::storeu(h, out);
    S::scatter(h, 1, out);
    S::scatter(h, indices, out);
    S::storeu(S::abs(a), static_cast<typename S::r::value_t*>(nullptr));
    S::abs2(a);
    S::fabs(a);
    S::arg(a);
    S::real(a);
    S::imag(a);
    { S::accumulate(a) } -> std::same_as<typename S::accumulation_t>;
};

template <class S> constexpr bool excludes_real_only()
{
    using v = typename S::simd_t;
    return !requires(v a) { S::gt(a, a); } && !requires(v a) { S::ge(a, a); } &&
           !requires(v a) { S::lt(a, a); } && !requires(v a) { S::le(a, a); } &&
           !requires(v a) { S::min(a, a); } && !requires(v a) { S::max(a, a); } &&
           !requires(v a) { S::hmin(a); } && !requires(v a) { S::hmax(a); } &&
           !requires(v a) { S::signcopy(a, a); } && !requires(v a) { S::hypot(a, a); } &&
           !requires(v a) { S::cbrt(a); } && !requires(v a) { S::cdf(a); } &&
           !requires(v a) { S::inv_cdf(a); };
}

static_assert(complex_packet_api<simd<scalars::complex<float>>>);
static_assert(complex_packet_api<simd<scalars::complex<double>>>);
static_assert(excludes_real_only<simd<scalars::complex<float>>>());
static_assert(excludes_real_only<simd<scalars::complex<double>>>());

// Normwise relative tolerance: errors are measured against `scale`, the size of
// the whole expected result (max(|re|, |im|) for complex values, |expected| for
// real ones), with a floor only at the smallest normal number. A fixed absolute
// floor of 1 would hide relative-accuracy regressions in small results (see
// plan item 4.1), while a per-component relative bound would reject legitimate
// rounding residue, such as a ~1e-16 imaginary part next to a real part near 1.
template <class T>
void expect_component(T actual, T expected, bool check_zero_sign = false, T scale = T(-1))
{
    if (std::isnan(expected))
    {
        EXPECT_TRUE(std::isnan(actual));
    }
    else if (std::isinf(expected))
    {
        EXPECT_EQ(actual, expected);
    }
    else
    {
        const T magnitude = std::max(scale, std::fabs(expected));
        const T tolerance =
            64 * std::numeric_limits<T>::epsilon() * magnitude + std::numeric_limits<T>::min();
        EXPECT_NEAR(actual, expected, tolerance);
        if (check_zero_sign && expected == T(0))
        {
            EXPECT_EQ(actual, T(0));
            EXPECT_EQ(std::signbit(actual), std::signbit(expected));
        }
    }
}
template <class T>
void expect_value(
    scalars::complex<T> actual, std::complex<T> expected, bool check_zero_sign = false)
{
    T scale = T(0);
    for (const T component : {expected.real(), expected.imag()})
    {
        if (std::isfinite(component))
        {
            scale = std::max(scale, std::fabs(component));
        }
    }
    expect_component(actual.real(), expected.real(), check_zero_sign, scale);
    expect_component(actual.imag(), expected.imag(), check_zero_sign, scale);
}
// Reference square root that stays accurate for subnormal inputs. libc++
// computes std::sqrt(z) as polar(sqrt(abs(z)), arg(z) / 2), and abs(z)
// rounds to the subnormal grid first: |denorm_min * (1 - i)| becomes
// denorm_min instead of sqrt(2) * denorm_min, a 2^(1/4) error in the result.
// Scaling by an even power of two, 2^(2k), is exact here and divides out as
// 2^k after the square root.
template <class T> std::complex<T> reference_sqrt(scalars::complex<T> value)
{
    const std::complex<T> z(value.real(), value.imag());
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag()) ||
        std::fabs(z.real()) >= std::numeric_limits<T>::min() ||
        std::fabs(z.imag()) >= std::numeric_limits<T>::min() ||
        (z.real() == T(0) && z.imag() == T(0)))
    {
        return std::sqrt(z);
    }
    constexpr int k = std::numeric_limits<T>::digits;
    const auto    r =
        std::sqrt(std::complex<T>(std::ldexp(z.real(), 2 * k), std::ldexp(z.imag(), 2 * k)));
    return {std::ldexp(r.real(), -k), std::ldexp(r.imag(), -k)};
}
template <class T> void expect_bits(scalars::complex<T> actual, scalars::complex<T> expected)
{
    using bits = typename simd<scalars::complex<T>>::int_t;
    EXPECT_EQ(std::bit_cast<bits>(actual.real()), std::bit_cast<bits>(expected.real()));
    EXPECT_EQ(std::bit_cast<bits>(actual.imag()), std::bit_cast<bits>(expected.imag()));
}

template <class T> void check_masks()
{
    using c    = scalars::complex<T>;
    using s    = simd<c>;
    using bits = typename s::int_t;
    static_assert(sizeof(bits) == sizeof(T));
    std::array<c, s::size>    x, y, out;
    std::array<bits, s::size> flags;
    for (int trial = 0; trial < 8; ++trial)
    {
        for (int i = 0; i < s::size; ++i)
        {
            x[i] = c(T(i), T(-i));
            y[i] = x[i];
            if ((i + trial) % 3 == 0)
            {
                y[i].imag(T(i + 1));
            }
            if ((i + trial) % 5 == 0)
            {
                x[i] = c(T(0), T(-0.0));
                y[i] = c(T(-0.0), T(0));
            }
            if ((i + trial) % 7 == 0)
            {
                x[i].real(std::numeric_limits<T>::quiet_NaN());
                y[i] = x[i];
            }
        }
        const auto a = s::loadu(x.data()), b = s::loadu(y.data());
        const auto equal = s::eq(a, b), different = s::neq(a, b);
        s::storeu(equal, flags.data());
        for (int i = 0; i < s::size; ++i)
        {
            EXPECT_EQ(flags[i], x[i] == y[i] ? ~bits(0) : bits(0));
        }
        s::storeu(different, flags.data());
        for (int i = 0; i < s::size; ++i)
        {
            EXPECT_EQ(flags[i], x[i] != y[i] ? ~bits(0) : bits(0));
        }
        s::storeu(s::if_else(equal, a, b), out.data());
        for (int i = 0; i < s::size; ++i)
        {
            expect_bits(out[i], x[i] == y[i] ? x[i] : y[i]);
        }
        s::storeu(s::and_mask(equal, different), flags.data());
        for (const auto flag : flags)
        {
            EXPECT_EQ(flag, bits(0));
        }
        s::storeu(s::or_mask(equal, different), flags.data());
        for (const auto flag : flags)
        {
            EXPECT_EQ(flag, ~bits(0));
        }
        s::storeu(s::xor_mask(equal, different), flags.data());
        for (const auto flag : flags)
        {
            EXPECT_EQ(flag, ~bits(0));
        }
        s::storeu(s::not_mask(equal), flags.data());
        for (int i = 0; i < s::size; ++i)
        {
            EXPECT_EQ(flags[i], x[i] != y[i] ? ~bits(0) : bits(0));
        }
    }
    for (int i = 0; i < s::size; ++i)
    {
        flags[i] = i % 2 ? bits(7) : bits(0);
    }
    const auto memory_mask = s::loadu(flags.data());
    s::storeu(memory_mask, flags.data());
    for (int i = 0; i < s::size; ++i)
    {
        EXPECT_EQ(flags[i], i % 2 ? ~bits(0) : bits(0));
    }
    for (int count = 0; count <= s::size; ++count)
    {
        std::array<bits, s::size + 2> partial;
        partial.fill(bits(42));
        s::storeu(s::loadu(flags.data(), count), partial.data() + 1, count);
        EXPECT_EQ(partial.front(), bits(42));
        for (int i = 0; i < count; ++i)
        {
            EXPECT_EQ(partial[i + 1], flags[i]);
        }
        for (int i = count + 1; i < s::size + 2; ++i)
        {
            EXPECT_EQ(partial[i], bits(42));
        }
    }
    s::storeu(s::loadu(static_cast<const bits*>(nullptr), 0), static_cast<bits*>(nullptr), 0);
    s::storeu(s::set_mask(bits(1)), flags.data());
    for (const auto flag : flags)
    {
        EXPECT_EQ(flag, ~bits(0));
    }
    s::storeu(s::set_mask(false), flags.data());
    for (const auto flag : flags)
    {
        EXPECT_EQ(flag, bits(0));
    }
    // Plain integer literals must compile (they were ambiguous between the bool
    // and int_t overloads) and follow the nonzero-selects-all rule.
    s::storeu(s::set_mask(1), flags.data());
    for (const auto flag : flags)
    {
        EXPECT_EQ(flag, ~bits(0));
    }
    s::storeu(s::set_mask(0), flags.data());
    for (const auto flag : flags)
    {
        EXPECT_EQ(flag, bits(0));
    }
    s::storeu(s::set_mask(std::int64_t(7)), flags.data());
    for (const auto flag : flags)
    {
        EXPECT_EQ(flag, ~bits(0));
    }
    s::storeu(s::set_mask(0u), flags.data());
    for (const auto flag : flags)
    {
        EXPECT_EQ(flag, bits(0));
    }
    s::storeu(s::set(2), out.data());
    for (const auto value : out)
    {
        EXPECT_EQ(value, c(T(2), T(0)));
    }
}

template <class T, int N> void check_transpose()
{
    using c = scalars::complex<T>;
    using s = simd<c>;
    std::array<typename s::simd_t, N>                    rows;
    std::array<c, static_cast<std::size_t>(N) * s::size> input, output;
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = c(T(i + 1), -T(i + 3));
    }
    for (int row = 0; row < N; ++row)
    {
        rows[row] = s::loadu(input.data() + row * s::size);
    }
    s::template ptranspose<N>(rows.data());
    for (int row = 0; row < N; ++row)
    {
        s::storeu(rows[row], output.data() + row * s::size);
    }
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        EXPECT_EQ(output[i], input[(i % N) * s::size + i / N]);
    }
}

template <class T> void check_memory()
{
    using c                    = scalars::complex<T>;
    using s                    = simd<c>;
    constexpr int            w = s::size;
    const c                  sentinel(T(97), T(-63));
    std::array<c, 2 * w + 8> input, output, expected;
    for (int i = 0; i < 2 * w + 8; ++i)
    {
        input[i] = c(T(i + 1) / 4, T(2 - i) / 8);
    }
    s::prefetch(input.data());
    output.fill(sentinel);
    s::copy(input.data() + 1, output.data() + 1);
    EXPECT_EQ(output.front(), sentinel);
    for (int i = 1; i <= w; ++i)
    {
        EXPECT_EQ(output[i], input[i]);
    }
    EXPECT_EQ(output[w + 1], sentinel);
    typename s::simd_t a, b, c0, d;
    s::broadcast(input.data(), a, b, c0, d);
    const std::array<typename s::simd_t, 4> broadcasts{a, b, c0, d};
    for (int i = 0; i < 4; ++i)
    {
        s::storeu(broadcasts[i], output.data());
        for (int j = 0; j < w; ++j)
        {
            EXPECT_EQ(output[j], input[i]);
        }
    }
    constexpr int repeats = std::is_same_v<T, float> && w == 4 ? 2 : 4;
    s::storeu(s::ploadquad(input.data()), output.data());
    for (int i = 0; i < w; ++i)
    {
        EXPECT_EQ(output[i], input[i / repeats]);
    }
    std::array<int, w> indices;
    for (int i = 0; i < w; ++i)
    {
        indices[i] = w - 1 - i - w / 2;
    }
    a = s::gather(input.data() + w, indices.data());
    s::storeu(a, output.data());
    for (int i = 0; i < w; ++i)
    {
        EXPECT_EQ(output[i], input[w + indices[i]]);
    }
    for (int i = 0; i < w; ++i)
    {
        indices[i] = i % 2;
    }
    output.fill(sentinel);
    expected = output;
    for (int i = 0; i < w; ++i)
    {
        expected[indices[i]] = input[i];
    }
    s::scatter(s::loadu(input.data()), indices.data(), output.data());
    EXPECT_EQ(output, expected);

    const auto h = s::loadu_half(input.data());
    const auto k = s::gather_half(input.data() + w, -1);
    output.fill(sentinel);
    s::storeu(s::add(h, h), output.data());
    for (int i = 0; i < s::half_size; ++i)
    {
        EXPECT_EQ(output[i], input[i] + input[i]);
    }
    for (int i = s::half_size; i < 2 * w + 8; ++i)
    {
        EXPECT_EQ(output[i], sentinel);
    }
    output.fill(sentinel);
    s::scatter(s::fma(h, k, h), 2, output.data());
    for (int i = 0; i < s::half_size; ++i)
    {
        const std::complex<T> x(input[i].real(), input[i].imag());
        const std::complex<T> y(input[w - i].real(), input[w - i].imag());
        expect_value(output[2 * i], x * y + x);
        EXPECT_EQ(output[2 * i + 1], sentinel);
    }
    output.fill(sentinel);
    s::storeu(s::set(input.data()).value, output.data());
    for (int i = 0; i < s::half_size; ++i)
    {
        EXPECT_EQ(output[i], input[0]);
    }
    for (int i = s::half_size; i < w; ++i)
    {
        EXPECT_EQ(output[i], c{});
    }
    for (int i = 0; i < w; ++i)
    {
        indices[i] = 2 * i;
    }
    output.fill(sentinel);
    s::scatter(s::gather_half(input.data(), indices.data()), indices.data(), output.data());
    for (int i = 0; i < s::half_size; ++i)
    {
        EXPECT_EQ(output[indices[i]], input[indices[i]]);
    }
    if constexpr (w >= 4)
    {
        s::storeu(s::predux_downto4(s::loadu(input.data())), output.data());
        for (int i = 0; i < s::half_size; ++i)
        {
            EXPECT_EQ(output[i], input[i] + input[i + s::half_size]);
        }
    }
    check_transpose<T, 1>();
    check_transpose<T, 2>();
    check_transpose<T, 4>();
    check_transpose<T, s::size>();
}

template <class T, class PacketFunction, class Reference>
void check_unary(PacketFunction packet, Reference reference, bool zero_sign = false)
{
    using c = scalars::complex<T>;
    using s = simd<c>;
    const std::array<c, 16> values{c(T(.25), T(-.5)),
        c(T(-1.5), T(.25)),
        c(T(2), T(-.75)),
        c(T(-4), T(0)),
        c(T(-4), T(-0.0)),
        c(T(0), T(0)),
        c(T(0), T(-0.0)),
        c(T(1), T(0)),
        c(T(-1), T(-0.0)),
        c(T(0), T(1)),
        c(T(0), T(-1)),
        c(T(-.125), T(.875)),
        c(T(.2), T(.3)),
        c(T(1.5), T(-.25)),
        c(T(0), T(.5)),
        c(T(-.2), T(.25))};
    std::array<c, s::size>  input, output;
    for (std::size_t start = 0; start < values.size(); start += s::size)
    {
        for (int i = 0; i < s::size; ++i)
        {
            input[i] = values[(start + i) % values.size()];
        }
        s::storeu(packet(s::loadu(input.data())), output.data());
        for (int i = 0; i < s::size; ++i)
        {
            expect_value(
                output[i], reference(std::complex<T>(input[i].real(), input[i].imag())), zero_sign);
        }
    }
}

template <class T> void check_math()
{
    using s = simd<scalars::complex<T>>;
    using v = typename s::simd_t;
    check_unary<T>([](v a) { return s::sqr(a); }, [](auto z) { return z * z; });
    check_unary<T>([](v a) { return s::sqrt(a); }, [](auto z) { return std::sqrt(z); }, true);
    check_unary<T>([](v a) { return s::invsqrt(a); }, [](auto z) { return T(1) / std::sqrt(z); });
    check_unary<T>([](v a) { return s::exp(a); }, [](auto z) { return std::exp(z); });
    check_unary<T>([](v a) { return s::expm1(a); }, [](auto z) { return std::exp(z) - T(1); });
    check_unary<T>([](v a) { return s::exp2(a); },
        [](auto z) { return std::exp(z * std::numbers::ln2_v<T>); });
    check_unary<T>([](v a) { return s::exp10(a); },
        [](auto z) { return std::exp(z * std::numbers::ln10_v<T>); });
    check_unary<T>([](v a) { return s::log(a); }, [](auto z) { return std::log(z); }, true);
    check_unary<T>([](v a) { return s::log1p(a); }, [](auto z) { return std::log(z + T(1)); });
    check_unary<T>([](v a) { return s::log2(a); },
        [](auto z) { return std::log(z) / std::numbers::ln2_v<T>; });
    check_unary<T>([](v a) { return s::log10(a); }, [](auto z) { return std::log10(z); });
    check_unary<T>([](v a) { return s::sin(a); }, [](auto z) { return std::sin(z); });
    check_unary<T>([](v a) { return s::cos(a); }, [](auto z) { return std::cos(z); });
    check_unary<T>([](v a) { return s::tan(a); }, [](auto z) { return std::tan(z); });
    check_unary<T>([](v a) { return s::asin(a); }, [](auto z) { return std::asin(z); });
    check_unary<T>([](v a) { return s::acos(a); }, [](auto z) { return std::acos(z); }, true);
    check_unary<T>([](v a) { return s::atan(a); }, [](auto z) { return std::atan(z); });
    check_unary<T>([](v a) { return s::sinh(a); }, [](auto z) { return std::sinh(z); });
    check_unary<T>([](v a) { return s::cosh(a); }, [](auto z) { return std::cosh(z); });
    check_unary<T>([](v a) { return s::tanh(a); }, [](auto z) { return std::tanh(z); });
    check_unary<T>([](v a) { return s::asinh(a); }, [](auto z) { return std::asinh(z); });
    check_unary<T>([](v a) { return s::acosh(a); }, [](auto z) { return std::acosh(z); });
    check_unary<T>([](v a) { return s::atanh(a); }, [](auto z) { return std::atanh(z); });
    check_unary<T>([](v a) { return s::ceil(a); },
        [](auto z) { return std::complex<T>(std::ceil(z.real()), std::ceil(z.imag())); },
        true);
    check_unary<T>([](v a) { return s::floor(a); },
        [](auto z) { return std::complex<T>(std::floor(z.real()), std::floor(z.imag())); },
        true);
    check_unary<T>([](v a) { return s::trunc(a); },
        [](auto z) { return std::complex<T>(std::trunc(z.real()), std::trunc(z.imag())); },
        true);
    const scalars::complex<T> exponent(T(.75), T(-.25));
    check_unary<T>([&](v a) { return s::pow(a, s::set(exponent)); },
        [&](auto z) { return std::pow(z, std::complex<T>(exponent.real(), exponent.imag())); });
}

template <class T> void check_sqrt_bounds()
{
    using c                      = scalars::complex<T>;
    using s                      = simd<c>;
    const T                lower = scalars::detail_complex_simd::division_lower<T>();
    const T                upper = scalars::detail_complex_simd::division_upper<T>();
    const std::array<T, 9> values{T(0),
        T(-0.0),
        lower,
        std::nextafter(lower, T(0)),
        std::nextafter(lower, upper),
        T(1),
        std::nextafter(upper, T(0)),
        upper,
        std::nextafter(upper, std::numeric_limits<T>::infinity())};
    std::array<c, s::size> input, output;
    for (T real : values)
    {
        for (T imag : values)
        {
            for (T sign : {T(1), T(-1)})
            {
                for (int lane = 0; lane < s::size; ++lane)
                {
                    input[lane] = c(sign * real, lane % 2 ? -imag : imag);
                }
                s::storeu(s::sqrt(s::loadu(input.data())), output.data());
                for (int lane = 0; lane < s::size; ++lane)
                {
                    // The scalar path normalizes before taking roots; compare
                    // relative component errors, including small branch-cut parts.
                    const auto reference = scalars::sqrt(input[lane]);
                    for (const auto pair : {std::array<T, 2>{output[lane].real(), reference.real()},
                             std::array<T, 2>{output[lane].imag(), reference.imag()}})
                    {
                        EXPECT_NEAR(pair[0],
                            pair[1],
                            T(16) * std::numeric_limits<T>::epsilon() * std::fabs(pair[1]));
                        if (pair[1] == T(0))
                        {
                            EXPECT_EQ(std::signbit(pair[0]), std::signbit(pair[1]));
                        }
                    }
                }
            }
        }
    }
}

template <class T> void check_magnitude_and_edges()
{
    using c          = scalars::complex<T>;
    using s          = simd<c>;
    using bits       = typename s::int_t;
    const T infinity = std::numeric_limits<T>::infinity(),
            nan      = std::numeric_limits<T>::quiet_NaN();
    const T maximum = std::numeric_limits<T>::max(), tiny = std::numeric_limits<T>::denorm_min();
    const std::array<c, 16>   edges{c(3, 4),
        c(maximum, maximum),
        c(tiny, -tiny),
        c(0, T(-0.0)),
        c(T(-0.0), T(-0.0)),
        c(infinity, 0),
        c(-infinity, T(-0.0)),
        c(1, infinity),
        c(nan, 1),
        c(infinity, nan),
        c(T(-4), T(-0.0)),
        c(T(-4), T(0)),
        c(-maximum, -maximum),
        c(T(0x1p31) + T(.5), -T(0x1p31) - T(.5)),
        c(T(0x1p23) - T(.5), -T(0x1p23) + T(.5)),
        c(T(.75), T(-.75))};
    std::array<c, s::size>    input, output;
    std::array<T, s::size>    real;
    std::array<bits, s::size> flags;
    for (const auto edge : edges)
    {
        input.fill(edge);
        const auto a = s::loadu(input.data());
        s::storeu(s::abs(a), real.data());
        for (const T value : real)
        {
            expect_component(value, std::hypot(edge.real(), edge.imag()));
        }
        s::storeu(s::fabs(a), real.data());
        for (const T value : real)
        {
            expect_component(value, std::hypot(edge.real(), edge.imag()));
        }
        s::storeu(s::arg(a), real.data());
        for (const T value : real)
        {
            expect_component(value, std::atan2(edge.imag(), edge.real()), true);
        }
        s::storeu(s::abs2(a), real.data());
        for (const T value : real)
        {
            expect_component(value, edge.real() * edge.real() + edge.imag() * edge.imag());
        }
        s::storeu(s::real(a), real.data());
        for (const T value : real)
        {
            EXPECT_EQ(std::bit_cast<bits>(value), std::bit_cast<bits>(edge.real()));
        }
        s::storeu(s::imag(a), real.data());
        for (const T value : real)
        {
            EXPECT_EQ(std::bit_cast<bits>(value), std::bit_cast<bits>(edge.imag()));
        }
        s::storeu(s::isfinite(a), flags.data());
        for (const auto flag : flags)
        {
            EXPECT_EQ(flag, scalars::isfinite(edge) ? ~bits(0) : bits(0));
        }
        s::storeu(s::isnan(a), flags.data());
        for (const auto flag : flags)
        {
            EXPECT_EQ(
                flag, std::isnan(edge.real()) || std::isnan(edge.imag()) ? ~bits(0) : bits(0));
        }
        s::storeu(s::isinf(a), flags.data());
        for (const auto flag : flags)
        {
            EXPECT_EQ(
                flag, std::isinf(edge.real()) || std::isinf(edge.imag()) ? ~bits(0) : bits(0));
        }
        s::storeu(s::sqrt(a), output.data());
        for (const auto value : output)
        {
            if (std::isfinite(edge.real()) && std::isfinite(edge.imag()) &&
                (std::fabs(edge.real()) > maximum / T(4) ||
                    std::fabs(edge.imag()) > maximum / T(4)))
            {
                // Verify the square-root identity after scaling. Some standard
                // libraries overflow their own reference sqrt(max + max*i).
                EXPECT_TRUE(std::isfinite(value.real()));
                EXPECT_TRUE(std::isfinite(value.imag()));
                const T               scale = std::sqrt(maximum);
                const std::complex<T> normalized(value.real() / scale, value.imag() / scale);
                expect_value(c(normalized * normalized),
                    std::complex<T>(edge.real() / maximum, edge.imag() / maximum));
                EXPECT_GE(value.real(), T(0));
                EXPECT_EQ(std::signbit(value.imag()), std::signbit(edge.imag()));
            }
            else
            {
                expect_value(value, reference_sqrt(edge), true);
            }
        }
        s::storeu(s::log(a), output.data());
        for (const auto value : output)
        {
            expect_value(value, std::log(std::complex<T>(edge.real(), edge.imag())), true);
        }
        const std::array<typename s::simd_t, 3> rounded{s::ceil(a), s::floor(a), s::trunc(a)};
        const std::array<std::complex<T>, 3>    references{
            std::complex<T>(std::ceil(edge.real()), std::ceil(edge.imag())),
            std::complex<T>(std::floor(edge.real()), std::floor(edge.imag())),
            std::complex<T>(std::trunc(edge.real()), std::trunc(edge.imag()))};
        for (std::size_t operation = 0; operation < rounded.size(); ++operation)
        {
            s::storeu(rounded[operation], output.data());
            for (const auto value : output)
            {
                expect_value(value, references[operation], true);
            }
        }
        s::storeu(s::neg(a), output.data());
        for (const auto value : output)
        {
            expect_bits(value, c(-edge.real(), -edge.imag()));
        }
    }
    for (T x : {T(0), T(-0.0), T(1e-8), T(-1e-8), tiny, infinity, -infinity})
    {
        const auto a = s::set(c(x, T(-0.0)));
        s::storeu(s::expm1(a), output.data());
        for (const auto value : output)
        {
            expect_value(value, std::complex<T>(std::expm1(x), T(-0.0)), true);
        }
        s::storeu(s::log1p(a), output.data());
        if (x >= T(-1))
        {
            for (const auto value : output)
            {
                expect_value(value, std::complex<T>(std::log1p(x), T(-0.0)), true);
            }
        }
    }
    static_assert(std::is_same_v<typename s::accumulation_t, scalars::complex<double>>);
    input.fill(c{});
    if constexpr (std::is_same_v<T, float>)
    {
        input[0]       = c(T(1e8), T(-1e8));
        input[1]       = c(T(-1e8), T(1e8));
        input[2]       = c(T(1), T(-1));
        const auto sum = s::accumulate(s::loadu(input.data()));
        EXPECT_EQ(sum.real(), 1.0);
        EXPECT_EQ(sum.imag(), -1.0);
    }
}

template <class T> void check_mixed_math_edges()
{
    using c                         = scalars::complex<T>;
    using s                         = simd<c>;
    const T                 inf     = std::numeric_limits<T>::infinity();
    const T                 nan     = std::numeric_limits<T>::quiet_NaN();
    const T                 tiny    = std::numeric_limits<T>::denorm_min();
    const T                 maximum = std::numeric_limits<T>::max();
    const T                 limit   = std::is_same_v<T, float> ? T(80) : T(700);
    const std::array<c, 19> edges{c(0, -0.0),
        c(-4, -0.0),
        c(-4, 0),
        c(tiny, -tiny),
        c(maximum, maximum),
        c(-maximum, T(-0.0)),
        c(inf, 0),
        c(-inf, -0.0),
        c(nan, 1),
        c(inf, nan),
        c(1, inf),
        c(T(.5), tiny),
        c(limit, T(.25)),
        c(-limit, T(-.25)),
        c(std::nextafter(limit, T(0)), T(-.5)),
        c(T(-0.0), T(-0.0)),
        c(T(1), T(0)),
        c(T(0), T(1)),
        c(T(-1), tiny)};
    std::array<c, s::size>  input, output;
    std::array<T, s::size>  magnitude;
    for (const auto edge : edges)
    {
        for (int lane = 0; lane < s::size; ++lane)
        {
            input.fill(c(T(.25), T(-.5)));
            input[lane]       = edge;
            const auto packet = s::loadu(input.data());
            s::storeu(s::abs(packet), magnitude.data());
            for (int i = 0; i < s::size; ++i)
            {
                const T expected = std::hypot(input[i].real(), input[i].imag());
                expect_component(magnitude[i], expected, true);
                if (std::isfinite(expected) && expected != T(0))
                {
                    EXPECT_NEAR(magnitude[i],
                        expected,
                        T(8) * std::numeric_limits<T>::epsilon() * expected + T(2) * tiny);
                }
            }
            s::storeu(s::exp(packet), output.data());
            for (int i = 0; i < s::size; ++i)
            {
                expect_value(
                    output[i], std::exp(std::complex<T>(input[i].real(), input[i].imag())), true);
            }
            s::storeu(s::log(packet), output.data());
            for (int i = 0; i < s::size; ++i)
            {
                expect_value(
                    output[i], std::log(std::complex<T>(input[i].real(), input[i].imag())), true);
            }
        }
    }
}
}  // namespace

TEST(SimdComplexParity, MasksAndSelection)
{
    check_masks<float>();
    check_masks<double>();
}
TEST(SimdComplexParity, MemoryPackingAndHalfWidth)
{
    check_memory<float>();
    check_memory<double>();
}
TEST(SimdComplexParity, PrincipalMathAndRounding)
{
    check_math<float>();
    check_math<double>();
}
TEST(SimdComplexParity, SquareRootRangeBoundaries)
{
    check_sqrt_bounds<float>();
    check_sqrt_bounds<double>();
}
TEST(SimdComplexParity, MagnitudeSpecialValuesAndAccumulation)
{
    check_magnitude_and_edges<float>();
    check_magnitude_and_edges<double>();
}
TEST(SimdComplexParity, MixedMathEdgesAndBranchCuts)
{
    check_mixed_math_edges<float>();
    check_mixed_math_edges<double>();
}

namespace
{
template <class T> void check_vector_math_ranges()
{
    using c = scalars::complex<T>;
    using s = simd<c>;
    // NOLINTNEXTLINE(bugprone-random-generator-seed): reproducible range coverage.
    std::mt19937                       random(12345);
    const T                            limit = std::is_same_v<T, float> ? T(85) : T(710);
    std::uniform_real_distribution<T>  exponent(-limit, limit), angle(T(-512), T(512));
    std::uniform_real_distribution<T>  mantissa(T(-1), T(1));
    std::uniform_int_distribution<int> power(
        std::numeric_limits<T>::min_exponent - 24, std::numeric_limits<T>::max_exponent - 2);
    std::array<c, s::size> input, output;
    std::array<T, s::size> magnitude;
    for (int trial = 0; trial < 512; ++trial)
    {
        for (auto& value : input)
        {
            value = c(exponent(random), angle(random));
        }
        s::storeu(s::exp(s::loadu(input.data())), output.data());
        for (int i = 0; i < s::size; ++i)
        {
            expect_value(output[i], std::exp(std::complex<T>(input[i].real(), input[i].imag())));
        }
        for (auto& value : input)
        {
            value = c(std::ldexp(mantissa(random), power(random)),
                std::ldexp(mantissa(random), power(random)));
        }
        if (trial % 4 == 0)
        {
            // Near-unit magnitudes expose cancellation in logarithm's real part.
            input[0] =
                c(std::nextafter(T(1), T(2)), std::ldexp(T(1), -std::numeric_limits<T>::digits));
        }
        s::storeu(s::abs(s::loadu(input.data())), magnitude.data());
        s::storeu(s::log(s::loadu(input.data())), output.data());
        for (int i = 0; i < s::size; ++i)
        {
            const auto reference = std::complex<T>(input[i].real(), input[i].imag());
            const T    expected  = std::abs(reference);
            EXPECT_NEAR(magnitude[i],
                expected,
                T(8) * std::numeric_limits<T>::epsilon() * expected +
                    T(2) * std::numeric_limits<T>::denorm_min());
            expect_value(output[i], std::log(reference));
        }
    }
}
}  // namespace

TEST(SimdComplexParity, MathExponentRanges)
{
    check_vector_math_ranges<float>();
    check_vector_math_ranges<double>();
}
namespace
{
// Reduced-precision complex packets: set_mask must accept plain integer
// literals as well as bool (7.8). The mask is observed through if_else.
template <class T> void check_reduced_set_mask()
{
    using c = scalars::complex<T>;
    using s = simd<c>;
    std::array<c, s::size> a{}, b{}, out{};
    for (int i = 0; i < s::size; ++i)
    {
        a[i] = c(T(float(i + 1)), T(float(-i)));
        b[i] = c(T(-2.f), T(3.f));
    }
    const auto pa = s::loadu(a.data()), pb = s::loadu(b.data());
    auto       expect_selects = [&](typename s::mask_t m, const std::array<c, s::size>& want)
    {
        s::storeu(s::if_else(m, pa, pb), out.data());
        for (int i = 0; i < s::size; ++i)
        {
            EXPECT_EQ(float(out[i].real()), float(want[i].real())) << i;
            EXPECT_EQ(float(out[i].imag()), float(want[i].imag())) << i;
        }
    };
    expect_selects(s::set_mask(1), a);
    expect_selects(s::set_mask(0), b);
    expect_selects(s::set_mask(true), a);
    expect_selects(s::set_mask(false), b);
    expect_selects(s::set_mask(std::int64_t(-3)), a);
    expect_selects(s::set_mask(typename s::int_t(0)), b);
}
}  // namespace

TEST(SimdComplexParity, ReducedSetMaskAcceptsIntegerLiterals)
{
    check_reduced_set_mask<scalars::half>();
    check_reduced_set_mask<scalars::bfloat16>();
}

#if SCALARS_HAS_SVML || SCALARS_HAS_MKL || SCALARS_HAS_SLEEF || SCALARS_HAS_ACCELERATE
namespace
{
// Reference log|z| for z near the unit circle, computed without cancellation:
// |z|^2 - 1 = (hi^2 - 1) + lo^2 with each square split exactly by fma. For
// float inputs the squares are exact in double, so the double result is far
// more accurate than float needs.
double near_unit_log_magnitude_reference(double re, double im)
{
    const double hi  = std::max(std::fabs(re), std::fabs(im));
    const double lo  = std::min(std::fabs(re), std::fabs(im));
    const double hi2 = hi * hi, hi2_error = std::fma(hi, hi, -hi2);
    const double lo2 = lo * lo, lo2_error = std::fma(lo, lo, -lo2);
    // Two-sum (not Fast2Sum) so the reference does not share the
    // implementation's ordering assumption.
    const double sum = hi2 + lo2, rounded_hi2 = sum - lo2;
    const double sum_error = (hi2 - rounded_hi2) + (lo2 - (sum - rounded_hi2));
    return 0.5 * std::log1p((sum - 1.0) + (sum_error + hi2_error + lo2_error));
}

// Re(log z) must be accurate relative to itself, not just to |log z|: near
// |z| = 1 it is tiny while Im(log z) = arg z is not, so the normwise check in
// expect_value cannot see errors in it (plan items 4.1 and 6.2).
template <class T> void check_log_real_part_near_unit_circle()
{
    using c = scalars::complex<T>;
    using s = simd<c>;
    std::vector<c> inputs;
    // The plan's 4.1 example: |z|^2 - 1 is about 4e-8.
    inputs.emplace_back(T(0.9323273301), T(0.3616154194));
    const int deepest = std::is_same_v<T, float> ? 22 : 50;
    for (int k = 3; k <= deepest; ++k)
    {
        for (const double sign : {-1.0, 1.0})
        {
            const double radius = 1.0 + sign * std::ldexp(1.0, -k);
            for (const double angle : {0.05, 0.4, 0.785, 1.2, 2.5, -2.0})
            {
                inputs.emplace_back(T(radius * std::cos(angle)), T(radius * std::sin(angle)));
            }
        }
    }
    while (inputs.size() % s::size != 0)
    {
        inputs.push_back(inputs.front());
    }
    std::vector<c> outputs(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); i += s::size)
    {
        s::storeu(s::log(s::loadu(inputs.data() + i)), outputs.data() + i);
    }
    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        const double expected =
            near_unit_log_magnitude_reference(double(inputs[i].real()), double(inputs[i].imag()));
        const double tolerance =
            16 * double(std::numeric_limits<T>::epsilon()) * std::fabs(expected) +
            double(std::numeric_limits<T>::min());
        EXPECT_NEAR(double(outputs[i].real()), expected, tolerance)
            << "z = (" << double(inputs[i].real()) << ", " << double(inputs[i].imag()) << ")";
    }
}
}  // namespace

TEST(SimdComplexParity, LogRealPartRelativeAccuracyNearUnitCircle)
{
    check_log_real_part_near_unit_circle<float>();
    check_log_real_part_near_unit_circle<double>();
}
#endif

#endif
