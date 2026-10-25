#include "backend/simd.h"

#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <concepts>
#include <limits>
#include <numbers>
#include <type_traits>

#include "scalars_test.h"

#if SCALARS_VECTORIZED
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
    S::set(typename S::int_t(1));
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

template <class T> void expect_component(T actual, T expected, bool check_zero_sign = false)
{
    if (std::isnan(expected))
        EXPECT_TRUE(std::isnan(actual));
    else if (std::isinf(expected))
        EXPECT_EQ(actual, expected);
    else
    {
        const T tolerance = 64 * std::numeric_limits<T>::epsilon() * (T(1) + std::fabs(expected));
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
    expect_component(actual.real(), expected.real(), check_zero_sign);
    expect_component(actual.imag(), expected.imag(), check_zero_sign);
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
                y[i].imag(T(i + 1));
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
            EXPECT_EQ(flags[i], x[i] == y[i] ? ~bits(0) : bits(0));
        s::storeu(different, flags.data());
        for (int i = 0; i < s::size; ++i)
            EXPECT_EQ(flags[i], x[i] != y[i] ? ~bits(0) : bits(0));
        s::storeu(s::if_else(equal, a, b), out.data());
        for (int i = 0; i < s::size; ++i)
            expect_bits(out[i], x[i] == y[i] ? x[i] : y[i]);
        s::storeu(s::and_mask(equal, different), flags.data());
        for (const auto flag : flags)
            EXPECT_EQ(flag, bits(0));
        s::storeu(s::or_mask(equal, different), flags.data());
        for (const auto flag : flags)
            EXPECT_EQ(flag, ~bits(0));
        s::storeu(s::xor_mask(equal, different), flags.data());
        for (const auto flag : flags)
            EXPECT_EQ(flag, ~bits(0));
        s::storeu(s::not_mask(equal), flags.data());
        for (int i = 0; i < s::size; ++i)
            EXPECT_EQ(flags[i], x[i] != y[i] ? ~bits(0) : bits(0));
    }
    for (int i = 0; i < s::size; ++i)
        flags[i] = i % 2 ? bits(7) : bits(0);
    const auto memory_mask = s::loadu(flags.data());
    s::storeu(memory_mask, flags.data());
    for (int i = 0; i < s::size; ++i)
        EXPECT_EQ(flags[i], i % 2 ? ~bits(0) : bits(0));
    for (int count = 0; count <= s::size; ++count)
    {
        std::array<bits, s::size + 2> partial;
        partial.fill(bits(42));
        s::storeu(s::loadu(flags.data(), count), partial.data() + 1, count);
        EXPECT_EQ(partial.front(), bits(42));
        for (int i = 0; i < count; ++i)
            EXPECT_EQ(partial[i + 1], flags[i]);
        for (int i = count + 1; i < s::size + 2; ++i)
            EXPECT_EQ(partial[i], bits(42));
    }
    s::storeu(s::loadu(static_cast<const bits*>(nullptr), 0), static_cast<bits*>(nullptr), 0);
    s::storeu(s::set(bits(1)), flags.data());
    for (const auto flag : flags)
        EXPECT_EQ(flag, ~bits(0));
    s::storeu(s::set_mask(false), flags.data());
    for (const auto flag : flags)
        EXPECT_EQ(flag, bits(0));
    s::storeu(s::set(2), out.data());
    for (const auto value : out)
        EXPECT_EQ(value, c(T(2), T(0)));
}

template <class T, int N> void check_transpose()
{
    using c = scalars::complex<T>;
    using s = simd<c>;
    std::array<typename s::simd_t, N>                    rows;
    std::array<c, static_cast<std::size_t>(N) * s::size> input, output;
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = c(T(i + 1), -T(i + 3));
    for (int row = 0; row < N; ++row)
        rows[row] = s::loadu(input.data() + row * s::size);
    s::template ptranspose<N>(rows.data());
    for (int row = 0; row < N; ++row)
        s::storeu(rows[row], output.data() + row * s::size);
    for (std::size_t i = 0; i < input.size(); ++i)
        EXPECT_EQ(output[i], input[(i % N) * s::size + i / N]);
}

template <class T> void check_memory()
{
    using c                    = scalars::complex<T>;
    using s                    = simd<c>;
    constexpr int            w = s::size;
    const c                  sentinel(T(97), T(-63));
    std::array<c, 2 * w + 8> input, output, expected;
    for (int i = 0; i < 2 * w + 8; ++i)
        input[i] = c(T(i + 1) / 4, T(2 - i) / 8);
    s::prefetch(input.data());
    output.fill(sentinel);
    s::copy(input.data() + 1, output.data() + 1);
    EXPECT_EQ(output.front(), sentinel);
    for (int i = 1; i <= w; ++i)
        EXPECT_EQ(output[i], input[i]);
    EXPECT_EQ(output[w + 1], sentinel);
    typename s::simd_t a, b, c0, d;
    s::broadcast(input.data(), a, b, c0, d);
    const std::array<typename s::simd_t, 4> broadcasts{a, b, c0, d};
    for (int i = 0; i < 4; ++i)
    {
        s::storeu(broadcasts[i], output.data());
        for (int j = 0; j < w; ++j)
            EXPECT_EQ(output[j], input[i]);
    }
    constexpr int repeats = std::is_same_v<T, float> && w == 4 ? 2 : 4;
    s::storeu(s::ploadquad(input.data()), output.data());
    for (int i = 0; i < w; ++i)
        EXPECT_EQ(output[i], input[i / repeats]);
    std::array<int, w> indices;
    for (int i = 0; i < w; ++i)
        indices[i] = w - 1 - i - w / 2;
    a = s::gather(input.data() + w, indices.data());
    s::storeu(a, output.data());
    for (int i = 0; i < w; ++i)
        EXPECT_EQ(output[i], input[w + indices[i]]);
    for (int i = 0; i < w; ++i)
        indices[i] = i % 2;
    output.fill(sentinel);
    expected = output;
    for (int i = 0; i < w; ++i)
        expected[indices[i]] = input[i];
    s::scatter(s::loadu(input.data()), indices.data(), output.data());
    EXPECT_EQ(output, expected);

    const auto h = s::loadu_half(input.data());
    const auto k = s::gather_half(input.data() + w, -1);
    output.fill(sentinel);
    s::storeu(s::add(h, h), output.data());
    for (int i = 0; i < s::half_size; ++i)
        EXPECT_EQ(output[i], input[i] + input[i]);
    for (int i = s::half_size; i < 2 * w + 8; ++i)
        EXPECT_EQ(output[i], sentinel);
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
        EXPECT_EQ(output[i], input[0]);
    for (int i = s::half_size; i < w; ++i)
        EXPECT_EQ(output[i], c{});
    for (int i = 0; i < w; ++i)
        indices[i] = 2 * i;
    output.fill(sentinel);
    s::scatter(s::gather_half(input.data(), indices.data()), indices.data(), output.data());
    for (int i = 0; i < s::half_size; ++i)
        EXPECT_EQ(output[indices[i]], input[indices[i]]);
    if constexpr (w >= 4)
    {
        s::storeu(s::predux_downto4(s::loadu(input.data())), output.data());
        for (int i = 0; i < s::half_size; ++i)
            EXPECT_EQ(output[i], input[i] + input[i + s::half_size]);
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
            input[i] = values[(start + i) % values.size()];
        s::storeu(packet(s::loadu(input.data())), output.data());
        for (int i = 0; i < s::size; ++i)
            expect_value(
                output[i], reference(std::complex<T>(input[i].real(), input[i].imag())), zero_sign);
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
    check_unary<T>([](v a) { return s::acos(a); }, [](auto z) { return std::acos(z); });
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

template <class T> void check_magnitude_and_edges()
{
    using c          = scalars::complex<T>;
    using s          = simd<c>;
    using bits       = typename s::int_t;
    const T infinity = std::numeric_limits<T>::infinity(),
            nan      = std::numeric_limits<T>::quiet_NaN();
    const T maximum = std::numeric_limits<T>::max(), tiny = std::numeric_limits<T>::denorm_min();
    const std::array<c, 12>   edges{c(3, 4),
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
        c(T(-4), T(0))};
    std::array<c, s::size>    input, output;
    std::array<T, s::size>    real;
    std::array<bits, s::size> flags;
    for (const auto edge : edges)
    {
        input.fill(edge);
        const auto a = s::loadu(input.data());
        s::storeu(s::abs(a), real.data());
        for (const T value : real)
            expect_component(value, std::hypot(edge.real(), edge.imag()));
        s::storeu(s::fabs(a), real.data());
        for (const T value : real)
            expect_component(value, std::hypot(edge.real(), edge.imag()));
        s::storeu(s::arg(a), real.data());
        for (const T value : real)
            expect_component(value, std::atan2(edge.imag(), edge.real()), true);
        s::storeu(s::abs2(a), real.data());
        for (const T value : real)
            expect_component(value, edge.real() * edge.real() + edge.imag() * edge.imag());
        s::storeu(s::real(a), real.data());
        for (const T value : real)
            EXPECT_EQ(std::bit_cast<bits>(value), std::bit_cast<bits>(edge.real()));
        s::storeu(s::imag(a), real.data());
        for (const T value : real)
            EXPECT_EQ(std::bit_cast<bits>(value), std::bit_cast<bits>(edge.imag()));
        s::storeu(s::isfinite(a), flags.data());
        for (const auto flag : flags)
            EXPECT_EQ(flag, scalars::isfinite(edge) ? ~bits(0) : bits(0));
        s::storeu(s::isnan(a), flags.data());
        for (const auto flag : flags)
            EXPECT_EQ(
                flag, std::isnan(edge.real()) || std::isnan(edge.imag()) ? ~bits(0) : bits(0));
        s::storeu(s::isinf(a), flags.data());
        for (const auto flag : flags)
            EXPECT_EQ(
                flag, std::isinf(edge.real()) || std::isinf(edge.imag()) ? ~bits(0) : bits(0));
        s::storeu(s::sqrt(a), output.data());
        for (const auto value : output)
            expect_value(value, std::sqrt(std::complex<T>(edge.real(), edge.imag())), true);
        s::storeu(s::log(a), output.data());
        for (const auto value : output)
            expect_value(value, std::log(std::complex<T>(edge.real(), edge.imag())), true);
    }
    for (T x : {T(0), T(-0.0), T(1e-8), T(-1e-8), tiny, infinity, -infinity})
    {
        const auto a = s::set(c(x, T(-0.0)));
        s::storeu(s::expm1(a), output.data());
        for (const auto value : output)
            expect_value(value, std::complex<T>(std::expm1(x), T(-0.0)), true);
        s::storeu(s::log1p(a), output.data());
        if (x >= T(-1))
            for (const auto value : output)
                expect_value(value, std::complex<T>(std::log1p(x), T(-0.0)), true);
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
TEST(SimdComplexParity, MagnitudeSpecialValuesAndAccumulation)
{
    check_magnitude_and_edges<float>();
    check_magnitude_and_edges<double>();
}
#endif
