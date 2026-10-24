#include "backend/simd.h"

#if SCALARS_VECTORIZED

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "scalars_test.h"

namespace
{
template <class T> class SimdReduced : public ::testing::Test
{
};
using ReducedTypes = ::testing::Types<scalars::half, scalars::bfloat16>;
TYPED_TEST_SUITE(SimdReduced, ReducedTypes);

template <class T> T from_bits(uint16_t bits)
{
    return T(bits, T::from_bits());
}

template <class T> void expect_value(T actual, T expected, float ulps = 0)
{
    const float a = static_cast<float>(actual);
    const float e = static_cast<float>(expected);
    if (std::isnan(e))
        EXPECT_TRUE(std::isnan(a));
    else if (std::isinf(e) || ulps == 0 || e == 0)
        EXPECT_EQ(actual.x, expected.x);
    else
    {
        const float eps  = std::is_same_v<T, scalars::half> ? 0x1p-10f : 0x1p-7f;
        const float tiny = std::is_same_v<T, scalars::half> ? 0x1p-24f : 0x1p-133f;
        EXPECT_LE(std::fabs(a - e), ulps * std::max(eps * std::fabs(e), tiny));
    }
}

TYPED_TEST(SimdReduced, DenseDistinctPackets)
{
    using S = simd<TypeParam>;
    static_assert(S::size == 2 * simd<float>::size);
    static_assert(S::half_size == simd<float>::size);
    static_assert(sizeof(typename S::simd_t) == S::size * sizeof(TypeParam));
    static_assert(sizeof(typename S::simd_half_t) == S::half_size * sizeof(TypeParam));
    static_assert(!std::is_same_v<typename S::simd_t, typename S::mask_t>);
    static_assert(!std::is_same_v<simd<scalars::half>::simd_t, simd<scalars::bfloat16>::simd_t>);
    static_assert(std::is_same_v<decltype(S::accumulate(S::setzero())), float>);
}

TYPED_TEST(SimdReduced, ExhaustiveEncodingWideningAndMemory)
{
    using T = TypeParam;
    using S = simd<T>;
    std::array<T, S::size>     input, output;
    std::array<float, S::size> widened;
    for (uint32_t first = 0; first < 65536; first += S::size)
    {
        for (int i = 0; i < S::size; ++i)
            input[i] = from_bits<T>(static_cast<uint16_t>(first + i));
        const auto packet = S::loadu(input.data());
        S::storeu(packet, output.data());
        const auto f = S::to_float_pair(packet);
        simd<float>::storeu(f.low, widened.data());
        simd<float>::storeu(f.high, widened.data() + S::half_size);
        for (int i = 0; i < S::size; ++i)
        {
            ASSERT_EQ(output[i].x, input[i].x);
            const float expected = static_cast<float>(input[i]);
            if (std::isnan(expected))
                ASSERT_TRUE(std::isnan(widened[i]));
            else
            {
                ASSERT_EQ(widened[i], expected) << first + i;
                ASSERT_EQ(std::signbit(widened[i]), std::signbit(expected));
            }
        }
    }
}

TYPED_TEST(SimdReduced, NarrowingMidpointsAndExceptionalValues)
{
    using T = TypeParam;
    using S = simd<T>;
    std::vector<float> cases;
    const uint32_t     max_finite = std::is_same_v<T, scalars::half> ? 0x7bff : 0x7f7f;
    for (uint32_t bits = 0; bits < max_finite; ++bits)
    {
        const float a        = static_cast<float>(from_bits<T>(static_cast<uint16_t>(bits)));
        const float b        = static_cast<float>(from_bits<T>(static_cast<uint16_t>(bits + 1)));
        const float midpoint = static_cast<float>((double(a) + double(b)) * 0.5);
        for (float value : {std::nextafter(midpoint, 0.f),
                 midpoint,
                 std::nextafter(midpoint, std::numeric_limits<float>::infinity())})
        {
            cases.push_back(value);
            cases.push_back(-value);
        }
    }
    for (float value : {0.f,
             -0.f,
             65504.f,
             65520.f,
             65536.f,
             std::numeric_limits<float>::max(),
             std::numeric_limits<float>::min(),
             std::numeric_limits<float>::denorm_min(),
             std::numeric_limits<float>::infinity(),
             std::numeric_limits<float>::quiet_NaN(),
             std::numeric_limits<float>::signaling_NaN()})
    {
        cases.push_back(value);
        cases.push_back(-value);
    }
    std::mt19937 rng(41267);
    for (int i = 0; i < 65536; ++i)
        cases.push_back(scalars::bit_cast<float>(static_cast<uint32_t>(rng())));
    cases.resize((cases.size() + S::size - 1) / S::size * S::size, 0.f);
    std::array<T, S::size> output;
    for (std::size_t offset = 0; offset < cases.size(); offset += S::size)
    {
        const auto low  = simd<float>::loadu(cases.data() + offset);
        const auto high = simd<float>::loadu(cases.data() + offset + S::half_size);
        S::storeu(S::from_float_pair(low, high), output.data());
        for (int i = 0; i < S::size; ++i)
        {
            const T expected(cases[offset + i]);
            if (std::isnan(cases[offset + i]))
            {
                ASSERT_TRUE(std::isnan(static_cast<float>(output[i])));
                if constexpr (std::is_same_v<T, scalars::bfloat16>)
                    ASSERT_EQ(output[i].x, 0x7fc0);
            }
            else
                ASSERT_EQ(output[i].x, expected.x) << "input=" << cases[offset + i];
        }
    }
}

TYPED_TEST(SimdReduced, ArithmeticAndFloatAccumulation)
{
    using T = TypeParam;
    using S = simd<T>;
    std::mt19937                          rng(87291);
    std::uniform_real_distribution<float> dist(-5, 5);
    std::array<T, S::size>                x, y, z, out;
    for (int trial = 0; trial < 128; ++trial)
    {
        for (int i = 0; i < S::size; ++i)
        {
            x[i] = T(dist(rng));
            y[i] = T(dist(rng) + 6.f);
            z[i] = T(dist(rng));
        }
        const auto a = S::loadu(x.data()), b = S::loadu(y.data()), c = S::loadu(z.data());
#define CHECK_BINARY(name, expression)                                                             \
    S::storeu(S::name(a, b), out.data());                                                          \
    for (int i = 0; i < S::size; ++i)                                                              \
    {                                                                                              \
        const float u = static_cast<float>(x[i]), v = static_cast<float>(y[i]);                    \
        expect_value(out[i], T(expression));                                                       \
    }
        CHECK_BINARY(add, u + v)
        CHECK_BINARY(sub, u - v)
        CHECK_BINARY(mul, u * v)
        CHECK_BINARY(div, u / v)
#undef CHECK_BINARY
        S::storeu(S::fma(a, b, c), out.data());
        for (int i = 0; i < S::size; ++i)
        {
            const float u = static_cast<float>(x[i]), v = static_cast<float>(y[i]),
                        w = static_cast<float>(z[i]);
#if defined(__FMA__) || SCALARS_HAS_NEON || SCALARS_HAS_SVE || SCALARS_HAS_AVX512
            const float expected = std::fma(u, v, w);
#else
            volatile float product  = u * v;
            const float    expected = product + w;
#endif
            expect_value(out[i], T(expected));
        }
    }
    // A reduced-precision sum overflows half here, while float stays finite.
    // For bf16, 256 + 1 cannot be represented as a stored bf16 result.
    x.fill(T(std::is_same_v<T, scalars::half> ? 10000.f : 1.f));
    if constexpr (std::is_same_v<T, scalars::bfloat16>)
        x[0] = T(256.f);
    const float expected =
        std::is_same_v<T, scalars::half> ? 10000.f * S::size : 256.f + S::size - 1;
    EXPECT_EQ(S::accumulate(S::loadu(x.data())), expected);
    for (int i = 0; i < S::size; ++i)
        x[i] = T(float(i - S::half_size));
    expect_value(S::hmin(S::loadu(x.data())), x.front());
    expect_value(S::hmax(S::loadu(x.data())), x.back());
}

TYPED_TEST(SimdReduced, MasksComparisonsAndBitPreservingSelect)
{
    using T = TypeParam;
    using S = simd<T>;
    std::array<T, S::size>                 x, y, out;
    std::array<typename S::int_t, S::size> mask;
    const T nan = from_bits<T>(std::is_same_v<T, scalars::half> ? 0x7e35 : 0x7fc5);
    for (int i = 0; i < S::size; ++i)
    {
        x[i] = i % 4 == 0 ? nan : T(float(i % 4 - 1));
        y[i] = T(float(2 - i % 4));
    }
    const auto a = S::loadu(x.data()), b = S::loadu(y.data());
#define CHECK_COMPARE(name, op)                                                                    \
    S::storeu(S::name(a, b), mask.data());                                                         \
    for (int i = 0; i < S::size; ++i)                                                              \
        EXPECT_EQ(mask[i], (static_cast<float>(x[i]) op static_cast<float>(y[i])) ? 0xffff : 0);
    CHECK_COMPARE(eq, ==)
    CHECK_COMPARE(neq, !=)
    CHECK_COMPARE(gt, >)
    CHECK_COMPARE(ge, >=)
    CHECK_COMPARE(lt, <)
    CHECK_COMPARE(le, <=)
#undef CHECK_COMPARE
    const auto m = S::neq(a, b);
    S::storeu(S::if_else(m, a, b), out.data());
    for (int i = 0; i < S::size; ++i)
        EXPECT_EQ(out[i].x, (static_cast<float>(x[i]) != static_cast<float>(y[i]) ? x[i] : y[i]).x);
    S::storeu(S::not_mask(S::not_mask(m)), mask.data());
    const auto loaded = S::load(mask.data());
    S::store(S::xor_mask(loaded, m), mask.data());
    for (auto bits : mask)
        EXPECT_EQ(bits, 0);
    S::storeu(S::and_mask(m, S::not_mask(m)), mask.data());
    for (auto bits : mask)
        EXPECT_EQ(bits, 0);
    S::storeu(S::or_mask(m, S::not_mask(m)), mask.data());
    for (auto bits : mask)
        EXPECT_EQ(bits, 0xffff);
    S::storeu(S::set(typename S::int_t(1)), mask.data());
    for (auto bits : mask)
        EXPECT_EQ(bits, 0xffff);
}

TYPED_TEST(SimdReduced, SignedZerosNaNsAndMinMax)
{
    using T = TypeParam;
    using S = simd<T>;
    std::array<T, S::size> out;
    const auto             plus = S::set(T(0.f)), minus = S::set(from_bits<T>(0x8000));
    S::storeu(S::min(plus, minus), out.data());
    for (auto v : out)
        EXPECT_EQ(v.x, 0x8000);
    S::storeu(S::max(minus, plus), out.data());
    for (auto v : out)
        EXPECT_EQ(v.x, 0);
    EXPECT_EQ(S::hmin(minus).x, 0x8000);
    EXPECT_EQ(S::hmax(minus).x, 0x8000);
    const T nan = from_bits<T>(std::is_same_v<T, scalars::half> ? 0xfe35 : 0xffc5);
    S::storeu(S::neg(S::set(nan)), out.data());
    for (auto v : out)
        EXPECT_EQ(v.x, (nan.x ^ 0x8000));
    S::storeu(S::fabs(S::set(nan)), out.data());
    for (auto v : out)
        EXPECT_EQ(v.x, (nan.x & 0x7fff));
    S::storeu(S::signcopy(S::set(nan), plus), out.data());
    for (auto v : out)
        EXPECT_EQ(v.x, (nan.x & 0x7fff));
    for (bool left : {false, true})
    {
        auto a = left ? S::set(nan) : S::set(1.f);
        auto b = left ? S::set(1.f) : S::set(nan);
        S::storeu(S::min(a, b), out.data());
        for (auto v : out)
            EXPECT_TRUE(std::isnan(static_cast<float>(v)));
        S::storeu(S::max(a, b), out.data());
        for (auto v : out)
            EXPECT_TRUE(std::isnan(static_cast<float>(v)));
    }
    out.fill(T(1.f));
    out[S::size / 2] = nan;
    EXPECT_TRUE(std::isnan(static_cast<float>(S::hmin(S::loadu(out.data())))));
    EXPECT_TRUE(std::isnan(static_cast<float>(S::hmax(S::loadu(out.data())))));
    S::storeu(S::invsqrt(plus), out.data());
    for (auto v : out)
        EXPECT_EQ(static_cast<float>(v), std::numeric_limits<float>::infinity());
    S::storeu(S::invsqrt(S::set(std::numeric_limits<float>::infinity())), out.data());
    for (auto v : out)
        EXPECT_EQ(v.x, 0);
}

TYPED_TEST(SimdReduced, MemoryTailsBroadcastAndGatherScatter)
{
    using T = TypeParam;
    using S = simd<T>;
    alignas(64) std::array<T, S::size + 2> input, out;
    for (int i = 0; i < S::size + 2; ++i)
        input[i] = from_bits<T>(static_cast<uint16_t>(i * 719));
    S::prefetch(input.data());
    S::copy(input.data() + 1, out.data());
    for (int i = 0; i < S::size; ++i)
        EXPECT_EQ(out[i].x, input[i + 1].x);
    S::store(S::load(input.data()), out.data());
    for (int i = 0; i < S::size; ++i)
        EXPECT_EQ(out[i].x, input[i].x);
    S::storeu(S::setzero(), nullptr, 0);
    S::storeu(S::loadu(nullptr, 0), out.data());
    for (int i = 0; i < S::size; ++i)
        EXPECT_EQ(out[i].x, 0);
    for (int count = 0; count <= S::size; ++count)
    {
        // An exact allocation makes an inactive-lane overread visible to ASan.
        std::vector<T> exact(input.begin(), input.begin() + count);
        const auto     packet = S::loadu(exact.data(), count);
        out.fill(from_bits<T>(0x4567));
        S::storeu(packet, out.data() + 1, count);
        EXPECT_EQ(out[0].x, 0x4567);
        for (int i = 0; i < count; ++i)
            EXPECT_EQ(out[i + 1].x, input[i].x);
        for (int i = count + 1; i < S::size + 2; ++i)
            EXPECT_EQ(out[i].x, 0x4567);
        S::storeu(packet, out.data());
        for (int i = count; i < S::size; ++i)
            EXPECT_EQ(out[i].x, 0);
    }
    std::array<typename S::simd_t, 4> broadcast;
    S::broadcast(input.data(), broadcast[0], broadcast[1], broadcast[2], broadcast[3]);
    for (int k = 0; k < 4; ++k)
    {
        S::storeu(broadcast[k], out.data());
        for (int i = 0; i < S::size; ++i)
            EXPECT_EQ(out[i].x, input[k].x);
    }
    S::storeu(S::ploadquad(input.data()), out.data());
    for (int i = 0; i < S::size; ++i)
        EXPECT_EQ(out[i].x, input[i / 4].x);
    std::array<int, S::size> indices;
    for (int i = 0; i < S::size; ++i)
        indices[i] = S::size - 1 - i;
    S::storeu(S::gather(input.data() + S::size - 1, -1), out.data());
    for (int i = 0; i < S::size; ++i)
        EXPECT_EQ(out[i].x, input[S::size - 1 - i].x);
    const auto gathered = S::gather(input.data(), indices.data());
    S::scatter(gathered, indices.data(), out.data());
    for (int i = 0; i < S::size; ++i)
        EXPECT_EQ(out[i].x, input[i].x);
    std::array<T, 2 * S::size> strided;
    S::scatter(gathered, 2, strided.data());
    S::storeu(S::gather(strided.data(), 2), out.data());
    for (int i = 0; i < S::size; ++i)
        EXPECT_EQ(out[i].x, input[indices[i]].x);
    indices.fill(0);
    S::scatter(gathered, indices.data(), out.data());
    EXPECT_EQ(out[0].x, input[0].x);  // last logical lane wins
}

template <class T, int Rows> void check_transpose()
{
    using S = simd<T>;
    std::array<T, Rows * S::size> values, output;
    typename S::simd_t            packets[Rows];
    for (int i = 0; i < Rows * S::size; ++i)
        values[i] = from_bits<T>(static_cast<uint16_t>(i * 191));
    for (int row = 0; row < Rows; ++row)
        packets[row] = S::loadu(values.data() + row * S::size);
    S::template ptranspose<Rows>(packets);
    for (int row = 0; row < Rows; ++row)
        S::storeu(packets[row], output.data() + row * S::size);
    for (int row = 0; row < Rows; ++row)
        for (int col = 0; col < S::size; ++col)
            EXPECT_EQ(output[col * Rows + row].x, values[row * S::size + col].x);
}

TYPED_TEST(SimdReduced, TransposeAndHalfWidthUtilities)
{
    using T = TypeParam;
    using S = simd<T>;
    check_transpose<T, S::size>();
    check_transpose<T, 4>();
    check_transpose<T, 2 * S::size>();
    std::array<T, S::size> x, out;
    for (int i = 0; i < S::size; ++i)
        x[i] = T(float(i));
    const auto a = S::loadu_half(x.data());
    const auto b = S::gather_half(x.data(), 2);
    S::scatter(S::add(a, b), 1, out.data());
    for (int i = 0; i < S::half_size; ++i)
        expect_value(out[i], T(float(i * 3)));
    const auto c = S::set(x.data() + 1);
    S::scatter(S::fma(a, b, c), 1, out.data());
    for (int i = 0; i < S::half_size; ++i)
        expect_value(out[i], T(float(i * i * 2 + 1)));
    S::scatter(S::predux_downto4(S::loadu(x.data())), 1, out.data());
    for (int i = 0; i < S::half_size; ++i)
        expect_value(out[i], T(float(i * 2 + S::half_size)));
    const auto f = S::load_float_pair(x.data());
    S::store_float_pair(f.low, f.high, out.data());
    for (int i = 0; i < S::size; ++i)
        EXPECT_EQ(out[i].x, x[i].x);
}

template <class T, class Op, class Ref>
void check_math(const char* name, Op op, Ref ref, float lo, float hi)
{
    SCOPED_TRACE(name);
    using S = simd<T>;
    std::array<T, S::size> input, output;
    for (int trial = 0; trial < 16; ++trial)
    {
        for (int i = 0; i < S::size; ++i)
            input[i] = T(lo + (hi - lo) * float(trial * S::size + i) / float(16 * S::size - 1));
        S::storeu(op(S::loadu(input.data())), output.data());
        for (int i = 0; i < S::size; ++i)
            expect_value(output[i], T(ref(static_cast<float>(input[i]))), 2.f);
    }
}

TYPED_TEST(SimdReduced, CompleteRealMathSurface)
{
    using T = TypeParam;
    using S = simd<T>;
#define CHECK_MATH(name, expression, lo, hi)                                                       \
    check_math<T>(                                                                                 \
        #name, [](auto v) { return S::name(v); }, [](float x) { return expression; }, lo, hi);
    CHECK_MATH(sqrt, std::sqrt(x), 0.f, 8.f)
    CHECK_MATH(invsqrt, 1.f / std::sqrt(x), 0.125f, 8.f)
    CHECK_MATH(sqr, x * x, -4.f, 4.f)
    CHECK_MATH(ceil, std::ceil(x), -4.f, 4.f)
    CHECK_MATH(floor, std::floor(x), -4.f, 4.f)
    CHECK_MATH(trunc, std::trunc(x), -4.f, 4.f)
    CHECK_MATH(exp, std::exp(x), -3.f, 3.f)
    CHECK_MATH(expm1, std::expm1(x), -0.25f, 0.25f)
    CHECK_MATH(exp2, std::exp2(x), -3.f, 3.f)
    CHECK_MATH(exp10, std::pow(10.f, x), -2.f, 2.f)
    CHECK_MATH(log, std::log(x), 0.125f, 8.f)
    CHECK_MATH(log1p, std::log1p(x), -0.25f, 0.25f)
    CHECK_MATH(log2, std::log2(x), 0.125f, 8.f)
    CHECK_MATH(log10, std::log10(x), 0.125f, 8.f)
    CHECK_MATH(sin, std::sin(x), -1.f, 1.f)
    CHECK_MATH(cos, std::cos(x), -1.f, 1.f)
    CHECK_MATH(tan, std::tan(x), -1.f, 1.f)
    CHECK_MATH(asin, std::asin(x), -1.f, 1.f)
    CHECK_MATH(acos, std::acos(x), -1.f, 1.f)
    CHECK_MATH(atan, std::atan(x), -1.f, 1.f)
    CHECK_MATH(sinh, std::sinh(x), -1.f, 1.f)
    CHECK_MATH(cosh, std::cosh(x), -1.f, 1.f)
    CHECK_MATH(tanh, std::tanh(x), -1.f, 1.f)
    CHECK_MATH(asinh, std::asinh(x), -1.f, 1.f)
    CHECK_MATH(acosh, std::acosh(x), 1.f, 4.f)
    CHECK_MATH(atanh, std::atanh(x), -0.5f, 0.5f)
    CHECK_MATH(cbrt, std::cbrt(x), -8.f, 8.f)
    CHECK_MATH(cdf, float(scalars::normalcdf(double(x))), -3.f, 3.f)
    CHECK_MATH(inv_cdf, float(scalars::inv_normalcdf(double(x))), 0.125f, 0.875f)
#undef CHECK_MATH
    check_math<T>(
        "pow",
        [](auto a) { return S::pow(a, S::set(1.5f)); },
        [](float x) { return std::pow(x, 1.5f); },
        0.125f,
        4.f);
    check_math<T>(
        "hypot",
        [](auto a) { return S::hypot(a, S::set(2.f)); },
        [](float x) { return std::hypot(x, 2.f); },
        -4.f,
        4.f);
}
}  // namespace
#endif
