#include "scalars_macros.h"

#if SCALARS_HAS_METAL

#include "backend/gpu/metal/metal_dispatch.h"
#include "metal_storage.h"
#include "types/bfloat16.h"
#include "types/half.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

namespace
{
namespace mb = scalars::metal_backend;
namespace ms = scalars::metal_storage;

template <class T>
constexpr mb::dtype metal_type =
    std::is_same_v<T, scalars::half> ? mb::dtype::half : mb::dtype::bfloat16;

template <class T> struct metal_buffer
{
    T* data;
    explicit metal_buffer(std::size_t n) : data(static_cast<T*>(ms::allocate(n * sizeof(T)))) {}
    ~metal_buffer() { ms::deallocate(data); }
    metal_buffer(const metal_buffer&)            = delete;
    metal_buffer& operator=(const metal_buffer&) = delete;
};

template <class T> void expect_bits(T actual, T expected)
{
    if (std::isnan(float(expected)))
    {
        EXPECT_TRUE(std::isnan(float(actual)));
    }
    else
    {
        EXPECT_EQ(actual.x, expected.x);
    }
}

template <class T> void run_metal_reduced_FillArithmeticAndFusedRounding()
{
    using t                    = T;
    constexpr auto        type = metal_type<t>;
    constexpr std::size_t n    = 519;  // Partial final threadgroup.
    metal_buffer<t>       a(n), b(n), c(n), out(n);
    const float           eps = std::is_same_v<t, scalars::half> ? 0x1p-10f : 0x1p-7f;
    mb::dispatch_fill(a.data, 1 + eps, n, type);
    mb::dispatch_fill(b.data, 1 - eps, n, type);
    mb::dispatch_fill(c.data, -1, n, type);
    const void* inputs[] = {a.data, b.data, c.data};
    for (const char* op : {"add", "sub", "mul", "div", "min", "max", "fma"})
    {
        const std::string name(op);
        mb::dispatch(op, inputs, name == "fma" ? 3 : 2, out.data, n, type);
        const float x = float(a.data[0]), y = float(b.data[0]);
        const float expected = name == "add"   ? x + y
                               : name == "sub" ? x - y
                               : name == "mul" ? x * y
                               : name == "div" ? x / y
                               : name == "min" ? y
                               : name == "max" ? x
                                               : std::fma(x, y, -1.f);
        for (std::size_t i = 0; i < n; ++i)
        {
            expect_bits(out.data[i], t(expected));
        }
    }
    // Multiplication rounded to T before the addition would produce zero.
    EXPECT_LT(float(out.data[0]), 0.f);
    mb::dispatch_fill(out.data, 1.00390625f, n, type);
    expect_bits(out.data[n - 1], t(1.00390625f));
    for (float value : {0.f,
             -0.f,
             1.f + eps / 2,
             1.f + 1.5f * eps,
             1.f + 2.5f * eps,
             INFINITY,
             -INFINITY,
             std::numeric_limits<float>::quiet_NaN(),
             0x1p-24f,
             0x1p-133f})
    {
        mb::dispatch_fill(out.data, value, n, type);
        expect_bits(out.data[0], t(value));
        expect_bits(out.data[n - 1], t(value));
    }
}

TEST(metal_reduced, FillArithmeticAndFusedRounding_half)
{
    if (!mb::device_available())
    {
        GTEST_SKIP() << "No Metal device";
    }
    run_metal_reduced_FillArithmeticAndFusedRounding<scalars::half>();
}
TEST(metal_reduced, FillArithmeticAndFusedRounding_bfloat16)
{
    if (!mb::device_available())
    {
        GTEST_SKIP() << "No Metal device";
    }
    run_metal_reduced_FillArithmeticAndFusedRounding<scalars::bfloat16>();
}

template <class T> void run_metal_reduced_UnaryMath()
{
    using t                    = T;
    constexpr auto        type = metal_type<t>;
    constexpr std::size_t n    = 513;
    metal_buffer<t>       input(n), out(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        input.data[i] = t(0.25f + float(i) / float(n));
    }
    const void* inputs[] = {input.data};
    const float eps      = std::is_same_v<t, scalars::half> ? 0x1p-10f : 0x1p-7f;
    for (const char* op : {"sqrt", "exp", "log", "sin", "cos", "tanh", "fabs", "neg"})
    {
        mb::dispatch(op, inputs, 1, out.data, n, type);
        const std::string name(op);
        for (std::size_t i = 0; i < n; ++i)
        {
            const float x        = float(input.data[i]);
            const float expected = name == "sqrt"   ? std::sqrt(x)
                                   : name == "exp"  ? std::exp(x)
                                   : name == "log"  ? std::log(x)
                                   : name == "sin"  ? std::sin(x)
                                   : name == "cos"  ? std::cos(x)
                                   : name == "tanh" ? std::tanh(x)
                                   : name == "fabs" ? std::fabs(x)
                                                    : -x;
            EXPECT_NEAR(float(out.data[i]), expected, eps * std::max(1.f, std::fabs(expected)));
        }
    }
}

TEST(metal_reduced, UnaryMath_half)
{
    if (!mb::device_available())
    {
        GTEST_SKIP() << "No Metal device";
    }
    run_metal_reduced_UnaryMath<scalars::half>();
}
TEST(metal_reduced, UnaryMath_bfloat16)
{
    if (!mb::device_available())
    {
        GTEST_SKIP() << "No Metal device";
    }
    run_metal_reduced_UnaryMath<scalars::bfloat16>();
}

template <class T> void run_metal_reduced_EveryEncodingPreservesSignAndPayload()
{
    using t                    = T;
    constexpr auto        type = metal_type<t>;
    constexpr std::size_t n    = 65536;
    metal_buffer<t>       input(n), out(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        input.data[i] = t(uint16_t(i), t::from_bits());
    }
    const void* inputs[] = {input.data};
    for (const char* op : {"neg", "fabs"})
    {
        mb::dispatch(op, inputs, 1, out.data, n, type);
        for (std::size_t i = 0; i < n; ++i)
        {
            const uint16_t expected = op[0] == 'n' ? uint16_t(i ^ 0x8000) : uint16_t(i & 0x7fff);
            ASSERT_EQ(out.data[i].x, expected) << op << " input bits=" << i;
        }
    }
}

TEST(metal_reduced, EveryEncodingPreservesSignAndPayload_half)
{
    if (!mb::device_available())
    {
        GTEST_SKIP() << "No Metal device";
    }
    run_metal_reduced_EveryEncodingPreservesSignAndPayload<scalars::half>();
}
TEST(metal_reduced, EveryEncodingPreservesSignAndPayload_bfloat16)
{
    if (!mb::device_available())
    {
        GTEST_SKIP() << "No Metal device";
    }
    run_metal_reduced_EveryEncodingPreservesSignAndPayload<scalars::bfloat16>();
}

template <class T> void run_metal_reduced_FloatReductionPartialsAndExceptionalValues()
{
    using t                    = T;
    constexpr auto        type = metal_type<t>;
    constexpr std::size_t n    = 65537;  // Three passes with 256-thread groups.
    metal_buffer<t>       input(n);
    mb::dispatch_fill(input.data, 1, n, type);
    EXPECT_EQ(mb::reduce_sum(input.data, n, type), float(n));
    EXPECT_EQ(mb::reduce_min(input.data, n, type), 1.f);
    EXPECT_EQ(mb::reduce_max(input.data, n, type), 1.f);
    input.data[n - 1] = t(std::numeric_limits<float>::quiet_NaN());
    EXPECT_TRUE(std::isnan(mb::reduce_min(input.data, n, type)));
    EXPECT_TRUE(std::isnan(mb::reduce_max(input.data, n, type)));
    EXPECT_TRUE(std::isnan(mb::reduce_sum(input.data, n, type)));

    mb::dispatch_fill(input.data, 0, n, type);
    input.data[n - 1] = t(uint16_t(0x8000), t::from_bits());
    EXPECT_TRUE(std::signbit(mb::reduce_min(input.data, n, type)));
    EXPECT_FALSE(std::signbit(mb::reduce_max(input.data, n, type)));
    mb::dispatch_fill(input.data, INFINITY, 1, type);
    EXPECT_EQ(mb::reduce_min(input.data, 1, type), INFINITY);
    mb::dispatch_fill(input.data, -INFINITY, 1, type);
    EXPECT_EQ(mb::reduce_max(input.data, 1, type), -INFINITY);
    EXPECT_EQ(mb::reduce_sum(nullptr, 0, type), 0.f);
    EXPECT_EQ(mb::reduce_min(nullptr, 0, type), INFINITY);
    EXPECT_EQ(mb::reduce_max(nullptr, 0, type), -INFINITY);
    mb::dispatch_fill(nullptr, 1, 0, type);
    mb::dispatch("add", nullptr, 2, nullptr, 0, type);
}

TEST(metal_reduced, FloatReductionPartialsAndExceptionalValues_half)
{
    if (!mb::device_available())
    {
        GTEST_SKIP() << "No Metal device";
    }
    run_metal_reduced_FloatReductionPartialsAndExceptionalValues<scalars::half>();
}
TEST(metal_reduced, FloatReductionPartialsAndExceptionalValues_bfloat16)
{
    if (!mb::device_available())
    {
        GTEST_SKIP() << "No Metal device";
    }
    run_metal_reduced_FloatReductionPartialsAndExceptionalValues<scalars::bfloat16>();
}

template <class T> void run_metal_reduced_TypedFusedEntryPoints()
{
    using t              = T;
    constexpr auto  type = metal_type<t>;
    metal_buffer<t> out(1);
    // One source with both entry points also exercises dtype-aware caching.
    const std::string source = R"(
        #include <metal_stdlib>
        using namespace metal;
        kernel void fused_half(device half* out [[buffer(0)]],
                               constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]])
        { if (i < n) out[i] = half(1.5f); }
        kernel void fused_bfloat16(device ushort* out [[buffer(0)]],
                                   constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]])
        { if (i < n) out[i] = ushort(0x4020); }
    )";
    mb::dispatch_fused(source, nullptr, 0, nullptr, 0, out.data, 1, type);
    expect_bits(out.data[0], t(std::is_same_v<t, scalars::half> ? 1.5f : 2.5f));
}

TEST(metal_reduced, TypedFusedEntryPoints_half)
{
    if (!mb::device_available())
    {
        GTEST_SKIP() << "No Metal device";
    }
    run_metal_reduced_TypedFusedEntryPoints<scalars::half>();
}
TEST(metal_reduced, TypedFusedEntryPoints_bfloat16)
{
    if (!mb::device_available())
    {
        GTEST_SKIP() << "No Metal device";
    }
    run_metal_reduced_TypedFusedEntryPoints<scalars::bfloat16>();
}

}  // namespace
#endif
