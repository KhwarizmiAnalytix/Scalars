#include "scalars_macros.h"

#if SCALARS_HAS_METAL

#include "backend/gpu/metal/metal_dispatch.h"
#include "metal_storage.h"
#include "types/complex.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace
{
namespace mb = scalars::metal_backend;
namespace ms = scalars::metal_storage;
using cfloat = scalars::complex<float>;

template <class T> struct metal_buffer
{
    T* data;
    explicit metal_buffer(std::size_t n) : data(static_cast<T*>(ms::allocate(n * sizeof(T)))) {}
    ~metal_buffer() { ms::deallocate(data); }
    metal_buffer(const metal_buffer&)            = delete;
    metal_buffer& operator=(const metal_buffer&) = delete;
};

class metal_complex : public ::testing::Test
{
    void SetUp() override
    {
        if (!mb::device_available())
        {
            GTEST_SKIP() << "No Metal device";
        }
    }
};

void expect_near(cfloat actual, std::complex<float> expected, float tol = 1e-4f)
{
    EXPECT_NEAR(actual.real(), expected.real(), tol);
    EXPECT_NEAR(actual.imag(), expected.imag(), tol);
}

}  // namespace

TEST_F(metal_complex, ArithmeticAndConj)
{
    constexpr std::size_t n = 517;  // Partial final threadgroup.
    metal_buffer<cfloat>  a(n), b(n), out(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        a.data[i] = cfloat(1.5f + float(i) * 0.01f, -0.5f - float(i) * 0.005f);
        b.data[i] = cfloat(2.0f, 1.0f);
    }
    const std::complex<float> ra(a.data[0].real(), a.data[0].imag());
    const std::complex<float> rb(b.data[0].real(), b.data[0].imag());
    const void*               inputs[] = {a.data, b.data};

    mb::dispatch("add", inputs, 2, out.data, n, mb::dtype::complex64);
    expect_near(out.data[0], ra + rb);

    mb::dispatch("sub", inputs, 2, out.data, n, mb::dtype::complex64);
    expect_near(out.data[0], ra - rb);

    mb::dispatch("mul", inputs, 2, out.data, n, mb::dtype::complex64);
    expect_near(out.data[0], ra * rb);

    mb::dispatch("div", inputs, 2, out.data, n, mb::dtype::complex64);
    expect_near(out.data[0], ra / rb);

    const void* fma_inputs[] = {a.data, b.data, b.data};
    mb::dispatch("fma", fma_inputs, 3, out.data, n, mb::dtype::complex64);
    expect_near(out.data[0], ra * rb + rb);

    mb::dispatch("neg", inputs, 1, out.data, n, mb::dtype::complex64);
    expect_near(out.data[0], -ra);

    mb::dispatch("conj", inputs, 1, out.data, n, mb::dtype::complex64);
    expect_near(out.data[0], std::conj(ra));

    // Every lane ran the same op, not just lane 0 (partial-threadgroup coverage).
    mb::dispatch("add", inputs, 2, out.data, n, mb::dtype::complex64);
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::complex<float> expect(
            a.data[i].real() + b.data[i].real(), a.data[i].imag() + b.data[i].imag());
        expect_near(out.data[i], expect);
    }
}

TEST_F(metal_complex, Transcendentals)
{
    constexpr std::size_t n = 64;
    metal_buffer<cfloat>  a(n), out(n);
    metal_buffer<float>   mag(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        a.data[i] = cfloat(1.2f + float(i) * 0.05f, -0.7f + float(i) * 0.02f);
    }

    const void* inputs[] = {a.data};

    mb::dispatch("sqrt", inputs, 1, out.data, n, mb::dtype::complex64);
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::complex<float> z(a.data[i].real(), a.data[i].imag());
        expect_near(out.data[i], std::sqrt(z), 1e-3f);
    }

    mb::dispatch("exp", inputs, 1, out.data, n, mb::dtype::complex64);
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::complex<float> z(a.data[i].real(), a.data[i].imag());
        expect_near(out.data[i], std::exp(z), 1e-3f);
    }

    mb::dispatch("log", inputs, 1, out.data, n, mb::dtype::complex64);
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::complex<float> z(a.data[i].real(), a.data[i].imag());
        expect_near(out.data[i], std::log(z), 1e-3f);
    }

    mb::dispatch("sin", inputs, 1, out.data, n, mb::dtype::complex64);
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::complex<float> z(a.data[i].real(), a.data[i].imag());
        expect_near(out.data[i], std::sin(z), 1e-2f);
    }

    mb::dispatch("cos", inputs, 1, out.data, n, mb::dtype::complex64);
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::complex<float> z(a.data[i].real(), a.data[i].imag());
        expect_near(out.data[i], std::cos(z), 1e-2f);
    }

    mb::dispatch("tanh", inputs, 1, out.data, n, mb::dtype::complex64);
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::complex<float> z(a.data[i].real(), a.data[i].imag());
        expect_near(out.data[i], std::tanh(z), 1e-2f);
    }

    mb::dispatch("fabs", inputs, 1, mag.data, n, mb::dtype::complex64);
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::complex<float> z(a.data[i].real(), a.data[i].imag());
        EXPECT_NEAR(mag.data[i], std::abs(z), 1e-4f);
    }
}

TEST_F(metal_complex, Fill)
{
    constexpr std::size_t n = 9;
    metal_buffer<cfloat>  out(n);
    mb::dispatch_fill(out.data, 3.5f, n, mb::dtype::complex64);
    for (std::size_t i = 0; i < n; ++i)
    {
        expect_near(out.data[i], std::complex<float>(3.5f, 0.0f));
    }
}

TEST_F(metal_complex, BatchedDependentLaunchesAndReuse)
{
    constexpr std::size_t n = 517;
    metal_buffer<float>   a(n), b(n), out(n);
    const void*           inputs[] = {a.data, b.data};
    mb::execution_context context;
    context.dispatch_fill(a.data, 2.f, n);
    context.dispatch_fill(b.data, 3.f, n);
    context.dispatch("add", inputs, 2, out.data, n);
    context.submit();
    const void* dependent[] = {out.data, b.data};
    context.dispatch("mul", dependent, 2, out.data, n);
    context.wait();
    context.wait();
    for (std::size_t i = 0; i < n; ++i)
    {
        EXPECT_EQ(out.data[i], 15.f);
    }

    // Backing storage stays alive until the encoded batch completes.
    metal_buffer<float> temporary(n);
    context.dispatch_fill(temporary.data, 4.f, n);
    const void* added[] = {temporary.data, b.data};
    context.dispatch("add", added, 2, out.data, n);
    context.wait();
    for (std::size_t i = 0; i < n; ++i)
    {
        EXPECT_EQ(out.data[i], 7.f);
    }

    // Empty launches accept null pointers and do not submit work.
    context.dispatch("add", nullptr, 2, nullptr, 0);
    context.dispatch_fill(nullptr, 1.f, 0);
    context.dispatch_fused("", nullptr, 0, nullptr, 0, nullptr, 0);
    context.wait();
}

TEST_F(metal_complex, BatchedFusedAndDestructorCompletion)
{
    constexpr std::size_t n = 259;
    metal_buffer<float>   input(n), out(n);
    const std::string     source = R"(
        #include <metal_stdlib>
        using namespace metal;
        kernel void fused_float(device const float* a [[buffer(0)]],
            constant float& factor [[buffer(1)]], device float* out [[buffer(2)]],
            constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
            if (i < n) out[i] = a[i] * factor;
        })";
    {
        mb::execution_context context;
        context.dispatch_fill(input.data, 2.f, n);
        const void* inputs[] = {input.data};
        const float factor   = 3.f;
        context.dispatch_fused(source, inputs, 1, &factor, 1, out.data, n);
        // Destruction drains the encoded batch, including copied scalar args.
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        EXPECT_EQ(out.data[i], 6.f);
    }
}

TEST_F(metal_complex, BatchRecoversFromEncodingError)
{
    metal_buffer<float>   out(3);
    mb::execution_context context;
    const void*           invalid[] = {nullptr};
    EXPECT_THROW(context.dispatch("neg", invalid, 1, out.data, 3), std::invalid_argument);
    context.dispatch_fill(out.data, 5.f, 3);
    context.wait();
    EXPECT_EQ(out.data[2], 5.f);
    EXPECT_THROW(
        context.dispatch_fill(
            out.data, 1.f, static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) + 1),
        std::invalid_argument);
}

TEST_F(metal_complex, ReductionWorkspaceReuseAndConcurrentCalls)
{
    constexpr std::size_t n = 300001;
    metal_buffer<float>   input(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        input.data[i] = 1.f;
    }
    for (std::size_t count : {n, std::size_t(1), std::size_t(257), std::size_t(65537), n})
    {
        EXPECT_EQ(mb::reduce_sum(input.data, count), float(count));
        EXPECT_EQ(mb::reduce_min(input.data, count), 1.f);
        EXPECT_EQ(mb::reduce_max(input.data, count), 1.f);
    }
    float       first = 0, second = 0;
    std::thread a([&] { first = mb::reduce_sum(input.data, n); });
    std::thread b([&] { second = mb::reduce_sum(input.data, n - 1); });
    a.join();
    b.join();
    EXPECT_EQ(first, float(n));
    EXPECT_EQ(second, float(n - 1));
}

#endif  // SCALARS_HAS_METAL
