#include "scalars_macros.h"

#if SCALARS_HAS_METAL

#include "backend/gpu/metal/metal_dispatch.h"
#include "types/complex.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>

namespace
{
namespace mb = scalars::metal_backend;
using cfloat = scalars::complex<float>;

template <class T> struct metal_buffer
{
    T* data;
    explicit metal_buffer(std::size_t n) : data(static_cast<T*>(mb::allocate(n * sizeof(T)))) {}
    ~metal_buffer() { mb::deallocate(data); }
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
    metal_buffer<cfloat>   a(n), b(n), out(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        a.data[i] = cfloat(1.5f + float(i) * 0.01f, -0.5f - float(i) * 0.005f);
        b.data[i] = cfloat(2.0f, 1.0f);
    }
    const std::complex<float> ra(a.data[0].real(), a.data[0].imag());
    const std::complex<float> rb(b.data[0].real(), b.data[0].imag());
    const void*                inputs[] = {a.data, b.data};

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
        const std::complex<float> expect(a.data[i].real() + b.data[i].real(),
            a.data[i].imag() + b.data[i].imag());
        expect_near(out.data[i], expect);
    }
}

TEST_F(metal_complex, Transcendentals)
{
    constexpr std::size_t n = 64;
    metal_buffer<cfloat>   a(n), out(n);
    metal_buffer<float>    mag(n);
    for (std::size_t i = 0; i < n; ++i)
        a.data[i] = cfloat(1.2f + float(i) * 0.05f, -0.7f + float(i) * 0.02f);

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
    metal_buffer<cfloat>   out(n);
    mb::dispatch_fill(out.data, 3.5f, n, mb::dtype::complex64);
    for (std::size_t i = 0; i < n; ++i)
        expect_near(out.data[i], std::complex<float>(3.5f, 0.0f));
}

#endif  // SCALARS_HAS_METAL
