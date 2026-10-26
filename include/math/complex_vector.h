#pragma once

#include <cassert>
#include <cstddef>

#include "backend/simd.h"
#include "types/complex.h"

namespace scalars
{
// Host vector primitives shared by matrix algorithms. Strides count complex<T>
// objects, may be negative, and the caller supplies the first logical element.
// Output strides must be nonzero. Outputs must not overlap other logical
// elements; axpy permits x == y with matching strides. Empty ranges accept null.
template <class T>
complex<T> complex_dot(const complex<T>* x,
    std::ptrdiff_t                       sx,
    const complex<T>*                    y,
    std::ptrdiff_t                       sy,
    std::ptrdiff_t                       count,
    bool                                 conjugate_x = false)
{
    assert(count >= 0);
    std::ptrdiff_t k = 0;
    complex<T>     result{};
#if SCALARS_VECTORIZED && !SCALARS_ON_GPU_DEVICE
    using p         = ::simd<complex<T>>;
    auto       sum0 = p::setzero(), sum1 = sum0, sum2 = sum0, sum3 = sum0;
    const auto product_add = [&](std::ptrdiff_t index, typename p::simd_t sum)
    {
        auto a = p::gather(x + index * sx, sx);
        if (conjugate_x)
        {
            a = p::conj(a);
        }
        return p::fma(a, p::gather(y + index * sy, sy), sum);
    };
    // Independent chains hide FMA latency without reassociating each packet's
    // arithmetic. Reduce to the wider accumulation type before combining them.
    for (; count - k >= 4 * p::size; k += 4 * p::size)
    {
        sum0 = product_add(k, sum0);
        sum1 = product_add(k + p::size, sum1);
        sum2 = product_add(k + 2 * p::size, sum2);
        sum3 = product_add(k + 3 * p::size, sum3);
    }
    for (; count - k >= p::size; k += p::size)
    {
        sum0 = product_add(k, sum0);
    }
    result = p::accumulate(sum0) + p::accumulate(sum1) + p::accumulate(sum2) + p::accumulate(sum3);
#endif
    for (; k < count; ++k)
    {
        result += (conjugate_x ? scalars::conj(x[k * sx]) : x[k * sx]) * y[k * sy];
    }
    return result;
}

template <class T>
void complex_axpy(complex<T>* y,
    std::ptrdiff_t            sy,
    const complex<T>*         x,
    std::ptrdiff_t            sx,
    std::ptrdiff_t            count,
    complex<T>                alpha,
    bool                      conjugate_x = false)
{
    assert(count >= 0 && (count <= 1 || sy != 0));
    std::ptrdiff_t k = 0;
#if SCALARS_VECTORIZED && !SCALARS_ON_GPU_DEVICE
    using p           = ::simd<complex<T>>;
    const auto factor = p::set(alpha);
    for (; count - k >= p::size; k += p::size)
    {
        auto a = p::gather(x + k * sx, sx);
        if (conjugate_x)
        {
            a = p::conj(a);
        }
        const auto b = p::gather(y + k * sy, sy);
        p::scatter(p::fma(factor, a, b), sy, y + k * sy);
    }
#endif
    for (; k < count; ++k)
    {
        y[k * sy] += alpha * (conjugate_x ? scalars::conj(x[k * sx]) : x[k * sx]);
    }
}

template <class T>
void complex_scale(complex<T>* x, std::ptrdiff_t stride, std::ptrdiff_t count, T scale)
{
    assert(count >= 0 && (count <= 1 || stride != 0));
    std::ptrdiff_t k = 0;
#if SCALARS_VECTORIZED && !SCALARS_ON_GPU_DEVICE
    using p           = ::simd<complex<T>>;
    using r           = ::simd<T>;
    const auto factor = r::set(scale);
    for (; count - k >= p::size; k += p::size)
    {
        auto a = p::gather(x + k * stride, stride);
        a.real = r::mul(a.real, factor);
        a.imag = r::mul(a.imag, factor);
        p::scatter(a, stride, x + k * stride);
    }
#endif
    for (; k < count; ++k)
    {
        x[k * stride] *= scale;
    }
}

// x' = c*x + s*y; y' = c*y - conj(s)*x. Both inputs are loaded before
// either store. The two output ranges must be disjoint; no scratch is needed.
template <class T>
void complex_rotate(complex<T>* x,
    std::ptrdiff_t              sx,
    complex<T>*                 y,
    std::ptrdiff_t              sy,
    std::ptrdiff_t              count,
    T                           c,
    complex<T>                  s)
{
    assert(count >= 0 && (count <= 1 || (sx != 0 && sy != 0)));
    std::ptrdiff_t k = 0;
#if SCALARS_VECTORIZED && !SCALARS_ON_GPU_DEVICE
    using p                       = ::simd<complex<T>>;
    using r                       = ::simd<T>;
    const auto cosine             = r::set(c);
    const auto sine               = p::set(s);
    const auto negative_conjugate = p::neg(p::conj(sine));
    for (; count - k >= p::size; k += p::size)
    {
        const auto               a = p::gather(x + k * sx, sx);
        const auto               b = p::gather(y + k * sy, sy);
        const typename p::simd_t ca{r::mul(a.real, cosine), r::mul(a.imag, cosine)};
        const typename p::simd_t cb{r::mul(b.real, cosine), r::mul(b.imag, cosine)};
        p::scatter(p::fma(sine, b, ca), sx, x + k * sx);
        p::scatter(p::fma(negative_conjugate, a, cb), sy, y + k * sy);
    }
#endif
    for (; k < count; ++k)
    {
        const auto a = x[k * sx];
        const auto b = y[k * sy];
        x[k * sx]    = a * c + s * b;
        y[k * sy]    = b * c - scalars::conj(s) * a;
    }
}
}  // namespace scalars
