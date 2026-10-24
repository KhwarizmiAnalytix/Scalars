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
    using P  = ::simd<complex<T>>;
    auto sum = P::setzero();
    for (; count - k >= P::size; k += P::size)
    {
        auto a = P::gather(x + k * sx, sx);
        if (conjugate_x)
            a = P::conj(a);
        sum = P::fma(a, P::gather(y + k * sy, sy), sum);
    }
    result = P::accumulate(sum);
#endif
    for (; k < count; ++k)
        result += (conjugate_x ? scalars::conj(x[k * sx]) : x[k * sx]) * y[k * sy];
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
    using P           = ::simd<complex<T>>;
    const auto factor = P::set(alpha);
    for (; count - k >= P::size; k += P::size)
    {
        auto a = P::gather(x + k * sx, sx);
        if (conjugate_x)
            a = P::conj(a);
        const auto b = P::gather(y + k * sy, sy);
        P::scatter(P::fma(factor, a, b), sy, y + k * sy);
    }
#endif
    for (; k < count; ++k)
        y[k * sy] += alpha * (conjugate_x ? scalars::conj(x[k * sx]) : x[k * sx]);
}

template <class T>
void complex_scale(complex<T>* x, std::ptrdiff_t stride, std::ptrdiff_t count, T scale)
{
    assert(count >= 0 && (count <= 1 || stride != 0));
    std::ptrdiff_t k = 0;
#if SCALARS_VECTORIZED && !SCALARS_ON_GPU_DEVICE
    using P           = ::simd<complex<T>>;
    using R           = ::simd<T>;
    const auto factor = R::set(scale);
    for (; count - k >= P::size; k += P::size)
    {
        auto a = P::gather(x + k * stride, stride);
        a.real = R::mul(a.real, factor);
        a.imag = R::mul(a.imag, factor);
        P::scatter(a, stride, x + k * stride);
    }
#endif
    for (; k < count; ++k)
        x[k * stride] *= scale;
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
    using P                       = ::simd<complex<T>>;
    using R                       = ::simd<T>;
    const auto cosine             = R::set(c);
    const auto sine               = P::set(s);
    const auto negative_conjugate = P::neg(P::conj(sine));
    for (; count - k >= P::size; k += P::size)
    {
        const auto               a = P::gather(x + k * sx, sx);
        const auto               b = P::gather(y + k * sy, sy);
        const typename P::simd_t ca{R::mul(a.real, cosine), R::mul(a.imag, cosine)};
        const typename P::simd_t cb{R::mul(b.real, cosine), R::mul(b.imag, cosine)};
        P::scatter(P::fma(sine, b, ca), sx, x + k * sx);
        P::scatter(P::fma(negative_conjugate, a, cb), sy, y + k * sy);
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
