#pragma once

#include <array>
#include <cassert>
#include <cstddef>

#include "backend/simd.h"
#include "types/complex.h"

namespace scalars::detail_complex_simd
{
// Complex arithmetic uses the existing real SIMD implementation directly.
// Separate registers avoid treating members of adjacent complex objects as a T array.
template <class T> struct complex_simd
{
    using r       = ::simd<T>;
    using value_t = complex<T>;
    struct simd_t
    {
        typename r::simd_t real, imag;
    };
    static constexpr int size = r::size;

    static simd_t gather(const value_t* p, std::ptrdiff_t stride)
    {
        std::array<T, size> real, imag;
        for (int i = 0; i < size; ++i)
        {
            real[i] = p[i * stride].real();
            imag[i] = p[i * stride].imag();
        }
        return {r::loadu(real.data()), r::loadu(imag.data())};
    }
    static void scatter(simd_t a, std::ptrdiff_t stride, value_t* p)
    {
        std::array<T, size> real, imag;
        r::storeu(a.real, real.data());
        r::storeu(a.imag, imag.data());
        for (int i = 0; i < size; ++i)
        {
            p[i * stride] = value_t(real[i], imag[i]);
        }
    }
    static simd_t loadu(const value_t* p) { return gather(p, 1); }
    static simd_t load(const value_t* p) { return loadu(p); }
    static void   storeu(simd_t a, value_t* p) { scatter(a, 1, p); }
    static void   store(simd_t a, value_t* p) { storeu(a, p); }
    static simd_t loadu(const value_t* p, int count)
    {
        assert(count >= 0 && count <= size);
        if (count == size)
        {
            return loadu(p);
        }
        std::array<value_t, size> lanes{};
        for (int i = 0; i < count; ++i)
        {
            lanes[i] = p[i];
        }
        return loadu(lanes.data());
    }
    static void storeu(simd_t a, value_t* p, int count)
    {
        assert(count >= 0 && count <= size);
        if (count == size)
        {
            storeu(a, p);
            return;
        }
        std::array<value_t, size> lanes;
        storeu(a, lanes.data());
        for (int i = 0; i < count; ++i)
        {
            p[i] = lanes[i];
        }
    }
    static simd_t set(value_t a) { return {r::set(a.real()), r::set(a.imag())}; }
    static simd_t setzero() { return {r::setzero(), r::setzero()}; }
    static simd_t add(simd_t a, simd_t b)
    {
        return {r::add(a.real, b.real), r::add(a.imag, b.imag)};
    }
    static simd_t sub(simd_t a, simd_t b)
    {
        return {r::sub(a.real, b.real), r::sub(a.imag, b.imag)};
    }
    static simd_t neg(simd_t a) { return {r::neg(a.real), r::neg(a.imag)}; }
    static simd_t conj(simd_t a) { return {a.real, r::neg(a.imag)}; }
    static simd_t mul(simd_t a, simd_t b)
    {
        return {r::sub(r::mul(a.real, b.real), r::mul(a.imag, b.imag)),
            r::add(r::mul(a.real, b.imag), r::mul(a.imag, b.real))};
    }
    static simd_t fma(simd_t a, simd_t b, simd_t c)
    {
        return {r::fma(r::neg(a.imag), b.imag, r::fma(a.real, b.real, c.real)),
            r::fma(a.imag, b.real, r::fma(a.real, b.imag, c.imag))};
    }
    // Division retains the scalar implementation's behavior at extreme magnitudes.
    static simd_t div(simd_t a, simd_t b)
    {
        std::array<value_t, size> x, y;
        storeu(a, x.data());
        storeu(b, y.data());
        for (int i = 0; i < size; ++i)
        {
            x[i] /= y[i];
        }
        return loadu(x.data());
    }
    static value_t accumulate(simd_t a)
    {
        return {static_cast<T>(r::accumulate(a.real)), static_cast<T>(r::accumulate(a.imag))};
    }
};
}  // namespace scalars::detail_complex_simd

template <> struct simd<scalars::complex<float>> : scalars::detail_complex_simd::complex_simd<float>
{
};
template <>
struct simd<scalars::complex<double>> : scalars::detail_complex_simd::complex_simd<double>
{
};
