#pragma once
#include <cmath>
#include <complex>
#include <type_traits>
#include <types/bfloat16.h>
#include <types/half.h>
namespace kh::scalars
{
template <class T> struct alignas(2 * sizeof(T)) complex
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, half> || std::is_same_v<T, bfloat16>);
    using value_type = T;
    T re{}, im{};
    constexpr complex() noexcept = default;
    constexpr complex(T r, T i = T{}) noexcept : re(r), im(i) {}
    template <class U>
    explicit constexpr complex(const complex<U>& v) noexcept : re(T(v.real())), im(T(v.imag()))
    {
    }
    template <class U>
    explicit complex(const std::complex<U>& v) noexcept : re(T(v.real())), im(T(v.imag()))
    {
    }
    [[nodiscard]] constexpr T real() const noexcept { return re; }
    [[nodiscard]] constexpr T imag() const noexcept { return im; }
    constexpr void            real(T v) noexcept { re = v; }
    constexpr void            imag(T v) noexcept { im = v; }
    constexpr complex&        operator+=(complex x) noexcept
    {
        re = T(re + x.re);
        im = T(im + x.im);
        return *this;
    }
    constexpr complex& operator-=(complex x) noexcept
    {
        re = T(re - x.re);
        im = T(im - x.im);
        return *this;
    }
    constexpr complex& operator*=(complex x) noexcept
    {
        using work_type = std::conditional_t<std::is_same_v<T, double>, double, float>;
        work_type a = work_type(re), b = work_type(im), c = work_type(x.re), d = work_type(x.im);
        re = T(a * c - b * d);
        im = T(a * d + b * c);
        return *this;
    }
    complex& operator/=(complex x) noexcept
    {
        using work_type   = std::conditional_t<std::is_same_v<T, double>, double, float>;
        const work_type a = work_type(re), b = work_type(im), c = work_type(x.re),
                        d = work_type(x.im);
        const auto out    = std::complex<work_type>(a, b) / std::complex<work_type>(c, d);
        re                = T(out.real());
        im                = T(out.imag());
        return *this;
    }
    template <class U = T>
        requires(std::is_same_v<U, float> || std::is_same_v<U, double>)
    explicit operator std::complex<U>() const
    {
        return {re, im};
    }
};
template <class T> constexpr complex<T> operator+(complex<T> a, complex<T> b) noexcept
{
    return a += b;
}
template <class T> constexpr complex<T> operator-(complex<T> a, complex<T> b) noexcept
{
    return a -= b;
}
template <class T> constexpr complex<T> operator*(complex<T> a, complex<T> b) noexcept
{
    return a *= b;
}
template <class T> complex<T> operator/(complex<T> a, complex<T> b) noexcept
{
    return a /= b;
}
template <class T> constexpr complex<T> operator-(complex<T> z) noexcept
{
    return {-z.re, -z.im};
}
template <class T> constexpr bool operator==(complex<T> a, complex<T> b) noexcept
{
    return a.re == b.re && a.im == b.im;
}
template <class T> constexpr bool operator!=(complex<T> a, complex<T> b) noexcept
{
    return !(a == b);
}
template <class T> constexpr complex<T> conj(complex<T> z) noexcept
{
    return {z.re, -z.im};
}
template <class T> constexpr auto norm(complex<T> z) noexcept
{
    using work_type = std::conditional_t<std::is_same_v<T, double>, double, float>;
    return work_type(z.re) * work_type(z.re) + work_type(z.im) * work_type(z.im);
}
template <class T> auto abs(complex<T> z) noexcept
{
    using work_type = std::conditional_t<std::is_same_v<T, double>, double, float>;
    return std::hypot(work_type(z.re), work_type(z.im));
}
using complex_float  = complex<float>;
using complex_double = complex<double>;
using complex_half   = complex<half>;
using complex_bf16   = complex<bfloat16>;
static_assert(sizeof(complex_float) == 8 && alignof(complex_float) == 8);
static_assert(sizeof(complex_double) == 16 && alignof(complex_double) == 16);
static_assert(sizeof(complex_half) == 4 && sizeof(complex_bf16) == 4);
static_assert(std::is_trivially_copyable_v<complex_float>);
}  // namespace kh::scalars
