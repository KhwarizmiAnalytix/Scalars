#pragma once
#include <Scalars/types/Half.h>
#include <Scalars/types/BFloat16.h>
#include <complex>
#include <type_traits>
#include <cmath>
namespace kh::scalars {
template<class T> struct alignas(2*sizeof(T)) Complex {
  static_assert(std::is_same_v<T,float> || std::is_same_v<T,double> ||
                std::is_same_v<T,Half> || std::is_same_v<T,BFloat16>);
  using value_type=T;
  T re{}, im{};
  constexpr Complex() noexcept=default;
  constexpr Complex(T r,T i=T{}) noexcept : re(r),im(i) {}
  template<class U> explicit constexpr Complex(const Complex<U>& v) noexcept : re(T(v.real())), im(T(v.imag())) {}
  template<class U> explicit Complex(const std::complex<U>& v) noexcept : re(T(v.real())),im(T(v.imag())) {}
  [[nodiscard]] constexpr T real() const noexcept { return re; }
  [[nodiscard]] constexpr T imag() const noexcept { return im; }
  constexpr void real(T v) noexcept { re=v; }
  constexpr void imag(T v) noexcept { im=v; }
  constexpr Complex& operator+=(Complex x) noexcept { re=T(re+x.re); im=T(im+x.im);return *this; }
  constexpr Complex& operator-=(Complex x) noexcept { re=T(re-x.re); im=T(im-x.im);return *this; }
  constexpr Complex& operator*=(Complex x) noexcept {
    using W=std::conditional_t<std::is_same_v<T,double>,double,float>;
    W a=W(re),b=W(im),c=W(x.re),d=W(x.im);
    re=T(a*c-b*d); im=T(a*d+b*c);return *this;
  }
  Complex& operator/=(Complex x) noexcept {
    using W=std::conditional_t<std::is_same_v<T,double>,double,float>;
    const W a=W(re),b=W(im),c=W(x.re),d=W(x.im);
    const auto out=std::complex<W>(a,b)/std::complex<W>(c,d);
    re=T(out.real()); im=T(out.imag()); return *this;
  }
  template<class U=T> requires(std::is_same_v<U,float> || std::is_same_v<U,double>)
  explicit operator std::complex<U>() const { return {re,im}; }
};
template<class T> constexpr Complex<T> operator+(Complex<T> a,Complex<T> b) noexcept {return a+=b;}
template<class T> constexpr Complex<T> operator-(Complex<T> a,Complex<T> b) noexcept {return a-=b;}
template<class T> constexpr Complex<T> operator*(Complex<T> a,Complex<T> b) noexcept {return a*=b;}
template<class T> Complex<T> operator/(Complex<T> a,Complex<T> b) noexcept {return a/=b;}
template<class T> constexpr Complex<T> operator-(Complex<T> z) noexcept {return {-z.re,-z.im};}
template<class T> constexpr bool operator==(Complex<T> a,Complex<T> b) noexcept {return a.re==b.re && a.im==b.im;}
template<class T> constexpr bool operator!=(Complex<T> a,Complex<T> b) noexcept {return !(a==b);}
template<class T> constexpr Complex<T> conj(Complex<T> z) noexcept { return {z.re,-z.im}; }
template<class T> constexpr auto norm(Complex<T> z) noexcept {using W=std::conditional_t<std::is_same_v<T,double>,double,float>;return W(z.re)*W(z.re)+W(z.im)*W(z.im);}
template<class T> auto abs(Complex<T> z) noexcept {using W=std::conditional_t<std::is_same_v<T,double>,double,float>;return std::hypot(W(z.re),W(z.im));}
using ComplexFloat=Complex<float>;
using ComplexDouble=Complex<double>;
using ComplexHalf=Complex<Half>;
using ComplexBF16=Complex<BFloat16>;
static_assert(sizeof(ComplexFloat)==8 && alignof(ComplexFloat)==8);
static_assert(sizeof(ComplexDouble)==16 && alignof(ComplexDouble)==16);
static_assert(sizeof(ComplexHalf)==4 && sizeof(ComplexBF16)==4);
static_assert(std::is_trivially_copyable_v<ComplexFloat>);
}
