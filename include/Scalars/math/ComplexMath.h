#pragma once
#include <Scalars/types/Complex.h>
#include <complex>
#include <cmath>
namespace kh::scalars {
namespace detail {
template<class T> using complex_work_t=std::conditional_t<std::is_same_v<T,double>,double,float>;
template<class T> std::complex<complex_work_t<T>> to_std(Complex<T> v) {
 return {static_cast<complex_work_t<T>>(v.real()),static_cast<complex_work_t<T>>(v.imag())};
}
template<class T> Complex<T> from_std(std::complex<complex_work_t<T>> v) {
 return {T(v.real()),T(v.imag())};
}
}
#define KH_COMPLEX_UNARY(NAME) template<class T> Complex<T> NAME(Complex<T> z) {return detail::from_std<T>(std::NAME(detail::to_std(z)));}
KH_COMPLEX_UNARY(exp)
KH_COMPLEX_UNARY(log)
KH_COMPLEX_UNARY(log10)
KH_COMPLEX_UNARY(sqrt)
KH_COMPLEX_UNARY(sin)
KH_COMPLEX_UNARY(cos)
KH_COMPLEX_UNARY(tan)
KH_COMPLEX_UNARY(asin)
KH_COMPLEX_UNARY(acos)
KH_COMPLEX_UNARY(atan)
KH_COMPLEX_UNARY(sinh)
KH_COMPLEX_UNARY(cosh)
KH_COMPLEX_UNARY(tanh)
KH_COMPLEX_UNARY(asinh)
KH_COMPLEX_UNARY(acosh)
KH_COMPLEX_UNARY(atanh)
#undef KH_COMPLEX_UNARY
template<class T> Complex<T> pow(Complex<T> a,Complex<T> b) {return detail::from_std<T>(std::pow(detail::to_std(a),detail::to_std(b)));}
template<class T> auto arg(Complex<T> z) {return std::arg(detail::to_std(z));}
}
