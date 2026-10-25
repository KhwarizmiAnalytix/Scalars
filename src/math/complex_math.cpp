#include "types/complex.h"

#include <cmath>
#include <complex>

// Note [ Complex Square root in libc++]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// In libc++ complex square root is computed using polar form
// This is a reasonably fast algorithm, but can result in significant
// numerical errors when arg is close to 0, pi/2, pi, or 3pi/4
// In that case provide a more conservative implementation which is
// slower but less prone to those kinds of errors
// In libstdc++ complex square root yield invalid results
// for -x-0.0j unless C99 csqrt/csqrtf fallbacks are used

#if defined(_LIBCPP_VERSION) || \
    (defined(__GLIBCXX__) && !defined(_GLIBCXX11_USE_C99_COMPLEX))

namespace {
template <typename T>
scalars::complex<T> compute_csqrt(const scalars::complex<T>& z) {
  constexpr auto half = T(.5);

  // Trust standard library to correctly handle infs and NaNs
  if (std::isinf(z.real()) || std::isinf(z.imag()) || std::isnan(z.real()) ||
      std::isnan(z.imag())) {
    return static_cast<scalars::complex<T>>(
        std::sqrt(static_cast<std::complex<T>>(z)));
  }

  const T scale = std::fmax(std::abs(z.real()), std::abs(z.imag()));
  if (scale == T(0))
  {
    return {T(0), z.imag()};
  }

  // An even power-of-two normalization keeps hypot and the subsequent sum
  // finite, and avoids halving a subnormal input down to zero.
  int exponent = 0;
  std::frexp(scale, &exponent);
  if (exponent % 2 != 0)
  {
    --exponent;
  }
  const T real = std::scalbn(z.real(), -exponent);
  const T imag = std::scalbn(z.imag(), -exponent);
  const T t = std::sqrt((std::hypot(real, imag) + std::abs(real)) * half);
  const T result_real = real >= T(0) ? t : half * std::abs(imag / t);
  const T result_imag = real >= T(0) ? half * (imag / t) : std::copysign(t, imag);
  return {std::scalbn(result_real, exponent / 2), std::scalbn(result_imag, exponent / 2)};
}

// Compute complex arccosine using formula from W. Kahan
// "Branch Cuts for Complex Elementary Functions" 1986 paper:
// cacos(z).re = 2*atan2(sqrt(1-z).re(), sqrt(1+z).re())
// cacos(z).im = asinh((sqrt(conj(1+z))*sqrt(1-z)).im())
template <typename T>
scalars::complex<T> compute_cacos(const scalars::complex<T>& z) {
  auto constexpr one = T(1);
  // Trust standard library to correctly handle infs and NaNs
  if (std::isinf(z.real()) || std::isinf(z.imag()) || std::isnan(z.real()) ||
      std::isnan(z.imag())) {
    return static_cast<scalars::complex<T>>(
        std::acos(static_cast<std::complex<T>>(z)));
  }
  auto a = compute_csqrt(scalars::complex<T>(one - z.real(), -z.imag()));
  auto b = compute_csqrt(scalars::complex<T>(one + z.real(), z.imag()));
  auto c = compute_csqrt(scalars::complex<T>(one + z.real(), -z.imag()));
  auto r = T(2) * std::atan2(a.real(), b.real());
  // Explicitly unroll (a*c).imag()
  auto i = std::asinh(a.real() * c.imag() + a.imag() * c.real());
  return scalars::complex<T>(r, i);
}
} // anonymous namespace

namespace scalars::complex_math {
namespace _detail {
scalars::complex<float> sqrt(const scalars::complex<float>& in) {
  return compute_csqrt(in);
}

scalars::complex<double> sqrt(const scalars::complex<double>& in) {
  return compute_csqrt(in);
}

scalars::complex<float> acos(const scalars::complex<float>& in) {
  return compute_cacos(in);
}

scalars::complex<double> acos(const scalars::complex<double>& in) {
  return compute_cacos(in);
}

} // namespace _detail
} // namespace scalars::complex_math
#endif
