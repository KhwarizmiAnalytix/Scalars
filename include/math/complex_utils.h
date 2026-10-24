#if !defined(SCALARS_INTERNAL_INCLUDE_COMPLEX_REMAINING_H)
#error \
    "math/complex_utils.h is not meant to be individually included. Include types/complex.h instead."
#endif

#include <limits>

namespace scalars {

template <typename T>
struct is_complex : public std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : public std::true_type {};

template <typename T>
struct is_complex<scalars::complex<T>> : public std::true_type {};

// Extract double from std::complex<double>; is identity otherwise
// TODO: Write in more idiomatic C++17
template <typename T>
struct scalar_value_type {
  using type = T;
};
template <typename T>
struct scalar_value_type<std::complex<T>> {
  using type = T;
};
template <typename T>
struct scalar_value_type<scalars::complex<T>> {
  using type = T;
};

} // namespace scalars

namespace std {
// NOLINTBEGIN(bugprone-std-namespace-modification) -- numeric_limits
// specialization is standard-sanctioned; the isnan overload mirrors it so
// generic code can call std::isnan uniformly on scalars::complex<T>.

template <typename T>
class numeric_limits<scalars::complex<T>> : public numeric_limits<T> {};

template <typename T>
bool isnan(const scalars::complex<T>& v) {
  return std::isnan(v.real()) || std::isnan(v.imag());
}

// NOLINTEND(bugprone-std-namespace-modification)
} // namespace std
