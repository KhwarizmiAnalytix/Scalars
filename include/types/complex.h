#pragma once

#include <cmath>
#include <complex>
#include <type_traits>

#include "scalars_macros.h"
#include "types/bfloat16.h"
#include "types/half.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/complex.h>
#endif

SCALARS_CLANG_DIAGNOSTIC_PUSH()
#if SCALARS_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
SCALARS_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
#endif
#if SCALARS_CLANG_HAS_WARNING("-Wfloat-conversion")
SCALARS_CLANG_DIAGNOSTIC_IGNORE("-Wfloat-conversion")
#endif

namespace scalars
{

namespace detail_complex
{
// Conservative bounds keep the direct division's products, denominator, and
// quotient in range. Extreme components retain the scaled implementation below.
template <class T> inline bool division_component_is_bounded(T value)
{
    constexpr T lower     = std::is_same_v<T, float> ? T(0x1p-30) : T(0x1p-250);
    constexpr T upper     = std::is_same_v<T, float> ? T(0x1p31) : T(0x1p251);
    const T     magnitude = std::fabs(value);
    return value == T(0) || (magnitude >= lower && magnitude < upper);
}

// A constexpr-safe absolute value, correct for signed zero (-0 maps to +0).
// std::abs/std::fabs are not guaranteed constexpr for floating-point types
// before C++23 (libstdc++ accepts it as an extension, libc++ historically
// does not), and this helper is used from a constexpr operator/=. NaN must
// pass through unchanged: mapping it to 0 would send a NaN divisor down
// operator/='s divide-by-zero branch and produce inf instead of NaN.
template <class T> SCALARS_HOST_DEVICE constexpr T division_abs(T value)
{
    return value < T(0) ? -value : (value == T(0) ? T(0) : value);
}

// Evaluate (x*u + y*v) / denominator * 2^exponent without overflowing
// products or losing a subnormal product before the final rescaling.
template <class T> inline T scaled_sum_products(T x, T u, T y, T v, T denominator, int exponent)
{
    int       ex = 0, eu = 0, ey = 0, ev = 0;
    const T   mx = std::frexp(x, &ex), mu = std::frexp(u, &eu);
    const T   my = std::frexp(y, &ey), mv = std::frexp(v, &ev);
    const T   first = mx * mu, second = my * mv;
    const int first_exponent = ex + eu, second_exponent = ey + ev;
    const int common_exponent =
        first == T(0)    ? second_exponent
        : second == T(0) ? first_exponent
                         : (first_exponent > second_exponent ? first_exponent : second_exponent);
    const T sum = std::scalbn(first, first_exponent - common_exponent) +
                  std::scalbn(second, second_exponent - common_exponent);
    return std::scalbn(sum / denominator, exponent + common_exponent);
}
}  // namespace detail_complex

// scalars::complex is an implementation of complex numbers that aims
// to work on all devices supported by PyTorch
//
// Most of the APIs duplicates std::complex
// Reference: https://en.cppreference.com/w/cpp/numeric/complex
//
// [NOTE: Complex Operator Unification]
// Operators currently use a mix of std::complex, thrust::complex, and
// scalars::complex internally. The end state is that all operators will use
// scalars::complex internally.  Until then, there may be some hacks to support all
// variants.
//
//
// [Note on Constructors]
//
// The APIs of constructors are mostly copied from C++ standard:
//   https://en.cppreference.com/w/cpp/numeric/complex/complex
//
// Since C++14, all constructors are constexpr in std::complex
//
// There are three types of constructors:
// - initializing from real and imag:
//     `constexpr complex( const T& re = T(), const T& im = T() );`
// - implicitly-declared copy constructor
// - converting constructors
//
// Converting constructors:
// - std::complex defines converting constructor between float/double/long
// double,
//   while we define converting constructor between float/double.
// - For these converting constructors, upcasting is implicit, downcasting is
//   explicit.
// - We also define explicit casting from std::complex/thrust::complex
//   - Note that the conversion from thrust is not constexpr, because
//     thrust does not define them as constexpr ????
//
//
// [Operator =]
//
// The APIs of operator = are mostly copied from C++ standard:
//   https://en.cppreference.com/w/cpp/numeric/complex/operator%3D
//
// Since C++20, all operator= are constexpr. Although we are not building with
// C++20, we also obey this behavior.
//
// There are three types of assign operator:
// - Assign a real value from the same scalar type
//   - In std, this is templated as complex& operator=(const T& x)
//     with specialization `complex& operator=(T x)` for float/double/long
//     double Since we only support float and double, on will use `complex&
//     operator=(T x)`
// - Copy assignment operator and converting assignment operator
//   - There is no specialization of converting assignment operators, which type
//   is
//     convertible is solely dependent on whether the scalar type is convertible
//
// In addition to the standard assignment, we also provide assignment operators
// with std and thrust
//
//
// [Casting operators]
//
// std::complex does not have casting operators. We define casting operators
// casting to std::complex and thrust::complex
//
//
// [Operator ""]
//
// std::complex has custom literals `i`, `if` and `il` defined in namespace
// `std::literals::complex_literals`. We define our own custom literals in the
// namespace `scalars::complex_literals`. Our custom literals does not follow the
// same behavior as in std::complex, instead, we define _if, _id to construct
// float/double complex literals.
//
//
// [real() and imag()]
//
// In C++20, there are two overload of these functions, one it to return the
// real/imag, another is to set real/imag, they are both constexpr. We follow
// this design.
//
//
// [Operator +=,-=,*=,/=]
//
// Since C++20, these operators become constexpr. In our implementation, they
// are also constexpr.
//
// There are two types of such operators: operating with a real number, or
// operating with another complex number. For the operating with a real number,
// the generic template form has argument type `const T &`, while the overload
// for float/double/long double has `T`. We will follow the same type as
// float/double/long double in std.
//
// [Unary operator +-]
//
// Since C++20, they are constexpr. We also make them expr
//
// [Binary operators +-*/]
//
// Each operator has three versions (taking + as example):
// - complex + complex
// - complex + real
// - real + complex
//
// [Operator ==, !=]
//
// Each operator has three versions (taking == as example):
// - complex == complex
// - complex == real
// - real == complex
//
// Some of them are removed on C++20, but we decide to keep them
//
// [Operator <<, >>]
//
// These are implemented by casting to std::complex
//
//
//
// TODO: scalars::complex<scalars::half> is not currently supported,
// because:
//  - lots of members and functions of scalars::half are not constexpr
//  - thrust::complex only support float and double

template <typename T> struct alignas(sizeof(T) * 2) complex
{
    using value_type = T;

    T real_ = T(0);
    T imag_ = T(0);

    constexpr complex() = default;
    // cppcheck-suppress noExplicitConstructor -- deliberate, matches std::complex: a real value
    // converts implicitly to a complex with a zero imaginary part.
    SCALARS_HOST_DEVICE constexpr complex(const T& re, const T& im = T()) : real_(re), imag_(im) {}
    template <typename U>
    explicit constexpr complex(const std::complex<U>& other) : complex(other.real(), other.imag())
    {
    }
#if defined(__CUDACC__) || defined(__HIPCC__)
    template <typename U>
    explicit SCALARS_HOST_DEVICE complex(const thrust::complex<U>& other)
        : real_(other.real()), imag_(other.imag())
    {
    }
// NOTE can not be implemented as follow due to ROCm bug:
//   explicit SCALARS_HOST_DEVICE complex(const thrust::complex<U> &other):
//   complex(other.real(), other.imag()) {}
#endif

    // Use SFINAE to specialize casting constructor for scalars::complex<float> and
    // scalars::complex<double>
    template <typename U = T>
    SCALARS_HOST_DEVICE explicit constexpr complex(
        const std::enable_if_t<std::is_same_v<U, float>, complex<double>>& other)
        : real_(other.real_), imag_(other.imag_)
    {
    }

    // Deliberate: complex<float> -> complex<double> is a widening conversion, implicit by
    // design (narrowing the other way stays explicit above).
    template <typename U = T>
    // cppcheck-suppress noExplicitConstructor
    SCALARS_HOST_DEVICE constexpr complex(
        const std::enable_if_t<std::is_same_v<U, double>, complex<float>>& other)
        : real_(other.real_), imag_(other.imag_)
    {
    }

    SCALARS_HOST_DEVICE constexpr complex<T>& operator=(T re)
    {
        real_ = re;
        imag_ = 0;
        return *this;
    }

    SCALARS_HOST_DEVICE constexpr complex<T>& operator+=(T re)
    {
        real_ += re;
        return *this;
    }

    SCALARS_HOST_DEVICE constexpr complex<T>& operator-=(T re)
    {
        real_ -= re;
        return *this;
    }

    SCALARS_HOST_DEVICE constexpr complex<T>& operator*=(T re)
    {
        real_ *= re;
        imag_ *= re;
        return *this;
    }

    SCALARS_HOST_DEVICE constexpr complex<T>& operator/=(T re)
    {
        real_ /= re;
        imag_ /= re;
        return *this;
    }

    template <typename U> SCALARS_HOST_DEVICE constexpr complex<T>& operator=(const complex<U>& rhs)
    {
        real_ = rhs.real();
        imag_ = rhs.imag();
        return *this;
    }

    template <typename U>
    SCALARS_HOST_DEVICE constexpr complex<T>& operator+=(const complex<U>& rhs)
    {
        real_ += rhs.real();
        imag_ += rhs.imag();
        return *this;
    }

    template <typename U>
    SCALARS_HOST_DEVICE constexpr complex<T>& operator-=(const complex<U>& rhs)
    {
        real_ -= rhs.real();
        imag_ -= rhs.imag();
        return *this;
    }

    template <typename U>
    SCALARS_HOST_DEVICE constexpr complex<T>& operator*=(const complex<U>& rhs)
    {
        // (a + bi) * (c + di) = (a*c - b*d) + (a * d + b * c) i
        T a   = real_;
        T b   = imag_;
        U c   = rhs.real();
        U d   = rhs.imag();
        real_ = a * c - b * d;
        imag_ = a * d + b * c;
        return *this;
    }

    template <typename U>
    SCALARS_HOST_DEVICE SCALARS_FORCE_INLINE constexpr complex<T>& operator/=(
        const complex<U>& rhs) SCALARS_UBSAN_IGNORE_FLOAT_DIVIDE_BY_ZERO
    {
        // (a + bi) / (c + di) = (ac + bd)/(c^2 + d^2) + (bc - ad)/(c^2 + d^2) i
        // the calculation below follows numpy's complex division
        T a = real_;
        T b = imag_;
        U c = rhs.real();
        U d = rhs.imag();

#if !defined(__CUDACC__) && !defined(__HIPCC__)
        if constexpr (std::is_same_v<T, U> && std::is_floating_point_v<T>)
        {
            if (!std::is_constant_evaluated() && std::isfinite(a) && std::isfinite(b) &&
                std::isfinite(c) && std::isfinite(d) && (c != T(0) || d != T(0)))
            {
                if (detail_complex::division_component_is_bounded(a) &&
                    detail_complex::division_component_is_bounded(b) &&
                    detail_complex::division_component_is_bounded(c) &&
                    detail_complex::division_component_is_bounded(d))
                {
                    const T denominator = c * c + d * d;
                    real_               = (a * c + b * d) / denominator;
                    imag_               = (b * c - a * d) / denominator;
                    return *this;
                }
                // Normalize the divisor, then scale each numerator product
                // independently: components may differ by thousands of powers
                // of two while one quotient component remains finite.
                int denominator_exponent = 0;
                std::frexp(std::fmax(std::fabs(c), std::fabs(d)), &denominator_exponent);
                const T    normalized_c = std::scalbn(c, -denominator_exponent);
                const T    normalized_d = std::scalbn(d, -denominator_exponent);
                const auto denominator  = normalized_c * normalized_c + normalized_d * normalized_d;
                real_                   = detail_complex::scaled_sum_products(
                    a, c, b, d, denominator, -2 * denominator_exponent);
                imag_ = detail_complex::scaled_sum_products(
                    b, c, -a, d, denominator, -2 * denominator_exponent);
                return *this;
            }
        }
#endif

        // detail_complex::division_abs, not std::abs: correctly normalizes a
        // signed zero to +0 and is constexpr on every supported compiler.
        auto abs_c = detail_complex::division_abs(c);
        auto abs_d = detail_complex::division_abs(d);

        if (abs_c >= abs_d)
        {
            if (abs_c == U(0) && abs_d == U(0))
            {
                /* divide by zeros should yield a complex inf or nan */
                // Divide by c*c + d*d, which is +0 for any signed zeros, not
                // by abs_c/abs_d: MSVC 14.51 (VS 18) folds division_abs's
                // `value == 0 ? 0 : value` to `value` in Release, keeping a
                // -0 divisor and flipping the sign of the infinite quotient.
                const auto zero = c * c + d * d;
                real_           = a / zero;
                imag_           = b / zero;
            }
            else
            {
                auto rat = d / c;
                auto scl = U(1.0) / (c + d * rat);
                real_    = (a + b * rat) * scl;
                imag_    = (b - a * rat) * scl;
            }
        }
        else
        {
            auto rat = c / d;
            auto scl = U(1.0) / (d + c * rat);
            real_    = (a * rat + b) * scl;
            imag_    = (b * rat - a) * scl;
        }
        return *this;
    }
    template <typename U> constexpr complex<T>& operator=(const std::complex<U>& rhs)
    {
        real_ = rhs.real();
        imag_ = rhs.imag();
        return *this;
    }

#if defined(__CUDACC__) || defined(__HIPCC__)
    template <typename U> SCALARS_HOST_DEVICE complex<T>& operator=(const thrust::complex<U>& rhs)
    {
        real_ = rhs.real();
        imag_ = rhs.imag();
        return *this;
    }
#endif

    template <typename U> explicit constexpr operator std::complex<U>() const
    {
        return std::complex<U>(std::complex<T>(real(), imag()));
    }

#if defined(__CUDACC__) || defined(__HIPCC__)
    template <typename U> SCALARS_HOST_DEVICE explicit operator thrust::complex<U>() const
    {
        return static_cast<thrust::complex<U>>(thrust::complex<T>(real(), imag()));
    }
#endif

    // consistent with NumPy behavior
    explicit SCALARS_HOST_DEVICE constexpr operator bool() const { return real() || imag(); }

    SCALARS_HOST_DEVICE constexpr T    real() const { return real_; }
    SCALARS_HOST_DEVICE constexpr void real(T value) { real_ = value; }
    SCALARS_HOST_DEVICE constexpr T    imag() const { return imag_; }
    SCALARS_HOST_DEVICE constexpr void imag(T value) { imag_ = value; }
};

namespace complex_literals
{

SCALARS_HOST_DEVICE constexpr complex<float> operator""_if(long double imag)
{
    return complex<float>(0.0f, static_cast<float>(imag));
}

SCALARS_HOST_DEVICE constexpr complex<double> operator""_id(long double imag)
{
    return complex<double>(0.0, static_cast<double>(imag));
}

SCALARS_HOST_DEVICE constexpr complex<float> operator""_if(unsigned long long imag)
{
    return complex<float>(0.0f, static_cast<float>(imag));
}

SCALARS_HOST_DEVICE constexpr complex<double> operator""_id(unsigned long long imag)
{
    return complex<double>(0.0, static_cast<double>(imag));
}

}  // namespace complex_literals

template <typename T> SCALARS_HOST_DEVICE constexpr complex<T> operator+(const complex<T>& val)
{
    return val;
}

template <typename T> SCALARS_HOST_DEVICE constexpr complex<T> operator-(const complex<T>& val)
{
    return complex<T>(-val.real(), -val.imag());
}

template <typename T>
SCALARS_HOST_DEVICE constexpr complex<T> operator+(const complex<T>& lhs, const complex<T>& rhs)
{
    complex<T> result = lhs;
    return result += rhs;
}

template <typename T>
SCALARS_HOST_DEVICE constexpr complex<T> operator+(const complex<T>& lhs, const T& rhs)
{
    complex<T> result = lhs;
    return result += rhs;
}

template <typename T>
SCALARS_HOST_DEVICE constexpr complex<T> operator+(const T& lhs, const complex<T>& rhs)
{
    return complex<T>(lhs + rhs.real(), rhs.imag());
}

template <typename T>
SCALARS_HOST_DEVICE constexpr complex<T> operator-(const complex<T>& lhs, const complex<T>& rhs)
{
    complex<T> result = lhs;
    return result -= rhs;
}

template <typename T>
SCALARS_HOST_DEVICE constexpr complex<T> operator-(const complex<T>& lhs, const T& rhs)
{
    complex<T> result = lhs;
    return result -= rhs;
}

template <typename T>
SCALARS_HOST_DEVICE constexpr complex<T> operator-(const T& lhs, const complex<T>& rhs)
{
    complex<T> result = -rhs;
    return result += lhs;
}

template <typename T>
SCALARS_HOST_DEVICE constexpr complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs)
{
    complex<T> result = lhs;
    return result *= rhs;
}

template <typename T>
SCALARS_HOST_DEVICE constexpr complex<T> operator*(const complex<T>& lhs, const T& rhs)
{
    complex<T> result = lhs;
    return result *= rhs;
}

template <typename T>
SCALARS_HOST_DEVICE constexpr complex<T> operator*(const T& lhs, const complex<T>& rhs)
{
    complex<T> result = rhs;
    return result *= lhs;
}

template <typename T>
SCALARS_HOST_DEVICE constexpr complex<T> operator/(const complex<T>& lhs, const complex<T>& rhs)
{
    complex<T> result = lhs;
    return result /= rhs;
}

template <typename T>
SCALARS_HOST_DEVICE constexpr complex<T> operator/(const complex<T>& lhs, const T& rhs)
{
    complex<T> result = lhs;
    return result /= rhs;
}

template <typename T>
SCALARS_HOST_DEVICE constexpr complex<T> operator/(const T& lhs, const complex<T>& rhs)
{
    complex<T> result(lhs, T());
    return result /= rhs;
}

// Define operators between integral scalars and scalars::complex. std::complex does
// not support this when T is a floating-point number. This is useful because it
// saves a lot of "static_cast" when operate a complex and an integer. This
// makes the code both less verbose and potentially more efficient.
#define COMPLEX_INTEGER_OP_TEMPLATE_CONDITION                                                      \
    typename std::enable_if_t<std::is_floating_point_v<fT> && std::is_integral_v<iT>, int> = 0

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
SCALARS_HOST_DEVICE constexpr scalars::complex<fT> operator+(
    const scalars::complex<fT>& a, const iT& b)
{
    return a + static_cast<fT>(b);
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
SCALARS_HOST_DEVICE constexpr scalars::complex<fT> operator+(
    const iT& a, const scalars::complex<fT>& b)
{
    return static_cast<fT>(a) + b;
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
SCALARS_HOST_DEVICE constexpr scalars::complex<fT> operator-(
    const scalars::complex<fT>& a, const iT& b)
{
    return a - static_cast<fT>(b);
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
SCALARS_HOST_DEVICE constexpr scalars::complex<fT> operator-(
    const iT& a, const scalars::complex<fT>& b)
{
    return static_cast<fT>(a) - b;
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
SCALARS_HOST_DEVICE constexpr scalars::complex<fT> operator*(
    const scalars::complex<fT>& a, const iT& b)
{
    return a * static_cast<fT>(b);
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
SCALARS_HOST_DEVICE constexpr scalars::complex<fT> operator*(
    const iT& a, const scalars::complex<fT>& b)
{
    return static_cast<fT>(a) * b;
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
SCALARS_HOST_DEVICE constexpr scalars::complex<fT> operator/(
    const scalars::complex<fT>& a, const iT& b)
{
    return a / static_cast<fT>(b);
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
SCALARS_HOST_DEVICE constexpr scalars::complex<fT> operator/(
    const iT& a, const scalars::complex<fT>& b)
{
    return static_cast<fT>(a) / b;
}

#undef COMPLEX_INTEGER_OP_TEMPLATE_CONDITION

template <typename T>
SCALARS_HOST_DEVICE constexpr bool operator==(const complex<T>& lhs, const complex<T>& rhs)
{
    return (lhs.real() == rhs.real()) && (lhs.imag() == rhs.imag());
}

template <typename T>
SCALARS_HOST_DEVICE constexpr bool operator==(const complex<T>& lhs, const T& rhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <typename T>
SCALARS_HOST_DEVICE constexpr bool operator==(const T& lhs, const complex<T>& rhs)
{
    return (lhs == rhs.real()) && (T() == rhs.imag());
}

template <typename T>
SCALARS_HOST_DEVICE constexpr bool operator!=(const complex<T>& lhs, const complex<T>& rhs)
{
    return !(lhs == rhs);
}

template <typename T>
SCALARS_HOST_DEVICE constexpr bool operator!=(const complex<T>& lhs, const T& rhs)
{
    return !(lhs == rhs);
}

template <typename T>
SCALARS_HOST_DEVICE constexpr bool operator!=(const T& lhs, const complex<T>& rhs)
{
    return !(lhs == rhs);
}

template <typename T, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os, const complex<T>& x)
{
    return (os << static_cast<std::complex<T>>(x));
}

template <typename T, typename CharT, typename Traits>
std::basic_istream<CharT, Traits>& operator>>(std::basic_istream<CharT, Traits>& is, complex<T>& x)
{
    std::complex<T> tmp;
    is >> tmp;
    x = tmp;
    return is;
}

template <typename T> SCALARS_HOST_DEVICE complex<T> polar(const T& r, const T& theta = T())
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return static_cast<complex<T>>(thrust::polar(r, theta));
#else
    // std::polar() requires r >= 0, so spell out the explicit implementation to
    // avoid a branch.
    return complex<T>(r * std::cos(theta), r * std::sin(theta));
#endif
}

namespace detail_complex
{
// Shared implementation of complex<half> and complex<bfloat16>. Arithmetic is
// computed in float (division in complex<float>) and narrowed once, as PyTorch
// does for c10::complex<c10::Half>, rather than instantiating the generic
// complex<T> with T = half/bfloat16, which would round every component
// operation in reduced precision and has no reduced-precision division path.
// Keeping both types on this one template stops their interfaces drifting
// apart.
template <class T> struct reduced_complex_base
{
    // Zero-initialized like the generic complex<T>: `complex<T> acc; acc += x;`
    // must start from 0.
    T real_{};
    T imag_{};

    reduced_complex_base() = default;
    // T's constructor is not constexpr, so these constructors can't be either.
    SCALARS_HOST_DEVICE explicit inline reduced_complex_base(const T& real, const T& imag)
        : real_(real), imag_(imag)
    {
    }
    // Implicit real -> complex conversion with a zero imaginary part, as the
    // generic complex<T> provides. Constrained to exactly T: an unconstrained
    // constructor from const T& would make complex<T>(1.0f) ambiguous with the
    // complex<float> converting constructor below.
    template <typename U, std::enable_if_t<std::is_same_v<U, T>, int> = 0>
    // cppcheck-suppress noExplicitConstructor -- deliberate, matches std::complex.
    SCALARS_HOST_DEVICE inline reduced_complex_base(const U& real) : real_(real), imag_()
    {
    }
    // cppcheck-suppress noExplicitConstructor -- deliberate, matches T's own implicit
    // narrowing conversion from float used throughout the reduced-precision types.
    SCALARS_HOST_DEVICE inline reduced_complex_base(const scalars::complex<float>& value)
        : real_(value.real()), imag_(value.imag())
    {
    }

    inline SCALARS_HOST_DEVICE operator scalars::complex<float>() const { return {real_, imag_}; }

    constexpr SCALARS_HOST_DEVICE T    real() const { return real_; }
    constexpr SCALARS_HOST_DEVICE void real(T value) { real_ = value; }
    constexpr SCALARS_HOST_DEVICE T    imag() const { return imag_; }
    constexpr SCALARS_HOST_DEVICE void imag(T value) { imag_ = value; }

    SCALARS_HOST_DEVICE complex<T>& operator+=(const complex<T>& other)
    {
        real_ = static_cast<float>(real_) + static_cast<float>(other.real_);
        imag_ = static_cast<float>(imag_) + static_cast<float>(other.imag_);
        return self();
    }

    SCALARS_HOST_DEVICE complex<T>& operator-=(const complex<T>& other)
    {
        real_ = static_cast<float>(real_) - static_cast<float>(other.real_);
        imag_ = static_cast<float>(imag_) - static_cast<float>(other.imag_);
        return self();
    }

    SCALARS_HOST_DEVICE complex<T>& operator*=(const complex<T>& other)
    {
        const auto a = static_cast<float>(real_);
        const auto b = static_cast<float>(imag_);
        const auto c = static_cast<float>(other.real_);
        const auto d = static_cast<float>(other.imag_);
        real_        = a * c - b * d;
        imag_        = a * d + b * c;
        return self();
    }

    // Division in complex<float> avoids the signed-zero and overflow pitfalls
    // of dividing directly in reduced precision.
    SCALARS_HOST_DEVICE complex<T>& operator/=(const complex<T>& other)
    {
        const scalars::complex<float> result =
            scalars::complex<float>(self()) / scalars::complex<float>(other);
        real_ = result.real();
        imag_ = result.imag();
        return self();
    }

    // Scalar compound operators, as the generic complex<T> provides; the
    // non-member complex<T> (op) T operators are implemented through these.
    SCALARS_HOST_DEVICE complex<T>& operator+=(const T& re)
    {
        real_ = static_cast<float>(real_) + static_cast<float>(re);
        return self();
    }

    SCALARS_HOST_DEVICE complex<T>& operator-=(const T& re)
    {
        real_ = static_cast<float>(real_) - static_cast<float>(re);
        return self();
    }

    SCALARS_HOST_DEVICE complex<T>& operator*=(const T& re)
    {
        real_ = static_cast<float>(real_) * static_cast<float>(re);
        imag_ = static_cast<float>(imag_) * static_cast<float>(re);
        return self();
    }

    SCALARS_HOST_DEVICE complex<T>& operator/=(const T& re)
    {
        real_ = static_cast<float>(real_) / static_cast<float>(re);
        imag_ = static_cast<float>(imag_) / static_cast<float>(re);
        return self();
    }

private:
    SCALARS_HOST_DEVICE complex<T>& self() { return static_cast<complex<T>&>(*this); }
    SCALARS_HOST_DEVICE const complex<T>& self() const
    {
        return static_cast<const complex<T>&>(*this);
    }
};
}  // namespace detail_complex

template <> struct alignas(4) complex<half> : detail_complex::reduced_complex_base<half>
{
    using reduced_complex_base::reduced_complex_base;
};

template <> struct alignas(4) complex<bfloat16> : detail_complex::reduced_complex_base<bfloat16>
{
    using reduced_complex_base::reduced_complex_base;
};

}  // namespace scalars

SCALARS_CLANG_DIAGNOSTIC_POP()

// std functions
//
// The implementation of these functions also follow the design of C++20

namespace std
{
// NOLINTBEGIN(bugprone-std-namespace-modification) -- deliberate, mirrors
// PyTorch's c10/util/complex.h: these overloads let generic code call
// std::abs/std::real/etc. uniformly on scalars::complex<T>.

template <typename T> SCALARS_HOST_DEVICE constexpr T real(const scalars::complex<T>& z)
{
    return z.real();
}

template <typename T> SCALARS_HOST_DEVICE constexpr T imag(const scalars::complex<T>& z)
{
    return z.imag();
}

template <typename T> SCALARS_HOST_DEVICE T abs(const scalars::complex<T>& z)
{
#if defined(__CUDACC__) || defined(__HIPCC__)
    return thrust::abs(static_cast<thrust::complex<T>>(z));
#else
    return std::abs(static_cast<std::complex<T>>(z));
#endif
}

#if defined(SCALARS_USE_HIP)
#define ROCm_Bug(x)
#else
#define ROCm_Bug(x) x
#endif

template <typename T> SCALARS_HOST_DEVICE T arg(const scalars::complex<T>& z)
{
    return ROCm_Bug(std)::atan2(std::imag(z), std::real(z));
}

#undef ROCm_Bug

template <typename T> SCALARS_HOST_DEVICE constexpr T norm(const scalars::complex<T>& z)
{
    return z.real() * z.real() + z.imag() * z.imag();
}

// For std::conj, there are other versions of it:
//   constexpr std::complex<float> conj( float z );
//   template< class DoubleOrInteger >
//   constexpr std::complex<double> conj( DoubleOrInteger z );
//   constexpr std::complex<long double> conj( long double z );
// These are not implemented
// TODO: implement them as scalars::conj
template <typename T>
SCALARS_HOST_DEVICE constexpr scalars::complex<T> conj(const scalars::complex<T>& z)
{
    return scalars::complex<T>(z.real(), -z.imag());
}

// Thrust does not have complex --> complex version of thrust::proj,
// so this function is not implemented here.
// TODO: implement it by ourselves

// There is no c10 version of std::polar, because std::polar always
// returns std::complex. Use scalars::polar instead;

// NOLINTEND(bugprone-std-namespace-modification)
}  // namespace std

#define SCALARS_INTERNAL_INCLUDE_COMPLEX_REMAINING_H
// math functions are included in a separate file
#include "math/complex_math.h"  // IWYU pragma: keep
// utilities for complex types
#include "math/complex_utils.h"  // IWYU pragma: keep
#undef SCALARS_INTERNAL_INCLUDE_COMPLEX_REMAINING_H

namespace scalars
{
using complex_float  = scalars::complex<float>;
using complex_double = scalars::complex<double>;
using complex_half   = scalars::complex<half>;
using complex_bf16   = scalars::complex<bfloat16>;

// Qualified Scalars access to component operations already provided above.
using std::abs;
using std::arg;
using std::conj;
using std::imag;
using std::norm;
using std::real;

template <class T> inline bool isfinite(const complex<T>& z)
{
    return std::isfinite(z.real()) && std::isfinite(z.imag());
}

using scalars::complex_math::acos;
using scalars::complex_math::acosh;
using scalars::complex_math::asin;
using scalars::complex_math::asinh;
using scalars::complex_math::atan;
using scalars::complex_math::atanh;
using scalars::complex_math::cos;
using scalars::complex_math::cosh;
using scalars::complex_math::exp;
using scalars::complex_math::expm1;
using scalars::complex_math::log;
using scalars::complex_math::log10;
using scalars::complex_math::log1p;
using scalars::complex_math::log2;
using scalars::complex_math::pow;
using scalars::complex_math::sin;
using scalars::complex_math::sinh;
using scalars::complex_math::sqrt;
using scalars::complex_math::tan;
using scalars::complex_math::tanh;
}  // namespace scalars
