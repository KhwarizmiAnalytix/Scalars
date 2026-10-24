#pragma once
#include <type_traits>
#include <types/complex.h>
namespace kh::scalars
{
template <class T> struct is_complex : std::false_type
{
};
template <class T> struct is_complex<complex<T>> : std::true_type
{
};
template <class T> inline constexpr bool is_complex_v = is_complex<T>::value;
template <class T> struct real_type
{
    using type = T;
};
template <class T> struct real_type<complex<T>>
{
    using type = T;
};
template <class T> using real_type_t = typename real_type<T>::type;
template <class T> struct accumulation_type
{
    using type = T;
};
template <> struct accumulation_type<half>
{
    using type = float;
};
template <> struct accumulation_type<bfloat16>
{
    using type = float;
};
template <class T> struct accumulation_type<complex<T>>
{
    using type = complex<typename accumulation_type<T>::type>;
};
template <class T> using accumulation_type_t = typename accumulation_type<T>::type;
}  // namespace kh::scalars
