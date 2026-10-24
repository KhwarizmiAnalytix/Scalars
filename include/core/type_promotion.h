#pragma once
#include <core/scalar_traits.h>
#include <type_traits>
namespace kh::scalars
{
template <class A, class B> struct promote_real
{
    using type = std::conditional_t<std::is_same_v<A, double> || std::is_same_v<B, double>,
        double,
        std::conditional_t<std::is_same_v<A, float> || std::is_same_v<B, float> ||
                               !std::is_same_v<A, B>,
            float,
            A>>;
};
template <class A, class B> struct promote
{
    using type = typename promote_real<A, B>::type;
};
template <class A, class B> struct promote<complex<A>, complex<B>>
{
    using type = complex<typename promote_real<A, B>::type>;
};
template <class A, class B> struct promote<complex<A>, B>
{
    using type = complex<typename promote_real<A, B>::type>;
};
template <class A, class B> struct promote<A, complex<B>>
{
    using type = complex<typename promote_real<A, B>::type>;
};
template <class A, class B> using promote_t = typename promote<A, B>::type;
}  // namespace kh::scalars
