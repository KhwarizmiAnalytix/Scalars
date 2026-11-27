#pragma once
#include "common/scalar_traits.h"
#include <type_traits>
namespace scalars
{
template <class A, class B> struct promote_real
{
    // Confirmed bug: long double used to fall through to the double/float
    // tiers below (e.g. promote_real<long double, double> gave double, and
    // promote_real<long double, float> gave float), silently narrowing it
    // away instead of it dominating, as the usual arithmetic conversions do.
    using type =
        std::conditional_t<std::is_same_v<A, long double> || std::is_same_v<B, long double>,
            long double,
            std::conditional_t<std::is_same_v<A, double> || std::is_same_v<B, double>,
                double,
                std::conditional_t<std::is_same_v<A, float> || std::is_same_v<B, float> ||
                                       !std::is_same_v<A, B>,
                    float,
                    A>>>;
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
}  // namespace scalars
