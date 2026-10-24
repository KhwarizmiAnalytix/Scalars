#pragma once
#include "common/scalar_traits.h"
#include <type_traits>

namespace scalars
{
template <class To, class From> To scalar_cast(const From& v)
{
    if constexpr (is_complex_v<From> && is_complex_v<To>)
    {
        return To(typename To::value_type(v.real()), typename To::value_type(v.imag()));
    }
    else if constexpr (is_complex_v<To>)
    {
        return To(typename To::value_type(v));
    }
    else if constexpr (is_complex_v<From>)
    {
        return To(v.real());  // Explicitly discard imaginary component, consistent with
                              // PyTorch-style cast.
    }
    else
    {
        return static_cast<To>(v);
    }
}
}  // namespace scalars
