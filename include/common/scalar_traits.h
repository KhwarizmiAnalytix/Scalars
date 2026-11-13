#pragma once
#include "common/accumulate_type.h"
#include "types/bfloat16.h"
#include "types/complex.h"
#include <type_traits>
namespace scalars
{
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

// accumulation_type_t<T> is the CPU form of acc_type<T, /*is_cuda*/false>
// (common/accumulate_type.h), kept as a single-argument alias for existing
// callers. These used to be two independently maintained tables that
// disagreed for float on CPU: accumulation_type_t<float> was float, while
// acc_type<float, false> is double (matching PyTorch's at::acc_type, and the
// documented reason acc_type exists at all: avoid compounding rounding
// error in intermediate computations). There is deliberately no fallback to
// T for a type with no acc_type entry -- querying the accumulation type of
// something nobody has thought about should fail to compile, not silently
// claim "itself".
template <class T> struct accumulation_type
{
    using type = acc_type<T, /*is_cuda=*/false>;
};
template <class T> using accumulation_type_t = typename accumulation_type<T>::type;
}  // namespace scalars
