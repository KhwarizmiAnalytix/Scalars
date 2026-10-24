#pragma once
#include "types/complex.h"
#include <cmath>
namespace scalars
{
#define KH_REAL_MATH(NAME)                                                                         \
    inline half NAME(half x)                                                                       \
    {                                                                                              \
        return half(std::NAME(float(x)));                                                          \
    }                                                                                              \
    inline bfloat16 NAME(bfloat16 x)                                                               \
    {                                                                                              \
        return bfloat16(std::NAME(float(x)));                                                      \
    }
KH_REAL_MATH(exp)
KH_REAL_MATH(log)
KH_REAL_MATH(sqrt)
KH_REAL_MATH(sin)
KH_REAL_MATH(cos)
KH_REAL_MATH(tan)
KH_REAL_MATH(sinh)
KH_REAL_MATH(cosh)
KH_REAL_MATH(tanh)
KH_REAL_MATH(floor)
KH_REAL_MATH(ceil)
KH_REAL_MATH(round)
KH_REAL_MATH(trunc)
#undef KH_REAL_MATH
}  // namespace scalars
