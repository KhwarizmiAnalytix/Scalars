#pragma once
#include "types/complex.h"
#include <cmath>
namespace scalars
{
#define SCALARS_REAL_MATH(NAME)                                                                     \
    inline half NAME(half x)                                                                       \
    {                                                                                               \
        return half(std::NAME(float(x)));                                                          \
    }                                                                                               \
    inline bfloat16 NAME(bfloat16 x)                                                                \
    {                                                                                               \
        return bfloat16(std::NAME(float(x)));                                                       \
    }
SCALARS_REAL_MATH(exp)
SCALARS_REAL_MATH(log)
SCALARS_REAL_MATH(sqrt)
SCALARS_REAL_MATH(sin)
SCALARS_REAL_MATH(cos)
SCALARS_REAL_MATH(tan)
SCALARS_REAL_MATH(sinh)
SCALARS_REAL_MATH(cosh)
SCALARS_REAL_MATH(tanh)
SCALARS_REAL_MATH(floor)
SCALARS_REAL_MATH(ceil)
SCALARS_REAL_MATH(round)
SCALARS_REAL_MATH(trunc)
#undef SCALARS_REAL_MATH
}  // namespace scalars
