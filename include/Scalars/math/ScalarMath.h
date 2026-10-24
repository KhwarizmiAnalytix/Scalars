#pragma once
#include <Scalars/types/Half.h>
#include <Scalars/types/BFloat16.h>
#include <Scalars/math/ComplexMath.h>
#include <cmath>
namespace kh::scalars {
#define KH_REAL_MATH(NAME) \
 inline Half NAME(Half x) {return Half(std::NAME(float(x)));} \
 inline BFloat16 NAME(BFloat16 x) {return BFloat16(std::NAME(float(x)));}
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
}
