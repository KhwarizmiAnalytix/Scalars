#pragma once

#include "common/intrinsics.h"

#if !defined(__ARM_FEATURE_SVE_BITS)
#error "SCALARS_HAS_SVE: compile with SVE enabled and a fixed vector length, e.g. \
-march=armv8-a+sve (or armv9-a) and -msve-vector-bits=128"
#endif

#if __ARM_FEATURE_SVE_BITS != 128
#error                                                                                             \
    "SCALARS_HAS_SVE: only -msve-vector-bits=128 is supported; use SCALARS_CPU_BACKEND=neon for variable-length or wider builds"
#endif

#include "../../neon/complex/simd.h"
