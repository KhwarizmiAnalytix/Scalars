#pragma once

#include "backend/cpu/real_simd.h"

// Select full-precision complex packets without pulling in reduced packets.
#if SCALARS_HAS_AVX512
#include "backend/cpu/avx512/complex/simd.h"
#elif SCALARS_HAS_AVX2 || SCALARS_HAS_AVX
#include "backend/cpu/avx/complex/simd.h"
#elif SCALARS_HAS_SSE
#include "backend/cpu/sse/complex/simd.h"
#elif SCALARS_HAS_SVE
#include "backend/cpu/sve/complex/simd.h"
#elif SCALARS_HAS_NEON
#include "backend/cpu/neon/complex/simd.h"
#endif
