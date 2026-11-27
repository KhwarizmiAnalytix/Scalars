#pragma once

#include "backend/simd_base.h"

// Real packets form the dependency layer below complex and reduced packets.
#if SCALARS_HAS_AVX512
#include "backend/cpu/avx512/double/simd.h"
#include "backend/cpu/avx512/float/simd.h"
#elif SCALARS_HAS_AVX2 || SCALARS_HAS_AVX
#include "backend/cpu/avx/double/simd.h"
#include "backend/cpu/avx/float/simd.h"
#elif SCALARS_HAS_SSE
#include "backend/cpu/sse/double/simd.h"
#include "backend/cpu/sse/float/simd.h"
#elif SCALARS_HAS_SVE
#include "backend/cpu/sve/double/simd.h"
#include "backend/cpu/sve/float/simd.h"
#elif SCALARS_HAS_NEON
#include "backend/cpu/neon/double/simd.h"
#include "backend/cpu/neon/float/simd.h"
#endif
