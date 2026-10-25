#pragma once
#include <cstdint>

#include "scalars_macros.h"

#if SCALARS_VECTORIZED
#include "common/intrinsics.h"

#if __SCALARS_COMPILER_PGI__
using prefetch_ptr_type = const void*;
#else
using prefetch_ptr_type = const char*;
#endif
#endif

template <typename T> struct simd
{
};

// Backend selection strategy:
//
//  GPU device pass (nvcc/hipcc compiling __device__ code):
//    Always use the scalar GPU backend so AVX intrinsics (unavailable on device)
//    are never included in device translation units.
//
//  CPU host pass:
//    Use the selected CPU SIMD backend (SCALARS_CPU_BACKEND != "no"). When no
//    CPU tier is selected, simd<T> has no specialization here -- the GPU
//    scalar backend below is not a valid host fallback (it calls CUDA/HIP
//    device-runtime-only math), so code using simd<T> requires a CPU SIMD
//    backend or an actual GPU device compile.

#if SCALARS_ON_GPU_DEVICE
// --- GPU device code: scalar simd<T> ---
// Shared by CUDA and HIP (identical scalar semantics); SCALARS_ON_GPU_DEVICE
// being set already guarantees nvcc or hipcc is doing the device pass.
#include "backend/gpu/double/simd.h"
#include "backend/gpu/float/simd.h"
#include "backend/gpu/reduced/simd.h"
#include "backend/gpu/complex/simd.h"

#else
// --- CPU host code: SIMD or scalar fallback ---
#if SCALARS_HAS_SSE || SCALARS_HAS_AVX || SCALARS_HAS_AVX2 || SCALARS_HAS_AVX512
#include "backend/cpu/x86/math_fallback.h"
#endif
#if SCALARS_HAS_AVX512
#include "backend/cpu/avx512/double/simd.h"
#include "backend/cpu/avx512/float/simd.h"

#include "backend/cpu/avx512/complex/simd.h"
#elif SCALARS_HAS_AVX2 || SCALARS_HAS_AVX
#include "backend/cpu/avx/double/simd.h"
#include "backend/cpu/avx/float/simd.h"

#include "backend/cpu/avx/complex/simd.h"
#elif SCALARS_HAS_SSE
#include "backend/cpu/sse/double/simd.h"
#include "backend/cpu/sse/float/simd.h"

#include "backend/cpu/sse/complex/simd.h"
#elif SCALARS_HAS_SVE
#include "backend/cpu/sve/double/simd.h"
#include "backend/cpu/sve/float/simd.h"

#include "backend/cpu/sve/complex/simd.h"
#elif SCALARS_HAS_NEON
#include "backend/cpu/neon/double/simd.h"
#include "backend/cpu/neon/float/simd.h"

#include "backend/cpu/neon/complex/simd.h"
#endif

// Reduced packets share the selected float backend's math and use explicit
// architecture-specific conversions. They are CPU-only, including SVE's
// existing fixed-width NEON compatibility path.
#if SCALARS_VECTORIZED
#include "backend/cpu/reduced/simd.h"
#include "backend/cpu/reduced/complex_simd.h"
#endif
// No CPU SIMD tier selected (SCALARS_CPU_BACKEND=no) and not compiling the
// device pass: simd<T> is intentionally left without a specialization here.
// backend/gpu/{double,float}/simd.h is not a portable host fallback -- it
// calls CUDA/HIP device-runtime-only math (exp10, erfinv, rsqrtf, ...) that
// does not exist in the host C library, so it is only ever valid under
// SCALARS_ON_GPU_DEVICE above.

#endif  // SCALARS_ON_GPU_DEVICE
