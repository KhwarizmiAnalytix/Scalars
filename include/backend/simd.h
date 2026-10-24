#pragma once
#include <cstdint>

#include "scalars_macros.h"

#if SCALARS_VECTORIZED
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
/* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#if _MSC_VER <= 1900
#define _mm256_extract_epi64(X, Y) (_mm_extract_epi64(_mm256_extractf128_si256(X, Y >> 1), Y % 2))
#define _mm256_extract_epi32(X, Y) (_mm_extract_epi32(_mm256_extractf128_si256(X, Y >> 2), Y % 4))
#define _mm256_extract_epi16(X, Y) (_mm_extract_epi16(_mm256_extractf128_si256(X, Y >> 3), Y % 8))
#define _mm256_extract_epi8(X, Y) (_mm_extract_epi8(_mm256_extractf128_si256(X, Y >> 4), Y % 16))
#endif
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC-compatible compiler, targeting x86/x86-64 */
#include <x86intrin.h>
#elif defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && SCALARS_HAS_SVE
#include <arm_sve.h>
#elif defined(__GNUC__) && defined(__ARM_NEON__)
/* GCC-compatible compiler, targeting ARM with NEON */
#include <arm_neon.h>
#elif defined(__GNUC__) && defined(__IWMMXT__)
/* GCC-compatible compiler, targeting ARM with WMMX */
#include <mmintrin.h>
#elif (defined(__GNUC__) || defined(__xlC__)) && (defined(__VEC__) || defined(__ALTIVEC__))
/* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
#include <altivec.h>
#elif defined(__GNUC__) && defined(__SPE__)
/* GCC-compatible compiler, targeting PowerPC with SPE */
#include <spe.h>
#endif

#if __SCALARS_COMPILER_PGI__
using PREFETCH_PTR_TYPE = const void*;
#else
using PREFETCH_PTR_TYPE = const char*;
#endif
#endif

template <typename T>
struct simd
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

#else
// --- CPU host code: SIMD or scalar fallback ---
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
// No CPU SIMD tier selected (SCALARS_CPU_BACKEND=no) and not compiling the
// device pass: simd<T> is intentionally left without a specialization here.
// backend/gpu/{double,float}/simd.h is not a portable host fallback -- it
// calls CUDA/HIP device-runtime-only math (exp10, erfinv, rsqrtf, ...) that
// does not exist in the host C library, so it is only ever valid under
// SCALARS_ON_GPU_DEVICE above.

#endif  // SCALARS_ON_GPU_DEVICE
