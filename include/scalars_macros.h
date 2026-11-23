#pragma once

#include <cstddef>

// Library macros for the half, bfloat16, complex, and backend/ SIMD
// implementations. Each PyTorch/c10 macro used by those types is defined as
// SCALARS_YYY, and the original name is an alias of that definition.
// Compiler-provided macros (__host__, __device__, __CUDACC__, __HIPCC__,
// __CUDA_ARCH__, __HIP_DEVICE_COMPILE__) are left unchanged; build-controlled
// toolchain flags (USE_ROCM, TORCH_HIP_VERSION, CPU_CAPABILITY_AVX2,
// CPU_CAPABILITY_AVX512) are set by CMake under their SCALARS_ names
// (SCALARS_USE_HIP, SCALARS_HIP_VERSION, SCALARS_CPU_CAPABILITY_AVX2,
// SCALARS_CPU_CAPABILITY_AVX512 -- see cmake/simd.cmake, cmake/gpu_backend.cmake)
// and there is no longer a PyTorch-named alias.

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

// ISA consistency guard. SCALARS_HAS_* (defined by cmake/simd.cmake,
// scalars_apply_simd_compile_definitions) reflects the instruction-set
// baseline SCALARS_CPU_BACKEND selected when Scalars was configured, and the
// matching compiler ISA flags (-mavx2, -march=armv8.2-a, ...) are PUBLIC on
// the Scalars target -- an ordinary CMake consumer linking Scalars::Scalars
// has those flags applied to its own translation units too, so this never
// fires. It exists for a translation unit that reaches this header some
// other way (a different build system, or compile options overridden after
// fetching Scalars::Scalars's usage requirements): without it, calling SIMD
// code built for an instruction set this TU does not have crashes with an
// illegal instruction at runtime instead of failing here, at compile time.
// GPU device passes (nvcc/clang-CUDA __CUDA_ARCH__, hipcc
// __HIP_DEVICE_COMPILE__) are skipped: they target nvptx/amdgcn, never define
// the host ISA macros, and do not run host SIMD code. The host pass of the
// same .cu/.hip file still receives the ISA flags (-Xcompiler) and is checked.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
// Device pass: no host ISA check.
#elif defined(SCALARS_HAS_AVX2) && SCALARS_HAS_AVX2 && !defined(__AVX2__)
#error \
    "Scalars was configured with SCALARS_CPU_BACKEND=avx2, but this translation unit is compiled without AVX2 (-mavx2) enabled. Compile it with the same ISA flags as Scalars, or rebuild Scalars with a narrower SCALARS_CPU_BACKEND."
#elif defined(SCALARS_HAS_AVX512) && SCALARS_HAS_AVX512 && !defined(__AVX512F__)
#error \
    "Scalars was configured with SCALARS_CPU_BACKEND=avx512, but this translation unit is compiled without AVX-512 (-mavx512f) enabled. Compile it with the same ISA flags as Scalars, or rebuild Scalars with a narrower SCALARS_CPU_BACKEND."
#elif defined(SCALARS_HAS_AVX) && SCALARS_HAS_AVX && !defined(__AVX__)
#error \
    "Scalars was configured with SCALARS_CPU_BACKEND=avx, but this translation unit is compiled without AVX (-mavx) enabled. Compile it with the same ISA flags as Scalars, or rebuild Scalars with a narrower SCALARS_CPU_BACKEND."
#elif defined(SCALARS_HAS_SSE) && SCALARS_HAS_SSE && !defined(__SSE2__) && !defined(_M_X64) && \
    !defined(_M_AMD64)
#error \
    "Scalars was configured with SCALARS_CPU_BACKEND=sse, but this translation unit is compiled without SSE2 (-msse2) enabled. Compile it with the same ISA flags as Scalars, or rebuild Scalars with -DSCALARS_CPU_BACKEND=no."
#elif defined(SCALARS_HAS_NEON) && SCALARS_HAS_NEON && defined(__aarch64__) && \
    !defined(__ARM_NEON)
#error \
    "Scalars was configured with SCALARS_CPU_BACKEND=neon, but this translation unit is compiled without NEON enabled. Compile it with the same ISA flags as Scalars, or rebuild Scalars with -DSCALARS_CPU_BACKEND=no."
#elif defined(SCALARS_HAS_SVE) && SCALARS_HAS_SVE && defined(__aarch64__) && \
    !defined(__ARM_FEATURE_SVE)
#error \
    "Scalars was configured with SCALARS_CPU_BACKEND=sve, but this translation unit is compiled without SVE enabled. Compile it with the same ISA flags as Scalars, or rebuild Scalars with -DSCALARS_CPU_BACKEND=no."
#endif

#if defined(__clang__)
#define SCALARS_UBSAN_IGNORE_FLOAT_DIVIDE_BY_ZERO \
    __attribute__((no_sanitize("float-divide-by-zero")))
#define SCALARS_UBSAN_IGNORE_UNDEFINED __attribute__((no_sanitize("undefined")))
#define SCALARS_UBSAN_IGNORE_SIGNED_INT_OVERFLOW \
    __attribute__((no_sanitize("signed-integer-overflow")))
#define SCALARS_UBSAN_IGNORE_POINTER_OVERFLOW __attribute__((no_sanitize("pointer-overflow")))
#define SCALARS_UBSAN_IGNORE_FUNCTION __attribute__((no_sanitize("function")))
#define SCALARS_UBSAN_IGNORE_FLOAT_CAST_OVERFLOW \
    __attribute__((no_sanitize("float-cast-overflow")))
#else
#define SCALARS_UBSAN_IGNORE_FLOAT_DIVIDE_BY_ZERO
#define SCALARS_UBSAN_IGNORE_UNDEFINED
#define SCALARS_UBSAN_IGNORE_SIGNED_INT_OVERFLOW
#define SCALARS_UBSAN_IGNORE_POINTER_OVERFLOW
#define SCALARS_UBSAN_IGNORE_FUNCTION
#define SCALARS_UBSAN_IGNORE_FLOAT_CAST_OVERFLOW
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define SCALARS_HOST_DEVICE __host__ __device__
#define SCALARS_DEVICE __device__
#define SCALARS_HOST __host__
#else
#define SCALARS_HOST_DEVICE
#define SCALARS_DEVICE
#define SCALARS_HOST
#endif


#define SCALARS_PRAGMA_IMPL(string) _Pragma(#string)
#define SCALARS_PRAGMA(string) SCALARS_PRAGMA_IMPL(string)

#ifdef __clang__
#define SCALARS_CLANG_DIAGNOSTIC_PUSH() _Pragma("clang diagnostic push")
#define SCALARS_CLANG_DIAGNOSTIC_POP() _Pragma("clang diagnostic pop")
#define SCALARS_CLANG_DIAGNOSTIC_IGNORE(flag) SCALARS_PRAGMA(clang diagnostic ignored flag)
#define SCALARS_CLANG_HAS_WARNING(flag) __has_warning(flag)
#else
#define SCALARS_CLANG_DIAGNOSTIC_PUSH()
#define SCALARS_CLANG_DIAGNOSTIC_POP()
#define SCALARS_CLANG_DIAGNOSTIC_IGNORE(flag)
#define SCALARS_CLANG_HAS_WARNING(flag) 0
#endif

#define SCALARS_HIDDEN_NS_GET_MACRO(_1, _2, _3, NAME, ...) NAME
#define SCALARS_HIDDEN_NS_1(n1) namespace n1 __attribute__((visibility("hidden"))) {
#define SCALARS_HIDDEN_NS_2(n1, n2) \
    namespace n1                    \
    {                               \
    namespace n2 __attribute__((visibility("hidden"))) {
#define SCALARS_HIDDEN_NS_3(n1, n2, n3) \
    namespace n1::n2                    \
    {                                   \
    namespace n3 __attribute__((visibility("hidden"))) {
#define SCALARS_HIDDEN_NS_END_1(n1) }
#define SCALARS_HIDDEN_NS_END_N(n1, ...) \
    }                                    \
    }
#define SCALARS_EXPAND(...) __VA_ARGS__
#define SCALARS_JOIN_GET_MACRO(_1, _2, _3, NAME, ...) NAME
#define SCALARS_JOIN_NS1(a) a
#define SCALARS_JOIN_NS2(a, b) a::b
#define SCALARS_JOIN_NS3(a, b, c) a::b::c

#if defined(__GNUG__) && !defined(_WIN32)
#define SCALARS_HIDDEN_NAMESPACE_BEGIN(...) \
    SCALARS_HIDDEN_NS_GET_MACRO(__VA_ARGS__, SCALARS_HIDDEN_NS_3, SCALARS_HIDDEN_NS_2, SCALARS_HIDDEN_NS_1)(__VA_ARGS__)
#define SCALARS_HIDDEN_NAMESPACE_END(...) \
    SCALARS_HIDDEN_NS_GET_MACRO(__VA_ARGS__, SCALARS_HIDDEN_NS_END_N, SCALARS_HIDDEN_NS_END_N, SCALARS_HIDDEN_NS_END_1)(__VA_ARGS__)
#else
#define SCALARS_HIDDEN_NAMESPACE_BEGIN(...) \
    namespace SCALARS_EXPAND(SCALARS_JOIN_GET_MACRO(__VA_ARGS__, SCALARS_JOIN_NS3, SCALARS_JOIN_NS2, SCALARS_JOIN_NS1)(__VA_ARGS__)) {
#define SCALARS_HIDDEN_NAMESPACE_END(...) }
#endif

#define HIDDEN_NAMESPACE_BEGIN(...) SCALARS_HIDDEN_NAMESPACE_BEGIN(__VA_ARGS__)
#define HIDDEN_NAMESPACE_END(...) SCALARS_HIDDEN_NAMESPACE_END(__VA_ARGS__)

// ---------------------------------------------------------------------------
// backend/ (ported SIMD CPU/GPU tree) and math/scalar_helper_functions.h
// macro vocabulary. Mirrors Vectorization's common/vectorization_macros.h,
// minus its Logging::Logging / Memory::Memory integration -- Scalars has no
// such dependency and these macros stay self-contained.
// ---------------------------------------------------------------------------

#ifndef SCALARS_FORCE_INLINE
#if defined(_MSC_VER)
#define SCALARS_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#define SCALARS_FORCE_INLINE __attribute__((always_inline)) inline
#else
#define SCALARS_FORCE_INLINE inline
#endif
#endif

#if defined(_MSC_VER)
#define SCALARS_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#define SCALARS_NOINLINE __attribute__((noinline))
#else
#define SCALARS_NOINLINE
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define SCALARS_VECTORCALL __vectorcall
#else
#define SCALARS_VECTORCALL
#endif

// True when compiling the device-side pass (GPU kernel body). Safe to use
// inside SCALARS_HOST_DEVICE functions to select device-only paths.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
// NOLINTNEXTLINE(modernize-macro-to-enum)
#define SCALARS_ON_GPU_DEVICE 1
#else
// NOLINTNEXTLINE(modernize-macro-to-enum)
#define SCALARS_ON_GPU_DEVICE 0
#endif

#ifdef NDEBUG
#define SCALARS_SIMD_RETURN_TYPE SCALARS_FORCE_INLINE SCALARS_HOST_DEVICE static void SCALARS_VECTORCALL
#else
#define SCALARS_SIMD_RETURN_TYPE SCALARS_HOST_DEVICE static void
#endif

#ifdef NDEBUG
#define SCALARS_SIMD_METHOD SCALARS_FORCE_INLINE SCALARS_HOST_DEVICE static
#else
#define SCALARS_SIMD_METHOD SCALARS_HOST_DEVICE static
#endif

#ifdef NDEBUG
#define SCALARS_FUNCTION_ATTRIBUTE SCALARS_FORCE_INLINE SCALARS_HOST_DEVICE
#else
#define SCALARS_FUNCTION_ATTRIBUTE SCALARS_HOST_DEVICE
#endif

#ifdef NDEBUG
#define SCALARS_HOST_FUNCTION_ATTRIBUTE SCALARS_FORCE_INLINE SCALARS_HOST
#else
#define SCALARS_HOST_FUNCTION_ATTRIBUTE SCALARS_HOST
#endif

#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#define SCALARS_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define SCALARS_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define SCALARS_LIKELY(expr) (expr)
#define SCALARS_UNLIKELY(expr) (expr)
#endif

#if defined(_MSC_VER)
#define SCALARS_ALIGN(alignment) __declspec(align(alignment))
#elif defined(__GNUC__) || defined(__clang__) || defined(__ICC) || defined(__INTEL_COMPILER)
#define SCALARS_ALIGN(alignment) __attribute__((aligned(alignment)))
#else
#define SCALARS_ALIGN(alignment)
#endif

// alignas() for an inline static data member defined in its class. nvcc's EDG
// front end (seen with CUDA 13.2 and cl.exe as host compiler) rejects that
// combination: "alignment attribute must also appear on definition". The
// members using this have SIMD vector types (__m256d, __m512i, ...), whose
// type alignment already equals the requested value, so dropping the
// redundant specifier under nvcc keeps the same storage alignment.
#if defined(__NVCC__) && !defined(__clang__)
#define SCALARS_ALIGNAS_INLINE_STATIC(alignment)
#else
#define SCALARS_ALIGNAS_INLINE_STATIC(alignment) alignas(alignment)
#endif

inline constexpr std::size_t SCALARS_ALIGNMENT = 64;

#if __cplusplus >= 201703L
#define SCALARS_NODISCARD [[nodiscard]]
#define SCALARS_UNUSED [[maybe_unused]]
#else
#define SCALARS_NODISCARD
#define SCALARS_UNUSED
#endif

// Manual-unroll factor for the vectorized hot loops in the ported backend/
// SIMD kernels. Falls back to 1 (no manual unroll) for any translation unit
// built outside the CMake target that doesn't get the -D from the build
// system. See cmake/simd.cmake.
#ifndef SCALARS_PACKET_SIZE
#define SCALARS_PACKET_SIZE 1
#endif
static_assert(SCALARS_PACKET_SIZE >= 1, "SCALARS_PACKET_SIZE must be >= 1");
