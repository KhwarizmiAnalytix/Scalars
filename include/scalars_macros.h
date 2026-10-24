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
