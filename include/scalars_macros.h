#pragma once

// Library macros for the half, bfloat16, and complex implementations.
// Each PyTorch/c10 macro used by those types is defined as SCALARS_YYY, and
// the original name is an alias of that definition. Compiler and toolchain
// macros (__host__, __device__, __CUDACC__, USE_ROCM, TORCH_HIP_VERSION,
// CPU_CAPABILITY_*) are left unchanged.

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

// These names are the attribute spellings used by the type implementations.
#define __ubsan_ignore_float_divide_by_zero__ SCALARS_UBSAN_IGNORE_FLOAT_DIVIDE_BY_ZERO
#define __ubsan_ignore_undefined__ SCALARS_UBSAN_IGNORE_UNDEFINED
#define __ubsan_ignore_signed_int_overflow__ SCALARS_UBSAN_IGNORE_SIGNED_INT_OVERFLOW
#define __ubsan_ignore_pointer_overflow__ SCALARS_UBSAN_IGNORE_POINTER_OVERFLOW
#define __ubsan_ignore_function__ SCALARS_UBSAN_IGNORE_FUNCTION
#define __ubsan_ignore_float_cast_overflow__ SCALARS_UBSAN_IGNORE_FLOAT_CAST_OVERFLOW

#if defined(__CUDACC__) || defined(__HIPCC__)
#define SCALARS_HOST_DEVICE __host__ __device__
#define SCALARS_DEVICE __device__
#define SCALARS_HOST __host__
#else
#define SCALARS_HOST_DEVICE
#define SCALARS_DEVICE
#define SCALARS_HOST
#endif


#define _SCALARS_PRAGMA_IMPL(string) _Pragma(#string)
#define _SCALARS_PRAGMA(string) _SCALARS_PRAGMA_IMPL(string)

#ifdef __clang__
#define SCALARS_CLANG_DIAGNOSTIC_PUSH() _Pragma("clang diagnostic push")
#define SCALARS_CLANG_DIAGNOSTIC_POP() _Pragma("clang diagnostic pop")
#define SCALARS_CLANG_DIAGNOSTIC_IGNORE(flag) _SCALARS_PRAGMA(clang diagnostic ignored flag)
#define SCALARS_CLANG_HAS_WARNING(flag) __has_warning(flag)
#else
#define SCALARS_CLANG_DIAGNOSTIC_PUSH()
#define SCALARS_CLANG_DIAGNOSTIC_POP()
#define SCALARS_CLANG_DIAGNOSTIC_IGNORE(flag)
#define SCALARS_CLANG_HAS_WARNING(flag) 0
#endif

#define _SCALARS_HIDDEN_NS_GET_MACRO(_1, _2, _3, NAME, ...) NAME
#define _SCALARS_HIDDEN_NS_1(n1) namespace n1 __attribute__((visibility("hidden"))) {
#define _SCALARS_HIDDEN_NS_2(n1, n2) \
    namespace n1                     \
    {                                \
    namespace n2 __attribute__((visibility("hidden"))) {
#define _SCALARS_HIDDEN_NS_3(n1, n2, n3) \
    namespace n1::n2                     \
    {                                    \
    namespace n3 __attribute__((visibility("hidden"))) {
#define _SCALARS_HIDDEN_NS_END_1(n1) }
#define _SCALARS_HIDDEN_NS_END_N(n1, ...) \
    }                                     \
    }
#define _SCALARS_EXPAND(...) __VA_ARGS__
#define _SCALARS_JOIN_GET_MACRO(_1, _2, _3, NAME, ...) NAME
#define _SCALARS_JOIN_NS1(a) a
#define _SCALARS_JOIN_NS2(a, b) a::b
#define _SCALARS_JOIN_NS3(a, b, c) a::b::c

#if defined(__GNUG__) && !defined(_WIN32)
#define SCALARS_HIDDEN_NAMESPACE_BEGIN(...) \
    _SCALARS_HIDDEN_NS_GET_MACRO(__VA_ARGS__, _SCALARS_HIDDEN_NS_3, _SCALARS_HIDDEN_NS_2, _SCALARS_HIDDEN_NS_1)(__VA_ARGS__)
#define SCALARS_HIDDEN_NAMESPACE_END(...) \
    _SCALARS_HIDDEN_NS_GET_MACRO(__VA_ARGS__, _SCALARS_HIDDEN_NS_END_N, _SCALARS_HIDDEN_NS_END_N, _SCALARS_HIDDEN_NS_END_1)(__VA_ARGS__)
#else
#define SCALARS_HIDDEN_NAMESPACE_BEGIN(...) \
    namespace _SCALARS_EXPAND(_SCALARS_JOIN_GET_MACRO(__VA_ARGS__, _SCALARS_JOIN_NS3, _SCALARS_JOIN_NS2, _SCALARS_JOIN_NS1)(__VA_ARGS__)) {
#define SCALARS_HIDDEN_NAMESPACE_END(...) }
#endif

#define HIDDEN_NAMESPACE_BEGIN(...) SCALARS_HIDDEN_NAMESPACE_BEGIN(__VA_ARGS__)
#define HIDDEN_NAMESPACE_END(...) SCALARS_HIDDEN_NAMESPACE_END(__VA_ARGS__)
