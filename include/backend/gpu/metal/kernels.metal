// Fixed, hand-written Metal kernels for fill, reductions, and metal_backend::dispatch()
// tests. Expression trees such as `x = y + a + 5*d` are emitted as one fused MSL kernel
// at runtime (see expressions_evaluator_metal.h) — there is no per-node lowering.
// Float, half and bf16 storage; MSL has no double type. One thread per element; buffer
// argument order is always (inputs..., out, n).
//
// This source is embedded into the Vectorization library as a C++ string at CMake
// configure time (see the "Metal GPU backend" block in CMakeLists.txt) and compiled at
// runtime via -[MTLDevice newLibraryWithSource:options:error:] — there is no .metallib
// build step.

#include <metal_stdlib>
using namespace metal;

// NaN propagation and deterministic signed-zero ties, matching reduced SIMD.
float reduced_min(float a, float b)
{
    if (isnan(a)) return a;
    if (isnan(b)) return b;
    if (a == 0.0f && b == 0.0f)
        return as_type<float>(as_type<uint>(a) | as_type<uint>(b));
    return min(a, b);
}
float reduced_max(float a, float b)
{
    if (isnan(a)) return a;
    if (isnan(b)) return b;
    if (a == 0.0f && b == 0.0f)
        return as_type<float>(as_type<uint>(a) & as_type<uint>(b));
    return max(a, b);
}

// -----------------------------------------------------------------------------------
// Fill
// -----------------------------------------------------------------------------------

kernel void fill_float(
    device float* out [[buffer(0)]],
    constant float& value [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = value;
}

// -----------------------------------------------------------------------------------
// Binary arithmetic
// -----------------------------------------------------------------------------------

kernel void add_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = a[tid] + b[tid];
}

kernel void sub_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = a[tid] - b[tid];
}

kernel void mul_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = a[tid] * b[tid];
}

kernel void div_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = a[tid] / b[tid];
}

// -----------------------------------------------------------------------------------
// Ternary: fused multiply-add
// -----------------------------------------------------------------------------------

kernel void fma_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* c [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = fma(a[tid], b[tid], c[tid]);
}

// -----------------------------------------------------------------------------------
// Unary math
// -----------------------------------------------------------------------------------

kernel void sqrt_float(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = sqrt(a[tid]);
}

kernel void exp_float(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = exp(a[tid]);
}

kernel void log_float(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = log(a[tid]);
}

kernel void sin_float(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = sin(a[tid]);
}

kernel void cos_float(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = cos(a[tid]);
}

kernel void tanh_float(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = tanh(a[tid]);
}

kernel void fabs_float(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = fabs(a[tid]);
}

kernel void neg_float(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = -a[tid];
}

// -----------------------------------------------------------------------------------
// Reductions: one threadgroup writes one partial (out[gid]). The host launcher
// walks the partials until a single value remains, so n is not limited to one
// threadgroup.
// -----------------------------------------------------------------------------------
kernel void reduce_sum_float(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    shared[lid] = (tid < n) ? in[tid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1)
    {
        if (lid < stride)
            shared[lid] += shared[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0)
        out[gid] = shared[0];
}

kernel void reduce_min_float(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    shared[lid] = (tid < n) ? in[tid] : INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1)
    {
        if (lid < stride)
            shared[lid] = reduced_min(shared[lid], shared[lid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0)
        out[gid] = shared[0];
}

kernel void reduce_max_float(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    shared[lid] = (tid < n) ? in[tid] : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1)
    {
        if (lid < stride)
            shared[lid] = reduced_max(shared[lid], shared[lid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0)
        out[gid] = shared[0];
}

// Reduced storage uses the same one-lane-per-thread contract as CUDA/HIP.
// ushort stores bf16 bits so this path also works without native MSL bfloat.
// Operation math and all reduction partials stay in FP32.
template <typename T> float widen_reduced(T value) { return float(value); }
template <> float widen_reduced<ushort>(ushort value)
{
    return as_type<float>(uint(value) << 16);
}
template <typename T> T narrow_reduced(float value) { return T(value); }
template <> ushort narrow_reduced<ushort>(float value)
{
    uint bits = as_type<uint>(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return ushort(0x7fc0);
    return ushort((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

#define SCALARS_METAL_UNARY(name, expr) \
    template <typename T> kernel void name##_reduced( \
        device const T* in [[buffer(0)]], device T* out [[buffer(1)]], \
        constant uint& n [[buffer(2)]], uint tid [[thread_position_in_grid]]) \
    { \
        if (tid < n) { const float x = widen_reduced(in[tid]); out[tid] = narrow_reduced<T>(expr); } \
    } \
    SCALARS_METAL_INSTANTIATE(name)
#define SCALARS_METAL_INSTANTIATE(name) \
    template [[host_name(#name "_half")]] kernel void name##_reduced<half>( \
        device const half*, device half*, constant uint&, uint); \
    template [[host_name(#name "_bfloat16")]] kernel void name##_reduced<ushort>( \
        device const ushort*, device ushort*, constant uint&, uint);
SCALARS_METAL_UNARY(sqrt, sqrt(x))
SCALARS_METAL_UNARY(exp, exp(x))
SCALARS_METAL_UNARY(log, log(x))
SCALARS_METAL_UNARY(sin, sin(x))
SCALARS_METAL_UNARY(cos, cos(x))
SCALARS_METAL_UNARY(tanh, tanh(x))
#undef SCALARS_METAL_UNARY
#undef SCALARS_METAL_INSTANTIATE

// Sign operations work on stored encodings, including signaling NaNs.
template <typename T> ushort reduced_bits(T x) { return as_type<ushort>(x); }
template <> ushort reduced_bits<ushort>(ushort x) { return x; }
template <typename T> T reduced_from_bits(ushort x) { return as_type<T>(x); }
template <> ushort reduced_from_bits<ushort>(ushort x) { return x; }
#define SCALARS_METAL_SIGN(name, expr) \
    template <typename T> kernel void name##_reduced( \
        device const T* in [[buffer(0)]], device T* out [[buffer(1)]], \
        constant uint& n [[buffer(2)]], uint tid [[thread_position_in_grid]]) \
    { \
        if (tid < n) { const ushort x = reduced_bits(in[tid]); out[tid] = reduced_from_bits<T>(expr); } \
    } \
    template [[host_name(#name "_half")]] kernel void name##_reduced<half>( \
        device const half*, device half*, constant uint&, uint); \
    template [[host_name(#name "_bfloat16")]] kernel void name##_reduced<ushort>( \
        device const ushort*, device ushort*, constant uint&, uint);
SCALARS_METAL_SIGN(fabs, ushort(x & 0x7fff))
SCALARS_METAL_SIGN(neg, ushort(x ^ 0x8000))
#undef SCALARS_METAL_SIGN

#define SCALARS_METAL_BINARY(name, expr) \
    template <typename T> kernel void name##_reduced( \
        device const T* a [[buffer(0)]], device const T* b [[buffer(1)]], \
        device T* out [[buffer(2)]], constant uint& n [[buffer(3)]], \
        uint tid [[thread_position_in_grid]]) \
    { \
        if (tid < n) { \
            const float x = widen_reduced(a[tid]), y = widen_reduced(b[tid]); \
            out[tid] = narrow_reduced<T>(expr); \
        } \
    } \
    template [[host_name(#name "_half")]] kernel void name##_reduced<half>( \
        device const half*, device const half*, device half*, constant uint&, uint); \
    template [[host_name(#name "_bfloat16")]] kernel void name##_reduced<ushort>( \
        device const ushort*, device const ushort*, device ushort*, constant uint&, uint);
SCALARS_METAL_BINARY(add, x + y)
SCALARS_METAL_BINARY(sub, x - y)
SCALARS_METAL_BINARY(mul, x * y)
SCALARS_METAL_BINARY(div, x / y)
SCALARS_METAL_BINARY(min, reduced_min(x, y))
SCALARS_METAL_BINARY(max, reduced_max(x, y))
#undef SCALARS_METAL_BINARY

template <typename T> kernel void fill_reduced(
    device T* out [[buffer(0)]], constant float& value [[buffer(1)]],
    constant uint& n [[buffer(2)]], uint tid [[thread_position_in_grid]])
{
    if (tid < n) out[tid] = narrow_reduced<T>(value);
}
template [[host_name("fill_half")]] kernel void fill_reduced<half>(
    device half*, constant float&, constant uint&, uint);
template [[host_name("fill_bfloat16")]] kernel void fill_reduced<ushort>(
    device ushort*, constant float&, constant uint&, uint);

template <typename T> kernel void fma_reduced(
    device const T* a [[buffer(0)]], device const T* b [[buffer(1)]],
    device const T* c [[buffer(2)]], device T* out [[buffer(3)]],
    constant uint& n [[buffer(4)]], uint tid [[thread_position_in_grid]])
{
    if (tid < n) out[tid] = narrow_reduced<T>(
        fma(widen_reduced(a[tid]), widen_reduced(b[tid]), widen_reduced(c[tid])));
}
template [[host_name("fma_half")]] kernel void fma_reduced<half>(
    device const half*, device const half*, device const half*, device half*, constant uint&, uint);
template [[host_name("fma_bfloat16")]] kernel void fma_reduced<ushort>(
    device const ushort*, device const ushort*, device const ushort*, device ushort*, constant uint&, uint);

#define SCALARS_METAL_REDUCE(name, identity, expr) \
    template <typename T> kernel void reduce_##name##_reduced( \
        device const T* in [[buffer(0)]], device float* out [[buffer(1)]], \
        constant uint& n [[buffer(2)]], uint tid [[thread_position_in_grid]], \
        uint lid [[thread_position_in_threadgroup]], uint gid [[threadgroup_position_in_grid]], \
        uint tg_size [[threads_per_threadgroup]], threadgroup float* shared [[threadgroup(0)]]) \
    { \
        shared[lid] = tid < n ? widen_reduced(in[tid]) : identity; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint stride = tg_size / 2; stride > 0; stride >>= 1) { \
            if (lid < stride) { \
                const float x = shared[lid], y = shared[lid + stride]; \
                shared[lid] = expr; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
        if (lid == 0) out[gid] = shared[0]; \
    } \
    template [[host_name("reduce_" #name "_half")]] kernel void reduce_##name##_reduced<half>( \
        device const half*, device float*, constant uint&, uint, uint, uint, uint, threadgroup float*); \
    template [[host_name("reduce_" #name "_bfloat16")]] kernel void reduce_##name##_reduced<ushort>( \
        device const ushort*, device float*, constant uint&, uint, uint, uint, uint, threadgroup float*);
SCALARS_METAL_REDUCE(sum, 0.0f, x + y)
SCALARS_METAL_REDUCE(min, INFINITY, reduced_min(x, y))
SCALARS_METAL_REDUCE(max, -INFINITY, reduced_max(x, y))
#undef SCALARS_METAL_REDUCE
