#pragma once

#include <cassert>
#include <cstdint>
#include <type_traits>

#include "backend/gpu/float/simd.h"
#include "types/bfloat16.h"
#include "types/half.h"

namespace scalars::detail_gpu
{
// CUDA/HIP packets have one logical lane per thread. Keep the reduced storage
// type distinct, widen for operation math, and round once on the way back.
// This follows ATen's opmath_type policy without changing the simd<T> contract.
template <class T> struct reduced_simd
{
    static_assert(std::is_same_v<T, half> || std::is_same_v<T, bfloat16>);
    using value_t             = T;
    using simd_t              = T;
    using int_t               = uint16_t;
    using mask_t              = uint16_t;
    using simd_int_t          = mask_t;
    using float_t             = float;
    static constexpr int size = 1;

    SCALARS_SIMD_RETURN_TYPE   prefetch(const value_t*) {}
    SCALARS_SIMD_METHOD simd_t load(const value_t* from) { return *from; }
    SCALARS_SIMD_METHOD simd_t loadu(const value_t* from) { return *from; }
    SCALARS_SIMD_RETURN_TYPE   store(simd_t x, value_t* to) { *to = x; }
    SCALARS_SIMD_RETURN_TYPE   storeu(simd_t x, value_t* to) { *to = x; }
    SCALARS_SIMD_METHOD simd_t loadu(const value_t* from, int count)
    {
        assert(count >= 0 && count <= size);
        return count ? *from : setzero();
    }
    SCALARS_SIMD_RETURN_TYPE storeu(simd_t x, value_t* to, int count)
    {
        assert(count >= 0 && count <= size);
        if (count)
            *to = x;
    }
    SCALARS_SIMD_METHOD simd_t set(value_t x) { return x; }
    template <class U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
    SCALARS_SIMD_METHOD simd_t set(U x)
    {
        return value_t(static_cast<float>(x));
    }
    SCALARS_SIMD_METHOD simd_t setzero() { return value_t(0, value_t::from_bits()); }

    // Explicit widened path for kernels that retain FP32 intermediates.
    SCALARS_SIMD_METHOD float  to_float(simd_t x) { return static_cast<float>(x); }
    SCALARS_SIMD_METHOD simd_t from_float(float x) { return value_t(x); }
    SCALARS_SIMD_METHOD float  load_float(const value_t* from) { return to_float(*from); }
    SCALARS_SIMD_RETURN_TYPE   store_float(float x, value_t* to) { *to = from_float(x); }

#define SCALARS_GPU_REDUCED_UNARY(name)                                                            \
    SCALARS_SIMD_METHOD simd_t name(simd_t x) { return from_float(simd<float>::name(to_float(x))); }
    SCALARS_GPU_REDUCED_UNARY(sqrt)
    SCALARS_GPU_REDUCED_UNARY(invsqrt)
    SCALARS_GPU_REDUCED_UNARY(sqr)
    SCALARS_GPU_REDUCED_UNARY(ceil)
    SCALARS_GPU_REDUCED_UNARY(floor)
    SCALARS_GPU_REDUCED_UNARY(trunc)
    SCALARS_GPU_REDUCED_UNARY(exp)
    SCALARS_GPU_REDUCED_UNARY(expm1)
    SCALARS_GPU_REDUCED_UNARY(exp2)
    SCALARS_GPU_REDUCED_UNARY(exp10)
    SCALARS_GPU_REDUCED_UNARY(log)
    SCALARS_GPU_REDUCED_UNARY(log1p)
    SCALARS_GPU_REDUCED_UNARY(log2)
    SCALARS_GPU_REDUCED_UNARY(log10)
    SCALARS_GPU_REDUCED_UNARY(cbrt)
    SCALARS_GPU_REDUCED_UNARY(sin)
    SCALARS_GPU_REDUCED_UNARY(cos)
    SCALARS_GPU_REDUCED_UNARY(tan)
    SCALARS_GPU_REDUCED_UNARY(asin)
    SCALARS_GPU_REDUCED_UNARY(acos)
    SCALARS_GPU_REDUCED_UNARY(atan)
    SCALARS_GPU_REDUCED_UNARY(sinh)
    SCALARS_GPU_REDUCED_UNARY(cosh)
    SCALARS_GPU_REDUCED_UNARY(tanh)
    SCALARS_GPU_REDUCED_UNARY(asinh)
    SCALARS_GPU_REDUCED_UNARY(acosh)
    SCALARS_GPU_REDUCED_UNARY(atanh)
    SCALARS_GPU_REDUCED_UNARY(cdf)
    SCALARS_GPU_REDUCED_UNARY(inv_cdf)
#undef SCALARS_GPU_REDUCED_UNARY

#define SCALARS_GPU_REDUCED_BINARY(name)                                                           \
    SCALARS_SIMD_METHOD simd_t name(simd_t x, simd_t y)                                            \
    {                                                                                              \
        return from_float(simd<float>::name(to_float(x), to_float(y)));                            \
    }
    SCALARS_GPU_REDUCED_BINARY(add)
    SCALARS_GPU_REDUCED_BINARY(sub)
    SCALARS_GPU_REDUCED_BINARY(mul)
    SCALARS_GPU_REDUCED_BINARY(div)
    SCALARS_GPU_REDUCED_BINARY(pow)
    SCALARS_GPU_REDUCED_BINARY(hypot)
#undef SCALARS_GPU_REDUCED_BINARY
    SCALARS_SIMD_METHOD simd_t fma(simd_t x, simd_t y, simd_t z)
    {
        return from_float(simd<float>::fma(to_float(x), to_float(y), to_float(z)));
    }

    // Manipulate encodings so signed zero and NaN payloads survive sign ops.
    SCALARS_SIMD_METHOD simd_t fabs(simd_t x)
    {
        return value_t(x.x & 0x7fff, value_t::from_bits());
    }
    SCALARS_SIMD_METHOD simd_t neg(simd_t x) { return value_t(x.x ^ 0x8000, value_t::from_bits()); }
    SCALARS_SIMD_METHOD simd_t signcopy(simd_t x, simd_t sign)
    {
        return value_t((x.x & 0x7fff) | (sign.x & 0x8000), value_t::from_bits());
    }

    // Match the CPU reduced packets: propagate NaNs; min(-0,+0)=-0,
    // max(-0,+0)=+0 regardless of operand order.
    template <bool IsMin> SCALARS_SIMD_METHOD simd_t extremum(simd_t x, simd_t y)
    {
        const float a = to_float(x), b = to_float(y);
        if (a != a)
            return x;
        if (b != b)
            return y;
        if (((x.x | y.x) & 0x7fff) == 0)
            return value_t(IsMin ? x.x | y.x : x.x & y.x, value_t::from_bits());
        return (IsMin ? a < b : a > b) ? x : y;
    }
    SCALARS_SIMD_METHOD simd_t  min(simd_t x, simd_t y) { return extremum<true>(x, y); }
    SCALARS_SIMD_METHOD simd_t  max(simd_t x, simd_t y) { return extremum<false>(x, y); }
    SCALARS_SIMD_METHOD float   accumulate(simd_t x) { return to_float(x); }
    SCALARS_SIMD_METHOD value_t hmin(simd_t x) { return x; }
    SCALARS_SIMD_METHOD value_t hmax(simd_t x) { return x; }

#define SCALARS_GPU_REDUCED_COMPARE(name, op)                                                      \
    SCALARS_SIMD_METHOD mask_t name(simd_t x, simd_t y)                                            \
    {                                                                                              \
        return to_float(x) op to_float(y) ? uint16_t(0xffff) : uint16_t(0);                        \
    }
    SCALARS_GPU_REDUCED_COMPARE(eq, ==)
    SCALARS_GPU_REDUCED_COMPARE(neq, !=)
    SCALARS_GPU_REDUCED_COMPARE(lt, <)
    SCALARS_GPU_REDUCED_COMPARE(le, <=)
    SCALARS_GPU_REDUCED_COMPARE(gt, >)
    SCALARS_GPU_REDUCED_COMPARE(ge, >=)
#undef SCALARS_GPU_REDUCED_COMPARE
    SCALARS_SIMD_METHOD mask_t load(const int_t* from) { return *from; }
    SCALARS_SIMD_METHOD mask_t loadu(const int_t* from) { return *from; }
    SCALARS_SIMD_RETURN_TYPE   store(mask_t x, int_t* to) { *to = x; }
    SCALARS_SIMD_RETURN_TYPE   storeu(mask_t x, int_t* to) { *to = x; }
    SCALARS_SIMD_METHOD mask_t and_mask(mask_t x, mask_t y) { return x & y; }
    SCALARS_SIMD_METHOD mask_t or_mask(mask_t x, mask_t y) { return x | y; }
    SCALARS_SIMD_METHOD mask_t xor_mask(mask_t x, mask_t y) { return x ^ y; }
    SCALARS_SIMD_METHOD mask_t not_mask(mask_t x) { return static_cast<mask_t>(~x); }
    SCALARS_SIMD_METHOD simd_t if_else(mask_t m, simd_t yes, simd_t no)
    {
        return value_t((m & yes.x) | (static_cast<mask_t>(~m) & no.x), value_t::from_bits());
    }

    SCALARS_SIMD_METHOD simd_t gather(const value_t* from, const int* offsets)
    {
        return from[offsets[0]];
    }
    // As in the float GPU backend, stride is this thread's absolute index.
    SCALARS_SIMD_METHOD simd_t gather(const value_t* from, int stride) { return from[stride]; }
    SCALARS_SIMD_RETURN_TYPE   scatter(simd_t x, const int* offsets, value_t* to)
    {
        to[offsets[0]] = x;
    }
    SCALARS_SIMD_RETURN_TYPE scatter(simd_t x, int stride, value_t* to) { to[stride] = x; }
    SCALARS_SIMD_RETURN_TYPE copy(const value_t* from, value_t* to) { *to = *from; }
};
}  // namespace scalars::detail_gpu

template <> struct simd<scalars::half> : scalars::detail_gpu::reduced_simd<scalars::half>
{
};
template <> struct simd<scalars::bfloat16> : scalars::detail_gpu::reduced_simd<scalars::bfloat16>
{
};
