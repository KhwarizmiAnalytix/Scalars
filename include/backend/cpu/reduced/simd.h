#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "backend/cpu/reduced/convert.h"

namespace scalars::detail_reduced
{
// Object-representation storage avoids aliasing scalar wrappers as uint16_t
// arrays. Fixed-size memcpy and the bitwise loops optimize to vector moves.
// The scalar tag keeps half, bf16, and masks distinct even at equal widths.
template <class Tag, int N> struct alignas(2 * N) packet
{
    std::array<uint16_t, N> bits;
};

struct float_pair
{
    float_packet low;
    float_packet high;
};

template <class T> struct reduced_simd
{
    static_assert(std::is_same_v<T, half> || std::is_same_v<T, bfloat16>);
    static_assert(sizeof(T) == sizeof(uint16_t) && std::is_trivially_copyable_v<T>);

    using value_t                  = T;
    using int_t                    = uint16_t;
    static constexpr int size      = 2 * float_backend::size;
    static constexpr int half_size = float_backend::size;
    using simd_t                   = packet<T, size>;
    using simd_half_t              = packet<T, half_size>;
    using mask_t                   = packet<void, size>;
    using simd_int_t               = mask_t;
    using float_pair_t             = float_pair;

    SCALARS_FORCE_INLINE static simd_t loadu(const value_t* from)
    {
        simd_t out;
        std::memcpy(out.bits.data(), from, sizeof(out));
        return out;
    }
    SCALARS_FORCE_INLINE static simd_t load(const value_t* from) { return loadu(from); }
    SCALARS_FORCE_INLINE static void   storeu(simd_t from, value_t* to)
    {
        std::memcpy(to, from.bits.data(), sizeof(from));
    }
    SCALARS_FORCE_INLINE static void store(simd_t from, value_t* to) { storeu(from, to); }

    // Counted operations never access inactive elements; zero lanes on load.
    SCALARS_FORCE_INLINE static simd_t loadu(const value_t* from, int count)
    {
        assert(count >= 0 && count <= size);
        simd_t out{};
        if (count != 0) {
            std::memcpy(out.bits.data(), from, count * sizeof(value_t));
        }
        return out;
    }
    SCALARS_FORCE_INLINE static void storeu(simd_t from, value_t* to, int count)
    {
        assert(count >= 0 && count <= size);
        if (count != 0) {
            std::memcpy(to, from.bits.data(), count * sizeof(value_t));
        }
    }
    SCALARS_FORCE_INLINE static simd_t set(value_t value)
    {
        simd_t out;
        out.bits.fill(value.x);
        return out;
    }
    template <class U, std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
    SCALARS_FORCE_INLINE static simd_t set(U value)
    {
        return set(value_t(static_cast<float>(value)));
    }
    SCALARS_FORCE_INLINE static simd_t setzero() { return {}; }
    SCALARS_FORCE_INLINE static void   prefetch(const value_t* from)
    {
#if defined(_MSC_VER)
        _mm_prefetch(reinterpret_cast<const char*>(from), _MM_HINT_T0);
#else
        __builtin_prefetch(from, 0, 3);
#endif
    }

    // Explicit widened interface for kernels retaining FP32 intermediates.
    // Ordinary packet operations below round after each reduced-result op.
    SCALARS_FORCE_INLINE static float_pair_t to_float_pair(simd_t from)
    {
        return {widen<T>(from.bits.data()), widen<T>(from.bits.data() + half_size)};
    }
    SCALARS_FORCE_INLINE static simd_t from_float_pair(float_packet low, float_packet high)
    {
        simd_t out;
        narrow<T>(low, out.bits.data());
        narrow<T>(high, out.bits.data() + half_size);
        return out;
    }
    SCALARS_FORCE_INLINE static float_pair_t load_float_pair(const value_t* from)
    {
        return to_float_pair(loadu(from));
    }
    SCALARS_FORCE_INLINE static void store_float_pair(
        float_packet low, float_packet high, value_t* to)
    {
        storeu(from_float_pair(low, high), to);
    }

private:
    template <class Op> SCALARS_FORCE_INLINE static simd_t map1(simd_t x, Op op)
    {
        const auto a = to_float_pair(x);
        return from_float_pair(op(a.low), op(a.high));
    }
    template <class Op> SCALARS_FORCE_INLINE static simd_t map2(simd_t x, simd_t y, Op op)
    {
        const auto a = to_float_pair(x);
        const auto b = to_float_pair(y);
        return from_float_pair(op(a.low, b.low), op(a.high, b.high));
    }
    template <class Op> SCALARS_FORCE_INLINE static mask_t compare(simd_t x, simd_t y, Op op)
    {
        const auto                    a = to_float_pair(x);
        const auto                    b = to_float_pair(y);
        typename float_backend::int_t low[half_size], high[half_size];
        float_backend::storeu(op(a.low, b.low), low);
        float_backend::storeu(op(a.high, b.high), high);
        mask_t out;
        for (int i = 0; i < half_size; ++i)
        {
            out.bits[i]             = low[i] != 0 ? uint16_t(0xffff) : uint16_t(0);
            out.bits[i + half_size] = high[i] != 0 ? uint16_t(0xffff) : uint16_t(0);
        }
        return out;
    }

public:
#define SCALARS_REDUCED_UNARY(name)                                                                \
    SCALARS_FORCE_INLINE static simd_t name(simd_t x)                                              \
    {                                                                                              \
        return map1(x, [](float_packet a) { return float_backend::name(a); });                     \
    }
    SCALARS_REDUCED_UNARY(sqrt)
    SCALARS_REDUCED_UNARY(sqr)
    SCALARS_REDUCED_UNARY(ceil)
    SCALARS_REDUCED_UNARY(floor)
    SCALARS_REDUCED_UNARY(trunc)
    SCALARS_REDUCED_UNARY(exp)
    SCALARS_REDUCED_UNARY(expm1)
    SCALARS_REDUCED_UNARY(exp2)
    SCALARS_REDUCED_UNARY(exp10)
    SCALARS_REDUCED_UNARY(log)
    SCALARS_REDUCED_UNARY(log1p)
    SCALARS_REDUCED_UNARY(log2)
    SCALARS_REDUCED_UNARY(log10)
    SCALARS_REDUCED_UNARY(sin)
    SCALARS_REDUCED_UNARY(cos)
    SCALARS_REDUCED_UNARY(tan)
    SCALARS_REDUCED_UNARY(asin)
    SCALARS_REDUCED_UNARY(acos)
    SCALARS_REDUCED_UNARY(atan)
    SCALARS_REDUCED_UNARY(sinh)
    SCALARS_REDUCED_UNARY(cosh)
    SCALARS_REDUCED_UNARY(tanh)
    SCALARS_REDUCED_UNARY(asinh)
    SCALARS_REDUCED_UNARY(acosh)
    SCALARS_REDUCED_UNARY(atanh)
    SCALARS_REDUCED_UNARY(cbrt)
    SCALARS_REDUCED_UNARY(cdf)
    SCALARS_REDUCED_UNARY(inv_cdf)
#undef SCALARS_REDUCED_UNARY

#define SCALARS_REDUCED_BINARY(name)                                                               \
    SCALARS_FORCE_INLINE static simd_t name(simd_t x, simd_t y)                                    \
    {                                                                                              \
        return map2(                                                                               \
            x, y, [](float_packet a, float_packet b) { return float_backend::name(a, b); });       \
    }
    SCALARS_REDUCED_BINARY(add)
    SCALARS_REDUCED_BINARY(sub)
    SCALARS_REDUCED_BINARY(mul)
    SCALARS_REDUCED_BINARY(div)
    SCALARS_REDUCED_BINARY(pow)
    SCALARS_REDUCED_BINARY(hypot)
#undef SCALARS_REDUCED_BINARY

    SCALARS_FORCE_INLINE static simd_t invsqrt(simd_t x)
    {
        // Division by sqrt handles zero, infinity and subnormals without the
        // exceptional-value failures of an unguarded Newton rsqrt refinement.
        return map1(x,
            [](float_packet a)
            { return float_backend::div(float_backend::set(1.f), float_backend::sqrt(a)); });
    }
    SCALARS_FORCE_INLINE static simd_t fma(simd_t x, simd_t y, simd_t z)
    {
        const auto a = to_float_pair(x);
        const auto b = to_float_pair(y);
        const auto c = to_float_pair(z);
        return from_float_pair(
            float_backend::fma(a.low, b.low, c.low), float_backend::fma(a.high, b.high, c.high));
    }
    SCALARS_FORCE_INLINE static simd_t fabs(simd_t x)
    {
        for (auto& b : x.bits) {
            b &= 0x7fff;
        }
        return x;
    }
    SCALARS_FORCE_INLINE static simd_t neg(simd_t x)
    {
        for (auto& b : x.bits) {
            b ^= 0x8000;
        }
        return x;
    }
    SCALARS_FORCE_INLINE static simd_t signcopy(simd_t x, simd_t sign)
    {
        for (int i = 0; i < size; ++i) {
            x.bits[i] = (x.bits[i] & 0x7fff) | (sign.bits[i] & 0x8000);
        }
        return x;
    }

#define SCALARS_REDUCED_COMPARE(name)                                                              \
    SCALARS_FORCE_INLINE static mask_t name(simd_t x, simd_t y)                                    \
    {                                                                                              \
        return compare(                                                                            \
            x, y, [](float_packet a, float_packet b) { return float_backend::name(a, b); });       \
    }
    SCALARS_REDUCED_COMPARE(eq)
    SCALARS_REDUCED_COMPARE(neq)
    SCALARS_REDUCED_COMPARE(gt)
    SCALARS_REDUCED_COMPARE(ge)
    SCALARS_REDUCED_COMPARE(lt)
    SCALARS_REDUCED_COMPARE(le)
#undef SCALARS_REDUCED_COMPARE

    SCALARS_FORCE_INLINE static mask_t loadu(const int_t* from)
    {
        mask_t out;
        std::memcpy(out.bits.data(), from, sizeof(out));
        return out;
    }
    SCALARS_FORCE_INLINE static mask_t load(const int_t* from) { return loadu(from); }
    SCALARS_FORCE_INLINE static void   storeu(mask_t from, int_t* to)
    {
        std::memcpy(to, from.bits.data(), sizeof(from));
    }
    SCALARS_FORCE_INLINE static void   store(mask_t from, int_t* to) { storeu(from, to); }
    SCALARS_FORCE_INLINE static mask_t set(int_t bits)
    {
        mask_t out;
        out.bits.fill(bits != 0 ? uint16_t(0xffff) : uint16_t(0));
        return out;
    }
    SCALARS_FORCE_INLINE static mask_t not_mask(mask_t x)
    {
        for (auto& b : x.bits) {
            b = static_cast<uint16_t>(~b);
        }
        return x;
    }
#define SCALARS_REDUCED_MASK(name, op)                                                             \
    SCALARS_FORCE_INLINE static mask_t name(mask_t x, mask_t y)                                    \
    {                                                                                              \
        for (int i = 0; i < size; ++i)                                                             \
            x.bits[i] = x.bits[i] op y.bits[i];                                                    \
        return x;                                                                                  \
    }
    SCALARS_REDUCED_MASK(and_mask, &)
    SCALARS_REDUCED_MASK(or_mask, |)
    SCALARS_REDUCED_MASK(xor_mask, ^)
#undef SCALARS_REDUCED_MASK
    SCALARS_FORCE_INLINE static simd_t if_else(mask_t mask, simd_t yes, simd_t no)
    {
        for (int i = 0; i < size; ++i) {
            yes.bits[i] = (mask.bits[i] & yes.bits[i]) | (uint16_t(~mask.bits[i]) & no.bits[i]);
        }
        return yes;
    }

    // Propagate NaNs and make signed-zero ties independent of the float ISA's
    // min/max instruction. Keep this policy identical for horizontal reductions.
    template <bool IsMin> SCALARS_FORCE_INLINE static simd_t extremum(simd_t x, simd_t y)
    {
        auto out = map2(x,
            y,
            [](float_packet a, float_packet b)
            {
                const auto result = IsMin ? float_backend::min(a, b) : float_backend::max(a, b);
                return float_backend::if_else(float_backend::neq(a, a),
                    a,
                    float_backend::if_else(float_backend::neq(b, b), b, result));
            });
        for (int i = 0; i < size; ++i)
        {
            if (((x.bits[i] | y.bits[i]) & 0x7fff) == 0)
            {
                out.bits[i] = IsMin ? (x.bits[i] | y.bits[i]) : (x.bits[i] & y.bits[i]);
            }
        }
        return out;
    }
    SCALARS_FORCE_INLINE static simd_t min(simd_t x, simd_t y) { return extremum<true>(x, y); }
    SCALARS_FORCE_INLINE static simd_t max(simd_t x, simd_t y) { return extremum<false>(x, y); }
    SCALARS_FORCE_INLINE static float  accumulate(simd_t x)
    {
        const auto a = to_float_pair(x);
        return static_cast<float>(float_backend::accumulate(float_backend::add(a.low, a.high)));
    }
    template <bool IsMin> SCALARS_FORCE_INLINE static value_t horizontal_extremum(simd_t x)
    {
        alignas(sizeof(float_packet)) float values[size];
        const auto                          a = to_float_pair(x);
        float_backend::storeu(a.low, values);
        float_backend::storeu(a.high, values + half_size);
        float out = values[0];
        for (int i = 0; i < size; ++i)
        {
            if (std::isnan(values[i])) {
                return value_t(values[i]);
            }
            if (values[i] == 0 && out == 0) {
                out = IsMin ? std::copysign(
                                  0.f, std::signbit(out) || std::signbit(values[i]) ? -1.f : 1.f)
                            : std::copysign(
                                  0.f, std::signbit(out) && std::signbit(values[i]) ? -1.f : 1.f);
            } else if (IsMin ? values[i] < out : values[i] > out) {
                out = values[i];
            }
        }
        return value_t(out);
    }
    SCALARS_FORCE_INLINE static value_t hmin(simd_t x) { return horizontal_extremum<true>(x); }
    SCALARS_FORCE_INLINE static value_t hmax(simd_t x) { return horizontal_extremum<false>(x); }

    SCALARS_FORCE_INLINE static simd_t gather(const value_t* from, int stride)
    {
        simd_t out;
        for (int i = 0; i < size; ++i) {
            out.bits[i] = from[static_cast<std::ptrdiff_t>(i) * stride].x;
        }
        return out;
    }
    SCALARS_FORCE_INLINE static simd_t gather(const value_t* from, const int* indices)
    {
        simd_t out;
        for (int i = 0; i < size; ++i) {
            out.bits[i] = from[indices[i]].x;
        }
        return out;
    }
    SCALARS_FORCE_INLINE static void scatter(simd_t from, int stride, value_t* to)
    {
        for (int i = 0; i < size; ++i) {
            to[static_cast<std::ptrdiff_t>(i) * stride] = value_t(from.bits[i], value_t::from_bits());
        }
    }
    SCALARS_FORCE_INLINE static void scatter(simd_t from, const int* indices, value_t* to)
    {
        for (int i = 0; i < size; ++i) {
            to[indices[i]] = value_t(from.bits[i], value_t::from_bits());
        }
    }
    SCALARS_FORCE_INLINE static void copy(const value_t* from, value_t* to)
    {
        storeu(loadu(from), to);
    }
    SCALARS_FORCE_INLINE static void broadcast(
        const value_t* from, simd_t& a, simd_t& b, simd_t& c, simd_t& d)
    {
        a = set(from[0]);
        b = set(from[1]);
        c = set(from[2]);
        d = set(from[3]);
    }
    SCALARS_FORCE_INLINE static simd_t ploadquad(const value_t* from)
    {
        simd_t out;
        for (int i = 0; i < size; ++i) {
            out.bits[i] = from[i / 4].x;
        }
        return out;
    }
    template <int N> SCALARS_FORCE_INLINE static void ptranspose(simd_t rows[N])
    {
        static_assert(N > 0 && (N % size == 0 || size % N == 0), "unsupported transpose shape");
        // Transpose N x size, then repack the row-major result into N packets.
        // This also handles rectangular blocks without converting NaN payloads.
        std::array<simd_t, N> out;
        for (int row = 0; row < N; ++row) {
            for (int col = 0; col < size; ++col)
            {
                const int index                      = col * N + row;
                out[index / size].bits[index % size] = rows[row].bits[col];
            }
        }
        for (int row = 0; row < N; ++row) {
            rows[row] = out[row];
        }
    }
    SCALARS_FORCE_INLINE static simd_half_t loadu_half(const value_t* from)
    {
        simd_half_t out;
        std::memcpy(out.bits.data(), from, sizeof(out));
        return out;
    }
    SCALARS_FORCE_INLINE static simd_half_t gather_half(const value_t* from, int stride)
    {
        simd_half_t out;
        for (int i = 0; i < half_size; ++i) {
            out.bits[i] = from[static_cast<std::ptrdiff_t>(i) * stride].x;
        }
        return out;
    }
    SCALARS_FORCE_INLINE static simd_half_t set(const value_t* from)
    {
        simd_half_t out;
        out.bits.fill(from->x);
        return out;
    }
    SCALARS_FORCE_INLINE static simd_half_t add(simd_half_t x, simd_half_t y)
    {
        simd_half_t out;
        narrow<T>(
            float_backend::add(widen<T>(x.bits.data()), widen<T>(y.bits.data())), out.bits.data());
        return out;
    }
    SCALARS_FORCE_INLINE static simd_half_t fma(simd_half_t x, simd_half_t y, simd_half_t z)
    {
        simd_half_t out;
        narrow<T>(float_backend::fma(
                      widen<T>(x.bits.data()), widen<T>(y.bits.data()), widen<T>(z.bits.data())),
            out.bits.data());
        return out;
    }
    // The legacy utility halves the packet width; for reduced types that is
    // one float packet's lane count (4, 8 or 16), not invariably four lanes.
    SCALARS_FORCE_INLINE static simd_half_t predux_downto4(simd_t x)
    {
        const auto  a = to_float_pair(x);
        simd_half_t out;
        narrow<T>(float_backend::add(a.low, a.high), out.bits.data());
        return out;
    }
    SCALARS_FORCE_INLINE static void scatter(simd_half_t from, int stride, value_t* to)
    {
        for (int i = 0; i < half_size; ++i) {
            to[static_cast<std::ptrdiff_t>(i) * stride] = value_t(from.bits[i], value_t::from_bits());
        }
    }
};
}  // namespace scalars::detail_reduced

template <> struct simd<scalars::half> : scalars::detail_reduced::reduced_simd<scalars::half>
{
};
template <>
struct simd<scalars::bfloat16> : scalars::detail_reduced::reduced_simd<scalars::bfloat16>
{
};
