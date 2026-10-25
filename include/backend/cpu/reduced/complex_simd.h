#pragma once

// complex<half> / complex<bfloat16> SIMD packets.
//
// Unlike backend/cpu/reduced/simd.h (real half/bfloat16), which narrows after
// every single operation, this widens the whole operand once to a pair of
// complex<float> packets, performs the complete operation through the
// already-correct simd<complex<float>> backend (including its bounded-division
// fallback), and narrows once at the end. Narrowing after every intermediate
// op -- e.g. squaring b.real/b.imag for a division denominator -- would
// overflow half's ~65504 range far sooner than a real division ever would,
// so this does not reuse detail_complex_simd::complex_simd<T, Ops> directly;
// that shared template narrows every real/imag sub-operation, which is only
// safe when T itself is the arithmetic type (float/double), not when T is a
// storage-only reduced type being promoted through float underneath.

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "backend/cpu/detail/complex_simd.h"
#include "backend/cpu/reduced/convert.h"
#include "common/accumulate_type.h"
#include "types/complex.h"

namespace scalars::detail_reduced_complex
{
using detail_complex_simd::complex_half_packet;
using detail_complex_simd::complex_packet;

template <class T> struct complex_float_pair
{
    complex_packet<float> low;
    complex_packet<float> high;
};

template <class T> struct complex_reduced_simd
{
    static_assert(std::is_same_v<T, half> || std::is_same_v<T, bfloat16>);

    using value_t            = complex<T>;
    using component_r        = ::simd<T>;                        // reduced_simd<T>
    using component_packet_t = typename component_r::simd_t;     // packet<T, size>
    using cf                 = ::simd<scalars::complex<float>>;  // complete complex<float> backend
    using simd_t             = complex_packet<T>;                // {component_packet_t real, imag}
    using real_packet_t      = component_packet_t;
    using mask_t             = typename component_r::mask_t;
    using int_t              = typename component_r::int_t;
    using simd_int_t         = typename component_r::simd_int_t;
    using simd_half_t        = complex_half_packet<T>;
    using accumulation_t =
        scalars::acc_type<value_t, false>;  // complex<float>, see accumulate_type.h
    using float_pair_t                      = complex_float_pair<T>;
    static constexpr int  size              = component_r::size;
    static constexpr int  half_size         = component_r::half_size;
    static constexpr bool supports_ordering = false;
    static_assert(std::is_trivially_copyable_v<value_t>);
    static_assert(sizeof(value_t) == 2 * sizeof(T));

private:
    static float_pair_t to_float_pair(simd_t x)
    {
        return {{detail_reduced::widen<T>(x.real.bits.data()),
                    detail_reduced::widen<T>(x.imag.bits.data())},
            {detail_reduced::widen<T>(x.real.bits.data() + half_size),
                detail_reduced::widen<T>(x.imag.bits.data() + half_size)}};
    }
    static simd_t from_float_pair(complex_packet<float> low, complex_packet<float> high)
    {
        simd_t out;
        detail_reduced::narrow<T>(low.real, out.real.bits.data());
        detail_reduced::narrow<T>(high.real, out.real.bits.data() + half_size);
        detail_reduced::narrow<T>(low.imag, out.imag.bits.data());
        detail_reduced::narrow<T>(high.imag, out.imag.bits.data() + half_size);
        return out;
    }
    template <class Op> static simd_t map1(simd_t x, Op op)
    {
        const auto p = to_float_pair(x);
        return from_float_pair(op(p.low), op(p.high));
    }
    template <class Op> static simd_t map2(simd_t x, simd_t y, Op op)
    {
        const auto a = to_float_pair(x), b = to_float_pair(y);
        return from_float_pair(op(a.low, b.low), op(a.high, b.high));
    }
    template <class Op> static real_packet_t map_real(simd_t x, Op op)
    {
        const auto    p = to_float_pair(x);
        real_packet_t out;
        detail_reduced::narrow<T>(op(p.low), out.bits.data());
        detail_reduced::narrow<T>(op(p.high), out.bits.data() + half_size);
        return out;
    }

public:
    static void prefetch(const value_t* p) { component_r::prefetch(reinterpret_cast<const T*>(p)); }

    static simd_t loadu(const value_t* p)
    {
        std::array<T, size> re, im;
        for (int i = 0; i < size; ++i)
        {
            re[i] = p[i].real();
            im[i] = p[i].imag();
        }
        return {component_r::loadu(re.data()), component_r::loadu(im.data())};
    }
    static simd_t load(const value_t* p) { return loadu(p); }
    static void   storeu(simd_t a, value_t* p)
    {
        std::array<T, size> re, im;
        component_r::storeu(a.real, re.data());
        component_r::storeu(a.imag, im.data());
        for (int i = 0; i < size; ++i) {
            p[i] = value_t(re[i], im[i]);
        }
    }
    static void   store(simd_t a, value_t* p) { storeu(a, p); }
    static simd_t loadu(const value_t* p, int count)
    {
        assert(count >= 0 && count <= size);
        if (count == size) {
            return loadu(p);
        }
        std::array<value_t, size> lanes{};
        for (int i = 0; i < count; ++i) {
            lanes[i] = p[i];
        }
        return loadu(lanes.data());
    }
    static void storeu(simd_t a, value_t* p, int count)
    {
        assert(count >= 0 && count <= size);
        if (count == size)
        {
            storeu(a, p);
            return;
        }
        std::array<value_t, size> lanes;
        storeu(a, lanes.data());
        for (int i = 0; i < count; ++i) {
            p[i] = lanes[i];
        }
    }
    static simd_t gather(const value_t* p, std::ptrdiff_t stride)
    {
        if (stride == 1) {
            return loadu(p);
        }
        std::array<T, size> re, im;
        for (int i = 0; i < size; ++i)
        {
            re[i] = p[i * stride].real();
            im[i] = p[i * stride].imag();
        }
        return {component_r::loadu(re.data()), component_r::loadu(im.data())};
    }
    static void scatter(simd_t a, std::ptrdiff_t stride, value_t* p)
    {
        if (stride == 1)
        {
            storeu(a, p);
            return;
        }
        std::array<T, size> re, im;
        component_r::storeu(a.real, re.data());
        component_r::storeu(a.imag, im.data());
        for (int i = 0; i < size; ++i) {
            p[i * stride] = value_t(re[i], im[i]);
        }
    }
    template <class Index>
        requires std::is_integral_v<Index>
    static simd_t gather(const value_t* p, const Index* indices)
    {
        std::array<value_t, size> lanes;
        for (int i = 0; i < size; ++i) {
            lanes[i] = p[indices[i]];
        }
        return loadu(lanes.data());
    }
    template <class Index>
        requires std::is_integral_v<Index>
    static void scatter(simd_t a, const Index* indices, value_t* p)
    {
        std::array<value_t, size> lanes;
        storeu(a, lanes.data());
        for (int i = 0; i < size; ++i) {
            p[indices[i]] = lanes[i];
        }
    }
    static void   copy(const value_t* from, value_t* to) { storeu(loadu(from), to); }
    static simd_t set(value_t a)
    {
        return {component_r::set(a.real()), component_r::set(a.imag())};
    }
    template <class U>
        requires(std::is_arithmetic_v<U> && !std::is_same_v<U, int_t>)
    static simd_t set(U a)
    {
        return set(value_t(static_cast<T>(static_cast<float>(a)), T(0.f)));
    }
    static simd_t setzero() { return {component_r::setzero(), component_r::setzero()}; }
    static mask_t set_mask(bool a)
    {
        return a ? component_r::eq(component_r::setzero(), component_r::setzero())
                 : component_r::neq(component_r::setzero(), component_r::setzero());
    }
    static mask_t set(int_t a) { return component_r::set(a); }
    static void   broadcast(const value_t* p, simd_t& a, simd_t& b, simd_t& c, simd_t& d)
    {
        a = set(p[0]);
        b = set(p[1]);
        c = set(p[2]);
        d = set(p[3]);
    }
    static simd_t ploadquad(const value_t* p)
    {
        constexpr int             repeats = size / 4;
        std::array<value_t, size> lanes;
        for (int i = 0; i < size; ++i) {
            lanes[i] = p[i / repeats];
        }
        return loadu(lanes.data());
    }
    template <int N>
        requires(N > 0)
    static void ptranspose(simd_t* rows)
    {
        std::array<value_t, static_cast<std::size_t>(N) * size> input, output;
        for (int row = 0; row < N; ++row) {
            storeu(rows[row], input.data() + row * size);
        }
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = input[(i % N) * size + i / N];
        }
        for (int row = 0; row < N; ++row) {
            rows[row] = loadu(output.data() + row * size);
        }
    }
    static simd_half_t loadu_half(const value_t* p) { return {loadu(p, half_size)}; }
    static simd_half_t set(const value_t* p)
    {
        std::array<value_t, size> lanes{};
        for (int i = 0; i < half_size; ++i) {
            lanes[i] = *p;
        }
        return {loadu(lanes.data())};
    }
    static simd_half_t gather_half(const value_t* p, std::ptrdiff_t stride)
    {
        std::array<value_t, size> lanes{};
        for (int i = 0; i < half_size; ++i) {
            lanes[i] = p[i * stride];
        }
        return {loadu(lanes.data())};
    }
    static void storeu(simd_half_t a, value_t* p) { storeu(a.value, p, half_size); }
    static void store(simd_half_t a, value_t* p) { storeu(a, p); }
    static void scatter(simd_half_t a, std::ptrdiff_t stride, value_t* p)
    {
        std::array<value_t, size> lanes;
        storeu(a.value, lanes.data());
        for (int i = 0; i < half_size; ++i) {
            p[i * stride] = lanes[i];
        }
    }
    static simd_half_t add(simd_half_t a, simd_half_t b) { return {add(a.value, b.value)}; }
    static simd_half_t sub(simd_half_t a, simd_half_t b) { return {sub(a.value, b.value)}; }
    static simd_half_t mul(simd_half_t a, simd_half_t b) { return {mul(a.value, b.value)}; }
    static simd_half_t fma(simd_half_t a, simd_half_t b, simd_half_t c)
    {
        return {fma(a.value, b.value, c.value)};
    }
    static simd_half_t predux_downto4(simd_t a)
        requires(size >= 4)
    {
        std::array<value_t, size> lanes;
        storeu(a, lanes.data());
        for (int i = 0; i < half_size; ++i) {
            lanes[i] += lanes[i + half_size];
        }
        return loadu_half(lanes.data());
    }

    // -- Arithmetic, powers, and transcendentals: widen once, compute through
    //    simd<complex<float>>, narrow once. --
#define SCALARS_REDUCED_COMPLEX_UNARY(name)                                                        \
    static simd_t name(simd_t x)                                                                   \
    {                                                                                              \
        return map1(x, [](complex_packet<float> a) { return cf::name(a); });                       \
    }
    SCALARS_REDUCED_COMPLEX_UNARY(sqrt)
    SCALARS_REDUCED_COMPLEX_UNARY(invsqrt)
    SCALARS_REDUCED_COMPLEX_UNARY(reciprocal)
    SCALARS_REDUCED_COMPLEX_UNARY(sqr)
    SCALARS_REDUCED_COMPLEX_UNARY(neg)
    SCALARS_REDUCED_COMPLEX_UNARY(conj)
    SCALARS_REDUCED_COMPLEX_UNARY(exp)
    SCALARS_REDUCED_COMPLEX_UNARY(expm1)
    SCALARS_REDUCED_COMPLEX_UNARY(exp2)
    SCALARS_REDUCED_COMPLEX_UNARY(exp10)
    SCALARS_REDUCED_COMPLEX_UNARY(log)
    SCALARS_REDUCED_COMPLEX_UNARY(log1p)
    SCALARS_REDUCED_COMPLEX_UNARY(log2)
    SCALARS_REDUCED_COMPLEX_UNARY(log10)
    SCALARS_REDUCED_COMPLEX_UNARY(sin)
    SCALARS_REDUCED_COMPLEX_UNARY(cos)
    SCALARS_REDUCED_COMPLEX_UNARY(tan)
    SCALARS_REDUCED_COMPLEX_UNARY(asin)
    SCALARS_REDUCED_COMPLEX_UNARY(acos)
    SCALARS_REDUCED_COMPLEX_UNARY(atan)
    SCALARS_REDUCED_COMPLEX_UNARY(sinh)
    SCALARS_REDUCED_COMPLEX_UNARY(cosh)
    SCALARS_REDUCED_COMPLEX_UNARY(tanh)
    SCALARS_REDUCED_COMPLEX_UNARY(asinh)
    SCALARS_REDUCED_COMPLEX_UNARY(acosh)
    SCALARS_REDUCED_COMPLEX_UNARY(atanh)
    SCALARS_REDUCED_COMPLEX_UNARY(ceil)
    SCALARS_REDUCED_COMPLEX_UNARY(floor)
    SCALARS_REDUCED_COMPLEX_UNARY(trunc)
#undef SCALARS_REDUCED_COMPLEX_UNARY

#define SCALARS_REDUCED_COMPLEX_BINARY(name)                                                       \
    static simd_t name(simd_t x, simd_t y)                                                         \
    {                                                                                              \
        return map2(x,                                                                             \
            y,                                                                                     \
            [](complex_packet<float> a, complex_packet<float> b) { return cf::name(a, b); });      \
    }
    SCALARS_REDUCED_COMPLEX_BINARY(add)
    SCALARS_REDUCED_COMPLEX_BINARY(sub)
    SCALARS_REDUCED_COMPLEX_BINARY(mul)
    SCALARS_REDUCED_COMPLEX_BINARY(div)
    SCALARS_REDUCED_COMPLEX_BINARY(pow)
#undef SCALARS_REDUCED_COMPLEX_BINARY

    static simd_t fma(simd_t x, simd_t y, simd_t z)
    {
        const auto a = to_float_pair(x), b = to_float_pair(y), c = to_float_pair(z);
        return from_float_pair(cf::fma(a.low, b.low, c.low), cf::fma(a.high, b.high, c.high));
    }

    // -- Magnitude and components: real() / imag() read the stored packet
    //    directly; abs/abs2/arg widen since they need a float-accurate formula. --
    static real_packet_t real(simd_t x) { return x.real; }
    static real_packet_t imag(simd_t x) { return x.imag; }
    static real_packet_t abs2(simd_t x)
    {
        return map_real(x, [](complex_packet<float> a) { return cf::abs2(a); });
    }
    static real_packet_t norm(simd_t x) { return abs2(x); }
    static real_packet_t abs(simd_t x)
    {
        return map_real(x, [](complex_packet<float> a) { return cf::abs(a); });
    }
    static real_packet_t fabs(simd_t x) { return abs(x); }
    static real_packet_t arg(simd_t x)
    {
        return map_real(x, [](complex_packet<float> a) { return cf::arg(a); });
    }

    // -- Equality, masks, classification: these stay in the reduced real
    //    backend directly; there is no precision reason to widen a comparison. --
    static mask_t eq(simd_t a, simd_t b)
    {
        return component_r::and_mask(
            component_r::eq(a.real, b.real), component_r::eq(a.imag, b.imag));
    }
    static mask_t neq(simd_t a, simd_t b)
    {
        return component_r::or_mask(
            component_r::neq(a.real, b.real), component_r::neq(a.imag, b.imag));
    }
    static mask_t not_mask(mask_t a) { return component_r::not_mask(a); }
    static mask_t and_mask(mask_t a, mask_t b) { return component_r::and_mask(a, b); }
    static mask_t or_mask(mask_t a, mask_t b) { return component_r::or_mask(a, b); }
    static mask_t xor_mask(mask_t a, mask_t b) { return component_r::xor_mask(a, b); }
    static simd_t if_else(mask_t m, simd_t a, simd_t b)
    {
        return {component_r::if_else(m, a.real, b.real), component_r::if_else(m, a.imag, b.imag)};
    }
    static mask_t isnan(simd_t a)
    {
        return component_r::or_mask(
            component_r::neq(a.real, a.real), component_r::neq(a.imag, a.imag));
    }
    static mask_t isinf(simd_t a)
    {
        const auto infinity = component_r::set(std::numeric_limits<float>::infinity());
        return component_r::or_mask(component_r::eq(component_r::fabs(a.real), infinity),
            component_r::eq(component_r::fabs(a.imag), infinity));
    }
    static mask_t isfinite(simd_t a)
    {
        const auto infinity = component_r::set(std::numeric_limits<float>::infinity());
        return component_r::and_mask(component_r::lt(component_r::fabs(a.real), infinity),
            component_r::lt(component_r::fabs(a.imag), infinity));
    }

    static accumulation_t accumulate(simd_t a)
    {
        return accumulation_t(component_r::accumulate(a.real), component_r::accumulate(a.imag));
    }

    // Complex values have no ordering. Real-only analytic contracts are also
    // deliberately excluded instead of silently applying them componentwise,
    // matching detail_complex_simd::complex_simd<T, Ops>.
    static mask_t  gt(simd_t, simd_t)       = delete;
    static mask_t  ge(simd_t, simd_t)       = delete;
    static mask_t  lt(simd_t, simd_t)       = delete;
    static mask_t  le(simd_t, simd_t)       = delete;
    static simd_t  min(simd_t, simd_t)      = delete;
    static simd_t  max(simd_t, simd_t)      = delete;
    static value_t hmin(simd_t)             = delete;
    static value_t hmax(simd_t)             = delete;
    static simd_t  signcopy(simd_t, simd_t) = delete;
    static simd_t  hypot(simd_t, simd_t)    = delete;
    static simd_t  cbrt(simd_t)             = delete;
    static simd_t  cdf(simd_t)              = delete;
    static simd_t  inv_cdf(simd_t)          = delete;
};
}  // namespace scalars::detail_reduced_complex

template <>
struct simd<scalars::complex<scalars::half>>
    : scalars::detail_reduced_complex::complex_reduced_simd<scalars::half>
{
};
template <>
struct simd<scalars::complex<scalars::bfloat16>>
    : scalars::detail_reduced_complex::complex_reduced_simd<scalars::bfloat16>
{
};
