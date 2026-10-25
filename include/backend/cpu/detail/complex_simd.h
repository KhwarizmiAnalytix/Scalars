#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <type_traits>

#include "common/accumulate_type.h"
#include "scalars_macros.h"
#include "types/complex.h"

namespace scalars::detail_complex_simd
{
template <class T> struct complex_packet
{
    typename ::simd<T>::simd_t real, imag;
};

// A distinct type keeps half-width overloads unambiguous even on backends
// whose real half-register aliases a full-register type. Upper lanes are zero.
template <class T> struct complex_half_packet
{
    complex_packet<T> value;
};

// Avoid GCC's ignored-attributes diagnostic for intrinsic vector template arguments.
template <class T> struct packed_registers
{
    using register_t = typename ::simd<T>::simd_t;
    register_t values[2];
    packed_registers() = default;
    packed_registers(register_t lo, register_t hi) : values{lo, hi} {}
    register_t&       operator[](std::size_t i) { return values[i]; }
    const register_t& operator[](std::size_t i) const { return values[i]; }
    register_t*       data() { return values; }
    const register_t* data() const { return values; }
};

// All backends use the same conservative range for ordinary vector division.
template <class T> constexpr T division_lower()
{
    if constexpr (std::is_same_v<T, float>)
        return T(0x1p-30);
    else
        return T(0x1p-250);
}
template <class T> constexpr T division_upper()
{
    if constexpr (std::is_same_v<T, float>)
        return T(0x1p31);
    else
        return T(0x1p251);
}

template <class T> SCALARS_FORCE_INLINE auto division_mask(complex_packet<T> a, complex_packet<T> b)
{
    using r            = ::simd<T>;
    const auto zero    = r::setzero();
    const auto bounded = [&](typename r::simd_t v)
    {
        const auto magnitude = r::fabs(v);
        return r::and_mask(r::lt(magnitude, r::set(division_upper<T>())),
            r::or_mask(r::ge(magnitude, r::set(division_lower<T>())), r::eq(v, zero)));
    };
    const auto components = r::and_mask(r::and_mask(bounded(a.real), bounded(a.imag)),
        r::and_mask(bounded(b.real), bounded(b.imag)));
    return r::and_mask(components, r::or_mask(r::neq(b.real, zero), r::neq(b.imag, zero)));
}

// ISA headers supply register packing, range reduction, and reciprocals.
// This layer contains no backend selection or architecture intrinsics.
template <class T, class Ops> struct complex_simd
{
    using r                         = ::simd<T>;
    using value_t                   = complex<T>;
    using simd_t                    = complex_packet<T>;
    using real_packet_t             = typename r::simd_t;
    using mask_t                    = typename r::mask_t;
    using int_t                     = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
    using simd_int_t                = typename r::simd_int_t;
    using simd_half_t               = complex_half_packet<T>;
    using accumulation_t            = scalars::acc_type<value_t, false>;
    using packed_t                  = packed_registers<T>;
    static constexpr int  size      = r::size;
    static constexpr int  half_size = size / 2;
    static constexpr bool supports_ordering = false;
    static_assert(std::is_trivially_copyable_v<value_t>);
    static_assert(sizeof(value_t) == 2 * sizeof(T));
    static_assert(sizeof(packed_t) == size * sizeof(value_t));
    static_assert(std::is_trivially_copyable_v<packed_t>);

private:
    template <class F> static simd_t map(simd_t a, F function)
    {
        std::array<value_t, size> lanes;
        storeu(a, lanes.data());
        for (auto& lane : lanes)
            lane = function(lane);
        return loadu(lanes.data());
    }
    template <class F> static simd_t map(simd_t a, simd_t b, F function)
    {
        std::array<value_t, size> x, y;
        storeu(a, x.data());
        storeu(b, y.data());
        for (int i = 0; i < size; ++i)
            x[i] = function(x[i], y[i]);
        return loadu(x.data());
    }
    template <class F> static real_packet_t map_real(simd_t a, F function)
    {
        std::array<value_t, size> lanes;
        std::array<T, size>       result;
        storeu(a, lanes.data());
        for (int i = 0; i < size; ++i)
            result[i] = function(lanes[i]);
        return r::loadu(result.data());
    }
    static SCALARS_NOINLINE simd_t divide_fallback(simd_t a, simd_t b)
    {
        std::array<value_t, size> x, y;
        storeu(a, x.data());
        storeu(b, y.data());
        for (int i = 0; i < size; ++i)
            x[i] /= y[i];
        return loadu(x.data());
    }

public:
    static void prefetch(const value_t* p)
    {
        // Prefetch uses only the address; it does not access a T through p.
        r::prefetch(reinterpret_cast<const T*>(p));
    }
    static simd_t gather(const value_t* p, std::ptrdiff_t stride)
    {
        if (stride == 1)
            return loadu(p);
        std::array<T, size> real, imag;
        for (int i = 0; i < size; ++i)
        {
            real[i] = p[i * stride].real();
            imag[i] = p[i * stride].imag();
        }
        return {r::loadu(real.data()), r::loadu(imag.data())};
    }
    static void scatter(simd_t a, std::ptrdiff_t stride, value_t* p)
    {
        if (stride == 1)
        {
            storeu(a, p);
            return;
        }
        std::array<T, size> real, imag;
        r::storeu(a.real, real.data());
        r::storeu(a.imag, imag.data());
        for (int i = 0; i < size; ++i)
            p[i * stride] = value_t(real[i], imag[i]);
    }
    template <class Index>
        requires std::is_integral_v<Index>
    static simd_t gather(const value_t* p, const Index* indices)
    {
        std::array<value_t, size> lanes;
        for (int i = 0; i < size; ++i)
            lanes[i] = p[indices[i]];
        return loadu(lanes.data());
    }
    template <class Index>
        requires std::is_integral_v<Index>
    static void scatter(simd_t a, const Index* indices, value_t* p)
    {
        std::array<value_t, size> lanes;
        storeu(a, lanes.data());
        // Increasing-lane order makes repeated indices deterministic.
        for (int i = 0; i < size; ++i)
            p[indices[i]] = lanes[i];
    }
    static simd_t loadu(const value_t* p)
    {
        // Byte copies preserve the complex object/aliasing rules. Optimizers
        // fold them into native loads/stores around the ISA's register shuffles.
        packed_t lanes;
        std::memcpy(lanes.data(), p, sizeof(lanes));
        return Ops::deinterleave(lanes);
    }
    static simd_t load(const value_t* p) { return loadu(p); }
    static void   storeu(simd_t a, value_t* p)
    {
        const auto lanes = Ops::interleave(a);
        std::memcpy(p, lanes.data(), sizeof(lanes));
    }
    static void   store(simd_t a, value_t* p) { storeu(a, p); }
    static simd_t loadu(const value_t* p, int count)
    {
        assert(count >= 0 && count <= size);
        if (count == size)
            return loadu(p);
        std::array<value_t, size> lanes{};
        for (int i = 0; i < count; ++i)
            lanes[i] = p[i];
        return loadu(lanes.data());
    }
    static simd_t loadu(std::nullptr_t, int count)
    {
        assert(count == 0);
        (void)count;
        return setzero();
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
        for (int i = 0; i < count; ++i)
            p[i] = lanes[i];
    }
    static simd_t set(value_t a) { return {r::set(a.real()), r::set(a.imag())}; }
    template <class U>
        requires(std::is_arithmetic_v<U> && !std::is_same_v<U, int_t>)
    static simd_t set(U a)
    {
        return set(value_t(static_cast<T>(a), T(0)));
    }
    static mask_t set(int_t a) { return set_mask(a != 0); }
    static simd_t setzero() { return {r::setzero(), r::setzero()}; }
    static mask_t set_mask(bool a)
    {
        return a ? r::eq(r::setzero(), r::setzero()) : r::neq(r::setzero(), r::setzero());
    }
    static mask_t loadu(const int_t* p, int count)
    {
        assert(count >= 0 && count <= size);
        std::array<T, size> flags{};
        for (int i = 0; i < count; ++i)
            flags[i] = p[i] != 0 ? T(1) : T(0);
        return r::neq(r::loadu(flags.data()), r::setzero());
    }
    static mask_t loadu(const int_t* p) { return loadu(p, size); }
    static mask_t load(const int_t* p) { return loadu(p); }
    static void   storeu(mask_t a, int_t* p, int count)
    {
        assert(count >= 0 && count <= size);
        std::array<T, size> flags;
        r::storeu(r::if_else(a, r::set(T(1)), r::setzero()), flags.data());
        for (int i = 0; i < count; ++i)
            p[i] = flags[i] != T(0) ? ~int_t(0) : int_t(0);
    }
    static void   storeu(mask_t a, int_t* p) { storeu(a, p, size); }
    static void   store(mask_t a, int_t* p) { storeu(a, p); }
    static void   storeu(real_packet_t a, T* p) { r::storeu(a, p); }
    static void   store(real_packet_t a, T* p) { r::storeu(a, p); }
    static mask_t eq(simd_t a, simd_t b)
    {
        return r::and_mask(r::eq(a.real, b.real), r::eq(a.imag, b.imag));
    }
    static mask_t neq(simd_t a, simd_t b)
    {
        return r::or_mask(r::neq(a.real, b.real), r::neq(a.imag, b.imag));
    }
    static mask_t not_mask(mask_t a) { return r::not_mask(a); }
    static mask_t and_mask(mask_t a, mask_t b) { return r::and_mask(a, b); }
    static mask_t or_mask(mask_t a, mask_t b) { return r::or_mask(a, b); }
    static mask_t xor_mask(mask_t a, mask_t b) { return r::xor_mask(a, b); }
    static simd_t if_else(mask_t mask, simd_t a, simd_t b)
    {
        return {r::if_else(mask, a.real, b.real), r::if_else(mask, a.imag, b.imag)};
    }
    static void copy(const value_t* from, value_t* to) { storeu(loadu(from), to); }
    static void broadcast(const value_t* p, simd_t& a, simd_t& b, simd_t& c, simd_t& d)
    {
        a = set(p[0]);
        b = set(p[1]);
        c = set(p[2]);
        d = set(p[3]);
    }
    static simd_t ploadquad(const value_t* p)
    {
        constexpr int             repeats = std::is_same_v<T, float> && size == 4 ? 2 : 4;
        std::array<value_t, size> lanes;
        for (int i = 0; i < size; ++i)
            lanes[i] = p[i / repeats];
        return loadu(lanes.data());
    }
    template <int N>
        requires(N > 0)
    static void ptranspose(simd_t* rows)
    {
        // Transpose N x size complete complex values, then repack into N packets.
        std::array<value_t, static_cast<std::size_t>(N) * size> input, output;
        for (int row = 0; row < N; ++row)
            storeu(rows[row], input.data() + row * size);
        for (std::size_t i = 0; i < input.size(); ++i)
            output[i] = input[(i % N) * size + i / N];
        for (int row = 0; row < N; ++row)
            rows[row] = loadu(output.data() + row * size);
    }
    static simd_half_t loadu_half(const value_t* p) { return {loadu(p, half_size)}; }
    static simd_half_t set(const value_t* p)
    {
        std::array<value_t, size> lanes{};
        for (int i = 0; i < half_size; ++i)
            lanes[i] = *p;
        return {loadu(lanes.data())};
    }
    static simd_half_t gather_half(const value_t* p, std::ptrdiff_t stride)
    {
        std::array<value_t, size> lanes{};
        for (int i = 0; i < half_size; ++i)
            lanes[i] = p[i * stride];
        return {loadu(lanes.data())};
    }
    template <class Index>
        requires std::is_integral_v<Index>
    static simd_half_t gather_half(const value_t* p, const Index* indices)
    {
        std::array<value_t, size> lanes{};
        for (int i = 0; i < half_size; ++i)
            lanes[i] = p[indices[i]];
        return {loadu(lanes.data())};
    }
    static void storeu(simd_half_t a, value_t* p) { storeu(a.value, p, half_size); }
    static void store(simd_half_t a, value_t* p) { storeu(a, p); }
    static void scatter(simd_half_t a, std::ptrdiff_t stride, value_t* p)
    {
        std::array<value_t, size> lanes;
        storeu(a.value, lanes.data());
        for (int i = 0; i < half_size; ++i)
            p[i * stride] = lanes[i];
    }
    template <class Index>
        requires std::is_integral_v<Index>
    static void scatter(simd_half_t a, const Index* indices, value_t* p)
    {
        std::array<value_t, size> lanes;
        storeu(a.value, lanes.data());
        for (int i = 0; i < half_size; ++i)
            p[indices[i]] = lanes[i];
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
        for (int i = 0; i < half_size; ++i)
            lanes[i] += lanes[i + half_size];
        return loadu_half(lanes.data());
    }
    static simd_t add(simd_t a, simd_t b)
    {
        const auto x = Ops::interleave(a), y = Ops::interleave(b);
        return Ops::deinterleave({r::add(x[0], y[0]), r::add(x[1], y[1])});
    }
    static simd_t sub(simd_t a, simd_t b)
    {
        const auto x = Ops::interleave(a), y = Ops::interleave(b);
        return Ops::deinterleave({r::sub(x[0], y[0]), r::sub(x[1], y[1])});
    }
    static simd_t neg(simd_t a) { return {r::neg(a.real), r::neg(a.imag)}; }
    static simd_t conj(simd_t a) { return {a.real, r::neg(a.imag)}; }
    static simd_t mul(simd_t a, simd_t b)
    {
        return {r::sub(r::mul(a.real, b.real), r::mul(a.imag, b.imag)),
            r::add(r::mul(a.real, b.imag), r::mul(a.imag, b.real))};
    }
    static simd_t        sqr(simd_t a) { return mul(a, a); }
    static real_packet_t real(simd_t a) { return a.real; }
    static real_packet_t imag(simd_t a) { return a.imag; }
    static real_packet_t abs2(simd_t a)
    {
        return r::add(r::mul(a.real, a.real), r::mul(a.imag, a.imag));
    }
    static real_packet_t norm(simd_t a) { return abs2(a); }
    static real_packet_t abs(simd_t a)
    {
        return map_real(a, [](value_t x) { return std::hypot(x.real(), x.imag()); });
    }
    // fabs is the magnitude alias, returning one real result per complex lane.
    static real_packet_t fabs(simd_t a) { return abs(a); }
    static real_packet_t arg(simd_t a)
    {
        return map_real(a, [](value_t x) { return std::atan2(x.imag(), x.real()); });
    }
    static mask_t isnan(simd_t a)
    {
        return r::or_mask(r::neq(a.real, a.real), r::neq(a.imag, a.imag));
    }
    static mask_t isinf(simd_t a)
    {
        const auto infinity = r::set(std::numeric_limits<T>::infinity());
        return r::or_mask(r::eq(r::fabs(a.real), infinity), r::eq(r::fabs(a.imag), infinity));
    }
    static mask_t isfinite(simd_t a)
    {
        const auto infinity = r::set(std::numeric_limits<T>::infinity());
        return r::and_mask(r::lt(r::fabs(a.real), infinity), r::lt(r::fabs(a.imag), infinity));
    }
    // Branch-sensitive functions deliberately map the scalar complex backend.
    // They are packet API operations, not claims of native vector math.
    static simd_t sqrt(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::sqrt(x); });
    }
    static simd_t reciprocal(simd_t a) { return div(set(value_t(T(1), T(0))), a); }
    static simd_t invsqrt(simd_t a) { return reciprocal(sqrt(a)); }
    static simd_t pow(simd_t a, simd_t b)
    {
        return map(a, b, [](value_t x, value_t y) { return scalars::pow(x, y); });
    }
    static simd_t exp(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::exp(x); });
    }
    static simd_t expm1(simd_t a)
    {
        return map(a,
            [](value_t x)
            {
                if (x.imag() == T(0))
                    return value_t(std::expm1(x.real()), x.imag());
                if (scalars::isfinite(x))
                    return scalars::expm1(x);
                const auto result = scalars::exp(x);
                return value_t(result.real() - T(1), result.imag());
            });
    }
    static simd_t exp2(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::exp(x * std::numbers::ln2_v<T>); });
    }
    static simd_t exp10(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::exp(x * std::numbers::ln10_v<T>); });
    }
    static simd_t log(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::log(x); });
    }
    static simd_t log1p(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::log1p(x); });
    }
    static simd_t log2(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::log(x) / std::numbers::ln2_v<T>; });
    }
    static simd_t log10(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::log10(x); });
    }
    static simd_t sin(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::sin(x); });
    }
    static simd_t cos(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::cos(x); });
    }
    static simd_t tan(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::tan(x); });
    }
    static simd_t asin(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::asin(x); });
    }
    static simd_t acos(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::acos(x); });
    }
    static simd_t atan(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::atan(x); });
    }
    static simd_t sinh(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::sinh(x); });
    }
    static simd_t cosh(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::cosh(x); });
    }
    static simd_t tanh(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::tanh(x); });
    }
    static simd_t asinh(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::asinh(x); });
    }
    static simd_t acosh(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::acosh(x); });
    }
    static simd_t atanh(simd_t a)
    {
        return map(a, [](value_t x) { return scalars::atanh(x); });
    }
    static simd_t ceil(simd_t a)
    {
        return map(a, [](value_t x) { return value_t(std::ceil(x.real()), std::ceil(x.imag())); });
    }
    static simd_t floor(simd_t a)
    {
        return map(
            a, [](value_t x) { return value_t(std::floor(x.real()), std::floor(x.imag())); });
    }
    static simd_t trunc(simd_t a)
    {
        return map(
            a, [](value_t x) { return value_t(std::trunc(x.real()), std::trunc(x.imag())); });
    }
    static simd_t fma(simd_t a, simd_t b, simd_t c)
    {
        return {r::fma(r::neg(a.imag), b.imag, r::fma(a.real, b.real, c.real)),
            r::fma(a.imag, b.real, r::fma(a.real, b.imag, c.imag))};
    }
    static SCALARS_FORCE_INLINE simd_t div(simd_t a, simd_t b)
    {
        if (Ops::division_is_safe(a, b))
        {
            const auto denominator = r::fma(b.imag, b.imag, r::mul(b.real, b.real));
            const auto inverse     = Ops::reciprocal(denominator);
            return {r::mul(r::fma(a.imag, b.imag, r::mul(a.real, b.real)), inverse),
                r::mul(r::fma(r::neg(a.real), b.imag, r::mul(a.imag, b.real)), inverse)};
        }
        return divide_fallback(a, b);
    }
    static accumulation_t accumulate(simd_t a)
    {
        std::array<T, size> real, imag;
        r::storeu(a.real, real.data());
        r::storeu(a.imag, imag.data());
        using acc_real    = decltype(accumulation_t{}.real());
        acc_real real_sum = 0, imag_sum = 0;
        for (int i = 0; i < size; ++i)
        {
            real_sum += static_cast<acc_real>(real[i]);
            imag_sum += static_cast<acc_real>(imag[i]);
        }
        return {real_sum, imag_sum};
    }
    // Complex values have no ordering. Real-only analytic contracts are also
    // deliberately excluded instead of silently applying them componentwise.
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
}  // namespace scalars::detail_complex_simd
