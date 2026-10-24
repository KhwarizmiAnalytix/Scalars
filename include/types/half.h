#pragma once
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
namespace kh::scalars
{
namespace detail
{
inline constexpr std::uint16_t float_to_half(float f) noexcept
{
    const std::uint32_t x    = std::bit_cast<std::uint32_t>(f);
    const std::uint16_t sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
    const std::uint32_t exp = (x >> 23) & 0xffu, mant = x & 0x7fffffu;
    if (exp == 255)
    {
        return static_cast<std::uint16_t>(sign | 0x7c00u | (mant ? (0x0200u | (mant >> 13)) : 0));
    }
    const int e = static_cast<int>(exp) - 127 + 15;
    if (e >= 31)
    {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (e <= 0)
    {
        if (e < -10)
        {
            return sign;
        }
        const std::uint32_t m         = mant | 0x800000u;
        const int           shift     = 14 - e;
        std::uint32_t       val       = m >> shift;
        const std::uint32_t remainder = m & ((1u << shift) - 1u);
        const std::uint32_t midpoint  = 1u << (shift - 1);
        if (remainder > midpoint || (remainder == midpoint && (val & 1u)))
        {
            ++val;
        }
        return static_cast<std::uint16_t>(sign | val);
    }
    std::uint32_t       val       = (static_cast<std::uint32_t>(e) << 10) | (mant >> 13);
    const std::uint32_t remainder = mant & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (val & 1u)))
    {
        ++val;
    }
    return static_cast<std::uint16_t>(sign | val);
}
inline constexpr float half_to_float(std::uint16_t h) noexcept
{
    const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000u) << 16);
    const std::uint32_t exp = (h >> 10) & 0x1fu, mant = h & 0x03ffu;
    std::uint32_t       bits;
    if (exp == 0)
    {
        if (!mant)
        {
            bits = sign;
        }
        else
        {
            std::uint32_t m = mant;
            int           e = -14;
            while ((m & 0x400u) == 0)
            {
                m <<= 1;
                --e;
            }
            bits = sign | (static_cast<std::uint32_t>(e + 127) << 23) | ((m & 0x3ffu) << 13);
        }
    }
    else if (exp == 31)
    {
        bits = sign | 0x7f800000u | (mant << 13);
    }
    else
    {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    return std::bit_cast<float>(bits);
}
}  // namespace detail
struct half
{
    std::uint16_t bits{};
    constexpr half() noexcept = default;
    explicit constexpr half(float v) noexcept : bits(detail::float_to_half(v)) {}
    explicit constexpr half(double v) noexcept : half(static_cast<float>(v)) {}
    [[nodiscard]] static constexpr half from_bits(std::uint16_t b) noexcept
    {
        half h;
        h.bits = b;
        return h;
    }
    [[nodiscard]] constexpr std::uint16_t to_bits() const noexcept { return bits; }
    constexpr       operator float() const noexcept { return detail::half_to_float(bits); }
    constexpr half& operator+=(half v) noexcept
    {
        *this = half(float(*this) + float(v));
        return *this;
    }
    constexpr half& operator-=(half v) noexcept
    {
        *this = half(float(*this) - float(v));
        return *this;
    }
    constexpr half& operator*=(half v) noexcept
    {
        *this = half(float(*this) * float(v));
        return *this;
    }
    constexpr half& operator/=(half v) noexcept
    {
        *this = half(float(*this) / float(v));
        return *this;
    }
};
inline constexpr half operator+(half a, half b) noexcept
{
    return a += b;
}
inline constexpr half operator-(half a, half b) noexcept
{
    return a -= b;
}
inline constexpr half operator*(half a, half b) noexcept
{
    return a *= b;
}
inline constexpr half operator/(half a, half b) noexcept
{
    return a /= b;
}
inline constexpr half operator-(half a) noexcept
{
    return half::from_bits(a.bits ^ 0x8000u);
}
inline constexpr bool operator==(half a, half b) noexcept
{
    return float(a) == float(b);
}
inline constexpr bool operator!=(half a, half b) noexcept
{
    return !(a == b);
}
inline constexpr bool operator<(half a, half b) noexcept
{
    return float(a) < float(b);
}
inline constexpr bool operator<=(half a, half b) noexcept
{
    return float(a) <= float(b);
}
inline constexpr bool operator>(half a, half b) noexcept
{
    return float(a) > float(b);
}
inline constexpr bool operator>=(half a, half b) noexcept
{
    return float(a) >= float(b);
}
static_assert(sizeof(half) == 2 && std::is_trivially_copyable_v<half>);
}  // namespace kh::scalars
namespace std
{
template <> class numeric_limits<kh::scalars::half>
{
public:
    static constexpr bool is_specialized = true;
    static constexpr int  digits = 11, digits10 = 3, max_digits10 = 5;
    static constexpr bool is_signed = true, is_integer = false, is_exact = false, is_iec559 = true;
    static constexpr int  radix = 2, min_exponent = -13, max_exponent = 16;
    static constexpr kh::scalars::half min() noexcept
    {
        return kh::scalars::half::from_bits(0x0400);
    }
    static constexpr kh::scalars::half max() noexcept
    {
        return kh::scalars::half::from_bits(0x7bff);
    }
    static constexpr kh::scalars::half lowest() noexcept
    {
        return kh::scalars::half::from_bits(0xfbff);
    }
    static constexpr kh::scalars::half epsilon() noexcept
    {
        return kh::scalars::half::from_bits(0x1400);
    }
    static constexpr kh::scalars::half infinity() noexcept
    {
        return kh::scalars::half::from_bits(0x7c00);
    }
    static constexpr kh::scalars::half quiet_NaN() noexcept
    {
        return kh::scalars::half::from_bits(0x7e00);
    }
    static constexpr kh::scalars::half denorm_min() noexcept
    {
        return kh::scalars::half::from_bits(0x0001);
    }
};
}  // namespace std
