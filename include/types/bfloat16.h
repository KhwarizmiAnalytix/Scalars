#pragma once

#include <bit>
#include <cstdint>
#include <type_traits>

namespace kh::scalars
{
namespace detail
{
inline constexpr std::uint16_t float_to_bfloat16(float x) noexcept
{
    const auto bits = std::bit_cast<std::uint32_t>(x);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu))
    {
        return static_cast<std::uint16_t>((bits >> 16) | 0x0040u);
    }
    return static_cast<std::uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
inline constexpr float bfloat16_to_float(std::uint16_t bits) noexcept
{
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}
}  // namespace detail
struct bfloat16
{
    std::uint16_t bits{};
    constexpr bfloat16() noexcept = default;
    explicit constexpr bfloat16(float v) noexcept : bits(detail::float_to_bfloat16(v)) {}
    explicit constexpr bfloat16(double v) noexcept : bfloat16(static_cast<float>(v)) {}
    static constexpr bfloat16 from_bits(std::uint16_t b) noexcept
    {
        bfloat16 x;
        x.bits = b;
        return x;
    }
    constexpr std::uint16_t to_bits() const noexcept { return bits; }
    constexpr           operator float() const noexcept { return detail::bfloat16_to_float(bits); }
    constexpr bfloat16& operator+=(bfloat16 b) noexcept
    {
        *this = bfloat16(float(*this) + float(b));
        return *this;
    }
    constexpr bfloat16& operator-=(bfloat16 b) noexcept
    {
        *this = bfloat16(float(*this) - float(b));
        return *this;
    }
    constexpr bfloat16& operator*=(bfloat16 b) noexcept
    {
        *this = bfloat16(float(*this) * float(b));
        return *this;
    }
    constexpr bfloat16& operator/=(bfloat16 b) noexcept
    {
        *this = bfloat16(float(*this) / float(b));
        return *this;
    }
};
inline constexpr bfloat16 operator+(bfloat16 a, bfloat16 b) noexcept
{
    return a += b;
}
inline constexpr bfloat16 operator-(bfloat16 a, bfloat16 b) noexcept
{
    return a -= b;
}
inline constexpr bfloat16 operator*(bfloat16 a, bfloat16 b) noexcept
{
    return a *= b;
}
inline constexpr bfloat16 operator/(bfloat16 a, bfloat16 b) noexcept
{
    return a /= b;
}
inline constexpr bfloat16 operator-(bfloat16 a) noexcept
{
    return bfloat16::from_bits(a.bits ^ 0x8000u);
}
inline constexpr bool operator==(bfloat16 a, bfloat16 b) noexcept
{
    return float(a) == float(b);
}
inline constexpr bool operator!=(bfloat16 a, bfloat16 b) noexcept
{
    return !(a == b);
}
inline constexpr bool operator<(bfloat16 a, bfloat16 b) noexcept
{
    return float(a) < float(b);
}
inline constexpr bool operator<=(bfloat16 a, bfloat16 b) noexcept
{
    return float(a) <= float(b);
}
inline constexpr bool operator>(bfloat16 a, bfloat16 b) noexcept
{
    return float(a) > float(b);
}
inline constexpr bool operator>=(bfloat16 a, bfloat16 b) noexcept
{
    return float(a) >= float(b);
}
static_assert(sizeof(bfloat16) == 2 && std::is_trivially_copyable_v<bfloat16>);
}  // namespace kh::scalars
