#pragma once
#include <bit>
#include <cstdint>
#include <type_traits>
namespace kh::scalars {
namespace detail {
inline constexpr std::uint16_t float_to_bfloat16(float x) noexcept {
  const auto bits=std::bit_cast<std::uint32_t>(x);
  if ((bits & 0x7f800000u)==0x7f800000u && (bits & 0x007fffffu))
    return static_cast<std::uint16_t>((bits >> 16) | 0x0040u);
  return static_cast<std::uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
inline constexpr float bfloat16_to_float(std::uint16_t bits) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}
}
struct BFloat16 {
  std::uint16_t bits{};
  constexpr BFloat16() noexcept=default;
  explicit constexpr BFloat16(float v) noexcept : bits(detail::float_to_bfloat16(v)) {}
  explicit constexpr BFloat16(double v) noexcept : BFloat16(static_cast<float>(v)) {}
  static constexpr BFloat16 from_bits(std::uint16_t b) noexcept { BFloat16 x; x.bits=b; return x; }
  constexpr std::uint16_t to_bits() const noexcept { return bits; }
  constexpr operator float() const noexcept { return detail::bfloat16_to_float(bits); }
  constexpr BFloat16& operator+=(BFloat16 b) noexcept { *this=BFloat16(float(*this)+float(b)); return *this; }
  constexpr BFloat16& operator-=(BFloat16 b) noexcept { *this=BFloat16(float(*this)-float(b)); return *this; }
  constexpr BFloat16& operator*=(BFloat16 b) noexcept { *this=BFloat16(float(*this)*float(b)); return *this; }
  constexpr BFloat16& operator/=(BFloat16 b) noexcept { *this=BFloat16(float(*this)/float(b)); return *this; }
};
inline constexpr BFloat16 operator+(BFloat16 a,BFloat16 b) noexcept { return a+=b; }
inline constexpr BFloat16 operator-(BFloat16 a,BFloat16 b) noexcept { return a-=b; }
inline constexpr BFloat16 operator*(BFloat16 a,BFloat16 b) noexcept { return a*=b; }
inline constexpr BFloat16 operator/(BFloat16 a,BFloat16 b) noexcept { return a/=b; }
inline constexpr BFloat16 operator-(BFloat16 a) noexcept { return BFloat16::from_bits(a.bits ^ 0x8000u); }
inline constexpr bool operator==(BFloat16 a,BFloat16 b) noexcept { return float(a)==float(b); }
inline constexpr bool operator!=(BFloat16 a,BFloat16 b) noexcept { return !(a==b); }
inline constexpr bool operator<(BFloat16 a,BFloat16 b) noexcept { return float(a)<float(b); }
inline constexpr bool operator<=(BFloat16 a,BFloat16 b) noexcept { return float(a)<=float(b); }
inline constexpr bool operator>(BFloat16 a,BFloat16 b) noexcept { return float(a)>float(b); }
inline constexpr bool operator>=(BFloat16 a,BFloat16 b) noexcept { return float(a)>=float(b); }
static_assert(sizeof(BFloat16)==2 && std::is_trivially_copyable_v<BFloat16>);
}
