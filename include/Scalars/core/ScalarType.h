#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>
namespace kh::scalars {
enum class ScalarType : std::uint8_t { Float16, BFloat16, Float32, Float64, ComplexHalf, ComplexBF16, ComplexFloat, ComplexDouble };
std::string_view scalar_type_name(ScalarType type) noexcept;
std::size_t scalar_type_size(ScalarType type) noexcept;
}
