#pragma once
#include "common/scalars_export.h"
#include <cstddef>
#include <cstdint>
#include <string_view>
namespace scalars
{
enum class scalar_type : std::uint8_t
{
    float16,
    bfloat16,
    float32,
    float64,
    complex_half,
    complex_bf16,
    complex_float,
    complex_double
};
SCALARS_API std::string_view scalar_type_name(scalar_type type) noexcept;
SCALARS_API std::size_t      scalar_type_size(scalar_type type) noexcept;
}  // namespace scalars
