#include "common/scalar_type.h"
namespace scalars
{
std::string_view scalar_type_name(scalar_type t) noexcept
{
    switch (t)
    {
    case scalar_type::float16:
        return "float16";
    case scalar_type::bfloat16:
        return "bfloat16";
    case scalar_type::float32:
        return "float32";
    case scalar_type::float64:
        return "float64";
    case scalar_type::complex_half:
        return "complex_half";
    case scalar_type::complex_bf16:
        return "complex_bfloat16";
    case scalar_type::complex_float:
        return "complex64";
    case scalar_type::complex_double:
        return "complex128";
    }
    return "unknown";
}
std::size_t scalar_type_size(scalar_type t) noexcept
{
    switch (t)
    {
    case scalar_type::float16:
    case scalar_type::bfloat16:
        return 2;
    case scalar_type::float32:
    case scalar_type::complex_half:
    case scalar_type::complex_bf16:
        return 4;
    case scalar_type::float64:
    case scalar_type::complex_float:
        return 8;
    case scalar_type::complex_double:
        return 16;
    }
    return 0;
}
}  // namespace scalars
