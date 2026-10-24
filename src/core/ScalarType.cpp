#include <Scalars/core/ScalarType.h>
namespace kh::scalars {
std::string_view scalar_type_name(ScalarType t) noexcept {
 switch(t) {
 case ScalarType::Float16:return "float16";case ScalarType::BFloat16:return "bfloat16";
 case ScalarType::Float32:return "float32";case ScalarType::Float64:return "float64";
 case ScalarType::ComplexHalf:return "complex_half";case ScalarType::ComplexBF16:return "complex_bfloat16";
 case ScalarType::ComplexFloat:return "complex64";case ScalarType::ComplexDouble:return "complex128";
 } return "unknown";
}
std::size_t scalar_type_size(ScalarType t) noexcept {
 switch(t) {
 case ScalarType::Float16:case ScalarType::BFloat16:return 2;
 case ScalarType::Float32:case ScalarType::ComplexHalf:case ScalarType::ComplexBF16:return 4;
 case ScalarType::Float64:case ScalarType::ComplexFloat:return 8;
 case ScalarType::ComplexDouble:return 16;
 } return 0;
}
}
