#include <Scalars/Scalars.h>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <iostream>
using namespace kh::scalars;
int main() {
 static_assert(sizeof(Half)==2 && sizeof(BFloat16)==2);
 static_assert(sizeof(ComplexFloat)==8 && sizeof(ComplexDouble)==16);
 static_assert(is_complex_v<ComplexDouble>);
 static_assert(std::is_same_v<promote_t<Half,BFloat16>,float>);
 static_assert(std::is_same_v<promote_t<Complex<Half>,double>,Complex<double>>);
 assert(Half(1.0f).to_bits()==0x3c00);
 assert(Half(65504.0f).to_bits()==0x7bff);
 assert(Half(0.000000059604644775390625f).to_bits()==0x0001);
 assert(Half::from_bits(0x8000).to_bits()==0x8000);
 assert(BFloat16(1.0f).to_bits()==0x3f80);
 assert(BFloat16::from_bits(0x3f80).operator float()==1.0f);
 assert(Half(1.0f)+Half(2.0f)==Half(3.0f));
 assert(BFloat16(2.0f)*BFloat16(3.0f)==BFloat16(6.0f));
 assert((ComplexFloat{1,2}*ComplexFloat{3,4})==(ComplexFloat{-5,10}));
 auto z=kh::scalars::exp(ComplexDouble{0.,1.});
 assert(std::abs(z.real()-std::cos(1.))<1e-14);
 assert(std::abs(z.imag()-std::sin(1.))<1e-14);
 assert(std::abs(kh::scalars::sqrt(ComplexDouble{-1.,0.}).imag()-1.)<1e-14);
 auto casted=scalar_cast<ComplexDouble>(ComplexFloat{1,2});
 assert(casted.real()==1. && casted.imag()==2.);
 assert(scalar_type_size(ScalarType::ComplexDouble)==16);
 assert(scalar_type_name(ScalarType::BFloat16)=="bfloat16");
 // Every possible 16-bit half encoding has a consistent representation round-trip.
 for (std::uint32_t i=0;i<65536;++i) {
   auto h=Half::from_bits(static_cast<std::uint16_t>(i));
   auto r=Half(static_cast<float>(h));
   if (((i & 0x7c00u)==0x7c00u) && (i & 0x03ffu)) assert(std::isnan(float(r)));
   else assert(r.to_bits()==i);
 }
 std::cout << "Scalars tests passed (including 65,536 Half patterns)\n";
}
