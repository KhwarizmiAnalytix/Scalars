#include <scalars.h>

int main()
{
    using namespace kh::scalars;
    const half value{1.0f};
    return float(value) == 1.f && scalar_type_size(scalar_type::complex_double) == 16 ? 0 : 1;
}
