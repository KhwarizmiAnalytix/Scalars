#include <core/scalar_traits.h>

#include "scalars_test.h"

using namespace kh::scalars;

TEST(scalar_traits, complex_type)
{
    static_assert(is_complex_v<complex_double>);
}
