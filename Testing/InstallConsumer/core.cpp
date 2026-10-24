#include "math/complex_vector.h"
#include "types/half.h"

int main()
{
    using c             = scalars::complex<double>;
    const c    values[] = {c(1, 2), c(3, -4)};
    const auto dot      = scalars::complex_dot(values, 1, values, 1, 2, true);
    return float(scalars::half{1.0f}) == 1.f && dot.real() == 30 && dot.imag() == 0 ? 0 : 1;
}
