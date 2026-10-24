#include <types/half.h>

int main()
{
    return float(kh::scalars::half{1.0f}) == 1.f ? 0 : 1;
}
