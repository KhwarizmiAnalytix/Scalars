#include "core/accumulate_type.h"
#include "core/scalar_traits.h"

#include <complex>
#include <cstdint>
#include <type_traits>

#include "scalars_test.h"

using namespace scalars;

namespace
{
template <class T, class Cpu, class Gpu> void expect_accumulate()
{
    static_assert(std::is_same_v<acc_type<T, false>, Cpu>);
    static_assert(std::is_same_v<acc_type<T, true>, Gpu>);
    static_assert(std::is_same_v<acc_type_device<T, device_type::cpu>, Cpu>);
    static_assert(std::is_same_v<acc_type_device<T, device_type::gpu>, Gpu>);
    static_assert(std::is_same_v<typename AccumulateType<T, false>::type, Cpu>);
    static_assert(std::is_same_v<typename AccumulateType<T, true>::type, Gpu>);
}
}  // namespace

TEST(ScalarTraits, ComplexRealAndAccumulation)
{
    static_assert(is_complex_v<complex<double>>);
    static_assert(is_complex_v<std::complex<float>>);
    static_assert(!is_complex_v<float>);
    static_assert(!is_complex<int>::value);
    static_assert(std::is_same_v<real_type_t<complex<half>>, half>);
    static_assert(std::is_same_v<real_type_t<float>, float>);
    static_assert(std::is_same_v<accumulation_type_t<half>, float>);
    static_assert(std::is_same_v<accumulation_type_t<bfloat16>, float>);
    static_assert(std::is_same_v<accumulation_type_t<float>, float>);
    static_assert(std::is_same_v<accumulation_type_t<complex<half>>, complex<float>>);
    static_assert(std::is_same_v<accumulation_type_t<complex<float>>, complex<float>>);
    static_assert(std::is_same_v<scalar_value_type<complex<double>>::type, double>);
    static_assert(std::is_same_v<scalar_value_type<std::complex<float>>::type, float>);
    static_assert(std::is_same_v<scalar_value_type<int>::type, int>);
    EXPECT_TRUE(is_complex_v<complex_double>);
}

TEST(ScalarTraits, AccumulateType)
{
    expect_accumulate<Half, float, float>();
    expect_accumulate<BFloat16, float, float>();
    expect_accumulate<float, double, float>();
    expect_accumulate<double, double, double>();
    expect_accumulate<std::int8_t, std::int64_t, std::int64_t>();
    expect_accumulate<std::uint8_t, std::int64_t, std::int64_t>();
    expect_accumulate<char, std::int64_t, std::int64_t>();
    expect_accumulate<std::int16_t, std::int64_t, std::int64_t>();
    expect_accumulate<std::int32_t, std::int64_t, std::int64_t>();
    expect_accumulate<std::int64_t, std::int64_t, std::int64_t>();
    expect_accumulate<bool, bool, bool>();
    expect_accumulate<complex<Half>, complex<float>, complex<float>>();
    expect_accumulate<complex<float>, complex<double>, complex<float>>();
    expect_accumulate<complex<double>, complex<double>, complex<double>>();

    const scalar_type types[] = {scalar_type::float16, scalar_type::bfloat16, scalar_type::float32,
        scalar_type::float64, scalar_type::complex_half, scalar_type::complex_bf16,
        scalar_type::complex_float, scalar_type::complex_double};
    for (const scalar_type type : types)
    {
        const auto cpu = to_accumulate_type(type, false);
        const auto gpu = to_accumulate_type(type, device_type::gpu);
        EXPECT_EQ(to_accumulate_type(type, true), gpu);
        EXPECT_EQ(to_accumulate_type(type, device_type::cpu), cpu);
        if (type == scalar_type::float32)
        {
            EXPECT_EQ(cpu, scalar_type::float64);
            EXPECT_EQ(gpu, scalar_type::float32);
        }
        if (type == scalar_type::complex_float)
        {
            EXPECT_EQ(cpu, scalar_type::complex_double);
            EXPECT_EQ(gpu, scalar_type::complex_float);
        }
        if (type == scalar_type::float16 || type == scalar_type::bfloat16)
            EXPECT_EQ(cpu, scalar_type::float32);
        if (type == scalar_type::float64 || type == scalar_type::complex_double)
            EXPECT_EQ(gpu, type);
        if (type == scalar_type::complex_half || type == scalar_type::complex_bf16)
            EXPECT_EQ(cpu, scalar_type::complex_float);
    }
}
