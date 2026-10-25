// Unit tests for the GPU simd<T> backend (CUDA or HIP).
//
// Each test group launches a device kernel that exercises a category of simd<T>
// ops on-device, copies the results back to the host, and compares against
// std:: math. Tests are skipped automatically when no GPU device is present
// at runtime. BackendProperties is a compile-time / host-side check that does
// not require a GPU.
#include "scalars_macros.h"

#if SCALARS_HAS_CUDA || SCALARS_HAS_HIP

#if SCALARS_HAS_CUDA
#include <cuda_runtime.h>
#elif SCALARS_HAS_HIP
#include <hip/hip_runtime.h>
#endif

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

// Exercise device dispatch and use the same one-lane types in the host pass.
// Including the CPU dispatcher on the host would redefine simd<float/double>.
#if SCALARS_ON_GPU_DEVICE
#include "backend/simd.h"
#else
#include "backend/gpu/complex/simd.h"
#include "backend/gpu/double/simd.h"
#include "backend/gpu/reduced/simd.h"
#endif
#include "scalars_test.h"
#include "types/complex.h"

// ---- CUDA/HIP runtime API aliases -----------------------------------------
// The two runtimes expose an identical API surface under different names;
// aliasing lets the kernels/tests below be written once for both backends.
#if SCALARS_HAS_CUDA
using gpu_error_t                 = cudaError_t;
constexpr gpu_error_t kGpuSuccess = cudaSuccess;
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuGetErrorString cudaGetErrorString
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetDeviceCount cudaGetDeviceCount
#elif SCALARS_HAS_HIP
using gpu_error_t                 = hipError_t;
constexpr gpu_error_t kGpuSuccess = hipSuccess;
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuGetErrorString hipGetErrorString
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetDeviceCount hipGetDeviceCount
#endif

namespace
{

// ---- CUDA/HIP error helper -------------------------------------------------

#define GPU_CHECK(expr)                                                                            \
    do                                                                                             \
    {                                                                                              \
        gpu_error_t _e = (expr);                                                                   \
        if (_e != kGpuSuccess)                                                                     \
            ADD_FAILURE() << "GPU error: " << gpuGetErrorString(_e) << " at " << __FILE__ << ":"   \
                          << __LINE__;                                                             \
    } while (0)

// ---- RAII device buffer --------------------------------------------------

template <typename T> struct DevBuf
{
    T*     d = nullptr;
    size_t n = 0;

    explicit DevBuf(size_t count) : n(count) { GPU_CHECK(gpuMalloc(&d, n * sizeof(T))); }
    ~DevBuf() { gpuFree(d); }

    void upload(const std::vector<T>& h) const
    {
        GPU_CHECK(gpuMemcpy(d, h.data(), n * sizeof(T), gpuMemcpyHostToDevice));
    }

    std::vector<T> download() const
    {
        std::vector<T> h(n);
        GPU_CHECK(gpuMemcpy(h.data(), d, n * sizeof(T), gpuMemcpyDeviceToHost));
        return h;
    }
};

// ---- Kernels ------------------------------------------------------------

// Arithmetic: add, sub, mul, div, fma(a,b,b) == a*b+b
template <typename T>
__global__ void k_arith(
    const T* a, const T* b, T* r_add, T* r_sub, T* r_mul, T* r_div, T* r_fma, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    T av = simd<T>::loadu(a + i), bv = simd<T>::loadu(b + i);
    simd<T>::storeu(simd<T>::add(av, bv), r_add + i);
    simd<T>::storeu(simd<T>::sub(av, bv), r_sub + i);
    simd<T>::storeu(simd<T>::mul(av, bv), r_mul + i);
    simd<T>::storeu(simd<T>::div(av, bv), r_div + i);
    simd<T>::storeu(simd<T>::fma(av, bv, bv), r_fma + i);
}

// Unary: sqrt(|a|), sqr, fabs, neg, ceil, floor, trunc
template <typename T>
__global__ void k_unary(const T* a,
    T*                           r_sqrt,
    T*                           r_sqr,
    T*                           r_fabs,
    T*                           r_neg,
    T*                           r_ceil,
    T*                           r_floor,
    T*                           r_trunc,
    size_t                       n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    T av = simd<T>::loadu(a + i);
    simd<T>::storeu(simd<T>::sqrt(simd<T>::fabs(av)), r_sqrt + i);
    simd<T>::storeu(simd<T>::sqr(av), r_sqr + i);
    simd<T>::storeu(simd<T>::fabs(av), r_fabs + i);
    simd<T>::storeu(simd<T>::neg(av), r_neg + i);
    simd<T>::storeu(simd<T>::ceil(av), r_ceil + i);
    simd<T>::storeu(simd<T>::floor(av), r_floor + i);
    simd<T>::storeu(simd<T>::trunc(av), r_trunc + i);
}

// Exp / log: exp, log(|a|+eps), exp2, log2, cbrt
template <typename T>
__global__ void k_explog(const T* a, T* r_exp, T* r_log, T* r_exp2, T* r_log2, T* r_cbrt, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    T av  = simd<T>::loadu(a + i);
    T pos = simd<T>::fabs(av) + static_cast<T>(0.1);
    simd<T>::storeu(simd<T>::exp(av), r_exp + i);
    simd<T>::storeu(simd<T>::log(pos), r_log + i);
    simd<T>::storeu(simd<T>::exp2(av), r_exp2 + i);
    simd<T>::storeu(simd<T>::log2(pos), r_log2 + i);
    simd<T>::storeu(simd<T>::cbrt(av), r_cbrt + i);
}

// Trig: sin, cos, tanh
template <typename T> __global__ void k_trig(const T* a, T* r_sin, T* r_cos, T* r_tanh, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    T av = simd<T>::loadu(a + i);
    simd<T>::storeu(simd<T>::sin(av), r_sin + i);
    simd<T>::storeu(simd<T>::cos(av), r_cos + i);
    simd<T>::storeu(simd<T>::tanh(av), r_tanh + i);
}

// Comparisons + identity reductions (size==1 -> accumulate/hmin/hmax are pass-through)
template <typename T>
__global__ void k_cmp_reduce(
    const T* a, const T* b, T* r_lt, T* r_eq_self, T* r_acc, T* r_hmin, T* r_hmax, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    T av = simd<T>::loadu(a + i), bv = simd<T>::loadu(b + i);
    r_lt[i]      = simd<T>::lt(av, bv) ? static_cast<T>(1) : static_cast<T>(0);
    r_eq_self[i] = simd<T>::eq(av, av) ? static_cast<T>(1) : static_cast<T>(0);
    simd<T>::storeu(simd<T>::accumulate(av), r_acc + i);
    simd<T>::storeu(simd<T>::hmin(av), r_hmin + i);
    simd<T>::storeu(simd<T>::hmax(av), r_hmax + i);
}

// ---- Templated test body -------------------------------------------------

template <typename T> void run_gpu_simd_test()
{
    constexpr size_t N    = 512;
    constexpr double kRel = std::is_same_v<T, scalars::bfloat16> ? 2e-2
                            : std::is_same_v<T, scalars::half>   ? 2e-3
                            : std::is_same_v<T, float>           ? 5e-5
                                                                 : 1e-11;
    constexpr double kAbs = kRel;

    // a in [-2, 2), b in [0.5, 1.5) -- chosen so neither is zero and lt has a mix
    std::vector<T> ha(N), hb(N);
    for (size_t i = 0; i < N; ++i)
    {
        double t = static_cast<double>(i) / static_cast<double>(N);
        ha[i]    = static_cast<T>(t * 4.0 - 2.0);
        hb[i]    = static_cast<T>(t + 0.5);
    }

    DevBuf<T> da(N), db(N);
    da.upload(ha);
    db.upload(hb);

    constexpr unsigned kBlock = 128u;
    const unsigned     kGrid  = static_cast<unsigned>((N + kBlock - 1u) / kBlock);

    // ---- Arithmetic -------------------------------------------------------
    {
        DevBuf<T> r_add(N), r_sub(N), r_mul(N), r_div(N), r_fma(N);
        k_arith<<<kGrid, kBlock>>>(da.d, db.d, r_add.d, r_sub.d, r_mul.d, r_div.d, r_fma.d, N);
        GPU_CHECK(gpuDeviceSynchronize());

        auto vadd = r_add.download(), vsub = r_sub.download(), vmul = r_mul.download(),
             vdiv = r_div.download(), vfma = r_fma.download();

        for (size_t i = 0; i < N; ++i)
        {
            double a = ha[i], b = hb[i];
            EXPECT_NEAR(vadd[i], a + b, std::abs(a + b) * kRel + kAbs);
            EXPECT_NEAR(vsub[i], a - b, std::abs(a - b) * kRel + kAbs);
            EXPECT_NEAR(vmul[i], a * b, std::abs(a * b) * kRel + kAbs);
            EXPECT_NEAR(vdiv[i], a / b, std::abs(a / b) * kRel + kAbs);
            EXPECT_NEAR(vfma[i], a * b + b, std::abs(a * b + b) * kRel + kAbs);
        }
    }

    // ---- Unary ------------------------------------------------------------
    {
        DevBuf<T> r_sqrt(N), r_sqr(N), r_fabs(N), r_neg(N), r_ceil(N), r_floor(N), r_trunc(N);
        k_unary<<<kGrid, kBlock>>>(
            da.d, r_sqrt.d, r_sqr.d, r_fabs.d, r_neg.d, r_ceil.d, r_floor.d, r_trunc.d, N);
        GPU_CHECK(gpuDeviceSynchronize());

        auto vsqrt = r_sqrt.download(), vsqr = r_sqr.download();
        auto vfabs = r_fabs.download(), vneg = r_neg.download();
        auto vceil = r_ceil.download(), vfloor = r_floor.download();
        auto vtrunc = r_trunc.download();

        for (size_t i = 0; i < N; ++i)
        {
            double a = ha[i];
            EXPECT_NEAR(vsqrt[i], std::sqrt(std::fabs(a)), kAbs);
            EXPECT_NEAR(vsqr[i], a * a, std::abs(a * a) * kRel + kAbs);
            EXPECT_NEAR(vfabs[i], std::fabs(a), kAbs);
            EXPECT_NEAR(vneg[i], -a, kAbs);
            EXPECT_NEAR(vceil[i], std::ceil(a), kAbs);
            EXPECT_NEAR(vfloor[i], std::floor(a), kAbs);
            EXPECT_NEAR(vtrunc[i], std::trunc(a), kAbs);
        }
    }

    // ---- Exp / log --------------------------------------------------------
    {
        DevBuf<T> r_exp(N), r_log(N), r_exp2(N), r_log2(N), r_cbrt(N);
        k_explog<<<kGrid, kBlock>>>(da.d, r_exp.d, r_log.d, r_exp2.d, r_log2.d, r_cbrt.d, N);
        GPU_CHECK(gpuDeviceSynchronize());

        auto vexp = r_exp.download(), vlog = r_log.download();
        auto vexp2 = r_exp2.download(), vlog2 = r_log2.download();
        auto vcbrt = r_cbrt.download();

        for (size_t i = 0; i < N; ++i)
        {
            double a = ha[i], pos = std::fabs(a) + 0.1;
            EXPECT_NEAR(vexp[i], std::exp(a), std::exp(a) * kRel + kAbs);
            EXPECT_NEAR(vlog[i], std::log(pos), std::abs(std::log(pos)) * kRel + kAbs);
            EXPECT_NEAR(vexp2[i], std::exp2(a), std::exp2(a) * kRel + kAbs);
            EXPECT_NEAR(vlog2[i], std::log2(pos), std::abs(std::log2(pos)) * kRel + kAbs);
            EXPECT_NEAR(vcbrt[i], std::cbrt(a), kAbs);
        }
    }

    // ---- Trig -------------------------------------------------------------
    {
        DevBuf<T> r_sin(N), r_cos(N), r_tanh(N);
        k_trig<<<kGrid, kBlock>>>(da.d, r_sin.d, r_cos.d, r_tanh.d, N);
        GPU_CHECK(gpuDeviceSynchronize());

        auto vsin = r_sin.download(), vcos = r_cos.download(), vtanh = r_tanh.download();

        for (size_t i = 0; i < N; ++i)
        {
            double a = ha[i];
            EXPECT_NEAR(vsin[i], std::sin(a), kAbs);
            EXPECT_NEAR(vcos[i], std::cos(a), kAbs);
            EXPECT_NEAR(vtanh[i], std::tanh(a), kAbs);
        }
    }

    // ---- Comparisons and reductions ---------------------------------------
    {
        DevBuf<T> r_lt(N), r_eq(N), r_acc(N), r_hmin(N), r_hmax(N);
        k_cmp_reduce<<<kGrid, kBlock>>>(da.d, db.d, r_lt.d, r_eq.d, r_acc.d, r_hmin.d, r_hmax.d, N);
        GPU_CHECK(gpuDeviceSynchronize());

        auto vlt = r_lt.download(), veq = r_eq.download();
        auto vacc = r_acc.download(), vhmin = r_hmin.download(), vhmax = r_hmax.download();

        for (size_t i = 0; i < N; ++i)
        {
            EXPECT_EQ(vlt[i] != static_cast<T>(0), ha[i] < hb[i]) << "lt at i=" << i;
            EXPECT_EQ(veq[i], static_cast<T>(1)) << "eq(x,x) must be true at i=" << i;
            EXPECT_EQ(vacc[i], ha[i]) << "accumulate is identity for size=1, i=" << i;
            EXPECT_EQ(vhmin[i], ha[i]) << "hmin is identity for size=1, i=" << i;
            EXPECT_EQ(vhmax[i], ha[i]) << "hmax is identity for size=1, i=" << i;
        }
    }
}

// ---- Complex arithmetic / math kernels -----------------------------------
// simd<complex<T>> on GPU is scalar (size == 1), same as simd<T>; there is no
// ordering (lt/hmin/hmax/etc. are deleted), so this does not reuse
// run_gpu_simd_test's generic template -- it is its own test body.

template <typename T>
__global__ void k_complex_arith(
    const T* a, const T* b, T* r_add, T* r_sub, T* r_mul, T* r_div, T* r_fma, T* r_conj, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    T av = simd<T>::loadu(a + i), bv = simd<T>::loadu(b + i);
    simd<T>::storeu(simd<T>::add(av, bv), r_add + i);
    simd<T>::storeu(simd<T>::sub(av, bv), r_sub + i);
    simd<T>::storeu(simd<T>::mul(av, bv), r_mul + i);
    simd<T>::storeu(simd<T>::div(av, bv), r_div + i);
    simd<T>::storeu(simd<T>::fma(av, bv, bv), r_fma + i);
    simd<T>::storeu(simd<T>::conj(av), r_conj + i);
}

template <typename T>
__global__ void k_complex_math(const T* a,
    T*                                  r_sqrt,
    T*                                  r_exp,
    T*                                  r_log,
    typename T::value_type*             r_abs,
    typename T::value_type*             r_arg,
    size_t                              n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    T av = simd<T>::loadu(a + i);
    simd<T>::storeu(simd<T>::sqrt(simd<T>::mul(av, av)), r_sqrt + i);
    simd<T>::storeu(simd<T>::exp(av), r_exp + i);
    simd<T>::storeu(simd<T>::log(simd<T>::exp(av)), r_log + i);
    r_abs[i] = simd<T>::abs(av);
    r_arg[i] = simd<T>::arg(av);
}

// Comparisons + identity accumulate (size==1 -> accumulate is pass-through)
template <typename T>
__global__ void k_complex_cmp_reduce(const T* a, uint32_t* r_eq_self, T* r_acc, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    T av           = simd<T>::loadu(a + i);
    r_eq_self[i]   = simd<T>::eq(av, av) != 0 ? 1u : 0u;
    const auto acc = simd<T>::accumulate(av);
    r_acc[i]       = T(static_cast<typename T::value_type>(acc.real()),
        static_cast<typename T::value_type>(acc.imag()));
}

template <typename T> void run_gpu_complex_simd_test()
{
    using component_t     = typename T::value_type;
    constexpr size_t N    = 512;
    constexpr double kRel = std::is_same_v<component_t, float> ? 5e-5 : 1e-11;
    constexpr double kAbs = kRel;

    // Real part stays positive so sqrt(a*a) == a on the principal branch.
    std::vector<T>                    ha(N), hb(N);
    std::vector<std::complex<double>> ra(N), rb(N);
    for (size_t i = 0; i < N; ++i)
    {
        double t  = static_cast<double>(i) / static_cast<double>(N);
        double re = t * 2.0 + 0.5, im = t * 4.0 - 2.0;
        double rb_re = t + 0.5, rb_im = 1.0 - t;
        ha[i] = T(static_cast<component_t>(re), static_cast<component_t>(im));
        hb[i] = T(static_cast<component_t>(rb_re), static_cast<component_t>(rb_im));
        ra[i] = {re, im};
        rb[i] = {rb_re, rb_im};
    }

    DevBuf<T> da(N), db(N);
    da.upload(ha);
    db.upload(hb);

    constexpr unsigned kBlock = 128u;
    const unsigned     kGrid  = static_cast<unsigned>((N + kBlock - 1u) / kBlock);

    auto expect_near = [&](const T& got, std::complex<double> expect, const char* what, size_t i)
    {
        double got_re = static_cast<double>(got.real()), got_im = static_cast<double>(got.imag());
        double scale = std::max({1.0, std::abs(expect.real()), std::abs(expect.imag())});
        EXPECT_NEAR(got_re, expect.real(), scale * kRel + kAbs) << what << " real at i=" << i;
        EXPECT_NEAR(got_im, expect.imag(), scale * kRel + kAbs) << what << " imag at i=" << i;
    };

    // ---- Arithmetic -------------------------------------------------------
    {
        DevBuf<T> r_add(N), r_sub(N), r_mul(N), r_div(N), r_fma(N), r_conj(N);
        k_complex_arith<<<kGrid, kBlock>>>(
            da.d, db.d, r_add.d, r_sub.d, r_mul.d, r_div.d, r_fma.d, r_conj.d, N);
        GPU_CHECK(gpuDeviceSynchronize());

        auto vadd = r_add.download(), vsub = r_sub.download(), vmul = r_mul.download(),
             vdiv = r_div.download(), vfma = r_fma.download(), vconj = r_conj.download();

        for (size_t i = 0; i < N; ++i)
        {
            expect_near(vadd[i], ra[i] + rb[i], "add", i);
            expect_near(vsub[i], ra[i] - rb[i], "sub", i);
            expect_near(vmul[i], ra[i] * rb[i], "mul", i);
            expect_near(vdiv[i], ra[i] / rb[i], "div", i);
            expect_near(vfma[i], ra[i] * rb[i] + rb[i], "fma", i);
            expect_near(vconj[i], std::conj(ra[i]), "conj", i);
        }
    }

    // ---- Roots, exp/log, magnitude/argument --------------------------------
    {
        DevBuf<T>           r_sqrt(N), r_exp(N), r_log(N);
        DevBuf<component_t> r_abs(N), r_arg(N);
        k_complex_math<<<kGrid, kBlock>>>(da.d, r_sqrt.d, r_exp.d, r_log.d, r_abs.d, r_arg.d, N);
        GPU_CHECK(gpuDeviceSynchronize());

        auto vsqrt = r_sqrt.download(), vexp = r_exp.download(), vlog = r_log.download();
        auto vabs = r_abs.download(), varg = r_arg.download();

        for (size_t i = 0; i < N; ++i)
        {
            expect_near(vsqrt[i], ra[i], "sqrt(a*a)", i);
            expect_near(vexp[i], std::exp(ra[i]), "exp", i);
            expect_near(vlog[i], ra[i], "log(exp(a))", i);
            EXPECT_NEAR(
                static_cast<double>(vabs[i]), std::abs(ra[i]), std::abs(ra[i]) * kRel + kAbs)
                << "abs at i=" << i;
            EXPECT_NEAR(static_cast<double>(varg[i]), std::arg(ra[i]), kAbs) << "arg at i=" << i;
        }
    }

    // ---- Comparisons and reduction -----------------------------------------
    {
        DevBuf<uint32_t> r_eq(N);
        DevBuf<T>        r_acc(N);
        k_complex_cmp_reduce<<<kGrid, kBlock>>>(da.d, r_eq.d, r_acc.d, N);
        GPU_CHECK(gpuDeviceSynchronize());

        auto veq  = r_eq.download();
        auto vacc = r_acc.download();
        for (size_t i = 0; i < N; ++i)
        {
            EXPECT_EQ(veq[i], 1u) << "eq(x,x) must be true at i=" << i;
            expect_near(vacc[i], ra[i], "accumulate is identity for size=1", i);
        }
    }
}

// Raw encodings exercise subnormals, signed zeros, infinities, and every NaN
// payload. Memory/sign/selection operations must not round these values.
template <class T>
__global__ void k_reduced_encoding_contract(
    const T* input, T* output, uint16_t* masks, float* widened)
{
    using S     = simd<T>;
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= 65536)
        return;
    const auto x        = S::gather(input, i);
    const int  offset[] = {i};
    S::scatter(S::loadu(input + i), offset, output);
    S::scatter(S::gather(input, offset), i, output + 65536);
    S::storeu(S::neg(x), output + 2 * 65536 + i);
    S::storeu(S::fabs(x), output + 3 * 65536 + i);
    S::storeu(S::signcopy(x, S::set(-1.f)), output + 4 * 65536 + i);
    S::storeu(S::if_else(S::eq(S::set(1.f), S::set(1.f)), x, S::setzero()), output + 5 * 65536 + i);
    S::storeu(S::if_else(S::not_mask(S::eq(S::set(1.f), S::set(1.f))), S::setzero(), x),
        output + 6 * 65536 + i);
    S::storeu(S::loadu(input + i, 1), output + 7 * 65536 + i, 1);
    // Count zero permits a null pointer and must leave the destination alone.
    S::storeu(S::loadu(static_cast<const T*>(nullptr), 0), static_cast<T*>(nullptr), 0);
    S::storeu(S::eq(x, x), masks + i);
    widened[i] = S::accumulate(x);
}

template <class T>
__global__ void k_reduced_rounding_contract(T* out, uint16_t* masks, float* widened)
{
    using S         = simd<T>;
    const float eps = std::is_same_v<T, scalars::half> ? 0x1p-10f : 0x1p-7f;
    const auto  a = S::set(1.f + eps), b = S::set(1.f - eps), c = S::set(-1.f);
    S::storeu(S::fma(a, b, c), out);
    S::storeu(S::add(S::mul(a, b), c), out + 1);
    widened[0] = S::to_float(a) * S::to_float(b) + S::to_float(c);
    S::store_float(widened[0], out + 2);
    const auto zero = S::setzero(), negzero = S::neg(zero);
    S::storeu(S::min(zero, negzero), out + 3);
    S::storeu(S::min(negzero, zero), out + 4);
    S::storeu(S::max(zero, negzero), out + 5);
    S::storeu(S::max(negzero, zero), out + 6);
    const T nan(uint16_t(std::is_same_v<T, scalars::half> ? 0x7e15 : 0x7fc5), T::from_bits());
    S::storeu(S::min(nan, zero), out + 7);
    S::storeu(S::max(zero, nan), out + 8);
    S::storeu(S::hmin(nan), out + 9);
    S::storeu(S::hmax(nan), out + 10);
    S::storeu(S::loadu(static_cast<const T*>(nullptr), 0), out + 11);
    S::copy(&nan, out + 12);
    const auto t = S::lt(c, a), f = S::ge(c, a);
    masks[0] = t;
    masks[1] = f;
    masks[2] = S::le(a, a);
    masks[3] = S::gt(a, c);
    masks[4] = S::neq(nan, nan);
    masks[5] = S::eq(nan, nan);
    masks[6] = S::and_mask(t, S::not_mask(f));
    masks[7] = S::or_mask(t, f);
    masks[8] = S::xor_mask(t, f);
}

template <class T> void run_reduced_contract_test()
{
    constexpr size_t n = 65536;
    std::vector<T>   values(n);
    for (size_t i = 0; i < n; ++i)
        values[i] = T(uint16_t(i), T::from_bits());
    DevBuf<T>        input(n), output(8 * n);
    DevBuf<uint16_t> masks(n);
    DevBuf<float>    widened(n);
    input.upload(values);
    k_reduced_encoding_contract<<<512, 128>>>(input.d, output.d, masks.d, widened.d);
    GPU_CHECK(gpuDeviceSynchronize());
    const auto result = output.download();
    const auto mask   = masks.download();
    const auto fp32   = widened.download();
    for (size_t i = 0; i < n; ++i)
    {
        for (int block : {0, 1, 5, 6, 7})
            ASSERT_EQ(result[block * n + i].x, i);
        ASSERT_EQ(result[2 * n + i].x, uint16_t(i ^ 0x8000));
        ASSERT_EQ(result[3 * n + i].x, uint16_t(i & 0x7fff));
        ASSERT_EQ(result[4 * n + i].x, uint16_t(i | 0x8000));
        const float expected = static_cast<float>(values[i]);
        ASSERT_EQ(mask[i], std::isnan(expected) ? 0 : 0xffff);
        if (std::isnan(expected))
            ASSERT_TRUE(std::isnan(fp32[i]));
        else
        {
            ASSERT_EQ(fp32[i], expected);
            if (expected == 0)
                ASSERT_EQ(std::signbit(fp32[i]), std::signbit(expected));
        }
    }
    k_reduced_rounding_contract<T><<<1, 1>>>(output.d, masks.d, widened.d);
    GPU_CHECK(gpuDeviceSynchronize());
    const auto  rounded = output.download();
    const auto  cmp     = masks.download();
    const float eps     = std::is_same_v<T, scalars::half> ? 0x1p-10f : 0x1p-7f;
    EXPECT_EQ(rounded[0].x, T(-eps * eps).x);
    EXPECT_EQ(rounded[1].x, 0);
    EXPECT_EQ(rounded[2].x, rounded[0].x);
    EXPECT_EQ(widened.download()[0], -eps * eps);
    EXPECT_EQ(rounded[3].x, 0x8000);
    EXPECT_EQ(rounded[4].x, 0x8000);
    EXPECT_EQ(rounded[5].x, 0);
    EXPECT_EQ(rounded[6].x, 0);
    for (int i : {7, 8, 9, 10, 12})
        EXPECT_EQ(rounded[i].x, (std::is_same_v<T, scalars::half> ? 0x7e15 : 0x7fc5));
    EXPECT_EQ(rounded[11].x, 0);
    for (int i : {0, 2, 3, 4, 6, 7, 8})
        EXPECT_EQ(cmp[i], 0xffff);
    for (int i : {1, 5})
        EXPECT_EQ(cmp[i], 0);
}

}  // namespace

// --------------------------------------------------------------------------
// Compile-time and host-side property checks -- no GPU device required.
// --------------------------------------------------------------------------
TEST(GpuSimd, BackendProperties)
{
    static_assert(simd<float>::size == 1, "GPU simd<float>::size must be 1");
    static_assert(simd<double>::size == 1, "GPU simd<double>::size must be 1");
    static_assert(simd<scalars::half>::size == 1);
    static_assert(simd<scalars::bfloat16>::size == 1);
    static_assert(sizeof(simd<scalars::half>::simd_t) == 2);
    static_assert(sizeof(simd<scalars::bfloat16>::simd_t) == 2);
    static_assert(
        std::is_same_v<decltype(simd<scalars::half>::accumulate(scalars::half{})), float>);
    static_assert(
        std::is_same_v<decltype(simd<scalars::bfloat16>::accumulate(scalars::bfloat16{})), float>);
    EXPECT_EQ(simd<float>::size, 1);
    EXPECT_EQ(simd<double>::size, 1);
}

// --------------------------------------------------------------------------
// On-device arithmetic and math tests
// --------------------------------------------------------------------------
TEST(GpuSimd, ArithmeticAndMathFloat)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device -- skipping GpuSimd float test";
    run_gpu_simd_test<float>();
}

TEST(GpuSimd, ArithmeticAndMathDouble)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device -- skipping GpuSimd double test";
    run_gpu_simd_test<double>();
}

TEST(GpuSimd, ArithmeticAndMathHalf)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";
    run_gpu_simd_test<scalars::half>();
    run_reduced_contract_test<scalars::half>();
}

TEST(GpuSimd, ArithmeticAndMathBFloat16)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";
    run_gpu_simd_test<scalars::bfloat16>();
    run_reduced_contract_test<scalars::bfloat16>();
}

TEST(GpuSimd, ArithmeticAndMathComplexFloat)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device -- skipping GpuSimd complex<float> test";
    run_gpu_complex_simd_test<scalars::complex<float>>();
}

TEST(GpuSimd, ArithmeticAndMathComplexDouble)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device -- skipping GpuSimd complex<double> test";
    run_gpu_complex_simd_test<scalars::complex<double>>();
}

#endif  // SCALARS_HAS_CUDA || SCALARS_HAS_HIP
