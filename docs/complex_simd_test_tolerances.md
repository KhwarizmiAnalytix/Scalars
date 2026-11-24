Complex SIMD test tolerances, 2026-11-24.

[`TestSimdComplex.cxx`](../Testing/Cxx/TestSimdComplex.cxx) holds every
one-, two-, and three-argument math function of the complex packets to the
smallest error budget it meets. This document explains how the budgets are
measured and set, which configurations they were measured on, and the
limitations found while measuring them. The test file is authoritative; the
table below matches it.

## Layout

The file replaces `TestSimdComplex.cxx` and `TestSimdComplexParity.cxx`.

| Suite | Types | Contents |
|---|---|---|
| `simd_complex` | `complex<float>`, `complex<double>`, `complex<half>`, `complex<bfloat16>` | `UnaryMath` (32 checks), `RealValuedFunctions` (8), `BinaryMath` (10), `FusedMultiplyAdd` (fma, with and without cancelling terms), `SetMaskAndSelect` |
| `simd_complex_native` | `complex<float>`, `complex<double>` | Masks, memory layout and half-width helpers, partial loads and stores, accumulation, matrix product, division edge lanes, square-root range boundaries, special values, mixed-lane edges, `Re(log z)` near the unit circle |

`Simd.MatrixPacking` tests the real `simd<T>` types and moved unchanged to
[`TestSimdUtility.cxx`](../Testing/Cxx/TestSimdUtility.cxx). Until this
suite, `complex<half>` and `complex<bfloat16>` packets had no accuracy tests.

## Error measure

Each lane is compared with `std::complex` evaluated in a wider type: `double`
for float, half, and bfloat16, and `long double` for double. `long double` is
128-bit on Linux AArch64 and 80-bit on x86; on Apple and MSVC targets it is
`double`.

The error is normwise: the largest component error divided by
`eps * max(|re|, |im|)` of the reference, where `eps` is the type's spacing at
1.0. A tiny residue next to a large component is therefore not over-weighted,
and a relative-accuracy loss in a small result is not hidden by an absolute
floor. A `tol` entry can set a `floor` (absolute error below that magnitude)
and `zero_sign` (exact zeros, with their sign, where the reference is zero).

The fma check measures error against `|a| |b| + |c|`, because the two real
products round separately even when each component is fused.

Random inputs come from `mt19937` through `genrand_res53` rather than a
standard distribution, so the samples are identical on every standard library
and a budget measured on one platform means the same thing on another.

A failure prints the measured error, the budget, and the inputs that produced
the worst lane, for example:

```text
complex<double>::log on (region{{0.999, 1.001}, {-1e-3, 1e-3}})
  inputs   = (1.00001701731076209079, 2.15431283442193433617e-05)
  simd     = (1.70173980145923840366e-05, 2.15427617410154674993e-05)
  expected = (1.70173980145923989871e-05, 2.15427617410154684869e-05)
  error    = 3.12545335012496128257 ulps (budget 1.7)
```

## Budget policy

- **float and double:** the worst error measured over all configurations
  below, rounded up to two significant digits, and never below 0.5 ULP, since
  a correctly rounded result may be off by that much.
- **half and bfloat16:** 0.51 ULP for every function. These packets compute
  through `complex<float>` and round once, so the budget is half a ULP plus
  the float intermediate's own error, which is far below 0.01 ULP of either
  type. Every measured value was at most 0.4999.
- **Exact operations** (`neg`, `conj`, `ceil`, `floor`, `trunc`, `real`,
  `imag`, the dyadic matrix product, half-width fma): 0 ULP, with signed
  zeros.
- **Double `acos` where `long double` is `double`:** 19 ULPs instead of 1.2.
  The reference there is the standard library's own `std::acos`, and its
  error is part of what is measured. `kWideReference` selects the budget.

Apart from `sqrt`, `acos`, and the packet fast paths (`exp`, `log`, `div`,
`abs`, `arg`, `fma`), the complex math forwards to `std::complex<T>`
([`complex_math.h`](../include/math/complex_math.h)), so the budgets also
bound the standard library. libc++'s float `asin`, `acosh`, `asinh`, and
`tanh` are near 18 ULPs, where libstdc++'s are near 1.

## Measured configurations

| Platform | Backend | Standard library | Double reference | Result |
|---|---|---|---|---|
| macOS arm64 | NEON + Accelerate | libc++ | `double` | 141 passed |
| macOS arm64 | NEON + SLEEF | libc++ | `double` | 141 passed |
| Linux arm64 (GCC 13) | NEON + SLEEF | libstdc++ | 128-bit | 141 passed |
| Linux x86_64 (GCC 13, emulated) | AVX2 + SVML | libstdc++ | 80-bit | 140 passed |
| Linux x86_64 (GCC 13, emulated) | SSE | libstdc++ | 80-bit | 140 passed |

The Linux runs used the `scalars-linux-build` and `scalars-linux-build-amd64`
Docker images. On macOS, `SCALARS_ENABLE_ACCELERATE=OFF` is overridden by
auto-detection, so plain NEON without a vendor library was not measured.

The budgets also cover the GitHub Actions hosts, whose first run raised the
entries the emulated Linux x86 runs had underestimated:

| Platform | Backend | Compiler and build | Raised |
|---|---|---|---|
| Linux x86_64 | AVX2 + SVML | GCC 15, Debug with ASan+UBSan, and coverage | double `expm1`, `exp2`, `log1p`, `div` |
| Linux x86_64 | AVX2 + SVML | GCC 11, Release (CUDA and HIP containers) | double `log1p` |
| Linux x86_64 | AVX2 + SVML | Clang 21, Release | double `mul` |
| Linux x86_64 | AVX-512 | GCC 15, Release | float `sqrt`, `invsqrt`, `reciprocal`, `exp`, `log2`, `log10`, `abs`, `fabs`, `fma`; double `reciprocal`, `expm1`, `exp10`, `log1p`, `sin`, `tan`, `acosh` |
| Windows x86_64 | AVX2 + SVML | MSVC 19.44, Release | double `exp` (narrow) |
| Linux x86_64 | AVX + SVML (`_e9`) | GCC 15, Release | float `exp` (wide), `MixedLaneEdgesAndBranchCuts` float `log` |

Not measured: MKL. Under Valgrind, x87 `long double` arithmetic runs in
double precision, so the `complex<double>` reference is no better than the
result it checks, and glibc's `clogl` can return NaN for tiny arguments. The
test detects this at run time (`1 + LDBL_EPSILON == 1`) and then runs every
`complex<double>` check without asserting its budget.

## Budgets

Values are ULPs for `complex<float>` / `complex<double>`. A double entry
marked "narrow" applies where `long double` is `double` (Apple, MSVC); the
other applies with an 80- or 128-bit reference. Every non-exact entry is 0.51
for half and bfloat16.

| Check | Domain | float | double |
|---|---|---|---|
| `sqrt` | sweep / box(10) | 0.99 / 1.1 | 1.3 / 1.3 |
| `invsqrt` | sweep / box(10) | 1.8 / 1.7 | 2.3 / 2.2 |
| `reciprocal` | sweep / box(10) | 1.3 / 1.3 | 1.1 / 2 |
| `sqr` | box(100) | 0.89 | 0.98 (1.2 narrow) |
| `exp` | re ±8, im ±25 / wide | 2.2 / 2.4 | 2.5 (3.1 narrow) / 2.8 (3.1 narrow) |
| `expm1` | re ±8, im ±25 / box(1e-3) | 2.3 / 2.2 | 1.8 / 1.4 |
| `exp2` | re ±12, im ±25 | 8.8 | 13 |
| `exp10` | re ±3, im ±10 | 12 | 20 |
| `log` | sweep / normal range / near 1 | 0.95 / 1.1 / 1.3 | 0.98 / 0.98 / 1.7 |
| `log1p` | re −0.9..50, im ±50 / box(1e-3) | 1.6 / 2.3 | 1.6 (1.4 narrow) / 2.2 (2.6 narrow) |
| `log2`, `log10` | sweep | 0.83, 1 | 0.92, 1.1 |
| `sin`, `cos`, `tan` | re ±25, im ±8 | 1.6, 1.8, 3.2 | 1.4, 1.3, 2.3 |
| `asin`, `acos`, `atan` | box(4) | 18, 1.2, 6.3 | 1.1, 1.2 (19 narrow), 1.2 |
| `sinh`, `cosh`, `tanh` | re ±8, im ±25 | 1.8, 1.6, 18 | 1.5, 1.5, 2.4 |
| `asinh`, `acosh`, `atanh` | box(4) | 15, 17, 4.6 | 1.1, 1.1, 0.69 |
| `abs`, `fabs` | sweep / full range | 1.2 / 1.1 | 1.5 / 0.88 |
| `abs2`, `norm` | box(100) / root sweep | 0.91 / 0.92 | 1 / 0.98 (0.99 narrow) |
| `arg` | sweep / box(10) | 0.86 / 1.1 | 0.64 / 1 |
| `add`, `sub` | box(40) / sweep | 0.5 | 0.5 |
| `mul` | box(25) / root sweep | 0.89 / 0.91 | 1.1 / 0.57 (0.82 narrow) |
| `div` | box(50) / sweep / full range | 1.7 / 1.4 / 1.4 | 2 (2.1 narrow) / 1.2 / 1.2 (1.7 narrow) |
| `pow` | x: re 0.05..12, im ±3; y: re ±3, im ±1 | 6.6 | 6.7 |
| `fma` | box(25), plain / cancelling | 0.79 / 0.76 | 0.97 / 0.91 (0.96 narrow) |

The double `exp2` and `exp10` budgets reflect the implementation's own
`exp(z * ln2)` and `exp(z * ln10)`: rounding the scaled argument shifts the
phase by up to about 18 ULPs for `|im|` near 10–25.

## Known limitations

- **`log` overflow and subnormals (libc++).** `scalars::log`, the fallback
  for lanes outside the packet fast path, delegates to
  `std::log(std::complex<T>)`, and libc++ forms `|z|` in `T` first:
  - `log(FLT_MAX (1 + i))` returns `+inf` for the real part; the correct
    value is 89.07.
  - Subnormal inputs lose up to about 28,000 ULPs:
    `log|denorm_min (1 - i)|` gives −103.28 instead of −102.93.

  The old tests used a `std::complex<T>` reference with the same flaw, so
  both went unnoticed. `log_reference` holds only these lanes to parity with
  the standard library, and the `log` sweep starts at the smallest normal
  number. Rescaling `z` by a power of two before taking `|z|` would fix both.
- **x86 SLEEF builds do not compile.**
  [`avx/svml.h`](../include/backend/cpu/avx/svml.h) calls
  `scalars::inv_normalcdf` without including `math/normal_cdf.h`, so AVX/AVX2
  builds with `SCALARS_ENABLE_SLEEF=ON` fail on Linux and macOS.
- **macOS x86_64 links Linux SVML.** Without SLEEF, CMake auto-enables SVML
  and links the vendored Linux `.so` files, so the build fails.
- **Emulated x87.** Rosetta and Docker on Apple silicon evaluate `log1pl` to
  only about 2^-64 absolute, which is far from relative accuracy when the
  result is tiny. `ref_log1p_real` sums an `atanh` series for small arguments
  when `long double` is wider than `double`, so the references stay accurate
  under emulation.

## Re-measuring

To find the current worst case of a check, set its `tol` entry to 0 and run
`ScalarsCxxTests --gtest_filter='simd_complex*'`: the failure message prints
the measured error and its inputs. Measure on every available configuration,
then apply the budget policy above.
