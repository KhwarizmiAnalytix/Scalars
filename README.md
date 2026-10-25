# Scalars

Standalone C++20 scalar library for **half**, **bfloat16**, and **complex<T>**, with dtype metadata, traits, promotion rules, casts, and standard-library-backed math functions. Inspired by the *architecture* of PyTorch `c10`, but independently implemented; no PyTorch source files are copied.

## Build

Requires CMake 3.24 or newer and a C++20 compiler. The build and lint setup
follows the sibling LinearAlgebra repository, with `SCALARS_` options and
the configuration summary in `cmake/summary.cmake`.

```bash
git submodule update --init ThirdParty/googletest ThirdParty/benchmark
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix /your/install/prefix
```

Consumers can use `find_package(Scalars CONFIG REQUIRED)` and link `Scalars::Scalars`.

Headers live directly under `include/`: use `<scalars.h>` for the full library,
or individual headers such as `<types/half.h>` and `<core/scalar_type.h>`.

The default standalone build produces a shared library. Use
`-DBUILD_SHARED_LIBS=OFF` for a static library. `add_subdirectory()` consumers
inherit the host's shared/static choice and do not build tests by default.

Common build options follow LinearAlgebra's naming and defaults:

| Option | Default | Purpose |
| --- | --- | --- |
| `SCALARS_CXX_STANDARD` | `20` | C++20 or C++23 |
| `SCALARS_ENABLE_TESTING` | `ON` standalone | Build the scalar test suite |
| `SCALARS_ENABLE_GTEST` | `ON` | Build GoogleTest C++ unit tests |
| `SCALARS_ENABLE_BENCHMARK` | `OFF` | Build Google Benchmark micro-benchmarks |
| `SCALARS_ENABLE_CACHE` | `ON` | Enable the selected compiler cache |
| `SCALARS_CACHE_BACKEND` | `none` | `none`, `ccache`, `sccache`, or `buildcache` |
| `SCALARS_LTO_MODE` | `off` | `off`, `thin`, `full`, `ipo`, or `auto` |
| `SCALARS_LINKER_CHOICE` | `default` | `default`, `lld`, `mold`, `gold`, or `lld-link` |
| `SCALARS_ENABLE_SANITIZER` | `OFF` | Instrument the library and tests |
| `SCALARS_SANITIZER_TYPE` | `address` | Sanitizer name or comma-separated combination |
| `SCALARS_ENABLE_COVERAGE` | `OFF` | Instrument the library and tests for coverage |
| `SCALARS_ENABLE_CLANGTIDY` | `OFF` | Run clang-tidy during compilation |
| `SCALARS_ENABLE_FIX` | `OFF` | Allow clang-tidy to apply fixes |
| `SCALARS_ENABLE_IWYU` | `OFF` | Run include-what-you-use |
| `SCALARS_ENABLE_SPELL` | `OFF` | Run codespell during the build |
| `SCALARS_ENABLE_VALGRIND` | `OFF` | Configure CTest memcheck and test timeouts |
| `SCALARS_ENABLE_ICECC` | `OFF` | Use Icecream distributed compilation |

`BUILD_TESTING=OFF` disables tests globally. The original
`SCALARS_BUILD_TESTS` switch still supplies the default for
`SCALARS_ENABLE_TESTING` when configuring an existing consumer.
Coverage and sanitizers skip LTO, as in LinearAlgebra.

The Python build helper uses the same dotted commands as LinearAlgebra:

```bash
python3 Scripts/setup.py --help
python3 Scripts/setup.py config.build.test
python3 Scripts/setup.py config.build.test.static.cxx23
python3 Scripts/setup.py config.build.test --sanitizer.address
python3 Scripts/setup.py config.build.test --lto.ipo
python3 Scripts/setup.py config.build.test.cppcheck
python3 Scripts/setup.py config.build.test.coverage
```

Run it from the repository root, or use `python3 setup.py ...` from `Scripts/`.
It defaults to Ninja, Clang, C++20, and Release, and creates `build_ninja/`
with suffixes for selected options. `config` reuses the existing build directory;
`build.test` reruns an already configured build. Compiler/generator tokens such
as `gcc`, `clang-18`, `xcode`, and `vs22` follow the reference CLI.
Sanitizer, coverage, and Valgrind builds use Debug. Optional analysis tools must
be installed separately; coverage reporting requires `coverage-tool` and uses
`coverage.toml`. `colorama` and `psutil` are optional.

Tests follow LinearAlgebra's structure, with CamelCase source filenames:

- `Testing/Cxx/Test*.cxx`: GoogleTest cases grouped by scalar type, math,
  traits, promotion, conversion, and metadata, built as `ScalarsCxxTests`.
- `Testing/Benchmark/Benchmark*.cxx`: Google Benchmark micro-benchmarks, built
  when `SCALARS_ENABLE_BENCHMARK=ON` (setup helper token `benchmark`).
- `Testing/scalars_test.h.in`: the shared test header configured by CMake.
- `Testing/CMake/test_setup.py`: Python build-helper regression checks,
  registered as `ScalarsSetupTests` when Python is available.
- `Testing/InstallConsumer/`: standalone consumers for the installed
  `Scalars::Scalars` target.

`ctest --test-dir build --output-on-failure` runs the C++ and Python suites.
GoogleTest and Google Benchmark are the `ThirdParty/googletest` and
`ThirdParty/benchmark` submodules, pinned to the same commits as LinearAlgebra.
Initialize them with `git submodule update --init`. A missing checkout falls
back to FetchContent. Library-only builds with `SCALARS_ENABLE_TESTING=OFF`
need no GoogleTest. Use `-DSCALARS_ENABLE_GTEST=OFF` (or the setup helper's
`gtest` token) to disable only the C++ suite. Run the Python checks directly
with `python3 -m unittest discover -s Testing/CMake -p test_setup.py`.

After installation, verify the exported package with:

```bash
cmake -S Testing/InstallConsumer -B build_consumer -DCMAKE_PREFIX_PATH=/your/install/prefix
cmake --build build_consumer --parallel
ctest --test-dir build_consumer --output-on-failure
```

## Lint

`.lintrunner.toml` and the formatter/analyzer configurations are copied from
LinearAlgebra, with paths adapted for `cmake/`, `Testing/`, and `examples/`.
The adapters come from the shared `lint-tool` package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install lintrunner -e ../lint-tool
lintrunner init
lintrunner --all-files
```

Install `clang-tidy` separately for C++ analysis. Configure CMake first to
generate `build/compile_commands.json`. Optional CMake analysis tools and
compiler cache backends must also be installed before enabling their options.

## Example

Code filenames and class/type names use `snake_case`, for example
`types/half.h`, `half`, `bfloat16`, `complex<T>`, and `scalar_type`.

```cpp
#include <scalars.h>
using namespace scalars;
int main() {
  half h{1.5f};
  bfloat16 b{2.0f};
  complex_float z{1.f, 2.f};
  auto w = scalars::exp(z);
  return float(h) + float(b) + w.real() > 0 ? 0 : 1;
}
```

## CPU SIMD

Include `<backend/simd.h>` for `simd<float>`, `simd<double>`,
`simd<scalars::half>`, and `simd<scalars::bfloat16>`. Select a CPU backend with
`-DSCALARS_CPU_BACKEND=sse|avx|avx2|avx512|neon|sve`; `no` leaves the host SIMD
interface unavailable. The existing SVE path requires a fixed 128-bit vector
length and reuses NEON packets.

Transcendental math (`exp`, `log`, `sin`, `pow`, ...) is provided by whichever
vector math library is enabled, in priority order per architecture: on x86, Intel
SVML (`-DSCALARS_ENABLE_SVML=ON`) or Intel MKL VML (`-DSCALARS_ENABLE_MKL=ON`),
then the bundled [SLEEF](https://sleef.org) (`-DSCALARS_ENABLE_SLEEF=ON`, built
from the `ThirdParty/sleef` submodule); on AArch64 NEON/SVE, Apple Accelerate
vForce (`-DSCALARS_ENABLE_ACCELERATE=ON`) then SLEEF. When Apple Accelerate is
unavailable on a NEON/SVE host and SLEEF has not been explicitly requested, or
when a non-Apple AArch64 host is detected, SLEEF is auto-enabled so the library
never falls further back than it needs to. Without any of these, transcendental
functions fall back to scalar `std::` calls applied per lane.

half and bfloat16 use dense 16-bit storage: 8 lanes on SSE/NEON, 16 on AVX/AVX2,
and 32 on AVX-512. Following the architecture of PyTorch's CPU reduced-precision
vectors, each packet widens into two float packets, computes in float, then
rounds back to the stored format. half uses hardware conversion where available
and an SSE2 software SIMD conversion otherwise. Bfloat16 conversion uses
round-to-nearest-even and canonicalizes narrowed NaNs to `0x7fc0`.

The reduced types provide the float backend's real arithmetic/math surface,
comparisons, masks, selection, gather/scatter, horizontal reductions, broadcast,
transpose, and half-width packet utilities. `accumulate` returns float and keeps
float intermediates. Reduced `min`/`max` and their horizontal variants propagate
NaNs and choose `-0` for minimum zero ties, `+0` for maximum zero ties. Masks use
one `uint16_t` per lane: all bits set for true, zero for false. Bitwise sign,
memory, selection, and permutation operations preserve stored NaN payloads.
`ploadquad` repeats each input four times. `ptranspose<N>` transposes an
`N * size` block and repacks the result into `N` packets; one dimension must
divide the other. The legacy `predux_downto4` helper adds the lower and upper
halves, returning `half_size` reduced lanes.

`loadu(ptr, count)` zero-fills inactive lanes, and `storeu(packet, ptr, count)`
writes only active lanes, for `0 <= count <= size`. The full-packet and
half-width memory operations accept scalar-aligned addresses. For kernels that
explicitly retain float intermediates across several operations, use
`to_float_pair` / `from_float_pair` or `load_float_pair` / `store_float_pair`;
the returned pair has `low` and `high` float packets. These helpers round only
when narrowing, so their results can differ from chaining reduced operations.

Math functions reuse the selected float backend/provider, including its accuracy
limits and scalar mappings. X86 builds without SVML/SLEEF use standard-library
scalar mappings for transcendental functions. FMA follows the float backend's
fused/nonfused behavior and narrows once. The numerical contract assumes the
default floating-point rounding environment. Complex SIMD is described below.

Reference architecture:
[PyTorch reduced CPU packets](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cpu/vec/vec256/vec256_16bit_float.h)
and [widened functional helpers](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cpu/vec/functional_bfloat16.h).

## SIMD and complex primitives for LinearAlgebra

LinearAlgebra can use the existing `<backend/simd.h>` interface directly.
`simd<scalars::complex<float>>` and `simd<scalars::complex<double>>` extend
that interface at parity with the real float/double operation set: load/store,
counted load/store, set/setzero, arithmetic/FMA/division, negation,
conjugation, roots/powers, the exponential/logarithmic/trigonometric/
hyperbolic functions, equality and mask selection, magnitude/argument/
real/imag, classification (`isnan`/`isinf`/`isfinite`), rounding, half-width
utilities, and strided or indexed gather/scatter. Real-only ordering
(`gt`/`lt`/`min`/`max`/`hmin`/`hmax`) and `hypot`/`cbrt`/`signcopy`/`cdf`/
`inv_cdf` are deliberately `= delete`d rather than silently applied
componentwise -- complex numbers have no total order and those others have no
agreed complex contract. See
[the operation audit](docs/complex_simd_operation_audit.md) for the full
table and what still maps the scalar complex implementation lane-by-lane
instead of using native vector transcendentals.

Complex specializations live beside the real types in
`backend/cpu/{sse,avx,avx512,neon,sve}/complex/simd.h` and are selected by the same
dispatcher. AVX and AVX2 share the `avx` folder. SVE follows the real types' fixed
128-bit NEON compatibility path. Backend headers implement their own native
register shuffles, division checks, and reciprocals; common packet operations
live in `backend/cpu/detail/complex_simd.h` without ISA conditionals.
`simd<scalars::complex<half>>`/`simd<scalars::complex<bfloat16>>`
(`backend/cpu/reduced/complex_simd.h`) widen the whole operand to a
`complex<float>` pair, run the operation through `simd<complex<float>>`, and
narrow once, rather than narrowing after every intermediate op the way the
real reduced backend does -- narrowing a squared term for a division
denominator would overflow half's range far sooner than a real division ever
would. `simd<scalars::complex<float>>`/`simd<scalars::complex<double>>` also
exist for CUDA/HIP device code (`backend/gpu/complex/simd.h`, scalar,
`size == 1`, same interface as `backend/gpu/{float,double}/simd.h`), and
Metal has a `complex64` kernel set at parity with its `float32` kernels
(`backend/gpu/metal/kernels.metal`; no `complex128`, since Metal has no
`double` either).

All CPU backends use contiguous interleaved memory operations and bounded vector
division. SSE, AVX/AVX2, and AVX-512 use hardware vector division for the reciprocal;
NEON uses Newton-refined reciprocal estimates. Packets containing extreme or
nonfinite components, or a zero divisor, use scaled scalar division. Each complex
SIMD value holds separate real and imaginary registers, with `size` complex elements.
No packet adapter or additional public abstraction is needed.

Link `Scalars::Scalars` to inherit the ISA, compiler flags, and math provider.
CPU complex SIMD (including the reduced-precision packets above) requires a
configured CPU backend, like the existing real SIMD types.

```cpp
#include <backend/simd.h>
using S = simd<scalars::complex<double>>;
auto a = S::loadu(input);
auto b = S::fma(a, S::set({2.0, -1.0}), S::setzero());
S::storeu(b, output);
```

`<math/complex_vector.h>` supplies `scalars::complex_dot`, `complex_axpy`,
`complex_scale`, and `complex_rotate` using this same SIMD API directly.
They handle tails with scalar arithmetic and use scalar loops when CPU SIMD
is disabled. Strides count complex objects and may be negative. Dot and AXPY
optionally conjugate x. Rotation computes `x' = c*x + s*y` and
`y' = c*y - conj(s)*x` without scratch storage. Output strides must be nonzero;
output ranges must be disjoint except for AXPY's exact in-place x/y case.
Empty ranges accept null pointers. Complex counted loads zero-fill inactive
lanes; counted stores write only active lanes (`0 <= count <= size`).
SIMD accumulation order and FMA semantics follow the selected real backend.
The division fast path follows the real backend's FMA behavior and has ordinary
floating-point rounding differences from scalar division. See the
[complex SIMD performance notes](docs/complex_simd_performance.md) for measured
PyTorch comparisons and the standalone benchmark command.

Real SIMD `ploadquad` uses PRETORIAN's implementation for each ISA:
four-wide SSE/NEON float loads duplicate pairs; AVX float and AVX-512 double
loads repeat each input four times; narrower double loads broadcast one value.
PRETORIAN does not provide AVX-512 float `ploadquad`, so that operation remains
unavailable. `ptranspose<4>` on wider float/double SIMD registers follows PRETORIAN's
rectangular transpose: a `4 x size` input becomes a contiguous `size x 4`
output repacked into four registers. Store those registers consecutively to
produce a packed RHS block. Existing square transposes and half-width helpers remain
on `simd<T>` itself. FMA returns its result; consumers must assign that result,
including in half-width remainder kernels.

Scalar helpers `scalars::abs`, `arg`, `conj`, `real`, `imag`, `norm`, and
`isfinite` are available for complex values. Finite host complex division and
the custom complex square root use power-of-two scaling at extreme magnitudes.
Only Scalars is changed; LinearAlgebra can consume these operations directly
when its local scalar/SIMD implementations are replaced.

## GPU backends

Select `-DSCALARS_GPU_BACKEND=cuda|hip|metal` independently of the CPU backend.
HIP builds also require ROCm's rocThrust development headers (`rocthrust-dev`),
which provide the `thrust/complex.h` dependency used by GPU complex operations.
CUDA/HIP device compilation of `<backend/simd.h>` provides `simd<float>`,
`simd<double>`, `simd<scalars::half>`, and `simd<scalars::bfloat16>`. Each packet
has `size == 1`, with one logical lane per GPU thread. Reduced packets retain
their distinct two-byte storage, compute through the float GPU backend, and
round once per operation. FMA uses a float fused operation before narrowing;
`accumulate` returns float. This follows PyTorch's
[operation-math policy](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/OpMathType.h).

Reduced device packets support the float backend's math surface, 16-bit
comparison masks, bitwise selection/sign operations, gather/scatter, and
counted load/store with counts zero or one. `min`/`max` propagate NaNs and use
the CPU reduced packets' signed-zero policy. `to_float` / `from_float` and
`load_float` / `store_float` let kernels retain FP32 intermediates explicitly.
Gather/scatter's integer argument is the thread's absolute element index.
CPU packet transpose and half-width utilities have no one-lane GPU counterpart.

Metal uses `<backend/gpu/metal/metal_dispatch.h>` rather than the C++ device
dispatcher. Pass `metal_backend::dtype::half` or `dtype::bfloat16` as the final
argument to `dispatch`, `dispatch_fill`, `dispatch_fused`, and reductions;
existing calls continue to use float. All buffers in an element-wise launch
use the selected storage type. Fill values and fused scalar arguments remain
float. Allocate buffers with `allocate(n * sizeof(T))` and release them with
`deallocate`. For example, after allocating half buffers `a`, `b`, and `out`:

```cpp
namespace mb = scalars::metal_backend;
const void* inputs[] = {a, b};
mb::dispatch("add", inputs, 2, out, n, mb::dtype::half);
float sum = mb::reduce_sum(out, n, mb::dtype::half);
```

Metal supplies reduced fill, add/sub/mul/div, min/max, FMA, sqrt/exp/log,
sin/cos/tanh, fabs/neg, and sum/min/max reductions. All reduction passes use
FP32 partials; empty reductions return zero, positive infinity, or negative
infinity respectively. BF16 uses software bit conversion with round-to-nearest-even
and canonical narrowed NaNs, without requiring native MSL bfloat support.
Fused sources define `fused_float`, `fused_half`, or `fused_bfloat16` for the
selected type and are responsible for their own storage conversions.

## Scope and limitations

- CUDA/HIP device tests require the corresponding compiler and GPU. Metal tests run on a Mac with a Metal device. This release does not claim full host/device parity with PyTorch.
- Arithmetic for half/bfloat16 uses float intermediates and rounds the result to the stored format.
- Scalar casts from complex to real discard the imaginary component by explicit request; review this policy for strict applications.
- Complex division uses scaled host arithmetic for finite float/double operands;
  transcendental functions use standard-library math and the existing custom
  square-root/arc-cosine implementation where required by the standard library.
- Type promotion supports the four defined real component types; broader integral promotion and checked casts remain future work.
- Numerical limits currently specialize half only; add complete bfloat16 limits and edge-case exhaustive tests before production use.
- complex_math.h is CPU-only because of the `std::complex` backend.
- Not a drop-in ABI replacement for c10, std::complex, cuComplex, or rocComplex.

## Reference code investigated

- https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/half.h
- https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/bfloat16.h
- https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/complex.h
- https://github.com/pytorch/pytorch/blob/main/c10/util/complex_math.h
- https://github.com/pytorch/pytorch/blob/main/c10/util/TypeCast.h
- https://github.com/pytorch/pytorch/blob/main/c10/core/ScalarType.h

This project is an independent implementation, not a verbatim port of PyTorch.
