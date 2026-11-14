# Scalars

Scalars is a C++20 library for **half**, **bfloat16**, and **complex numbers**, with
scalar math, type traits, promotion, explicit casts, dtype metadata, and CPU/GPU
kernel primitives. Use it as the scalar and SIMD foundation of numerical code,
including linear algebra and custom compute kernels.

The main entry points are **`<scalars.h>`**, **namespace `scalars`**, and the CMake
target **`Scalars::Scalars`**. SIMD, vector algorithms, and GPU dispatch have
separate headers described below.

[Quick start](#quick-start) · [Public API](#public-headers-and-api) ·
[CPU SIMD](#cpu-simd-and-vector-algorithms) · [GPU](#gpu-entry-points) ·
[Build and tests](#build-test-and-development-options)

## Why use Scalars?

| Capability | Practical benefit |
| --- | --- |
| Compact real types | `half` and `bfloat16` occupy two bytes each, reducing storage and memory traffic compared with `float`. |
| Explicit compute precision | Reduced arithmetic uses float intermediates; accumulation traits let kernels choose wider intermediate types. |
| Shared scalar vocabulary | Real/complex types, casts, promotion, and runtime dtype metadata can be reused across numerical components. |
| CPU SIMD | The same `scalars::simd<T>` interface selects SSE, AVX, AVX2, AVX-512, NEON, or the fixed-width SVE compatibility backend. |
| GPU building blocks | CUDA/HIP device operations and Metal dispatch support custom kernels and reduced-precision storage. |
| CMake integration | An installed package or `add_subdirectory()` exposes the same target, including required compiler flags and linked math providers. |

Scalars builds independently of PyTorch. Some implementations derive from
PyTorch c10 and FP16; see [License and provenance](#license-and-provenance).
Performance depends on the operation, backend, math provider, and workload;
[complex SIMD measurements](docs/complex_simd_performance.md) describe the
benchmark methodology and measured results.

## Quick start

Requires **CMake 3.24+** and a **C++20 compiler**; C++23 is also supported.
Run these commands from the repository root:

```sh
git submodule update --init ThirdParty/googletest ThirdParty/sleef
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix /your/install/prefix
```

The standalone build defaults to a shared library. Set `-DBUILD_SHARED_LIBS=OFF`
for a static library, or `-DSCALARS_ENABLE_TESTING=OFF` for a library-only build.
GoogleTest is needed for the C++ tests; SLEEF is used when that math provider is
selected. Google Benchmark is optional.

### Link from your application

```cmake
cmake_minimum_required(VERSION 3.24)
project(scalars_example LANGUAGES CXX)

find_package(Scalars CONFIG REQUIRED)
add_executable(scalars_example main.cpp)
target_link_libraries(scalars_example PRIVATE Scalars::Scalars)
```

Configure your application with the installation prefix:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/your/install/prefix
cmake --build build --parallel
```

For a source checkout inside your project, replace `find_package(...)` with
`add_subdirectory(path/to/Scalars)`. Link the same `Scalars::Scalars` target.
Subdirectory builds inherit the parent's shared/static choice and normally leave
Scalars tests disabled; `BUILD_TESTING` can override that default.

### First program

Save this as `main.cpp` for the application above:

```cpp
#include <scalars.h>
#include <iostream>

int main()
{
    scalars::half h{1.5f};
    scalars::bfloat16 b{2.0f};
    scalars::complex_float z{1.0f, 2.0f};

    const auto sum = h + scalars::half{0.5f};
    const auto root = scalars::sqrt(h);
    const auto w = scalars::exp(z);

    std::cout << static_cast<float>(sum) << '\n';  // 2
    std::cout << static_cast<float>(b) << '\n';    // 2
    std::cout << static_cast<float>(root) << '\n';
    std::cout << w.real() << ", " << w.imag() << '\n';
    std::cout << scalars::scalar_type_name(scalars::scalar_type::complex_float)
              << '\n';  // complex64
}
```

## Public headers and API

Headers are installed directly under `include/`, so use `<types/half.h>` rather
than a `Scalars/` prefix. Prefer the dispatcher headers over ISA-specific files.

| Include | Public entry points | Included by `<scalars.h>`? |
| --- | --- | --- |
| [`<scalars.h>`](include/scalars.h) | Scalar umbrella: types, scalar math, traits, casts, promotion, metadata | — |
| [`<types/half.h>`](include/types/half.h) | `half`, conversions, arithmetic, comparisons, `std::numeric_limits` | Yes |
| [`<types/bfloat16.h>`](include/types/bfloat16.h) | `bfloat16`, conversions, arithmetic, comparisons, `std::numeric_limits` | Yes |
| [`<types/complex.h>`](include/types/complex.h) | `complex<T>`, aliases, component access, arithmetic, complex math | Yes |
| [`<math/scalar_math.h>`](include/math/scalar_math.h) | Real math overloads for `half` and `bfloat16` | Yes |
| [`<common/scalar_type.h>`](include/common/scalar_type.h) | `scalar_type`, `scalar_type_name`, `scalar_type_size` | Yes |
| [`<common/scalar_traits.h>`](include/common/scalar_traits.h) | `is_complex_v`, `real_type_t`, `accumulation_type_t` | Yes |
| [`<common/type_promotion.h>`](include/common/type_promotion.h) | `promote_t<A, B>` | Yes |
| [`<common/type_cast.h>`](include/common/type_cast.h) | `scalar_cast<To>(value)` | Yes |
| [`<common/accumulate_type.h>`](include/common/accumulate_type.h) | `acc_type`, `acc_type_device`, `device_type`, `to_accumulate_type` | Yes |
| [`<math/normal_cdf.h>`](include/math/normal_cdf.h) | `normalcdf`, `inv_normalcdf` | No |
| [`<backend/simd.h>`](include/backend/simd.h) | `simd<T>` and backend selection | No |
| [`<math/complex_vector.h>`](include/math/complex_vector.h) | `complex_dot`, `complex_axpy`, `complex_scale`, `complex_rotate` | No |
| [`<backend/gpu/metal/metal_dispatch.h>`](include/backend/gpu/metal/metal_dispatch.h) | `metal_backend::dispatch`, reductions, `execution_context`, fused-pipeline cache controls | No |

All listed library names live in `scalars`; Metal names live in
`scalars::metal_backend`. Use `scalars::simd<T>` explicitly. The
`math/complex_math.h` and `math/complex_utils.h` headers are implementation
companions included through `<types/complex.h>` and reject direct inclusion.

### Scalar types and math

| Type | Storage | Use |
| --- | --- | --- |
| `half` | 2 bytes | IEEE binary16: 11 significant binary digits, including the implicit leading bit |
| `bfloat16` | 2 bytes | 8 significant binary digits with float's exponent range |
| `complex_half` / `complex<half>` | 4 bytes | Reduced-precision real and imaginary components |
| `complex_bf16` / `complex<bfloat16>` | 4 bytes | Bfloat16 real and imaginary components |
| `complex_float` / `complex<float>` | 8 bytes | Float real and imaginary components |
| `complex_double` / `complex<double>` | 16 bytes | Double real and imaginary components |

Reduced real arithmetic computes in float and rounds back to the stored format.
Use `static_cast<float>(value)` to widen explicitly. Both reduced real types
provide `std::numeric_limits` specializations, including infinities, NaNs,
epsilon, and subnormal limits.

`scalars::exp`, `log`, `sqrt`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`,
`floor`, `ceil`, `round`, and `trunc` accept reduced real values. Use the
standard `<cmath>` functions for native `float` and `double` values.

For `complex<float>` and `complex<double>`, Scalars supplies arithmetic,
`real()`/`imag()` getters and setters, `polar`, and the exponential, logarithmic,
power, trigonometric, inverse-trigonometric, and hyperbolic families, plus
`log1p` and `expm1`. Component helpers include `abs`, `arg`, `conj`, `real`,
`imag`, `norm`, and `isfinite`. Host transcendental math uses the standard
library with custom square-root/arc-cosine handling where needed; CUDA/HIP
complex math uses Thrust. Reduced complex types support arithmetic; explicitly
widen to `complex<float>` for scalar transcendental operations.

The normal-distribution functions have a separate entry point:

```cpp
#include <math/normal_cdf.h>

const double probability = scalars::normalcdf(0.0);     // 0.5
const double quantile = scalars::inv_normalcdf(0.975);  // approximately 1.96
```

### Traits, promotion, casts, and metadata

```cpp
#include <scalars.h>
#include <type_traits>

static_assert(scalars::is_complex_v<scalars::complex_float>);
static_assert(std::is_same_v<
    scalars::real_type_t<scalars::complex_double>, double>);
static_assert(std::is_same_v<
    scalars::promote_t<scalars::half, scalars::bfloat16>, float>);
static_assert(std::is_same_v<
    scalars::promote_t<scalars::complex_float, double>, scalars::complex_double>);

const auto z = scalars::scalar_cast<scalars::complex_double>(2.0);
const double real_part = scalars::scalar_cast<double>(
    scalars::complex_double{3.0, 4.0});  // 3: discards the imaginary component

const auto type = scalars::scalar_type::complex_double;
const auto name = scalars::scalar_type_name(type);  // "complex128"
const auto bytes = scalars::scalar_type_size(type);  // 16
```

`promote_t` selects a result type for generic code; it does not insert casts into
an expression. Real promotion favors `long double`, then `double`, then `float`;
a half/bfloat16 pair promotes to float. A complex operand makes the promoted
result complex. Casts follow the requested destination type and provide no
range checking; complex-to-real casts explicitly discard the imaginary part.

The dtype enum covers `float16`, `bfloat16`, `float32`, `float64`,
`complex_half`, `complex_bf16`, `complex_float`, and `complex_double`.
It describes types at runtime; it does not own storage or dispatch arbitrary
algorithms automatically.

Choose accumulation precision independently of storage:

| Input `T` | CPU `acc_type<T, false>` | GPU `acc_type<T, true>` |
| --- | --- | --- |
| `half`, `bfloat16` | `float` | `float` |
| `float` | `double` | `float` |
| `double` | `double` | `double` |
| `complex<half>`, `complex<bfloat16>` | `complex<float>` | `complex<float>` |
| `complex<float>` | `complex<double>` | `complex<float>` |
| `complex<double>` | `complex<double>` | `complex<double>` |

`accumulation_type_t<T>` is the CPU alias. The equivalent device-enum form is
`acc_type_device<T, device_type::cpu>` or `device_type::gpu`; use
`to_accumulate_type(dtype, device)` for runtime metadata. Only types with an
explicit accumulation-table entry are supported. These traits select types;
callers choose where to widen their intermediate calculations.

## CPU SIMD and vector algorithms

Include `<backend/simd.h>` and link `Scalars::Scalars`. The `simd<T>` API covers
`float`, `double`, `half`, `bfloat16`, and their supported complex forms.
Packet types and lane counts are backend-dependent: query `simd_t`, `value_t`,
`mask_t`, and `size` instead of hard-coding register widths.

```cpp
#include <backend/simd.h>
#include <array>

#if SCALARS_VECTORIZED
void packet_example()
{
    using packet = scalars::simd<scalars::complex<double>>;
    using value = packet::value_t;
    std::array<value, packet::size> input{};
    std::array<value, packet::size> output{};
    input.fill(value{1.0, 2.0});

    const auto a = packet::loadu(input.data());
    const auto b = packet::fma(
        a, packet::set(value{2.0, -1.0}), packet::setzero());
    packet::storeu(b, output.data());  // every element is 4 + 3i
}
#endif
```

The common surface includes load/store, broadcast, arithmetic, FMA, math,
comparisons, masks, selection, gather/scatter, and reductions. Complex packets
add conjugation, magnitude, argument, components, and classification. Complex
ordering, min/max, `hypot`, `cbrt`, `signcopy`, and normal-CDF operations are
deliberately unavailable. Some complex transcendental operations map scalar
math lane by lane; see the [operation audit](docs/complex_simd_operation_audit.md).

Reduced real packets store 8 lanes on SSE/NEON, 16 on AVX/AVX2, and 32 on
AVX-512. They widen into two float packets, compute, and narrow. For a chain
that retains float intermediates, use `to_float_pair`/`from_float_pair` or
`load_float_pair`/`store_float_pair`. Reduced complex packets widen the whole
operation through the complex-float backend before narrowing once.

### Strided complex algorithms

Use `<math/complex_vector.h>` for host loops that handle packets and scalar tails:

| Function | Operation |
| --- | --- |
| `complex_dot(x, sx, y, sy, count, conjugate_x)` | Sum of `x*y`, optionally conjugating `x` |
| `complex_axpy(y, sy, x, sx, count, alpha, conjugate_x)` | `y += alpha*x`, optionally conjugating `x` |
| `complex_scale(x, stride, count, scale)` | Scale complex values by a real factor |
| `complex_rotate(x, sx, y, sy, count, c, s)` | `x' = c*x + s*y`, `y' = c*y - conj(s)*x` |

```cpp
#include <math/complex_vector.h>
#include <array>

void vector_example()
{
    using value = scalars::complex<double>;
    std::array<value, 3> x{value{1, 0}, value{2, 0}, value{3, 0}};
    std::array<value, 3> y{value{4, 0}, value{5, 0}, value{6, 0}};

    const auto dot = scalars::complex_dot(x.data(), 1, y.data(), 1, 3);
    // dot == 32 + 0i; passing true below selects a conjugated dot product.
    const auto hermitian_dot = scalars::complex_dot(
        x.data(), 1, y.data(), 1, 3, true);
    scalars::complex_axpy(y.data(), 1, x.data(), 1, 3, value{2, 0});
}
```

Strides count complex objects and may be negative; pass the first logical
element. Counts must be nonnegative. Output strides must be nonzero for multiple
elements. Keep output ranges disjoint; AXPY permits exact in-place `x == y` with
matching strides. Empty ranges accept null pointers. These algorithms use scalar
loops when CPU SIMD is disabled. Accumulation order and FMA rounding can differ
between scalar and SIMD execution.

### Backend selection and numerical contracts

| CMake option | Selection |
| --- | --- |
| `SCALARS_CPU_BACKEND` | `no`, `sse`, `avx`, `avx2`, `avx512`, `neon`, `sve` |
| `SCALARS_ENABLE_SVML` | Bundled Intel SVML on supported x86 platforms |
| `SCALARS_ENABLE_MKL` | Intel MKL VML on x86 |
| `SCALARS_ENABLE_SLEEF` | SLEEF from `ThirdParty/sleef` |
| `SCALARS_ENABLE_ACCELERATE` | Apple Accelerate vForce on AArch64 |

The default CPU backend is AVX2 on x86, NEON on AArch64, and `no` elsewhere,
subject to compiler support. SVE currently reuses NEON packets and requires
`SCALARS_SVE_VECTOR_BITS=128`. With `SCALARS_CPU_BACKEND=no`, host `simd<T>`
specializations are unavailable; scalar APIs and complex vector loops remain usable.

Backend selection fixes the CPU instruction-set baseline at build time. There
is **no runtime ISA dispatch**. The CMake target propagates matching ISA flags to
consumers. Select the narrowest backend supported by your deployment machines;
use `-DSCALARS_CPU_BACKEND=no` to disable CPU packet code.

On x86, math uses SVML or MKL, then SLEEF when selected. On AArch64, it uses
Accelerate or SLEEF. Configuration may auto-enable SVML on x86, Accelerate on
Apple AArch64, or SLEEF on other AArch64 hosts. Inspect the configuration summary
for the actual provider. Without a vector math provider, transcendental functions
map standard-library calls over individual lanes.

<details>
<summary>Packet memory, precision, and packing details</summary>

- Counted `loadu(ptr, count)` zero-fills inactive lanes; counted
  `storeu(packet, ptr, count)` writes only active lanes, with `0 <= count <= size`.
  Reduced and complex counted memory operations accept scalar-aligned pointers.
- Reduced real `accumulate` returns float. Reduced min/max propagate NaNs and
  choose negative zero for minimum ties and positive zero for maximum ties.
  Reduced masks hold one `uint16_t` per lane, all bits set for true and zero for
  false. Bitwise sign, selection, memory, and permutation preserve stored payloads.
- Reduced conversions use round-to-nearest-even; narrowed bfloat16 NaNs are
  canonicalized to `0x7fc0`. Hardware half conversion is used where available.
- CPU complex packets keep separate real and imaginary registers and load/store
  interleaved complex objects. Ordinary division stays vectorized; extreme,
  nonfinite, and zero-divisor lanes use scalar repair. SSE, AVX, AVX-512, and
  AArch64 NEON use hardware division for their complex reciprocal step.
- Reduced `ploadquad` repeats each input four times. Four-wide real SSE/NEON float
  `ploadquad` duplicates pairs; wider AVX float and AVX-512 double repeat each
  input four times. Narrow double packets broadcast one value. AVX-512 float
  `ploadquad` is unavailable.
- `ptranspose<N>` repacks a rectangular transpose; supported dimensions depend on
  the packet type. Reduced packets require one dimension to divide the other.
  The legacy `predux_downto4` combines packet halves. FMA returns a result that
  callers must assign, including in half-width operations.
- Math accuracy and fused/nonfused behavior follow the chosen provider and
  backend. The numerical contract assumes the default rounding environment.

</details>

## GPU entry points

Select `-DSCALARS_GPU_BACKEND=cuda`, `hip`, or `metal` independently of the CPU
backend. The default is `none`.

### CUDA and HIP

Compile device kernels with the corresponding toolchain and include
`<backend/simd.h>`. Device `simd<T>` packets have `size == 1`, one logical lane
per thread. The device interface covers float/double, reduced real values, and
complex float/double. The dispatcher selects the GPU implementation during the
device pass and the configured CPU implementation during the host pass.

Reduced device packets compute through float, support 16-bit comparison masks,
and offer `to_float`/`from_float` and `load_float`/`store_float` for explicit FP32
intermediates. Gather/scatter's integer argument is the absolute element index.
Counted loads/stores accept zero or one; CPU transpose and half-width utilities
have no device counterpart. HIP complex math requires ROCm's rocThrust
headers, commonly provided by `rocthrust-dev`.

### Metal

Include `<backend/gpu/metal/metal_dispatch.h>` and use
`scalars::metal_backend`. Supported buffer types are `dtype::float32`,
`dtype::half`, `dtype::bfloat16`, and `dtype::complex64`. Metal has no double or
complex128 support; complex64 reductions are unavailable.

Free dispatch functions are synchronous. For preallocated, page-aligned shared
buffers `a`, `b`, and `out` covering `n` half elements:

```cpp
#include <backend/gpu/metal/metal_dispatch.h>

void add_half_buffers(void* a, void* b, void* out, std::size_t n)
{
    namespace mb = scalars::metal_backend;
    const void* inputs[] = {a, b};
    mb::execution_context batch;
    batch.dispatch_fill(a, 2.0f, n, mb::dtype::half);
    batch.dispatch_fill(b, 3.0f, n, mb::dtype::half);
    batch.dispatch("add", inputs, 2, out, n, mb::dtype::half);
    batch.submit();  // asynchronous submission
    batch.wait();    // wait before accessing shared buffers from the CPU
    const float sum = mb::reduce_sum(out, n, mb::dtype::half);
}
```

Each backing allocation must be page-aligned, have a page-multiple length, and
cover all elements. Keep it alive until `wait()` completes. Execution contexts
are confined to one thread; destruction drains outstanding work without
throwing, so call `wait()` to observe errors.

Reduced Metal operations include fill, arithmetic/FMA, min/max, sqrt/exp/log,
sin/cos/tanh, fabs/neg, and sum/min/max reductions. Reduction partials and results
use float; empty sum/min/max return zero/positive infinity/negative infinity.
Bfloat16 conversions are implemented in software. Fused sources define
`fused_float`, `fused_half`, or `fused_bfloat16` for the selected type and handle
their storage conversions. Compiled fused pipelines use an LRU cache with a
default capacity of 256; `set_fused_pipeline_cache_capacity`,
`fused_pipeline_cache_capacity`, and `fused_pipeline_cache_size` control or inspect it.

## Build, test, and development options

Common options, in addition to the CPU/GPU selections above:

| Option | Default | Purpose |
| --- | --- | --- |
| `SCALARS_CXX_STANDARD` | `20` | C++20 or C++23 |
| `SCALARS_ENABLE_TESTING` | `ON` standalone | Build tests; `BUILD_TESTING=OFF` disables them globally |
| `SCALARS_ENABLE_GTEST` | `ON` | Enable the C++ GoogleTest suite |
| `SCALARS_ENABLE_BENCHMARK` | `OFF` | Build Google Benchmark micro-benchmarks |
| `SCALARS_ENABLE_CACHE` / `SCALARS_CACHE_BACKEND` | `ON` / `none` | Select `none`, `ccache`, `sccache`, or `buildcache` |
| `SCALARS_LTO_MODE` | `off` | `off`, `thin`, `full`, `ipo`, or `auto` |
| `SCALARS_LINKER_CHOICE` | `default` | `default`, `lld`, `mold`, `gold`, or `lld-link` |
| `SCALARS_ENABLE_SANITIZER` / `SCALARS_SANITIZER_TYPE` | `OFF` / `address` | Instrument with one sanitizer or a comma-separated combination |
| `SCALARS_ENABLE_COVERAGE` | `OFF` | Coverage instrumentation |
| `SCALARS_ENABLE_CLANGTIDY` / `SCALARS_ENABLE_FIX` | `OFF` / `OFF` | Build-integrated analysis; the second option allows source edits |
| `SCALARS_ENABLE_IWYU` / `SCALARS_ENABLE_SPELL` | `OFF` / `OFF` | Include analysis / spell checking |
| `SCALARS_ENABLE_VALGRIND` / `SCALARS_ENABLE_ICECC` | `OFF` / `OFF` | CTest memcheck / distributed compilation |

Coverage and sanitizers skip LTO. The legacy `SCALARS_BUILD_TESTS` option still
supplies the testing default. GoogleTest, Google Benchmark, and SLEEF are
submodules under `ThirdParty/`; initialize Benchmark before enabling benchmarks.
Missing GoogleTest or Benchmark checkouts can fall back to FetchContent.

The optional Python helper provides the same build actions from the repository
root, or through `python3 setup.py` from `Scripts/`:

```sh
python3 Scripts/setup.py --help
python3 Scripts/setup.py config.build.test
python3 Scripts/setup.py config.build.test.static.cxx23
python3 Scripts/setup.py config.build.test --sanitizer.address
python3 Scripts/setup.py config.build.test --cpu-backend=neon --gpu-backend=metal
python3 Scripts/setup.py config.build.test --lto.ipo
python3 Scripts/setup.py config.build.test.coverage
```

It defaults to Ninja, Clang, C++20, and Release, creating `build_ninja/` with
option suffixes. `config` reuses a build directory; `build.test` reruns an
existing configuration. Sanitizer, coverage, and Valgrind helper builds select
Debug. Use named options from `--help` for paths, toolchains, architectures,
backends, filters, and installation settings. Raw build-system arguments and
`--` passthrough are rejected. Windows Ninja builds require a Visual Studio
developer environment. Coverage reporting requires `coverage-tool` and
`coverage.toml`; optional analysis tools and cache backends are installed separately.

| Location | Purpose |
| --- | --- |
| `Testing/Cxx/Test*.cxx` | C++ unit tests, built as `ScalarsCxxTests` |
| `Testing/Cxx/Test*.cu` | CUDA/HIP device tests when the corresponding backend is enabled |
| `Testing/CMake/test_setup.py` | Python helper checks, registered as `ScalarsSetupTests` |
| `Testing/Benchmark/Benchmark*.cxx` | Optional native micro-benchmarks |
| `Testing/InstallConsumer/` | Standalone installed-package integration checks |

Run the Python checks alone with
`python3 -m unittest discover -s Testing/CMake -p test_setup.py`.
Use Release without coverage or sanitizers for benchmarks. GPU runtime tests
require a compatible device and toolchain.

Verify an installation independently with:

```sh
cmake -S Testing/InstallConsumer -B build_consumer -DCMAKE_PREFIX_PATH=/your/install/prefix
cmake --build build_consumer --parallel
ctest --test-dir build_consumer --output-on-failure
```

Lint uses `.lintrunner.toml` and the shared `lint-tool` adapters:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install lintrunner -e ../lint-tool
lintrunner init
lintrunner --all-files
```

The `../lint-tool` path assumes that sibling checkout is available. Install
`clang-tidy` separately and configure CMake to generate `compile_commands.json`
before C++ analysis.

## Scope and compatibility

Scalars supplies numerical building blocks. Array/tensor ownership, expression
graphs, full matrix algorithms, runtime dtype dispatch, and checked conversions
belong to the consuming application. It provides no drop-in ABI compatibility
with c10, `std::complex`, cuComplex, or rocComplex. Host/device math availability
and accuracy depend on the backend; full PyTorch parity is not claimed.

Source changes relevant to existing consumers:

- SIMD lives in `scalars::simd<T>`; update uses of the former global name.
- `accumulation_type_t<float>` is now `double`, and complex-float CPU
  accumulation is complex-double. Types without an accumulation-table entry,
  including `long double` and `uint32_t`, fail to instantiate that trait.
- Half comparisons accept half and arithmetic operands and resolve alongside
  CUDA/HIP's native half operators.
- Reduced complex types share an implementation, default to zero, and support
  scalar compound arithmetic using their component type.
- Complex-packet `set_mask` accepts Boolean and integral arguments; nonzero
  enables every lane.
- Complex arithmetic/component helpers and `bit_cast` carry host/device
  annotations for CUDA/HIP compilation.

## License and provenance

The project license is [MIT](LICENSE). Sections of
[`types/half.h`](include/types/half.h) and
[`types/bfloat16.h`](include/types/bfloat16.h) derive from PyTorch c10
`Half-inl.h` and `BFloat16-inl.h` under BSD-3-Clause. Half conversion routines
also derive from Marat Dukhan's FP16 library under MIT. [NOTICE](NOTICE)
contains the attributions and license text; vendored dependencies retain their
own notices, including [Intel SVML's license](ThirdParty/svml/LICENSE).
