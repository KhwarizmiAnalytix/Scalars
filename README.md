# Scalars

Standalone C++20 scalar library for **Half**, **BFloat16**, and **Complex<T>**, with dtype metadata, traits, promotion rules, casts, and standard-library-backed math functions. Inspired by the *architecture* of PyTorch `c10`, but independently implemented; no PyTorch source files are copied.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix /your/install/prefix
```

Consumers can use `find_package(Scalars CONFIG REQUIRED)` and link `Scalars::Core` (headers) or `Scalars::Scalars` (runtime dtype metadata).

## Example

```cpp
#include <Scalars/Scalars.h>
using namespace kh::scalars;
int main() {
  Half h{1.5f};
  BFloat16 b{2.0f};
  ComplexFloat z{1.f, 2.f};
  auto w = kh::scalars::exp(z);
  return float(h) + float(b) + w.real() > 0 ? 0 : 1;
}
```

## Scope and limitations

- Portable CPU implementation; **CUDA/HIP kernels are not implemented yet**. This release does not claim host/device parity with PyTorch.
- Arithmetic for Half/BFloat16 uses float intermediates and rounds the result to the stored format.
- Scalar casts from complex to real discard the imaginary component by explicit request; review this policy for strict applications.
- Complex division and transcendental functions delegate to `std::complex`, with intermediate float for reduced precision.
- Type promotion supports the four defined real component types; broader integral promotion and checked casts remain future work.
- Numerical limits currently specialize Half only; add complete BFloat16 limits and edge-case exhaustive tests before production use.
- ComplexMath.h is CPU-only because of the `std::complex` backend.
- Not a drop-in ABI replacement for c10, std::complex, cuComplex, or rocComplex.

## Reference code investigated

- https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/Half.h
- https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/BFloat16.h
- https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/complex.h
- https://github.com/pytorch/pytorch/blob/main/c10/util/complex_math.h
- https://github.com/pytorch/pytorch/blob/main/c10/util/TypeCast.h
- https://github.com/pytorch/pytorch/blob/main/c10/core/ScalarType.h

This project is an independent implementation, not a verbatim port of PyTorch.
