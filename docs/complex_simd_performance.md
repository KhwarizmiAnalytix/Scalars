Complex SIMD measurements on macOS ARM64, 2026-10-25, using PyTorch 2.13.0
(`cf30153c4c131c8164ee7798e5022d810682e2cb`) and Clang 22.1.2.

The CPU implementations follow the interleaved memory/arithmetic approach of
[PyTorch's complex packets](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/cpu/vec/vec256/vec256_complex_float.h).
Loads/stores use `memcpy` and vector shuffles, preserving object aliasing rules.
Add/subtract operate on interleaved registers so the compiler can eliminate
surrounding shuffles. The public packet still exposes separate real/imaginary
registers for multiply, multiply-add, reductions, and existing callers.

Division checks that each nonzero component is in `[2^-30, 2^31)` for
float or `[2^-250, 2^251)` for double, and that every divisor is nonzero. Products,
the denominator, and reciprocal then stay in range. NEON uses a reciprocal estimate
with two float or three double Newton refinements. SSE, AVX/AVX2, and AVX-512 use
their hardware vector division for the reciprocal. Intermediates follow the real
backend's FMA support. This has rounding differences from scalar
division and does not promise correctly rounded results. If any lane fails the
check, the whole packet uses the existing scaled scalar implementation.
Zero-valued components are allowed through a separate check, including signed
zero; a zero complex divisor always takes the fallback. The NEON double high-word
shortcut is restricted to little-endian targets.

The fallback policy resembles PyTorch's scalar handling of exceptional complex
operations. The bounded vector division is an additional optimization;
PyTorch's AVX2 complex division itself still maps scalar division over lanes.
The fixed-width SVE compatibility backend uses the same NEON implementation.
Each ISA owns a `backend/cpu/<isa>/complex/simd.h` next to its `float` and `double`
folders. AVX2 uses `avx`, matching the real types. Shared code in
`backend/cpu/detail/complex_simd.h` contains no ISA conditionals or intrinsic types.

To reproduce with an installed PyTorch wheel and C++ compiler:

```sh
python3 Scripts/benchmark_complex_simd.py --cxx clang++ \
    --elements 64 4096 65536 --iterations 2000
```

The benchmark compiles the same four packet loops against Scalars and the actual
installed ATen headers with `-O3 -DNDEBUG -std=c++20` and matching ISA flags.
It uses preallocated arrays, seed 123, independent real/imaginary values in
`[-2, 2]`, 20 warmups, and medians of nine batches. Measurement order alternates
between implementations. A memory barrier after each loop invocation prevents
discarding output writes. PyTorch is an optional dependency for this standalone
comparison; the regular CMake build and tests do not depend on it.
Use `--backend sse|avx|avx2|avx512|neon|sve` to select a supported backend on
the corresponding host architecture; the default is NEON on ARM64 and AVX2 on x86.

One measured run with 4,096 complex elements, nanoseconds per complex element:

| Operation | Scalars complex64 | ATen complex64 | Speedup | Scalars complex128 | ATen complex128 | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Add | 0.158 | 0.194 | 1.22x | 0.263 | 0.260 | 0.99x |
| Multiply | 0.265 | 0.376 | 1.42x | 0.527 | 0.613 | 1.16x |
| Divide | 0.773 | 1.149 | 1.49x | 1.482 | 1.933 | 1.30x |
| Multiply-add | 0.286 | 0.507 | 1.77x | 0.663 | 0.755 | 1.14x |

An earlier run at 65,536 elements measured float-complex multiply/divide/multiply-add at
1.36x/1.48x/1.35x ATen throughput; double-complex results were
1.10x/1.32x/1.01x. Double-complex addition was approximately tied, trailing ATen
by 1-3% across these measurements. Array size, cache state, compiler, and CPU can
change these ratios. There is no universal claim that every operation is faster.

These measurements exclude Python dispatch, TensorIterator, allocation,
multithreading, and BLAS. Installed ATen on this ARM build uses generic complex
packets (2 complex64 or 1 complex128 value), which the compiler may vectorize.
Scalars uses two NEON registers per packet (4 complex64 or 2 complex128 values).
No x86 or scalable-vector performance claims follow from these results.

For the 65,536-element comparison, the maximum observed normalized difference
across these operations was 3.2e-7 for complex64 and 6.5e-16 for complex128,
using `abs(Scalars - ATen) / max(1, abs(ATen))`. Permanent tests additionally
compare division against scaled scalar arithmetic over broad exponent ranges,
zeros, range boundaries, subnormals, infinities, NaNs, and mixed exceptional
packets. Existing tests cover counted loads/stores, negative strides, matrix
products, and in-place vector operations. The scalar extreme-magnitude division
behavior is retained, including `(max + max*i) / (max + max*i) == 1 + 0*i`.

These measurements cover arithmetic, FMA, and division — the operations with
genuine native-vector implementations. The packet API's broader operation
set (roots/powers, exponential/logarithmic, trigonometric, hyperbolic,
masks/selection, half-width utilities, rounding) is covered by
[`complex_simd_operation_audit.md`](complex_simd_operation_audit.md), not by
this performance comparison; most of those map the existing scalar complex
implementation lane-by-lane rather than using native vector transcendentals,
so no PyTorch throughput comparison is claimed for them here.

After splitting the implementation by backend, the full native NEON suite passed
(66 unit tests), as did the CMake setup and both installed-package consumer tests.
The six focused complex SIMD/vector tests passed for SSE2, AVX without AVX2/FMA,
and AVX2/FMA under Rosetta. AVX-512 test executables and fixed-width SVE test
objects compiled successfully; those two targets were not run on matching
hardware. The existing CI SIMD matrix also builds each selected backend.
