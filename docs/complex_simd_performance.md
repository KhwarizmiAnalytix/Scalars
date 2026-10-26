Historical complex SIMD measurements on macOS ARM64, 2026-10-25, using PyTorch 2.13.0
(`cf30153c4c131c8164ee7798e5022d810682e2cb`) and Clang 22.1.2.

The table below records the implementation measured on that date. Current
loads/stores still use `memcpy` and vector shuffles, preserving object aliasing
rules. Add/subtract now operate directly on the packet's separate real/imaginary
registers, avoiding internal packing shuffles in arithmetic chains.

Division checks that each nonzero component is in `[2^-30, 2^31)` for
float or `[2^-250, 2^251)` for double, and that every divisor is nonzero. Products,
the denominator, and reciprocal then stay in range. NEON uses a reciprocal estimate
with two float or three double Newton refinements. SSE, AVX/AVX2, and AVX-512 use
their hardware vector division for the reciprocal. Intermediates follow the real
backend's FMA support. This has rounding differences from scalar
division and does not promise correctly rounded results. If any lane fails the
check, only the affected lanes use scalar repair. Benign operands replace unsafe
lanes before the vector formula runs, so exceptional lanes do not execute invalid
fast arithmetic. The scalar implementation now also uses bounded direct division
for ordinary components and retains scaling for extreme magnitudes.
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

Square root now uses real SIMD square roots, arithmetic, division, and branch
selection within the same conservative component bounds. Packets containing a
zero complex lane or an out-of-range/nonfinite component keep the scaled scalar
fallback. Arccosine uses two packet roots in Kahan's branch-preserving formula,
then scalar `atan2` and `asinh`; it retains the scalar fallback outside those
bounds. Componentwise rounding and negation operate on interleaved registers so
surrounding load/store shuffles can cancel. NEON's real backend now uses native
truncation and negation instructions. SSE2 rounding preserves large values,
infinities, NaNs, and signed zeros instead of overflowing its integer conversion.

Magnitude uses a scaled SIMD ratio rather than scalar lane mapping. With a math
provider, exponential uses real vector math for finite inputs with real component
in `(-80, 80)` for float or `(-700, 700)` for double; other lanes retain scalar
semantics. NEON/SVE share sine/cosine range reduction and use provider `atan2`
for argument. SLEEF/SVML logarithms have a bounded squared-magnitude vector path;
Accelerate retains scalar complex logarithms because its small-array composition
was slower in local measurements. Other complex transcendentals still use scalar
lane mappings. These paths inherit the real provider's accuracy limits.

Dot products use four packet accumulators and combine their widened reductions.
Their accumulation order differs from both scalar loops and the former single
packet accumulator. Native CMake benchmarks now cover bounded input arrays,
arithmetic chains, reductions, tails, strides, exceptional lanes, and several
working-set sizes. Use a Release build with instrumentation disabled.

To reproduce with an installed PyTorch wheel and C++ compiler:

```sh
python3 Scripts/benchmark_complex_simd.py --backend neon --cxx clang++ \
    --elements 4096 --iterations 1000 --operations all
```

The comparison now covers 41 operations in both complex64 and complex128.
Use `--operations sqrt acos` to select a subset; the default remains
`add mul div fma`. Use `--backend sse|avx|avx2|avx512|neon|sve` on the
corresponding host architecture; the default is NEON on ARM64 and AVX2 on x86.

The benchmark compiles identical packet loops against Scalars and the actual
installed ATen headers with `-O3 -DNDEBUG -std=c++20` and matching ISA flags.
It compiles Scalars' scalar complex math source and links the installed
`torch_cpu`/`c10` libraries for ATen's compiled math helpers, including its
square-root/arccosine implementation. It uses preallocated arrays, seed 123,
independent real/imaginary values in `[-2, 2]`, 20 warmups, and medians of nine
batches of 1,000 iterations. Equality tests make half of the inputs equal.
Measurement order alternates between implementations, with a memory barrier
after each loop invocation to preserve output writes. PyTorch is optional for
this standalone benchmark; regular CMake builds and tests do not depend on it.

The table records the final complete run with 4,096 complex elements.
Times are nanoseconds per complex element; speedup is `ATen / Scalars`, so
values below 1 mean Scalars is slower. Small margins can change between runs.

| Operation | Scalars complex64 | ATen complex64 | Speedup | Scalars complex128 | ATen complex128 | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| `add` | 0.132 | 0.179 | 1.36x | 0.263 | 0.264 | 1.00x |
| `sub` | 0.137 | 0.178 | 1.30x | 0.263 | 0.266 | 1.01x |
| `mul` | 0.264 | 0.375 | 1.42x | 0.528 | 0.617 | 1.17x |
| `div` | 0.749 | 1.156 | 1.54x | 1.415 | 1.926 | 1.36x |
| `fma` | 0.265 | 0.511 | 1.93x | 0.706 | 0.732 | 1.04x |
| `sqr` | 0.204 | 0.293 | 1.43x | 0.397 | 0.440 | 1.11x |
| `abs` | 1.557 | 1.239 | 0.80x | 1.530 | 1.757 | 1.15x |
| `arg` | 3.273 | 3.138 | 0.96x | 7.494 | 7.495 | 1.00x |
| `sqrt` | 0.736 | 5.131 | 6.97x | 1.386 | 6.691 | 4.83x |
| `invsqrt` | 1.568 | 8.456 | 5.39x | 3.369 | 12.015 | 3.57x |
| `reciprocal` | 0.705 | 1.800 | 2.55x | 2.997 | 3.895 | 1.30x |
| `exp` | 5.513 | 5.078 | 0.92x | 13.352 | 14.014 | 1.05x |
| `expm1` | 10.427 | 9.082 | 0.87x | 16.220 | 14.889 | 0.92x |
| `exp2` | 5.405 | 5.394 | 1.00x | 8.775 | 9.622 | 1.10x |
| `exp10` | 5.999 | 6.102 | 1.02x | 8.324 | 8.966 | 1.08x |
| `log` | 6.334 | 6.337 | 1.00x | 11.617 | 11.931 | 1.03x |
| `log1p` | 8.201 | 8.475 | 1.03x | 12.402 | 12.450 | 1.00x |
| `log2` | 6.527 | 6.401 | 0.98x | 11.670 | 12.113 | 1.04x |
| `log10` | 6.542 | 6.243 | 0.95x | 11.638 | 11.952 | 1.03x |
| `sin` | 7.882 | 7.864 | 1.00x | 14.403 | 14.699 | 1.02x |
| `cos` | 9.250 | 8.863 | 0.96x | 14.159 | 14.480 | 1.02x |
| `tan` | 10.885 | 11.220 | 1.03x | 15.916 | 16.473 | 1.03x |
| `asin` | 28.050 | 30.297 | 1.08x | 50.036 | 44.244 | 0.88x |
| `acos` | 8.875 | 27.011 | 3.04x | 20.282 | 39.174 | 1.93x |
| `atan` | 39.238 | 40.361 | 1.03x | 47.781 | 41.547 | 0.87x |
| `sinh` | 7.980 | 8.385 | 1.05x | 13.629 | 14.439 | 1.06x |
| `cosh` | 8.946 | 8.550 | 0.96x | 14.174 | 15.014 | 1.06x |
| `tanh` | 10.997 | 11.726 | 1.07x | 15.966 | 16.413 | 1.03x |
| `asinh` | 27.187 | 28.196 | 1.04x | 42.938 | 46.231 | 1.08x |
| `acosh` | 28.318 | 30.048 | 1.06x | 49.084 | 55.595 | 1.13x |
| `atanh` | 35.607 | 38.230 | 1.07x | 32.650 | 29.743 | 0.91x |
| `pow` | 12.519 | 13.587 | 1.09x | 23.364 | 22.652 | 0.97x |
| `ceil` | 0.224 | 0.194 | 0.87x | 0.193 | 0.197 | 1.02x |
| `floor` | 0.213 | 0.196 | 0.92x | 0.191 | 0.195 | 1.02x |
| `trunc` | 0.226 | 0.220 | 0.97x | 0.205 | 0.188 | 0.92x |
| `neg` | 0.215 | 0.242 | 1.13x | 0.194 | 0.193 | 1.00x |
| `conj` | 0.241 | 0.287 | 1.19x | 0.219 | 0.183 | 0.84x |
| `real` | 0.162 | 0.202 | 1.25x | 0.212 | 0.268 | 1.27x |
| `imag` | 0.221 | 0.229 | 1.04x | 0.226 | 0.269 | 1.19x |
| `eq` | 0.223 | 0.403 | 1.81x | 0.449 | 0.635 | 1.41x |
| `neq` | 0.245 | 0.423 | 1.72x | 0.499 | 0.678 | 1.36x |

These results exclude Python dispatch, TensorIterator, allocation,
multithreading, and BLAS. Installed ATen on this ARM build uses generic complex
packets (2 complex64 or 1 complex128 value), which the compiler may vectorize.
Scalars uses two NEON registers per packet (4 complex64 or 2 complex128 values).
The timed loops use matching compiler/ISA flags, but compiled scalar math helpers
come from their respective libraries. No x86 or scalable-vector performance
claims follow from this ARM run.

Magnitude, angle, component projections, and equality results are encoded as
`complex(result, 0)` to match ATen's complex packet output. Scalars' actual public
APIs return real packets or masks; these timings therefore do not represent
PyTorch tensor projection or boolean-output kernels. `exp10` has no ATen packet
method, so its baseline composes ATen's scalar map with `exp(z * ln(10))`.
Reductions are excluded because Scalars widens CPU complex64 accumulation to
complex128, which differs from ATen's complex64 packet accumulation contract.

The maximum observed normalized differences were 2.9e-07 for complex64
and 6.5e-16 for complex128, using
`abs(Scalars - ATen) / max(1, abs(ATen))`. All 82 operation/dtype comparisons passed
the benchmark limits of 2e-6 and 2e-14 respectively. These random inputs exercise
ordinary finite values. Permanent tests separately cover branch cuts, signed
zeros, exceptional/mixed division packets, broad exponent ranges, subnormals,
range boundaries, infinities, NaNs, and extreme-magnitude square roots.
The square-root tests check the scaled square identity at maximum magnitudes,
where some standard-library references overflow. They also compare relative
component errors around fast-path bounds against the scaled scalar algorithm.

Scalars is faster for several arithmetic and root operations in this run, but
some operations remain slower, including complex64 magnitude and exponential,
and complex128 conjugation. Most exponential/logarithmic/trigonometric/hyperbolic
operations still map scalar complex math over lanes. API completeness does not
imply native vector transcendentals or universally faster execution; see
[`complex_simd_operation_audit.md`](complex_simd_operation_audit.md) for the
operation contracts.

Validation for the historical measurements: all 71 native NEON unit tests and the CMake setup
tests passed. The focused complex test executables passed 11 cases each for SSE2,
AVX without AVX2/FMA, and AVX2/FMA under Rosetta. AVX-512 test executables and
fixed-width SVE test objects compiled successfully; those targets were not run
on matching hardware.
