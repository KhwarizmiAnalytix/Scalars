Complex SIMD operation audit, 2026-10-25 (revised).

This document previously claimed complex SIMD exposed only 16 of the
applicable float/double operation names. That count described the packet
before it was split per CPU ISA; the split itself
(`include/backend/cpu/detail/complex_simd.h`) already closed almost the
entire gap in the same change, and this document was not updated to match.
The stale "16 operations" table has been removed; this revision audits the
code as it stands.

## CPU complex<float> / complex<double>: at parity

All six CPU backends (SSE, AVX, AVX2, AVX-512, NEON, SVE fixed-width
compatibility) specialize `simd<complex<float>>` / `simd<complex<double>>`
from the same shared `scalars::detail_complex_simd::complex_simd<T, Ops>` in
[`complex_simd.h`](../include/backend/cpu/detail/complex_simd.h), which now
implements the applicable float/double surface:

| Family | Status |
|---|---|
| Basic arithmetic | `add`, `sub`, `mul`, `div`, `neg`, `conj`, `sqr`, `fma` |
| Roots and powers | `sqrt`, `invsqrt`, `pow`, `reciprocal` |
| Exponential/logarithmic | `exp`, `expm1`, `exp2`, `exp10`, `log`, `log1p`, `log2`, `log10` |
| Trigonometric | `sin`, `cos`, `tan`, `asin`, `acos`, `atan` |
| Hyperbolic | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` |
| Equality and selection | `eq`, `neq`, `if_else`, `mask_t` |
| Mask logic | `not_mask`, `and_mask`, `or_mask`, `xor_mask` |
| Memory/packing | `load`/`loadu`/`store`/`storeu` (plain and counted), `prefetch`, `copy`, four-value `broadcast`, `ptranspose`, `ploadquad`, uniform-stride and indexed `gather`/`scatter` |
| Half-width utilities | `simd_half_t`, `half_size`, `loadu_half`, `gather_half`, `predux_downto4`, and half-width `add`/`sub`/`mul`/`fma`/`scatter` |
| Integer packet metadata | `simd_int_t`, `int_t` |
| Magnitude and components | `conj`, `abs`, `abs2`/`norm`, `arg`, `real`, `imag` |
| Classification | `isnan`, `isinf`, `isfinite` |
| Sum reduction | `accumulate`, returning the CPU `complex<T> -> complex<double>` accumulation type from [`accumulate_type.h`](../include/common/accumulate_type.h) |
| Rounding | `ceil`, `floor`, `trunc` (componentwise) |

Deliberately absent, by design rather than oversight: `gt`, `ge`, `lt`, `le`,
`min`, `max`, `hmin`, `hmax` (complex numbers have no total order),
`signcopy`, `hypot`, `cbrt`, `cdf`, `inv_cdf` (no agreed complex contract).
These are `= delete`d on `complex_simd<T, Ops>` so misuse is a compile error
rather than a silent componentwise real-function application. An analytic
complex extension of these would be a distinct, separately-reviewed API
decision, not a parity gap.

`pow`/`exp`/the log family/trig/hyperbolic generally map the
existing scalar complex implementation
([`types/complex.h`](../include/types/complex.h),
[`complex_math.h`](../include/math/complex_math.h)) lane-by-lane rather than
claiming native vector transcendentals; this is a packet API completeness
decision. `sqrt` now uses real SIMD square roots, arithmetic, division,
and branch selection within the conservative division bounds, retaining
the scaled scalar fallback for extreme/exceptional inputs and packets
containing a zero complex lane. `acos` uses two such packet roots in
Kahan's formula, followed by scalar `atan2`/`asinh`, with a scalar fallback
outside those bounds. Rounding reuses the real backend's componentwise
operations, including native NEON truncation and negation. Componentwise
rounding and negation use interleaved registers so surrounding packing
shuffles can cancel. Arithmetic, FMA, and bounded division also use native vectors (see
[`complex_simd_performance.md`](complex_simd_performance.md)).

`Testing/Cxx/TestSimdComplexParity.cxx` now checks the applicable API and
return types with C++20 `requires` expressions, and verifies masks,
selection, packing, indexed memory, transpose, half-width helpers, math,
signed zeros, special values, square-root range boundaries, and widened
accumulation. The focused complex suites run on NEON and on SSE2,
AVX without AVX2/FMA, and AVX2/FMA under Rosetta. AVX-512 executables and
fixed-width SVE objects compile; those targets have not run on matching
hardware. See the performance document for fresh measurements.

## Reduced precision, GPU, and Metal: now closed

1. **CPU `complex<half>` / `complex<bfloat16>`**
   ([`backend/cpu/reduced/complex_simd.h`](../include/backend/cpu/reduced/complex_simd.h)):
   stores real/imag lanes as the existing reduced real packet type, widens
   the whole operand once to a pair of `complex<float>` packets, runs the
   complete operation through `simd<complex<float>>`, and narrows once at
   the end -- deliberately *not* reusing `complex_simd<T, Ops>` directly for
   T=half/bfloat16, since narrowing after every intermediate real/imag
   sub-operation (as that shared template does) would overflow half's
   ~65504 range far sooner than a real division ever would. This gets the
   entire existing `complex<float>` operation set (arithmetic,
   roots/powers/transcendentals, masks, half-width utilities) without
   depending on `scalars::complex<half>`/`complex<bfloat16>` supporting
   those operations themselves. `accumulate` returns `complex<float>`,
   matching the (now also added) `complex<bfloat16> -> complex<float>`
   policy in `accumulate_type.h` (previously only `complex<half>` had one --
   an oversight relative to every other half/bfloat16 pair in that table).
2. **GPU `simd<complex<float>>` / `simd<complex<double>>`**
   ([`backend/gpu/complex/simd.h`](../include/backend/gpu/complex/simd.h)):
   scalar (`size == 1`), shared by CUDA and HIP, wrapping the scalar complex
   implementation already in `types/complex.h`/`complex_math.h` (already
   `SCALARS_HOST_DEVICE`) behind the same `simd<T>` interface
   `gpu/float/simd.h` and `gpu/double/simd.h` use for real types. Unverified
   on this machine beyond compilation (no CUDA/HIP device or toolchain
   present) -- same caveat as AVX-512/SVE in the performance doc.
3. **Metal complex64**
   ([`kernels.metal`](../include/backend/gpu/metal/kernels.metal)'s complex64
   block, `dtype::complex64` in `metal_dispatch.h`): `float2`-backed
   add/sub/mul/div/fma/neg/conj/sqrt/exp/log/sin/cos/tanh/fabs kernels, at
   parity with Metal's existing `float32` kernel set (not complex128 --
   Metal has no `double`, so there is no float32/float64 parity target for
   complex128 to reach either). `reduce_sum`/`min`/`max` are not provided:
   min/max have no complex ordering (consistent with every other backend
   here), and a complex sum needs a two-wide accumulator, which is a new
   public API shape (`metal_dispatch.h` currently returns `float`), not just
   a kernel -- left for follow-up work. Verified on this machine's Metal
   device (`Testing/Cxx/TestMetalComplex.cxx`), not just compiled.
