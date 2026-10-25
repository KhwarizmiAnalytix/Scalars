Complex SIMD operation audit, 2026-10-25.

Complex SIMD does **not** currently provide the applicable float/double operation
set. All CPU complex backends inherit the same 16 public operation names from
[`complex_simd.h`](../include/backend/cpu/detail/complex_simd.h). Moving the native
implementation into per-backend folders did not fill these API gaps.

The implemented names are `load`, `loadu`, `store`, `storeu`, `set`, `setzero`,
`gather`, `scatter`, `add`, `sub`, `mul`, `div`, `fma`, `neg`, `conj`, and
`accumulate`. `loadu` and `storeu` also have counted overloads. Gather/scatter
support a uniform stride, including negative strides, but do not support an
array of lane indices.

Public operation-name inventory:

| CPU backend | Float | Double | complex<float> | complex<double> |
|---|---:|---:|---:|---:|
| SSE | 71 | 68 | 16 | 16 |
| AVX | 71 | 68 | 16 | 16 |
| AVX2 | 71 | 68 | 16 | 16 |
| AVX-512 | 67 | 71 | 16 | 16 |
| NEON | 71 | 68 | 16 | 16 |
| SVE fixed-width compatibility | 71 | 68 | 16 | 16 |

Counts distinguish names, not overloads, return types, or mathematical
applicability. Internal helpers are excluded. The compiler probes found 15 of
the real operation names on complex packets; `conj` is the additional complex
operation. AVX-512 float lacks `ploadquad` and the three half-width/reduction
names present in its double backend. SSE/AVX/NEON double lack those three
half-width/reduction names. SVE forwards to the fixed-width NEON implementations.

| Family | Complex status and gap |
|---|---|
| Basic arithmetic | `add`, `sub`, `mul`, `div`, `neg`, `fma` present; `sqr` missing |
| Roots and powers | `sqrt`, `invsqrt`, `pow` missing |
| Exponential/logarithmic | `exp`, `expm1`, `exp2`, `exp10`, `log`, `log1p`, `log2`, `log10` missing |
| Trigonometric | `sin`, `cos`, `tan`, `asin`, `acos`, `atan` missing |
| Hyperbolic | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` missing |
| Equality and selection | `eq`, `neq`, `if_else` missing; no `mask_t` alias |
| Mask logic | `not_mask`, `and_mask`, `or_mask`, `xor_mask` missing |
| Memory/packing | Basic and counted loads/stores present; `prefetch`, `copy`, four-value `broadcast`, `ptranspose`, `ploadquad`, indexed gather/scatter missing |
| Half-width utilities | No `simd_half_t`, `half_size`, `loadu_half`, `gather_half`, or `predux_downto4`; applicable shapes need a logical-complex-lane contract |
| Integer packet metadata | No `simd_int_t`/`int_t` aliases for generic packet code |
| Magnitude and components | `conj` present; `abs`, `abs2`, `arg`, callable `real`/`imag` helpers absent. Real/imaginary register fields are available on the packet value |
| Sum reduction | `accumulate` present, with a return-type difference described below |
| Rounding | `ceil`, `floor`, `trunc` absent; adding them requires explicit componentwise semantics |

Real-only ordering should not be copied into complex packets without a separate
policy. This applies to `gt`, `ge`, `lt`, `le`, `min`, `max`, `hmin`, and `hmax`.
`fabs` should have a documented complex magnitude/real-result interpretation
rather than silently applying a real function to components. `signcopy`, `hypot`,
`cbrt`, `cdf`, and `inv_cdf` also require an explicit complex contract; the current
[dtype parity plan](simd_dtype_parity_plan.md) excludes these real-function
contracts for complex packets. An analytic complex extension would be a distinct
API decision.

There is also a reduction contract difference. `simd<float>::accumulate` returns
`double`, while `simd<complex<float>>::accumulate` returns `complex<float>` and
explicitly casts each component reduction back to float. It does not use the
CPU `complex<float> -> complex<double>` policy in
[`accumulate_type.h`](../include/common/accumulate_type.h). A double return type
on the real operation does not itself imply every intermediate is accumulated
in double; those details remain backend-specific.

Scalar complex mathematics is separate from packet support. Scalar `sqrt`,
`pow`, `exp`, `log`, and trigonometric functions already exist via
[`types/complex.h`](../include/types/complex.h) and
[`complex_math.h`](../include/math/complex_math.h). Their existence does not make
`simd<complex<T>>::sqrt`, `pow`, or the other packet methods callable.

The existing real unary/binary/utility suites instantiate float and double.
The dedicated complex suite checks the implemented arithmetic, memory,
reduction, division edge cases, matrix products, and vector primitives. It does
not enforce operation parity, so passing these tests does not establish parity.

Scope and verification: C++20 `requires` probes compiled and ran on native NEON
and on SSE, AVX, and AVX2 under Rosetta. AVX-512 and fixed-width SVE probes
compiled successfully; their remaining name counts were checked against the
headers/compatibility forwarding. These probes check callable declarations and
selected return types, not numerical behavior or every transpose shape. Generated
probe sources and CSV results are in `build_ninja/complex_api_audit/`.

Only CPU `complex<float>` and `complex<double>` SIMD specializations exist.
Complex half/bfloat16 packets and GPU complex SIMD specializations remain absent
from the dispatcher.

The next applicable additions are equality/masks/selection and memory/packing
utilities, followed by `sqr`, roots/power, and complex math methods. Scalar lane
mapping can provide correct branch-cut behavior for the latter before native
vector optimization; it should be documented as scalar mapping.
