# SIMD parity for half, bfloat16, and complex

Status: proposed implementation plan; no SIMD implementation changes.
Repository inspected: current working tree, 2026-10-24.

## 1. Goal and scope

Extend the existing global `simd<T>` interface to these six existing scalar types:

- `scalars::half` and `scalars::bfloat16`;
- `scalars::complex<float>` and `scalars::complex<double>`;
- `scalars::complex<half>` and `scalars::complex<bfloat16>`.

CPU parity means that every selected CPU backend provides the applicable memory,
arithmetic, math, mask, reduction, and packet utility operations. CUDA/HIP parity
is a separate, required completion milestone for their existing scalar-per-thread
backend. Metal uses a different float-only dispatch interface today; extending
that interface is a separate project, not a dependency of `simd<T>` CPU parity.

Parity does not imply identical throughput across types. It requires documented
precision, actual vector execution of the core CPU paths, and explicit visibility
of operations that use scalar math within a packet. Complex ordering and min/max
are unsupported because this library defines no complex ordering.

## 2. Findings in this working tree

| Area | Current implementation | Consequence |
| --- | --- | --- |
| Dispatch | `include/backend/simd.h` includes only float/double specializations | Add dtype dispatch after the underlying real packets are defined |
| CPU backends | SSE, shared AVX/AVX2, AVX-512, NEON | All four implementation families need the new dtypes |
| SVE | `cpu/sve/{float,double}/simd.h` requires 128-bit SVE and includes NEON | Reuse the same new NEON packets under this existing restriction; do not claim scalable SVE |
| Reduced storage | `half` and `bfloat16` are aligned two-byte wrappers | Existing scalar storage and public aliases can be preserved |
| half conversion | `detail/vec_half.h` contains scalar conversion helpers under AVX capability macros | Introduce packet conversion separately; gate by actual conversion features |
| Complex storage | Interleaved public `real_`, `imag_`; alignment is twice component size | Verify layout, then preserve it in vector memory operations |
| Reduced complex | `complex<half>` has a limited specialization; bf16 uses the general template | Repair the scalar contract before using it as a SIMD oracle |
| Accumulation | `acc_type` promotes half/bf16 to float and CPU complex<float> to complex<double> | Add missing complex<bf16> specializations and distinguish computation from accumulation |
| Existing float reduction | For example, AVX float `accumulate` sums in float and then returns double | A double return type alone is not evidence of double accumulation |
| Tests | SIMD F1/F2/H/utility suites are instantiated for float/double | Add dtype-aware coverage and numerical references |
| CPU disabled | `SCALARS_CPU_BACKEND=no` intentionally has no host SIMD specialization | Preserve that behavior; expose unsupported capabilities without borrowing GPU headers |
| CUDA/HIP | GPU float/double `size == 1` | Keep one logical element per thread when adding dtypes |

Existing uncommitted implementation work is the baseline for this plan. Do not
replace or reset it while carrying out these changes.

## 3. How to follow PyTorch and Eigen

Use their architectural patterns while retaining Scalars' existing API. Add no
PyTorch or Eigen runtime dependency. Resolve reference revisions to commit IDs
in the first implementation change; the links below are browsing references and
the `main`/nightly links can move. Follow the repository's independent
implementation policy rather than importing source files.

| Reference | Pattern to use in Scalars |
| --- | --- |
| [PyTorch vec256 16-bit base](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cpu/vec/vec256/vec256_16bit_float.h) | A dense reduced packet widens to two float packets; share conversion and operation plumbing |
| [PyTorch half specialization](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cpu/vec/vec256/vec256_half.h) | Implement reduced arithmetic through float operations and narrow the result |
| [PyTorch reduced functional layer](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cpu/vec/functional_bfloat16.h) | Offer widened compute helpers so an explicitly widened kernel avoids conversion after every intermediate operation |
| [PyTorch complex float](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cpu/vec/vec256/vec256_complex_float.h) | Interleaved complex storage; logical complex lanes; scalar mapping where exceptional-value semantics make a vector implementation incomplete |
| [PyTorch AVX-512 reduced packets](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cpu/vec/vec512/vec512_bfloat16.h) | Wide packet packing, lane-order correction, and software bf16 conversion |
| [PyTorch NEON bf16](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cpu/vec/vec128/vec128_bfloat16_neon.h) | Separate storage support from optional compiler/hardware bf16 features |
| [Eigen generic packet traits](https://libeigen.gitlab.io/eigen/docs-nightly/GenericPacketMath_8h_source.html) | Advertise capabilities per operation instead of assuming every packet supports every real-number function |
| [Eigen AVX packet math](https://libeigen.gitlab.io/eigen/docs-nightly/AVX_2PacketMath_8h_source.html) | Typed reduced packet wrappers, conversion-based arithmetic, and packet utilities |
| [Eigen AVX complex packets](https://libeigen.gitlab.io/eigen/docs-nightly/AVX_2Complex_8h_source.html) | Complex multiply/conjugation, complex-aware transpose, and shared complex division helpers |

One deliberate choice: use PyTorch-style dense reduced storage. The inspected
Eigen AVX `Packet8h`/`Packet8bf` store eight reduced values, corresponding to one
eight-lane float compute packet. Scalars should store sixteen reduced values in
a 256-bit packet and widen to two eight-lane float packets. These are different
packet-width policies; following both projects does not mean mixing them.

## 4. Contract and packet design

### 4.1 Preserve the entry point

Keep `simd<T>::simd_t`, `value_t`, `size`, and existing method names. Add internal
strongly typed wrappers for new packets rather than aliasing half and bf16 to
the same integer vector type. Preserve float/double signatures and behavior.
Do not change scalar sizes, alignments, or installed aliases.

Add an auxiliary `scalars::simd_traits<T>` with:

- `supported`, `size`, alignment, component type, compute type, and accumulation
  type;
- per-operation status: unsupported, scalar mapping, composed SIMD, or native
  SIMD; composed SIMD means arithmetic assembled from vector operations;
- capabilities for partial load/store, half-width packets, transpose shapes,
  and ordered comparisons.

Populate the traits for float/double too, so callers can use one contract.
Keep capability detection separate from instantiation of an unsupported
`simd<T>`. Compile-time coverage must verify advertised operations exist.

### 4.2 Dense CPU lane counts

`size` always counts logical scalar values, including complete complex values.

| Selected backend | float | double | half / bf16 | complex<float> | complex<double> | complex<half> / complex<bf16> |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SSE | 4 | 2 | 8 | 2 | 1 | 4 |
| AVX / AVX2 | 8 | 4 | 16 | 4 | 2 | 8 |
| AVX-512 | 16 | 8 | 32 | 8 | 4 | 16 |
| NEON | 4 | 2 | 8 | 2 | 1 | 4 |
| Existing SVE compatibility path | 4 | 2 | 8 | 2 | 1 | 4 |
| CUDA / HIP device | 1 | 1 | 1 | 1 | 1 | 1 |

The new CPU counts are proposed contract values. SVE remains subject to its
current fixed-width build restrictions. No native SVE implementation is implied.

### 4.3 Storage, compute, and accumulation

For half/bf16, store packed 16-bit encodings. Provide:

```cpp
// Sketch: final helper names can follow local conventions.
to_float_pair(packed)             // {low_float_packet, high_float_packet}
from_float_pair(low, high)        // packed result, round to nearest even
load_float_pair(ptr)              // load and widen
store_float_pair(low, high, ptr)  // narrow and store
```

Implement ordinary packet operations as widen -> compute -> narrow. Round at
each public reduced-result operation, matching stored scalar arithmetic. Keep
an explicitly widened API for kernels that choose float intermediates across
multiple operations. Such kernels have a different rounding contract and must
not silently replace ordinary reduced arithmetic. This distinction follows the
functional-layer motivation in the PyTorch reference above.

For reduced complex, widen one packed complex packet to two `complex<float>`
packets, invoke shared complex operations, then narrow each output component
once. Never evaluate complex multiplication through repeated half/bf16 scalar
products: premature rounding changes both accuracy and overflow behavior.

Use `acc_type<T, false>` for new CPU reductions and `acc_type<T, true>` on GPU.
That gives float for reduced real, complex<float> for reduced complex, and
complex<double> for CPU complex<float>. Widen before reduction when the trait
requires it. Preserve existing float/double reduction behavior; do not claim
that it already implements these wider intermediates. Clarify the role of
`accumulation_type_t`, which currently follows a different policy.

### 4.4 Masks, memory, and packet utilities

- One truth value per logical lane. For complex equality, combine real and
  imaginary comparisons; `neq` is true when either component differs, including
  unordered NaN cases. Duplicate the logical truth value into both physical
  component positions when needed for selection.
- For new integer mask memory overloads, use `uint16_t` per reduced real lane,
  `uint32_t` per float/reduced complex lane, and `uint64_t` per double complex
  lane. Encode true as all bits set. Define mask logic using integer bits, never
  floating comparisons of all-ones masks (which are NaN bit patterns).
- `if_else` selects a complete complex value. Gather/scatter strides and index
  arrays count logical values, not underlying components. Repeated scatter
  indices use a documented increasing-lane overwrite order; native scatter
  requires unique indices or a collision fallback.
- Verify `sizeof`, `alignof`, trivial copyability, standard layout, and component
  offsets. Use safe object-representation loads/stores or component access;
  adjacency alone does not authorize arbitrary pointer aliasing across objects.
- `load`/`store` retain an explicit alignment precondition; `loadu`/`storeu`
  accept arbitrary scalar alignment. `copy` retains its aligned destination
  precondition where that is the existing backend contract.
- Add count-limited load/store for `0 <= count <= size`. Inactive lanes are zero
  and never read or write memory. These are additive helpers, not prerequisites
  for users of the existing full-packet API.
- Support `set(value_t)`, compatible real broadcasts, `setzero`, prefetch, copy,
  four-value broadcast, gather/scatter, and valid transpose shapes. Complex real
  broadcasts have zero imaginary part.
- Preserve complete complex pairs during broadcast, shuffle, and transpose.
  Distinguish half-width packet utilities from the `half` dtype. Disable
  half-width utilities for one-logical-lane packets. Define a new four-lane
  reduction helper only where logical width makes it meaningful; advertise
  support for the legacy `predux_downto4` name accordingly.
- Implement supported half-width load/gather/add/fma/scatter overloads and
  `ploadquad` semantics in logical lanes. Make unsupported utility shapes fail
  clearly at compile time rather than silently doing nothing.

## 5. Applicable operation matrix

| Operation family | half / bf16 | All four complex component types |
| --- | --- | --- |
| Memory, broadcast, gather/scatter, copy, select | Full parity | Full parity, operating on complete complex lanes |
| `add`, `sub`, `mul`, `div`, `neg`, `sqr` | Float compute, reduced result | Complex arithmetic; float compute for reduced components |
| `fma` | Follow underlying float fused/nonfused capability; narrow once | Defined as complex multiply-add with documented component evaluation order, without claiming one correctly rounded complex FMA |
| `eq`, `neq`, mask logic | Full parity | Complete-value equality/inequality and logical mask operations |
| `gt`, `ge`, `lt`, `le` | Full parity | Unsupported |
| `min`, `max`, `hmin`, `hmax` | Full parity with explicit NaN/zero policy | Unsupported |
| `fabs`, `signcopy` | Bitwise sign handling where applicable | Add `abs`, `abs2`, `arg`, `real`, `imag`, `conj`; no componentwise `fabs` or `signcopy` disguised as complex math |
| `accumulate` | Float result and intermediates | Corresponding complex accumulation type |
| `sqrt`, `invsqrt`, `pow` | Reuse float backend, narrow | Principal complex semantics, with scalar mapping until vector cases are validated |
| `exp`, `expm1`, `exp2`, `exp10` | Reuse float backend, narrow | Complex extensions; stable small-input `expm1` |
| `log`, `log1p`, `log2`, `log10` | Reuse float backend, narrow | Principal branch; stable small-input `log1p` |
| trig, inverse trig, hyperbolic, inverse hyperbolic | Reuse float backend, narrow | Corresponding complex functions |
| `ceil`, `floor`, `trunc` | Full parity | Explicit componentwise semantics, matching the inspected PyTorch packet convention |
| `hypot`, real `cbrt`, `cdf`, `inv_cdf` | Full real parity through float | Unsupported in this contract; any analytic extension needs a separate named API |

Complex-to-real functions return a `real_packet_t` containing exactly the same
number of logical results, with its own load/store support. Do not pretend that
a full `simd<component>::simd_t` has the same lane count as a complex packet.
Provide a documented `fabs` alias to magnitude only if generic callers require
that spelling, with the same real-result return contract.

For new reduced `min`/`max`, choose NaN propagation and deterministic zero ties
(`min(+0,-0) = -0`, `max(+0,-0) = +0`), including horizontal versions. Normalize
backend primitives to that policy. Record existing float/double differences
without silently changing their behavior in this project.

## 6. Ordered implementation changes

### Change 1 — Freeze the contract and repair scalar prerequisites

Files: `types/complex.h`, `core/accumulate_type.h`, scalar traits/math headers,
existing scalar tests, and a new SIMD traits header.

1. Inventory actual float/double signatures per backend, including overloads
   and utility shapes. Establish the matrix above as the review checklist.
2. Complete reduced complex division and conversions through complex<float>;
   make bf16 multiplication/division follow the same widened policy as half.
   Provide a reliable promoted scalar math path instead of requiring
   `std::complex<half>` or `std::complex<bf16>` math implementations.
3. Add missing CPU/GPU `AccumulateTypeDevice<complex<bfloat16>, ...>` entries.
   Check promotion and runtime accumulation metadata agree for all eight dtypes.
4. Verify layout and add traits for current float/double backends. Record
   exceptional-value behavior for functions that new packets will reuse.
5. Pin reference commits and record relevant algorithms, without source imports.

Exit: scalar reduced complex is a trustworthy reference; new type contracts,
return types, and unsupported operations are compile-time testable. Existing
float/double tests remain green.

### Change 2 — Conversion foundation and ISA feature probes

Files: proposed `backend/detail/reduced_convert.h` plus architecture conversion
helpers; `cmake/simd.cmake`, exported configuration, and `cmake/summary.cmake`.

1. Implement exact 16-bit storage moves and packet widening/narrowing.
2. half: use F16C on supported x86 targets, AVX-512 conversion when available,
   and ARM conversion instructions behind exact feature checks. Provide SIMD
   software conversions for SSE2 and targets lacking hardware conversion.
   A temporary scalar reference fallback is marked as such and is not the
   completed core conversion path.
3. Bf16: widen through zero-extension and a 16-bit shift; narrow with round-to-
   nearest-even bias, explicit NaN handling, and correct packing order.
   Match the repository's scalar NaN policy (`0x7fc0` for bf16 narrowing), even
   when the reference projects or native instructions use another payload.
4. Probe F16C, AVX-512 subsets, optional AVX-512 FP16/BF16, ARM FP16 conversion
   and arithmetic, and ARM BF16 independently. Compile a representative
   intrinsic for each, rather than defining a feature macro without probing it.
5. Retain AVX without AVX2: use 128-bit integer chunks for unsupported 256-bit
   integer operations. Do not infer FMA, FP16, or BF16 arithmetic from AVX2 or
   AVX-512 selection. Preserve compile-only probing for cross-compilation.
6. Export actual feature definitions through installed CMake targets. Document
   that an ISA-selected binary still needs matching hardware at execution;
   runtime CPU multiversioning is a separate concern.

Exit: exhaustive widening tests and narrowing boundary tests pass, including
lane order, NaN policy, subnormals, and builds without optional ISA extensions.

### Change 3 — Dense half/bfloat16 packets

Implement AVX2 and NEON first as independent reference backends, followed by
SSE, AVX-only, AVX-512, and the current SVE compatibility path.

Files: proposed `backend/detail/reduced_simd.h`,
`backend/cpu/<isa>/{half,bfloat16}/simd.h`, dispatcher, and reduced SIMD tests.

1. Add storage/masks, aligned and unaligned memory, counted memory, broadcast,
   and arithmetic. Share operation plumbing between half and bf16.
2. Reuse the float backend for every real math function in the current surface,
   including `hypot`, `cbrt`, `cdf`, and `inv_cdf`. Inherit and document provider
   accuracy; identify scalar-mapped math through traits.
3. Implement bit-preserving sign operations and masks directly on storage.
4. Complete horizontal reductions, gather/scatter, transposes, half-width
   overloads, and remaining utility capabilities. Keep float intermediates in
   sum reductions until the final float scalar result.
5. Add widened load/compute/store helpers and demonstrate a float-intermediate
   kernel with distinct, explicitly tested rounding semantics.

Exit: every applicable real float/double operation has a reduced counterpart on
each CPU backend; core conversions/arithmetic/masks run as SIMD. Scalar math
fallbacks remain visible. No dependency on optional native 16-bit arithmetic.

### Change 4 — Complex<float> and complex<double> packets

Files: proposed `backend/detail/complex_simd.h`, architecture complex packet
helpers and dtype headers, plus complex SIMD tests.

1. Add interleaved packets, complete-value masks, memory operations, gather/
   scatter, broadcast, conjugation, negation, addition, and subtraction.
2. Implement multiplication with real/imag shuffles and multiply/add-subtract;
   add component FMA variants only under a documented evaluation policy.
3. Begin division with a correct scalar mapping. Add a scaled vector algorithm
   for proven-safe finite lanes, with a conservative eligibility mask and scalar
   repair for exceptional or extreme lanes. A ratio formula alone does not
   establish overflow safety. Validate that unselected computations cannot
   introduce unwanted floating exceptions in a strict mode.
4. Implement accurate magnitude via scaled hypot, phase via atan2, and principal
   sqrt/log/exp paths. Add the remaining applicable functions from the matrix.
   Use explicit scalar mapping where needed for branch cuts and exceptional
   inputs; the PyTorch complex reference also uses this approach for several
   functions. Do not advertise those functions as native SIMD.
5. Provide stable `log1p`/`expm1` near zero. Preserve signed imaginary zero on
   branch cuts. Add reductions and transpose operations on whole complex lanes.

Exit: complete applicable complex API on every CPU backend; core memory,
selection, add/sub/mul/conjugation and validated finite division cases use SIMD.
Exceptional division lanes have a tested scalar repair path. A scalar-mapped
division implementation is an intermediate delivery, not this completion gate.

### Change 5 — Complex<half> and complex<bfloat16> packets

Files: proposed reduced-complex shared adapter and architecture dtype headers.

1. Compose reduced conversion with the validated complex<float> implementation.
2. Add storage-level conjugation/negation, complete-lane comparison and select,
   memory utilities, and dtype-specific mask conversion.
3. Keep float components throughout each public complex operation and narrow
   only its final real/imag results. Sum in complex<float>, including across
   both widened packets.
4. Return compact reduced real packets for magnitude/phase/component outputs;
   provide widened helper variants for callers retaining float results.
5. Reuse all meaningful complex math functions and utilities from Change 4.

Exit: both reduced complex dtypes satisfy the same applicable operation matrix
and backend coverage as complex<float>, subject to documented precision and
scalar-mapped math.

### Change 6 — CUDA/HIP device parity

Files: proposed GPU dtype specializations, device dispatch, scalar host/device
annotations where needed, and `Testing/Cxx/TestGpuSimd.cu` extensions.

1. Preserve `size == 1`; use float / complex<float> intermediates for reduced
   types, and device-supported complex math or explicit component formulas.
2. Keep CPU intrinsic headers out of device passes. Do not call host-only
   standard complex math from device code. Test nvcc/hipcc host passes too.
3. Apply the same applicable API, rounding, masks, and accumulation contracts.
   Mark CPU-only packet utilities unsupported where they have no GPU analogue.
4. Treat `half2`/packed bf16 GPU work as an optional later optimization. It must
   not change the public logical width of the current scalar backend.

Exit: CUDA and HIP device tests pass for all supported new types and mixed
host/device translation units. Do not label this milestone validated on GPU
hardware when only compiler checks have run.

### Change 7 — Native acceleration and release gates

1. Evaluate optional native FP16 arithmetic and native BF16 conversion/dot
   features. Keep widened semantics as the baseline. Native BF16 dot-product
   instructions do not replace general elementwise BF16 arithmetic.
2. Compare rounding, NaN handling, FTZ/DAZ behavior, and FMA semantics before
   enabling any native path. A direct 16-bit FMA need not match float FMA
   followed by narrowing. Put semantically different fast paths behind an
   explicit policy rather than changing results by hardware accident.
3. Benchmark complete kernels, inspect generated assembly, and finish the
   published per-dtype/per-operation/per-backend capability matrix.
4. Update installation consumers and README, including current CPU/GPU scope,
   precision policies, SVE limitations, and math-provider requirements.

Exit: all required coverage gates below pass, and performance claims have
measurements on the corresponding hardware. Native 16-bit arithmetic is an
optimization milestone, not a prerequisite for meaningful SIMD support.

## 7. Verification and completion gates

### Correctness

- Instantiate memory/arithmetic/mask/utility tests for all eight dtypes.
  Adapt generators to produce float/double values before narrowing; standard
  uniform distributions and `std::complex<T>` math cannot be assumed to work
  directly with custom reduced scalar types.
- Exhaust all 65,536 encodings of each reduced real type for widening. For
  narrowing, test every adjacent finite representable pair's midpoint and
  neighboring float values, exponent transitions, overflow/underflow, infinities,
  signed zeros, and representative signaling/quiet NaN payloads.
- Require exact storage preservation for memory/permutation operations. For
  conversion and finite basic arithmetic, require agreement with an independent
  scalar bit-level reference under the stated rounding/FMA policy. Do not use
  the same packet conversion helper to construct the expected result.
- Record per-function error bounds: destination-format ULP where meaningful,
  relative error away from zero, absolute error near zero, and separate NaN,
  infinity, and signed-zero checks. Float-provider error can change a final
  reduced rounding decision; do not claim correctly rounded transcendentals.
- Test complex cancellation, huge/tiny components, division by zero, purely
  real/imaginary values, and NaNs/infinities independently in both components.
  Check branch cuts of sqrt/log/pow and inverse functions from both sides,
  explicitly including imaginary `+0` and `-0`.
- Exercise all tail counts, mask truth tables, arbitrary scalar alignment,
  canary-protected stores, lane-unique shuffles, transpose shapes, gather/scatter
  indices, and reductions. Add cancellation tests that distinguish promoted
  accumulation from summing narrowly and casting the result afterward.
- Test the default rounding environment plus selected alternative rounding and
  FTZ/DAZ settings. Document any environmental restrictions or normalize the
  relevant conversions; never change global floating-point state silently.
- Use optional pinned PyTorch/Eigen comparison harnesses for overlapping
  semantics. Their packet widths, NaN payloads, and reduced complex coverage
  need not match Scalars, so they are comparison references, not universal
  expected-output generators.

Suggested files: `TestSimdTraits.cxx`, `TestSimdConvert.cxx`,
`TestSimdReduced.cxx`, `TestSimdComplex.cxx`, `TestSimdComplexReduced.cxx`, and
shared dtype-aware test helpers. Reuse the existing F1/F2/H/utility organization
where practical; avoid duplicating a complete suite per architecture.

### Build and hardware matrix

- Separate SSE2, AVX-only, AVX2, and AVX-512 configurations. Deliberately disable
  optional features to catch accidental AVX2/FMA/F16C/FP16/BF16 dependencies.
- Native AArch64 NEON execution, ARM feature variants, and SVE-128 compatibility
  checks. Wider/scalable SVE remains unsupported under the current contract.
- GCC and Clang on x86/Linux, AppleClang on AArch64/macOS, MSVC on x86/Windows;
  C++20/23 where already supported. CUDA and HIP compile/run configurations.
- Default math provider and enabled SVML/SLEEF configurations where configured
  by the project, plus installed-package consumers. Classify numerical checks
  against the provider actually used.
- `SCALARS_CPU_BACKEND=no` must remain a clean build without exposing a false
  CPU SIMD specialization. Test host/device header isolation and standalone
  inclusion of public entry points.
- Run ASan/UBSan for memory and aliasing mistakes. Distinguish compile coverage
  from hardware execution in CI reports; compiling AVX-512 is not permission to
  execute it on an unsupported runner.

### Performance

Add SIMD benchmarks alongside the existing scalar benchmarks. Measure conversion,
load/store, arithmetic, FMA, masks/select, gather/scatter, transpose, reductions,
complex multiplication/division, math functions, and a widened composite kernel.
Include short arrays/tails and cache-resident/streaming sizes. Report elements/s,
bytes/s, dtype, ISA, compiler, math provider, and numerical policy. Compare with
scalar reference loops and the same ISA's float/double implementation.

Inspect assembly to confirm core paths use vector conversion/arithmetic and to
identify spills or scalar loops. Compare dense reduced packets with a one-float-
packet implementation before asserting a speed benefit. Establish regression
thresholds from measured baselines; do not promise a universal 2x or 4x gain.

## 8. Recommended delivery order

Contract and scalar repair -> conversions -> half/bf16 on AVX2 and NEON ->
remaining CPU reduced backends -> complex float/double -> reduced complex ->
CUDA/HIP parity -> optional native acceleration and final documentation.

Each implementation change includes its relevant tests and documentation.
CPU dtype parity is complete after Change 5 and its CPU gates. Cross-backend
parity includes Change 6. Native acceleration can then be assessed independently
without weakening the numerical baseline.
