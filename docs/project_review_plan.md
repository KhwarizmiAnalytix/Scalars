# Project review remediation plan

Status: Phases 0 to 7 are implemented (Phases 0 to 6 and 7.1 to 7.5 in
`fd796ab`). Verification on Windows x86 with an NVIDIA GPU on 2026-11-13
closed the remaining open items (7.6 to 7.10, 6.1, 6.2) and found three more
defects that the arm64-only work could not see; they are in
[Phase 8](#10-phase-8-windows-x86-and-cuda-verification). The only open item
is the long-term half of 2.3, a design decision.
Repository inspected: `main` at `647564c`, 2026-11-12; Phase 8 against
`fd796ab`.

## 1. Goal and scope

Fix the defects found in the in-depth project review, in priority order. Each
work item lists the files to change, the intended fix, and the test or check
that proves it. Items are grouped into phases so that each phase can land as
one or a few focused pull requests.

Evidence labels:

- **Confirmed**: reproduced with a probe program or compiler error on arm64
  (NEON + Accelerate, Homebrew Clang, Release).
- **Unverified**: found by reading code; not reproduced end to end. The first
  step for these items is a reproducer.

Baseline checks at the time of review:

| Check | Result |
| --- | --- |
| Release build (Ninja, Clang, NEON + Accelerate) | Clean, no warnings |
| `ScalarsCxxTests` (74 cases, 17 suites) and `ScalarsSetupTests` | Pass |
| ASan + UBSan Debug build and tests | Clean |
| clang-tidy on `src/` and `TestSimdComplex.cxx` | No findings |
| x86 (SSE/AVX/AVX2/AVX-512), SVE, CUDA, HIP, Metal, cppcheck | Not exercised locally |

Later in the session, real x86_64 execution became available via
`docker run --platform linux/amd64 ... ubuntu:24.04` (QEMU emulation on this
arm64 host, not cross-compilation only -- see the Phase 6 install-consumer
entry and the note below for what that covered).

## 2. Phase 0: licensing and provenance

These items carry legal risk, so they come first.

### 0.1 Correct the README provenance claim

- [x] [README.md:3](../README.md#L3) says no PyTorch source files are copied.
      [half.h](../include/types/half.h) and
      [bfloat16.h](../include/types/bfloat16.h) state they are copied from
      `c10/util/*-inl.h` and still contain PyTorch TODOs and "ATen" comments.
- [x] Rewrite the claim to say which files derive from PyTorch c10.
- [x] Add a `ThirdParty`-style notice (or `NOTICE` file) carrying the PyTorch
      BSD-3 copyright and license text.
- [x] Attribute the FP16 conversion code to Marat Dukhan's FP16 library (MIT)
      and include its notice.
- [x] Remove leftover PyTorch TODO owner tags and "ATen" comments.

### 0.2 Committed Intel SVML binaries

- [x] `ThirdParty/svml/` contains prebuilt `libsvml`, `libintlc`, `libirc`, and
      `svml_dispmd` binaries with no license file.
- [x] Add the Intel redistribution license that covers these files
      (`ThirdParty/svml/LICENSE`, Intel Simplified Software License, Version
      October 2022). The file flags that this is the current public version
      and should be checked against the specific installer the binaries were
      extracted from.

Acceptance: every derived or vendored file has a traceable license notice; the
README statement matches the source tree.

## 3. Phase 1: confirmed correctness bugs

Each fix lands with a regression test in `Testing/Cxx/`. Use the
[new-test](../.agents/skills/new-test/SKILL.md) skill and match adjacent
GoogleTest suites.

### 1.1 `half` mixed comparisons do not compile (confirmed) -- Fixed (CUDA verified; see 7.6)

- File: [half.h:739-762](../include/types/half.h#L739-L762).
- Problem: `(half, half)` comparison operators make `h == 1.0f` and `h < 1.0`
  ambiguous. c10 accepts them through implicit conversion to `float`.
- Fix: remove the `(half, half)` overloads and rely on conversion, as c10
  does, or add explicit `(half, float)`, `(float, half)`, `(half, double)`,
  and `(double, half)` overloads.
- Test: compile-and-run checks for `==`, `!=`, `<`, `<=`, `>`, `>=` against
  `half`, `float`, `double`, and `int`.

### 1.2 `numeric_limits<bfloat16>::signaling_NaN()` returns +inf (confirmed) -- Fixed

- File: [bfloat16.h:516-519](../include/types/bfloat16.h#L516-L519).
- Problem: returns bits `0x7F80`, which is +infinity; `isnan()` is false.
  Inherited from c10.
- Fix: return `0x7FA0` (exponent all ones, quiet bit clear, nonzero payload).
- Test: `isnan(signaling_NaN())`, quiet bit clear, distinct from `infinity()`.
  Add the same checks for `half`.

### 1.3 `bfloat16` bitwise operators mutate the left operand (confirmed) -- Fixed

- File: [bfloat16.h:248-264](../include/types/bfloat16.h#L248-L264).
- Problem: `operator|`, `operator^`, `operator&` take `bfloat16&` and write the
  result into it, so `c = a | b` changes `a`.
- Fix: take both operands by value or `const&` and return a new value.
- Test: operands are unchanged after each operator; results match bitwise
  operations on `uint16_t`.
- Related cleanup: `bfloat16` has no `==`, `<=`, `>=`, while `half` does. Make
  the two types consistent after 1.1 is decided.

### 1.4 Reduced-precision complex division (confirmed) -- Fixed

- File: [complex.h:291-368](../include/types/complex.h#L291-L368) and the
  `complex<half>` specialization at
  [complex.h:626-676](../include/types/complex.h#L626-L676).
- Problems:
  - `complex<bfloat16>` division fails to compile: `c < 0` is ambiguous
    between `bfloat16` and `int`. The public `complex_bf16` alias and
    `scalar_type::complex_bf16` still advertise the type.
  - `complex<half>` has no `/` or `/=`, no `real(T)`/`imag(T)` setters, and is
    not `constexpr` like the generic template.
  - `abs_c = c < 0 ? -c : c` keeps `-0.0`, so `(1+1i)/(-0-0i)` returns
    `(-inf, -inf)`.
  - The fast division path is disabled for CUDA host compiles, so host results
    differ between a CUDA and a non-CUDA build.
- Fix: compute reduced-precision complex arithmetic in `complex<float>` and
  narrow the result. This is how PyTorch handles `c10::complex<c10::Half>`.
  Use `T(0)` in comparisons and `std::abs`/`fabs` for magnitudes. Give
  `complex<half>` the same member set as the generic template.
- Test: `+ - * /` and compound forms for `complex<half>` and
  `complex<bfloat16>`; division by signed zeros; results match `complex<float>`
  narrowed.
- Follow-up: the signed-zero fix mapped a NaN divisor to 0, and the new
  `complex<bfloat16>` specialization dropped part of the generic interface.
  Both are fixed in 7.1 and 7.2.

### 1.5 Scalar `expm1(inf + 0i)` returns `(inf, NaN)` (confirmed) -- Fixed

- File: [complex_math.h:351-364](../include/math/complex_math.h#L351-L364).
- Problem: `exp(x) * sin(y)` evaluates `inf * 0`. The SIMD layer guards this
  case, so the scalar and SIMD results differ.
- Fix: return `y` unchanged for the imaginary part when `y == 0`, matching
  C99 Annex G and the SIMD path.
- Test: Annex G special values for `expm1`: `±0`, `±inf`, and NaN in each
  component. Compare scalar and SIMD.

### 1.6 Mixed-precision complex `pow` computes in the narrower type (confirmed) -- Fixed

- File: [complex_math.h:138-170](../include/math/complex_math.h#L138-L170).
- Problem: `pow(complex<float>, complex<double>)` returns `complex<double>` but
  computes in `float`. It gives 1.04880881 where the true value is 1.04880886.
  Inherited from c10.
- Fix: cast both operands to the promoted type before calling `pow`.
- Test: mixed float/double `pow` matches the all-double result to double
  tolerance.

### 1.7 NEON `hypot` overflows and underflows (confirmed) -- Fixed

- Files: [neon/float/simd.h:78](../include/backend/cpu/neon/float/simd.h#L78)
  and the matching NEON double header.
- Problem: plain `sqrt(x*x + y*y)`. `hypot(1e20, 1e20) = inf` and
  `hypot(3e-25, 4e-25) = 0`. The x86 backends use SVML, so results depend on
  the platform.
- Fix: scale by `max(|x|, |y|)`, or by a power of two from the exponent so the
  scaling stays exact. Handle inf and NaN as `std::hypot` does: `hypot(inf,
  NaN) = inf`.
- Test: a range sweep over huge, tiny, subnormal, inf, and NaN inputs compared
  with `std::hypot` in the SIMD parity suite.

### 1.8 `normalcdf` lower tail has 1.3e-7 relative error (confirmed) -- Fixed

- File: [normal_cdf.cpp:51](../src/math/normal_cdf.cpp#L51).
- Problem: the asymptotic series for `z < -7.071` is truncated too early.
  Elsewhere the error is about 1e-15.
- Fix: compute the tail as `0.5 * erfc(-z / sqrt(2))`, or add enough series
  terms to reach double precision. Make the coefficient arrays
  `static constexpr`.
- Test: relative error below 1e-14 against `0.5 * erfc(-z / √2)` for
  `z` in `[-38, 8]`.

### 1.9 `inv_normalcdf` returns NaN at the endpoints (confirmed) -- Fixed

- File: [normal_cdf.cpp:119](../src/math/normal_cdf.cpp#L119).
- Problem: `p = 0` and `p = 1` return NaN instead of `-inf` and `+inf`.
- Fix: return `∓infinity` for exact 0 and 1, and NaN outside `[0, 1]`.
- Test: endpoint values, out-of-domain inputs, and the round trip
  `normalcdf(inv_normalcdf(p)) ≈ p`.

## 4. Phase 2: packaging and install

### 2.1 Installed SVML consumers may fail to link (unverified) -- Fixed (SVML paths unverified end-to-end: no x86 hardware here)

- Files: [utils.cmake:223](../cmake/utils.cmake#L223) and
  [simd.cmake:266](../cmake/simd.cmake#L266).
- Problem: on x86 with Clang or GCC, SVML is turned on automatically, so the
  inline headers emit `__svml_*` calls. SVML is linked only through
  `$<BUILD_INTERFACE:...>`. `SCALARS_HAS_SVML=1` also stays defined when SVML
  is not found.
- Steps:
  - [x] Reproduce: confirmed by inspection that `SCALARS_HAS_SVML` was set
        from `SCALARS_ENABLE_SVML` alone with no `TARGET SVML::SVML` check
        (unlike the already-correct `SCALARS_HAS_MKL`), so it is wrongly 1
        whenever SVML is requested but not found, including on non-x86
        targets where it is never probed. Could not build a real x86 SVML
        consumer here (arm64 host; the `ThirdParty/svml` binaries are
        Linux/Windows x86 only).
  - [x] Installed and exported the SVML import target: `SVML::SVML`/`IRC`/
        `INTLC` are now linked `PUBLIC` (unrestricted, matching MKL/Sleef)
        instead of `$<BUILD_INTERFACE:...>`; `FindSVML.cmake`/`FindMKL.cmake`
        are installed to `<prefix>/lib/cmake/Scalars/Modules`, and
        `scalars_config.cmake.in` prepends that to `CMAKE_MODULE_PATH` and
        calls `find_dependency(SVML)` before including `ScalarsTargets.cmake`.
        Also fixed `FindSVML.cmake` to search the installed-prefix layout
        (previously only the in-tree `ThirdParty/svml` layout), and install
        the Windows `.lib` import library (previously only the `.dll`).
  - [x] `SCALARS_HAS_SVML` now also requires `TARGET SVML::SVML`.
  - [x] Extend the CI install-consumer job to call `exp`, `log`, and `sin`.
        Added `check_simd_math<T>()` to
        [InstallConsumer/main.cpp](../Testing/InstallConsumer/main.cpp): loads
        a packet, calls `simd<T>::exp/log/sin`, and checks the result against
        `std::exp/log/sin` to a loose tolerance, for both `float` and
        `double`. Added a new `install-consumer-svml` job to
        [ci.yml](../.github/workflows/ci.yml) (Linux and Windows only --
        `ThirdParty/svml` ships no macOS binaries) that installs with
        `--svml=ON --cpu-backend=avx2` explicitly, matching the existing
        `svml` build-only job, so the consumer actually links the exported
        `SVML::SVML`/`IRC`/`INTLC` targets from 2.1 rather than relying on
        `cmake/utils.cmake`'s own auto-enable (which already turns SVML on
        by default for GCC/Clang on x86 whenever the compiler lacks builtin
        SVML-equivalent intrinsics, but not for MSVC, which has its own
        builtin equivalents and would otherwise never exercise real SVML
        linking from a consumer).
        While verifying this end-to-end (install + consume locally, NEON +
        Accelerate), found and fixed an unrelated, pre-existing break: 3.1's
        namespace migration moved `simd<T>` into `scalars::simd<T>` but never
        updated `InstallConsumer/main.cpp`'s own `check_reduced_simd<T>()`
        helper, which used unqualified `simd<T>` outside the `using namespace
        scalars;` in `main()`'s scope -- the installed-consumer build was
        broken on this branch before this session touched it. Fixed both
        helpers to `scalars::simd<T>`. Verified twice: a full local
        `cmake --install` + separate consumer `find_package(Scalars CONFIG
        REQUIRED)` + build + run passed end-to-end on NEON + Accelerate
        (this machine's native backend); then the same cycle was re-run
        inside `docker run --platform linux/amd64 ubuntu:24.04` (QEMU
        emulation, real x86_64 instruction execution, not just
        cross-compilation) with the exact `install-consumer-svml` job
        configuration (`--svml=ON --cpu-backend=avx2`, gcc, shared build) --
        `find_dependency(SVML)` found the installed `libsvml.so`, the
        consumer linked against it, and `check_simd_math()`'s
        `simd<float>::exp/log/sin` calls (backed by the real Intel SVML
        `.so` binaries) produced correct results. (The `--no-as-needed`
        and `--as-needed` linker flags around these targets were later put
        back in `BUILD_INTERFACE`; see 7.4.) This is the exact
        link failure 2.1 was worried about, now confirmed fixed end-to-end
        on real (emulated) x86_64, not reasoned from source. Also used the
        same container to rebuild and re-run the full `ScalarsCxxTests`
        suite on plain Linux x86_64 with the default backend (gcc and
        clang, both auto-select avx2 + auto-enable SVML per
        `cmake/utils.cmake`) and explicitly with `--cpu-backend=sse/avx/
        avx2`: all passed (86/86 each). `--cpu-backend=avx512` compiled
        cleanly but the test binary hit `Illegal instruction` at runtime --
        QEMU's TCG does not implement AVX-512 opcodes, which is exactly why
        `ci.yml`'s own `simd` job already compiles every backend
        unconditionally but only runs tests when `/proc/cpuinfo` on the
        real runner advertises the matching flag; not a Scalars defect.

### 2.2 The package config does not re-find SLEEF and MKL -- Fixed (verified for the no-extra-deps default; SLEEF/MKL paths reasoned from source, not built end-to-end here)

- Files: [simd.cmake:325](../cmake/simd.cmake#L325) (MKL),
  [simd.cmake:334](../cmake/simd.cmake#L334) (SLEEF), and
  [scalars_config.cmake.in](../cmake/scalars_config.cmake.in).
- Problem: `Sleef::sleef` and `MKL::MKL` are linked `PUBLIC`, but the config
  file only calls `find_dependency` for CUDA and HIP. SLEEF is enabled
  automatically on non-Apple AArch64, so installed packages there are
  probably broken.
- Fix: added conditional `find_dependency(SVML)`/`find_dependency(MKL)`/
  `find_dependency(sleef)` to `scalars_config.cmake.in`, each gated on the
  matching `SCALARS_HAS_*` flag substituted at configure time.
  `Accelerate` needs no `find_dependency`: it is linked as a plain
  `"-framework Accelerate"` string, not an imported target, so it already
  propagates through the exported `INTERFACE_LINK_LIBRARIES` as-is.
  For SLEEF specifically: `ThirdParty/sleef`'s own `CMakeLists.txt` installs
  and exports its target as `sleef::sleef` (confirmed by reading it), not
  `Sleef::sleef` (Scalars' local alias) -- `scalars_config.cmake.in` now also
  adds `add_library(sleef ALIAS sleef::sleef)` after `find_dependency(sleef)`,
  since CMake records Scalars' own link requirement under the alias's
  underlying bare name, `sleef`.
- Test: a full local `cmake --install` + a separate consumer project's
  `find_package(Scalars CONFIG REQUIRED)` + build + run, for the default
  config (NEON + Accelerate, no SVML/MKL/SLEEF), passed end-to-end. Forcing
  `SCALARS_ENABLE_SLEEF=ON` to test that path hit a separate, pre-existing
  build failure unrelated to this fix: `ThirdParty/sleef`'s vendored
  `tlfloat` dependency (built via `ExternalProject_Add`, installed to
  `CMAKE_INSTALL_PREFIX`) fails to link into `libScalars` with `library
  'tlfloat' not found`, because nothing adds its install directory to the
  linker search path. That is an upstream SLEEF/tlfloat packaging gap in a
  vendored submodule, not a Scalars `cmake/*.cmake` bug, so it is out of
  scope here, but it blocks fully verifying the SLEEF path end-to-end on
  this machine. An install-consumer CI job on Linux AArch64 with SLEEF (as
  the plan's own Test line asks for) is still the way to close that out.

### 2.3 SIMD compile flags leak to consumers -- Short term done; long term is a design decision, left to the user

- File: [simd.cmake](../cmake/simd.cmake).
- Problem: `-mavx2 -mf16c -mfma` and `-march=armv8.2-a` are `PUBLIC`. The
  default x86 backend is `avx2` with no runtime dispatch, so every consumer is
  compiled for AVX2 and the installed binary will not run on older CPUs.
- Design references:
  - PyTorch compiles kernels several times under `CPU_CAPABILITY` namespaces
    and picks one at runtime.
  - Eigen leaves the ISA to the consumer's translation-unit flags.
- Short term:
  - [x] Document the ISA requirement in the README.
  - [x] Export the selected backend as a property or a generated
        `scalars_config.h` macro so a mismatch fails at compile time. Added
        to `scalars_macros.h` (always included) rather than a new generated
        header: `#error`s if `SCALARS_HAS_{AVX2,AVX512,AVX,SSE,NEON,SVE}`
        says an ISA was selected but this translation unit's own predefined
        macros (`__AVX2__`, ...) say it isn't actually enabled. Verified by
        preprocessing a stub TU cross-targeted at x86_64 with
        `SCALARS_HAS_AVX2=1` and no `-mavx2`: errors with a clear message;
        adding `-mavx2` passes. (NEON's guard cannot be exercised on real
        AArch64 hardware: `__ARM_NEON` is unconditionally predefined there,
        since NEON is mandatory on AArch64, not optional like the x86 ISAs.)
        GPU device passes are now excluded from the guard; see 7.3.
- Long term: decide between Eigen-style consumer-selected ISA (header-only
  packets and `INTERFACE` flags only on request) and PyTorch-style runtime
  dispatch for the compiled sources. Not attempted here -- this is the
  architecture decision the plan itself defers to a separate design pass.

### 2.4 The standard-library ABI choice is fixed at build time -- Fixed

- File: [complex_math.h](../include/math/complex_math.h).
- Problem: `_detail::sqrt` and `_detail::acos` exist only for libc++, or for
  libstdc++ without C99 complex. A consumer built against a different
  standard library gets link errors.
- Fix: made these helpers `inline` templates directly in `complex_math.h`
  (previously declared `SCALARS_API` and defined in `src/math/complex_math.cpp`,
  now removed -- they used only portable `<cmath>` calls on `scalars::complex<T>`,
  no actual standard-library-specific dependency of their own, so there was no
  reason for them to be part of libScalars's ABI surface at all).

### 2.5 Smaller build issues -- Fixed

- [x] Validate `SCALARS_SANITIZER_TYPE`. A CMake list such as
      `address;undefined` currently becomes `-fsanitize=address undefined`.
      Accept both `,` and `;`, and reject unknown sanitizers. Fixed by joining
      the already-validated, semicolon-normalized list back with commas
      before building the `-fsanitize=` flag, instead of using the raw,
      unnormalized input string. Verified: configuring with
      `-DSCALARS_SANITIZER_TYPE="address;undefined"` now produces a single
      `-fsanitize=address,undefined` on both the compile and link lines
      (previously a stray bare `undefined` token); an unknown sanitizer name
      is still rejected with `FATAL_ERROR`.
- [x] Tests compile with `gnu++20` while the library uses
      `CXX_EXTENSIONS OFF`. Set `CXX_EXTENSIONS OFF` on test targets too.
      `ScalarsCxxTests` now also sets `CXX_STANDARD`/`CXX_STANDARD_REQUIRED`/
      `CXX_EXTENSIONS OFF`; full test suite still passes under `-std=c++20`.
- [x] Fix the stale path `include/detail/vec_half.h` in
      [simd.cmake](../cmake/simd.cmake). The file is at
      `include/common/vec_half.h`. (This was a stale comment, not a used
      path/glob pattern.)

## 5. Phase 3: API and design cleanup

These changes can break consumers, so batch them and note them in release
notes.

### 3.1 Remove global namespace pollution -- Fixed (NEON + reduced + detail built and tested; AVX/AVX2/AVX512/SSE syntax-verified by cross-compilation; CUDA/HIP edited by inspection only, unverified; SVE/Metal unaffected)

- Files: [simd.h:16](../include/backend/simd.h#L16), every backend header that
  specializes `::simd<T>`, and
  [complex_math.h:370](../include/math/complex_math.h#L370).
- Problems:
  - `template <class T> struct simd` and `prefetch_ptr_type` are at global
    scope.
  - `using scalars::complex_math::*` declarations sit at global scope and
    inside `std`.
- Fix:
  - [x] Moved to `scalars::simd<T>` / `scalars::prefetch_ptr_type`. No
        deprecated global alias was added -- every internal reference was
        updated instead (see below), so there is nothing left relying on the
        old global name to preserve compatibility for.
    - `include/backend/simd.h`: primary template and `prefetch_ptr_type`
      moved into `namespace scalars`.
    - 18 backend specialization files updated to specialize
      `scalars::simd<T>` instead of `::simd<T>`: NEON
      (float/double/complex), AVX (float/double/complex, shared by the
      `avx`/`avx2` backends), AVX-512 (float/double/complex), SSE
      (float/double/complex), the CPU-generic `reduced/simd.h` and
      `reduced/complex_simd.h` (half/bfloat16 and their complex forms), and
      the GPU `float`/`double`/`complex`/`reduced` scalar backends (CUDA/HIP).
      `backend/cpu/sve/{float,double,complex}/simd.h` needed no change: each
      is a pure `#include` of the matching NEON header, reusing its packets
      exactly as before. Metal does not use the `simd<T>` template at all.
    - 6 library headers with explicit global-scope `::simd<T>` qualification
      fixed to resolve through `scalars::simd` instead:
      `backend/cpu/detail/complex_simd.h`, `backend/cpu/neon/complex/simd.h`,
      `backend/cpu/reduced/convert.h`, `backend/cpu/reduced/complex_simd.h`,
      `math/complex_vector.h`.
    - 11 `Testing/Cxx/*` files and 3 `Testing/Benchmark/*` files had no
      `using namespace scalars;` and relied on `simd<T>` resolving at global
      scope; each now has `using scalars::simd;`.
    - Verification: `ScalarsCxxTests` (NEON + Accelerate) builds and passes
      in both Release and the ASan+UBSan Debug configuration;
      `SCALARS_ENABLE_BENCHMARK=ON` also builds cleanly (exercises
      `BenchmarkComplex.cxx`, `BenchmarkReduced.h` via the half/bfloat16
      benchmarks). AVX, AVX2 (shares AVX's files), AVX-512, and SSE were
      syntax-checked with `clang++ -fsyntax-only -target x86_64-apple-darwin`
      plus each backend's real ISA flags (`-mavx2 -mfma -mf16c`,
      `-mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512cd -mfma -mf16c`,
      `-msse -msse2`) -- this machine has no x86 hardware to actually run
      them, but clang's front end can still parse real AVX/SSE intrinsics
      header-only when cross-targeting x86_64, which caught several
      namespace-boundary mistakes before they were fixed. The CUDA/HIP GPU
      backend files could not be checked at all (no nvcc/hipcc here); they
      were edited by the same mechanical pattern, by inspection.
  - [x] Dropped the global-scope `using scalars::complex_math::*;` block.
        Checked every caller first: within `include/`/`src/`, nothing relied
        on it (the one apparent hit, `backend/gpu/metal/kernels.metal`, is
        Metal Shading Language, an unrelated language with its own built-in
        `sqrt`/`exp`/etc., not C++). Exactly one test file,
        `Testing/Cxx/TestComplexMath.cxx`, called `sqrt`/`exp`/`pow`/etc.
        unqualified relying on it; it now has
        `using namespace scalars::complex_math;` of its own instead of
        depending on the library leaking those names globally.
  - [x] Kept the `std` injection (`namespace std { using
        scalars::complex_math::sqrt; ... }`): it already carried the
        deliberate, documented justification the plan asks for (a
        `NOLINTBEGIN/END(bugprone-std-namespace-modification)` comment
        explaining it lets generic code call `std::sqrt`-style on
        `scalars::complex<T>` uniformly), and removing it would break any
        such generic caller outside this repository with no equivalent
        replacement available.

### 3.2 Unify the accumulation-type systems -- Fixed; release note added (see 7.7)

- Files: [scalar_traits.h:17](../include/common/scalar_traits.h#L17) and
  [accumulate_type.h](../include/common/accumulate_type.h).
- Problem: `accumulation_type<float>` is `float`, but `acc_type<float, cpu>` is
  `double`.
- Fix: kept `acc_type` (which matches PyTorch's `at::acc_type`) and turned
  `accumulation_type<T>` into `acc_type<T, /*is_cuda*/false>` (an alias, not
  a removal, since test code still uses the single-argument
  `accumulation_type_t<T>` form). Deliberately no fallback to `T` for a type
  with no `acc_type` entry -- unlike the old primary template, an
  unspecialized type is now a compile error, matching `at::acc_type`.
  `expect_accumulate<T, Cpu, Gpu>()` in `TestScalarTraits.cxx` (already an
  exhaustive table test over every `acc_type`-covered type) now also asserts
  `accumulation_type_t<T> == Cpu` for every row -- the actual "do these two
  systems now agree" check. Fixed the two call sites that asserted the old,
  wrong value (`accumulation_type_t<float>` was asserted to be `float`; it
  is `double`).
- Also:
  - [x] Added the missing return after the `switch` in `to_accumulate_type`
        ([accumulate_type.h:121](../include/common/accumulate_type.h#L121)):
        `return type;` (identity) after the switch, which is exhaustive over
        today's `scalar_type` enumerators but not provably so to the
        compiler since the enum is backed by `uint8_t`.
  - [x] Stopped `promote_real` mapping `long double` to `double` (and, found
        while fixing it, to `float` too: `promote_real<long double, float>`
        also collapsed to `float`)
        ([type_promotion.h:6-14](../include/common/type_promotion.h#L6-L14)).
        Added a `long double`-dominates tier above the existing
        double/float tiers, matching the usual arithmetic conversions.
        `promote_real`/`promote_t` have no other callers in this repo
        (checked), so this only affects their own test, extended with
        `long double` cases.

### 3.3 Make `complex_simd::set` unambiguous -- Fixed (see 7.8)

- File: [complex_simd.h:330](../include/backend/cpu/detail/complex_simd.h#L330).
- Problem: `set(1)` returns a packet but `set(1u)` returns a mask. For
  `double`, any `uint64_t` or `size_t` argument also returns a mask.
- Fix: renamed the mask overload to `set_mask` everywhere this pattern
  existed (confirmed by grep, not just the cited file), since
  `complex_simd<T, Ops>` already had a `set_mask(bool)` sibling to match:
  - `backend/cpu/detail/complex_simd.h` (the cited file).
  - `backend/cpu/reduced/simd.h` and `backend/cpu/reduced/complex_simd.h`
    (half/bfloat16 and their complex forms) -- same trap, independently
    reimplemented.
  - The underlying real-packet backends themselves, not just their complex
    wrappers: `backend/cpu/neon/{float,double}/simd.h` and
    `backend/cpu/avx512/{float,double}/simd.h`. AVX/AVX2 and SSE have no
    second `set` overload at all, so they were never affected.
  - Updated the handful of call sites in `Testing/Cxx/` that relied on the
    old ambiguity resolving to the mask overload
    (`TestSimdComplexParity.cxx`, `TestSimdReduced.cxx`) to call `set_mask`
    explicitly.
- Verification: `ScalarsCxxTests` (NEON) passes in Release and ASan+UBSan
  Debug; AVX-512 re-verified clean with the same `-fsyntax-only
  -target x86_64-apple-darwin -mavx512f ...` cross-compilation check used
  for 3.1.

### 3.4 Naming and export consistency -- Already fixed

- [x] Rename `KH_REAL_MATH` in `scalar_math.h` to a `SCALARS_*` macro.
      Checked: already `SCALARS_REAL_MATH` throughout
      [scalar_math.h](../include/math/scalar_math.h); no `KH_REAL_MATH`
      left anywhere in the tree.
- [x] Mark `scalar_type_name` and `scalar_type_size` `SCALARS_API` if they
      are defined out of line. Checked: both already declared
      `SCALARS_API` in
      [scalar_type.h:19-20](../include/common/scalar_type.h#L19-L20) and
      defined out of line in `src/core/scalar_type.cpp`.
- [x] Choose one dtype naming scheme (`complex_half` versus `complex64`) and
      keep the other as aliases. Checked: the public `scalar_type` enum and
      its `complex_half`/`complex_bf16` aliases
      ([scalar_type.h](../include/common/scalar_type.h),
      [complex.h:848-849](../include/types/complex.h#L848-L849)) already
      consistently use the `complex_<basetype>` scheme throughout the
      public API, with no `complex64`/`complex128` competing there. The
      only `complex64` in the tree is
      `scalars::metal_backend::dtype::complex64`
      ([metal_dispatch.h:27-33](../include/backend/gpu/metal/metal_dispatch.h#L27-L33)),
      a private, size-based enum scoped to the Metal GPU-dispatch boundary
      (mirrors NumPy/PyTorch's size-based dtype convention there, same as
      its sibling `float32`); it never appears in the public dtype API, so
      there is no actual collision to alias away.

## 6. Phase 4: numerical consistency and performance

### 4.1 Complex `log` fast path near |z| = 1 -- Fixed (see 7.9 and 8.4)

- File: [complex_simd.h:165](../include/backend/cpu/detail/complex_simd.h#L165).
- Problem: `0.5 * log(re² + im²)` loses all relative precision near `|z| = 1`.
  For example, it returns 0 where the true value is about 5e-9. libc++
  behaves the same, but glibc does not. The path is active by default on x86.
- Fix: in `log_fast`, use `0.5 * log1p(hi² - 1 + lo²)` -- with
  `hi = max(|re|, |im|)`, `lo = min(|re|, |im|)`, and `hi² - 1` computed as
  `(hi+1)(hi-1)` (one `fma`) rather than `(re²+im²) - 1` -- **only** in the
  band where `hi` is within `[0.5, 2]`. There, `hi - 1` is exact by
  Sterbenz's lemma, so the signal is limited only by `lo²`'s own rounding,
  not by cancellation in `re²+im²` itself. Outside that band this
  refinement does not apply -- Sterbenz's lemma needs `hi` within a factor
  of 2 of *1 specifically*, not of whatever `re²+im²` happens to be, so for
  tiny or huge `hi` the `hi - 1` subtraction is itself a lossy
  cancellation (e.g. `hi = 1e-15` gives `hi - 1 == -1` exactly, losing
  `hi` entirely, where the true `log(re²+im²)` is a large finite negative,
  not `log1p(-1) = -inf`) -- so the two formulas are selected with a
  per-lane mask (`r::if_else`), the same pattern this file already uses
  for `isnan`/`isinf` substitution in `abs()`. First attempt at this fix
  applied the `log1p` formula unconditionally (reasoning it was
  algebraically identical to the old formula everywhere); that broke
  `SimdComplexParity.MathExponentRanges` with `-inf` results once log's
  fast path was also enabled for Accelerate (below) and so actually
  exercised locally -- caught by the existing test suite, not inspection.
- Note: `log_fast` only ran when `(SCALARS_HAS_SVML || SCALARS_HAS_SLEEF)
  && !SCALARS_HAS_ACCELERATE` (see `log()` just above it); the default
  local NEON+Accelerate config always took the scalar `scalars::log(x)`
  fallback instead, which is why this bug was never caught by the existing
  test suite before this change. Fixed as part of 4.2 below (the
  exp/log Accelerate asymmetry) by enabling the fast path for Accelerate
  too, which is also what caught the masking mistake above.
- Verification: `ScalarsCxxTests` now actually exercises `log_fast` on
  this machine (NEON+Accelerate, see 4.2) -- 86/86 pass, including
  `MathExponentRanges`'s random sweep across the full float/double
  exponent range (denormal to near-max) and its explicit near-unit-magnitude
  case; also clean under the ASan+UBSan Debug build. Before wiring
  Accelerate through `log()`, separately built a throwaway NEON+SLEEF
  (`SCALARS_ENABLE_ACCELERATE=OFF`) configuration to compile and run
  `log_fast` directly (library *linking* hits the same pre-existing
  vendored `tlfloat` gap noted under 2.2, but a standalone driver linked
  against the built `libsleef.a`/`libtlfloat.a` worked): for
  `z = (0.9323273301, 0.3616154194)` (`|z|² = 1` off by ~1e-8), the old
  formula gives `-2.980232328e-08` against a true value of
  `-1.899317544e-08` (57% relative error); the fixed formula gives
  `-1.95955181e-08` (3% relative error). Also cross-compiled
  `-fsyntax-only -target x86_64-apple-darwin` for AVX2, AVX512, and
  SSE-only (the non-SVML/SLEEF case, where `log_fast` still has to compile
  even though it is unreachable at runtime there) -- all pass.
- Fix: when `|z|` is near 1, use `0.5 * log1p((re - 1)(re + 1) + im²)`, with
  the larger component first, or reuse the scalar fallback lanes.

### 4.2 Backend asymmetries

- [x] The `exp` fast path is enabled under Accelerate, but `log` is disabled.
      Either justify this in a comment or make them match. Made them
      match: `log()`'s fast-path condition now reads
      `SCALARS_HAS_SVML || SCALARS_HAS_MKL || SCALARS_HAS_SLEEF ||
      SCALARS_HAS_ACCELERATE`, the same as `exp()`'s, instead of
      `(SCALARS_HAS_SVML || SCALARS_HAS_SLEEF) && !SCALARS_HAS_ACCELERATE`.
      The old, narrower condition was only ever load-bearing while
      `log_fast()`'s formula lost precision near `|z|=1` (4.1); now that
      4.1 is fixed, there is no remaining reason for `log()` to exclude
      Accelerate. This also means `log_fast()` is now actually exercised
      on this machine's default local build (NEON+Accelerate) rather than
      only the scalar fallback -- which is what surfaced the masking
      mistake described under 4.1, so this item and 4.1 were fixed and
      verified together.
- [x] NEON complex `reciprocal` uses `vrecpe` plus Newton steps
      ([neon/complex/simd.h:120-137](../include/backend/cpu/neon/complex/simd.h#L120-L137)),
      while x86 uses exact division. Benchmarked `vdivq` directly on this
      machine (Apple M-series, the actual NEON hardware available here) with
      a standalone microbenchmark (2^20-element sweep, 200 iterations):
      `vdivq_f32` is both exact (max relative error 5.96e-8, i.e. correctly
      rounded) **and faster** than the 2-step Newton approximation (0.095ms
      vs 0.233ms/iter; the approximation's own error was ~1.18e-7, about 1
      ULP, matching its "no correct-rounding promise" comment).
      `vdivq_f64` likewise: exact (0 measured relative error) and faster
      (0.195ms vs 0.533ms/iter for 3 Newton steps). No tradeoff to document
      on this target, so switched `reciprocal()` to plain
      `vdivq_f32`/`vdivq_f64`, matching the x86 backends' own
      `_mm*_div_p{s,d}(1, a)`. Verified: `ScalarsCxxTests` passes (86/86)
      in both Release and ASan+UBSan Debug.
- [x] `neg`, `ceil`, `floor`, and `trunc` in `complex_simd.h` do needless
      interleave shuffles. Fixed: all four act independently per lane
      regardless of AoS/SoA layout (same as `add`/`sub`/`conj`, which
      already applied `r::add`/`r::sub`/`r::neg` straight to `a.real`/
      `a.imag`), so the `Ops::interleave(a)` / `Ops::deinterleave(...)`
      round trip they went through bought nothing. `ceil`/`floor`/`trunc`
      carried a comment claiming "surrounding memory shuffles can cancel"
      for standalone rounding; that doesn't hold for a standalone call (no
      surrounding shuffle exists to cancel with), which is exactly the
      case the plan flagged. Simplified all four to apply `r::neg`/
      `r::ceil`/`r::floor`/`r::trunc` directly to `a.real`/`a.imag`.
      Verified: `ScalarsCxxTests` passes (86/86) in both Release and
      ASan+UBSan Debug; cross-compiled `-fsyntax-only
      -target x86_64-apple-darwin` for AVX2, AVX512, and SSE -- all pass.
- [x] `ploadquad` for 4-wide float differs from Eigen's semantics. Rename it
      or document the difference. Chose "document": for every other packet
      width/type (AVX512 double 8-wide, AVX double 4-wide, SSE double
      2-wide, `reduced/simd.h`), this repo's `ploadquad` already matches
      Eigen's own rule (read `size/4` scalars, each repeated 4x -- which
      for a packet of 4 or fewer collapses to reading 1 scalar and
      broadcasting it, i.e. `pload1`). Only the SSE and NEON 4-wide
      **float** backends deviate (`_mm_set_ps(from[1], from[1], from[0],
      from[0])` / the NEON equivalent: 2 scalars, each repeated 2x). This
      is not an oversight: `complex_simd.h`'s own `ploadquad` already
      special-cases it explicitly (`repeats = (T=float && size==4) ? 2 :
      4`), and `TestSimdUtility.cxx`/`TestSimdComplex.cxx` test this exact
      lane pattern. Renaming or changing the behavior risks breaking that
      already-tested, intentional special case for no benefit, so
      documented the deviation with a comment at both backends'
      `ploadquad` and at the `complex_simd.h` special case explaining why
      it exists and pointing at the other two. Verified: `ScalarsCxxTests`
      passes (86/86) -- comment-only change.

### 4.3 Metal fused-kernel cache -- Fixed

- [x] The pipeline cache is keyed by kernel source and has no size limit. Add
      an LRU bound or a configurable cap. This machine has real Metal (Apple
      Silicon) GPU hardware, so built and ran this with
      `SCALARS_GPU_BACKEND=metal` rather than just inspecting it.
      `pipeline_for_fused_source()` and the fixed-kernel-name
      `pipeline_for()` shared one `pipeline_cache()` map; split the fused
      one out into its own LRU-bounded structure (`fused_pipeline_map()` +
      a recency `std::list<std::string>`, default capacity 256, in
      [metal_dispatch.mm](../include/backend/gpu/metal/metal_dispatch.mm))
      under its own mutex, so bounding it never evicts or contends with
      the small, fixed kernel-name set `pipeline_for()` still caches
      unbounded. Added
      `set_fused_pipeline_cache_capacity`/`fused_pipeline_cache_capacity`/
      `fused_pipeline_cache_size` to the public
      [metal_dispatch.h](../include/backend/gpu/metal/metal_dispatch.h) API
      for the "configurable cap" half of the ask and so a test could
      observe eviction.
- Test: added `metal_complex.FusedPipelineCacheEviction` to
  [TestMetalComplex.cxx](../Testing/Cxx/TestMetalComplex.cxx): sets
  capacity to 2, compiles three distinct fused sources, asserts the cache
  never exceeds 2 entries, and that re-dispatching the evicted first
  source still produces the correct result (recompiles rather than
  returning something stale). Verified on real Metal hardware
  (`SCALARS_GPU_BACKEND=metal`): all 18 pre-existing Metal tests plus the
  new one pass (19/19 Metal, 105/105 overall); the default
  NEON+Accelerate (no-Metal) build and its 86 tests are unaffected.

## 7. Phase 5: dead code, docs, and project rules

### 5.1 Dead code -- Fixed

- [x] `SCALARS_X86_F16` is never defined. Defined it from the F16C capability
      check (chose this over removing the branches, since it is a real
      perf win: hardware `VCVTPH2PS`/`VCVTPS2PH` instead of the portable
      bit-twiddling fallback -- PyTorch's own `c10/util/Half.h` takes the
      same approach on x86). Added, right after the existing
      `common/intrinsics.h` include in `half.h`:
      `#if defined(__F16C__) && (x86) && !defined(__CUDACC__) && !defined(__HIPCC__)`
      `#define SCALARS_X86_F16 1`. `__F16C__` is predefined by GCC/Clang
      whenever `-mf16c` is active and MSVC gets `/D__F16C__` from
      `cmake/utils.cmake` under the same condition (confirmed: `-mf16c`/
      `/D__F16C__` is added to `SCALARS_SIMD_COMPILER_FLAGS` for the AVX,
      AVX2, and AVX512 backend probes, never for SSE-only), so this mirrors
      the compiler's own ISA selection. Excluded for CUDA/HIP TUs since
      `common/intrinsics.h` (which pulls in the x86 intrinsic headers that
      declare `_cvtsh_ss`/`_cvtss_sh`) is itself not included there.
      Verified: `ScalarsCxxTests` (NEON) still passes (86/86, no change on
      arm64, where this macro stays undefined); cross-compiled
      `-fsyntax-only -target x86_64-apple-darwin` for AVX2, AVX512, and
      SSE-only (no `-mf16c`) all pass; emitting `-S` for the AVX2 case shows
      `fp16_ieee_to_fp32_value` now compiles down to `vcvtph2ps` instead of
      the bit-twiddling sequence, confirming the hardware path is actually
      taken, not just dead-code-eliminated.
- [x] Removed the unreachable `#elif defined(__aarch64__)` in
      `half::operator float()` (was at half.h:457, now part of the `#else`
      arm of the outer `#if defined(__aarch64__) && !__CUDACC__ &&
      !__HIPCC__`, where that same condition is already known false --
      provably dead, not just empirically untaken).
- [x] Fixed the HIP branches in `bfloat16.h` (the `cuda_bf16.h` include
      guard, the two `__nv_bfloat16` member-declaration/definition guards,
      the `operator float()` guard, and the `bfloat16(float)` constructor's
      compound condition): all previously required `defined(__CUDACC__)`
      even for their `SCALARS_USE_HIP` disjunct, but every type/intrinsic
      guarded (`__nv_bfloat16`, `__bfloat16_as_ushort`, `__float2bfloat16`,
      `__bfloat162float`) is NVIDIA-only, with no AMD/ROCm equivalent; real
      HIP (hipcc on AMD, `__HIPCC__` without `__CUDACC__`) could never take
      these branches, and HIP's NVIDIA backend that could has since been
      removed from HIP itself. Chose "remove the branch" over "fix the
      condition": gating the HIP disjunct on `__HIPCC__` instead would
      newly enable an actually-unreachable-today code path that calls
      CUDA-only functions, which would only surface as a build break for
      whoever first builds HIP here -- not verifiable on this machine (no
      HIP toolchain). Simplified every guard to plain
      `defined(__CUDACC__) && !defined(SCALARS_USE_HIP)`, a pure
      simplification with no behavior change for any currently-reachable
      configuration (CUDA, HIP, or neither), and documented why at the
      first occurrence. Verified: `ScalarsCxxTests` still passes (86/86);
      `bfloat16.h` has no remaining `SCALARS_HIP_VERSION` reference.
- [x] Removed the redundant `using half = scalars::half;` (was, inside
      `namespace scalars`, an alias from `half` to `scalars::half` --
      i.e. to itself).

### 5.2 Documentation drift -- Already fixed

- [x] The README refers to `<core/scalar_type.h>`. Change it to
      `<common/scalar_type.h>`. Checked: no longer present. README.md's only
      remaining `core/` hit is the correct upstream link to
      `c10/core/ScalarType.h`; it was dropped as part of the 0.1 README
      rewrite and needs no further change.

### 5.3 CMake summary alignment (CLAUDE.md rule) -- Fixed

- Files: [summary.cmake:4-52](../cmake/summary.cmake#L4-L52) and
  [lto.cmake:70-123](../cmake/lto.cmake#L70-L123).
- Problem: every summary line places the colon at column 25 instead of 24.
  Line 77 is at 27, and the dynamic lines 70 and 123 are at 20 and 21.
- Fix: use two spaces, then the label padded to 20 characters, then `:`.
  Shorten "Default C++ standard" (20 characters, which leaves no room for
  padding) to something like "C++ standard".

```cmake
message("  Icecc               : ${SCALARS_ENABLE_ICECC}")
```

- Check: added `test_summary_message_colon_alignment` to
  `Testing/CMake/test_setup.py`; it scans `cmake/summary.cmake` and
  `cmake/lto.cmake` and asserts that the colon sits at index 23 of every
  literal-label `message("  ...:` line (the two dynamic
  `${_summary_var}` lines, whose label length varies, are left out of the
  fixed-column check and were instead normalized to the same single-space
  spacing as each other). Verified: `python3 -m unittest Testing.CMake.test_setup`
  passes (17 tests).
- Follow-up: the two dynamic lines now go through a padding helper, so short
  variable names also put the colon at column 24; see 7.5.

## 8. Phase 6: CI and tests

### 6.1 CI hygiene

- File: [ci.yml](../.github/workflows/ci.yml).
- [x] Add a lint job that runs lintrunner, clang-format, and clang-tidy.
      `lint-tool` is public at `KhwarizmiAnalytix/lint-tool`; the new `lint`
      job checks it out at a pinned commit next to this repository and pins
      `lintrunner` and `pyyaml`. `lintrunner --all-files` reports 254
      findings on `main`, so it cannot gate changes yet. Instead
      [lint_changes.py](../.github/scripts/lint_changes.py) fails only on
      what a change introduces: `git clang-format --diff` against the merge
      base (changed lines, with the linter's own clang-format binary), and
      every other linter, clang-tidy included, on the changed files at the
      merge base and at HEAD, reporting findings HEAD has and the base does
      not. Verified in Ubuntu 24.04 (WSL): this change set reports 0 new;
      an added reserved identifier and an unformatted line both fail it.
- [x] Pin actions by commit SHA, especially third-party ones
      (`Jimver/cuda-toolkit`, `ilammy/msvc-dev-cmd`). Also pinned
      `mozilla-actions/sccache-action` and `codecov/codecov-action` (both
      third-party) the same way, each with a `# vX.Y.Z` comment recording
      the tag the pinned commit corresponds to. First-party `actions/*`
      actions (`checkout`, `setup-python`, `upload-artifact`) left on their
      major-version tags, consistent with the plan's "especially third-party
      ones" framing. SHAs resolved via `git ls-remote --tags` against each
      action's own repository (peeled `^{}` ref for the one annotated tag,
      `mozilla-actions/sccache-action`'s `v0.0.9`).
- [x] Use one version of `actions/setup-python` (currently a mix of v5 and v7).
      `toolchains.yml` was on v5 while `ci.yml` was already on v7; moved
      `toolchains.yml` to v7 to match.
- [x] Pin the buildcache installer to a commit, not tag `v0.33.1`.
      [install_buildcache.py](../.github/scripts/install_buildcache.py) now
      clones the repository in full (a shallow `--depth 1` clone cannot
      fetch an arbitrary commit, only branch/tag tips) and checks out
      `aa10336e06bc4a6caa55e039b7887047e82dbf20` (the commit `v0.33.1`
      currently points at) explicitly, then initializes submodules.

### 6.2 Test coverage gaps

Every Phase 1 bug sits in an untested area. Checked each bullet below
against the current test files rather than re-adding tests blindly: every
item except the last was already satisfied by tests added alongside the
matching Phase 1 fix above, just never checked off here.

- [x] `numeric_limits` special values for `half` and `bfloat16`. Covered by
      `TestHalf.cxx` and `TestBFloat16.cxx` (1.2).
- [x] Mixed-type comparisons and bitwise operators. Covered by `TestHalf.cxx`
      (1.1) and `TestBFloat16.cxx` (1.3).
- [x] `complex<half>` and `complex<bfloat16>` arithmetic. Covered by
      `TestComplex.cxx` (1.4).
- [x] Annex G special values for the scalar complex math functions. Covered
      by `TestComplexMath.cxx` (1.5, 1.6).
- [x] `hypot` range across all CPU backends. Covered by `TestSimdF2.cxx`
      (1.7).
- [x] `normalcdf` tails and the `inv_normalcdf` domain. Covered by
      `TestNormalCdf.cxx` (1.8, 1.9).
- [x] Relative-error parity checks. Done in two parts on 2026-11-13.
      `expect_value` now uses a normwise relative bound,
      `64 * eps * max(|Re|, |Im|) + numeric_limits<T>::min()`, scaled by the
      whole expected complex value: it absorbs the residue case below (a
      ~1e-16 component next to one near 1) but no longer lets small results
      pass with an absolute error of 64 eps. A normwise bound still cannot
      see 4.1, because `Im(log z) = arg z` dominates the scale there, so
      `SimdComplexParity.LogRealPartRelativeAccuracyNearUnitCircle` checks
      `Re(log z)` componentwise (16 eps relative) for `|z| = 1 ± 2^-k`
      against a cancellation-free reference. It fails on both the original
      formula (696 lanes) and the first 4.1 fix (628 lanes); see 8.4. The
      earlier attempt, kept for the record: the old tolerance,
      `64 * eps * (1 + |expected|)` at
      [TestSimdComplexParity.cxx:135](../Testing/Cxx/TestSimdComplexParity.cxx#L135),
      is effectively absolute near zero, which hides relative-accuracy
      regressions such as 4.1. Attempted: replaced the `(1 + |expected|)`
      floor with `max(|expected|, numeric_limits<T>::min())`, making the
      bound purely relative with a floor only at the denormal boundary.
      Reverted: this broke two real passing cases in
      `SimdComplexParity.PrincipalMathAndRounding` and
      `.MagnitudeSpecialValuesAndAccumulation` where the true value is a
      ULP-scale cancellation residue (e.g. `1.2246467991473532e-16`, the
      double rounding error in a value that is mathematically zero) and one
      SIMD path legitimately rounds that residue to exactly `0` while
      another keeps it -- both are valid results of the same computation,
      not a bug, and the `+1` floor exists precisely to absorb this class of
      case. A correct fix needs a floor sized from each call site's own
      input scale (not a type-wide constant), which `expect_component`
      cannot see from `actual`/`expected` alone -- left as unverified
      end-to-end here; built and ran the full suite
      (`ScalarsCxxTests`, 86/86) to confirm the revert is clean.

## 9. Phase 7: review of the uncommitted changes

A detailed review of the working-tree diff against `647564c` (72 files) on
2026-11-13 found ten issues, some of them regressions introduced by the fixes
above. All ten are resolved; 7.6 to 7.10 were closed on 2026-11-13 together
with Phase 8.

Verification of the fixed items: `python3 Scripts/setup.py config.build.test`
(Release, Ninja, Clang, NEON + Accelerate) builds cleanly, and
`ScalarsCxxTests` (88 cases, 17 suites) and `ScalarsSetupTests` pass.
clang-format reports nothing new on the touched lines. Not run: the sanitizer
build, clang-tidy, and any x86, CUDA, HIP, or SVML build.

### 7.1 NaN divisor in complex division returns inf (confirmed) -- Fixed

- File: [complex.h:44](../include/types/complex.h#L44).
- Problem: `detail_complex::division_abs`, added for 1.4, returned 0 for NaN,
  because both `value > 0` and `value < 0` are false. `operator/=` then took
  the divide-by-zero branch, so `(1+1i) / (NaN+NaNi)` gave `(inf, inf)`. This
  path is always used on CUDA/HIP and in constexpr, and on the host whenever
  a component is not finite.
- Fix: `value < 0 ? -value : (value == 0 ? 0 : value)`. NaN passes through
  unchanged, -0 still maps to +0, and the function stays constexpr.
- Test: `Complex.DivisionByNaNYieldsNaN` in
  [TestComplex.cxx](../Testing/Cxx/TestComplex.cxx), with NaN in both
  components, the real component only, and the imaginary component only.

### 7.2 `complex<bfloat16>` specialization dropped the generic interface (confirmed) -- Fixed

- File: [complex.h:706](../include/types/complex.h#L706).
- Problem: before 1.4, `complex<bfloat16>` was the generic template. The new
  specialization left `real_`/`imag_` uninitialized by default, made the only
  two-argument constructor `explicit` with no real-only form, and had no
  scalar `+=`, `-=`, `*=`, `/=`. `complex<bfloat16> acc; acc += x;` started
  from garbage, and `complex<bfloat16> z = bfloat16(1)` and
  `operator+(complex<T>, T)` no longer compiled.
- Fix: default member initializers (`bfloat16 real_{}`), an implicit
  real-only constructor constrained to exactly `bfloat16`, and scalar compound
  operators computed in `float`. An unconstrained `complex(const bfloat16&)`
  would make `complex<bfloat16>(1.5f)` ambiguous with the `complex<float>`
  converting constructor.
- Test: `Complex.BFloat16KeepsGenericComplexInterface` checks the zero
  default, the implicit constructor, `complex<bfloat16>(1.5f)`, the scalar
  compound and binary operators, and an accumulator loop.
- Not changed: `complex<half>` still has the uninitialized default and no
  scalar operators. That is not a regression (it was always like this), but
  the two types now differ. 7.10 would fix both.

### 7.3 ISA guard fails GPU device compiles (unverified) -- Fixed, verified on CUDA

- File: [scalars_macros.h:35](../include/scalars_macros.h#L35).
- Problem: the 2.3 guard `#error`s when `SCALARS_HAS_AVX2` is set but
  `__AVX2__` is not. The `SCALARS_HAS_*` definitions are PUBLIC, so `.cu` and
  HIP sources get them. Device passes target nvptx or amdgcn and never define
  the host ISA macros, so every GPU build with an x86 CPU backend would stop
  at the `#error`.
- Fix: skip the guard when `__CUDA_ARCH__` or `__HIP_DEVICE_COMPILE__` is
  defined. The host pass still gets the ISA flags through
  `-Xcompiler=${flag}` ([simd.cmake:380](../cmake/simd.cmake#L380)) and is
  still checked.
- Verified 2026-11-13: a CUDA 13.2 build (MSVC host, `cpu_backend=avx2`,
  sm_89) compiles every device pass without the `#error`, and the GPU tests
  run on an RTX 4060 Ti. HIP is still unverified.

### 7.4 SVML linker flags exported to consumers (unverified) -- Fixed, verified on Linux

- File: [simd.cmake:286](../cmake/simd.cmake#L286).
- Problem: when 2.1 exported the SVML targets, it also removed the
  `BUILD_INTERFACE` wrapper from the `-Wl,--no-as-needed` and
  `-Wl,--as-needed` flags around them. Exported, these are positional raw
  linker flags in the consumer's link line. CMake may reorder or deduplicate
  them, and the trailing `--as-needed` applies to every library the consumer
  links after Scalars.
- Fix: only the two flags go back into `BUILD_INTERFACE`. `SVML::SVML`,
  `SVML::IRC`, and `SVML::INTLC` stay exported, as 2.1 requires.
- Verified 2026-11-13 on Ubuntu 24.04 (WSL, x86_64), mirroring the
  `install-consumer-svml` job: the exported targets carry only `SVML::SVML;
  SVML::IRC;SVML::INTLC`, no installed CMake file or consumer link line has
  `--as-needed`, `libScalars.so` has RUNPATH `$ORIGIN/../lib:$ORIGIN`, and
  after moving the install prefix the consumer resolves `libintlc.so.5`
  and its tests pass.

### 7.5 Dynamic summary labels not aligned (CLAUDE.md rule) -- Fixed

- File: [summary.cmake:1](../cmake/summary.cmake#L1).
- Problem: the two `message("  ${_summary_var} : ...")` lines put the colon
  wherever the variable name ended.
- Fix: a `_scalars_summary_variable()` helper pads the label to 20
  characters, so the colon is at column 24. Names of 20 characters or more,
  such as `CMAKE_CXX_FLAGS_RELEASE`, cannot fit and get one space.
- Check: the configure output aligns for short names (`CMAKE_LINKER`,
  `MEMORYCHECK_COMMAND`); `ScalarsSetupTests` passes.

### 7.6 `half` comparisons are ambiguous on CUDA/HIP (confirmed) -- Fixed

- File: [half.h:748](../include/types/half.h#L748).
- Problem: 1.1 removed every `(half, half)` comparison operator. The deleted
  comment said they were "the single exact match for CUDA, where half also
  converts to `__half`". Without them, `h1 < h2` in device code is ambiguous
  between the built-in float comparison and `cuda_fp16.h`'s global
  `operator<(const __half&, const __half&)`.
- Confirmed: with CUDA 13.2, `TestGpuSimd.cu` failed to compile with "more
  than one operator `<` matches" for `half < half`.
- Fix: constrained comparison templates in [half.h](../include/types/half.h)
  for `half` against `half` or any arithmetic type. Both operands match
  exactly, so they beat the built-in float comparison and the `__half`
  operators, and `h == 1.0f` stays unambiguous (the reason 1.1 removed the
  non-template overloads). Operands compare in the usual arithmetic common
  type with `half` as `float`, so results are unchanged. This needs no
  `__CUDA_NO_HALF_OPERATORS__`, which would have changed what all device
  code sees. `__hneg` now gets an explicit `__half`.
- Test: `GpuSimd.HalfComparisonsOnDevice` runs all six operators for
  `(half, half)`, `(half, float)`, `(float, half)`, `double` and `int` in a
  kernel over 81 input pairs including signed zeros, 65504 and NaN.

### 7.7 `accumulation_type_t<float>` is now `double` (API change) -- Release note added

- File: [scalar_traits.h:19](../include/common/scalar_traits.h#L19).
- Problem: 3.2 made this change deliberately. It is still a public API
  break: `complex<float>` now accumulates in `complex<double>`,
  `simd<accumulation_type_t<float>>` becomes `simd<double>`, and types with
  no `acc_type` entry (`long double`, `uint32_t`) no longer compile.
- Done: kept the new behaviour and added a README "Upgrade notes" section
  describing it, with the other caller-visible changes from this plan.
  Keeping the old alias under a deprecation remains possible if a
  downstream project needs it.

### 7.8 `set_mask` literal calls are ambiguous -- Fixed

- Files: [complex_simd.h:361](../include/backend/cpu/detail/complex_simd.h#L361)
  and [reduced/complex_simd.h:207](../include/backend/cpu/reduced/complex_simd.h#L207).
- Problem: after 3.3, `set_mask(int_t)` sits next to `set_mask(bool)`.
  `set_mask(1)` does not compile, because `int` to `bool` and `int` to
  `uint32_t`/`uint64_t` are equally good conversions. That is the obvious way
  to migrate from the old `set(1)` mask call.
- Fix: `set_mask(int_t)` became `set_mask(I)` constrained to integral,
  non-`bool` `I`, in both complex packet templates. It is an exact match for
  every integer type, and `set_mask(true)` still picks the `bool` overload.
- Test: `SimdComplexParity.MasksAndSelection` stores
  `set_mask(1)`, `set_mask(0)`, `set_mask(int64_t(7))`, `set_mask(0u)`;
  `SimdComplexParity.ReducedSetMaskAcceptsIntegerLiterals` covers the
  `complex<half>`/`complex<bfloat16>` packets through `if_else`.

### 7.9 `log_fast` computes two transcendentals per packet -- Fixed

- File: [complex_simd.h:185](../include/backend/cpu/detail/complex_simd.h#L185).
- Problem: the 4.1 fix computes both `log1p` and `log` on every lane and
  selects one with `if_else`. This doubles the cost of every complex `log`
  packet, and 4.2 also made this the fast path for Accelerate and MKL builds.
- Fix: `log_fast` evaluates the refined formula only if some lane is in
  the band and the plain one only if some lane is outside it, using the
  backends' existing `Ops::all`.
- Benchmark (MSVC, AVX2, SVML, i9-14900K, median of 5): complex `log`
  improved 1-4% for float and about 8% for double, e.g. double/262144 from
  2.34 ms to 2.15 ms. The benchmark's inputs straddle the band edge, so
  most packets still need both formulas; packets wholly inside or outside
  the band now make one transcendental call instead of two.

### 7.10 Duplicated reduced-precision complex code -- Fixed

- File: [complex.h:634](../include/types/complex.h#L634) and
  [complex.h:706](../include/types/complex.h#L706).
- Problem: `complex<half>` and `complex<bfloat16>` are near-copies, and they
  already differ after 7.2.
- Fix: `detail_complex::reduced_complex_base<T>` holds the whole
  implementation (float intermediates, division in `complex<float>`, zero
  default, implicit real constructor, scalar compound operators); the two
  specializations only inherit its constructors. `complex<half>` gains the
  7.2 interface.
- Test: the 7.2 test is now a template run as
  `Complex.BFloat16KeepsGenericComplexInterface` and
  `Complex.HalfKeepsGenericComplexInterface`.

## 10. Phase 8: Windows x86 and CUDA verification

Run on 2026-11-13 against `fd796ab` on Windows 11 (i9-14900K: AVX2, no
AVX-512; RTX 4060 Ti, sm_89; CUDA 13.2; MSVC 14.44; LLVM/Clang 22) and in
WSL Ubuntu 24.04 (Clang 22, GCC 13). Phases 0 to 7 had only been built on
arm64, with x86 cross-compiled or emulated, and none of the CUDA work had
been compiled. Three defects turned up, besides confirming 7.6.

### 8.1 MSVC builds fail in `half.h` after 5.1 (confirmed) -- Fixed

- File: [half.h](../include/types/half.h).
- Problem: 5.1 enabled `SCALARS_X86_F16`, whose path calls `_cvtsh_ss` and
  `_cvtss_sh`. Those scalar helpers exist in GCC and Clang only; MSVC has
  just the packed `_mm_cvtph_ps`/`_mm_cvtps_ph`. Every MSVC AVX/AVX2/AVX-512
  build stopped with C3861. 5.1 was verified with Clang cross-compiles only.
- Fix: under MSVC (not clang-cl), convert lane 0 with the packed forms, as
  PyTorch does.

### 8.2 Complex arithmetic returned zeros in CUDA device code (confirmed) -- Fixed

- Files: [complex.h](../include/types/complex.h),
  [bit_cast.h](../include/common/bit_cast.h),
  [Testing/Cxx/CMakeLists.txt](../Testing/Cxx/CMakeLists.txt).
- Problem: the free `complex` operators, the member compound operators, and
  `std::real`/`imag`/`norm`/`conj` were `constexpr` but not
  `SCALARS_HOST_DEVICE`, and `bit_cast` was `std::bit_cast`. PyTorch, the
  origin of this code, compiles CUDA with `--expt-relaxed-constexpr`; this
  repository does not, so nvcc only warned (#20013) and the device calls
  produced nothing: `GpuSimd.ArithmeticAndMathComplex{Float,Double}` read
  back all zeros. Pre-existing, not a regression from this plan.
- Fix: annotate those functions `SCALARS_HOST_DEVICE`; CUDA/HIP translation
  units use the `SCALARS_HOST_DEVICE` `memcpy` form of `bit_cast`. The test
  target now builds CUDA with `--diag-error=20011,20013,20014,20015`, so a
  host-only call from device code fails the build instead of miscomputing.
  `TestGpuSimd.cu` also checks `cudaGetLastError` after every launch.

### 8.3 Clang on Windows called Intel SVML with the wrong convention (confirmed) -- Fixed

- Files: [sse/svml.h](../include/backend/cpu/sse/svml.h),
  [avx/svml.h](../include/backend/cpu/avx/svml.h),
  [avx512/svml.h](../include/backend/cpu/avx512/svml.h).
- Problem: SVML takes vectors in registers (`__vectorcall`). The shim cast
  each entry point to a `SCALARS_VECTORCALL` pointer, but that macro is
  empty under Clang, so Clang used the default x64 convention (vector by
  pointer in `rcx`) and SVML computed on whatever was left in `ymm0`.
  Single calls often passed by luck; `log_fast`, which makes two calls,
  returned e.g. `log(0.3)` for `log|0.2+0.3i|` and `-inf` for `log|-4|`.
  `SimdComplexParity` failed 4 tests with Clang + SVML on Windows, and an
  older build from before this plan failed 2. An earlier workaround for
  Debug `pow`/`hypot` hid the same cause.
- Fix: under Clang on Windows, declare each entry point `__vectorcall` with
  an asm label giving its undecorated DLL symbol (`"__svml_logf8"`), and
  drop the default-convention prototypes there, which Clang would otherwise
  merge with the new declarations. The Debug `pow`/`hypot` fallback now
  applies to cl.exe only.

### 8.4 `log_fast` real part still lost accuracy near |z| = 1 (confirmed) -- Fixed

- File: [complex_simd.h](../include/backend/cpu/detail/complex_simd.h).
- Problem: 4.1's `(hi+1)(hi-1) + lo^2` keeps the rounding errors of `hi+1`
  and `lo^2`, which are as large as the result when `|z|^2 - 1` is tiny
  (the 4.1 entry itself measured 3% relative error). `P - 1` alone is not
  exact either: at `re ~ im ~ 0.707`, `P = hi^2 < 0.5`.
- Fix: split both squares exactly with `fma` (`x^2 = P + E`), add
  `P + p` with Fast2Sum, and subtract 1 from that sum, which is exact by
  Sterbenz's lemma because it is close to `|z|^2`. Two extra `fma` per lane
  in the band; without hardware `fma` the error terms are 0.
- Test: `SimdComplexParity.LogRealPartRelativeAccuracyNearUnitCircle` (see
  6.2); runs when SVML, MKL, SLEEF, or Accelerate enables `log_fast`.

### 8.5 Verification

| Configuration | Result |
| --- | --- |
| MSVC, AVX2 | 89/89 |
| MSVC, AVX2, SVML (compiler intrinsics) | 90/90 |
| Clang, AVX2, SVML DLL, Release and Debug | 90/90 each |
| Clang, SSE, SVML DLL | 90/90 |
| Clang, AVX-512, SVML DLL | Builds; this CPU cannot run AVX-512 |
| MSVC + CUDA 13.2, AVX2, RTX 4060 Ti | 97/97, including 8 GPU tests |
| Ubuntu 24.04: Clang and GCC, AVX2, SVML | 90/90 each |
| Ubuntu 24.04: Clang, ASan + UBSan, Debug | 90/90 |
| Ubuntu 24.04: install, move, consumer (7.4) | Passes |
| Lint gate (6.1) on this change set | 0 new findings |

`ScalarsSetupTests` passes in every configuration. Not run: HIP, Metal,
SVE, and AVX-512 at run time. While building, this machine's compilers
crashed at random (GCC internal errors, Clang segfaults) on files that
compiled cleanly on retry; every result above comes from a complete,
clean build.

## 11. Verification for each phase

For each pull request:

1. `python3 Scripts/setup.py config.build.test` (Release).
2. A sanitizer build with `SCALARS_SANITIZER_TYPE=address,undefined`.
3. clang-tidy on the touched first-party files
   ([clang-tidy skill](../.agents/skills/clang-tidy/SKILL.md)).
4. The CI matrix for the x86, CUDA, HIP, and Metal backends, which cannot be
   tested locally on arm64.
5. The [session checklist](../.agents/skills/session-checklist/SKILL.md)
   before handoff.

## 12. Suggested pull request order

| Order | Pull request | Items | Risk |
| --- | --- | --- | --- |
| 1 | Provenance and notices | 0.1, 0.2 | None to code |
| 2 | Scalar type fixes | 1.1, 1.2, 1.3, 5.1 | Low; 1.1 changes overload resolution |
| 3 | Reduced complex arithmetic | 1.4 | Medium |
| 4 | Complex math and `normalcdf` | 1.5, 1.6, 1.8, 1.9 | Low |
| 5 | NEON `hypot` and backend consistency | 1.7, 4.1, 4.2 | Medium; needs benchmarks |
| 6 | Install and package config | 2.1, 2.2, 2.4, 2.5 | Medium; needs x86 and AArch64 CI |
| 7 | Summary alignment and CI lint | 5.2, 5.3, 6.1 | Low |
| 8 | Namespace and type-system cleanup | 3.1 to 3.4 | High; breaks the API |
| 9 | ISA dispatch design | 2.3 (long term) | High; design decision first |
| 10 | Review follow-ups, fixed | 7.1 to 7.5 | Low; land with PRs 3, 6, and 7 |
| 11 | Review follow-ups | 7.6 to 7.10 | Medium; 7.6 changes overload resolution, 7.10 class structure |
| 12 | Windows x86 and CUDA fixes, lint gate | 8.1 to 8.4, 6.1, 6.2 | Medium; 8.2 changes device-code behaviour |

Phase 6.2 tests land with the matching fix rather than as a separate pull
request.
