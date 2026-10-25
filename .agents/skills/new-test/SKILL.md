---
name: new-test
description: Add or extend a unit test in Scalars, preserving its framework, test registration, and error-handling contracts. Use when writing tests for changed behavior.
---

# new-test

Read [CLAUDE.md](../../../CLAUDE.md) and the existing test closest to
the changed behavior before choosing a filename, fixture, or macro.

Follow neighboring Google Test cases. C++ tests are `Test*.cxx` under
`Testing/Cxx/` and are picked up by the CMake glob into `ScalarsCxxTests`.
`Test*.cu` is compiled only when the GPU backend is `cuda` or `hip`. Metal
tests stay `Test*.cxx`. The shared header is generated from
`Testing/scalars_test.h.in`. Use namespace `scalars` and `SCALARS_*` macros.
Do not add a Bazel target; this repository has none.

1. Search for existing coverage with `rg`; extend the relevant test file
   instead of creating a duplicate suite.
2. Match the neighboring includes, namespace, fixtures, naming, and license
   conventions. Do not bring in test helpers from a different repository.
3. Cover the meaningful success, boundary, and failure cases for the change.
   Assert public behavior and the repository's documented error contract.
4. Check that the filename matches the CMake glob (`Test*.cxx`, or `Test*.cu`
   for CUDA/HIP) so the new cases actually execute. A file that misses the
   glob is not registered.
5. Run the affected tests using [project-build](../project-build/SKILL.md).
   Report the test command and result, including any unavailable CPU or GPU
   backend.
