# Tests and coverage

When the task asks to add or change tests, load the `new-test` skill. For an
ordinary code change without a test request, do not load it automatically.

Cover changed behavior with focused tests for success, boundaries, invalid
inputs, failure handling, and relevant state changes. Keep tests independent
and deterministic; use fixtures when they reduce repeated setup.

Match the language, test framework, macros, filenames, and directory layout
already used by this repository. Consult `CLAUDE.md` and the `new-test` skill
when adding or extending tests. Do not import another project's test macros
or helper headers.
Preserve existing exception assertions when exceptions are the public API.

Check how tests are registered. Verify glob patterns actually include new
files. This repository registers C++ tests with a CMake glob; there is no
Bazel package to update.

Run affected tests and the repository's configured coverage checks when
relevant. Report measured coverage honestly; do not assert a fixed 98%
merge gate unless the repository actually enforces it.
