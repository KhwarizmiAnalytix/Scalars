# Scalars

Standalone C++20 scalar library for half, bfloat16, and complex<T>, with dtype
metadata, traits, promotion, casts, and math. Public headers live in
`include/`; the umbrella header is `<scalars.h>`. The library target is
`Scalars::Scalars`. Tests live in
`Testing/`. Use namespace `scalars` and `SCALARS_*` macros from
`include/scalars_macros.h` and `include/common/scalars_export.h`. Dependencies
are under `ThirdParty/`.

## Shared agent guidance

Adapted from the Logging repository's rules and skills, which follow the public
[XSigma rules and skills](https://github.com/KhwarizmiAnalytix/XSigma/tree/89848c54492abef57fd0d0dc53b9da96b7cd1d5d)
at revision `89848c54492abef57fd0d0dc53b9da96b7cd1d5d`. Local API, dependency,
language, and build conventions below specialize that guidance for this
standalone repository.

The cross-tool entry point is [.agent-rules.md](.agent-rules.md). Cursor and
Codex load it through [.cursor/rules/scalars.mdc](.cursor/rules/scalars.mdc)
and [AGENTS.md](AGENTS.md); Claude should use it as the index for the same
authoritative rules under `.agent-rules/`.

Read the applicable rules before editing. They apply to Claude, Cursor, and
Codex; C++ rules apply only when working on C++:

- [C++ coding](.agent-rules/coding.md) and [builders](.agent-rules/builder.md)
- [Python](.agent-rules/python.md)
- [Testing](.agent-rules/testing.md) and [builds](.agent-rules/build%20rule.md)
- [Dependencies](.agent-rules/ThirdParty.md)
- [Portability](.agent-rules/must-have.md) and [documentation](.agent-rules/markdown.md)

Use these task-specific skills as needed:

- [project-build](.agents/skills/project-build/SKILL.md): configure, build, and test
- [new-test](.agents/skills/new-test/SKILL.md): add tests using local conventions
- [clang-tidy](.agents/skills/clang-tidy/SKILL.md): analyze first-party C++ when applicable
- [session-checklist](.agents/skills/session-checklist/SKILL.md): verify completed work

## Build and test

Use the setup helper from `Scripts/`; inspect its help before adding
feature flags:

```sh
cd Scripts
python3 setup.py --help
python3 setup.py config.build.test
```

The same helper runs from the repository root as `python3 Scripts/setup.py`.
It defaults to Ninja, Clang, C++20, and Release, and writes `build_ninja/`
with suffixes for selected options. Direct CMake commands in `README.md` and
CI are also valid. This repository has no Bazel build.

CPU SIMD is selected with `cpu_backend` (`no`, `sse`, `avx`, `avx2`, `avx512`,
`neon`, `sve`). GPU is selected independently with `gpu_backend` (`none`,
`cuda`, `hip`, `metal`). Optional math libraries use `sleef` and the CMake
options documented in `README.md` (`SCALARS_ENABLE_SVML`, `SCALARS_ENABLE_MKL`,
`SCALARS_ENABLE_ACCELERATE`). Do not pass another repository's backend flags.

## Test conventions

Match adjacent Google Test cases. C++ tests are `Test*.cxx` under
`Testing/Cxx/` and are picked up by the CMake glob into `ScalarsCxxTests`.
`Test*.cu` is added only when `SCALARS_GPU_BACKEND` is `cuda` or `hip`. Metal
coverage is `Test*.cxx` and is part of the default glob. The shared test
header is generated from `Testing/scalars_test.h.in`.

Python build-helper checks live in `Testing/CMake/test_setup.py` and use
unittest (`ScalarsSetupTests`). Benchmarks are `Testing/Benchmark/Benchmark*.cxx`
and build only when `SCALARS_ENABLE_BENCHMARK=ON`. There is no Bazel test
list to update.

## Verification and scope

For non-trivial source or build changes, run affected tests, review the diff,
and run configured lint/static-analysis checks relevant to touched files.
Follow the session checklist and report checks run, failures, and unavailable
tools explicitly. Guidance-only changes need frontmatter/link/whitespace
validation, not compilation.

Keep unrelated user edits and dependency sources intact. Share review
findings in the response or pull request; do not create unsolicited status
documents. Follow this repository's existing license and contribution policy.
The default branch is `main`.

## CMake configuration message alignment

All `message("  LABEL : value")` configuration summary messages in
`cmake/summary.cmake` and `cmake/lto.cmake` must align colons at exactly
**24 characters from the opening quote** (inclusive).

Format: `message("  LABEL{PADDING}: VALUE")`
- Opening `"`: position 1
- Two spaces + label + padding: positions 2-23 (22 chars total)
- Colon `:`: position 24

Example:
```cmake
message("  Icecc               : ${SCALARS_ENABLE_ICECC}")
```

This keeps the colons in the configuration summary on one vertical line.
