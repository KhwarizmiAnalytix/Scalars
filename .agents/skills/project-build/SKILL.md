---
name: project-build
description: Configure, build, and test Scalars with its own supported tooling. Use for build, test, coverage, or sanitizer requests; consult local capabilities before selecting optional flags.
---

# project-build

Adapted from Logging's `project-build` skill for Scalars.
Read [CLAUDE.md](../../../CLAUDE.md) for repository boundaries and test conventions.

Use this skill only when the task mentions build, test, coverage, sanitizer,
configuration, or a build failure. Do not invoke it for an ordinary source
edit with no build or test request.

Use the setup helper from `Scripts/`; inspect its help before adding
feature flags:

```sh
cd Scripts
python3 setup.py --help
python3 setup.py config.build.test
```

`python3 Scripts/setup.py` from the repository root is the same helper.
For compiler or generator requirements, follow `README.md` and CI.
The repository also documents direct CMake commands for integration and CI.
There is no Bazel build; do not invent one.

Select a CPU SIMD backend with a `cpu_backend` token (`no`, `sse`, `avx`,
`avx2`, `avx512`, `neon`, `sve`) and a GPU backend with `gpu_backend`
(`none`, `cuda`, `hip`, `metal`). Optional coverage, sanitizer, LTO, and
analysis tokens such as `coverage`, `clangtidy`, and `cppcheck` are documented
by `setup.py --help`. Do not reuse another repository's backend flags.

Scope repeated test runs to affected behavior when the framework supports
it, then run the required broader checks before handoff. The C++ suite is
`ScalarsCxxTests`. Use only options documented in this repository.

Distinguish missing prerequisites from build or test failures. Report the
command, selected configuration, and result; do not report unrun checks as
passing. Keep generated files in the normal build or temporary directories.
