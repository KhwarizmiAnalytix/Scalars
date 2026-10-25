# Build and verification

When the task mentions building, testing, coverage, sanitizers, configuring,
or a build failure, load the `project-build` skill. Do not load it for an
ordinary source edit with no build or test request. Then read `CLAUDE.md` for
this repository's actual build commands. Use existing setup helpers where
provided. Python packages use their packaging and unittest workflow instead
of a C++ build.

Configure before the first build and after changing build options or source
registration. Run tests for changed behavior. This repository builds with
CMake only; do not invent a second build system.

Load the `session-checklist` skill only when the user requests a handoff,
commit-readiness, full verification, or review checklist. For guidance-only
edits, check links, syntax, and the diff; compilation is not required. Report
what actually ran and any blockers.
