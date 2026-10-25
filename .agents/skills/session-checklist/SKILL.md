---
name: session-checklist
description: Verify non-trivial changes in Scalars before handoff or commit, including the CMake build, tests, configured lint, and diff review. Use only when a handoff, commit-readiness, full verification, or review checklist is requested.
---

# session-checklist

Do not invoke this skill for every edit or commit. Use it only when the task
explicitly requests a full verification or handoff review.

Read [CLAUDE.md](../../../CLAUDE.md) and review the working diff without
discarding unrelated user changes. Scale the checks to what changed.

1. **Build definitions.** For source or build changes, check the affected
   CMake options, glob patterns, and dependencies. For the Python helper
   checks, confirm unittest discovery in `Testing/CMake/`. Do not invent a
   second build system; this repository is CMake only.
2. **Build and test.** Use [project-build](../project-build/SKILL.md). Run
   affected tests and the required broader checks. Explain any
   platform or toolchain blocker instead of skipping it silently.
3. **Lint and analysis.** Inspect `.lintrunner.toml`, `pyproject.toml`, and
   CI where present. Run configured checks for touched files. The default
   branch is `main`. Use the clang-tidy skill for applicable C++ changes.
   Scope automatic fixes and review their diff.
4. **Review.** Check names, API/exception contracts, ownership, portability,
   dependency boundaries, and test registration against local rules. Use
   available review tools when appropriate, without assuming a particular
   slash-command plugin exists. Report actual findings.
5. **Report.** State what passed, what failed, and what was not run and why.
   Identify known unrelated failures without silently fixing them or
   treating new failures as pre-existing.

For changes limited to these markdown rules and skills, validate frontmatter,
relative links, command references, and whitespace, then inspect the diff.
Compilation, runtime tests, and clang-tidy are unnecessary for that scope.
