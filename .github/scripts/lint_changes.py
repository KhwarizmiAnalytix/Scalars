#!/usr/bin/env python3
"""Fail only on lint findings that a change introduces.

The repository has pre-existing findings, so `lintrunner --all-files` cannot
gate pull requests. This script checks a change against its merge base:

* Formatting (clang-format) is checked on the changed lines only, with
  `git clang-format --diff`, using the same binary as the CLANGFORMAT linter.
* Every other lintrunner linter runs on the changed files twice, once at the
  merge base (in a temporary worktree) and once at HEAD. A finding is new when
  HEAD has more findings with the same (path, linter, name, severity,
  description) than the merge base. Line numbers are left out of that key so
  that unrelated edits that move existing findings do not count as new.

Run from the repository root after `lintrunner init`.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FORMATTER_EXTENSIONS = "h,hpp,hxx,c,cc,cpp,cxx,cu,cuh"
FindingKey = tuple[str, str, str, str, str]


def run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True)


def changed_files(root: Path, base: str) -> list[str]:
    out = run(["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base}...HEAD"], root)
    return [line for line in out.stdout.splitlines() if line.strip()]


def merge_base(root: Path, base: str) -> str:
    return run(["git", "merge-base", base, "HEAD"], root).stdout.strip()


def lint(tree: Path, paths: list[str], skip: list[str], configure: str | None) -> list[dict]:
    existing = [p for p in paths if (tree / p).is_file()]
    if not existing:
        return []
    if configure:
        result = subprocess.run(configure, cwd=tree, shell=True, text=True, capture_output=True)
        if result.returncode != 0:
            sys.stderr.write(result.stdout + result.stderr)
            raise SystemExit(f"configure command failed in {tree}")
    cmd = ["lintrunner", "--output", "json"]
    if skip:
        cmd += ["--skip", ",".join(skip)]
    result = run(cmd + existing, tree, check=False)
    findings = []
    resolved_tree = tree.resolve()
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            finding = json.loads(line)
            # lintrunner reports absolute paths; make them relative so findings
            # from the merge-base worktree and from HEAD compare equal.
            path = finding.get("path")
            if path:
                try:
                    finding["path"] = Path(path).resolve().relative_to(resolved_tree).as_posix()
                except ValueError:
                    pass
            findings.append(finding)
    if result.returncode not in (0, 1):
        sys.stderr.write(result.stderr)
        raise SystemExit(f"lintrunner failed in {tree} (exit {result.returncode})")
    return findings


def key(finding: dict) -> FindingKey:
    return (
        str(finding.get("path") or ""),
        str(finding.get("code") or ""),
        str(finding.get("name") or ""),
        str(finding.get("severity") or ""),
        str(finding.get("description") or ""),
    )


def new_findings(base: list[dict], head: list[dict]) -> list[dict]:
    remaining = collections.Counter(key(f) for f in base)
    fresh = []
    for finding in head:
        k = key(finding)
        if remaining[k] > 0:
            remaining[k] -= 1
        else:
            fresh.append(finding)
    return fresh


def check_format(root: Path, base: str, binary: Path) -> str:
    result = run(
        [
            "git",
            "clang-format",
            "--binary",
            str(binary),
            "--extensions",
            FORMATTER_EXTENSIONS,
            "--diff",
            base,
        ],
        root,
        check=False,
    )
    text = result.stdout.strip()
    if text.startswith("no modified files") or "did not modify any files" in text:
        return ""
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="Revision to compare against")
    parser.add_argument(
        "--configure",
        help="Shell command that creates a compilation database in a tree "
        "(needed by CLANGTIDY); run in both the merge base and HEAD",
    )
    parser.add_argument(
        "--skip", default="", help="Comma-separated lintrunner linters to skip"
    )
    args = parser.parse_args()

    root = Path(run(["git", "rev-parse", "--show-toplevel"], Path.cwd()).stdout.strip())
    base = merge_base(root, args.base)
    paths = changed_files(root, base)
    print(f"Merge base {base}; {len(paths)} changed file(s).")
    if not paths:
        return 0

    failed = False
    clang_format = root / ".lintbin" / "clang-format"
    if clang_format.is_file():
        diff = check_format(root, base, clang_format)
        if diff:
            failed = True
            print("::error::Changed lines are not clang-formatted. Run "
                  f"`git clang-format --binary .lintbin/clang-format {base}`.")
            print(diff)
        else:
            print("clang-format: changed lines are formatted.")
    else:
        print("clang-format: .lintbin/clang-format missing; run `lintrunner init`.")
        failed = True

    skip = ["CLANGFORMAT"] + [s for s in args.skip.split(",") if s]
    head_findings = lint(root, paths, skip, args.configure)
    worktree = Path(tempfile.mkdtemp(prefix="lint-base-"))
    try:
        run(["git", "worktree", "add", "--detach", str(worktree), base], root)
        if (root / ".lintbin").is_dir():
            shutil.copytree(root / ".lintbin", worktree / ".lintbin")
        base_findings = lint(worktree, paths, skip, args.configure)
    finally:
        run(["git", "worktree", "remove", "--force", str(worktree)], root, check=False)
        shutil.rmtree(worktree, ignore_errors=True)

    fresh = new_findings(base_findings, head_findings)
    print(f"lintrunner: {len(head_findings)} finding(s) in changed files at HEAD, "
          f"{len(base_findings)} at the merge base, {len(fresh)} new.")
    for f in fresh:
        location = f"{f.get('path')}:{f.get('line') or ''}"
        print(f"::error file={f.get('path')},line={f.get('line') or 1}::"
              f"[{f.get('code')}/{f.get('name')}] {f.get('description') or ''}")
        print(f"  {location} [{f.get('code')}/{f.get('name')}] {f.get('description') or ''}")
    return 1 if failed or fresh else 0


if __name__ == "__main__":
    sys.exit(main())
