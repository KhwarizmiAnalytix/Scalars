#!/usr/bin/env python3
"""Compile identical C++ packet loops against Scalars and installed PyTorch."""

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, nargs="+", default=[4096])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "clang++"))
    parser.add_argument(
        "--backend", choices=("sse", "avx", "avx2", "avx512", "neon", "sve")
    )
    args = parser.parse_args()

    import torch

    root = Path(__file__).resolve().parents[1]
    compiler = shutil.which(args.cxx)
    if compiler is None:
        parser.error(f"C++ compiler not found: {args.cxx}")
    torch_include = Path(torch.__file__).parent / "include"
    machine = platform.machine().lower()
    options = {
        "sse": ["-msse2", "-DSCALARS_HAS_SSE=1"],
        "avx": ["-mavx", "-mf16c", "-DSCALARS_HAS_AVX=1"],
        "avx2": [
            "-mavx2",
            "-mf16c",
            "-mfma",
            "-DSCALARS_HAS_AVX=1",
            "-DSCALARS_HAS_AVX2=1",
            "-DSCALARS_CPU_CAPABILITY_AVX2",
            "-DCPU_CAPABILITY_AVX2",
        ],
        "avx512": [
            "-mavx512f",
            "-mavx512dq",
            "-mavx512vl",
            "-mavx512bw",
            "-mavx512cd",
            "-mf16c",
            "-mfma",
            "-DSCALARS_HAS_AVX512=1",
            "-DSCALARS_CPU_CAPABILITY_AVX512",
            "-DCPU_CAPABILITY_AVX512",
        ],
        "neon": ["-march=armv8.2-a", "-DSCALARS_HAS_NEON=1"],
        "sve": ["-march=armv8-a+sve", "-msve-vector-bits=128", "-DSCALARS_HAS_SVE=1"],
    }
    if machine in ("arm64", "aarch64"):
        backend = args.backend or "neon"
        if backend not in ("neon", "sve"):
            parser.error("x86 backends require an x86 host")
    elif machine in ("x86_64", "amd64"):
        backend = args.backend or "avx2"
        if backend not in ("sse", "avx", "avx2", "avx512"):
            parser.error("ARM backends require an ARM host")
    else:
        parser.error(f"Unsupported benchmark architecture: {machine}")
    width = {"sse": 4, "avx": 8, "avx2": 8, "avx512": 16, "neon": 4, "sve": 4}[backend]
    if args.iterations <= 0 or any(n <= 0 or n % width for n in args.elements):
        parser.error(
            f"elements must be positive multiples of {width}; iterations must be positive"
        )
    print(
        f"PyTorch {torch.__version__}, {torch.version.git_version}; compiler: {compiler}; backend: {backend}",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="scalars-complex-benchmark-") as build:
        executable = Path(build) / "compare"
        subprocess.run(
            [
                compiler,
                "-O3",
                "-DNDEBUG",
                "-std=c++20",
                "-DSCALARS_VECTORIZED=1",
                *options[backend],
                f"-I{root / 'include'}",
                f"-I{torch_include}",
                str(root / "Testing/Benchmark/CompareComplexSimdPytorch.cpp"),
                "-o",
                str(executable),
            ],
            check=True,
        )
        for elements in args.elements:
            subprocess.run(
                [str(executable), str(elements), str(args.iterations)], check=True
            )


if __name__ == "__main__":
    main()
