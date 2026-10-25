#!/usr/bin/env python3
"""Compile identical C++ packet loops against Scalars and installed PyTorch."""

import argparse
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    """Parse CLI arguments, compile the comparison harness for the selected
    backend, and run it against the installed PyTorch build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, nargs="+", default=[4096])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "clang++"))
    parser.add_argument(
        "--backend", choices=("sse", "avx", "avx2", "avx512", "neon", "sve")
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        default=["add", "mul", "div", "fma"],
        choices=(
            "all",
            "add",
            "sub",
            "mul",
            "div",
            "fma",
            "sqr",
            "abs",
            "arg",
            "sqrt",
            "invsqrt",
            "reciprocal",
            "exp",
            "expm1",
            "exp2",
            "exp10",
            "log",
            "log1p",
            "log2",
            "log10",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            "pow",
            "ceil",
            "floor",
            "trunc",
            "neg",
            "conj",
            "real",
            "imag",
            "eq",
            "neq",
        ),
    )
    args = parser.parse_args()

    import torch

    root = Path(__file__).resolve().parents[1]
    compiler = shutil.which(args.cxx)
    if compiler is None:
        parser.error(f"C++ compiler not found: {args.cxx}")
    torch_include = Path(torch.__file__).parent / "include"
    torch_library = Path(torch.__file__).parent / "lib"
    if os.name == "nt":
        library_flags = [
            str(torch_library / "torch_cpu.lib"),
            str(torch_library / "c10.lib"),
        ]
    else:
        library_flags = [
            f"-L{torch_library}",
            "-ltorch_cpu",
            "-lc10",
            f"-Wl,-rpath,{torch_library}",
        ]
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
            f"elements must be positive multiples of {width}; "
            "iterations must be positive"
        )
    print(
        f"PyTorch {torch.__version__}, {torch.version.git_version}; "
        f"compiler: {compiler}; backend: {backend}",
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
                "-DSCALARS_STATIC_DEFINE=1",
                *options[backend],
                f"-I{root / 'include'}",
                f"-I{torch_include}",
                str(root / "Testing/Benchmark/CompareComplexSimdPytorch.cpp"),
                str(root / "src/math/complex_math.cpp"),
                *library_flags,
                "-o",
                str(executable),
            ],
            check=True,
        )
        run_environment = os.environ.copy()
        if os.name == "nt":
            run_environment["PATH"] = (
                str(torch_library) + os.pathsep + run_environment.get("PATH", "")
            )
        for elements in args.elements:
            subprocess.run(
                [
                    str(executable),
                    str(elements),
                    str(args.iterations),
                    "all" if "all" in args.operations else ",".join(args.operations),
                ],
                check=True,
                env=run_environment,
            )


if __name__ == "__main__":
    main()
