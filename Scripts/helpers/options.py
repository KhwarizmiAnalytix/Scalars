"""Supported, documented setup options. Arbitrary build-tool flags are rejected."""

from typing import TypedDict


class _RequiredOption(TypedDict):
    name: str
    cache: str
    kind: str


class SetupOption(_RequiredOption, total=False):
    """One documented setup option and the CMake cache entry it sets."""

    choices: list[str]


OPTIONS: list[SetupOption] = [
    {"name": "shared", "cache": "BUILD_SHARED_LIBS", "kind": "bool"},
    {
        "name": "cuda-architectures",
        "cache": "CMAKE_CUDA_ARCHITECTURES",
        "kind": "string",
    },
    {
        "name": "compile-commands",
        "cache": "CMAKE_EXPORT_COMPILE_COMMANDS",
        "kind": "bool",
    },
    {"name": "hip-architectures", "cache": "CMAKE_HIP_ARCHITECTURES", "kind": "string"},
    {
        "name": "install-includedir",
        "cache": "CMAKE_INSTALL_INCLUDEDIR",
        "kind": "string",
    },
    {"name": "install-libdir", "cache": "CMAKE_INSTALL_LIBDIR", "kind": "string"},
    {"name": "install-rpath", "cache": "CMAKE_INSTALL_RPATH", "kind": "string"},
    {"name": "prefix-path", "cache": "CMAKE_PREFIX_PATH", "kind": "string"},
    {
        "name": "cache-backend",
        "cache": "SCALARS_CACHE_BACKEND",
        "kind": "enum",
        "choices": ["none", "ccache", "sccache", "buildcache"],
    },
    {
        "name": "cpu-backend",
        "cache": "SCALARS_CPU_BACKEND",
        "kind": "enum",
        "choices": ["no", "sse", "avx", "avx2", "avx512", "neon", "sve"],
    },
    {"name": "cxx-standard", "cache": "SCALARS_CXX_STANDARD", "kind": "string"},
    {"name": "benchmark", "cache": "SCALARS_ENABLE_BENCHMARK", "kind": "bool"},
    {"name": "cache", "cache": "SCALARS_ENABLE_CACHE", "kind": "bool"},
    {"name": "clangtidy", "cache": "SCALARS_ENABLE_CLANGTIDY", "kind": "bool"},
    {"name": "coverage", "cache": "SCALARS_ENABLE_COVERAGE", "kind": "bool"},
    {"name": "fix", "cache": "SCALARS_ENABLE_FIX", "kind": "bool"},
    {"name": "gtest", "cache": "SCALARS_ENABLE_GTEST", "kind": "bool"},
    {"name": "icecc", "cache": "SCALARS_ENABLE_ICECC", "kind": "bool"},
    {"name": "iwyu", "cache": "SCALARS_ENABLE_IWYU", "kind": "bool"},
    {"name": "sanitizer", "cache": "SCALARS_ENABLE_SANITIZER", "kind": "bool"},
    {"name": "spell", "cache": "SCALARS_ENABLE_SPELL", "kind": "bool"},
    {"name": "testing", "cache": "SCALARS_ENABLE_TESTING", "kind": "bool"},
    {"name": "valgrind", "cache": "SCALARS_ENABLE_VALGRIND", "kind": "bool"},
    {
        "name": "gpu-backend",
        "cache": "SCALARS_GPU_BACKEND",
        "kind": "enum",
        "choices": ["none", "cuda", "hip", "metal"],
    },
    {"name": "linker-choice", "cache": "SCALARS_LINKER_CHOICE", "kind": "string"},
    {
        "name": "lto-mode",
        "cache": "SCALARS_LTO_MODE",
        "kind": "enum",
        "choices": ["off", "thin", "full", "ipo", "auto"],
    },
    {
        "name": "math-backend",
        "cache": "SCALARS_MATH_BACKEND",
        "kind": "enum",
        "choices": ["auto", "svml", "sleef", "accelerate", "native"],
    },
    {"name": "sanitizer-types", "cache": "SCALARS_SANITIZER_TYPE", "kind": "string"},
    {"name": "package-dir", "cache": "Scalars_DIR", "kind": "string"},
]
