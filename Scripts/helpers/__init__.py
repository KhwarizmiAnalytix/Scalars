"""
Scalars Build System Helper Modules

Modular helper functions for Scripts/setup.py, split out for readability and
reuse (mirrors the KhwarizmiAnalytix/XSigma Scripts/helpers/ package this
standalone Scalars repo is built alongside).

Modules:
    - config: CMake configure-step invocation
    - build: Build-step invocation (ninja/xcodebuild/msvc)
    - test: ctest / valgrind invocation
    - cppcheck: Static analysis with cppcheck

Coverage analysis is the coverage-tool PyPI package (pip install coverage-tool),
invoked from Scripts/setup.py when the coverage token is set.
"""

__version__ = "0.1.0"
__all__ = ["config", "build", "test", "cppcheck"]
