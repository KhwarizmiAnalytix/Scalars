"""Regression checks for the Scalars build helper."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[2] / "Scripts"
SPEC = importlib.util.spec_from_file_location("scalars_setup", SCRIPTS / "setup.py")
setup = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    SPEC.loader.exec_module(setup)


class setup_tests(unittest.TestCase):
    def flags(self, *args):
        flags = setup.ScalarsFlags(setup.parse_args(list(args)))
        command = []
        build_type = flags.create_cmake_flags(command, "Release", "Darwin")
        values = dict(arg[2:].split("=", 1) for arg in command)
        return flags, values, build_type

    def test_defaults_match_cmake(self):
        _, values, build_type = self.flags("config.build.test")
        self.assertEqual(values["BUILD_SHARED_LIBS"], "ON")
        self.assertEqual(values["SCALARS_ENABLE_TESTING"], "ON")
        self.assertEqual(values["SCALARS_ENABLE_GTEST"], "ON")
        self.assertEqual(values["SCALARS_ENABLE_CACHE"], "ON")
        self.assertEqual(values["SCALARS_CACHE_BACKEND"], "none")
        self.assertEqual(build_type, "Release")
        cmake = (SCRIPTS.parent / "CMakeLists.txt").read_text()
        for name in values:
            self.assertIn(name, cmake)

    def test_static_cxx23_and_lto(self):
        flags, values, _ = self.flags("config.build.test.static.cxx23", "--lto.ipo")
        self.assertEqual(values["BUILD_SHARED_LIBS"], "OFF")
        self.assertEqual(values["SCALARS_CXX_STANDARD"], "23")
        self.assertEqual(values["SCALARS_LTO_MODE"], "ipo")
        self.assertIn("_static", flags.builder_suffix)
        self.assertIn("_cxx23", flags.builder_suffix)

    def test_sanitizer_enables_debug(self):
        _, values, build_type = self.flags("config.build.test", "--sanitizer.undefined")
        self.assertEqual(values["SCALARS_ENABLE_SANITIZER"], "ON")
        self.assertEqual(values["SCALARS_SANITIZER_TYPE"], "undefined")
        self.assertEqual(build_type, "Debug")

    def test_linker_default_and_cache_toggle(self):
        _, values, _ = self.flags("config.cache", "--linker.default")
        self.assertEqual(values["SCALARS_LINKER_CHOICE"], "default")
        self.assertEqual(values["SCALARS_ENABLE_CACHE"], "OFF")

    def test_gtest_toggle(self):
        _, values, _ = self.flags("config.gtest")
        self.assertEqual(values["SCALARS_ENABLE_GTEST"], "OFF")

    def test_benchmark_toggle(self):
        _, values, _ = self.flags("config.benchmark")
        self.assertEqual(values["SCALARS_ENABLE_BENCHMARK"], "ON")

    def test_cpu_backend_defaults_to_empty(self):
        _, values, _ = self.flags("config.build.test")
        self.assertNotIn("SCALARS_CPU_BACKEND", values)
        self.assertNotIn("SCALARS_GPU_BACKEND", values)

    def test_cpu_and_gpu_backend_tokens(self):
        _, values, _ = self.flags("config.avx2.metal")
        self.assertEqual(values["SCALARS_CPU_BACKEND"], "avx2")
        self.assertEqual(values["SCALARS_GPU_BACKEND"], "metal")
        cmake = (SCRIPTS.parent / "cmake" / "simd.cmake").read_text()
        self.assertIn("SCALARS_CPU_BACKEND", cmake)
        cmake = (SCRIPTS.parent / "cmake" / "gpu_backend.cmake").read_text()
        self.assertIn("SCALARS_GPU_BACKEND", cmake)

    def test_cpu_and_gpu_backend_equals_form(self):
        flags, values, _ = self.flags(
            "config", "--cpu_backend=sve", "--gpu-backend=hip"
        )
        self.assertEqual(values["SCALARS_CPU_BACKEND"], "sve")
        self.assertEqual(values["SCALARS_GPU_BACKEND"], "hip")
        self.assertIn("_sve", flags.builder_suffix)
        self.assertIn("_hip", flags.builder_suffix)

    def test_invalid_options_fail(self):
        for arg in [
            "mkl",
            "cxx17",
            "c++17",
            "--lto.invalid",
            "--sanitizer.invalid",
            "--cpu_backend=avx256",
            "--gpu_backend=opencl",
        ]:
            with self.subTest(arg=arg), self.assertRaises(SystemExit) as error:
                self.flags("config", arg)
            self.assertEqual(error.exception.code, 1)

    def test_compiler_paths_preserve_spaces(self):
        compiler = "/tmp/compiler tools/clang++"
        self.assertEqual(setup.parse_args([compiler]), [compiler])
        with patch.object(setup, "check_dependencies", return_value=[]):
            config = setup.ScalarsConfiguration(["config", compiler])
        values = config._ScalarsConfiguration__value
        self.assertEqual(
            values["cmake_cxx_compiler"], f"-DCMAKE_CXX_COMPILER={compiler}"
        )
        self.assertEqual(
            values["cmake_c_compiler"], "-DCMAKE_C_COMPILER=/tmp/compiler tools/clang"
        )

    def test_configure_preserves_existing_build_and_uses_script_location(self):
        original_dir = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project with spaces"
            (root / "Scripts").mkdir(parents=True)
            with (
                patch.object(setup, "__file__", str(root / "Scripts/setup.py")),
                patch.object(setup, "check_dependencies", return_value=[]),
            ):
                config = setup.ScalarsConfiguration(["config", "static"])
                build = root / "build_ninja_static"
                build.mkdir()
                sentinel = build / "keep.txt"
                sentinel.write_text("preserve existing build data")
                try:
                    self.assertEqual(
                        Path(config.move_to_build_folder()), build.resolve()
                    )
                    self.assertEqual(
                        sentinel.read_text(), "preserve existing build data"
                    )
                finally:
                    os.chdir(original_dir)

    def test_configure_passes_paths_as_separate_arguments(self):
        with patch.object(setup.config_helper.subprocess, "check_call") as run:
            result = setup.config_helper.configure_build(
                "/tmp/source folder",
                "/tmp/build folder",
                "Ninja",
                "",
                "",
                [],
                "--loglevel=NOTICE",
                False,
            )
        self.assertEqual(result, 0)
        self.assertEqual(
            run.call_args.args[0][:5],
            ["cmake", "-S", "/tmp/source folder", "-B", "/tmp/build folder"],
        )

    def test_test_failure_propagates(self):
        with patch.object(
            setup.test_helper.subprocess,
            "check_call",
            side_effect=subprocess.CalledProcessError(1, "ctest"),
        ):
            self.assertEqual(
                setup.test_helper.run_ctest("ninja", "Release", "Darwin", "", False), 1
            )

    def test_analysis_failure_propagates(self):
        for stage in ["cppcheck", "coverage", "test"]:
            with self.subTest(stage=stage):
                config = Mock(source_path=SCRIPTS.parent)
                config.cppcheck.return_value = 0
                config.coverage.return_value = 0
                config.test.return_value = 0
                getattr(config, stage).return_value = 1
                with (
                    patch.object(setup, "ScalarsConfiguration", return_value=config),
                    patch.object(sys, "argv", ["setup.py", "config.build.test"]),
                    self.assertRaises(SystemExit) as error,
                ):
                    setup.main()
                self.assertEqual(error.exception.code, 1)

    def test_help_without_optional_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-S", str(SCRIPTS / "setup.py"), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Scalars Build Configuration Helper", result.stdout)


if __name__ == "__main__":
    unittest.main()
