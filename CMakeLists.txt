cmake_minimum_required(VERSION 3.24)

# Standalone package and add_subdirectory() dependency, following LinearAlgebra.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(
        Scalars
        VERSION 0.1.0
        DESCRIPTION "C++20 scalar types, traits, conversions, and math"
        LANGUAGES CXX C
    )
    set(SCALARS_STANDALONE ON)
    option(BUILD_SHARED_LIBS "Build shared libraries" ON)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
else()
    set(SCALARS_STANDALONE OFF)
endif()

set(SCALARS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(SCALARS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
include(GNUInstallDirs)

# --- Feature flags and toolchain controls ---
include("${SCALARS_SOURCE_DIR}/cmake/lto.cmake")
scalars_lto_compute_default(_lto_default PROFILE SUPPORT)
set(SCALARS_LTO_MODE "${_lto_default}" CACHE STRING
                                             "LTO mode for Scalars: off | thin | full | ipo | auto"
)
set_property(CACHE SCALARS_LTO_MODE PROPERTY STRINGS off thin full ipo auto)
option(SCALARS_ENABLE_COVERAGE "Enable coverage instrumentation for Scalars" OFF)

set(_scalars_testing_default ${SCALARS_STANDALONE})
if(DEFINED BUILD_TESTING)
    set(_scalars_testing_default ${BUILD_TESTING})
endif()
# Keep the original Scalars switch working for existing consumers.
if(DEFINED SCALARS_BUILD_TESTS)
    set(_scalars_testing_default ${SCALARS_BUILD_TESTS})
endif()
option(SCALARS_ENABLE_TESTING "Build Scalars test suite" ${_scalars_testing_default})
option(SCALARS_ENABLE_GTEST "Enable GoogleTest for Scalars" ON)
# Vendored submodule under ThirdParty/benchmark. Off by default, as in LinearAlgebra.
option(SCALARS_ENABLE_BENCHMARK "Enable Google Benchmark for Scalars micro-benchmarks" OFF)
if(DEFINED BUILD_TESTING AND NOT BUILD_TESTING)
    set(SCALARS_ENABLE_TESTING OFF)
endif()

option(SCALARS_ENABLE_ICECC "Use Icecream (icecc) distributed compiler for Scalars" OFF)
option(SCALARS_ENABLE_CACHE "Enable compiler caching for the Scalars target" ON)
option(SCALARS_ENABLE_CLANGTIDY "Enable clang-tidy static analysis for Scalars" OFF)
option(SCALARS_ENABLE_FIX "Enable clang-tidy fix-errors and fix options for Scalars" OFF)
option(SCALARS_ENABLE_IWYU "Enable include-what-you-use analysis for Scalars" OFF)
option(SCALARS_ENABLE_SANITIZER "Build Scalars with sanitizer support (Clang/GCC only)" OFF)
set(SCALARS_SANITIZER_TYPE "address"
    CACHE STRING "Sanitizer type for Scalars. Options: address, undefined, thread, memory, leak"
)
set_property(CACHE SCALARS_SANITIZER_TYPE PROPERTY STRINGS address undefined thread memory leak)
option(SCALARS_ENABLE_SPELL "Enable check-only spell checking for Scalars" OFF)
option(SCALARS_ENABLE_VALGRIND "Execute Scalars test suite with Valgrind" OFF)
set(SCALARS_LINKER_CHOICE "default" CACHE STRING
                                          "Linker for Scalars: default, lld, mold, gold, lld-link"
)
set_property(CACHE SCALARS_LINKER_CHOICE PROPERTY STRINGS default lld mold gold lld-link)
set(SCALARS_CACHE_BACKEND "none"
    CACHE STRING "Compiler cache backend for Scalars. Options: none, ccache, sccache, buildcache"
)
set_property(CACHE SCALARS_CACHE_BACKEND PROPERTY STRINGS none ccache sccache buildcache)
mark_as_advanced(
    SCALARS_ENABLE_ICECC
    SCALARS_ENABLE_CACHE
    SCALARS_ENABLE_CLANGTIDY
    SCALARS_ENABLE_FIX
    SCALARS_ENABLE_IWYU
    SCALARS_ENABLE_SANITIZER
    SCALARS_SANITIZER_TYPE
    SCALARS_ENABLE_SPELL
    SCALARS_ENABLE_VALGRIND
    SCALARS_LINKER_CHOICE
    SCALARS_CACHE_BACKEND
)

# Scalars uses C++20 facilities such as std::bit_cast.
set(SCALARS_CXX_STANDARD "20" CACHE STRING "C++ standard for Scalars. Options: 20 23")
set_property(CACHE SCALARS_CXX_STANDARD PROPERTY STRINGS 20 23)
if(NOT SCALARS_CXX_STANDARD MATCHES "^(20|23)$")
    message(FATAL_ERROR "SCALARS_CXX_STANDARD must be one of: 20 23")
endif()

# --- Toolchain helpers: checks, platform, cache, instrumentation, analysis ---
include(CheckCXXSourceRuns)
include(CheckCXXSymbolExists)
include("${SCALARS_SOURCE_DIR}/cmake/checks.cmake")
include("${SCALARS_SOURCE_DIR}/cmake/build_type.cmake")
include("${SCALARS_SOURCE_DIR}/cmake/platform.cmake")
include("${SCALARS_SOURCE_DIR}/cmake/cache.cmake")
include(CMakeDependentOption)
include("${SCALARS_SOURCE_DIR}/cmake/helper_macros.cmake")
if(SCALARS_ENABLE_SANITIZER)
    include("${SCALARS_SOURCE_DIR}/cmake/sanitize.cmake")
endif()
if(SCALARS_ENABLE_COVERAGE)
    include("${SCALARS_SOURCE_DIR}/cmake/coverage.cmake")
endif()
if(SCALARS_ENABLE_SPELL)
    include("${SCALARS_SOURCE_DIR}/cmake/spell.cmake")
endif()
if(SCALARS_ENABLE_VALGRIND)
    include("${SCALARS_SOURCE_DIR}/cmake/valgrind.cmake")
endif()
if(SCALARS_STANDALONE)
    include(CTest)
endif()
if(SCALARS_ENABLE_TESTING)
    enable_testing()
endif()
if(SCALARS_ENABLE_CLANGTIDY)
    include("${SCALARS_SOURCE_DIR}/cmake/clang_tidy.cmake")
endif()
if(SCALARS_ENABLE_IWYU)
    include("${SCALARS_SOURCE_DIR}/cmake/iwyu.cmake")
endif()

function(scalars_configure_target target)
    scalars_target_apply_cache(${target} SCALARS_ENABLE_CACHE SCALARS_CACHE_BACKEND)
    scalars_target_optional_icecc(${target} SCALARS_ENABLE_ICECC)
    if(SCALARS_ENABLE_SANITIZER)
        scalars_configure_sanitizer(${target} SCALARS)
    endif()
    if(SCALARS_ENABLE_COVERAGE)
        scalars_configure_coverage(${target})
    endif()
    scalars_target_apply_lto(${target} SCALARS)
    scalars_find_linker(${SCALARS_LINKER_CHOICE} ${target})
    scalars_apply_platform_compile_flags(${target})
    if(SCALARS_ENABLE_CLANGTIDY)
        scalars_target_clang_tidy(${target} ${SCALARS_ENABLE_FIX})
    endif()
    if(SCALARS_ENABLE_IWYU)
        scalars_apply_iwyu(${target})
    endif()
endfunction()

# --- Public targets ---
set(CMAKE_CXX_STANDARD ${SCALARS_CXX_STANDARD})
add_library(ScalarsCore INTERFACE)
add_library(Scalars::Core ALIAS ScalarsCore)
target_compile_features(ScalarsCore INTERFACE cxx_std_${SCALARS_CXX_STANDARD})
target_include_directories(
    ScalarsCore INTERFACE $<BUILD_INTERFACE:${SCALARS_SOURCE_DIR}/include>
                          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set_target_properties(ScalarsCore PROPERTIES EXPORT_NAME Core)

add_library(Scalars src/core/scalar_type.cpp src/math/complex_math.cpp)
add_library(Scalars::Scalars ALIAS Scalars)
target_link_libraries(Scalars PUBLIC ScalarsCore)
target_compile_definitions(Scalars PRIVATE SCALARS_BUILD_MAIN_LIB)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(Scalars PUBLIC SCALARS_BUILD_SHARED_LIBS)
endif()
set_target_properties(
    Scalars
    PROPERTIES EXPORT_NAME Scalars
               CXX_STANDARD ${SCALARS_CXX_STANDARD}
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
               WINDOWS_EXPORT_ALL_SYMBOLS ON
)
scalars_configure_target(Scalars)

# --- Installation and find_package(Scalars) support ---
install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
    PATTERN "*.h"
)
install(
    TARGETS Scalars ScalarsCore
    EXPORT ScalarsTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(EXPORT ScalarsTargets NAMESPACE Scalars:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Scalars)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${SCALARS_SOURCE_DIR}/cmake/scalars_config.cmake.in"
    "${SCALARS_BINARY_DIR}/ScalarsConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Scalars
)
write_basic_package_version_file(
    "${SCALARS_BINARY_DIR}/ScalarsConfigVersion.cmake" VERSION 0.1.0 COMPATIBILITY SameMajorVersion
)
install(FILES "${SCALARS_BINARY_DIR}/ScalarsConfig.cmake"
              "${SCALARS_BINARY_DIR}/ScalarsConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Scalars
)

# --- Testing ---
if(SCALARS_ENABLE_TESTING)
    add_subdirectory(Testing/Cxx)
    add_subdirectory(Testing/CMake)
endif()
if(SCALARS_ENABLE_BENCHMARK)
    add_subdirectory(Testing/Benchmark)
endif()
if(SCALARS_ENABLE_VALGRIND AND SCALARS_ENABLE_TESTING)
    scalars_apply_valgrind_timeouts()
endif()

# --- Configuration summary ---
include("${SCALARS_SOURCE_DIR}/cmake/summary.cmake")
