# GoogleTest and Google Benchmark resolution, following LinearAlgebra.
include_guard(GLOBAL)

set(SCALARS_THIRD_PARTY_DIR "${SCALARS_SOURCE_DIR}/ThirdParty"
    CACHE PATH "Root of Scalars' bundled third-party sources"
)

# Bundled projects can flip BUILD_TESTING. Restore the caller's cache entry
# after add_subdirectory, matching LinearAlgebra.
function(scalars_add_dependency_subdirectory source_dir binary_dir)
    if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT
                                                            STREQUAL "GNU"
    )
        set(MSVC FALSE)
    endif()
    get_property(
        _had_testing
        CACHE BUILD_TESTING
        PROPERTY TYPE
        SET
    )
    if(_had_testing)
        get_property(_testing_value CACHE BUILD_TESTING PROPERTY VALUE)
        get_property(_testing_help CACHE BUILD_TESTING PROPERTY HELPSTRING)
        get_property(_testing_type CACHE BUILD_TESTING PROPERTY TYPE)
    endif()
    add_subdirectory("${source_dir}" "${binary_dir}" ${ARGN})
    if(_had_testing)
        set(BUILD_TESTING "${_testing_value}" CACHE ${_testing_type} "${_testing_help}" FORCE)
    else()
        unset(BUILD_TESTING CACHE)
    endif()
endfunction()

function(scalars_setup_gtest)
    if(TARGET gtest_main OR TARGET GTest::gtest_main)
        # Fall through to alias normalization below.
    elseif(COMMAND xsigma_add_googletest)
        xsigma_add_googletest()
    elseif(EXISTS "${SCALARS_THIRD_PARTY_DIR}/googletest/CMakeLists.txt")
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        scalars_add_dependency_subdirectory(
            "${SCALARS_THIRD_PARTY_DIR}/googletest"
            "${SCALARS_BINARY_DIR}/ThirdParty/googletest_build" EXCLUDE_FROM_ALL
        )
    else()
        include(FetchContent)
        FetchContent_Declare(
            googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.18.0
        )
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT
                                                                STREQUAL "GNU"
        )
            set(MSVC FALSE)
        endif()
        FetchContent_MakeAvailable(googletest)
    endif()
    if(TARGET gtest AND NOT TARGET GTest::gtest)
        add_library(GTest::gtest ALIAS gtest)
    endif()
    if(TARGET gtest_main AND NOT TARGET GTest::gtest_main)
        add_library(GTest::gtest_main ALIAS gtest_main)
    endif()
    foreach(_name IN ITEMS gtest gtest_main)
        if(TARGET GTest::${_name})
            set(_target GTest::${_name})
        elseif(TARGET ${_name})
            set(_target ${_name})
        else()
            message(FATAL_ERROR "GoogleTest must provide ${_name}")
        endif()
        if(NOT TARGET Gtest::${_name})
            add_library(Gtest::${_name} INTERFACE IMPORTED)
            set_target_properties(Gtest::${_name} PROPERTIES INTERFACE_LINK_LIBRARIES "${_target}")
        endif()
    endforeach()
endfunction()

function(scalars_setup_sleef)
    if(NOT TARGET sleef AND NOT TARGET Sleef::sleef)
        set(_sleef_src "${SCALARS_THIRD_PARTY_DIR}/sleef")
        if(NOT EXISTS "${_sleef_src}/CMakeLists.txt")
            message(WARNING
                "SCALARS_ENABLE_SLEEF=ON but ${_sleef_src} is missing (run "
                "'git submodule update --init -- ThirdParty/sleef'). Disabling SLEEF."
            )
            set(SCALARS_ENABLE_SLEEF OFF
                CACHE BOOL
                "Enable SLEEF SIMD math library (SSE/AVX/AVX2/AVX512/NEON); mutually exclusive with SVML"
                FORCE
            )
            return()
        endif()

        # SLEEF's install() rules read CMAKE_INSTALL_PREFIX unconditionally, even for an
        # EXCLUDE_FROM_ALL subdirectory build, so it requires a non-default prefix.
        if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
            set(CMAKE_INSTALL_PREFIX "${SCALARS_BINARY_DIR}/ThirdParty/install"
                CACHE PATH "Install path prefix, prepended onto install directories." FORCE
            )
            set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT FALSE)
        endif()

        set(SLEEF_BUILD_TESTS OFF CACHE BOOL "Disable SLEEF tests" FORCE)
        set(SLEEF_BUILD_DFT OFF CACHE BOOL "Disable SLEEF DFT library" FORCE)
        set(SLEEF_BUILD_SCALAR_LIB OFF CACHE BOOL "Disable SLEEF scalar library" FORCE)
        set(SLEEF_BUILD_QUAD OFF CACHE BOOL "Disable SLEEF quad library" FORCE)
        set(SLEEF_BUILD_GNUABI_LIBS OFF CACHE BOOL "Disable SLEEF GNU ABI variants" FORCE)
        set(SLEEF_DISABLE_OPENMP ON CACHE BOOL "Disable SLEEF OpenMP usage" FORCE)
        set(SLEEF_BUILD_INLINE_HEADERS OFF CACHE BOOL "Disable SLEEF inline headers" FORCE)
        set(SLEEF_ENABLE_TESTER OFF CACHE BOOL "Disable SLEEF tester binaries" FORCE)
        set(SLEEF_ENABLE_TESTER4 OFF CACHE BOOL "Disable SLEEF tester4 binaries" FORCE)
        # SLEEF 3.x gates OpenSSL behind SLEEF_DISABLE_SSL (the old SLEEF_ENABLE_SSL name no
        # longer exists). Without this, a system OpenSSL makes src/common register
        # test_psha/test_psha_capi in CTest even though SLEEF_BUILD_TESTS is OFF -- those
        # phantom third-party tests then fail as "Not Run" in Scalars' own ctest run.
        set(SLEEF_DISABLE_SSL ON CACHE BOOL "Disable SLEEF SSL-backed tester checks" FORCE)
        set(SLEEF_SHOW_CONFIG OFF CACHE BOOL "Suppress SLEEF config output" FORCE)
        set(SLEEF_BUILD_SHARED_LIBS OFF CACHE BOOL "Build SLEEF as a static library" FORCE)

        scalars_add_dependency_subdirectory(
            "${_sleef_src}" "${SCALARS_BINARY_DIR}/ThirdParty/sleef_build" EXCLUDE_FROM_ALL
        )
    endif()

    if(TARGET sleef AND NOT TARGET Sleef::sleef)
        add_library(Sleef::sleef ALIAS sleef)
        # SLEEF's own CMakeLists.txt defaults sleef_BINARY_DIR to CMAKE_BINARY_DIR when built
        # as a subdirectory (see ThirdParty/sleef/CMakeLists.txt), so the generated sleef.h
        # lands at <top-level-build-dir>/include/sleef.h, not under the sleef_build tree.
        # sleef.h also checks SLEEF_STATIC_LIBS to suppress __declspec(dllimport) on Windows;
        # the generated sleef-config.h defines it but sleef.h never includes that header, so
        # propagate the definition explicitly to all consumers.
        # Wrapped in $<BUILD_INTERFACE:...> because SLEEF's own CMakeLists.txt installs
        # and exports the "sleef" target; CMake rejects a raw build-tree absolute path in
        # an exported target's INTERFACE_INCLUDE_DIRECTORIES.
        target_include_directories(sleef INTERFACE "$<BUILD_INTERFACE:${SCALARS_BINARY_DIR}/include>")
        target_compile_definitions(sleef INTERFACE SLEEF_STATIC_LIBS)
    endif()
    if(NOT TARGET Sleef::sleef)
        message(FATAL_ERROR "SLEEF must provide a 'sleef' target")
    endif()
endfunction()

function(scalars_setup_benchmark)
    if(NOT TARGET benchmark AND NOT TARGET benchmark::benchmark)
        set(_bench_src "")
        if(EXISTS "${SCALARS_THIRD_PARTY_DIR}/benchmark/CMakeLists.txt")
            set(_bench_src "${SCALARS_THIRD_PARTY_DIR}/benchmark")
        endif()
        if(NOT DEFINED HAVE_STD_REGEX)
            set(HAVE_STD_REGEX 1 CACHE INTERNAL "")
        endif()
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests" FORCE)
        set(BENCHMARK_ENABLE_EXCEPTIONS ON CACHE BOOL "Enable benchmark exceptions" FORCE)
        set(BENCHMARK_ENABLE_LTO OFF CACHE BOOL "Disable benchmark LTO" FORCE)
        set(BENCHMARK_USE_LIBCXX OFF CACHE BOOL "Disable benchmark libcxx" FORCE)
        set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "Disable benchmark werror" FORCE)
        set(BENCHMARK_FORCE_WERROR OFF CACHE BOOL "Disable benchmark force werror" FORCE)
        set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install" FORCE)
        set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "Disable benchmark docs install" FORCE)
        set(BENCHMARK_ENABLE_DOXYGEN OFF CACHE BOOL "Disable benchmark doxygen" FORCE)
        set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable benchmark gtest tests" FORCE)
        set(BENCHMARK_USE_BUNDLED_GTEST OFF CACHE BOOL "Don't use bundled gtest for benchmark"
                                                  FORCE
        )
        set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "Don't download dependencies" FORCE)
        if(_bench_src)
            scalars_add_dependency_subdirectory(
                "${_bench_src}" "${SCALARS_BINARY_DIR}/ThirdParty/benchmark_build" EXCLUDE_FROM_ALL
            )
        else()
            include(FetchContent)
            FetchContent_Declare(
                benchmark GIT_REPOSITORY https://github.com/google/benchmark.git GIT_TAG v1.9.5
            )
            FetchContent_MakeAvailable(benchmark)
        endif()
    endif()
    if(TARGET benchmark AND NOT TARGET benchmark::benchmark)
        add_library(benchmark::benchmark ALIAS benchmark)
    endif()
    if(TARGET benchmark_main AND NOT TARGET benchmark::benchmark_main)
        add_library(benchmark::benchmark_main ALIAS benchmark_main)
    endif()
    foreach(_name IN ITEMS benchmark benchmark_main)
        if(TARGET benchmark::${_name})
            set(_target benchmark::${_name})
        elseif(TARGET ${_name})
            set(_target ${_name})
        else()
            message(FATAL_ERROR "Google Benchmark must provide ${_name}")
        endif()
        if(NOT TARGET Benchmark::${_name})
            add_library(Benchmark::${_name} INTERFACE IMPORTED)
            set_target_properties(
                Benchmark::${_name} PROPERTIES INTERFACE_LINK_LIBRARIES "${_target}"
            )
        endif()
    endforeach()
endfunction()
