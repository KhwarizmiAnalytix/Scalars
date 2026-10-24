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
