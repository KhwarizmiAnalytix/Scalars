# Micro-benchmarks, following LinearAlgebra's Testing/Benchmark layout.
include("${SCALARS_SOURCE_DIR}/cmake/scalars_dependencies.cmake")

scalars_setup_benchmark()
if(NOT TARGET Benchmark::benchmark)
    message(FATAL_ERROR "Google Benchmark target unavailable after scalars_setup_benchmark()")
endif()

# Eigen headers for BenchmarkVectorMath's comparison: an Eigen source or include
# directory (the one holding Eigen/Core), else an installed Eigen3 package.
# Without either, the benchmark measures Scalars alone.
set(SCALARS_BENCHMARK_EIGEN_DIR ""
    CACHE PATH "Directory containing Eigen/Core for the vector math benchmark (optional)"
)
set(_scalars_bench_eigen "")
if(SCALARS_BENCHMARK_EIGEN_DIR)
    if(NOT EXISTS "${SCALARS_BENCHMARK_EIGEN_DIR}/Eigen/Core")
        message(FATAL_ERROR "SCALARS_BENCHMARK_EIGEN_DIR has no Eigen/Core: "
                            "${SCALARS_BENCHMARK_EIGEN_DIR}"
        )
    endif()
    set(_scalars_bench_eigen "${SCALARS_BENCHMARK_EIGEN_DIR}")
else()
    find_package(Eigen3 CONFIG QUIET)
    if(TARGET Eigen3::Eigen)
        set(_scalars_bench_eigen "Eigen3::Eigen")
    endif()
endif()

# Adds Eigen and SCALARS_BENCHMARK_HAS_EIGEN to <target> when Eigen was found.
function(scalars_bench_use_eigen target)
    if(NOT _scalars_bench_eigen)
        return()
    endif()
    if(TARGET ${_scalars_bench_eigen})
        target_link_libraries(${target} PRIVATE ${_scalars_bench_eigen})
    else()
        target_include_directories(${target} SYSTEM PRIVATE "${_scalars_bench_eigen}")
    endif()
    target_compile_definitions(${target} PRIVATE SCALARS_BENCHMARK_HAS_EIGEN=1)
    message(STATUS "Scalars: ${target} compares against Eigen (${_scalars_bench_eigen})")
endfunction()

file(GLOB_RECURSE bench_sources CONFIGURE_DEPENDS LIST_DIRECTORIES false
     "${CMAKE_CURRENT_SOURCE_DIR}/Benchmark*.cxx"
)

foreach(_bench_source IN ITEMS ${bench_sources})
    get_filename_component(bench_name "${_bench_source}" NAME_WE)
    string(REPLACE "Benchmark" "" bench_suffix "${bench_name}")
    string(TOLOWER "${bench_suffix}" bench_suffix_lower)
    set(target_name "benchmark_scalars_${bench_suffix_lower}")

    add_executable(${target_name} "${_bench_source}")
    target_include_directories(${target_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
    target_link_libraries(
        ${target_name} PRIVATE Benchmark::benchmark Benchmark::benchmark_main Scalars::Scalars
    )
    scalars_configure_target(${target_name})
    if(bench_name STREQUAL "BenchmarkVectorMath")
        scalars_bench_use_eigen(${target_name})
    endif()
    set_target_properties(
        ${target_name} PROPERTIES FOLDER "Benchmarks" VS_DEBUGGER_WORKING_DIRECTORY
                                                      "$<TARGET_FILE_DIR:${target_name}>"
    )

    if(SCALARS_ENABLE_TESTING)
        add_test(NAME ${target_name} COMMAND ${target_name} --benchmark_min_time=0.01s
                 WORKING_DIRECTORY "$<TARGET_FILE_DIR:${target_name}>"
        )
        set_tests_properties(${target_name} PROPERTIES TIMEOUT 120 LABELS "Scalars;Benchmark")
    endif()
endforeach()

# Error measurement of the vector math benchmark's functions. It runs for about a
# minute, so it is a tool rather than a test.
add_executable(accuracy_scalars_vectormath "${CMAKE_CURRENT_SOURCE_DIR}/AccuracyVectorMath.cxx")
target_include_directories(accuracy_scalars_vectormath PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(accuracy_scalars_vectormath PRIVATE Scalars::Scalars)
scalars_configure_target(accuracy_scalars_vectormath)
scalars_bench_use_eigen(accuracy_scalars_vectormath)
set_target_properties(accuracy_scalars_vectormath PROPERTIES FOLDER "Benchmarks")
