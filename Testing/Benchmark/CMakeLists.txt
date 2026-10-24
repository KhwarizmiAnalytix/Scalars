# Micro-benchmarks, following LinearAlgebra's Testing/Benchmark layout.
include("${SCALARS_SOURCE_DIR}/cmake/scalars_dependencies.cmake")

scalars_setup_benchmark()
if(NOT TARGET Benchmark::benchmark)
    message(FATAL_ERROR "Google Benchmark target unavailable after scalars_setup_benchmark()")
endif()

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
