# C++ unit tests, following LinearAlgebra's Testing/Cxx layout.
if(SCALARS_ENABLE_GTEST)
    include("${SCALARS_SOURCE_DIR}/cmake/scalars_dependencies.cmake")
    scalars_setup_gtest()

    configure_file(
        "${SCALARS_SOURCE_DIR}/Testing/scalars_test.h.in"
        "${SCALARS_BINARY_DIR}/Testing/scalars_test.h" @ONLY
    )
    file(GLOB_RECURSE test_sources CONFIGURE_DEPENDS LIST_DIRECTORIES false
         "${CMAKE_CURRENT_SOURCE_DIR}/Test*.cxx"
    )

    # TestGpuSimd.cu exercises backend/gpu/{float,double,reduced}/simd.h directly (no tensor<T>
    # expression templates, so none of Vectorization's ODR cross-TU tagging applies here) --
    # only compiled in when a GPU backend is actually active.
    set(_scalars_gpu_test_sources "")
    if(SCALARS_GPU_BACKEND STREQUAL "cuda" OR SCALARS_GPU_BACKEND STREQUAL "hip")
        file(GLOB_RECURSE _scalars_gpu_test_sources CONFIGURE_DEPENDS LIST_DIRECTORIES false
             "${CMAKE_CURRENT_SOURCE_DIR}/Test*.cu"
        )
        list(APPEND test_sources ${_scalars_gpu_test_sources})
    endif()

    add_executable(ScalarsCxxTests ${test_sources})
    target_include_directories(
        ScalarsCxxTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_CURRENT_BINARY_DIR}/.."
    )
    target_compile_definitions(
        ScalarsCxxTests PRIVATE SCALARS_HAS_GTEST=1 SCALARS_GOOGLE_TEST USE_GTEST _VARIADIC_MAX=10
    )
    target_link_libraries(ScalarsCxxTests PRIVATE Gtest::gtest Gtest::gtest_main Scalars::Scalars)

    if(_scalars_gpu_test_sources)
        if(SCALARS_GPU_BACKEND STREQUAL "cuda")
            set_target_properties(
                ScalarsCxxTests
                PROPERTIES CUDA_STANDARD ${SCALARS_CXX_STANDARD} CUDA_STANDARD_REQUIRED ON
                           CUDA_ARCHITECTURES "${SCALARS_CUDA_ARCHITECTURES}"
            )
            # A __host__ __device__ function that calls a host-only (or host-only
            # constexpr) function is only a warning in nvcc, and the device call
            # then silently does nothing. Make it an error so device paths that
            # lose their SCALARS_HOST_DEVICE annotation fail the build.
            target_compile_options(
                ScalarsCxxTests
                PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--diag-error=20011,20013,20014,20015>
            )
            if(MSVC)
                # See cmake/gpu_backend.cmake: cl.exe's traditional preprocessor
                # trips a hard error in CCCL's headers under nvcc.
                target_compile_options(
                    ScalarsCxxTests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
                )
            endif()
            # See cmake/gpu_backend.cmake: CMAKE_CUDA_RUNTIME_LIBRARY=None is set for
            # WIN32+GNU to avoid MSVC-style link flags. Add the runtime explicitly here.
            if(WIN32 AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"
               AND CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES
            )
                list(GET CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES 0 _scalars_cuda_lib_dir)
                foreach(_scalars_cuda_lib IN ITEMS cudart cudadevrt)
                    if(EXISTS "${_scalars_cuda_lib_dir}/${_scalars_cuda_lib}.lib")
                        target_link_libraries(
                            ScalarsCxxTests
                            PRIVATE "${_scalars_cuda_lib_dir}/${_scalars_cuda_lib}.lib"
                        )
                    endif()
                endforeach()
                unset(_scalars_cuda_lib_dir)
            endif()
        elseif(SCALARS_GPU_BACKEND STREQUAL "hip")
            set_source_files_properties(${_scalars_gpu_test_sources} PROPERTIES LANGUAGE HIP)
            set_target_properties(
                ScalarsCxxTests PROPERTIES HIP_STANDARD ${SCALARS_CXX_STANDARD}
                                           HIP_ARCHITECTURES "${SCALARS_GPU_ARCHITECTURES}"
            )
        endif()
        message(STATUS "ScalarsCxxTests: GPU simd tests enabled (backend: ${SCALARS_GPU_BACKEND})")
    endif()
    unset(_scalars_gpu_test_sources)

    scalars_configure_target(ScalarsCxxTests)
    set_target_properties(
        ScalarsCxxTests
        PROPERTIES FOLDER "Tests"
                   VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:ScalarsCxxTests>"
                   CXX_STANDARD ${SCALARS_CXX_STANDARD}
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF
    )
    add_test(NAME ScalarsCxxTests COMMAND ScalarsCxxTests
             WORKING_DIRECTORY "$<TARGET_FILE_DIR:ScalarsCxxTests>"
    )
    set_tests_properties(ScalarsCxxTests PROPERTIES TIMEOUT 120 LABELS "Scalars;Unit")

    # Each generated source starts with the header under test, with no umbrella
    # header or precompiled header supplying missing declarations for it.
    set(_scalars_header_check_sources "")
    function(scalars_add_simd_header_check name)
        set(source "${CMAKE_CURRENT_BINARY_DIR}/header_checks/${name}.cxx")
        set(content "")
        foreach(header IN LISTS ARGN)
            string(APPEND content "#include \"${header}\"\n")
        endforeach()
        file(GENERATE OUTPUT "${source}" CONTENT "${content}")
        set(_scalars_header_check_sources ${_scalars_header_check_sources} "${source}" PARENT_SCOPE)
    endfunction()

    set(_scalars_headers backend/simd_base.h backend/simd.h backend/cpu/real_simd.h
                         backend/cpu/complex_simd.h backend/cpu/detail/complex_simd.h
    )
    if(SCALARS_VECTORIZED)
        list(APPEND _scalars_headers backend/cpu/reduced/convert.h backend/cpu/reduced/simd.h
             backend/cpu/reduced/complex_simd.h
        )
        set(_scalars_header_backend "${SCALARS_CPU_BACKEND}")
        if(_scalars_header_backend STREQUAL "avx2")
            set(_scalars_header_backend avx)
        endif()
        foreach(kind float double complex)
            list(APPEND _scalars_headers "backend/cpu/${_scalars_header_backend}/${kind}/simd.h")
        endforeach()
    endif()
    foreach(header IN LISTS _scalars_headers)
        string(MAKE_C_IDENTIFIER "${header}" name)
        scalars_add_simd_header_check("single_${name}" "${header}" "${header}")
        if(NOT header STREQUAL "backend/simd.h")
            scalars_add_simd_header_check("before_${name}" "${header}" backend/simd.h)
            scalars_add_simd_header_check("after_${name}" backend/simd.h "${header}")
        endif()
    endforeach()
    scalars_add_simd_header_check(all_headers ${_scalars_headers})
    list(REVERSE _scalars_headers)
    scalars_add_simd_header_check(all_headers_reversed ${_scalars_headers})

    if(SCALARS_VECTORIZED)
        # Exercise all six permutations of the headers whose dependencies
        # previously required callers to arrange their includes manually.
        foreach(first convert simd complex_simd)
            foreach(second convert simd complex_simd)
                foreach(third convert simd complex_simd)
                    if(NOT first STREQUAL second AND NOT first STREQUAL third AND NOT second
                                                                                  STREQUAL third
                    )
                        scalars_add_simd_header_check(
                            "reduced_${first}_${second}_${third}" "backend/cpu/reduced/${first}.h"
                            "backend/cpu/reduced/${second}.h" "backend/cpu/reduced/${third}.h"
                        )
                    endif()
                endforeach()
            endforeach()
        endforeach()
    endif()
    add_library(ScalarsSimdHeaderChecks OBJECT ${_scalars_header_check_sources})
    target_link_libraries(ScalarsSimdHeaderChecks PRIVATE Scalars::Scalars)
    set_target_properties(
        ScalarsSimdHeaderChecks
        PROPERTIES DISABLE_PRECOMPILE_HEADERS ON
                   UNITY_BUILD OFF
                   CXX_STANDARD ${SCALARS_CXX_STANDARD}
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF
    )
    # Use the public consumer requirements only. In particular, do not apply
    # scalars_configure_target's forced <cstdlib> include to these probes.
    if(SCALARS_ENABLE_CLANGTIDY)
        scalars_target_clang_tidy(ScalarsSimdHeaderChecks ${SCALARS_ENABLE_FIX})
    endif()
endif()
