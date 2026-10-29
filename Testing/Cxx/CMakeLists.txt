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
               AND CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES)
                list(GET CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES 0 _scalars_cuda_lib_dir)
                foreach(_scalars_cuda_lib IN ITEMS cudart cudadevrt)
                    if(EXISTS "${_scalars_cuda_lib_dir}/${_scalars_cuda_lib}.lib")
                        target_link_libraries(
                            ScalarsCxxTests PRIVATE
                            "${_scalars_cuda_lib_dir}/${_scalars_cuda_lib}.lib"
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
        ScalarsCxxTests PROPERTIES FOLDER "Tests" VS_DEBUGGER_WORKING_DIRECTORY
                                                  "$<TARGET_FILE_DIR:ScalarsCxxTests>"
    )
    add_test(NAME ScalarsCxxTests COMMAND ScalarsCxxTests
             WORKING_DIRECTORY "$<TARGET_FILE_DIR:ScalarsCxxTests>"
    )
    set_tests_properties(ScalarsCxxTests PROPERTIES TIMEOUT 120 LABELS "Scalars;Unit")
endif()
