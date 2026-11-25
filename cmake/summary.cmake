# Print a variable-named summary line, padding the label to 20 characters so
# the colon lands at column 24 like the fixed labels. Names of 20 or more
# characters (e.g. CMAKE_CXX_FLAGS_RELEASE) cannot fit and get one space.
function(_scalars_summary_variable _name)
    string(LENGTH "${_name}" _len)
    set(_pad " ")
    if(_len LESS 20)
        math(EXPR _pad_len "20 - ${_len}")
        string(REPEAT " " ${_pad_len} _pad)
    endif()
    message("  ${_name}${_pad}: ${${_name}}")
endfunction()

if(SCALARS_STANDALONE)
    message("")
    message("******** Scalars module (CMake flags) ********")
    message("  Shared libs         : ${BUILD_SHARED_LIBS}")
    message("  Cxx standard        : ${SCALARS_CXX_STANDARD}")
    message("  Cache               : ${SCALARS_ENABLE_CACHE}")
    message("  Cache backend       : ${SCALARS_CACHE_BACKEND}")
    message("  Icecc               : ${SCALARS_ENABLE_ICECC}")
    scalars_print_lto_linker_summary(SCALARS)
    message("  Coverage            : ${SCALARS_ENABLE_COVERAGE}")
    message("  Testing             : ${SCALARS_ENABLE_TESTING}")
    message("  Gtest               : ${SCALARS_ENABLE_GTEST}")
    message("  Benchmark           : ${SCALARS_ENABLE_BENCHMARK}")
    message("  CTest dashboards    : ${BUILD_TESTING}")
    message("  Clang-tidy          : ${SCALARS_ENABLE_CLANGTIDY}")
    message("  Fix                 : ${SCALARS_ENABLE_FIX}")
    message("  Iwyu                : ${SCALARS_ENABLE_IWYU}")
    message("  Sanitizer           : ${SCALARS_ENABLE_SANITIZER}")
    message("  Sanitizer type      : ${SCALARS_SANITIZER_TYPE}")
    message("  Spell               : ${SCALARS_ENABLE_SPELL}")
    message("  Valgrind            : ${SCALARS_ENABLE_VALGRIND}")
    message("")
    message("SIMD / GPU backend:")
    message("  CPU backend         : ${SCALARS_CPU_BACKEND}")
    message("  SIMD compiler flags : ${SCALARS_SIMD_COMPILER_FLAGS}")
    message("  Packet size         : ${SCALARS_PACKET_SIZE}")
    message("  Math backend        : ${SCALARS_MATH_BACKEND}")
    message("  Math backend used   : ${SCALARS_MATH_BACKEND_RESOLVED}")
    message("  MKL                 : ${SCALARS_ENABLE_MKL}")
    message("  GPU backend         : ${SCALARS_GPU_BACKEND}")
    if(SCALARS_GPU_BACKEND STREQUAL "hip")
        message("  HIP version         : ${SCALARS_HIP_VERSION}")
        message("  HIP architectures   : ${SCALARS_GPU_ARCHITECTURES}")
    elseif(SCALARS_GPU_BACKEND STREQUAL "cuda")
        message("  CUDA architectures  : ${SCALARS_CUDA_ARCHITECTURES}")
        message("  CUDA compiler       : ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")
        list(JOIN SCALARS_CUDA_HOST_SIMD_FLAGS " " _summary_cuda_host_flags)
        message("  CUDA host SIMD flags: ${_summary_cuda_host_flags}")
        unset(_summary_cuda_host_flags)
    endif()
    message("")
    message("Build:")
    message("  Build type          : ${CMAKE_BUILD_TYPE}")
    message("  Generator           : ${CMAKE_GENERATOR}")
    message("  C compiler          : ${CMAKE_C_COMPILER}")
    message("  CXX compiler        : ${CMAKE_CXX_COMPILER}")
    message("  C++ standard        : ${CMAKE_CXX_STANDARD}")
    message("  Export compile cmds : ${CMAKE_EXPORT_COMPILE_COMMANDS}")
    message("  Install prefix      : ${CMAKE_INSTALL_PREFIX}")
    message("  C flags             : ${CMAKE_C_FLAGS}")
    message("  CXX flags           : ${CMAKE_CXX_FLAGS}")
    message("  Exe linker flags    : ${CMAKE_EXE_LINKER_FLAGS}")
    message("  Shared linker flags : ${CMAKE_SHARED_LINKER_FLAGS}")
    message("  Static linker flags : ${CMAKE_STATIC_LINKER_FLAGS}")
    message("  Module linker flags : ${CMAKE_MODULE_LINKER_FLAGS}")
    # Include configuration-specific flags for both single- and multi-config generators.
    set(_summary_configs ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE})
    list(REMOVE_DUPLICATES _summary_configs)
    foreach(_summary_config IN LISTS _summary_configs)
        string(TOUPPER "${_summary_config}" _summary_config_upper)
        foreach(
            _summary_flag IN
            ITEMS CMAKE_C_FLAGS
                  CMAKE_CXX_FLAGS
                  CMAKE_CUDA_FLAGS
                  CMAKE_EXE_LINKER_FLAGS
                  CMAKE_SHARED_LINKER_FLAGS
                  CMAKE_STATIC_LINKER_FLAGS
                  CMAKE_MODULE_LINKER_FLAGS
        )
            set(_summary_var "${_summary_flag}_${_summary_config_upper}")
            if(DEFINED ${_summary_var})
                _scalars_summary_variable(${_summary_var})
            endif()
        endforeach()
    endforeach()
    get_directory_property(
        _summary_compile_defs DIRECTORY ${SCALARS_SOURCE_DIR} COMPILE_DEFINITIONS
    )
    message("  Compile defs        : ${_summary_compile_defs}")
    message("")

    # Optional overrides and tool paths from the cmake/ helpers. Keep unset
    # platform/backend-specific settings out of the summary.
    message("Additional CMake settings (when defined):")
    foreach(
        _summary_var IN
        ITEMS CMAKE_CONFIGURATION_TYPES
              CMAKE_GENERATOR_PLATFORM
              CMAKE_GENERATOR_TOOLSET
              CMAKE_TOOLCHAIN_FILE
              CMAKE_MAKE_PROGRAM
              CMAKE_LINKER
              CMAKE_PREFIX_PATH
              CMAKE_FIND_ROOT_PATH
              CMAKE_INTERPROCEDURAL_OPTIMIZATION
              CMAKE_MSVC_RUNTIME_LIBRARY
              CMAKE_CXX_MP_FLAG
              CMAKE_CXX_MP_NUM_PROCESSORS
              PROJECT_LLVM_INSTALL_PREFIX
              CMAKE_OSX_ARCHITECTURES
              CMAKE_OSX_DEPLOYMENT_TARGET
              CMAKE_OSX_SYSROOT
              CMAKE_CUDA_COMPILER
              CMAKE_CUDA_HOST_COMPILER
              CMAKE_CUDA_STANDARD
              CMAKE_CUDA_ARCHITECTURES
              CMAKE_CUDA_FLAGS
              CMAKE_CUDA_RUNTIME_LIBRARY
              CUDAToolkit_ROOT
              SCALARS_CCACHE_PROGRAM
              SCALARS_SCCACHE_PROGRAM
              SCALARS_BUILDCACHE_PROGRAM
              SCALARS_ICECC_EXECUTABLE
              CLANG_TIDY_PATH
              IWYU_EXECUTABLE
              CODESPELL_EXECUTABLE
              PROJECT_VALGRIND_TIMEOUT_MULTIPLIER
              DART_TESTING_TIMEOUT
              MEMORYCHECK_COMMAND
              MEMORYCHECK_COMMAND_OPTIONS
              MEMORYCHECK_SUPPRESSIONS_FILE
              CMAKE_PDB_OUTPUT_DIRECTORY
    )
        if(DEFINED ${_summary_var})
            _scalars_summary_variable(${_summary_var})
        endif()
    endforeach()
    message("")
    unset(_summary_configs)
    unset(_summary_config)
    unset(_summary_config_upper)
    unset(_summary_flag)
    unset(_summary_var)
    unset(_summary_compile_defs)
endif()
