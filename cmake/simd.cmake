# CPU SIMD backend selection for Scalars. Compiler probes live in utils.cmake,
# ported from Vectorization's include/Cmake/utils.cmake.
#
# SCALARS_CPU_BACKEND (cache: no|sse|avx|avx2|avx512|neon|sve) selects the CPU SIMD
# backend used by include/backend/cpu/*. C++ preprocessor defines:
#   sse -> SCALARS_HAS_SSE=1; avx/avx2 -> SCALARS_HAS_AVX=1; avx2 -> SCALARS_HAS_AVX2=1;
#   avx512 -> SCALARS_HAS_AVX512=1; neon -> SCALARS_HAS_NEON=1; sve -> SCALARS_HAS_SVE=1.
# SCALARS_CPU_CAPABILITY_AVX2 / SCALARS_CPU_CAPABILITY_AVX512 are defined (ifdef-style,
# no value) exactly when backend is avx2 / avx512 -- these are what
# include/types/half.h and include/detail/vec_half.h test via #if defined(...).
include_guard(GLOBAL)

# Host-aware default: x86 family -> avx2, aarch64 -> neon, anything else -> no.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _scalars_simd_proc)
if(_scalars_simd_proc MATCHES "^(x86_64|amd64|i686|i386)$")
    set(_scalars_simd_is_x86 TRUE)
    set(_scalars_simd_is_aarch64 FALSE)
    set(_scalars_cpu_backend_default "avx2")
elseif(_scalars_simd_proc MATCHES "^(aarch64|arm64)$")
    set(_scalars_simd_is_x86 FALSE)
    set(_scalars_simd_is_aarch64 TRUE)
    set(_scalars_cpu_backend_default "neon")
else()
    set(_scalars_simd_is_x86 FALSE)
    set(_scalars_simd_is_aarch64 FALSE)
    set(_scalars_cpu_backend_default "no")
endif()

set(SCALARS_CPU_BACKEND "${_scalars_cpu_backend_default}"
    CACHE STRING "CPU SIMD backend for Scalars. Options: no, sse, avx, avx2, avx512, neon, sve"
)
set_property(CACHE SCALARS_CPU_BACKEND PROPERTY STRINGS no sse avx avx2 avx512 neon sve)
if(NOT SCALARS_CPU_BACKEND MATCHES "^(no|sse|avx|avx2|avx512|neon|sve)$")
    message(FATAL_ERROR
        "SCALARS_CPU_BACKEND must be one of: no, sse, avx, avx2, avx512, neon, sve "
        "(got '${SCALARS_CPU_BACKEND}')"
    )
endif()

option(SCALARS_ENABLE_SVML "Enable Intel SVML short vector math library" OFF)
mark_as_advanced(SCALARS_ENABLE_SVML)
option(SCALARS_ENABLE_MKL "Enable Intel MKL VML vector math backend (x86 only, static link)" OFF)
mark_as_advanced(SCALARS_ENABLE_MKL)
option(SCALARS_ENABLE_SLEEF
       "Enable SLEEF SIMD math library (SSE/AVX/AVX2/AVX512/NEON); mutually exclusive with SVML"
       OFF
)
mark_as_advanced(SCALARS_ENABLE_SLEEF)
option(SCALARS_ENABLE_ACCELERATE "Enable Apple Accelerate vForce on AArch64 NEON" OFF)
mark_as_advanced(SCALARS_ENABLE_ACCELERATE)
option(USE_NATIVE_ARCH "Prefer host CPU instruction tuning (-march=native, Clang/GCC only)" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/packages")

set(SCALARS_HAS_SVML 0)
set(SCALARS_HAS_MKL 0)
if(SCALARS_ENABLE_MKL)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _scalars_mkl_proc)
    if(NOT _scalars_mkl_proc MATCHES "^(x86_64|amd64|i686|i386)$")
        message(WARNING "SCALARS_ENABLE_MKL=ON is only supported on x86_64; disabling.")
        set(SCALARS_ENABLE_MKL OFF CACHE BOOL "Enable Intel MKL VML vector math backend (x86 only, static link)" FORCE)
    else()
        set(MKL_THREADING "SEQ" CACHE STRING "MKL threading flavour used by Scalars" FORCE)
        set(ENABLE_STATIC_MKL OFF CACHE BOOL "Link MKL statically for Scalars" FORCE)
        find_package(MKL QUIET)
        if(MKL_FOUND)
            set(SCALARS_HAS_MKL 1)
            message(STATUS "Scalars: MKL VML found (version ${MKL_VERSION}). Include: ${MKL_INCLUDE_DIR}")
            if(NOT TARGET MKL::MKL)
                add_library(MKL::MKL INTERFACE IMPORTED GLOBAL)
                target_include_directories(MKL::MKL INTERFACE "${MKL_INCLUDE_DIR}")
                target_link_libraries(MKL::MKL INTERFACE ${MKL_LIBRARIES} ${CMAKE_DL_LIBS})
            endif()
        else()
            message(WARNING
                "SCALARS_ENABLE_MKL=ON but MKL was not found -- disabling MKL VML backend. "
                "Set MKLROOT or INTEL_MKL_DIR to your MKL installation root."
            )
            set(SCALARS_ENABLE_MKL OFF CACHE BOOL "Enable Intel MKL VML vector math backend (x86 only, static link)" FORCE)
        endif()
    endif()
    unset(_scalars_mkl_proc)
endif()

# SVE vector register width. include/backend/cpu/sve/* only supports 128 bits.
# Set before utils.cmake, which passes it to -msve-vector-bits.
set(SCALARS_SVE_VECTOR_BITS "128" CACHE STRING "SVE vector register width in bits: 128, 256, 512")
set_property(CACHE SCALARS_SVE_VECTOR_BITS PROPERTY STRINGS 128 256 512)
if(NOT SCALARS_SVE_VECTOR_BITS MATCHES "^(128|256|512)$")
    message(FATAL_ERROR "SCALARS_SVE_VECTOR_BITS must be 128, 256, or 512 (got '${SCALARS_SVE_VECTOR_BITS}')")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/utils.cmake")

# compile_definition() in Vectorization maps VECTORIZATION_ENABLE_SVML to
# VECTORIZATION_HAS_SVML=1/0 from the option itself, including when the probe
# above forces it on. A missing ThirdParty package warns; it does not clear the flag.
if(SCALARS_ENABLE_SVML)
    set(SCALARS_HAS_SVML 1)
else()
    set(SCALARS_HAS_SVML 0)
endif()

if(NOT SCALARS_CPU_BACKEND STREQUAL "no" AND NOT _scalars_has_sse AND NOT _scalars_has_avx
   AND NOT _scalars_has_avx2 AND NOT _scalars_has_avx512 AND NOT _scalars_has_neon AND NOT _scalars_has_sve
)
    message(WARNING
        "SCALARS_CPU_BACKEND '${SCALARS_CPU_BACKEND}' is not supported by this compiler/architecture. "
        "Falling back to SCALARS_CPU_BACKEND=no."
    )
    set(SCALARS_CPU_BACKEND "no"
        CACHE STRING "CPU SIMD backend for Scalars. Options: no, sse, avx, avx2, avx512, neon, sve" FORCE
    )
endif()

set(SCALARS_HAS_SSE ${_scalars_has_sse})
set(SCALARS_HAS_AVX ${_scalars_has_avx})
set(SCALARS_HAS_AVX2 ${_scalars_has_avx2})
set(SCALARS_HAS_AVX512 ${_scalars_has_avx512})
set(SCALARS_HAS_NEON ${_scalars_has_neon})
set(SCALARS_HAS_SVE ${_scalars_has_sve})

# Build/link ThirdParty SLEEF (vendored submodule) when requested, or auto-enabled
# above for a NEON/SVE host without Apple Accelerate. Ported from Vectorization's
# include/CMakeLists.txt; include/backend/cpu/{sse,avx,avx512,neon}/svml.h and
# backend/cpu/x86/math_fallback.h branch on SCALARS_HAS_SLEEF.
set(SCALARS_HAS_SLEEF 0)
if(SCALARS_ENABLE_SLEEF)
    include("${CMAKE_CURRENT_LIST_DIR}/scalars_dependencies.cmake")
    scalars_setup_sleef()
    if(TARGET Sleef::sleef)
        set(SCALARS_HAS_SLEEF 1)
    endif()
endif()

# Apple Accelerate vForce is a system framework -- no ThirdParty build needed,
# just a link flag applied in scalars_apply_simd_compile_definitions() below.
set(SCALARS_HAS_ACCELERATE 0)
if(SCALARS_ENABLE_ACCELERATE)
    if(NOT APPLE)
        message(WARNING "SCALARS_ENABLE_ACCELERATE=ON is only supported on Apple platforms; disabling.")
        set(SCALARS_ENABLE_ACCELERATE OFF
            CACHE BOOL "Enable Apple Accelerate vForce on AArch64 NEON" FORCE
        )
    else()
        set(SCALARS_HAS_ACCELERATE 1)
    endif()
endif()

# 1 when any CPU SIMD backend/cpu/* tree is active (GPU does not affect this) --
# gates backend/simd.h's system-intrinsic-header include block.
if(_scalars_has_sse OR _scalars_has_avx OR _scalars_has_avx2 OR _scalars_has_avx512
   OR _scalars_has_neon OR _scalars_has_sve
)
    set(SCALARS_VECTORIZED 1)
else()
    set(SCALARS_VECTORIZED 0)
endif()

unset(_scalars_simd_proc)
unset(_scalars_simd_is_x86)
unset(_scalars_simd_is_aarch64)
unset(_scalars_cpu_backend_default)
unset(_scalars_has_sse)
unset(_scalars_has_avx)
unset(_scalars_has_avx2)
unset(_scalars_has_avx512)
unset(_scalars_has_neon)
unset(_scalars_has_sve)

# scalars_apply_simd_compile_definitions(<target>)
#
# Applies the one-hot SCALARS_HAS_* defines, the ifdef-style
# SCALARS_CPU_CAPABILITY_AVX2/AVX512 defines, and SCALARS_SIMD_COMPILER_FLAGS
# as PUBLIC settings on <target>.
function(scalars_apply_simd_compile_definitions target)
    target_compile_definitions(
        ${target}
        PUBLIC SCALARS_HAS_SSE=${SCALARS_HAS_SSE}
               SCALARS_HAS_AVX=${SCALARS_HAS_AVX}
               SCALARS_HAS_AVX2=${SCALARS_HAS_AVX2}
               SCALARS_HAS_AVX512=${SCALARS_HAS_AVX512}
               SCALARS_HAS_NEON=${SCALARS_HAS_NEON}
               SCALARS_HAS_SVE=${SCALARS_HAS_SVE}
               SCALARS_HAS_SVML=${SCALARS_HAS_SVML}
               SCALARS_HAS_MKL=${SCALARS_HAS_MKL}
               SCALARS_HAS_SLEEF=${SCALARS_HAS_SLEEF}
               SCALARS_HAS_ACCELERATE=${SCALARS_HAS_ACCELERATE}
               SCALARS_VECTORIZED=${SCALARS_VECTORIZED}
               SCALARS_PACKET_SIZE=${SCALARS_PACKET_SIZE}
    )
    if(SCALARS_CPU_BACKEND STREQUAL "avx2")
        target_compile_definitions(${target} PUBLIC SCALARS_CPU_CAPABILITY_AVX2)
    endif()
    if(SCALARS_CPU_BACKEND STREQUAL "avx512")
        target_compile_definitions(${target} PUBLIC SCALARS_CPU_CAPABILITY_AVX512)
    endif()
    # Link ThirdParty SVML and stage its runtime next to the library, matching
    # Vectorization. On Linux that directory is already on the test RPATH; on
    # Windows it is the directory of the DLL, next to the test executables.
    if(SCALARS_ENABLE_SVML AND TARGET SVML::SVML)
        target_link_libraries(${target} PUBLIC SVML::SVML)
        if(TARGET SVML::IRC)
            target_link_libraries(${target} PUBLIC SVML::IRC)
        endif()
        if(TARGET SVML::INTLC)
            target_link_libraries(${target} PUBLIC SVML::INTLC)
        endif()
        foreach(_svml_rt IN LISTS SVML_RUNTIME_LIBRARIES)
            if(_svml_rt)
                add_custom_command(
                    TARGET ${target}
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_svml_rt}"
                            "$<TARGET_FILE_DIR:${target}>"
                    COMMENT "Staging SVML runtime: ${_svml_rt}"
                    VERBATIM
                )
            endif()
        endforeach()
        unset(_svml_rt)
    endif()
    if(SCALARS_HAS_MKL AND TARGET MKL::MKL)
        target_link_libraries(${target} PUBLIC MKL::MKL)
    endif()
    if(SCALARS_HAS_SLEEF AND TARGET Sleef::sleef)
        target_link_libraries(${target} PUBLIC Sleef::sleef)
    endif()
    if(SCALARS_HAS_ACCELERATE)
        target_link_libraries(${target} PUBLIC "-framework Accelerate")
    endif()
    if(SCALARS_SIMD_COMPILER_FLAGS)
        if(MSVC)
            separate_arguments(_scalars_simd_flags_list WINDOWS_COMMAND "${SCALARS_SIMD_COMPILER_FLAGS}")
        else()
            separate_arguments(_scalars_simd_flags_list UNIX_COMMAND "${SCALARS_SIMD_COMPILER_FLAGS}")
        endif()
        target_compile_options(${target} PUBLIC ${_scalars_simd_flags_list})
    endif()
endfunction()

if(NOT DEFINED SCALARS_PACKET_SIZE)
    set(SCALARS_PACKET_SIZE "1" CACHE STRING
        "SIMD registers processed per unrolled loop iteration in the ported backend/ kernels (1,2,3,...)"
    )
endif()
if(NOT SCALARS_PACKET_SIZE MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "SCALARS_PACKET_SIZE must be a positive decimal integer (got '${SCALARS_PACKET_SIZE}')")
endif()

message(STATUS "Scalars: SIMD compiler flags: ${SCALARS_SIMD_COMPILER_FLAGS}")
