# CPU SIMD backend selection and compiler probing for Scalars, ported from
# Vectorization's include/Cmake/utils.cmake (xsigma_/VECTORIZATION_ -> scalars_/SCALARS_),
# trimmed of the Ubuntu-GCC-version check and INTERN_BUILD_MOBILE branch (not
# applicable to Scalars).
#
# SCALARS_CPU_BACKEND (cache: no|sse|avx|avx2|avx512|neon|sve) selects the CPU SIMD
# backend used by include/backend/cpu/*. C++ preprocessor defines:
#   sse -> SCALARS_HAS_SSE=1; avx/avx2 -> SCALARS_HAS_AVX=1; avx2 -> SCALARS_HAS_AVX2=1;
#   avx512 -> SCALARS_HAS_AVX512=1; neon -> SCALARS_HAS_NEON=1; sve -> SCALARS_HAS_SVE=1.
# SCALARS_CPU_CAPABILITY_AVX2 / SCALARS_CPU_CAPABILITY_AVX512 are defined (ifdef-style,
# no value) exactly when backend is avx2 / avx512 -- these are what
# include/types/half.h and include/detail/vec_half.h test via #if defined(...).
include_guard(GLOBAL)

include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)
include(CMakePushCheckState)

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

option(SCALARS_ENABLE_SVML "Enable Intel SVML short vector math library for Scalars backend/" OFF)
mark_as_advanced(SCALARS_ENABLE_SVML)
option(SCALARS_ENABLE_MKL "Enable Intel MKL VML vector math backend (x86 only, static link)" OFF)
mark_as_advanced(SCALARS_ENABLE_MKL)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/packages")

set(SCALARS_HAS_SVML 0)
set(SCALARS_HAS_MKL 0)
if(SCALARS_ENABLE_SVML)
    find_package(SVML QUIET)
    if(SVML_FOUND)
        set(SCALARS_HAS_SVML 1)
        message(STATUS "Scalars: SVML ThirdParty package found.")
    else()
        message(WARNING
            "SCALARS_ENABLE_SVML=ON but ThirdParty SVML binaries were not found. "
            "Ensure ThirdParty/svml/ contains the required libraries."
        )
    endif()
endif()
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

set(SCALARS_SIMD_COMPILER_FLAGS "")
set(_scalars_has_sse 0)
set(_scalars_has_avx 0)
set(_scalars_has_avx2 0)
set(_scalars_has_avx512 0)
set(_scalars_has_neon 0)
set(_scalars_has_sve 0)

if(_scalars_simd_is_x86)
    cmake_push_check_state(RESET)
    if(MSVC)
        set(CMAKE_REQUIRED_FLAGS "/arch:SSE2")
    else()
        set(CMAKE_REQUIRED_FLAGS "-msse -msse2")
    endif()
    check_cxx_source_compiles(
        "#include <immintrin.h>
         int main() { __m128 a = _mm_set1_ps(1); _mm_add_ps(a, a); return 0; }"
        SCALARS_COMPILER_SUPPORTS_SSE
    )
    if(SCALARS_COMPILER_SUPPORTS_SSE AND SCALARS_CPU_BACKEND STREQUAL "sse")
        set(_scalars_has_sse 1)
        set(SCALARS_SIMD_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
    endif()
    cmake_pop_check_state()

    cmake_push_check_state(RESET)
    if(MSVC)
        set(CMAKE_REQUIRED_FLAGS "/arch:AVX /D__F16C__")
    else()
        set(CMAKE_REQUIRED_FLAGS "-mavx -mf16c")
    endif()
    check_cxx_source_compiles(
        "#include <immintrin.h>
         int main() { __m256 a = _mm256_set1_ps(1); _mm256_add_ps(a, a); return 0; }"
        SCALARS_COMPILER_SUPPORTS_AVX
    )
    if(SCALARS_COMPILER_SUPPORTS_AVX AND SCALARS_CPU_BACKEND STREQUAL "avx")
        set(_scalars_has_avx 1)
        set(SCALARS_SIMD_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
    endif()
    cmake_pop_check_state()

    cmake_push_check_state(RESET)
    if(MSVC)
        set(CMAKE_REQUIRED_FLAGS "/arch:AVX2 /D__F16C__")
    else()
        set(CMAKE_REQUIRED_FLAGS "-mavx2 -mf16c")
    endif()
    check_cxx_source_compiles(
        "#include <immintrin.h>
         int main() {
           __m256i a = _mm256_set1_epi8(1);
           _mm256_add_epi8(a, a);
           __m256i x; _mm256_extract_epi64(x, 0);
           return 0;
         }"
        SCALARS_COMPILER_SUPPORTS_AVX2
    )
    if(SCALARS_COMPILER_SUPPORTS_AVX2 AND SCALARS_CPU_BACKEND STREQUAL "avx2")
        set(_scalars_has_avx 1)
        set(_scalars_has_avx2 1)
        set(SCALARS_SIMD_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
    endif()
    cmake_pop_check_state()

    # Compile-only: do not switch to check_cxx_source_runs() -- hosted CI can
    # emit AVX-512 without avx512f silicon present.
    cmake_push_check_state(RESET)
    if(MSVC)
        set(CMAKE_REQUIRED_FLAGS "/arch:AVX512 /D__F16C__")
    else()
        set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512cd -mf16c")
    endif()
    check_cxx_source_compiles(
        "#if defined(_MSC_VER)
         #include <intrin.h>
         #else
         #include <immintrin.h>
         #endif
         __m512 addConstant(__m512 arg) { return _mm512_add_ps(arg, _mm512_set1_ps(1.f)); }
         __m512 andConstant(__m512 arg) { return _mm512_and_ps(arg, _mm512_set1_ps(1.f)); }
         int main() {
           __m512i a = _mm512_set1_epi32(1);
           __m256i ymm = _mm512_extracti64x4_epi64(a, 0);
           ymm = _mm256_abs_epi64(ymm);
           __mmask16 m = _mm512_cmp_epi32_mask(a, a, _MM_CMPINT_EQ);
           __m512i r = _mm512_andnot_si512(a, a);
           return 0;
         }"
        SCALARS_COMPILER_SUPPORTS_AVX512
    )
    if(SCALARS_COMPILER_SUPPORTS_AVX512 AND SCALARS_CPU_BACKEND STREQUAL "avx512")
        set(_scalars_has_avx512 1)
        set(SCALARS_SIMD_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
    endif()
    cmake_pop_check_state()
endif()

if(_scalars_simd_is_aarch64)
    cmake_push_check_state(RESET)
    if(NOT MSVC)
        check_cxx_compiler_flag("-march=armv8.2-a" _scalars_neon_armv82)
        if(_scalars_neon_armv82)
            set(CMAKE_REQUIRED_FLAGS "-march=armv8.2-a")
        else()
            set(CMAKE_REQUIRED_FLAGS "-march=armv8-a")
        endif()
    endif()
    check_cxx_source_compiles(
        "#include <arm_neon.h>
         int main() { float32x4_t a = vdupq_n_f32(1.f); float32x4_t b = vfmaq_f32(a, a, a); (void)b; return 0; }"
        SCALARS_COMPILER_SUPPORTS_NEON
    )
    if(SCALARS_COMPILER_SUPPORTS_NEON AND SCALARS_CPU_BACKEND STREQUAL "neon")
        set(_scalars_has_neon 1)
        if(NOT MSVC)
            set(SCALARS_SIMD_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
        endif()
    endif()
    cmake_pop_check_state()

    # SVE vector register width. include/backend/cpu/sve/* only supports 128 bits.
    set(SCALARS_SVE_VECTOR_BITS "128" CACHE STRING "SVE vector register width in bits: 128, 256, 512")
    set_property(CACHE SCALARS_SVE_VECTOR_BITS PROPERTY STRINGS 128 256 512)

    cmake_push_check_state(RESET)
    if(NOT MSVC)
        set(CMAKE_REQUIRED_FLAGS "-march=armv8-a+sve -msve-vector-bits=${SCALARS_SVE_VECTOR_BITS}")
    endif()
    check_cxx_source_compiles(
        "#include <arm_sve.h>
         int main() { svbool_t pg = svptrue_b32(); (void)pg; return 0; }"
        SCALARS_COMPILER_SUPPORTS_SVE
    )
    if(SCALARS_COMPILER_SUPPORTS_SVE AND SCALARS_CPU_BACKEND STREQUAL "sve")
        set(_scalars_has_sve 1)
        if(NOT MSVC)
            set(SCALARS_SIMD_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
        endif()
    endif()
    cmake_pop_check_state()
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
               SCALARS_VECTORIZED=${SCALARS_VECTORIZED}
               SCALARS_PACKET_SIZE=${SCALARS_PACKET_SIZE}
    )
    if(SCALARS_CPU_BACKEND STREQUAL "avx2")
        target_compile_definitions(${target} PUBLIC SCALARS_CPU_CAPABILITY_AVX2)
    endif()
    if(SCALARS_CPU_BACKEND STREQUAL "avx512")
        target_compile_definitions(${target} PUBLIC SCALARS_CPU_CAPABILITY_AVX512)
    endif()
    if(SCALARS_HAS_SVML AND TARGET SVML::SVML)
        target_link_libraries(${target} PUBLIC SVML::SVML)
    endif()
    if(SCALARS_HAS_MKL AND TARGET MKL::MKL)
        target_link_libraries(${target} PUBLIC MKL::MKL)
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
