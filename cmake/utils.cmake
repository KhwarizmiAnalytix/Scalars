# Compiler SIMD probes for Scalars, from Vectorization's include/Cmake/utils.cmake
# (xsigma_/VECTORIZATION_ -> scalars_/SCALARS_). The Ubuntu GCC-version check and
# INTERN_BUILD_MOBILE branch are omitted; they do not apply here.
#
# Requires SCALARS_CPU_BACKEND, SCALARS_ENABLE_SVML, SCALARS_ENABLE_SLEEF,
# SCALARS_ENABLE_ACCELERATE, and SCALARS_SVE_VECTOR_BITS to be defined.
# Sets _scalars_has_{sse,avx,avx2,avx512,neon,sve} and SCALARS_SIMD_COMPILER_FLAGS.
# May FORCE SCALARS_ENABLE_SVML, SCALARS_ENABLE_SLEEF, and SCALARS_ENABLE_ACCELERATE.
include_guard(GLOBAL)

include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)
include(CMakePushCheckState)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _scalars_vec_proc)
if(_scalars_vec_proc MATCHES "^(x86_64|amd64|i686|i386)$")
    set(_scalars_vec_is_x86 TRUE)
else()
    set(_scalars_vec_is_x86 FALSE)
endif()
if(_scalars_vec_proc MATCHES "^(aarch64|arm64)$")
    set(_scalars_vec_is_aarch64 TRUE)
else()
    set(_scalars_vec_is_aarch64 FALSE)
endif()

set(_scalars_simd_active FALSE)
set(_scalars_has_sse 0)
set(_scalars_has_avx 0)
set(_scalars_has_avx2 0)
set(_scalars_has_avx512 0)
set(_scalars_has_neon 0)
set(_scalars_has_sve 0)
set(SCALARS_SIMD_COMPILER_FLAGS "")

if(_scalars_vec_is_x86)
    # ---[ Check if the compiler has SSE support.
    cmake_push_check_state(RESET)
    if(MSVC)
        set(CMAKE_REQUIRED_FLAGS "/arch:SSE2")
    else()
        set(CMAKE_REQUIRED_FLAGS "-msse -msse2")
    endif()
    check_cxx_source_compiles(
        "#include <immintrin.h>
         int main() {
           __m128 a, b;
           a = _mm_set1_ps(1);
           _mm_add_ps(a, a);
           return 0;
         }"
        SCALARS_COMPILER_SUPPORTS_SSE
    )
    if(SCALARS_COMPILER_SUPPORTS_SSE)
        message(STATUS "Scalars: compiler supports SSE.")
        if(SCALARS_CPU_BACKEND STREQUAL "sse")
            set(_scalars_has_sse 1)
            set(_scalars_simd_active TRUE)
            set(SCALARS_SIMD_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
        endif()
    endif()
    cmake_pop_check_state()

    # ---[ Check if the compiler has AVX support.
    cmake_push_check_state(RESET)
    if(MSVC)
        set(CMAKE_REQUIRED_FLAGS "/arch:AVX /D__F16C__")
    else()
        set(CMAKE_REQUIRED_FLAGS "-mavx -mf16c")
    endif()
    check_cxx_source_compiles(
        "#include <immintrin.h>
         int main() {
           __m256 a, b;
           a = _mm256_set1_ps(1);
           b = a;
           _mm256_add_ps(a, a);
           return 0;
         }"
        SCALARS_COMPILER_SUPPORTS_AVX
    )
    if(SCALARS_COMPILER_SUPPORTS_AVX)
        message(STATUS "Scalars: compiler supports AVX.")
        if(SCALARS_CPU_BACKEND STREQUAL "avx")
            set(_scalars_has_avx 1)
            set(_scalars_simd_active TRUE)
            set(SCALARS_SIMD_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
        endif()
    endif()
    cmake_pop_check_state()

    # ---[ Check if the compiler has AVX2 support.
    cmake_push_check_state(RESET)
    if(MSVC)
        set(CMAKE_REQUIRED_FLAGS "/arch:AVX2 /D__F16C__")
    else()
        set(CMAKE_REQUIRED_FLAGS "-mavx2 -mf16c")
    endif()
    check_cxx_source_compiles(
        "#include <immintrin.h>
         int main() {
           __m256i a, b;
           a = _mm256_set1_epi8(1);
           b = a;
           _mm256_add_epi8(a, a);
           __m256i x;
           _mm256_extract_epi64(x, 0);
           return 0;
         }"
        SCALARS_COMPILER_SUPPORTS_AVX2
    )
    if(SCALARS_COMPILER_SUPPORTS_AVX2)
        message(STATUS "Scalars: compiler supports AVX2.")
        if(SCALARS_CPU_BACKEND STREQUAL "avx2")
            # AVX2 implies AVX. Vectorization emits both defines from the backend name.
            set(_scalars_has_avx 1)
            set(_scalars_has_avx2 1)
            set(_scalars_simd_active TRUE)
            set(SCALARS_SIMD_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
        endif()
    endif()
    cmake_pop_check_state()

    # ---[ Check if the compiler has AVX512 support.
    # Compile-only: do not change this to check_cxx_source_runs(). Hosted CI
    # can emit AVX-512 but typically has no avx512f silicon; a run-time probe
    # would disable the backend.
    cmake_push_check_state(RESET)
    if(MSVC)
        # /arch:AVX512 implies AVX512F, AVX512CD, AVX512BW, AVX512DQ, and AVX512VL.
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
         __m512 addConstant(__m512 arg) {
           return _mm512_add_ps(arg, _mm512_set1_ps(1.f));
         }
         __m512 andConstant(__m512 arg) {
           return _mm512_and_ps(arg, _mm512_set1_ps(1.f));
         }
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
    if(SCALARS_COMPILER_SUPPORTS_AVX512)
        message(STATUS "Scalars: compiler supports AVX-512.")
        if(SCALARS_CPU_BACKEND STREQUAL "avx512")
            set(_scalars_has_avx512 1)
            set(_scalars_simd_active TRUE)
            set(SCALARS_SIMD_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
        endif()
    endif()
    cmake_pop_check_state()

    # ---[ Check if the compiler has FMA support.
    cmake_push_check_state(RESET)
    if(MSVC)
        set(CMAKE_REQUIRED_FLAGS "${SCALARS_SIMD_COMPILER_FLAGS} /D__FMA__")
    else()
        set(CMAKE_REQUIRED_FLAGS "${SCALARS_SIMD_COMPILER_FLAGS} -mfma")
    endif()
    check_cxx_source_compiles(
        "#if defined(_MSC_VER)
         #include <intrin.h>
         #else
         #include <immintrin.h>
         #endif
         int main() {
           __m128 a, b;
           a = _mm_set1_ps(1);
           b = _mm_set1_ps(1);
           a = _mm_fmadd_ps(a, b, b);
           return 0;
         }"
        SCALARS_COMPILER_SUPPORTS_FMA
    )
    if(SCALARS_COMPILER_SUPPORTS_FMA)
        message(STATUS "Scalars: compiler supports FMA.")
        # Only apply -mfma when an x86 CPU SIMD backend was actually selected.
        # This probe used to run unconditionally and leaked -mfma into every
        # consumer, including CUDA/HIP host passes that do not use CPU SIMD.
        if(_scalars_simd_active)
            set(SCALARS_SIMD_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
        endif()
    endif()
    cmake_pop_check_state()
endif()

if(_scalars_vec_is_x86)
    # ---[ Check if the compiler has SVML support.
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_FLAGS "${SCALARS_SIMD_COMPILER_FLAGS}")
    check_cxx_source_compiles(
        "#if defined(_MSC_VER)
         #include <intrin.h>
         #else
         #include <immintrin.h>
         #endif
         int main() {
           __m256 a, b;
           a = _mm256_setzero_ps();
           b = _mm256_exp_ps(a);
           b = _mm256_cos_ps(a);
           b = _mm256_tanh_ps(a);
           return 0;
         }"
        SCALARS_COMPILER_SUPPORTS_SVML_EXTENSIONS
    )
    if(NOT SCALARS_COMPILER_SUPPORTS_SVML_EXTENSIONS AND _scalars_simd_active AND NOT SCALARS_ENABLE_SLEEF)
        message(WARNING
            "Scalars: compiler does not provide SVML functions. Turning SCALARS_ENABLE_SVML ON."
        )
        set(SCALARS_ENABLE_SVML ON
            CACHE BOOL "Enable Intel SVML short vector math library" FORCE
        )
    endif()
    cmake_pop_check_state()

    if(SCALARS_ENABLE_SVML)
        find_package(SVML QUIET)
        if(SVML_FOUND)
            message(STATUS
                "Scalars: SVML ThirdParty package found. Runtime files: ${SVML_RUNTIME_LIBRARIES}"
            )
        else()
            message(WARNING
                "SCALARS_ENABLE_SVML=ON but ThirdParty SVML binaries were not found. "
                "Ensure ThirdParty/svml/ contains the required libraries."
            )
        endif()
    endif()
endif()

if(_scalars_vec_is_aarch64)
    # ---[ AArch64 NEON. Probe armv8.2-a first, then baseline armv8-a.
    cmake_push_check_state(RESET)
    if(NOT MSVC)
        check_cxx_compiler_flag("-march=armv8.2-a" _scalars_vec_neon_armv82)
        if(_scalars_vec_neon_armv82)
            set(CMAKE_REQUIRED_FLAGS "-march=armv8.2-a")
        else()
            set(CMAKE_REQUIRED_FLAGS "-march=armv8-a")
        endif()
    endif()
    check_cxx_source_compiles(
        "#include <arm_neon.h>
         int main() {
           float32x4_t a = vdupq_n_f32(1.f);
           float32x4_t b = vfmaq_f32(a, a, a);
           (void)b;
           return 0;
         }"
        SCALARS_COMPILER_SUPPORTS_NEON
    )
    if(SCALARS_COMPILER_SUPPORTS_NEON AND SCALARS_CPU_BACKEND STREQUAL "neon")
        message(STATUS "Scalars: compiler supports AArch64 NEON.")
        set(_scalars_has_neon 1)
        set(_scalars_simd_active TRUE)
        if(NOT MSVC)
            set(SCALARS_SIMD_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
        endif()
    endif()
    cmake_pop_check_state()

    # ---[ AArch64 SVE. Width comes from SCALARS_SVE_VECTOR_BITS.
    cmake_push_check_state(RESET)
    if(NOT MSVC)
        set(CMAKE_REQUIRED_FLAGS "-march=armv8-a+sve -msve-vector-bits=${SCALARS_SVE_VECTOR_BITS}")
    endif()
    check_cxx_source_compiles(
        "#include <arm_sve.h>
         int main() {
           svbool_t pg = svptrue_b32();
           (void)pg;
           return 0;
         }"
        SCALARS_COMPILER_SUPPORTS_SVE
    )
    if(SCALARS_COMPILER_SUPPORTS_SVE AND SCALARS_CPU_BACKEND STREQUAL "sve")
        message(STATUS
            "Scalars: compiler supports AArch64 SVE (${SCALARS_SVE_VECTOR_BITS}-bit vector length)."
        )
        set(_scalars_has_sve 1)
        set(_scalars_simd_active TRUE)
        if(NOT MSVC)
            set(SCALARS_SIMD_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
        endif()
    endif()
    cmake_pop_check_state()

    # ---[ Probe Apple Accelerate vForce (vvexpf/vvsinf/vvcosf/vvlogf).
    # include/backend/cpu/neon/svml.h is also used for SVE's fixed 128-bit layout,
    # so either tier triggers the probe. Without Accelerate, SLEEF is turned on.
    if(_scalars_has_neon OR _scalars_has_sve)
        if(APPLE AND NOT SCALARS_ENABLE_SLEEF)
            cmake_push_check_state(RESET)
            set(CMAKE_REQUIRED_LIBRARIES "-framework Accelerate")
            check_cxx_source_compiles(
                "#include <Accelerate/Accelerate.h>
                 int main() {
                   const int n = 4;
                   float x[4] = {1.f, 2.f, 3.f, 4.f};
                   float y[4];
                   vvexpf(y, x, &n);
                   vvsinf(y, x, &n);
                   vvcosf(y, x, &n);
                   vvlogf(y, x, &n);
                   return 0;
                 }"
                SCALARS_COMPILER_SUPPORTS_ACCELERATE
            )
            cmake_pop_check_state()
            if(SCALARS_COMPILER_SUPPORTS_ACCELERATE)
                message(STATUS "Scalars: Accelerate vForce available -- "
                               "auto-enabling SCALARS_ENABLE_ACCELERATE."
                )
                set(SCALARS_ENABLE_ACCELERATE ON
                    CACHE BOOL "Enable Apple Accelerate vForce on AArch64 NEON (auto-enabled)" FORCE
                )
            else()
                message(STATUS "Scalars: Accelerate vForce unavailable -- "
                               "auto-enabling SCALARS_ENABLE_SLEEF."
                )
                set(SCALARS_ENABLE_SLEEF ON
                    CACHE BOOL "Enable SLEEF SIMD math library (auto-enabled: Accelerate vForce unavailable)"
                    FORCE
                )
            endif()
        elseif(NOT APPLE AND NOT SCALARS_ENABLE_SLEEF)
            message(STATUS "Scalars: non-Apple AArch64 detected -- Accelerate vForce is "
                           "unavailable, auto-enabling SCALARS_ENABLE_SLEEF."
            )
            set(SCALARS_ENABLE_SLEEF ON
                CACHE BOOL "Enable SLEEF SIMD math library (auto-enabled: Accelerate unavailable on non-Apple)"
                FORCE
            )
        endif()
    endif()
endif()

if(USE_NATIVE_ARCH)
    check_cxx_compiler_flag("-march=native" SCALARS_COMPILER_SUPPORTS_MARCH_NATIVE)
    if(SCALARS_COMPILER_SUPPORTS_MARCH_NATIVE)
        string(APPEND CMAKE_C_FLAGS " -march=native")
        string(APPEND CMAKE_CXX_FLAGS " -march=native")
    else()
        message(WARNING "The compiler does not support -march=native. "
                        "Turn off this warning by setting -DUSE_NATIVE_ARCH=OFF."
        )
    endif()
endif()

unset(_scalars_vec_proc)
unset(_scalars_vec_is_x86)
unset(_scalars_vec_is_aarch64)
unset(_scalars_vec_neon_armv82)
unset(_scalars_simd_active)
